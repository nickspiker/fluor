//! Text rendering — verbatim port of photon's [text_rasterizing.rs](/mnt/Harbor/Code/photon/src/ui/text_rasterizing.rs). Built on cosmic-text + swash. Bundles Open Sans as one variable face, every weight 300..800 real (OFL-licensed) so the default fluor build renders text out of the box; consumers can load additional fonts via the `font_system_mut()` accessor.
//!
//! Photon's full version bundles Oxanium and Josefin Slab too; those are dropped here to keep the v0 crate small (~260 KB instead of ~2 MB). They'll be added back when a consumer needs them.

use crate::canvas::{Canvas, Damage};
use crate::paint::{AlphaMask, Clip, Transform};
use crate::pixel::{BlendMode, Pixel};
use cosmic_text::fontdb::Database;
use cosmic_text::{Attrs, Buffer, Fallback, Family, FontSystem, Metrics, Shaping, SwashCache, Weight};
use unicode_script::Script;
use swash::scale::{Render, ScaleContext, Source, image::Image as SwashImage};
use swash::zeno::Transform as PathTransform;

// The 8-bit-only blits (legacy byte buffers, additive tints), kept apart from the α + darkness path every depth shares.
mod argb8;

/// Report the painted text bbox to the canvas damage accumulator. Bbox is the alignment-adjusted text rect clipped to buffer dimensions — slightly conservative versus the per-glyph union, but cheap and never under-reports. `alignment`: 0 = center, 1 = left, 2 = right (matching `render_buffer_u32`'s convention).
#[inline]
fn damage_text_bbox(
    damage: &mut Damage,
    buf_w: usize,
    buf_h: usize,
    anchor_x: f32,
    anchor_y: f32,
    text_width: f32,
    text_height: f32,
    alignment: u8,
) {
    let (lx, ty) = match alignment {
        0 => (anchor_x - text_width * 0.5, anchor_y - text_height * 0.5),
        1 => (anchor_x, anchor_y - text_height * 0.5),
        2 => (anchor_x - text_width, anchor_y - text_height * 0.5),
        _ => (anchor_x, anchor_y),
    };
    let rx = lx + text_width;
    let by = ty + text_height;
    let w = buf_w as f32;
    let h = buf_h as f32;
    // WHY/PROOF: the anchor is the caller's text position, past the buffer whenever the text scrolls off an edge — the min keeps the damage rect inside the buffer; a negative edge needs no floor, since the f32 → usize cast saturates it to 0.
    let x0 = lx.min(w) as usize;
    let y0 = ty.min(h) as usize;
    let x1 = rx.min(w) as usize;
    let y1 = by.min(h) as usize;
    damage.add_bounds(x0, y0, x1, y1);
}

/// Cache key for rasterized transformed-glyph images. The full linear part (`a, b, c, d`) of the transform is included as bit-pattern equality so any unique linear transform — pure rotation, scale, skew, mirror, mix — gets its own cache entry. Translation is excluded because it doesn't affect the glyph bitmap, only where it gets composited. Callers that want cache hits across small angle changes should snap with [`crate::paint::snap_rotation`] before constructing the Transform.
#[derive(Copy, Clone, PartialEq, Eq)]
struct TransformGlyphKey {
    font_id: cosmic_text::fontdb::ID,
    /// The weight the glyph is drawn at: one variable font draws every weight, so the weight is part of the glyph's identity.
    weight: u16,
    glyph_id: u16,
    size_bits: u32,
    a_bits: u32,
    b_bits: u32,
    c_bits: u32,
    d_bits: u32,
}

/// Linear-scan + bump-to-front LRU cache for swash-rasterized transformed glyphs. Capacity-bounded; tail-evicts on overflow. No `HashMap` — the keyspace is small (visible glyphs at active rotations) and a Vec scan with cache-friendly contiguous layout beats hashing at the N we care about.
struct TransformGlyphCache {
    entries: Vec<(TransformGlyphKey, SwashImage)>,
    cap: usize,
}

impl TransformGlyphCache {
    fn new(cap: usize) -> Self {
        Self {
            entries: Vec::with_capacity(cap.min(64)),
            cap,
        }
    }

    /// Look up a key. On hit, bumps the entry to the front (LRU). Returns the slice index of the (front) entry on hit, `None` on miss.
    fn lookup(&mut self, key: TransformGlyphKey) -> Option<()> {
        let pos = self.entries.iter().position(|(k, _)| *k == key)?;
        if pos != 0 {
            self.entries[..=pos].rotate_right(1);
        }
        Some(())
    }

    /// Insert at the front, evict tail if over capacity.
    fn insert(&mut self, key: TransformGlyphKey, image: SwashImage) {
        self.entries.insert(0, (key, image));
        if self.entries.len() > self.cap {
            self.entries.pop();
        }
    }
}

pub struct TextRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    /// Owned by `TextRenderer` for the transform-aware glyph path. Cosmic-text's `SwashCache` only handles identity-transform glyphs; for any rotated / scaled / skewed text we go to swash directly so the glyph contour transforms in font space (proper hinting + AA), then composite the result.
    scale_context: ScaleContext,
    /// Cache for transform-rasterized glyph images. Identity transforms use cosmic-text's own cache; only non-identity rasterizations land here.
    transform_cache: TransformGlyphCache,
    /// OPTICAL WEIGHT (Nick 2026-10-14, with variable fonts): see [`TextRenderer::set_optical_weight`]. Off by default.
    optical: bool,
    /// Font packs ([`crate::packs`]), as bits by index: looked for in the db, loaded (there, bundled or fetched), wanted by text drawn without them, and offered to the app.
    packs_checked: u64,
    packs_loaded: u64,
    packs_wanted: u64,
    packs_offered: u64,
    /// The regional face for Han text with no kana or Hangul to say otherwise.
    han: crate::packs::Han,
    /// Bumped whenever a face is added, so whatever measured text before (a label's advances) measures again.
    fonts_version: u64,
}

/// How text draws — everything beyond "which glyphs, where". REQUIRED values (size, colour) live in [`TextStyle::new`]; every OPTION defaults to none and is added by a chainable builder. Fields are PRIVATE on purpose: no struct literals means adding the next option (kerning, vertical shear, …) can never break a call site — the forward-compat contract this API was rebuilt for (2026-07-17).
///
/// The transform options compose in font space as `rotate(rotation) ∘ [stretch, skew; 0, 1]` and feed the SAME contour+position machinery rotated titles already use — so skewed text is a true synthetic oblique (contours skew before rasterization, proper AA), stretch scales advances automatically, and the transformed-glyph cache keys on the resulting matrix with no extra plumbing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextStyle<'a> {
    size: f32,
    colour: u32,
    weight: u16,
    font: &'a str,
    shear: f32,
    stretch: f32,
    rotation: f32,
}

impl<'a> TextStyle<'a> {
    /// The required pair: `size` (px, span-scaled by the caller like everything in fluor) and `colour` (α+darkness u32). There is deliberately NO `Default` — a defaulted size or colour is an invisible-text bug, so required-ness is structural.
    pub const fn new(size: f32, colour: u32) -> Self {
        Self { size, colour, weight: 400, font: "Open Sans", shear: 0., stretch: 1., rotation: 0. }
    }
    /// Font weight (cosmic-text face selection). Default 400.
    pub const fn weight(mut self, w: u16) -> Self {
        self.weight = w;
        self
    }
    /// Font family name. Default "Open Sans".
    pub const fn font(mut self, f: &'a str) -> Self {
        self.font = f;
        self
    }
    /// Forward slant: the x-SHEAR coefficient applied to glyph CONTOURS (`x' = x + shear·y`). 0 = upright; `tan 12° ≈ 0.2126` is the classic oblique amount. Named for the mathematics — CSS says "skew" and parameterizes by angle; this is the shear COEFFICIENT (tan of that angle). Not "italic": a true italic is a different FACE; this is honest geometry, and it works on faces that ship no italic at all. Advances and baselines are untouched — only the contours lean (the shear rides the contour path alone, never the position transform).
    pub const fn shear(mut self, s: f32) -> Self {
        self.shear = s;
        self
    }
    /// Horizontal stretch of contours AND advances: 1 = natural, <1 condensed, >1 extended.
    pub const fn stretch(mut self, s: f32) -> Self {
        self.stretch = s;
        self
    }
    /// Rotation in radians. Snapping to the 1-pixel-arc cache grid stays internal.
    pub const fn rotation(mut self, r: f32) -> Self {
        self.rotation = r;
        self
    }
    /// The composed POSITION/rotation-component matrix (rotation ∘ stretch), or `None` at defaults. The shear is deliberately NOT here: it must never touch glyph positions (baseline drift) nor `base_theta` (the rotation snapper would read shear as rotation) — it composes separately at the contour handoff.
    fn transform(&self) -> Option<Transform> {
        if self.rotation == 0. && self.stretch == 1. {
            return None;
        }
        let m = Transform::new(self.stretch, 0., 0., 1., 0., 0.);
        Some(if self.rotation == 0. { m } else { m.then(Transform::rotate(self.rotation)) })
    }
}

/// The ONE fallback chain, hardcoded and identical on every platform.
/// cosmic-text's default `PlatformFallback` returns real system family names on Linux/macOS/Windows but an EMPTY list on Android (its `other.rs` impl), so a codepoint absent from the primary font resolved through a system font on desktop and rendered as tofu on Android — the "heart shows a square on her phone, fine on mine" divergence.
/// This impl names ONLY our bundled families, so glyph selection is deterministic: the same bytes render the same glyph everywhere, with no dependency on what fonts happen to be installed.
/// A codepoint we don't bundle a glyph for is tofu on ALL platforms (consistently) rather than silently working on one — bundle another font to cover it, never rely on the host.
struct BundledFallback;

impl Fallback for BundledFallback {
    fn common_fallback(&self) -> &[&'static str] {
        // After the primary font misses, in order:
        //   Oxanium — the app's glyph face (photon's `+glyphs` carries the dozenal digits at 0x10..0x1B). First and by name: cosmic-text tries the PRIMARY face before any of this, so a font carrying its own matching glyphs still wins; this only catches the miss. A family that isn't loaded is skipped, so naming it costs nothing to fluor users who don't bundle it.
        //   Noto Color Emoji BEFORE the monochrome symbol fonts (Nick 2026-09-08: ☎ ☠ ☢ came out as tinted line-art while ☮ ☯ 😀 🍄 came out in colour — the only difference was which font sat earlier in this list). Colour wins whenever a colour glyph exists; a string that wants a symbol as a tinted ICON pins it with U+FE0E (see set_text_vs).
        //   Then the monochrome coverage, broad to narrow: Symbols 2 (hearts, stars, weather, dice, geometric), Symbols (☮ ☯ ♻ and the dingbat/misc-technical ranges), Math (arrows U+2190–21FF, operators U+2200–22FF, the mathematical alphanumerics), Mono (box drawing U+2500–257F in full, currency U+20A0–20CF).
        // Every family here is a bundled file — the app loads them into this db at startup. NEVER a host family name.
        &["Oxanium", "Noto Color Emoji", "Noto Sans Symbols 2", "Noto Sans Symbols", "Noto Sans Math", "Noto Sans Mono"]
    }
    fn forbidden_fallback(&self) -> &[&'static str] {
        &[]
    }
    fn script_fallback(&self, script: Script, _locale: &str) -> &[&'static str] {
        // Per-script bundled faces, tried right after the primary font and before the common chain. Still hardcoded, still bundle-only. CJK is deliberately absent (Nick 2026-09-08: skipped for now — a pan-CJK face is 10–20 MB and roughly doubles the binary; revisit when it earns its place).
        // A script's family whether bundled by the app or loaded from a font pack later ([`crate::packs`], Nick 2026-10-16): a family not in the db is skipped, so naming it before it arrives costs nothing.
        match script {
            Script::Arabic => &["Noto Sans Arabic"],
            Script::Devanagari => &["Noto Sans Devanagari"],
            Script::Thai => &["Noto Sans Thai"],
            Script::Armenian => &["Noto Sans Armenian"],
            Script::Georgian => &["Noto Sans Georgian"],
            Script::Runic => &["Noto Sans Runic"],
            Script::Hebrew => &["Noto Sans Hebrew"],
            Script::Syriac => &["Noto Sans Syriac"],
            Script::Thaana => &["Noto Sans Thaana"],
            Script::Bengali => &["Noto Sans Bengali"],
            Script::Gurmukhi => &["Noto Sans Gurmukhi"],
            Script::Gujarati => &["Noto Sans Gujarati"],
            Script::Oriya => &["Noto Sans Oriya"],
            Script::Tamil => &["Noto Sans Tamil"],
            Script::Telugu => &["Noto Sans Telugu"],
            Script::Kannada => &["Noto Sans Kannada"],
            Script::Malayalam => &["Noto Sans Malayalam"],
            Script::Sinhala => &["Noto Sans Sinhala"],
            Script::Lao => &["Noto Sans Lao"],
            Script::Tibetan => &["Noto Serif Tibetan"],
            Script::Myanmar => &["Noto Sans Myanmar"],
            Script::Ethiopic => &["Noto Sans Ethiopic"],
            Script::Cherokee => &["Noto Sans Cherokee"],
            Script::Canadian_Aboriginal => &["Noto Sans Canadian Aboriginal"],
            Script::Khmer => &["Noto Sans Khmer"],
            Script::Mongolian => &["Noto Sans Mongolian"],
            Script::Tifinagh => &["Noto Sans Tifinagh"],
            Script::Han | Script::Bopomofo => &["Noto Sans SC", "Noto Sans TC", "Noto Sans HK", "Noto Sans JP", "Noto Sans KR"],
            Script::Hiragana | Script::Katakana => &["Noto Sans JP"],
            Script::Hangul => &["Noto Sans KR"],
            _ => &[],
        }
    }
}


/// Emoji-presentation routing by VARIATION SELECTOR, done at span level before shaping.
/// U+FE0F after a base character forces the colour-emoji face for that character; U+FE0E forces the monochrome symbol face; a bare character follows the fallback chain (colour first — see BundledFallback).
/// Why here and not in the font: cosmic-text's fallback triggers per MISSING glyph, not per selector. A selector the winning font doesn't map is just a default-ignorable zero-width glyph, so `☎\u{FE0E}` would still land on whichever face the chain reached first. Splitting the pair into its own span with an explicit family is the only lever that actually chooses the face.
/// The selector stays in the span's text: the face it routes to treats it as default-ignorable, so it draws nothing and costs no advance.
fn set_text_vs<'a>(buffer: &mut Buffer, text: &'a str, attrs: &Attrs<'a>, cjk: Option<&'static str>) {
    if cjk.is_none() && !text.contains(['\u{FE0E}', '\u{FE0F}']) {
        buffer.set_text(text, attrs, Shaping::Advanced, None);
        return;
    }
    // Each character's family override: a variation-selector pair its emoji or symbol face, a CJK character the text's regional face (Nick 2026-10-16: Japanese forms for Japanese names), the rest none. Runs of one override become one span.
    let mut spans: Vec<(&str, Attrs)> = Vec::new();
    let span = |spans: &mut Vec<(&'a str, Attrs<'a>)>, part: &'a str, family: Option<&'static str>| {
        if !part.is_empty() {
            spans.push((part, match family {
                Some(f) => attrs.clone().family(Family::Name(f)),
                None => attrs.clone(),
            }));
        }
    };
    let (mut run_start, mut run_family) = (0usize, None::<&'static str>);
    let mut it = text.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let family = match it.peek() {
            Some(&(_, vs)) if vs == '\u{FE0F}' || vs == '\u{FE0E}' => {
                it.next();
                Some(if vs == '\u{FE0F}' { "Noto Color Emoji" } else { "Noto Sans Symbols 2" })
            }
            _ if crate::packs::is_cjk(c) => cjk,
            _ => None,
        };
        if family != run_family {
            span(&mut spans, &text[run_start..i], run_family);
            (run_start, run_family) = (i, family);
        }
    }
    span(&mut spans, &text[run_start..], run_family);
    buffer.set_rich_text(spans, attrs, Shaping::Advanced, None);
}

impl TextRenderer {
    pub fn new() -> Self {
        // Build the FontSystem from an EXPLICIT db + our BundledFallback — NOT `FontSystem::new()`, whose `PlatformFallback` diverges per-OS (see BundledFallback). Locale is fixed too: cosmic-text only uses it to pick han-unification CJK order, which our bundled set doesn't cover, so a constant keeps rendering identical regardless of the host's `LANG`.
        let mut db = Database::new();

        // Open Sans, VARIABLE (wght 300..800, wdth 75..100) — the primary Latin family, every weight a real one (Nick 2026-10-14, "get it all variable"). It replaced the static Regular + Bold, between which every 500 and 600 a consumer asked for used to snap to 400 or 700.
        db.load_font_data(include_bytes!("../assets/Open_Sans/OpenSans-VariableFont_wdth,wght.ttf").to_vec());
        // Noto Sans Symbols 2 — the deterministic MONOCHROME symbol fallback. Covers hearts ♥♡, stars, checks, weather, geometric shapes, arrows, dingbats, box-drawing, and math that Open Sans lacks. Swash rasterizes these as a coverage Mask (tinted with the style colour like any glyph).
        db.load_font_data(
            include_bytes!("../assets/Noto_Sans_Symbols2/NotoSansSymbols2-Regular.ttf").to_vec(),
        );
        // Noto Color Emoji — the deterministic COLOUR-emoji fallback (CBDT/CBLC bitmap strikes, family name "Noto Color Emoji"). swash returns these as `Content::Color` RGBA images (not a coverage mask), and the blit's Color branch composites the glyph's OWN sRGB pixels into the darkness buffer, ignoring the style colour — snowman ☃, faces, hands, flags, ZWJ families, and skin tones show in true colour. The old "doesn't rasterize cleanly" note is obsolete now the Color path exists. ~10.7 MB embedded so a fluor build renders emoji with NO host-font read.
        // Gated behind the `emoji` feature: the font is 11 MB, which pushes the crate tarball over crates.io's 10 MiB limit, so it is excluded from the published package (see `exclude` in Cargo.toml). Enable `emoji` for a git/local checkout that has assets/NotoColorEmoji.ttf present.
        #[cfg(feature = "emoji")]
        db.load_font_data(include_bytes!("../assets/NotoColorEmoji.ttf").to_vec());

        db.set_sans_serif_family("Open Sans");

        let font_system = FontSystem::new_with_locale_and_db_and_fallback(
            "en-US".to_string(),
            db,
            BundledFallback,
        );

        Self {
            font_system,
            swash_cache: SwashCache::new(),
            scale_context: ScaleContext::new(),
            // Cap at 1024 glyph-images. At ~256 visible glyphs × 4 active rotation bins it never trips; bigger workloads tail-evict the coldest entries first.
            transform_cache: TransformGlyphCache::new(1024),
            optical: false,
            packs_checked: 0,
            packs_loaded: 0,
            packs_wanted: 0,
            packs_offered: 0,
            han: crate::packs::Han::Simplified,
            fonts_version: 0,
        }
    }

    /// The packs this text needs, noted as wanted where none is loaded; returns the CJK face its CJK runs draw in (kana means Japanese, Hangul Korean, Han alone the renderer's default), if it has any.
    fn note_packs(&mut self, text: &str) -> Option<&'static str> {
        use crate::packs::{Han, Need, PACKS, need};
        if text.is_ascii() {
            return None;
        }
        let (mut kana, mut hangul, mut shared) = (false, false, false);
        for c in text.chars() {
            match need(c) {
                Some(Need::Script(i)) => self.need_pack(i),
                Some(Need::Kana) => kana = true,
                Some(Need::Hangul) => hangul = true,
                Some(Need::Shared) => shared = true,
                None => {}
            }
        }
        if !(kana || hangul || shared) {
            return None;
        }
        let region = if kana { Han::Japanese } else if hangul { Han::Korean } else { self.han };
        let i = region.pack();
        self.need_pack(i);
        Some(PACKS[i].family)
    }

    /// One pack noted as needed: looked for in the db the first time (a face an app bundled counts as loaded), wanted if absent.
    fn need_pack(&mut self, i: usize) {
        let bit = 1u64 << i;
        if self.packs_loaded & bit != 0 {
            return;
        }
        if self.packs_checked & bit == 0 {
            self.packs_checked |= bit;
            let family = crate::packs::PACKS[i].family;
            if self.font_system.db().faces().any(|f| f.families.iter().any(|(n, _)| n == family)) {
                self.packs_loaded |= bit;
                return;
            }
        }
        self.packs_wanted |= bit;
    }

    /// Packs text has needed since the last call and none is loaded for, each offered once: the app fetches them and hands each to [`TextRenderer::load_pack`] (or [`TextRenderer::add_pack_face`] for a face it checked itself).
    pub fn wanted_packs(&mut self) -> Vec<&'static crate::packs::Pack> {
        let new = self.packs_wanted & !self.packs_loaded & !self.packs_offered;
        self.packs_offered |= new;
        (0..crate::packs::PACKS.len()).filter(|i| new & (1u64 << i) != 0).map(|i| &crate::packs::PACKS[i]).collect()
    }

    /// Offer a pack again (its fetch failed): the next text needing it brings it back through [`TextRenderer::wanted_packs`].
    pub fn offer_again(&mut self, id: &str) {
        if let Some(i) = crate::packs::index(id) {
            self.packs_offered &= !(1u64 << i);
        }
    }

    /// The regional face for Han text with no kana or Hangul in it (Simplified Chinese until set).
    pub fn set_han(&mut self, region: crate::packs::Han) {
        self.han = region;
    }

    /// Bumped whenever a face is added: anything that measured text before measures again.
    pub fn fonts_version(&self) -> u64 {
        self.fonts_version
    }

    /// A pack's face added, checked by the caller already: it must declare the pack's family. The next frame draws that script.
    pub fn add_pack_face(&mut self, id: &str, face: Vec<u8>) -> Result<(), String> {
        let i = crate::packs::index(id).ok_or_else(|| format!("unknown font pack {id}"))?;
        let family = crate::packs::PACKS[i].family;
        let ids = self.font_system.db_mut().load_font_source(cosmic_text::fontdb::Source::Binary(std::sync::Arc::new(face)));
        let db = self.font_system.db();
        if !ids.iter().any(|id| db.face(*id).is_some_and(|f| f.families.iter().any(|(n, _)| n == family))) {
            for id in ids {
                self.font_system.db_mut().remove_face(id);
            }
            return Err(format!("font pack {id}: the face is not {family}"));
        }
        self.packs_loaded |= 1u64 << i;
        self.packs_checked |= 1u64 << i;
        self.fonts_version += 1;
        Ok(())
    }

    /// A signed pack document checked against the author's key and its face added ([`crate::packs::read`]); returns the pack.
    #[cfg(feature = "packs")]
    pub fn load_pack(&mut self, doc: &[u8], author: [u8; 32]) -> Result<&'static crate::packs::Pack, String> {
        let (i, face) = crate::packs::read(doc, author)?;
        let pack = &crate::packs::PACKS[i];
        self.add_pack_face(pack.id, face)?;
        Ok(pack)
    }

    /// OPTICAL WEIGHT (Nick 2026-10-14, with variable fonts): the weight a style asks for is the weight at 16 px; smaller text draws heavier and larger text lighter, so a 9 px label and a 40 px name read as the same weight — a thin stem under a pixel wide renders as grey, a heavy stem at display size reads as shouting. 4 weight units per pixel away from 16, at most +60 (small) and −40 (large), within 100..900. A variable face draws the result exactly; a static face snaps it to its nearest cut. Off by default, so a consumer opts in.
    pub fn set_optical_weight(&mut self, on: bool) {
        self.optical = on;
    }

    /// The weight actually drawn for a style weight at `size` px (see [`set_optical_weight`](Self::set_optical_weight)). Measuring and drawing both go through it, so a measured width is the drawn width.
    pub fn optical_weight(&self, weight: u16, size: f32) -> u16 {
        if !self.optical || !size.is_finite() {
            return weight;
        }
        // The size → weight lift is the old linear ramp `(16 − size) × 4` eased into its old limits (−40 and +60) by a tanh centred on their midpoint: C∞ where the clamp had kinks at 26 px and 1 px, and within about two weight units of the ramp across 12–20 px.
        let d = 10.0 + 50.0 * libm::tanhf(((16.0 - size) * 4.0 - 10.0) / 50.0);
        // The u16 cast saturates; a weight past 100..=900 is bounded by the variable font's own wght axis at the scaler (and matched to the nearest face otherwise).
        (weight as f32 + d).round() as u16
    }

    /// Mutable access to the underlying `FontSystem` so consumers can load additional fonts via `text.font_system_mut().db_mut().load_font_data(...)`.
    pub fn font_system_mut(&mut self) -> &mut FontSystem {
        &mut self.font_system
    }

    /// Load a font from raw bytes and return its primary family name, for use as the `font: &str`
    /// argument to the `draw_text_*` methods. Consumers that ship their own font (e.g. a capsule carrying a monospace face) call this once and cache the returned family name. Returns `None`
    /// if the bytes register no face.
    pub fn load_font_data_named(&mut self, data: Vec<u8>) -> Option<String> {
        let db = self.font_system.db_mut();
        db.load_font_data(data);
        db.faces()
            .last()
            .and_then(|f| f.families.first().map(|(name, _)| name.clone()))
    }


    /// Draw left-aligned styled text. `x, y` = left edge, vertical center. Returns the drawn width. The whole "how" lives in [`TextStyle`]; clip and mask stay positional because they're render-target concerns, not style.
    pub fn draw_text_left<P: Pixel>(&mut self, canvas: &mut Canvas<P>, text: &str, x: f32, y: f32, style: &TextStyle, clip: Option<Clip>, mask: Option<&AlphaMask>) -> f32 {
        self.draw_text_left_u32(canvas, text, x, y, style.size, style.weight, style.colour, style.font, clip, mask, style.transform(), style.shear)
    }
    /// Draw center-aligned styled text (`x` = center).
    pub fn draw_text_center<P: Pixel>(&mut self, canvas: &mut Canvas<P>, text: &str, x: f32, y: f32, style: &TextStyle, clip: Option<Clip>, mask: Option<&AlphaMask>) -> f32 {
        self.draw_text_center_u32(canvas, text, x, y, style.size, style.weight, style.colour, style.font, clip, mask, style.transform(), style.shear)
    }
    /// Draw right-aligned styled text (`x` = right edge).
    pub fn draw_text_right<P: Pixel>(&mut self, canvas: &mut Canvas<P>, text: &str, x: f32, y: f32, style: &TextStyle, clip: Option<Clip>, mask: Option<&AlphaMask>) -> f32 {
        self.draw_text_right_u32(canvas, text, x, y, style.size, style.weight, style.colour, style.font, clip, mask, style.transform(), style.shear)
    }
    /// Measure styled text width: summed advances × stretch (skew and rotation don't change advances). Build ONE style and share it between measure and draw — that's the intended pattern.
    pub fn measure_text(&mut self, text: &str, style: &TextStyle) -> f32 {
        self.measure_text_width_impl(text, style.size, style.weight, style.font) * style.stretch
    }

    fn draw_text_center_u32<P: Pixel>(
        &mut self,
        canvas: &mut Canvas<P>,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: u32,
        font: &str,
        clip: Option<Clip>,
        mask: Option<&AlphaMask>,
        transform: Option<Transform>,
        shear: f32,
    ) -> f32 {
        // Clip cull BEFORE shaping (photon 2026-10-06, scroll as a memmove): shaping is the cost of a text draw, and a clip that excludes the whole line box — an empty clip, or one whose rows lie entirely above or below `y ± 1.2·size` (y is the box centre; the margin covers ascender/descender overhang) — means nothing would land anyway. Returns 0.0: a caller laying out BY the returned width is laying out inside a culled band, where nothing paints either.
        if let Some(c) = clip {
            if c.x_end <= c.x_start || c.y_end <= c.y_start || y + size * 1.2 < c.y_start as f32 || y - size * 1.2 > c.y_end as f32 {
                return 0.0;
            }
        }
        let buf_w = canvas.width;
        let buf_h = canvas.height;
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));
        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            for glyph in run.glyphs {
                min_x = min_x.min(glyph.x);
                max_x = max_x.max(glyph.x + glyph.w);
            }
            let text_width = max_x - min_x;
            let text_height = run.line_height;
            damage_text_bbox(
                canvas.damage,
                buf_w,
                buf_h,
                x,
                y,
                text_width,
                text_height,
                0,
            );
            self.render_buffer_u32(
                &mut buffer,
                canvas.pixels,
                buf_w,
                buf_h,
                x,
                y,
                text_width,
                text_height,
                colour,
                0,
                clip,
                mask,
                transform,
                shear,
            );
            text_width
        } else {
            0.
        }
    }

    fn draw_text_left_u32<P: Pixel>(
        &mut self,
        canvas: &mut Canvas<P>,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: u32,
        font: &str,
        clip: Option<Clip>,
        mask: Option<&AlphaMask>,
        transform: Option<Transform>,
        shear: f32,
    ) -> f32 {
        // Clip cull BEFORE shaping (photon 2026-10-06, scroll as a memmove): shaping is the cost of a text draw, and a clip that excludes the whole line box — an empty clip, or one whose rows lie entirely above or below `y ± 1.2·size` (y is the box centre; the margin covers ascender/descender overhang) — means nothing would land anyway. Returns 0.0: a caller laying out BY the returned width is laying out inside a culled band, where nothing paints either.
        if let Some(c) = clip {
            if c.x_end <= c.x_start || c.y_end <= c.y_start || y + size * 1.2 < c.y_start as f32 || y - size * 1.2 > c.y_end as f32 {
                return 0.0;
            }
        }
        let buf_w = canvas.width;
        let buf_h = canvas.height;
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));
        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;
            damage_text_bbox(
                canvas.damage,
                buf_w,
                buf_h,
                x,
                y,
                text_width,
                text_height,
                1,
            );
            self.render_buffer_u32(
                &mut buffer,
                canvas.pixels,
                buf_w,
                buf_h,
                x,
                y,
                text_width,
                text_height,
                colour,
                1,
                clip,
                mask,
                transform,
                shear,
            );
            text_width
        } else {
            0.
        }
    }

    fn draw_text_right_u32<P: Pixel>(
        &mut self,
        canvas: &mut Canvas<P>,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: u32,
        font: &str,
        clip: Option<Clip>,
        mask: Option<&AlphaMask>,
        transform: Option<Transform>,
        shear: f32,
    ) -> f32 {
        // Clip cull BEFORE shaping (photon 2026-10-06, scroll as a memmove): shaping is the cost of a text draw, and a clip that excludes the whole line box — an empty clip, or one whose rows lie entirely above or below `y ± 1.2·size` (y is the box centre; the margin covers ascender/descender overhang) — means nothing would land anyway. Returns 0.0: a caller laying out BY the returned width is laying out inside a culled band, where nothing paints either.
        if let Some(c) = clip {
            if c.x_end <= c.x_start || c.y_end <= c.y_start || y + size * 1.2 < c.y_start as f32 || y - size * 1.2 > c.y_end as f32 {
                return 0.0;
            }
        }
        let buf_w = canvas.width;
        let buf_h = canvas.height;
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));
        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);
        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;
            damage_text_bbox(
                canvas.damage,
                buf_w,
                buf_h,
                x,
                y,
                text_width,
                text_height,
                2,
            );
            self.render_buffer_u32(
                &mut buffer,
                canvas.pixels,
                buf_w,
                buf_h,
                x,
                y,
                text_width,
                text_height,
                colour,
                2,
                clip,
                mask,
                transform,
                shear,
            );
            text_width
        } else {
            0.
        }
    }

    /// Render shaped text into a u32 ARGB pixel buffer with explicit clip + optional mask.
    ///
    /// **Per-glyph entry-point clamp:** the glyph's natural pixel rect `[glyph_x, glyph_x + glyph_w) × [glyph_y, glyph_y + glyph_h)` is intersected with `clip` once at the start of each glyph. The `cx`/`cy` inner-loop ranges are then pre-clamped so the loop body carries **no per-pixel bounds checks** — the math at entry is the proof. Photon's old per-pixel `if final_x < 0 || final_y >= height` is gone.
    ///
    /// **Mask:** if `Some(&AlphaMask)`, `mask.pixels[idx]` multiplies into the glyph alpha (effective_alpha = glyph_alpha × mask_alpha / 256). Used for soft-clipping text by textbox shape, scroll fade, etc.
    fn render_buffer_u32<P: Pixel>(
        &mut self,
        buffer: &mut Buffer,
        pixels: &mut [P],
        buf_w: usize,
        buf_h: usize,
        anchor_x: f32,
        anchor_y: f32,
        text_width: f32,
        text_height: f32,
        colour: u32,
        alignment: u8, // 0=center, 1=left, 2=right
        clip: Option<Clip>,
        mask: Option<&AlphaMask>,
        transform: Option<Transform>,
        shear: f32,
    ) {
        let clip = Clip::resolve(clip, buf_w, buf_h);
        if let Some(m) = mask {
            crate::paint::assert_mask_matches_buffer(m, buf_w, buf_h);
        }

        let (offset_x, offset_y) = match alignment {
            0 => (anchor_x - text_width / 2., anchor_y - text_height / 2.),
            1 => (anchor_x, anchor_y - text_height / 2.),
            2 => (anchor_x - text_width, anchor_y - text_height / 2.),
            _ => (anchor_x, anchor_y),
        };

        // Topmost-first: split colour into its α channel (combined with per-pixel glyph coverage and optional mask to form the glyph pixel's effective α) and its RGB-darkness payload (passed straight thru to under()). Replaces the legacy SWAR over-blend — text now composites with the same under() kernel as every other rasterizer.
        // The style colour is authored 8-bit `g2t`; widened once here to the buffer's depth.
        let colour = P::from_g2t8(colour);
        let colour_alpha = colour.alpha();
        let colour_dark = colour.rgb();

        // Identity-or-None: cosmic-text's SwashCache fast path. Non-identity transform splits two ways:
        //   1. **Contour transform** (linear part only — `a, b, c, d`, *no translation*) → fed to swash::scale to rotate/skew/scale the glyph outline in font space. Translation is intentionally stripped here because swash applies its transform to the contour in glyph-local coords (origin at 0,0 baseline-left); leaving translation in would translate the contour itself, blowing the placement bbox out into absolute pixel space. The linear part is *snapped* per-glyph to the 1-pixel-arc rotation grid via [`crate::paint::snap_rotation`], so consecutive frames within the same rotation bin reuse the same rasterized glyph (cache hit). Snapping the contour but **not** the position lets the run slide smoothly along its arc while each glyph's orientation steps in cache-friendly bins.
        //   2. **Position transform** (full transform — linear *plus* translation, *unsnapped*) → applied to each glyph's run-local position, so the run translates + rotates as a rigid body around the chosen anchor with sub-bin precision.
        // swash's Transform constructor takes (xx, xy, yx, yy, x, y) where the matrix is `[xx xy x; yx yy y]`. Our `Transform` stores `[a c tx; b d ty]` — same layout, just renamed: xx=a, xy=c, yx=b, yy=d, x=tx, y=ty.
        let active_transform = if shear != 0. {
            // Shear-only text must still route thru the contour machinery (the identity fast path would drop it).
            Some(transform.unwrap_or(Transform::IDENTITY))
        } else {
            transform.filter(|t| !t.is_identity())
        };
        // Continuous rotation angle of the linear part (radians). atan2 of the rotated +X axis. For pure rotation `R(θ)` this returns θ exactly; for rotation+uniform-scale the magnitude factors out; for skew/non-uniform scale this picks the angle of the first column as the "rotation component" — fine for our text use case (titles rotate, scale stays uniform).
        let base_theta = active_transform.map(|t| crate::math::atan2(t.b, t.a));

        for run in buffer.layout_runs() {
            let baseline_offset = run.line_y;
            for glyph in run.glyphs {
                let physical_glyph = glyph.physical((offset_x, offset_y), 1.);

                // Per-glyph snapped contour matrix. Snap radius is `glyph.font_size / 2` (the 1-pixel-arc rule, evaluated at the font size of *this* glyph since runs may mix sizes). Compose `R(δ) ∘ M_linear` where δ = θ_snapped − θ_actual: this rotates the original linear matrix forward by the snap delta, preserving any scale/skew while landing the rotation component on a bin boundary.
                let contour_transform = match (active_transform, base_theta) {
                    (Some(t), Some(theta)) => {
                        let theta_snapped = crate::paint::snap_rotation(theta, glyph.font_size, 8);
                        let delta = theta_snapped - theta;
                        if delta == 0.0 {
                            Some(Transform {
                                a: t.a,
                                b: t.b,
                                c: t.c,
                                d: t.d,
                                tx: 0.0,
                                ty: 0.0,
                            })
                        } else {
                            let linear_only = Transform {
                                a: t.a,
                                b: t.b,
                                c: t.c,
                                d: t.d,
                                tx: 0.0,
                                ty: 0.0,
                            };
                            Some(linear_only.then(Transform::rotate(delta)))
                        }
                    }
                    _ => None,
                };
                // The shear folds in HERE — contour-only, post-snap, BEFORE the cache key is taken (sheared and upright glyphs must never share an entry). It deliberately never enters `active_transform`: positions would drift (∝ shear·x) and base_theta's snapper would read shear as rotation. Slot empirics (2026-07-17): the c-slot renders as VERTICAL shear thru swash — the layout comment above notwithstanding; rotations can't distinguish a transposition, shears can — so the horizontal-oblique coefficient rides the b-slot.
                let contour_transform = contour_transform.map(|t| {
                    if shear != 0. {
                        Transform::new(1., shear, 0., 1., 0., 0.).then(t)
                    } else {
                        t
                    }
                });
                let path_transform =
                    contour_transform.map(|t| PathTransform::new(t.a, t.c, t.b, t.d, 0.0, 0.0));

                // `glyph_is_colour` distinguishes a `Content::Color` RGBA bitmap (colour emoji — 4 bytes/pixel, composite its own sRGB) from a `Content::Mask` coverage image (every other glyph — 1 byte/pixel, tint with the style colour). Carried out of the match so the shared blit below branches on it once.
                let (glyph_data, glyph_is_colour, placement_left, placement_top, placement_w, placement_h) =
                    match path_transform {
                        None => {
                            // Identity fast path: cosmic-text's SwashCache owns the rasterized-image cache for non-transformed glyphs.
                            let Some(image) = self
                                .swash_cache
                                .get_image(&mut self.font_system, physical_glyph.cache_key)
                            else {
                                continue;
                            };
                            let p = image.placement;
                            let is_colour = image.content == swash::scale::image::Content::Color;
                            (&image.data[..], is_colour, p.left, p.top, p.width, p.height)
                        }
                        Some(zt) => {
                            // Transform path: check our own LRU cache first; on miss, rasterize via swash directly (contour transformed in font space → proper hinting + AA), then insert at front. Cache key uses the *snapped* linear part so consecutive frames in the same rotation bin hit.
                            let t = contour_transform
                                .expect("path_transform set implies contour_transform set");
                            let key = TransformGlyphKey {
                                font_id: glyph.font_id,
                                weight: glyph.font_weight.0,
                                glyph_id: glyph.glyph_id,
                                size_bits: glyph.font_size.to_bits(),
                                a_bits: t.a.to_bits(),
                                b_bits: t.b.to_bits(),
                                c_bits: t.c.to_bits(),
                                d_bits: t.d.to_bits(),
                            };
                            if self.transform_cache.lookup(key).is_none() {
                                let Some(font) = self.font_system.get_font(glyph.font_id, glyph.font_weight) else {
                                    continue;
                                };
                                let image = {
                                    let swash_font = font.as_swash();
                                    let mut scaler = self.scale_context.builder(swash_font).size(glyph.font_size);
                                    // A variable font drawn at the glyph's weight on its wght axis, as cosmic-text's own cache draws the upright glyphs (Nick 2026-10-14: variable weights in the render, no split fonts).
                                    let wght = swash::Tag::from_be_bytes(*b"wght");
                                    if let Some(axis) = swash_font.variations().find_by_tag(wght) {
                                        // WHY/PROOF: the axis range is read from the font file, and a weight past it is outside what the face defines — swash's variation coords would extrapolate the outline.
                                        let w = f32::from(glyph.font_weight.0).clamp(axis.min_value(), axis.max_value());
                                        scaler = scaler.normalized_coords(swash_font.variations().normalized_coords([(wght, w)]));
                                    }
                                    let mut scaler = scaler.build();
                                    // Request the colour-bitmap and colour-outline sources ALONGSIDE the outline so a rotated colour emoji still produces a `Content::Color` image (swash picks the first source the face supports; Open Sans/Symbols glyphs fall thru to Outline as before). The swash transform rotates the colour strike in place, so rotated emoji show in true colour rather than being skipped.
                                    let Some(img) = Render::new(&[
                                        Source::ColorBitmap(swash::scale::StrikeWith::BestFit),
                                        Source::ColorOutline(0),
                                        Source::Outline,
                                    ])
                                        .transform(Some(zt))
                                        .render(&mut scaler, glyph.glyph_id)
                                    else {
                                        continue;
                                    };
                                    img
                                };
                                self.transform_cache.insert(key, image);
                            }
                            // Cache front entry is the one we just looked up or inserted.
                            let entry = &self.transform_cache.entries[0];
                            let p = entry.1.placement;
                            let is_colour = entry.1.content == swash::scale::image::Content::Color;
                            (&entry.1.data[..], is_colour, p.left, p.top, p.width, p.height)
                        }
                    };

                // Glyph origin: cosmic-text gives integer (px, py) at the run-local position. Under transform we rotate that position too so the run reads as one rotated piece.
                let (origin_x, origin_y) = match active_transform {
                    Some(t) => {
                        let (rx, ry) = t.apply(
                            physical_glyph.x as f32,
                            (physical_glyph.y + baseline_offset as i32) as f32,
                        );
                        (rx as i32, ry as i32)
                    }
                    None => (physical_glyph.x, physical_glyph.y + baseline_offset as i32),
                };
                let glyph_x = origin_x + placement_left;
                let glyph_y = origin_y - placement_top;
                let gw = placement_w as i32;
                let gh = placement_h as i32;

                // Per-glyph clamp: intersect glyph rect with clip → cx/cy ranges that are guaranteed in-bounds for both image.data[] and pixels[] indexing, no per-pixel checks.
                let cx_min = (clip.x_start as i32 - glyph_x).max(0).min(gw) as usize;
                let cy_min = (clip.y_start as i32 - glyph_y).max(0).min(gh) as usize;
                let cx_max = (clip.x_end as i32 - glyph_x).max(0).min(gw) as usize;
                let cy_max = (clip.y_end as i32 - glyph_y).max(0).min(gh) as usize;
                if cx_min >= cx_max || cy_min >= cy_max {
                    continue;
                }

                let gw_us = gw as usize;
                if glyph_is_colour {
                    // COLOUR-EMOJI blit. `glyph_data` is RGBA, 4 bytes/pixel (swash `Content::Color` is straight/unpremultiplied sRGB — verified against swash's CBDT/CBLC decoder, which copies the strike's RGBA through without premultiply). The glyph carries its OWN colour, so the style `colour` is IGNORED for the RGB payload; only its α (`colour_alpha`) and the optional mask still modulate opacity, so a fading-out toast fades the emoji too.
                    // COLOUR DOCTRINE (photon): this bitmap is sRGB. fluor emits visible-sRGB and photon's theme.rs `to_display` owns the sRGB→BT.2020 edge on every platform, so we do NOT colour-convert here — we just land true sRGB into the darkness buffer exactly as fluor does for any other RGB it rasterizes. Darkness = 255 − channel per the crate's α+darkness convention (see pixel.rs); after finalize's `^ 0x00FFFFFF` the emoji reappears in true colour.
                    for cy in cy_min..cy_max {
                        let final_y = (glyph_y + cy as i32) as usize;
                        let row_offset_buf = final_y * buf_w;
                        let row_offset_glyph = (cy * gw_us) * 4;
                        for cx in cx_min..cx_max {
                            let px = row_offset_glyph + cx * 4;
                            let sa = glyph_data[px + 3];
                            if sa == 0 {
                                continue;
                            }
                            let final_x = (glyph_x + cx as i32) as usize;
                            let idx = row_offset_buf + final_x;

                            let mask_alpha = match mask {
                                Some(m) => m.pixels[idx] as u32,
                                None => 0xFF,
                            };
                            // Combine the glyph's own α with the style α and the mask, same product the Mask path uses (just with `sa` in place of the coverage byte).
                            // Glyph and mask are bytes (÷ 255 each) and the style α is the depth's own, so the result is in the depth's scale; at 16 bits the product peaks at 255 · 65535 · 255 < 2³², still a u32.
                            let effective_alpha =
                                (sa as u32 * colour_alpha * mask_alpha) / (255 * 255);
                            if effective_alpha == 0 {
                                continue;
                            }
                            // Per-pixel darkness payload = bitwise-complement of the source's visible sRGB. `under()` reads these RGB bytes straight thru as the "bottom" darkness (BlendMode::Normal), identical to how `colour_dark` feeds the Mask path — so the domain matches exactly.
                            let dark_r = 255 - glyph_data[px] as u32;
                            let dark_g = 255 - glyph_data[px + 1] as u32;
                            let dark_b = 255 - glyph_data[px + 2] as u32;
                            // Platform channel order via theme::fmt (identity on desktop, R↔B on Android) — every colour that reaches the buffer rides fmt; the emoji payload skipping it was the "R and B emoji colours appear swapped" Android bug (same class as the old swapped ring). fmt reorders only the RGB bytes and preserves α, so it's correct on the darkness-packed value.
                            // The bitmap is γ2-ish display bytes; the buffer is `g2t`, so the darkness takes the same light's code.
                            let glyph_pixel = P::from_g2t8(crate::theme::darkness_from_gamma2(crate::theme::fmt((dark_r << 16) | (dark_g << 8) | dark_b)))
                                .with_alpha(effective_alpha);
                            pixels[idx] = pixels[idx].under(glyph_pixel, BlendMode::Normal);
                        }
                    }
                } else {
                    for cy in cy_min..cy_max {
                        let final_y = (glyph_y + cy as i32) as usize;
                        let row_offset_buf = final_y * buf_w;
                        let row_offset_glyph = cy * gw_us;
                        for cx in cx_min..cx_max {
                            let alpha = glyph_data[row_offset_glyph + cx];
                            if alpha == 0 {
                                continue;
                            }
                            let final_x = (glyph_x + cx as i32) as usize;
                            let idx = row_offset_buf + final_x;

                            let mask_alpha = match mask {
                                Some(m) => m.pixels[idx] as u32,
                                None => 0xFF,
                            };
                            let effective_alpha =
                                (alpha as u32 * colour_alpha * mask_alpha) / (255 * 255);
                            if effective_alpha == 0 {
                                continue;
                            }
                            let glyph_pixel = colour_dark.with_alpha(effective_alpha);
                            pixels[idx] = pixels[idx].under(glyph_pixel, BlendMode::Normal);
                        }
                    }
                }
            }
        }
    }

    /// Measure the width of text without rendering it
    fn measure_text_width_impl(&mut self, text: &str, size: f32, weight: u16, font: &str) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size));
        buffer.set_size(Some(10000.0), Some(size * 2.0));
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        // cosmic-text 0.19 shapes on demand: lay the text out before reading its runs.
        buffer.shape_until_scroll(&mut self.font_system, false);

        // Sum each glyph's ADVANCE (glyph.w), don't take max(glyph.x + glyph.w). A trailing space has zero visual extent, so cosmic-text gives it glyph.x = glyph.w = 0 when it's the last (or only)
        // glyph in the run — max(x+w) then returns 0, so a lone " " measures 0 and words that end in a space collapse into the next (text looks truncated / spaces vanish). Summing advances counts the space's advance regardless of position, matching `measure_text_widths_per_char` below.
        buffer.layout_runs().fold(0.0, |max_width, run| {
            let run_width = run.glyphs.iter().fold(0.0, |w, glyph| w + glyph.w);
            max_width.max(run_width)
        })
    }

    /// Per-char widths for the entire string, computed by shaping ONCE and attributing each glyph's advance back to its source char via `glyph.start` (byte offset). For 1-glyph-per-char scripts (typical Latin), the result is exact. For ligatures (e.g. "fi" → 1 glyph), the ligature width gets attributed to the FIRST source char and subsequent chars in the ligature get 0 — approximate but stable for cursor positioning. Massively faster than per-char `measure_text_width` calls when the input is long: one cosmic-text shape pass instead of N.
    ///
    /// Returns a Vec sized `text.chars().count()`. Each entry is the advance contribution of all glyphs originating from that char.
    pub fn measure_text_widths_per_char(
        &mut self,
        text: &str,
        size: f32,
        weight: u16,
        font: &str,
    ) -> alloc::vec::Vec<f32> {
        let char_count = text.chars().count();
        let mut widths = alloc::vec![0.0f32; char_count];
        if text.is_empty() {
            return widths;
        }
        // Build (byte_offset, char_idx) pairs in increasing byte order — for pure ASCII this is the identity; for multi-byte UTF-8 it remaps each glyph.start (byte offset) back to its char_idx.
        let mut byte_to_char: alloc::vec::Vec<(usize, usize)> =
            alloc::vec::Vec::with_capacity(char_count);
        for (char_idx, (byte_idx, _ch)) in text.char_indices().enumerate() {
            byte_to_char.push((byte_idx, char_idx));
        }
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));
        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size));
        buffer.set_size(Some(10000.0), Some(size * 2.0));
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        // cosmic-text 0.19 shapes on demand: lay the text out before reading its runs.
        buffer.shape_until_scroll(&mut self.font_system, false);
        for run in buffer.layout_runs() {
            for glyph in run.glyphs {
                // Binary search for the largest byte_idx <= glyph.start: that's the char whose UTF-8 encoding starts at or before this glyph's source byte.
                let byte_start = glyph.start;
                let char_idx = match byte_to_char.binary_search_by(|&(b, _)| b.cmp(&byte_start)) {
                    Ok(i) => byte_to_char[i].1,
                    Err(i) if i > 0 => byte_to_char[i - 1].1,
                    Err(_) => continue,
                };
                // char_idx comes from this string's own char_indices, so it is < char_count = widths.len().
                widths[char_idx] += glyph.w;
            }
        }
        widths
    }

}

#[cfg(test)]
mod colour_emoji_tests {
    use super::*;
    use crate::canvas::{Canvas, Damage};

    /// Finalize-domain read: fluor stores α+darkness; the host applies `^ 0x00FFFFFF` at present to flip darkness → visible RGB. Mirror that here so the test reasons in the visible colours a screen would show.
    fn visible_rgb(pixel: u32) -> (u8, u8, u8) {
        let v = pixel ^ 0x00FF_FFFF;
        (((v >> 16) & 0xFF) as u8, ((v >> 8) & 0xFF) as u8, (v & 0xFF) as u8)
    }

    /// A colour emoji ("hi ☃😀") must (a) not panic, and (b) deposit at least one NON-GREYSCALE pixel (r≠g or g≠b) — proving it composited the glyph's own colour, not a monochrome tint. Gated on the emoji feature: without the Noto Color Emoji bundle the fallback renders monochrome BY DESIGN and this assertion is testing a font that isn't there.
    #[cfg(feature = "emoji")]
    #[test]
    fn colour_emoji_renders_non_greyscale() {
        let mut text = TextRenderer::new();
        let w = 128usize;
        let h = 48usize;
        let mut pixels = vec![0u32; w * h];
        let mut damage = Damage::new();
        let mut canvas = Canvas::new(&mut pixels, w, h, &mut damage);

        // Opaque BLACK text tint (α=0xFF, darkness=0xFFFFFF). Any monochrome glyph tinted with this stays r=g=b in visible space, so a non-greyscale pixel can ONLY come from the colour-emoji Color path.
        let style = TextStyle::new(28.0, 0xFFFF_FFFF);
        text.draw_text_left(&mut canvas, "hi \u{2603}\u{1F600}", 2.0, 24.0, &style, None, None);

        let mut found_colour = false;
        for &p in pixels.iter() {
            let (r, g, b) = visible_rgb(p);
            if r != g || g != b {
                found_colour = true;
                break;
            }
        }
        assert!(found_colour, "expected at least one non-greyscale (colour-emoji) pixel; got an all-greyscale buffer");
    }

    /// Pure text must render IDENTICALLY across two renders (the Mask path is untouched) AND stay fully greyscale (a monochrome tint never introduces colour). Guards the Color branch against leaking into the text path.
    #[test]
    fn pure_text_is_greyscale_and_stable() {
        let render = |s: &str| -> Vec<u32> {
            let mut text = TextRenderer::new();
            let w = 96usize;
            let h = 32usize;
            let mut pixels = vec![0u32; w * h];
            let mut damage = Damage::new();
            let mut canvas = Canvas::new(&mut pixels, w, h, &mut damage);
            let style = TextStyle::new(22.0, 0xFFFF_FFFF);
            text.draw_text_left(&mut canvas, s, 2.0, 16.0, &style, None, None);
            pixels
        };

        let a = render("hello world");
        let b = render("hello world");
        assert_eq!(a, b, "pure-text render must be deterministic (Mask path unchanged)");

        for &p in a.iter() {
            let (r, g, b) = visible_rgb(p);
            assert!(r == g && g == b, "pure text tinted black must be greyscale; found ({r},{g},{b})");
        }
    }
}
