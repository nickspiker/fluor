//! Pixel-buffer paint primitives. Packed layout is `0xααRRGGBB` (α-byte high, blue low) — top byte is **α (opacity)**, industry-standard direction (`α = 0` transparent, `α = 0xFF` opaque). RGB bytes store darkness (`0 = white`, `255 = black`). See [`crate::pixel`] for the locked convention. All inputs are pixel-space, not RU — convert via [`Viewport::ru_to_px`](crate::Viewport::ru_to_px) before calling.
//!
//! Internal to fluor's render pipeline. Per `## API / Implementation Separation` in AGENT.md, these are not part of the consumer-facing API: future SIMD kernels (NEON, SSE2) will dispatch thru the same entry points without changing call sites in `pane` or `Compositor`.
//!
//! Blend model is α + darkness front-to-back: `dst` is the partial composite already accumulated above (its α-byte = accumulated opacity, RGB = accumulated darkness), `src` is the new layer going behind. Per-pixel early-out fires when `dst >= 0xFF000000` (dst α saturated = opaque) via a single u32 compare. Math thruout is `>> 8` with the `(256 − top_α)` trick — never `/ 255`, never floats in the inner loop. Multi-layer composition is additive on BOTH halves (α adds, darkness adds); the buffer carries the accumulator state across Group boundaries so the early-out chain survives between flatten passes.
//!
//! Every blending primitive accepts an optional [`Clip`] (defaults to full buffer when `None`) and an optional [`AlphaMask`] (full-frame, multiplies into per-pixel alpha for soft clipping — rounded textboxes, squircle pane corners, scroll fades). The clip is resolved once at entry into `(x_min, y_min, x_max, y_max)` loop bounds, so the inner loops carry **zero per-pixel bounds checks** — the math at the entry is the proof. AlphaMask dimensions must equal the buffer's `(buf_w, buf_h)`; mismatches panic per AGENT.md "fail loud."

use crate::canvas::Canvas;
use crate::coord::Coord;

// Submodules — extracted from this file as part of the paint-organisation pass. Each is re-exported flat so existing call sites (`crate::paint::draw_squircle_pill`, etc.) continue to resolve without any caller changes.
pub mod finalize;
pub mod glow;
pub mod shadow;
pub mod shape;
pub mod squircle;
pub use finalize::*;
pub use glow::*;
pub use shadow::*;
pub use shape::*;
pub use squircle::*;

/// Clipping rectangle in buffer pixel coordinates. `x_end` and `y_end` are exclusive (matches Rust ranges). Construct via [`Clip::new`] or [`Clip::buffer`] for a full-buffer clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clip {
    pub x_start: usize,
    pub y_start: usize,
    pub x_end: usize,
    pub y_end: usize,
}

impl Clip {
    pub const fn new(x_start: usize, y_start: usize, x_end: usize, y_end: usize) -> Self {
        Self {
            x_start,
            y_start,
            x_end,
            y_end,
        }
    }

    /// Full-buffer clip. Equivalent to passing `None` to a primitive.
    pub const fn buffer(buf_w: usize, buf_h: usize) -> Self {
        Self {
            x_start: 0,
            y_start: 0,
            x_end: buf_w,
            y_end: buf_h,
        }
    }

    /// Resolve an optional clip — `None` defaults to the full buffer. Used by every primitive at entry so the rest of the function reads from a single concrete `Clip`.
    #[inline]
    pub fn resolve(opt: Option<Clip>, buf_w: usize, buf_h: usize) -> Self {
        match opt {
            Some(c) => c,
            None => Self::buffer(buf_w, buf_h),
        }
    }

    /// Intersect a primitive's `i32` bbox with `opt` (resolved against the buffer extent), returning integer pixel bounds suitable for `for` loops. Returns `None` if the intersection is empty (whole primitive is offscreen or fully clipped). Used by every rasterizer's entry path so the clip story is one call: pass `clip` thru, get back either `(x_start, y_start, x_end, y_end)` to iterate or an early-return signal.
    #[inline]
    pub fn intersect_bbox(
        opt: Option<Clip>,
        buf_w: usize,
        buf_h: usize,
        x_min: i32,
        x_max: i32,
        y_min: i32,
        y_max: i32,
    ) -> Option<(usize, usize, usize, usize)> {
        let c = Self::resolve(opt, buf_w, buf_h);
        let x_start = (x_min.max(0) as usize).max(c.x_start);
        let y_start = (y_min.max(0) as usize).max(c.y_start);
        let x_end = (x_max.max(0) as usize).min(buf_w).min(c.x_end);
        let y_end = (y_max.max(0) as usize).min(buf_h).min(c.y_end);
        if x_start >= x_end || y_start >= y_end {
            None
        } else {
            Some((x_start, y_start, x_end, y_end))
        }
    }
}

/// 2D affine transform — a 2×3 matrix laid out as `[a c tx; b d ty]`. Applied to a point `(x, y)` as `(a*x + c*y + tx, b*x + d*y + ty)`. Composes via [`then`](Self::then) (`a.then(b)` = "do `a` first, then `b`"). Used by the text path so glyph contours rotate / scale / skew *before* swash rasterizes them — proper hinting and AA on rotated glyphs, not a post-rotation pixel-shuffle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub a: Coord,
    pub b: Coord,
    pub c: Coord,
    pub d: Coord,
    pub tx: Coord,
    pub ty: Coord,
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    #[inline]
    pub const fn new(a: Coord, b: Coord, c: Coord, d: Coord, tx: Coord, ty: Coord) -> Self {
        Self { a, b, c, d, tx, ty }
    }

    #[inline]
    pub fn rotate(radians: Coord) -> Self {
        let (s, co) = crate::math::sin_cos(radians);
        Self {
            a: co,
            b: s,
            c: -s,
            d: co,
            tx: 0.0,
            ty: 0.0,
        }
    }

    #[inline]
    pub fn scale(sx: Coord, sy: Coord) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            tx: 0.0,
            ty: 0.0,
        }
    }

    #[inline]
    pub fn skew(kx: Coord, ky: Coord) -> Self {
        Self {
            a: 1.0,
            b: ky,
            c: kx,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    #[inline]
    pub fn translate(tx: Coord, ty: Coord) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx,
            ty,
        }
    }

    /// Compose `self` then `other` (i.e. `other ∘ self` in math notation). The result transform applies `self` to the point first, then `other` to the result.
    #[inline]
    pub fn then(self, other: Self) -> Self {
        Self {
            a: other.a * self.a + other.c * self.b,
            b: other.b * self.a + other.d * self.b,
            c: other.a * self.c + other.c * self.d,
            d: other.b * self.c + other.d * self.d,
            tx: other.a * self.tx + other.c * self.ty + other.tx,
            ty: other.b * self.tx + other.d * self.ty + other.ty,
        }
    }

    /// Apply to a point.
    #[inline]
    pub fn apply(self, x: Coord, y: Coord) -> (Coord, Coord) {
        (
            self.a * x + self.c * y + self.tx,
            self.b * x + self.d * y + self.ty,
        )
    }

    /// Bit-exact identity check. For "approximately identity" use a tolerance compare on the field deltas.
    #[inline]
    pub fn is_identity(self) -> bool {
        self.a == 1.0
            && self.b == 0.0
            && self.c == 0.0
            && self.d == 1.0
            && self.tx == 0.0
            && self.ty == 0.0
    }

    /// Axis-aligned bounding box of the transformed rectangle `[0, w] × [0, h]`. Returns `(min_x, min_y, max_x, max_y)` in transformed coordinates. Used by text rasterizers to compute the clip-clamp range for a transformed glyph.
    pub fn aabb_of_rect(self, w: Coord, h: Coord) -> (Coord, Coord, Coord, Coord) {
        let p0 = self.apply(0.0, 0.0);
        let p1 = self.apply(w, 0.0);
        let p2 = self.apply(0.0, h);
        let p3 = self.apply(w, h);
        let min_x = p0.0.min(p1.0).min(p2.0).min(p3.0);
        let max_x = p0.0.max(p1.0).max(p2.0).max(p3.0);
        let min_y = p0.1.min(p1.1).min(p2.1).min(p3.1);
        let max_y = p0.1.max(p1.1).max(p2.1).max(p3.1);
        (min_x, min_y, max_x, max_y)
    }
}

/// Quantize a continuous rotation angle (radians) into a bin index `0..N` where `N = ceil_to_multiple_of(K, ceil(2π × radius))`. The "1-pixel-arc rule": `radius = font_size_px / 2` makes one bin = one pixel of arc travel at the glyph's outer edge — the perceptual minimum step. Ceiling to a multiple of `K` (typically 8) lands cardinal + octant angles exactly on bin boundaries.
///
/// Use `.floor()` (not `.round()`) for monotonic bin assignment — `rem_euclid` handles negative angles cleanly and `.floor()` is single-uop on aarch64 NEON / x86 SSE4 ROUNDSS, so the legacy "subtract epsilon, use round" workaround is unnecessary.
pub fn quantize_rotation(radians: f32, font_size_px: f32, k: u32) -> u16 {
    let radius = font_size_px * 0.5;
    let raw_divs = crate::math::ceil(core::f32::consts::TAU * radius) as u32;
    let divs = ((raw_divs + k - 1) / k) * k;
    let theta = crate::math::rem_euclid(radians, core::f32::consts::TAU);
    let bin = crate::math::floor(theta / core::f32::consts::TAU * divs as f32) as u32;
    (bin % divs.max(1)) as u16
}

/// Snap a continuous rotation angle to the nearest quantization bin and return the bin's representative angle (in radians, range `[0, 2π)`). Same quantization grid as [`quantize_rotation`]. Use this when constructing a [`Transform`] for text that should benefit from the rasterized-glyph cache: animated rotation that varies continuously per frame would miss the cache every frame; pre-snapping to bins makes consecutive frames within the same bin cache-hit.
pub fn snap_rotation(radians: f32, font_size_px: f32, k: u32) -> f32 {
    let radius = font_size_px * 0.5;
    let raw_divs = crate::math::ceil(core::f32::consts::TAU * radius) as u32;
    let divs = ((raw_divs + k - 1) / k) * k;
    if divs == 0 {
        return 0.0;
    }
    let step = core::f32::consts::TAU / divs as f32;
    let theta = crate::math::rem_euclid(radians, core::f32::consts::TAU);
    crate::math::floor(theta / step) * step
}

/// Per-pixel alpha mask sized to the framebuffer. Multiplies into rendered alpha for soft clipping (textbox shapes, squircle pane corners, scroll fades). Carries its dimensions so primitives can panic on mismatch (per AGENT.md: init bugs fail loud, not silently render garbage).
pub struct AlphaMask<'a> {
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
}

impl<'a> AlphaMask<'a> {
    /// Construct an alpha mask. Panics if `pixels.len() != width * height`.
    pub fn new(pixels: &'a [u8], width: usize, height: usize) -> Self {
        assert_eq!(
            pixels.len(),
            width * height,
            "AlphaMask: pixels.len() ({}) != width * height ({} * {} = {})",
            pixels.len(),
            width,
            height,
            width * height,
        );
        Self {
            pixels,
            width,
            height,
        }
    }
}

/// Assert mask dimensions match buffer; panic with a descriptive message if not. Per AGENT.md a mask attached to the wrong buffer is an initialization bug — fail loud.
#[inline]
pub(crate) fn assert_mask_matches_buffer(mask: &AlphaMask, buf_w: usize, buf_h: usize) {
    assert!(
        mask.width == buf_w && mask.height == buf_h,
        "AlphaMask dimensions {}×{} don't match buffer {}×{}",
        mask.width,
        mask.height,
        buf_w,
        buf_h,
    );
}

/// Pack four 8-bit channels into a fluor internal pixel (`0xααRRGGBB`, α + darkness convention). The public API takes visible RGB and opacity α (`a = 255` means fully opaque) so consumer code reads naturally. Inside, RGB is stored as darkness (`255 − channel`); α is stored direct. This is the canonical external→internal boundary.
#[inline]
pub fn pack_argb(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((a as u32) << 24) | (((255 - r) as u32) << 16) | (((255 - g) as u32) << 8) | ((255 - b) as u32)
}

/// Per-pixel hit-test ID type. `u16` gives 65 535 registerable interactive zones — enough for "every list row is a hit zone" patterns without revisiting. Cost vs `u8` at 4 K is +8 MB on a ~100 MB+ render-state footprint, which is well inside the engineering budget for "never have to think about it again." Lives in `paint` because every hit-stamping rasterizer is here; [`crate::host::chrome`] and [`crate::host::app::HitRegistry`] re-export and consume.
pub type HitId = u16;

/// Reserved "no widget here" ID. Always 0 — registrations start at 1 so the value cannot collide.
pub const HIT_NONE: HitId = 0;

/// Wrap-add the RGB bytes of `delta_rgb` into `pixels[i]` for every `i` where `hit_map[i] == target_id`. The α byte is preserved. Used to apply a "tint" effect (hover, focus, etc.) on top of an already-painted region whose pixels were stamped with a known hit ID — mirrors the chrome's bake_hover pattern but generic over any hit ID. `wrapping_sub` of the same delta exactly reverses the add, so callers maintain old_delta/new_delta state and unbake/bake atomically.
pub fn wrap_add_rgb_where(pixels: &mut [u32], hit_map: &[HitId], target_id: HitId, delta_rgb: u32) {
    let dr = ((delta_rgb >> 16) & 0xFF) as u8;
    let dg = ((delta_rgb >> 8) & 0xFF) as u8;
    let db = (delta_rgb & 0xFF) as u8;
    let n = pixels.len().min(hit_map.len());
    for i in 0..n {
        if hit_map[i] == target_id {
            let p = pixels[i];
            let a = p & 0xFF00_0000;
            let r = (((p >> 16) & 0xFF) as u8).wrapping_add(dr) as u32;
            let g = (((p >> 8) & 0xFF) as u8).wrapping_add(dg) as u32;
            let b = ((p & 0xFF) as u8).wrapping_add(db) as u32;
            pixels[i] = a | (r << 16) | (g << 8) | b;
        }
    }
}

/// Generic per-hit-id overlay pass: walk `hit_test_map`, for each pixel whose hit id is currently tinted OR was tinted last frame, COPY the corresponding scratch pixel into persistent_screen — XOR'd to visible RGB, forced opaque, with an optional per-channel wrap-sub of `deltas[id]` if the id is currently tinted. The hit_test_map IS the silhouette trace of each interactive element (same flood-fill identification Photon uses), and we use it strictly to bound which pixels we touch — there's no diff math, no `last_applied_delta` accumulation, just "read scratch, maybe-adjust, write screen." Every frame's overlay is independently correct against scratch's content, so nothing the persistent_screen does between frames (fade, debug toggles, anything) can corrupt the tint.
///
/// Overlay-delta flag (bit 24, above the three channel bytes): apply the per-pixel sqrt gamma lift instead of a constant wrap-sub. The low 24 bits are zero, so a consumer that predates the flag degrades to a visible no-op write rather than a wrong colour.
pub const OVERLAY_SQRT_BRIGHTEN: u32 = 0x0100_0000;

/// `SQRT_BRIGHTEN_LUT[v] = round(sqrt(v/255)·255)` — the visible-RGB gamma-lift curve for [`OVERLAY_SQRT_BRIGHTEN`]. Endpoints are fixed points (0→0, 255→255); midtones lift hard (64→128).
static SQRT_BRIGHTEN_LUT: [u8; 256] = build_sqrt_brighten_lut();
const fn build_sqrt_brighten_lut() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut v = 0usize;
    while v < 256 {
        let n = v * 255;
        let mut r = 0usize;
        while (r + 1) * (r + 1) <= n {
            r += 1;
        }
        // Round to nearest: bump when the remainder passes the (r + 0.5)² midpoint.
        t[v] = if n - r * r > r { r + 1 } else { r } as u8;
        v += 1;
    }
    t
}

/// `deltas[id] == 0` means "this id is not currently tinted." `last_active[id] == true` means "we wrote this id last frame, so this frame we still need to restore its pixels to clean scratch baseline even if the tint is now zero." After the walk, `last_active` is updated to reflect which ids had non-zero delta this frame.
///
/// Why wrap-sub when applying: chrome's hover-colour constants and `wrap_sub_rgb(THEME_HOVER, THEME_FILL)` are darkness-space deltas. The persistent screen is post-finalize visible-RGB. Since `visible = 255 − darkness` per channel, adding `delta` in darkness = subtracting `delta` in visible. So the same delta constants the bake path used apply here with a flipped operator.
///
/// `win_ox / win_oy` are the screen-space offset of the window's top-left. `hit_test_map` and `scratch` are window-space (both length `win_w × win_h`). Pixels translating outside the screen are skipped.
///
/// Force-α=0xFF on every write is safe because pixels with non-`HIT_NONE` id are always interior to the window shape (corners and the squircle AA rim are stamped `HIT_NONE`), so no clip_mask carve gets stomped.
///
/// `deltas` and `last_active` are caller-owned slices the same length, sized to `registry.next_id` (= 1 + number of registered hit zones). With the u16 hit-id widening a fixed `[T; 256]` would either truncate the ID space or balloon to 256 KB per frame; slices keep the per-frame cost proportional to live widget count. IDs past `deltas.len()` are treated as `HIT_NONE` for safety — defensive against a stale paint stamping with an unregistered ID.
///
/// Besides the constant wrap-sub deltas, a delta with [`OVERLAY_SQRT_BRIGHTEN`] set applies a per-pixel gamma lift instead: each visible channel becomes `sqrt(v/255)·255` (LUT'd). Nonlinear, so it can't be a constant delta — it brightens midtones hard while leaving black and white alone, which reads as the whole image glowing rather than a flat tint wash. Used by the chrome app-icon orb's hover.
pub fn apply_overlay(
    scratch: &[u32],
    persistent_screen: &mut [u32],
    scr_w: usize,
    win_ox: i32,
    win_oy: i32,
    hit_test_map: &[HitId],
    win_w: usize,
    win_h: usize,
    deltas: &[u32],
    bboxes: &[Option<crate::canvas::PixelRect>],
    last_active: &mut [bool],
) {
    debug_assert_eq!(
        deltas.len(),
        last_active.len(),
        "deltas / last_active must be sized identically to registry.next_id"
    );
    // Quick reject: if no id is currently tinted AND none were tinted last frame, the overlay has zero work.
    let any_current = deltas.iter().any(|&d| d != 0);
    let any_last = last_active.iter().any(|&a| a);
    if !any_current && !any_last {
        return;
    }
    if scr_w == 0
        || win_w == 0
        || win_h == 0
        || hit_test_map.len() < win_w * win_h
        || scratch.len() < win_w * win_h
    {
        return;
    }
    let table_len = deltas.len();
    let scr_h = persistent_screen.len() / scr_w;
    // The full on-screen window range — the fallback scan for an id with no bbox.
    let full_y_lo = ((-win_oy).max(0) as usize).min(win_h);
    let full_y_hi = win_h.min(((scr_h as i32) - win_oy).max(0) as usize);
    let full_x_lo = ((-win_ox).max(0) as usize).min(win_w);
    let full_x_hi = win_w.min(((scr_w as i32) - win_ox).max(0) as usize);
    if full_y_lo >= full_y_hi || full_x_lo >= full_x_hi {
        for a in last_active.iter_mut() {
            *a = false;
        }
        return;
    }

    // Snapshot prev → new. With u16 IDs the previous fixed `[bool; 256]` swap can't survive, but the table is bounded by registered widget count (~tens, not 65 k) so a per-frame Vec pair is cheap.
    let prev_active: alloc::vec::Vec<bool> = last_active.to_vec();
    let mut new_active = alloc::vec![false; table_len];

    // Per-id BOUNDED scan: for each id that's tinted now OR was last frame, walk ONLY its bbox (clamped to the on-screen range; falling back to the full window if it has none), tinting the pixels where `hit_map == id`.
    // This makes hover cost O(hovered widget), not O(screen) — the whole point of the bbox table.
    // A just-left id (delta == 0, was_active) restores its pixels to clean scratch.
    for id in 0..table_len {
        let delta = deltas[id];
        let was_active = prev_active[id];
        if delta == 0 && !was_active {
            continue;
        }
        if delta != 0 {
            new_active[id] = true;
        }
        let (y0, y1, x0, x1) = match bboxes.get(id).copied().flatten() {
            Some(r) => (
                r.y0.clamp(full_y_lo, full_y_hi),
                r.y1.clamp(full_y_lo, full_y_hi),
                r.x0.clamp(full_x_lo, full_x_hi),
                r.x1.clamp(full_x_lo, full_x_hi),
            ),
            None => (full_y_lo, full_y_hi, full_x_lo, full_x_hi),
        };
        let sqrt = delta & OVERLAY_SQRT_BRIGHTEN != 0;
        let dr = ((delta >> 16) & 0xFF) as u8;
        let dg = ((delta >> 8) & 0xFF) as u8;
        let db = (delta & 0xFF) as u8;
        let id_hit = id as HitId;
        for y in y0..y1 {
            let scr_y = (y as i32 + win_oy) as usize;
            let map_row = y * win_w;
            let scr_row = scr_y * scr_w;
            for x in x0..x1 {
                let map_idx = map_row + x;
                if hit_test_map[map_idx] != id_hit {
                    continue;
                }
                let raw = scratch[map_idx];
                let visible = raw ^ 0x00FF_FFFF;
                let (r, g, b) = if sqrt {
                    // Per-pixel gamma lift: v' = sqrt(v/255)·255 per channel. See the doc note above.
                    (
                        SQRT_BRIGHTEN_LUT[((visible >> 16) & 0xFF) as usize] as u32,
                        SQRT_BRIGHTEN_LUT[((visible >> 8) & 0xFF) as usize] as u32,
                        SQRT_BRIGHTEN_LUT[(visible & 0xFF) as usize] as u32,
                    )
                } else {
                    (
                        (((visible >> 16) & 0xFF) as u8).wrapping_sub(dr) as u32,
                        (((visible >> 8) & 0xFF) as u8).wrapping_sub(dg) as u32,
                        ((visible & 0xFF) as u8).wrapping_sub(db) as u32,
                    )
                };
                let scr_x = (x as i32 + win_ox) as usize;
                persistent_screen[scr_row + scr_x] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
            }
        }
    }
    last_active.copy_from_slice(&new_active);
}

/// Compose `src` underneath `dst` where `src` is in *pre-composed* α + darkness form (the result of a chain of `under()` writes starting from empty — so `src.dark` is already attenuated by `src.α`). Unlike [`flatten`] with `BlendMode::Normal`, this kernel scales src's contribution by `(256 − dst.α)` only, NOT by `(256 − dst.α) × src.α`, avoiding the second premult that would dim every AA pixel.
///
/// Use case: blit a cached widget layer (built by repeated `under()` from empty) onto a target buffer.
pub fn flatten_premult(dst: &mut [u32], src: &[u32]) {
    let n = dst.len().min(src.len());
    for i in 0..n {
        let d = dst[i];
        if d >= 0xFF000000 {
            continue;
        }
        let s = src[i];
        let d_a = d >> 24;
        let factor = 256 - d_a;
        let s_a = (s >> 24) & 0xFF;
        let s_r = (s >> 16) & 0xFF;
        let s_g = (s >> 8) & 0xFF;
        let s_b = s & 0xFF;
        let d_r = (d >> 16) & 0xFF;
        let d_g = (d >> 8) & 0xFF;
        let d_b = d & 0xFF;
        let new_a = d_a + ((factor * s_a) >> 8);
        let new_r = d_r + ((factor * s_r) >> 8);
        let new_g = d_g + ((factor * s_g) >> 8);
        let new_b = d_b + ((factor * s_b) >> 8);
        dst[i] = (new_a << 24) | (new_r << 16) | (new_g << 8) | new_b;
    }
}

/// Per-byte wrap-subtract of `b` from `a`'s RGB bytes, returning a packed RGB delta suitable for [`wrap_add_rgb_where`]. The α byte is zeroed in the result. Used to derive a tint delta from two theme colours at runtime (e.g. `wrap_sub_rgb(TEXTBOX_HOVER, TEXTBOX_FILL)` → the per-channel offset that, wrap-added to a `TEXTBOX_FILL` pixel, lands at `TEXTBOX_HOVER`).
#[inline]
pub fn wrap_sub_rgb(a: u32, b: u32) -> u32 {
    let ar = (a >> 16) & 0xFF;
    let ag = (a >> 8) & 0xFF;
    let ab = a & 0xFF;
    let br = (b >> 16) & 0xFF;
    let bg = (b >> 8) & 0xFF;
    let bb = b & 0xFF;
    ((ar.wrapping_sub(br) & 0xFF) << 16)
        | ((ag.wrapping_sub(bg) & 0xFF) << 8)
        | (ab.wrapping_sub(bb) & 0xFF)
}

/// Like [`wrap_sub_rgb`] but scales the SIGNED per-channel delta `(target − base)` by `num/den` before wrapping — a fractional-strength tint.
/// The overlay's visible-space wrap-sub of the result lands a base-coloured pixel `num/den` of the way toward `target` (e.g. `(1, 4)` = quarter-opacity hover), so a vivid target hue reads as a gentle wash instead of a full-saturation flood.
/// α byte is dropped (0).
pub fn wrap_sub_rgb_scaled(target: u32, base: u32, num: i32, den: i32) -> u32 {
    let mut out = 0u32;
    for shift in [16u32, 8, 0] {
        let t = ((target >> shift) & 0xFF) as i32;
        let b = ((base >> shift) & 0xFF) as i32;
        let d = (t - b) * num / den;
        out |= ((d.rem_euclid(256)) as u32) << shift;
    }
    out
}

/// Unpack a fluor internal pixel into `(r, g, b, a)` with visible RGB and opacity α — the inverse of [`pack_argb`]. Flips darkness back to visible RGB; α passes thru.
#[inline]
pub fn unpack_argb(packed: u32) -> (u8, u8, u8, u8) {
    let a = (packed >> 24) as u8;
    let r = 255 - ((packed >> 16) as u8);
    let g = 255 - ((packed >> 8) as u8);
    let b = 255 - (packed as u8);
    (r, g, b, a)
}

use crate::pixel::Blend;
pub use crate::pixel::BlendMode;

/// Flatten `src` underneath `dst` across the whole slice, pixel-by-pixel, via [`Blend::under`]. `dst` is the partial composite already accumulated above (its α-byte = accumulated opacity); `src` is the new layer going behind. Per-pixel early-out fires when `dst.α == 0xFF`. Both slices must be the same length.
///
/// For `BlendMode::Normal` (the 99% case) the kernel uses 8-wide SIMD + Rayon parallelism. Other modes fall back to scalar per-pixel with Rayon row-chunking still applied — the math just stays in the scalar [`Blend::under`] kernel.
#[inline]
pub fn flatten(dst: &mut [u32], src: &[u32], mode: BlendMode) {
    let n = dst.len().min(src.len());
    if n == 0 {
        return;
    }
    let dst = &mut dst[..n];
    let src = &src[..n];
    const CHUNK: usize = 4096;
    if mode == BlendMode::Normal {
        crate::par::par_chunks(dst, CHUNK, |off, chunk| {
            let src_chunk = &src[off..off + chunk.len()];
            under_chunk_normal_dispatch(chunk, src_chunk);
        });
    } else {
        crate::par::par_chunks(dst, CHUNK, |off, chunk| {
            let src_chunk = &src[off..off + chunk.len()];
            for i in 0..chunk.len() {
                chunk[i] = chunk[i].under(src_chunk[i], mode);
            }
        });
    }
}

/// Per-chunk Normal-under dispatcher: SIMD with the `simd` feature, scalar fallback otherwise. Output is bit-identical between paths.
#[inline]
pub(crate) fn under_chunk_normal_dispatch(dst: &mut [u32], src: &[u32]) {
    #[cfg(feature = "simd")]
    {
        under_chunk_normal_simd(dst, src);
    }
    #[cfg(not(feature = "simd"))]
    {
        under_chunk_normal_scalar(dst, src);
    }
}

/// 8-wide SIMD chunk kernel for Normal-mode under. Tail (0..7 leftover pixels) runs scalar.
#[cfg(feature = "simd")]
fn under_chunk_normal_simd(dst: &mut [u32], src: &[u32]) {
    use crate::simd::{LANES, u32x8};
    let n = dst.len();
    let mut i = 0;
    while i + LANES <= n {
        let d_arr: [u32; 8] = dst[i..i + LANES].try_into().unwrap();
        let s_arr: [u32; 8] = src[i..i + LANES].try_into().unwrap();
        let out = crate::pixel::under_x8_normal(u32x8::from(d_arr), u32x8::from(s_arr));
        dst[i..i + LANES].copy_from_slice(out.as_array_ref());
        i += LANES;
    }
    while i < n {
        dst[i] = dst[i].under(src[i], BlendMode::Normal);
        i += 1;
    }
}

/// Scalar Normal-under chunk kernel for `--no-default-features` (no `simd`) builds.
#[cfg(not(feature = "simd"))]
fn under_chunk_normal_scalar(dst: &mut [u32], src: &[u32]) {
    for i in 0..dst.len() {
        dst[i] = dst[i].under(src[i], BlendMode::Normal);
    }
}

/// Per-chunk Normal-under dispatcher for a CONSTANT src pixel — what the rasterizer interior fast paths need: every dst pixel composes the same `full_pixel` underneath. Saves the cost of materializing an 8-pixel src array when all lanes are the same value (the SIMD path uses `u32x8::splat` instead).
#[inline]
pub(crate) fn under_chunk_const_dispatch(dst: &mut [u32], src_const: u32) {
    #[cfg(feature = "simd")]
    {
        under_chunk_const_simd(dst, src_const);
    }
    #[cfg(not(feature = "simd"))]
    {
        under_chunk_const_scalar(dst, src_const);
    }
}

/// 8-wide SIMD constant-src under kernel. `src_const` is broadcast to all 8 lanes once outside the inner loop. Tail scalar.
#[cfg(feature = "simd")]
fn under_chunk_const_simd(dst: &mut [u32], src_const: u32) {
    use crate::simd::{LANES, u32x8};
    let src = u32x8::splat(src_const);
    let n = dst.len();
    let mut i = 0;
    while i + LANES <= n {
        let d_arr: [u32; 8] = dst[i..i + LANES].try_into().unwrap();
        let out = crate::pixel::under_x8_normal(u32x8::from(d_arr), src);
        dst[i..i + LANES].copy_from_slice(out.as_array_ref());
        i += LANES;
    }
    while i < n {
        dst[i] = dst[i].under(src_const, BlendMode::Normal);
        i += 1;
    }
}

/// Scalar constant-src under for `--no-default-features` (no `simd`) builds.
#[cfg(not(feature = "simd"))]
fn under_chunk_const_scalar(dst: &mut [u32], src_const: u32) {
    for i in 0..dst.len() {
        dst[i] = dst[i].under(src_const, BlendMode::Normal);
    }
}

/// Intersect a caller-supplied `(x, y, w, h)` rect (in pixels, top-left origin, may be negative or extend off-clip) with a `Clip`. Returns `(x_min, y_min, x_max, y_max)` in `usize`, all guaranteed in-bounds for `pixels[y * buf_w + x]` indexing **as long as the supplied `Clip` is itself within the buffer**. Returns an empty range (x_min >= x_max or y_min >= y_max) if the rect lies entirely outside the clip.
///
/// **Rule 0 — WHY/PROOF/PREVENTS:** rect coords are external inputs (caller can pass a pane dragged off the window edge). WHY: compositor semantics demand "draw the intersection with the clip." PROOF without it: a negative `x as usize` wraps to a huge value, indexing past the pixel slice panics. PREVENTS: panic on partial-offscreen rects, which is a normal use case. The clip happens once per rect; inner loops trust the math.
#[inline]
fn clip_rect(
    clip: Clip,
    x: isize,
    y: isize,
    rect_w: isize,
    rect_h: isize,
) -> (usize, usize, usize, usize) {
    // Negative isize → huge usize after cast; .min(clip.x_end) clamps it down. .max(clip.x_start) ensures we never index before the clip's left edge.
    let x_end = x + rect_w;
    let y_end = y + rect_h;
    let x_min = if x < 0 {
        clip.x_start
    } else {
        (x as usize).clamp(clip.x_start, clip.x_end)
    };
    let y_min = if y < 0 {
        clip.y_start
    } else {
        (y as usize).clamp(clip.y_start, clip.y_end)
    };
    let x_max = if x_end < 0 {
        clip.x_start
    } else {
        (x_end as usize).clamp(clip.x_start, clip.x_end)
    };
    let y_max = if y_end < 0 {
        clip.y_start
    } else {
        (y_end as usize).clamp(clip.y_start, clip.y_end)
    };
    (x_min, y_min, x_max, y_max)
}

/// Fill a rectangle by under-blending `colour` into the buffer — the single rect paint primitive.
///
/// Per-pixel: `pixels[idx] = pixels[idx].under(effective_colour, BlendMode::Normal)`. The dst-opaque early-out fires automatically where a topmost paint has already claimed the pixel (`dst.α == 0xFF`), so the topmost-first doctrine is honored without callers having to think about z-order.
///
/// **Hairline convention:** a `0` width or height means "1 pixel thick" along that axis — pass `rect_h = 0` for a horizontal hairline, `rect_w = 0` for a vertical one, both `0` for a single-pixel dot. This lets callers draw rules/dividers without inventing a thickness; for a thicker line pass the real pixel count. (Negative dimensions stay a no-op.)
///
/// `mask: Some(&AlphaMask)` multiplies each pixel's mask alpha into `colour`'s opacity (`effective_opacity = colour_opacity * mask_alpha >> 8`) — used for shaped textbox interiors, scroll fades, etc. `mask: None` skips that math entirely.
pub fn fill_rect(
    canvas: &mut Canvas,
    x: isize,
    y: isize,
    rect_w: isize,
    rect_h: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    // 0 → 1-pixel hairline along that axis (both 0 = single-pixel dot). Coerce before clipping so the rect always covers at least one pixel line.
    let rect_w = if rect_w == 0 { 1 } else { rect_w };
    let rect_h = if rect_h == 0 { 1 } else { rect_h };
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    let clip = Clip::resolve(clip, buf_w, buf_h);
    if let Some(m) = mask {
        assert_mask_matches_buffer(m, buf_w, buf_h);
    }
    let (x_min, y_min, x_max, y_max) = clip_rect(clip, x, y, rect_w, rect_h);
    canvas.damage.add_bounds(x_min, y_min, x_max, y_max);
    let pixels: &mut [u32] = canvas.pixels;
    match mask {
        None => {
            for row in y_min..y_max {
                let base = row * buf_w;
                for col in x_min..x_max {
                    let idx = base + col;
                    pixels[idx] = pixels[idx].under(colour, BlendMode::Normal);
                }
            }
        }
        Some(m) => {
            // Mask convention matches `TextRenderer::render_buffer_u32`: `effective_α = colour_α × mask_α / 255` (opacity multiplies). Mask=0 → pixel fully clipped; mask=255 → colour passes thru at full α. The previous implementation here used `colour_opacity = 255 - α`, which inverted the semantics — for an opaque colour (α=0xFF) the mask was multiplied by zero and had no effect, so masks were silently ineffective on solid fills.
            let colour_alpha = (colour >> 24) & 0xFF;
            let colour_rgb = colour & 0x00FF_FFFF;
            for row in y_min..y_max {
                let base = row * buf_w;
                for col in x_min..x_max {
                    let idx = base + col;
                    let mask_a = m.pixels[idx] as u32;
                    // Floor `>> 8` with the +1 weight bump instead of `/ 255` (house convention, paint.rs module doc): mask 255 → +1 → ×256 passes colour_α thru EXACTLY, mask 0 → ×1 floors to 0, everything between lands within 1 LSB of the exact 255-division — and the shift is far cheaper than the division in this per-pixel loop.
                    let effective_alpha = (colour_alpha * (mask_a + 1)) >> 8;
                    if effective_alpha == 0 {
                        continue;
                    }
                    let masked = colour_rgb | (effective_alpha << 24);
                    pixels[idx] = pixels[idx].under(masked, BlendMode::Normal);
                }
            }
        }
    }
}

/// Stroke (outline) an axis-aligned rectangle. Draws four filled rect strips along the edges via [`fill_rect`]; corners are not joined separately because at 90° angles the strips meet cleanly.
pub fn stroke_rect(
    canvas: &mut Canvas,
    x: isize,
    y: isize,
    rect_w: isize,
    rect_h: isize,
    stroke: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    if stroke <= 0 || rect_w <= 0 || rect_h <= 0 {
        return;
    }
    let inner_h = rect_h - 2 * stroke;
    let edges: [(isize, isize, isize, isize); 4] = [
        (x, y, rect_w, stroke),                             // top
        (x, y + rect_h - stroke, rect_w, stroke),           // bottom
        (x, y + stroke, stroke, inner_h),                   // left
        (x + rect_w - stroke, y + stroke, stroke, inner_h), // right
    ];
    // Damage accumulates via the four fill_rect calls, each reporting its own clipped bbox.
    for &(ex, ey, ew, eh) in &edges {
        fill_rect(canvas, ex, ey, ew, eh, colour, clip, mask);
    }
}

/// Fill the buffer with photon's signature procedural background — symmetric organic noise. Rows are RNG-independent (each row reseeds from `logical_row`), so the outer loop parallelizes cleanly via [`crate::par::par_rows`]. Mirrored left/right halves like photon. Set `fullscreen=true` to fill the whole buffer; `false` leaves a 1px border for the window edge stroke. `shimmer` is mixed into each row's starting colour so animating it cycles the colour bias across rows without changing the noise topology; `scroll_offset` shifts the texture vertically (for content scroll integration). The speckle gate (the rare bright/dim flash branch inside each row) fires at a constant 1/256 rate independent of `shimmer`.
///
/// SIMD inside a row is intentionally not done — the per-pixel RNG (`rng ^= rng.rotate_left(13).wrapping_add(const)`) is a serial dependency chain; vectorizing it would require N independent RNG streams per lane and would change photon's visual pattern. If profiling shows the per-row scalar work still dominating after Rayon, that's the next lever.
///
/// Clip restricts the row range. Mask isn't supported here (background is bg — masking it would mean "draw nothing where mask is zero" which is the same as just clearing afterward; if you need that, do it explicitly).
pub fn background_noise(
    canvas: &mut Canvas,
    shimmer: usize,
    fullscreen: bool,
    scroll_offset: isize,
    clip: Option<Clip>,
    base: Option<u32>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if buf_w < 2 || buf_h < 2 {
        return;
    }
    let clip = Clip::resolve(clip, buf_w, buf_h);
    // Clip first, then `fullscreen` further insets by 1 px for the edge hairline.
    let (row_start, row_end, x_start, x_end) = if fullscreen {
        (clip.y_start, clip.y_end, clip.x_start, clip.x_end)
    } else {
        (
            (clip.y_start + 1).min(clip.y_end),
            clip.y_end.saturating_sub(1).max(clip.y_start),
            (clip.x_start + 1).min(clip.x_end),
            clip.x_end.saturating_sub(1).max(clip.x_start),
        )
    };
    if row_start >= row_end || x_start >= x_end {
        return;
    }
    canvas.damage.add_bounds(x_start, row_start, x_end, row_end);
    let pixels: &mut [u32] = canvas.pixels;
    let resolved_base = base.unwrap_or(crate::theme::BG_BASE);
    crate::par::par_rows(pixels, buf_w, row_start, row_end, |row_idx, row_pixels| {
        let logical_row = row_idx as isize - scroll_offset;
        background_row(
            row_pixels,
            buf_w,
            logical_row,
            buf_h,
            x_start,
            x_end,
            shimmer,
            resolved_base,
        );
    });
}

#[inline]
fn background_row(
    row_pixels: &mut [u32],
    width: usize,
    logical_row: isize,
    height: usize,
    x_start: usize,
    x_end: usize,
    shimmer: usize,
    base: u32,
) {
    use crate::theme::{BG_MASK, BG_SPECKLE};
    // Noise math runs in visible-RGB space (matching photon's reference). At the store site we flip the visible result to stored darkness via XOR, then OR α=0xFF for opaque. Mask off the top byte first to strip any carry from `wrapping_add`.
    const VISIBLE_TO_DARK_FLIP: u32 = 0x00FFFFFF;
    const RGB_MASK: u32 = 0x00FFFFFF;
    const OPAQUE_ALPHA: u32 = 0xFF000000;
    // Hybrid 2-pass: pass 1 fills `noise_buf` with the row's chunk of noise values via the serial RNG/colour chain (branches stay scalar — predicating the speckle gate would cost as much as it saves). Pass 2 hands the chunk to the 8-wide SIMD under-blend kernel (`under_chunk_normal_dispatch`), which composites it into row_pixels at ~1 cycle/pixel amortized. Output is bit-identical to the old straight-scalar version.
    const CHUNK: usize = 64;
    let mut noise_buf = [0u32; CHUNK];
    let ones = 0x0001_0101u32;
    // RNG chain is explicitly u64 (not usize) so the noise pattern is bit-identical on 32-bit
    // targets (wasm32 browser renderer) and 64-bit desktops — same seed, same wrap points.
    let seed: u64 = 0xDEAD_BEEF_0123_4567u64
        ^ (logical_row as i64 as u64)
            .wrapping_sub(height as u64 / 2)
            .wrapping_mul(0x9E37_79B9_4517_B397);

    // Right half — left to right. Noise composes UNDER existing content (topmost-first): an empty pixel gets the noise; a non-empty pixel (e.g. a topmost rect already painted) has the noise blended behind it.
    let mut rng = seed;
    let mut colour = rng.wrapping_add(shimmer as u64) as u32 & BG_MASK;
    let mut x = width / 2;
    while x < x_end {
        let chunk_len = (x_end - x).min(CHUNK);
        for i in 0..chunk_len {
            rng ^= rng.rotate_left(13).wrapping_add(12_345_678_942);
            let adder = rng as u32 & ones;
            if rng < u64::MAX / 256 {
                colour = (rng as u32 >> 8) & BG_SPECKLE;
            } else {
                colour = colour.wrapping_add(adder) & BG_MASK;
                let subtractor = (rng >> 5) as u32 & ones;
                colour = colour.wrapping_sub(subtractor) & BG_MASK;
            }
            noise_buf[i] =
                ((colour.wrapping_add(base) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA;
        }
        under_chunk_normal_dispatch(&mut row_pixels[x..x + chunk_len], &noise_buf[..chunk_len]);
        x += chunk_len;
    }

    // Left half — right to left, same RNG seed (mirror), SUB instead of ADD on the rng step. Within each chunk the RNG iterates rightmost-pixel-first; we store into `noise_buf` in left-to-right order (`i = chunk_len-1` down to 0) so the chunk dispatch can scan the buffer sequentially.
    rng = seed;
    let mut colour = rng.wrapping_add(shimmer as u64) as u32 & BG_MASK;
    let mut x_hi = width / 2;
    while x_hi > x_start {
        let chunk_lo = x_hi.saturating_sub(CHUNK).max(x_start);
        let chunk_len = x_hi - chunk_lo;
        for i in (0..chunk_len).rev() {
            rng ^= rng.rotate_left(13).wrapping_sub(12_345_678_942);
            let adder = rng as u32 & ones;
            if rng < u64::MAX / 256 {
                colour = (rng as u32 >> 8) & BG_SPECKLE;
            } else {
                colour = colour.wrapping_add(adder) & BG_MASK;
                let subtractor = (rng >> 5) as u32 & ones;
                colour = colour.wrapping_sub(subtractor) & BG_MASK;
            }
            noise_buf[i] =
                ((colour.wrapping_add(base) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA;
        }
        under_chunk_normal_dispatch(&mut row_pixels[chunk_lo..x_hi], &noise_buf[..chunk_len]);
        x_hi = chunk_lo;
    }
}

/// Debug toggle that lets the `[]p` chord skip the boundary premultiply at runtime — A/B the Linux premult fix without recompiling. Stays `false` by default.
pub static DEBUG_SKIP_PREMULT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug cycle bound to the `[]a` chord. Three states (rotate each press): `0` = off (normal boundary conversion), `1` = α-as-grayscale (replace each pixel with `(final_α, final_α, final_α, 0xFF)` — inspect alpha distribution), `2` = force-opaque (force every pixel's α to 255 and pass the visible RGB thru unmodified — inspect what the kernel produced BEFORE the clip mask + premultiply trimmed it).
pub static DEBUG_SHOW_ALPHA: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
pub const DEBUG_SHOW_ALPHA_OFF: u8 = 0;
pub const DEBUG_SHOW_ALPHA_GRAYSCALE: u8 = 1;
pub const DEBUG_SHOW_ALPHA_FORCE_OPAQUE: u8 = 2;

/// Debug toggle bound to the `[]h` chord. When set, `finalize_into_screen` routes thru the FORCE_OPAQUE debug branch (XOR darkness → visible RGB, ignore clip_mask trim, force α=0xFF, skip premult) so the per-id colours the consumer paints into scratch land in `persistent_screen` exactly as written — no AA edges, no corner cutouts, no shadow boost on the perimeter. The host additionally skips `paint_shadow` while this is on so the band outside the window doesn't disturb the hit-id view at the chrome edge.
pub static DEBUG_SHOW_HITMASK: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug toggle bound to the `[]d` chord (Decay). When set, the host saturating-subtracts 1 from every persistent_screen pixel's RGB at the top of each frame (before finalize) and self-requests a continuous redraw chain. Fresh pixels written by finalize / overlay land at full brightness; pixels that aren't repainted fade visibly toward black. Used to verify the incremental left/right opaque-scan finalize is actually copying the regions it should — anything that doesn't get touched on a given frame visibly decays.
pub static DEBUG_SHOW_FADE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug toggle bound to the `[]b` chord. When set, the **incremental** finalize path saturating-adds 16 to the blue byte of every pixel inside the per-row `[l, r)` opaque-scan range, AFTER the chunk dispatch writes the post-XOR visible-RGB pixel. Makes exactly the pixels the incremental scan touches each frame glow blue — full_repaint frames are deliberately NOT tinted because a uniform interior wash on every focus change / resize / drag release would drown out the actual diagnostic signal (which is "what does the incremental scan land on?"). Pairs naturally with `DEBUG_SHOW_FADE` (the blue stack reaches equilibrium where incremental finalize hits often, decays elsewhere).
pub static DEBUG_SHOW_OPAQUE_SCAN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug toggle that suppresses chrome layer rasterization (perimeter hairline + future controls + title) so consumers can see the background / panes / textbox underneath without chrome on top. Bound to the `[]c` chord. The clip_mask is still carved at the boundary, so the window-shape trim remains visible. Stays `false` by default.
pub static DEBUG_SKIP_CHROME: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug toggle that suppresses ONLY the controls strip (curves + hairlines + glyphs + dividers + strip-bg fill) while keeping the window perimeter intact. Bound to the `[]l` chord (controLs). Useful for isolating perimeter rendering from controls rendering. Stays `false` by default.
pub static DEBUG_SKIP_CONTROLS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug toggle that overlays a one-line diagnostic strip across the bottom of the window showing live render-pipeline stats: composite-FPS (= `1.0 / composite_time`, NOT the vsync-capped frame rate) and the cumulative frame counter. Bound to the `[]f` chord. The composite-FPS is the actual headroom — a 144 Hz display showing "1240 FPS" means each composite took ~0.8 ms, leaving 6.1 ms of slack against vsync. `false` by default. Counter bumped by primitives that perform genuine *rasterize* work (geometric paint, glyph shaping, etc.) — NOT by blits/copies/tint applications. The host reads-and-resets this with `.swap(0, ...)` after `app.render` to decide whether to call `DebugStats::record_rasterize` or only `record_present`. Lets the F (frame) counter climb on hover-only frames while R (rasterize) stays put.
pub static RASTERIZE_OPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub static DEBUG_SHOW_FPS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug toggle that overlays a 1-px magenta hairline around the damage rect the host repaints this frame. Bound to the `[]w` chord ("Where"). Drawn directly into the platform back buffer AFTER `persistent_screen` has been copied in, BEFORE `present()`. The outline never enters `persistent_screen`, never flows thru finalize, and never survives more than one frame — so toggling it on/off needs no full-repaint promotion and there is no stale-bbox state to carry between frames. `false` by default.
pub static DEBUG_SHOW_DAMAGE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Stamp a 1-pixel magenta hairline around `bbox` directly into the screen-sized, visible-RGB back buffer. `bbox` is window-local; `offset_x` / `offset_y` are the window's top-left in screen space. Pure overwrite — no blending, no `under` path, no damage tracking. Caller invokes between `copy_from_slice(&persistent_screen)` and `buffer.present()` so the magenta lives for exactly one frame.
pub fn stamp_damage_outline_visible(
    buf: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    bbox: crate::canvas::PixelRect,
    offset_x: i32,
    offset_y: i32,
) {
    const MAGENTA: u32 = 0xFFFF_00FF;
    if bbox.is_empty() || scr_w == 0 || scr_h == 0 {
        return;
    }
    let sx0_i = offset_x + bbox.x0 as i32;
    let sy0_i = offset_y + bbox.y0 as i32;
    let sx1_i = offset_x + bbox.x1 as i32;
    let sy1_i = offset_y + bbox.y1 as i32;
    let x0 = sx0_i.max(0) as usize;
    let y0 = sy0_i.max(0) as usize;
    let x1 = (sx1_i.max(0) as usize).min(scr_w);
    let y1 = (sy1_i.max(0) as usize).min(scr_h);
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    // Top row.
    let top_row = y0 * scr_w;
    for px in x0..x1 {
        buf[top_row + px] = MAGENTA;
    }
    // Bottom row — only paint if distinct from the top (i.e., rect is ≥ 2 tall); otherwise the top row already covered it.
    if y1 - 1 > y0 {
        let bot_row = (y1 - 1) * scr_w;
        for px in x0..x1 {
            buf[bot_row + px] = MAGENTA;
        }
    }
    // Left and right columns (interior rows only — corners are claimed by the top/bottom rows).
    let interior_y0 = y0 + 1;
    let interior_y1 = y1.saturating_sub(1);
    if interior_y0 < interior_y1 {
        let right_col = x1 - 1;
        for py in interior_y0..interior_y1 {
            let row = py * scr_w;
            buf[row + x0] = MAGENTA;
            if right_col > x0 {
                buf[row + right_col] = MAGENTA;
            }
        }
    }
}

/// Overlay a chord-hint panel listing the consumer's debug shortcuts. Painted while the consumer's debug chord is armed so users discover bindings without consulting docs. `hints` is `(chord_label, description)` pairs. `span` is the viewport's effective span (`2wh/(w+h)`) — all sizes derive from it so the panel scales with the user's zoom.
///
/// Topmost-first: text glyphs paint first; the semi-opaque panel background fills the gaps behind them.
#[cfg(feature = "text")]
pub fn draw_chord_hint(
    canvas: &mut Canvas,
    text: &mut crate::text::TextRenderer,
    hints: &[(&str, &str)],
    span: Coord,
) {
    if hints.is_empty() || canvas.width < 200 || canvas.height < 120 {
        return;
    }
    // RU-coherent typography. `font_size = span × 0.014` lands at ~14 px at span ≈ 1000, ~28 px at 2000, etc. Vertical rhythm + padding follow as ratios of the font size so the panel scales as one unit. No pixel floor — span already scales with the viewport, and a hardcoded minimum would break resolution independence (and desync from photon's matching bbox math).
    let font_size = span * 0.014;
    let header_size = font_size * 1.18;
    let line_h = font_size * 1.55;
    let pad = font_size * 1.25;

    let line_count = hints.len() as f32 + 1.5;
    let panel_h = line_count * line_h + pad * 2.0;
    // Width clamped against the font_size so the panel never gets narrower than a long binding line. Upper clamp keeps it from spanning the whole viewport on wide windows.
    let panel_w = (span * 0.45).clamp(font_size * 22.0, font_size * 36.0);

    let cx = canvas.width as f32 * 0.5;
    let cy = canvas.height as f32 * 0.4;
    let panel_y = cy - panel_h * 0.5;

    let title_colour = pack_argb(255, 255, 255, 0xFF);
    let body_colour = pack_argb(220, 220, 220, 0xFF);

    text.draw_text_center_u32(
        canvas,
        "Debug chord — [ + ] then …",
        cx,
        panel_y + pad + header_size * 0.5,
        header_size,
        500,
        title_colour,
        "Open Sans",
        None,
        None,
        None,
    );

    for (i, (chord, desc)) in hints.iter().enumerate() {
        let line_y = panel_y + pad + header_size + line_h * (i as f32 + 0.5) + font_size * 0.5;
        let line = alloc::format!("{}  —  {}", chord, desc);
        text.draw_text_center_u32(
            canvas,
            &line,
            cx,
            line_y,
            font_size,
            400,
            body_colour,
            "Open Sans",
            None,
            None,
            None,
        );
    }

    // Panel background fills behind the glyphs (text rows already occupied; bar's under() only lands on the gaps).
    let bg = pack_argb(0, 0, 0, 0xD8);
    draw_rect(canvas, cx, cy, panel_w, panel_h, bg, None);
}

/// Live diagnostic counters owned by the host's render loop and read by [`draw_debug_strip`]. All fields are simple POD; the host updates them every frame when [`DEBUG_SHOW_FPS`] is on and the helper renders them as a single line of text into the bottom-of-window scratch region before the boundary pass runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugStats {
    /// Raw work time of the most recent frame, per stage, in seconds. NOT smoothed — each value is exactly the last measurement so SIMD/Rayon toggles produce immediately legible swings. FPS shown in the strip is `1.0 / stage_secs` per stage and `1.0 / sum` for total.
    pub app_secs: f32,
    pub fill_secs: f32,
    pub finalize_secs: f32,
    pub shadow_secs: f32,
    /// Frames presented to the OS since start — includes "frame-only" updates (tint diff, cursor blink) that did no rasterization. Wraps at `u64::MAX`.
    pub frame_count: u64,
    /// Frames where actual rasterization work happened. Diverges from `frame_count` once dirty-tracking lands and lets the host skip rasterize on hover-only / tint-only updates.
    pub rasterize_count: u64,
    /// Fraction (`0.0..=1.0`) of the rasterizable area touched by the most recent rasterize. `1.0` = full repaint; smaller values appear once damage-clipped partial rasterization is wired. Held verbatim from the last `record_rasterize` call.
    pub last_rasterize_pct: f32,
    /// Fraction (`0.0..=1.0`) of the viewport area covered by this frame's `damage_clip` — what the host actually cleared + asked the app to repaint. Updated EVERY frame (on present), so it reflects the live per-frame damage even when `last_rasterize_pct` is sticky from the last full rasterize. Drops on hover/focus frames where damage is just the textbox bbox.
    pub last_damage_pct: f32,
}

impl DebugStats {
    /// Bump `frame_count` only — for frames that present without rasterizing (cursor blink off-frame, tint-diff-only hover update once dirty tracking lands). No stage timings updated; the strip continues to show whatever the last rasterize recorded. `damage_pct` updates every present so the live damage area shows even when no rasterize fired.
    #[inline]
    pub fn record_present(&mut self, damage_pct: f32) {
        self.frame_count = self.frame_count.wrapping_add(1);
        self.last_damage_pct = damage_pct;
    }

    /// Store this rasterize's stage times, bump `rasterize_count`, and record the fraction of the rasterizable area that was touched. Pass `pct = 1.0` for a full repaint; once damage-clipped rasterization is wired, pass `damage_area / full_area`.
    #[inline]
    pub fn record_rasterize(&mut self, app: f32, fill: f32, finalize: f32, shadow: f32, pct: f32) {
        self.app_secs = app;
        self.fill_secs = fill;
        self.finalize_secs = finalize;
        self.shadow_secs = shadow;
        self.rasterize_count = self.rasterize_count.wrapping_add(1);
        self.last_rasterize_pct = pct;
    }

    /// Shorthand for the current "every frame is a full rasterize" world — bumps both counters and stores timings with `pct = 1.0`. Replace per-call with `record_rasterize` + `record_present` once dirty tracking can drop the rasterize on tint-only updates.
    #[inline]
    pub fn record_frame(&mut self, app: f32, fill: f32, finalize: f32, shadow: f32) {
        self.record_rasterize(app, fill, finalize, shadow, 1.0);
        self.record_present(1.0);
    }

    #[inline]
    pub fn total_secs(&self) -> f32 {
        self.app_secs + self.fill_secs + self.finalize_secs + self.shadow_secs
    }
}

/// Overlay a one-line diagnostic strip across the bottom of `pixels` showing the live stats in [`DebugStats`]. Gated by [`DEBUG_SHOW_FPS`] — the host should check that flag before calling. Paints into the α + darkness scratch buffer BEFORE the boundary pass so the strip flows thru `finalize_*` like any other content (no special handling needed downstream).
///
/// The strip is `~24` pixels tall, semi-opaque black background, bright green monospace text (terminal-style for readability against any underlying content). Positioned at the very bottom of `pixels`; clipped to the buffer if the window is too short to fit the strip (returns early in that case — diagnostic, not load-bearing).
///
/// `strip_y` is the canvas-relative row where the strip's top edge lands. Callers can pass any value that fits inside the canvas (`strip_y + DEBUG_STRIP_H ≤ canvas.height`); the host typically points the canvas at a dedicated `DEBUG_STRIP_H`-tall staging buffer with `strip_y = 0` so the strip never touches the app's scratch.
#[cfg(feature = "text")]
pub fn draw_debug_strip(
    canvas: &mut Canvas,
    text: &mut crate::text::TextRenderer,
    stats: &DebugStats,
    strip_y: usize,
) {
    const FONT_SIZE: f32 = 13.0;
    let width = canvas.width;
    let height = canvas.height;
    if width == 0 || height < strip_y + DEBUG_STRIP_H {
        return;
    }

    let app_ms = stats.app_secs * 1000.0;
    let fill_ms = stats.fill_secs * 1000.0;
    let fin_ms = stats.finalize_secs * 1000.0;
    let shadow_ms = stats.shadow_secs * 1000.0;
    let total_ms = stats.total_secs() * 1000.0;

    let damage_pct = (stats.last_damage_pct * 100.0).clamp(0.0, 100.0);
    let stats_line = alloc::format!(
        "app {app_ms:>6.3} ms  fill {fill_ms:>6.3} ms  fin {fin_ms:>6.3} ms  shdw {shadow_ms:>6.3} ms    tot {total_ms:>6.3} ms    F {:>7} R {:>7} ({damage_pct:>5.1}%)",
        stats.frame_count,
        stats.rasterize_count,
    );

    // Topmost-first ordering: text glyphs paint FIRST so the bar's under() writes are rejected by the glyph pixels, leaving the green characters visible against the black. If the bar were painted first it would fill all strip pixels opaque and every glyph would be eaten.
    let fg = pack_argb(80, 255, 120, 0xFF);
    let text_cy = strip_y as f32 + DEBUG_STRIP_H as f32 * 0.5;
    text.draw_text_center_u32(
        canvas,
        &stats_line,
        width as f32 * 0.5,
        text_cy,
        FONT_SIZE,
        400,
        fg,
        "monospace",
        None,
        None,
        None,
    );

    // Bar fills behind the glyphs (topmost-first: text rows already occupied, bar's under() only lands on the gaps).
    let bg = pack_argb(0, 0, 0, 0xE0);
    draw_rect(
        canvas,
        width as Coord * 0.5,
        strip_y as Coord + DEBUG_STRIP_H as Coord * 0.5,
        width as Coord,
        DEBUG_STRIP_H as Coord,
        bg,
        None,
    );
}

/// Height in pixels of the [`draw_debug_strip`] band — the consumer / host uses it to size the staging buffer and pick the screen-space y position.
#[cfg(feature = "text")]
pub const DEBUG_STRIP_H: usize = 24;



pub fn draw_blinkey(canvas: &mut Canvas, bx: usize, by: usize, height: usize, top_bright: bool) {
    let buf_w = canvas.width;
    // Damage: ±7 horizontal spread × `height` vertical band. Caller's bounds invariant (bx ≥ 7, bx < buf_w-7) guarantees this stays in-buffer.
    canvas.damage.add_bounds(bx - 7, by, bx + 8, by + height);
    let pixels: &mut [u32] = canvas.pixels;
    let half = height / 2;
    for y in by..by + height {
        let idx = y * buf_w + bx;
        let t = (y - by - half) as isize as f32 / half as f32;
        let wave = if top_bright {
            (1.0 - t * t) * (1.0 - t) * (1.0 - t) * crate::theme::CURSOR_BRIGHTNESS
        } else {
            (1.0 - t * t) * (1.0 + t) * (1.0 + t) * crate::theme::CURSOR_BRIGHTNESS
        };
        let w = wave as u32;
        for dx in -7i32..=7 {
            // The cursor is a BRIGHT wave. In the α + darkness convention (RGB bytes are
            // darkness, `0 = white`), brightening means REDUCING darkness — a per-channel
            // saturating subtract, NOT the add the visible-RGB original used. (Photon's
            // buffer was visible-space, so it added; the port kept the `+=` which silently
            // darkened the cursor into invisibility against a dark field.) α is preserved.
            let k = w >> dx.unsigned_abs();
            let p = &mut pixels[(idx as isize + dx as isize) as usize];
            let a = *p & 0xFF00_0000;
            let r = ((*p >> 16) & 0xFF).saturating_sub(k);
            let g = ((*p >> 8) & 0xFF).saturating_sub(k);
            let b = (*p & 0xFF).saturating_sub(k);
            *p = a | (r << 16) | (g << 8) | b;
        }
    }
}


/// Glyph rasterizers for window controls. Ported verbatim from photon's [compositing.rs](/mnt/Octopus/Code/photon/src/ui/compositing.rs) — the squircle minus / squircle ring / capsule X — so chrome looks identical to photon.
pub mod glyph {
    /// Draw a horizontal squircle "minus" stroke centered at `(x, y)` inside a button of pixel radius `r`. Uses a 4-power squircle with widened axis to make a flat horizontal pill.
    pub fn minimize_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
    ) {
        let r = r + 1;
        let r_render = r / 4 + 1;
        let r_2 = r_render * r_render;
        let r_4 = r_2 * r_2;
        let r_3 = r_render * r_render * r_render;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;

        for h in -(r_render as isize)..=(r_render as isize) {
            for w in -(r as isize)..=(r as isize) {
                let h2 = h * h;
                let h4 = h2 * h2;
                let a = (w.abs() - (r * 3 / 4) as isize).max(0);
                let w2 = a * a;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 <= r_4 {
                    let px = (x as isize + w) as usize;
                    let py = (y as isize + h + (r / 2) as isize) as usize;
                    let idx = py * width + px;
                    let gradient = ((r_4 - dist_4) << 8) / (r_3 << 2);
                    if gradient > 255 {
                        pixels[idx] = stroke_packed;
                    } else {
                        pixels[idx] = blend_swar(pixels[idx], stroke_packed, gradient as u64);
                    }
                }
            }
        }
    }

    /// Draw a square "maximize" symbol — squircle ring with stroke + interior fill — centered at `(x, y)` with pixel radius `r`.
    pub fn maximize_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
        fill_colour: u32,
    ) {
        let r = r + 1;
        let mut r_4 = r * r;
        r_4 *= r_4;
        let r_3 = r * r * r;

        let r_inner = r * 4 / 5;
        let mut r_inner_4 = r_inner * r_inner;
        r_inner_4 *= r_inner_4;
        let r_inner_3 = r_inner * r_inner * r_inner;

        let outer_edge_threshold = r_3 << 2;
        let inner_edge_threshold = r_inner_3 << 2;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;
        let fill_packed = fill_colour & 0x00FF_FFFF;

        for h in -(r as isize)..=r as isize {
            for w in -(r as isize)..=r as isize {
                let h2 = h * h;
                let h4 = h2 * h2;
                let w2 = w * w;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 > r_4 {
                    continue;
                }
                let px = (x as isize + w) as usize;
                let py = (y as isize + h) as usize;
                let idx = py * width + px;

                let dist_from_outer = r_4 - dist_4;
                if dist_4 <= r_inner_4 {
                    let dist_from_inner = r_inner_4 - dist_4;
                    if dist_from_inner <= inner_edge_threshold {
                        let gradient = (dist_from_inner << 8) / inner_edge_threshold;
                        pixels[idx] = blend_swar(stroke_packed, fill_packed, gradient as u64);
                    } else {
                        pixels[idx] = fill_packed;
                    }
                } else if dist_from_outer <= outer_edge_threshold {
                    let gradient = (dist_from_outer << 8) / outer_edge_threshold;
                    pixels[idx] = blend_swar(pixels[idx], stroke_packed, gradient as u64);
                } else {
                    pixels[idx] = stroke_packed;
                }
            }
        }
    }

    /// Draw an antialiased "X" close symbol — two crossed capsule lines — centered at `(x, y)` with pixel radius `r`.
    pub fn close_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
    ) {
        let r = r + 1;
        let thickness = (r / 3).max(1) as f32;
        let radius = thickness / 2.0;
        let size = (r * 2) as f32;
        let cxf = x as f32;
        let cyf = y as f32;
        let end = size / 3.0;

        let x1_start = cxf - end;
        let y1_start = cyf - end;
        let x1_end = cxf + end;
        let y1_end = cyf + end;
        let x2_start = cxf + end;
        let y2_start = cyf - end;
        let x2_end = cxf - end;
        let y2_end = cyf + end;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;
        let cxi = x as i32;
        let cyi = y as i32;
        let height = (pixels.len() / width) as i32;
        let min_x = ((x as i32) - (r as i32)).max(0);
        let max_x = ((x as i32) + (r as i32)).min(width as i32);
        let min_y = ((y as i32) - (r as i32)).max(0);
        let max_y = ((y as i32) + (r as i32)).min(height);

        // Each quadrant samples one of the two diagonals (whichever passes thru it).
        let quadrants: [(i32, i32, i32, i32, f32, f32, f32, f32); 4] = [
            (min_x, cxi, min_y, cyi, x1_start, y1_start, x1_end, y1_end), // top-left, diag1
            (cxi, max_x, min_y, cyi, x2_start, y2_start, x2_end, y2_end), // top-right, diag2
            (min_x, cxi, cyi, max_y, x2_start, y2_start, x2_end, y2_end), // bottom-left, diag2
            (cxi, max_x, cyi, max_y, x1_start, y1_start, x1_end, y1_end), // bottom-right, diag1
        ];
        for (qx0, qx1, qy0, qy1, x0, y0, x1, y1) in quadrants {
            for py in qy0..qy1 {
                for px in qx0..qx1 {
                    let dist = distance_to_capsule(
                        px as f32 + 0.5,
                        py as f32 + 0.5,
                        x0,
                        y0,
                        x1,
                        y1,
                        radius,
                    );
                    let alpha_f = if dist < -0.5 {
                        1.0
                    } else if dist < 0.5 {
                        0.5 - dist
                    } else {
                        0.0
                    };
                    if alpha_f > 0.0 {
                        let idx = py as usize * width + px as usize;
                        let alpha = (alpha_f * 256.0) as u64;
                        pixels[idx] = blend_swar(pixels[idx], stroke_packed, alpha);
                    }
                }
            }
        }
    }

    /// Distance from a point to a capsule (line segment + radius). Negative inside the capsule, positive outside, used as an SDF for AA.
    #[inline]
    fn distance_to_capsule(
        px: f32,
        py: f32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        radius: f32,
    ) -> f32 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;
        let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
        let t_clamped = t.clamp(0.0, 1.0);
        let cx = x1 + t_clamped * dx;
        let cy = y1 + t_clamped * dy;
        let ex = px - cx;
        let ey = py - cy;
        crate::math::sqrt(ex * ex + ey * ey) - radius
    }

    /// Pre-multiplied SWAR blend of `fg` over `bg` with explicit `alpha` (0..=256). Photon's exact pattern: widen each 32-bit pixel to 64 bits with each channel in its own 16-bit slot, do `bg*(256-α) + fg*α` in parallel, narrow back.
    #[inline]
    fn blend_swar(bg: u32, fg: u32, alpha: u64) -> u32 {
        let inv = 256 - alpha;
        let mut bg64 = bg as u64;
        bg64 = (bg64 | (bg64 << 16)) & 0x0000_FFFF_0000_FFFF;
        bg64 = (bg64 | (bg64 << 8)) & 0x00FF_00FF_00FF_00FF;
        let mut fg64 = fg as u64;
        fg64 = (fg64 | (fg64 << 16)) & 0x0000_FFFF_0000_FFFF;
        fg64 = (fg64 | (fg64 << 8)) & 0x00FF_00FF_00FF_00FF;
        let mut blended = bg64 * inv + fg64 * alpha;
        blended = (blended >> 8) & 0x00FF_00FF_00FF_00FF;
        blended = (blended | (blended >> 8)) & 0x0000_FFFF_0000_FFFF;
        blended = blended | (blended >> 16);
        blended as u32
    }
}

/// Fill a circle with a 1-pixel-wide AA edge ring. Center `(cx, cy)` and `radius` are in pixels; `colour` is straight-alpha ARGB (the AA coverage modulates the supplied alpha, so a translucent fill stays translucent at the edge). Optional `mask` multiplies into the per-pixel alpha for soft clipping.
///
/// AA via gradient-magnitude (no sqrt): for a pixel at squared distance `d²`, coverage is `(r_outer² - d²) / (r_outer² - r_inner²)` where `r_outer = radius` and `r_inner = radius - 1`. Gives a smooth 0→1 ramp across one pixel of edge.
pub fn circle_filled(
    canvas: &mut Canvas,
    cx: isize,
    cy: isize,
    radius: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if radius <= 0 {
        return;
    }
    // Damage = bbox of the circle clipped to buffer + caller clip.
    {
        let c = Clip::resolve(clip, buf_w, buf_h);
        let x0 = (cx - radius).max(c.x_start as isize).max(0) as usize;
        let y0 = (cy - radius).max(c.y_start as isize).max(0) as usize;
        let x1 = (cx + radius + 1)
            .min(c.x_end as isize)
            .min(buf_w as isize)
            .max(0) as usize;
        let y1 = (cy + radius + 1)
            .min(c.y_end as isize)
            .min(buf_h as isize)
            .max(0) as usize;
        canvas.damage.add_bounds(x0, y0, x1, y1);
    }
    let pixels: &mut [u32] = canvas.pixels;
    let clip = Clip::resolve(clip, buf_w, buf_h);
    if let Some(m) = mask {
        assert_mask_matches_buffer(m, buf_w, buf_h);
    }
    let r_outer = radius;
    let r_outer2 = r_outer * r_outer;
    let r_inner = radius - 1;
    let r_inner2 = r_inner * r_inner;
    let edge_range = r_outer2 - r_inner2;

    // Circle's bounding box, clipped. Side length is 2r + 1 (inclusive on both ends).
    let (x_min, y_min, x_max, y_max) = clip_rect(
        clip,
        cx - r_outer,
        cy - r_outer,
        2 * r_outer + 1,
        2 * r_outer + 1,
    );

    // α + darkness: top byte IS opacity; use directly. RGB darkness intact.
    let fg_opacity = (colour >> 24) & 0xFF;
    let colour_rgb = colour & 0x00FF_FFFF;

    for py in y_min..y_max {
        let dy = py as isize - cy;
        let dy2 = dy * dy;
        let base = py * buf_w;
        for px in x_min..x_max {
            let dx = px as isize - cx;
            let dist2 = dx * dx + dy2;
            if dist2 > r_outer2 {
                continue;
            }
            let coverage: u32 = if dist2 <= r_inner2 {
                256
            } else {
                (((r_outer2 - dist2) << 8) / edge_range) as u32
            };
            let scaled_opacity = (fg_opacity * coverage) >> 8;
            let idx = base + px;
            let final_opacity = match mask {
                Some(m) => (scaled_opacity * m.pixels[idx] as u32) >> 8,
                None => scaled_opacity,
            };
            let scaled_colour = colour_rgb | (final_opacity << 24);
            pixels[idx] = pixels[idx].under(scaled_colour, BlendMode::Normal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        let cases = [
            (0, 0, 0, 0),
            (255, 255, 255, 255),
            (12, 34, 56, 78),
            (200, 100, 50, 200),
        ];
        for &(r, g, b, a) in &cases {
            let p = pack_argb(r, g, b, a);
            assert_eq!(unpack_argb(p), (r, g, b, a));
        }
    }

    #[test]
    fn pack_layout_stores_alpha_in_top_byte() {
        // α + darkness storage: α = 0x12 in the top byte, RGB stored as darkness (255 − channel). pack_argb(0xAB, 0xCD, 0xEF, 0x12) → α=0x12, dark=(0x54, 0x32, 0x10).
        assert_eq!(pack_argb(0xAB, 0xCD, 0xEF, 0x12), 0x12_54_32_10);
    }

    #[test]
    fn under_fully_transparent_top_yields_bottom_within_one_lsb() {
        // Canonical empty top (α=0, dark=0 → 0x00000000) over opaque bottom → result visible ≈ bottom with ≤1 LSB drift from the >>8 truncation.
        let top = pack_argb(255, 255, 255, 0);
        let bottom = pack_argb(100, 150, 200, 255);
        let result = top.under(bottom, BlendMode::Normal);
        let (r, g, b, _) = unpack_argb(result);
        assert!((r as i32 - 100).abs() <= 1, "r got {}", r);
        assert!((g as i32 - 150).abs() <= 1, "g got {}", g);
        assert!((b as i32 - 200).abs() <= 1, "b got {}", b);
    }

    #[test]
    fn under_opaque_top_returns_top_exact() {
        // top with α=0xFF (opaque) — early-out returns top unchanged regardless of bottom.
        let top = pack_argb(50, 80, 110, 255);
        let bottom = pack_argb(100, 150, 200, 255);
        let result = top.under(bottom, BlendMode::Normal);
        assert_eq!(result, top);
        let (r, g, b, _) = unpack_argb(result);
        assert_eq!((r, g, b), (50, 80, 110));
    }

    #[test]
    fn stroke_rect_only_touches_edges() {
        // Buffer starts fully transparent (α=0, dark=0 → 0x00000000) — the canonical empty state for under-blend painting.
        let mut buf = vec![0u32; 10 * 10];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 10, 10, &mut damage);
        stroke_rect(
            &mut canvas,
            2,
            2,
            6,
            6,
            1,
            pack_argb(255, 0, 0, 255),
            None,
            None,
        );
        // Interior of the rect — never touched, stays empty.
        assert_eq!(buf[5 * 10 + 5], 0);
        let (r, _, _, _) = unpack_argb(buf[2 * 10 + 2]);
        assert!(r > 240, "top-left stroke pixel r={}", r);
        // Outside the rect — untouched, stays empty.
        assert_eq!(buf[1 * 10 + 1], 0);
    }

    #[test]
    fn circle_filled_center_is_colour() {
        let mut buf = vec![0u32; 16 * 16];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 16, 16, &mut damage);
        circle_filled(&mut canvas, 8, 8, 5, pack_argb(255, 0, 0, 255), None, None);
        let (r, g, b, _) = unpack_argb(buf[8 * 16 + 8]);
        assert!(
            r > 240 && g < 16 && b < 16,
            "center = ({}, {}, {})",
            r,
            g,
            b
        );
        // Corner outside the circle — untouched, stays empty.
        assert_eq!(buf[0], 0);
    }

    #[test]
    fn circle_filled_clips_partial_offscreen() {
        let mut buf = vec![0u32; 8 * 8];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 8, 8, &mut damage);
        circle_filled(
            &mut canvas,
            -2,
            -2,
            4,
            pack_argb(255, 255, 255, 255),
            None,
            None,
        );
        let (r, _, _, _) = unpack_argb(buf[0]);
        assert!(r > 200, "buf[0] r={}", r);
    }

    #[test]
    fn under_half_top_blends_with_opaque_bottom() {
        // Half-α white top (α=128, visible white) over opaque black bottom. Porter-Duff over: 1 − (1−0.5)×1 = 1.0 α, visible ≈ 0.5×white + 0.5×black = mid-gray (~128 per channel).
        let top = pack_argb(255, 255, 255, 128);
        let bottom = pack_argb(0, 0, 0, 255);
        let result = top.under(bottom, BlendMode::Normal);
        let (r, g, b, _) = unpack_argb(result);
        assert!((r as i32 - 128).abs() <= 2, "r got {}", r);
        assert!((g as i32 - 128).abs() <= 2, "g got {}", g);
        assert!((b as i32 - 128).abs() <= 2, "b got {}", b);
    }

    #[test]
    fn fill_rect_full_buffer() {
        // Buffer starts EMPTY (α=0, dark=0). Paint opaque visible RGB(0x11,0x22,0x33) under it. Result: ~opaque colour (1-LSB drift per channel from the >>8 normalization).
        let mut buf = vec![0u32; 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        // Visible (0x11, 0x22, 0x33) with α=0xFF → α + darkness = α=0xFF, dark=(0xEE, 0xDD, 0xCC).
        fill_rect(
            &mut canvas,
            0,
            0,
            4,
            4,
            pack_argb(0x11, 0x22, 0x33, 0xFF),
            None,
            None,
        );
        for &p in &buf {
            assert_eq!(p >> 24, 0xFF, "pixel α should be 0xFF (opaque): {p:#x}");
            // Darkness of visible 0x11 is 0xEE; minus 1-LSB drift.
            assert!(((p >> 16) & 0xFF) >= 0xEC);
        }
    }

    #[test]
    fn fill_rect_partial() {
        let mut buf = vec![0u32; 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        fill_rect(
            &mut canvas,
            1,
            1,
            2,
            2,
            pack_argb(0xAA, 0xBB, 0xCC, 0xFF),
            None,
            None,
        );
        for y in 0..4usize {
            for x in 0..4usize {
                let p = buf[y * 4 + x];
                let inside = x >= 1 && x < 3 && y >= 1 && y < 3;
                if inside {
                    assert_eq!(
                        p >> 24,
                        0xFF,
                        "inside pixel ({x},{y}) should be opaque: {p:#x}"
                    );
                } else {
                    assert_eq!(p, 0, "outside pixel ({x},{y}) untouched, got {p:#x}");
                }
            }
        }
    }

    #[test]
    fn fill_rect_clips_negative_origin() {
        let mut buf = vec![0u32; 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        fill_rect(
            &mut canvas,
            -2,
            -2,
            4,
            4,
            pack_argb(0xFF, 0xFF, 0xFE, 0xFF),
            None,
            None,
        );
        for y in 0..4usize {
            for x in 0..4usize {
                let p = buf[y * 4 + x];
                let inside = x < 2 && y < 2;
                if inside {
                    assert_eq!(
                        p >> 24,
                        0xFF,
                        "inside pixel ({x},{y}) should be opaque: {p:#x}"
                    );
                } else {
                    assert_eq!(p, 0, "outside pixel ({x},{y}) untouched");
                }
            }
        }
    }

    #[test]
    fn fill_rect_fully_offscreen_is_noop() {
        let mut buf = vec![0u32; 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        fill_rect(
            &mut canvas,
            100,
            100,
            5,
            5,
            pack_argb(0, 0, 0, 0xFF),
            None,
            None,
        );
        assert!(buf.iter().all(|&p| p == 0));
        let mut damage2 = crate::canvas::Damage::new();
        let mut canvas2 = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage2);
        fill_rect(
            &mut canvas2,
            -10,
            -10,
            5,
            5,
            pack_argb(0, 0, 0, 0xFF),
            None,
            None,
        );
        assert!(buf.iter().all(|&p| p == 0));
    }

    #[test]
    fn fill_rect_transparent_src_into_opaque_dst_is_noop() {
        // Dst is opaque (α=0xFF). Under doctrine: dst-opaque early-out fires for every pixel. Src never read.
        let mut buf = vec![pack_argb(50, 60, 70, 255); 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        fill_rect(&mut canvas, 0, 0, 4, 4, pack_argb(255, 0, 0, 0), None, None);
        // Buffer unchanged — every pixel was opaque dst, every under call early-out'd.
        assert!(buf.iter().all(|&p| p == pack_argb(50, 60, 70, 255)));
    }

    #[test]
    fn transform_identity_round_trip() {
        let p = Transform::IDENTITY.apply(3.0, 5.0);
        assert_eq!(p, (3.0, 5.0));
        assert!(Transform::IDENTITY.is_identity());
        let composed = Transform::IDENTITY.then(Transform::IDENTITY);
        assert!(composed.is_identity());
    }

    #[test]
    fn transform_rotate_quarter_turn_maps_x_to_y() {
        // 90° rotation: (1, 0) → (0, 1).
        let r = Transform::rotate(core::f32::consts::FRAC_PI_2);
        let (x, y) = r.apply(1.0, 0.0);
        assert!(x.abs() < 1e-6, "x = {}", x);
        assert!((y - 1.0).abs() < 1e-6, "y = {}", y);
    }

    #[test]
    fn transform_compose_translate_then_rotate() {
        // translate(1,0) then rotate(90°): point (0,0) → (1,0) → (0,1).
        let t =
            Transform::translate(1.0, 0.0).then(Transform::rotate(core::f32::consts::FRAC_PI_2));
        let (x, y) = t.apply(0.0, 0.0);
        assert!(x.abs() < 1e-6 && (y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn transform_aabb_of_rotated_unit_square() {
        // Unit square rotated 45° → bbox is a 1.414×1.414 square centered at origin's diagonal.
        let r = Transform::rotate(core::f32::consts::FRAC_PI_4);
        let (min_x, min_y, max_x, max_y) = r.aabb_of_rect(1.0, 1.0);
        let w = max_x - min_x;
        let h = max_y - min_y;
        assert!((w - core::f32::consts::SQRT_2).abs() < 1e-5, "w = {}", w);
        assert!((h - core::f32::consts::SQRT_2).abs() < 1e-5, "h = {}", h);
    }

    #[test]
    fn quantize_rotation_landings_for_k8() {
        // 10 px font: ceil(π × 10) = 32 (already a multiple of 8). Cardinal angles must land on bins 0, 8, 16, 24.
        let n = 32u16;
        assert_eq!(quantize_rotation(0.0, 10.0, 8), 0);
        assert_eq!(quantize_rotation(core::f32::consts::FRAC_PI_2, 10.0, 8), 8);
        assert_eq!(quantize_rotation(core::f32::consts::PI, 10.0, 8), 16);
        assert_eq!(quantize_rotation(core::f32::consts::PI * 1.5, 10.0, 8), 24);
        // Octant: 45° → bin 4, 135° → 12, etc.
        assert_eq!(quantize_rotation(core::f32::consts::FRAC_PI_4, 10.0, 8), 4);
        // Wrap-around: 2π == 0.
        assert_eq!(quantize_rotation(core::f32::consts::TAU, 10.0, 8), 0);
        // Negative angles wrap correctly via rem_euclid.
        assert_eq!(
            quantize_rotation(-core::f32::consts::FRAC_PI_2, 10.0, 8),
            n - 8
        );
    }

    #[test]
    fn quantize_rotation_scales_with_font_size() {
        // Bigger font → more divisions.
        let small = {
            let raw = crate::math::ceil(core::f32::consts::TAU * 10.0 * 0.5) as u32;
            ((raw + 7) / 8) * 8
        };
        let big = {
            let raw = crate::math::ceil(core::f32::consts::TAU * 100.0 * 0.5) as u32;
            ((raw + 7) / 8) * 8
        };
        assert!(big > small);
        // 100 px font: ceil(π × 100) = 315, ceil to multiple of 8 = 320.
        assert_eq!(big, 320);
        // 10 px font: ceil(π × 10) = 32.
        assert_eq!(small, 32);
    }
}
