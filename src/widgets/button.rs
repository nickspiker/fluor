//! Pill-shaped button widget. Visually a Textbox stripped of the editing apparatus: same squircle silhouette, same two-tone AA edge, same focus-glow, same hover tint via the host's overlay-deltas pipe; just a static centred label, no cursor / scroll / selection / keystroke routing. Adapted from `widgets::textbox` so the pill-cache + text-cache-clipped-by-`AlphaMask` pattern is identical and the two widgets read as a coherent family.
//!
//! **Action model.** Button doesn't carry a callback or an action enum — it owns a click counter. App polls [`Button::take_click`] each frame; `true` means the button fired since last poll. That's the smallest stateful API that decouples "the widget knows it was clicked" from "the app knows what to do about it" without dragging closures + lifetime juggling into the widget. For richer action dispatch (per-button intent codes, multi-target routing), the app keeps a `Vec<(HitId, Action)>` keyed by `button.hit_id()` (a linear scan over a handful of buttons) and matches at the dispatch layer.

use crate::canvas::PixelRect;
use crate::text::TextStyle;
use crate::coord::Coord;
use crate::paint::{self, Clip, HitId};
use crate::region::Region;
use crate::text::TextRenderer;
use crate::pixel::{Argb8, Pixel};
use crate::theme;
use crate::widgets::textbox::{blit_cache_to_target, region_to_pixelrect};
use alloc::string::String;
use alloc::vec::Vec;

pub struct Button<P: Pixel = Argb8> {
    /// Allocated at construction; stamped into the host hit map at every opaque pill pixel.
    hit_id: HitId,
    /// Where the last paint showed ([`paint::Shown`]): the hit claim rides it, so the widget claims exactly the pixels it shows on and none it does not.
    shown: paint::Shown,
    /// Label text — single line, no editing. Mutate via [`Self::set_label`] so caches invalidate.
    label: String,
    font: &'static str,

    /// Stroke thickness as a fraction of `font_size`. Final pixel width = `(stroke_ru × font_size) as isize` — NO floor: the two-tone edge is the ring between the outer squircle and the inset inner fill, and both are anti-aliased silhouettes, so below 1px the ring just truncates to nothing and the pill reads as a clean filled shape with a soft AA edge (no fixed-pixel hairline needed — that's only for bare LINE strokes with no fill behind them, e.g. a chevron). Default `1.0 / (1 << 5)` (= 1/32 of font_size) yields 0 (borderless) thru typical desktop and a growing ring on 4K + zoom; same convention as Textbox so a Button and Textbox at the same `stroke_ru` render identically.
    pub stroke_ru: f32,
    pub center_x: Coord,
    pub center_y: Coord,
    pub width: Coord,
    pub height: Coord,
    pub font_size: Coord,

    /// `true` while the button is the focused widget (Tab / click). Drives the glow and the active fill colour.
    focused: bool,
    /// `true` while the cursor is over the button. Drives the hover fill colour via the host's overlay-delta pipe (Button doesn't bake hover into its own cache — same pattern as Textbox).
    hovered: bool,
    /// `true` while a pointer is DOWN on this button and a release here would fire it (press-hold-release — see [`crate::host::pointer`]). Set by [`crate::host::widget::apply_pressed`] from the host arbiter; drives the [`theme::BUTTON_HELD`] "held" fill via the overlay-delta pipe, ranked above hover and focus. Clears on release or drag-off.
    pressed: bool,
    /// `true` (default) while the button accepts input. When `false` the [`crate::host::widget::Widget`] capability accessors return `None`, so dispatch + tab-cycle skip it — the "frozen sibling" of a disabled textbox (an Attest / plus button that goes inert while its query is in flight). Visual unchanged; inert, not greyed. Toggle via [`Self::set_enabled`].
    enabled: bool,

    /// Optional per-instance hovered fill colour (α+darkness), replacing the shared [`theme::BUTTON_HOVER`] for this button only. `None` = use the default. Lets a host give specific controls their own hover colour (the pre-fluor model, where the send/query button had a distinct subtle tint from a generic action button). Set via [`Self::set_hover_fill`].
    hover_fill: Option<u32>,

    /// Optional per-instance resting fill colour (α+darkness) for the pill interior, replacing the shared [`theme::BUTTON_FILL`] for this button only. `None` = use the default theme fill. Lets a host tint a specific control (e.g. an orange "nuke" button) without changing the shared theme. Baked into `pill_cache`, so [`Self::set_fill`] dirties the cache. Set via [`Self::set_fill`].
    fill: Option<u32>,

    /// Optional per-instance PRESSED ("held") fill (α+darkness), replacing [`theme::BUTTON_HELD`] / the derived ramp. `None` = default. Set via [`Self::set_held_fill`]. Read by [`Self::effective_fill`] and, in overlay mode, by `tint_delta`.
    held_fill: Option<u32>,

    /// When `true`, `render_content_into` bakes the current STATE fill ([`Self::effective_fill`]) directly into the pill instead of only the idle fill — for headless hosts (e.g. toka in the browser) that don't run the host's overlay-delta hover pass. When `false` (default, desktop/Android), the pill bakes the idle fill and hover/pressed come from the overlay via `tint_delta`. Set via [`Self::set_bake_states`].
    bake_states: bool,

    /// When `true` the silhouette is a circle (an ellipse in a box that is not square): every corner at half the height, traced at exponent 2, in place of the house pill's asymmetric squircle. For a control that is a bead rather than a pill (photon's Wave and Beam beads round the orb on ferros, owner 2026-11-18). Set via [`Self::set_round`].
    round: bool,

    /// Last effective fill actually rasterized into `pill_cache` — lets baked-state mode re-dirty the cache only when the state fill genuinely changes (hover/press transition), not every frame.
    last_baked_fill: Option<u32>,

    /// Number of times [`Click::on_click`] has fired since construction. Monotonic. Consumers compare against [`Self::last_seen_click_counter`] via [`Self::take_click`] to know "has this button fired since I last looked."
    click_counter: u32,
    last_seen_click_counter: u32,

    // --- Caches (same layering as Textbox: pill_cache holds squircle bg + AA edges, text_cache holds the label glyphs clipped to the inner-pill silhouette, both blit topmost-first into target via under()) ---
    pill_cache: Vec<P>,
    pill_cache_w: usize,
    pill_cache_h: usize,
    pill_cache_dirty: bool,
    /// The look generation the caches were drawn under ([`crate::theme::look_generation`]): a new look redraws them.
    look_gen: u32,
    text_cache: Vec<P>,
    text_cache_w: usize,
    text_cache_h: usize,
    text_cache_dirty: bool,
    inner_pill_mask: Vec<u8>,

    // --- Damage protocol ---
    last_painted_bbox: Option<PixelRect>,
    last_painted_focused: bool,
    last_painted_hovered: bool,
}

impl Button {
    /// The 8-bit widget; [`Button::with_depth`] builds one at another pixel depth.
    /// `new` lives on the 8-bit type alone so `Button::new(..)` needs no annotation where nothing else fixes the depth.
    pub fn new(
        hit_counter: &mut HitId,
        center_x: Coord,
        center_y: Coord,
        width: Coord,
        height: Coord,
        font_size: Coord,
        label: impl Into<String>,
    ) -> Self {
        Self::with_depth(hit_counter, center_x, center_y, width, height, font_size, label)
    }
}

impl<P: Pixel> Button<P> {
    /// `hit_counter` is the app's monotonic [`HitId`] allocator (see [`crate::host::widget::next_id`]). Each Button claims one ID at construction.
    pub fn with_depth(
        hit_counter: &mut HitId,
        center_x: Coord,
        center_y: Coord,
        width: Coord,
        height: Coord,
        font_size: Coord,
        label: impl Into<String>,
    ) -> Self {
        Self {
            hit_id: crate::host::widget::next_id(hit_counter),
            shown: paint::Shown::default(),
            label: label.into(),
            font: "Open Sans",
            stroke_ru: 1.0 / (1 << 5) as f32,
            center_x,
            center_y,
            width,
            height,
            font_size,
            focused: false,
            hovered: false,
            pressed: false,
            enabled: true,
            hover_fill: None,
            fill: None,
            held_fill: None,
            bake_states: false,
            round: false,
            last_baked_fill: None,
            click_counter: 0,
            last_seen_click_counter: 0,
            pill_cache: Vec::new(),
            pill_cache_w: 0,
            pill_cache_h: 0,
            pill_cache_dirty: true,
            look_gen: crate::theme::look_generation(),
            text_cache: Vec::new(),
            text_cache_w: 0,
            text_cache_h: 0,
            text_cache_dirty: true,
            inner_pill_mask: Vec::new(),
            last_painted_bbox: None,
            last_painted_focused: false,
            last_painted_hovered: false,
        }
    }

    pub fn hit_id(&self) -> HitId {
        self.hit_id
    }
    pub fn is_focused(&self) -> bool {
        self.focused
    }
    pub fn is_hovered(&self) -> bool {
        self.hovered
    }
    /// Give this button a specific hovered fill colour (α+darkness), instead of the shared
    /// [`theme::BUTTON_HOVER`]. `None` restores the default. Used to reproduce the pre-fluor per-control hover colours (e.g. a subtle neutral tint on the send/query button).
    pub fn set_hover_fill(&mut self, colour: Option<u32>) {
        self.hover_fill = colour;
    }

    /// Give this button a specific resting pill fill colour (α+darkness), instead of the shared
    /// [`theme::BUTTON_FILL`]. `None` restores the default. Dirties the pill cache so the new colour rasterizes on the next paint.
    pub fn set_fill(&mut self, colour: Option<u32>) {
        if colour != self.fill {
            self.fill = colour;
            self.pill_cache_dirty = true;
        }
    }

    /// Give this button a specific PRESSED ("held") fill (α+darkness), instead of [`theme::BUTTON_HELD`]
    /// / the derived ramp. `None` restores the default.
    pub fn set_held_fill(&mut self, colour: Option<u32>) {
        self.held_fill = colour;
    }

    /// Select baked-state rendering (`true`) vs. host-overlay hover/pressed (`false`, default). See the
    /// `bake_states` field. Headless hosts (toka) set `true`; the desktop/Android host leaves it `false`.
    pub fn set_bake_states(&mut self, bake: bool) {
        self.bake_states = bake;
    }

    /// A circle (`true`) or the house pill (`false`, default). See the `round` field. Changing it redraws the silhouette and the label clipped to it.
    pub fn set_round(&mut self, round: bool) {
        if round != self.round {
            self.round = round;
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
        }
    }

    /// Ratios that turn a resting fill into its hover / held variant — chosen so a default
    /// [`theme::BUTTON_FILL`] scales to ≈[`theme::BUTTON_HOVER`] / ≈[`theme::BUTTON_HELD`], and any custom idle fill inherits the same luminance ramp.
    const HOVER_FACTOR: (u16, u16) = (3, 2); // ×1.5
    const HELD_FACTOR: (u16, u16) = (11, 5); // ×2.2

    /// The pill fill for the button's CURRENT state, precedence pressed → focused → hovered → idle.
    /// Each state prefers its explicit override; an unset override on a *custom-idle* button derives from the idle fill (so a coloured button gets a matching ramp), while a default button (no idle override) falls back to the exact `theme::BUTTON_*` constants — pixel-identical to today.
    fn effective_fill(&self) -> u32 {
        let idle = self.fill.unwrap_or(crate::theme::look().button);
        if self.pressed {
            self.held_fill.unwrap_or_else(|| match self.fill {
                Some(_) => crate::paint::scale_brightness(idle, Self::HELD_FACTOR.0, Self::HELD_FACTOR.1),
                None => crate::theme::look().button_held,
            })
        } else if self.focused {
            crate::theme::look().button_active
        } else if self.hovered {
            self.hover_fill.unwrap_or_else(|| match self.fill {
                Some(_) => crate::paint::scale_brightness(idle, Self::HOVER_FACTOR.0, Self::HOVER_FACTOR.1),
                None => crate::theme::look().button_hover,
            })
        } else {
            idle
        }
    }
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Set the focused state. Idempotent. Side-effects nothing else — the painter consults `focused` directly to decide whether to draw the glow + active fill.
    pub fn set_focused(&mut self, focused: bool) {
        if focused != self.focused {
            self.focused = focused;
        }
    }

    pub fn set_hovered(&mut self, hovered: bool) {
        // No-op while disabled: the disabled button still stamps its hit silhouette, so a hover over it must not re-light it.
        if !self.enabled {
            return;
        }
        if hovered != self.hovered {
            self.hovered = hovered;
        }
    }

    /// Set the "held" (pointer-down) state. No-op while disabled (a disabled button still stamps its hit silhouette, but must never light up). Idempotent; the painter's `tint_delta` reads `pressed` to pick [`theme::BUTTON_HELD`].
    pub fn set_pressed(&mut self, pressed: bool) {
        if !self.enabled {
            return;
        }
        if pressed != self.pressed {
            self.pressed = pressed;
        }
    }

    /// `true` (default) while the button accepts input.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable / disable input. Disabling drops the button out of click / key / focus / hover dispatch (the [`crate::host::widget::Widget`] accessors return `None`) and clears `focused` + `hovered`. As with [`Textbox::set_enabled`], move focus elsewhere first if this button currently holds it.
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled == self.enabled {
            return;
        }
        self.enabled = enabled;
        if !enabled {
            self.focused = false;
            self.hovered = false;
        }
        // The label colour follows enabled (dim when off — see render), so re-rasterize the glyphs.
        self.text_cache_dirty = true;
    }

    /// Replace the label text. Marks both caches dirty so the next render re-rasterizes glyphs (text_cache) — the pill silhouette doesn't depend on label content so technically only text_cache needs it, but a label swap typically goes with a re-layout, and marking pill dirty too is cheap insurance.
    pub fn set_label(&mut self, label: impl Into<String>) {
        let new = label.into();
        if new != self.label {
            self.label = new;
            self.text_cache_dirty = true;
        }
    }

    /// Replace the label font family (e.g. "Oxanium" for the dozenal control-block glyphs Open Sans lacks). Dirties the text cache so the glyphs re-rasterize in the new face. `&'static str` because font names live for the program.
    pub fn set_font_family(&mut self, font: &'static str) {
        if font != self.font {
            self.font = font;
            self.text_cache_dirty = true;
        }
    }

    /// Reposition the pill. No scroll state to reconcile (Button has no horizontal scroll) so just dirty the caches and store.
    pub fn set_rect(&mut self, center_x: Coord, center_y: Coord, width: Coord, height: Coord) {
        if self.center_x != center_x
            || self.center_y != center_y
            || self.width != width
            || self.height != height
        {
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
        }
        self.center_x = center_x;
        self.center_y = center_y;
        self.width = width;
        self.height = height;
    }

    pub fn set_font_size(&mut self, font_size: Coord) {
        if self.font_size != font_size {
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
        }
        self.font_size = font_size;
    }

    /// Set the label size, SHRUNK so the label and its padding fit the pill's current width — the fit rule [`Self::draw_pill_immediate`] uses for immediate pills, for a retained one: never grown past `desired`, floored by nothing, so a narrow slot yields smaller text instead of a clipped label. Call after [`Self::set_rect`].
    pub fn set_font_size_fitted(&mut self, desired: Coord, text: &mut TextRenderer) {
        let tw = text.measure_text(&self.label, &TextStyle::new(desired, 0).font(self.font));
        let need = tw + desired * 1.6;
        let size = if need > self.width { desired * self.width / need } else { desired };
        self.set_font_size(size);
    }

    /// Returns `true` if the button has been clicked since the last call to `take_click`. The internal counter is monotonic; this call advances the consumer's read pointer to the current value. Multiple clicks between polls coalesce into a single `true` — buttons aren't a rate-limited event source, they're "did the user activate this thing recently."
    pub fn take_click(&mut self) -> bool {
        if self.click_counter != self.last_seen_click_counter {
            self.last_seen_click_counter = self.click_counter;
            true
        } else {
            false
        }
    }

    /// Raw counter — total clicks since construction. Public for apps that want to display a counter or detect rapid clicks via deltas.
    pub fn click_count(&self) -> u32 {
        self.click_counter
    }

    /// Mark the click counter advanced. Called from the Click trait impl; pub(crate) so internal consumers can fire it without going thru the trait if they need to (e.g. a hypothetical keyboard accelerator outside Key::on_key).
    pub(crate) fn fire(&mut self) {
        self.click_counter += 1;
    }

    /// Symmetric inner padding inside the pill, in pixels. Same model as Textbox so a Button and Textbox at the same `font_size` have visually identical pill margins.
    pub fn padding(&self) -> Coord {
        self.font_size * 0.4
    }

    pub fn bbox(&self) -> Region {
        Region::new(
            self.center_x - self.width * 0.5,
            self.center_y - self.height * 0.5,
            self.width,
            self.height,
        )
    }

    /// Glow envelope — wider than `bbox` because the focus glow rays extend past the pill. Used by `damage_rect` on focus-on/off transitions so the host's damage clip covers the area the glow will paint into. Pad sizes come from [`paint::ray_reach_px`], the canonical "how many pixels does this seed decay across at this factor" helper — same numbers the rasterizer actually walks — so the bbox contains exactly what gets painted, with no early cutoff and no over-clearing.
    pub fn glow_bbox(&self) -> Region {
        let horiz_factor = glow_factor_256(self.font_size, 1.5);
        let vert_factor = glow_factor_256(self.font_size, 0.75);
        let horiz_pad = paint::ray_reach_px(0x80, horiz_factor) as f32;
        let vert_pad = paint::ray_reach_px(0x40, vert_factor) as f32;
        Region::new(
            self.center_x - self.width * 0.5 - horiz_pad,
            self.center_y - self.height * 0.5 - vert_pad,
            self.width + 2.0 * horiz_pad,
            self.height + 2.0 * vert_pad,
        )
    }

    /// Damage region. Returns `None` if nothing has changed since the last paint (host can persist scratch). Returns `Some(rect)` covering whichever of (current bbox, prior bbox, glow envelope) need a fresh paint. Glow envelope enters damage only on focus on / off transitions, matching Textbox's per-keystroke optimisation that keeps steady-state damage tight to the bare pill.
    pub fn damage_rect(&self, viewport_w: usize, viewport_h: usize) -> Option<PixelRect> {
        let focus_changed = self.focused != self.last_painted_focused;
        let hover_changed = self.hovered != self.last_painted_hovered;
        let pill_dirty = self.pill_cache_dirty || self.text_cache_dirty;
        // Never painted: nothing on screen to clear (see Textbox::damage_rect) — a button in the app's walk but not yet revealed (attest before a handle is typed, the contacts + before a search) must not leak its dirty-from-birth caches as phantom damage.
        if self.last_painted_bbox.is_none() {
            return None;
        }
        if !pill_dirty && !focus_changed && !hover_changed {
            return None;
        }
        let mut combined: Option<PixelRect> = None;
        if let Some(prev) = self.last_painted_bbox {
            combined = Some(prev);
        }
        let current = if focus_changed {
            region_to_pixelrect(self.glow_bbox(), viewport_w, viewport_h)
        } else {
            region_to_pixelrect(self.bbox(), viewport_w, viewport_h)
        };
        combined = Some(combined.map_or(current, |c| c.union(current)));
        combined
    }

    /// Paint the button into `canvas` at its viewport-space `center_*` / `width` / `height`, claiming `hit_id` in `hit_map` at every pixel it shows on (label, pill, rim, glow) that nothing in front claimed.
    /// Same three-step blit as Textbox: text_cache (glyphs) → pill_cache (squircle bg + AA edges) → focus glow behind.
    /// No selection rect, no blinkey: just label + pill + optional glow.
    pub fn render_content_into(
        &mut self,
        canvas: &mut crate::canvas::Canvas<P>,
        offset_x: Coord,
        offset_y: Coord,
        text: &mut TextRenderer,
        clip: Option<Clip>,
        hit_map: Option<&mut [HitId]>,
        hit_id: HitId,
    ) {
        let pill_x_target = (self.center_x - self.width * 0.5 - offset_x) as isize;
        let pill_y_target = (self.center_y - self.height * 0.5 - offset_y) as isize;
        let pill_w = self.width as isize;
        let pill_h = self.height as isize;
        if pill_w <= 0 || pill_h <= 0 {
            return;
        }
        let cw = pill_w as usize;
        let ch = pill_h as usize;
        // A look changed since the caches were drawn (a platform theme set: the device's, as the person changed it): drawn again in it.
        let look_gen = crate::theme::look_generation();
        if look_gen != self.look_gen {
            self.look_gen = look_gen;
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
        }
        // Fractional squirdleyness — slots between an ellipse (2) and a diamond (1). `1.5` reads as a noticeably-rounder, slightly-faceted pill: distinctly more curved than the textbox's `3.0` "slightly squared" but not as soft as a full ellipse. Routes thru paint's `_f` (powf) variant; the textbox / chrome keep the integer (powi) fast path. Adjustable per-instance via this constant; future API could expose it as a Button field if more shapes are desired.
        let squirdleyness = if self.round { 2. } else { paint::SHAPE_POWER as f32 };
        // The corners at a radius: the house split (TL+BR deep, TR+BL half), or every corner at it for a circle.
        let round = self.round;
        let radii = move |r: f32| if round { [r; 4] } else { paint::asymmetric_radii(r) };
        // The `+ 1` idiom every other widget uses (textbox.rs): a 1 px floor on the ring. Without it the ring truncates to 0 below a 32 px font, the inner fill (painted first, so on top under under-blend) covers the whole pill, and the two-tone edge vanishes — the contact-screen action pills lost their hairline while Send kept its (2026-09-10).
        let stroke_px = (self.stroke_ru * self.font_size) as isize + 1;

        // Fill for THIS paint: baked-state mode folds hover/pressed/focus into the pill fill (headless hosts have no overlay pass); otherwise just the idle fill and the overlay tints on top. A state transition re-dirties the cache so only the fill re-rasterizes — stroke is unchanged.
        let pill_fill = if self.bake_states {
            self.effective_fill()
        } else {
            self.fill.unwrap_or(crate::theme::look().button)
        };
        if self.bake_states && self.last_baked_fill != Some(pill_fill) {
            self.pill_cache_dirty = true;
        }

        // --- Pill cache rasterize: squircle fill + AA edges, gated by pill_cache_dirty. Identical to Textbox's pill rasterize so the two widgets share the visual family. ---
        if self.pill_cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.pill_cache.clear();
            self.pill_cache.resize(cw * ch, P::EMPTY);
            self.pill_cache_w = cw;
            self.pill_cache_h = ch;
            let inner_x = stroke_px;
            let inner_y = stroke_px;
            let inner_w = pill_w - 2 * stroke_px;
            let inner_h = pill_h - 2 * stroke_px;
            let mut cache_damage = crate::canvas::Damage::new();
            {
                let mut cache_canvas =
                    crate::canvas::Canvas::new(&mut self.pill_cache, cw, ch, &mut cache_damage);
                if inner_w > 0 && inner_h > 0 {
                    // Asymmetric corners like the window perimeter: TL+BR at the pill's natural radius, TR+BL at half (Nick 2026-09-09); the inner fill's corners sit one stroke inside the edge's.
                    paint::draw_squircle_rrect_f(
                        &mut cache_canvas,
                        inner_x,
                        inner_y,
                        inner_w,
                        inner_h,
                        pill_fill,
                        radii(pill_h as f32 * 0.5 - stroke_px as f32),
                        squirdleyness,
                    );
                }
            }
            self.inner_pill_mask.clear();
            self.inner_pill_mask.resize(cw * ch, 0);
            for i in 0..(cw * ch) {
                // The mask is bytes at every depth: the top byte of α.
                self.inner_pill_mask[i] = (self.pill_cache[i].alpha() >> (P::DEPTH - 8)) as u8;
            }
            {
                let mut cache_canvas =
                    crate::canvas::Canvas::new(&mut self.pill_cache, cw, ch, &mut cache_damage);
                // Lit from the top left (the ferros owner, 2026-11-09: "TL should be brightish like the keymap and the bottom right should be darker"): light above the anti-diagonal, shadow below, as the honeycomb's keys and the textbox. The swap that put the shadow on top as "raised" read sunken on the glass.
                paint::draw_squircle_rrect_two_tone_f(
                    &mut cache_canvas,
                    0,
                    0,
                    pill_w,
                    pill_h,
                    crate::theme::look().light_edge,
                    crate::theme::look().shadow_edge,
                    radii(pill_h as f32 * 0.5),
                    squirdleyness,
                    None,
                    0,
                );
            }
            self.pill_cache_dirty = false;
            self.last_baked_fill = Some(pill_fill);
        }

        // --- Text cache: centred label clipped by inner-pill mask. No scroll, no selection — anchor at pill centre, draw_text_left_u32 with x = centre − text_width / 2. ---
        if self.text_cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.text_cache.clear();
            self.text_cache.resize(cw * ch, P::EMPTY);
            self.text_cache_w = cw;
            self.text_cache_h = ch;
            if !self.label.is_empty() && self.font_size > 0.0 {
                let tw = text.measure_text(&self.label, &TextStyle::new(self.font_size, 0).font(self.font));
                let local_text_left = pill_w as Coord * 0.5 - tw * 0.5;
                let local_y_center = pill_h as Coord * 0.5;
                let mut text_damage = crate::canvas::Damage::new();
                let mut text_canvas =
                    crate::canvas::Canvas::new(&mut self.text_cache, cw, ch, &mut text_damage);
                let mask_buffer = paint::AlphaMask::new(&self.inner_pill_mask, cw, ch);
                // Disabled buttons dim their label to the secondary grey (inert, still legible) — the hand-rolled pills did this and callers expect it.
                let label_colour = if self.enabled { crate::theme::look().text } else { crate::theme::look().label };
                text.draw_text_left(&mut text_canvas, &self.label, local_text_left, local_y_center, &TextStyle::new(self.font_size, label_colour).font(self.font), None, Some(&mask_buffer));
            }
            self.text_cache_dirty = false;
        }

        // The hits ride the paint: where the α stood before the label, the pill and the glow, so the button claims exactly the pixels it shows on.
        let reach = crate::widgets::textbox::reach_of(if self.focused { self.glow_bbox() } else { self.bbox() }, offset_x, offset_y, canvas.width, canvas.height);
        self.shown.before(canvas.pixels, canvas.width, canvas.height, reach);
        // --- Composition: text (topmost) → pill (bottom) ---
        blit_cache_to_target(
            &self.text_cache,
            self.text_cache_w,
            self.text_cache_h,
            pill_x_target,
            pill_y_target,
            canvas,
            clip,
        );
        blit_cache_to_target(
            &self.pill_cache,
            self.pill_cache_w,
            self.pill_cache_h,
            pill_x_target,
            pill_y_target,
            canvas,
            clip,
        );

        // --- Focus glow on target, paints fresh each frame. Same four-direction RU-invariant decay as Textbox. ---
        if self.focused {
            let horiz_factor = glow_factor_256(self.font_size, 1.5);
            let vert_factor = glow_factor_256(self.font_size, 0.75);
            paint::apply_textbox_glow_right(
                canvas,
                pill_x_target,
                pill_y_target,
                pill_w,
                pill_h,
                theme::GLOW_DARK,
                0x80,
                horiz_factor,
                clip,
            );
            paint::apply_textbox_glow_left(
                canvas,
                pill_x_target,
                pill_y_target,
                pill_w,
                pill_h,
                theme::GLOW_DARK,
                0x80,
                horiz_factor,
                clip,
            );
            paint::apply_textbox_glow_top(
                canvas,
                pill_x_target,
                pill_y_target,
                pill_w,
                pill_h,
                theme::GLOW_DARK,
                0x40,
                vert_factor,
                clip,
            );
            paint::apply_textbox_glow_bottom(
                canvas,
                pill_x_target,
                pill_y_target,
                pill_w,
                pill_h,
                theme::GLOW_DARK,
                0x40,
                vert_factor,
                clip,
            );
        }

        // Record what we painted so next frame's damage_rect can union prev∪current.
        let viewport_w = canvas.width;
        let viewport_h = canvas.height;
        self.last_painted_bbox = Some(region_to_pixelrect(self.bbox(), viewport_w, viewport_h));
        // Where the button showed (label, pill, its anti-aliased rim, its glow), settled whether or not a map came with this paint: `stamp_hit_into` claims from the same mask later.
        self.shown.settle(canvas.pixels);
        if let Some(map) = hit_map {
            self.shown.claim_shape(map, hit_id, crate::widgets::textbox::cache_covers(&self.pill_cache, self.pill_cache_w, self.pill_cache_h, pill_x_target, pill_y_target));
        }
        self.last_painted_focused = self.focused;
        self.last_painted_hovered = self.hovered;
    }

    /// Immediate-mode pill draw (no retained cache): THE canonical Button look — squircle fill + two-tone raised edge + centred label, fit to `rect`, hit-stamped — for a host with MANY one-off action pills where a persistent Button per pill isn't worth the click-counter state (photon's settings panel). Visually indistinguishable from a retained Button because it draws with the same primitives, the same shared `SHAPE_POWER`, the same two-tone edge, and the same no-floor `1/32·font_size` stroke.
    /// State fill mirrors [`Self::effective_fill`] precedence (pressed > hover > idle) using the same brightness factors, so an immediate pill hovers and presses identically to a retained one — this is the hover the hand-rolled pills never had. `fill = None` uses the theme navy; `Some((idle, held))` overrides for a ramp (Security's green→red destructiveness). Disabled dims the label to the theme grey; pass `hit_map = None` for a disabled pill so it stamps nothing and a dead tap dispatches nowhere.
    /// Fit-to-slot: the font shrinks so a label never overruns its slot (pills sharing a row don't collide), scaling all the way down; the pill grows only when the label needs more than `rect.w` (never shrinks the pill).
    #[allow(clippy::too_many_arguments)]
    pub fn draw_pill_immediate(
        canvas: &mut crate::canvas::Canvas<P>,
        text: &mut TextRenderer,
        hit_map: Option<&mut [HitId]>,
        buf_w: usize,
        buf_h: usize,
        rect: Region,
        label: &str,
        font: &'static str,
        hit_id: HitId,
        hovered: bool,
        pressed: bool,
        enabled: bool,
        fill: Option<(u32, u32)>,
    ) {
        if rect.w <= 0.0 || rect.h <= 0.0 {
            return;
        }
        let idle = fill.map(|f| f.0).unwrap_or(crate::theme::look().button);
        // State fill — same precedence and factors as effective_fill (no focus for immediate pills). Only an enabled pill lights up; a disabled one stays at idle.
        let state_fill = if pressed && enabled {
            fill.map(|f| f.1).unwrap_or_else(|| match fill {
                Some(_) => paint::scale_brightness(idle, Self::HELD_FACTOR.0, Self::HELD_FACTOR.1),
                None => crate::theme::look().button_held,
            })
        } else if hovered && enabled {
            match fill {
                Some(_) => paint::scale_brightness(idle, Self::HOVER_FACTOR.0, Self::HOVER_FACTOR.1),
                None => crate::theme::look().button_hover,
            }
        } else {
            idle
        };
        // Fit order: SHRINK the font toward the slot so pills sharing a row don't collide; the shrink is capped at the original size (never grows a label) and floored by nothing — a smaller slot yields smaller text, scaling all the way down like everything else.
        let mut font_size = rect.h * 0.5;
        let mut tw = text.measure_text(label, &TextStyle::new(font_size, 0).font(font));
        let max_w = rect.w * 0.96;
        if tw + font_size * 1.6 > max_w {
            let scaled = font_size * max_w / (tw + font_size * 1.6);
            font_size = scaled;
            tw = text.measure_text(label, &TextStyle::new(font_size, 0).font(font));
        }
        let need_w = tw + font_size * 1.6;
        let (px, pw) = if need_w > rect.w {
            (rect.center_x() - need_w * 0.5, need_w)
        } else {
            (rect.x, rect.w)
        };
        let w = pw as isize;
        let h = rect.h as isize;
        let x0 = px as isize;
        let y0 = rect.y as isize;
        // Same stroke_ru default as a retained Button (1/32 of font_size) with the same `+ 1` floor: below a 32 px font a floorless ring truncates to 0 and the fill hides the edge entirely (2026-09-10).
        let stroke = (font_size * (1.0 / (1u32 << 5) as f32)) as isize + 1;
        // Disabled dims the label to the secondary grey (inert, still legible) — same fill and edges, so the pill reads "present but inert" rather than vanished. Matches a retained Button's disabled label.
        let label_colour = if enabled {
            crate::theme::look().text
        } else {
            crate::theme::look().label
        };
        // THE HIT CLAIM RIDES THE PAINT (photon 2026-10-01; the owner 2026-11-17: "if it's showing on screen, it NEEDS to be hitting the hitmap, if it's already opaque and won't show? SKIP!").
        // The α over the pill before the label and the pill go in, compared after: the pill claims its declared shape, its silhouette, which is exactly what it draws (fill and two-tone rim, the label inside it), so the pixels it showed on are that shape; none already opaque before it (the chrome's strip or orb, flattened first), none outside its silhouette, and none something in front claimed.
        // Until 2026-11-17 it claimed its whole bbox wherever the pixel was not yet opaque, the corners outside the silhouette included.
        // A disabled pill passes hit_map = None and claims nothing.
        let mut shown = hit_map.is_some().then(paint::Shown::default);
        if let Some(s) = shown.as_mut() {
            // The pill holds its slot (it only grows, centred on it) and the label sits in the slot, so the pill and its rim are the reach.
            let reach = paint::reach_rect(x0 - 1, y0 - 1, x0 + w + 1, y0 + h + 1, buf_w, buf_h);
            s.before(canvas.pixels, buf_w, buf_h, reach);
        }
        // Label first (topmost under-blend), centred in the SLOT (not the grown pill) so it stays put.
        text.draw_text_left(
            canvas,
            label,
            rect.center_x() - tw * 0.5,
            rect.center_y(),
            &TextStyle::new(font_size, label_colour).font(font),
            None,
            None,
        );
        let inner_w = w - 2 * stroke;
        let inner_h = h - 2 * stroke;
        if inner_w > 0 && inner_h > 0 {
            // Asymmetric corners (TL+BR deep, TR+BL half) — the same split the retained Button and the window perimeter carry.
            paint::draw_squircle_rrect_f(
                canvas,
                x0 + stroke,
                y0 + stroke,
                inner_w,
                inner_h,
                state_fill,
                paint::asymmetric_radii(h as f32 * 0.5 - stroke as f32),
                paint::SHAPE_POWER as f32,
            );
        }
        // Two-tone edge lit from the top left: light top/left, shadow bottom/right (ferros owner, 2026-11-09).
        paint::draw_squircle_rrect_two_tone_f(
            canvas,
            x0,
            y0,
            w,
            h,
            crate::theme::look().light_edge,
            crate::theme::look().shadow_edge,
            paint::asymmetric_radii(h as f32 * 0.5),
            paint::SHAPE_POWER as f32,
            None,
            0,
        );
        if let (Some(s), Some(map)) = (shown.as_mut(), hit_map) {
            s.settle(canvas.pixels);
            s.claim(map, hit_id);
        }
    }

    /// Claim this button's `hit_id` in `hit_map` by its last paint: its declared shape (the pill's silhouette) wherever the pixel was not already opaque before it (a layer in front hid it there), and its glow wherever it showed.
    /// It is the button's hits as a step of their own, for a pass that painted it with `hit_map = None`: call it right after that paint, before anything behind it claims, since something drawn behind a translucent part of the button claims there first otherwise.
    /// A pixel something in front already claimed stays theirs, so overlap is settled by what is in front, never by which call came last.
    /// Until 2026-11-17 this overwrote what was there and its doc said to call it after the widgets the button overlaps so it would "re-win" as the last writer, which let any later widget take the button's clicks (the owner: "that's real bad"); until the same day's second pass it claimed every pixel the pill's cache covered, shown or not.
    pub fn stamp_hit_into(&self, hit_map: &mut [HitId], buf_w: usize, buf_h: usize, hit_id: HitId) {
        // A button never painted has an empty mask (width 0) and shows nowhere, so it claims nothing; otherwise the mask indexes the map at the width the paint drew at, and a map of another width is another frame (fail loud).
        if self.shown.width() == 0 {
            return;
        }
        assert_eq!(self.shown.width(), buf_w, "stamp_hit_into on a map {buf_w} wide after a paint {} wide", self.shown.width());
        let _ = buf_h;
        // The pill where the last paint drew it (an offset paint draws it shifted; this claims at the viewport place, as it always did).
        let (ox, oy) = ((self.center_x - self.width * 0.5) as isize, (self.center_y - self.height * 0.5) as isize);
        self.shown.claim_shape(hit_map, hit_id, crate::widgets::textbox::cache_covers(&self.pill_cache, self.pill_cache_w, self.pill_cache_h, ox, oy));
    }
}

/// Same RU-invariant glow factor as Textbox's [`Textbox::glow_factor_256`]; duplicated here so Button doesn't pull a `pub(crate)` from Textbox just for one tiny formula. Returns a factor in `[0, 255]` (the `as u8` cast saturates both ends; 0 at a tiny font turns the glow off) where smaller = steeper decay = shorter visual reach; matches the chrome shadow's `factor_256` math so glows around buttons and textboxes share the same look.
fn glow_factor_256(font_size: f32, radius_scale: f32) -> u32 {
    let target_radius = font_size * radius_scale;
    let drop = (1240.0 / target_radius) as u32;
    (256.0 - drop as f32) as u8 as u32
}

mod widget_impls {
    //! [`crate::host::widget`] capability traits for [`Button`]. Mirrors `textbox::widget_impls`: Click increments the action counter, Focus / Hover route thru Button's setters, Key activates on Enter / Space. Speaks fluor-native event types so the same impls work on every host (host-winit on desktop, host-android on Android).

    use super::Button;
    use crate::pixel::Pixel;
    use crate::coord::Coord;
    use crate::event::{ElementState, Key as FKey, KeyEvent, ModifiersState, NamedKey};
    use crate::host::widget::{Click, Focus, Hover, Key, PaintCtx, Widget};
    use crate::paint::HitId;
    use crate::text::TextRenderer;

    impl<P: Pixel> Widget for Button<P> {
        fn id(&self) -> HitId {
            self.hit_id()
        }
        fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
            // No-op for the same reason Textbox's is: panes drives the actual render via [`Button::render_content_into`] with its ad-hoc parameter list. The trait makes Button a participant in dispatch (click / focus / hover / key) without forcing every consumer onto PaintCtx today.
        }
        // Disabled → every accessor returns `None`, so dispatch + tab-cycle skip the button (see the matching note on Textbox's Widget impl).
        fn click(&mut self) -> Option<&mut dyn Click> {
            self.enabled.then_some(self as &mut dyn Click)
        }
        fn key(&mut self) -> Option<&mut dyn Key> {
            self.enabled.then_some(self as &mut dyn Key)
        }
        fn focus(&mut self) -> Option<&mut dyn Focus> {
            self.enabled.then_some(self as &mut dyn Focus)
        }
        fn hover(&mut self) -> Option<&mut dyn Hover> {
            self.enabled.then_some(self as &mut dyn Hover)
        }
        fn damage_rect(&self, viewport_w: usize, viewport_h: usize) -> Option<crate::canvas::PixelRect> {
            Button::damage_rect(self, viewport_w, viewport_h)
        }
    }

    impl<P: Pixel> Click for Button<P> {
        fn on_click(
            &mut self,
            _x: Coord,
            _y: Coord,
            _mods: ModifiersState,
        ) -> crate::host::EventResponse {
            self.fire();
            crate::host::EventResponse::Handled
        }
        fn set_pressed(&mut self, pressed: bool) {
            Button::set_pressed(self, pressed);
        }
    }

    impl<P: Pixel> Key for Button<P> {
        fn on_key(
            &mut self,
            kev: &KeyEvent,
            _mods: ModifiersState,
            _text: &mut TextRenderer,
        ) -> crate::host::EventResponse {
            if kev.state != ElementState::Pressed {
                return crate::host::EventResponse::Pass;
            }
            match &kev.logical_key {
                FKey::Named(NamedKey::Enter) | FKey::Named(NamedKey::Space) => {
                    self.fire();
                    crate::host::EventResponse::Handled
                }
                _ => crate::host::EventResponse::Pass,
            }
        }
    }

    impl<P: Pixel> Focus for Button<P> {
        fn set_focused(&mut self, focused: bool) {
            Button::set_focused(self, focused);
        }
        fn focus_bbox(&self) -> Option<crate::canvas::PixelRect> {
            let r = self.bbox();
            let x0 = r.x as usize;
            let y0 = r.y as usize;
            let x1 = (r.x + r.w) as usize;
            let y1 = (r.y + r.h) as usize;
            Some(crate::canvas::PixelRect::new(x0, y0, x1, y1))
        }
    }

    impl<P: Pixel> Hover for Button<P> {
        fn set_hovered(&mut self, hovered: bool) {
            Button::set_hovered(self, hovered);
        }
        fn tint_delta(&self) -> u32 {
            // Baked-state mode folds the state colour into the pill fill itself, so there is no overlay tint to add — the delta is zero (a host that runs the overlay must not double-apply).
            if self.bake_states {
                return 0;
            }
            // Otherwise the pill baked its RESTING fill and the host overlay wrap-adds this delta on the hovered/pressed pixels. `effective_fill()` picks the state colour (held > focused > hover >
            // idle) honouring per-instance overrides + the derived ramp; the delta carries it there off the resting fill, so `idle` yields a zero delta (no tint).
            let idle = self.fill.unwrap_or(crate::theme::look().button);
            crate::paint::wrap_sub_rgb(self.effective_fill(), idle)
        }
        fn hover_bbox(
            &self,
            viewport_w: usize,
            viewport_h: usize,
        ) -> Option<crate::canvas::PixelRect> {
            // The pill bbox — bounds the host's overlay tint scan to this button, not the whole window.
            Some(crate::widgets::textbox::region_to_pixelrect(
                self.bbox(),
                viewport_w,
                viewport_h,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{Canvas, Damage};
    use alloc::vec;

    /// A round button's silhouette is a circle: every pixel it shows on lies within its radius of its centre (plus the anti-aliased rim), it shows at the four points of the compass just inside, and its four corners are bare, where the house pill's deep corners would cover two of them.
    #[test]
    fn a_round_button_is_a_circle() {
        let (w, h) = (128usize, 128usize);
        let mut text = TextRenderer::new();
        let mut ids: HitId = paint::HIT_NONE;
        let mut b = Button::new(&mut ids, 64., 64., 96., 96., 20., "");
        b.set_round(true);
        let mut px = vec![Argb8::EMPTY; w * h];
        let mut dmg = Damage::new();
        {
            let mut c = Canvas::new(&mut px, w, h, &mut dmg);
            let id = b.hit_id();
            b.render_content_into(&mut c, 0., 0., &mut text, None, None, id);
        }
        let shows = |x: usize, y: usize| px[y * w + x].alpha() != 0;
        for y in 0..h {
            for x in 0..w {
                if shows(x, y) {
                    let d = ((x as f32 + 0.5 - 64.).powi(2) + (y as f32 + 0.5 - 64.).powi(2)).sqrt();
                    assert!(d <= 48. + 1.5, "({x}, {y}) shows {d} from the centre");
                }
            }
        }
        for (x, y) in [(64, 18), (64, 109), (18, 64), (109, 64)] {
            assert!(shows(x, y), "({x}, {y}) inside the circle does not show");
        }
        // The box's corners, a twelfth in from each: outside a circle of radius 48, inside the pill's TL and BR squircle corners.
        for (x, y) in [(24, 24), (103, 24), (24, 103), (103, 103)] {
            assert!(!shows(x, y), "({x}, {y}) in a corner shows");
        }
    }
}
