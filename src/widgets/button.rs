//! Pill-shaped button widget. Visually a Textbox stripped of the editing apparatus: same squircle silhouette, same two-tone AA edge, same focus-glow, same hover tint via the host's overlay-deltas pipe; just a static centred label, no cursor / scroll / selection / keystroke routing. Adapted from `widgets::textbox` so the pill-cache + text-cache-clipped-by-`AlphaMask` pattern is identical and the two widgets read as a coherent family.
//!
//! **Action model.** Button doesn't carry a callback or an action enum — it owns a click counter. App polls [`Button::take_click`] each frame; `true` means the button fired since last poll. That's the smallest stateful API that decouples "the widget knows it was clicked" from "the app knows what to do about it" without dragging closures + lifetime juggling into the widget. For richer action dispatch (per-button intent codes, multi-target routing), the app builds a `HashMap<HitId, Action>` keyed by `button.hit_id()` and matches at the dispatch layer.

use crate::canvas::PixelRect;
use crate::coord::Coord;
use crate::paint::{self, Clip, HitId};
use crate::region::Region;
use crate::text::TextRenderer;
use crate::theme;
use crate::widgets::textbox::{blit_cache_to_target, region_to_pixelrect};
use alloc::string::String;
use alloc::vec::Vec;

pub struct Button {
    /// Allocated at construction; stamped into the host hit map at every opaque pill pixel.
    hit_id: HitId,
    /// Label text — single line, no editing. Mutate via [`Self::set_label`] so caches invalidate.
    label: String,
    font: &'static str,

    /// Stroke thickness in RU (× `font_size`). `0.0` → 1 px minimum via the `+ 1` idiom in render. Matches Textbox's stroke convention so a Button and a Textbox at the same `stroke_ru` render with identical edge weight.
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

    /// Number of times [`Click::on_click`] has fired since construction. Monotonic. Consumers compare against [`Self::last_seen_click_counter`] via [`Self::take_click`] to know "has this button fired since I last looked."
    click_counter: u32,
    last_seen_click_counter: u32,

    // --- Caches (same layering as Textbox: pill_cache holds squircle bg + AA edges, text_cache holds the label glyphs clipped to the inner-pill silhouette, both blit topmost-first into target via under()) ---
    pill_cache: Vec<u32>,
    pill_cache_w: usize,
    pill_cache_h: usize,
    pill_cache_dirty: bool,
    text_cache: Vec<u32>,
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
    /// `hit_counter` is the app's monotonic [`HitId`] allocator (see [`crate::host::widget::next_id`]). Each Button claims one ID at construction.
    pub fn new(
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
            label: label.into(),
            font: "Open Sans",
            stroke_ru: 0.0,
            center_x,
            center_y,
            width,
            height,
            font_size,
            focused: false,
            hovered: false,
            click_counter: 0,
            last_seen_click_counter: 0,
            pill_cache: Vec::new(),
            pill_cache_w: 0,
            pill_cache_h: 0,
            pill_cache_dirty: true,
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
        if hovered != self.hovered {
            self.hovered = hovered;
        }
    }

    /// Replace the label text. Marks both caches dirty so the next render re-rasterizes glyphs (text_cache) — the pill silhouette doesn't depend on label content so technically only text_cache needs it, but a label swap typically goes with a re-layout, and marking pill dirty too is cheap insurance.
    pub fn set_label(&mut self, label: impl Into<String>) {
        let new = label.into();
        if new != self.label {
            self.label = new;
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

    /// Mark the click counter advanced. Called from the Click trait impl; pub(crate) so internal consumers can fire it without going through the trait if they need to (e.g. a hypothetical keyboard accelerator outside Key::on_key).
    pub(crate) fn fire(&mut self) {
        self.click_counter = self.click_counter.wrapping_add(1);
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
        if !pill_dirty && !focus_changed && !hover_changed && self.last_painted_bbox.is_some() {
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

    /// Paint the button into `canvas` at its viewport-space `center_*` / `width` / `height`, stamping `hit_id` into `hit_map` at every opaque pill pixel. Same three-step blit as Textbox: text_cache (glyphs) → pill_cache (squircle bg + AA edges) → focus glow on top. No selection rect, no blinkey — just label + pill + optional glow.
    pub fn render_content_into(
        &mut self,
        canvas: &mut crate::canvas::Canvas,
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
        // Fractional squirdleyness — slots between an ellipse (2) and a diamond (1). `1.5` reads as a noticeably-rounder, slightly-faceted pill: distinctly more curved than the textbox's `3.0` "slightly squared" but not as soft as a full ellipse. Routes through paint's `_f` (powf) variant; the textbox / chrome keep the integer (powi) fast path. Adjustable per-instance via this constant; future API could expose it as a Button field if more shapes are desired.
        let squirdleyness = 1.75;
        let stroke_px = (self.stroke_ru * self.font_size) as isize + 1;

        // --- Pill cache rasterize: squircle fill + AA edges, gated by pill_cache_dirty. Identical to Textbox's pill rasterize so the two widgets share the visual family. ---
        if self.pill_cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.pill_cache.clear();
            self.pill_cache.resize(cw * ch, 0);
            self.pill_cache_w = cw;
            self.pill_cache_h = ch;
            let inner_x = stroke_px;
            let inner_y = stroke_px;
            let inner_w = (pill_w - 2 * stroke_px).max(0);
            let inner_h = (pill_h - 2 * stroke_px).max(0);
            let mut cache_damage = crate::canvas::Damage::new();
            {
                let mut cache_canvas =
                    crate::canvas::Canvas::new(&mut self.pill_cache, cw, ch, &mut cache_damage);
                if inner_w > 0 && inner_h > 0 {
                    paint::draw_squircle_pill_f(
                        &mut cache_canvas,
                        inner_x,
                        inner_y,
                        inner_w,
                        inner_h,
                        theme::BUTTON_FILL,
                        squirdleyness,
                    );
                }
            }
            self.inner_pill_mask.clear();
            self.inner_pill_mask.resize(cw * ch, 0);
            for i in 0..(cw * ch) {
                self.inner_pill_mask[i] = ((self.pill_cache[i] >> 24) & 0xFF) as u8;
            }
            {
                let mut cache_canvas =
                    crate::canvas::Canvas::new(&mut self.pill_cache, cw, ch, &mut cache_damage);
                // Bevel direction is INVERTED relative to Textbox: shadow on the top/left (where light would normally hit a sunken edge), light on the bottom/right. Reads visually as "raised" — the button protrudes toward the viewer — whereas a textbox with the canonical orientation reads as "inset / carved into the surface." Single argument swap, zero extra render cost, classic UI lighting convention preserved.
                paint::draw_squircle_pill_two_tone_f(
                    &mut cache_canvas,
                    0,
                    0,
                    pill_w,
                    pill_h,
                    theme::TEXTBOX_SHADOW_EDGE,
                    theme::TEXTBOX_LIGHT_EDGE,
                    squirdleyness,
                    None,
                    0,
                );
            }
            self.pill_cache_dirty = false;
        }

        // --- Text cache: centred label clipped by inner-pill mask. No scroll, no selection — anchor at pill centre, draw_text_left_u32 with x = centre − text_width / 2. ---
        if self.text_cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.text_cache.clear();
            self.text_cache.resize(cw * ch, 0);
            self.text_cache_w = cw;
            self.text_cache_h = ch;
            if !self.label.is_empty() && self.font_size > 0.0 {
                let tw = text.measure_text_width(&self.label, self.font_size, 400, self.font);
                let local_text_left = pill_w as Coord * 0.5 - tw * 0.5;
                let local_y_center = pill_h as Coord * 0.5;
                let mut text_damage = crate::canvas::Damage::new();
                let mut text_canvas =
                    crate::canvas::Canvas::new(&mut self.text_cache, cw, ch, &mut text_damage);
                let mask_buffer = paint::AlphaMask::new(&self.inner_pill_mask, cw, ch);
                text.draw_text_left_u32(
                    &mut text_canvas,
                    &self.label,
                    local_text_left,
                    local_y_center,
                    self.font_size,
                    400,
                    theme::TEXTBOX_TEXT,
                    self.font,
                    None,
                    Some(&mask_buffer),
                    None,
                );
            }
            self.text_cache_dirty = false;
        }

        // --- Composition: text (topmost) → pill (bottom, stamps hit_map) ---
        blit_cache_to_target(
            &self.text_cache,
            self.text_cache_w,
            self.text_cache_h,
            pill_x_target,
            pill_y_target,
            canvas,
            None,
            0,
            clip,
        );
        blit_cache_to_target(
            &self.pill_cache,
            self.pill_cache_w,
            self.pill_cache_h,
            pill_x_target,
            pill_y_target,
            canvas,
            hit_map,
            hit_id,
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
        self.last_painted_focused = self.focused;
        self.last_painted_hovered = self.hovered;
    }
}

/// Same RU-invariant glow factor as Textbox's [`Textbox::glow_factor_256`]; duplicated here so Button doesn't pull a `pub(crate)` from Textbox just for one tiny formula. Returns a factor in `[96, 254]` where smaller = steeper decay = shorter visual reach; matches the chrome shadow's `factor_256` math so glows around buttons and textboxes share the same look.
fn glow_factor_256(font_size: f32, radius_scale: f32) -> u32 {
    let target_radius = (font_size * radius_scale).max(8.0);
    let drop = (1240.0 / target_radius) as u32;
    (256u32.saturating_sub(drop)).clamp(96, 254)
}

#[cfg(feature = "host-winit")]
mod widget_impls {
    //! [`crate::host::widget`] capability traits for [`Button`]. Mirrors `textbox::widget_impls`: Click increments the action counter, Focus / Hover route through Button's setters, Key activates on Enter / Space.

    use super::Button;
    use crate::coord::Coord;
    use crate::host::widget::{Click, Focus, Hover, Key, PaintCtx, Widget};
    use crate::paint::HitId;
    use crate::text::TextRenderer;
    use winit::event::KeyEvent;
    use winit::keyboard::{Key as WKey, ModifiersState, NamedKey};

    impl Widget for Button {
        fn id(&self) -> HitId {
            self.hit_id()
        }
        fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
            // No-op for the same reason Textbox's is: panes drives the actual render via [`Button::render_content_into`] with its ad-hoc parameter list. The trait makes Button a participant in dispatch (click / focus / hover / key) without forcing every consumer onto PaintCtx today.
        }
        fn click(&mut self) -> Option<&mut dyn Click> {
            Some(self)
        }
        fn key(&mut self) -> Option<&mut dyn Key> {
            Some(self)
        }
        fn focus(&mut self) -> Option<&mut dyn Focus> {
            Some(self)
        }
        fn hover(&mut self) -> Option<&mut dyn Hover> {
            Some(self)
        }
    }

    impl Click for Button {
        fn on_click(
            &mut self,
            _x: Coord,
            _y: Coord,
            _mods: ModifiersState,
        ) -> crate::host::app::EventResponse {
            self.fire();
            crate::host::app::EventResponse::Handled
        }
    }

    impl Key for Button {
        fn on_key(
            &mut self,
            kev: &KeyEvent,
            _mods: ModifiersState,
            _text: &mut TextRenderer,
        ) -> crate::host::app::EventResponse {
            if kev.state != winit::event::ElementState::Pressed {
                return crate::host::app::EventResponse::Pass;
            }
            match &kev.logical_key {
                WKey::Named(NamedKey::Enter) | WKey::Named(NamedKey::Space) => {
                    self.fire();
                    crate::host::app::EventResponse::Handled
                }
                _ => crate::host::app::EventResponse::Pass,
            }
        }
    }

    impl Focus for Button {
        fn set_focused(&mut self, focused: bool) {
            Button::set_focused(self, focused);
        }
        fn focus_bbox(&self) -> Option<crate::canvas::PixelRect> {
            let r = self.bbox();
            let x0 = r.x.max(0.0) as usize;
            let y0 = r.y.max(0.0) as usize;
            let x1 = (r.x + r.w).max(0.0) as usize;
            let y1 = (r.y + r.h).max(0.0) as usize;
            Some(crate::canvas::PixelRect::new(x0, y0, x1, y1))
        }
    }

    impl Hover for Button {
        fn set_hovered(&mut self, hovered: bool) {
            Button::set_hovered(self, hovered);
        }
        fn tint_delta(&self) -> u32 {
            // Three states from the BUTTON_* palette: idle (no tint, lands on BUTTON_FILL — slate-grey-blue), hovered → BUTTON_HOVER (slightly more saturated blue, signals "clickable"), focused → BUTTON_ACTIVE (darkens back toward TEXTBOX_FILL for the conventional "pressed in" inverse-bevel reading). Focus dominates hover so a focused-and-hovered button stays at ACTIVE rather than flickering to HOVER while the cursor passes over.
            if self.is_focused() {
                crate::paint::wrap_sub_rgb(
                    crate::theme::BUTTON_ACTIVE,
                    crate::theme::BUTTON_FILL,
                )
            } else if self.is_hovered() {
                crate::paint::wrap_sub_rgb(
                    crate::theme::BUTTON_HOVER,
                    crate::theme::BUTTON_FILL,
                )
            } else {
                0
            }
        }
    }
}
