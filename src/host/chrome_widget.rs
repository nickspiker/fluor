//! `DefaultChrome` — the reusable borderless-window frame consumers compose into their `FluorApp`.
//!
//! Owns a full-viewport [`Group`] containing three layers — bg (caller paints), chrome (controls + edges + hairlines + title text), hover (button delta) — plus the per-pixel `hit_test_map` byte buffer that records which button (if any) covers each pixel. Hover state, the cached hover-pixel list, and the title string all live here so the consumer can drop in chrome with one struct field.
//!
//! Built on the verbatim photon primitives in [`super::chrome`] — `draw_window_controls`, `draw_window_edges_and_mask`, `draw_button_hairlines`, `draw_button_hover_by_pixels`, `pixels_for_button`. Those stay; this module is a stateful wrapper that schedules them against the chrome group's dirty layers.
//!
//! Pattern: `chrome.rasterize_bg(|bg, w, h| { /* paint into bg */ });` → `chrome.rasterize_chrome(text);` → `chrome.flatten_into(target, w, h);`. Each rasterize_* checks the layer's dirty bit internally and is a no-op on clean. Hover / focus tint is NOT rasterized — it's applied by the host's post-finalize overlay pass against `persistent_screen` via [`super::widget::build_overlay_deltas`] reading [`Hover::tint_delta`] off each chrome button.

use super::EventResponse;
use super::chrome::{self, HIT_NONE, HitId};
use super::widget::{self, Click, Container, Hover, PaintCtx, Widget};
use crate::coord::Coord;
use crate::geom::Viewport;
use crate::group::Group;
use crate::paint::BlendMode;
use crate::region::Region;
use crate::stack::Op;
use crate::text::TextRenderer;
use crate::event::{Key as FKey, KeyEvent, ModifiersState, NamedKey};
use crate::theme;
use alloc::string::String;
use alloc::vec::Vec;

// Hover colour mapping moved to [`DefaultChrome::hover_colour_for`] — needs the live button IDs allocated at chrome construction time, so it can no longer be a free function.

/// Action a [`ChromeButton`] dispatches when clicked or activated via keyboard. Closed enum because the four canonical window-frame buttons aren't user-extensible — new chrome elements would be new widget types, not new variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromeAction {
    Minimize,
    ToggleMaximized,
    Close,
    /// The app-icon "orb" slot. Currently a no-op on click but kept in the tab cycle for keyboard-discoverability (apps may eventually wire it to a window menu).
    AppIcon,
}

impl ChromeAction {
    /// The [`EventResponse`] this action emits when the button fires. Centralised so click and keyboard-activate paths can't drift.
    fn response(&self) -> EventResponse {
        match self {
            ChromeAction::Minimize => EventResponse::Minimize,
            ChromeAction::ToggleMaximized => EventResponse::ToggleMaximized,
            ChromeAction::Close => EventResponse::Close,
            ChromeAction::AppIcon => EventResponse::Handled,
        }
    }
}

/// One of the four window-frame buttons (minimize / maximize / close / app icon). Carries the dense hit-id allocated at chrome construction, the click action, and the live focused / hovered state. **Does not paint itself** — [`DefaultChrome::rasterize_chrome`] paints the four buttons collectively in one pass because the hit-fill walls and slot dividers are shared geometry that's painful to split per-button. [`Widget::paint`] is therefore a structural no-op; chrome is still responsible for stamping `self.id` into the hit map at the right pixels via the existing `paint_button_hit_row_scan` helper.
pub struct ChromeButton {
    id: HitId,
    pub action: ChromeAction,
    pub focused: bool,
    pub hovered: bool,
}

impl ChromeButton {
    fn new(id: HitId, action: ChromeAction) -> Self {
        Self {
            id,
            action,
            focused: false,
            hovered: false,
        }
    }

    /// The button's hit-id. Mirrors [`Widget::id`] so callers that hold a `&ChromeButton` directly (not thru `dyn Widget`) don't need to import the trait.
    pub fn id(&self) -> HitId {
        self.id
    }
}

impl Widget for ChromeButton {
    fn id(&self) -> HitId {
        self.id
    }
    fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
        // Intentional no-op — chrome paints buttons collectively in `rasterize_chrome`. See struct-level doc for the why.
    }
    fn click(&mut self) -> Option<&mut dyn Click> {
        Some(self)
    }
    fn hover(&mut self) -> Option<&mut dyn Hover> {
        Some(self)
    }
    fn focus(&mut self) -> Option<&mut dyn widget::Focus> {
        Some(self)
    }
    fn key(&mut self) -> Option<&mut dyn widget::Key> {
        Some(self)
    }
}

impl Click for ChromeButton {
    fn on_click(&mut self, _x: Coord, _y: Coord, _mods: ModifiersState) -> EventResponse {
        self.action.response()
    }
}

impl Hover for ChromeButton {
    fn set_hovered(&mut self, hovered: bool) {
        self.hovered = hovered;
    }
    fn tint_delta(&self) -> u32 {
        if !self.focused && !self.hovered {
            return 0;
        }
        // The CLOSE/MAX/MIN theme constants are TARGET colours, so return the darkness-space DELTA from the control's base fill to the target — `wrap_sub_rgb(target, WINDOW_CONTROLS_BG)`, exactly the form the overlay's visible-space wrap-sub expects (lands a base-fill pixel at the target hue), and the same shape `Button::tint_delta` produces.
        // Returning the raw target here made the overlay subtract the full colour off the fill instead → a muddy/grey shift, not the cyan/magenta/yellow.
        // The app-icon orb takes the per-pixel sqrt gamma lift instead of a flat tint — a flat delta over a multi-colour starburst washes it toward one hue, while the sqrt lift makes the whole icon glow.
        // Quarter-strength (num/den = 1/4) so the vivid red/green/blue targets read as a gentle wash over the control fill, not a full-saturation flood.
        match self.action {
            ChromeAction::Close => {
                crate::paint::wrap_sub_rgb_scaled(theme::CLOSE_HOVER, theme::WINDOW_CONTROLS_BG, 1, 4)
            }
            ChromeAction::ToggleMaximized => crate::paint::wrap_sub_rgb_scaled(
                theme::MAXIMIZE_HOVER,
                theme::WINDOW_CONTROLS_BG,
                1,
                4,
            ),
            ChromeAction::Minimize => crate::paint::wrap_sub_rgb_scaled(
                theme::MINIMIZE_HOVER,
                theme::WINDOW_CONTROLS_BG,
                1,
                4,
            ),
            ChromeAction::AppIcon => crate::paint::OVERLAY_SQRT_BRIGHTEN,
        }
    }
}

impl widget::Focus for ChromeButton {
    fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }
    // focus_bbox returns None — chrome layout is recomputed per-frame; spatial tab-order would need a different shape (query chrome at walk time). Defer.
}

impl widget::Key for ChromeButton {
    fn on_key(
        &mut self,
        kev: &KeyEvent,
        _mods: ModifiersState,
        _text: &mut TextRenderer,
    ) -> EventResponse {
        match &kev.logical_key {
            FKey::Named(NamedKey::Enter) | FKey::Named(NamedKey::Space) => self.action.response(),
            _ => EventResponse::Pass,
        }
    }
}

/// Reusable window frame: controls, edges, hairlines, title, hover overlay.
pub struct DefaultChrome {
    /// Full-viewport Group with 3 layers (bg, chrome, hover) composed via Stack Notation. Topmost-first: `Push chrome, Push bg, Under(Normal)` for the minimal scaffold; expand to include hover via `Push hover, Push chrome, Under(Add), Push bg, Under(Normal)` as the design grows.
    pub group: Group,
    /// Per-pixel button-id map. `HIT_NONE` (0) for pixels outside any chrome button; otherwise the dense `HitId` of whichever button the pixel belongs to (min / max / close / app-icon — each id is allocated at chrome construction time via [`super::widget::next_id`]). Sized to `width * height` pixels of the current viewport.
    pub hit_test_map: Vec<HitId>,
    /// Window title rendered into the chrome layer (left-aligned in the controls strip). Empty string = skip text rendering.
    pub title: String,
    /// Optional bottom status bar. `None` = no bar, panes/bg fill all the way to the squircle bottom (default behaviour). `Some(text)` = paint a `button_size / 2`-tall band at the bottom with the given text left-aligned. Mutate via [`set_status_text`](Self::set_status_text) to mark the chrome layer dirty automatically.
    pub status_text: Option<String>,
    /// Optional app-icon orb painted in the top-left chrome slot. `None` = no orb, title text starts at the left margin. When `Some`, [`chrome::draw_app_icon`] runs after the perimeter and the title text shifts right by `button_size + button_size/4` so it doesn't overlap.
    pub app_icon: Option<crate::host::icon::Icon>,
    /// Window-focus state. `true` = active (full edge bevel, bright title, ring follows perimeter, icon at full saturation). `false` = inactive (edges + title + orb ring collapse to `LABEL_COLOUR`; orb image desaturates 50 % toward grey when `orb_tint` is `FollowFocus`). Host wires this from `WindowEvent::Focused`. Mutate via [`set_focused`](Self::set_focused) to mark the chrome layer dirty automatically.
    pub focused: bool,
    /// Orb visual state. Default `OrbTint::FollowFocus` makes the orb a window-state indicator; `OrbTint::Custom` lets the app turn it into a network/recording/presence badge. Mutate via [`set_orb_tint`](Self::set_orb_tint) to mark the chrome layer dirty automatically.
    pub orb_tint: chrome::OrbTint,
    /// Currently-hovered button id (HIT_NONE if none). Consumed by the host's overlay pass to derive the visible-RGB tint delta to apply at matching `hit_test_map` pixels in persistent_screen.
    pub hover_state: HitId,
    /// "Full edge" / maximized mode. When `true`, [`Self::rasterize_chrome`] skips [`chrome::draw_window_edges_and_mask`] entirely: no perimeter hairline, no corner cutout in `clip_mask`, no AA fringe — the chrome flows straight to the four screen edges. The OS surface is fullscreen anyway, the WM can't show a shadow against the screen border, and AA on a corner that's flush with the screen is wasted work. Buttons / title / app icon still rasterize as usual. Toggle via [`Self::set_full_edge`]; sync from [`super::app::Context::is_maximized`].
    pub full_edge: bool,
    /// Last viewport passed to `new` or `resize`. Stored so chrome rasterization can read `effective_span` (= `span * ru`) and pick up the user's zoom multiplier automatically — chrome control sizing scales with the same `ceil(effective_span/32)` formula, so Ctrl+/ Ctrl-/ Ctrl+scroll zoom the chrome together with content.
    viewport: Viewport,
    layer_bg: usize,
    layer_chrome: usize,
    /// Minimize-button widget. ID allocated at chrome construction time. Allocation order (min → max → close → app-icon) is the tab-cycle order chrome exposes via [`Container::visit`]; ids are otherwise opaque — callers query [`Self::owns_hit`] / [`Self::hover_colour_for`] instead of comparing numerically.
    pub min_btn: ChromeButton,
    /// Maximize / restore button.
    pub max_btn: ChromeButton,
    /// Close button.
    pub close_btn: ChromeButton,
    /// App-icon orb button.
    pub app_icon_btn: ChromeButton,
}

impl DefaultChrome {
    /// Allocate the chrome group + hit_test_map sized to `viewport`. Three layers (bg, chrome, hover) all start dirty so the first frame paints from scratch.
    ///
    /// **Topmost-first scaffold:** the Stack program is the minimal front-to-back composite — `Push chrome, Push bg, Under(Normal)`. Chrome is the topmost layer (controls, edges, hairlines, title), bg is the layer behind it (background_noise + panes). Stack order matches the visual stack: first push lands on the bottom of the eval stack and is the topmost layer; second push goes underneath via `Under`. The hover layer still exists and is rasterized so the API surface is stable; it's omitted from the program until the hover overlay is wired back up via `Push hover, Under(Add)` as the topmost step. Corner knockout (formerly a separate silhouette layer + `Op::Or`) is handled at chrome rasterization time by writing `t=255` directly into the chrome layer's corner pixels — no separate Stack op under the unified Under model. Construct. `hit_counter` is the app's monotonic [`HitId`] allocator (see [`super::widget::next_id`]) — chrome registers four IDs (min, max, close, app icon, in that order). Construction order against the rest of the app is free: ids are opaque and queried thru [`Self::owns_hit`] / [`Self::hover_colour_for`], not compared by value.
    pub fn new(
        viewport: Viewport,
        title: impl Into<String>,
        app_icon: Option<crate::host::icon::Icon>,
        status_text: Option<String>,
        hit_counter: &mut HitId,
    ) -> Self {
        let region = Region::new(
            0.0,
            0.0,
            viewport.width_px as Coord,
            viewport.height_px as Coord,
        );
        let mut group = Group::new(region, BlendMode::Normal);
        let layer_bg = group.new_layer();
        let layer_chrome = group.new_layer();
        // Front-to-back: chrome on top (controls + edges + hairlines), bg underneath (panes + background_noise). Hover / focus tint is NOT a separate layer — applied by the host's post-finalize overlay pass against persistent_screen instead of baked into chrome_buf, so chrome's partial-α AA edges stay clean (a separate premultiplied hover layer would re-premultiply them and trash the edge).
        group.set_program(alloc::vec![
            Op::Push(layer_chrome),
            Op::Push(layer_bg),
            Op::Under(BlendMode::Normal),
        ]);
        let map_len = (viewport.width_px as usize).saturating_mul(viewport.height_px as usize);
        let min_id = widget::next_id(hit_counter);
        let max_id = widget::next_id(hit_counter);
        let close_id = widget::next_id(hit_counter);
        let app_icon_id = widget::next_id(hit_counter);
        Self {
            group,
            hit_test_map: alloc::vec![HIT_NONE; map_len],
            title: title.into(),
            status_text,
            app_icon,
            focused: true,
            orb_tint: chrome::OrbTint::FollowFocus,
            hover_state: HIT_NONE,
            full_edge: false,
            viewport,
            layer_bg,
            layer_chrome,
            min_btn: ChromeButton::new(min_id, ChromeAction::Minimize),
            max_btn: ChromeButton::new(max_id, ChromeAction::ToggleMaximized),
            close_btn: ChromeButton::new(close_id, ChromeAction::Close),
            app_icon_btn: ChromeButton::new(app_icon_id, ChromeAction::AppIcon),
        }
    }

    /// Resize the chrome group + hit_test_map to a new viewport. Also called when only zoom (`viewport.ru`) changed without size, so chrome re-rasterizes at the new effective span. All layers go dirty either way.
    pub fn resize(&mut self, viewport: Viewport) {
        let region = Region::new(
            0.0,
            0.0,
            viewport.width_px as Coord,
            viewport.height_px as Coord,
        );
        self.group.resize(region);
        let map_len = (viewport.width_px as usize).saturating_mul(viewport.height_px as usize);
        self.hit_test_map.resize(map_len, HIT_NONE);
        self.viewport = viewport;
    }

    /// Buffer dimensions (full viewport).
    /// The app-icon orb's current geometry `(cx, cy, radius)` in buffer pixels, or `None` when no orb slot is active — the same math `rasterize_chrome` lays it out with, exposed so a host app can paint press effects (photon's glow) around the orb AFTER the chrome flatten.
    pub fn orb_geometry(&self) -> Option<(isize, isize, isize)> {
        let orb_present = self.app_icon.is_some()
            || matches!(self.orb_tint, chrome::OrbTint::Custom { .. });
        if !orb_present {
            return None;
        }
        let span = self.viewport.effective_span();
        let button_size = crate::math::ceil(span / 32.0) as usize;
        let orb_radius = (button_size as isize * 3) / 4;
        let c = orb_radius + button_size as isize / 2;
        Some((c, c, orb_radius))
    }

    pub fn dims(&self) -> (usize, usize) {
        self.group.dims()
    }

    /// Paint the bg layer with consumer-supplied content. The closure receives a [`crate::canvas::Canvas`] backed by the bg layer's pixel buffer and the caller-supplied `damage` accumulator — so any rasterizer the consumer invokes reports its painted bbox into the frame-level damage automatically. No-op if the layer is clean.
    pub fn rasterize_bg(
        &mut self,
        damage: &mut crate::canvas::Damage,
        paint: impl FnOnce(&mut crate::canvas::Canvas),
    ) {
        let (w, h) = self.dims();
        let layer = &mut self.group.rpn.layers[self.layer_bg];
        if !layer.dirty {
            return;
        }
        crate::paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        // α + darkness: transparent init (α=0) so pixels the closure doesn't paint stay transparent rather than becoming spurious opaque content. The closure is expected to fully cover the bg, but defaulting to transparent is the safe failure mode. Zero-init is calloc-free.
        layer.pixels.fill(0);
        let mut canvas = crate::canvas::Canvas::new(&mut layer.pixels, w, h, damage);
        paint(&mut canvas);
    }

    /// Paint the window-perimeter hairline DIRECTLY into the consumer's `target` buffer and (re)carve the window-shape `clip_mask` — the first writer of the frame, run BEFORE the consumer paints any content into `target`.
    ///
    /// **Why this is split out of [`rasterize_chrome`]:** fluor is under-blend only ("topmost paints first wins"). The chrome group composites UNDER `target` at [`flatten_into`](Self::flatten_into) time (`target.under(chrome)`), so any content the consumer draws directly into `target` ALWAYS wins over chrome at shared edge pixels — burying the perimeter hairline wherever full-bleed content reaches the window edge. Painting the hairline into `target` first makes it the top of the under-chain at those edge pixels: content then composes UNDER it and the hairline survives. Buttons / orb / controls-strip / title stay in the chrome group (they never sit at the window edge, so no content conflicts there) and keep compositing under content as before.
    ///
    /// **clip_mask ownership moved here.** The `clip_mask.fill(255)` reset and the corner cutout carve (a side effect of [`chrome::draw_window_edges_and_mask`]) are inseparable — same pixels, one pass — so they MUST stay on the same cycle. This pass owns BOTH now and runs **every frame** (not dirty-gated): the consumer overwrites `target`'s edge pixels each frame, so the hairline must be repainted each frame, and the mask is recomputed identically each frame (it's a pure function of viewport + zoom), keeping it coherent with the dirty-gated `hit_test_map` that [`rasterize_chrome`] stamps against it. The single window-shape α-trim still happens exactly once, at the OS boundary in [`crate::paint::finalize_for_os`] — this pass only writes the mask, never multiplies it into α.
    ///
    /// `full_edge` / `DEBUG_SKIP_CHROME` skip the hairline + carve entirely, leaving `clip_mask` at 255 (rectangular window) — same contract as the old in-chrome path.
    pub fn rasterize_perimeter(
        &mut self,
        target: &mut [u32],
        target_w: usize,
        target_h: usize,
        clip_mask: &mut [u8],
    ) {
        let (buf_w, buf_h) = self.dims();
        let vp_w = buf_w as u32;
        let vp_h = buf_h as u32;

        // Reset clip_mask to 255 (fully visible) BEFORE re-carving — the carve is a side effect of `draw_window_edges_and_mask`, so the reset rides the same pass. Recomputed identically every frame so it never desyncs from the dirty-gated hit_test_map in `rasterize_chrome`.
        clip_mask.fill(255);

        if vp_w < 2 || vp_h < 2 || target_w != buf_w || target_h != buf_h {
            return;
        }
        if self.full_edge || crate::paint::DEBUG_SKIP_CHROME.load(std::sync::atomic::Ordering::Relaxed) {
            return; // rectangular window: no perimeter hairline, no corner cutout
        }

        // Geometry shared with `rasterize_chrome` (recomputed here so the perimeter is self-contained per frame). `effective_span` folds in the user's zoom so the corners scale with Ctrl+/Ctrl-/Ctrl+scroll.
        let span = self.viewport.effective_span();
        let (start, crossings) = compute_squircle_crossings(span / 4.0, 24);
        if crossings.is_empty() {
            return;
        }
        let (start_big, crossings_big) = compute_squircle_crossings(span / 2.0, 24);

        // Same focus-driven bevel palette as `rasterize_chrome` — top/left light, bottom/right shadow, dimmed when unfocused.
        let (edge_light, edge_shadow) = if self.focused {
            (theme::WINDOW_LIGHT_EDGE, theme::WINDOW_SHADOW_EDGE)
        } else {
            (
                theme::WINDOW_LIGHT_EDGE_UNFOCUSED,
                theme::WINDOW_SHADOW_EDGE_UNFOCUSED,
            )
        };

        // Hairline RGB lands in `target` (NOT chrome_buf) so it wins the under-chain over content drawn afterward; the window-shape cutout + AA coverage lands in `clip_mask`. The `hit_test_map` argument is passed thru untouched — `draw_window_edges_and_mask` never writes it (the trailing `let _ = hit_test_map;` in that function proves it), so handing it the chrome's own map is harmless and avoids allocating a throwaway.
        chrome::draw_window_edges_and_mask(
            target,
            &mut self.hit_test_map,
            clip_mask,
            vp_w,
            vp_h,
            start_big,
            &crossings_big,
            start,
            &crossings,
            edge_light,
            edge_shadow,
        );
    }

    /// **Scaffold step 1 (top of stack: AA rounded hairline only):** paint the controls strip, buttons, app-icon orb and title text into the chrome layer. The window-perimeter hairline + window-shape `clip_mask` carve were split out into [`rasterize_perimeter`](Self::rasterize_perimeter) (which must run BEFORE the consumer's content so the hairline wins the under-chain); this pass assumes `clip_mask` is already carved and only READS it (button hit-scan walls, silhouette restriction).
    ///
    /// `text` is used by the title text rasterization pass (Open Sans, span-relative font size, left-aligned in the area to the left of the controls strip). `damage` is the frame-level accumulator; chrome routes its migrated rasterizers (title text, status bar) thru Canvas instances backed by it, so chrome's contribution to the damage rect flows to the host.
    pub fn rasterize_chrome(
        &mut self,
        damage: &mut crate::canvas::Damage,
        text: &mut TextRenderer,
        clip_mask: &mut [u8],
    ) {
        let (buf_w, buf_h) = self.dims();
        let vp_w = buf_w as u32;
        let vp_h = buf_h as u32;

        let chrome_dirty = self.group.rpn.layers[self.layer_chrome].dirty;

        // Hover is no longer baked into chrome_buf — it lives entirely in the host's post-finalize overlay pass via `current_overlay_deltas` + `apply_overlay_diff`. chrome_buf stays tint-free; hit_test_map is the only thing chrome owes the overlay.
        if !chrome_dirty {
            return;
        }
        crate::paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // clip_mask is owned + carved by `rasterize_perimeter` (runs before content, every frame). This pass only READS it (button hit-scan walls below, silhouette restriction post-pass). hit_test_map is still owned here — wiped on each dirty cycle before re-stamping the buttons.
        self.hit_test_map.fill(HIT_NONE);

        let chrome_buf = &mut self.group.rpn.layers[self.layer_chrome].pixels;
        // α + darkness: transparent init (α=0, dark=0) so the bg shows thru everywhere except the hairline + AA pixels.
        chrome_buf.fill(0);

        if vp_w < 2 || vp_h < 2 {
            return;
        }

        // Compute span + button size shared by controls and squircle. Use the viewport's `effective_span` (= `span * ru`) so chrome scales with the user's zoom — Ctrl+/Ctrl-/Ctrl+scroll zoom the chrome together with content.
        let span = self.viewport.effective_span();
        // Span-relative: button height is span/32, where span is the harmonic mean of viewport dims times zoom. Strip layout bails downstream if the result is too small to render glyphs.
        let button_size = crate::math::ceil(span / 32.0) as usize;

        // Controls unit. The close/min/max strip and its buttons are 2× the base button size — a taller strip with bigger glyphs — while the orb and title keep the base size. There's no BL swoop any more: the strip bottom is a straight hairline. The orb/title y-centre is `button_size`, which is exactly `ctl/2`, so they land vertically centred in the taller strip with no position change.
        // ctl ≥ 2 always: button_size = ceil(span/32) ≥ 1 for any live viewport (the vp_w/vp_h < 2 bail above already excluded degenerate surfaces).
        let ctl = button_size * 2;

        // Controls-strip layout. Computed early so the title text pass can clip against `strip_x` (title shouldn't paint over the buttons even at long titles or narrow windows). The strip lives in the top-right `ctl`-tall band, `strip_w` wide.
        let strip_w = ctl * 7 / 2;
        let strip_x = buf_w.saturating_sub(strip_w);

        // App-icon orb layout: centered in the top-left `button_size`-tall band, mirroring the right-side controls strip. Diameter is half the band height so the orb reads as a tasteful badge rather than a full button. Title text shifts right by the orb's footprint when an icon is present. `draw_app_icon` has an `r < 2` early-return so degenerate sizes pass thru without drawing — no min-size guard needed here.
        //
        // Orb slot is also reserved when `OrbTint::Custom` is active even without an icon — that's the "status badge" use case (network indicator, recording light, presence). `draw_app_icon`'s no-icon path fills the disk with `ring_colour`, so the slot reads as a coloured dot.
        let orb_present = self.app_icon.is_some()
            || matches!(self.orb_tint, chrome::OrbTint::Custom { .. });
        // Orb diameter is 1.5× `button_size` (grown 2026-07-17 from the full-button-size badge — the brand mark earns the real estate). Centre keeps a constant button_size/2 clearance from the top-left corner, so the tuck into the TL squircle survives the growth; the title-margin math below tracks the orb's actual right edge automatically.
        let orb_diameter = if orb_present {
            (button_size as isize * 3) / 2
        } else {
            0
        };
        let orb_radius = orb_diameter / 2;
        let orb_cx = orb_radius + button_size as isize / 2;
        let orb_cy = orb_radius + button_size as isize / 2;
        // Title clears the orb's actual right edge. `draw_title_text`'s base left margin is `button_size/2`, so `left_extra` is the extra push needed to land the title just past `orb_cx + orb_radius` (plus a `button_size/4` gap). Tracks the orb wherever it sits, so moving the orb right keeps the title from sliding under it.
        let title_left_extra = if orb_present {
            ((orb_cx + orb_radius) as usize + button_size / 4).saturating_sub(button_size / 2)
        } else {
            0
        };
        // Title row: level with the orb when one is present, else the original top-band centre.
        let title_y_center = if orb_present {
            orb_cy as Coord
        } else {
            button_size as Coord * 0.5
        };

        // Front-to-back chrome rendering. Earliest writers WIN — `pixels[i].under(...)`'s opaque-top early-out makes later writes a no-op on pixels a previous step already claimed opaque.
        //
        // Order (top → down):
        //   1. Window perimeter — writes chrome + carves clip_mask at window boundary.
        //   2. Title text — left-aligned in the area to the left of the controls strip, on top of whatever strip_bg would later paint.
        //   3. Maximize / minimize / close glyphs (per-button).
        //   4. Strip vertical hairlines (dividers + bottom hairline).
        //   5. Strip BL squircle curves.
        //   6. Strip background fill (lowest — fills remaining empty pixels in the strip).
        //   7. Hover-state tint baked into chrome (wrap-add on hit_test_map matches).
        // `[]c`: skip the window edge/perimeter AND title text (both are "decoration"). Controls still render. clip_mask stays at host default (255 everywhere), so the window appears as a rectangle (no rounded corners). Focus-driven palette. Each element pulls from a named theme constant so a downstream consumer can override (e.g. an app that wants a totally different unfocused look) by swapping the theme module rather than re-implementing the rasterizer wiring.
        let (edge_light, title_colour) = if self.focused {
            (theme::WINDOW_LIGHT_EDGE, theme::TITLE_TEXT)
        } else {
            (theme::WINDOW_LIGHT_EDGE_UNFOCUSED, theme::TITLE_TEXT_UNFOCUSED)
        };

        // Orb tint: FollowFocus → ring matches the active perimeter colour, icon gets `theme::ORB_DARKEN_UNFOCUSED` blend when the window is unfocused. Custom → app dictates ring + brighten; window-focus state doesn't dim a Custom orb (apps using it as a status indicator want it stable).
        let (orb_ring, orb_brighten, orb_darken) = match self.orb_tint {
            chrome::OrbTint::FollowFocus => (
                edge_light,
                false,
                if self.focused {
                    0
                } else {
                    theme::ORB_DARKEN_UNFOCUSED
                },
            ),
            chrome::OrbTint::Custom { ring, brighten } => (ring, brighten, 0),
        };

        if !crate::paint::DEBUG_SKIP_CHROME.load(std::sync::atomic::Ordering::Relaxed) {
            // Window perimeter hairline + clip_mask carve moved to `rasterize_perimeter` (runs into `target` before content so the hairline wins the under-chain). This pass paints only the chrome-group elements (orb, title, strip, buttons).
            if orb_present {
                chrome::draw_app_icon(
                    chrome_buf,
                    Some(&mut self.hit_test_map),
                    self.app_icon_btn.id(),
                    buf_w,
                    buf_h,
                    orb_cx,
                    orb_cy,
                    orb_radius,
                    self.app_icon.as_ref(),
                    Some(orb_ring),
                    orb_darken,
                    orb_brighten,
                );
            }
            {
                // Title-text rasterization thru the frame-level damage accumulator. Other chrome rasterizers (perimeter, app icon, button glyphs) still write into `chrome_buf` directly without damage tracking — they'll migrate when the rest of the chrome surface gets the Canvas treatment.
                let mut canvas = crate::canvas::Canvas::new(chrome_buf, buf_w, buf_h, damage);
                chrome::draw_title_text(
                    &mut canvas,
                    &self.title,
                    text,
                    button_size,
                    strip_x,
                    title_left_extra,
                    title_y_center,
                    title_colour,
                );
            }
        }

        // `[]l`: skip ONLY the controls strip (perimeter + title stay).
        if crate::paint::DEBUG_SKIP_CONTROLS.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let button_area_x = strip_x + ctl / 4;
        let glyph_y = ctl / 2;
        let glyph_r = ctl / 4;
        let min_cx = button_area_x + ctl / 2;
        let max_cx = button_area_x + ctl + ctl / 2;
        let close_cx = button_area_x + ctl * 2 + ctl / 2;
        let strip_w = ctl * 7 / 2;

        // Hairlines (dividers + bottom) BEFORE curves: solid lines have to win at intersection pixels, otherwise the curve's inner-AA hairline (which is mostly transparent in the linear region) fragments them. With this order, dividers and bottom hairline claim their pixels first; the BL curve's hairlines fill in only the gaps the straight lines didn't reach.
        //
        // Strip-frame colours follow the focus palette: vertical dividers + bottom hairline take `edge_light` (same as the top/left window perimeter), so the strip reads as a continuation of the window edge, not a separate piece.
        // Straight strip frame — dividers + bottom hairline, no BL swoop. Passing the strip height as `start` with empty crossings puts both hairlines and the bg fill on their no-curve path (`curve_active = start < height` is false), so the bottom-left is a clean right-angle.
        chrome::draw_strip_hairlines(
            chrome_buf,
            vp_w,
            vp_h,
            ctl,
            ctl,
            &[],
            edge_light,
        );

        // Per-row directional fills — runs AFTER hairlines but BEFORE symbols + bg fill. Each button anchors at the slot's inner edge (one pixel past its adjacent divider on the side opposite the slot's content boundary) and scans outward across the row toward the silhouette / strip edge. No inward scan needed: the divider itself is the inner boundary, and we start past it. MAX is the only button with dividers on BOTH sides — handled by splitting top half / bottom half between the two directions, so each half-row scan still only crosses one direction.
        let div1_col = button_area_x + ctl;
        let div2_col = button_area_x + 2 * ctl;
        let bound_x_min = strip_x;
        let bound_x_max = strip_x + strip_w;
        let min_id = self.min_btn.id();
        let max_id = self.max_btn.id();
        let close_id = self.close_btn.id();

        // MIN: anchor just left of div1, scan LEFT toward strip edge.
        chrome::paint_button_hit_row_scan(
            chrome_buf,
            clip_mask,
            &mut self.hit_test_map,
            buf_w,
            div1_col - 1,
            false,
            min_id,
            0,
            ctl,
            bound_x_min,
            bound_x_max,
        );

        // CLOSE: anchor just right of div2, scan RIGHT toward perimeter / silhouette.
        chrome::paint_button_hit_row_scan(
            chrome_buf,
            clip_mask,
            &mut self.hit_test_map,
            buf_w,
            div2_col + 1,
            true,
            close_id,
            0,
            ctl,
            bound_x_min,
            bound_x_max,
        );

        // MAX top half: anchor just right of div1, scan RIGHT. Stops at div2 (static wall).
        chrome::paint_button_hit_row_scan(
            chrome_buf,
            clip_mask,
            &mut self.hit_test_map,
            buf_w,
            div1_col + 1,
            true,
            max_id,
            0,
            ctl / 2,
            bound_x_min,
            bound_x_max,
        );

        // MAX bottom half: anchor just left of div2, scan LEFT. Stops at div1.
        chrome::paint_button_hit_row_scan(
            chrome_buf,
            clip_mask,
            &mut self.hit_test_map,
            buf_w,
            div2_col - 1,
            false,
            max_id,
            ctl / 2,
            ctl,
            bound_x_min,
            bound_x_max,
        );

        // Symbols painted AFTER flood-fill: now that hit_test_map is populated, glyph opacity in chrome_buf no longer affects the hit map (and the hit fill won't see them as walls since it already ran).
        chrome::draw_maximize_symbol(
            chrome_buf,
            buf_w,
            buf_h,
            max_cx,
            glyph_y,
            glyph_r,
            theme::MAXIMIZE_GLYPH,
            theme::MAXIMIZE_GLYPH_INTERIOR,
            theme::WINDOW_CONTROLS_BG,
        );
        chrome::draw_minimize_symbol(
            chrome_buf,
            buf_w,
            buf_h,
            min_cx,
            glyph_y,
            glyph_r,
            theme::MINIMIZE_GLYPH,
            theme::WINDOW_CONTROLS_BG,
        );
        chrome::draw_close_symbol(
            chrome_buf,
            buf_w,
            buf_h,
            close_cx,
            glyph_y,
            glyph_r,
            theme::CLOSE_GLYPH,
            theme::WINDOW_CONTROLS_BG,
        );

        chrome::draw_strip_bg(
            chrome_buf,
            &mut self.hit_test_map,
            vp_w,
            vp_h,
            ctl,
            ctl,
            &[],
        );

        // Status bar — bottom band, half the height of the top strip. Painted last (after every top-side chrome element) but the regions never overlap, so order is just for readability. `band_h = 0` ⇒ no-op rasterizer when `status_text` is `None` or empty. Hairline uses `edge_light` (same colour as the top strip's dividers); bg matches the top strip's `WINDOW_CONTROLS_BG` so both bands read as the same material. Title-text colour reuses the same focus-driven `title_colour` so the status text dims when the window is unfocused.
        let status_band_h = match self.status_text.as_deref() {
            Some(t) if !t.is_empty() => button_size / 2,
            _ => 0,
        };
        if status_band_h > 0 {
            let mut canvas = crate::canvas::Canvas::new(chrome_buf, buf_w, buf_h, damage);
            chrome::draw_status_bar(
                &mut canvas,
                status_band_h,
                theme::WINDOW_CONTROLS_BG,
                edge_light,
                self.status_text.as_deref().unwrap_or(""),
                text,
                title_colour,
            );
        }

        // Hover paint moved to the host overlay path; chrome_buf is intentionally tint-free here.

        // Silhouette restriction post-pass — photon-style. Any pixel the perimeter dropped below full coverage (clip_mask < 255 = corner cutouts, AA fringes) gets HIT_NONE so a non-strip rasterizer (currently the app icon at the top-left) can't leave a stale hit_id at a pixel that's geometrically outside the silhouette.
        for i in 0..clip_mask.len() {
            if clip_mask[i] < 255 {
                self.hit_test_map[i] = HIT_NONE;
            }
        }
    }

    /// Composite the chrome group (bg + chrome layers via internal Stack `Push chrome, Push bg, Under(Normal)`) and flatten under the present buffer. Front-to-back: chrome's composited result is blended `under` whatever's already in target, so chrome wins where opaque and bg shows thru where chrome is transparent.
    ///
    /// `clip`: optional damage-clip in target pixel coords; passed straight thru to [`Group::flatten_into`]. `None` = full target (current behavior).
    pub fn flatten_into(
        &mut self,
        target: &mut [u32],
        target_w: usize,
        target_h: usize,
        clip: Option<crate::paint::Clip>,
    ) {
        self.group.flatten_into(target, target_w, target_h, clip);
    }

    /// Hit query at `(x, y)` in viewport pixel coordinates. Returns the chrome button id at that pixel (one of `min_btn.id()` / `max_btn.id()` / `close_btn.id()` / `app_icon_btn.id()`) or `HIT_NONE` for pixels outside any chrome button or outside the viewport entirely.
    ///
    /// **Rule 0 — WHY/PROOF/PREVENTS:** WHY: a negative `x` cast to `usize` wraps to a huge value; without the bound check, indexing `hit_test_map[idx]` panics. PROOF: the host receives cursor coords from winit which can land outside the window during drag-resize. PREVENTS: panic on out-of-window cursor.
    pub fn hit_at(&self, x: Coord, y: Coord) -> HitId {
        let (w, h) = self.dims();
        let mx = x as i32;
        let my = y as i32;
        if (mx as usize) < w && (my as usize) < h {
            self.hit_test_map[(my as usize) * w + (mx as usize)]
        } else {
            HIT_NONE
        }
    }

    /// Damage region this chrome contributes to the host's per-frame clip rect. Returns `Some(viewport)` when bg or chrome layer needs a fresh rasterize (resize, focus change, debug toggle, scroll-driven bg); `None` otherwise. Hover is NOT damage anymore — it's an overlay operation against persistent_screen via `hit_test_map`, not a scratch repaint.
    pub fn damage_rect(&self) -> Option<crate::canvas::PixelRect> {
        let (w, h) = self.dims();
        let bg_dirty = self.group.rpn.layers[self.layer_bg].dirty;
        let chrome_dirty = self.group.rpn.layers[self.layer_chrome].dirty;
        if bg_dirty || chrome_dirty {
            Some(crate::canvas::PixelRect::new(0, 0, w, h))
        } else {
            None
        }
    }

    /// Read-only accessor so the host overlay pass can walk the hit-test map without taking ownership.
    pub fn hit_test_map(&self) -> &[HitId] {
        &self.hit_test_map
    }

    /// Wrap-add hover colour for a chrome button id. Returns `None` for ids that don't belong to chrome, ids that belong to chrome but don't have a hover tint (app-icon orb), or a theme entry of `0`. Single source of truth for the per-id overlay delta — consumers build their `overlay_deltas` table by asking chrome for its tint, then filling in their own widget tints.
    pub fn hover_colour_for(&self, hit: HitId) -> Option<u32> {
        let c = if hit == self.close_btn.id() {
            theme::CLOSE_HOVER
        } else if hit == self.max_btn.id() {
            theme::MAXIMIZE_HOVER
        } else if hit == self.min_btn.id() {
            theme::MINIMIZE_HOVER
        } else {
            return None;
        };
        if c == 0 { None } else { Some(c) }
    }

    /// True when `hit` is one of this chrome's button ids (min / max / close / app-icon). Use to decide chrome-vs-content routing without comparing against four ids manually — for example, panes' `cursor_for` shows the pointer for any chrome button id.
    pub fn owns_hit(&self, hit: HitId) -> bool {
        hit != HIT_NONE
            && (hit == self.min_btn.id()
                || hit == self.max_btn.id()
                || hit == self.close_btn.id()
                || hit == self.app_icon_btn.id())
    }

    /// Update the hover state if `new_hit` differs from the current. Returns `true` iff the state changed (so the consumer knows to request a redraw — the host's overlay pass picks up the new state and applies the visible-RGB delta to persistent_screen, no scratch repaint needed).
    ///
    /// Side effect: synchronises each of the four [`ChromeButton`]s' `hovered` field so widget-tree walks (e.g. [`super::widget::build_overlay_deltas`]) can read per-button state directly via [`Hover`]. Without this sync chrome buttons would always report `hovered = false` to the walker.
    pub fn set_hover(&mut self, new_hit: HitId) -> bool {
        if new_hit == self.hover_state {
            return false;
        }
        self.hover_state = new_hit;
        self.min_btn.hovered = new_hit == self.min_btn.id();
        self.max_btn.hovered = new_hit == self.max_btn.id();
        self.close_btn.hovered = new_hit == self.close_btn.id();
        self.app_icon_btn.hovered = new_hit == self.app_icon_btn.id();
        true
    }

    /// Update window-focus state. Returns `true` iff the value changed. Host wires this from `WindowEvent::Focused`. Marks the chrome layer dirty so the focus palette swap re-rasterizes on the next paint.
    pub fn set_focused(&mut self, focused: bool) -> bool {
        if focused == self.focused {
            return false;
        }
        self.focused = focused;
        self.group.rpn.layers[self.layer_chrome].dirty = true;
        true
    }

    /// Toggle full-edge / maximized rendering. Returns `true` iff the value changed. App calls this from `on_resize` (or right after [`super::app::EventResponse::ToggleMaximized`] takes effect) — read [`super::app::Context::is_maximized`] for the host's source-of-truth state. Marks the chrome layer dirty so the next paint either drops or restores the perimeter hairline.
    pub fn set_full_edge(&mut self, full_edge: bool) -> bool {
        if full_edge == self.full_edge {
            return false;
        }
        self.full_edge = full_edge;
        self.group.rpn.layers[self.layer_chrome].dirty = true;
        true
    }

    /// Update the orb tint. Returns `true` iff the value changed. App calls this when the orb's semantic state shifts (network came online, recording started, presence flipped). Marks the chrome layer dirty.
    pub fn set_orb_tint(&mut self, tint: chrome::OrbTint) -> bool {
        if tint == self.orb_tint {
            return false;
        }
        self.orb_tint = tint;
        self.group.rpn.layers[self.layer_chrome].dirty = true;
        true
    }

    /// Update the status bar text. `None` or empty string hides the band entirely. Returns `true` iff the value changed. Marks the chrome layer dirty so the band re-rasterizes (or vanishes) on the next paint.
    pub fn set_status_text(&mut self, text: Option<String>) -> bool {
        let changed = match (&self.status_text, &text) {
            (None, None) => false,
            (Some(a), Some(b)) => a != b,
            _ => true,
        };
        if !changed {
            return false;
        }
        self.status_text = text;
        self.group.rpn.layers[self.layer_chrome].dirty = true;
        true
    }

    /// Mark the bg layer dirty (consumer should call when their bg content needs repaint — pane edits, animation tick, etc.).
    pub fn invalidate_bg(&mut self) {
        self.group.rpn.layers[self.layer_bg].dirty = true;
    }

    /// Mark the chrome layer dirty (consumer calls when title changes; chrome is otherwise stable across a viewport size).
    pub fn invalidate_chrome(&mut self) {
        self.group.rpn.layers[self.layer_chrome].dirty = true;
    }

    /// Set the title-bar text, marking the chrome layer dirty only when it actually changed. Returns `true` if a repaint is needed. Mirrors [`set_status_text`](Self::set_status_text) — the idiomatic way to drive a dynamic title (e.g. a per-screen label) without re-rasterizing chrome every frame.
    pub fn set_title(&mut self, title: impl Into<String>) -> bool {
        let title = title.into();
        if self.title == title {
            return false;
        }
        self.title = title;
        self.group.rpn.layers[self.layer_chrome].dirty = true;
        true
    }
}

/// Visit the four chrome buttons in tab order (app-icon → minimize → maximize → close). Order is the keyboard-discoverability convention: the orb sits visually leftmost so it leads in left-to-right reading order, then the window controls flow right-to-left from minimize to close. The app's outer [`Container`] decides where the chrome's buttons land in the overall cycle (typically AFTER content widgets like textboxes, matching macOS / GNOME).
impl Container for DefaultChrome {
    fn visit(&mut self, f: &mut dyn FnMut(&mut dyn Widget)) {
        f(&mut self.app_icon_btn);
        f(&mut self.min_btn);
        f(&mut self.max_btn);
        f(&mut self.close_btn);
    }
}

/// Compute squircle crossings table for a corner with the given `radius` and `squirdleyness`. Returns `(start, crossings)` where `start` is the distance from the corner-of-corner inward to the curve's first integer-row crossing, and `crossings` is the rev'd table indexed by `i in 0..count` such that at row offset `start + i` the curve is at column `inset_i` with AA values `h_cov_i` and `l_i`. Used for both the window perimeter (radius = span/4) and the controls-strip BL curve (radius = button_size).
pub(crate) fn compute_squircle_crossings(
    radius: Coord,
    squirdleyness: i32,
) -> (usize, Vec<(u16, u8, u8)>) {
    let mut crossings: Vec<(u16, u8, u8)> = Vec::new();
    if radius <= 0.0 {
        return (0, crossings);
    }
    let mut y = 1f32;
    loop {
        let y_norm = y / radius;
        // For y_norm > 1, the squircle equation gives a negative inner term — clamp to 0 so the (1/p) root is well-defined. This makes x = 0 → x < y → break, preventing the loop from spinning forever on tiny radii.
        let inner = (1.0 - crate::math::powi(y_norm, squirdleyness)).max(0.0);
        let x_norm = crate::math::powf(inner, 1.0 / squirdleyness as Coord);
        let x = x_norm * radius;
        let inset = radius - x;
        if inset > 0.0 {
            crossings.push((
                inset as u16,
                (crate::math::sqrt(crate::math::fract(inset)) * 256.0) as u8,
                (crate::math::sqrt(1.0 - crate::math::fract(inset)) * 256.0) as u8,
            ));
        }
        if x < y {
            break;
        }
        y += 1.0;
    }
    let start = (radius - y) as usize;
    let crossings: Vec<(u16, u8, u8)> = crossings.into_iter().rev().collect();
    (start, crossings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allocates_full_viewport_buffers() {
        let mut counter: HitId = HIT_NONE;
        let chrome = DefaultChrome::new(Viewport::new(800, 600), "test", None, None, &mut counter);
        assert_eq!(chrome.dims(), (800, 600));
        assert_eq!(chrome.hit_test_map.len(), 800 * 600);
        assert_eq!(chrome.title, "test");
        assert_eq!(chrome.hover_state, HIT_NONE);
    }

    #[test]
    fn hit_at_outside_viewport_returns_hit_none() {
        let mut counter: HitId = HIT_NONE;
        let chrome = DefaultChrome::new(Viewport::new(100, 100), "", None, None, &mut counter);
        assert_eq!(chrome.hit_at(-1.0, 50.0), HIT_NONE);
        assert_eq!(chrome.hit_at(50.0, -1.0), HIT_NONE);
        assert_eq!(chrome.hit_at(101.0, 50.0), HIT_NONE);
        assert_eq!(chrome.hit_at(50.0, 101.0), HIT_NONE);
    }

    #[test]
    fn set_hover_returns_true_on_change_only() {
        let mut counter: HitId = HIT_NONE;
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "", None, None, &mut counter);
        assert!(chrome.set_hover(chrome.close_btn.id())); // changed
        assert!(!chrome.set_hover(chrome.close_btn.id())); // same
        assert!(chrome.set_hover(HIT_NONE)); // changed back
    }

    #[test]
    fn set_hover_does_not_dirty_chrome_layer() {
        let mut counter: HitId = HIT_NONE;
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "", None, None, &mut counter);
        // Run a flatten cycle so StackCompositor::evaluate clears all initial-dirty flags.
        let mut target = alloc::vec![0u32; 100 * 100];
        chrome.flatten_into(&mut target, 100, 100, None);
        assert!(!chrome.group.rpn.layers[chrome.layer_chrome].dirty);
        chrome.set_hover(chrome.close_btn.id());
        // Hover tint lives entirely in the host overlay pass against persistent_screen; no scratch repaint is needed, so the chrome layer must stay clean.
        assert!(!chrome.group.rpn.layers[chrome.layer_chrome].dirty);
    }

    #[test]
    fn resize_marks_layers_dirty_and_resizes_hit_map() {
        let mut counter: HitId = HIT_NONE;
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "", None, None, &mut counter);
        let mut target = alloc::vec![0u32; 100 * 100];
        chrome.flatten_into(&mut target, 100, 100, None);
        chrome.resize(Viewport::new(200, 150));
        assert_eq!(chrome.dims(), (200, 150));
        assert_eq!(chrome.hit_test_map.len(), 200 * 150);
        // Group::resize marks all layers dirty.
        assert!(chrome.group.rpn.layers[chrome.layer_bg].dirty);
        assert!(chrome.group.rpn.layers[chrome.layer_chrome].dirty);
    }
}
