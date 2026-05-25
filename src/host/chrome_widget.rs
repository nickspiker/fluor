//! `DefaultChrome` — the reusable borderless-window frame consumers compose into their `FluorApp`.
//!
//! Owns a full-viewport [`Group`] containing three layers — bg (caller paints), chrome (controls + edges + hairlines + title text), hover (button delta) — plus the per-pixel `hit_test_map` byte buffer that records which button (if any) covers each pixel. Hover state, the cached hover-pixel list, and the title string all live here so the consumer can drop in chrome with one struct field.
//!
//! Built on the verbatim photon primitives in [`super::chrome`] — `draw_window_controls`, `draw_window_edges_and_mask`, `draw_button_hairlines`, `draw_button_hover_by_pixels`, `pixels_for_button`. Those stay; this module is a stateful wrapper that schedules them against the chrome group's dirty layers.
//!
//! Pattern: `chrome.rasterize_bg(|bg, w, h| { /* paint into bg */ });` → `chrome.rasterize_chrome(text);` → `chrome.rasterize_hover();` → `chrome.flatten_into(target, w, h);`. Each rasterize_* checks the layer's dirty bit internally and is a no-op on clean.

use super::chrome::{self, HIT_NONE};
use crate::coord::Coord;
use crate::geom::Viewport;
use crate::group::Group;
use crate::paint::BlendMode;
use crate::region::Region;
use crate::stack::Op;
use crate::text::TextRenderer;
use crate::theme;
use alloc::string::String;
use alloc::vec::Vec;

/// Reusable window frame: controls, edges, hairlines, title, hover overlay.
pub struct DefaultChrome {
    /// Full-viewport Group with 3 layers (bg, chrome, hover) composed via Stack Notation. Topmost-first: `Push chrome, Push bg, Under(Normal)` for the minimal scaffold; expand to include hover via `Push hover, Push chrome, Under(Add), Push bg, Under(Normal)` as the design grows.
    pub group: Group,
    /// Per-pixel button-id map. `0` = HIT_NONE; non-zero = a chrome button (`HIT_MINIMIZE_BUTTON` / `HIT_MAXIMIZE_BUTTON` / `HIT_CLOSE_BUTTON`). Sized to `width * height` pixels of the current viewport.
    pub hit_test_map: Vec<u8>,
    /// Window title rendered into the chrome layer (left-aligned in the controls strip). Empty string = skip text rendering.
    pub title: String,
    /// Optional app-icon orb painted in the top-left chrome slot. `None` = no orb, title text starts at the left margin. When `Some`, [`chrome::draw_app_icon`] runs after the perimeter and the title text shifts right by `button_size + button_size/4` so it doesn't overlap.
    pub app_icon: Option<crate::host::icon::Icon>,
    /// Window-focus state. `true` = active (full edge bevel, bright title, ring follows perimeter, icon at full saturation). `false` = inactive (edges + title + orb ring collapse to `LABEL_COLOUR`; orb image desaturates 50 % toward grey when `orb_tint` is `FollowFocus`). Host wires this from `WindowEvent::Focused`. Mutate via [`set_focused`](Self::set_focused) to mark the chrome layer dirty automatically.
    pub focused: bool,
    /// Orb visual state. Default `OrbTint::FollowFocus` makes the orb a window-state indicator; `OrbTint::Custom` lets the app turn it into a network/recording/presence badge. Mutate via [`set_orb_tint`](Self::set_orb_tint) to mark the chrome layer dirty automatically.
    pub orb_tint: chrome::OrbTint,
    /// Currently-hovered button id (HIT_NONE if none). Rasterized as a colour delta on the hover layer.
    pub hover_state: u8,
    /// Cached pixel index list for the currently hovered button — recomputed on hover-state change.
    pub hover_pixel_list: Vec<usize>,
    /// Last viewport passed to `new` or `resize`. Stored so chrome rasterization can read `effective_span` (= `span * ru`) and pick up the user's zoom multiplier automatically — chrome control sizing scales with the same `ceil(effective_span/32)` formula, so Ctrl+/ Ctrl-/ Ctrl+scroll zoom the chrome together with content.
    viewport: Viewport,
    layer_bg: usize,
    layer_chrome: usize,
    layer_hover: usize,
}

impl DefaultChrome {
    /// Allocate the chrome group + hit_test_map sized to `viewport`. Three layers (bg, chrome, hover) all start dirty so the first frame paints from scratch.
    ///
    /// **Topmost-first scaffold:** the Stack program is the minimal front-to-back composite — `Push chrome, Push bg, Under(Normal)`. Chrome is the topmost layer (controls, edges, hairlines, title), bg is the layer behind it (background_noise + panes). Stack order matches the visual stack: first push lands on the bottom of the eval stack and is the topmost layer; second push goes underneath via `Under`. The hover layer still exists and is rasterized so the API surface is stable; it's omitted from the program until the hover overlay is wired back up via `Push hover, Under(Add)` as the topmost step. Corner knockout (formerly a separate silhouette layer + `Op::Or`) is handled at chrome rasterization time by writing `t=255` directly into the chrome layer's corner pixels — no separate Stack op under the unified Under model.
    pub fn new(viewport: Viewport, title: impl Into<String>, app_icon: Option<crate::host::icon::Icon>) -> Self {
        let region = Region::new(
            0.0,
            0.0,
            viewport.width_px as Coord,
            viewport.height_px as Coord,
        );
        let mut group = Group::new(region, BlendMode::Normal);
        let layer_bg = group.new_layer();
        let layer_chrome = group.new_layer();
        let layer_hover = group.new_layer();
        // Front-to-back: chrome on top (controls + edges + hairlines + hover tint baked in), bg underneath (panes + background_noise). The hover layer is allocated for forward-compat with future designs that promote it to a separate Stack operand, but it's NOT in the program — hover tint is baked into the chrome layer at rasterization time via `pixels[i].under(tint, Normal)` (which composes correctly because rasterizer's Under expects straight-α bottoms; stacking a separate premultiplied hover layer would re-premultiply chrome's partial-α AA edges and trash them).
        group.set_program(alloc::vec![
            Op::Push(layer_chrome),
            Op::Push(layer_bg),
            Op::Under(BlendMode::Normal),
        ]);
        let map_len = (viewport.width_px as usize).saturating_mul(viewport.height_px as usize);
        Self {
            group,
            hit_test_map: alloc::vec![HIT_NONE; map_len],
            title: title.into(),
            app_icon,
            focused: true,
            orb_tint: chrome::OrbTint::FollowFocus,
            hover_state: HIT_NONE,
            hover_pixel_list: Vec::new(),
            viewport,
            layer_bg,
            layer_chrome,
            layer_hover,
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
    pub fn dims(&self) -> (usize, usize) {
        self.group.dims()
    }

    /// Paint the bg layer with consumer-supplied content. The closure receives `(pixels, width, height)` for the bg layer (full-viewport sized). No-op if the layer is clean — call [`invalidate_bg`](Self::invalidate_bg) to force a repaint (e.g., when pane content changes).
    pub fn rasterize_bg(&mut self, paint: impl FnOnce(&mut [u32], usize, usize)) {
        let (w, h) = self.dims();
        let layer = &mut self.group.rpn.layers[self.layer_bg];
        if !layer.dirty {
            return;
        }
        // α + darkness: transparent init (α=0) so pixels the closure doesn't paint stay transparent rather than becoming spurious opaque content. The closure is expected to fully cover the bg, but defaulting to transparent is the safe failure mode. Zero-init is calloc-free.
        layer.pixels.fill(0);
        paint(&mut layer.pixels, w, h);
    }

    /// **Scaffold step 1 (top of stack: AA rounded hairline only):** paint the squircle-cornered window-perimeter hairline into the chrome layer via [`chrome::draw_window_edges_and_mask`]. Chrome layer carries OPAQUE RGB only at the hairline; partial-α window-shape coverage (corner curve AA + outside-curve cutout) is written into `clip_mask` so the OS boundary can fold it into the final alpha in one pass.
    ///
    /// `text` is used by the title text rasterization pass (Open Sans, span-relative font size, left-aligned in the area to the left of the controls strip).
    pub fn rasterize_chrome(&mut self, text: &mut TextRenderer, clip_mask: &mut [u8]) {
        let (buf_w, buf_h) = self.dims();
        let vp_w = buf_w as u32;
        let vp_h = buf_h as u32;

        if !self.group.rpn.layers[self.layer_chrome].dirty {
            return;
        }

        // Reset clip_mask to 255 (fully visible) BEFORE re-carving. The corner cutout is a side effect of `chrome::draw_window_edges_and_mask`, so it MUST run on the same dirty-cycle that resets the mask — otherwise the old (larger or smaller) carving persists. Done here (inside the dirty check) rather than in the host's render_frame, because a per-frame reset there would wipe the carving on frames where chrome is clean and never re-carve, producing rectangular windows whenever the cursor is idle for a tick.
        clip_mask.fill(255);
        self.hit_test_map.fill(HIT_NONE);

        let chrome_buf = &mut self.group.rpn.layers[self.layer_chrome].pixels;
        // α + darkness: transparent init (α=0, dark=0) so the bg shows through everywhere except the hairline + AA pixels.
        chrome_buf.fill(0);

        if vp_w < 2 || vp_h < 2 {
            return;
        }

        // Compute span + button size shared by controls and squircle. Use the viewport's `effective_span` (= `span * ru`) so chrome scales with the user's zoom — Ctrl+/Ctrl-/Ctrl+scroll zoom the chrome together with content.
        let span = self.viewport.effective_span();
        // Span-relative: button height is span/32, where span is the harmonic mean of viewport dims times zoom. Strip layout bails downstream if the result is too small to render glyphs.
        let button_size = crate::math::ceil(span / 32.0) as usize;

        // Single squircle (radius = span/4, squirdleyness 24) shared by the window perimeter AND the controls-strip BL curve — same shape as photon. At typical viewport sizes the curve is too big to fit in the strip and degrades to a rectangular bottom; at high zoom it appears.
        let (start, crossings) = compute_squircle_crossings(span / 4.0, 24);
        // No curve to draw — bail. start=0 is fine (just means the corner-of-corner cutout is empty); curve walks handle it.
        if crossings.is_empty() {
            return;
        }

        // Controls-strip layout. Computed early so the title text pass can clip against `strip_x` (title shouldn't paint over the buttons even at long titles or narrow windows). The strip lives in the top-right `button_size`-tall band, `strip_w` wide.
        let strip_w = button_size * 7 / 2;
        let strip_x = buf_w.saturating_sub(strip_w);

        // App-icon orb layout: centered in the top-left `button_size`-tall band, mirroring the right-side controls strip. Diameter is half the band height so the orb reads as a tasteful badge rather than a full button. Title text shifts right by the orb's footprint when an icon is present. `draw_app_icon` has an `r < 2` early-return so degenerate sizes pass through without drawing — no min-size guard needed here.
        let orb_present = self.app_icon.is_some();
        let orb_diameter = if orb_present {
            (button_size / 2) as isize
        } else {
            0
        };
        let orb_radius = orb_diameter / 2;
        let orb_cx = (button_size / 2) as isize;
        let orb_cy = (button_size / 2) as isize;
        let title_left_extra = if orb_present {
            orb_diameter as usize
        } else {
            0
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
        // Ctrl+Shift+D+C: skip the window edge/perimeter AND title text (both are "decoration"). Controls still render. clip_mask stays at host default (255 everywhere), so the window appears as a rectangle (no rounded corners).
        // Focus-driven palette. Each element pulls from a named theme constant so a downstream consumer can override (e.g. an app that wants a totally different unfocused look) by swapping the theme module rather than re-implementing the rasterizer wiring.
        let (edge_light, edge_shadow, title_color) = if self.focused {
            (
                theme::WINDOW_LIGHT_EDGE,
                theme::WINDOW_SHADOW_EDGE,
                theme::TEXT_COLOUR,
            )
        } else {
            (
                theme::WINDOW_LIGHT_EDGE_UNFOCUSED,
                theme::WINDOW_SHADOW_EDGE_UNFOCUSED,
                theme::TEXT_COLOUR_UNFOCUSED,
            )
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
            chrome::draw_window_edges_and_mask(
                chrome_buf,
                &mut self.hit_test_map,
                clip_mask,
                vp_w,
                vp_h,
                start,
                &crossings,
                edge_light,
                edge_shadow,
            );
            if orb_present {
                chrome::draw_app_icon(
                    chrome_buf,
                    Some(&mut self.hit_test_map),
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
            chrome::draw_title_text(
                chrome_buf,
                buf_w,
                buf_h,
                &self.title,
                text,
                button_size,
                strip_x,
                title_left_extra,
                title_color,
            );
        }

        // Ctrl+Shift+D+X: skip ONLY the controls strip (perimeter + title stay).
        if crate::paint::DEBUG_SKIP_CONTROLS.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }

        let button_area_x = strip_x + button_size / 4;
        let glyph_y = button_size / 2;
        let glyph_r = button_size / 4;
        let min_cx = button_area_x + button_size / 2;
        let max_cx = button_area_x + button_size + button_size / 2;
        let close_cx = button_area_x + button_size * 2 + button_size / 2;

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

        // Hairlines (dividers + bottom) BEFORE curves: solid lines have to win at intersection pixels, otherwise the curve's inner-AA hairline (which is mostly transparent in the linear region) fragments them. With this order, dividers and bottom hairline claim their pixels first; the BL curve's hairlines fill in only the gaps the straight lines didn't reach.
        //
        // Strip-frame colours follow the focus palette: vertical dividers + bottom hairline take `edge_light` (same as the top/left window perimeter), the BL squircle curve's vertical face takes `edge_light` (continues the left-of-strip vertical hairline) and its horizontal face takes `edge_shadow` (continues the bottom-of-strip hairline if no curve is present, and matches the bottom-of-window shadow) — so the strip reads as a continuation of the window edge, not a separate piece.
        chrome::draw_strip_hairlines(
            chrome_buf,
            vp_w,
            vp_h,
            button_size,
            start,
            &crossings,
            edge_light,
        );

        chrome::draw_strip_curves(
            chrome_buf,
            &mut self.hit_test_map,
            vp_w,
            vp_h,
            button_size,
            start,
            &crossings,
            edge_light,
            edge_shadow,
        );

        chrome::draw_strip_bg(
            chrome_buf,
            &mut self.hit_test_map,
            vp_w,
            vp_h,
            button_size,
            start,
            &crossings,
        );

        // Hover overlay: raw wrap-add of the hover colour's darkness into every chrome pixel whose hit_test_map matches the current hover_state. Done LAST so `hit_test_map` is fully populated. This is photon's intentional-wrap hover effect — `chrome_dark.wrapping_add(hover_dark)` per channel, α stays opaque. The wrap is the point: a cyan/magenta/yellow hover colour added to chrome's existing darkness shifts the visible RGB by exactly the hover colour (no overflow = identity), giving a distinct hover state without obscuring the glyph. `.under()` won't work here because chrome's button pixels are opaque after `draw_strip_bg`, so the opaque-top early-out fires and any blend gets silently dropped.
        if self.hover_state != HIT_NONE {
            let hover_color = match self.hover_state {
                chrome::HIT_CLOSE_BUTTON => theme::CLOSE_HOVER,
                chrome::HIT_MAXIMIZE_BUTTON => theme::MAXIMIZE_HOVER,
                chrome::HIT_MINIMIZE_BUTTON => theme::MINIMIZE_HOVER,
                _ => 0,
            };
            if hover_color != 0 {
                let h_r = ((hover_color >> 16) & 0xFF) as u8;
                let h_g = ((hover_color >> 8) & 0xFF) as u8;
                let h_b = (hover_color & 0xFF) as u8;
                for (i, &hit) in self.hit_test_map.iter().enumerate() {
                    if hit == self.hover_state {
                        let p = chrome_buf[i];
                        let a = p & 0xFF000000;
                        let r = (((p >> 16) & 0xFF) as u8).wrapping_add(h_r) as u32;
                        let g = (((p >> 8) & 0xFF) as u8).wrapping_add(h_g) as u32;
                        let b = ((p & 0xFF) as u8).wrapping_add(h_b) as u32;
                        chrome_buf[i] = a | (r << 16) | (g << 8) | b;
                    }
                }
            }
        }
    }

    /// **No-op stub.** Hover tint is baked into the chrome layer in `rasterize_chrome` directly (last pass), not maintained as a separate Stack layer — see the explanation in `new()` for why. Kept as a public method for forward-compat with future designs that may promote hover to a separate Stack operand once an `under_premult` for premultiplied-layer stacking exists.
    pub fn rasterize_hover(&mut self) {}

    /// Composite the chrome group (bg + chrome layers via internal Stack `Push chrome, Push bg, Under(Normal)`) and flatten under the present buffer. Front-to-back: chrome's composited result is blended `under` whatever's already in target, so chrome wins where opaque and bg shows through where chrome is transparent.
    pub fn flatten_into(&mut self, target: &mut [u32], target_w: usize, target_h: usize) {
        self.group.flatten_into(target, target_w, target_h);
    }

    /// Hit query at `(x, y)` in viewport pixel coordinates. Returns the chrome button id (HIT_NONE / HIT_MINIMIZE_BUTTON / HIT_MAXIMIZE_BUTTON / HIT_CLOSE_BUTTON). `(x, y)` outside the viewport returns HIT_NONE.
    ///
    /// **Rule 0 — WHY/PROOF/PREVENTS:** WHY: a negative `x` cast to `usize` wraps to a huge value; without the bound check, indexing `hit_test_map[idx]` panics. PROOF: the host receives cursor coords from winit which can land outside the window during drag-resize. PREVENTS: panic on out-of-window cursor.
    pub fn hit_at(&self, x: Coord, y: Coord) -> u8 {
        let (w, h) = self.dims();
        let mx = x as i32;
        let my = y as i32;
        if (mx as usize) < w && (my as usize) < h {
            self.hit_test_map[(my as usize) * w + (mx as usize)]
        } else {
            HIT_NONE
        }
    }

    /// Update the hover state if `new_hit` differs from the current. Returns `true` iff the state changed (so the consumer knows to request a redraw). Marks the chrome layer dirty (not a separate hover layer) — hover tint is baked into chrome at rasterize time.
    pub fn set_hover(&mut self, new_hit: u8) -> bool {
        if new_hit == self.hover_state {
            return false;
        }
        self.hover_state = new_hit;
        self.group.rpn.layers[self.layer_chrome].dirty = true;
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

    /// Update the orb tint. Returns `true` iff the value changed. App calls this when the orb's semantic state shifts (network came online, recording started, presence flipped). Marks the chrome layer dirty.
    pub fn set_orb_tint(&mut self, tint: chrome::OrbTint) -> bool {
        if tint == self.orb_tint {
            return false;
        }
        self.orb_tint = tint;
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
        let chrome = DefaultChrome::new(Viewport::new(800, 600), "test", None);
        assert_eq!(chrome.dims(), (800, 600));
        assert_eq!(chrome.hit_test_map.len(), 800 * 600);
        assert_eq!(chrome.title, "test");
        assert_eq!(chrome.hover_state, HIT_NONE);
    }

    #[test]
    fn hit_at_outside_viewport_returns_hit_none() {
        let chrome = DefaultChrome::new(Viewport::new(100, 100), "", None);
        assert_eq!(chrome.hit_at(-1.0, 50.0), HIT_NONE);
        assert_eq!(chrome.hit_at(50.0, -1.0), HIT_NONE);
        assert_eq!(chrome.hit_at(101.0, 50.0), HIT_NONE);
        assert_eq!(chrome.hit_at(50.0, 101.0), HIT_NONE);
    }

    #[test]
    fn set_hover_returns_true_on_change_only() {
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "", None);
        assert!(chrome.set_hover(chrome::HIT_CLOSE_BUTTON)); // changed
        assert!(!chrome.set_hover(chrome::HIT_CLOSE_BUTTON)); // same
        assert!(chrome.set_hover(HIT_NONE)); // changed back
    }

    #[test]
    fn set_hover_marks_chrome_layer_dirty() {
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "", None);
        // Run a flatten cycle so StackCompositor::evaluate clears all initial-dirty flags.
        let mut target = alloc::vec![0u32; 100 * 100];
        chrome.flatten_into(&mut target, 100, 100);
        assert!(!chrome.group.rpn.layers[chrome.layer_chrome].dirty);
        chrome.set_hover(chrome::HIT_CLOSE_BUTTON);
        // Hover tint is baked into the chrome layer (not a separate hover stack operand), so a hover state change invalidates chrome.
        assert!(chrome.group.rpn.layers[chrome.layer_chrome].dirty);
    }

    #[test]
    fn resize_marks_layers_dirty_and_resizes_hit_map() {
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "", None);
        let mut target = alloc::vec![0u32; 100 * 100];
        chrome.flatten_into(&mut target, 100, 100);
        chrome.resize(Viewport::new(200, 150));
        assert_eq!(chrome.dims(), (200, 150));
        assert_eq!(chrome.hit_test_map.len(), 200 * 150);
        // Group::resize marks all layers dirty.
        assert!(chrome.group.rpn.layers[chrome.layer_hover].dirty);
    }
}
