//! `DefaultChrome` — the reusable borderless-window frame consumers compose into their `FluorApp`.
//!
//! Owns a full-viewport [`Group`] containing three layers — bg (caller paints), chrome (controls + edges + hairlines + title text), hover (button delta) — plus the per-pixel `hit_test_map` byte buffer that records which button (if any) covers each pixel. Hover state, the cached hover-pixel list, and the title string all live here so the consumer can drop in chrome with one struct field.
//!
//! Built on the verbatim photon primitives in [`super::chrome`] — `draw_window_controls`, `draw_window_edges_and_mask`, `draw_button_hairlines`, `draw_button_hover_by_pixels`, `pixels_for_button`. Those stay; this module is a stateful wrapper that schedules them against the chrome group's dirty layers.
//!
//! Pattern: `chrome.rasterize_bg(|bg, w, h| { /* paint into bg */ });` → `chrome.rasterize_chrome(text);` → `chrome.rasterize_hover();` → `chrome.flatten_into(target, w, h);`. Each rasterize_* checks the layer's dirty bit internally and is a no-op on clean.

use super::chrome::{self, HIT_NONE, MIN_BUTTON_HEIGHT_PX};
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
    /// Currently-hovered button id (HIT_NONE if none). Rasterized as a colour delta on the hover layer.
    pub hover_state: u8,
    /// Cached pixel index list for the currently hovered button — recomputed on hover-state change.
    pub hover_pixel_list: Vec<usize>,
    layer_bg: usize,
    layer_chrome: usize,
    layer_hover: usize,
}

impl DefaultChrome {
    /// Allocate the chrome group + hit_test_map sized to `viewport`. Three layers (bg, chrome, hover) all start dirty so the first frame paints from scratch.
    ///
    /// **Topmost-first scaffold:** the Stack program is the minimal front-to-back composite — `Push chrome, Push bg, Under(Normal)`. Chrome is the topmost layer (controls, edges, hairlines, title), bg is the layer behind it (background_noise + panes). Stack order matches the visual stack: first push lands on the bottom of the eval stack and is the topmost layer; second push goes underneath via `Under`. The hover layer still exists and is rasterized so the API surface is stable; it's omitted from the program until the hover overlay is wired back up via `Push hover, Under(Add)` as the topmost step. Corner knockout (formerly a separate silhouette layer + `Op::Or`) is handled at chrome rasterization time by writing `t=255` directly into the chrome layer's corner pixels — no separate Stack op under the unified Under model.
    pub fn new(viewport: Viewport, title: impl Into<String>) -> Self {
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
            hover_state: HIT_NONE,
            hover_pixel_list: Vec::new(),
            layer_bg,
            layer_chrome,
            layer_hover,
        }
    }

    /// Resize the chrome group + hit_test_map to a new viewport. All layers go dirty.
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
        // t-convention: transparent init so any pixels the closure doesn't paint don't end up
        // as opaque black (t=0). The closure is expected to fully cover the bg, but defaulting
        // to transparent is the safe failure mode.
        layer.pixels.fill(0xFFFFFFFF);
        paint(&mut layer.pixels, w, h);
    }

    /// **Scaffold step 1 (top of stack: AA rounded hairline only):** paint the squircle-cornered window-perimeter hairline into the chrome layer via [`chrome::draw_window_edges_and_mask`]. Chrome layer carries OPAQUE RGB only at the hairline; partial-α window-shape coverage (corner curve AA + outside-curve cutout) is written into `clip_mask` so the OS boundary can fold it into the final alpha in one pass.
    ///
    /// The `text` parameter is accepted for forward-compat with the title text pass; currently unused.
    pub fn rasterize_chrome(&mut self, _text: &mut TextRenderer, clip_mask: &mut [u8]) {
        let (buf_w, buf_h) = self.dims();
        let vp_w = buf_w as u32;
        let vp_h = buf_h as u32;

        if !self.group.rpn.layers[self.layer_chrome].dirty {
            return;
        }

        self.hit_test_map.fill(HIT_NONE);

        let chrome_buf = &mut self.group.rpn.layers[self.layer_chrome].pixels;
        // t-convention: transparent init so the bg shows through everywhere except the hairline + AA pixels.
        chrome_buf.fill(0xFFFFFFFF);

        if vp_w < 2 || vp_h < 2 {
            return;
        }

        // Compute squircle start + crossings (lifted from `chrome::draw_window_controls`, button bits stripped).
        let span = 2.0 * vp_w as Coord * vp_h as Coord / (vp_w as Coord + vp_h as Coord);
        let radius = span / 4.0;
        let squirdleyness = 24i32;
        let mut crossings: Vec<(u16, u8, u8)> = Vec::new();
        let mut y = 1f32;
        loop {
            let y_norm = y / radius;
            let x_norm = crate::math::powf(
                1.0 - crate::math::powi(y_norm, squirdleyness),
                1.0 / squirdleyness as Coord,
            );
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

        if start == 0 || crossings.is_empty() {
            return;
        }

        chrome::draw_window_edges_and_mask(
            chrome_buf,
            &mut self.hit_test_map,
            clip_mask,
            vp_w,
            vp_h,
            start,
            &crossings,
        );

        // Ctrl+Shift+D+C: suppress chrome RGB so the layers underneath show through. Clip-mask carving (above) is preserved so the window-shape trim stays visible.
        if crate::paint::DEBUG_SKIP_CHROME.load(std::sync::atomic::Ordering::Relaxed) {
            chrome_buf.fill(0xFFFFFFFF);
        }
    }

    /// Paint the hover-overlay delta if the hover layer is dirty. The delta is added (per-channel wrap) onto the chrome layer at the currently-hovered button's pixel positions; on hover_state == HIT_NONE the layer is just zeroed (Add of 0 = no-op).
    /// Stub for the future hover-overlay scaffold step. Currently leaves the hover layer at the
    /// canonical empty value (no-op). The button hover effect depends on the controls scaffold
    /// (which builds the per-pixel hit_test_map of button regions); it returns alongside that.
    pub fn rasterize_hover(&mut self) {
        let layer = &mut self.group.rpn.layers[self.layer_hover];
        if !layer.dirty {
            return;
        }
        layer.pixels.fill(0xFFFFFFFF);
        layer.dirty = false;
    }

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

    /// Update the hover state if `new_hit` differs from the current. Returns `true` iff the state changed (so the consumer knows to invalidate the hover layer + request_redraw).
    pub fn set_hover(&mut self, new_hit: u8) -> bool {
        if new_hit == self.hover_state {
            return false;
        }
        self.hover_state = new_hit;
        self.group.rpn.layers[self.layer_hover].dirty = true;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allocates_full_viewport_buffers() {
        let chrome = DefaultChrome::new(Viewport::new(800, 600), "test");
        assert_eq!(chrome.dims(), (800, 600));
        assert_eq!(chrome.hit_test_map.len(), 800 * 600);
        assert_eq!(chrome.title, "test");
        assert_eq!(chrome.hover_state, HIT_NONE);
    }

    #[test]
    fn hit_at_outside_viewport_returns_hit_none() {
        let chrome = DefaultChrome::new(Viewport::new(100, 100), "");
        assert_eq!(chrome.hit_at(-1.0, 50.0), HIT_NONE);
        assert_eq!(chrome.hit_at(50.0, -1.0), HIT_NONE);
        assert_eq!(chrome.hit_at(101.0, 50.0), HIT_NONE);
        assert_eq!(chrome.hit_at(50.0, 101.0), HIT_NONE);
    }

    #[test]
    fn set_hover_returns_true_on_change_only() {
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "");
        assert!(chrome.set_hover(chrome::HIT_CLOSE_BUTTON)); // changed
        assert!(!chrome.set_hover(chrome::HIT_CLOSE_BUTTON)); // same
        assert!(chrome.set_hover(HIT_NONE)); // changed back
    }

    #[test]
    fn set_hover_marks_hover_layer_dirty() {
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "");
        // Run a flatten cycle so StackCompositor::evaluate clears all initial-dirty flags.
        let mut target = alloc::vec![0u32; 100 * 100];
        chrome.flatten_into(&mut target, 100, 100);
        assert!(!chrome.group.rpn.layers[chrome.layer_hover].dirty);
        chrome.set_hover(chrome::HIT_CLOSE_BUTTON);
        assert!(chrome.group.rpn.layers[chrome.layer_hover].dirty);
    }

    #[test]
    fn resize_marks_layers_dirty_and_resizes_hit_map() {
        let mut chrome = DefaultChrome::new(Viewport::new(100, 100), "");
        let mut target = alloc::vec![0u32; 100 * 100];
        chrome.flatten_into(&mut target, 100, 100);
        chrome.resize(Viewport::new(200, 150));
        assert_eq!(chrome.dims(), (200, 150));
        assert_eq!(chrome.hit_test_map.len(), 200 * 150);
        // Group::resize marks all layers dirty.
        assert!(chrome.group.rpn.layers[chrome.layer_hover].dirty);
    }
}
