//! `DefaultChrome` — the reusable borderless-window frame consumers compose into their `FluorApp`.
//!
//! Owns a full-viewport [`Group`] containing three layers — bg (caller paints), chrome (controls + edges + hairlines + title text), hover (button delta) — plus the per-pixel `hit_test_map` byte buffer that records which button (if any) covers each pixel. Hover state, the cached hover-pixel list, and the title string all live here so the consumer can drop in chrome with one struct field.
//!
//! Built on the verbatim photon primitives in [`super::chrome`] — `draw_window_controls`, `draw_window_edges_and_mask`, `draw_button_hairlines`, `draw_button_hover_by_pixels`, `pixels_for_button`. Those stay; this module is a stateful wrapper that schedules them against the chrome group's dirty layers.
//!
//! Pattern: `chrome.rasterize_bg(|bg, w, h| { /* paint into bg */ });` → `chrome.rasterize_chrome(text);` → `chrome.rasterize_hover();` → `chrome.flatten_into(target, w, h);`. Each rasterize_* checks the layer's dirty bit internally and is a no-op on clean.

use alloc::string::String;
use alloc::vec::Vec;
use crate::coord::Coord;
use crate::geom::Viewport;
use crate::group::Group;
use crate::paint::BlendMode;
use crate::region::Region;
use crate::stack::Op;
use crate::text::TextRenderer;
use crate::theme;
use super::chrome::{self, HIT_NONE, MIN_BUTTON_HEIGHT_PX};

/// Reusable window frame: controls, edges, hairlines, title, hover overlay.
pub struct DefaultChrome {
    /// Full-viewport Group with 3 layers (bg, chrome, hover) composed via Stack Notation: `Push bg, Push chrome, AlphaOver, Push hover, Add`. Blend onto target = `Replace`.
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
    /// Allocate the chrome group + hit_test_map sized to `viewport`. All three layers start dirty so the first frame paints from scratch.
    pub fn new(viewport: Viewport, title: impl Into<String>) -> Self {
        let region = Region::new(0.0, 0.0, viewport.width_px as Coord, viewport.height_px as Coord);
        let mut group = Group::new(region, BlendMode::Replace);
        let layer_bg = group.new_layer();
        let layer_chrome = group.new_layer();
        let layer_hover = group.new_layer();
        group.set_program(alloc::vec![
            Op::Push(layer_bg),
            Op::Push(layer_chrome),
            Op::AlphaOver,
            Op::Push(layer_hover),
            Op::Add,
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
        let region = Region::new(0.0, 0.0, viewport.width_px as Coord, viewport.height_px as Coord);
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
        if !layer.dirty { return; }
        layer.pixels.fill(0);
        paint(&mut layer.pixels, w, h);
    }

    /// Paint controls + edges + hairlines + title text into the chrome layer if dirty. Clears + rewrites `hit_test_map` as a side effect (chrome buttons stamp their IDs there). Title is skipped silently when the computed title size falls below 6 px (tiny windows where text would be unreadable).
    pub fn rasterize_chrome(&mut self, text: &mut TextRenderer) {
        let (buf_w, buf_h) = self.dims();
        let vp_w = buf_w as u32;
        let vp_h = buf_h as u32;

        let layer = &mut self.group.rpn.layers[self.layer_chrome];
        if !layer.dirty { return; }

        // Reset hit_test_map for the new chrome rasterization.
        self.hit_test_map.fill(HIT_NONE);

        let chrome_buf = &mut layer.pixels;
        chrome_buf.fill(0);

        let (start, crossings, button_x_start, button_height) = chrome::draw_window_controls(
            chrome_buf, &mut self.hit_test_map, vp_w, vp_h, 1.0,
        );
        chrome::draw_window_edges_and_mask(
            chrome_buf, &mut self.hit_test_map, vp_w, vp_h, start, &crossings,
        );
        chrome::draw_button_hairlines(
            chrome_buf, &mut self.hit_test_map, vp_w, vp_h,
            button_x_start, button_height, start, &crossings,
        );

        // Title text.
        let span = 2.0 * vp_w as Coord * vp_h as Coord / (vp_w as Coord + vp_h as Coord);
        let bw = MIN_BUTTON_HEIGHT_PX as Coord + crate::math::ceil(span / 32.0);
        let title_size = bw * 0.55;
        if title_size >= 6.0 && !self.title.is_empty() {
            let pad = bw * 0.5;
            let baseline_y = bw * 0.5;
            let _ = text.draw_text_left_u32(
                chrome_buf, buf_w, buf_h, &self.title,
                pad, baseline_y, title_size, 400,
                theme::TEXT_COLOUR, "Open Sans", None, None, None,
            );
        }
    }

    /// Paint the hover-overlay delta if the hover layer is dirty. The delta is added (per-channel wrap) onto the chrome layer at the currently-hovered button's pixel positions; on hover_state == HIT_NONE the layer is just zeroed (Add of 0 = no-op).
    pub fn rasterize_hover(&mut self) {
        let layer = &mut self.group.rpn.layers[self.layer_hover];
        if !layer.dirty { return; }

        let buf = &mut layer.pixels;
        buf.fill(0);

        // Recompute pixel list for the current hover state.
        self.hover_pixel_list = chrome::pixels_for_button(&self.hit_test_map, self.hover_state);
        let hover_delta = match self.hover_state {
            chrome::HIT_CLOSE_BUTTON => theme::CLOSE_HOVER,
            chrome::HIT_MAXIMIZE_BUTTON => theme::MAXIMIZE_HOVER,
            chrome::HIT_MINIMIZE_BUTTON => theme::MINIMIZE_HOVER,
            _ => 0,
        };
        if hover_delta != 0 {
            for &idx in &self.hover_pixel_list {
                buf[idx] = hover_delta;
            }
        }
    }

    /// Flatten the chrome group onto `target` at the chrome's bbox (full viewport).
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
        if new_hit == self.hover_state { return false; }
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
        assert!(chrome.set_hover(chrome::HIT_CLOSE_BUTTON));     // changed
        assert!(!chrome.set_hover(chrome::HIT_CLOSE_BUTTON));    // same
        assert!(chrome.set_hover(HIT_NONE));                     // changed back
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
