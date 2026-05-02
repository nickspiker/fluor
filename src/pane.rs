//! Panes — the unit of UI in fluor — and the [`Compositor`] that owns them.
//!
//! A `Pane` is a center-origin rectangle in RU space with a stable `PaneId` handle, a z-order index, and an ARGB background. Consumers (rhe, photon, basecalc) attach content to a pane via a future `PaneContent` trait; v0 just paints the background rectangle.
//!
//! `Compositor` owns the pane tree, the active `Viewport`, and the focus state. Per `## API / Implementation Separation` in AGENT.md, the renderer underneath is an implementation detail — `Compositor::render` currently calls into [`crate::paint`] directly, but a future enum-dispatched backend (CPU SIMD / GPU / Spirix-AA) plugs in here without changing the API.

use alloc::vec::Vec;
use crate::coord::RuVec2;
use crate::geom::Viewport;
use crate::paint;

/// Stable handle to a pane. Returned from [`Compositor::insert`]; remains valid until the pane is removed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PaneId(u32);

/// A center-origin rectangular pane in RU space. `extent` is the half-width / half-height — a pane with `extent = (0.25, 0.25)` spans `[center - 0.25, center + 0.25]` on each axis.
#[derive(Clone, Copy, Debug)]
pub struct Pane {
    id: PaneId,
    z: u8,
    pub center: RuVec2,
    pub extent: RuVec2,
    pub background: u32,
}

impl Pane {
    pub fn id(&self) -> PaneId { self.id }
    pub fn z(&self) -> u8 { self.z }

    /// True if `p` lies inside the pane (inclusive on the edges).
    pub fn contains(&self, p: RuVec2) -> bool {
        let dx = (p.x - self.center.x).abs();
        let dy = (p.y - self.center.y).abs();
        dx <= self.extent.x && dy <= self.extent.y
    }

    pub fn min_corner(&self) -> RuVec2 { self.center - self.extent }
    pub fn max_corner(&self) -> RuVec2 { self.center + self.extent }
}

/// The compositor — owner of the pane tree and the viewport. Public API surface that consumers program against.
pub struct Compositor {
    /// Panes in z-order, lowest at index 0, highest at end. `bring_to_front` moves a pane to the end; render iterates this slice; `hit_test` iterates it in reverse.
    panes: Vec<Pane>,
    next_id: u32,
    focused: Option<PaneId>,
    viewport: Viewport,
}

impl Compositor {
    pub fn new(viewport: Viewport) -> Self {
        Self { panes: Vec::new(), next_id: 0, focused: None, viewport }
    }

    pub fn viewport(&self) -> Viewport { self.viewport }

    /// Resize the viewport. Recomputes `span` / `perimeter` / `diagonal_sq` from the new pixel dimensions; pane RU coordinates are unchanged so layout scales naturally.
    pub fn resize(&mut self, width_px: u32, height_px: u32) {
        let ru = self.viewport.ru;
        self.viewport = Viewport::new(width_px, height_px).with_ru(ru);
    }

    /// Insert a new pane on top of the stack. Returns its handle.
    pub fn insert(&mut self, center: RuVec2, extent: RuVec2, background: u32) -> PaneId {
        let id = PaneId(self.next_id);
        self.next_id += 1;
        let z = self.panes.len() as u8;
        self.panes.push(Pane { id, z, center, extent, background });
        id
    }

    pub fn remove(&mut self, id: PaneId) {
        if let Some(pos) = self.panes.iter().position(|p| p.id == id) {
            self.panes.remove(pos);
            self.renumber_z();
            if self.focused == Some(id) { self.focused = None; }
        }
    }

    pub fn get(&self, id: PaneId) -> Option<&Pane> {
        self.panes.iter().find(|p| p.id == id)
    }

    pub fn get_mut(&mut self, id: PaneId) -> Option<&mut Pane> {
        self.panes.iter_mut().find(|p| p.id == id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Pane> { self.panes.iter() }
    pub fn len(&self) -> usize { self.panes.len() }
    pub fn is_empty(&self) -> bool { self.panes.is_empty() }

    /// Find the topmost pane containing `p` (in RU). Returns `None` if no pane covers it.
    pub fn hit_test(&self, p: RuVec2) -> Option<PaneId> {
        self.panes.iter().rev().find(|pane| pane.contains(p)).map(|pane| pane.id)
    }

    pub fn focus(&mut self, id: PaneId) {
        if self.panes.iter().any(|p| p.id == id) {
            self.focused = Some(id);
        }
    }

    pub fn focused(&self) -> Option<PaneId> { self.focused }

    /// Move `id` to the top of the z-stack. No-op if `id` is unknown.
    pub fn bring_to_front(&mut self, id: PaneId) {
        if let Some(pos) = self.panes.iter().position(|p| p.id == id) {
            let pane = self.panes.remove(pos);
            self.panes.push(pane);
            self.renumber_z();
        }
    }

    /// Move `id` to the bottom of the z-stack. No-op if `id` is unknown.
    pub fn send_to_back(&mut self, id: PaneId) {
        if let Some(pos) = self.panes.iter().position(|p| p.id == id) {
            let pane = self.panes.remove(pos);
            self.panes.insert(0, pane);
            self.renumber_z();
        }
    }

    /// Render every pane, bottom-up, into the target ARGB buffer. Future versions will route through a renderer enum and call into the squircle rasterizer for rounded corners.
    ///
    /// Fully-opaque panes (alpha == 255) take the exact `fill_rect_solid` path; translucent panes go through `fill_rect_blend`. Reason: the SWAR blend divides by 256 (not 255) and produces a 1/256 channel error per blend — invisible by itself, but compounded over many opaque panes it would dim the buffer by a measurable amount over time. Solid is also faster.
    pub fn render(&self, target: &mut [u32], buf_w: usize, buf_h: usize) {
        for pane in &self.panes {
            let (cx, cy) = self.viewport.ru_to_px(pane.center);
            let ex = self.viewport.ru_to_px_d(pane.extent.x);
            let ey = self.viewport.ru_to_px_d(pane.extent.y);
            let x = cx - ex;
            let y = cy - ey;
            let w = ex + ex;
            let h = ey + ey;
            if (pane.background >> 24) == 0xFF {
                paint::fill_rect_solid(target, buf_w, buf_h, x, y, w, h, pane.background, None);
            } else {
                paint::fill_rect_blend(target, buf_w, buf_h, x, y, w, h, pane.background, None, None);
            }
        }
    }

    fn renumber_z(&mut self) {
        for (i, pane) in self.panes.iter_mut().enumerate() {
            pane.z = i as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::{pack_argb, unpack_argb};

    fn make_compositor(w: u32, h: u32) -> Compositor {
        Compositor::new(Viewport::new(w, h))
    }

    #[test]
    fn insert_assigns_unique_ids() {
        let mut c = make_compositor(800, 600);
        let a = c.insert(RuVec2::ZERO, RuVec2::splat(0.1), 0xFF000000);
        let b = c.insert(RuVec2::ZERO, RuVec2::splat(0.1), 0xFF000000);
        assert_ne!(a, b);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn contains_inclusive_on_edges() {
        let mut c = make_compositor(800, 600);
        let id = c.insert(RuVec2::new(0.0, 0.0), RuVec2::new(0.2, 0.1), 0xFF000000);
        let pane = c.get(id).unwrap();
        assert!(pane.contains(RuVec2::new(0.0, 0.0)));
        assert!(pane.contains(RuVec2::new(0.2, 0.1)));
        assert!(pane.contains(RuVec2::new(-0.2, -0.1)));
        assert!(!pane.contains(RuVec2::new(0.21, 0.0)));
        assert!(!pane.contains(RuVec2::new(0.0, 0.11)));
    }

    #[test]
    fn hit_test_picks_topmost() {
        let mut c = make_compositor(800, 600);
        let bottom = c.insert(RuVec2::new(0.0, 0.0), RuVec2::splat(0.5), 0xFF000000);
        let top = c.insert(RuVec2::new(0.0, 0.0), RuVec2::splat(0.5), 0xFF000000);
        assert_eq!(c.hit_test(RuVec2::ZERO), Some(top));
        // Outside the smaller pane but inside the larger? Both same size — make a smaller top pane.
        c.remove(top);
        let small_top = c.insert(RuVec2::new(0.0, 0.0), RuVec2::splat(0.1), 0xFF000000);
        assert_eq!(c.hit_test(RuVec2::ZERO), Some(small_top));
        assert_eq!(c.hit_test(RuVec2::new(0.3, 0.0)), Some(bottom));
    }

    #[test]
    fn hit_test_returns_none_for_empty_space() {
        let mut c = make_compositor(800, 600);
        c.insert(RuVec2::new(0.4, 0.0), RuVec2::splat(0.05), 0xFF000000);
        assert_eq!(c.hit_test(RuVec2::ZERO), None);
    }

    #[test]
    fn bring_to_front_moves_pane_to_top() {
        let mut c = make_compositor(800, 600);
        let a = c.insert(RuVec2::ZERO, RuVec2::splat(0.5), 0xFF000000);
        let b = c.insert(RuVec2::ZERO, RuVec2::splat(0.5), 0xFF000000);
        let cc = c.insert(RuVec2::ZERO, RuVec2::splat(0.5), 0xFF000000);
        assert_eq!(c.hit_test(RuVec2::ZERO), Some(cc));
        c.bring_to_front(a);
        assert_eq!(c.hit_test(RuVec2::ZERO), Some(a));
        assert_eq!(c.get(a).unwrap().z(), 2);
        assert_eq!(c.get(b).unwrap().z(), 0);
        assert_eq!(c.get(cc).unwrap().z(), 1);
    }

    #[test]
    fn send_to_back_moves_pane_to_bottom() {
        let mut c = make_compositor(800, 600);
        let a = c.insert(RuVec2::ZERO, RuVec2::splat(0.5), 0xFF000000);
        let b = c.insert(RuVec2::ZERO, RuVec2::splat(0.5), 0xFF000000);
        c.send_to_back(b);
        assert_eq!(c.get(b).unwrap().z(), 0);
        assert_eq!(c.get(a).unwrap().z(), 1);
        assert_eq!(c.hit_test(RuVec2::ZERO), Some(a));
    }

    #[test]
    fn focus_clears_on_remove() {
        let mut c = make_compositor(800, 600);
        let id = c.insert(RuVec2::ZERO, RuVec2::splat(0.5), 0xFF000000);
        c.focus(id);
        assert_eq!(c.focused(), Some(id));
        c.remove(id);
        assert_eq!(c.focused(), None);
    }

    #[test]
    fn render_fills_pane_pixels() {
        let mut c = make_compositor(8, 8);
        // Pane at center, half-extent 0.05 RU. With span = harmonic mean of 8x8 = 8, ru = 1.0,
        // half-extent in pixels = 0.05 * 8 * 1 = 0.4 px → 0 after isize cast. Use larger extent.
        c.insert(RuVec2::new(0.0, 0.0), RuVec2::splat(0.25), pack_argb(255, 0, 0, 255));
        let mut buf = vec![0u32; 8 * 8];
        c.render(&mut buf, 8, 8);
        // Center pixel at (4, 4) should be opaque red.
        let center = buf[4 * 8 + 4];
        let (r, g, b, _) = unpack_argb(center);
        assert_eq!((r, g, b), (255, 0, 0), "center pixel = {:#010x}", center);
    }

    #[test]
    fn resize_preserves_pane_ru_layout() {
        let mut c = make_compositor(800, 600);
        let id = c.insert(RuVec2::new(0.1, -0.2), RuVec2::splat(0.15), 0xFF112233);
        let center_before = c.get(id).unwrap().center;
        c.resize(1920, 1080);
        let center_after = c.get(id).unwrap().center;
        assert_eq!(center_before, center_after);
        assert_eq!(c.viewport().width_px, 1920);
    }
}
