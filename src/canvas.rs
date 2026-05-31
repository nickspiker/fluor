//! Drawing surface that bundles a pixel buffer with automatic damage tracking. Every public rasterizer takes a `&mut Canvas` instead of a bare `&mut [u32] + width + height` triple; rasterizers report the pixel rectangle they touched into the canvas's [`Damage`] accumulator. Damage is unforgettable because it's part of what you draw on, not a side parameter — consumer widgets get differential-redraw support automatically.
//!
//! Pixel-space rectangles use [`PixelRect`] (half-open `[x0, x1) × [y0, y1)` in `usize`), matching what the rasterizers already compute internally via `Clip::intersect_bbox`. We do NOT reuse [`crate::region::Region`] here because Region carries a precomputed `span` (harmonic mean) which is meaningless overhead for damage rects and pollutes the equality/hash semantics.

use core::cmp::{max, min};

/// Half-open pixel rectangle: covers pixels at `(x, y)` with `x0 ≤ x < x1`, `y0 ≤ y < y1`. Empty if `x0 >= x1` or `y0 >= y1`. Matches the integer ranges rasterizers produce after clipping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PixelRect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl PixelRect {
    #[inline]
    pub fn new(x0: usize, y0: usize, x1: usize, y1: usize) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// Empty rect; `is_empty` returns true for any such value.
    #[inline]
    pub fn empty() -> Self {
        Self::default()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x0 >= self.x1 || self.y0 >= self.y1
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.x1.saturating_sub(self.x0)
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.y1.saturating_sub(self.y0)
    }

    /// Bounding union with `other`. If either is empty the other is returned; otherwise the smallest rect containing both.
    #[inline]
    pub fn union(self, other: PixelRect) -> PixelRect {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        PixelRect {
            x0: min(self.x0, other.x0),
            y0: min(self.y0, other.y0),
            x1: max(self.x1, other.x1),
            y1: max(self.y1, other.y1),
        }
    }
}

/// Accumulator for the pixel area painted into a [`Canvas`] this frame. Currently a bounding rectangle (union of all reported rects); the host can read it after consumer render to drive damage-clipped composite and present. Will gain a multi-rect representation if profile shows wasted bandwidth on big disjoint damages.
#[derive(Clone, Copy, Debug, Default)]
pub struct Damage {
    bbox: PixelRect,
}

impl Damage {
    /// Empty damage — nothing painted yet this frame.
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Report a painted rectangle. Unions into the running bbox.
    #[inline]
    pub fn add(&mut self, rect: PixelRect) {
        self.bbox = self.bbox.union(rect);
    }

    /// Convenience for the rasterizer's internal `(x_start, y_start, x_end, y_end)` shape.
    #[inline]
    pub fn add_bounds(&mut self, x_start: usize, y_start: usize, x_end: usize, y_end: usize) {
        self.add(PixelRect::new(x_start, y_start, x_end, y_end));
    }

    /// Current bounding rect of everything painted so far. Empty if no rasterizer reported damage.
    #[inline]
    pub fn bbox(&self) -> PixelRect {
        self.bbox
    }

    /// True when nothing has been painted (no rasterizer reported damage).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bbox.is_empty()
    }

    /// Reset to "nothing painted" — host calls this at the start of each frame.
    #[inline]
    pub fn clear(&mut self) {
        self.bbox = PixelRect::empty();
    }
}

/// Bundle of "what you're drawing on": a pixel buffer (with its dimensions) plus the [`Damage`] accumulator for this frame. Every public rasterizer takes `&mut Canvas`; the pixel buffer and damage are borrowed together so reporting damage cannot be skipped at a call site.
///
/// The lifetimes are tied together — a Canvas borrows the pixel slice and damage from the host (or from a test scratch). It does not own them, which keeps it cheap to construct per-frame.
pub struct Canvas<'a> {
    pub pixels: &'a mut [u32],
    pub width: usize,
    pub height: usize,
    pub damage: &'a mut Damage,
}

impl<'a> Canvas<'a> {
    /// Build a canvas from caller-owned pixel storage and damage accumulator.
    #[inline]
    pub fn new(pixels: &'a mut [u32], width: usize, height: usize, damage: &'a mut Damage) -> Self {
        Self {
            pixels,
            width,
            height,
            damage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixel_rect_union_with_empty_is_other() {
        let a = PixelRect::new(10, 20, 30, 40);
        let e = PixelRect::empty();
        assert_eq!(a.union(e), a);
        assert_eq!(e.union(a), a);
    }

    #[test]
    fn pixel_rect_union_two_non_overlapping() {
        let a = PixelRect::new(0, 0, 10, 10);
        let b = PixelRect::new(50, 60, 70, 80);
        assert_eq!(a.union(b), PixelRect::new(0, 0, 70, 80));
    }

    #[test]
    fn damage_starts_empty_and_unions_on_add() {
        let mut d = Damage::new();
        assert!(d.is_empty());
        d.add_bounds(5, 5, 15, 15);
        assert_eq!(d.bbox(), PixelRect::new(5, 5, 15, 15));
        d.add_bounds(100, 100, 110, 110);
        assert_eq!(d.bbox(), PixelRect::new(5, 5, 110, 110));
    }

    #[test]
    fn damage_clear_resets_to_empty() {
        let mut d = Damage::new();
        d.add_bounds(1, 2, 3, 4);
        assert!(!d.is_empty());
        d.clear();
        assert!(d.is_empty());
    }
}
