//! Drawing surface that bundles a pixel buffer with automatic damage tracking. Every public rasterizer takes a `&mut Canvas` instead of a bare `&mut [u32] + width + height` triple; rasterizers report the pixel rectangle they touched into the canvas's [`Damage`] accumulator. Damage is unforgettable because it's part of what you draw on, not a side parameter — consumer widgets get differential-redraw support automatically.
//!
//! Pixel-space rectangles use [`PixelRect`] (half-open `[x0, x1) × [y0, y1)` in `usize`), matching what the rasterizers already compute internally via `Clip::intersect_bbox`. We do NOT reuse [`crate::region::Region`] here because Region carries a precomputed `span` (harmonic mean) which is meaningless overhead for damage rects and pollutes the equality/hash semantics.

use crate::pixel::{Argb8, Pixel};
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
        // WHY/PROOF: an inverted rect (x0 > x1) is a legal empty value by this type's contract and apps build their own damage rects — plain subtraction would panic on one in debug and report a huge width in release.
        self.x1.saturating_sub(self.x0)
    }

    #[inline]
    pub fn height(&self) -> usize {
        // WHY/PROOF: an inverted rect (y0 > y1) is a legal empty value by this type's contract — plain subtraction would underflow on one.
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

/// The most rects a frame's damage is named by ([`crate::host::FluorApp::damage_rects`]).
pub const DAMAGE_RECTS: usize = 4;

/// A frame's damage as up to [`DAMAGE_RECTS`] rects (2026-11-17, the Pro: a scroll's exposed band and a fixed overlay at the other end of the pane made one bounding box of 16 % of the frame for a 20-row move, and every pixel of it was cleared, drawn and converted). The host clears each and converts each; the app draws inside them alone ([`crate::host::Context::damage_rects`]). Disjoint rects, from the app: a pixel in two would be drawn twice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DamageRects {
    rects: [PixelRect; DAMAGE_RECTS],
    n: usize,
}

impl DamageRects {
    /// None at all.
    pub const EMPTY: Self = Self { rects: [PixelRect { x0: 0, y0: 0, x1: 0, y1: 0 }; DAMAGE_RECTS], n: 0 };

    /// One rect (the whole damage of an app that names one, or a host's whole frame).
    #[inline]
    pub fn one(r: PixelRect) -> Self {
        let mut d = Self::EMPTY;
        d.push(r);
        d
    }

    /// The first `n` of `rects` (an app's [`crate::host::FluorApp::damage_rects`]), the empty ones left out.
    pub fn from_slice(rects: &[PixelRect]) -> Self {
        let mut d = Self::EMPTY;
        for &r in rects {
            d.push(r);
        }
        d
    }

    /// `r` added unless it is empty. More than [`DAMAGE_RECTS`] is a bug in the caller (the app's count is held to the array it filled).
    #[inline]
    pub fn push(&mut self, r: PixelRect) {
        if !r.is_empty() {
            self.rects[self.n] = r;
            self.n += 1;
        }
    }

    /// The rects, none empty.
    #[inline]
    pub fn as_slice(&self) -> &[PixelRect] {
        &self.rects[..self.n]
    }

    /// The smallest rect holding them all (empty for none).
    pub fn bbox(&self) -> PixelRect {
        self.as_slice().iter().fold(PixelRect::empty(), |a, &r| a.union(r))
    }

    /// Their pixels (as disjoint rects).
    pub fn area(&self) -> usize {
        self.as_slice().iter().map(|r| r.width() * r.height()).sum()
    }

    /// Each cut to a `w × h` frame (an app's rects may overhang it; one cut away entirely is left out).
    pub fn cut(&self, w: usize, h: usize) -> Self {
        let mut d = Self::EMPTY;
        for r in self.as_slice() {
            // WHY/PROOF: the rects are the app's, which may overhang its frame (fluor's own primitives cut theirs at the buffer the same way, `Clip::resolve`); the hosts index the scratch and the surface up to their far edges.
            d.push(PixelRect::new(r.x0.min(w), r.y0.min(h), r.x1.min(w), r.y1.min(h)));
        }
        d
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
///
/// `P` is the pixel depth ([`Pixel`]).
/// It defaults to [`Argb8`], so `Canvas`, `&mut Canvas` and `Canvas<'a>` written anywhere still mean the 8-bit canvas, and `Canvas::new` takes its depth from the slice it is given.
pub struct Canvas<'a, P: Pixel = Argb8> {
    pub pixels: &'a mut [P],
    pub width: usize,
    pub height: usize,
    pub damage: &'a mut Damage,
}

impl<'a, P: Pixel> Canvas<'a, P> {
    /// Build a canvas from caller-owned pixel storage and damage accumulator.
    #[inline]
    pub fn new(pixels: &'a mut [P], width: usize, height: usize, damage: &'a mut Damage) -> Self {
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

    /// An app's rects: the empty ones left out, the box round the rest, each cut at the frame (one past it gone).
    #[test]
    fn damage_rects_keep_what_is_there() {
        let band = PixelRect::new(0, 2900, 1344, 2992);
        let orb = PixelRect::new(40, 140, 200, 300);
        let d = DamageRects::from_slice(&[band, PixelRect::empty(), orb]);
        assert_eq!(d.as_slice(), &[band, orb]);
        assert_eq!(d.bbox(), PixelRect::new(0, 140, 1344, 2992));
        assert_eq!(d.area(), 1344 * 92 + 160 * 160);
        let cut = DamageRects::from_slice(&[PixelRect::new(0, 2900, 2000, 3100), PixelRect::new(0, 3000, 10, 3010)]).cut(1344, 2992);
        assert_eq!(cut.as_slice(), &[band]);
        assert_eq!(DamageRects::EMPTY.bbox(), PixelRect::empty());
    }
}
