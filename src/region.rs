//! Proportional layout primitive — unidirectional nesting, no solver.
//!
//! `Region` is a pixel-space rectangle that carries its own harmonic-mean span. Subdivide it proportionally via [`split_v`](Region::split_v) / [`split_h`](Region::split_h), nest arbitrarily deep — each level is one `split` call, O(N) total. Size flows strictly parent → child, never back. No tree nodes, no parent pointers, no constraint solver, no content-dependent sizing.
//!
//! Each Region's `span = 2wh/(w+h)` is local to that region's dimensions, so `region.size(16.0)` returns a value proportional to *that region's* shape — automatic scaling at every nesting level, like SVG viewBox coordinate mapping.
//!
//! The consumer's code IS the tree:
//! ```ignore
//! let root = Region::from_viewport(&vp);
//! let [_, content, _] = root.split_h([1.0, 6.0, 1.0]);
//! let [header, body, footer] = content.split_v([2.0, 12.0, 2.0]);
//! let font = header.size(16.0);  // scales with header, not viewport
//! ```
//! Resize = call the function again. No invalidation, no dirty flags.

use crate::coord::Coord;
use crate::geom::Viewport;
use crate::paint::Clip;

/// Pixel-space rectangle with region-local harmonic-mean span.
/// 20 bytes, `Copy`, no lifetimes, no allocator, `no_std`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    /// Left edge, pixel coordinates (top-left origin).
    pub x: Coord,
    /// Top edge, pixel coordinates.
    pub y: Coord,
    /// Width in pixels.
    pub w: Coord,
    /// Height in pixels.
    pub h: Coord,
    /// Region-local harmonic mean: `2*w*h / (w+h)`. Zero when either dimension is zero.
    pub span: Coord,
}

impl Region {
    /// Construct from explicit pixel bounds. Computes span internally.
    #[inline]
    pub fn new(x: Coord, y: Coord, w: Coord, h: Coord) -> Self {
        let sum = w + h;
        let span = if sum == 0.0 { 0.0 } else { 2.0 * w * h / sum };
        Self { x, y, w, h, span }
    }

    /// Root region spanning the full viewport.
    #[inline]
    pub fn from_viewport(vp: &Viewport) -> Self {
        Self::new(0.0, 0.0, vp.width_px as Coord, vp.height_px as Coord)
    }

    // --- Edges and center ---

    /// Right edge: `x + w`.
    #[inline]
    pub fn right(&self) -> Coord { self.x + self.w }

    /// Bottom edge: `y + h`.
    #[inline]
    pub fn bottom(&self) -> Coord { self.y + self.h }

    /// Center point in pixel coordinates.
    #[inline]
    pub fn center(&self) -> (Coord, Coord) { (self.x + self.w * 0.5, self.y + self.h * 0.5) }

    /// Center x in pixel coordinates.
    #[inline]
    pub fn center_x(&self) -> Coord { self.x + self.w * 0.5 }

    /// Center y in pixel coordinates.
    #[inline]
    pub fn center_y(&self) -> Coord { self.y + self.h * 0.5 }

    // --- Hit testing ---

    /// True if `(px, py)` is inside this region. Inclusive on left/top, exclusive on right/bottom.
    #[inline]
    pub fn contains(&self, px: Coord, py: Coord) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    // --- Sizing ---

    /// Derive a size from the region's span: `span / divisor`.
    /// Use for font sizes, margins, padding, border widths — anything that should
    /// scale with this region's dimensions.
    #[inline]
    pub fn size(&self, divisor: Coord) -> Coord { self.span / divisor }

    // --- Subdivision ---

    /// Split into `N` vertical bands (top-to-bottom rows) by proportional weights.
    /// Each returned Region spans the full width of `self` with height proportional to its weight.
    /// The last band absorbs rounding so sub-regions tile the parent exactly.
    pub fn split_v<const N: usize>(&self, weights: [Coord; N]) -> [Region; N] {
        let total: Coord = weights.iter().sum();
        let mut result = [*self; N];
        let mut cursor = self.y;
        for i in 0..N - 1 {
            let band_h = self.h * weights[i] / total;
            result[i] = Region::new(self.x, cursor, self.w, band_h);
            cursor += band_h;
        }
        // Last band: remainder to prevent accumulation drift.
        let last_h = self.bottom() - cursor;
        result[N - 1] = Region::new(self.x, cursor, self.w, last_h);
        result
    }

    /// Split into `N` horizontal bands (left-to-right columns) by proportional weights.
    /// Each returned Region spans the full height of `self` with width proportional to its weight.
    /// The last band absorbs rounding so sub-regions tile the parent exactly.
    pub fn split_h<const N: usize>(&self, weights: [Coord; N]) -> [Region; N] {
        let total: Coord = weights.iter().sum();
        let mut result = [*self; N];
        let mut cursor = self.x;
        for i in 0..N - 1 {
            let band_w = self.w * weights[i] / total;
            result[i] = Region::new(cursor, self.y, band_w, self.h);
            cursor += band_w;
        }
        let last_w = self.right() - cursor;
        result[N - 1] = Region::new(cursor, self.y, last_w, self.h);
        result
    }

    // --- Reshaping ---

    /// Shrink by `frac` of each dimension on each side.
    /// `inset(0.1)` removes 10% of width from left AND right (20% total width reduction),
    /// same for height. `inset(0.0)` returns self. `inset(0.5)` collapses to a point.
    #[inline]
    pub fn inset(&self, frac: Coord) -> Region {
        let dx = self.w * frac;
        let dy = self.h * frac;
        Region::new(self.x + dx, self.y + dy, self.w - dx - dx, self.h - dy - dy)
    }

    /// Shrink by independent fractions of each dimension on each side.
    #[inline]
    pub fn inset_xy(&self, frac_x: Coord, frac_y: Coord) -> Region {
        let dx = self.w * frac_x;
        let dy = self.h * frac_y;
        Region::new(self.x + dx, self.y + dy, self.w - dx - dx, self.h - dy - dy)
    }

    /// Centered sub-region using `frac` of this region's width, full height.
    #[inline]
    pub fn center_h(&self, frac: Coord) -> Region {
        let new_w = self.w * frac;
        let dx = (self.w - new_w) * 0.5;
        Region::new(self.x + dx, self.y, new_w, self.h)
    }

    /// Centered sub-region using `frac` of this region's height, full width.
    #[inline]
    pub fn center_v(&self, frac: Coord) -> Region {
        let new_h = self.h * frac;
        let dy = (self.h - new_h) * 0.5;
        Region::new(self.x, self.y + dy, self.w, new_h)
    }

    /// Largest centered square that fits inside this region.
    #[inline]
    pub fn square(&self) -> Region {
        let side = self.w.min(self.h);
        let dx = (self.w - side) * 0.5;
        let dy = (self.h - side) * 0.5;
        Region::new(self.x + dx, self.y + dy, side, side)
    }

    // --- Conversion ---

    /// Convert to a [`Clip`] for paint primitives. Truncates to `usize`.
    #[inline]
    pub fn to_clip(&self) -> Clip {
        Clip::new(
            self.x as usize,
            self.y as usize,
            self.right() as usize,
            self.bottom() as usize,
        )
    }
}

/// Harmonic mean of two values: `2ab / (a + b)`.
///
/// Use for blending two sizing constraints — e.g., span-based unit vs height-based unit
/// (photon's `ContactsUnifiedLayout` pattern). Returns 0.0 if both inputs are zero.
#[inline]
pub fn harmonic(a: Coord, b: Coord) -> Coord {
    let sum = a + b;
    if sum == 0.0 { 0.0 } else { 2.0 * a * b / sum }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::Viewport;

    const EPSILON: f32 = 1e-4;

    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < EPSILON }

    #[test]
    fn from_viewport_produces_correct_region() {
        let vp = Viewport::new(800, 600);
        let r = Region::from_viewport(&vp);
        assert_eq!(r.x, 0.0);
        assert_eq!(r.y, 0.0);
        assert_eq!(r.w, 800.0);
        assert_eq!(r.h, 600.0);
        // span = 2*800*600 / (800+600) = 960000/1400 ≈ 685.71
        assert!(approx(r.span, 2.0 * 800.0 * 600.0 / 1400.0));
    }

    #[test]
    fn span_zero_when_dimension_zero() {
        let r = Region::new(0.0, 0.0, 100.0, 0.0);
        assert_eq!(r.span, 0.0);
        let r2 = Region::new(0.0, 0.0, 0.0, 100.0);
        assert_eq!(r2.span, 0.0);
        let r3 = Region::new(0.0, 0.0, 0.0, 0.0);
        assert_eq!(r3.span, 0.0);
    }

    #[test]
    fn split_v_tiles_parent_exactly() {
        let parent = Region::new(10.0, 20.0, 300.0, 400.0);
        let bands = parent.split_v([1.0, 3.0, 1.0]);
        // First band starts at parent.y
        assert_eq!(bands[0].y, parent.y);
        // Each band starts where the previous one ends
        assert!(approx(bands[1].y, bands[0].bottom()));
        assert!(approx(bands[2].y, bands[1].bottom()));
        // Last band ends at parent.bottom()
        assert_eq!(bands[2].bottom(), parent.bottom());
        // All bands have parent's width and x
        for b in &bands {
            assert_eq!(b.x, parent.x);
            assert_eq!(b.w, parent.w);
        }
        // Heights are proportional: 1/5, 3/5, 1/5
        assert!(approx(bands[0].h, 80.0));
        assert!(approx(bands[1].h, 240.0));
        assert!(approx(bands[2].h, 80.0));
    }

    #[test]
    fn split_h_tiles_parent_exactly() {
        let parent = Region::new(10.0, 20.0, 300.0, 400.0);
        let bands = parent.split_h([1.0, 6.0, 1.0]);
        assert_eq!(bands[0].x, parent.x);
        assert!(approx(bands[1].x, bands[0].right()));
        assert!(approx(bands[2].x, bands[1].right()));
        assert_eq!(bands[2].right(), parent.right());
        for b in &bands {
            assert_eq!(b.y, parent.y);
            assert_eq!(b.h, parent.h);
        }
        // 1/8, 6/8, 1/8 of 300
        assert!(approx(bands[0].w, 37.5));
        assert!(approx(bands[1].w, 225.0));
        assert!(approx(bands[2].w, 37.5));
    }

    #[test]
    fn split_single_element_returns_parent() {
        let parent = Region::new(5.0, 10.0, 200.0, 100.0);
        let [only] = parent.split_v([1.0]);
        assert_eq!(only.x, parent.x);
        assert_eq!(only.y, parent.y);
        assert_eq!(only.w, parent.w);
        assert_eq!(only.h, parent.h);
    }

    #[test]
    fn split_equal_weights_produces_equal_regions() {
        let parent = Region::new(0.0, 0.0, 400.0, 300.0);
        let bands = parent.split_h([1.0, 1.0, 1.0, 1.0]);
        for b in &bands {
            assert!(approx(b.w, 100.0));
        }
    }

    #[test]
    fn nested_splits_stay_within_parent() {
        let root = Region::new(0.0, 0.0, 1920.0, 1080.0);
        let [_, content, _] = root.split_h([1.0, 6.0, 1.0]);
        let [header, body, footer] = content.split_v([2.0, 12.0, 2.0]);
        // All children within content bounds
        for r in &[header, body, footer] {
            assert!(r.x >= content.x - EPSILON);
            assert!(r.y >= content.y - EPSILON);
            assert!(r.right() <= content.right() + EPSILON);
            assert!(r.bottom() <= content.bottom() + EPSILON);
        }
        // Further nesting
        let [a, b] = header.split_h([1.0, 4.0]);
        assert!(a.x >= header.x - EPSILON);
        assert!(b.right() <= header.right() + EPSILON);
    }

    #[test]
    fn each_region_has_local_span() {
        let root = Region::new(0.0, 0.0, 1000.0, 500.0);
        let [narrow, wide] = root.split_h([1.0, 9.0]);
        // narrow is 100x500, wide is 900x500
        // Their spans should differ because their aspect ratios differ
        assert!(narrow.span != wide.span);
        // narrow.span = 2*100*500/(100+500) = 100000/600 ≈ 166.67
        assert!(approx(narrow.span, 2.0 * 100.0 * 500.0 / 600.0));
    }

    #[test]
    fn inset_zero_returns_self() {
        let r = Region::new(10.0, 20.0, 300.0, 200.0);
        let inset = r.inset(0.0);
        assert_eq!(inset.x, r.x);
        assert_eq!(inset.y, r.y);
        assert_eq!(inset.w, r.w);
        assert_eq!(inset.h, r.h);
    }

    #[test]
    fn inset_shrinks_symmetrically() {
        let r = Region::new(0.0, 0.0, 100.0, 200.0);
        let inset = r.inset(0.1);
        // 10% of 100 = 10 from each side → x=10, w=80
        // 10% of 200 = 20 from each side → y=20, h=160
        assert!(approx(inset.x, 10.0));
        assert!(approx(inset.y, 20.0));
        assert!(approx(inset.w, 80.0));
        assert!(approx(inset.h, 160.0));
    }

    #[test]
    fn center_h_75_percent() {
        let r = Region::new(0.0, 0.0, 400.0, 300.0);
        let centered = r.center_h(0.75);
        assert!(approx(centered.w, 300.0));
        assert!(approx(centered.x, 50.0)); // (400-300)/2
        assert_eq!(centered.y, r.y);
        assert_eq!(centered.h, r.h);
    }

    #[test]
    fn center_v_half() {
        let r = Region::new(100.0, 100.0, 400.0, 300.0);
        let centered = r.center_v(0.5);
        assert!(approx(centered.h, 150.0));
        assert!(approx(centered.y, 175.0)); // 100 + (300-150)/2
        assert_eq!(centered.x, r.x);
        assert_eq!(centered.w, r.w);
    }

    #[test]
    fn square_landscape() {
        let r = Region::new(0.0, 0.0, 400.0, 200.0);
        let sq = r.square();
        assert!(approx(sq.w, 200.0));
        assert!(approx(sq.h, 200.0));
        assert!(approx(sq.x, 100.0)); // centered horizontally
        assert!(approx(sq.y, 0.0));
    }

    #[test]
    fn square_portrait() {
        let r = Region::new(0.0, 0.0, 200.0, 400.0);
        let sq = r.square();
        assert!(approx(sq.w, 200.0));
        assert!(approx(sq.h, 200.0));
        assert!(approx(sq.x, 0.0));
        assert!(approx(sq.y, 100.0)); // centered vertically
    }

    #[test]
    fn size_is_span_over_divisor() {
        let r = Region::new(0.0, 0.0, 800.0, 600.0);
        let expected_span = 2.0 * 800.0 * 600.0 / 1400.0;
        assert!(approx(r.size(16.0), expected_span / 16.0));
        assert!(approx(r.size(32.0), expected_span / 32.0));
    }

    #[test]
    fn contains_edges() {
        let r = Region::new(10.0, 20.0, 100.0, 50.0);
        // Inclusive on left/top
        assert!(r.contains(10.0, 20.0));
        // Exclusive on right/bottom
        assert!(!r.contains(110.0, 20.0));
        assert!(!r.contains(10.0, 70.0));
        // Inside
        assert!(r.contains(50.0, 40.0));
        // Outside
        assert!(!r.contains(9.9, 20.0));
    }

    #[test]
    fn to_clip_truncates() {
        let r = Region::new(10.5, 20.7, 100.3, 50.9);
        let clip = r.to_clip();
        assert_eq!(clip.x_start, 10);
        assert_eq!(clip.y_start, 20);
        assert_eq!(clip.x_end, 110); // (10.5 + 100.3) as usize = 110
        assert_eq!(clip.y_end, 71);  // (20.7 + 50.9) as usize = 71
    }

    #[test]
    fn center_point() {
        let r = Region::new(10.0, 20.0, 100.0, 200.0);
        let (cx, cy) = r.center();
        assert!(approx(cx, 60.0));
        assert!(approx(cy, 120.0));
    }

    #[test]
    fn harmonic_basic() {
        // harmonic(100, 200) = 2*100*200 / 300 = 133.33...
        assert!(approx(super::harmonic(100.0, 200.0), 133.3333));
    }

    #[test]
    fn harmonic_equal_values() {
        // harmonic(x, x) = x
        assert!(approx(super::harmonic(50.0, 50.0), 50.0));
    }

    #[test]
    fn harmonic_zero_input() {
        assert_eq!(super::harmonic(0.0, 100.0), 0.0);
        assert_eq!(super::harmonic(100.0, 0.0), 0.0);
        assert_eq!(super::harmonic(0.0, 0.0), 0.0);
    }
}
