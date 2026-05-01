//! Viewport geometry: pixel dimensions, derived universal scaling units, and conversions between RU coordinates and integer pixel coordinates.
//!
//! Universal scaling units (matching photon's AGENT.md "Universal Scaling Units"):
//! - `span = 2wh/(w+h)` — harmonic mean of pixel dimensions; the project's default scaling base. Slope 1 along `w==h`, smooth at the diagonal, biased toward the smaller dimension on narrow displays.
//! - `perimeter = w + h` — for edge-aware calculations.
//! - `diagonal_sq = w² + h²` — for distance calculations without sqrt.
//!
//! All three are `f32` so consumers can derive sizes via plain arithmetic (e.g. `let margin = vp.span / 64.0;`).
//!
//! Coordinate convention: origin is at viewport center, +x right, +y down. The y-down choice is deliberate — text engines, image scanline order, and pixel storage are all y-down, so +y down means zero flip points below the layout layer.

use crate::coord::{Coord, RuVec2};

/// Viewport state. Recomputed every time the host window resizes.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub width_px: u32,
    pub height_px: u32,
    /// Harmonic mean of width and height in pixel units.
    pub span: Coord,
    /// `width + height` in pixel units.
    pub perimeter: Coord,
    /// `width² + height²` in pixel units.
    pub diagonal_sq: Coord,
    /// RU multiplier: 1 RU corresponds to `span * ru` pixels. Default 1.0 — consumers scale this to match their UI density (e.g. set to 1/64 for em-like sizing).
    pub ru: Coord,
    half_w: Coord,
    half_h: Coord,
}

impl Viewport {
    /// Construct a viewport from integer pixel dimensions. `ru` defaults to 1.0; call [`with_ru`](Self::with_ru) to override.
    pub fn new(width_px: u32, height_px: u32) -> Self {
        let w = width_px as Coord;
        let h = height_px as Coord;
        let perimeter = w + h;
        let span = (2.0 * w * h) / perimeter;
        let diagonal_sq = w * w + h * h;
        Self {
            width_px,
            height_px,
            span,
            perimeter,
            diagonal_sq,
            ru: 1.0,
            half_w: w * 0.5,
            half_h: h * 0.5,
        }
    }

    /// Override the RU multiplier. Returns a new `Viewport` with all other derived units preserved.
    pub fn with_ru(mut self, ru: Coord) -> Self {
        self.ru = ru;
        self
    }

    /// Convert an RU x-coordinate (center-origin) to a pixel x-coordinate (top-left-origin).
    #[inline]
    pub fn ru_to_px_x(&self, x_ru: Coord) -> isize {
        (self.half_w + x_ru * self.span * self.ru) as isize
    }

    /// Convert an RU y-coordinate (center-origin, +y down) to a pixel y-coordinate (top-left-origin, +y down).
    #[inline]
    pub fn ru_to_px_y(&self, y_ru: Coord) -> isize {
        (self.half_h + y_ru * self.span * self.ru) as isize
    }

    /// Convert an RU width/height/distance (no center offset) to a pixel distance.
    #[inline]
    pub fn ru_to_px_d(&self, d_ru: Coord) -> isize {
        (d_ru * self.span * self.ru) as isize
    }

    /// Convert an `RuVec2` point to a `(px_x, px_y)` integer pixel coordinate pair.
    #[inline]
    pub fn ru_to_px(&self, p: RuVec2) -> (isize, isize) {
        (self.ru_to_px_x(p.x), self.ru_to_px_y(p.y))
    }

    /// Convert a pixel coordinate (top-left origin) to an `RuVec2` (center-origin). Inverse of [`ru_to_px`](Self::ru_to_px) up to integer rounding.
    #[inline]
    pub fn px_to_ru(&self, px: i32, py: i32) -> RuVec2 {
        let span_ru = self.span * self.ru;
        RuVec2 {
            x: (px as Coord - self.half_w) / span_ru,
            y: (py as Coord - self.half_h) / span_ru,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_is_harmonic_mean() {
        // 1920x1080: 2*1920*1080 / (1920+1080) = 4147200 / 3000 = 1382.4
        let vp = Viewport::new(1920, 1080);
        assert!((vp.span - 1382.4).abs() < 0.01, "span = {}, expected ~1382.4", vp.span);
    }

    #[test]
    fn center_origin_round_trip() {
        let vp = Viewport::new(800, 600);
        let (px, py) = vp.ru_to_px(RuVec2::ZERO);
        assert_eq!((px, py), (400, 300));
    }

    #[test]
    fn px_to_ru_inverse() {
        let vp = Viewport::new(1024, 768);
        let original = RuVec2::new(0.25, -0.125);
        let (px, py) = vp.ru_to_px(original);
        let recovered = vp.px_to_ru(px as i32, py as i32);
        let dx = (recovered.x - original.x).abs();
        let dy = (recovered.y - original.y).abs();
        let one_px_ru = (vp.span * vp.ru).recip();
        assert!(dx <= one_px_ru, "dx={} > 1px_ru={}", dx, one_px_ru);
        assert!(dy <= one_px_ru, "dy={} > 1px_ru={}", dy, one_px_ru);
    }

    #[test]
    fn perimeter_and_diagonal_sq() {
        let vp = Viewport::new(3, 4);
        assert!((vp.perimeter - 7.0).abs() < 1e-6);
        assert!((vp.diagonal_sq - 25.0).abs() < 1e-6);
    }
}
