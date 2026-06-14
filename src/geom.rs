//! Viewport geometry: pixel dimensions, derived universal scaling units, and conversions between RU coordinates and integer pixel coordinates.
//!
//! Relative Units, not pixels (ru)
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
    /// RU multiplier (zoom). 1 RU corresponds to `span * ru` pixels. Default 1.0 = 100%. Modified at runtime via [`adjust_zoom`](Self::adjust_zoom) / [`reset_zoom`](Self::reset_zoom) bound to `Ctrl/Cmd + plus/minus/0/scroll` in the host. Use [`effective_span`](Self::effective_span) instead of `span` directly anywhere zoom should apply (chrome button size, widget font size, etc.) — bare `span` is the pixel harmonic mean and ignores zoom.
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
        let span = (2. * w * h) / perimeter;
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

    /// Pixel size of 1 RU under the current zoom (`span * ru`). Use this everywhere "scale by viewport" math used to use bare `span` — chrome button size, glyph font size, widget hairlines, anything the user-facing zoom should affect. Bare `span` is the unzoomed pixel harmonic mean and only matters for OS-edge-adjacent things (e.g. WM resize-border hit zones).
    #[inline]
    pub fn effective_span(&self) -> Coord {
        self.span * self.ru
    }

    /// Adjust zoom by `steps` (positive = zoom in, negative = zoom out). Asymmetric photon-style log curve: each in-step multiplies `ru` by `32/31` (≈ +3.23%), each out-step by `32/33` (≈ −3.03%). The slight asymmetry means in/out aren't exact inverses — `in_then_out` drifts by `1024/1023 ≈ 1.001` per pair, which is visually imperceptible but matches photon's behaviour exactly so cross-codebase muscle memory transfers. **Unbounded by design** — clamp at the consumer layer if a particular widget needs guardrails; the host applies no min/max.
    pub fn adjust_zoom(&mut self, steps: f32) {
        let factor = if steps < 0.0 {
            (33.0_f32 / 32.0).powf(steps)
        } else {
            (31.0_f32 / 32.0).powf(-steps)
        };
        self.ru *= factor;
        // Production zoom clamp (requested + justified by Nick): below 1/8 (12.5%) or above 3× (300%) the layout/scratch math starts to break, so the shipping build pins ru to that range. WHY this clamp is allowed despite AGENT.md Rule 0: it's a user-requested, user-justified bound on EXTERNAL input (scroll/zoom gestures), not defensive masking of internal math — and it's release-gated so it never hides a bug in dev (debug builds stay unclamped so the breakage threshold can still be probed).
        #[cfg(not(debug_assertions))]
        {
            const ZOOM_MIN: f32 = 1.0 / (1 << 3) as f32; // 12.5%
            const ZOOM_MAX: f32 = 3.0; // 300% — chosen ceiling, deliberately not a power of two
            self.ru = self.ru.clamp(ZOOM_MIN, ZOOM_MAX);
        }
    }

    /// Reset zoom to 1.0 (bound to `Ctrl/Cmd + 0` in the host).
    pub fn reset_zoom(&mut self) {
        self.ru = 1.0;
    }
}
