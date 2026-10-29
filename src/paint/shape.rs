//! AA shape rasterizers — axis-aligned rect, rotated rect, axis-aligned ellipse / circle, rotated ellipse — plus the per-row pixel-range solver and reciprocal-sqrt helper they share. Every entry point honours sub-pixel position + size, composes UNDER existing pixel content via the [`Blend::under`](crate::pixel::Blend::under) kernel, and reports its bbox to the canvas damage accumulator.
//!
//! Conventions: colour is an authored 8-bit α + darkness `g2t` pixel (build with [`crate::paint::pack_argb`]), widened to the canvas depth at entry by `Pixel::from_g2t8`; `Coord` is fluor's `f32` alias. AA coverage is the standard analytical approximation — `0.5 + signed_distance_inside` saturated to the depth's field by [`Pixel::cov`]'s cast, multiplied across edges for rotated rects, or solved analytically via implicit-form math for ellipses (sqrt-and-divide fused into a single hardware reciprocal-sqrt where available).

use super::Clip;
use crate::canvas::Canvas;
use crate::coord::Coord;
use crate::pixel::{BlendMode, Pixel};

/// α × the product of four edge coverages (each `0..=ONE`) ÷ ONE⁴: the rotated rect's corner coverage.
/// The same formula at both depths in an integer wide enough for it: at 8 bits α × 255⁴ fits a u64, at 16 bits 65535⁵ needs a u128, and the 8-bit build keeps its u64 (the branch is on a constant and folds away).
#[inline(always)]
fn alpha_of_four<P: Pixel>(alpha: u32, c: [u32; 4]) -> u32 {
    if P::DEPTH <= 8 {
        let one = P::ONE as u64;
        (alpha as u64 * (c[0] * c[1] * c[2] * c[3]) as u64 / (one * one * one * one)) as u32
    } else {
        let one = P::ONE as u128;
        (alpha as u128 * c[0] as u128 * c[1] as u128 * c[2] as u128 * c[3] as u128 / (one * one * one * one)) as u32
    }
}

/// Filled rectangle, anti-aliased, axis-aligned. Centered at `(cx, cy)` with fractional dimensions `(rect_w, rect_h)` — sub-pixel position + size both honoured. Colour is α + darkness packed (build with [`crate::paint::pack_argb`]); the rect blends UNDER any existing pixel content in the buffer via [`Blend::under`](crate::pixel::Blend::under).
///
/// AA: each pixel's coverage = `0.5 + distance_inside` against the nearest rect edge, saturated to the depth's field by [`Pixel::cov`]'s cast. Interior pixels saturate to coverage = 1.0 (full colour). Edge pixels get a fraction of the source α. Off-buffer pixels are clipped by the iteration bounds.
///
/// Rule 0: iteration bounds clamp `x_min/y_min ≥ 0` and `x_max/y_max ≤ buffer dim` because the rect can be partially or fully off-screen (caller passes arbitrary cx/cy); without the clamps an i32→usize cast on negative values wraps to a huge number → OOB panic. Coverage saturates at ONE (the cast in [`Pixel::cov`]) because pixels inside the rect have unbounded `d_inside`; without the cap, the `α × coverage` multiply would exceed the field.
pub fn draw_rect<P: Pixel>(
    canvas: &mut Canvas<P>,
    cx: Coord,
    cy: Coord,
    rect_w: Coord,
    rect_h: Coord,
    colour: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if rect_w < 0.0 || rect_h < 0.0 || width == 0 || height == 0 {
        return;
    }
    // A zero (or sub-pixel) dimension is a 1px line: floor the half-extent to 0.5 so the coverage band spans exactly one pixel centred on the axis. This is fluor's line primitive — `draw_rect(.., w, 0.0, ..)` is a horizontal hairline.
    let hw = (rect_w * 0.5).max(0.5);
    let hh = (rect_h * 0.5).max(0.5);
    let x_min = (cx - hw - 0.5) as i32;
    let x_max = (cx + hw + 1.5) as i32;
    let y_min = (cy - hh - 0.5) as i32;
    let y_max = (cy + hh + 1.5) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [P] = canvas.pixels;

    let colour = P::from_g2t8(colour);
    let colour_a = colour.alpha();
    let colour_dark = colour.rgb();

    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy_abs = ((py as Coord + 0.5) - cy).abs();
        let dy_inside = hh - dy_abs;
        for px in x_start..x_end {
            let dx_abs = ((px as Coord + 0.5) - cx).abs();
            let dx_inside = hw - dx_abs;
            let d_inside = dx_inside.min(dy_inside);
            let coverage = P::cov(0.5 + d_inside);
            if coverage == 0 {
                continue;
            }
            let new_alpha = colour_a * coverage / P::ONE;
            if new_alpha == 0 {
                continue;
            }
            let rect_pixel = colour_dark.with_alpha(new_alpha);
            row[px] = row[px].under(rect_pixel, BlendMode::Normal);
        }
    });
}

/// Blit a decoded image (`src`, α + darkness packed, row-major `src_w × src_h`) scaled into a dest rect centred at `(cx, cy)` with px size `(dst_w, dst_h)`. Nearest-neighbour sampling. Each sampled pixel composes UNDER existing content via [`Blend::under`](crate::pixel::Blend::under) exactly like the other primitives, so draw order controls z. Source pixels are treated as opaque — the VSF image decoder packs α = `0xFF`; a source pixel's own α still rides thru `under`, so a decoder that emits transparency (future masked avatars) composites correctly with no change here.
///
/// Bounds are clamped/clipped like [`draw_rect`] (the dest rect may be partly off-buffer). Edges are hard (no boundary AA) — fine behind a rounded mask or at icon scale; add coverage later if bare-edged blits need softening.
pub fn draw_image<P: Pixel>(
    canvas: &mut Canvas<P>,
    src: &[P],
    src_w: usize,
    src_h: usize,
    cx: Coord,
    cy: Coord,
    dst_w: Coord,
    dst_h: Coord,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if src_w == 0 || src_h == 0 || dst_w <= 0.0 || dst_h <= 0.0 || width == 0 || height == 0 {
        return;
    }
    // WHY/PROOF: src and its dimensions come from a decoded image (an avatar off the wire) — a short buffer would index past its end.
    if src.len() < src_w * src_h {
        return;
    }
    let hw = dst_w * 0.5;
    let hh = dst_h * 0.5;
    let left = cx - hw;
    let top = cy - hh;
    let x_min = crate::math::floor(left) as i32;
    let x_max = crate::math::ceil(cx + hw) as i32;
    let y_min = crate::math::floor(top) as i32;
    let y_max = crate::math::ceil(cy + hh) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [P] = canvas.pixels;
    // Source pixels per dest pixel along each axis (nearest-neighbour scale factor).
    let step_x = src_w as Coord / dst_w;
    let step_y = src_h as Coord / dst_h;
    let src_w_i = src_w as i32;
    let src_h_i = src_h as i32;
    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let sy_f = ((py as Coord + 0.5) - top) * step_y;
        if sy_f < 0.0 {
            return;
        }
        // sy_f ≥ 0 was just checked; the bbox rounds out to whole pixels, so the last row's centre can sample one past the source's bottom edge.
        let sy = (sy_f as i32).min(src_h_i - 1) as usize;
        let src_row_base = sy * src_w;
        for px in x_start..x_end {
            let sx_f = ((px as Coord + 0.5) - left) * step_x;
            if sx_f < 0.0 {
                continue;
            }
            // sx_f ≥ 0 was just checked; the rounded-out bbox's last column can sample one past the source's right edge.
            let sx = (sx_f as i32).min(src_w_i - 1) as usize;
            let sp = src[src_row_base + sx];
            row[px] = row[px].under(sp, BlendMode::Normal);
        }
    });
}

/// Filled rectangle, anti-aliased, rotated by `angle` radians around `(cx, cy)`. Positive angle rotates counter-clockwise (standard math convention). Other semantics match [`draw_rect`] — α + darkness colour, AA edges, UNDER-blend onto existing pixel content.
///
/// Scanline + per-pixel edge AA. Per row we analytically solve the px range where each pixel's centre lies in the rect's local-coord interior (full coverage) vs the wider "any coverage" band (AA needed). The interior loop is a tight UNDER-blend with a precomputed `(α<<24)|RGB` value — no per-pixel rotation, no coverage math, no abs/clamp. Only the two AA strips at the row endpoints (and full-AA rows near top/bottom corners) pay for the 4-edge product coverage. Bounding box is the tight rotated extent `(hw|cos| + hh|sin|, hw|sin| + hh|cos|)`.
///
/// Coverage at AA pixels: product of four [`Pixel::cov`]`(0.5 + signed_dist_inside)` coverages, one per edge.
pub fn draw_rect_rotated<P: Pixel>(
    canvas: &mut Canvas<P>,
    cx: Coord,
    cy: Coord,
    rect_w: Coord,
    rect_h: Coord,
    angle: Coord,
    colour: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if rect_w <= 0.0 || rect_h <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let hw = rect_w * 0.5;
    let hh = rect_h * 0.5;
    let (sin_a, cos_a) = crate::math::sin_cos(angle);
    let abs_cos = cos_a.abs();
    let abs_sin = sin_a.abs();
    let bbox_hw = hw * abs_cos + hh * abs_sin;
    let bbox_hh = hw * abs_sin + hh * abs_cos;
    let x_min = (cx - bbox_hw - 0.5) as i32;
    let x_max = (cx + bbox_hw + 1.5) as i32;
    let y_min = (cy - bbox_hh - 0.5) as i32;
    let y_max = (cy + bbox_hh + 1.5) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };

    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [P] = canvas.pixels;

    let colour = P::from_g2t8(colour);
    let colour_dark = colour.rgb();
    let full_alpha = colour.alpha();
    let full_top = colour;

    let dlx = cos_a;
    let dly = -sin_a;

    let lx_outer = hw + 0.5;
    let ly_outer = hh + 0.5;
    let lx_inner = hw - 0.5;
    let ly_inner = hh - 0.5;
    let has_inner = lx_inner >= 0.0 && ly_inner >= 0.0;

    let x_start_f = x_start as Coord + 0.5;
    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy_row = (py as Coord + 0.5) - cy;
        let dx0 = x_start_f - cx;
        let lx0 = dx0 * cos_a + dy_row * sin_a;
        let ly0 = -dx0 * sin_a + dy_row * cos_a;

        let (ox_lo, ox_hi) = px_range(lx0, dlx, -lx_outer, lx_outer, x_start, x_end);
        let (oy_lo, oy_hi) = px_range(ly0, dly, -ly_outer, ly_outer, x_start, x_end);
        let outer_lo = ox_lo.max(oy_lo);
        let outer_hi = ox_hi.min(oy_hi);
        if outer_lo >= outer_hi {
            return;
        }

        let (inner_lo, inner_hi) = if has_inner {
            let (ix_lo, ix_hi) = px_range(lx0, dlx, -lx_inner, lx_inner, x_start, x_end);
            let (iy_lo, iy_hi) = px_range(ly0, dly, -ly_inner, ly_inner, x_start, x_end);
            let lo = ix_lo.max(iy_lo).max(outer_lo);
            let hi = ix_hi.min(iy_hi).min(outer_hi);
            if lo >= hi {
                (outer_hi, outer_hi)
            } else {
                (lo, hi)
            }
        } else {
            (outer_hi, outer_hi)
        };

        let mut lx = lx0 + (outer_lo as Coord - x_start as Coord) * dlx;
        let mut ly = ly0 + (outer_lo as Coord - x_start as Coord) * dly;
        for px in outer_lo..inner_lo {
            let coverage = [P::cov(0.5 + (hw - lx)), P::cov(0.5 + (lx + hw)), P::cov(0.5 + (hh - ly)), P::cov(0.5 + (ly + hh))];
            let na = alpha_of_four::<P>(full_alpha, coverage);
            if na > 0 {
                let rect_pixel = colour_dark.with_alpha(na);
                row[px] = row[px].under(rect_pixel, BlendMode::Normal);
            }
            lx += dlx;
            ly += dly;
        }

        P::under_chunk_const(&mut row[inner_lo..inner_hi], full_top);

        let mut lx = lx0 + (inner_hi as Coord - x_start as Coord) * dlx;
        let mut ly = ly0 + (inner_hi as Coord - x_start as Coord) * dly;
        for px in inner_hi..outer_hi {
            let coverage = [P::cov(0.5 + (hw - lx)), P::cov(0.5 + (lx + hw)), P::cov(0.5 + (hh - ly)), P::cov(0.5 + (ly + hh))];
            let na = alpha_of_four::<P>(full_alpha, coverage);
            if na > 0 {
                let rect_pixel = colour_dark.with_alpha(na);
                row[px] = row[px].under(rect_pixel, BlendMode::Normal);
            }
            lx += dlx;
            ly += dly;
        }
    });
}

/// Pixel-range solver for `lo ≤ v0 + (px − x_start)·dv ≤ hi`. Returns half-open `[px_lo, px_hi)` clamped to `[x_start, x_end]`. Empty range is returned as `(x_end, x_end)`.
fn px_range(
    v0: Coord,
    dv: Coord,
    lo: Coord,
    hi: Coord,
    x_start: usize,
    x_end: usize,
) -> (usize, usize) {
    if dv.abs() < 1e-6 {
        if v0 >= lo && v0 <= hi {
            return (x_start, x_end);
        }
        return (x_end, x_end);
    }
    let i_a = (lo - v0) / dv;
    let i_b = (hi - v0) / dv;
    let i_lo = i_a.min(i_b);
    let i_hi = i_a.max(i_b);
    let i_min = crate::math::ceil(i_lo) as i64;
    let i_max = crate::math::floor(i_hi) as i64;
    let start_i = x_start as i64;
    let end_i = x_end as i64;
    // WHY/PROOF: the solved range runs wherever the rotated edge crosses the row, past the clip on either side — the clamps keep the AA strips inside the row's clipped span.
    let lo_px = (start_i + i_min).clamp(start_i, end_i);
    let hi_px = (start_i + i_max + 1).clamp(start_i, end_i);
    if lo_px >= hi_px {
        (x_end, x_end)
    } else {
        (lo_px as usize, hi_px as usize)
    }
}

/// Hardware fast reciprocal square root — `1/sqrt(x)` as ONE operation when the target supports it. LLVM won't lower `1.0/sqrtf(x)` to `rsqrtss` without `-ffast-math` (verified in asm: it emits `SQRTSS` + `DIVSS`, ~25 cycles), so we reach for the platform intrinsic explicitly. RSQRTSS gives ~12 bits of precision in ~5 cycles — far beyond what the 8-bit AA coverage byte needs. On non-x86_64 targets, fall back to the portable `1/sqrt` path; LLVM still picks the platform-best sqrt instruction underneath.
///
/// Precondition: `x > 0`. Inputs that hit `x == 0` should be screened by the caller (we never call this on the AA path for an ellipse center pixel because the cheap `4·f² ≥ |∇f|²` classifier promotes it to the interior fast path first).
#[inline(always)]
fn fast_inv_sqrt(x: Coord) -> Coord {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm_set_ss(x);
        let r = _mm_rsqrt_ss(v);
        _mm_cvtss_f32(r)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0 / libm::sqrtf(x)
    }
}

/// Filled circle, anti-aliased. Centered at `(cx, cy)` with fractional radius `r`. Colour is α + darkness packed; composes topmost-first via `pixels[idx].under(circle_pixel, Normal)`.
///
/// Stays in squared-distance space — no `sqrt` anywhere. Pre-squares the inner and outer AA thresholds (`(r−½)²`, `(r+½)²`); per pixel computes `dist² = dx² + dy²` and classifies into interior (full coverage), AA band, or outside.
pub fn draw_circle<P: Pixel>(
    canvas: &mut Canvas<P>,
    cx: Coord,
    cy: Coord,
    r: Coord,
    colour: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if r <= 0.0 || width == 0 || height == 0 {
        return;
    }
    // WHY/PROOF: r is the caller's radius, and under half a pixel r − ½ goes negative — squared, it would claim a full-coverage interior the circle doesn't have.
    let r_in = (r - 0.5).max(0.0);
    let r_out = r + 0.5;
    let r_in2 = r_in * r_in;
    let r_out2 = r_out * r_out;
    let inv_diff = 1.0 / (r_out2 - r_in2);
    let x_min = (cx - r_out) as i32;
    let x_max = (cx + r_out + 1.0) as i32;
    let y_min = (cy - r_out) as i32;
    let y_max = (cy + r_out + 1.0) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [P] = canvas.pixels;

    let colour = P::from_g2t8(colour);
    let colour_alpha = colour.alpha() as Coord;
    let colour_dark = colour.rgb();
    let full_pixel = colour;

    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy = (py as Coord + 0.5) - cy;
        let dy2 = dy * dy;
        for px in x_start..x_end {
            let dx = (px as Coord + 0.5) - cx;
            let dist2 = dx * dx + dy2;
            if dist2 <= r_in2 {
                row[px] = row[px].under(full_pixel, BlendMode::Normal);
            } else if dist2 < r_out2 {
                let t = (r_out2 - dist2) * inv_diff;
                let na = (colour_alpha * t) as u32;
                if na > 0 {
                    let circle_pixel = colour_dark.with_alpha(na);
                    row[px] = row[px].under(circle_pixel, BlendMode::Normal);
                }
            }
        }
    });
}

/// A RING split into `colours.len()` equal segments, anti-aliased: centre `(cx, cy)`, mid radius `r`, stroke `stroke` pixels, segment 0 starting at 12 o'clock and running clockwise.
/// Each segment takes its own colour (α + darkness packed; α = 0 draws nothing there); `gap` pixels of arc separate neighbours, none when there is only one segment, so one segment is a whole ring.
/// The same count at any size: one member or a thousand share one drawing.
pub fn draw_ring_segments<P: Pixel>(
    canvas: &mut Canvas<P>,
    cx: Coord,
    cy: Coord,
    r: Coord,
    stroke: Coord,
    gap: Coord,
    colours: &[u32],
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    let n = colours.len();
    if n == 0 || r <= 0.0 || stroke <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let half = stroke * 0.5;
    let r_out = r + half + 1.0;
    let x_min = (cx - r_out) as i32;
    let x_max = (cx + r_out + 1.0) as i32;
    let y_min = (cy - r_out) as i32;
    let y_max = (cy + r_out + 1.0) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [P] = canvas.pixels;
    let tau = core::f32::consts::TAU;
    let seg = tau / n as Coord;
    // The gap, as arc length at the mid radius, split evenly to both sides of every boundary.
    let half_gap = if n > 1 { gap * 0.5 } else { 0.0 };

    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy = (py as Coord + 0.5) - cy;
        for px in x_start..x_end {
            let dx = (px as Coord + 0.5) - cx;
            let dist = libm::sqrtf(dx * dx + dy * dy);
            // Radial coverage: a one-pixel ramp at each edge of the stroke.
            let radial = P::cov(half + 0.5 - (dist - r).abs());
            if radial == 0 {
                continue;
            }
            // Clockwise from 12 o'clock (screen y grows downward).
            let mut a = libm::atan2f(dx, -dy);
            if a < 0.0 {
                a += tau;
            }
            // WHY/PROOF: atan2 of a hair-negative angle plus τ rounds to exactly τ in f32, and a / seg then lands on n — colours[n] would panic.
            let i = ((a / seg) as usize).min(n - 1);
            let colour = P::from_g2t8(colours[i]);
            let colour_a = colour.alpha();
            if colour_a == 0 {
                continue;
            }
            // Angular coverage: the arc distance to the nearer boundary of this segment, less the half gap, as a one-pixel ramp.
            let angular = if n > 1 {
                let into = a - i as Coord * seg;
                let edge = into.min(seg - into) * r;
                P::cov(edge - half_gap + 0.5)
            } else {
                P::ONE
            };
            // Three factors in the depth's scale over ONE²: a u64 holds 65535³ (a u32 held 255³).
            let one = P::ONE as u64;
            let na = (colour_a as u64 * radial as u64 * angular as u64 / (one * one)) as u32;
            if na > 0 {
                row[px] = row[px].under(colour.rgb().with_alpha(na), BlendMode::Normal);
            }
        }
    });
}

/// Filled axis-aligned ellipse, anti-aliased. Centered at `(cx, cy)` with fractional radii `(rx, ry)`. Colour is α + darkness packed.
///
/// Per-pixel implicit form `f = (dx/rx)² + (dy/ry)² − 1` plus its gradient-magnitude-squared `|∇f|² = 4·((dx/rx²)² + (dy/ry²)²)`. AA via fused reciprocal-sqrt.
pub fn draw_ellipse<P: Pixel>(
    canvas: &mut Canvas<P>,
    cx: Coord,
    cy: Coord,
    rx: Coord,
    ry: Coord,
    colour: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if rx <= 0.0 || ry <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let inv_rx2 = 1.0 / (rx * rx);
    let inv_ry2 = 1.0 / (ry * ry);
    let x_min = (cx - rx - 0.5) as i32;
    let x_max = (cx + rx + 1.5) as i32;
    let y_min = (cy - ry - 0.5) as i32;
    let y_max = (cy + ry + 1.5) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [P] = canvas.pixels;

    let colour = P::from_g2t8(colour);
    let colour_alpha = colour.alpha() as Coord;
    let colour_dark = colour.rgb();
    let full_pixel = colour;

    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy = (py as Coord + 0.5) - cy;
        let dy2 = dy * dy;
        let dy2_b = dy2 * inv_ry2;
        for px in x_start..x_end {
            let dx = (px as Coord + 0.5) - cx;
            let dx2 = dx * dx;
            let dx2_a = dx2 * inv_rx2;
            let f = dx2_a + dy2_b - 1.0;
            let grad2 = 4.0 * (dx2_a * inv_rx2 + dy2_b * inv_ry2);
            let f_sq_4 = 4.0 * f * f;
            if f <= 0.0 && f_sq_4 >= grad2 {
                row[px] = row[px].under(full_pixel, BlendMode::Normal);
            } else if f < 0.0 || f_sq_4 < grad2 {
                let inv_g = fast_inv_sqrt(grad2);
                // In this branch |f · inv_g| < 0.5, so coverage already sits in [0, 1] (fast_inv_sqrt's ~12-bit error moves it by under 1/255, and the `as u32` cast floors any hair below 0).
                let coverage = 0.5 - f * inv_g;
                let na = (colour_alpha * coverage) as u32;
                if na > 0 {
                    let ellipse_pixel = colour_dark.with_alpha(na);
                    row[px] = row[px].under(ellipse_pixel, BlendMode::Normal);
                }
            }
        }
    });
}

/// Filled ellipse, anti-aliased, rotated by `angle` radians around `(cx, cy)`. Other semantics match [`draw_ellipse`].
pub fn draw_ellipse_rotated<P: Pixel>(
    canvas: &mut Canvas<P>,
    cx: Coord,
    cy: Coord,
    rx: Coord,
    ry: Coord,
    angle: Coord,
    colour: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if rx <= 0. || ry <= 0. || width == 0 || height == 0 {
        return;
    }
    let inv_rx2 = 1. / (rx * rx);
    let inv_ry2 = 1. / (ry * ry);
    let (sin_a, cos_a) = crate::math::sin_cos(angle);
    let abs_cos = cos_a.abs();
    let abs_sin = sin_a.abs();
    let bbox_hw = rx * abs_cos + ry * abs_sin;
    let bbox_hh = rx * abs_sin + ry * abs_cos;
    let x_min = (cx - bbox_hw - 0.5) as i32;
    let x_max = (cx + bbox_hw + 1.5) as i32;
    let y_min = (cy - bbox_hh - 0.5) as i32;
    let y_max = (cy + bbox_hh + 1.5) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [P] = canvas.pixels;

    let colour = P::from_g2t8(colour);
    let colour_alpha = colour.alpha() as Coord;
    let colour_dark = colour.rgb();
    let full_pixel = colour;

    let x_start_f = x_start as Coord + 0.5;
    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy_screen = (py as Coord + 0.5) - cy;
        let dx_screen0 = x_start_f - cx;
        let mut lx = dx_screen0 * cos_a + dy_screen * sin_a;
        let mut ly = -dx_screen0 * sin_a + dy_screen * cos_a;
        for px in x_start..x_end {
            let lx2_a = lx * lx * inv_rx2;
            let ly2_b = ly * ly * inv_ry2;
            let f = lx2_a + ly2_b - 1.;
            let grad2 = 4. * (lx2_a * inv_rx2 + ly2_b * inv_ry2);
            let f_sq_4 = 4. * f * f;
            if f <= 0. && f_sq_4 >= grad2 {
                row[px] = row[px].under(full_pixel, BlendMode::Normal);
            } else if f < 0. || f_sq_4 < grad2 {
                let inv_g = fast_inv_sqrt(grad2);
                // In this branch |f · inv_g| < 0.5, so coverage already sits in [0, 1] (see draw_ellipse).
                let coverage = 0.5 - f * inv_g;
                let na = (colour_alpha * coverage) as u32;
                if na > 0 {
                    let ellipse_pixel = colour_dark.with_alpha(na);
                    row[px] = row[px].under(ellipse_pixel, BlendMode::Normal);
                }
            }
            lx += cos_a;
            ly -= sin_a;
        }
    });
}
