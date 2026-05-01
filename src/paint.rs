//! Pixel-buffer paint primitives. ARGB layout is `0xAARRGGBB` (alpha high byte, blue low). All inputs are pixel-space, not RU — convert via [`Viewport::ru_to_px`](crate::Viewport::ru_to_px) before calling.
//!
//! Internal to fluor's render pipeline. Per `## API / Implementation Separation` in AGENT.md, these are not part of the consumer-facing API: future SIMD kernels (NEON, SSE2) will dispatch through the same entry points without changing call sites in `pane` or `Compositor`.
//!
//! Blend model is straight (non-premultiplied) alpha lerp: `result = bg * (1 - α) + fg * α`. For an opaque target framebuffer (the common case — the host window's backbuffer) the alpha channel of the result is don't-care; for layered translucency a Porter-Duff over would be needed and is not provided here.

/// Pack four 8-bit channels into a single 32-bit ARGB value (`0xAARRGGBB`).
#[inline]
pub fn pack_argb(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Unpack a 32-bit ARGB value into `(r, g, b, a)`.
#[inline]
pub fn unpack_argb(packed: u32) -> (u8, u8, u8, u8) {
    let a = (packed >> 24) as u8;
    let r = (packed >> 16) as u8;
    let g = (packed >> 8) as u8;
    let b = packed as u8;
    (r, g, b, a)
}

/// Straight-alpha lerp of `fg` onto `bg`. SWAR pattern: widen each 32-bit ARGB pixel to 64 bits with each 8-bit channel in its own 16-bit slot, do four channel multiplies in parallel via u64 arithmetic, narrow back. The `>>8` divisor is 256 (not 255) — the canonical fast-blend approximation; per-channel error is below 1/256 and imperceptible.
#[inline]
pub fn blend(bg: u32, fg: u32) -> u32 {
    let alpha = ((fg >> 24) & 0xFF) as u64;
    let inv_alpha = 256 - alpha;

    let mut bg64 = bg as u64;
    bg64 = (bg64 | (bg64 << 16)) & 0x0000_FFFF_0000_FFFF;
    bg64 = (bg64 | (bg64 << 8)) & 0x00FF_00FF_00FF_00FF;

    let mut fg64 = fg as u64;
    fg64 = (fg64 | (fg64 << 16)) & 0x0000_FFFF_0000_FFFF;
    fg64 = (fg64 | (fg64 << 8)) & 0x00FF_00FF_00FF_00FF;

    let mut blended = bg64 * inv_alpha + fg64 * alpha;

    blended = (blended >> 8) & 0x00FF_00FF_00FF_00FF;
    blended = (blended | (blended >> 8)) & 0x0000_FFFF_0000_FFFF;
    blended = blended | (blended >> 16);
    blended as u32
}

/// Compute the intersection of a caller-supplied `(x, y, w, h)` rect (in pixels, top-left origin, may be negative or extend off-buffer) with the buffer `(0, 0, buf_w, buf_h)`. Returns `(x_min, y_min, x_max, y_max)` in `usize`, all guaranteed in-bounds for `pixels[y * buf_w + x]` indexing. Returns an empty range (x_min == x_max or y_min == y_max) if the rect lies entirely outside.
///
/// **Why this clamp is justified (Rule 0):** `x` / `y` / `rect_w` / `rect_h` are external inputs. Compositor semantics demand "draw the intersection with the buffer" — partial off-screen rects (e.g. a pane dragged past the window edge) are normal, not an error. Without clipping, a negative `x` cast to `usize` wraps to a huge value and panics or segfaults inner-loop indexing. The clip happens once per rect, not per pixel; inner loops trust the math from there.
#[inline]
fn clip_rect(buf_w: usize, buf_h: usize, x: isize, y: isize, rect_w: isize, rect_h: isize) -> (usize, usize, usize, usize) {
    let x_min = if x < 0 { 0 } else if (x as usize) > buf_w { buf_w } else { x as usize };
    let y_min = if y < 0 { 0 } else if (y as usize) > buf_h { buf_h } else { y as usize };
    let x_end = x.saturating_add(rect_w);
    let y_end = y.saturating_add(rect_h);
    let x_max = if x_end < 0 { 0 } else if (x_end as usize) > buf_w { buf_w } else { x_end as usize };
    let y_max = if y_end < 0 { 0 } else if (y_end as usize) > buf_h { buf_h } else { y_end as usize };
    (x_min, y_min, x_max, y_max)
}

/// Fill a rectangle with a solid (opaque-replace) ARGB color. The rect is clipped to the buffer; off-screen portions are silently dropped.
pub fn fill_rect_solid(pixels: &mut [u32], buf_w: usize, buf_h: usize, x: isize, y: isize, rect_w: isize, rect_h: isize, color: u32) {
    let (x_min, y_min, x_max, y_max) = clip_rect(buf_w, buf_h, x, y, rect_w, rect_h);
    for row in y_min..y_max {
        let base = row * buf_w;
        for col in x_min..x_max {
            pixels[base + col] = color;
        }
    }
}

/// Fill a rectangle by alpha-blending `color` over the existing buffer contents. The rect is clipped to the buffer.
pub fn fill_rect_blend(pixels: &mut [u32], buf_w: usize, buf_h: usize, x: isize, y: isize, rect_w: isize, rect_h: isize, color: u32) {
    let (x_min, y_min, x_max, y_max) = clip_rect(buf_w, buf_h, x, y, rect_w, rect_h);
    for row in y_min..y_max {
        let base = row * buf_w;
        for col in x_min..x_max {
            let idx = base + col;
            pixels[idx] = blend(pixels[idx], color);
        }
    }
}

/// Stroke (outline) an axis-aligned rectangle. Draws four filled rect strips along the edges; corners are not joined separately because at 90° angles the strips meet cleanly.
pub fn stroke_rect(pixels: &mut [u32], buf_w: usize, buf_h: usize, x: isize, y: isize, rect_w: isize, rect_h: isize, stroke: isize, color: u32) {
    if stroke <= 0 || rect_w <= 0 || rect_h <= 0 { return; }
    let solid = (color >> 24) == 0xFF;
    let inner_h = rect_h - 2 * stroke;
    let edges: [(isize, isize, isize, isize); 4] = [
        (x, y, rect_w, stroke),                                       // top
        (x, y + rect_h - stroke, rect_w, stroke),                     // bottom
        (x, y + stroke, stroke, inner_h),                             // left  (between top & bottom strips)
        (x + rect_w - stroke, y + stroke, stroke, inner_h),           // right
    ];
    for &(ex, ey, ew, eh) in &edges {
        if solid {
            fill_rect_solid(pixels, buf_w, buf_h, ex, ey, ew, eh, color);
        } else {
            fill_rect_blend(pixels, buf_w, buf_h, ex, ey, ew, eh, color);
        }
    }
}

/// Fill the buffer with photon's signature procedural background — symmetric organic noise plus speckle. Sequential (no rayon dep at this layer) but mirrored left/right halves like photon. Set `fullscreen=true` to fill the whole buffer; `false` leaves a 1px border for the window edge stroke. `speckle` is an animation counter (constant 0 for static); `scroll_offset` shifts the texture vertically (for content scroll integration).
pub fn background_noise(pixels: &mut [u32], buf_w: usize, buf_h: usize, speckle: usize, fullscreen: bool, scroll_offset: isize) {
    if buf_w < 2 || buf_h < 2 { return; }
    let (row_start, row_end, x_start, x_end) = if fullscreen {
        (0, buf_h, 0, buf_w)
    } else {
        (1, buf_h - 1, 1, buf_w - 1)
    };
    for row_idx in row_start..row_end {
        let logical_row = row_idx as isize - scroll_offset;
        let row_pixels = &mut pixels[row_idx * buf_w..(row_idx + 1) * buf_w];
        background_row(row_pixels, buf_w, logical_row, buf_h, x_start, x_end, speckle);
    }
}

#[inline]
fn background_row(row_pixels: &mut [u32], width: usize, logical_row: isize, height: usize, x_start: usize, x_end: usize, speckle: usize) {
    use crate::theme::{BG_ALPHA, BG_BASE, BG_MASK, BG_SPECKLE};
    let mut rng: usize = (0xDEAD_BEEF_0123_4567)
        ^ ((logical_row as usize).wrapping_sub(height / 2).wrapping_mul(0x9E37_79B9_4517_B397));
    let ones = 0x0001_0101u32;
    let mut colour = rng as u32 & BG_MASK | BG_ALPHA;

    // Right half — left to right.
    for x in (width / 2)..x_end {
        rng ^= rng.rotate_left(13).wrapping_add(12_345_678_942);
        let adder = rng as u32 & ones;
        if rng.wrapping_add(speckle) < usize::MAX / 256 {
            colour = (rng as u32 >> 8) & BG_SPECKLE | BG_ALPHA;
        } else {
            colour = colour.wrapping_add(adder) & BG_MASK;
            let subtractor = (rng >> 5) as u32 & ones;
            colour = colour.wrapping_sub(subtractor) & BG_MASK;
        }
        row_pixels[x] = colour.wrapping_add(BG_BASE) | BG_ALPHA;
    }

    // Left half — right to left, same RNG seed (mirror).
    rng = 0xDEAD_BEEF_0123_4567
        ^ ((logical_row as usize).wrapping_sub(height / 2).wrapping_mul(0x9E37_79B9_4517_B397));
    colour = rng as u32 & BG_MASK | BG_ALPHA;
    for x in (x_start..(width / 2)).rev() {
        rng ^= rng.rotate_left(13).wrapping_sub(12_345_678_942);
        let adder = rng as u32 & ones;
        if rng.wrapping_add(speckle) < usize::MAX / 256 {
            colour = (rng as u32 >> 8) & BG_SPECKLE | BG_ALPHA;
        } else {
            colour = colour.wrapping_add(adder) & BG_MASK;
            let subtractor = (rng >> 5) as u32 & ones;
            colour = colour.wrapping_sub(subtractor) & BG_MASK;
        }
        row_pixels[x] = colour.wrapping_add(BG_BASE) | BG_ALPHA;
    }
}

/// Photon's `PREMULTIPLIED` cfg flag: when true (Linux/Windows/macOS targets) the framebuffer expects premultiplied alpha; transparent edge pixels need their RGB scaled by alpha. False elsewhere (Android, etc.) — straight ARGB.
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
pub const PREMULTIPLIED: bool = true;
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
pub const PREMULTIPLIED: bool = false;

/// Photon's `scale_alpha` helper. Verbatim port from [compositing.rs:5809](/mnt/Octopus/Code/photon/src/ui/compositing.rs#L5809). Multiplies all four channels of `colour` by `alpha/256` using SWAR — premultiplies RGB so a fully transparent pixel reads as `0x00000000`.
pub fn scale_alpha(colour: u32, alpha: u8) -> u32 {
    let mut c = colour as u64;
    c = (c | (c << 16)) & 0x0000FFFF0000FFFF;
    c = (c | (c << 8)) & 0x00FF00FF00FF00FF;
    let mut scaled = c * alpha as u64;
    scaled = (scaled >> 8) & 0x00FF00FF00FF00FF;
    scaled = (scaled | (scaled >> 8)) & 0x0000FFFF0000FFFF;
    scaled = scaled | (scaled >> 16);
    scaled as u32
}

/// Photon's `blend_rgb_only` helper: weighted RGB blend of two colors with explicit per-pixel weights. Verbatim port from [compositing.rs:5821](/mnt/Octopus/Code/photon/src/ui/compositing.rs#L5821). Used by `draw_window_controls` for AA squircle edges.
pub fn blend_rgb_only(bg_colour: u32, fg_colour: u32, weight_bg: u8, weight_fg: u8) -> u32 {
    let mut bg = bg_colour as u64;
    bg = (bg | (bg << 16)) & 0x0000FFFF0000FFFF;
    bg = (bg | (bg << 8)) & 0x00FF00FF00FF00FF;

    let mut fg = fg_colour as u64;
    fg = (fg | (fg << 16)) & 0x0000FFFF0000FFFF;
    fg = (fg | (fg << 8)) & 0x00FF00FF00FF00FF;

    let mut blended = bg * weight_bg as u64 + fg * weight_fg as u64;
    blended = (blended >> 8) & 0x00FF00FF00FF00FF;
    blended = (blended | (blended >> 8)) & 0x0000FFFF0000FFFF;
    blended = blended | (blended >> 16) | 0xFF000000;

    blended as u32
}

/// Glyph rasterizers for window controls. Ported verbatim from photon's [compositing.rs](/mnt/Octopus/Code/photon/src/ui/compositing.rs) — the squircle minus / squircle ring / capsule X — so chrome looks identical to photon.
pub mod glyph {
    /// Draw a horizontal squircle "minus" stroke centered at `(x, y)` inside a button of pixel radius `r`. Uses a 4-power squircle with widened axis to make a flat horizontal pill.
    pub fn minimize_symbol(pixels: &mut [u32], width: usize, x: usize, y: usize, r: usize, stroke_colour: u32) {
        let r = r + 1;
        let r_render = r / 4 + 1;
        let r_2 = r_render * r_render;
        let r_4 = r_2 * r_2;
        let r_3 = r_render * r_render * r_render;

        let stroke_packed = stroke_colour | 0xFF00_0000;

        for h in -(r_render as isize)..=(r_render as isize) {
            for w in -(r as isize)..=(r as isize) {
                let h2 = h * h;
                let h4 = h2 * h2;
                let a = (w.abs() - (r * 3 / 4) as isize).max(0);
                let w2 = a * a;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 <= r_4 {
                    let px = (x as isize + w) as usize;
                    let py = (y as isize + h + (r / 2) as isize) as usize;
                    let idx = py * width + px;
                    let gradient = ((r_4 - dist_4) << 8) / (r_3 << 2);
                    if gradient > 255 {
                        pixels[idx] = stroke_packed;
                    } else {
                        pixels[idx] = blend_swar(pixels[idx], stroke_packed, gradient as u64);
                    }
                }
            }
        }
    }

    /// Draw a square "maximize" symbol — squircle ring with stroke + interior fill — centered at `(x, y)` with pixel radius `r`.
    pub fn maximize_symbol(pixels: &mut [u32], width: usize, x: usize, y: usize, r: usize, stroke_colour: u32, fill_colour: u32) {
        let r = r + 1;
        let mut r_4 = r * r;
        r_4 *= r_4;
        let r_3 = r * r * r;

        let r_inner = r * 4 / 5;
        let mut r_inner_4 = r_inner * r_inner;
        r_inner_4 *= r_inner_4;
        let r_inner_3 = r_inner * r_inner * r_inner;

        let outer_edge_threshold = r_3 << 2;
        let inner_edge_threshold = r_inner_3 << 2;

        let stroke_packed = stroke_colour | 0xFF00_0000;
        let fill_packed = fill_colour | 0xFF00_0000;

        for h in -(r as isize)..=r as isize {
            for w in -(r as isize)..=r as isize {
                let h2 = h * h;
                let h4 = h2 * h2;
                let w2 = w * w;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 > r_4 { continue; }
                let px = (x as isize + w) as usize;
                let py = (y as isize + h) as usize;
                let idx = py * width + px;

                let dist_from_outer = r_4 - dist_4;
                if dist_4 <= r_inner_4 {
                    let dist_from_inner = r_inner_4 - dist_4;
                    if dist_from_inner <= inner_edge_threshold {
                        let gradient = (dist_from_inner << 8) / inner_edge_threshold;
                        pixels[idx] = blend_swar(stroke_packed, fill_packed, gradient as u64);
                    } else {
                        pixels[idx] = fill_packed;
                    }
                } else if dist_from_outer <= outer_edge_threshold {
                    let gradient = (dist_from_outer << 8) / outer_edge_threshold;
                    pixels[idx] = blend_swar(pixels[idx], stroke_packed, gradient as u64);
                } else {
                    pixels[idx] = stroke_packed;
                }
            }
        }
    }

    /// Draw an antialiased "X" close symbol — two crossed capsule lines — centered at `(x, y)` with pixel radius `r`.
    pub fn close_symbol(pixels: &mut [u32], width: usize, x: usize, y: usize, r: usize, stroke_colour: u32) {
        let r = r + 1;
        let thickness = (r / 3).max(1) as f32;
        let radius = thickness / 2.0;
        let size = (r * 2) as f32;
        let cxf = x as f32;
        let cyf = y as f32;
        let end = size / 3.0;

        let x1_start = cxf - end; let y1_start = cyf - end;
        let x1_end   = cxf + end; let y1_end   = cyf + end;
        let x2_start = cxf + end; let y2_start = cyf - end;
        let x2_end   = cxf - end; let y2_end   = cyf + end;

        let stroke_packed = stroke_colour | 0xFF00_0000;
        let cxi = x as i32;
        let cyi = y as i32;
        let height = (pixels.len() / width) as i32;
        let min_x = ((x as i32) - (r as i32)).max(0);
        let max_x = ((x as i32) + (r as i32)).min(width as i32);
        let min_y = ((y as i32) - (r as i32)).max(0);
        let max_y = ((y as i32) + (r as i32)).min(height);

        // Each quadrant samples one of the two diagonals (whichever passes through it).
        let quadrants: [(i32, i32, i32, i32, f32, f32, f32, f32); 4] = [
            (min_x, cxi, min_y, cyi, x1_start, y1_start, x1_end, y1_end),  // top-left, diag1
            (cxi,   max_x, min_y, cyi, x2_start, y2_start, x2_end, y2_end),// top-right, diag2
            (min_x, cxi, cyi,   max_y, x2_start, y2_start, x2_end, y2_end),// bottom-left, diag2
            (cxi,   max_x, cyi, max_y, x1_start, y1_start, x1_end, y1_end),// bottom-right, diag1
        ];
        for (qx0, qx1, qy0, qy1, x0, y0, x1, y1) in quadrants {
            for py in qy0..qy1 {
                for px in qx0..qx1 {
                    let dist = distance_to_capsule(
                        px as f32 + 0.5, py as f32 + 0.5,
                        x0, y0, x1, y1, radius,
                    );
                    let alpha_f = if dist < -0.5 { 1.0 } else if dist < 0.5 { 0.5 - dist } else { 0.0 };
                    if alpha_f > 0.0 {
                        let idx = py as usize * width + px as usize;
                        let alpha = (alpha_f * 256.0) as u64;
                        pixels[idx] = blend_swar(pixels[idx], stroke_packed, alpha);
                    }
                }
            }
        }
    }

    /// Distance from a point to a capsule (line segment + radius). Negative inside the capsule, positive outside, used as an SDF for AA.
    #[inline]
    fn distance_to_capsule(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32, radius: f32) -> f32 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;
        let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
        let t_clamped = t.clamp(0.0, 1.0);
        let cx = x1 + t_clamped * dx;
        let cy = y1 + t_clamped * dy;
        let ex = px - cx;
        let ey = py - cy;
        (ex * ex + ey * ey).sqrt() - radius
    }

    /// Pre-multiplied SWAR blend of `fg` over `bg` with explicit `alpha` (0..=256). Photon's exact pattern: widen each 32-bit pixel to 64 bits with each channel in its own 16-bit slot, do `bg*(256-α) + fg*α` in parallel, narrow back.
    #[inline]
    fn blend_swar(bg: u32, fg: u32, alpha: u64) -> u32 {
        let inv = 256 - alpha;
        let mut bg64 = bg as u64;
        bg64 = (bg64 | (bg64 << 16)) & 0x0000_FFFF_0000_FFFF;
        bg64 = (bg64 | (bg64 << 8)) & 0x00FF_00FF_00FF_00FF;
        let mut fg64 = fg as u64;
        fg64 = (fg64 | (fg64 << 16)) & 0x0000_FFFF_0000_FFFF;
        fg64 = (fg64 | (fg64 << 8)) & 0x00FF_00FF_00FF_00FF;
        let mut blended = bg64 * inv + fg64 * alpha;
        blended = (blended >> 8) & 0x00FF_00FF_00FF_00FF;
        blended = (blended | (blended >> 8)) & 0x0000_FFFF_0000_FFFF;
        blended = blended | (blended >> 16);
        blended as u32
    }
}

/// Fill a circle with a 1-pixel-wide AA edge ring. Center `(cx, cy)` and `radius` are in pixels; `color` is straight-alpha ARGB (the AA coverage modulates the supplied alpha, so a translucent fill stays translucent at the edge).
///
/// AA via gradient-magnitude (no sqrt): for a pixel at squared distance `d²`, coverage is `(r_outer² - d²) / (r_outer² - r_inner²)` where `r_outer = radius` and `r_inner = radius - 1`. Gives a smooth 0→1 ramp across one pixel of edge.
pub fn circle_filled(pixels: &mut [u32], buf_w: usize, buf_h: usize, cx: isize, cy: isize, radius: isize, color: u32) {
    if radius <= 0 { return; }
    let r_outer = radius;
    let r_outer2 = r_outer * r_outer;
    let r_inner = radius - 1;
    let r_inner2 = r_inner * r_inner;
    let edge_range = r_outer2 - r_inner2;

    // Bounding box of the circle clipped to buffer. Inclusive on both ends → side length is 2r + 1.
    let (x_min, y_min, x_max, y_max) = clip_rect(
        buf_w, buf_h,
        cx - r_outer, cy - r_outer,
        2 * r_outer + 1, 2 * r_outer + 1,
    );

    let fg_alpha = (color >> 24) & 0xFF;
    let color_rgb = color & 0x00FF_FFFF;

    for py in y_min..y_max {
        let dy = py as isize - cy;
        let dy2 = dy * dy;
        let base = py * buf_w;
        for px in x_min..x_max {
            let dx = px as isize - cx;
            let dist2 = dx * dx + dy2;
            if dist2 > r_outer2 { continue; }
            // coverage in 0..=256: 256 = fully inside, 0 = at outer edge
            let coverage: u32 = if dist2 <= r_inner2 {
                256
            } else {
                (((r_outer2 - dist2) << 8) / edge_range) as u32
            };
            let scaled_alpha = (fg_alpha * coverage) >> 8;  // back to 0..=255
            let scaled_color = color_rgb | (scaled_alpha << 24);
            let idx = base + px;
            pixels[idx] = blend(pixels[idx], scaled_color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        let cases = [(0, 0, 0, 0), (255, 255, 255, 255), (12, 34, 56, 78), (200, 100, 50, 200)];
        for &(r, g, b, a) in &cases {
            let p = pack_argb(r, g, b, a);
            assert_eq!(unpack_argb(p), (r, g, b, a));
        }
    }

    #[test]
    fn pack_layout_is_argb() {
        assert_eq!(pack_argb(0xAB, 0xCD, 0xEF, 0x12), 0x12AB_CDEF);
    }

    #[test]
    fn blend_alpha_zero_preserves_bg() {
        let bg = pack_argb(100, 150, 200, 255);
        let fg = pack_argb(255, 0, 0, 0);
        let result = blend(bg, fg);
        let (r, g, b, _) = unpack_argb(result);
        assert_eq!((r, g, b), (100, 150, 200));
    }

    #[test]
    fn blend_alpha_full_replaces_rgb() {
        let bg = pack_argb(100, 150, 200, 255);
        let fg = pack_argb(50, 80, 110, 255);
        let result = blend(bg, fg);
        let (r, g, b, _) = unpack_argb(result);
        // alpha=255, inv_alpha=1: result.r = (100*1 + 50*255) / 256 = (100 + 12750)/256 ≈ 50
        // Off by at most 1 from fg per channel.
        assert!((r as i32 - 50).abs() <= 1, "r got {}", r);
        assert!((g as i32 - 80).abs() <= 1, "g got {}", g);
        assert!((b as i32 - 110).abs() <= 1, "b got {}", b);
    }

    #[test]
    fn stroke_rect_only_touches_edges() {
        let mut buf = vec![0u32; 10 * 10];
        stroke_rect(&mut buf, 10, 10, 2, 2, 6, 6, 1, pack_argb(255, 0, 0, 255));
        // Center pixel is interior — should not be touched.
        assert_eq!(buf[5 * 10 + 5], 0);
        // Top-left corner of stroke region should be set.
        let (r, _, _, _) = unpack_argb(buf[2 * 10 + 2]);
        assert!(r > 240, "top-left stroke pixel r={}", r);
        // Just-outside-stroke pixels remain 0.
        assert_eq!(buf[1 * 10 + 1], 0);
    }

    #[test]
    fn circle_filled_center_is_color() {
        let mut buf = vec![0u32; 16 * 16];
        circle_filled(&mut buf, 16, 16, 8, 8, 5, pack_argb(255, 0, 0, 255));
        // Center is fully inside → off by ≤1 from the supplied opaque red.
        let (r, g, b, _) = unpack_argb(buf[8 * 16 + 8]);
        assert!(r > 240 && g < 16 && b < 16, "center = ({}, {}, {})", r, g, b);
        // Corner of the bounding box (well outside the circle) should be untouched.
        assert_eq!(buf[0], 0);
    }

    #[test]
    fn circle_filled_clips_partial_offscreen() {
        let mut buf = vec![0u32; 8 * 8];
        // Circle center at (-2, -2), radius 4 — only the bottom-right quadrant lands in the buffer.
        circle_filled(&mut buf, 8, 8, -2, -2, 4, pack_argb(255, 255, 255, 255));
        // No panic = success. Some pixels in the top-left should be set.
        let (r, _, _, _) = unpack_argb(buf[0]);
        assert!(r > 200, "buf[0] r={}", r);
    }

    #[test]
    fn blend_alpha_half_is_midpoint() {
        let bg = pack_argb(0, 0, 0, 255);
        let fg = pack_argb(200, 200, 200, 128);
        let result = blend(bg, fg);
        let (r, g, b, _) = unpack_argb(result);
        // (0 * 128 + 200 * 128) / 256 = 100
        assert!((r as i32 - 100).abs() <= 1, "r got {}", r);
        assert!((g as i32 - 100).abs() <= 1, "g got {}", g);
        assert!((b as i32 - 100).abs() <= 1, "b got {}", b);
    }

    #[test]
    fn fill_rect_solid_full_buffer() {
        let mut buf = vec![0u32; 4 * 4];
        fill_rect_solid(&mut buf, 4, 4, 0, 0, 4, 4, 0xFF112233);
        assert!(buf.iter().all(|&p| p == 0xFF112233));
    }

    #[test]
    fn fill_rect_solid_partial() {
        let mut buf = vec![0u32; 4 * 4];
        fill_rect_solid(&mut buf, 4, 4, 1, 1, 2, 2, 0xFFAABBCC);
        let expected: [u32; 16] = [
            0, 0, 0, 0,
            0, 0xFFAABBCC, 0xFFAABBCC, 0,
            0, 0xFFAABBCC, 0xFFAABBCC, 0,
            0, 0, 0, 0,
        ];
        assert_eq!(buf.as_slice(), &expected);
    }

    #[test]
    fn fill_rect_solid_clips_negative_origin() {
        let mut buf = vec![0u32; 4 * 4];
        // Rect from (-2, -2) of size 4x4: only (0,0)..(2,2) intersects the buffer.
        fill_rect_solid(&mut buf, 4, 4, -2, -2, 4, 4, 0xFF000001);
        let expected: [u32; 16] = [
            0xFF000001, 0xFF000001, 0, 0,
            0xFF000001, 0xFF000001, 0, 0,
            0,          0,          0, 0,
            0,          0,          0, 0,
        ];
        assert_eq!(buf.as_slice(), &expected);
    }

    #[test]
    fn fill_rect_solid_fully_offscreen_is_noop() {
        let mut buf = vec![0u32; 4 * 4];
        fill_rect_solid(&mut buf, 4, 4, 100, 100, 5, 5, 0xFFFFFFFF);
        assert!(buf.iter().all(|&p| p == 0));
        fill_rect_solid(&mut buf, 4, 4, -10, -10, 5, 5, 0xFFFFFFFF);
        assert!(buf.iter().all(|&p| p == 0));
    }

    #[test]
    fn fill_rect_blend_alpha_zero_no_change() {
        let mut buf = vec![pack_argb(50, 60, 70, 255); 4 * 4];
        fill_rect_blend(&mut buf, 4, 4, 0, 0, 4, 4, pack_argb(255, 0, 0, 0));
        assert!(buf.iter().all(|&p| {
            let (r, g, b, _) = unpack_argb(p);
            (r, g, b) == (50, 60, 70)
        }));
    }

    #[test]
    fn fill_rect_blend_clips_partial() {
        let mut buf = vec![pack_argb(0, 0, 0, 255); 4 * 4];
        fill_rect_blend(&mut buf, 4, 4, 2, 2, 10, 10, pack_argb(200, 200, 200, 128));
        // Pixels at (2,2), (3,2), (2,3), (3,3) should be ~(100, 100, 100); rest unchanged.
        for y in 0..4usize {
            for x in 0..4usize {
                let (r, g, b, _) = unpack_argb(buf[y * 4 + x]);
                if x >= 2 && y >= 2 {
                    assert!((r as i32 - 100).abs() <= 1);
                    assert!((g as i32 - 100).abs() <= 1);
                    assert!((b as i32 - 100).abs() <= 1);
                } else {
                    assert_eq!((r, g, b), (0, 0, 0));
                }
            }
        }
    }
}
