//! Focus glow rasterizers. Four directional ray passes (right / left / top / bottom) that scan the canvas's α channel to find the silhouette of the pill to glow around, then emit decaying rays of partial-α `glow_colour` pixels outward. Plus a 4-way blur variant used by photon's textbox glow shape, and the shared decay-table helper.
//!
//! The `factor_256` curve: `alpha[k] = (alpha[k-1] × factor_256) >> 8`. Geometric decay starting from `seed` (typically `0x80` for horizontal, `0x40` for vertical). [`ray_reach_px`] reports how many steps until α decays to zero — callers use it to pre-size damage rects so the host clears + finalizes exactly the area the rasterizer will paint.

use super::HitId;
use crate::canvas::Canvas;
use crate::paint::Clip;
use crate::pixel::{Blend, BlendMode};

// Suppress the unused-import warning when this file gets re-included via `pub use glow::*;` — HitId isn't referenced here, but the import keeps the symbol available for future glow variants that might want to stamp a hit map.
#[allow(dead_code)]
fn _unused_hit_id_ref(_: HitId) {}

/// Precompute the per-step α decay table for a ray. `alphas[0] = seed`; `alphas[k] = (alphas[k-1] * factor_256) >> 8`. Returns the populated prefix length (where `alphas[len-1] > 0`). Replaces the per-step mul on the hot path with an array lookup. 1024 entries cover the worst case (seed 0x40, factor 254 → ~528 non-zero steps).
#[inline]
fn compute_alphas(seed: u32, factor_256: u32) -> ([u32; 1024], usize) {
    let mut alphas = [0u32; 1024];
    alphas[0] = seed;
    let mut a = seed;
    let mut len = 1;
    while len < 1024 {
        a = (a * factor_256) >> 8;
        if a == 0 {
            break;
        }
        alphas[len] = a;
        len += 1;
    }
    (alphas, len)
}

/// Same decay iteration as [`compute_alphas`] but returns only the length (`alpha_len`). Callers that need the per-ray pixel reach to pre-size damage rects (textbox focus glow padding, chrome shadow extent) use this so the geometry the host clears + finalizes matches *exactly* what the rasterizer will paint — no early cutoff at the bbox edge, no over-clearing past where rays actually decay to zero.
pub fn ray_reach_px(seed: u32, factor_256: u32) -> usize {
    if seed == 0 || factor_256 == 0 || factor_256 >= 256 {
        return 0;
    }
    let mut a = seed;
    let mut len = 1usize;
    while len < 1024 {
        a = (a * factor_256) >> 8;
        if a == 0 {
            break;
        }
        len += 1;
    }
    len
}

/// Right-edge glow ray pass. Per row in the pill's vertical span, scans the buffer's α byte from the right end of the bbox leftward to find that row's first fully-opaque pixel (α=0xFF — the pill's interior, set by the squircle's writes), then walks rightward writing a glow pixel each step whose α decays exponentially from `seed` via `factor_256` — same curve as [`crate::paint::paint_shadow`], just emitting white at 0°/180° instead of black at 45°. Each write composes UNDER the existing target via [`Blend::under`] (Normal).
///
/// `factor_256 ∈ [1, 255]`: 240 ≈ ×0.9375 per step (~60-pixel reach at seed=0x80), 250 ≈ ×0.9766 (~150-pixel). Caller derives this from `effective_span` (or font_size) so glow reach is RU-invariant. Uses [`compute_alphas`] to precompute the decay table — hot loop is a table lookup.
pub fn apply_textbox_glow_right(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    glow_colour: u32,
    seed: u32,
    factor_256: u32,
    clip: Option<Clip>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if pill_w <= 0 || pill_h <= 0 || seed == 0 || factor_256 == 0 || factor_256 >= 256 {
        return;
    }
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    let clip_rect = Clip::resolve(clip, buf_w, buf_h);
    let y0 = (pill_y.max(0) as usize).max(clip_rect.y_start);
    let y1 = ((pill_y + pill_h).min(buf_h_i).max(0) as usize).min(clip_rect.y_end);
    if y0 >= y1 {
        return;
    }
    let scan_left = (pill_x.max(0) as usize).max(clip_rect.x_start);
    let scan_right = ((pill_x + pill_w).min(buf_w_i).max(0) as usize).min(clip_rect.x_end);
    if scan_left >= scan_right {
        return;
    }

    let (alphas, alpha_len) = compute_alphas(seed, factor_256);
    {
        let dx0 = scan_left;
        let dx1 = (scan_right + alpha_len).min(buf_w).min(clip_rect.x_end);
        canvas.damage.add_bounds(dx0, y0, dx1, y1);
    }

    let glow_rgb = glow_colour & 0x00FF_FFFF;
    let pixels: &mut [u32] = canvas.pixels;
    let ray_x_end = clip_rect.x_end.min(buf_w);

    for y in y0..y1 {
        let row_base = y * buf_w;
        let mut opaque_x: Option<usize> = None;
        for x in (scan_left..scan_right).rev() {
            if (pixels[row_base + x] >> 24) == 0xFF {
                opaque_x = Some(x);
                break;
            }
        }
        let Some(ox) = opaque_x else {
            continue;
        };
        let ray_start = ox + 1;
        for k in 0..alpha_len {
            let bx = ray_start + k;
            if bx >= ray_x_end {
                break;
            }
            let new_pixel = (alphas[k] << 24) | glow_rgb;
            let idx = row_base + bx;
            pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
        }
    }
}

/// Left-edge glow ray pass. Mirror of [`apply_textbox_glow_right`]: per row in the pill's vertical span, scans the buffer's α byte from the LEFT end of the bbox rightward to find the first fully-opaque pixel, then walks `reach_px` steps LEFT writing a linearly-tapered glow underneath. Each write composes UNDER the existing target via [`Blend::under`] (Normal).
pub fn apply_textbox_glow_left(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    glow_colour: u32,
    seed: u32,
    factor_256: u32,
    clip: Option<Clip>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if pill_w <= 0 || pill_h <= 0 || seed == 0 || factor_256 == 0 || factor_256 >= 256 {
        return;
    }
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    let clip_rect = Clip::resolve(clip, buf_w, buf_h);
    let y0 = (pill_y.max(0) as usize).max(clip_rect.y_start);
    let y1 = ((pill_y + pill_h).min(buf_h_i).max(0) as usize).min(clip_rect.y_end);
    if y0 >= y1 {
        return;
    }
    let scan_left = (pill_x.max(0) as usize).max(clip_rect.x_start);
    let scan_right = ((pill_x + pill_w).min(buf_w_i).max(0) as usize).min(clip_rect.x_end);
    if scan_left >= scan_right {
        return;
    }

    let (alphas, alpha_len) = compute_alphas(seed, factor_256);
    {
        let dx0 = scan_left.saturating_sub(alpha_len).max(clip_rect.x_start);
        let dx1 = scan_right;
        canvas.damage.add_bounds(dx0, y0, dx1, y1);
    }

    let glow_rgb = glow_colour & 0x00FF_FFFF;
    let pixels: &mut [u32] = canvas.pixels;
    let ray_x_start = clip_rect.x_start;

    for y in y0..y1 {
        let row_base = y * buf_w;
        let mut opaque_x: Option<usize> = None;
        for x in scan_left..scan_right {
            if (pixels[row_base + x] >> 24) == 0xFF {
                opaque_x = Some(x);
                break;
            }
        }
        let Some(ox) = opaque_x else {
            continue;
        };
        if ox == 0 {
            continue;
        }
        let ray_start = ox - 1;
        for k in 0..alpha_len {
            if ray_start < k {
                break;
            }
            let bx = ray_start - k;
            if bx < ray_x_start {
                break;
            }
            let new_pixel = (alphas[k] << 24) | glow_rgb;
            let idx = row_base + bx;
            pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
        }
    }
}

/// Top-edge glow ray pass. Per column in the pill's horizontal span, scans the buffer's α byte from the TOP of the bbox downward to find the first fully-opaque pixel (α=0xFF — the squircle's interior), then walks `reach_px` steps UP writing a glow pixel each step whose α tapers linearly from `seed` down toward 0. Each write composes UNDER existing target via [`Blend::under`] (Normal).
///
/// Why the scan only catches squircle pixels: the horizontal glow passes (left/right) write α ≤ `seed_horizontal` (typically 0x80) which is well below 0xFF, so they don't trigger the "fully opaque" scan condition — the vertical pass cleanly walks past horizontal glow pixels until it reaches the silhouette's actual α=0xFF interior.
pub fn apply_textbox_glow_top(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    glow_colour: u32,
    seed: u32,
    factor_256: u32,
    clip: Option<Clip>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if pill_w <= 0 || pill_h <= 0 || seed == 0 || factor_256 == 0 || factor_256 >= 256 {
        return;
    }
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    let clip_rect = Clip::resolve(clip, buf_w, buf_h);
    let x0 = (pill_x.max(0) as usize).max(clip_rect.x_start);
    let x1 = ((pill_x + pill_w).min(buf_w_i).max(0) as usize).min(clip_rect.x_end);
    if x0 >= x1 {
        return;
    }
    let scan_top = (pill_y.max(0) as usize).max(clip_rect.y_start);
    let scan_bot = ((pill_y + pill_h).min(buf_h_i).max(0) as usize).min(clip_rect.y_end);
    if scan_top >= scan_bot {
        return;
    }

    let (alphas, alpha_len) = compute_alphas(seed, factor_256);
    {
        let dy0 = scan_top.saturating_sub(alpha_len).max(clip_rect.y_start);
        let dy1 = scan_bot;
        canvas.damage.add_bounds(x0, dy0, x1, dy1);
    }

    let glow_rgb = glow_colour & 0x00FF_FFFF;
    let pixels: &mut [u32] = canvas.pixels;
    let ray_y_start = clip_rect.y_start;

    for x in x0..x1 {
        let mut opaque_y: Option<usize> = None;
        for y in scan_top..scan_bot {
            if (pixels[y * buf_w + x] >> 24) == 0xFF {
                opaque_y = Some(y);
                break;
            }
        }
        let Some(oy) = opaque_y else {
            continue;
        };
        if oy == 0 {
            continue;
        }
        let ray_start = oy - 1;
        for k in 0..alpha_len {
            if ray_start < k {
                break;
            }
            let by = ray_start - k;
            if by < ray_y_start {
                break;
            }
            let new_pixel = (alphas[k] << 24) | glow_rgb;
            let idx = by * buf_w + x;
            pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
        }
    }
}

/// Bottom-edge glow ray pass. Mirror of [`apply_textbox_glow_top`]: per column in the pill's horizontal span, scans the buffer's α byte from the BOTTOM of the bbox upward to find the first fully-opaque pixel, then walks `reach_px` steps DOWN writing a linearly-tapered glow underneath.
pub fn apply_textbox_glow_bottom(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    glow_colour: u32,
    seed: u32,
    factor_256: u32,
    clip: Option<Clip>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if pill_w <= 0 || pill_h <= 0 || seed == 0 || factor_256 == 0 || factor_256 >= 256 {
        return;
    }
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    let clip_rect = Clip::resolve(clip, buf_w, buf_h);
    let x0 = (pill_x.max(0) as usize).max(clip_rect.x_start);
    let x1 = ((pill_x + pill_w).min(buf_w_i).max(0) as usize).min(clip_rect.x_end);
    if x0 >= x1 {
        return;
    }
    let scan_top = (pill_y.max(0) as usize).max(clip_rect.y_start);
    let scan_bot = ((pill_y + pill_h).min(buf_h_i).max(0) as usize).min(clip_rect.y_end);
    if scan_top >= scan_bot {
        return;
    }

    let (alphas, alpha_len) = compute_alphas(seed, factor_256);
    {
        let dy0 = scan_top;
        let dy1 = (scan_bot + alpha_len).min(buf_h).min(clip_rect.y_end);
        canvas.damage.add_bounds(x0, dy0, x1, dy1);
    }

    let glow_rgb = glow_colour & 0x00FF_FFFF;
    let pixels: &mut [u32] = canvas.pixels;
    let ray_y_end = clip_rect.y_end.min(buf_h);

    for x in x0..x1 {
        let mut opaque_y: Option<usize> = None;
        for y in (scan_top..scan_bot).rev() {
            if (pixels[y * buf_w + x] >> 24) == 0xFF {
                opaque_y = Some(y);
                break;
            }
        }
        let Some(oy) = opaque_y else {
            continue;
        };
        let ray_start = oy + 1;
        for k in 0..alpha_len {
            let by = ray_start + k;
            if by >= ray_y_end {
                break;
            }
            let new_pixel = (alphas[k] << 24) | glow_rgb;
            let idx = by * buf_w + x;
            pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
        }
    }
}

/// Paint a 4-directional blur glow around a textbox pill silhouette.
///
/// Loop structure + adder/intensity math ported from photon's `apply_textbox_glow` (`compositing.rs:4479`). Pixel writes compose UNDER whatever's already in the buffer via [`Blend::under`] (Normal) — same kernel as every other fluor primitive. Caller paints squircle FIRST (topmost), then this; the squircle's opaque interior takes the under() early-out so glow stays outside the silhouette, AA-edge pixels (α<255) let glow bleed through the soft transition.
pub fn apply_textbox_glow(
    canvas: &mut Canvas,
    mask: &[u8],
    center_x: isize,
    center_y: isize,
    box_width: usize,
    box_height: usize,
    glow_colour: u32,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    let blur_h = 32usize;
    let blur_v = 16usize;

    let half_h = (box_height / 2) as isize;
    if center_y < half_h || (center_y + half_h) as usize >= buf_h {
        return;
    }
    if center_x < 0 || (center_x as usize) >= buf_w {
        return;
    }
    let cy = center_y as usize;
    let cx = center_x as usize;

    let y_top = cy - box_height / 2;
    let y_bot = cy + box_height / 2;
    // Damage = glow halo bbox (box + lateral blur padding, vertical box + vertical blur).
    {
        let x0 = cx.saturating_sub(box_width / 2 + blur_h);
        let x1 = (cx + box_width / 2 + blur_h).min(buf_w);
        let y0 = y_top.saturating_sub(blur_v);
        let y1 = (y_bot + blur_v).min(buf_h);
        canvas.damage.add_bounds(x0, y0, x1, y1);
    }
    let pixels: &mut [u32] = canvas.pixels;

    // Find horizontal bounds by scanning mask at center row, starting at the pill center.
    let mut x_left = cx;
    let mut x_right = cx;
    let scan = cy * buf_w;
    for lx in (0..cx).rev() {
        if mask[scan + lx] > 0 {
            x_left = lx;
        } else {
            break;
        }
    }
    for rx in cx..buf_w {
        if mask[scan + rx] > 0 {
            x_right = rx;
        } else {
            break;
        }
    }

    let corner_r = 2 * box_width * box_height / (box_width + box_height);
    let xvs = x_left + corner_r;
    let xve = x_right - corner_r;
    let yhs = y_top + corner_r;
    let yhe = y_bot - corner_r;

    let glow_rgb = glow_colour & 0x00FF_FFFF;

    // Right blur.
    for y in y_top..y_bot {
        let mut adder = 0u32;
        let start = x_right
            - (yhs as isize - y as isize).max(0) as usize
            - (y as isize - yhe as isize).max(0) as usize;
        for bx in start..x_right + blur_h {
            if bx >= buf_w {
                break;
            }
            let idx = y * buf_w + bx;
            if bx > 0 && mask[idx] < mask[idx - 1] {
                adder += (mask[idx - 1] - mask[idx]) as u32;
            }
            adder = (adder * 15 >> 4).min(71);
            let intensity = (adder * (255 - mask[idx]) as u32) >> 8;
            if intensity > 0 {
                let new_pixel = (intensity << 24) | glow_rgb;
                pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
            }
        }
    }
    // Left blur.
    for y in y_top..y_bot {
        let mut adder = 0u32;
        let end = x_left
            + (yhs as isize - y as isize).max(0) as usize
            + (y as isize - yhe as isize).max(0) as usize;
        for bx in (x_left.saturating_sub(blur_h)..=end).rev() {
            let idx = y * buf_w + bx;
            if bx + 1 < buf_w && mask[idx] < mask[idx + 1] {
                adder += (mask[idx + 1] - mask[idx]) as u32;
            }
            adder = (adder * 15 >> 4).min(71);
            let intensity = (adder * (255 - mask[idx]) as u32) >> 8;
            if intensity > 0 {
                let new_pixel = (intensity << 24) | glow_rgb;
                pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
            }
        }
    }
    // Down blur.
    for bx in x_left..x_right {
        let mut adder = 0u32;
        let start = y_bot
            - (xvs as isize - bx as isize).max(0) as usize
            - (bx as isize - xve as isize).max(0) as usize;
        for by in start..y_bot + blur_v {
            if by >= buf_h {
                break;
            }
            let idx = by * buf_w + bx;
            if by > 0 {
                let ia = (by - 1) * buf_w + bx;
                if mask[idx] < mask[ia] {
                    adder += (mask[ia] - mask[idx]) as u32;
                }
            }
            adder = (adder * 3 >> 2).min(70);
            let intensity = (adder * (255 - mask[idx]) as u32) >> 8;
            if intensity > 0 {
                let new_pixel = (intensity << 24) | glow_rgb;
                pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
            }
        }
    }
    // Up blur.
    for bx in x_left..x_right {
        let mut adder = 0u32;
        let end = y_top
            + (xvs as isize - bx as isize).max(0) as usize
            + (bx as isize - xve as isize).max(0) as usize;
        for by in (0..=end).rev() {
            if by + blur_v < y_top {
                break;
            }
            if by >= buf_h {
                continue;
            }
            let idx = by * buf_w + bx;
            if by + 1 < buf_h {
                let ib = (by + 1) * buf_w + bx;
                if mask[idx] < mask[ib] {
                    adder += (mask[ib] - mask[idx]) as u32;
                }
            }
            adder = (adder * 3 >> 2).min(70);
            let intensity = (adder * (255 - mask[idx]) as u32) >> 8;
            if intensity > 0 {
                let new_pixel = (intensity << 24) | glow_rgb;
                pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
            }
        }
    }
}
