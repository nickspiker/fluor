//! Focus glow rasterizers. Four directional ray passes (right / left / top / bottom) that scan the canvas's α channel to find the silhouette of the pill to glow around, then emit decaying rays of partial-α `glow_colour` pixels outward. Plus a 4-way blur variant used by photon's textbox glow shape, and the shared decay-table helper.
//!
//! The `factor_256` curve: `alpha[k] = (alpha[k-1] × factor_256) >> 8`. Geometric decay starting from `seed` (typically `0x80` for horizontal, `0x40` for vertical). [`ray_reach_px`] reports how many steps until α decays to zero — callers use it to pre-size damage rects so the host clears + finalizes exactly the area the rasterizer will paint.

use super::HitId;
use crate::canvas::Canvas;
use crate::paint::Clip;
use crate::pixel::{BlendMode, Pixel};

// Suppress the unused-import warning when this file gets re-included via `pub use glow::*;` — HitId isn't referenced here, but the import keeps the symbol available for future glow variants that might want to stamp a hit map.
#[allow(dead_code)]
fn _unused_hit_id_ref(_: HitId) {}

/// Precompute the per-step α decay table for a ray, in the depth's α scale. `alphas[0] = seed`; `alphas[k] = (alphas[k-1] * factor_256) >> 8`. Returns the populated prefix length. Replaces the per-step mul on the hot path with an array lookup. 1024 entries cover the worst case (seed 0x40, factor 254 → ~528 non-zero steps).
/// `seed` is an 8-bit α (the callers' 0x80 / 0x40) and the ray's length is always the 8-bit decay's, so [`ray_reach_px`] sizes the damage exactly at every depth; at 16 bits the values run the same recurrence from `seed × 257`, keeping the fraction each floor drops at 8 bits, so the ray fades in 2⁸ times the steps over the same reach.
#[inline]
fn compute_alphas<P: Pixel>(seed: u32, factor_256: u32) -> ([u32; 1024], usize) {
    let mut alphas = [0u32; 1024];
    // ONE / 255 is 1 at 8 bits (so `a` is `a8` there, step for step) and 257 at 16.
    let mut a = seed * (P::ONE / 255);
    alphas[0] = a;
    let mut a8 = seed;
    let mut len = 1;
    while len < 1024 {
        a8 = (a8 * factor_256) >> 8;
        if a8 == 0 {
            break;
        }
        a = (a * factor_256) >> 8;
        alphas[len] = a;
        len += 1;
    }
    (alphas, len)
}

/// The silhouette scan's "inside": α full to 8 bits.
/// A squircle AA weight of `0xFF00` or more is a whole byte at 8 bits (α 255) and 99.6 % at 16; reading it as inside at every depth puts each ray's first pixel in the same place at both, so the glow's geometry does not depend on the depth (with `α == ONE` the 16-bit rays started a pixel further out wherever an edge weight sat in that last byte).
#[inline(always)]
fn interior<P: Pixel>(p: P) -> bool {
    p.alpha() >> (P::DEPTH - 8) == (1 << 8) - 1
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
pub fn apply_textbox_glow_right<P: Pixel>(
    canvas: &mut Canvas<P>,
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
    // WHY/PROOF: the pill's position is the caller's layout, partly or wholly off the buffer whenever a widget scrolls past an edge — the max(0) and clip mins below keep every scan and ray inside the pixel buffer.
    let clip_rect = Clip::resolve(clip, buf_w, buf_h);
    let y0 = (pill_y.max(0) as usize).max(clip_rect.y_start);
    let y1 = ((pill_y + pill_h).max(0) as usize).min(clip_rect.y_end);
    if y0 >= y1 {
        return;
    }
    let scan_left = (pill_x.max(0) as usize).max(clip_rect.x_start);
    let scan_right = ((pill_x + pill_w).max(0) as usize).min(clip_rect.x_end);
    if scan_left >= scan_right {
        return;
    }

    let (alphas, alpha_len) = compute_alphas::<P>(seed, factor_256);
    {
        let dx0 = scan_left;
        let dx1 = (scan_right + alpha_len).min(clip_rect.x_end);
        canvas.damage.add_bounds(dx0, y0, dx1, y1);
    }

    let glow_rgb = P::from_g2t8(glow_colour).rgb();
    let pixels: &mut [P] = canvas.pixels;
    let ray_x_end = clip_rect.x_end;

    for y in y0..y1 {
        let row_base = y * buf_w;
        let mut opaque_x: Option<usize> = None;
        for x in (scan_left..scan_right).rev() {
            if interior(pixels[row_base + x]) {
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
            let new_pixel = glow_rgb.with_alpha(alphas[k]);
            let idx = row_base + bx;
            pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
        }
    }
}

/// Left-edge glow ray pass. Mirror of [`apply_textbox_glow_right`]: per row in the pill's vertical span, scans the buffer's α byte from the LEFT end of the bbox rightward to find the first fully-opaque pixel, then walks `reach_px` steps LEFT writing a linearly-tapered glow underneath. Each write composes UNDER the existing target via [`Blend::under`] (Normal).
pub fn apply_textbox_glow_left<P: Pixel>(
    canvas: &mut Canvas<P>,
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
    // WHY/PROOF: the pill's position is the caller's layout, partly or wholly off the buffer whenever a widget scrolls past an edge — the max(0) and clip mins below keep every scan and ray inside the pixel buffer.
    let clip_rect = Clip::resolve(clip, buf_w, buf_h);
    let y0 = (pill_y.max(0) as usize).max(clip_rect.y_start);
    let y1 = ((pill_y + pill_h).max(0) as usize).min(clip_rect.y_end);
    if y0 >= y1 {
        return;
    }
    let scan_left = (pill_x.max(0) as usize).max(clip_rect.x_start);
    let scan_right = ((pill_x + pill_w).max(0) as usize).min(clip_rect.x_end);
    if scan_left >= scan_right {
        return;
    }

    let (alphas, alpha_len) = compute_alphas::<P>(seed, factor_256);
    {
        let dx0 = (scan_left as isize - alpha_len as isize).max(clip_rect.x_start as isize) as usize;
        let dx1 = scan_right;
        canvas.damage.add_bounds(dx0, y0, dx1, y1);
    }

    let glow_rgb = P::from_g2t8(glow_colour).rgb();
    let pixels: &mut [P] = canvas.pixels;
    let ray_x_start = clip_rect.x_start;

    for y in y0..y1 {
        let row_base = y * buf_w;
        let mut opaque_x: Option<usize> = None;
        for x in scan_left..scan_right {
            if interior(pixels[row_base + x]) {
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
            let new_pixel = glow_rgb.with_alpha(alphas[k]);
            let idx = row_base + bx;
            pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
        }
    }
}

/// Top-edge glow ray pass. Per column in the pill's horizontal span, scans the buffer's α byte from the TOP of the bbox downward to find the first fully-opaque pixel (α=0xFF — the squircle's interior), then walks `reach_px` steps UP writing a glow pixel each step whose α tapers linearly from `seed` down toward 0. Each write composes UNDER existing target via [`Blend::under`] (Normal).
///
/// Why the scan only catches squircle pixels: the horizontal glow passes (left/right) write α ≤ `seed_horizontal` (typically 0x80) which is well below 0xFF, so they don't trigger the "fully opaque" scan condition — the vertical pass cleanly walks past horizontal glow pixels until it reaches the silhouette's actual α=0xFF interior.
pub fn apply_textbox_glow_top<P: Pixel>(
    canvas: &mut Canvas<P>,
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
    // WHY/PROOF: the pill's position is the caller's layout, partly or wholly off the buffer whenever a widget scrolls past an edge — the max(0) and clip mins below keep every scan and ray inside the pixel buffer.
    let clip_rect = Clip::resolve(clip, buf_w, buf_h);
    let x0 = (pill_x.max(0) as usize).max(clip_rect.x_start);
    let x1 = ((pill_x + pill_w).max(0) as usize).min(clip_rect.x_end);
    if x0 >= x1 {
        return;
    }
    let scan_top = (pill_y.max(0) as usize).max(clip_rect.y_start);
    let scan_bot = ((pill_y + pill_h).max(0) as usize).min(clip_rect.y_end);
    if scan_top >= scan_bot {
        return;
    }

    let (alphas, alpha_len) = compute_alphas::<P>(seed, factor_256);
    {
        let dy0 = (scan_top as isize - alpha_len as isize).max(clip_rect.y_start as isize) as usize;
        let dy1 = scan_bot;
        canvas.damage.add_bounds(x0, dy0, x1, dy1);
    }

    let glow_rgb = P::from_g2t8(glow_colour).rgb();
    let pixels: &mut [P] = canvas.pixels;
    let ray_y_start = clip_rect.y_start;

    for x in x0..x1 {
        let mut opaque_y: Option<usize> = None;
        for y in scan_top..scan_bot {
            if interior(pixels[y * buf_w + x]) {
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
            let new_pixel = glow_rgb.with_alpha(alphas[k]);
            let idx = by * buf_w + x;
            pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
        }
    }
}

/// Bottom-edge glow ray pass. Mirror of [`apply_textbox_glow_top`]: per column in the pill's horizontal span, scans the buffer's α byte from the BOTTOM of the bbox upward to find the first fully-opaque pixel, then walks `reach_px` steps DOWN writing a linearly-tapered glow underneath.
pub fn apply_textbox_glow_bottom<P: Pixel>(
    canvas: &mut Canvas<P>,
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
    // WHY/PROOF: the pill's position is the caller's layout, partly or wholly off the buffer whenever a widget scrolls past an edge — the max(0) and clip mins below keep every scan and ray inside the pixel buffer.
    let clip_rect = Clip::resolve(clip, buf_w, buf_h);
    let x0 = (pill_x.max(0) as usize).max(clip_rect.x_start);
    let x1 = ((pill_x + pill_w).max(0) as usize).min(clip_rect.x_end);
    if x0 >= x1 {
        return;
    }
    let scan_top = (pill_y.max(0) as usize).max(clip_rect.y_start);
    let scan_bot = ((pill_y + pill_h).max(0) as usize).min(clip_rect.y_end);
    if scan_top >= scan_bot {
        return;
    }

    let (alphas, alpha_len) = compute_alphas::<P>(seed, factor_256);
    {
        let dy0 = scan_top;
        let dy1 = (scan_bot + alpha_len).min(clip_rect.y_end);
        canvas.damage.add_bounds(x0, dy0, x1, dy1);
    }

    let glow_rgb = P::from_g2t8(glow_colour).rgb();
    let pixels: &mut [P] = canvas.pixels;
    let ray_y_end = clip_rect.y_end;

    for x in x0..x1 {
        let mut opaque_y: Option<usize> = None;
        for y in (scan_top..scan_bot).rev() {
            if interior(pixels[y * buf_w + x]) {
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
            let new_pixel = glow_rgb.with_alpha(alphas[k]);
            let idx = by * buf_w + x;
            pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
        }
    }
}

/// Where the glow's ceiling starts to bend: below it the adder passes untouched.
const GLOW_KNEE: u32 = 48;
/// What the glow's ceiling approaches and never passes.
const GLOW_CEIL: u32 = 74;

/// The glow's peak brightness as a smooth ceiling (Nick 2026-10-25) where the legacy curve had a hard min(71) and min(70).
/// Identity up to the knee, then the hyperbola knee + e·span / (e + span) with e the excess and span = ceil − knee: slope 1 on both sides of the knee (C¹), approaching the ceiling asymptotically (the quotient rounds, so a full step peaks where the legacy cap did).
/// A full 255 edge step lands where the hard caps sat: 239 → 71 across, 191 → 70 down and up.
#[inline]
fn glow_ceiling(x: u32) -> u32 {
    if x <= GLOW_KNEE {
        return x;
    }
    let (e, span) = (x - GLOW_KNEE, GLOW_CEIL - GLOW_KNEE);
    GLOW_KNEE + (e * span + (e + span) / 2) / (e + span)
}

/// Paint a 4-directional blur glow around a textbox pill silhouette.
///
/// Loop structure + adder/intensity math ported from photon's `apply_textbox_glow` (`compositing.rs:4479`). Pixel writes compose UNDER whatever's already in the buffer via [`Blend::under`] (Normal) — same kernel as every other fluor primitive. Caller paints squircle FIRST (topmost), then this; the squircle's opaque interior takes the under() early-out so glow stays outside the silhouette, AA-edge pixels (α<255) let glow bleed thru the soft transition.
pub fn apply_textbox_glow<P: Pixel>(
    canvas: &mut Canvas<P>,
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
    // WHY/PROOF: a pill near the buffer's edge puts the blur padding past it — the saturating subs and buffer mins here (and the left blur's start below) keep the damage rect and the left ray inside the buffer.
    // Damage = glow halo bbox (box + lateral blur padding, vertical box + vertical blur).
    {
        let x0 = cx.saturating_sub(box_width / 2 + blur_h);
        let x1 = (cx + box_width / 2 + blur_h).min(buf_w);
        let y0 = y_top.saturating_sub(blur_v);
        let y1 = (y_bot + blur_v).min(buf_h);
        canvas.damage.add_bounds(x0, y0, x1, y1);
    }
    let pixels: &mut [P] = canvas.pixels;

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

    let glow_rgb = P::from_g2t8(glow_colour).rgb();
    // The adder and the mask are bytes; their product over 255 · 256 is the 8-bit `(adder · (255 − m)) >> 8` exactly at 8 bits (the 255 cancels) and the same fraction of ONE at 16, where the bits the shift dropped stay.
    let intensity_of = |adder: u32, m: u8| adder * (255 - m) as u32 * P::ONE / (255 << 8);

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
            adder = glow_ceiling(adder * 15 >> 4);
            let intensity = intensity_of(adder, mask[idx]);
            if intensity > 0 {
                let new_pixel = glow_rgb.with_alpha(intensity);
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
            adder = glow_ceiling(adder * 15 >> 4);
            let intensity = intensity_of(adder, mask[idx]);
            if intensity > 0 {
                let new_pixel = glow_rgb.with_alpha(intensity);
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
            adder = glow_ceiling(adder * 3 >> 2);
            let intensity = intensity_of(adder, mask[idx]);
            if intensity > 0 {
                let new_pixel = glow_rgb.with_alpha(intensity);
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
            adder = glow_ceiling(adder * 3 >> 2);
            let intensity = intensity_of(adder, mask[idx]);
            if intensity > 0 {
                let new_pixel = glow_rgb.with_alpha(intensity);
                pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smooth ceiling keeps the legacy peaks, rises monotonically and never passes its ceiling.
    #[test]
    fn glow_ceiling_keeps_the_legacy_peaks() {
        assert_eq!(glow_ceiling(239), 71, "a full step across");
        assert_eq!(glow_ceiling(191), 70, "a full step down or up");
        assert_eq!(glow_ceiling(GLOW_KNEE), GLOW_KNEE);
        let mut last = 0;
        for x in 0..=4096u32 {
            let y = glow_ceiling(x);
            assert!(y >= last && y <= GLOW_CEIL, "x {x}: {y} after {last}");
            last = y;
        }
    }
}
