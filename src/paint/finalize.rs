//! OS-boundary finalize step — folds darkness→visible XOR, clip-mask α multiply, and (Linux-only) RGB premultiplication into one pass per pixel. Two entry points: [`finalize_for_os`] (in-place) and [`finalize_into_screen`] (read scratch, write to screen sub-rect, used by the fullscreen-compositor architecture). The shared `_chunk_dispatch` / `_chunk_simd` / `_chunk_scalar` family applies the same math via either SIMD lanes (8 px / iter) or scalar; both paths produce bit-identical output.
//!
//! Also home to [`shift_screen_wrap`], the in-place 2-D wrap-shift used by the drag-to-move fast path. Lives here because it operates on the same post-finalize screen buffer.

use super::{
    DEBUG_SHOW_ALPHA, DEBUG_SHOW_ALPHA_FORCE_OPAQUE, DEBUG_SHOW_ALPHA_GRAYSCALE,
    DEBUG_SHOW_HITMASK, DEBUG_SHOW_OPAQUE_SCAN, DEBUG_SKIP_PREMULT,
};

/// Boundary step that finalizes the present buffer for the OS in a **single pass** per pixel — folds the darkness→visible flip, the window-shape clip-mask multiply, the Linux RGB premultiply, and the pack into one go. Walks `pixels` and `clip_mask` in lockstep; both slices must be the same length.
///
/// Per pixel:
/// 1. `v = pixel ^ 0x00FFFFFF` — single XOR flips RGB darkness to visible (255 − dark). α stays as α (already opacity-direction in storage).
/// 2. `final_α = (α × clip_mask_α) >> 8` — multiply with the window-shape clip; trims to the window's actual shape while preserving any partial α the under-chain produced.
/// 3. Premultiply RGB by `final_α / 256`. Required on all platforms for clean AA edges.
/// 4. Pack back into `0xααRRGGBB`.
///
/// Debug toggles:
/// * `DEBUG_SHOW_ALPHA` (`[]a`): replace each pixel with `(final_α, final_α, final_α, 0xFF)` — grayscale α visualization, opaque so the OS shows it.
/// * `DEBUG_SKIP_PREMULT` (`[]p`): skip the Linux RGB×α step.
pub fn finalize_for_os(pixels: &mut [u32], clip_mask: &[u8]) {
    let alpha_mode = DEBUG_SHOW_ALPHA.load(std::sync::atomic::Ordering::Relaxed);
    let n = pixels.len().min(clip_mask.len());
    if n == 0 {
        return;
    }

    // Debug-visualization paths stay scalar — they're rare (toggle-only) and not worth SIMD-izing. Hitmask piggybacks on FORCE_OPAQUE here too.
    let hitmask = DEBUG_SHOW_HITMASK.load(std::sync::atomic::Ordering::Relaxed);
    let effective_alpha_mode = if hitmask {
        DEBUG_SHOW_ALPHA_FORCE_OPAQUE
    } else {
        alpha_mode
    };
    if effective_alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE
        || effective_alpha_mode == DEBUG_SHOW_ALPHA_FORCE_OPAQUE
    {
        finalize_scalar_debug_inplace(&mut pixels[..n], &clip_mask[..n], effective_alpha_mode);
        return;
    }

    // Premultiply RGB by final_α before handing to the OS compositor. Required on both Linux (X11 composite assumes premultiplied) and macOS (Metal PostMultiplied in practice needs it for clean AA edges — without it, edge pixels with fractional α show harsh checkerboard artifacts).
    let skip_premult = DEBUG_SKIP_PREMULT.load(std::sync::atomic::Ordering::Relaxed);

    let pixels = &mut pixels[..n];
    let clip = &clip_mask[..n];

    // Chunk size of 4096 pixels (16 KiB of u32). Large enough to amortize Rayon's ~1 µs task-dispatch overhead against ~10 µs of SIMD work per chunk; small enough that a typical 8-core system pulls hundreds of tasks from a 4K (8M pixel) finalize and load-balances cleanly. On non-Rayon builds, this is just a sequential walk in 4096-pixel windows.
    const CHUNK: usize = 4096;

    crate::par::par_chunks(pixels, CHUNK, |off, chunk| {
        let clip_chunk = &clip[off..off + chunk.len()];
        finalize_chunk_dispatch(chunk, clip_chunk, skip_premult);
    });
}

/// Per-chunk dispatcher: SIMD path when the `simd` feature is enabled, scalar fallback otherwise. Both branches produce bit-identical output (the SIMD path is a straight lane-wise translation of the scalar math, not an approximation).
#[inline]
fn finalize_chunk_dispatch(chunk: &mut [u32], clip: &[u8], skip_premult: bool) {
    #[cfg(feature = "simd")]
    {
        finalize_chunk_simd(chunk, clip, skip_premult);
    }
    #[cfg(not(feature = "simd"))]
    {
        finalize_chunk_scalar(chunk, clip, skip_premult);
    }
}

/// SIMD finalize kernel: 8 pixels per inner iter (u32x8), scalar tail for the leftover 0..7. Same math as [`finalize_chunk_scalar`] lane-by-lane.
#[cfg(feature = "simd")]
fn finalize_chunk_simd(chunk: &mut [u32], clip: &[u8], skip_premult: bool) {
    use crate::simd::{LANES, u32x8};
    let n = chunk.len();
    let mask_ff = u32x8::splat(0xFF);
    let xor_flip = u32x8::splat(0x00FFFFFF);
    let const_256 = u32x8::splat(256);

    let mut i = 0;
    while i + LANES <= n {
        let raw: [u32; 8] = chunk[i..i + LANES].try_into().unwrap();
        let v = u32x8::from(raw) ^ xor_flip;
        let m = u32x8::from([
            clip[i] as u32,
            clip[i + 1] as u32,
            clip[i + 2] as u32,
            clip[i + 3] as u32,
            clip[i + 4] as u32,
            clip[i + 5] as u32,
            clip[i + 6] as u32,
            clip[i + 7] as u32,
        ]);
        let inner_a = (v >> 24) & mask_ff;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { const_256 } else { final_a };
        let r = (((v >> 16) & mask_ff) * s) >> 8;
        let g = (((v >> 8) & mask_ff) * s) >> 8;
        let b = ((v & mask_ff) * s) >> 8;
        let out: u32x8 = (final_a << 24) | (r << 16) | (g << 8) | b;
        chunk[i..i + LANES].copy_from_slice(out.as_array_ref());
        i += LANES;
    }
    while i < n {
        let v = chunk[i] ^ 0x00FFFFFF;
        let m = clip[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { 256u32 } else { final_a };
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        chunk[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
        i += 1;
    }
}

/// Scalar finalize kernel for `--no-default-features` (no `simd`) builds. Identical math.
#[cfg(not(feature = "simd"))]
fn finalize_chunk_scalar(chunk: &mut [u32], clip: &[u8], skip_premult: bool) {
    for i in 0..chunk.len() {
        let v = chunk[i] ^ 0x00FFFFFF;
        let m = clip[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { 256u32 } else { final_a };
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        chunk[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
    }
}

/// Debug-visualization fallback: GRAYSCALE replaces each pixel with `(final_α, final_α, final_α, 0xFF)`; FORCE_OPAQUE keeps the kernel's visible RGB exactly and sets α=255 (lets you see what the kernel produced BEFORE the clip mask + premult trimmed it). Both stay scalar — they're debug-toggle paths, not on the hot path.
fn finalize_scalar_debug_inplace(pixels: &mut [u32], clip_mask: &[u8], alpha_mode: u8) {
    for i in 0..pixels.len() {
        let v = pixels[i] ^ 0x00FFFFFF;
        let m = clip_mask[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        if alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE {
            pixels[i] = 0xFF000000 | (final_a << 16) | (final_a << 8) | final_a;
        } else {
            // DEBUG_SHOW_ALPHA_FORCE_OPAQUE
            pixels[i] = 0xFF000000 | (v & 0x00FFFFFF);
        }
    }
}

/// 2D wrap-shift the screen buffer in place. Two passes — one per axis. The X pass walks each row, memmoves the row by `dx` columns, and pastes the wrap segment (the pixels that fall off one edge) at the opposite edge. The Y pass treats the whole buffer as a stack of rows, memmoves the rows by `dy` rows, and pastes the wrap row-block at the opposite end.
///
/// Used during in-buffer drag-to-move to skip the chrome / panes / shadow re-rasterization entirely — the window just slides thru the screen buffer with its existing pixels, and pixels that fall off any edge wrap around to the opposite end. On drag release the host does one full re-render to clear the wrap artefacts.
///
/// Per-pixel cost: one read + one write, via `slice::copy_within` which lowers to platform `memmove`. No branches inside the inner copy loops — the direction (right/left, up/down) selects between two precomputed `(src_range, dst_offset, wrap_src_range, wrap_dst_range)` tuples, then the copies execute unconditionally.
///
/// `dx` / `dy` are typically bounded by per-frame cursor motion (≪ screen dimensions). For oversized deltas (cursor teleport, stalled frame, multi-monitor span) the signed remainder is used: a shift by exactly `scr_w` is a full wrap = no-op, so `dx = scr_w + 100` is equivalent to `dx = 100` (same wrapped result). Keeps the function panic-free for any input.
pub fn shift_screen_wrap(screen: &mut [u32], scr_w: usize, scr_h: usize, dx: i32, dy: i32) {
    // Normalize to the (-scr_w, +scr_w) range via signed remainder. Direction (sign) is preserved.
    let signed_dx = if scr_w == 0 { 0 } else { dx % (scr_w as i32) };
    let signed_dy = if scr_h == 0 { 0 } else { dy % (scr_h as i32) };
    let nx = signed_dx.unsigned_abs() as usize;
    let ny = signed_dy.unsigned_abs() as usize;

    if signed_dx != 0 {
        let mut tmp_x = alloc::vec![0u32; nx];
        let (wrap_src, body_src, body_dst, wrap_dst) = if signed_dx > 0 {
            (scr_w - nx..scr_w, 0..scr_w - nx, nx, 0..nx)
        } else {
            (0..nx, nx..scr_w, 0, scr_w - nx..scr_w)
        };
        for y in 0..scr_h {
            let row_start = y * scr_w;
            let row = &mut screen[row_start..row_start + scr_w];
            tmp_x.copy_from_slice(&row[wrap_src.clone()]);
            row.copy_within(body_src.clone(), body_dst);
            row[wrap_dst.clone()].copy_from_slice(&tmp_x);
        }
    }

    if signed_dy != 0 {
        let mut tmp_y = alloc::vec![0u32; ny * scr_w];
        let split = (scr_h - ny) * scr_w;
        let (wrap_src, body_src, body_dst, wrap_dst) = if signed_dy > 0 {
            (split..scr_h * scr_w, 0..split, ny * scr_w, 0..ny * scr_w)
        } else {
            (
                0..ny * scr_w,
                ny * scr_w..scr_h * scr_w,
                0,
                split..scr_h * scr_w,
            )
        };
        tmp_y.copy_from_slice(&screen[wrap_src]);
        screen.copy_within(body_src, body_dst);
        screen[wrap_dst].copy_from_slice(&tmp_y);
    }
}

/// Combined finalize + blit: same per-pixel math as [`finalize_for_os`] (XOR darkness→visible, multiply clip_mask into α, Linux RGB×α premultiply) but reads from a `(win_w × win_h)` scratch buffer and writes into a `(scr_w × scr_h)` screen buffer at the offset `(rect_x, rect_y)`. Used by the fullscreen-compositor host path: the consumer renders into the scratch (window-space, contiguous), this function reads scratch + clip_mask once per pixel and writes the OS-ready ARGB into the screen buffer's sub-rect. The scratch buffer is **not mutated** — its α + darkness convention is preserved so a future incremental-rendering path can reuse it across frames without forcing a full re-render.
///
/// Pre-conditions: `rect_x + win_w ≤ scr_w` and `rect_y + win_h ≤ scr_h` (rect fits inside the screen). Caller is responsible for clearing the destination region (typically the whole screen buffer cleared to `0` so pixels outside `rect` stay α=0 and the OS compositor shows whatever's behind us).
///
/// One pass over pixels — same cost per pixel as `finalize_for_os` plus one address calculation for the screen-buffer offset.
pub fn finalize_into_screen(
    scratch: &[u32],
    clip_mask: &[u8],
    win_w: usize,
    win_h: usize,
    screen: &mut [u32],
    scr_w: usize,
    rect_x: i32,
    rect_y: i32,
    damage_clip: crate::canvas::PixelRect,
    full_repaint: bool,
) {
    let alpha_mode = DEBUG_SHOW_ALPHA.load(std::sync::atomic::Ordering::Relaxed);
    if scr_w == 0 || damage_clip.is_empty() {
        return;
    }
    let scr_h = screen.len() / scr_w;

    let rect_sy_min = (-rect_y).max(0) as usize;
    let rect_sx_min = (-rect_x).max(0) as usize;
    let rect_sy_max = win_h.min(((scr_h as i32) - rect_y).max(0) as usize);
    let rect_sx_max = win_w.min(((scr_w as i32) - rect_x).max(0) as usize);
    let sy_min = rect_sy_min.max(damage_clip.y0);
    let sx_min = rect_sx_min.max(damage_clip.x0);
    let sy_max = rect_sy_max.min(damage_clip.y1);
    let sx_max = rect_sx_max.min(damage_clip.x1);
    if sy_min >= sy_max || sx_min >= sx_max {
        return;
    }

    let dst_y_min = (rect_y + sy_min as i32) as usize;
    let dst_y_max = (rect_y + sy_max as i32) as usize;
    let dst_x_min = (rect_x + sx_min as i32) as usize;
    let row_len = sx_max - sx_min;

    let hitmask = DEBUG_SHOW_HITMASK.load(std::sync::atomic::Ordering::Relaxed);
    let effective_alpha_mode = if hitmask {
        DEBUG_SHOW_ALPHA_FORCE_OPAQUE
    } else {
        alpha_mode
    };
    if effective_alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE
        || effective_alpha_mode == DEBUG_SHOW_ALPHA_FORCE_OPAQUE
    {
        finalize_into_scalar_debug(
            scratch,
            clip_mask,
            screen,
            scr_w,
            win_w,
            effective_alpha_mode,
            sy_min,
            sy_max,
            sx_min,
            sx_max,
            rect_x,
            rect_y,
        );
        return;
    }

    let skip_premult = DEBUG_SKIP_PREMULT.load(std::sync::atomic::Ordering::Relaxed);

    let tint_scan = DEBUG_SHOW_OPAQUE_SCAN.load(std::sync::atomic::Ordering::Relaxed);
    if full_repaint {
        crate::par::par_rows(screen, scr_w, dst_y_min, dst_y_max, |dst_y, screen_row| {
            let sy = (dst_y as i32 - rect_y) as usize;
            let scratch_off = sy * win_w + sx_min;
            let src_chunk = &scratch[scratch_off..scratch_off + row_len];
            let clip_chunk = &clip_mask[scratch_off..scratch_off + row_len];
            let dst_chunk = &mut screen_row[dst_x_min..dst_x_min + row_len];
            finalize_into_chunk_dispatch(src_chunk, clip_chunk, dst_chunk, skip_premult);
        });
    } else {
        crate::par::par_rows(screen, scr_w, dst_y_min, dst_y_max, |dst_y, screen_row| {
            let sy = (dst_y as i32 - rect_y) as usize;
            let scratch_off = sy * win_w + sx_min;
            let clip_row = &clip_mask[scratch_off..scratch_off + row_len];
            let mut l = 0;
            while l < row_len && clip_row[l] != 255 {
                l += 1;
            }
            if l == row_len {
                return;
            }
            let mut r = row_len;
            while r > l && clip_row[r - 1] != 255 {
                r -= 1;
            }
            let len = r - l;
            let src_row = &scratch[scratch_off..scratch_off + row_len];
            let clip_chunk = &clip_row[l..l + len];
            let src_chunk = &src_row[l..l + len];
            let dst_chunk = &mut screen_row[dst_x_min + l..dst_x_min + l + len];
            finalize_into_chunk_dispatch(src_chunk, clip_chunk, dst_chunk, skip_premult);
            if tint_scan {
                for px in dst_chunk.iter_mut() {
                    let b = ((*px & 0xFF) as u8).saturating_add(16) as u32;
                    *px = (*px & 0xFFFF_FF00) | b;
                }
            }
        });
    }
}

/// Per-row src→dst dispatcher — same shape as [`finalize_chunk_dispatch`] but reading from a separate src buffer rather than in-place.
#[inline]
fn finalize_into_chunk_dispatch(src: &[u32], clip: &[u8], dst: &mut [u32], skip_premult: bool) {
    #[cfg(feature = "simd")]
    {
        finalize_into_chunk_simd(src, clip, dst, skip_premult);
    }
    #[cfg(not(feature = "simd"))]
    {
        finalize_into_chunk_scalar(src, clip, dst, skip_premult);
    }
}

/// SIMD finalize+blit kernel: reads from `src` + `clip`, writes to `dst`. Same math as the in-place [`finalize_chunk_simd`]; the only difference is the read/write split.
#[cfg(feature = "simd")]
fn finalize_into_chunk_simd(src: &[u32], clip: &[u8], dst: &mut [u32], skip_premult: bool) {
    use crate::simd::{LANES, u32x8};
    let n = src.len();
    let mask_ff = u32x8::splat(0xFF);
    let xor_flip = u32x8::splat(0x00FFFFFF);
    let const_256 = u32x8::splat(256);

    let mut i = 0;
    while i + LANES <= n {
        let raw: [u32; 8] = src[i..i + LANES].try_into().unwrap();
        let v = u32x8::from(raw) ^ xor_flip;
        let m = u32x8::from([
            clip[i] as u32,
            clip[i + 1] as u32,
            clip[i + 2] as u32,
            clip[i + 3] as u32,
            clip[i + 4] as u32,
            clip[i + 5] as u32,
            clip[i + 6] as u32,
            clip[i + 7] as u32,
        ]);
        let inner_a = (v >> 24) & mask_ff;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { const_256 } else { final_a };
        let r = (((v >> 16) & mask_ff) * s) >> 8;
        let g = (((v >> 8) & mask_ff) * s) >> 8;
        let b = ((v & mask_ff) * s) >> 8;
        let out: u32x8 = (final_a << 24) | (r << 16) | (g << 8) | b;
        dst[i..i + LANES].copy_from_slice(out.as_array_ref());
        i += LANES;
    }
    while i < n {
        let v = src[i] ^ 0x00FFFFFF;
        let m = clip[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { 256u32 } else { final_a };
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        dst[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
        i += 1;
    }
}

/// Scalar finalize+blit kernel for `--no-default-features` (no `simd`) builds.
#[cfg(not(feature = "simd"))]
fn finalize_into_chunk_scalar(src: &[u32], clip: &[u8], dst: &mut [u32], skip_premult: bool) {
    for i in 0..src.len() {
        let v = src[i] ^ 0x00FFFFFF;
        let m = clip[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { 256u32 } else { final_a };
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        dst[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
    }
}

/// Debug-visualization fallback for [`finalize_into_screen`]: GRAYSCALE / FORCE_OPAQUE paths stay scalar (debug-toggle, off the hot path). Sequential — Rayon adds no value here.
#[allow(clippy::too_many_arguments)]
fn finalize_into_scalar_debug(
    scratch: &[u32],
    clip_mask: &[u8],
    screen: &mut [u32],
    scr_w: usize,
    win_w: usize,
    alpha_mode: u8,
    sy_min: usize,
    sy_max: usize,
    sx_min: usize,
    sx_max: usize,
    rect_x: i32,
    rect_y: i32,
) {
    for sy in sy_min..sy_max {
        let dst_y = (rect_y + sy as i32) as usize;
        let dst_row = dst_y * scr_w;
        let src_row = sy * win_w;
        for sx in sx_min..sx_max {
            let scratch_idx = src_row + sx;
            if scratch_idx >= scratch.len() || scratch_idx >= clip_mask.len() {
                break;
            }
            let dst_idx = dst_row + (rect_x + sx as i32) as usize;
            let v = scratch[scratch_idx] ^ 0x00FFFFFF;
            let m = clip_mask[scratch_idx] as u32;
            let inner_a = (v >> 24) & 0xFF;
            let final_a = (inner_a * m) >> 8;
            if alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE {
                screen[dst_idx] = 0xFF000000 | (final_a << 16) | (final_a << 8) | final_a;
            } else {
                screen[dst_idx] = 0xFF000000 | (v & 0x00FFFFFF);
            }
        }
    }
}
