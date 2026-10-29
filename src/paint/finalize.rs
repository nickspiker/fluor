//! OS-boundary finalize step — folds darkness→visible XOR, clip-mask α multiply, and RGB premultiplication (every platform: X11 compositing, Core Animation, Android and Windows ULW_ALPHA all take premultiplied pixels, so no presenter premultiplies again) into one pass per pixel. Two entry points: [`finalize_for_os`] (in-place) and [`finalize_into_screen`] (read scratch, write to screen sub-rect, used by the fullscreen-compositor architecture). The shared `_chunk_dispatch` / `_chunk_simd` / `_chunk_scalar` family applies the same math via either SIMD lanes (8 px / iter) or scalar; both paths produce bit-identical output.
//!
//! Also home to [`shift_screen_wrap`], the in-place 2-D wrap-shift used by the drag-to-move fast path. Lives here because it operates on the same post-finalize screen buffer.

use super::{
    DEBUG_SHOW_ALPHA, DEBUG_SHOW_ALPHA_FORCE_OPAQUE, DEBUG_SHOW_ALPHA_GRAYSCALE,
    DEBUG_SHOW_HITMASK, DEBUG_SHOW_OPAQUE_SCAN, DEBUG_SKIP_PREMULT,
};

/// Boundary step that finalizes the present buffer for the OS in a **single pass** per pixel — folds the darkness→visible flip, the window-shape clip-mask multiply, the RGB premultiply (every platform), and the pack into one go. Walks `pixels` and `clip_mask` in lockstep; both slices must be the same length.
///
/// Per pixel:
/// 1. `v = theme::to_sdr_visible(pixel)` — one table lookup per channel flips the stored darkness and maps the buffer's `g2t` code to the surface's display code (γ2, light above white clipped to white on SDR). α stays as α (already opacity-direction in storage).
/// 2. `final_α = (α × clip_mask_α) >> 8` — multiply with the window-shape clip; trims to the window's actual shape while preserving any partial α the under-chain produced.
/// 3. Premultiply RGB by `final_α / 256`. Required on all platforms for clean AA edges.
/// 4. Pack back into `0xααRRGGBB`.
///
/// Debug toggles: * `DEBUG_SHOW_ALPHA` (`[]a`): replace each pixel with `(final_α, final_α, final_α, 0xFF)` — grayscale α visualization, opaque so the OS shows it.
/// * `DEBUG_SKIP_PREMULT` (`[]p`): skip the RGB×α step (debug only: every presenter expects premultiplied pixels).
pub fn finalize_for_os(pixels: &mut [u32], clip_mask: &[u8]) {
    let alpha_mode = DEBUG_SHOW_ALPHA.load(core::sync::atomic::Ordering::Relaxed);
    // The clip mask is the window's own shape at the window's size: a shorter one is a sizing bug upstream and panics at the slice below.
    let n = pixels.len();
    if n == 0 {
        return;
    }

    // Debug-visualization paths stay scalar — they're rare (toggle-only) and not worth SIMD-izing. Hitmask piggybacks on FORCE_OPAQUE here too.
    let hitmask = DEBUG_SHOW_HITMASK.load(core::sync::atomic::Ordering::Relaxed);
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

    // Premultiply RGB by final_α before handing to the OS compositor, the one premultiply on every platform: X11 composite assumes premultiplied; Core Animation composites premultiplied whatever wgpu's PostMultiplied label says (it only clears the layer's `opaque` flag); Android's surface and Windows' ULW_ALPHA take it as-is. Each channel lands ≤ α ((c·α) >> 8), which ULW_ALPHA requires.
    let skip_premult = DEBUG_SKIP_PREMULT.load(core::sync::atomic::Ordering::Relaxed);

    let pixels = &mut pixels[..n];
    let clip = &clip_mask[..n];

    // Chunk size of 4096 pixels (16 KiB of u32). Large enough to amortize Rayon's ~1 µs task-dispatch overhead against ~10 µs of SIMD work per chunk; small enough that a typical 8-core system pulls hundreds of tasks from a 4K (8M pixel) finalize and load-balances cleanly. On non-Rayon builds, this is just a sequential walk in 4096-pixel windows.
    const CHUNK: usize = 4096;

    let opaque = crate::paint::OPAQUE_BACKDROP.load(core::sync::atomic::Ordering::Relaxed);
    crate::par::par_chunks(pixels, CHUNK, |off, chunk| {
        let clip_chunk = &clip[off..off + chunk.len()];
        finalize_chunk_dispatch(chunk, clip_chunk, skip_premult);
        if opaque {
            // Premultiplied RGB over black with α forced solid — the backdrop is black, never the compositor's.
            for px in chunk.iter_mut() {
                *px |= 0xFF00_0000;
            }
        }
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
    let const_256 = u32x8::splat(256);

    let mut i = 0;
    while i + LANES <= n {
        // The colour step is a table lookup per channel (`theme::to_sdr_visible`: flip + `g2t` → SDR display), done lane by lane before the SIMD premultiply.
        let raw: [u32; 8] = core::array::from_fn(|k| crate::theme::to_sdr_visible(chunk[i + k]));
        let v = u32x8::from(raw);
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
        let v = crate::theme::to_sdr_visible(chunk[i]);
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
        let v = crate::theme::to_sdr_visible(chunk[i]);
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
        let v = crate::theme::to_sdr_visible(pixels[i]);
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

/// Scroll-copy fast path: vertically shift the contents of a sub-rectangle `[x0,x1) × [y0,y1)` of the surface buffer by `dy` backing pixels, IN PLACE and NON-wrapping. The rigid bulk of a scrolling pane is a `memmove`; the caller repaints ONLY the freshly-exposed band (via the frame's damage clip) instead of redrawing the whole pane — a scroll becomes a copy plus a sliver.
///
/// Distinct from [`shift_screen_wrap`] in three ways that matter for scrolling (not window-dragging): (1) confined to a rect, so pinned chrome OUTSIDE the rect (title bar, perimeter, a compose bar) is never touched; (2) never wraps — content that scrolls off the far edge is dropped, not brought around; (3) vertical only. The `|dy|`-tall band vacated at the near edge is left as whatever was there — correctness depends on the caller including that band in `damage_clip` so finalize overwrites it with the newly-scrolled-in content this frame.
///
/// Sign: `dy > 0` moves content DOWN (band exposed at the TOP of the rect); `dy < 0` moves content UP (band exposed at the BOTTOM). `|dy| ≥ rect height` is a no-op — nothing survives the shift, so the caller should just repaint the whole rect (a full repaint, not a scroll-copy). All bounds are clamped to the buffer, so out-of-range input is safe (no panic, no-op on a degenerate rect).
///
/// Per row: one `copy_within` (lowers to `memmove`). Rows are copied in the order that never reads an already-overwritten source (bottom-up for a down-shift, top-down for an up-shift); source and destination rows are always distinct, so no single copy overlaps itself.
#[allow(clippy::too_many_arguments)]
pub fn scroll_copy_rect<T: Copy>(
    screen: &mut [T],
    scr_w: usize,
    scr_h: usize,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    dy: i32,
) {
    // WHY/PROOF: the rect is the app's scroll hint, which can overhang the screen buffer (a pane taller than a shrunken window) — the row copies would index past the buffer.
    let x0 = x0.min(scr_w);
    let x1 = x1.min(scr_w);
    let y0 = y0.min(scr_h);
    let y1 = y1.min(scr_h);
    if dy == 0 || x1 <= x0 || y1 <= y0 {
        return;
    }
    let rows = y1 - y0;
    let n = dy.unsigned_abs() as usize;
    if n >= rows {
        return; // Shifted entirely out — nothing survives; caller repaints the whole rect.
    }
    let span = x1 - x0;
    if dy > 0 {
        // Content moves DOWN: dst row (y0+n .. y1) ← src row (y0 .. y1−n). Walk bottom-up so a source row is copied before a lower destination overwrites it.
        for dst in (y0 + n..y1).rev() {
            let src = dst - n;
            let s = src * scr_w + x0;
            let d = dst * scr_w + x0;
            screen.copy_within(s..s + span, d);
        }
    } else {
        // Content moves UP: dst row (y0 .. y1−n) ← src row (y0+n .. y1). Walk top-down for the same non-clobber reason.
        for dst in y0..y1 - n {
            let src = dst + n;
            let s = src * scr_w + x0;
            let d = dst * scr_w + x0;
            screen.copy_within(s..s + span, d);
        }
    }
}

/// Combined finalize + blit: same per-pixel math as [`finalize_for_os`] (XOR darkness→visible, multiply clip_mask into α, RGB×α premultiply) but reads from a `(win_w × win_h)` scratch buffer and writes into a `(scr_w × scr_h)` screen buffer at the offset `(rect_x, rect_y)`. Used by the fullscreen-compositor host path: the consumer renders into the scratch (window-space, contiguous), this function reads scratch + clip_mask once per pixel and writes the OS-ready ARGB into the screen buffer's sub-rect. The scratch buffer is **not mutated** — its α + darkness convention is preserved so a future incremental-rendering path can reuse it across frames without forcing a full re-render.
///
/// `over`: the scratch goes over an under layer the GPU present lays beneath it — partly covering pixels are finalized as their own colour (see [`finalize_over_chunk`]), not as the pixel over white.
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
    over: bool,
) {
    let alpha_mode = DEBUG_SHOW_ALPHA.load(core::sync::atomic::Ordering::Relaxed);
    if scr_w == 0 || damage_clip.is_empty() {
        return;
    }
    let scr_h = screen.len() / scr_w;

    // WHY/PROOF: rect_x and rect_y are the OS window's place on the screen buffer, negative or past its edge while the window hangs off-screen, and damage_clip is the app's rect — without these the copy would index outside both buffers.
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

    let hitmask = DEBUG_SHOW_HITMASK.load(core::sync::atomic::Ordering::Relaxed);
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

    let skip_premult = DEBUG_SKIP_PREMULT.load(core::sync::atomic::Ordering::Relaxed);
    // Over an under layer the layer is the backdrop: the chrome keeps its own α and the present fills the rest of the window from the layer.
    let opaque = crate::paint::OPAQUE_BACKDROP.load(core::sync::atomic::Ordering::Relaxed) && !over;

    let tint_scan = DEBUG_SHOW_OPAQUE_SCAN.load(core::sync::atomic::Ordering::Relaxed);
    if full_repaint {
        crate::par::par_rows(screen, scr_w, dst_y_min, dst_y_max, |dst_y, screen_row| {
            let sy = (dst_y as i32 - rect_y) as usize;
            let scratch_off = sy * win_w + sx_min;
            let src_chunk = &scratch[scratch_off..scratch_off + row_len];
            let clip_chunk = &clip_mask[scratch_off..scratch_off + row_len];
            let dst_chunk = &mut screen_row[dst_x_min..dst_x_min + row_len];
            if over {
                finalize_over_chunk(src_chunk, clip_chunk, dst_chunk);
            } else {
                finalize_into_chunk_dispatch(src_chunk, clip_chunk, dst_chunk, skip_premult);
            }
            if opaque {
                for px in dst_chunk.iter_mut() {
                    *px |= 0xFF00_0000;
                }
            }
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
            if over {
                finalize_over_chunk(src_chunk, clip_chunk, dst_chunk);
            } else {
                finalize_into_chunk_dispatch(src_chunk, clip_chunk, dst_chunk, skip_premult);
            }
            if opaque {
                for px in dst_chunk.iter_mut() {
                    *px |= 0xFF00_0000;
                }
            }
            if tint_scan {
                for px in dst_chunk.iter_mut() {
                    // WHY/PROOF: not a guard — the debug tint's saturation is what lets repeated incremental hits pile up to full blue; wrapping would blink back to dark every 16 hits.
                    let b = ((*px & 0xFF) as u8).saturating_add(16) as u32;
                    *px = (*px & 0xFFFF_FF00) | b;
                }
            }
        });
    }
}

/// Finalize for a scratch that goes OVER an under layer (the GPU present lays the app's own picture beneath the chrome, `host::gpu::UnderLayer`): the scratch's darkness is premultiplied by α (the under-chains build it so), so a partly covering pixel's own colour is its darkness ÷ α — that thru the display map, then premultiplied by α × clip, the way the present adds the layer's light under it.
/// The plain finalize takes the darkness as it is, which is the pixel over WHITE: right for a window whose content is opaque, too light here, where the map shows thru the anti-aliased edges of the chrome's text and controls.
/// An opaque or empty pixel comes out exactly as the plain finalize gives it.
fn finalize_over_chunk(src: &[u32], clip: &[u8], dst: &mut [u32]) {
    for i in 0..src.len() {
        let p = src[i];
        let a = p >> 24;
        let v = if a == 0 || a == 255 {
            crate::theme::to_sdr_visible(p)
        } else {
            let k = (255 << 16) / a;
            // WHY/PROOF: an app may write straight darkness with partial α (pack_argb takes visible RGB and α apart), whose darkness exceeds α — un-premultiplying it would carry past the byte into the next channel.
            let un = |d: u32| ((d * k + 32768) >> 16).min(255);
            crate::theme::to_sdr_visible((a << 24) | (un((p >> 16) & 0xFF) << 16) | (un((p >> 8) & 0xFF) << 8) | un(p & 0xFF))
        };
        let m = clip[i] as u32;
        let final_a = (a * m) >> 8;
        let r = (((v >> 16) & 0xFF) * final_a) >> 8;
        let g = (((v >> 8) & 0xFF) * final_a) >> 8;
        let b = ((v & 0xFF) * final_a) >> 8;
        dst[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
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
    let const_256 = u32x8::splat(256);

    let mut i = 0;
    while i + LANES <= n {
        let raw: [u32; 8] = core::array::from_fn(|k| crate::theme::to_sdr_visible(src[i + k]));
        let v = u32x8::from(raw);
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
        let v = crate::theme::to_sdr_visible(src[i]);
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
        let v = crate::theme::to_sdr_visible(src[i]);
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

/// The EXTENDED-RANGE finalize (HDR plan, Android first): the whole window from `scratch` (α + darkness, the `g2t` buffer) into an RGBA F16 surface whose dataspace is linear extended (1.0 = SDR white), `dst_stride` pixels per row.
/// Per pixel: final α = α × clip / 255; an opaque pixel takes three half lookups in `table` (the darkness flip, the display map and the half pack in one read each); a partly-covered one is premultiplied in Q16 integers from `table.q16` and packed ([`crate::theme::q16_to_half`]); α is written as half, or 1.0 over black when [`crate::paint::OPAQUE_BACKDROP`] is set, like the 8-bit path.
/// Channels go in the order they are stored — byte 0 to the first half — which is R, G, B on Android, where theme::fmt already put R in the low byte.
/// Returns the frame's brightest `g2t` code (255 − the smallest darkness under any visible pixel): the light the next frame asks the display for.
pub fn finalize_into_f16(
    scratch: &[u32],
    clip_mask: &[u8],
    win_w: usize,
    win_h: usize,
    dst: &mut [u64],
    dst_stride: usize,
    table: &crate::theme::HdrTable,
) -> u8 {
    if dst_stride == 0 || win_w == 0 {
        return 0;
    }
    // WHY/PROOF: dst is the OS-allocated surface buffer, whose height and stride can lag the window size across a resize — the rows would index past it.
    let rows = win_h.min(dst.len() / dst_stride);
    let w = win_w.min(dst_stride);
    let opaque = crate::paint::OPAQUE_BACKDROP.load(core::sync::atomic::Ordering::Relaxed);
    let min_dark = core::sync::atomic::AtomicU32::new(255);
    crate::par::par_rows(dst, dst_stride, 0, rows, |y, row| {
        let off = y * win_w;
        let m = finalize_f16_row(&scratch[off..off + w], &clip_mask[off..off + w], &mut row[..w], table, opaque);
        min_dark.fetch_min(m, core::sync::atomic::Ordering::Relaxed);
    });
    255 - min_dark.into_inner() as u8
}

/// The 10-BIT extended-brightness finalize (HDR plan, Android: RGBA_1010102, BT.2020, γ2.2, extended range): the whole window from `scratch` into a 2:10:10:10 surface, `dst_stride` pixels per row, R in the low bits.
/// Per pixel: final α = α × clip / 255; each channel one read of `table` (darkness flip + light ÷ declared ratio + γ2.2 in one); premultiplied ON CODES like the 8-bit path (Nick: blending stays on codes); α as 2 bits, or opaque when [`crate::paint::OPAQUE_BACKDROP`] is set.
/// Channels go in stored order — byte 0 to bits 0–9 — which is R, G, B on Android. Returns the frame's brightest `g2t` code, as [`finalize_into_f16`] does.
pub fn finalize_into_1010102(
    scratch: &[u32],
    clip_mask: &[u8],
    win_w: usize,
    win_h: usize,
    dst: &mut [u32],
    dst_stride: usize,
    table: &crate::theme::Hdr10Table,
) -> u8 {
    if dst_stride == 0 || win_w == 0 {
        return 0;
    }
    // WHY/PROOF: dst is the OS-allocated surface buffer, whose height and stride can lag the window size across a resize — the rows would index past it.
    let rows = win_h.min(dst.len() / dst_stride);
    let w = win_w.min(dst_stride);
    let opaque = crate::paint::OPAQUE_BACKDROP.load(core::sync::atomic::Ordering::Relaxed);
    let min_dark = core::sync::atomic::AtomicU32::new(255);
    crate::par::par_rows(dst, dst_stride, 0, rows, |y, row| {
        let off = y * win_w;
        let src = &scratch[off..off + w];
        let clip = &clip_mask[off..off + w];
        let mut m = 255u32;
        for (i, out) in row[..w].iter_mut().enumerate() {
            let px = src[i];
            let a = (((px >> 24) & 0xFF) * clip[i] as u32 + 127) / 255;
            if a == 0 {
                *out = if opaque { 3 << 30 } else { 0 };
                continue;
            }
            let (d0, d1, d2) = ((px & 0xFF) as usize, ((px >> 8) & 0xFF) as usize, ((px >> 16) & 0xFF) as usize);
            m = m.min(d0 as u32).min(d1 as u32).min(d2 as u32);
            let ch = |d: usize| {
                let c = table.code[d] as u32;
                if a == 255 { c } else { (c * a + 127) / 255 }
            };
            let a2 = if opaque { 3 } else { (a * 3 + 127) / 255 };
            *out = ch(d0) | ch(d1) << 10 | ch(d2) << 20 | a2 << 30;
        }
        min_dark.fetch_min(m, core::sync::atomic::Ordering::Relaxed);
    });
    255 - min_dark.into_inner() as u8
}

/// One row of [`finalize_into_f16`]; returns the row's smallest darkness under a visible pixel.
#[inline]
fn finalize_f16_row(src: &[u32], clip: &[u8], dst: &mut [u64], t: &crate::theme::HdrTable, opaque: bool) -> u32 {
    let one = crate::theme::ALPHA_HALF[255] as u64;
    let mut min_dark = 255u32;
    for i in 0..src.len() {
        let px = src[i];
        let a = (((px >> 24) & 0xFF) * clip[i] as u32 + 127) / 255;
        if a == 0 {
            dst[i] = if opaque { one << 48 } else { 0 };
            continue;
        }
        let (d0, d1, d2) = ((px & 0xFF) as usize, ((px >> 8) & 0xFF) as usize, ((px >> 16) & 0xFF) as usize);
        min_dark = min_dark.min(d0 as u32).min(d1 as u32).min(d2 as u32);
        let (h0, h1, h2) = if a == 255 {
            (t.half[d0], t.half[d1], t.half[d2])
        } else {
            let pm = |d: usize| crate::theme::q16_to_half(((t.q16[d] as u64 * a as u64 + 127) / 255) as u32);
            (pm(d0), pm(d1), pm(d2))
        };
        let alpha = if opaque { one } else { crate::theme::ALPHA_HALF[a as usize] as u64 };
        dst[i] = h0 as u64 | (h1 as u64) << 16 | (h2 as u64) << 32 | alpha << 48;
    }
    min_dark
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
            // sy < sy_max ≤ win_h and sx < sx_max ≤ win_w (the caller's clip), so the index is inside the window-sized scratch and clip mask.
            let scratch_idx = src_row + sx;
            let dst_idx = dst_row + (rect_x + sx as i32) as usize;
            let v = crate::theme::to_sdr_visible(scratch[scratch_idx]);
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

#[cfg(test)]
mod premultiply_contract_tests {
    /// Every presenter takes finalize's output as premultiplied and none premultiplies again; Windows' ULW_ALPHA also requires every channel ≤ α. Exhaustive over α, channel and clip coverage.
    #[test]
    fn finalized_channels_never_exceed_alpha() {
        for clip in [255u8, 200, 128, 1] {
            for a in 0..=255u32 {
                let mut px: Vec<u32> = (0..=255u32).map(|dark| (a << 24) | (dark << 16) | (dark << 8) | dark).collect();
                let mask = vec![clip; px.len()];
                super::finalize_for_os(&mut px, &mask);
                for p in px {
                    let fa = p >> 24;
                    for sh in [16, 8, 0] {
                        assert!((p >> sh) & 0xFF <= fa, "α {a}, clip {clip}: channel {} > α {fa}", (p >> sh) & 0xFF);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod extended_range_tests {
    use crate::theme::{q16_to_half, HdrTable, ALPHA_HALF, WHITE};

    /// The F16 finalize: an opaque pixel is its three table halves with α 1; a half-covered one is premultiplied (≤ its opaque light, α ½); a transparent one is zero; the stride gap is left alone; the returned peak is the brightest code under a visible pixel.
    #[test]
    fn f16_finalize_maps_premultiplies_and_reports_the_peak() {
        let t = HdrTable::new(2.0, 227);
        let glow = 0xFF00_0000 | ((255 - 227) << 16) | ((255 - 227) << 8) | (255 - 227); // code 227 = 2× white
        let scratch = [WHITE, glow, WHITE & 0x00FF_FFFF | 0x8000_0000, 0x0000_0000];
        let clip = [255u8; 4];
        let mut dst = [0xDEAD_BEEFu64; 5];
        let was = crate::paint::OPAQUE_BACKDROP.swap(false, core::sync::atomic::Ordering::Relaxed);
        let peak = super::finalize_into_f16(&scratch, &clip, 4, 1, &mut dst, 5, &t);
        crate::paint::OPAQUE_BACKDROP.store(was, core::sync::atomic::Ordering::Relaxed);
        let ch = |p: u64, i: u32| ((p >> (16 * i)) & 0xFFFF) as u16;
        let d0 = (WHITE & 0xFF) as usize;
        assert_eq!(ch(dst[0], 0), t.half[d0]);
        assert_eq!(ch(dst[0], 3), ALPHA_HALF[255]);
        assert_eq!(ch(dst[1], 0), t.half[255 - 227]);
        assert!(ch(dst[2], 0) < t.half[d0] && ch(dst[2], 0) > 0, "premultiplied below the opaque light");
        assert_eq!(ch(dst[2], 3), ALPHA_HALF[128]);
        assert_eq!(dst[3], 0);
        assert_eq!(dst[4], 0xDEAD_BEEF, "the stride gap is not written");
        assert_eq!(peak, 227);
        assert_eq!(t.half[255 - 227], q16_to_half(t.q16[255 - 227]));
    }
}

#[cfg(test)]
mod ten_bit_tests {
    use crate::theme::{Hdr10Table, WHITE};

    /// The 10-bit finalize: an opaque pixel is its three table codes with α 3, a half-covered one is premultiplied on codes, a transparent one is zero, the stride gap is untouched, and the peak is the brightest visible code.
    #[test]
    fn ten_bit_finalize_packs_premultiplies_and_reports_the_peak() {
        let t = Hdr10Table::new(2.0);
        let glow = 0xFF00_0000 | ((255 - 227) << 16) | ((255 - 227) << 8) | (255 - 227);
        let scratch = [WHITE, glow, WHITE & 0x00FF_FFFF | 0x8000_0000, 0];
        let mut dst = [0xDEAD_BEEFu32; 5];
        let was = crate::paint::OPAQUE_BACKDROP.swap(false, core::sync::atomic::Ordering::Relaxed);
        let peak = super::finalize_into_1010102(&scratch, &[255; 4], 4, 1, &mut dst, 5, &t);
        crate::paint::OPAQUE_BACKDROP.store(was, core::sync::atomic::Ordering::Relaxed);
        let ch = |p: u32, i: u32| (p >> (10 * i)) & 0x3FF;
        let w = t.code[(WHITE & 0xFF) as usize] as u32;
        assert_eq!((ch(dst[0], 0), dst[0] >> 30), (w, 3));
        assert_eq!(ch(dst[1], 0), t.code[255 - 227] as u32);
        assert_eq!(ch(dst[2], 0), (w * 128 + 127) / 255);
        assert_eq!(dst[2] >> 30, 2);
        assert_eq!(dst[3], 0);
        assert_eq!(dst[4], 0xDEAD_BEEF);
        assert_eq!(peak, 227);
    }
}
