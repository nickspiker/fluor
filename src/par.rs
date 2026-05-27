//! Row-parallel iteration over a pixel buffer.
//!
//! Single entry point [`par_rows`]: takes a contiguous `width × height` `u32` pixel buffer, a
//! row range (clip-aware), and a closure invoked per row. With the `rayon` feature on, rows are
//! dispatched across the rayon thread pool; off, the same iteration runs sequentially. The
//! closure signature is identical either way — rasterizers and boundary passes don't branch on
//! the feature gate, they just call this wrapper and the right thing happens.
//!
//! ## Why `Send + Sync` on the closure
//!
//! When rayon is on, rows run on different worker threads and the closure must be safe to call
//! from multiple threads concurrently — that's `Send + Sync`. Rasterizer closures are tiny
//! captures of small `f32` / `u32` / packed-pixel parameters with no interior mutability, so
//! the bound is satisfied trivially. We require it unconditionally (even in sequential builds)
//! so the call-site code is identical across feature combos — no `#[cfg]` per call.

/// Iterate `pixels` row-by-row over `row_start..row_end`, calling `f(row_index, &mut row_slice)`
/// for each row. Each `row_slice` is exactly `width` `u32`s. With `rayon` enabled, rows are
/// dispatched in parallel; without it, sequentially.
///
/// Pre-conditions: `pixels.len() >= row_end * width` and `row_start <= row_end`. No internal
/// clamping — the caller (a clipped rasterizer or a boundary pass) is responsible for shaping
/// the range to fit the buffer.
#[inline]
pub fn par_rows<F>(pixels: &mut [u32], width: usize, row_start: usize, row_end: usize, f: F)
where
    F: Fn(usize, &mut [u32]) + Send + Sync,
{
    if row_start >= row_end || width == 0 {
        return;
    }
    let sub = &mut pixels[row_start * width..row_end * width];
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        sub.par_chunks_mut(width).enumerate().for_each(|(i, row)| {
            f(row_start + i, row);
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, row) in sub.chunks_mut(width).enumerate() {
            f(row_start + i, row);
        }
    }
}

/// Iterate a flat `pixels` slice in equal-sized chunks of `chunk_len` (handles trailing tail of
/// any length). Closure receives `(chunk_offset_in_pixels, &mut chunk_slice)`. Same Rayon-vs-
/// sequential dispatch as [`par_rows`] — used for whole-buffer passes (finalize, flatten,
/// Op::Under) where row geometry doesn't matter and chunking by a multiple of cache-line size
/// gives the best throughput.
#[inline]
pub fn par_chunks<F>(pixels: &mut [u32], chunk_len: usize, f: F)
where
    F: Fn(usize, &mut [u32]) + Send + Sync,
{
    if chunk_len == 0 || pixels.is_empty() {
        return;
    }
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        pixels
            .par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, chunk)| {
                f(i * chunk_len, chunk);
            });
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (i, chunk) in pixels.chunks_mut(chunk_len).enumerate() {
            f(i * chunk_len, chunk);
        }
    }
}
