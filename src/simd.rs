//! SIMD lane re-exports, gated on the `simd` feature. Currently scoped to `u32x8` because that's all the hot path needs — the finalize and opaque-scan paths in [`crate::paint::finalize`] operate directly on `u32x8`-packed pixels via lane-wide bitops, no per-channel unpack/repack required. When a future pass needs other widths (f32x8 for coverage math, etc.) bring them in alongside.
//!
//! On x86_64 with AVX2 enabled at compile time, `wide::u32x8` lowers to single 256-bit ops (`vpxor`, `vpmullw`, etc.). Without AVX2 (pre-Haswell), it emits two 128-bit SSE ops — still better than scalar but half the throughput. The fluor convention is to assume AVX2 on x86_64 and let pre-Haswell users either rebuild with `-C target-cpu=native` (gets AVX2 if present) or take the SSE fallback transparently. With `simd` off, this module is empty and callers cfg-gate their SIMD code paths back to the scalar primitives in [`crate::pixel`] / [`crate::paint`].

#![cfg(feature = "simd")]

pub use wide::u32x8;

/// SIMD lane count used by every hot-path vectorized op in fluor.
///
/// 8 is the sweet spot: * Fits in one AVX2 register (256 bits = 8 × f32 or 8 × u32).
/// * Degrades cleanly to two 128-bit SSE ops on pre-AVX2 hardware via `wide`'s internal split.
/// * Matches typical AA-band width (1-2 pixels) so under-utilization at the boundary is rare.
/// * One cache line of u32 pixels = 16, so each cache line holds exactly 2 lanes — clean sequential prefetch behavior with no straddling.
pub const LANES: usize = 8;
