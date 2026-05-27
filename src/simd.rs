//! SIMD types and pixel pack/unpack helpers, gated on the `simd` feature.
//!
//! When `simd` is on, this module re-exports `wide`'s f32 / u32 lane types and provides
//! ARGB-pixel pack/unpack helpers built on top. When `simd` is off, the module is empty —
//! callers cfg-gate their SIMD code paths and fall back to the scalar primitives in
//! [`crate::pixel`] / [`crate::paint`].
//!
//! ## Lane width
//!
//! [`LANES`] = 8 everywhere we vectorize: `u32x8` packed pixels, `f32x8` coverage math. On
//! x86_64 with AVX2 enabled at compile time, `wide`'s 8-wide types lower to single 256-bit
//! ops (`vpxor`, `vpmullw`, `vfmadd…`, `vrsqrtps`). Without AVX2 (e.g., pre-Haswell or `-march`
//! restricted), they emit two 128-bit SSE ops apiece — still better than scalar but half the
//! throughput. The fluor convention is to assume AVX2 on x86_64 and let pre-Haswell users
//! either rebuild with `-C target-cpu=native` (gets AVX2 if present) or take the SSE fallback
//! transparently.
//!
//! ## Pack / unpack
//!
//! Packed `u32x8` ARGB → 4 lanes of `u32x8` (α, R, G, B), each holding `u32` values in `0..=255`
//! sitting in the low byte of each lane. The math (consumed, contrib, etc) all happens in
//! lane-wide `u32x8` arithmetic — no bit-twiddling between lanes — and the result is packed
//! back into a `u32x8`. This is the canonical shape for SIMD-izing [`crate::pixel::Blend::under`]
//! and the boundary passes.

#![cfg(feature = "simd")]

pub use wide::{f32x4, f32x8, i32x4, i32x8, u32x4, u32x8};

/// SIMD lane count used by every hot-path vectorized op in fluor.
///
/// 8 is the sweet spot:
/// * Fits in one AVX2 register (256 bits = 8 × f32 or 8 × u32).
/// * Degrades cleanly to two 128-bit SSE ops on pre-AVX2 hardware via `wide`'s internal split.
/// * Matches typical AA-band width (1-2 pixels) so under-utilization at the boundary is rare.
/// * One cache line of u32 pixels = 16, so each cache line holds exactly 2 lanes — clean
///   sequential prefetch behavior with no straddling.
pub const LANES: usize = 8;

/// Unpack a `u32x8` of packed ARGB pixels (`0xAARRGGBB`) into 4 separate channel lanes, each
/// `u32x8` holding values `0..=255` in the low byte of each lane. Inverse of [`pack_argb_x8`].
///
/// Uses lane-wide right-shift + mask — single ops on AVX2 (`vpsrld`, `vpand`).
#[inline]
pub fn unpack_argb_x8(pix: u32x8) -> (u32x8, u32x8, u32x8, u32x8) {
    let mask = u32x8::splat(0xFF);
    let a = (pix >> 24) & mask;
    let r = (pix >> 16) & mask;
    let g = (pix >> 8) & mask;
    let b = pix & mask;
    (a, r, g, b)
}

/// Pack 4 channel lanes back into a `u32x8` of `0xAARRGGBB` pixels. Inverse of [`unpack_argb_x8`].
/// Caller is responsible for ensuring each lane fits in `0..=255` (the under-math invariants
/// guarantee this; see [`crate::pixel::Blend::under`]'s overflow proof).
#[inline]
pub fn pack_argb_x8(a: u32x8, r: u32x8, g: u32x8, b: u32x8) -> u32x8 {
    (a << 24) | (r << 16) | (g << 8) | b
}
