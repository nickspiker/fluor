//! Pixel format types with per-channel arithmetic.
//!
//! `Argb8` wraps a packed `u32` (`0xAARRGGBB`) and provides channel ops via SWAR (SIMD Within A Register) — four u8 channels processed in parallel via u64 widening. The `>> 8` normalization (divide by 256, not 255) is the canonical fast-blend approximation; per-channel error is below 1/256 and imperceptible.
//!
//! `#[repr(transparent)]` guarantees `Argb8` has the same layout as `u32`, so `&[Argb8]` and `&[u32]` are safely transmutable for zero-cost interop with paint primitives and GPU upload.

/// Packed ARGB pixel: `0xAARRGGBB`, 8 bits per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Argb8(pub u32);

impl Argb8 {
    pub const ZERO: Argb8 = Argb8(0);

    /// Additive blend: `dst + src` per channel via wrapping add. Overflow wraps intentionally — for small values (glow, blinkey) channels don't interfere. For large values, use SWAR saturating add.
    #[inline]
    pub fn add(a: Argb8, b: Argb8) -> Argb8 {
        Argb8(a.0.wrapping_add(b.0))
    }

    /// Subtractive: `dst - src` per channel via wrapping sub.
    #[inline]
    pub fn sub(a: Argb8, b: Argb8) -> Argb8 {
        Argb8(a.0.wrapping_sub(b.0))
    }

    /// Channel multiply: `(a * b) >> 8` per channel.
    ///
    /// Two packed vectors can't share the SWAR `widened * scalar` trick used by `alpha_over` (slot products would carry into neighbouring slots). Instead we extract each channel pair, do four independent `(u32 × u32) >> 8` ops in isolated registers, and pack back. LLVM recognises the pattern and emits one NEON `vmul.i16` on aarch64 (or `pmullw` on x86) — true four-lane SIMD via auto-vectorization, no platform intrinsics, channel-order agnostic.
    #[inline]
    pub fn mul(a: Argb8, b: Argb8) -> Argb8 {
        // High pair: A and R channels
        let a_hi = ((a.0 >> 24) & 0xFF, (a.0 >> 16) & 0xFF);
        let b_hi = ((b.0 >> 24) & 0xFF, (b.0 >> 16) & 0xFF);
        let ra = (a_hi.0 * b_hi.0) >> 8;
        let rr = (a_hi.1 * b_hi.1) >> 8;

        // Low pair: G and B channels
        let a_lo = ((a.0 >> 8) & 0xFF, a.0 & 0xFF);
        let b_lo = ((b.0 >> 8) & 0xFF, b.0 & 0xFF);
        let rg = (a_lo.0 * b_lo.0) >> 8;
        let rb = (a_lo.1 * b_lo.1) >> 8;

        Argb8((ra << 24) | (rr << 16) | (rg << 8) | rb)
    }

    /// Channel divide: `(a << 8) / b` per channel via integer Euclidean division. Saturates at 0xFF when the result would exceed channel range (small `b`) or `b == 0`. Bit-exact, deterministic, no IEEE. Inverse of [`mul`](Self::mul) up to ±1 LSB per channel (mul truncates the low byte; div recovers the high byte). Slow path: aarch64 NEON has no SIMD integer divide, so this is four scalar `udiv`s per pixel — completeness op for un-premultiply / calibration / exact ratios, not for hot loops.
    ///
    /// **Branchless divide-by-zero handling:** when `b == 0`, the numerator is forced to `0xFF` and the denominator to `1`, so the divide always runs and the channel saturates to `0xFF`. Style match with `mul` / `alpha_over` (no branches).
    ///
    /// **Rule 0 — WHY/PROOF/PREVENTS:** WHY: `(255 << 8) / 1 = 65280` exceeds u8 channel range; raw cast wraps to 0x00. PROOF: max numerator = 255 << 8 = 65280; max valid u8 = 255. PREVENTS: wraparound producing nonsense channel values. The `min(0xFF)` is the saturation semantic of channel arithmetic, not a safety clamp.
    #[inline]
    pub fn div(a: Argb8, b: Argb8) -> Argb8 {
        let ch = |a: u32, b: u32| -> u32 {
            // mask = 1 if b == 0, 0 otherwise. `numer = a | (0xFF * mask)` becomes 0xFF when b == 0
            // (a is already ≤ 0xFF). `denom = b + mask` becomes 1 when b == 0. Result then saturates to 0xFF.
            let mask = (b == 0) as u32;
            let numer = a | (0xFF * mask);
            let denom = b + mask;
            ((numer << 8) / denom).min(0xFF)
        };
        let ra = ch((a.0 >> 24) & 0xFF, (b.0 >> 24) & 0xFF);
        let rr = ch((a.0 >> 16) & 0xFF, (b.0 >> 16) & 0xFF);
        let rg = ch((a.0 >>  8) & 0xFF, (b.0 >>  8) & 0xFF);
        let rb = ch( a.0        & 0xFF,  b.0        & 0xFF);
        Argb8((ra << 24) | (rr << 16) | (rg << 8) | rb)
    }

    /// Per-channel invert: `255 - x` for each channel.
    #[inline]
    pub fn inv(a: Argb8) -> Argb8 {
        Argb8(0xFFFF_FFFF - a.0)
    }

    /// Per-channel XOR across all four channels (uniform with `add` / `sub` / `mul` / `div` / `inv`). For "invert RGB but preserve destination alpha" semantics — the selection-highlight idiom — use `BlendMode::Xor` (kernel-level) which handles the alpha preservation; or XOR explicitly against `0x00FFFFFF`.
    #[inline]
    pub fn xor(a: Argb8, b: Argb8) -> Argb8 {
        Argb8(a.0 ^ b.0)
    }

    /// Porter-Duff source-over: `src * α + dst * (1 - α)` where α = src's alpha channel. SWAR with `>> 8` normalization (divide by 256, not 255).
    ///
    /// **Branchless and uniform across all α values.** α = 0 gives exact `dst` from the math (`bg * 256 + 0 = 256 * bg`, then `>> 8 = bg`). α = 255 gives `0.996·src + 0.004·dst` — *not* exact src, by design: the `>> 8` shortcut produces ≤1 LSB error per channel across the whole α range, and special-casing 255 to return src exactly would create a discontinuity at exactly one α value while leaving the rest noisy. Same trade-off `mul` makes. Callers that need a "fully transparent → skip work" optimization should short-circuit at the kernel level (see `paint::flatten_alpha_over`) where it composes with the per-pixel test for free.
    #[inline]
    pub fn alpha_over(dst: Argb8, src: Argb8) -> Argb8 {
        let alpha = ((src.0 >> 24) & 0xFF) as u64;
        let inv = 256 - alpha;

        let mut bg = dst.0 as u64;
        bg = (bg | (bg << 16)) & 0x0000_FFFF_0000_FFFF;
        bg = (bg | (bg << 8)) & 0x00FF_00FF_00FF_00FF;

        let mut fg = src.0 as u64;
        fg = (fg | (fg << 16)) & 0x0000_FFFF_0000_FFFF;
        fg = (fg | (fg << 8)) & 0x00FF_00FF_00FF_00FF;

        let mut r = bg * inv + fg * alpha;
        r = (r >> 8) & 0x00FF_00FF_00FF_00FF;
        r = (r | (r >> 8)) & 0x0000_FFFF_0000_FFFF;
        r = r | (r >> 16);
        Argb8(r as u32)
    }

    /// Screen: `MAX - (MAX - a) * (MAX - b) >> 8` per channel.
    #[inline]
    pub fn screen(a: Argb8, b: Argb8) -> Argb8 {
        // screen(a, b) = inv(mul(inv(a), inv(b)))
        Argb8::inv(Argb8::mul(Argb8::inv(a), Argb8::inv(b)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_small_values() {
        let a = Argb8(0xFF_10_20_30);
        let b = Argb8(0x00_01_02_03);
        assert_eq!(Argb8::add(a, b), Argb8(0xFF_11_22_33));
    }

    #[test]
    fn sub_reverses_add() {
        let a = Argb8(0xFF_10_20_30);
        let b = Argb8(0x00_01_02_03);
        let sum = Argb8::add(a, b);
        assert_eq!(Argb8::sub(sum, b), a);
    }

    #[test]
    fn mul_by_white_is_identity() {
        let a = Argb8(0xFF_80_40_C0);
        let white = Argb8(0xFF_FF_FF_FF);
        let result = Argb8::mul(a, white);
        // >>8 approximation: each channel = ch * 255 >> 8 ≈ ch - 1 (off by at most 1)
        let r = result.0;
        assert!(((r >> 16) & 0xFF) >= 0x7F && ((r >> 16) & 0xFF) <= 0x80);
    }

    #[test]
    fn mul_by_zero_is_zero() {
        let a = Argb8(0xFF_FF_FF_FF);
        let zero = Argb8(0x00_00_00_00);
        assert_eq!(Argb8::mul(a, zero), Argb8(0));
    }

    #[test]
    fn inv_of_zero_is_max() {
        assert_eq!(Argb8::inv(Argb8(0x00_00_00_00)), Argb8(0xFF_FF_FF_FF));
    }

    #[test]
    fn inv_of_max_is_zero() {
        assert_eq!(Argb8::inv(Argb8(0xFF_FF_FF_FF)), Argb8(0x00_00_00_00));
    }

    #[test]
    fn xor_is_uniform_four_channel() {
        let a = Argb8(0xFF_AA_BB_CC);
        let b = Argb8(0x12_FF_00_FF);
        let result = Argb8::xor(a, b);
        // All four channels XOR independently — no alpha special case.
        assert_eq!(result.0 >> 24, 0xFF ^ 0x12);
        assert_eq!((result.0 >> 16) & 0xFF, 0xAA ^ 0xFF);
        assert_eq!((result.0 >> 8) & 0xFF, 0xBB ^ 0x00);
        assert_eq!(result.0 & 0xFF, 0xCC ^ 0xFF);
    }

    #[test]
    fn xor_self_inverse() {
        // a ^ b ^ b == a (uniform XOR is its own inverse).
        let a = Argb8(0xDE_AD_BE_EF);
        let b = Argb8(0xCA_FE_BA_BE);
        assert_eq!(Argb8::xor(Argb8::xor(a, b), b), a);
    }

    #[test]
    fn alpha_over_opaque_within_one_lsb_of_src() {
        // α = 255 gives `0.996·src + 0.004·dst` under >>8 — each channel within 1 LSB of src.
        let dst = Argb8(0xFF_11_22_33);
        let src = Argb8(0xFF_AA_BB_CC);
        let result = Argb8::alpha_over(dst, src);
        for shift in [24, 16, 8, 0] {
            let s = ((src.0 >> shift) & 0xFF) as i32;
            let r = ((result.0 >> shift) & 0xFF) as i32;
            assert!((s - r).abs() <= 1, "channel at shift {} > 1 LSB off: src={:#x} got={:#x}", shift, s, r);
        }
    }

    #[test]
    fn alpha_over_transparent_preserves_dst_exact() {
        // α = 0: math gives `bg·256 + fg·0`, then `>>8 = bg`. Exact, no LSB error.
        let dst = Argb8(0xFF_11_22_33);
        let src = Argb8(0x00_AA_BB_CC);
        assert_eq!(Argb8::alpha_over(dst, src), dst);
    }

    #[test]
    fn div_by_one_scales_to_full() {
        // (a << 8) / 1 = a * 256, saturated to 0xFF per channel.
        let a = Argb8(0xFF_01_02_03);
        let one = Argb8(0x01_01_01_01);
        let result = Argb8::div(a, one);
        // Every channel >= 1 → result.channel = (ch * 256) min 255 = 255.
        assert_eq!(result, Argb8(0xFF_FF_FF_FF));
    }

    #[test]
    fn div_by_zero_saturates_to_max() {
        let a = Argb8(0xFF_80_40_C0);
        let zero = Argb8(0x00_00_00_00);
        assert_eq!(Argb8::div(a, zero), Argb8(0xFF_FF_FF_FF));
    }

    #[test]
    fn div_round_trips_mul_within_one_lsb() {
        // mul truncates the low byte; div recovers the high byte. Round trip is exact at multiples
        // of 256/b, off by at most 1 LSB elsewhere — Argb8::mul does (a*b)>>8, then div does ((a*b)>>8 << 8) / b which differs from a only by the truncation (a*b mod b)/b range.
        let a = Argb8(0xFF_80_40_C0);
        let b = Argb8(0xFF_FF_80_C0);
        let mid = Argb8::mul(a, b);
        let recovered = Argb8::div(mid, b);
        for shift in [24, 16, 8, 0] {
            let orig = ((a.0 >> shift) & 0xFF) as i32;
            let got = ((recovered.0 >> shift) & 0xFF) as i32;
            assert!((orig - got).abs() <= 1, "channel at shift {} off by more than 1: orig={:#x} got={:#x}", shift, orig, got);
        }
    }

    #[test]
    fn screen_of_zero_is_identity() {
        let a = Argb8(0xFF_80_40_C0);
        let zero = Argb8(0x00_00_00_00);
        // screen(a, 0) = inv(mul(inv(a), inv(0))) = inv(mul(inv(a), FF)) ≈ inv(inv(a)) ≈ a
        let result = Argb8::screen(a, zero);
        // May be off by 1 due to >>8 approximation
        let diff = (a.0 as i64 - result.0 as i64).unsigned_abs();
        assert!(diff <= 0x01_01_01_01);
    }
}
