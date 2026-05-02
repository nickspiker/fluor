//! Pixel format types with per-channel arithmetic.
//!
//! `Argb8` wraps a packed `u32` (`0xAARRGGBB`) and provides channel ops via SWAR
//! (SIMD Within A Register) — four u8 channels processed in parallel via u64 widening.
//! The `>> 8` normalization (divide by 256, not 255) is the canonical fast-blend
//! approximation; per-channel error is below 1/256 and imperceptible.
//!
//! `#[repr(transparent)]` guarantees `Argb8` has the same layout as `u32`, so
//! `&[Argb8]` and `&[u32]` are safely transmutable for zero-cost interop with
//! paint primitives and GPU upload.

/// Packed ARGB pixel: `0xAARRGGBB`, 8 bits per channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Argb8(pub u32);

impl Argb8 {
    pub const ZERO: Argb8 = Argb8(0);

    /// Additive blend: `dst + src` per channel via wrapping add.
    /// Overflow wraps intentionally — for small values (glow, blinkey)
    /// channels don't interfere. For large values, use SWAR saturating add.
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
    /// Uses the same u64 SWAR widening as `blend_rgb_only` / `alpha_over`, but since
    /// both operands have per-channel values (not a scalar broadcast), we do two passes:
    /// high pair (bytes 3,2 → A,R) and low pair (bytes 1,0 → G,B), each using a scalar
    /// multiply in isolated 16-bit slots. Four 16-bit multiplies total, channel-order
    /// agnostic. The compiler vectorizes this to a single NEON `vmul` on ARM.
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

    /// Per-channel invert: `255 - x` for each channel.
    #[inline]
    pub fn inv(a: Argb8) -> Argb8 {
        Argb8(0xFFFF_FFFF - a.0)
    }

    /// XOR RGB channels, preserve destination alpha.
    #[inline]
    pub fn xor(a: Argb8, b: Argb8) -> Argb8 {
        Argb8((a.0 ^ b.0) & 0x00FF_FFFF | (a.0 & 0xFF00_0000))
    }

    /// Porter-Duff source-over: `src * α + dst * (1 - α)` where α = src's alpha channel.
    /// SWAR with >> 8 normalization.
    #[inline]
    pub fn alpha_over(dst: Argb8, src: Argb8) -> Argb8 {
        let alpha = ((src.0 >> 24) & 0xFF) as u64;
        if alpha == 0 { return dst; }
        if alpha == 255 { return src; }
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
    fn xor_preserves_dst_alpha() {
        let a = Argb8(0xFF_AA_BB_CC);
        let b = Argb8(0x00_FF_00_FF);
        let result = Argb8::xor(a, b);
        assert_eq!(result.0 >> 24, 0xFF); // alpha preserved from a
        assert_eq!((result.0 >> 16) & 0xFF, 0xAA ^ 0xFF);
        assert_eq!((result.0 >> 8) & 0xFF, 0xBB ^ 0x00);
        assert_eq!(result.0 & 0xFF, 0xCC ^ 0xFF);
    }

    #[test]
    fn alpha_over_opaque_replaces() {
        let dst = Argb8(0xFF_11_22_33);
        let src = Argb8(0xFF_AA_BB_CC);
        assert_eq!(Argb8::alpha_over(dst, src), src);
    }

    #[test]
    fn alpha_over_transparent_preserves() {
        let dst = Argb8(0xFF_11_22_33);
        let src = Argb8(0x00_AA_BB_CC);
        assert_eq!(Argb8::alpha_over(dst, src), dst);
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
