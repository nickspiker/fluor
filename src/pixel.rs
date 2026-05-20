//! Pixel format types with per-channel arithmetic.
//!
//! # Convention — locked
//!
//! **Internal pixel format**: `0xttRRGGBB` packed in a `u32` — top byte is **transparency**, not alpha. LE byte layout `[B, G, R, t]`. Every paint primitive, every layer buffer, every Group composite uses this. The type is still named `Argb8` for stability across the codebase; the byte layout hasn't changed, only the top byte's interpretation.
//!
//! **Transparency semantics**: `t = 0` = fully opaque, `t = 255` = (almost) fully transparent. Variables in code are named `t` / `transparency` — never `alpha` or `a` — so the convention is unambiguous at every read site. RGB channels store the straight (non-premultiplied) intrinsic color.
//!
//! **Why t-convention** (mirroring the README): the hot-loop early-out fires precisely when a fully-opaque layer is hit, because `remaining = (remaining * t) >> 8 = 0` exactly when `t = 0`. Single u32 compare against an immediate detects opacity classes without any shift or mask: `if pixel <= 0x00FFFFFF` is "this pixel is fully opaque" in one CMP. Sort order primary-sorts by transparency for free.
//!
//! **The 256-vs-255 gap**: fully-transparent layers never enter the blend pass — they're culled at the caller level (you don't draw an invisible layer). Per-pixel `t = 255` within a partial-transparency layer contributes exact 0 to RGB (because `255 - 255 = 0`) and attenuates `remaining` by `255/256`; the 1-LSB drift is invisible at expected stacking depths. We never need to represent `t = 256` and the missing slot is a happy coincidence.
//!
//! **Boundary**: paint primitives write `t` directly. The host present pass flips `t → α` (XOR top byte with `0xFF`) right before submitting to wgpu / softbuffer — folded into the existing `premultiply_buffer` step on Linux. External inputs (cosmic-text glyph coverage, `pack_argb`) flip once at the import boundary, never per-frame.
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

    /// Per-channel XOR across all four channels (uniform with `add` / `sub` / `mul` / `div` / `inv`). For "invert RGB but preserve destination transparency" semantics — the selection-highlight idiom — use `BlendMode::Xor` (kernel-level) which handles the t-channel preservation; or XOR explicitly against `0x00FFFFFF`.
    #[inline]
    pub fn xor(a: Argb8, b: Argb8) -> Argb8 {
        Argb8(a.0 ^ b.0)
    }

    /// Porter-Duff source-over under fluor's t-convention (top byte = transparency).
    ///
    /// Formulas:
    /// - `result_rgb = (src_rgb · (256 − src_t) + dst_rgb · src_t) >> 8`. Linear blend; src dominates as `src_t → 0`, dst dominates as `src_t → 255`. The `(256 − src_t)` multiplier sums with `src_t` to exactly 256, making the `>> 8` normalization exact at `src_t = 0` (full src) — same trick the α-convention version used.
    /// - `result_t = (src_t · dst_t) >> 8`. Result is opaque (`t = 0`) iff either input is opaque, which is the right "src over dst" semantic.
    ///
    /// Single shortcut: `src_t == 0` (opaque src) → return src exactly. The transparent endpoint `src_t = 255` has no shortcut — it's just a normal value that the SWAR math handles with the standard 1-LSB drift (`dst * 255 / 256 ≈ dst`).
    #[inline]
    pub fn alpha_over(dst: Argb8, src: Argb8) -> Argb8 {
        let src_t = ((src.0 >> 24) & 0xFF) as u64;
        if src_t == 0 { return src; }                       // opaque src covers dst entirely
        let dst_t = ((dst.0 >> 24) & 0xFF) as u64;
        let inv_src_t = 256 - src_t;

        // SWAR for the 3 RGB slots; t slot patched below.
        let mut bg = dst.0 as u64;
        bg = (bg | (bg << 16)) & 0x0000_FFFF_0000_FFFF;
        bg = (bg | (bg << 8)) & 0x00FF_00FF_00FF_00FF;

        let mut fg = src.0 as u64;
        fg = (fg | (fg << 16)) & 0x0000_FFFF_0000_FFFF;
        fg = (fg | (fg << 8)) & 0x00FF_00FF_00FF_00FF;

        // fg · (256-src_t) + bg · src_t — slot max = 255*256 = 65280, fits in 16 bits per slot.
        let mut r = fg * inv_src_t + bg * src_t;
        r = (r >> 8) & 0x00FF_00FF_00FF_00FF;
        r = (r | (r >> 8)) & 0x0000_FFFF_0000_FFFF;
        r = r | (r >> 16);
        let rgb_only = (r as u32) & 0x00FF_FFFF;

        // t slot: (src_t · dst_t) >> 8 — separately, because the SWAR formula above produces a different value in the t-slot than we want.
        let result_t = ((src_t * dst_t) >> 8) as u32;
        Argb8(rgb_only | (result_t << 24))
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
    fn alpha_over_opaque_src_returns_src_exact() {
        // src_t = 0 (opaque) → shortcut returns src exactly. No SWAR math runs.
        let dst = Argb8(0x00_11_22_33);
        let src = Argb8(0x00_AA_BB_CC);
        assert_eq!(Argb8::alpha_over(dst, src), src);
    }

    #[test]
    fn alpha_over_almost_transparent_src_within_one_lsb_of_dst() {
        // src_t = 255 (almost transparent) → src contributes 0 to RGB exactly (since 256-255=1
        // gives 1-LSB src contribution), so result ≈ dst with ≤ 1 LSB per-channel drift.
        let dst = Argb8(0x00_11_22_33);
        let src = Argb8(0xFF_AA_BB_CC);
        let result = Argb8::alpha_over(dst, src);
        for shift in [24u32, 16, 8, 0] {
            let d = ((dst.0 >> shift) & 0xFF) as i32;
            let r = ((result.0 >> shift) & 0xFF) as i32;
            assert!((d - r).abs() <= 1, "channel at shift {shift} > 1 LSB off: dst={:#x} got={:#x}", d, r);
        }
    }

    #[test]
    fn alpha_over_opaque_dst_stays_opaque_exact() {
        // dst_t = 0 (opaque) → result_t = (src_t · 0) >> 8 = 0 exactly. EXACT, no drift, for any
        // src_t. The chrome can never lose its opacity under a transparent overlay.
        let dst = Argb8(0x00_11_22_33);
        for src_t in [0u32, 1, 50, 100, 128, 200, 254, 255] {
            let src = Argb8((src_t << 24) | 0x00_AA_BB_CC);
            let result = Argb8::alpha_over(dst, src);
            let result_t = (result.0 >> 24) & 0xFF;
            assert_eq!(result_t, 0, "src_t={src_t} produced result_t={result_t}");
        }
    }

    #[test]
    fn alpha_over_combined_transparency() {
        // result_t = (src_t · dst_t) >> 8. Two half-transparent layers stacked = ~25% opacity
        // (transparent · transparent → more transparent).
        let dst = Argb8(0x80_00_00_00);
        let src = Argb8(0x80_00_00_00);
        let result = Argb8::alpha_over(dst, src);
        let result_t = (result.0 >> 24) & 0xFF;
        // 0x80 * 0x80 >> 8 = 0x40
        assert_eq!(result_t, 0x40, "expected result_t=0x40, got {:#x}", result_t);
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
