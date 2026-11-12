//! The ferros display's working values from a fluor scratch buffer: the present the ferros host runs, at either pixel depth.
//! The display composes 16-bit surfaces (`Px` u64: R, G, B in bits 0, 16 and 32 as gamma 2 with white at 2¹⁴; bit 63 = see-through), so every depth's present ends in the same place: each channel's `g2t` code to `16384 × atanh(code / 2ⁿ)`, VSF's decode at the bucket's bottom (light = atanh(code / 2ⁿ)², and gamma 2 is its root).
//! A pixel's α composites it over black in darkness space: fluor's darkness is premultiplied (`under` deposits darkness × α, so darkness ≤ α), what is still transparent is white until something is put under it, and black under it adds darkness `top − α`; so the visible code is `α − darkness`. (Until 2026-11-07 the darkness was taken as straight, `(top − darkness) × α`, right only where α is whole: every translucent edge came out too bright.)
//! At 16 bits the display reads the pixels as fluor drew them (owner, 2026-10-29: fluor the one compositor, passed straight through): the window is shown with format `G2T16` (op 22's last byte) and the display decodes each pixel itself (ferros payloads/display `working`, the composite and table below), so the 16-bit present is a copy. The 8-bit present still converts here.

//! A display that bakes (ferros COMPOSITOR.md "Baked", 2026-11-12; owner: "if the GPU touches it last and we precompute the tanh highlight colours for all elements then it's a straight thru render") takes a third format, `BAKED`: each pixel already the panel's own codes, so the display only dithers and packs it. Then the present here is the bake (`Bake`, its parameters the display's op 65), for both depths: fluor's code over black, its light by the display's roll-off, the panel's matrix and gamma, the code in Q16. A display without op 65 gets the formats above, as before.

use crate::pixel::{Argb8, G2t16, Pixel};

/// The display's see-through bit: a pixel the window does not cover, so whatever is behind it shows.
pub const SEE_THROUGH: u64 = 1 << 63;

/// The surface format of pixels already in the panel's codes (ferros `window::BAKED`).
pub const BAKED: u8 = 2;

/// A pixel depth the ferros display can be fed: one scratch row into a row of the surface, in the format the window is shown with.
pub trait PanelPixel: Pixel {
    /// The surface format op 22 names (ferros `window::WORKING` 0, `window::G2T16` 1) when the display does not bake.
    const FORMAT: u8;
    /// `dst[i]` from `src[i]`; the slices are the same length.
    fn to_panel(src: &[Self], dst: &mut [u64]);
    /// `dst[i]` from `src[i]` baked (`BAKED`): each channel's code over black, α − darkness, its light by `bake`'s table, then the panel's matrix and code (`Bake::row`). The transparency in bits 49-62 (Q14), exactly the see-through bit where α is 0, as the display's own decode of a fluor surface has it.
    fn to_baked(src: &[Self], dst: &mut [u64], bake: &Bake);
}

/// The bake a ferros display's `BAKED` surface is made with (display op 65, ferros_sys `window::bake`): per channel of a working value v (VSF RGB, gamma 2, white at 1), the light l = tanh²(k·v); the panel's light p = m·l (its rows the panel's red, green and blue); its code p^g × 65472 in Q16, held at the field's ends. A fluor code c (g2t, light atanh(c / 2ⁿ)²) is the working value atanh(c / 2ⁿ), so its light is tanh²(k · atanh(c / 2¹⁶)): one table over every 16-bit code (an 8-bit code is its 16-bit `c × 2⁸`). The panel's part is the display's own NEON kernels (ferros payloads/display neon.rs `matrix8`, `code16`, odt.rs `odt_tables`), carried here: fluor does not depend on ferros.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub struct Bake {
    /// The display's generation of this bake: a brightness change moves it (the window's event `EV_BAKE`).
    pub generation: u32,
    pub k: f32,
    pub m: [[f32; 3]; 3],
    pub g: f32,
    /// Every 16-bit g2t code's light, f16 bits, the E peak at 2¹⁴.
    light: alloc::vec::Vec<u16>,
    /// The panel's tables (the display's `Odt` less its working-value light): the gains + K as f16 bits, and the code's per-exponent scale and per-mantissa nudge and slope.
    c0: [u16; 8],
    c1: [u16; 8],
    nudge_hi: [u8; 64],
    nudge_lo: [u8; 64],
    slope: [u8; 64],
    scale_hi: [u8; 32],
    scale_lo: [u8; 32],
}

/// f32 to f16 bits, rounded to nearest even; the subnormals and past the range as IEEE has them. Portable: built once a bake.
fn f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = (b >> 16 & 0x8000) as u16;
    let e = (b >> 23 & 0xFF) as i32 - 127 + 15;
    let m = b & 0x7F_FFFF;
    if e >= 31 {
        return sign | 0x7C00;
    }
    if e <= 0 {
        // Subnormal (or zero): the mantissa with its hidden bit, shifted down, rounded.
        if e < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let s = (14 - e) as u32;
        let r = (m >> s) + ((m >> (s - 1)) & 1);
        return sign | r as u16;
    }
    let r = (e as u32) << 10 | m >> 13;
    // Round to nearest, ties to even; a carry into the exponent is the next binade, exactly.
    let rest = m & 0x1FFF;
    let r = r + (rest > 0x1000 || (rest == 0x1000 && r & 1 == 1)) as u32;
    sign | r as u16
}

impl Bake {
    /// From the display's op 65 reply: [0][generation u32][k f32][m 9 × f32][g f32], little-endian. None for anything else (a display without the op answers otherwise).
    pub fn from_reply(r: &[u8]) -> Option<Bake> {
        if r.len() < 49 || r[0] != 0 {
            return None;
        }
        let f = |i: usize| f32::from_le_bytes([r[i], r[i + 1], r[i + 2], r[i + 3]]);
        Some(Bake::new(u32::from_le_bytes([r[1], r[2], r[3], r[4]]), f(5), core::array::from_fn(|i| core::array::from_fn(|j| f(9 + 4 * (3 * i + j)))), f(45)))
    }

    /// The tables for these parameters (about 65536 tanh and atanh: a few ms, once a bake).
    pub fn new(generation: u32, k: f32, m: [[f32; 3]; 3], g: f32) -> Bake {
        let light = (0..1u32 << 16)
            .map(|c| {
                let t = libm::tanh(k as f64 * libm::atanh(c as f64 * (1. / 65536.)));
                f16((t * t * 16384.) as f32)
            })
            .collect();
        let gains: [f32; 3] = core::array::from_fn(|r| m[r][0] + m[r][1] + m[r][2]);
        let kk = |r: usize, c: usize| m[r][c] - if r == c { gains[r] } else { 0. };
        let c0 = [gains[0], gains[1], gains[2], kk(0, 0), kk(0, 2), kk(1, 0), kk(1, 2), kk(2, 0)].map(f16);
        let c1 = [kk(2, 2), 0., 0., 0., 0., 0., 0., 0.].map(f16);
        let pow = |x: f32, p: f32| libm::exp2f(p * libm::log2f(x));
        let nudge = |i: usize| (pow(1. + i as f32 / 64., g) * 32768. + 0.5) as u32;
        let (mut nudge_hi, mut nudge_lo, mut slope) = ([0u8; 64], [0u8; 64], [0u8; 64]);
        for i in 0..64 {
            let n = nudge(i);
            nudge_hi[i] = (n >> 8) as u8;
            nudge_lo[i] = n as u8;
            slope[i] = (nudge(i + 1) - n) as u8;
        }
        let (mut scale_hi, mut scale_lo) = ([0u8; 32], [0u8; 32]);
        for e in 1..32 {
            let s = if e < 30 { (2. * 16368. * pow(2., (e as f32 - 29.) * g) + 0.5) as u32 } else { 65535 };
            scale_hi[e] = (s >> 8) as u8;
            scale_lo[e] = s as u8;
        }
        Bake { generation, k, m, g, light, c0, c1, nudge_hi, nudge_lo, slope, scale_hi, scale_lo }
    }

    /// `dst.len()` pixels, pixel i given by `px(i)` as its channels' 16-bit g2t codes over black (red first) and its fourth lane (bits 48-63): the light by the table, then the panel's matrix and code, 16 at a time.
    pub fn row(&self, px: impl Fn(usize) -> ([u16; 3], u16), dst: &mut [u64]) {
        let n = dst.len();
        let mut x = 0;
        while x < n {
            let k = (n - x).min(16);
            let (mut l, mut a) = ([0u16; 48], [0u16; 16]);
            for j in 0..k {
                let (c, a3) = px(x + j);
                l[j] = self.light[c[0] as usize];
                l[16 + j] = self.light[c[1] as usize];
                l[32 + j] = self.light[c[2] as usize];
                a[j] = a3;
            }
            let mut out = [0u64; 16];
            self.sixteen(&l, &a, &mut out);
            dst[x..x + k].copy_from_slice(&out[..k]);
            x += k;
        }
    }

    /// 16 pixels' lights (f16 bits: red, green, blue sixteen each) to baked pixels with their fourth lanes `a`.
    #[cfg(target_arch = "aarch64")]
    fn sixteen(&self, l: &[u16; 48], a: &[u16; 16], out: &mut [u64; 16]) {
        use core::arch::aarch64::*;
        unsafe {
            let v = |i: usize| vld1q_u16(l.as_ptr().add(8 * i));
            let (c0, c1) = (vld1q_u16(self.c0.as_ptr()), vld1q_u16(self.c1.as_ptr()));
            let (r0, g0, b0) = matrix8(c0, c1, v(0), v(2), v(4));
            let (r1, g1, b1) = matrix8(c0, c1, v(1), v(3), v(5));
            let t = Code {
                nudge_hi: vld1q_u8_x4(self.nudge_hi.as_ptr()),
                nudge_lo: vld1q_u8_x4(self.nudge_lo.as_ptr()),
                slope: vld1q_u8_x4(self.slope.as_ptr()),
                scale_hi: vld1q_u8_x2(self.scale_hi.as_ptr()),
                scale_lo: vld1q_u8_x2(self.scale_lo.as_ptr()),
            };
            let (cr, cg, cb) = (code16(&t, r0, r1), code16(&t, g0, g1), code16(&t, b0, b1));
            // ×4 into Q16, saturating: a light past the panel's peak (a colour beyond its primaries) codes past 16383, and is the top rather than a carry into a dark code (the display's `bake_store`).
            let q = |c: uint16x8_t| vqshlq_n_u16::<2>(c);
            vst4q_u16(out.as_mut_ptr() as *mut u16, uint16x8x4_t(q(cr.0), q(cg.0), q(cb.0), vld1q_u16(a.as_ptr())));
            vst4q_u16((out.as_mut_ptr() as *mut u16).add(32), uint16x8x4_t(q(cr.1), q(cg.1), q(cb.1), vld1q_u16(a.as_ptr().add(8))));
        }
    }

    /// The same in plain f32 where there is no NEON (the host's tests): the lights from their f16 bits, the matrix, the code p^g × 65472, the cast holding it at the field's ends.
    #[cfg(not(target_arch = "aarch64"))]
    fn sixteen(&self, l: &[u16; 48], a: &[u16; 16], out: &mut [u64; 16]) {
        let from16 = |h: u16| {
            let (s, e, m) = (h >> 15, (h >> 10 & 31) as i32, (h & 1023) as f32);
            let v = if e == 0 { m * libm::exp2f(-24.) } else { (1. + m / 1024.) * libm::exp2f((e - 15) as f32) };
            if s == 1 { -v } else { v }
        };
        for j in 0..16 {
            let lj = [from16(l[j]), from16(l[16 + j]), from16(l[32 + j])].map(|v| v * (1. / 16384.));
            let ch = |r: usize| {
                let p = self.m[r][0] * lj[0] + self.m[r][1] * lj[1] + self.m[r][2] * lj[2];
                // A negative light (outside the panel's gamut) has no power: NaN, which the cast makes 0, as the NEON tables do.
                (libm::exp2f(self.g * libm::log2f(p)) * 65472.) as u16 as u64
            };
            out[j] = ch(0) | ch(1) << 16 | ch(2) << 32 | (a[j] as u64) << 48;
        }
    }
}

/// The panel's code tables as registers (the display's `Odt`, its code part).
#[cfg(target_arch = "aarch64")]
struct Code {
    nudge_hi: core::arch::aarch64::uint8x16x4_t,
    nudge_lo: core::arch::aarch64::uint8x16x4_t,
    slope: core::arch::aarch64::uint8x16x4_t,
    scale_hi: core::arch::aarch64::uint8x16x2_t,
    scale_lo: core::arch::aarch64::uint8x16x2_t,
}

/// Eight pixels' light thru the panel's matrix as gains + K (f16, by asm: Rust's f16 is unstable): p = g∘l + K_R·(l_R − l_G) + K_B·(l_B − l_G). The display's `matrix8`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn matrix8(c0: core::arch::aarch64::uint16x8_t, c1: core::arch::aarch64::uint16x8_t, lr: core::arch::aarch64::uint16x8_t, lg: core::arch::aarch64::uint16x8_t, lb: core::arch::aarch64::uint16x8_t) -> (core::arch::aarch64::uint16x8_t, core::arch::aarch64::uint16x8_t, core::arch::aarch64::uint16x8_t) {
    unsafe {
        let (pr, pg, pb);
        core::arch::asm!(
            "fsub {dr:v}.8h, {lr:v}.8h, {lg:v}.8h",
            "fsub {db:v}.8h, {lb:v}.8h, {lg:v}.8h",
            "fmul {pr:v}.8h, {lr:v}.8h, {c0:v}.h[0]",
            "fmul {pg:v}.8h, {lg:v}.8h, {c0:v}.h[1]",
            "fmul {pb:v}.8h, {lb:v}.8h, {c0:v}.h[2]",
            "fmla {pr:v}.8h, {dr:v}.8h, {c0:v}.h[3]",
            "fmla {pg:v}.8h, {dr:v}.8h, {c0:v}.h[5]",
            "fmla {pb:v}.8h, {dr:v}.8h, {c0:v}.h[7]",
            "fmla {pr:v}.8h, {db:v}.8h, {c0:v}.h[4]",
            "fmla {pg:v}.8h, {db:v}.8h, {c0:v}.h[6]",
            "fmla {pb:v}.8h, {db:v}.8h, {c1:v}.h[0]",
            lr = in(vreg) lr, lg = in(vreg) lg, lb = in(vreg) lb,
            c0 = in(vreg_low16) c0, c1 = in(vreg_low16) c1,
            dr = out(vreg) _, db = out(vreg) _,
            pr = out(vreg) pr, pg = out(vreg) pg, pb = out(vreg) pb,
            options(pure, nomem, nostack),
        );
        (pr, pg, pb)
    }
}

/// 16 panel lights (f16 bits, two halves) to their Q4 10-bit codes (the display's `code16`): the exponent picks the code at its octave's foot, the top six mantissa bits the nudge across the octave and its slope, interpolated on the low four; a negative light's sign puts its exponent index past the table, which answers 0.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn code16(o: &Code, a: core::arch::aarch64::uint16x8_t, b: core::arch::aarch64::uint16x8_t) -> (core::arch::aarch64::uint16x8_t, core::arch::aarch64::uint16x8_t) {
    use core::arch::aarch64::*;
    unsafe {
        let (a8, b8) = (vreinterpretq_u8_u16(a), vreinterpretq_u8_u16(b));
        let (hi, lo) = (vuzp2q_u8(a8, b8), vuzp1q_u8(a8, b8));
        let e = vshrq_n_u8::<2>(hi);
        let m6 = vandq_u8(vsliq_n_u8::<4>(vshrq_n_u8::<4>(lo), hi), vdupq_n_u8(63));
        let m4 = vandq_u8(lo, vdupq_n_u8(15));
        let (nh, nl, sl) = (vqtbl4q_u8(o.nudge_hi, m6), vqtbl4q_u8(o.nudge_lo, m6), vqtbl4q_u8(o.slope, m6));
        let (sh, sl2) = (vqtbl2q_u8(o.scale_hi, e), vqtbl2q_u8(o.scale_lo, e));
        let f0 = vrsraq_n_u16::<4>(vreinterpretq_u16_u8(vzip1q_u8(nl, nh)), vmull_u8(vget_low_u8(sl), vget_low_u8(m4)));
        let f1 = vrsraq_n_u16::<4>(vreinterpretq_u16_u8(vzip2q_u8(nl, nh)), vmull_high_u8(sl, m4));
        let (s0, s1) = (vreinterpretq_u16_u8(vzip1q_u8(sl2, sh)), vreinterpretq_u16_u8(vzip2q_u8(sl2, sh)));
        let c = |f: uint16x8_t, s: uint16x8_t| vrshrn_high_n_u32::<16>(vrshrn_n_u32::<16>(vmull_u16(vget_low_u16(f), vget_low_u16(s))), vmull_high_u16(f, s));
        (c(f0, s0), c(f1, s1))
    }
}

/// 8-bit `g2t` code to the display's gamma-2 Q14 working value: `16384 × atanh(c / 256)`.
fn q14_from_g2t8() -> &'static [u16; 256] {
    static TABLE: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u16; 256];
        for (c, v) in t.iter_mut().enumerate() {
            *v = (16384.0 * libm::atanhf(c as f32 / 256.0) + 0.5) as u16;
        }
        t
    })
}

impl PanelPixel for Argb8 {
    const FORMAT: u8 = 0;
    /// The 8-bit present: the visible code over black, `α − darkness` (no subtraction can underflow: fluor's own buffer, where `under` keeps darkness ≤ α), thru the 256-entry table. A transparent pixel is black (no see-through bit at this depth).
    fn to_panel(src: &[Self], dst: &mut [u64]) {
        let lut = q14_from_g2t8();
        for (d, &px) in dst.iter_mut().zip(src.iter()) {
            let a = px.alpha();
            let [r, g, b] = px.channels().map(|c| a - c);
            *d = lut[r as usize] as u64 | (lut[g as usize] as u64) << 16 | (lut[b as usize] as u64) << 32;
        }
    }
    /// Baked: the visible code over black as `to_panel` has it, at its 16-bit code (`c × 2⁸`, the same light); opaque, as at this depth always.
    fn to_baked(src: &[Self], dst: &mut [u64], bake: &Bake) {
        bake.row(
            |i| {
                let px = src[i];
                let a = px.alpha();
                (px.channels().map(|c| ((a - c) << 8) as u16), 0)
            },
            dst,
        );
    }
}

impl PanelPixel for G2t16 {
    const FORMAT: u8 = 1;
    /// The pixels as drawn: the display decodes them (`G2T16`).
    fn to_panel(src: &[Self], dst: &mut [u64]) {
        dst.copy_from_slice(src);
    }
    /// Baked: each channel's code over black, α − darkness (no subtraction can underflow: fluor's own buffer, where `under` keeps darkness ≤ α); the transparency (16384 − α / 4) in bits 49-62, which where α is 0 is exactly the see-through bit.
    fn to_baked(src: &[Self], dst: &mut [u64], bake: &Bake) {
        bake.row(
            |i| {
                let px = src[i];
                let a = px.alpha();
                (px.channels().map(|c| (a - c) as u16), ((16384 - (a >> 2)) << 1) as u16)
            },
            dst,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-bit `g2t` code to the display's gamma-2 Q14 working value: `16384 × atanh(c / 65536)`, every code, 128 KiB built once: the display's own `G2T16_Q14`, kept here as the reference its decode is held to.
    /// The display's channel stops at 65535 (4.0, 16× white light); codes above `tanh(4) × 2¹⁶` (65 457 up, 22× to 34.7× white) saturate there.
    fn q14_from_g2t16() -> &'static [u16] {
        static TABLE: std::sync::OnceLock<alloc::vec::Vec<u16>> = std::sync::OnceLock::new();
        TABLE.get_or_init(|| {
            (0..1u32 << 16)
                .map(|c| {
                    let v = 16384.0 * libm::atanh(c as f64 / 65536.0) + 0.5;
                    // WHY/PROOF: the display's channel is 16 bits (4.0); the top 79 codes decode past it (atanh(65535/65536) × 2¹⁴ ≈ 90 850), and the `as u16` cast would saturate them anyway — the min names the ceiling the plan set ("saturating at the surface's 4.0") instead of leaving it to the cast.
                    v.min(65535.) as u16
                })
                .collect()
        })
    }

    /// The display's decode of a G2t16 pixel (ferros payloads/display `working`), the contract a G2T16 window is shown under: each channel's visible code over black, α less its (premultiplied) darkness, thru the table, and α 0 see-through.
    fn display_working(p: G2t16) -> u64 {
        let (lut, a) = (q14_from_g2t16(), p.alpha());
        if a == 0 {
            return SEE_THROUGH;
        }
        let [r, g, b] = p.channels().map(|c| a - c);
        lut[r as usize] as u64 | (lut[g as usize] as u64) << 16 | (lut[b as usize] as u64) << 32
    }

    /// The 16-bit table is the 8-bit one at every 8-bit code's own 16-bit code (`c × 2⁸`, the same light), and rises to the display's ceiling.
    #[test]
    fn the_tables_agree_on_the_same_light() {
        let (t8, t16) = (q14_from_g2t8(), q14_from_g2t16());
        for c in 0..256usize {
            assert!(t16[c << 8].abs_diff(t8[c]) <= 1, "code {c}: {} vs {}", t16[c << 8], t8[c]);
        }
        assert_eq!(t16[0], 0);
        assert_eq!(t16[0xFFFF], 0xFFFF);
        assert!(t16.windows(2).all(|w| w[0] <= w[1]), "monotone");
        // Diffuse white (light 1) is code 194.97 of 256: its bucket's bottom decodes to atanh(194/256) = 0.9915, just under 2¹⁴, at both depths, and the 16-bit code for light 1 itself (tanh(1) × 2¹⁶ = 49912) is 2¹⁴ to the step.
        assert_eq!(t16[194 << 8], t8[194]);
        assert!(t16[49912].abs_diff(16384) <= 1, "{}", t16[49912]);
    }

    /// A translucent pixel decodes as its light over black: black laid at half α by `under` stays black (the straight-darkness decode showed it as a grey of code 2¹⁴), and white at half α is half of white's code.
    #[test]
    fn translucent_decodes_over_black() {
        use crate::pixel::{Blend, BlendMode};
        let half = 1 << 15;
        let black = G2t16::EMPTY.under(u64::MAX.with_alpha(half), BlendMode::Normal);
        assert_eq!(display_working(black) & 0xFFFF, q14_from_g2t16()[1] as u64);
        let white = G2t16::EMPTY.under(G2t16::OPAQUE.with_alpha(half), BlendMode::Normal);
        assert_eq!(display_working(white) & 0xFFFF, q14_from_g2t16()[half as usize] as u64);
    }

    /// The same opaque colour reaches the display alike at both depths (the 8-bit present against the display's 16-bit decode), and an empty 16-bit pixel is see-through.
    #[test]
    fn both_depths_present_alike() {
        let colours: [u32; 4] = [crate::theme::WHITE, crate::theme::BUTTON_FILL, crate::theme::TEXTBOX_SELECTION_BG, 0xFF00_0000];
        for c in colours {
            let (mut d8, mut d16) = ([0u64; 1], [0u64; 1]);
            Argb8::to_panel(&[c], &mut d8);
            d16[0] = display_working(G2t16::from_g2t8(c));
            for sh in [0, 16, 32] {
                let (v8, v16) = ((d8[0] >> sh) & 0xFFFF, (d16[0] >> sh) & 0xFFFF);
                assert!(v8.abs_diff(v16) <= 1, "{c:08x} channel at {sh}: {v8} vs {v16}");
            }
        }
        assert_eq!(display_working(G2t16::EMPTY), SEE_THROUGH);
        // And the 16-bit present hands the pixels over as drawn.
        let mut d = [0u64; 1];
        G2t16::to_panel(&[G2t16::from_g2t8(colours[1])], &mut d);
        assert_eq!(d[0], G2t16::from_g2t8(colours[1]));
    }

    /// husky's ODT (ferros payloads/display odt.rs `ODT_HUSKY`) and its gamma, at a white about a ninth of the E peak (k as the display has it there).
    fn husky_bake() -> Bake {
        let m = [[1.359550, -0.372458, 0.012909], [-0.098444, 0.863965, -0.026116], [0.001826, -0.001101, 0.816481]];
        Bake::new(7, libm::atanhf(libm::sqrtf(0.113)), m, 1. / 2.19)
    }

    /// The bake's code for a g2t code over black, worked out whole in f64: l = tanh²(k · atanh(c / 2¹⁶)), p = m · l, p^g × 65472.
    fn baked_f64(b: &Bake, c: [u16; 3]) -> [f64; 3] {
        let l = c.map(|c| libm::tanh(b.k as f64 * libm::atanh(c as f64 / 65536.)).powi(2));
        core::array::from_fn(|r| {
            let p: f64 = (0..3).map(|j| b.m[r][j] as f64 * l[j]).sum();
            if p > 0. { p.powf(b.g as f64) * 65472. } else { 0. }
        })
    }

    /// Opaque pixels bake to their formula within a few hundredths of a code of the 8-bit link (16 of 65536; the table is f16), black to black, at both depths alike.
    #[test]
    fn bake_matches_its_formula() {
        let b = husky_bake();
        let cases: [[u16; 3]; 6] = [[0, 0, 0], [49912, 49912, 49912], [65280, 0, 0], [0, 65280, 0], [12800, 25600, 51200], [256, 512, 1024]];
        for c in cases {
            let px = G2t16::from_channels(65535, c.map(|v| 65535 - v as u32));
            let mut d = [0u64; 1];
            G2t16::to_baked(&[px], &mut d, &b);
            let want = baked_f64(&b, c);
            for (ch, w) in want.iter().enumerate() {
                let got = (d[0] >> (16 * ch) & 0xFFFF) as f64;
                assert!((got - w.min(65535.)).abs() <= 16. + w * (1. / 512.), "{c:?} channel {ch}: {got} vs {w}");
            }
            // The transparency of an opaque pixel: one Q14 step (α is 65535, 16384 − 16383).
            assert_eq!(d[0] >> 48, 2, "{c:?}");
        }
        let mut z = [0u64; 1];
        G2t16::to_baked(&[G2t16::from_channels(65535, [65535; 3])], &mut z, &b);
        assert_eq!(z[0] & 0xFFFF_FFFF_FFFF, 0, "black");
    }

    /// An empty 16-bit pixel bakes to exactly the see-through bit, with no light; an 8-bit pixel bakes as its 16-bit code would.
    #[test]
    fn bake_see_through_and_depths() {
        let b = husky_bake();
        let mut d = [0u64; 1];
        G2t16::to_baked(&[G2t16::EMPTY], &mut d, &b);
        assert_eq!(d[0], SEE_THROUGH);
        for c in [crate::theme::WHITE, crate::theme::BUTTON_FILL, crate::theme::TEXTBOX_SELECTION_BG] {
            let (mut d8, mut d16) = ([0u64; 1], [0u64; 1]);
            Argb8::to_baked(&[c], &mut d8, &b);
            G2t16::to_baked(&[G2t16::from_g2t8(c)], &mut d16, &b);
            for sh in [0, 16, 32] {
                let (v8, v16) = ((d8[0] >> sh) & 0xFFFF, (d16[0] >> sh) & 0xFFFF);
                assert!(v8.abs_diff(v16) <= 64, "{c:08x} channel at {sh}: {v8} vs {v16}");
            }
        }
    }

    /// The display's op 65 reply round trips.
    #[test]
    fn bake_from_reply() {
        let b = husky_bake();
        let mut r = [0u8; 49];
        r[1..5].copy_from_slice(&b.generation.to_le_bytes());
        r[5..9].copy_from_slice(&b.k.to_le_bytes());
        for (i, v) in b.m.iter().flatten().enumerate() {
            r[9 + 4 * i..13 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        r[45..49].copy_from_slice(&b.g.to_le_bytes());
        let c = Bake::from_reply(&r).expect("a reply");
        assert_eq!((c.generation, c.k, c.m, c.g), (b.generation, b.k, b.m, b.g));
        r[0] = 1;
        assert!(Bake::from_reply(&r).is_none());
    }

    /// The portable f16 against known bit patterns: 1, 2^14, the smallest normal, a subnormal, past the range.
    #[test]
    fn f16_bits() {
        assert_eq!(f16(1.), 0x3C00);
        assert_eq!(f16(16384.), 0x7400);
        assert_eq!(f16(libm::exp2f(-14.)), 0x0400);
        assert_eq!(f16(libm::exp2f(-24.)), 0x0001);
        assert_eq!(f16(1e9), 0x7C00);
        assert_eq!(f16(0.), 0);
    }
}
