//! Pixel format + blend trait.
//!
//! # Convention — locked
//!
//! **Internal pixel format**: `0xααRRGGBB` packed in a `u32` — top byte is **α (opacity)**; RGB bytes are **darkness** (`0 = white potential`, `255 = black ink`). LE byte layout `[B, G, R, α]`. Every paint primitive, every layer buffer, every Group composite uses this. `Argb8` is a type alias for `u32`; the byte layout matches a typical compositor pixel, but the RGB bytes are the *bitwise complement* of visible RGB.
//!
//! **Two depths, one convention** ([`Pixel`]): [`Argb8`] is the layout above; [`G2t16`] is the same layout widened to a `u64` (`α << 48 | R << 32 | G << 16 | B`, each field a 16-bit `g2t` code's darkness).
//! Every rendering kernel is generic over the depth and reaches the fields thru the trait, so every 255, 256 and `>> 8` below reads as the depth's ONE, ONE + 1 and `>> DEPTH`; colours stay authored at 8 bits and widen at a kernel's entry ([`Pixel::from_g2t8`]).
//!
//! **Opacity semantics**: `α = 0` = fully transparent, `α = 255` = fully opaque — industry-standard direction. Empty pixel = `0x00000000` (no opacity, no darkness — calloc-free zero-init). Both halves of the pixel accumulate up from 0: α adds, darkness adds.
//!
//! **Why this convention** (mirroring the README): the Under accumulator is pure addition (`new_dark = top_dark + (mod_dark × consumed >> 8)`, `new_α = top_α + consumed`) — no `255 − x` anywhere on the hot path. Three subtractions saved per pixel per Under call versus the old visible-RGB convention. Hot-loop early-out fires when the top is fully opaque: `if self >= 0xFF000000` is "top α-byte is 0xFF" in one CMP (α-byte saturating implies the lower bits don't matter for the result — bottom is invisible). Empty marker is the zero pattern, so `vec![0u32; n]` uses calloc → zero pages → genuinely free init.
//!
//! **Overflow proof**: with `top_α ∈ [0, 254]` on the math path (255 hits the early-out), let `k = 256 − top_α ∈ [2, 256]`. Then `consumed = floor(k × bot_α / 256) ≤ floor(k × 255 / 256) = k − 1 = 255 − top_α`. So `new_α = top_α + consumed ≤ 255`, never overflows. The invariant `dark ≤ α` is preserved inductively (base `(0, 0)`; step uses `contrib − consumed ≤ 0` from the same floor argument). Therefore `new_dark ≤ new_α ≤ 255` strictly — **plain `+` is safe everywhere, no `saturating_add` needed**.
//!
//! **Boundary**: paint primitives write α + darkness directly. The host present pass applies a single `pixel ^= 0x00FFFFFF` right before submitting to wgpu / softbuffer — flips darkness → visible RGB; α stays. Folded into `finalize_for_os` alongside the clip-mask multiply and (on Linux) the premultiply step. External inputs (cosmic-text glyph coverage, `pack_argb`) flip once at the import boundary, never per-frame.
//!
//! # The unified `Blend::under` kernel
//!
//! Every compositing op in fluor — Normal source-under, Multiply, Screen, Add, Subtract, Overlay, Darken, Lighten — flows thru one trait method: `top.under(bottom, mode)`. `top` is the partial composite already accumulated above (its α-byte = opacity so far; its RGB = darkness accumulated so far, with the canonical empty value `0x00000000` representing "nothing here yet"). `bottom` is the new layer going behind. The mode shapes only how `bottom`'s darkness is interpreted before contributing.
//!
//! Convention: each new layer ADDS darkness to the buffer at `mod_dark × consumed >> 8` per channel, where `consumed = ((256 − top_α) × bot_α) >> 8` (how much of the remaining opacity budget the new layer fills). The opacity accumulates as `new_α = top_α + consumed`. Mode kernels operate in *darkness space* but preserve the *visible-space* semantic the mode name promises (e.g., `BlendMode::Multiply` still darkens the visible result like Photoshop multiply).
//!
//! # When `under()` is worth it (vs read-modify-write)
//!
//! Two rasterizer patterns exist in fluor. Both compose a new layer over a partial buffer; they differ in where per-pixel computation happens.
//!
//! * **`under()` topmost-first**: caller computes `src = f(x, y)` (some pure function of position + constants), then `dst[i] = dst[i].under(src, mode)`. Buffer holds the running composite; `src` is independent of what's already there.
//! * **Direct read-modify-write**: caller reads `dst[i]`, computes `new = g(dst[i], x, y)`, writes back. `g` reads the bg pixel as an INPUT — the wave's per-pixel sqrt-blend `(c_wave · scale + c_bg²).sqrt()` is a canonical example; the noise's shimmer-mixed-into-seed is another.
//!
//! Choice depends on the cost of the src computation, the availability of an early-out, and whether the pattern SIMDs.
//!
//! **Scalar `Blend::under` early-out** — `if self >= 0xFF000000 { return self; }` fires whenever the top is fully opaque; bottom is invisible so the mode math is skipped. Release-mode LLVM inlines the trait method into the caller's loop, sees the src computation is pure, and successfully hoists the opaque check ABOVE the src work — DCE elides the whole `f(x, y)` pipeline on opaque pixels. Verified against a wave-mimicking loop (sqrt + trig + matmul-shaped src) built at `opt-level = 3`: LLVM emits `cmp dst, 0xFEFFFFFF; ja end_of_pixel_body` right after the dst load, jumping past every `sqrtss` / `mulss` / `call sinf` on the opaque branch. Debug (`opt-level = 0`) does NOT reorder — src is computed unconditionally, then `under()` early-outs and discards. Any perf-relevant DEBUG use of `under()` with expensive src pays the full compute cost on every pixel.
//!
//! **SIMD `under_x8_normal`** has no per-lane early-out. Every lane runs the full RGB math; the opacity mask is applied at the end via `(opaque_mask & dst) | (not_mask & result)`. Zero DCE possible — the src computation always executes for all eight lanes regardless of opt level. Vectorized rasterizers layered under a mostly-opaque top pay full compute on every lane.
//!
//! **Guidelines**:
//!
//! * `under()` is right for **cheap or constant src** (uniform tint, gradients, single-glyph blits, chrome buttons) — waste is negligible in both dirs, doctrine cleanliness dominates. All chrome / widget / textbox rasterizers use this path.
//! * Direct read-modify-write is right when the src computation is **expensive AND depends on the bg pixel value AND / OR you want a predictable SIMD-vectorization ceiling**. The chromatic wave (photon) and background noise (fluor) both fall here: the bg is a mathematical input (wave's per-channel sqrt-blend of bg squared; noise's row-seeded RNG chain), not something an under-blend could recover cleanly. Rewriting them as `under()` would either lose the visual (the sqrt-blend isn't expressible as `mode_kernel(top, bot)`) or waste per-pixel work on any SIMD-ification.
//! * If you're unsure, prefer `under()` — the doctrine cleanliness is real and release performance is usually fine. Reach for direct read-modify-write only when profiling shows it, or when the effect literally needs the bg value as an argument.

/// Packed pixel in α + darkness convention: `0xααRRGGBB`. Type alias for `u32` — `&[Argb8]` and `&[u32]` are the same slice. The name carries the convention contract, not a new type.
pub type Argb8 = u32;

/// How `bottom`'s darkness is mixed into the partial composite `top` when composing front-to-back.
///
/// Each mode is a pure channel-wise function of `(top_dark, bottom_dark)` that produces a "modulated bottom darkness" `(mr, mg, mb)`. The opacity-accumulator math around it is identical across all modes — see [`Blend::under`]. Mode names describe the *visible-space* semantic (e.g., "Multiply" darkens like Photoshop multiply on visible RGB); the formulas below operate on darkness operands and produce darkness output that round-trips thru the OS XOR boundary to match the named visible-space behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Source-under: bottom passes thru unchanged. `mr = bd`. Standard front-to-back compositing.
    Normal,
    /// Visible-multiply (darken-darken): `mr = 255 − (((255 − td) × (255 − bd)) >> 8)`. Bottom is "filtered thru" top's intensity in visible space.
    Multiply,
    /// Visible-screen (brighten-brighten): `mr = (td × bd) >> 8`. The darkness-space dual of visible-multiply.
    Screen,
    /// Saturating per-channel visible-add (brightens): `mr = (td + bd).saturating_sub(255)`. Additive light in visible space.
    Add,
    /// Saturating per-channel visible-subtract (darkens): `mr = 255 − bd.saturating_sub(td)`. Subtractive light in visible space.
    Subtract,
    /// Contrast-shifting Photoshop overlay: multiply where bottom is dark, screen where bottom is bright. Pivot at `bd > 0x80` (= `bv < 0x80`).
    Overlay,
    /// Per-channel visible-min (keeps the darker channel): `mr = td.max(bd)`.
    Darken,
    /// Per-channel visible-max (keeps the brighter channel): `mr = td.min(bd)`.
    Lighten,
}

/// The single compositing operation in fluor. Front-to-back source-under with selectable blend mode.
pub trait Blend {
    /// Compose `bottom` *underneath* `self` (the partial composite from layers above). `self`'s α-byte is the accumulated opacity; `self`'s RGB is the accumulated darkness (canonical empty value `0x00000000` = "nothing yet").
    ///
    /// Single early-out: `self >= 0xFF000000` ⇔ `self.α == 0xFF` (top opaque). Returns `self` unchanged — bottom is invisible behind a fully-opaque top.
    ///
    /// Math (all `>> 8`, no `/ 255`, no floats):
    /// ```text
    /// consumed     = ((256 − top_α) × bot_α) >> 8         (0..=255 − top_α) — amount the new layer contributes (mr, mg, mb) = mode_kernel(top_dark, bot_dark)      (0..=255 per channel) in darkness space contrib_ch   = (m_ch × consumed) >> 8               (darkness this layer deposits) out_dark_ch  = top_dark_ch + contrib_ch             (plain +, bounded ≤ 255 by floor proof) new_α        = top_α + consumed                     (plain +, bounded ≤ 255 by floor proof)
    /// ```
    /// Starting from `0x00000000` (no opacity, no darkness), each new layer adds darkness and opacity to the running total. Multi-layer stacking is repeated addition until α saturates or layers exhaust.
    ///
    /// The operands are `Self` so the same kernel name serves every pixel depth: [`Pixel`] has `Blend` as a supertrait, which is how generic code reaches `under` thru a `P: Pixel` bound.
    /// At 16 bits every 8, 255 and 256 above is the depth's own (16, 65535, 65536), and the floor proof carries over word for word.
    fn under(self, bottom: Self, mode: BlendMode) -> Self;
}

/// A pixel depth fluor can render at.
/// Every depth holds the same thing, VSF `g2t` codes as α + darkness: empty is all zeros, opaque black is all ones, and a channel's darkness is (top code − code) × α.
/// Only the width of the fields differs, so a kernel written against these constants and methods runs unchanged at 8 bits ([`Argb8`]) and 16 ([`G2t16`]).
///
/// `under` comes from the [`Blend`] supertrait rather than a method of its own: with both traits in scope a concrete `u32` would otherwise see two `under`s and every call would be ambiguous.
pub trait Pixel: Blend + Copy + Eq + Send + Sync + core::fmt::Debug + 'static {
    /// Bits per field (α and each darkness channel).
    /// Named DEPTH, not BITS: on the `u32` depth `Argb8::BITS` is u32's own inherent constant (32) and would silently outrank a trait constant of that name.
    const DEPTH: u32;
    /// A full field: opaque α, or black's darkness in one channel.
    const ONE: u32;
    /// Where α sits: `pixel >> ALPHA_SHIFT` is the α field.
    const ALPHA_SHIFT: u32;
    /// The three darkness channels.
    const RGB_MASK: Self;
    /// α full, no darkness: the α field alone.
    const OPAQUE: Self;
    /// Nothing here yet: no opacity, no darkness.
    const EMPTY: Self;

    /// The α field, `0..=ONE`.
    fn alpha(self) -> u32;
    /// The darkness channels with α cleared.
    fn rgb(self) -> Self;
    /// These darkness channels under a new α (`0..=ONE`, not checked: a wider value spills out of the field, and that is a bug in the caller's math).
    fn with_alpha(self, a: u32) -> Self;
    /// The three darkness channels, red first, each `0..=ONE`.
    fn channels(self) -> [u32; 3];
    /// A pixel from α and three darkness channels, each `0..=ONE` (not checked, as [`Pixel::with_alpha`]).
    fn from_channels(a: u32, rgb: [u32; 3]) -> Self;
    /// A pixel of darkness `rgb` at coverage `cov_q16`, coverage in Q16 with `0xFFFF` the whole pixel: α is the coverage's top DEPTH bits, the darkness passing thru as it is, because `under` scales darkness by the α it consumes.
    /// This is the squircle crossings' AA weight (`(√frac × 2¹⁶) as u16`): its top 8 bits are exactly the byte `(√frac × 2⁸) as u8` the 8-bit kernel always wrote, and at 16 bits the whole weight is the α.
    fn from_cov(rgb: Self, cov_q16: u32) -> Self;
    /// A one-pixel edge ramp `v` (≤ 0 outside, ≥ 1 inside) as coverage in this depth's own scale, `0..=ONE`: `v × ONE`, the cast to the field's width saturating both ends, which is the whole clamp.
    fn cov(v: f32) -> u32;
    /// A colour authored as an 8-bit `g2t` α + darkness pixel (the theme's palette, `theme::dark(fmt(vsf(..)))`) at this depth: the same light, widened.
    fn from_g2t8(c: u32) -> Self;
    /// An 8-bit α (`0..=255`, a coverage byte a rasterizer computed) as this depth's α: the same fraction, `a × ONE / 255`, which is how [`Pixel::from_g2t8`] widens α (× 1 at 8 bits, × 257 at 16).
    #[inline(always)]
    fn alpha_from8(a8: u32) -> u32 {
        a8 * (Self::ONE / 255)
    }

    /// `dst[i] = dst[i].under(src[i], Normal)` over a run: the hook a depth's SIMD kernel takes over (Argb8 runs 8 lanes of `wide::u32x8` with the `simd` feature).
    #[inline]
    fn under_chunk_normal(dst: &mut [Self], src: &[Self]) {
        for i in 0..dst.len() {
            dst[i] = dst[i].under(src[i], BlendMode::Normal);
        }
    }
    /// [`Pixel::under_chunk_normal`] with every source pixel the same: the interior run of a solid fill.
    #[inline]
    fn under_chunk_const(dst: &mut [Self], src: Self) {
        for i in 0..dst.len() {
            dst[i] = dst[i].under(src, BlendMode::Normal);
        }
    }
}

impl Pixel for Argb8 {
    const DEPTH: u32 = 8;
    const ONE: u32 = (1 << Self::DEPTH) - 1;
    const ALPHA_SHIFT: u32 = 3 * Self::DEPTH;
    const RGB_MASK: Self = (1 << Self::ALPHA_SHIFT) - 1;
    const OPAQUE: Self = Self::ONE << Self::ALPHA_SHIFT;
    const EMPTY: Self = 0;

    #[inline(always)]
    fn alpha(self) -> u32 {
        self >> Self::ALPHA_SHIFT
    }
    #[inline(always)]
    fn rgb(self) -> Self {
        self & Self::RGB_MASK
    }
    #[inline(always)]
    fn with_alpha(self, a: u32) -> Self {
        (a << Self::ALPHA_SHIFT) | (self & Self::RGB_MASK)
    }
    #[inline(always)]
    fn channels(self) -> [u32; 3] {
        [(self >> (2 * Self::DEPTH)) & Self::ONE, (self >> Self::DEPTH) & Self::ONE, self & Self::ONE]
    }
    #[inline(always)]
    fn from_channels(a: u32, rgb: [u32; 3]) -> Self {
        (a << Self::ALPHA_SHIFT) | (rgb[0] << (2 * Self::DEPTH)) | (rgb[1] << Self::DEPTH) | rgb[2]
    }
    #[inline(always)]
    fn from_cov(rgb: Self, cov_q16: u32) -> Self {
        ((cov_q16 >> (16 - Self::DEPTH)) << Self::ALPHA_SHIFT) | (rgb & Self::RGB_MASK)
    }
    #[inline(always)]
    fn cov(v: f32) -> u32 {
        (v * Self::ONE as f32) as u8 as u32
    }
    /// The 8-bit palette is this depth's own encoding, so it passes as it is.
    #[inline(always)]
    fn from_g2t8(c: u32) -> Self {
        c
    }

    #[inline]
    fn under_chunk_normal(dst: &mut [Self], src: &[Self]) {
        #[cfg(feature = "simd")]
        {
            use crate::simd::{LANES, u32x8};
            let n = dst.len();
            let mut i = 0;
            while i + LANES <= n {
                let d_arr: [u32; 8] = dst[i..i + LANES].try_into().unwrap();
                let s_arr: [u32; 8] = src[i..i + LANES].try_into().unwrap();
                let out = under_x8_normal(u32x8::from(d_arr), u32x8::from(s_arr));
                dst[i..i + LANES].copy_from_slice(out.as_array_ref());
                i += LANES;
            }
            while i < n {
                dst[i] = dst[i].under(src[i], BlendMode::Normal);
                i += 1;
            }
        }
        #[cfg(not(feature = "simd"))]
        for i in 0..dst.len() {
            dst[i] = dst[i].under(src[i], BlendMode::Normal);
        }
    }

    /// The source is broadcast to all 8 lanes once, outside the loop.
    #[inline]
    fn under_chunk_const(dst: &mut [Self], src_const: Self) {
        #[cfg(feature = "simd")]
        {
            use crate::simd::{LANES, u32x8};
            let src = u32x8::splat(src_const);
            let n = dst.len();
            let mut i = 0;
            while i + LANES <= n {
                let d_arr: [u32; 8] = dst[i..i + LANES].try_into().unwrap();
                let out = under_x8_normal(u32x8::from(d_arr), src);
                dst[i..i + LANES].copy_from_slice(out.as_array_ref());
                i += LANES;
            }
            while i < n {
                dst[i] = dst[i].under(src_const, BlendMode::Normal);
                i += 1;
            }
        }
        #[cfg(not(feature = "simd"))]
        for i in 0..dst.len() {
            dst[i] = dst[i].under(src_const, BlendMode::Normal);
        }
    }
}

/// Packed 16-bit pixel in the α + darkness convention: the 8-bit layout widened, `α << 48 | R << 32 | G << 16 | B`, each field a 16-bit VSF `g2t` darkness (opaque: 65535 − code, code / 2¹⁶ the encoded value, light = atanh(code / 2¹⁶)²).
/// A type alias for `u64` as [`Argb8`] is for `u32`: the name carries the convention.
/// The top code is 34.7× diffuse white (8 bits stop at 9.7×), and every glow and edge has 2⁸ times the steps.
pub type G2t16 = u64;

impl Pixel for G2t16 {
    const DEPTH: u32 = 16;
    const ONE: u32 = (1 << Self::DEPTH) - 1;
    const ALPHA_SHIFT: u32 = 3 * Self::DEPTH;
    const RGB_MASK: Self = (1 << Self::ALPHA_SHIFT) - 1;
    const OPAQUE: Self = (Self::ONE as u64) << Self::ALPHA_SHIFT;
    const EMPTY: Self = 0;

    #[inline(always)]
    fn alpha(self) -> u32 {
        (self >> Self::ALPHA_SHIFT) as u32
    }
    #[inline(always)]
    fn rgb(self) -> Self {
        self & Self::RGB_MASK
    }
    #[inline(always)]
    fn with_alpha(self, a: u32) -> Self {
        ((a as u64) << Self::ALPHA_SHIFT) | (self & Self::RGB_MASK)
    }
    #[inline(always)]
    fn channels(self) -> [u32; 3] {
        let one = Self::ONE as u64;
        [((self >> (2 * Self::DEPTH)) & one) as u32, ((self >> Self::DEPTH) & one) as u32, (self & one) as u32]
    }
    #[inline(always)]
    fn from_channels(a: u32, rgb: [u32; 3]) -> Self {
        ((a as u64) << Self::ALPHA_SHIFT) | ((rgb[0] as u64) << (2 * Self::DEPTH)) | ((rgb[1] as u64) << Self::DEPTH) | rgb[2] as u64
    }
    #[inline(always)]
    fn from_cov(rgb: Self, cov_q16: u32) -> Self {
        ((cov_q16 as u64) << Self::ALPHA_SHIFT) | (rgb & Self::RGB_MASK)
    }
    #[inline(always)]
    fn cov(v: f32) -> u32 {
        (v * Self::ONE as f32) as u16 as u32
    }
    /// The same light at 16 bits, field by field.
    /// `g2t` decodes a code at its bucket bottom, `atanh(code / 2ⁿ)²`, so 8-bit code `c` and 16-bit code `c × 2⁸` are the same encoded value and the same light, and VSF's settled encoder sends that light back to exactly `c × 2⁸` (held to vsf's codec by a test over all 256 codes).
    /// In darkness that is `65535 − 256·(255 − d) = 256·d + 255`.
    /// α is a fraction of full, not a code: `α × 257` takes 255 to 65535 and every byte to the same fraction.
    #[inline(always)]
    fn from_g2t8(c: u32) -> Self {
        let wide = |d: u32| (d << 8) | ((1 << 8) - 1);
        let [r, g, b] = Argb8::channels(c);
        Self::from_channels((c >> 24) * 257, [wide(r), wide(g), wide(b)])
    }

    #[inline]
    fn under_chunk_normal(dst: &mut [Self], src: &[Self]) {
        #[cfg(feature = "simd")]
        {
            use crate::simd::LANES;
            let n = dst.len();
            let mut i = 0;
            while i + LANES <= n {
                let (dh, dl) = g2t16_halves(&dst[i..i + LANES]);
                let (sh, sl) = g2t16_halves(&src[i..i + LANES]);
                let (h, l) = under_x8_normal_g2t16(dh, dl, sh, sl);
                g2t16_join(h, l, &mut dst[i..i + LANES]);
                i += LANES;
            }
            while i < n {
                dst[i] = dst[i].under(src[i], BlendMode::Normal);
                i += 1;
            }
        }
        #[cfg(not(feature = "simd"))]
        for i in 0..dst.len() {
            dst[i] = dst[i].under(src[i], BlendMode::Normal);
        }
    }

    /// The source's halves are split and broadcast once, outside the loop.
    #[inline]
    fn under_chunk_const(dst: &mut [Self], src_const: Self) {
        #[cfg(feature = "simd")]
        {
            use crate::simd::{LANES, u32x8};
            let (sh, sl) = (u32x8::splat((src_const >> 32) as u32), u32x8::splat(src_const as u32));
            let n = dst.len();
            let mut i = 0;
            while i + LANES <= n {
                let (dh, dl) = g2t16_halves(&dst[i..i + LANES]);
                let (h, l) = under_x8_normal_g2t16(dh, dl, sh, sl);
                g2t16_join(h, l, &mut dst[i..i + LANES]);
                i += LANES;
            }
            while i < n {
                dst[i] = dst[i].under(src_const, BlendMode::Normal);
                i += 1;
            }
        }
        #[cfg(not(feature = "simd"))]
        for i in 0..dst.len() {
            dst[i] = dst[i].under(src_const, BlendMode::Normal);
        }
    }
}

/// Eight G2t16 pixels as two `u32x8` halves: the high words (`α << 16 | R`) and the low (`G << 16 | B`).
#[cfg(feature = "simd")]
#[inline(always)]
fn g2t16_halves(p: &[G2t16]) -> (wide::u32x8, wide::u32x8) {
    let hi: [u32; 8] = core::array::from_fn(|k| (p[k] >> 32) as u32);
    let lo: [u32; 8] = core::array::from_fn(|k| p[k] as u32);
    (wide::u32x8::from(hi), wide::u32x8::from(lo))
}

/// [`g2t16_halves`] undone into eight pixels.
#[cfg(feature = "simd")]
#[inline(always)]
fn g2t16_join(hi: wide::u32x8, lo: wide::u32x8, out: &mut [G2t16]) {
    let (h, l) = (hi.as_array_ref(), lo.as_array_ref());
    for k in 0..8 {
        out[k] = ((h[k] as u64) << 32) | l[k] as u64;
    }
}

/// [`under_x8_normal`] at 16 bits: eight G2t16 pixels as their two `u32x8` halves ([`g2t16_halves`]), the scalar kernel lane by lane and bit for bit.
/// Every product fits a u32 lane, which is why the halves suffice and no 64-bit multiply is needed: `consumed = (2¹⁶ − top_α) × bot_α ≤ 2¹⁶ × (2¹⁶ − 1) < 2³²` before its shift, and each deposit `bot_d × consumed ≤ (2¹⁶ − 1)² < 2³²`.
/// The sums stay inside their fields by the floor proof on [`Blend::under`] (dark ≤ α ≤ ONE).
/// Lanes whose top is opaque keep it, as the scalar early-out does.
#[cfg(feature = "simd")]
#[inline]
pub fn under_x8_normal_g2t16(dst_hi: wide::u32x8, dst_lo: wide::u32x8, src_hi: wide::u32x8, src_lo: wide::u32x8) -> (wide::u32x8, wide::u32x8) {
    use wide::u32x8;
    let field = u32x8::splat((1 << 16) - 1);
    let whole = u32x8::splat(1 << 16);
    let da: u32x8 = dst_hi >> 16;
    let dr: u32x8 = dst_hi & field;
    let dg: u32x8 = dst_lo >> 16;
    let db: u32x8 = dst_lo & field;
    let sa: u32x8 = src_hi >> 16;
    let consumed: u32x8 = ((whole - da) * sa) >> 16;
    let nr: u32x8 = dr + (((src_hi & field) * consumed) >> 16);
    let ng: u32x8 = dg + (((src_lo >> 16) * consumed) >> 16);
    let nb: u32x8 = db + (((src_lo & field) * consumed) >> 16);
    let na: u32x8 = da + consumed;
    let opaque = da.cmp_eq(field);
    let keep = opaque ^ u32x8::splat(u32::MAX);
    ((opaque & dst_hi) | (keep & ((na << 16) | nr)), (opaque & dst_lo) | (keep & ((ng << 16) | nb)))
}

/// 8-wide SIMD version of [`Blend::under`] for `BlendMode::Normal` only — the 99% case in real compositing workloads. Same math as the scalar kernel lane-by-lane, plus a SIMD version of the `dst >= 0xFF000000` early-out: lanes where `dst.α == 0xFF` keep their original value via a masked blend with the SIMD result.
///
/// Other blend modes (Multiply, Screen, Add, Subtract, Overlay, Darken, Lighten) are NOT covered here — they branch per lane on bot_dark thresholds (Overlay) or call `saturating_sub` per channel, which doesn't SIMD-vectorize cleanly. Those modes stay on the scalar [`Blend::under`] kernel; only Normal gets the wide path.
#[cfg(feature = "simd")]
#[inline]
pub fn under_x8_normal(dst: wide::u32x8, src: wide::u32x8) -> wide::u32x8 {
    use wide::u32x8;
    let mask_ff = u32x8::splat(0xFF);
    let const_256 = u32x8::splat(256);
    let all_ones = u32x8::splat(0xFFFFFFFF);

    let dst_a: u32x8 = (dst >> 24) & mask_ff;
    let dst_r: u32x8 = (dst >> 16) & mask_ff;
    let dst_g: u32x8 = (dst >> 8) & mask_ff;
    let dst_b: u32x8 = dst & mask_ff;
    let src_a: u32x8 = (src >> 24) & mask_ff;
    let src_r: u32x8 = (src >> 16) & mask_ff;
    let src_g: u32x8 = (src >> 8) & mask_ff;
    let src_b: u32x8 = src & mask_ff;

    let consumed = ((const_256 - dst_a) * src_a) >> 8;
    let nr = dst_r + ((src_r * consumed) >> 8);
    let ng = dst_g + ((src_g * consumed) >> 8);
    let nb = dst_b + ((src_b * consumed) >> 8);
    let na = dst_a + consumed;
    let result: u32x8 = (na << 24) | (nr << 16) | (ng << 8) | nb;

    // SIMD early-out: lanes where dst.α was 0xFF keep dst (top was already opaque, bot invisible). cmp_eq returns 0xFFFFFFFF for true lanes, 0 for false — bitwise blend.
    let opaque_mask = dst_a.cmp_eq(mask_ff);
    let not_mask = opaque_mask ^ all_ones;
    (opaque_mask & dst) | (not_mask & result)
}

/// The one `under` kernel, written once and stamped out per depth in that depth's own integer width (`u32` for 8 bits, `u64` for 16), so the 8-bit build keeps its 32-bit arithmetic and 16 bits gets the room its products need (Overlay's `2·td·bd` reaches 2³³).
/// `$d` is the depth: `one` = 2^d − 1 is the 8-bit `0xFF`, `one + 1` its `256`, `>> $d` its `>> 8`, and `1 << ($d − 1)` Overlay's `0x80` pivot, so at `$d = 8` every expression is the one the 8-bit kernel always had.
macro_rules! impl_under {
    ($t:ty, $d:expr) => {
        impl Blend for $t {
            #[inline]
            fn under(self, bottom: $t, mode: BlendMode) -> $t {
                const D: u32 = $d;
                let one: $t = (1 << D) - 1;
                if self >= one << (3 * D) {
                    return self;
                } // top opaque — top wins, bottom invisible.
                let top_a = self >> (3 * D);
                let bot_a = bottom >> (3 * D);

                let consumed = ((one + 1 - top_a) * bot_a) >> D;

                let td_r = (self >> (2 * D)) & one;
                let td_g = (self >> D) & one;
                let td_b = self & one;
                let bd_r = (bottom >> (2 * D)) & one;
                let bd_g = (bottom >> D) & one;
                let bd_b = bottom & one;

                let pivot: $t = 1 << (D - 1);
                let (mr, mg, mb) = match mode {
                    BlendMode::Normal => (bd_r, bd_g, bd_b),
                    BlendMode::Multiply => (
                        one - (((one - td_r) * (one - bd_r)) >> D),
                        one - (((one - td_g) * (one - bd_g)) >> D),
                        one - (((one - td_b) * (one - bd_b)) >> D),
                    ),
                    BlendMode::Screen => ((td_r * bd_r) >> D, (td_g * bd_g) >> D, (td_b * bd_b) >> D),
                    // WHY/PROOF: the saturation IS the Add and Subtract blend (light past white stays white, darkness past black stays black), not a guard — plain arithmetic would wrap a bright pair to a dark result and underflow into the next lane.
                    BlendMode::Add => (
                        (td_r + bd_r).saturating_sub(one),
                        (td_g + bd_g).saturating_sub(one),
                        (td_b + bd_b).saturating_sub(one),
                    ),
                    BlendMode::Subtract => (
                        one - bd_r.saturating_sub(td_r),
                        one - bd_g.saturating_sub(td_g),
                        one - bd_b.saturating_sub(td_b),
                    ),
                    // Multiply where the bottom is dark (darkness past the pivot): its colour is 2·cs·cb, so its darkness 1 − 2(1 − td)(1 − bd); screen where it is light: darkness 2·td·bd. Each branch stays within one by its own side of the pivot. They were the other way round until 2026-10-29, which let 2·td·bd on a dark bottom reach 2·ONE and carry into the next field, α included.
                    BlendMode::Overlay => (
                        if bd_r > pivot { one - ((2 * (one - td_r) * (one - bd_r)) >> D) } else { (2 * td_r * bd_r) >> D },
                        if bd_g > pivot { one - ((2 * (one - td_g) * (one - bd_g)) >> D) } else { (2 * td_g * bd_g) >> D },
                        if bd_b > pivot { one - ((2 * (one - td_b) * (one - bd_b)) >> D) } else { (2 * td_b * bd_b) >> D },
                    ),
                    BlendMode::Darken => (td_r.max(bd_r), td_g.max(bd_g), td_b.max(bd_b)),
                    BlendMode::Lighten => (td_r.min(bd_r), td_g.min(bd_g), td_b.min(bd_b)),
                };

                let nr = td_r + ((mr * consumed) >> D);
                let ng = td_g + ((mg * consumed) >> D);
                let nb = td_b + ((mb * consumed) >> D);
                let na = top_a + consumed;

                (na << (3 * D)) | (nr << (2 * D)) | (ng << D) | nb
            }
        }
    };
}

impl_under!(Argb8, 8);
impl_under!(G2t16, 16);

/// A composite G2t16 pixel as the nearest 8-bit one, the cross-check's yardstick: α to the nearest 8-bit fraction, and each channel un-premultiplied to the 16-bit code it holds at full α, taken to the 8-bit code whose light (VSF's own decoder, bucket bottom) is nearest, and premultiplied back by the 8-bit α.
#[cfg(test)]
pub(crate) fn narrow_to_8(p: G2t16) -> Argb8 {
    use vsf::colour::convert::{linearize_gamma2tanh_u16_f32, linearize_gamma2tanh_u8_f32};
    // The 256 8-bit lights and the nearest 8-bit code for every 16-bit code, built once.
    static NEAREST: std::sync::OnceLock<alloc::vec::Vec<u8>> = std::sync::OnceLock::new();
    let nearest = NEAREST.get_or_init(|| {
        let lights: alloc::vec::Vec<f64> = (0..=255u32).map(|c| linearize_gamma2tanh_u8_f32(c as u8) as f64).collect();
        (0..=0xFFFFu32)
            .map(|c16| {
                let light = linearize_gamma2tanh_u16_f32(c16 as u16) as f64;
                let up = lights.partition_point(|&l| l <= light).max(1) - 1;
                if up < 255 && lights[up + 1] - light < light - lights[up] { (up + 1) as u8 } else { up as u8 }
            })
            .collect()
    });
    let a16 = p.alpha();
    let a8 = (a16 + 128) / 257;
    if a16 == 0 {
        return Argb8::from_channels(0, p.channels().map(|d| (d + 128) >> 8));
    }
    let channel = |d: u32| {
        // Darkness past α holds no code (Multiply and Overlay can leave it there, at either depth); it narrows linearly, the α share of the `256·d + α` widening taken back off.
        if d > a16 {
            return ((d - a16 / 257) + 128) >> 8;
        }
        let code16 = (65535. - d as f64 * 65535. / a16 as f64).round() as usize;
        let code8 = nearest[code16] as u32;
        (((255 - code8) * a8) as f64 / 255.).round() as u32
    };
    Argb8::from_channels(a8, p.channels().map(channel))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_opaque_top_returns_top_unchanged() {
        // α=255 means fully opaque — early-out fires regardless of RGB bits.
        let top: Argb8 = 0xFF_AA_BB_CC;
        let bottom: Argb8 = 0xFF_11_22_33;
        for mode in [
            BlendMode::Normal,
            BlendMode::Multiply,
            BlendMode::Screen,
            BlendMode::Add,
            BlendMode::Subtract,
            BlendMode::Overlay,
            BlendMode::Darken,
            BlendMode::Lighten,
        ] {
            assert_eq!(top.under(bottom, mode), top, "mode={mode:?}");
        }
    }

    #[test]
    fn under_opaque_top_skips_bottom_read() {
        // top α==0xFF → early-out fires; bottom value is irrelevant.
        let top: Argb8 = 0xFF_AA_BB_CC;
        let r1 = top.under(0x00_00_00_00, BlendMode::Normal);
        let r2 = top.under(0xFF_FF_FF_FF, BlendMode::Normal);
        assert_eq!(r1, r2);
        assert_eq!(r1, top);
    }

    #[test]
    fn under_empty_top_opaque_black_yields_opaque_black() {
        // Canonical empty (0x00000000 = no opacity, no darkness) + opaque black (α=255, dark=255) → opaque black. consumed = 255. contrib = (255 × 255) >> 8 = 254. nr ≈ 254 (≈ 255 after rounding). new_α = 255.
        let top: Argb8 = 0x00000000;
        let bottom: Argb8 = 0xFF_FF_FF_FF;
        let result = top.under(bottom, BlendMode::Normal);
        let new_a = result >> 24;
        let nr = (result >> 16) & 0xFF;
        assert_eq!(new_a, 0xFF, "new_α expected 0xFF (opaque)");
        assert!(nr >= 0xFD, "nr expected ~0xFF, got {nr:#x}");
    }

    #[test]
    fn under_empty_top_opaque_mid_gray_yields_mid_gray() {
        // Empty + opaque mid-gray (α=255, dark=128 ≈ visible 127 ≈ mid-gray) → ~mid-gray. consumed = 255. contrib = (128 × 255) >> 8 = 127. nr ≈ 127.
        let top: Argb8 = 0x00000000;
        let bottom: Argb8 = 0xFF_80_80_80;
        let result = top.under(bottom, BlendMode::Normal);
        let new_a = result >> 24;
        let nr = (result >> 16) & 0xFF;
        assert_eq!(new_a, 0xFF);
        assert!(nr >= 0x7E && nr <= 0x80, "nr expected ~0x7F, got {nr:#x}");
    }

    #[test]
    fn under_empty_top_20pct_opaque_black_yields_20pct_black() {
        // Empty + 20% opaque pure black (α=51, dark=255) → 20% α, 20% darkness. consumed = (256 × 51) >> 8 = 51. contrib = (255 × 51) >> 8 = 50. new_α = 51.
        let top: Argb8 = 0x00000000;
        let bottom: Argb8 = 0x33_FF_FF_FF;
        let result = top.under(bottom, BlendMode::Normal);
        let new_a = result >> 24;
        let nr = (result >> 16) & 0xFF;
        assert_eq!(new_a, 0x33, "new_α expected 0x33 (~20%)");
        assert!(nr >= 0x31 && nr <= 0x33, "nr expected ~0x32, got {nr:#x}");
    }

    /// The trait's constants are the layout the rest of fluor spells out by hand today, so moving a kernel onto them changes nothing.
    #[test]
    fn argb8_constants_are_the_8_bit_layout() {
        assert_eq!(Argb8::DEPTH, 8);
        assert_eq!(<Argb8 as Pixel>::ONE, 255);
        assert_eq!(<Argb8 as Pixel>::ALPHA_SHIFT, 24);
        assert_eq!(<Argb8 as Pixel>::RGB_MASK, 0x00FF_FFFF);
        assert_eq!(<Argb8 as Pixel>::OPAQUE, 0xFF00_0000);
        assert_eq!(<Argb8 as Pixel>::EMPTY, 0);
    }

    #[test]
    fn argb8_fields_round_trip() {
        let p: Argb8 = 0x80_12_34_56;
        assert_eq!(p.alpha(), 0x80);
        assert_eq!(p.rgb(), 0x00_12_34_56);
        assert_eq!(p.with_alpha(0xFF), 0xFF_12_34_56);
        assert_eq!(p.with_alpha(0), 0x00_12_34_56);
        assert_eq!(<Argb8 as Pixel>::from_g2t8(p), p);
    }

    /// Q16 coverage lands as its top DEPTH bits at each depth, and the squircle weights' identity holds: `(v·2¹⁶) as u16 >> 8` is `(v·2⁸) as u8` for every f32 `v`, since the products differ by an exact power of two and both casts saturate at their top.
    #[test]
    fn from_cov_takes_the_top_bits_of_q16() {
        let rgb: Argb8 = 0x00_AB_CD_EF;
        for w in 0..=0xFFFFu32 {
            assert_eq!(<Argb8 as Pixel>::from_cov(rgb, w), ((w >> 8) << 24) | rgb, "weight {w}");
            assert_eq!(<G2t16 as Pixel>::from_cov(0x1234, w).alpha(), w);
        }
        assert_eq!(<Argb8 as Pixel>::from_cov(0xFF_AB_CD_EF, 1 << 15).alpha(), 128, "α in the colour is replaced, not kept");
        let mut v = -0.25f32;
        while v < 1.25 {
            assert_eq!(((v * 65536.) as u16 >> 8) as u8, (v * 256.) as u8, "v {v}");
            v += 1. / ((1 << 14) as f32);
        }
    }

    /// The edge ramp at each depth: 0 and 1 at the ends, saturating past them, and the 8-bit one exactly the old `(v × 255) as u8`.
    #[test]
    fn cov_saturates_at_each_depth() {
        assert_eq!(<Argb8 as Pixel>::cov(-3.), 0);
        assert_eq!(<Argb8 as Pixel>::cov(1.), 255);
        assert_eq!(<Argb8 as Pixel>::cov(7.), 255);
        assert_eq!(<G2t16 as Pixel>::cov(1.), 65535);
        assert_eq!(<G2t16 as Pixel>::cov(7.), 65535);
        assert_eq!(<G2t16 as Pixel>::cov(-0.5), 0);
        let mut v = -0.5f32;
        while v < 1.5 {
            assert_eq!(<Argb8 as Pixel>::cov(v), (v * 255.0) as u8 as u32);
            v += 1. / 1024.;
        }
    }

    /// The 16-bit layout is the 8-bit one widened, and `from_g2t8` keeps the light of every authored code: VSF's own codec sends 8-bit code `c`'s light to 16-bit code `c × 2⁸`, which in darkness is `256·d + 255`.
    #[test]
    fn g2t16_from_g2t8_is_vsfs_light() {
        use vsf::colour::convert::{delinearize_gamma2tanh_u16_f32, linearize_gamma2tanh_u8_f32};
        assert_eq!(G2t16::DEPTH, 16);
        assert_eq!(G2t16::ONE, 65535);
        assert_eq!(G2t16::ALPHA_SHIFT, 48);
        assert_eq!(G2t16::RGB_MASK, 0x0000_FFFF_FFFF_FFFF);
        assert_eq!(G2t16::OPAQUE, 0xFFFF_0000_0000_0000);
        for code in 0..=255u32 {
            let code16 = delinearize_gamma2tanh_u16_f32(linearize_gamma2tanh_u8_f32(code as u8)) as u32;
            assert_eq!(code16, code << 8, "vsf's 16-bit code for 8-bit code {code}");
            let d8 = 255 - code;
            let p = G2t16::from_g2t8(0xFF00_0000 | (d8 << 16) | ((d8 ^ 0x5A) << 8) | (255 - d8));
            assert_eq!(p.alpha(), 65535);
            assert_eq!(p.channels()[0], 65535 - code16, "darkness for code {code}");
            assert_eq!(p.channels()[1], ((d8 ^ 0x5A) << 8) | 0xFF);
            assert_eq!(p.channels()[2], (code << 8) | 0xFF);
        }
        for a in 0..=255u32 {
            assert_eq!(G2t16::from_g2t8(a << 24).alpha(), a * 65535 / 255, "α {a} keeps its fraction");
        }
        let p = G2t16::from_channels(0x1234, [0x5678, 0x9ABC, 0xDEF0]);
        assert_eq!(p, 0x1234_5678_9ABC_DEF0);
        assert_eq!(p.channels(), [0x5678, 0x9ABC, 0xDEF0]);
        assert_eq!(p.rgb().with_alpha(7), 0x0007_5678_9ABC_DEF0);
    }

    /// The 8-bit kernel exactly as it was before it moved into `impl_under!`, kept as the reference the macro's 8-bit stamp is held to (with Overlay's branches put the right way round, as the kernel's were, 2026-10-29).
    fn reference_under_8(top: u32, bottom: u32, mode: BlendMode) -> u32 {
        if top >= 0xFF000000 {
            return top;
        }
        let top_a = top >> 24;
        let bot_a = bottom >> 24;
        let consumed = ((256 - top_a) * bot_a) >> 8;
        let t = [(top >> 16) & 0xFF, (top >> 8) & 0xFF, top & 0xFF];
        let b = [(bottom >> 16) & 0xFF, (bottom >> 8) & 0xFF, bottom & 0xFF];
        let m = |td: u32, bd: u32| match mode {
            BlendMode::Normal => bd,
            BlendMode::Multiply => 0xFF - (((0xFF - td) * (0xFF - bd)) >> 8),
            BlendMode::Screen => (td * bd) >> 8,
            BlendMode::Add => (td + bd).saturating_sub(0xFF),
            BlendMode::Subtract => 0xFF - bd.saturating_sub(td),
            BlendMode::Overlay => if bd > 0x80 { 0xFF - ((2 * (0xFF - td) * (0xFF - bd)) >> 8) } else { (2 * td * bd) >> 8 },
            BlendMode::Darken => td.max(bd),
            BlendMode::Lighten => td.min(bd),
        };
        let n: [u32; 3] = core::array::from_fn(|i| t[i] + ((m(t[i], b[i]) * consumed) >> 8));
        ((top_a + consumed) << 24) | (n[0] << 16) | (n[1] << 8) | n[2]
    }

    const MODES: [BlendMode; 8] = [BlendMode::Normal, BlendMode::Multiply, BlendMode::Screen, BlendMode::Add, BlendMode::Subtract, BlendMode::Overlay, BlendMode::Darken, BlendMode::Lighten];

    /// A pixel that keeps the `dark ≤ α` invariant every buffer holds, from a xorshift state.
    fn invariant_pixel(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        let a = *s >> 24;
        let d = |sh: u32| (((*s >> sh) & 0xFF) * a) >> 8;
        (a << 24) | (d(16) << 16) | (d(8) << 8) | d(0)
    }

    /// The macro's 8-bit stamp is bit for bit the kernel it replaced, over every mode and a spread of pixels (all α, opaque and empty included).
    #[test]
    fn argb8_under_is_the_old_kernel() {
        let mut s = 0x2545_F491u32;
        for _ in 0..1 << 16 {
            let (top, bottom) = (invariant_pixel(&mut s), invariant_pixel(&mut s));
            for mode in MODES {
                assert_eq!(top.under(bottom, mode), reference_under_8(top, bottom, mode), "{top:08x} under {bottom:08x} {mode:?}");
            }
        }
        for a in 0..=255u32 {
            for mode in MODES {
                let (top, bottom) = (a << 24, 0xFF00_0000 | (a << 16) | a);
                assert_eq!(top.under(bottom, mode), reference_under_8(top, bottom, mode));
            }
        }
    }

    /// The 16-bit stamp tracks the 8-bit one: the same layers widened and composed at 16 bits, narrowed back to the nearest 8-bit codes ([`narrow_to_8`]), land within 3 steps per channel and 1 in α of the 8-bit composite.
    /// The bottom is a source colour, widened by `from_g2t8`; the top is an accumulated composite, whose darkness is (top code − code) × α, so its widening is `256·d + α` (`from_g2t8`'s `+ 255` is that at full α).
    /// Why 3 and not 1: the 8-bit kernel takes `>> 8` for `÷ 255` twice (`consumed` and the deposit) and floors twice, which leaves a dark deposit under partial cover up to 3 steps short of the exact composite (32 ÷ 255 of 232 is 2.3 short in `29221021 under f4e8a14a`); at 16 bits the same steps cost 1/256 of that, so the difference is the 8-bit kernel's, not the 16-bit one's.
    #[test]
    fn g2t16_under_tracks_argb8() {
        let widen_composite = |p: u32| G2t16::from_channels(p.alpha() * 257, p.channels().map(|d| (d << 8) + p.alpha()));
        let mut s = 0x9E37_79B9u32;
        for _ in 0..1 << 14 {
            let (top, bottom) = (invariant_pixel(&mut s), invariant_pixel(&mut s));
            for mode in MODES {
                let p8 = top.under(bottom, mode);
                let n = narrow_to_8(widen_composite(top).under(G2t16::from_g2t8(bottom), mode));
                assert!(n.alpha().abs_diff(p8.alpha()) <= 1, "{mode:?} α {} vs {}", n.alpha(), p8.alpha());
                for (c16, c8) in n.channels().into_iter().zip(p8.channels()) {
                    assert!(c16.abs_diff(c8) <= 3, "{mode:?} {top:08x} under {bottom:08x}: {c16} vs {c8}");
                }
            }
        }
    }

    /// The 16-bit chunk kernels (SIMD with the `simd` feature, the scalar loop without) are the scalar `under` pixel for pixel, over runs that are not a whole number of lanes, opaque and empty tops included.
    #[test]
    fn g2t16_chunks_are_the_scalar_kernel() {
        let mut s = 0x1234_5678u32;
        let mut widened = || {
            let p = invariant_pixel(&mut s);
            G2t16::from_channels(p.alpha() * 257, p.channels().map(|d| (d << 8) + p.alpha()))
        };
        for n in [0usize, 1, 7, 8, 9, 31, 64, 67] {
            let mut dst: Vec<G2t16> = (0..n).map(|_| widened()).collect();
            let src: Vec<G2t16> = (0..n).map(|_| widened()).collect();
            if n > 2 {
                dst[1] = G2t16::OPAQUE | 0x1234;
                dst[2] = G2t16::EMPTY;
            }
            let want: Vec<G2t16> = dst.iter().zip(&src).map(|(&d, &s)| d.under(s, BlendMode::Normal)).collect();
            let mut got = dst.clone();
            G2t16::under_chunk_normal(&mut got, &src);
            assert_eq!(got, want, "run of {n}");
            if n > 0 {
                let want: Vec<G2t16> = dst.iter().map(|&d| d.under(src[0], BlendMode::Normal)).collect();
                let mut got = dst.clone();
                G2t16::under_chunk_const(&mut got, src[0]);
                assert_eq!(got, want, "const run of {n}");
            }
        }
    }

    /// An 8-bit α widens to the same fraction `from_g2t8` gives it.
    #[test]
    fn alpha_from8_is_from_g2t8s_alpha() {
        for a in 0..=255u32 {
            assert_eq!(Argb8::alpha_from8(a), a);
            assert_eq!(G2t16::alpha_from8(a), G2t16::from_g2t8(a << 24).alpha());
        }
    }

    #[test]
    fn under_two_translucent_layers_accumulates_opacity() {
        // Two 50%-α opaque-white layers (dark=0 to preserve the `dark ≤ α` invariant): top α=128 dark=0, bot α=128 dark=0 → consumed = (128 × 128) >> 8 = 64, new_α = 128 + 64 = 192. Porter-Duff over: 1 − (1 − 0.5)(1 − 0.5) = 0.75 ≈ 192/255.
        let top: Argb8 = 0x80_00_00_00;
        let bottom: Argb8 = 0x80_00_00_00;
        let result = top.under(bottom, BlendMode::Normal);
        let new_a = result >> 24;
        assert_eq!(new_a, 192);
    }
}
