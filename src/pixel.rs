//! Pixel format + blend trait.
//!
//! # Convention — locked
//!
//! **Internal pixel format**: `0xααRRGGBB` packed in a `u32` — top byte is **α (opacity)**; RGB bytes are **darkness** (`0 = white potential`, `255 = black ink`). LE byte layout `[B, G, R, α]`. Every paint primitive, every layer buffer, every Group composite uses this. `Argb8` is a type alias for `u32`; the byte layout matches a typical compositor pixel, but the RGB bytes are the *bitwise complement* of visible RGB.
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
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8;
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

impl Blend for Argb8 {
    #[inline]
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8 {
        if self >= 0xFF000000 {
            return self;
        } // top opaque — top wins, bottom invisible.
        let top_a = self >> 24;
        let bot_a = bottom >> 24;

        let consumed = ((256 - top_a) * bot_a) >> 8;

        let td_r = (self >> 16) & 0xFF;
        let td_g = (self >> 8) & 0xFF;
        let td_b = self & 0xFF;
        let bd_r = (bottom >> 16) & 0xFF;
        let bd_g = (bottom >> 8) & 0xFF;
        let bd_b = bottom & 0xFF;

        let (mr, mg, mb) = match mode {
            BlendMode::Normal => (bd_r, bd_g, bd_b),
            BlendMode::Multiply => (
                0xFF - (((0xFF - td_r) * (0xFF - bd_r)) >> 8),
                0xFF - (((0xFF - td_g) * (0xFF - bd_g)) >> 8),
                0xFF - (((0xFF - td_b) * (0xFF - bd_b)) >> 8),
            ),
            BlendMode::Screen => ((td_r * bd_r) >> 8, (td_g * bd_g) >> 8, (td_b * bd_b) >> 8),
            BlendMode::Add => (
                (td_r + bd_r).saturating_sub(0xFF),
                (td_g + bd_g).saturating_sub(0xFF),
                (td_b + bd_b).saturating_sub(0xFF),
            ),
            BlendMode::Subtract => (
                0xFF - bd_r.saturating_sub(td_r),
                0xFF - bd_g.saturating_sub(td_g),
                0xFF - bd_b.saturating_sub(td_b),
            ),
            BlendMode::Overlay => (
                if bd_r > 0x80 {
                    (2 * td_r * bd_r) >> 8
                } else {
                    0xFF - ((2 * (0xFF - td_r) * (0xFF - bd_r)) >> 8)
                },
                if bd_g > 0x80 {
                    (2 * td_g * bd_g) >> 8
                } else {
                    0xFF - ((2 * (0xFF - td_g) * (0xFF - bd_g)) >> 8)
                },
                if bd_b > 0x80 {
                    (2 * td_b * bd_b) >> 8
                } else {
                    0xFF - ((2 * (0xFF - td_b) * (0xFF - bd_b)) >> 8)
                },
            ),
            BlendMode::Darken => (td_r.max(bd_r), td_g.max(bd_g), td_b.max(bd_b)),
            BlendMode::Lighten => (td_r.min(bd_r), td_g.min(bd_g), td_b.min(bd_b)),
        };

        let nr = td_r + ((mr * consumed) >> 8);
        let ng = td_g + ((mg * consumed) >> 8);
        let nb = td_b + ((mb * consumed) >> 8);
        let na = top_a + consumed;

        (na << 24) | (nr << 16) | (ng << 8) | nb
    }
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
