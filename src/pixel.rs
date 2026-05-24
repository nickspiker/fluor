//! Pixel format + blend trait.
//!
//! # Convention — locked
//!
//! **Internal pixel format**: `0xttRRGGBB` packed in a `u32` — top byte is **transparency**, not alpha. LE byte layout `[B, G, R, t]`. Every paint primitive, every layer buffer, every Group composite uses this. `Argb8` is a type alias for `u32`; the byte layout hasn't changed, only the top byte's interpretation.
//!
//! **Transparency semantics**: `t = 0` = fully opaque, `t = 255` = (almost) fully transparent. Variables in code are named `t` / `transparency` — never `alpha` or `a` — so the convention is unambiguous at every read site. RGB channels store the straight (non-premultiplied) intrinsic color.
//!
//! **Why t-convention** (mirroring the README): the hot-loop early-out fires precisely when a fully-opaque layer is hit, because `remaining = (remaining * t) >> 8 = 0` exactly when `t = 0`. Single u32 compare against an immediate detects opacity classes without any shift or mask: `if pixel < 0x01000000` is "this pixel is fully opaque" in one CMP. Sort order primary-sorts by transparency for free.
//!
//! **The 256-vs-255 gap**: fully-transparent layers never enter the blend pass — they're culled at the caller level (you don't draw an invisible layer). Per-pixel `t = 255` within a partial-transparency layer contributes exact 0 to RGB and attenuates `remaining` by `255/256`; the 1-LSB drift is invisible at expected stacking depths. We never need to represent `t = 256` and the missing slot is a happy coincidence.
//!
//! **Boundary**: paint primitives write `t` directly. The host present pass flips `t → α` (XOR top byte with `0xFF`) right before submitting to wgpu / softbuffer — folded into the existing `premultiply_buffer` step on Linux. External inputs (cosmic-text glyph coverage, `pack_argb`) flip once at the import boundary, never per-frame.
//!
//! # The unified `Blend::under` kernel
//!
//! Every compositing op in fluor — Normal source-under, Multiply, Screen, Add, Subtract, Overlay, Darken, Lighten — flows through one trait method: `top.under(bottom, mode)`. `top` is the partial composite already accumulated above (its t-byte = remaining transparency budget; its RGB = the "visible-over-white" running color, with the canonical empty value `0xFFFFFFFF` representing "no paint yet, FF-white default fill in the remaining budget"). `bottom` is the new layer going behind. The mode shapes only how `bottom`'s RGB is interpreted.
//!
//! Convention: each new layer DARKENS the buffer's RGB from white by `(255 - layer_rgb) * consumed >> 8` per channel, where `consumed = (top_t * (256 - bot_t)) >> 8` (how much of the remaining budget the new layer fills). The transparency budget attenuates as `new_t = (top_t * bot_t) >> 8`. This way, the buffer's RGB always represents the visible color assuming the unfilled portion is white — preserving the `0xFFFFFFFF` invariant.

/// Packed pixel in t-convention: `0xttRRGGBB`. Type alias for `u32` — `&[Argb8]` and `&[u32]` are the same slice. The name carries the convention contract, not a new type.
pub type Argb8 = u32;

/// How `bottom`'s RGB is mixed into the partial composite `top` when composing front-to-back.
///
/// Each mode is a pure channel-wise function of `(top_rgb, bottom_rgb)` that produces a "modulated bottom" `(mr, mg, mb)`. The transparency-budget math around it is identical across all modes — see [`Blend::under`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Source-under: bottom passes through unchanged. `mr = br`. Standard front-to-back compositing.
    Normal,
    /// Multiplicative darkening per channel: `mr = (tr * br) >> 8`. Bottom is "filtered through" top's intensity.
    Multiply,
    /// Inverse-multiply (Photoshop screen): `mr = 255 - (((255 - tr) * (255 - br)) >> 8)`. Brightens.
    Screen,
    /// Saturating per-channel add: `mr = (tr + br).min(255)`. Additive light.
    Add,
    /// Saturating per-channel subtract: `mr = tr.saturating_sub(br)`. Darkening light.
    Subtract,
    /// Contrast-shifting Photoshop overlay: multiply where bottom is dark, screen where bottom is bright. Pivot at 0x80.
    Overlay,
    /// Per-channel min: `mr = tr.min(br)`. Keeps the darker channel.
    Darken,
    /// Per-channel max: `mr = tr.max(br)`. Keeps the brighter channel.
    Lighten,
}

/// The single compositing operation in fluor. Front-to-back source-under with selectable blend mode.
pub trait Blend {
    /// Compose `bottom` *underneath* `self` (the partial composite from layers above). `self`'s t-byte is the remaining transparency budget; `self`'s RGB is the visible-over-white running color (canonical empty value `0xFFFFFFFF` = "FF-white fill in the remaining budget").
    ///
    /// Single early-out: `self < 0x01000000` ⇔ `self.t == 0` (top opaque). Returns `self` unchanged — bottom is invisible behind a fully-opaque top.
    ///
    /// Math (all `>> 8`, no `/ 255`, no floats):
    /// ```text
    /// consumed     = (top_t * (256 - bot_t)) >> 8     (0..=255) — how much of the remaining budget the new layer fills
    /// (mr, mg, mb) = mode_kernel(top_rgb, bot_rgb)    (0..=255 per channel)
    /// delta_ch     = ((255 - m_ch) * consumed) >> 8   (loss of FF-white from new layer's darkening)
    /// out_ch       = top_ch.saturating_sub(delta_ch)
    /// new_t        = (top_t * bot_t) >> 8
    /// ```
    /// Starting from `0xFFFFFFFF` (white potential, full budget), each new layer subtracts `(255 - layer_color) × consumed / 256` per channel — the loss of white as the layer fills part of the budget. Multiple stacked layers compose by repeatedly attenuating from the running white potential.
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8;
}

impl Blend for Argb8 {
    #[inline]
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8 {
        if self < 0x01000000 {
            return self;
        } // top opaque — top wins, bottom invisible.
        let top_t = self >> 24;
        let bot_t = bottom >> 24;

        let contrib = (top_t * (256 - bot_t)) >> 8;

        let tr = (self >> 16) & 0xFF;
        let tg = (self >> 8) & 0xFF;
        let tb = self & 0xFF;
        let br = (bottom >> 16) & 0xFF;
        let bg = (bottom >> 8) & 0xFF;
        let bb = bottom & 0xFF;

        let (mr, mg, mb) = match mode {
            BlendMode::Normal => (br, bg, bb),
            BlendMode::Multiply => ((tr * br) >> 8, (tg * bg) >> 8, (tb * bb) >> 8),
            BlendMode::Screen => (
                0xFF - (((0xFF - tr) * (0xFF - br)) >> 8),
                0xFF - (((0xFF - tg) * (0xFF - bg)) >> 8),
                0xFF - (((0xFF - tb) * (0xFF - bb)) >> 8),
            ),
            BlendMode::Add => (
                (tr + br).min(0xFF),
                (tg + bg).min(0xFF),
                (tb + bb).min(0xFF),
            ),
            BlendMode::Subtract => (
                tr.saturating_sub(br),
                tg.saturating_sub(bg),
                tb.saturating_sub(bb),
            ),
            BlendMode::Overlay => (
                if br < 0x80 {
                    (2 * tr * br) >> 8
                } else {
                    0xFF - ((2 * (0xFF - tr) * (0xFF - br)) >> 8)
                },
                if bg < 0x80 {
                    (2 * tg * bg) >> 8
                } else {
                    0xFF - ((2 * (0xFF - tg) * (0xFF - bg)) >> 8)
                },
                if bb < 0x80 {
                    (2 * tb * bb) >> 8
                } else {
                    0xFF - ((2 * (0xFF - tb) * (0xFF - bb)) >> 8)
                },
            ),
            BlendMode::Darken => (tr.min(br), tg.min(bg), tb.min(bb)),
            BlendMode::Lighten => (tr.max(br), tg.max(bg), tb.max(bb)),
        };

        // The new layer darkens `top` from FF-white by `(255 - layer) * consumed >> 8` per channel.
        let delta_r = ((255 - mr) * contrib) >> 8;
        let delta_g = ((255 - mg) * contrib) >> 8;
        let delta_b = ((255 - mb) * contrib) >> 8;

        let nr = tr.saturating_sub(delta_r);
        let ng = tg.saturating_sub(delta_g);
        let nb = tb.saturating_sub(delta_b);

        ((top_t * bot_t) >> 8) << 24 | (nr << 16) | (ng << 8) | nb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn under_opaque_top_returns_top_unchanged() {
        let top: Argb8 = 0x00_AA_BB_CC;
        let bottom: Argb8 = 0x00_11_22_33;
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
        // top_t == 0 → early-out fires; bottom value is irrelevant.
        let top: Argb8 = 0x00_AA_BB_CC;
        let r1 = top.under(0x00_00_00_00, BlendMode::Normal);
        let r2 = top.under(0xFF_FF_FF_FF, BlendMode::Normal);
        assert_eq!(r1, r2);
        assert_eq!(r1, top);
    }

    #[test]
    fn under_empty_top_opaque_black_yields_opaque_black() {
        // Canonical empty (0xFFFFFFFF = white potential, full budget) + opaque black under → opaque black.
        // consumed = 255. delta = (255 - 0) * 255 >> 8 = 254. nr = 255 - 254 = 1 (≈ 0).
        let top: Argb8 = 0xFFFFFFFF;
        let bottom: Argb8 = 0x00_00_00_00;
        let result = top.under(bottom, BlendMode::Normal);
        let new_t = result >> 24;
        let nr = (result >> 16) & 0xFF;
        assert_eq!(new_t, 0, "new_t expected 0 (opaque bottom)");
        assert!(nr <= 0x02, "nr expected ~0x00, got {nr:#x}");
    }

    #[test]
    fn under_empty_top_opaque_mid_gray_yields_mid_gray() {
        // Empty + opaque mid-gray under → ~mid-gray.
        // consumed = 255. delta = (255 - 128) * 255 >> 8 = 127. nr = 255 - 127 = 128.
        let top: Argb8 = 0xFFFFFFFF;
        let bottom: Argb8 = 0x00_80_80_80;
        let result = top.under(bottom, BlendMode::Normal);
        let new_t = result >> 24;
        let nr = (result >> 16) & 0xFF;
        assert_eq!(new_t, 0);
        assert!(nr >= 0x7F && nr <= 0x81, "nr expected ~0x80, got {nr:#x}");
    }

    #[test]
    fn under_empty_top_20pct_mid_gray_preserves_white() {
        // Empty + 20% mid-gray (t=0xCD ≈ 205) → ~0xCC_E6_E6_E6 (mostly white, slightly darkened).
        // consumed = (255 * 51) >> 8 = 50. delta = (255 - 128) * 50 >> 8 = 24. nr = 255 - 24 = 231 ≈ 0xE7.
        let top: Argb8 = 0xFFFFFFFF;
        let bottom: Argb8 = 0xCD_80_80_80;
        let result = top.under(bottom, BlendMode::Normal);
        let new_t = result >> 24;
        let nr = (result >> 16) & 0xFF;
        assert!(new_t >= 0xCB && new_t <= 0xCD, "new_t expected ~0xCC, got {new_t:#x}");
        assert!(nr >= 0xE4 && nr <= 0xE8, "nr expected ~0xE6, got {nr:#x}");
    }

    #[test]
    fn under_two_translucent_layers_attenuates_budget() {
        // top_t=128, bot_t=128 → new_t = (128*128)>>8 = 64. Multi-layer translucency composes.
        let top: Argb8 = 0x80_FF_FF_FF;
        let bottom: Argb8 = 0x80_FF_FF_FF;
        let result = top.under(bottom, BlendMode::Normal);
        let new_t = result >> 24;
        assert_eq!(new_t, 64);
    }
}
