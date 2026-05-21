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
//! Every compositing op in fluor — Normal source-over, Multiply, Screen, Add, Subtract, Overlay, Darken, Lighten — flows through one trait method: `top.under(bottom, mode)`. `top` is the partial composite already accumulated above (its t-byte = remaining transparency budget). `bottom` is the new layer going behind. The mode shapes only how `bottom`'s RGB is interpreted; the outer transparency-budget math is identical across modes. One early-out (`top < 0x01000000`), one t-attenuation (`(top_t * bot_t) >> 8`), one channel-blend pattern (`(tr * top_opacity + mr * contrib) >> 8`). All `>> 8`, no `/ 255`, no floats.
//!
//! Invariant: `top_opacity + contrib ≤ 256`, so `(tr * top_opacity + mr * contrib) >> 8 ≤ 255` exactly when both RGB inputs are u8 — no per-channel saturation needed in the hot loop.

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
    /// Compose `bottom` *underneath* `self` (the partial composite from layers above). `self`'s t-byte is the remaining transparency budget; `bottom` contributes proportional to that budget through the chosen `mode`. Both operands are straight-α (RGB is the intrinsic colour, not premultiplied); the OS conversion layer handles any platform-specific premultiplication at the boundary.
    ///
    /// Single early-out: `self < 0x01000000` ⇔ `self.t == 0` (top opaque). Returns `self` unchanged — bottom is invisible behind a fully-opaque top. One u32 compare against an immediate.
    ///
    /// Math (all `>> 8`, no `/ 255`, no floats):
    /// ```text
    /// top_opacity = 256 - top_t                      (1..=256)
    /// contrib     = (top_t * (256 - bot_t)) >> 8    (0..=255)
    /// (mr, mg, mb) = mode_kernel(top_rgb, bot_rgb)   (0..=255 per channel)
    /// out_ch      = (top_ch * top_opacity + m_ch * contrib) >> 8
    /// new_t       = (top_t * bot_t) >> 8
    /// ```
    /// The invariant `top_opacity + contrib ≤ 256` keeps `out_ch ≤ 255` without explicit saturation.
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

        let top_opacity = 256 - top_t;
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

        let nr = (tr * top_opacity + mr * contrib) >> 8;
        let ng = (tg * top_opacity + mg * contrib) >> 8;
        let nb = (tb * top_opacity + mb * contrib) >> 8;

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
    fn under_normal_transparent_top_opaque_bottom_yields_bottom() {
        // top_t=255 (full budget) + bot_t=0 (opaque) → new_t=0, RGB ≈ bottom RGB.
        let top: Argb8 = 0xFF_00_00_00;
        let bottom: Argb8 = 0x00_FE_FE_FE;
        let result = top.under(bottom, BlendMode::Normal);
        // top_opacity=1, contrib=(255*256)>>8=255. nr = (0*1 + 0xFE*255) >> 8 = 64770>>8 = 253.
        let nr = (result >> 16) & 0xFF;
        let new_t = result >> 24;
        assert_eq!(new_t, 0, "new_t expected 0, got {new_t:#x}");
        assert!(nr >= 0xFD && nr <= 0xFE, "nr expected ~0xFE, got {nr:#x}");
    }

    #[test]
    fn under_normal_half_top_opaque_bottom_blends_50_50() {
        // top_t=128 → top_opacity=128, contrib=(128*256)>>8=128. Result is 50/50 of top/bottom RGB.
        let top: Argb8 = 0x80_FF_00_00;
        let bottom: Argb8 = 0x00_00_00_FF;
        let result = top.under(bottom, BlendMode::Normal);
        let nr = (result >> 16) & 0xFF;
        let nb = result & 0xFF;
        let new_t = result >> 24;
        assert_eq!(new_t, 0, "new_t expected 0 (opaque bottom kills budget)");
        // nr = (0xFF * 128 + 0 * 128) >> 8 = 32640 >> 8 = 127
        // nb = (0 * 128 + 0xFF * 128) >> 8 = 127
        assert!(nr >= 0x7E && nr <= 0x80, "nr expected ~0x7F, got {nr:#x}");
        assert!(nb >= 0x7E && nb <= 0x80, "nb expected ~0x7F, got {nb:#x}");
    }

    #[test]
    fn under_two_translucent_layers_attenuates_budget() {
        // top_t=128, bot_t=128 → new_t = (128*128)>>8 = 64. Multi-layer translucency composes.
        let top: Argb8 = 0x80_00_00_00;
        let bottom: Argb8 = 0x80_FF_00_00;
        let result = top.under(bottom, BlendMode::Normal);
        let new_t = result >> 24;
        assert_eq!(new_t, 64);
    }

    #[test]
    fn under_multiply_darkens() {
        // Multiply: mr = (tr * br) >> 8. With top opaque white over bottom opaque mid-gray, result ≈ mid-gray.
        // But top opaque → early-out, so use translucent top.
        let top: Argb8 = 0x80_FF_FF_FF; // 50% transparent white
        let bottom: Argb8 = 0x00_80_80_80; // opaque mid gray
        let result = top.under(bottom, BlendMode::Multiply);
        // top_opacity=128, contrib=(128*256)>>8=128
        // mr = (0xFF * 0x80) >> 8 = 127
        // nr = (0xFF * 128 + 127 * 128) >> 8 = (32640 + 16256) >> 8 = 191
        let nr = (result >> 16) & 0xFF;
        assert!(
            nr >= 0xBE && nr <= 0xC0,
            "Multiply result nr expected ~0xBF, got {nr:#x}"
        );
    }
}
