//! The ferros display's working values from a fluor scratch buffer: the present the ferros host runs, at either pixel depth.
//! The display composes 16-bit surfaces (`Px` u64: R, G, B in bits 0, 16 and 32 as gamma 2 with white at 2¹⁴; bit 63 = see-through), so every depth's present ends in the same place: each channel's `g2t` code to `16384 × atanh(code / 2ⁿ)`, VSF's decode at the bucket's bottom (light = atanh(code / 2ⁿ)², and gamma 2 is its root).
//! A pixel's α composites it the way the 8-bit present always has: the visible code `(top − darkness)`, scaled by α.
//! This is the stopgap until the display reads G2t16 surfaces itself (the plan's phase 5: a surface format flag at op 22, the display's own g2t decode), when the 16-bit present becomes a copy.

use crate::pixel::{Argb8, G2t16, Pixel};

/// The display's see-through bit: a pixel the window does not cover, so whatever is behind it shows.
pub const SEE_THROUGH: u64 = 1 << 63;

/// A pixel depth the ferros display can be fed: one scratch row into a row of the display's working values.
pub trait PanelPixel: Pixel {
    /// `dst[i]` from `src[i]`; the slices are the same length.
    fn to_panel(src: &[Self], dst: &mut [u64]);
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

/// 16-bit `g2t` code to the display's gamma-2 Q14 working value: `16384 × atanh(c / 65536)`, every code, 128 KiB built once.
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

impl PanelPixel for Argb8 {
    /// The 8-bit present as it was: the visible code over the pixel's α (rounded), thru the 256-entry table. A transparent pixel is black, as before (no see-through bit at this depth: the 8-bit shell keeps its present unchanged).
    fn to_panel(src: &[Self], dst: &mut [u64]) {
        let lut = q14_from_g2t8();
        for (d, &px) in dst.iter_mut().zip(src.iter()) {
            let a = px.alpha();
            let [r, g, b] = px.channels().map(|c| ((Argb8::ONE - c) * a + 127) / 255);
            *d = lut[r as usize] as u64 | (lut[g as usize] as u64) << 16 | (lut[b as usize] as u64) << 32;
        }
    }
}

impl PanelPixel for G2t16 {
    /// The same composite at 16 bits, thru the 2¹⁶-entry table; a pixel with no α at all is see-through, so the window takes its own shape.
    fn to_panel(src: &[Self], dst: &mut [u64]) {
        let lut = q14_from_g2t16();
        for (d, &px) in dst.iter_mut().zip(src.iter()) {
            let a = px.alpha();
            if a == 0 {
                *d = SEE_THROUGH;
                continue;
            }
            // (65535 − d) × α + 32767 peaks at 4 294 868 992, inside a u32.
            let [r, g, b] = px.channels().map(|c| ((G2t16::ONE - c) * a + (G2t16::ONE >> 1)) / G2t16::ONE);
            *d = lut[r as usize] as u64 | (lut[g as usize] as u64) << 16 | (lut[b as usize] as u64) << 32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The same opaque colour presents alike at both depths, a half-covered one within the rounding of its α, and an empty 16-bit pixel is see-through.
    #[test]
    fn both_depths_present_alike() {
        let colours: [u32; 4] = [crate::theme::WHITE, crate::theme::BUTTON_FILL, crate::theme::TEXTBOX_SELECTION_BG, 0xFF00_0000];
        for c in colours {
            let (mut d8, mut d16) = ([0u64; 1], [0u64; 1]);
            Argb8::to_panel(&[c], &mut d8);
            G2t16::to_panel(&[G2t16::from_g2t8(c)], &mut d16);
            for sh in [0, 16, 32] {
                let (v8, v16) = ((d8[0] >> sh) & 0xFFFF, (d16[0] >> sh) & 0xFFFF);
                assert!(v8.abs_diff(v16) <= 1, "{c:08x} channel at {sh}: {v8} vs {v16}");
            }
        }
        let mut d = [0u64; 1];
        G2t16::to_panel(&[G2t16::EMPTY], &mut d);
        assert_eq!(d[0], SEE_THROUGH);
    }
}
