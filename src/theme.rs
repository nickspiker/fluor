//! Theme constants — colors and geometry shared between paint primitives, chrome, and (eventually) widgets. Sourced verbatim from photon's [theme.rs](/mnt/Octopus/Code/photon/src/ui/theme.rs); names match photon exactly so cross-codebase greps work.
//!
//! All colors are u32 in packed ARGB format `0xAARRGGBB`. Android byte-swap (ARGB → ABGR) lives behind a cfg here so call sites stay platform-neutral.

#[cfg(target_os = "android")]
const fn fmt(argb: u32) -> u32 {
    let a = (argb >> 24) & 0xFF;
    let r = (argb >> 16) & 0xFF;
    let g = (argb >> 8) & 0xFF;
    let b = argb & 0xFF;
    (a << 24) | (b << 16) | (g << 8) | r
}

#[cfg(not(target_os = "android"))]
const fn fmt(argb: u32) -> u32 { argb }

// Background texture (organic noise, scrollable).
pub const BG_BASE: u32 = fmt(0xFF_0C_14_0E);
pub const BG_MASK: u32 = fmt(0xFF_0F_07_1F);
pub const BG_ALPHA: u32 = fmt(0xFF_00_00_00);
pub const BG_SPECKLE: u32 = fmt(0x00_3F_1F_7F);

// Window edges + controls background.
pub const WINDOW_LIGHT_EDGE: u32 = fmt(0xFF_44_41_37);
pub const WINDOW_SHADOW_EDGE: u32 = fmt(0xFF_2B_34_37);
pub const WINDOW_CONTROLS_BG: u32 = fmt(0xFF_1E_1E_1E);
pub const WINDOW_CONTROLS_HAIRLINE: u32 = fmt(0xFF_44_41_37);

// Button hover deltas (RGB channels wrap intentionally; 0xFF alpha absorbs carry).
pub const CLOSE_HOVER: u32 = fmt(0xFF_21_FD_F9);
pub const MAXIMIZE_HOVER: u32 = fmt(0xFF_FA_10_FA);
pub const MINIMIZE_HOVER: u32 = fmt(0xFF_F7_FA_25);

// Generic UI text.
pub const TEXT_COLOUR: u32 = fmt(0xFF_D0_D0_D0);
pub const LABEL_COLOUR: u32 = fmt(0xFF_80_80_80);

// Window control glyph colours (drawn on top of WINDOW_CONTROLS_BG).
pub const CLOSE_GLYPH: u32 = fmt(0xFF_80_20_20);
pub const MAXIMIZE_GLYPH: u32 = fmt(0xFF_48_6B_3A);
pub const MAXIMIZE_GLYPH_INTERIOR: u32 = fmt(0xFF_28_2D_2E);
pub const MINIMIZE_GLYPH: u32 = fmt(0xFF_33_30_C7);
