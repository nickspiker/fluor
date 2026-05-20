//! Theme constants — colours and geometry shared between paint primitives, chrome, and (eventually) widgets. Sourced from photon's [theme.rs](/mnt/Octopus/Code/photon/src/ui/theme.rs); names match photon exactly so cross-codebase greps work.
//!
//! Colour layout: `0xttRRGGBB` (fluor's t-convention internal format — top byte is transparency, `t=0` opaque). The `0x00_xx_yy_zz` constants here are all opaque. Android byte-swap (RGB byte-flip) lives behind a cfg here so call sites stay platform-neutral.

#[cfg(target_os = "android")]
const fn fmt(trgb: u32) -> u32 {
    let t = (trgb >> 24) & 0xFF;
    let r = (trgb >> 16) & 0xFF;
    let g = (trgb >> 8) & 0xFF;
    let b = trgb & 0xFF;
    (t << 24) | (b << 16) | (g << 8) | r
}

#[cfg(not(target_os = "android"))]
const fn fmt(trgb: u32) -> u32 { trgb }

// Background texture (organic noise, scrollable). All opaque (t=0).
pub const BG_BASE: u32 = fmt(0x00_0C_14_0E);
pub const BG_MASK: u32 = fmt(0x00_0F_07_1F);
pub const BG_SPECKLE: u32 = fmt(0x00_3F_1F_7F);

// Window edges + controls background.
pub const WINDOW_LIGHT_EDGE: u32 = fmt(0x00_44_41_37);
pub const WINDOW_SHADOW_EDGE: u32 = fmt(0x00_2B_34_37);
pub const WINDOW_CONTROLS_BG: u32 = fmt(0x00_1E_1E_1E);
pub const WINDOW_CONTROLS_HAIRLINE: u32 = fmt(0x00_44_41_37);

// Button hover deltas (RGB channels wrap intentionally; the t-byte is forced opaque after).
pub const CLOSE_HOVER: u32 = fmt(0x00_21_FD_F9);
pub const MAXIMIZE_HOVER: u32 = fmt(0x00_FA_10_FA);
pub const MINIMIZE_HOVER: u32 = fmt(0x00_F7_FA_25);

// Generic UI text.
pub const TEXT_COLOUR: u32 = fmt(0x00_D0_D0_D0);
pub const LABEL_COLOUR: u32 = fmt(0x00_80_80_80);

// Window control glyph colours (drawn on top of WINDOW_CONTROLS_BG).
pub const CLOSE_GLYPH: u32 = fmt(0x00_80_20_20);
pub const MAXIMIZE_GLYPH: u32 = fmt(0x00_48_6B_3A);
pub const MAXIMIZE_GLYPH_INTERIOR: u32 = fmt(0x00_28_2D_2E);
pub const MINIMIZE_GLYPH: u32 = fmt(0x00_33_30_C7);

// Textbox.
pub const TEXTBOX_FILL: u32 = fmt(0x00_06_08_09);
pub const TEXTBOX_HOVER: u32 = fmt(0x00_12_16_18);
pub const TEXTBOX_ACTIVE: u32 = fmt(0x00_00_00_00);
pub const TEXTBOX_LIGHT_EDGE: u32 = fmt(0x00_44_41_37);
pub const TEXTBOX_SHADOW_EDGE: u32 = fmt(0x00_2B_34_37);

// Cursor (blinkey).
pub const CURSOR_BRIGHTNESS: f32 = 100.0;

// Textbox glow colours (RGB only — t-byte is set per-pixel by the glow function).
pub const GLOW_DEFAULT: u32 = fmt(0x00_FF_FF_FF);
pub const GLOW_SUCCESS: u32 = fmt(0x00_40_FF_40);
pub const GLOW_ERROR: u32 = fmt(0x00_FF_60_60);
