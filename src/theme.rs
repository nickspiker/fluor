//! Theme constants — colours and geometry shared between paint primitives, chrome, and (eventually) widgets. Sourced from photon's [theme.rs](/mnt/Octopus/Code/photon/src/ui/theme.rs); names match photon exactly so cross-codebase greps work.
//!
//! Colour layout: `0xααRRGGBB` (fluor internal format — top byte is α opacity, `α=0xFF` opaque; RGB bytes are DARKNESS, `0 = white potential, 255 = black`). The `0x00_xx_yy_zz` literals here are written with `t = 0` (which would have been opaque under the old convention) and visible RGB; [`fmt`] handles platform byte-swap and [`dark`] inverts the RGB at compile time. Constants here keep `α = 0` — colour palette only; the α-byte is set per-pixel at the use site (`opacity << 24 | dark(theme_const)`).
//!
//! At the OS boundary, [`crate::paint::finalize_for_os`] does a single `pixel ^= 0x00FFFFFF` that flips RGB darkness back to visible; α passes through (already opacity-direction in storage) — putting the pixel in the format the host compositor wants.

#[cfg(target_os = "android")]
const fn fmt(trgb: u32) -> u32 {
    let t = (trgb >> 24) & 0xFF;
    let r = (trgb >> 16) & 0xFF;
    let g = (trgb >> 8) & 0xFF;
    let b = trgb & 0xFF;
    (t << 24) | (b << 16) | (g << 8) | r
}

#[cfg(not(target_os = "android"))]
const fn fmt(trgb: u32) -> u32 {
    trgb
}

/// Compile-time visible-RGB → stored α + darkness conversion: flips the RGB bytes (`255 − R, 255 − G, 255 − B`) AND sets α=0xFF (opaque). Theme colour constants default to OPAQUE since most use sites paint them as solid fills; partial-α sites (AA edges, glow accumulation) explicitly mask α off first via `(theme_const & 0x00FFFFFF) | (modulated_α << 24)`. Glow colour constants use [`dark_rgb_only`] to keep α=0 since the glow function sets α per pixel.
const fn dark(trgb: u32) -> u32 {
    (trgb ^ 0x00FFFFFF) | 0xFF000000
}

/// Variant of [`dark`] that leaves α=0 instead of setting it to 0xFF. Used for colour constants where the α-byte is filled in per-pixel at the use site (the textbox glow function).
const fn dark_rgb_only(trgb: u32) -> u32 {
    trgb ^ 0x00FFFFFF
}

// Background texture (organic noise, scrollable). These are NOISE-MATH constants — bit-patterns the noise function uses (base colour + low-bit variance mask + speckle mask), not display colours. They operate in visible-RGB space (matching photon's reference); the noise function does its math then flips the result to stored darkness at the store site (`result ^ 0x00FFFFFF`). NOT wrapped with `dark()` so the photon-original patterns survive.
pub const BG_BASE: u32 = fmt(0x00_0C_14_0E);
pub const BG_MASK: u32 = fmt(0x00_0F_07_1F);
pub const BG_SPECKLE: u32 = fmt(0x00_3F_1F_7F);

// Window edges (focused). Saturated warm top/left + saturated cool bottom/right give the chrome its 3D bevel cue. Brighter than the unfocused variants below — an active window earns the eye's attention.
pub const WINDOW_LIGHT_EDGE: u32 = dark(fmt(0x00_5C_4F_35));
pub const WINDOW_SHADOW_EDGE: u32 = dark(fmt(0x00_29_3A_4A));

// Window edges (unfocused). Desaturated (channels pulled toward grey, keeping a slight warm vs cool hint so the bevel survives) and darker than the focused variants. Reads as "this window is quiet, but I can still see it's a window."
pub const WINDOW_LIGHT_EDGE_UNFOCUSED: u32 = dark(fmt(0x00_36_34_30));
pub const WINDOW_SHADOW_EDGE_UNFOCUSED: u32 = dark(fmt(0x00_2A_2D_32));

// Controls strip background. The strip stays functional/clickable even when the window is unfocused, so the bg is focus-invariant. Strip hairlines + BL curve are NOT constants here — they now follow the focus-driven edge palette (vertical dividers + bottom hairline = `WINDOW_LIGHT_EDGE[_UNFOCUSED]`; BL squircle = `WINDOW_SHADOW_EDGE[_UNFOCUSED]`) so the strip's framing dims with the rest of the window.
pub const WINDOW_CONTROLS_BG: u32 = dark(fmt(0x00_1E_1E_1E));

// Button hover deltas (RGB channels wrap intentionally; α is 0xFF opaque from `dark()`).
pub const CLOSE_HOVER: u32 = dark(fmt(0x00_21_FD_F9));
pub const MAXIMIZE_HOVER: u32 = dark(fmt(0x00_FA_10_FA));
pub const MINIMIZE_HOVER: u32 = dark(fmt(0x00_F7_FA_25));

// Generic UI text. `TEXT_COLOUR` is the focused title + primary body text (brighter than the previous 0xD0 for a stronger active-window contrast). `TEXT_COLOUR_UNFOCUSED` dims for inactive-window titles (between `LABEL_COLOUR` and the focused value — readable but obviously quiet). `LABEL_COLOUR` stays as the secondary/labels grey used everywhere else.
pub const TEXT_COLOUR: u32 = dark(fmt(0x00_E8_E8_E8));
pub const TEXT_COLOUR_UNFOCUSED: u32 = dark(fmt(0x00_6A_6A_6A));
pub const LABEL_COLOUR: u32 = dark(fmt(0x00_80_80_80));

// Orb image desaturation factor applied when an unfocused window uses `OrbTint::FollowFocus`. `0 = full colour`, `255 = fully grey`. 128 = 50% lerp toward mid-grey — visibly quieted without losing app recognition. `Custom` orb tints ignore this (apps use the orb as a status indicator and want it stable).
pub const ORB_DARKEN_UNFOCUSED: u8 = 128;

// Window control glyph colours (drawn on top of WINDOW_CONTROLS_BG).
pub const CLOSE_GLYPH: u32 = dark(fmt(0x00_80_20_20));
pub const MAXIMIZE_GLYPH: u32 = dark(fmt(0x00_48_6B_3A));
pub const MAXIMIZE_GLYPH_INTERIOR: u32 = dark(fmt(0x00_28_2D_2E));
pub const MINIMIZE_GLYPH: u32 = dark(fmt(0x00_33_30_C7));

// Textbox.
pub const TEXTBOX_FILL: u32 = dark(fmt(0x00_06_08_09));
pub const TEXTBOX_HOVER: u32 = dark(fmt(0x00_12_16_18));
pub const TEXTBOX_ACTIVE: u32 = dark(fmt(0x00_00_00_00));
pub const TEXTBOX_LIGHT_EDGE: u32 = dark(fmt(0x00_44_41_37));
pub const TEXTBOX_SHADOW_EDGE: u32 = dark(fmt(0x00_2B_34_37));
pub const TEXTBOX_TEXT: u32 = dark(fmt(0x00_E0_E0_DC));
pub const TEXTBOX_SELECTION_BG: u32 = dark(fmt(0x00_3A_5A_8C));

// Cursor (blinkey).
pub const CURSOR_BRIGHTNESS: f32 = 100.0;

// Textbox glow colours (RGB only — α-byte is set per-pixel by the glow function).
pub const GLOW_DEFAULT: u32 = dark_rgb_only(fmt(0x00_FF_FF_FF));
pub const GLOW_SUCCESS: u32 = dark_rgb_only(fmt(0x00_40_FF_40));
pub const GLOW_ERROR: u32 = dark_rgb_only(fmt(0x00_FF_60_60));
/// Black halo — the rays add darkness instead of brightness, drawing the surrounding pixels toward 0. Used by [`crate::widgets::Button`] so its focus motif reads as a "deepening shadow / pressing into the surface" rather than the textbox glow's "lit-from-within" feel. Distinct visual vocabulary at the same pixel cost.
pub const GLOW_DARK: u32 = dark_rgb_only(fmt(0x00_00_00_00));
