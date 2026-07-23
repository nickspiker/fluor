//! Theme constants — colours and geometry shared between paint primitives, chrome, and (eventually) widgets. Sourced from photon's [theme.rs](/mnt/Octopus/Code/photon/src/ui/theme.rs); names match photon exactly so cross-codebase greps work.
//!
//! Colour layout: `0xααRRGGBB` (fluor internal format — top byte is α opacity, `α=0xFF` opaque; RGB bytes are DARKNESS, `0 = white potential, 255 = black`). The `0x00_xx_yy_zz` literals here are written with `t = 0` (which would have been opaque under the old convention) and visible RGB; [`fmt`] handles platform byte-swap and [`dark`] inverts the RGB at compile time. Constants here keep `α = 0` — colour palette only; the α-byte is set per-pixel at the use site (`opacity << 24 | dark(theme_const)`).
//!
//! At the OS boundary, [`crate::paint::finalize_for_os`] does a single `pixel ^= 0x00FFFFFF` that flips RGB darkness back to visible; α passes thru (already opacity-direction in storage) — putting the pixel in the format the host compositor wants.

/// Display colour-space matrix slot.
///
/// On Android, photon's Activity queries `display.preferredWideGamutColorSpace` and pushes the panel's RGB→CIE-XYZ-D50 3x3 matrix here thru a JNI shim. Consumers (chromatic_wave, future LMS-based painters) read it via [`display_rgb_to_xyz`] and compose with their own LMS→XYZ matrix to land samples in the actual device's primaries instead of falling thru a hardcoded REC2020 approximation. `None` until the JNI shim fires, and on desktop builds; consumers fall back to whatever default they want in that case.
static DISPLAY_RGB_TO_XYZ: std::sync::Mutex<Option<[f32; 9]>> = std::sync::Mutex::new(None);

/// Display chromaticity primaries (R, G, B as 1931-xy pairs — 6 floats: Rx Ry Gx Gy Bx By). Companion to [`display_rgb_to_xyz`]; useful when a consumer wants to do its own gamut mapping rather than going thru XYZ.
static DISPLAY_PRIMARIES: std::sync::Mutex<Option<[f32; 6]>> = std::sync::Mutex::new(None);

/// Push the device's display colour-space data. Called from the JNI shim on Android after the Activity's display is available. Idempotent — safe to call multiple times (e.g. on display reconfiguration).
pub fn set_display_color_space(rgb_to_xyz: [f32; 9], primaries: [f32; 6]) {
    if let Ok(mut g) = DISPLAY_RGB_TO_XYZ.lock() {
        *g = Some(rgb_to_xyz);
    }
    if let Ok(mut g) = DISPLAY_PRIMARIES.lock() {
        *g = Some(primaries);
    }
}

/// Read the device's display RGB→XYZ matrix if available. Consumers fall back to a hardcoded approximation (REC2020 in chromatic_wave's case) when this returns `None`.
pub fn display_rgb_to_xyz() -> Option<[f32; 9]> {
    DISPLAY_RGB_TO_XYZ.lock().ok().and_then(|g| *g)
}

/// Read the device's display chromaticity primaries `[Rx, Ry, Gx, Gy, Bx, By]` if available.
pub fn display_primaries() -> Option<[f32; 6]> {
    DISPLAY_PRIMARIES.lock().ok().and_then(|g| *g)
}

/// Platform-aware byte-order pack: identity on desktop, R↔B swap on Android (the ANativeWindow buffer is RGBA_8888, which reads as `0xAABBGGRR` when interpreted as a little-endian u32, so theme constants written in canonical `0xAARRGGBB` order need their R and B bytes swapped at compile time to land in the right slots without a per-pixel runtime swap). Pub so downstream crates (photon's chromatic-wave + per-screen colour constants) can adopt the same convention.
#[cfg(target_os = "android")]
pub const fn fmt(trgb: u32) -> u32 {
    let t = (trgb >> 24) & 0xFF;
    let r = (trgb >> 16) & 0xFF;
    let g = (trgb >> 8) & 0xFF;
    let b = trgb & 0xFF;
    (t << 24) | (b << 16) | (g << 8) | r
}

#[cfg(not(target_os = "android"))]
pub const fn fmt(trgb: u32) -> u32 {
    trgb
}

/// Compile-time visible-RGB → stored α + darkness conversion: flips the RGB bytes (`255 − R, 255 − G, 255 − B`) AND sets α=0xFF (opaque). Theme colour constants default to OPAQUE since most use sites paint them as solid fills; partial-α sites (AA edges, glow accumulation) explicitly mask α off first via `(theme_const & 0x00FFFFFF) | (modulated_α << 24)`. Glow colour constants use [`dark_rgb_only`] to keep α=0 since the glow function sets α per pixel.
pub const fn dark(trgb: u32) -> u32 {
    (trgb ^ 0x00FFFFFF) | 0xFF000000
}

/// Variant of [`dark`] that leaves α=0 instead of setting it to 0xFF. Used for colour constants where the α-byte is filled in per-pixel at the use site (the textbox glow function).
const fn dark_rgb_only(trgb: u32) -> u32 {
    trgb ^ 0x00FFFFFF
}

// Background texture (organic noise, scrollable). Three NOISE-MATH constants: an additive BASE colour, and two bit-MASKS the noise function ANDs random bits into (`rng & BG_MASK`, `>>8 & BG_SPECKLE`) — the masks carve variance into the channels, they are NOT colours, so a colour-space matrix on them is meaningless and they stay as-authored. Only BG_BASE is a real colour.
// COLOUR DOCTRINE (see photon theme.rs): the display surface is assumed BT.2020 γ2, so BG_BASE is HAND-CONVERTED VSF-RGB→Rec.2020 ONCE at authoring time and the Rec.2020 literal is baked here — the noise runs per-pixel on every frame, so it must never take a runtime matrix (that is the whole reason this is a const bake, not a to_display call). The VSF-authored originals are kept in the comment so the value is re-derivable. macOS ships the raw VSF value via its ICC-tagged surface — the shift is small enough for a dark near-neutral base (the matrix is near-identity on low-saturation darks) that one baked literal serves both paths acceptably; revisit only if a wide BG tint is ever introduced.
// The noise function does its math then flips the result to stored darkness at the store site (`result ^ 0x00FFFFFF`) — so these are NOT wrapped with `dark()`.
#[cfg(not(feature = "amber"))]
pub const BG_BASE: u32 = fmt(0x00_0C_14_0D); // VSF 0x0C140E → Rec.2020
#[cfg(not(feature = "amber"))]
pub const BG_MASK: u32 = fmt(0x00_0F_07_1F);
#[cfg(not(feature = "amber"))]
pub const BG_SPECKLE: u32 = fmt(0x00_3F_1F_7F);
// amber (dev builds): the same noise-math bit patterns re-biased from purple to debug orange — #FFA000's FF:A0:00 channel ratio at each constant's original magnitude (base warm-dark, variance orange-heavy, speckle an orange flash).
#[cfg(feature = "amber")]
pub const BG_BASE: u32 = fmt(0x00_16_0B_02); // VSF 0x140D00 → Rec.2020
#[cfg(feature = "amber")]
pub const BG_MASK: u32 = fmt(0x00_1F_0F_03);
#[cfg(feature = "amber")]
pub const BG_SPECKLE: u32 = fmt(0x00_7F_50_00);

// Window edges (focused). Saturated warm top/left + saturated cool bottom/right give the chrome its 3D bevel cue. Brighter than the unfocused variants below — an active window earns the eye's attention.
// amber (dev builds): the perimeter hairline IS the debug badge — full #FFA000 on the light edge; a darker orange shadow edge keeps the bevel cue.
#[cfg(not(feature = "amber"))]
pub const WINDOW_LIGHT_EDGE: u32 = dark(fmt(0x00_5C_4F_35));
#[cfg(not(feature = "amber"))]
pub const WINDOW_SHADOW_EDGE: u32 = dark(fmt(0x00_29_3A_4A));
#[cfg(feature = "amber")]
pub const WINDOW_LIGHT_EDGE: u32 = dark(fmt(0x00_FF_A0_00));
#[cfg(feature = "amber")]
pub const WINDOW_SHADOW_EDGE: u32 = dark(fmt(0x00_80_50_00));

// Window edges (unfocused). Desaturated (channels pulled toward grey, keeping a slight warm vs cool hint so the bevel survives) and darker than the focused variants. Reads as "this window is quiet, but I can still see it's a window."
#[cfg(not(feature = "amber"))]
pub const WINDOW_LIGHT_EDGE_UNFOCUSED: u32 = dark(fmt(0x00_36_34_30));
#[cfg(not(feature = "amber"))]
pub const WINDOW_SHADOW_EDGE_UNFOCUSED: u32 = dark(fmt(0x00_2A_2D_32));
#[cfg(feature = "amber")]
pub const WINDOW_LIGHT_EDGE_UNFOCUSED: u32 = dark(fmt(0x00_8A_58_00));
#[cfg(feature = "amber")]
pub const WINDOW_SHADOW_EDGE_UNFOCUSED: u32 = dark(fmt(0x00_45_2C_00));

// Controls strip background. The strip stays functional/clickable even when the window is unfocused, so the bg is focus-invariant. Strip hairlines + BL curve are NOT constants here — they now follow the focus-driven edge palette (vertical dividers + bottom hairline = `WINDOW_LIGHT_EDGE[_UNFOCUSED]`; BL squircle = `WINDOW_SHADOW_EDGE[_UNFOCUSED]`) so the strip's framing dims with the rest of the window.
pub const WINDOW_CONTROLS_BG: u32 = dark(fmt(0x00_1E_1E_1E));

// Window-control hover TARGET colours: close = red, maximize = green, minimize = blue.
// `dark(fmt(rgb))` stores visible RGB → darkness; the migration previously carried the legacy DARKNESS-encoded values (0x21FDF9 etc.) and wrapped them in dark() again, double-inverting to cyan/magenta/yellow.
// These are the plain visible primaries so ChromeButton::tint_delta's `wrap_sub_rgb(target, WINDOW_CONTROLS_BG)` lands the control at the right hue.
pub const CLOSE_HOVER: u32 = dark(fmt(0x00_DE_02_06));
pub const MAXIMIZE_HOVER: u32 = dark(fmt(0x00_03_78_03));
pub const MINIMIZE_HOVER: u32 = dark(fmt(0x00_08_05_DA));

// Generic UI text. `TEXT_COLOUR` is the focused title + primary body text (brighter than the previous 0xD0 for a stronger active-window contrast). `TEXT_COLOUR_UNFOCUSED` dims for inactive-window titles (between `LABEL_COLOUR` and the focused value — readable but obviously quiet). `LABEL_COLOUR` stays as the secondary/labels grey used everywhere else.
pub const TEXT_COLOUR: u32 = dark(fmt(0x00_E8_E8_E8));
pub const TEXT_COLOUR_UNFOCUSED: u32 = dark(fmt(0x00_6A_6A_6A));
pub const LABEL_COLOUR: u32 = dark(fmt(0x00_80_80_80));

// Title-bar text — normally just TEXT_COLOUR[_UNFOCUSED]; a named constant so the amber dev theme can badge the TITLE orange without touching body text everywhere.
#[cfg(not(feature = "amber"))]
pub const TITLE_TEXT: u32 = TEXT_COLOUR;
#[cfg(not(feature = "amber"))]
pub const TITLE_TEXT_UNFOCUSED: u32 = TEXT_COLOUR_UNFOCUSED;
#[cfg(feature = "amber")]
pub const TITLE_TEXT: u32 = dark(fmt(0x00_FF_A0_00));
#[cfg(feature = "amber")]
pub const TITLE_TEXT_UNFOCUSED: u32 = dark(fmt(0x00_8A_58_00));

// Hint / placeholder text — pure white at 1/4 opacity (α=64), stored directly in α+darkness format (the version watermark is the same treatment at 1/8). Glyph coverage multiplies into this α, so every hint reads as faint light over the dark background rather than flat ink — the eye reads real content first, the hint only on attention. One colour for every hint, by design.
pub const HINT_COLOUR: u32 = 0x40_00_00_00;

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

// Button. Distinct from textbox FILL so the eye reads "this is an action surface, not a typing surface" at a glance — slate-grey-blue, noticeably lighter than TEXTBOX_FILL's near-black. Hover lightens further toward a more saturated blue; active (pressed) drops back near TEXTBOX_FILL for the "pressed in" effect that conventionally inverts the normal raised-button shading.
pub const BUTTON_FILL: u32 = dark(fmt(0x00_1A_22_4E));
pub const BUTTON_HOVER: u32 = dark(fmt(0x00_28_34_71));
pub const BUTTON_ACTIVE: u32 = dark(fmt(0x00_0C_10_32));
// Held: a finger/button is DOWN on this control and a release here will fire it (press-hold-release — see host::pointer). A brighter, more-saturated azure than HOVER so touch-down visibly LIGHTS UP the control (distinct from ACTIVE's darker "pressed-in" focus reading); slides back to FILL if the pointer drags off before release. Also the fill used for photon's non-widget stamped elements (pills, contact/nav rows, orb) while held.
pub const BUTTON_HELD: u32 = dark(fmt(0x00_3C_4C_A8));

// Cursor (blinkey).
pub const CURSOR_BRIGHTNESS: f32 = 100.0;

// Textbox glow colours (RGB only — α-byte is set per-pixel by the glow function).
pub const GLOW_DEFAULT: u32 = dark_rgb_only(fmt(0x00_FF_FF_FF));
pub const GLOW_SUCCESS: u32 = dark_rgb_only(fmt(0x00_40_FF_40));
pub const GLOW_ERROR: u32 = dark_rgb_only(fmt(0x00_FF_60_60));
/// Black halo — the rays add darkness instead of brightness, drawing the surrounding pixels toward 0. Used by [`crate::widgets::Button`] so its focus motif reads as a "deepening shadow / pressing into the surface" rather than the textbox glow's "lit-from-within" feel. Distinct visual vocabulary at the same pixel cost.
pub const GLOW_DARK: u32 = dark_rgb_only(fmt(0x00_00_00_00));
