//! Theme constants — colours and geometry shared between paint primitives, chrome, and (eventually) widgets. Sourced from photon's [theme.rs](/mnt/Harbor/Code/photon/src/ui/theme.rs); names match photon exactly so cross-codebase greps work.
//!
//! Colour layout: `0xααRRGGBB` (fluor internal format — top byte is α opacity, `α=0xFF` opaque; RGB bytes are DARKNESS, `0 = white potential, 255 = black`). The `0x00_xx_yy_zz` literals here are written with `t = 0` (which would have been opaque under the old convention) and visible RGB; [`fmt`] handles platform byte-swap and [`dark`] inverts the RGB at compile time. Constants here keep `α = 0` — colour palette only; the α-byte is set per-pixel at the use site (`opacity << 24 | dark(theme_const)`).
//!
//! At the OS boundary, [`crate::paint::finalize_for_os`] does a single `pixel ^= 0x00FFFFFF` that flips RGB darkness back to visible; α passes thru (already opacity-direction in storage) — putting the pixel in the format the host compositor wants.

/// Display colour-space matrix slot.
///
/// On Android, photon's Activity queries `display.preferredWideGamutColorSpace` and pushes the panel's RGB→CIE-XYZ-D50 3x3 matrix here thru a JNI shim. Consumers (chromatic_wave, future LMS-based painters) read it via [`display_rgb_to_xyz`] and compose with their own LMS→XYZ matrix to land samples in the actual device's primaries instead of falling thru a hardcoded REC2020 approximation. `None` until the JNI shim fires, and on desktop builds; consumers fall back to whatever default they want in that case.
///
/// `std` only (the slot is a `Mutex`, and the only writer is a JNI shim): a no_std build has no display query to feed it, so the slot and its accessors don't exist there rather than silently reading `None`.
#[cfg(feature = "std")]
static DISPLAY_RGB_TO_XYZ: std::sync::Mutex<Option<[f32; 9]>> = std::sync::Mutex::new(None);

/// Display chromaticity primaries (R, G, B as 1931-xy pairs — 6 floats: Rx Ry Gx Gy Bx By). Companion to [`display_rgb_to_xyz`]; useful when a consumer wants to do its own gamut mapping rather than going thru XYZ. `std` only, like [`DISPLAY_RGB_TO_XYZ`].
#[cfg(feature = "std")]
static DISPLAY_PRIMARIES: std::sync::Mutex<Option<[f32; 6]>> = std::sync::Mutex::new(None);

/// Push the device's display colour-space data. Called from the JNI shim on Android after the Activity's display is available. Idempotent — safe to call multiple times (e.g. on display reconfiguration).
#[cfg(feature = "std")]
pub fn set_display_color_space(rgb_to_xyz: [f32; 9], primaries: [f32; 6]) {
    if let Ok(mut g) = DISPLAY_RGB_TO_XYZ.lock() {
        *g = Some(rgb_to_xyz);
    }
    if let Ok(mut g) = DISPLAY_PRIMARIES.lock() {
        *g = Some(primaries);
    }
}

/// Read the device's display RGB→XYZ matrix if available. Consumers fall back to a hardcoded approximation (REC2020 in chromatic_wave's case) when this returns `None`.
#[cfg(feature = "std")]
pub fn display_rgb_to_xyz() -> Option<[f32; 9]> {
    DISPLAY_RGB_TO_XYZ.lock().ok().and_then(|g| *g)
}

/// Read the device's display chromaticity primaries `[Rx, Ry, Gx, Gy, Bx, By]` if available.
#[cfg(feature = "std")]
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

/// Light held by each 8-bit `g2t` code (VSF RGB with headroom, vsf `colour.md`: linear = atanh(code / 256)², ×256 bucket bottom), Q16: diffuse white = 65536. Code 194 is diffuse white's bucket (0.982 up to 1.0006, the code VSF's encoder gives 1.0), 255 is 9.7× white. Generated from vsf's own decoder; `tests::g2t_table_is_vsfs_decoder` holds it to that.
pub const G2T_LIGHT_Q16: [u32; 256] = [
    0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 226,
    257, 290, 325, 362, 402, 443, 486, 532, 579, 629, 681, 734, 790, 848, 908, 971,
    1035, 1101, 1170, 1240, 1313, 1388, 1466, 1545, 1627, 1710, 1796, 1885, 1975, 2068, 2163, 2260,
    2359, 2461, 2565, 2672, 2781, 2892, 3006, 3122, 3240, 3361, 3484, 3610, 3738, 3868, 4001, 4137,
    4275, 4416, 4559, 4705, 4854, 5005, 5159, 5316, 5475, 5637, 5802, 5970, 6140, 6313, 6490, 6669,
    6851, 7036, 7223, 7414, 7608, 7805, 8006, 8209, 8415, 8625, 8838, 9054, 9274, 9496, 9723, 9952,
    10185, 10422, 10662, 10906, 11153, 11404, 11659, 11918, 12180, 12447, 12717, 12991, 13269, 13552, 13838, 14129,
    14424, 14723, 15026, 15335, 15647, 15964, 16286, 16612, 16944, 17280, 17621, 17967, 18318, 18674, 19036, 19402,
    19775, 20152, 20536, 20925, 21320, 21720, 22127, 22540, 22959, 23384, 23816, 24255, 24700, 25152, 25610, 26076,
    26549, 27030, 27518, 28013, 28517, 29028, 29547, 30075, 30611, 31156, 31709, 32272, 32844, 33425, 34016, 34617,
    35228, 35849, 36481, 37124, 37777, 38442, 39119, 39808, 40508, 41222, 41948, 42687, 43440, 44207, 44988, 45784,
    46595, 47422, 48264, 49123, 49999, 50892, 51803, 52733, 53682, 54651, 55640, 56649, 57681, 58735, 59812, 60913,
    62039, 63191, 64369, 65575, 66809, 68074, 69369, 70696, 72057, 73452, 74884, 76353, 77862, 79412, 81005, 82642,
    84327, 86061, 87847, 89687, 91584, 93540, 95560, 97647, 99803, 102034, 104344, 106737, 109219, 111795, 114471, 117255,
    120153, 123173, 126326, 129620, 133068, 136681, 140474, 144462, 148664, 153099, 157792, 162769, 168060, 173703, 179740, 186220,
    193205, 200766, 208993, 217996, 227912, 238921, 251251, 265216, 281241, 299939, 322228, 349566, 384469, 431821, 503081, 637212,
];

/// The `g2t` code for a light (Q16, diffuse white = 65536) by VSF's rule: the bucket it falls in, i.e. the largest code whose bucket bottom is at or below it (×2ⁿ truncation, the same answer as `vsf::colour::convert::delinearize_gamma2tanh_bits_f32`). Binary search over [`G2T_LIGHT_Q16`]; light past the top code's lands on 255.
pub const fn g2t_code(light_q16: u32) -> u32 {
    let (mut lo, mut hi) = (0usize, 255usize);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if G2T_LIGHT_Q16[mid] <= light_q16 {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo as u32
}

/// `vsf::colour::VSF_RGB2REC2020`, column-major, Q16, colorimetric (no chromatic adaptation since vsf 02076e6): out_row = m[row]·R + m[3+row]·G + m[6+row]·B. fluor deliberately has no runtime vsf dependency; `tests::rec2020_matrix_is_vsfs` holds this copy to vsf's.
pub const VSF_RGB2REC2020_Q16: [i64; 9] = [
    79325,
    -3661,
    24,
    -9351,
    72777,
    671,
    3466,
    -5762,
    61068,
];

/// Round-to-nearest integer sqrt over 0..=65025: the γ2 encode ((255x)² folds back to 255x exactly).
const fn isqrt_round(v: i64) -> u32 {
    let mut lo: i64 = 0;
    let mut hi: i64 = 256;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if mid * mid <= v {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if lo < 255 && (lo + 1) * (lo + 1) - v < v - lo * lo {
        (lo + 1) as u32
    } else {
        lo as u32
    }
}

/// Light (Q16) → γ2 over 0..=255 with 255 = white, nearest, clipped at black and white: an SDR display code.
pub const fn gamma2_code(light_q16: i64) -> u32 {
    let l = if light_q16 < 0 { 0 } else if light_q16 > 65536 { 65536 } else { light_q16 };
    isqrt_round((l * 65025) >> 16)
}

/// Light (Q16) → the buffer's code: `g2t` by VSF's rule ([`g2t_code`]), light above white kept up to the top code's 9.7×, below black clipped. The buffer has been `g2t` since 2026-10-20; the screen gets it thru [`G2T_TO_SDR`] (or, later, an HDR surface's table) in finalize.
pub const fn buffer_code(light_q16: i64) -> u32 {
    let l = if light_q16 < 0 { 0 } else if light_q16 > u32::MAX as i64 { u32::MAX as i64 } else { light_q16 };
    g2t_code(l as u32)
}

/// The final step for an SDR surface: each `g2t` code → the γ2 display code of its light, clipped at white. One table, 256 bytes, applied per channel in `paint::finalize`; codes above white (195..=255) all show as white until an HDR surface has room for them.
pub const G2T_TO_SDR: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut c = 0;
    while c < 256 {
        t[c] = gamma2_code(G2T_LIGHT_Q16[c] as i64) as u8;
        c += 1;
    }
    t
};

/// [`G2T_TO_SDR`] indexed by the stored DARKNESS byte (255 − code), so finalize's flip and the display map are one lookup.
pub const DARK_TO_SDR: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut d = 0;
    while d < 256 {
        t[d] = G2T_TO_SDR[255 - d];
        d += 1;
    }
    t
};

/// [`DARK_TO_SDR`] at 16 bits, indexed by a stored 16-bit DARKNESS (65535 − code), with 8 bits of fraction kept for the present's dither: the SDR display code × 2⁸, white 255 × 2⁸ = 65280.
/// The display code is γ2 of the light and a `g2t` code's light is atanh(code / 2¹⁶)², so the code is atanh(code / 2¹⁶) itself, clipped at white, × 255.
/// Built at compile time: atanh accumulated code by code as ∫ dt / (1 − t²), Simpson's rule over each step in f64, stopping at white (the codes past tanh(1) × 2¹⁶ all show as white); the accumulated error is far below the 2⁻⁸ step each entry is rounded to, and `tests::dark16_to_sdr_is_atanh` holds the table to libm's atanh.
/// A static, not a const: 128 KiB is one table in the binary, read in place.
pub static DARK16_TO_SDR_Q8: [u16; 1 << 16] = {
    const WHITE_Q8: f64 = (255 << 8) as f64;
    let mut t = [WHITE_Q8 as u16; 1 << 16];
    let h = 1. / ((1 << 16) as f64);
    let mut acc = 0f64;
    let mut code = 0usize;
    while code < 1 << 16 && acc <= 1. {
        t[(1 << 16) - 1 - code] = (acc * WHITE_Q8 + 0.5) as u16;
        let x = code as f64 * h;
        let m = x + h / 2.;
        let e = x + h;
        acc += h / 6. * (1. / (1. - x * x) + 4. / (1. - m * m) + 1. / (1. - e * e));
        code += 1;
    }
    t
};

/// Where the display map's roll-off starts, as a fraction of the display's peak, when content is brighter than the display can show — vsf's `colour::display::DISPLAY_KNEE` (a test holds the copy to it).
pub const DISPLAY_KNEE: f32 = 0.75;

/// vsf's `colour::display::display_light`, copied (fluor carries no runtime vsf on every target; `tests::display_light_is_vsfs` holds it to the original): scene light `l` (diffuse white 1) on a display whose peak is `headroom` × white, for content whose brightest light is `content_peak` → display light as a fraction of the display's peak.
/// Linear while the content fits; past the knee an extended-Reinhard shoulder that reaches exactly 1 at `content_peak`, so nothing clips and nothing brighter than the content is assumed.
pub fn display_light(l: f32, headroom: f32, content_peak: f32) -> f32 {
    if !(l > 0.) {
        return 0.;
    }
    let h = if headroom > 1. { headroom } else { 1. };
    if !(content_peak > h) {
        return (l / h).min(1.);
    }
    let k = DISPLAY_KNEE * h;
    if l <= k {
        return l / h;
    }
    let r = h - k;
    let u = (l - k) / r;
    let top = (content_peak - k) / r;
    let s = u * (1. + u / (top * top)) / (1. + u);
    ((k + r * s) / h).min(1.)
}

/// Q16 light (65536 = 1.0) → IEEE half bits, integer only: normals from 2⁻¹⁴ up, subnormals below, truncating like VSF's bucket rule; past half's range saturates at the largest finite half.
pub const fn q16_to_half(v: u32) -> u16 {
    if v == 0 {
        return 0;
    }
    let top = 31 - v.leading_zeros(); // bit index of the leading 1: the value is 2^(top − 16)
    if top < 2 {
        return (v << 8) as u16; // subnormal: value / 2⁻²⁴ = v × 2⁸
    }
    let exp = top - 1; // (top − 16) + 15
    if exp >= 31 {
        return 0x7BFF;
    }
    let mant = ((v << (31 - top)) >> 21) & 0x3FF;
    ((exp << 10) | mant) as u16
}

/// The extended-range finalize table (HDR plan, Android first): for every stored DARKNESS byte, the light the surface gets in its own linear extended range (1.0 = SDR white), as half bits for the opaque fast path and as Q16 for partly-covered pixels (premultiplied by α in integers, then packed).
/// Built for the headroom the display GRANTED and the frame's brightest light; rebuilt only when either changes. 512 + 1024 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HdrTable {
    pub half: [u16; 256],
    pub q16: [u32; 256],
    pub headroom_q16: u32,
    pub peak_code: u8,
}

impl HdrTable {
    /// `headroom` = what the display grants now (×white, ≥ 1); `peak_code` = the brightest `g2t` code in the frame.
    pub fn new(headroom: f32, peak_code: u8) -> Self {
        let h = if headroom > 1. { headroom } else { 1. };
        let peak = G2T_LIGHT_Q16[peak_code as usize] as f32 / 65536.;
        let mut half = [0u16; 256];
        let mut q16 = [0u32; 256];
        for d in 0..256 {
            let l = G2T_LIGHT_Q16[255 - d] as f32 / 65536.;
            let out = display_light(l, h, peak) * h;
            let q = (out * 65536. + 0.5) as u32;
            q16[d] = q;
            half[d] = q16_to_half(q);
        }
        HdrTable { half, q16, headroom_q16: (h * 65536. + 0.5) as u32, peak_code }
    }

    /// The light (×white) the buffer's brightest value reaches with this table: what the surface reports as its current ratio.
    pub fn buffer_ratio(&self) -> f32 {
        let top = self.q16.iter().copied().max().unwrap_or(65536);
        // WHY/PROOF: Android's extended-range brightness API rejects a buffer ratio below 1 (white) — a dark frame whose brightest code sits below white would otherwise report one.
        (top.max(65536)) as f32 / 65536.
    }
}

/// The 10-BIT extended-brightness table (HDR plan, Android: RGBA_1010102, BT.2020, γ2.2, extended range — the format the Pixel's display processor composites itself, where F16 forced a GPU pass into an 8-bit sRGB target).
/// On that surface 1023 is SDR white × the DECLARED ratio (SurfaceFlinger: layerBrightnessNits = sdrWhitePointNits × currentHdrSdrRatio), so every stored DARKNESS byte maps to (light ÷ declared)^(1/2.2) × 1023, clipped at the top. Declared 1 is a plain SDR surface (white near 1015); declared 9.7 still leaves SDR white about 360 codes, more than the `g2t` buffer's 194 below white.
/// Rebuilt only when the declared ratio changes (an SDR frame never does).
/// A 16-bit buffer reads `wide` instead: every 16-bit darkness → the 10-bit code × 2⁶ (the 6 bits of fraction the present dithers away), empty unless built by [`Hdr10Table::new_wide`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hdr10Table {
    pub code: [u16; 256],
    pub declared_q16: u32,
    pub wide: alloc::vec::Vec<u16>,
}

impl Hdr10Table {
    pub fn new(declared: f32) -> Self {
        let d = if declared > 1. { declared } else { 1. };
        let mut code = [0u16; 256];
        for dark in 0..256 {
            let l = G2T_LIGHT_Q16[255 - dark] as f32 / 65536.;
            // WHY/PROOF: content past the declared ratio is brighter than the surface's top code — without the clip the code would exceed 1023 and carry into the neighbouring channel of the packed RGBA_1010102 word.
            let v = (l / d).min(1.);
            code[dark] = (libm::powf(v, 1. / 2.2) * 1023. + 0.5) as u16;
        }
        Hdr10Table { code, declared_q16: (d * 65536. + 0.5) as u32, wide: alloc::vec::Vec::new() }
    }

    /// [`Hdr10Table::new`] with the 16-bit table as well: each 16-bit code's own light, atanh(code / 2¹⁶)², over the declared ratio, γ2.2, × 1023 × 2⁶.
    /// 2¹⁶ entries (128 KiB), one atanh and one pow each, a few milliseconds: rebuilt only when the declared ratio moves, as the 8-bit table is.
    pub fn new_wide(declared: f32) -> Self {
        let mut t = Self::new(declared);
        let d = t.declared();
        t.wide = (0..1u32 << 16)
            .map(|dark| {
                let a = libm::atanhf(((1 << 16) - 1 - dark) as f32 / ((1 << 16) as f32));
                // WHY/PROOF: as in `new` — light past the declared ratio would code above 1023 × 2⁶ and carry into the next channel once the dither's fraction is added.
                let v = (a * a / d).min(1.);
                (libm::powf(v, 1. / 2.2) * ((1023 << 6) as f32) + 0.5) as u16
            })
            .collect();
        t
    }

    pub fn declared(&self) -> f32 {
        self.declared_q16 as f32 / 65536.
    }
}

/// α (0..=255) → half bits of α / 255: the F16 surface's alpha channel and the opaque α (255 → 1.0).
pub const ALPHA_HALF: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut a = 0;
    while a < 256 {
        t[a] = q16_to_half(((a as u32) * 65536 + 127) / 255);
        a += 1;
    }
    t
};

/// A stored α + darkness pixel → visible SDR display RGB (γ2, white at 255), α passing thru: finalize's per-pixel colour step.
#[inline(always)]
pub const fn to_sdr_visible(px: u32) -> u32 {
    (px & 0xFF00_0000) | ((DARK_TO_SDR[((px >> 16) & 0xFF) as usize] as u32) << 16) | ((DARK_TO_SDR[((px >> 8) & 0xFF) as usize] as u32) << 8) | DARK_TO_SDR[(px & 0xFF) as usize] as u32
}

/// γ2 code (÷255, white at 255) → the `g2t` code of the same light by VSF's rule: for pixels that arrive γ2-encoded (decoded images, avatars, emoji bitmaps, video frames, hand-written darkness constants) on their way into the `g2t` buffer.
pub const GAMMA2_TO_G2T: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut c = 0;
    while c < 256 {
        t[c] = g2t_code(((c * c * 65536 + 65025 / 2) / 65025) as u32) as u8;
        c += 1;
    }
    t
};

/// Visible γ2 RGB → visible `g2t` RGB, per channel by [`GAMMA2_TO_G2T`], α passing thru.
#[inline(always)]
pub const fn from_gamma2(visible: u32) -> u32 {
    (visible & 0xFF00_0000) | ((GAMMA2_TO_G2T[((visible >> 16) & 0xFF) as usize] as u32) << 16) | ((GAMMA2_TO_G2T[((visible >> 8) & 0xFF) as usize] as u32) << 8) | GAMMA2_TO_G2T[(visible & 0xFF) as usize] as u32
}

/// A darkness value written for a γ2 buffer → the same light's darkness in the `g2t` buffer, α passing thru.
#[inline(always)]
pub const fn darkness_from_gamma2(dark: u32) -> u32 {
    from_gamma2(dark ^ 0x00FF_FFFF) ^ 0x00FF_FFFF
}

/// Compile-time authored colour → buffer value. The whole palette is AUTHORED in VSF RGB `g2t` (Nick 2026-10-20: UI colours are VSF RGB g2t, an HDR stack) and lands on the platform's surface correctly: each code decodes to light ([`G2T_LIGHT_Q16`]), takes the working primaries, and encodes into the buffer ([`buffer_code`]). Integer only: Q16 multiplies and shifts, no floats.
/// macOS: no matrix, because the surface is ICC-tagged VSF RGB (renderer_wgpu) and the OS compositor converts.
/// Everywhere else: VSF RGB → Rec.2020 by [`VSF_RGB2REC2020_Q16`], the primaries the surface is tagged (or assumed) with.
pub const fn vsf(trgb: u32) -> u32 {
    let r = G2T_LIGHT_Q16[((trgb >> 16) & 0xFF) as usize] as i64;
    let g = G2T_LIGHT_Q16[((trgb >> 8) & 0xFF) as usize] as i64;
    let b = G2T_LIGHT_Q16[(trgb & 0xFF) as usize] as i64;
    let (lr, lg, lb) = to_working(r, g, b);
    (trgb & 0xFF00_0000) | (buffer_code(lr) << 16) | (buffer_code(lg) << 8) | buffer_code(lb)
}

#[cfg(target_os = "macos")]
const fn to_working(r: i64, g: i64, b: i64) -> (i64, i64, i64) {
    (r, g, b)
}

#[cfg(not(target_os = "macos"))]
const fn to_working(r: i64, g: i64, b: i64) -> (i64, i64, i64) {
    let m = VSF_RGB2REC2020_Q16;
    ((m[0] * r + m[3] * g + m[6] * b) >> 16, (m[1] * r + m[4] * g + m[7] * b) >> 16, (m[2] * r + m[5] * g + m[8] * b) >> 16)
}

/// Diffuse white in the buffer, stored (α 0xFF + darkness): authored `g2t` 0xC2C2C2 thru [`vsf`].
/// Darkness 0 is NOT white since the buffer went `g2t`: it is the top code, 9.7× white, which an SDR screen clips to white but an HDR surface shows as a glow — anything meant as white light uses this, and anything meant to glow above white is chosen on purpose.
pub const WHITE: u32 = dark(fmt(vsf(0x00_C2_C2_C2)));

/// White light at opacity `alpha` (0..=255): [`WHITE`]'s darkness under that α byte, for watermarks, hints and glows.
pub const fn white_at(alpha: u32) -> u32 {
    ((alpha & 0xFF) << 24) | (WHITE & 0x00FF_FFFF)
}

/// Variant of [`dark`] that leaves α=0 instead of setting it to 0xFF. Used for colour constants where the α-byte is filled in per-pixel at the use site (the textbox glow function).
const fn dark_rgb_only(trgb: u32) -> u32 {
    trgb ^ 0x00FFFFFF
}

// Background texture (organic noise, scrollable). Three NOISE-MATH constants: an additive BASE colour, and two bit-MASKS the noise function ANDs random bits into (`rng & BG_MASK`, `>>8 & BG_SPECKLE`) — the masks carve variance into the channels, they are NOT colours, so a colour-space matrix on them is meaningless and they stay as-authored. Only BG_BASE is a real colour.
// COLOUR DOCTRINE: every literal in this file is AUTHORED VSF RGB `g2t` (diffuse white = 0xC2) and goes thru [`vsf`] — a compile-time bake (identity on macOS's ICC-tagged surface, the γ2/matrix/γ2 conversion everywhere else), so the noise still never takes a runtime matrix and each platform gets the right buffer value instead of one shared compromise.
// The noise function does its math then flips the result to stored darkness at the store site (`result ^ 0x00FFFFFF`) — so these are NOT wrapped with `dark()`.
#[cfg(not(feature = "amber"))]
pub const BG_BASE: u32 = fmt(vsf(0x00_0C_14_0E));
#[cfg(not(feature = "amber"))]
pub const BG_MASK: u32 = fmt(0x00_0F_07_1F);
#[cfg(not(feature = "amber"))]
pub const BG_SPECKLE: u32 = fmt(0x00_3F_1F_7F);
// amber (dev builds): the same noise-math bit patterns re-biased from purple to debug orange — #FFA000's FF:A0:00 channel ratio at each constant's original magnitude (base warm-dark, variance orange-heavy, speckle an orange flash).
#[cfg(feature = "amber")]
pub const BG_BASE: u32 = fmt(vsf(0x00_14_0D_00));
#[cfg(feature = "amber")]
pub const BG_MASK: u32 = fmt(0x00_1F_0F_03);
#[cfg(feature = "amber")]
pub const BG_SPECKLE: u32 = fmt(0x00_7F_50_00);

// Window edges (focused). Saturated warm top/left + saturated cool bottom/right give the chrome its 3D bevel cue. Brighter than the unfocused variants below — an active window earns the eye's attention.
// amber (dev builds): the perimeter hairline IS the debug badge — full #FFA000 on the light edge; a darker orange shadow edge keeps the bevel cue.
#[cfg(not(feature = "amber"))]
pub const WINDOW_LIGHT_EDGE: u32 = dark(fmt(vsf(0x00_58_4C_34)));
#[cfg(not(feature = "amber"))]
pub const WINDOW_SHADOW_EDGE: u32 = dark(fmt(vsf(0x00_28_39_48)));
#[cfg(feature = "amber")]
pub const WINDOW_LIGHT_EDGE: u32 = dark(fmt(vsf(0x00_C2_8E_00)));
#[cfg(feature = "amber")]
pub const WINDOW_SHADOW_EDGE: u32 = dark(fmt(vsf(0x00_76_4D_00)));

// Window edges (unfocused). Desaturated (channels pulled toward grey, keeping a slight warm vs cool hint so the bevel survives) and darker than the focused variants. Reads as "this window is quiet, but I can still see it's a window."
#[cfg(not(feature = "amber"))]
pub const WINDOW_LIGHT_EDGE_UNFOCUSED: u32 = dark(fmt(vsf(0x00_35_33_2F)));
#[cfg(not(feature = "amber"))]
pub const WINDOW_SHADOW_EDGE_UNFOCUSED: u32 = dark(fmt(vsf(0x00_29_2C_31)));
#[cfg(feature = "amber")]
pub const WINDOW_LIGHT_EDGE_UNFOCUSED: u32 = dark(fmt(vsf(0x00_7E_55_00)));
#[cfg(feature = "amber")]
pub const WINDOW_SHADOW_EDGE_UNFOCUSED: u32 = dark(fmt(vsf(0x00_43_2B_00)));

// Controls strip background. The strip stays functional/clickable even when the window is unfocused, so the bg is focus-invariant. Strip hairlines + BL curve are NOT constants here — they now follow the focus-driven edge palette (vertical dividers + bottom hairline = `WINDOW_LIGHT_EDGE[_UNFOCUSED]`; BL squircle = `WINDOW_SHADOW_EDGE[_UNFOCUSED]`) so the strip's framing dims with the rest of the window.
pub const WINDOW_CONTROLS_BG: u32 = dark(fmt(vsf(0x00_1D_1D_1D)));

// Window-control hover TARGET colours: close = red, maximize = green, minimize = blue.
// `dark(fmt(rgb))` stores visible RGB → darkness; the migration previously carried the legacy DARKNESS-encoded values (0x21FDF9 etc.) and wrapped them in dark() again, double-inverting to cyan/magenta/yellow.
// These are the plain visible primaries so ChromeButton::tint_delta's `wrap_sub_rgb(target, WINDOW_CONTROLS_BG)` lands the control at the right hue.
pub const CLOSE_HOVER: u32 = dark(fmt(vsf(0x00_B3_02_06)));
pub const MAXIMIZE_HOVER: u32 = dark(fmt(vsf(0x00_03_70_03)));
pub const MINIMIZE_HOVER: u32 = dark(fmt(vsf(0x00_08_05_B1)));

// Generic UI text. `TEXT_COLOUR` is the focused title + primary body text (brighter than the previous 0xD0 for a stronger active-window contrast). `TEXT_COLOUR_UNFOCUSED` dims for inactive-window titles (between `LABEL_COLOUR` and the focused value — readable but obviously quiet). `LABEL_COLOUR` stays as the secondary/labels grey used everywhere else.
pub const TEXT_COLOUR: u32 = dark(fmt(vsf(0x00_B8_B8_B8)));
pub const TEXT_COLOUR_UNFOCUSED: u32 = dark(fmt(vsf(0x00_64_64_64)));
pub const LABEL_COLOUR: u32 = dark(fmt(vsf(0x00_76_76_76)));

// Title-bar text — normally just TEXT_COLOUR[_UNFOCUSED]; a named constant so the amber dev theme can badge the TITLE orange without touching body text everywhere.
#[cfg(not(feature = "amber"))]
pub const TITLE_TEXT: u32 = TEXT_COLOUR;
#[cfg(not(feature = "amber"))]
pub const TITLE_TEXT_UNFOCUSED: u32 = TEXT_COLOUR_UNFOCUSED;
#[cfg(feature = "amber")]
pub const TITLE_TEXT: u32 = dark(fmt(vsf(0x00_C2_8E_00)));
#[cfg(feature = "amber")]
pub const TITLE_TEXT_UNFOCUSED: u32 = dark(fmt(vsf(0x00_7E_55_00)));

// Hint / placeholder text — diffuse white ([`WHITE`], never darkness 0) at 1/4 opacity (α=64) (the version watermark is the same treatment at 1/8). Glyph coverage multiplies into this α, so every hint reads as faint light over the dark background rather than flat ink — the eye reads real content first, the hint only on attention. One colour for every hint, by design.
pub const HINT_COLOUR: u32 = white_at(0x40);

// Orb image desaturation factor applied when an unfocused window uses `OrbTint::FollowFocus`. `0 = full colour`, `255 = fully grey`. 128 = 50% lerp toward mid-grey — visibly quieted without losing app recognition. `Custom` orb tints ignore this (apps use the orb as a status indicator and want it stable).
pub const ORB_DARKEN_UNFOCUSED: u8 = 128;

// Window control glyph colours (drawn on top of WINDOW_CONTROLS_BG).
pub const CLOSE_GLYPH: u32 = dark(fmt(vsf(0x00_76_1F_1F)));
pub const MAXIMIZE_GLYPH: u32 = dark(fmt(vsf(0x00_46_65_39)));
pub const MAXIMIZE_GLYPH_INTERIOR: u32 = dark(fmt(vsf(0x00_27_2C_2D)));
pub const MINIMIZE_GLYPH: u32 = dark(fmt(vsf(0x00_32_2F_A7)));

// Textbox.
pub const TEXTBOX_FILL: u32 = dark(fmt(vsf(0x00_06_08_09)));
pub const TEXTBOX_HOVER: u32 = dark(fmt(vsf(0x00_12_16_18)));
pub const TEXTBOX_ACTIVE: u32 = dark(fmt(vsf(0x00_00_00_00)));
pub const TEXTBOX_LIGHT_EDGE: u32 = dark(fmt(vsf(0x00_42_3F_36)));
pub const TEXTBOX_SHADOW_EDGE: u32 = dark(fmt(vsf(0x00_2A_33_36)));
pub const TEXTBOX_TEXT: u32 = dark(fmt(vsf(0x00_B4_B4_B2)));
pub const TEXTBOX_SELECTION_BG: u32 = dark(fmt(vsf(0x00_39_56_7F)));

// Button. Distinct from textbox FILL so the eye reads "this is an action surface, not a typing surface" at a glance — slate-grey-blue, noticeably lighter than TEXTBOX_FILL's near-black. Hover lightens further toward a more saturated blue; active (pressed) drops back near TEXTBOX_FILL for the "pressed in" effect that conventionally inverts the normal raised-button shading.
pub const BUTTON_FILL: u32 = dark(fmt(vsf(0x00_1A_21_4B)));
pub const BUTTON_HOVER: u32 = dark(fmt(vsf(0x00_27_33_6A)));
pub const BUTTON_ACTIVE: u32 = dark(fmt(vsf(0x00_0C_10_31)));
// Held: a finger/button is DOWN on this control and a release here will fire it (press-hold-release — see host::pointer). A brighter, more-saturated azure than HOVER so touch-down visibly LIGHTS UP the control (distinct from ACTIVE's darker "pressed-in" focus reading); slides back to FILL if the pointer drags off before release. Also the fill used for photon's non-widget stamped elements (pills, contact/nav rows, orb) while held.
pub const BUTTON_HELD: u32 = dark(fmt(vsf(0x00_3B_4A_93)));

// Cursor (blinkey).
pub const CURSOR_BRIGHTNESS: f32 = 100.0;

// Textbox glow colours (RGB only — α-byte is set per-pixel by the glow function).
pub const GLOW_DEFAULT: u32 = dark_rgb_only(fmt(vsf(0x00_C2_C2_C2)));
pub const GLOW_SUCCESS: u32 = dark_rgb_only(fmt(vsf(0x00_3E_C2_3E)));
pub const GLOW_ERROR: u32 = dark_rgb_only(fmt(vsf(0x00_C2_5C_5C)));
/// Black halo — the rays add darkness instead of brightness, drawing the surrounding pixels toward 0. Used by [`crate::widgets::Button`] so its focus motif reads as a "deepening shadow / pressing into the surface" rather than the textbox glow's "lit-from-within" feel. Distinct visual vocabulary at the same pixel cost.
pub const GLOW_DARK: u32 = dark_rgb_only(fmt(vsf(0x00_00_00_00)));

#[cfg(test)]
mod tests {
    use super::*;

    /// The integer half pack agrees with IEEE half at the landmarks and with a float reference (truncating) everywhere a Q16 light lands.
    #[test]
    fn q16_to_half_is_ieee_half() {
        assert_eq!(q16_to_half(0), 0);
        assert_eq!(q16_to_half(65536), 0x3C00, "1.0");
        assert_eq!(q16_to_half(131072), 0x4000, "2.0");
        assert_eq!(q16_to_half(32768), 0x3800, "0.5");
        assert_eq!(q16_to_half(4), 0x0400, "2⁻¹⁴, the smallest normal");
        assert_eq!(q16_to_half(1), 0x0100, "2⁻¹⁶, subnormal");
        for v in [3u32, 5, 100, 4095, 65535, 65537, 98304, 637212, 1 << 30] {
            let f = v as f32 / 65536.;
            let b = f.to_bits();
            let want = ((((b >> 23) & 0xFF) as i32 - 112) as u32) << 10 | ((b >> 13) & 0x3FF);
            if v >= 4 && v < (1 << 31) / 2 {
                assert_eq!(q16_to_half(v) as u32, want, "{v}");
            }
        }
        assert_eq!(ALPHA_HALF[255], 0x3C00);
        assert_eq!(ALPHA_HALF[0], 0);
    }

    /// fluor's copy of the display map is vsf's.
    #[test]
    fn display_light_is_vsfs() {
        assert_eq!(DISPLAY_KNEE, vsf::colour::display::DISPLAY_KNEE);
        for &(h, p) in &[(1.0f32, 1.0f32), (1.0, 9.7), (2.0, 2.0), (2.0, 9.7), (1.65, 4.0), (8.0, 2.0)] {
            for i in 0..=200 {
                let l = i as f32 * 0.05;
                assert_eq!(display_light(l, h, p), vsf::colour::display::display_light(l, h, p), "l={l} h={h} p={p}");
            }
        }
    }

    /// The extended-range table: on an SDR grant white stays white and nothing passes 1; with 2× granted and content peaking at 2×, the map is plain linear, so white is 1 and code 227 (2×) is 2; content brighter than the grant rolls off into exactly the grant.
    #[test]
    fn hdr_table_maps_light_into_the_granted_headroom() {
        let light = |t: &HdrTable, code: usize| t.q16[255 - code] as f32 / 65536.;
        let sdr = HdrTable::new(1.0, 194);
        assert!((light(&sdr, 194) - 0.982).abs() < 0.01, "{}", light(&sdr, 194));
        assert!(sdr.q16.iter().all(|&q| q <= 65536));
        let fits = HdrTable::new(2.0, 227);
        assert!((light(&fits, 194) - 0.982).abs() < 0.01);
        assert!((light(&fits, 227) - 1.978).abs() < 0.02, "{}", light(&fits, 227));
        let tall = HdrTable::new(2.0, 255);
        assert!((light(&tall, 255) - 2.0).abs() < 1e-3, "the content's peak lands on the grant");
        assert!(tall.q16.windows(2).all(|w| w[0] >= w[1]), "monotone (darkness up = light down)");
        assert_eq!(tall.half[255 - 194], q16_to_half(tall.q16[255 - 194]));
    }

    /// The 10-bit table: declared 1 is a plain SDR surface (white near the top, the top code clipped to 1023); declared 2 puts white at 2^(−1/2.2) of the range and code 227 (2×) at the top; every declaration is monotone and SDR white keeps far more than 194 codes.
    #[test]
    fn hdr10_table_spans_the_declared_ratio() {
        let code = |t: &Hdr10Table, c: usize| t.code[255 - c] as i32;
        let sdr = Hdr10Table::new(1.0);
        assert!((code(&sdr, 194) - 1015).abs() <= 2, "{}", code(&sdr, 194));
        assert_eq!(code(&sdr, 255), 1023);
        let two = Hdr10Table::new(2.0);
        assert!((code(&two, 194) - 741).abs() <= 3, "{}", code(&two, 194));
        assert!(code(&two, 227) >= 1018);
        let tall = Hdr10Table::new(9.72);
        assert!(code(&tall, 194) > 340, "{}", code(&tall, 194));
        for t in [&sdr, &two, &tall] {
            assert!(t.code.windows(2).all(|w| w[0] >= w[1]));
        }
    }

    /// White is diffuse white, never darkness 0: every channel of [`WHITE`] sits near code 194 (the matrix may lift one a little — VSF white is about 1.12× a D65 display's red), far below the top code a bare darkness 0 meant; [`white_at`] carries the same darkness under any α, and the hint uses it.
    #[test]
    fn white_is_diffuse_white_not_the_top_code() {
        for shift in [16, 8, 0] {
            let code = 255 - ((WHITE >> shift) & 0xFF);
            assert!((185..=205).contains(&code), "white's buffer code {code} on channel >>{shift}");
        }
        assert_eq!(WHITE >> 24, 0xFF);
        assert_eq!(white_at(0x40), 0x4000_0000 | (WHITE & 0x00FF_FFFF));
        assert_eq!(HINT_COLOUR, white_at(0x40));
    }

    #[test]
    fn g2t_table_is_vsfs_decoder() {
        for c in 0..256u32 {
            let want = vsf::colour::convert::linearize_gamma2tanh_bits_f32(c, 8) as f64 * 65536.0;
            let got = G2T_LIGHT_Q16[c as usize] as f64;
            assert!((got - want).abs() <= 1.0 + want * 1e-6, "code {c}: {got} vs {want}");
        }
    }

    #[test]
    fn rec2020_matrix_is_vsfs() {
        for (i, v) in vsf::colour::VSF_RGB2REC2020.iter().enumerate() {
            assert_eq!(VSF_RGB2REC2020_Q16[i], (*v as f64 * 65536.0).round() as i64, "element {i}");
        }
    }

    /// The compile-time 16-bit SDR table is libm's atanh, clipped at white, to the rounding of its last step, and it is the 8-bit table at every 8-bit code's own 16-bit code (`c × 2⁸`, the same light) to within a code.
    /// Not half a code: the 8-bit table goes thru the light in Q16 and floors it on the way to γ2 (`(l × 65025) >> 16`), which near black, where the light is a few Q16 units, drops it most of a code (code 1: 0.996 exact, 0 in the 8-bit table).
    #[test]
    fn dark16_to_sdr_is_atanh() {
        for d in 0..1usize << 16 {
            let code = ((1 << 16) - 1 - d) as f64;
            let want = libm::atanh(code / 65536.).min(1.) * 65280.;
            let got = DARK16_TO_SDR_Q8[d] as f64;
            assert!((got - want).abs() <= 0.5 + 1e-6, "darkness {d}: {got} vs {want}");
        }
        for c in 0..256usize {
            let d16 = (255 - c) * 256 + 255;
            let (q8, sdr) = (DARK16_TO_SDR_Q8[d16] as f64 / 256., G2T_TO_SDR[c] as f64);
            assert!((q8 - sdr).abs() <= 1., "code {c}: {q8} vs {sdr}");
        }
        assert_eq!(DARK16_TO_SDR_Q8[0], 65280, "the top code shows as white");
        assert_eq!(DARK16_TO_SDR_Q8[(1 << 16) - 1], 0, "black");
    }

    /// The 16-bit 10-bit table agrees with the 8-bit one at every 8-bit code's own 16-bit code.
    #[test]
    fn hdr10_wide_table_is_the_8_bit_one_widened() {
        for declared in [1., 2.5, 9.7] {
            let t = Hdr10Table::new_wide(declared);
            assert_eq!(t.wide.len(), 1 << 16);
            for dark in 0..256usize {
                let wide = t.wide[dark * 256 + 255] as f64 / 64.;
                assert!((wide - t.code[dark] as f64).abs() <= 1., "declared {declared}, darkness {dark}: {wide} vs {}", t.code[dark]);
            }
            assert!(t.wide.iter().all(|&c| c <= 1023 << 6));
        }
    }

    /// An SDR screen sees what it saw before: every γ2 code taken into the buffer and back out thru the final table lands within 3 codes (2.4 γ2 codes per `g2t` step near white), white and black exactly; above white clips to white.
    #[test]
    fn gamma2_round_trips_thru_the_buffer_and_the_sdr_table() {
        for c in 0..256u32 {
            let back = G2T_TO_SDR[GAMMA2_TO_G2T[c as usize] as usize] as u32;
            assert!(back.abs_diff(c) <= 3, "γ2 {c} → g2t {} → SDR {back}", GAMMA2_TO_G2T[c as usize]);
        }
        assert_eq!(GAMMA2_TO_G2T[255], 194);
        assert_eq!(G2T_TO_SDR[194], 253, "white's bucket decodes at its bottom, 0.982 of white (VSF's rule)");
        assert_eq!((G2T_TO_SDR[0], G2T_TO_SDR[255]), (0, 255));
        assert_eq!(to_sdr_visible(0x80_00_00_00), 0x80_FF_FF_FF, "darkness 0 is peak, and peak shows as white on SDR");
    }

    #[test]
    fn g2t_code_inverts_the_table() {
        for c in 0..256u32 {
            assert_eq!(g2t_code(G2T_LIGHT_Q16[c as usize]), c);
        }
        assert_eq!(g2t_code(65536), 194, "diffuse white, as VSF's encoder puts it");
        for l in [0u32, 1000, 30000, 65536, 300000, 600000] {
            assert_eq!(g2t_code(l), vsf::colour::convert::delinearize_gamma2tanh_bits_f32(l as f32 / 65536.0, 8), "light {l}");
        }
        assert_eq!(g2t_code(u32::MAX), 255);
    }

    /// The palette moved from γ2 to `g2t` authoring with its light kept: each pair is (the γ2 value it had, the `g2t` value it has now), and what the old pipeline (γ2 decode ÷255, the same matrix, γ2 encode) put on an SDR screen and what the new one does (`g2t` buffer thru [`G2T_TO_SDR`]) land within 5 codes. Not 0: `g2t` spends 194 codes below white where γ2 spent 255, so near white one `g2t` step is about 2.4 γ2 codes, and VSF's rule decodes at the bucket's bottom twice here: the authored colour, then the buffer code after the matrix.
    #[test]
    fn the_g2t_palette_keeps_the_gamma2_palettes_light() {
        const PAIRS: &[(u32, u32)] = &[
            (0x0C140E, 0x0C140E),
            (0x140D00, 0x140D00),
            (0x5C4F35, 0x584C34),
            (0x293A4A, 0x283948),
            (0xFFA000, 0xC28E00),
            (0x805000, 0x764D00),
            (0x363430, 0x35332F),
            (0x2A2D32, 0x292C31),
            (0x8A5800, 0x7E5500),
            (0x452C00, 0x432B00),
            (0x1E1E1E, 0x1D1D1D),
            (0xDE0206, 0xB30206),
            (0x037803, 0x037003),
            (0x0805DA, 0x0805B1),
            (0xE8E8E8, 0xB8B8B8),
            (0x6A6A6A, 0x646464),
            (0x808080, 0x767676),
            (0xFFA000, 0xC28E00),
            (0x8A5800, 0x7E5500),
            (0x802020, 0x761F1F),
            (0x486B3A, 0x466539),
            (0x282D2E, 0x272C2D),
            (0x3330C7, 0x322FA7),
            (0x060809, 0x060809),
            (0x121618, 0x121618),
            (0x000000, 0x000000),
            (0x444137, 0x423F36),
            (0x2B3437, 0x2A3336),
            (0xE0E0DC, 0xB4B4B2),
            (0x3A5A8C, 0x39567F),
            (0x1A224E, 0x1A214B),
            (0x283471, 0x27336A),
            (0x0C1032, 0x0C1031),
            (0x3C4CA8, 0x3B4A93),
            (0xFFFFFF, 0xC2C2C2),
            (0x40FF40, 0x3EC23E),
            (0xFF6060, 0xC25C5C),
            (0x000000, 0x000000),
        ];
        let old = |hex: u32| {
            let ch = |c: u32| ((c * c) as i64 * 65536) / 65025;
            let (r, g, b) = to_working(ch((hex >> 16) & 0xFF), ch((hex >> 8) & 0xFF), ch(hex & 0xFF));
            [gamma2_code(r), gamma2_code(g), gamma2_code(b)]
        };
        for &(was, now) in PAIRS {
            let (o, n) = (old(was), vsf(now));
            let n = [G2T_TO_SDR[((n >> 16) & 0xFF) as usize] as u32, G2T_TO_SDR[((n >> 8) & 0xFF) as usize] as u32, G2T_TO_SDR[(n & 0xFF) as usize] as u32];
            for i in 0..3 {
                assert!(o[i].abs_diff(n[i]) <= 5, "{was:06X} → {now:06X}: channel {i} {} vs {}", o[i], n[i]);
            }
        }
    }
}
