//! Golden scenes: representative renders hashed byte for byte, so a refactor of the pixel layer (the `Pixel` trait, a generic `Canvas`, the kernels moving onto trait constants) can prove it changed no output.
//! Each scene draws into a fixed-size buffer with the public API and compares an FNV-1a 64 hash of its bytes to the value recorded from the code before the refactor.
//! A moved hash means the output moved: find out why before re-recording one, since re-recording is exactly how a silent change would get in.
//!
//! The values are for x86_64 only, and were recorded on an AMD Zen 3.
//! The rotated rect, the circle and the ellipses take their AA from `rsqrtss` (paint/shape.rs `fast_inv_sqrt`), whose low bits are the vendor's own approximation, and the text goes thru float glyph rasterization; another architecture or CPU vendor can land a few edge pixels differently with nothing wrong.
//! Those scenes are kept apart from the integer-only ones so a vendor difference shows up as exactly those tests and nothing else.
//! The scenes draw in the release theme: the `amber` feature swaps theme colours and moves every hash, so these hold only without it.

use crate::canvas::{Canvas, Damage, PixelRect};
use crate::paint::{self, AlphaMask, BlendMode, Clip};
use crate::pixel::{Argb8, Pixel};
use crate::theme;
use alloc::vec;
use alloc::string::String;
use alloc::vec::Vec;

/// Scene width: wide enough for a pill with a glow on both sides and a run of text.
const W: usize = 320;
/// Scene height.
const H: usize = 192;

/// FNV-1a 64 over the little-endian bytes of the pixels: no dependency, and any one bit moving changes it.
/// The multiply wraps because FNV is defined modulo 2⁶⁴; the wrap is the hash, not a guard.
fn fnv(pixels: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &p in pixels {
        for b in p.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The same over a byte buffer (clip masks).
fn fnv8(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Compare a scene's hash to its recorded value, printing the actual one so a deliberate change can be recorded from the failure.
/// A scene that drew nothing would hash the same before and after any refactor and prove nothing, so an all-empty buffer fails here.
fn check(name: &str, pixels: &[u32], want: u64) {
    assert!(pixels.iter().any(|&p| p != 0), "golden scene `{name}` drew nothing");
    check_hash(name, fnv(pixels), want);
}

/// [`check`] for a byte buffer (clip masks).
fn check8(name: &str, bytes: &[u8], want: u64) {
    assert!(bytes.iter().any(|&b| b != 0), "golden scene `{name}` is empty");
    check_hash(name, fnv8(bytes), want);
}

/// The comparison itself, with the actual hash in the message.
fn check_hash(name: &str, got: u64, want: u64) {
    assert_eq!(got, want, "golden scene `{name}` moved: now {got:016x}, recorded {want:016x}");
}

/// A fresh empty buffer and damage accumulator at the scene size, at depth `P`.
fn scene<P: Pixel>() -> (Vec<P>, Damage) {
    (vec![P::EMPTY; W * H], Damage::new())
}

/// Every integer-only primitive in paint.rs: fill and stroke rects (plain, clipped, masked, hairline), the filled circle and the blinkey.
fn paint_rects<P: Pixel>() -> Vec<P> {
    let (mut px, mut dmg) = scene::<P>();
    let mask_bytes: Vec<u8> = (0..W * H).map(|i| ((i % W) * 255 / W) as u8).collect();
    let mask = AlphaMask::new(&mask_bytes, W, H);
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::fill_rect(&mut c, 8, 8, 120, 60, theme::BUTTON_FILL, None, None);
        paint::fill_rect(&mut c, 40, 30, 200, 80, (theme::TEXTBOX_SELECTION_BG & 0x00FF_FFFF) | 0x8000_0000, Some(Clip::new(0, 0, 180, 100)), None);
        paint::fill_rect(&mut c, 0, 120, W as isize, 40, theme::TEXTBOX_TEXT, None, Some(&mask));
        paint::fill_rect(&mut c, 10, 170, 300, 0, theme::WINDOW_LIGHT_EDGE, None, None);
        paint::stroke_rect(&mut c, 150, 10, 150, 90, 3, theme::WINDOW_SHADOW_EDGE, None, None);
        paint::circle_filled(&mut c, 260, 150, 24, theme::CLOSE_HOVER, None, None);
        paint::circle_filled(&mut c, 300, 30, 40, (theme::MAXIMIZE_HOVER & 0x00FF_FFFF) | 0x6000_0000, None, Some(&mask));
    }
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::draw_blinkey(&mut c, 100, 20, 40, true);
        paint::draw_blinkey_tinted(&mut c, 120, 20, 40, false, (160, 80, 255));
    }
    px
}

#[test]
fn golden_paint_rects() {
    check("paint_rects", &paint_rects::<Argb8>(), 0x219a_55dc_a5c7_3aca);
}

/// The float AA shapes in paint/shape.rs: axis rect with fractional edges, hairlines, rotated rect, AA circle, segmented ring.
fn shapes<P: Pixel>() -> Vec<P> {
    let (mut px, mut dmg) = scene::<P>();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::draw_rect(&mut c, 60.3, 40.7, 90.4, 50.2, theme::BUTTON_HOVER, None);
        paint::draw_rect(&mut c, 160., 20.5, 120., 0., theme::TEXT_COLOUR, None);
        paint::draw_rect(&mut c, 20.5, 120., 0., 60., theme::TEXT_COLOUR, None);
        paint::draw_rect_rotated(&mut c, 200., 100., 110., 36., 0.6, (theme::MINIMIZE_HOVER & 0x00FF_FFFF) | 0xC000_0000, None);
        paint::draw_rect_rotated(&mut c, 90., 140., 60., 20., -1.1, theme::CLOSE_GLYPH, Some(Clip::new(70, 120, 200, 192)));
        paint::draw_circle(&mut c, 270.5, 150.25, 30.7, theme::MAXIMIZE_GLYPH, None);
        paint::draw_ring_segments(&mut c, 120., 96., 44., 6., 4., &[theme::CLOSE_HOVER, theme::MAXIMIZE_HOVER, theme::MINIMIZE_HOVER, 0, theme::WHITE], None);
    }
    px
}

#[test]
fn golden_shapes() {
    check("shapes", &shapes::<Argb8>(), 0x6777_2b5d_1819_8f0d);
}

/// The ellipses (implicit form with the fast reciprocal square root).
fn ellipses<P: Pixel>() -> Vec<P> {
    let (mut px, mut dmg) = scene::<P>();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::draw_ellipse(&mut c, 90.5, 80.25, 70.3, 40.6, theme::BUTTON_FILL, None);
        paint::draw_ellipse_rotated(&mut c, 220., 110., 80., 30., 0.7, (theme::TEXTBOX_SELECTION_BG & 0x00FF_FFFF) | 0xA000_0000, None);
    }
    px
}

#[test]
fn golden_ellipses() {
    check("ellipses", &ellipses::<Argb8>(), 0x21fd_a6cb_4194_1455);
}

/// Squircle pills and rounded rects, integer and fractional exponent, single colour and two-tone, asymmetric corners, stacked so the second composes under the first.
fn squircles<P: Pixel>() -> Vec<P> {
    let (mut px, mut dmg) = scene::<P>();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::draw_squircle_pill(&mut c, 8, 8, 140, 44, theme::TEXTBOX_FILL, paint::SHAPE_POWER);
        paint::draw_squircle_pill_f(&mut c, 160, 8, 150, 44, theme::BUTTON_FILL, 1.5);
        paint::draw_squircle_pill_two_tone(&mut c, 4, 4, 148, 52, theme::TEXTBOX_LIGHT_EDGE, theme::TEXTBOX_SHADOW_EDGE, paint::SHAPE_POWER, None, 0);
        paint::draw_squircle_pill_two_tone_f(&mut c, 156, 4, 158, 52, theme::TEXTBOX_LIGHT_EDGE, theme::TEXTBOX_SHADOW_EDGE, 1.5, None, 0);
        paint::draw_squircle_rrect(&mut c, 12, 72, 130, 100, theme::BUTTON_HOVER, paint::asymmetric_radii(36.), paint::SHAPE_POWER);
        paint::draw_squircle_rrect_f(&mut c, 170, 72, 136, 100, theme::WINDOW_CONTROLS_BG, [8., 30., 50., 16.], 2.5);
        paint::draw_squircle_rrect_two_tone(&mut c, 8, 68, 138, 108, theme::WINDOW_LIGHT_EDGE, theme::WINDOW_SHADOW_EDGE, paint::asymmetric_radii(40.), paint::SHAPE_POWER, None, 0);
        paint::draw_squircle_rrect_two_tone_f(&mut c, 166, 68, 144, 108, theme::WINDOW_LIGHT_EDGE, theme::WINDOW_SHADOW_EDGE, [12., 34., 54., 20.], 2.5, None, 0);
    }
    px
}

#[test]
fn golden_squircles() {
    check("squircles", &squircles::<Argb8>(), 0x6041_ac40_23c4_68b4);
}

/// The textbox focus glow around a pill, its mask taken from the pill's own α in the scene (the top byte of α at any depth), and the four directional ray glows the textbox and button paint.
fn glows<P: Pixel>() -> (Vec<P>, Vec<P>, Vec<P>) {
    let (bw, bh) = (180usize, 48usize);
    let (mut px, mut dmg) = scene::<P>();
    let (cx, cy) = (W / 2, H / 2);
    let (x, y) = ((cx - bw / 2) as isize, (cy - bh / 2) as isize);
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::draw_squircle_pill(&mut c, x, y, bw as isize, bh as isize, theme::TEXTBOX_ACTIVE, paint::SHAPE_POWER);
    }
    let mask: Vec<u8> = px.iter().map(|p| (p.alpha() >> (P::DEPTH - 8)) as u8).collect();
    let pill = px.clone();
    let mut rays = px.clone();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::apply_textbox_glow(&mut c, &mask, cx as isize, cy as isize, bw, bh, theme::WHITE);
    }
    {
        let mut c = Canvas::new(&mut rays, W, H, &mut dmg);
        let (w, h) = (bw as isize, bh as isize);
        paint::apply_textbox_glow_right(&mut c, x, y, w, h, theme::GLOW_DEFAULT, 0x80, 236, None);
        paint::apply_textbox_glow_left(&mut c, x, y, w, h, theme::GLOW_DEFAULT, 0x80, 236, None);
        paint::apply_textbox_glow_top(&mut c, x, y, w, h, theme::GLOW_DEFAULT, 0x40, 220, None);
        paint::apply_textbox_glow_bottom(&mut c, x, y, w, h, theme::GLOW_DEFAULT, 0x40, 220, Some(Clip::new(0, 0, 200, H)));
    }
    (pill, px, rays)
}

#[test]
fn golden_glow() {
    let (pill, px, rays) = glows::<Argb8>();
    assert_ne!(px, pill, "the glow painted nothing");
    assert_ne!(rays, pill, "the ray glows painted nothing");
    check("glow", &px, 0xbb0c_f158_2b71_398d);
    check("glow_rays", &rays, 0x9f2c_8327_2e06_eceb);
}

/// The window drop shadow's diagonal rays, cast from a squircle-cornered window already finalized into a screen buffer.
#[test]
fn golden_shadow() {
    let (sw, sh) = (W, H);
    let (ww, wh) = (200usize, 120usize);
    let mut win = vec![0u32; ww * wh];
    let mut wd = Damage::new();
    {
        let mut c = Canvas::new(&mut win, ww, wh, &mut wd);
        paint::draw_squircle_rrect(&mut c, 0, 0, ww as isize, wh as isize, theme::WINDOW_CONTROLS_BG, paint::asymmetric_radii(30.), paint::SHAPE_POWER);
    }
    let mut screen = vec![0u32; sw * sh];
    let clip = vec![255u8; ww * wh];
    let (rx, ry) = (40i32, 20i32);
    paint::finalize_into_screen(&win, &clip, ww, wh, &mut screen, sw, rx, ry, PixelRect::new(0, 0, ww, wh), true, false);
    paint::paint_shadow(&mut screen, sw, 240, 0x80, (rx, ry, ww as i32, wh as i32));
    check("shadow", &screen, 0x4c5b_f4f6_0458_97bb);
}

/// The present boundary's scene at depth `P`: a fill, a two-tone pill, a translucent band and a translucent white pill, with a clip mask that is opaque in the middle, empty at the very edge and ramps in between, so every finalize branch (full, partial, cut) sees pixels.
fn finalize_scene<P: Pixel>() -> (Vec<P>, Vec<u8>) {
    let (mut px, mut dmg) = scene::<P>();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::fill_rect(&mut c, 0, 0, W as isize / 2, H as isize, theme::WINDOW_CONTROLS_BG, None, None);
        paint::draw_squircle_pill_two_tone(&mut c, 20, 20, 200, 60, theme::TEXTBOX_LIGHT_EDGE, theme::TEXTBOX_SHADOW_EDGE, paint::SHAPE_POWER, None, 0);
        paint::fill_rect(&mut c, 30, 100, 260, 60, (theme::TEXTBOX_SELECTION_BG & 0x00FF_FFFF) | 0x7000_0000, None, None);
        paint::draw_squircle_pill(&mut c, 230, 20, 80, 60, (theme::WHITE & 0x00FF_FFFF) | 0x9000_0000, paint::SHAPE_POWER);
    }
    let clip: Vec<u8> = (0..W * H)
        .map(|i| {
            let (x, y) = (i % W, i / W);
            let d = x.min(y).min(W - 1 - x).min(H - 1 - y);
            if d >= 16 { 255 } else { (d * 16) as u8 }
        })
        .collect();
    (px, clip)
}

/// The scene finalized into a larger screen three ways: full, damage-clipped with the window hanging off the left edge, and over an under layer.
fn finalize_presents<P: paint::Finalize>(px: &[P], clip: &[u8]) -> [Vec<u32>; 3] {
    let (sw, sh) = (W + 64, H + 48);
    let mut full = vec![0u32; sw * sh];
    paint::finalize_into_screen(px, clip, W, H, &mut full, sw, 24, 16, PixelRect::new(0, 0, W, H), true, false);
    let mut damaged = vec![0u32; sw * sh];
    paint::finalize_into_screen(px, clip, W, H, &mut damaged, sw, -8, 30, PixelRect::new(10, 12, 250, 170), false, false);
    let mut over = vec![0u32; sw * sh];
    paint::finalize_into_screen(px, clip, W, H, &mut over, sw, 24, 16, PixelRect::new(0, 0, W, H), true, true);
    [full, damaged, over]
}

/// The present boundary: in-place finalize and finalize into a larger screen, full and damage-clipped, plain and over an under layer, with a clip mask that has partial coverage.
#[test]
fn golden_finalize() {
    let (px, clip) = finalize_scene::<Argb8>();
    let mut in_place = px.clone();
    paint::finalize_for_os(&mut in_place, &clip);
    let [full, damaged, over] = finalize_presents(&px, &clip);

    check("finalize_in_place", &in_place, 0x3d1a_9c93_0655_c607);
    check("finalize_into_full", &full, 0x90a8_55b0_fc87_b707);
    check("finalize_into_damaged", &damaged, 0x62bb_dade_acd6_ce07);
    assert_ne!(over, full, "the over finalize took the plain path");
    check("finalize_into_over", &over, 0xb183_73ac_ecee_80ca);
    check8("finalize_clip_mask", &clip, 0xa5b2_9342_0fa7_8f25);
}

/// The 16-bit present of the same scene rendered at 16 bits: the dithered 8-bit screen pixels hashed (the dither is a function of value and screen position, so the hash is as fixed as the 8-bit one), and held against the 8-bit present byte for byte.
/// Measured 16 − 8 in [−1, 3] (full, damaged) and [−2, 3] (over), under a quarter of the bytes differing at all: two biases that partly cancel, the 16-bit render up to 3 of its steps above the 8-bit one (the cross-check below, these primitives' floors) and the present itself 0 to 4 above (`paint::finalize`'s `g2t16_present_tracks_argb8`), against the dither, which is a step either way; the over finalize divides darkness by a partial α, which can stretch an 8-bit floor of under one step to two.
#[test]
fn golden_finalize_g2t16() {
    use crate::pixel::G2t16;
    let (px8, clip) = finalize_scene::<Argb8>();
    let (px16, _) = finalize_scene::<G2t16>();
    let p8 = finalize_presents(&px8, &clip);
    let p16 = finalize_presents(&px16, &clip);
    for (name, a, b) in [("full", &p8[0], &p16[0]), ("damaged", &p8[1], &p16[1]), ("over", &p8[2], &p16[2])] {
        let (mut low, mut high, mut differing) = (0i32, 0i32, 0usize);
        for (&e, &w) in a.iter().zip(b.iter()) {
            for sh in [24, 16, 8, 0] {
                let d = ((w >> sh) & 0xFF) as i32 - ((e >> sh) & 0xFF) as i32;
                low = low.min(d);
                high = high.max(d);
                differing += (d != 0) as usize;
            }
        }
        std::eprintln!("present {name:<8} 16 − 8 in [{low:>2}, {high:>2}]  differing {:>6.2} %", 100. * differing as f64 / (4 * a.len()) as f64);
        assert!(low >= -2 && high <= 3, "{name}: the 16-bit present sits in [{low}, {high}] around the 8-bit one");
    }
    assert_ne!(p16[2], p16[0], "the over finalize took the plain path");
    check("finalize_g2t16_full", &p16[0], 0x5cda_03db_9ce9_cdbd);
    check("finalize_g2t16_damaged", &p16[1], 0x2955_7ddc_6b0b_ab92);
    check("finalize_g2t16_over", &p16[2], 0x5195_f6f9_f203_92e4);
}

/// Two layers of random pixels that keep the `dark ≤ α` invariant every buffer holds, 8-bit, from fixed seeds.
fn random_layers() -> (Vec<u32>, Vec<u32>) {
    let make = |seed: u32| -> Vec<u32> {
        let mut v = vec![0u32; W * H];
        let mut s = seed;
        for p in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let a = s >> 24;
            // Keep each channel's darkness at or under α, the invariant every primitive maintains.
            let d = |sh: u32| ((s >> sh) & 0xFF) * a >> 8;
            *p = (a << 24) | (d(16) << 16) | (d(8) << 8) | d(0);
        }
        v
    };
    (make(0x1234_5678), make(0x9abc_def1))
}

/// An 8-bit composite (a layer built by `under` from empty, darkness (top code − code) × α) at depth `P`: its darkness widens as `256·d + α` per 8 bits of depth gained, which at full α is `from_g2t8`'s `+ 255`.
fn widen_composite<P: Pixel>(p: u32) -> P {
    let gain = P::DEPTH - 8;
    let a = p >> 24;
    P::from_channels(a * (P::ONE / 255), Argb8::channels(p).map(|d| (d << gain) + ((a << gain) - a) / 255))
}

const MODES: [BlendMode; 8] = [
    BlendMode::Normal,
    BlendMode::Multiply,
    BlendMode::Screen,
    BlendMode::Add,
    BlendMode::Subtract,
    BlendMode::Overlay,
    BlendMode::Darken,
    BlendMode::Lighten,
];

/// `flatten` in every blend mode (Normal takes the depth's chunk kernel, SIMD at 8 bits where enabled): the top is a composite, the bottom a source layer, each widened to `P` as such.
fn flatten_modes<P: Pixel>() -> Vec<P> {
    let (top, bottom) = random_layers();
    let top: Vec<P> = top.iter().map(|&p| widen_composite(p)).collect();
    let bottom: Vec<P> = bottom.iter().map(|&p| P::from_g2t8(p)).collect();
    let mut all = Vec::with_capacity(W * H * MODES.len());
    for mode in MODES {
        let mut dst = top.clone();
        paint::flatten(&mut dst, &bottom, mode);
        all.extend_from_slice(&dst);
    }
    all
}

/// The stack compositor's program of pushes, a constant and unders, over two composite layers.
fn stack<P: Pixel>() -> Vec<P> {
    use crate::stack::{Op, StackCompositor};
    let (top, bottom) = random_layers();
    let mut stack = StackCompositor::<P>::with_depth(W, H);
    let l0 = stack.new_layer();
    let l1 = stack.new_layer();
    for (dst, src) in [(l0, &top), (l1, &bottom)] {
        for (d, &s) in stack.layers[dst].pixels.iter_mut().zip(src.iter()) {
            *d = widen_composite(s);
        }
    }
    stack.set_program(vec![Op::Push(l0), Op::Push(l1), Op::Under(BlendMode::Normal), Op::Constant(theme::BG_BASE | 0xFF00_0000), Op::Under(BlendMode::Screen)]);
    stack.evaluate().to_vec()
}

#[test]
fn golden_compositing() {
    check("flatten_modes", &flatten_modes::<Argb8>(), 0xab87_5179_3d8c_183c);
    check("stack", &stack::<Argb8>(), 0xfb5e_fc65_bbc7_6bb3);
}

/// The background noise field, its own base and a given one, scrolled.
fn noise<P: Pixel>() -> (Vec<P>, Vec<P>) {
    let (mut px, mut dmg) = scene::<P>();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        paint::background_noise(&mut c, 3, false, 0, None, None);
    }
    let (mut px2, mut dmg2) = scene::<P>();
    {
        let mut c = Canvas::new(&mut px2, W, H, &mut dmg2);
        paint::background_noise(&mut c, 7, true, 13, Some(Clip::new(0, 20, W, 150)), Some(theme::BUTTON_FILL));
    }
    (px, px2)
}

#[test]
fn golden_background_noise() {
    let (px, px2) = noise::<Argb8>();
    check("noise", &px, 0xe5d4_3657_bbf2_f4cd);
    check("noise_based_clipped", &px2, 0xdec6_5eba_0ebb_1243);
}

#[cfg(feature = "text")]
mod text_scenes {
    use super::*;
    use crate::paint::HitId;
    use crate::text::{TextRenderer, TextStyle};
    use crate::widgets::{Button, Dropdown, Textbox};

    /// Left, centred and right-aligned text, a bold weight, a sheared and a rotated run (the transform glyph path), and a clipped, masked run.
    pub(super) fn text<P: Pixel>() -> Vec<P> {
        let mut text = TextRenderer::new();
        let (mut px, mut dmg) = scene::<P>();
        let mask_bytes: Vec<u8> = (0..W * H).map(|i| (255 - (i % W) * 255 / W) as u8).collect();
        let mask = AlphaMask::new(&mask_bytes, W, H);
        {
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            text.draw_text_left(&mut c, "Photon fluor Ag 0189", 6., 24., &TextStyle::new(20., theme::TEXT_COLOUR), None, None);
            text.draw_text_center(&mut c, "centred ♥ ✓", W as f32 / 2., 56., &TextStyle::new(24., theme::TEXTBOX_TEXT).weight(700), None, None);
            text.draw_text_right(&mut c, "right side", W as f32 - 6., 88., &TextStyle::new(16., theme::LABEL_COLOUR), None, None);
            text.draw_text_left(&mut c, "sheared", 6., 120., &TextStyle::new(22., theme::TEXT_COLOUR).shear(0.2126), None, None);
            text.draw_text_center(&mut c, "rotated", 220., 140., &TextStyle::new(22., theme::TEXT_COLOUR).rotation(0.3), None, None);
            text.draw_text_left(&mut c, "clipped and masked text", 6., 176., &TextStyle::new(26., theme::WHITE), Some(Clip::new(20, 150, 260, 192)), Some(&mask));
        }
        px
    }

    #[test]
    fn golden_text() {
        // With the colour-emoji face bundled the fallback chain reaches it first for ♥ and ✓, so the scene has two recorded values.
        check("text", &text::<Argb8>(), if cfg!(feature = "emoji") { 0xbb03_89af_6962_a770 } else { 0xe5f7_3920_8b1e_bb53 });
    }

    /// A button at rest, then with baked states (as a headless host draws it) hovered, pressed on a custom fill (the derived held ramp), and focused (its glow), each a fresh render.
    pub(super) fn button<P: Pixel>() -> Vec<Vec<P>> {
        let mut text = TextRenderer::new();
        let mut ids: HitId = 0;
        let mut b = Button::<P>::with_depth(&mut ids, W as f32 / 2., H as f32 / 2., 160., 44., 20., "Attest");
        let mut frames: Vec<Vec<P>> = Vec::new();
        let mut render = |b: &mut Button<P>, text: &mut TextRenderer| {
            let (mut px, mut dmg) = scene::<P>();
            let mut hit = vec![0 as HitId; W * H];
            {
                let mut c = Canvas::new(&mut px, W, H, &mut dmg);
                let id = b.hit_id();
                b.render_content_into(&mut c, 0., 0., text, None, Some(&mut hit), id);
            }
            frames.push(px);
        };
        render(&mut b, &mut text);
        b.set_bake_states(true);
        b.set_hovered(true);
        render(&mut b, &mut text);
        b.set_hovered(false);
        b.set_fill(Some(theme::CLOSE_HOVER));
        b.set_pressed(true);
        render(&mut b, &mut text);
        b.set_pressed(false);
        b.set_focused(true);
        render(&mut b, &mut text);
        frames
    }

    #[test]
    fn golden_button() {
        let frames = button::<Argb8>();
        for i in 1..frames.len() {
            assert_ne!(frames[i], frames[i - 1], "button frame {i} repeats the one before it");
        }
        check("button", &frames.concat(), 0x8dab_682d_768a_5774);
    }

    /// A textbox with text, unfocused, then focused with a selection (glow, selection band), then its blinkey.
    pub(super) fn textbox<P: Pixel>() -> Vec<Vec<P>> {
        let mut text = TextRenderer::new();
        let mut ids: HitId = 0;
        let mut t = Textbox::<P>::with_depth(&mut ids, W as f32 / 2., H as f32 / 2., 240., 48., 22.);
        t.set_text("hello golden world", &mut text);
        let mut frames = Vec::new();
        {
            let (mut px, mut dmg) = scene::<P>();
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            t.render_content_into(&mut c, 0., 0., &mut text, None, None, None, t.hit_id());
            frames.push(px);
        }
        t.set_focused(true);
        t.select_word_at(7);
        {
            let (mut px, mut dmg) = scene::<P>();
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            t.render_content_into(&mut c, 0., 0., &mut text, None, None, None, t.hit_id());
            frames.push(px);
        }
        t.cursor_end();
        t.flip_blinkey();
        {
            let (mut px, mut dmg) = scene::<P>();
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            t.render_content_into(&mut c, 0., 0., &mut text, None, None, None, t.hit_id());
            t.render_blinkey_into(&mut c, 0., 0.);
            frames.push(px);
        }
        frames
    }

    #[test]
    fn golden_textbox() {
        let frames = textbox::<Argb8>();
        for i in 1..frames.len() {
            assert_ne!(frames[i], frames[i - 1], "textbox frame {i} repeats the one before it");
        }
        check("textbox", &frames.concat(), 0x5bb9_fae3_3104_3831);
    }

    /// A dropdown closed, then open with its popup painted first.
    pub(super) fn dropdown<P: Pixel>() -> Vec<Vec<P>> {
        let mut text = TextRenderer::new();
        let mut ids: HitId = 0;
        let options = vec!["Zil".into(), "Zila".into(), "Zilor".into(), "Ter".into()];
        let mut d = Dropdown::<P>::with_depth(&mut ids, W as f32 / 2., 30., 160., 36., 18., options);
        d.set_selected(2);
        let mut frames = Vec::new();
        {
            let (mut px, mut dmg) = scene::<P>();
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            d.render_content_into(&mut c, 0., 0., &mut text, None, None);
            frames.push(px);
        }
        d.open();
        {
            let (mut px, mut dmg) = scene::<P>();
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            d.render_popup_into(&mut c, &mut text, None, None);
            d.render_content_into(&mut c, 0., 0., &mut text, None, None);
            frames.push(px);
        }
        frames
    }

    #[test]
    fn golden_dropdown() {
        let frames = dropdown::<Argb8>();
        for i in 1..frames.len() {
            assert_ne!(frames[i], frames[i - 1], "dropdown frame {i} repeats the one before it");
        }
        check("dropdown", &frames.concat(), 0x9097_5d06_bf10_3139);
    }

    /// Colour emoji thru the glyph's own pixels, only where the font is bundled.
    #[cfg(feature = "emoji")]
    #[test]
    fn golden_emoji() {
        let mut text = TextRenderer::new();
        let (mut px, mut dmg) = scene();
        {
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            text.draw_text_left(&mut c, "hi \u{2603}\u{1F600}\u{1F44D}\u{1F3FD}", 6., 60., &TextStyle::new(40., theme::TEXT_COLOUR), None, None);
        }
        check("emoji", &px, 0xb5ed_6608_8f8e_854a);
    }
}

/// The window chrome without a window at depth `P`: edges and the clip-mask carve, the button strip's background, curves and hairlines, the three control glyphs and the title text.
#[cfg(all(feature = "text", feature = "icon"))]
fn chrome_scene<P: Pixel>() -> (Vec<P>, Vec<u8>) {
    use crate::host::chrome;
    use crate::host::chrome_widget::compute_squircle_crossings;
    use crate::paint::{HitId, HIT_NONE};
    let (w, h) = (480usize, 320usize);
    let mut px = vec![P::EMPTY; w * h];
    let mut hit = vec![HIT_NONE as HitId; w * h];
    let mut clip = vec![255u8; w * h];
    let span = 2. * w as f32 * h as f32 / (w as f32 + h as f32);
    let base = span / 4.;
    let (start_big, cross_big) = compute_squircle_crossings(base * 2., 24);
    let (start_small, cross_small) = compute_squircle_crossings(base, 24);
    chrome::draw_window_edges_and_mask(&mut px, &mut hit, &mut clip, w as u32, h as u32, start_big, &cross_big, start_small, &cross_small, theme::WINDOW_LIGHT_EDGE, theme::WINDOW_SHADOW_EDGE);
    let button = 40usize;
    let (bs, bc) = compute_squircle_crossings(button as f32 / 2., 8);
    chrome::draw_strip_bg(&mut px, &mut hit, w as u32, h as u32, button, bs, &bc);
    chrome::draw_strip_curves(&mut px, &mut hit, w as u32, h as u32, button, bs, &bc, theme::WINDOW_LIGHT_EDGE, theme::WINDOW_SHADOW_EDGE);
    chrome::draw_strip_hairlines(&mut px, w as u32, h as u32, button, bs, &bc, theme::WINDOW_LIGHT_EDGE);
    chrome::draw_minimize_symbol(&mut px, w, h, w - 100, 20, 8, theme::MINIMIZE_GLYPH, theme::WINDOW_CONTROLS_BG);
    chrome::draw_maximize_symbol(&mut px, w, h, w - 60, 20, 8, theme::MAXIMIZE_GLYPH, theme::MAXIMIZE_GLYPH_INTERIOR, theme::WINDOW_CONTROLS_BG);
    chrome::draw_close_symbol(&mut px, w, h, w - 20, 20, 8, theme::CLOSE_GLYPH, theme::WINDOW_CONTROLS_BG);
    let mut text = crate::text::TextRenderer::new();
    let mut dmg = Damage::new();
    {
        let mut c = Canvas::new(&mut px, w, h, &mut dmg);
        chrome::draw_title_text(&mut c, "fluor golden", &mut text, button, w - 3 * button, 0, 20., theme::TITLE_TEXT);
        chrome::draw_status_bar(&mut c, 24, theme::WINDOW_CONTROLS_BG, theme::WINDOW_LIGHT_EDGE, "status ok", &mut text, theme::LABEL_COLOUR);
    }
    (px, clip)
}

#[cfg(all(feature = "text", feature = "icon"))]
#[test]
fn golden_chrome() {
    let (px, clip) = chrome_scene::<Argb8>();
    check("chrome", &px, 0xa74e_33d5_c701_6453);
    check8("chrome_clip", &clip, 0x6ef3_84cb_654d_5c65);
    let (_, clip16) = chrome_scene::<crate::pixel::G2t16>();
    assert_eq!(clip16, clip, "the clip mask is the crossings' top byte at either depth");
}

/// The app-icon orb at depth `P`: a gradient icon sampled into the disk with its ring (plain, darkened as unfocused, brightened as online), a ring-only status dot, and the press glow under one of them.
#[cfg(all(feature = "text", feature = "icon"))]
fn orb_scene<P: Pixel>() -> Vec<P> {
    use crate::host::chrome;
    use crate::host::icon::Icon;
    use crate::paint::{HitId, HIT_NONE};
    let icon = Icon { width: 16, height: 16, pixels: (0..256u32).map(|i| 0xFF00_0000 | ((i & 15) * 16) << 16 | ((i >> 4) * 16) << 8 | (255 - i)).collect() };
    let mut px = vec![P::EMPTY; W * H];
    let mut hit = vec![HIT_NONE as HitId; W * H];
    chrome::draw_app_icon(&mut px, Some(&mut hit), 7, W, H, 50, 50, 36, Some(&icon), Some(theme::WINDOW_LIGHT_EDGE), 0, false);
    chrome::draw_orb_press_glow(&mut px, W, H, 50, 50, 36);
    chrome::draw_app_icon(&mut px, None, 0, W, H, 140, 60, 30, Some(&icon), Some(theme::LABEL_COLOUR), theme::ORB_DARKEN_UNFOCUSED, false);
    chrome::draw_app_icon(&mut px, None, 0, W, H, 220, 60, 30, Some(&icon), Some(theme::CLOSE_HOVER), 0, true);
    chrome::draw_app_icon(&mut px, None, 0, W, H, 290, 140, 20, None, Some(theme::MAXIMIZE_HOVER), 0, false);
    px
}

#[cfg(all(feature = "text", feature = "icon"))]
#[test]
fn golden_orb() {
    check("orb", &orb_scene::<Argb8>(), 0xe61d_7d15_318b_fe24);
}

/// How far a G2t16 render, narrowed to the nearest 8-bit codes ([`crate::pixel::narrow_to_8`]), sits from the Argb8 render of the same scene, byte by byte (α and each channel), signed as 16-bit minus 8-bit: the extremes, the share of bytes that differ at all, and the share that differ by more than one step.
struct Agreement {
    low: i32,
    high: i32,
    differing: usize,
    beyond_one: usize,
    bytes: usize,
}

fn agreement(p8: &[u32], p16: &[u64]) -> Agreement {
    assert_eq!(p8.len(), p16.len());
    let mut a = Agreement { low: 0, high: 0, differing: 0, beyond_one: 0, bytes: 4 * p8.len() };
    for (&e, &w) in p8.iter().zip(p16.iter()) {
        let n = crate::pixel::narrow_to_8(w);
        for sh in [24, 16, 8, 0] {
            let d = ((n >> sh) & 0xFF) as i32 - ((e >> sh) & 0xFF) as i32;
            a.low = a.low.min(d);
            a.high = a.high.max(d);
            a.differing += (d != 0) as usize;
            a.beyond_one += (d.abs() > 1) as usize;
        }
    }
    a
}

/// The cross-check: every depth-generic golden scene rendered at both depths, the 16-bit one narrowed to the nearest 8-bit codes and compared byte for byte with the 8-bit render.
///
/// The bar the plan set (within one step, under 0.1 % of bytes differing) is not reachable while the 8-bit kernels stay bit-identical, and the differences all point one way: the 16-bit render, narrowed, is at or above the 8-bit one.
/// The 8-bit `under` takes `>> 8` for `÷ 255` and floors, so an opaque colour laid into an empty pixel deposits `(d × 255) >> 8 = d − 1` (the noise scenes: every byte off by exactly one, none by more); partial cover floors twice more (paint_rects, shapes, squircles, stack, up to 3–4); the rotated rect's four coverage bytes floor before they multiply; and the glow's α table floors at every step of its decay, drifting up to 7 steps low over a ray (glow_rays).
/// At 16 bits each of those floors costs 1/256 of a step, so the 16-bit render is the nearer one to the exact composite, and what this test can hold is the shape of the difference: no byte where the 16-bit render falls more than one step below the 8-bit one (a kernel still thinking in bytes would lose light or coverage at 16 bits, and it lands far below, not one step), and the excess above it within each scene's floors.
#[test]
fn crosscheck_g2t16_against_argb8() {
    use crate::pixel::G2t16;
    let (_, glow8, rays8) = glows::<Argb8>();
    let (_, glow16, rays16) = glows::<G2t16>();
    let (noise8, noise_b8) = noise::<Argb8>();
    let (noise16, noise_b16) = noise::<G2t16>();
    // Name, 8-bit render, 16-bit render, and the range the narrowed 16-bit render may sit in around the 8-bit one (16 − 8, in 8-bit steps).
    let mut scenes: Vec<(String, Vec<u32>, Vec<u64>, (i32, i32))> = vec![
        ("paint_rects".into(), paint_rects(), paint_rects(), (-1, 3)),
        ("shapes".into(), shapes(), shapes(), (-1, 4)),
        ("ellipses".into(), ellipses(), ellipses(), (-1, 3)),
        ("squircles".into(), squircles(), squircles(), (-1, 3)),
        ("glow".into(), glow8, glow16, (-1, 3)),
        ("glow_rays".into(), rays8, rays16, (-1, 7)),
        ("stack".into(), stack(), stack(), (-1, 2)),
        ("noise".into(), noise8, noise16, (-1, 1)),
        ("noise_based_clipped".into(), noise_b8, noise_b16, (-1, 1)),
    ];
    let (modes8, modes16) = (flatten_modes::<Argb8>(), flatten_modes::<G2t16>());
    for (i, mode) in MODES.into_iter().enumerate() {
        let r = i * W * H..(i + 1) * W * H;
        // Multiply's 8-bit `255 − (product >> 8)` rounds the product down and so the result up, where the 8-bit render can sit above the 16-bit one by two; Overlay's dark branch is the same expression doubled.
        let low = if mode == BlendMode::Multiply || mode == BlendMode::Overlay { -2 } else { -1 };
        scenes.push((alloc::format!("flatten {mode:?}"), modes8[r.clone()].to_vec(), modes16[r].to_vec(), (low, 3)));
    }
    #[cfg(feature = "text")]
    scenes.push(("text".into(), text_scenes::text(), text_scenes::text(), (-1, 2)));
    // The widgets, frame by frame, each frame its own row: a pill with its label sits within the shapes' floors (3); a focused frame carries the focus glow, whose 8-bit decay table drifts low as glow_rays does (7) — button frame 3, textbox frames 1 and 2.
    #[cfg(feature = "text")]
    for (name, f8, f16, highs) in [
        ("button", text_scenes::button::<Argb8>(), text_scenes::button::<G2t16>(), &[3, 3, 3, 7][..]),
        ("textbox", text_scenes::textbox::<Argb8>(), text_scenes::textbox::<G2t16>(), &[3, 7, 7][..]),
        ("dropdown", text_scenes::dropdown::<Argb8>(), text_scenes::dropdown::<G2t16>(), &[3, 3][..]),
    ] {
        assert_eq!(f8.len(), highs.len());
        for (i, (p8, p16)) in f8.into_iter().zip(f16).enumerate() {
            scenes.push((alloc::format!("{name} frame {i}"), p8, p16, (-1, highs[i])));
        }
    }
    // The window chrome and the orb: the perimeter, strip and glyph edges now carry their coverage at the depth's own precision (the crossings' Q16 weights, the glyph ramps × ONE), so the 8-bit floors on them are the same kind as the shapes'.
    #[cfg(all(feature = "text", feature = "icon"))]
    {
        scenes.push(("chrome".into(), chrome_scene::<Argb8>().0, chrome_scene::<G2t16>().0, (-1, 3)));
        scenes.push(("orb".into(), orb_scene(), orb_scene(), (-1, 3)));
    }
    let mut failures = Vec::new();
    for (name, p8, p16, (low, high)) in &scenes {
        let a = agreement(p8, p16);
        let pct = |n: usize| 100. * n as f64 / a.bytes as f64;
        std::eprintln!("cross-check {name:<20} 16 − 8 in [{:>2}, {:>2}]  differing {:>6.2} %  beyond one {:>5.2} %", a.low, a.high, pct(a.differing), pct(a.beyond_one));
        if a.low < *low {
            failures.push(alloc::format!("{name}: the 16-bit render falls {} steps below the 8-bit one (its floors allow {})", -a.low, -low));
        }
        if a.high > *high {
            failures.push(alloc::format!("{name}: the 16-bit render sits {} steps above the 8-bit one (its floors allow {high})", a.high));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
