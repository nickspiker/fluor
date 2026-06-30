//! Phase-1 spike for the toka→fluor render migration.
//!
//! Proves the contract toka's `Canvas` will rely on, end to end:
//!   1. colour funnel: visible RGBA → `pack_argb` → α+darkness u32
//!   2. draw via `draw_rect` / `draw_rect_rotated` into an α+darkness buffer
//!   3. output flip: `pixel ^= 0x00FFFFFF` → visible sRGB RGBA bytes
//!   4. the line convention: a zero-dimension rect renders a 1px hairline
//!
//! None of this touches toka; it validates the fluor side in isolation.

use fluor::canvas::{Canvas, Damage};
use fluor::paint::{draw_rect, draw_rect_rotated, pack_argb};

const W: usize = 64;
const H: usize = 64;

/// Flip α+darkness → visible RGBA bytes, exactly as toka's `to_rgba_bytes` will.
fn to_rgba(pixels: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.len() * 4);
    for &p in pixels {
        let v = p ^ 0x00FF_FFFF; // darkness → visible RGB; α byte untouched
        out.push((v >> 16) as u8); // R
        out.push((v >> 8) as u8); // G
        out.push(v as u8); // B
        out.push(0xFF); // opaque surface
    }
    out
}

fn px(rgba: &[u8], x: usize, y: usize) -> (u8, u8, u8) {
    let i = (y * W + x) * 4;
    (rgba[i], rgba[i + 1], rgba[i + 2])
}

#[test]
fn pack_argb_round_trips_through_output_flip() {
    // Pure red, opaque, should survive pack → flip back to (255,0,0).
    let c = pack_argb(255, 0, 0, 255);
    let flipped = c ^ 0x00FF_FFFF;
    assert_eq!((flipped >> 16) & 0xFF, 255, "R");
    assert_eq!((flipped >> 8) & 0xFF, 0, "G");
    assert_eq!(flipped & 0xFF, 0, "B");
    assert_eq!((c >> 24) & 0xFF, 255, "α stays opaque");
}

#[test]
fn filled_rect_lands_opaque_at_centre() {
    let mut pixels = vec![0u32; W * H];
    let mut damage = Damage::new();
    let mut canvas = Canvas::new(&mut pixels, W, H, &mut damage);

    let blue = pack_argb(0, 0, 255, 255);
    draw_rect(&mut canvas, 32.0, 32.0, 20.0, 20.0, blue, None);

    let rgba = to_rgba(&pixels);
    // fluor's blend deposits `(255 × consumed) >> 8 = 254` darkness for an opaque layer — a documented ±1 floor of the integer `>>8` kernel — so an "opaque black" channel reads 1, not 0, after the output flip. Tolerate ≤1.
    let (r, g, b) = px(&rgba, 32, 32);
    assert!(r <= 1 && g <= 1 && b >= 254, "centre is ~blue fill, got ({r},{g},{b})");
    assert_eq!(px(&rgba, 0, 0), (255, 255, 255), "corner is bg (white)");
}

#[test]
fn rotated_rect_renders() {
    let mut pixels = vec![0u32; W * H];
    let mut damage = Damage::new();
    let mut canvas = Canvas::new(&mut pixels, W, H, &mut damage);

    let green = pack_argb(0, 200, 0, 255);
    draw_rect_rotated(&mut canvas, 32.0, 32.0, 24.0, 10.0, 0.6, green, None);

    let rgba = to_rgba(&pixels);
    let (_, g, _) = px(&rgba, 32, 32);
    assert!(g > 100, "centre of rotated rect carries the green fill, got g={g}");
}

#[test]
fn zero_dimension_rect_is_a_one_pixel_line() {
    // The line convention: height 0 → a horizontal 1px hairline at y=20.
    let mut pixels = vec![0u32; W * H];
    let mut damage = Damage::new();
    let mut canvas = Canvas::new(&mut pixels, W, H, &mut damage);

    // Centre the hairline ON pixel row 20 — pixel centres sit at y+0.5, so cy=20.5 puts the 1px coverage band fully inside row 20 (cy=20.0 would split it 50/50 across rows 19 and 20, which is correct AA but not what a crisp grid line wants).
    let black = pack_argb(0, 0, 0, 255);
    draw_rect(&mut canvas, 32.0, 20.5, 40.0, 0.0, black, None);

    let rgba = to_rgba(&pixels);
    let (r_on, _, _) = px(&rgba, 32, 20);
    assert!(r_on <= 1, "hairline row is fully inked, got r={r_on}");
    assert_eq!(px(&rgba, 32, 17), (255, 255, 255), "three rows above is clean bg");
    assert_eq!(px(&rgba, 32, 23), (255, 255, 255), "three rows below is clean bg");
}
