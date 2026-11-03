//! The window's edge, as geometry: the superellipse corners' crossings and the four-corner walk that carves a clip mask and lays a one-pixel hairline along the perimeter.
//! Shared by the desktop chrome ([`crate::host::chrome::draw_window_edges_and_mask`] is a wrapper that paints its bevel colours), by Photon on Android glass (the same chrome with [`glass_corner_params`]) and by ferros's display server, which draws the screen's own edge from this walk (payloads/display, 2026-11-02).
//! The walk owns no buffer and knows no pixel type: it reports every clip-mask value and every hairline pixel it would write, in the order it writes them, and the caller stores them as it likes (the chrome into its pixel layer and byte mask, ferros into a per-row plan). Coverages are Q16, as [`crate::paint::squircle_crossings`] gives them.

use alloc::vec::Vec;

/// Which part of the perimeter a hairline pixel belongs to: a straight edge, or a corner (the corner's bevel colour changes along its curve on the desktop).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EdgeSide {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// The glass-mode corner geometry `(r_small, r_big, exponent)`: the superellipse `((R−x)/R)^n + ((R−y)/R)^n = 1` sits INSIDE a circular glass arc of radius `g` iff it cuts at least as deep at the diagonal, where the two curves are closest — the circle cuts `g·(1 − 1/√2)` ≈ 0.293 g in from the corner, the superellipse `R·(1 − 2^(−1/n))` — so the small radius is `R = 0.293 g / (1 − 2^(−1/n))` (a hair more for the AA), and the big TL/BR diagonal is twice it, the desktop's 2:1. At the desktop's exponent 24 that is ~10.3 g — a 68 px glass wants a 700 px small corner and a 1400 px big one, more than a phone's width holds beside each other (the top edge carries one of each). So the exponent is the LARGEST n ≤ 24 whose corners fit: `r_small + r_big ≤ 0.9·min(w,h)`, i.e. `1 − 2^(−1/n) ≥ 3·0.293 g / (0.9·min(w,h))`; a 1008 px portrait phone with 68 px glass lands at n = 10 (297 / 594 px). Never below 3: at 2 the corner is a plain circle and reads as a lozenge.
pub fn glass_corner_params(g: f32, w: f32, h: f32) -> (f32, f32, i32) {
    // ONE AND A HALF glass insets at the diagonal, exponent 6 (Nick 2026-09-17: "exponent 6 and 1.5", after 2× at exponent 4 read as a soft blob): the small corner cuts half a glass-inset deeper than the arc, the big one three insets, and the squarer superellipse is what makes the corner read tight rather than round.
    let cut = 1.5 * (g * (1.0 - core::f32::consts::FRAC_1_SQRT_2) + 1.0);
    // FIT is the renderer's own topology, not a radius sum: an edge splits into straight run + two corner CAPS (the rows where the curve is at least a pixel off the edge — near r itself for a high exponent), and `walk_window_edges` falls back to a plain rectangle when the two caps on one edge don't fit. So the exponent is the largest n ≤ 24 whose two caps fit the short edge with a tenth to spare. A doubled inset at high n wants caps wider than a phone (n 9 → 553 + 1106 px on a 1008 px edge), so the phone lands lower (n 4 at 68 px glass: 252 + 504); the desktop keeps 24.
    let short = w.min(h) * 0.9;
    let mut n = 6;
    while n > 3 {
        let r = cut / (1.0 - crate::math::powf(2.0, -1.0 / n as f32));
        let (s1, c1) = compute_squircle_crossings(r, n);
        let (s2, c2) = compute_squircle_crossings(r * 2.0, n);
        if ((s1 + c1.len() + s2 + c2.len()) as f32) <= short {
            break;
        }
        n -= 1;
    }
    // Nine tenths of the fitted size (Nick 2026-09-17, "let's try making both radius on Android 0.9 what they are now (slightly smaller)") — a smaller corner at the same exponent always fits, so the trim is applied after the search.
    let r_small = cut / (1.0 - crate::math::powf(2.0, -1.0 / n as f32)) * 0.9;
    (r_small, r_small * 2.0, n)
}

/// Compute squircle crossings table for a corner with the given `radius` and `squirdleyness`. Returns `(start, crossings)` where `start` is the distance from the corner-of-corner inward to the curve's first integer-row crossing, and `crossings` is the rev'd table indexed by `i in 0..count` such that at row offset `start + i` the curve is at column `inset_i` with AA values `h_cov_i` and `l_i`. Used for both the window perimeter (radius = span/4) and the controls-strip BL curve (radius = button_size).
/// The AA values are Q16 (`√frac × 2¹⁶` and `√(1 − frac) × 2¹⁶`, each cast to u16, a whole pixel saturating at `0xFFFF`), as [`crate::paint::squircle_crossings`] gives them: a 16-bit layer takes the whole weight as its α ([`crate::pixel::Pixel::from_cov`]), an 8-bit layer and the clip mask the top byte, which is bit for bit the `× 256` byte this table held before.
pub fn compute_squircle_crossings(radius: f32, squirdleyness: i32) -> (usize, Vec<(u16, u16, u16)>) {
    let mut crossings: Vec<(u16, u16, u16)> = Vec::new();
    if radius <= 0.0 {
        return (0, crossings);
    }
    let mut y = 1f32;
    loop {
        let y_norm = y / radius;
        // For y_norm > 1, the squircle equation gives a negative inner term — clamp to 0 so the (1/p) root is well-defined. This makes x = 0 → x < y → break, preventing the loop from spinning forever on tiny radii.
        let inner = (1.0 - crate::math::powi(y_norm, squirdleyness)).max(0.0);
        let x_norm = crate::math::powf(inner, 1.0 / squirdleyness as f32);
        let x = x_norm * radius;
        let inset = radius - x;
        if inset > 0.0 {
            crossings.push((
                inset as u16,
                (crate::math::sqrt(crate::math::fract(inset)) * 65536.) as u16,
                (crate::math::sqrt(1.0 - crate::math::fract(inset)) * 65536.) as u16,
            ));
        }
        if x < y {
            break;
        }
        y += 1.0;
    }
    let start = (radius - y) as usize;
    crossings.reverse();
    (start, crossings)
}

/// The four-corner walk of a `width × height` perimeter whose top-left and bottom-right corners come from the big table and top-right and bottom-left from the small one (an asymmetric rounded rectangle built from one curve at two scales), reported as writes: `clip(index, coverage)` for every clip-mask pixel the shape carves (0 outside the curve, the Q16 coverage on the curve's outer pixel, `0xFFFF` on the inner one) and `hairline(index, side, alpha)` for every hairline pixel (opaque along the straight edges and on the curve's outer pixel, `0xFFFF − coverage` on the inner pixel, so the hairline's own antialiasing is the coverage's complement). Both in the order the chrome writes them: a later write to the same pixel is the one the chrome keeps.
/// Each corner's row walk covers its near-vertical segment and its column walk the near-horizontal one; the squircle is self-symmetric within a corner, so both share that corner's table, which is what keeps the curvature continuous.
/// Returns false when the window is too small for two corner caps per axis: then only a plain one-pixel rectangle is reported and the mask is left as it was, so extreme shrinks stay panic-free.
#[allow(clippy::too_many_arguments)]
pub fn walk_window_edges(
    width: u32,
    height: u32,
    start_big: usize,
    crossings_big: &[(u16, u16, u16)],
    start_small: usize,
    crossings_small: &[(u16, u16, u16)],
    clip: &mut impl FnMut(usize, u32),
    hairline: &mut impl FnMut(usize, EdgeSide, u32),
) -> bool {
    let w = width as usize;
    let h = height as usize;
    let cap_big = start_big + crossings_big.len();
    let cap_small = start_small + crossings_small.len();

    // Degenerate-window guard: below two corner caps per axis the per-corner walks underflow usize (`w - cap` panics) and index out of bounds.
    if w < cap_big + cap_small + 2 || h < cap_big + cap_small + 2 {
        if w == 0 || h == 0 {
            return false;
        }
        for x in 0..w {
            hairline(x, EdgeSide::Top, 0xFFFF);
            hairline((h - 1) * w + x, EdgeSide::Bottom, 0xFFFF);
        }
        for y in 0..h {
            hairline(y * w, EdgeSide::Left, 0xFFFF);
            hairline(y * w + (w - 1), EdgeSide::Right, 0xFFFF);
        }
        return false;
    }

    // Straight edges. Each spans between the caps of its two end corners, which differ: the top edge runs from the TL big cap to the TR small cap. The clip mask along them stays as the caller left it (fully visible).
    for x in cap_big..w - cap_small {
        hairline(x, EdgeSide::Top, 0xFFFF);
    }
    for x in cap_small..w - cap_big {
        hairline((h - 1) * w + x, EdgeSide::Bottom, 0xFFFF);
    }
    for y in cap_big..h - cap_small {
        hairline(y * w, EdgeSide::Left, 0xFFFF);
    }
    for y in cap_small..h - cap_big {
        hairline(y * w + (w - 1), EdgeSide::Right, 0xFFFF);
    }

    // Corner-of-corner cutout (start × start at each corner): the small outer square that the curve never reaches. Each corner uses its OWN start (big corners a larger square, small corners a smaller one).
    for r in 0..start_big {
        for c in 0..start_big {
            clip(r * w + c, 0); // TL
        }
    }
    for r in 0..start_small {
        for c in (w - start_small)..w {
            clip(r * w + c, 0); // TR
        }
    }
    for r in (h - start_small)..h {
        for c in 0..start_small {
            clip(r * w + c, 0); // BL
        }
    }
    for r in (h - start_big)..h {
        for c in (w - start_big)..w {
            clip(r * w + c, 0); // BR
        }
    }

    // Two-sided AA convention. OUTER pixel = on the curve, partially outside the window: clip = coverage trims against the OS bg; the hairline stays opaque. INNER pixel = one step inside: clip = full; the hairline's α carries the hairline-vs-bg AA = full − coverage.

    // TL corner (big). Row-walk: near-vertical segment; col-walk: near-horizontal segment. Both use the big table.
    for (i, &(inset_raw, _l, h_cov)) in crossings_big.iter().enumerate() {
        let inset = inset_raw as usize;
        if inset >= cap_big {
            continue;
        }
        let inner_alpha = 0xFFFF - h_cov as u32;
        let row_top = start_big + i;
        let col_left = start_big + i;

        for c in 0..inset {
            clip(row_top * w + c, 0);
        }
        let idx = row_top * w + inset;
        hairline(idx, EdgeSide::TopLeft, 0xFFFF);
        clip(idx, h_cov as u32);
        let idx = row_top * w + inset + 1;
        hairline(idx, EdgeSide::TopLeft, inner_alpha);
        clip(idx, 0xFFFF);

        for r in 0..inset {
            clip(r * w + col_left, 0);
        }
        let idx = inset * w + col_left;
        hairline(idx, EdgeSide::TopLeft, 0xFFFF);
        clip(idx, h_cov as u32);
        let idx = (inset + 1) * w + col_left;
        hairline(idx, EdgeSide::TopLeft, inner_alpha);
        clip(idx, 0xFFFF);
    }

    // BR corner (big).
    for (i, &(inset_raw, _l, h_cov)) in crossings_big.iter().enumerate() {
        let inset = inset_raw as usize;
        if inset >= cap_big {
            continue;
        }
        let inner_alpha = 0xFFFF - h_cov as u32;
        let row_bot = h - 1 - start_big - i;
        let col_right = w - 1 - start_big - i;

        for c in (w - inset)..w {
            clip(row_bot * w + c, 0);
        }
        let idx = row_bot * w + (w - 1 - inset);
        hairline(idx, EdgeSide::BottomRight, 0xFFFF);
        clip(idx, h_cov as u32);
        let idx = row_bot * w + (w - 2 - inset);
        hairline(idx, EdgeSide::BottomRight, inner_alpha);
        clip(idx, 0xFFFF);

        for r in (h - inset)..h {
            clip(r * w + col_right, 0);
        }
        let idx = (h - 1 - inset) * w + col_right;
        hairline(idx, EdgeSide::BottomRight, 0xFFFF);
        clip(idx, h_cov as u32);
        let idx = (h - 2 - inset) * w + col_right;
        hairline(idx, EdgeSide::BottomRight, inner_alpha);
        clip(idx, 0xFFFF);
    }

    // TR corner (small).
    for (i, &(inset_raw, _l, h_cov)) in crossings_small.iter().enumerate() {
        let inset = inset_raw as usize;
        if inset >= cap_small {
            continue;
        }
        let inner_alpha = 0xFFFF - h_cov as u32;
        let row_top = start_small + i;
        let col_right = w - 1 - start_small - i;

        for c in (w - inset)..w {
            clip(row_top * w + c, 0);
        }
        let idx = row_top * w + (w - 1 - inset);
        hairline(idx, EdgeSide::TopRight, 0xFFFF);
        clip(idx, h_cov as u32);
        let idx = row_top * w + (w - 2 - inset);
        hairline(idx, EdgeSide::TopRight, inner_alpha);
        clip(idx, 0xFFFF);

        for r in 0..inset {
            clip(r * w + col_right, 0);
        }
        let idx = inset * w + col_right;
        hairline(idx, EdgeSide::TopRight, 0xFFFF);
        clip(idx, h_cov as u32);
        let idx = (inset + 1) * w + col_right;
        hairline(idx, EdgeSide::TopRight, inner_alpha);
        clip(idx, 0xFFFF);
    }

    // BL corner (small).
    for (i, &(inset_raw, _l, h_cov)) in crossings_small.iter().enumerate() {
        let inset = inset_raw as usize;
        if inset >= cap_small {
            continue;
        }
        let inner_alpha = 0xFFFF - h_cov as u32;
        let row_bot = h - 1 - start_small - i;
        let col_left = start_small + i;

        for c in 0..inset {
            clip(row_bot * w + c, 0);
        }
        let idx = row_bot * w + inset;
        hairline(idx, EdgeSide::BottomLeft, 0xFFFF);
        clip(idx, h_cov as u32);
        let idx = row_bot * w + inset + 1;
        hairline(idx, EdgeSide::BottomLeft, inner_alpha);
        clip(idx, 0xFFFF);

        for r in (h - inset)..h {
            clip(r * w + col_left, 0);
        }
        let idx = (h - 1 - inset) * w + col_left;
        hairline(idx, EdgeSide::BottomLeft, 0xFFFF);
        clip(idx, h_cov as u32);
        let idx = (h - 2 - inset) * w + col_left;
        hairline(idx, EdgeSide::BottomLeft, inner_alpha);
        clip(idx, 0xFFFF);
    }
    true
}
