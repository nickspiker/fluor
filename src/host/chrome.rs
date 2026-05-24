//! Window chrome — minimal top-down rasterization. Each pixel in the chrome layer is written by exactly one site. No painter's algorithm anywhere.
//!
//! Currently scoped to **window perimeter hairline** with squircle corner AA. The chrome layer starts at the canonical empty value (`0xFFFFFFFF`); this function paints only the hairline pixels. Everywhere else in the chrome layer stays transparent so panes / bg can pass through the chrome group's Stack composition. Buttons, glyphs, title text, hover overlay — all deferred to subsequent scaffold steps; reintroduce them only when each can be added without overwriting earlier writes within this same layer.
//!
//! Hit-test IDs and the `ResizeEdge` enum live here so the desktop host's mouse routing can reference them without depending on the (future, larger) controls implementation.
//!
//! The squircle crossings table is consumed but not computed here; the caller (chrome_widget) computes it once per resize and passes it in.
//!
//! All RGB values stored in the chrome layer are straight-α (the canonical buffer convention). The OS conversion layer at the present boundary handles platform-specific premultiplication.

use crate::coord::Coord;
use crate::math;
use crate::pixel::{Blend, BlendMode};
use crate::theme;

/// Hit-test IDs that the per-pixel hit_test_map can carry. `HIT_NONE` = clicks pass through. Button IDs are placeholders for the future controls scaffold step.
pub const HIT_NONE: u8 = 0;
pub const HIT_MINIMIZE_BUTTON: u8 = 1;
pub const HIT_MAXIMIZE_BUTTON: u8 = 2;
pub const HIT_CLOSE_BUTTON: u8 = 3;

/// Resize-edge classification returned by [`get_resize_edge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeEdge {
    None,
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Classify a cursor position as one of nine resize zones (or None for the window interior). Geometry only — no rasterization. Edge band thickness derived from harmonic-mean span so the hit zone scales with viewport size.
pub fn get_resize_edge(window_width: u32, window_height: u32, x: Coord, y: Coord) -> ResizeEdge {
    let span = 2.0 * window_width as Coord * window_height as Coord
        / (window_width as Coord + window_height as Coord);
    let resize_border = math::ceil(span / 32.0);

    let at_left = x < resize_border;
    let at_right = x > (window_width as Coord - resize_border);
    let at_top = y < resize_border;
    let at_bottom = y > (window_height as Coord - resize_border);

    if at_top && at_left {
        ResizeEdge::TopLeft
    } else if at_top && at_right {
        ResizeEdge::TopRight
    } else if at_bottom && at_left {
        ResizeEdge::BottomLeft
    } else if at_bottom && at_right {
        ResizeEdge::BottomRight
    } else if at_top {
        ResizeEdge::Top
    } else if at_bottom {
        ResizeEdge::Bottom
    } else if at_left {
        ResizeEdge::Left
    } else if at_right {
        ResizeEdge::Right
    } else {
        ResizeEdge::None
    }
}

/// Rasterize the window-perimeter hairline into `pixels` (the chrome layer) AND the per-pixel window-shape `clip_mask`. Two outputs, single pass per crossing — the chrome layer carries opaque RGB only (no partial-t), and ALL partial-α information lives in the clip mask. The boundary's [`crate::paint::finalize_for_os`] multiplies the clip mask into each pixel's α before the OS sees it.
///
/// Pre-conditions: `pixels` already at the canonical empty value `0xFFFFFFFF`, `clip_mask` already at the host's default of `255` (fully visible window-interior assumption).
///
/// Topology: straight edges paint opaque RGB in non-corner ranges (`cap..(end-cap)`) and leave the clip mask alone (= 255, fully visible). Each crossing entry handles **one row** of the curve region for the four corners: zero out the cutout cols, write opaque hairline RGB at the curve's outer + inner pixel positions, and write `h_cov` / `l` into the clip mask at those same positions. Above-the-curve rows (`0..start`) and below-the-curve rows (`h-start..h`) are *entirely* cutout — the curve never enters them — so the full cap-width at those rows is zeroed in the clip mask.
///
/// Two-tone bevel (light from upper-left): top + left straight edges are light, bottom + right are shadow. TL and BR corners are uniform (both adjacent edges agree); TR and BL transition along the curve. The per-pixel colour test (`tr_color`, `bl_color`) is the same one we settled on previously — closer-to-light-edge wins.
///
/// `hit_test_map` is preserved as a parameter for forward compatibility with the controls scaffold step but is not modified here.
pub fn draw_window_edges_and_mask(
    pixels: &mut [u32],
    hit_test_map: &mut [u8],
    clip_mask: &mut [u8],
    width: u32,
    height: u32,
    start: usize,
    crossings: &[(u16, u8, u8)],
) {
    let _ = hit_test_map;
    if width < 2 || height < 2 {
        return;
    }
    let w = width as usize;
    let h = height as usize;
    if start * 2 >= w || start * 2 >= h {
        return;
    }

    let light = theme::WINDOW_LIGHT_EDGE;
    let shadow = theme::WINDOW_SHADOW_EDGE;
    let count = crossings.len();
    let cap = start + count;
    if cap * 2 >= w || cap * 2 >= h {
        return;
    }

    // Straight edges — opaque chrome composed via Under. Clip mask along these edges stays at the host's 255 default (fully visible).
    for x in cap..(w - cap) {
        pixels[x] = pixels[x].under(light, BlendMode::Normal); // top row
        let idx = (h - 1) * w + x;
        pixels[idx] = pixels[idx].under(shadow, BlendMode::Normal); // bottom row
    }
    for y in cap..(h - cap) {
        let lidx = y * w;
        pixels[lidx] = pixels[lidx].under(light, BlendMode::Normal); // left col
        let ridx = y * w + (w - 1);
        pixels[ridx] = pixels[ridx].under(shadow, BlendMode::Normal); // right col
    }

    // Corner-of-corner cutout (start × start at each corner): the small outer square that the curve never reaches under any squircle parameter. The rest of the cap (the inner L-shape: rows 0..start × cols start..cap, and rows start..cap × cols 0..start) is handled per-pixel by the curve row-walks (zero c in 0..inset at row=start+i) and col-walks (zero r in 0..inset at col=start+i). Curve interior (rows start..cap × cols start..cap, inside the squircle) stays at the default 255.
    for r in 0..start {
        for c in 0..start {
            clip_mask[r * w + c] = 0;
        }
        for c in (w - start)..w {
            clip_mask[r * w + c] = 0;
        }
    }
    for r in (h - start)..h {
        for c in 0..start {
            clip_mask[r * w + c] = 0;
        }
        for c in (w - start)..w {
            clip_mask[r * w + c] = 0;
        }
    }

    let tr_color =
        |row: usize, col: usize| -> u32 { if row < (w - 1 - col) { light } else { shadow } };
    let bl_color =
        |row: usize, col: usize| -> u32 { if col < (h - 1 - row) { light } else { shadow } };

    // Curve rows AND curve cols: the squircle is symmetric under x↔y swap, so the same crossings table walks both axes. The row-walk handles the corner's near-vertical segment (one row, two-pixel hairline at the curve crossing); the col-walk handles the near-horizontal segment (one col, two-pixel hairline). Both walks together fully cover the corner — without the col-walk, the near-horizontal portion of the corner (where the curve travels many cols per row) shows visible gaps.
    for (i, &(inset_raw, l, h_cov)) in crossings.iter().enumerate() {
        let inset = inset_raw as usize;
        // Guard: in degenerate geometries the curve terminal can sit past the cap boundary. Skip those only. Letting `inset+1 == cap` through is required — that's the curve's natural last hairline pixel meeting the straight edge at the cap join.
        if inset >= cap {
            continue;
        }

        let row_top = start + i;
        let row_bot = h - 1 - start - i;
        let col_left = start + i;
        let col_right = w - 1 - start - i;

        // Two-sided AA convention. OUTER pixel = on the curve, partially outside the window: clip_mask = h_cov trims against the OS bg; chrome stays opaque (t=0) because the entire inside-window portion IS hairline. INNER pixel = one step inside the curve: clip_mask = 255 (fully inside the window shape); chrome's t-byte carries the hairline-vs-bg AA, set to `inner_t` so the Under blend mixes h_cov/256 of hairline color over (256-h_cov)/256 of window bg. The `l` slot in each crossing entry is the linear-coverage counterpart of h_cov, retained for the outer-pixel chrome-t AA when we move from a 2-pixel hairline to a 1-pixel-with-halo hairline.
        let _ = l;
        let inner_t = (h_cov as u32) << 24;
        let light_inner = (light & 0x00FFFFFF) | inner_t;
        let shadow_inner = (shadow & 0x00FFFFFF) | inner_t;

        // TL row-walk
        for c in 0..inset {
            clip_mask[row_top * w + c] = 0;
        }
        let idx = row_top * w + inset;
        pixels[idx] = pixels[idx].under(light, BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = row_top * w + inset + 1;
        pixels[idx] = pixels[idx].under(light_inner, BlendMode::Normal);
        clip_mask[idx] = 255;

        // TR row-walk
        for c in (w - inset)..w {
            clip_mask[row_top * w + c] = 0;
        }
        let tr_out_col = w - 1 - inset;
        let tr_in_col = w - 2 - inset;
        let idx = row_top * w + tr_out_col;
        pixels[idx] = pixels[idx].under(tr_color(row_top, tr_out_col), BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = row_top * w + tr_in_col;
        let layer = (tr_color(row_top, tr_in_col) & 0x00FFFFFF) | inner_t;
        pixels[idx] = pixels[idx].under(layer, BlendMode::Normal);
        clip_mask[idx] = 255;

        // BL row-walk
        for c in 0..inset {
            clip_mask[row_bot * w + c] = 0;
        }
        let idx = row_bot * w + inset;
        pixels[idx] = pixels[idx].under(bl_color(row_bot, inset), BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = row_bot * w + inset + 1;
        let layer = (bl_color(row_bot, inset + 1) & 0x00FFFFFF) | inner_t;
        pixels[idx] = pixels[idx].under(layer, BlendMode::Normal);
        clip_mask[idx] = 255;

        // BR row-walk
        for c in (w - inset)..w {
            clip_mask[row_bot * w + c] = 0;
        }
        let idx = row_bot * w + (w - 1 - inset);
        pixels[idx] = pixels[idx].under(shadow, BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = row_bot * w + (w - 2 - inset);
        pixels[idx] = pixels[idx].under(shadow_inner, BlendMode::Normal);
        clip_mask[idx] = 255;

        // TL col-walk (near-horizontal portion of TL corner).
        for r in 0..inset {
            clip_mask[r * w + col_left] = 0;
        }
        let idx = inset * w + col_left;
        pixels[idx] = pixels[idx].under(light, BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = (inset + 1) * w + col_left;
        pixels[idx] = pixels[idx].under(light_inner, BlendMode::Normal);
        clip_mask[idx] = 255;

        // TR col-walk.
        for r in 0..inset {
            clip_mask[r * w + col_right] = 0;
        }
        let idx = inset * w + col_right;
        pixels[idx] = pixels[idx].under(tr_color(inset, col_right), BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = (inset + 1) * w + col_right;
        let layer = (tr_color(inset + 1, col_right) & 0x00FFFFFF) | inner_t;
        pixels[idx] = pixels[idx].under(layer, BlendMode::Normal);
        clip_mask[idx] = 255;

        // BL col-walk.
        for r in (h - inset)..h {
            clip_mask[r * w + col_left] = 0;
        }
        let bl_out_row = h - 1 - inset;
        let bl_in_row = h - 2 - inset;
        let idx = bl_out_row * w + col_left;
        pixels[idx] = pixels[idx].under(bl_color(bl_out_row, col_left), BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = bl_in_row * w + col_left;
        let layer = (bl_color(bl_in_row, col_left) & 0x00FFFFFF) | inner_t;
        pixels[idx] = pixels[idx].under(layer, BlendMode::Normal);
        clip_mask[idx] = 255;

        // BR col-walk.
        for r in (h - inset)..h {
            clip_mask[r * w + col_right] = 0;
        }
        let idx = (h - 1 - inset) * w + col_right;
        pixels[idx] = pixels[idx].under(shadow, BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = (h - 2 - inset) * w + col_right;
        pixels[idx] = pixels[idx].under(shadow_inner, BlendMode::Normal);
        clip_mask[idx] = 255;
    }
}

/// Strip geometry consumed by the three `draw_strip_*` functions. Returns `None` if the strip can't fit in the viewport.
fn strip_layout(
    width: u32,
    height: u32,
    button_size: usize,
) -> Option<(usize, usize, usize, usize, usize, usize)> {
    if width < 2 || height < 2 || button_size < 4 {
        return None;
    }
    let w = width as usize;
    let h = height as usize;
    let strip_w = button_size * 7 / 2;
    if strip_w >= w || button_size >= h {
        return None;
    }
    let strip_x = w - strip_w;
    let button_area_offset = button_size / 4;
    let last_row = button_size - 1;
    Some((w, strip_w, strip_x, button_area_offset, last_row, h))
}

#[inline]
fn hit_for_dx(dx: usize, button_size: usize, button_area_offset: usize) -> u8 {
    if dx < button_area_offset {
        HIT_MINIMIZE_BUTTON
    } else {
        let x_in = dx - button_area_offset;
        if x_in < button_size {
            HIT_MINIMIZE_BUTTON
        } else if x_in < button_size * 2 {
            HIT_MAXIMIZE_BUTTON
        } else {
            HIT_CLOSE_BUTTON
        }
    }
}

/// **Step 2** in the chrome rasterizer (after window perimeter). Paint the BL squircle hairline of the controls strip — row-walk (the curve's near-vertical leg) and col-walk (the near-horizontal leg). Uses [`paint_if_empty`] so writes from the window perimeter are not overwritten. Each curve pixel gets at most ONE writer (this function or the perimeter, whichever ran first).
pub fn draw_strip_curves(
    pixels: &mut [u32],
    hit_test_map: &mut [u8],
    width: u32,
    height: u32,
    button_size: usize,
    start: usize,
    crossings: &[(u16, u8, u8)],
) {
    let Some((w, strip_w, strip_x, button_area_offset, last_row, _h)) =
        strip_layout(width, height, button_size)
    else {
        return;
    };
    if start >= button_size {
        return;
    }
    let edge = theme::WINDOW_LIGHT_EDGE;
    // Hairline geometry: a 1-pixel line extending inward from the curve into the strip body.
    //   Outer pixel coverage = 1 - fract → opacity = l (sqrt-gamma'd) → chrome t = 255 - l.
    //   Inner pixel coverage = fract → opacity = h_cov → chrome t = 255 - h_cov.
    // Under composition handles the actual blending with whatever bg is below the chrome layer.

    // Row-walk.
    for (i, &(inset_raw, h_cov, l)) in crossings.iter().enumerate() {
        let dy = start + i;
        if dy >= button_size {
            break;
        }
        let inset = inset_raw as usize;
        if inset >= strip_w {
            continue;
        }
        let py = last_row - dy;
        let outer_v = (edge & 0x00FFFFFF) | (((255u32).saturating_sub(l as u32)) << 24);
        let outer_idx = py * w + strip_x + inset;
        pixels[outer_idx] = pixels[outer_idx].under(outer_v, BlendMode::Normal);
        if inset + 1 < strip_w {
            let inner_v = (edge & 0x00FFFFFF) | (((255u32).saturating_sub(h_cov as u32)) << 24);
            let inner_idx = py * w + strip_x + inset + 1;
            pixels[inner_idx] = pixels[inner_idx].under(inner_v, BlendMode::Normal);
        }
    }

    // Col-walk (mirror of row-walk by x↔y symmetry).
    for (i, &(inset_raw, h_cov, l)) in crossings.iter().enumerate() {
        let dx = start + i;
        if dx >= strip_w {
            break;
        }
        let inset = inset_raw as usize;
        if inset >= button_size {
            continue;
        }
        let outer_py = last_row - inset;
        let outer_v = (edge & 0x00FFFFFF) | (((255u32).saturating_sub(l as u32)) << 24);
        let outer_idx = outer_py * w + strip_x + dx;
        pixels[outer_idx] = pixels[outer_idx].under(outer_v, BlendMode::Normal);
        if inset + 1 < button_size {
            let inner_v = (edge & 0x00FFFFFF) | (((255u32).saturating_sub(h_cov as u32)) << 24);
            let inner_py = last_row - (inset + 1);
            let inner_idx = inner_py * w + strip_x + dx;
            pixels[inner_idx] = pixels[inner_idx].under(inner_v, BlendMode::Normal);
        }
    }
    let _ = (hit_test_map, button_area_offset);
}

/// **Step 3** in the chrome rasterizer. Vertical divider hairlines between min/max and max/close buttons, plus the linear bottom hairline (only relevant when the BL curve doesn't fit). Uses [`paint_if_empty`].
pub fn draw_strip_hairlines(
    pixels: &mut [u32],
    width: u32,
    height: u32,
    button_size: usize,
    start: usize,
    crossings: &[(u16, u8, u8)],
) {
    let Some((w, _strip_w, strip_x, button_area_offset, last_row, _h)) =
        strip_layout(width, height, button_size)
    else {
        return;
    };
    let edge = theme::WINDOW_LIGHT_EDGE;
    let div1 = button_area_offset + button_size;
    let div2 = button_area_offset + 2 * button_size;
    let cap = start + crossings.len();
    let curve_active = start < button_size;

    // Vertical dividers — full height of the strip.
    for py in 0..button_size {
        let row_base = py * w;
        let idx = row_base + strip_x + div1;
        pixels[idx] = pixels[idx].under(edge, BlendMode::Normal);
        let idx = row_base + strip_x + div2;
        pixels[idx] = pixels[idx].under(edge, BlendMode::Normal);
    }

    // Bottom hairline. When the BL curve is active and cap ≫ strip_w (the typical case), the col-walk's `inset=0` outer pixels already form the visible bottom hairline; this loop only paints the fallback rectangular case (no curve) or the linear region beyond cap.
    let bottom_row = last_row * w;
    for px in strip_x..w {
        let dx = px - strip_x;
        if !curve_active || dx >= cap {
            pixels[bottom_row + px] = pixels[bottom_row + px].under(edge, BlendMode::Normal);
        }
    }
}

/// **Step 6** (last). Strip background fill. For every pixel in the strip's geometric interior (= NOT in the BL cutout, NOT the curve's outer pixel), compose `WINDOW_CONTROLS_BG` under via `paint_if_empty`. Empty pixels get filled with strip bg directly; partial-opacity pixels (curve inner, glyph AA) compose strip bg underneath, darkening them toward strip bg — making the chrome layer fully opaque in the strip area.
///
/// The curve's OUTER pixel is explicitly skipped because geometrically it sits on the strip's boundary; its "behind" is the bg-layer (panes), not strip bg. Leaving it partial preserves the correct visible-over-panes composite at the Stack step.
pub fn draw_strip_bg(
    pixels: &mut [u32],
    hit_test_map: &mut [u8],
    width: u32,
    height: u32,
    button_size: usize,
    start: usize,
    crossings: &[(u16, u8, u8)],
) {
    let Some((w, _strip_w, strip_x, button_area_offset, last_row, _h)) =
        strip_layout(width, height, button_size)
    else {
        return;
    };
    let bg = theme::WINDOW_CONTROLS_BG;
    let curve_active = start < button_size;
    let cap = start + crossings.len();

    let in_strip_interior = |dx: usize, dy: usize| -> bool {
        if !curve_active {
            return true;
        }
        if dy >= cap || dx >= cap {
            return true;
        }
        // Corner-of-corner cutout — always outside.
        if dy < start && dx < start {
            return false;
        }
        // Curve row: outer at dx = inset. Inside iff dx > inset.
        if dy >= start {
            let inset = crossings[dy - start].0 as usize;
            return dx > inset;
        }
        // Curve col (dy < start, dx >= start): outer at dy = inset.
        let inset = crossings[dx - start].0 as usize;
        dy > inset
    };

    for py in 0..button_size {
        let dy = last_row - py;
        let row_base = py * w;
        for px in strip_x..w {
            let dx = px - strip_x;
            if !in_strip_interior(dx, dy) {
                continue;
            }
            let idx = row_base + px;
            pixels[idx] = pixels[idx].under(bg, BlendMode::Normal);
            // Hit map is independent of chrome layering — every in-strip-interior pixel registers as the button at this dx, regardless of whether a higher-priority paint (divider/curve/glyph) already claimed the chrome pixel.
            hit_test_map[idx] = hit_for_dx(dx, button_size, button_area_offset);
        }
    }
}

/// Rasterize the minimize glyph (a small horizontal squircle dash) centered at `(cx, cy)` with radius `r`. Top-down per-pixel: each pixel inside the squircle footprint computes its coverage and writes either the solid `stroke` color or a `stroke`-blended-with-`bg` color. The chrome layer is opaque at the button bg before this call; this function only overwrites pixels INSIDE the glyph footprint.
pub fn draw_minimize_symbol(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    r: usize,
    stroke: u32,
    bg: u32,
) {
    let _ = bg;
    let r = r + 1;
    let r_render = r / 4 + 1;
    let r2 = r_render * r_render;
    let r4 = r2 * r2;
    let r3 = r_render * r_render * r_render;

    for h in -(r_render as isize)..=(r_render as isize) {
        for ww in -(r as isize)..=(r as isize) {
            let h2 = h * h;
            let h4 = h2 * h2;
            let a = (ww.abs() - (r * 3 / 4) as isize).max(0);
            let w2 = a * a;
            let w4 = w2 * w2;
            let dist4 = (h4 + w4) as usize;
            if dist4 > r4 {
                continue;
            }
            let px = cx as isize + ww;
            let py = cy as isize + h + (r / 2) as isize;
            if px < 0 || py < 0 || (px as usize) >= width || (py as usize) >= height {
                continue;
            }
            let idx = (py as usize) * width + (px as usize);
            let gradient = ((r4 - dist4) << 8) / (r3 << 2);
            // AA via t-byte: opacity = gradient/256 (clamped to 256 = fully opaque).
            let opacity = gradient.min(256) as u32;
            if opacity == 0 {
                continue;
            }
            let chrome_t = (256 - opacity).min(255);
            let value = (stroke & 0x00FFFFFF) | (chrome_t << 24);
            pixels[idx] = pixels[idx].under(value, BlendMode::Normal);
        }
    }
}

/// Rasterize the maximize glyph (a squircle ring — outer stroke, inner fill) centered at `(cx, cy)`. Top-down per-pixel inside the outer squircle footprint.
pub fn draw_maximize_symbol(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    r: usize,
    stroke: u32,
    fill: u32,
    bg: u32,
) {
    let r = r + 1;
    let mut r4 = r * r;
    r4 *= r4;
    let r3 = r * r * r;
    let r_inner = r * 4 / 5;
    let mut r_inner4 = r_inner * r_inner;
    r_inner4 *= r_inner4;
    let r_inner3 = r_inner * r_inner * r_inner;
    let outer_thresh = r3 << 2;
    let inner_thresh = r_inner3 << 2;
    let stroke_rgb = (
        ((stroke >> 16) & 0xFF) as u32,
        ((stroke >> 8) & 0xFF) as u32,
        (stroke & 0xFF) as u32,
    );
    let fill_rgb = (
        ((fill >> 16) & 0xFF) as u32,
        ((fill >> 8) & 0xFF) as u32,
        (fill & 0xFF) as u32,
    );
    let _ = bg;

    for h in -(r as isize)..=(r as isize) {
        for ww in -(r as isize)..=(r as isize) {
            let h2 = h * h;
            let h4 = h2 * h2;
            let w2 = ww * ww;
            let w4 = w2 * w2;
            let dist4 = (h4 + w4) as usize;
            if dist4 > r4 {
                continue;
            }
            let px = cx as isize + ww;
            let py = cy as isize + h;
            if px < 0 || py < 0 || (px as usize) >= width || (py as usize) >= height {
                continue;
            }
            let idx = (py as usize) * width + (px as usize);

            let value = if dist4 <= r_inner4 {
                // INSIDE inner squircle = fill region. Inner edge (stroke ↔ fill) is between two known glyph colors, so pre-blending is correct here (both colors are deterministic, no bg layer involvement).
                let dist_from_inner = r_inner4 - dist4;
                if dist_from_inner <= inner_thresh {
                    let gradient = (dist_from_inner << 8) / inner_thresh;
                    let alpha = gradient as u32;
                    let inv = 256 - alpha;
                    let r_blend = (stroke_rgb.0 * inv + fill_rgb.0 * alpha) >> 8;
                    let g_blend = (stroke_rgb.1 * inv + fill_rgb.1 * alpha) >> 8;
                    let b_blend = (stroke_rgb.2 * inv + fill_rgb.2 * alpha) >> 8;
                    (r_blend << 16) | (g_blend << 8) | b_blend
                } else {
                    fill
                }
            } else {
                // RING region (between inner and outer). Outer edge AA against the bg layer goes via the chrome t-byte — no pre-blending.
                let dist_from_outer = r4 - dist4;
                if dist_from_outer <= outer_thresh {
                    let gradient = (dist_from_outer << 8) / outer_thresh;
                    let opacity = gradient.min(256) as u32;
                    if opacity == 0 {
                        continue;
                    }
                    let chrome_t = (256 - opacity).min(255);
                    (stroke & 0x00FFFFFF) | (chrome_t << 24)
                } else {
                    stroke
                }
            };
            pixels[idx] = pixels[idx].under(value, BlendMode::Normal);
        }
    }
}

/// Distance from `(px, py)` to the capsule (rounded-line) `[(x1,y1)..(x2,y2)]` with radius `rad`. Negative inside, positive outside. Used by [`draw_close_symbol`] to rasterize the two diagonals.
fn distance_to_capsule(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32, rad: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    let t = if len_sq > 0.0 {
        let raw = ((px - x1) * dx + (py - y1) * dy) / len_sq;
        if raw < 0.0 {
            0.0
        } else if raw > 1.0 {
            1.0
        } else {
            raw
        }
    } else {
        0.0
    };
    let cx = x1 + t * dx;
    let cy = y1 + t * dy;
    let ddx = px - cx;
    let ddy = py - cy;
    math::sqrt(ddx * ddx + ddy * ddy) - rad
}

/// Rasterize the close glyph (an X made of two diagonal capsules) centered at `(cx, cy)` with arm half-length `r`. Top-down per-pixel inside the X's bounding box.
pub fn draw_close_symbol(
    pixels: &mut [u32],
    width: usize,
    height: usize,
    cx: usize,
    cy: usize,
    r: usize,
    stroke: u32,
    bg: u32,
) {
    let r = r + 1;
    let thickness = ((r / 3).max(1)) as f32;
    let radius = thickness * 0.5;
    let end = (r * 2) as f32 / 3.0;
    let cxf = cx as f32;
    let cyf = cy as f32;
    let x1s = cxf - end;
    let y1s = cyf - end;
    let x1e = cxf + end;
    let y1e = cyf + end;
    let x2s = cxf + end;
    let y2s = cyf - end;
    let x2e = cxf - end;
    let y2e = cyf + end;
    let stroke_rgb = (
        ((stroke >> 16) & 0xFF) as u32,
        ((stroke >> 8) & 0xFF) as u32,
        (stroke & 0xFF) as u32,
    );
    let _ = bg;
    let _ = stroke_rgb;

    let min_x = cx.saturating_sub(r);
    let max_x = (cx + r).min(width);
    let min_y = cy.saturating_sub(r);
    let max_y = (cy + r).min(height);

    for py in min_y..max_y {
        for px in min_x..max_x {
            let pxf = px as f32 + 0.5;
            let pyf = py as f32 + 0.5;
            // Choose the diagonal whose orientation matches this quadrant.
            let use_d1 = (px >= cx && py >= cy) || (px < cx && py < cy);
            let dist = if use_d1 {
                distance_to_capsule(pxf, pyf, x1s, y1s, x1e, y1e, radius)
            } else {
                distance_to_capsule(pxf, pyf, x2s, y2s, x2e, y2e, radius)
            };
            let alpha_f = if dist < -0.5 {
                1.0
            } else if dist < 0.5 {
                0.5 - dist
            } else {
                0.0
            };
            if alpha_f <= 0.0 {
                continue;
            }
            let idx = py * width + px;
            // AA via t-byte: opacity = alpha_f (clamped to 1.0 = fully opaque).
            let opacity = (alpha_f * 256.0).min(256.0) as u32;
            if opacity == 0 {
                continue;
            }
            let chrome_t = (256 - opacity).min(255);
            let value = (stroke & 0x00FFFFFF) | (chrome_t << 24);
            pixels[idx] = pixels[idx].under(value, BlendMode::Normal);
        }
    }
}
