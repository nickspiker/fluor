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
use crate::theme;

/// Hit-test IDs that the per-pixel hit_test_map can carry. `HIT_NONE` = clicks pass through. Button IDs are placeholders for the future controls scaffold step.
pub const HIT_NONE: u8 = 0;
pub const HIT_MINIMIZE_BUTTON: u8 = 1;
pub const HIT_MAXIMIZE_BUTTON: u8 = 2;
pub const HIT_CLOSE_BUTTON: u8 = 3;

/// Minimum pixel height for a control button. The button-sizing formula in higher scaffold steps is `MIN_BUTTON_HEIGHT_PX + ceil(span/32 * ru)` — the floor guarantees controls remain visible (and that symbol-rasterizer integer math never floors to zero) at any window size the WM permits. Kept as a public const so the host can compute layout pre-rasterization.
pub const MIN_BUTTON_HEIGHT_PX: u32 = 24;

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

    // Straight edges — opaque RGB only. Clip mask along these edges stays at the host's 255 default (fully visible).
    for x in cap..(w - cap) {
        pixels[x] = light; // top row
        pixels[(h - 1) * w + x] = shadow; // bottom row
    }
    for y in cap..(h - cap) {
        pixels[y * w] = light; // left col
        pixels[y * w + (w - 1)] = shadow; // right col
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
        pixels[row_top * w + inset] = light;
        clip_mask[row_top * w + inset] = h_cov;
        pixels[row_top * w + inset + 1] = light_inner;
        clip_mask[row_top * w + inset + 1] = 255;

        // TR row-walk
        for c in (w - inset)..w {
            clip_mask[row_top * w + c] = 0;
        }
        let tr_out_col = w - 1 - inset;
        let tr_in_col = w - 2 - inset;
        pixels[row_top * w + tr_out_col] = tr_color(row_top, tr_out_col);
        clip_mask[row_top * w + tr_out_col] = h_cov;
        pixels[row_top * w + tr_in_col] =
            (tr_color(row_top, tr_in_col) & 0x00FFFFFF) | inner_t;
        clip_mask[row_top * w + tr_in_col] = 255;

        // BL row-walk
        for c in 0..inset {
            clip_mask[row_bot * w + c] = 0;
        }
        pixels[row_bot * w + inset] = bl_color(row_bot, inset);
        clip_mask[row_bot * w + inset] = h_cov;
        pixels[row_bot * w + inset + 1] =
            (bl_color(row_bot, inset + 1) & 0x00FFFFFF) | inner_t;
        clip_mask[row_bot * w + inset + 1] = 255;

        // BR row-walk
        for c in (w - inset)..w {
            clip_mask[row_bot * w + c] = 0;
        }
        pixels[row_bot * w + (w - 1 - inset)] = shadow;
        clip_mask[row_bot * w + (w - 1 - inset)] = h_cov;
        pixels[row_bot * w + (w - 2 - inset)] = shadow_inner;
        clip_mask[row_bot * w + (w - 2 - inset)] = 255;

        // TL col-walk (near-horizontal portion of TL corner).
        for r in 0..inset {
            clip_mask[r * w + col_left] = 0;
        }
        pixels[inset * w + col_left] = light;
        clip_mask[inset * w + col_left] = h_cov;
        pixels[(inset + 1) * w + col_left] = light_inner;
        clip_mask[(inset + 1) * w + col_left] = 255;

        // TR col-walk.
        for r in 0..inset {
            clip_mask[r * w + col_right] = 0;
        }
        pixels[inset * w + col_right] = tr_color(inset, col_right);
        clip_mask[inset * w + col_right] = h_cov;
        pixels[(inset + 1) * w + col_right] =
            (tr_color(inset + 1, col_right) & 0x00FFFFFF) | inner_t;
        clip_mask[(inset + 1) * w + col_right] = 255;

        // BL col-walk.
        for r in (h - inset)..h {
            clip_mask[r * w + col_left] = 0;
        }
        let bl_out_row = h - 1 - inset;
        let bl_in_row = h - 2 - inset;
        pixels[bl_out_row * w + col_left] = bl_color(bl_out_row, col_left);
        clip_mask[bl_out_row * w + col_left] = h_cov;
        pixels[bl_in_row * w + col_left] =
            (bl_color(bl_in_row, col_left) & 0x00FFFFFF) | inner_t;
        clip_mask[bl_in_row * w + col_left] = 255;

        // BR col-walk.
        for r in (h - inset)..h {
            clip_mask[r * w + col_right] = 0;
        }
        pixels[(h - 1 - inset) * w + col_right] = shadow;
        clip_mask[(h - 1 - inset) * w + col_right] = h_cov;
        pixels[(h - 2 - inset) * w + col_right] = shadow_inner;
        clip_mask[(h - 2 - inset) * w + col_right] = 255;
    }
}

// Removed in the "ONLY THE HAIRLINE" simplification (each was painter's-algorithm internally):
//   - draw_window_controls         — button strip; opaque bg painted first, then read back by blend_rgb_only at AA crossings (classic painter's).
//   - draw_button_hairlines        — walked vertically reading just-painted pixels to detect where to stop.
//   - pixels_for_button            — consumer of the hit_test_map after the controls scaffold.
//   - draw_button_hover_by_pixels  — additive overlay; depends on the controls scaffold.
//   - rasterize_window_silhouette  — was for the deleted Op::Or knockout.
//
// Each will be re-added as a separate scaffold step, redesigned top-down (compute layout once, write each pixel from a single deterministic source) when the time comes.
