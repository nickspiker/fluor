//! Window chrome — minimal top-down rasterization. Each pixel in the chrome layer is written by exactly one site. No painter's algorithm anywhere.
//!
//! Currently scoped to **window perimeter hairline** with squircle corner AA. The chrome layer starts at the canonical empty value (`0x00000000` = α=0 transparent, darkness=0); this function paints only the hairline pixels. Everywhere else in the chrome layer stays transparent so panes / bg can pass through the chrome group's Stack composition. Buttons, glyphs, title text, hover overlay — all deferred to subsequent scaffold steps; reintroduce them only when each can be added without overwriting earlier writes within this same layer.
//!
//! Hit-test IDs and the `ResizeEdge` enum live here so the desktop host's mouse routing can reference them without depending on the (future, larger) controls implementation.
//!
//! The squircle crossings table is consumed but not computed here; the caller (chrome_widget) computes it once per resize and passes it in.
//!
//! All RGB values stored in the chrome layer are straight-α (the canonical buffer convention). The OS conversion layer at the present boundary handles platform-specific premultiplication.

use crate::coord::Coord;
use crate::host::icon::Icon;
use crate::math;
use crate::paint::Clip;
use crate::pixel::{Blend, BlendMode};
use crate::text::TextRenderer;
use crate::theme;

pub use crate::paint::{HIT_NONE, HitId};

/// Orb visual state. The app sets this to give the orb a meaning beyond window-focus (network indicator, recording badge, presence light). Layered defaults: `FollowFocus` means "ring matches the perimeter, image dims when the window is unfocused" with zero app code; `Custom` lets the app dictate ring colour + brightness regardless of window state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrbTint {
    /// Default — ring colour equals the active perimeter colour and the orb image desaturates to 50 % grey when the window is unfocused. Window-state-as-orb-state, no app intervention.
    FollowFocus,
    /// App-driven override. `ring` paints the AA ring (already darkness-packed, e.g. a `theme::*` constant or `dark(fmt(0x00_FF_FF_FF))`). `brighten = true` applies photon's 3/2 lift to the icon image (online/active state), `false` leaves it as decoded.
    Custom { ring: u32, brighten: bool },
}

impl Default for OrbTint {
    fn default() -> Self {
        OrbTint::FollowFocus
    }
}

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
/// Pre-conditions: `pixels` already at the canonical empty value `0x00000000` (α=0 transparent, darkness=0 — calloc-free), `clip_mask` already at the host's default of `255` (fully visible window-interior assumption).
///
/// Topology: straight edges paint opaque RGB in non-corner ranges (`cap..(end-cap)`) and leave the clip mask alone (= 255, fully visible). Each crossing entry handles **one row** of the curve region for the four corners: zero out the cutout cols, write opaque hairline RGB at the curve's outer + inner pixel positions, and write `h_cov` / `l` into the clip mask at those same positions. Above-the-curve rows (`0..start`) and below-the-curve rows (`h-start..h`) are *entirely* cutout — the curve never enters them — so the full cap-width at those rows is zeroed in the clip mask.
///
/// Two-tone bevel (light from upper-left): top + left straight edges are light, bottom + right are shadow. TL and BR corners are uniform (both adjacent edges agree); TR and BL transition along the curve. The per-pixel colour test (`tr_colour`, `bl_colour`) is the same one we settled on previously — closer-to-light-edge wins.
///
/// `hit_test_map` is preserved as a parameter for forward compatibility with the controls scaffold step but is not modified here.
pub fn draw_window_edges_and_mask(
    pixels: &mut [u32],
    hit_test_map: &mut [HitId],
    clip_mask: &mut [u8],
    width: u32,
    height: u32,
    start: usize,
    crossings: &[(u16, u8, u8)],
    light: u32,
    shadow: u32,
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

    let tr_colour =
        |row: usize, col: usize| -> u32 { if row < (w - 1 - col) { light } else { shadow } };
    let bl_colour =
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

        // Two-sided AA convention. OUTER pixel = on the curve, partially outside the window: clip_mask = h_cov trims against the OS bg; chrome stays opaque (α=0xFF) because the entire inside-window portion IS hairline. INNER pixel = one step inside the curve: clip_mask = 255 (fully inside the window shape); chrome's α-byte carries the hairline-vs-bg AA, set to `inner_α = 255 − h_cov` so the Under blend mixes (255−h_cov)/256 of hairline colour over h_cov/256 of window bg. The `l` slot in each crossing entry is the linear-coverage counterpart of h_cov, retained for the outer-pixel chrome-α AA when we move from a 2-pixel hairline to a 1-pixel-with-halo hairline.
        let _ = l;
        let inner_alpha = ((255 - h_cov) as u32) << 24;
        let light_inner = (light & 0x00FFFFFF) | inner_alpha;
        let shadow_inner = (shadow & 0x00FFFFFF) | inner_alpha;

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
        pixels[idx] = pixels[idx].under(tr_colour(row_top, tr_out_col), BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = row_top * w + tr_in_col;
        let layer = (tr_colour(row_top, tr_in_col) & 0x00FFFFFF) | inner_alpha;
        pixels[idx] = pixels[idx].under(layer, BlendMode::Normal);
        clip_mask[idx] = 255;

        // BL row-walk
        for c in 0..inset {
            clip_mask[row_bot * w + c] = 0;
        }
        let idx = row_bot * w + inset;
        pixels[idx] = pixels[idx].under(bl_colour(row_bot, inset), BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = row_bot * w + inset + 1;
        let layer = (bl_colour(row_bot, inset + 1) & 0x00FFFFFF) | inner_alpha;
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
        pixels[idx] = pixels[idx].under(tr_colour(inset, col_right), BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = (inset + 1) * w + col_right;
        let layer = (tr_colour(inset + 1, col_right) & 0x00FFFFFF) | inner_alpha;
        pixels[idx] = pixels[idx].under(layer, BlendMode::Normal);
        clip_mask[idx] = 255;

        // BL col-walk.
        for r in (h - inset)..h {
            clip_mask[r * w + col_left] = 0;
        }
        let bl_out_row = h - 1 - inset;
        let bl_in_row = h - 2 - inset;
        let idx = bl_out_row * w + col_left;
        pixels[idx] = pixels[idx].under(bl_colour(bl_out_row, col_left), BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = bl_in_row * w + col_left;
        let layer = (bl_colour(bl_in_row, col_left) & 0x00FFFFFF) | inner_alpha;
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

/// Rasterize the window title text into the chrome layer, left-aligned in the area between the perimeter hairline (left edge) and the controls strip (right edge). Vertically centered in the strip-tall band. `left_extra` shifts the start position right to make room for the app-icon orb (pass `0` when no orb). `colour` is darkness-packed (typically `theme::TEXT_COLOUR` when focused, `theme::LABEL_COLOUR` when unfocused). Bails on empty title or impractically small `button_size` (below readability — the text wouldn't be legible anyway). Clip rect prevents the title from painting over the controls strip even at long titles or narrow windows. Font is "Open Sans" regular at `button_size * 0.55` — proportional to the rest of the chrome under the current zoom (since button_size is derived from `effective_span`).
pub fn draw_title_text(
    canvas: &mut crate::canvas::Canvas,
    title: &str,
    text_renderer: &mut TextRenderer,
    button_size: usize,
    strip_x: usize,
    left_extra: usize,
    colour: u32,
) {
    if title.is_empty() || button_size < 8 {
        return;
    }
    let left_margin = button_size / 2 + left_extra;
    let right_margin = button_size / 4;
    // strip_x = buf_w − strip_w can collapse to 0 on tiny viewports (buf_w ≤ strip_w). Without saturating_sub, `strip_x − right_margin` would underflow usize and wrap to ~usize::MAX, producing a clip rect that spans the whole row. saturating_sub returns 0, the `left_margin >= clip_x_end` check below catches it, and the function returns cleanly without drawing.
    let clip_x_end = strip_x.saturating_sub(right_margin);
    if left_margin >= clip_x_end {
        return;
    }
    let font_size = button_size as Coord * 0.55;
    let y_center = button_size as Coord * 0.5;
    let clip = Clip::new(left_margin, 0, clip_x_end, button_size);
    text_renderer.draw_text_left_u32(
        canvas,
        title,
        left_margin as f32,
        y_center,
        font_size,
        400,
        colour,
        "Open Sans",
        Some(clip),
        None,
        None,
    );
}

/// Rasterize the bottom status band: a thin strip at `height − band_h .. height` filled with `bg`, topped by a 1-px `hairline_colour` divider where the band meets the pane content. Optional left-aligned `text` paints in `text_colour` (Open Sans, font size = `band_h × 0.55`). The band is short — `band_h` is typically `button_size / 2` — so it reads as a secondary surface, distinct from the top controls strip.
///
/// The window perimeter's clip_mask carving handles the BL/BR squircle corners for free: chrome pixels in the corner cutout are written but masked off at the OS boundary, so the band's rectangular fill becomes a rounded bottom edge without per-pixel geometry here. Bails on `band_h == 0` or impractical `band_h` (>= height) so the rasterizer can be called unconditionally from a `chrome_widget` that always has a status field, with `band_h = 0` meaning "no status bar".
pub fn draw_status_bar(
    canvas: &mut crate::canvas::Canvas,
    band_h: usize,
    bg: u32,
    hairline_colour: u32,
    text: &str,
    text_renderer: &mut TextRenderer,
    text_colour: u32,
) {
    let width = canvas.width;
    let height = canvas.height;
    if band_h == 0 || band_h + 1 >= height || width == 0 {
        return;
    }
    let y_top = height - band_h;
    // Damage = full-width band [y_top, height).
    canvas.damage.add_bounds(0, y_top, width, height);
    let pixels: &mut [u32] = canvas.pixels;

    // Top hairline (1 px) — claims y_top across the full width. The squircle perimeter's clip_mask handles rounding at BL/BR.
    let hairline = 0xFF000000 | (hairline_colour & 0x00FFFFFF);
    let row_top = y_top * width;
    for x in 0..width {
        let idx = row_top + x;
        pixels[idx] = pixels[idx].under(hairline, BlendMode::Normal);
    }

    // BG fill — opaque pixels in (y_top, height). Front-to-back under-blend means earlier writers (perimeter hairline + corner curve pixels) keep their values; bg fills only the empty interior.
    let bg_opaque = 0xFF000000 | (bg & 0x00FFFFFF);
    for y in (y_top + 1)..height {
        let row_base = y * width;
        for x in 0..width {
            let idx = row_base + x;
            pixels[idx] = pixels[idx].under(bg_opaque, BlendMode::Normal);
        }
    }

    // Status text (optional). Horizontally centered in the band; vertically centered in the band's height. `band_h / 2` of side padding on both edges defines the clip so very long status strings don't bleed past the curves. Font size proportional to band height.
    if text.is_empty() {
        return;
    }
    let side_margin = band_h / 2;
    let clip_x_end = width.saturating_sub(side_margin);
    if side_margin >= clip_x_end {
        return;
    }
    let font_size = band_h as Coord * 0.55;
    let x_center = width as Coord * 0.5;
    let y_center = y_top as Coord + band_h as Coord * 0.5;
    let clip = Clip::new(side_margin, y_top, clip_x_end, height);
    text_renderer.draw_text_center_u32(
        canvas,
        text,
        x_center,
        y_center,
        font_size,
        400,
        text_colour,
        "Open Sans",
        Some(clip),
        None,
        None,
    );
}

/// Rasterize the top-left app-icon orb: a circular sample of `icon` clipped to `radius`, wrapped in an optional 1-px AA ring stroked in `ring_colour`. Topology mirrors [`draw_window_edges_and_mask`]'s two-sided AA: ring is 1px solid + 1px inner-AA + 1px outer-AA. `cx`/`cy` give the orb centre in pixel coords; `radius` is the icon sampling radius (ring extends outward from it). Without an `icon`, the interior fills with `ring_colour` (treated as a solid dark disk). Without a `ring_colour`, the orb is just the icon clipped to a circle (1-pixel outer-AA against the chrome).
///
/// Pixel sampling is nearest-neighbour from `icon`'s `width × height` source — the source is square in practice (`vsfimg` doesn't reshape) but the math doesn't assume that. Per-pixel cost is one map index + one `under` composite; total work is `O(diameter²)`, well under a millisecond at typical chrome sizes (~30–100 px orbs).
///
/// Hit-test: every pixel inside `r_outer²` (excluding the AA fringe) is tagged with `hit_id` so the host can route clicks to the chrome's app-icon widget. Decorative-only consumers can pass `None` for `hit_test_map` to skip the tag (in which case `hit_id` is ignored).
pub fn draw_app_icon(
    pixels: &mut [u32],
    hit_test_map: Option<&mut [HitId]>,
    hit_id: HitId,
    width: usize,
    height: usize,
    cx: isize,
    cy: isize,
    radius: isize,
    icon: Option<&Icon>,
    ring_colour: Option<u32>,
    darken: u8,
    brighten: bool,
) {
    let r = radius;
    if r < 2 {
        return;
    }
    // Stroke matches photon: `r / 16` with no floor. At small orbs (r < 16) this is 0 — the ring degrades to just the 1-px outer-AA edge instead of a forced 2-px band, so the orb stays proportional at chrome-button sizes.
    let stroke_width = r / 16;

    let r_inner = r - 1;
    let r_inner2 = r_inner * r_inner;
    let r_inner_inner = r - 2;
    let r_inner_inner2 = r_inner_inner * r_inner_inner;
    let r_outer = r + stroke_width;
    let r_outer2 = r_outer * r_outer;
    let r_outer_outer = r_outer + 1;
    let r_outer_outer2 = r_outer_outer * r_outer_outer;
    // diff_inner = r_inner² − r_inner_inner² = (r−1)² − (r−2)² = 2r − 3, which is ≥ 1 given the `r < 2` early return above. diff_outer = (r+sw+1)² − (r+sw)² = 2(r+sw) + 1 ≥ 5. Both are safe divisors; no max() guard needed.
    let diff_inner = r_inner2 - r_inner_inner2;
    let diff_outer = r_outer_outer2 - r_outer2;

    // BBox intersection with screen. WHY: caller can pass any (cx, cy) — orb may be partially or fully off-screen (e.g. scroll offsets in a future viewport). PROOF: clip the circle bounding box to `[0, width) × [0, height)`, returning early if the intersection is empty. PREVENTS: a negative isize converting to usize would wrap to a huge value and the iteration would index well past the buffer end (out-of-bounds → panic, or in release with overflow-checks=false → undefined behaviour).
    let max_r = if ring_colour.is_some() {
        r_outer_outer
    } else {
        r_inner
    };
    let y_min_i = (cy - max_r).max(0);
    let y_max_i = (cy + max_r + 1).min(height as isize);
    let x_min_i = (cx - max_r).max(0);
    let x_max_i = (cx + max_r + 1).min(width as isize);
    if y_max_i <= y_min_i || x_max_i <= x_min_i {
        return;
    }
    let (y_min, y_max, x_min, x_max) = (
        y_min_i as usize,
        y_max_i as usize,
        x_min_i as usize,
        x_max_i as usize,
    );

    let mut htm = hit_test_map;

    for y in y_min..y_max {
        let dy = y as isize - cy;
        let dy2 = dy * dy;
        for x in x_min..x_max {
            let dx = x as isize - cx;
            let dist2 = dx * dx + dy2;
            let idx = y * width + x;

            if let Some(map) = htm.as_mut() {
                if dist2 <= r_outer2 {
                    map[idx] = hit_id;
                }
            }

            if let Some(ring) = ring_colour {
                let ring_rgb = ring & 0x00FFFFFF;
                if dist2 <= r_inner_inner2 {
                    let top = sample_icon(icon, dx, dy, r, ring, darken, brighten);
                    pixels[idx] = pixels[idx].under(top, BlendMode::Normal);
                } else if dist2 < r_inner2 {
                    let icon_pixel = sample_icon(icon, dx, dy, r, ring, darken, brighten);
                    // dist2 ∈ (r_inner_inner², r_inner²) (strict on both sides). numerator < diff_inner, (numerator << 8) < diff_inner << 8, division < 256 — fits a u8 cleanly with no clamp.
                    let t = ((dist2 - r_inner_inner2) << 8) / diff_inner;
                    let mixed = mix_rgb(icon_pixel, 0xFF000000 | ring_rgb, t as u32);
                    pixels[idx] = pixels[idx].under(mixed, BlendMode::Normal);
                } else if dist2 <= r_outer2 {
                    pixels[idx] = pixels[idx].under(0xFF000000 | ring_rgb, BlendMode::Normal);
                } else if dist2 <= r_outer_outer2 {
                    // dist2 ∈ (r_outer², r_outer_outer²]. numerator ∈ [0, diff_outer), (numerator << 8) < diff_outer << 8, division < 256.
                    let edge_a = ((r_outer_outer2 - dist2) << 8) / diff_outer;
                    let top = ((edge_a as u32) << 24) | ring_rgb;
                    pixels[idx] = pixels[idx].under(top, BlendMode::Normal);
                }
            } else {
                if dist2 > r_inner2 {
                    continue;
                }
                if dist2 <= r_inner_inner2 {
                    let top = sample_icon(icon, dx, dy, r, 0, darken, brighten);
                    pixels[idx] = pixels[idx].under(top, BlendMode::Normal);
                } else {
                    let icon_pixel = sample_icon(icon, dx, dy, r, 0, darken, brighten);
                    // dist2 ∈ (r_inner_inner², r_inner²]. r_inner² − dist2 ∈ [0, diff_inner), so (numerator << 8)/diff_inner < 256 — fits u8 with no clamp.
                    let edge_a = ((r_inner2 - dist2) << 8) / diff_inner;
                    let top = ((edge_a as u32) << 24) | (icon_pixel & 0x00FFFFFF);
                    pixels[idx] = pixels[idx].under(top, BlendMode::Normal);
                }
            }
        }
    }
}

/// Nearest-neighbour fetch from `icon` for offset `(dx, dy)` from the orb centre, scaled to fit `radius`. Returns an opaque α + darkness pixel after applying `brighten` (photon's 3/2 visible-RGB lift) and `darken` (linear blend toward mid-grey: 0 = icon as-is, 255 = fully grey). Falls back to a solid `fallback_ring` (or dark grey) when no icon is present.
///
/// Precondition: caller only invokes this when `dx² + dy² ≤ r_inner² = (r−1)²`, so `|dx|, |dy| ≤ r−1` and `u = (dx+r+0.5)/(2r) ∈ (0, 1)` strictly — `sx = (u * img.width) as usize` is therefore `< img.width`. Violating that precondition panics on the index (fail loud).
fn sample_icon(
    icon: Option<&Icon>,
    dx: isize,
    dy: isize,
    radius: isize,
    fallback_ring: u32,
    darken: u8,
    brighten: bool,
) -> u32 {
    let raw = if let Some(img) = icon {
        let diameter = (radius * 2) as f32;
        let u = ((dx + radius) as f32 + 0.5) / diameter;
        let v = ((dy + radius) as f32 + 0.5) / diameter;
        let sx = (u * img.width as f32) as usize;
        let sy = (v * img.height as f32) as usize;
        img.pixels[sy * img.width as usize + sx]
    } else if fallback_ring != 0 {
        0xFF000000 | (fallback_ring & 0x00FFFFFF)
    } else {
        0xFF7F7F7F
    };
    modulate_icon_pixel(raw, darken, brighten)
}

/// Apply photon-style brighten (visible_RGB × 3/2 saturating) and a linear-blend-toward-mid-grey darken in one pass. α byte is preserved (always opaque for icon pixels).
///
/// In α + darkness terms: brighten visible_R = `min(255, vR × 3/2)` becomes `dR_new = dR.saturating_sub((255 − dR) / 2)`. saturating_sub kept because brightening already-dark pixels (`dR < 85`) would wrap u32 below zero without it — clamping at 0 is the correct "can't brighten past full visible" outcome.
fn modulate_icon_pixel(pixel: u32, darken: u8, brighten: bool) -> u32 {
    let mut dr = (pixel >> 16) & 0xFF;
    let mut dg = (pixel >> 8) & 0xFF;
    let mut db = pixel & 0xFF;

    if brighten {
        dr = dr.saturating_sub((255 - dr) / 2);
        dg = dg.saturating_sub((255 - dg) / 2);
        db = db.saturating_sub((255 - db) / 2);
    }

    if darken > 0 {
        let f = darken as u32;
        let inv = 255 - f;
        // Mid-grey in darkness space (= mid-grey in visible space, since 0x80 ≈ 255 − 0x7F). Linear lerp on each darkness channel toward this neutral.
        let grey = 0x80u32;
        dr = (dr * inv + grey * f) / 255;
        dg = (dg * inv + grey * f) / 255;
        db = (db * inv + grey * f) / 255;
    }

    (pixel & 0xFF000000) | (dr << 16) | (dg << 8) | db
}

/// Per-channel linear interpolation in darkness space: `t = 0` returns `a`, `t = 255` returns `b`. Keeps the α byte from `a`. Used for blending the icon with the ring across the inner-AA edge.
fn mix_rgb(a: u32, b: u32, t: u32) -> u32 {
    let inv = 256 - t;
    let alpha = a & 0xFF000000;
    let ar = (a >> 16) & 0xFF;
    let ag = (a >> 8) & 0xFF;
    let ab = a & 0xFF;
    let br = (b >> 16) & 0xFF;
    let bg = (b >> 8) & 0xFF;
    let bb = b & 0xFF;
    let r = (ar * inv + br * t) >> 8;
    let g = (ag * inv + bg * t) >> 8;
    let bch = (ab * inv + bb * t) >> 8;
    alpha | (r << 16) | (g << 8) | bch
}

/// Strip geometry consumed by the three `draw_strip_*` functions. Returns `None` if the strip can't fit in the viewport.
fn strip_layout(
    width: u32,
    height: u32,
    button_size: usize,
) -> Option<(usize, usize, usize, usize, usize, usize)> {
    if width < 2 || height < 2 {
        return None;
    }
    let w = width as usize;
    let h = height as usize;
    let strip_w = button_size * 7 / 2;
    // Strip can't render larger than the window — geometric, not pixel-arbitrary.
    if strip_w >= w || button_size >= h {
        return None;
    }
    let strip_x = w - strip_w;
    let button_area_offset = button_size / 4;
    // saturating_sub keeps button_size=0 from underflowing; all the `for ... in 0..button_size` loops fall through naturally with empty range.
    let last_row = button_size.saturating_sub(1);
    Some((w, strip_w, strip_x, button_area_offset, last_row, h))
}

/// Per-row directional fill. For each row in `[row_start, row_end)`, anchor at `start_col` (typically one pixel past the inner divider — i.e. just inside the slot) and walk in the direction given by `scan_right` until hitting a wall or peaking past one. Stamps `hit_id` at each accepted pixel.
///
/// Stop conditions per step: static wall (`chrome α == 0xFF` OR `clip_mask < 128`), or peak-descent (`current α < prev α`, single check). Since `start_col` is positioned one pixel inside the divider on the inner side, the scan goes **outward** across the slot toward the curve / silhouette / strip edge on the far side — no inward scan is needed because the divider itself is the inner boundary and we start past it.
///
/// CRITICAL ordering: called AFTER hairlines + curves are painted, BEFORE symbols + bg fill paint — symbol and bg pixels are opaque and would be misread as walls, collapsing the scan immediately.
pub fn paint_button_hit_row_scan(
    chrome_buf: &[u32],
    clip_mask: &[u8],
    hit_test_map: &mut [HitId],
    width: usize,
    start_col: usize,
    scan_right: bool,
    hit_id: HitId,
    row_start: usize,
    row_end: usize,
    bound_x_min: usize,
    bound_x_max: usize,
) {
    if row_start >= row_end || width == 0 || start_col < bound_x_min || start_col >= bound_x_max {
        return;
    }
    let static_wall =
        |idx: usize| -> bool { (chrome_buf[idx] >> 24) == 0xFF || clip_mask[idx] < 128 };
    for row in row_start..row_end {
        let row_base = row * width;
        let start_idx = row_base + start_col;
        if static_wall(start_idx) {
            continue;
        }
        hit_test_map[start_idx] = hit_id;
        let mut prev_a = (chrome_buf[start_idx] >> 24) & 0xFF;
        if scan_right {
            let mut col = start_col + 1;
            while col < bound_x_max {
                let idx = row_base + col;
                if static_wall(idx) {
                    break;
                }
                let a = (chrome_buf[idx] >> 24) & 0xFF;
                if a < prev_a {
                    break;
                }
                hit_test_map[idx] = hit_id;
                prev_a = a;
                col += 1;
            }
        } else {
            let mut col = start_col;
            while col > bound_x_min {
                col -= 1;
                let idx = row_base + col;
                if static_wall(idx) {
                    break;
                }
                let a = (chrome_buf[idx] >> 24) & 0xFF;
                if a < prev_a {
                    break;
                }
                hit_test_map[idx] = hit_id;
                prev_a = a;
            }
        }
    }
}

/// **Step 2** in the chrome rasterizer (after window perimeter). Paint the BL squircle hairline of the controls strip — row-walk (the curve's near-vertical leg) and col-walk (the near-horizontal leg). Uses [`paint_if_empty`] so writes from the window perimeter are not overwritten. Each curve pixel gets at most ONE writer (this function or the perimeter, whichever ran first).
pub fn draw_strip_curves(
    pixels: &mut [u32],
    hit_test_map: &mut [HitId],
    width: u32,
    height: u32,
    button_size: usize,
    start: usize,
    crossings: &[(u16, u8, u8)],
    edge_vert: u32,
    edge_horiz: u32,
) {
    let Some((w, strip_w, strip_x, button_area_offset, last_row, _h)) =
        strip_layout(width, height, button_size)
    else {
        return;
    };
    if start >= button_size {
        return;
    }
    // Hairline geometry: a 1-pixel line extending inward from the curve into the strip body.
    //   Outer pixel coverage = 1 − fract → α = l (sqrt-gamma'd, stored directly).
    //   Inner pixel coverage = fract → α = h_cov (stored directly).
    // Under composition handles the actual blending with whatever bg is below the chrome layer.
    //
    // Two colours: row-walk paints pixels along the curve's *vertical* face (extends UP the strip's left edge — continues the left-of-window light bevel), col-walk paints along the curve's *horizontal* face (extends RIGHT along the strip's bottom — continues the bottom-of-window shadow bevel). Same shape, two colours because two edges meet at this corner.

    // Row-walk — vertical face → light edge.
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
        // α-conv: opacity α=l for the outer pixel (=255−old_t where old_t=255−l).
        let outer_v = (edge_vert & 0x00FFFFFF) | ((l as u32) << 24);
        let outer_idx = py * w + strip_x + inset;
        pixels[outer_idx] = pixels[outer_idx].under(outer_v, BlendMode::Normal);
        if inset + 1 < strip_w {
            let inner_v = (edge_vert & 0x00FFFFFF) | ((h_cov as u32) << 24);
            let inner_idx = py * w + strip_x + inset + 1;
            pixels[inner_idx] = pixels[inner_idx].under(inner_v, BlendMode::Normal);
        }
    }

    // Col-walk — horizontal face → shadow edge.
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
        let outer_v = (edge_horiz & 0x00FFFFFF) | ((l as u32) << 24);
        let outer_idx = outer_py * w + strip_x + dx;
        pixels[outer_idx] = pixels[outer_idx].under(outer_v, BlendMode::Normal);
        if inset + 1 < button_size {
            let inner_v = (edge_horiz & 0x00FFFFFF) | ((h_cov as u32) << 24);
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
    edge: u32,
) {
    let Some((w, _strip_w, strip_x, button_area_offset, last_row, _h)) =
        strip_layout(width, height, button_size)
    else {
        return;
    };
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
    hit_test_map: &mut [HitId],
    width: u32,
    height: u32,
    button_size: usize,
    start: usize,
    crossings: &[(u16, u8, u8)],
) {
    // hit_test_map is no longer written here — population happens via per-button directional row scans (`paint_button_hit_row_scan`) BEFORE this bg pass runs, using the chrome buffer's post-hairlines/post-curves state as the wall geometry. Param retained for caller-signature stability.
    let _ = hit_test_map;
    let Some((w, _strip_w, strip_x, _button_area_offset, last_row, _h)) =
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
        }
    }
}

/// Rasterize the minimize glyph (a small horizontal squircle dash) centered at `(cx, cy)` with radius `r`. Top-down per-pixel: each pixel inside the squircle footprint computes its coverage and writes either the solid `stroke` colour or a `stroke`-blended-with-`bg` colour. The chrome layer is opaque at the button bg before this call; this function only overwrites pixels INSIDE the glyph footprint.
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
            // AA via α-byte: opacity = gradient/256 (clamped to 255 = fully opaque).
            let opacity = gradient.min(256) as u32;
            if opacity == 0 {
                continue;
            }
            let chrome_alpha = opacity.min(255);
            let value = (stroke & 0x00FFFFFF) | (chrome_alpha << 24);
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
    // `.max(1)` guards the degenerate r=0 case (very small button_size). Without it, r_inner3 would be 0 and `inner_thresh` would divide-by-zero in the gradient calc below.
    let r_inner = (r * 4 / 5).max(1);
    let mut r_inner4 = r_inner * r_inner;
    r_inner4 *= r_inner4;
    let r_inner3 = r_inner * r_inner * r_inner;
    let outer_thresh = (r3 << 2).max(1);
    let inner_thresh = (r_inner3 << 2).max(1);
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
                // INSIDE inner squircle = fill region. Inner edge (stroke ↔ fill) is between two known glyph colours, so pre-blending is correct here (both colours are deterministic, no bg layer involvement). Both `stroke` and `fill` are stored in darkness (theme constants); linear interpolation in darkness space = linear interpolation in visible space, so the formula is identical to the visible-space version. Theme constants are α=0xFF (opaque) by default.
                let dist_from_inner = r_inner4 - dist4;
                if dist_from_inner <= inner_thresh {
                    let gradient = (dist_from_inner << 8) / inner_thresh;
                    let alpha = gradient as u32;
                    let inv = 256 - alpha;
                    let r_blend = (stroke_rgb.0 * inv + fill_rgb.0 * alpha) >> 8;
                    let g_blend = (stroke_rgb.1 * inv + fill_rgb.1 * alpha) >> 8;
                    let b_blend = (stroke_rgb.2 * inv + fill_rgb.2 * alpha) >> 8;
                    0xFF000000 | (r_blend << 16) | (g_blend << 8) | b_blend
                } else {
                    fill
                }
            } else {
                // RING region (between inner and outer). Outer edge AA against the bg layer goes via the chrome α-byte — strip the theme const's default α=0xFF and replace with the AA-modulated value.
                let dist_from_outer = r4 - dist4;
                if dist_from_outer <= outer_thresh {
                    let gradient = (dist_from_outer << 8) / outer_thresh;
                    let opacity = gradient.min(256) as u32;
                    if opacity == 0 {
                        continue;
                    }
                    let chrome_alpha = opacity.min(255);
                    (stroke & 0x00FFFFFF) | (chrome_alpha << 24)
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
            // AA via α-byte: opacity = alpha_f (clamped to 1.0 = fully opaque).
            let opacity = (alpha_f * 256.0).min(256.0) as u32;
            if opacity == 0 {
                continue;
            }
            let chrome_alpha = opacity.min(255);
            let value = (stroke & 0x00FFFFFF) | (chrome_alpha << 24);
            pixels[idx] = pixels[idx].under(value, BlendMode::Normal);
        }
    }
}
