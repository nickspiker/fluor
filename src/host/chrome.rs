//! Window chrome — minimal top-down rasterization. Each pixel in the chrome layer is written by exactly one site. No painter's algorithm anywhere.
//!
//! Currently scoped to **window perimeter hairline** with squircle corner AA. The chrome layer starts at the canonical empty value (`0x00000000` = α=0 transparent, darkness=0); this function paints only the hairline pixels. Everywhere else in the chrome layer stays transparent so panes / bg can pass thru the chrome group's Stack composition. Buttons, glyphs, title text, hover overlay — all deferred to subsequent scaffold steps; reintroduce them only when each can be added without overwriting earlier writes within this same layer.
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
    let span = 2. * window_width as Coord * window_height as Coord
        / (window_width as Coord + window_height as Coord);
    let resize_border = math::ceil(span / 32.);

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
    start_big: usize,
    crossings_big: &[(u16, u8, u8)],
    start_small: usize,
    crossings_small: &[(u16, u8, u8)],
    light: u32,
    shadow: u32,
) {
    let w = width as usize;
    let h = height as usize;

    // Per-corner cap = how far the corner curve reaches inward from its corner. The TL+BR diagonal uses the big table, TR+BL the small one — an asymmetric rounded rect built from one curve at two scales. Each corner is now independent, so the symmetric single-table x↔y walk is replaced by an explicit per-corner walk; the squircle is still self-symmetric within each corner (its own row-walk and col-walk share that corner's table), which is what keeps each corner's curvature continuous.
    let cap_big = start_big + crossings_big.len();
    let cap_small = start_small + crossings_small.len();

    // Straight edges — opaque chrome composed via Under. Each edge spans between the caps of its two end corners, which now differ: e.g. the top edge runs from the TL big cap to the TR small cap. Clip mask along these edges stays at the host's 255 default (fully visible).
    let top_lo = cap_big; // TL
    let top_hi = w - cap_small; // TR
    for x in top_lo..top_hi {
        pixels[x] = pixels[x].under(light, BlendMode::Normal); // top row
    }
    let bot_lo = cap_small; // BL
    let bot_hi = w - cap_big; // BR
    for x in bot_lo..bot_hi {
        let idx = (h - 1) * w + x;
        pixels[idx] = pixels[idx].under(shadow, BlendMode::Normal); // bottom row
    }
    let left_lo = cap_big; // TL
    let left_hi = h - cap_small; // BL
    for y in left_lo..left_hi {
        let lidx = y * w;
        pixels[lidx] = pixels[lidx].under(light, BlendMode::Normal); // left col
    }
    let right_lo = cap_small; // TR
    let right_hi = h - cap_big; // BR
    for y in right_lo..right_hi {
        let ridx = y * w + (w - 1);
        pixels[ridx] = pixels[ridx].under(shadow, BlendMode::Normal); // right col
    }

    // Corner-of-corner cutout (start × start at each corner): the small outer square that the curve never reaches. Each corner uses its OWN start (big corners a larger square, small corners a smaller one).
    for r in 0..start_big {
        for c in 0..start_big {
            clip_mask[r * w + c] = 0; // TL
        }
    }
    for r in 0..start_small {
        for c in (w - start_small)..w {
            clip_mask[r * w + c] = 0; // TR
        }
    }
    for r in (h - start_small)..h {
        for c in 0..start_small {
            clip_mask[r * w + c] = 0; // BL
        }
    }
    for r in (h - start_big)..h {
        for c in (w - start_big)..w {
            clip_mask[r * w + c] = 0; // BR
        }
    }

    let tr_colour =
        |row: usize, col: usize| -> u32 { if row < (w - 1 - col) { light } else { shadow } };
    let bl_colour =
        |row: usize, col: usize| -> u32 { if col < (h - 1 - row) { light } else { shadow } };

    // Two-sided AA convention (unchanged from the symmetric version). OUTER pixel = on the curve, partially outside the window: clip_mask = h_cov trims against the OS bg; chrome stays opaque. INNER pixel = one step inside: clip_mask = 255; chrome's α-byte carries the hairline-vs-bg AA = 255 − h_cov.

    // TL corner (big). Row-walk: near-vertical segment; col-walk: near-horizontal segment. Both use the big table.
    for (i, &(inset_raw, _l, h_cov)) in crossings_big.iter().enumerate() {
        let inset = inset_raw as usize;
        if inset >= cap_big {
            continue;
        }
        let inner_alpha = ((255 - h_cov) as u32) << 24;
        let light_inner = (light & 0x00FFFFFF) | inner_alpha;
        let row_top = start_big + i;
        let col_left = start_big + i;

        for c in 0..inset {
            clip_mask[row_top * w + c] = 0;
        }
        let idx = row_top * w + inset;
        pixels[idx] = pixels[idx].under(light, BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = row_top * w + inset + 1;
        pixels[idx] = pixels[idx].under(light_inner, BlendMode::Normal);
        clip_mask[idx] = 255;

        for r in 0..inset {
            clip_mask[r * w + col_left] = 0;
        }
        let idx = inset * w + col_left;
        pixels[idx] = pixels[idx].under(light, BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = (inset + 1) * w + col_left;
        pixels[idx] = pixels[idx].under(light_inner, BlendMode::Normal);
        clip_mask[idx] = 255;
    }

    // BR corner (big). The shadow diagonal — both adjacent edges are shadow, so the corner is uniform.
    for (i, &(inset_raw, _l, h_cov)) in crossings_big.iter().enumerate() {
        let inset = inset_raw as usize;
        if inset >= cap_big {
            continue;
        }
        let inner_alpha = ((255 - h_cov) as u32) << 24;
        let shadow_inner = (shadow & 0x00FFFFFF) | inner_alpha;
        let row_bot = h - 1 - start_big - i;
        let col_right = w - 1 - start_big - i;

        for c in (w - inset)..w {
            clip_mask[row_bot * w + c] = 0;
        }
        let idx = row_bot * w + (w - 1 - inset);
        pixels[idx] = pixels[idx].under(shadow, BlendMode::Normal);
        clip_mask[idx] = h_cov;
        let idx = row_bot * w + (w - 2 - inset);
        pixels[idx] = pixels[idx].under(shadow_inner, BlendMode::Normal);
        clip_mask[idx] = 255;

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

    // TR corner (small). Light↔shadow transition along the curve via tr_colour.
    for (i, &(inset_raw, _l, h_cov)) in crossings_small.iter().enumerate() {
        let inset = inset_raw as usize;
        if inset >= cap_small {
            continue;
        }
        let inner_alpha = ((255 - h_cov) as u32) << 24;
        let row_top = start_small + i;
        let col_right = w - 1 - start_small - i;

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
    }

    // BL corner (small). Light↔shadow transition along the curve via bl_colour.
    for (i, &(inset_raw, _l, h_cov)) in crossings_small.iter().enumerate() {
        let inset = inset_raw as usize;
        if inset >= cap_small {
            continue;
        }
        let inner_alpha = ((255 - h_cov) as u32) << 24;
        let row_bot = h - 1 - start_small - i;
        let col_left = start_small + i;

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
    }

    let _ = hit_test_map;
}

/// Rasterize the window title text into the chrome layer, left-aligned in the area between the perimeter hairline (left edge) and the controls strip (right edge). Vertically centered on `y_center` (the app-icon orb's row, so the title sits level with the orb wherever it's placed). `left_extra` shifts the start position right to make room for the orb (pass `0` when no orb). `colour` is darkness-packed (typically `theme::TEXT_COLOUR` when focused, `theme::LABEL_COLOUR` when unfocused). Bails on empty title or impractically small `button_size` (below readability — the text wouldn't be legible anyway). Clip rect prevents the title from painting over the controls strip even at long titles or narrow windows. Font is "Open Sans" regular at `button_size * 0.55` — proportional to the rest of the chrome under the current zoom (since button_size is derived from `effective_span`).
pub fn draw_title_text(
    canvas: &mut crate::canvas::Canvas,
    title: &str,
    text_renderer: &mut TextRenderer,
    button_size: usize,
    strip_x: usize,
    left_extra: usize,
    y_center: Coord,
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
    // Clip band is centred on the title's row (a full button_size tall, centred on y_center) so the glyphs aren't clipped after the title drops to follow the orb.
    let half_band = button_size as Coord * 0.5;
    let clip_y0 = (y_center - half_band).max(0.0) as usize;
    let clip_y1 = (y_center + half_band) as usize;
    let clip = Clip::new(left_margin, clip_y0, clip_x_end, clip_y1);
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

    // Status text FIRST. Chrome composites front-to-back ("topmost paints first wins"), so
    // the text must be drawn before the band bg — otherwise it under-blends BEHIND the
    // opaque fill and vanishes. Horizontally centered, vertically centered in the band;
    // `band_h / 2` side padding clips long strings off the BL/BR curves.
    if !text.is_empty() {
        let side_margin = band_h / 2;
        let clip_x_end = width.saturating_sub(side_margin);
        if side_margin < clip_x_end {
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
    }

    let pixels: &mut [u32] = canvas.pixels;

    // Top hairline (1 px) — claims y_top across the full width. The squircle perimeter's clip_mask handles rounding at BL/BR.
    let hairline = 0xFF000000 | (hairline_colour & 0x00FFFFFF);
    let row_top = y_top * width;
    for x in 0..width {
        let idx = row_top + x;
        pixels[idx] = pixels[idx].under(hairline, BlendMode::Normal);
    }

    // BG fill — under-blends beneath the already-drawn text + hairline, filling the rest of
    // the band interior. (Earlier writers — perimeter hairline, corner curve, status text —
    // keep their pixels; bg only lands where the band is still empty.)
    let bg_opaque = 0xFF000000 | (bg & 0x00FFFFFF);
    for y in (y_top + 1)..height {
        let row_base = y * width;
        for x in 0..width {
            let idx = row_base + x;
            pixels[idx] = pixels[idx].under(bg_opaque, BlendMode::Normal);
        }
    }
}

/// Rasterize the top-left app-icon orb: a circular sample of `icon` clipped to `radius`, wrapped in an optional ring stroked in `ring_colour`. The ring is a thin hairline — the icon fills crisply to `r−1`, then a `stroke_width + 1`-px solid band (1px when `stroke_width` is 0), then a 1px outer AA against the chrome. The icon/ring seam is opaque→opaque so it needs no AA; only the outer silhouette is anti-aliased. `cx`/`cy` give the orb centre in pixel coords; `radius` is the icon sampling radius (ring extends outward from it). Without an `icon`, the interior fills with `ring_colour` (treated as a solid dark disk). Without a `ring_colour`, the orb is just the icon clipped to a circle (1-pixel outer-AA against the chrome).
///
/// Pixel sampling is nearest-neighbour from `icon`'s `width × height` source — the source is square in practice (`vsfimg` doesn't reshape) but the math doesn't assume that. Per-pixel cost is one map index + one `under` composite; total work is `O(diameter²)`, well under a millisecond at typical chrome sizes (~30–100 px orbs).
///
/// Hit-test: every pixel inside `r_outer²` (excluding the AA fringe) is tagged with `hit_id` so the host can route clicks to the chrome's app-icon widget. Decorative-only consumers can pass `None` for `hit_test_map` to skip the tag (in which case `hit_id` is ignored).
/// Rasterize the top-left app-icon orb: the icon sampled into a circle of radius `r−1`, wrapped in a thin ring of `ring_colour`. Integer `dist²` bands — same approach as the avatar / window-edge rasterizers: icon interior (`≤ (r−1)²`), then a `stroke_width`-px solid ring, then a 1-px outer AA against the chrome (`(r_aa²−dist²) << 8 / diff`). No floating point, no sqrt.
///
/// `stroke_width = (r >> 5) + 1`: the `+1` is the textbox "minimum 1-px stroke" idiom (an additive floor, not a clamp), and `r >> 5` adds a proportional pixel only on large / zoomed orbs.
///
/// Precondition: the orb is the fixed top-left chrome badge, always fully on-screen (`cx, cy ≥ r_aa` and `cx+r_aa, cy+r_aa < dims` for the chrome's `button_size/2`-centred, `button_size/4`-radius orb), so the bbox is in-bounds by construction — no clamping. A violated precondition wraps a `usize` and panics on the index (fail loud). Without an `icon`, the interior fills with `ring_colour` as a solid disk; `hit_test_map` (when present) tags every pixel inside the solid ring with `hit_id`.
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
        return; // Contract relied on by chrome_widget.rs:309 — degenerate sizes pass thru without drawing, so the layout needs no min-size guard. r ≥ 2 keeps (r-2) non-negative below.
    }
    let stroke_width = (r >> 5) + 1;
    // TOP-DOWN, not partitioned. The icon is the top layer; the ring sits beneath it. Paint the icon FIRST with a soft edge, then fill the ring as a FULL disk underneath. Under-blend is "topmost paints first, the layer beneath shows thru", so the ring bleeds thru the icon's partial-alpha rim → the icon↔ring boundary anti-aliases with no hard step; the background, painted under the whole orb afterward, gives the outer AA.
    //
    // Each edge is a 1-px coverage ramp biased INWARD, not centred and not a band tacked outside. From d ≈ R + (dist²−R²)/2R, coverage = (R²−dist²)/(2R−1): full (255) at dist²=(R−1)², zero at dist²=R², the single AA pixel landing at R. Biasing inward (rather than half-straddling R) lands the edge cleanly on the grid with no 50%-at-the-radius pixel and pulls the solid radius in by 1. BOTH edges are biased the same way so the icon reaches 0 exactly where the ring is still solid — two straddling ramps would overlap into a translucent dip (a see-thru ring) at the boundary. The icon image is unchanged: same sampling, just a clean AA rim.
    let r_icon = r - 1;
    let r_icon2 = r_icon * r_icon;
    let two_icon = 2 * r_icon - 1; // ramp denominator for the icon edge: (R²)−(R−1)²
    let r_ring = r - 1 + stroke_width;
    let r_ring2 = r_ring * r_ring;
    let two_ring = 2 * r_ring - 1; // ramp denominator for the ring edge
    let r_bbox = r_ring; // the inward ramp reaches 0 at r_ring², so the disk never paints beyond r_ring

    let mut htm = hit_test_map;
    let y0 = (cy - r_bbox) as usize;
    let y1 = (cy + r_bbox + 1) as usize;
    let x0 = (cx - r_bbox) as usize;
    let x1 = (cx + r_bbox + 1) as usize;

    for y in y0..y1 {
        let dy = y as isize - cy;
        let dy2 = dy * dy;
        for x in x0..x1 {
            let dx = x as isize - cx;
            let dist2 = dx * dx + dy2;
            if dist2 > r_ring2 {
                continue; // beyond the ring edge — a bbox corner, not part of the disk
            }
            let idx = y * width + x;

            if let Some(map) = htm.as_mut() {
                map[idx] = hit_id; // dist2 ≤ r_ring2 here by the continue above
            }

            // Icon, topmost: inward 1-px coverage ramp on r_icon. Sample only fires while a_icon ≠ 0, i.e. dist2 < r_icon² = (r-1)² → sample_icon's |dx|,|dy| ≤ r-2 precondition holds with margin.
            let num_icon = r_icon2 - dist2;
            let a_icon: u32 = if num_icon <= 0 {
                0
            } else if num_icon >= two_icon {
                255
            } else {
                (num_icon * 255 / two_icon) as u32
            };
            if a_icon != 0 {
                let s = sample_icon(icon, dx, dy, r, ring_colour.unwrap_or(0), darken, brighten);
                pixels[idx] = pixels[idx].under((a_icon << 24) | (s & 0x00FF_FFFF), BlendMode::Normal);
            }
            // Ring, painted UNDER the icon: inward 1-px coverage ramp on r_ring, full disk within. Skipped where the icon is fully opaque (invisible there anyway); where the icon's rim is partial it shows thru → the inner-boundary AA. The ring is solid out to (r_ring−1)² ≥ r_icon², so it backs the entire icon rim — no translucent gap.
            if a_icon != 255 {
                if let Some(ring) = ring_colour {
                    let num_ring = r_ring2 - dist2;
                    let a_ring: u32 = if num_ring <= 0 {
                        0
                    } else if num_ring >= two_ring {
                        255
                    } else {
                        (num_ring * 255 / two_ring) as u32
                    };
                    if a_ring != 0 {
                        pixels[idx] =
                            pixels[idx].under((a_ring << 24) | (ring & 0x00FF_FFFF), BlendMode::Normal);
                    }
                }
            }
        }
    }
}

/// Nearest-neighbour fetch from `icon` for offset `(dx, dy)` from the orb centre, scaled to fit `radius`. Returns an opaque α + darkness pixel after applying `brighten` (photon's 3/2 visible-RGB lift) and `darken` (linear blend toward mid-grey: 0 = icon as-is, 255 = fully grey). Falls back to a solid `fallback_ring` (or dark grey) when no icon is present.
///
/// Precondition: caller only invokes this when `dx² + dy² ≤ (r−1)²`, so `|dx|, |dy| ≤ r−1`, giving `2(dx+r)+1 ∈ [3, 4r−1]` and therefore `sx = (2(dx+r)+1)·width / 4r ∈ [0, width)`. Violating that precondition panics on the index (fail loud).
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
        // Integer nearest-neighbour: the float map `((d+r)+0.5)/(2r)·dim` doubled is `(2(d+r)+1)·dim / (4r)`, exact in integers. The precondition |dx|,|dy| ≤ r−1 keeps `sx, sy ∈ [0, dim)`.
        let denom = 4 * radius;
        let sx = ((2 * (dx + radius) + 1) * img.width as isize / denom) as usize;
        let sy = ((2 * (dy + radius) + 1) * img.height as isize / denom) as usize;
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
    // saturating_sub keeps button_size=0 from underflowing; all the `for ... in 0..button_size` loops fall thru naturally with empty range.
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
    // Hairline geometry: a 1-pixel line extending inward from the curve into the strip body. Outer pixel coverage = 1 − fract → α = l (sqrt-gamma'd, stored directly). Inner pixel coverage = fract → α = h_cov (stored directly). Under composition handles the actual blending with whatever bg is below the chrome layer.
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

/// Rasterize the close glyph — an ✕ of two 45° diagonal capsules — centred at `(cx, cy)` with arm radius `r`. Integer math, same spirit as [`draw_minimize_symbol`]/[`draw_maximize_symbol`]: per pixel, the squared distance to the relevant diagonal is compared against the stroke radius in a scaled (`×8`) integer domain, with a 1-pixel AA band. No floating point, no `sqrt`.
///
/// Coordinates run in a doubled, pixel-centred frame (`ax = 2·dx + 1 = 2·x_actual`), so the half-pixel sample offset and the diagonal's `1/√2` factor fall out as exact integers: a diagonal's perpendicular squared distance is `perp²/8` of the true value and a cap endpoint's is `[(ax−epx)²+(ay−epy)²]/4`. Multiplying thru by 8 (`d8`) clears both, so every test is against `radius²·8 = 2·rad²`. The only division is the AA ramp, whose divisor `denom = 8·rad ≥ 8` is provably non-zero and whose result is provably `< 256` in the ramp branch (there `d8 > inner8`, so `outer8 − d8 < denom`).
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
    let _ = bg;
    let r = r + 1;
    // Doubled-frame stroke half-width (`rad = thickness`, since rad = 2·radius_actual) and arm half-length (`end = 2·(2r/3)`). `thickness ≥ 1` is a visible-stroke floor — a 0-px glyph draws nothing — not a degenerate-input guard.
    let thickness = (r / 3).max(1) as isize;
    let rad = thickness;
    let end = (4 * r / 3) as isize;
    let cxi = cx as isize;
    let cyi = cy as isize;

    // AA band straddling the stroke surface (±0.5 px = ±1 in the doubled frame), in the ×8 squared-distance domain.
    let inner8 = 2 * (rad - 1) * (rad - 1);
    let outer8 = 2 * (rad + 1) * (rad + 1);
    let denom = outer8 - inner8; // = 8·rad ≥ 8

    let r_i = r as isize;
    let x0 = (cxi - r_i).max(0) as usize;
    let x1 = ((cxi + r_i + 1).max(0) as usize).min(width);
    let y0 = (cyi - r_i).max(0) as usize;
    let y1 = ((cyi + r_i + 1).max(0) as usize).min(height);

    for py in y0..y1 {
        for px in x0..x1 {
            let ax = 2 * (px as isize - cxi) + 1;
            let ay = 2 * (py as isize - cyi) + 1;
            // ╲ diagonal for the TL/BR quadrants, ╱ for TR/BL. `perp` = across-stroke axis, `along` = down-stroke axis.
            let backslash = (px >= cx && py >= cy) || (px < cx && py < cy);
            let (perp, along) = if backslash {
                (ax - ay, ax + ay)
            } else {
                (ax + ay, ax - ay)
            };
            let d8 = if along.abs() <= 2 * end {
                perp * perp // beside the segment — distance is purely perpendicular
            } else {
                // past a cap — distance to the rounded endpoint at (±end, ±end).
                let s = if along > 0 { 1 } else { -1 };
                let (epx, epy) = if backslash { (s * end, s * end) } else { (s * end, -s * end) };
                2 * ((ax - epx) * (ax - epx) + (ay - epy) * (ay - epy))
            };
            if d8 > outer8 {
                continue;
            }
            let alpha = if d8 <= inner8 {
                255u32
            } else {
                (((outer8 - d8) << 8) / denom) as u32
            };
            if alpha == 0 {
                continue;
            }
            let idx = py * width + px;
            pixels[idx] =
                pixels[idx].under((stroke & 0x00FF_FFFF) | (alpha << 24), BlendMode::Normal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::chrome_widget::compute_squircle_crossings;

    /// The asymmetric corner split actually lands at a realistic window size: TL+BR (big diagonal) consume a deeper corner than TR+BL (small diagonal), so the top straight edge starts further in on the left (cap_big) than it ends short on the right (cap_small), and the clip-mask cutout is deeper at TL than TR. Verifies the wiring (right table → right corner) and the cap math at the geometry level, no display needed. Uses a realistic base radius (span/4 for a normal window) so the squircle is well-formed rather than the degenerate near-square corner you get at tiny radii.
    #[test]
    fn asymmetric_corners_big_tl_br_small_tr_bl() {
        let (w, h) = (800usize, 600usize);
        let mut pixels = vec![0u32; w * h];
        let mut hit = vec![HIT_NONE; w * h];
        let mut clip = vec![255u8; w * h];

        // Mirror the chrome_widget scheme: TR+BL keep the original base radius (span/4), TL+BR are a literal 2× (span/2).
        let span = 2.0 * w as f32 * h as f32 / (w as f32 + h as f32);
        let base = span / 4.0;
        let (start_big, cross_big) = compute_squircle_crossings(base * 2.0, 24);
        let (start_small, cross_small) = compute_squircle_crossings(base, 24);
        let cap_big = start_big + cross_big.len();
        let cap_small = start_small + cross_small.len();
        assert!(cap_big > cap_small, "big corner must reach deeper than small");

        draw_window_edges_and_mask(
            &mut pixels,
            &mut hit,
            &mut clip,
            w as u32,
            h as u32,
            start_big,
            &cross_big,
            start_small,
            &cross_small,
            theme::WINDOW_LIGHT_EDGE,
            theme::WINDOW_SHADOW_EDGE,
        );

        // Both outer corner-of-corner pixels are cut from the window shape.
        assert_eq!(clip[0], 0, "TL outer corner cut out");
        assert_eq!(clip[w - 1], 0, "TR outer corner cut out");

        // Asymmetry in the clip mask: at the top row, the TL cutout reaches column start_big−1 (deep), while the TR cutout only reaches start_small in from the right. Since start_big > start_small the left corner is carved deeper — sample a column that is inside the big-corner cutout but, mirrored on the right, would already be straight edge.
        assert!(start_big > start_small, "big corner-of-corner is deeper");
        assert_eq!(clip[start_big - 1], 0, "TL still cut at start_big−1");

        // The top straight run is [cap_big, w−cap_small): non-empty, painted at both ends, and positioned asymmetrically (starts deeper on the left than it stops short on the right).
        let top_run_lo = cap_big;
        let top_run_hi = w - cap_small;
        assert!(top_run_lo < top_run_hi, "top straight run non-empty");
        assert_ne!(pixels[top_run_lo] >> 24, 0, "top edge painted at left start");
        assert_ne!(
            pixels[top_run_hi - 1] >> 24,
            0,
            "top edge painted at right end"
        );
        // Just LEFT of the run start is still corner (not straight edge): the pixel at cap_big−1 on the top row is not a straight-edge write.
        // (It may be a curve hairline or cut, but the contiguous straight run begins exactly at cap_big.)
        assert!(
            cap_big > cap_small,
            "left edge consumes more than right — the asymmetry"
        );

        // Bottom edge is the mirror: runs [cap_small, w−cap_big) — starts shallow on the left (BL small), stops deep on the right (BR big).
        let bot_row = (h - 1) * w;
        assert_ne!(
            pixels[bot_row + cap_small] >> 24,
            0,
            "bottom edge painted at left start (small)"
        );
        assert_ne!(
            pixels[bot_row + (w - cap_big) - 1] >> 24,
            0,
            "bottom edge painted at right end (big)"
        );
    }
}
