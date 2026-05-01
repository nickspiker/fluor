//! Window chrome — verbatim port of photon's window controls. **Do not "improve" or "simplify" the math here.** The function below is photon's [draw_window_controls](/mnt/Octopus/Code/photon/src/ui/compositing.rs) with `Self::` prefixes stripped to free-function calls. The squircle bottom-left corner, the per-pixel `hit_test_map` write, the symbol placement at `bw/4` offset — all unchanged.
//!
//! Symbol rendering and `blend_rgb_only` live in [`crate::paint`]. Hit IDs are integers so the desktop host can index `hit_test_map[y * w + x]` directly on click.
//!
//! Chrome here = window controls strip + edges. Title-bar text, scroll bars, status bar, tooltips, context menus, per-pane chrome — all not yet built.

use crate::paint;
use crate::theme;

/// Hit-test IDs written into the per-pixel map by [`draw_window_controls`]. Same numbering as photon.
pub const HIT_NONE: u8 = 0;
pub const HIT_MINIMIZE_BUTTON: u8 = 1;
pub const HIT_MAXIMIZE_BUTTON: u8 = 2;
pub const HIT_CLOSE_BUTTON: u8 = 3;

/// Resize edge classification, photon's enum verbatim.
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

/// Photon's `get_resize_edge`. Edge thickness derived from harmonic-mean span.
pub fn get_resize_edge(window_width: u32, window_height: u32, x: f32, y: f32) -> ResizeEdge {
    let span = 2.0 * window_width as f32 * window_height as f32 / (window_width as f32 + window_height as f32);
    let resize_border = (span / 32.0).ceil();

    let at_left = x < resize_border;
    let at_right = x > (window_width as f32 - resize_border);
    let at_top = y < resize_border;
    let at_bottom = y > (window_height as f32 - resize_border);

    if at_top && at_left { ResizeEdge::TopLeft }
    else if at_top && at_right { ResizeEdge::TopRight }
    else if at_bottom && at_left { ResizeEdge::BottomLeft }
    else if at_bottom && at_right { ResizeEdge::BottomRight }
    else if at_top { ResizeEdge::Top }
    else if at_bottom { ResizeEdge::Bottom }
    else if at_left { ResizeEdge::Left }
    else if at_right { ResizeEdge::Right }
    else { ResizeEdge::None }
}

/// Verbatim port of photon's `draw_window_controls`. Renders the top-right control strip — grey background, two-tone hairline edge, squircle bottom-left corner, plus three glyph buttons (minimize, maximize, close) — and writes per-pixel button IDs into `hit_test_map` for downstream click routing.
///
/// `ru` is the resolution-units multiplier (default `1.0`); button height = `ceil(span / 32 * ru)`.
pub fn draw_window_controls(
    pixels: &mut [u32],
    hit_test_map: &mut [u8],
    window_width: u32,
    window_height: u32,
    ru: f32,
) -> (usize, Vec<(u16, u8, u8)>, usize, usize) {
    let window_width = window_width as usize;
    let window_height = window_height as usize;

    // Calculate button dimensions using harmonic mean (span) scaled by ru span = 2wh/(w+h), base button size = span/32, scaled by ru (zoom multiplier)
    let span = 2.0 * window_width as f32 * window_height as f32
        / (window_width as f32 + window_height as f32);
    let button_height = (span / 32.0 * ru).ceil() as usize;
    let button_width = button_height;
    let total_width = button_width * 7 / 2;

    // Buttons extend to top-right corner of window
    let mut x_start = window_width - total_width;
    let y_start = 0;

    // Build squircle crossings for bottom-left corner
    let radius = span * ru / 4.;
    let squirdleyness = 24;

    let mut crossings: Vec<(u16, u8, u8)> = Vec::new();
    let mut y = 1f32;
    loop {
        let y_norm = y / radius;
        let x_norm = (1.0 - y_norm.powi(squirdleyness)).powf(1.0 / squirdleyness as f32);
        let x = x_norm * radius;
        let inset = radius - x;
        if inset > 0. {
            crossings.push((
                inset as u16,
                (inset.fract().sqrt() * 256.) as u8,
                ((1. - inset.fract()).sqrt() * 256.) as u8,
            ));
        }
        if x < y {
            break;
        }
        y += 1.;
    }
    let start = (radius - y) as usize;
    let crossings: Vec<(u16, u8, u8)> = crossings.into_iter().rev().collect();

    let edge_colour = theme::WINDOW_LIGHT_EDGE;
    let bg_colour = theme::WINDOW_CONTROLS_BG;

    // Left edge (vertical) - draw light hairline following squircle curve
    let mut y_offset = start;
    for (inset, l, h) in &crossings {
        if y_offset >= button_height {
            break;
        }
        let py = y_start + button_height - 1 - y_offset;

        // Fill grey to the right of the curve and populate hit test map
        let col_end = total_width.min(window_width - x_start);
        for col in (*inset as usize + 2)..col_end - 1 {
            let px = x_start + col;
            let pixel_idx = py * window_width + px;

            pixels[pixel_idx] = bg_colour;

            // Determine which button this pixel belongs to. Buttons are drawn with a button_width / 4 offset.
            let button_area_x_start = x_start + button_width / 4;
            let button_id = if px < button_area_x_start {
                HIT_MINIMIZE_BUTTON
            } else {
                let x_in_button_area = px - button_area_x_start;
                if x_in_button_area < button_width {
                    HIT_MINIMIZE_BUTTON
                } else if x_in_button_area < button_width * 2 {
                    HIT_MAXIMIZE_BUTTON
                } else {
                    HIT_CLOSE_BUTTON
                }
            };
            hit_test_map[pixel_idx] = button_id;
        }

        let px = x_start + *inset as usize;
        let pixel_idx = py * window_width + px;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], edge_colour, *l, *h);

        let px = x_start + *inset as usize + 1;
        let pixel_idx = py * window_width + px;
        pixels[pixel_idx] = paint::blend_rgb_only(bg_colour, edge_colour, *h, *l);

        let button_area_x_start = x_start + button_width / 4;
        let button_id = if px < button_area_x_start {
            HIT_MINIMIZE_BUTTON
        } else {
            let x_in_button_area = px - button_area_x_start;
            if x_in_button_area < button_width {
                HIT_MINIMIZE_BUTTON
            } else if x_in_button_area < button_width * 2 {
                HIT_MAXIMIZE_BUTTON
            } else {
                HIT_CLOSE_BUTTON
            }
        };
        hit_test_map[pixel_idx] = button_id;

        y_offset += 1;
    }

    // Bottom edge (horizontal)
    let mut x_offset = start;
    let crossing_limit = crossings.len().min(window_width - (x_start + start));
    for &(inset, l, h) in &crossings[..crossing_limit] {
        let i = inset as usize;
        let px = x_start + x_offset;

        let py = y_start + button_height - 1 - i;
        let pixel_idx = py * window_width + px;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], edge_colour, l, h);

        for row in (i + 2)..start {
            let py = y_start + button_height - 1 - row;
            let pixel_idx = py * window_width + px;

            pixels[pixel_idx] = bg_colour;

            let button_area_x_start = x_start + button_width / 4;
            let button_id = if px < button_area_x_start {
                HIT_MINIMIZE_BUTTON
            } else {
                let x_in_button_area = px - button_area_x_start;
                if x_in_button_area < button_width {
                    HIT_MINIMIZE_BUTTON
                } else if x_in_button_area < button_width * 2 {
                    HIT_MAXIMIZE_BUTTON
                } else {
                    HIT_CLOSE_BUTTON
                }
            };
            hit_test_map[pixel_idx] = button_id;
        }

        let py = y_start + button_height - 1 - (i + 1);
        let pixel_idx = py * window_width + px;
        pixels[pixel_idx] = paint::blend_rgb_only(bg_colour, edge_colour, h, l);

        let button_area_x_start = x_start + button_width / 4;
        let button_id = if px < button_area_x_start {
            HIT_MINIMIZE_BUTTON
        } else {
            let x_in_button_area = px - button_area_x_start;
            if x_in_button_area < button_width {
                HIT_MINIMIZE_BUTTON
            } else if x_in_button_area < button_width * 2 {
                HIT_MAXIMIZE_BUTTON
            } else {
                HIT_CLOSE_BUTTON
            }
        };
        hit_test_map[pixel_idx] = button_id;

        x_offset += 1;
    }

    // Continue bottom edge linearly from where squircle ends to window edge
    let linear_start_x = x_start + start + crossings.len();
    let edge_y = y_start + button_height - 1;

    for px in linear_start_x..window_width {
        let pixel_idx = edge_y * window_width + px;
        pixels[pixel_idx] = edge_colour;

        for row in 1..start {
            let py = edge_y - row;
            let pixel_idx = py * window_width + px;
            pixels[pixel_idx] = bg_colour;

            // All pixels past the squircle belong to close button
            hit_test_map[pixel_idx] = HIT_CLOSE_BUTTON;
        }
    }

    x_start += button_width / 4;

    // Draw button symbols using glyph colours
    paint::glyph::minimize_symbol(
        pixels,
        window_width,
        x_start + button_width / 2,
        y_start + button_width / 2,
        button_width / 4,
        theme::MINIMIZE_GLYPH,
    );

    paint::glyph::maximize_symbol(
        pixels,
        window_width,
        x_start + button_width + button_width / 2,
        y_start + button_width / 2,
        button_width / 4,
        theme::MAXIMIZE_GLYPH,
        theme::MAXIMIZE_GLYPH_INTERIOR,
    );

    paint::glyph::close_symbol(
        pixels,
        window_width,
        x_start + button_width * 2 + button_width / 2,
        y_start + button_width / 2,
        button_width / 4,
        theme::CLOSE_GLYPH,
    );
    (start, crossings, x_start, button_height)
}

/// Verbatim port of photon's `draw_window_edges_and_mask`. Paints the two-tone window perimeter (light top/left, shadow bottom/right) and clips out squircle corners by zeroing pixels outside the squircle (those pixels become fully transparent under `with_transparent(true)`). Takes `start` and `crossings` from the matching call to [`draw_window_controls`].
pub fn draw_window_edges_and_mask(
    pixels: &mut [u32],
    hit_test_map: &mut [u8],
    width: u32,
    height: u32,
    start: usize,
    crossings: &[(u16, u8, u8)],
) {
    let light_colour = theme::WINDOW_LIGHT_EDGE;
    let shadow_colour = theme::WINDOW_SHADOW_EDGE;

    // Fill all four edges with white before squircle clipping
    // Top edge
    for x in 0..width {
        let idx = 0 * width + x;
        pixels[idx as usize] = light_colour;
    }

    // Bottom edge
    for x in 0..width {
        let idx = (height - 1) * width + x;
        pixels[idx as usize] = shadow_colour;
    }

    // Left edge
    for y in 0..height {
        let idx = y * width + 0;
        pixels[idx as usize] = light_colour;
    }

    // Right edge
    for y in 0..height {
        let idx = y * width + (width - 1);
        pixels[idx as usize] = shadow_colour;
    }

    // Fill four corner squares and clear hitmap
    for row in 0..start {
        for col in 0..start {
            let idx = row * width as usize + col;
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }
    }
    for row in 0..start {
        for col in (width as usize - start)..width as usize {
            let idx = row * width as usize + col;
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }
    }
    for row in (height as usize - start)..height as usize {
        for col in 0..start {
            let idx = row * width as usize + col;
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }
    }
    for row in (height as usize - start)..height as usize {
        for col in (width as usize - start)..width as usize {
            let idx = row * width as usize + col;
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }
    }

    // Top left/right edges
    let mut y_top = start;
    for crossing in 0..crossings.len() {
        let (inset, l, h) = crossings[crossing];
        // Left edge fill
        for idx in y_top * width as usize..y_top * width as usize + inset as usize {
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }

        // Left edge outer pixel
        let pixel_idx = y_top * width as usize + inset as usize;
        if paint::PREMULTIPLIED {
            pixels[pixel_idx] = paint::scale_alpha(light_colour, h);
        } else {
            pixels[pixel_idx] = (light_colour & 0x00FFFFFF) | ((h as u32) << 24);
        }
        if h < 255 {
            hit_test_map[pixel_idx] = HIT_NONE; // NEEDS FIXED!!!
        }

        // Left edge inner pixel
        let pixel_idx = pixel_idx + 1;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], light_colour, h, l);

        // Right edge inner pixel
        let pixel_idx = y_top * width as usize + width as usize - 2 - inset as usize;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], shadow_colour, h, l);

        // Right edge outer pixel
        let pixel_idx = pixel_idx + 1;
        if paint::PREMULTIPLIED {
            pixels[pixel_idx] = paint::scale_alpha(shadow_colour, h);
        } else {
            pixels[pixel_idx] = (shadow_colour & 0x00FFFFFF) | ((h as u32) << 24);
        }
        if h < 255 {
            hit_test_map[pixel_idx] = HIT_NONE;
        }

        // Right edge fill
        for idx in (y_top * width as usize + width as usize - inset as usize)
            ..((y_top + 1) * width as usize)
        {
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }
        y_top += 1;
    }

    // Bottom left/right edges
    let mut y_bottom = height as usize - start - 1;
    for crossing in 0..crossings.len() {
        let (inset, l, h) = crossings[crossing];

        // Left edge fill
        for idx in y_bottom * width as usize..y_bottom * width as usize + inset as usize {
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }

        // Left outer edge pixel
        let pixel_idx = y_bottom * width as usize + inset as usize;
        if paint::PREMULTIPLIED {
            pixels[pixel_idx] = paint::scale_alpha(light_colour, h);
        } else {
            pixels[pixel_idx] = (light_colour & 0x00FFFFFF) | ((h as u32) << 24);
        }
        if h < 255 {
            hit_test_map[pixel_idx] = HIT_NONE;
        }

        // Left inner edge pixel
        let pixel_idx = pixel_idx + 1;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], light_colour, h, l);

        // Right edge inner pixel
        let pixel_idx = y_bottom * width as usize + width as usize - 2 - inset as usize;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], shadow_colour, h, l);

        // Right edge outer pixel
        let pixel_idx = pixel_idx + 1;
        if paint::PREMULTIPLIED {
            pixels[pixel_idx] = paint::scale_alpha(shadow_colour, h);
        } else {
            pixels[pixel_idx] = (shadow_colour & 0x00FFFFFF) | ((h as u32) << 24);
        }
        if h < 255 {
            hit_test_map[pixel_idx] = HIT_NONE;
        }

        // Right edge fill
        for idx in (y_bottom * width as usize + width as usize - inset as usize)
            ..((y_bottom + 1) * width as usize)
        {
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }

        y_bottom -= 1;
    }

    // Left side top/bottom edges
    let mut x_left = start;
    for crossing in 0..crossings.len() {
        let (inset, l, h) = crossings[crossing];

        // Top edge fill
        for row in 0..inset as usize {
            let idx = row * width as usize + x_left;
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }

        // Top outer edge pixel
        let pixel_idx = inset as usize * width as usize + x_left;
        if paint::PREMULTIPLIED {
            pixels[pixel_idx] = paint::scale_alpha(light_colour, h);
        } else {
            pixels[pixel_idx] = (light_colour & 0x00FFFFFF) | ((h as u32) << 24);
        }
        if h < 255 {
            hit_test_map[pixel_idx] = HIT_NONE;
        }

        // Top inner edge pixel
        let pixel_idx = (inset as usize + 1) * width as usize + x_left;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], light_colour, h, l);

        // Bottom outer edge pixel
        let pixel_idx = (height as usize - 1 - inset as usize) * width as usize + x_left;
        if paint::PREMULTIPLIED {
            pixels[pixel_idx] = paint::scale_alpha(shadow_colour, h);
        } else {
            pixels[pixel_idx] = (shadow_colour & 0x00FFFFFF) | ((h as u32) << 24);
        }
        if h < 255 {
            hit_test_map[pixel_idx] = HIT_NONE;
        }

        // Bottom inner edge pixel
        let pixel_idx = (height as usize - 2 - inset as usize) * width as usize + x_left;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], shadow_colour, h, l);

        // Bottom edge fill
        for row in (height as usize - inset as usize)..height as usize {
            let idx = row * width as usize + x_left;
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }

        x_left += 1;
    }

    // Right side top/bottom edges
    let mut x_right = width as usize - start - 1;
    for crossing in 0..crossings.len() {
        let (inset, l, h) = crossings[crossing];

        // Top edge fill
        for row in 0..inset as usize {
            let idx = row * width as usize + x_right;
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }

        // Top outer edge pixel
        let pixel_idx = inset as usize * width as usize + x_right;
        if paint::PREMULTIPLIED {
            pixels[pixel_idx] = paint::scale_alpha(light_colour, h);
        } else {
            pixels[pixel_idx] = (light_colour & 0x00FFFFFF) | ((h as u32) << 24);
        }
        if h < 255 {
            hit_test_map[pixel_idx] = HIT_NONE;
        }

        // Top inner edge pixel
        let pixel_idx = (inset as usize + 1) * width as usize + x_right;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], light_colour, h, l);

        // Bottom outer edge pixel
        let pixel_idx = (height as usize - 1 - inset as usize) * width as usize + x_right;
        if paint::PREMULTIPLIED {
            pixels[pixel_idx] = paint::scale_alpha(shadow_colour, h);
        } else {
            pixels[pixel_idx] = (shadow_colour & 0x00FFFFFF) | ((h as u32) << 24);
        }
        if h < 255 {
            hit_test_map[pixel_idx] = HIT_NONE;
        }

        // Bottom inner edge pixel
        let pixel_idx = (height as usize - 2 - inset as usize) * width as usize + x_right;
        pixels[pixel_idx] = paint::blend_rgb_only(pixels[pixel_idx], shadow_colour, h, l);

        // Bottom edge fill
        for row in (height as usize - inset as usize)..height as usize {
            let idx = row * width as usize + x_right;
            pixels[idx] = 0;
            hit_test_map[idx] = HIT_NONE;
        }

        x_right -= 1;
    }
}

/// Verbatim port of photon's `draw_button_hairlines`. Vertical 1-px hairlines between minimize/maximize and maximize/close, drawn from the strip's vertical center outward until they hit a pixel of a different colour (so they stop cleanly at the squircle edge above and the bottom edge below).
pub fn draw_button_hairlines(
    pixels: &mut [u32],
    hit_test_map: &mut [u8],
    window_width: u32,
    _window_height: u32,
    button_x_start: usize,
    button_height: usize,
    _start: usize,
    _crossings: &[(u16, u8, u8)],
) {
    let width = window_width as usize;
    let y_start = 0;

    // button_width equals button_height (passed in, already scaled with span * ru)
    let button_width = button_height;

    // Two hairlines: at 1.0 and 2.0 button widths from button area start
    let left_px = button_x_start + button_width;
    let right_px = button_x_start + button_width * 2;

    let center_y = y_start + button_height / 2;
    let edge_colour = theme::WINDOW_CONTROLS_HAIRLINE;

    // Left hairline — upward from center
    let center_colour = pixels[center_y * width + left_px];
    for py in (y_start..=center_y).rev() {
        let idx = py * width + left_px;
        let diff = pixels[idx] != center_colour;
        pixels[idx] = edge_colour;
        hit_test_map[idx] = HIT_NONE;
        if diff {
            break;
        }
    }
    // Left hairline — downward from center+1
    for py in (center_y + 1)..(y_start + button_height) {
        let idx = py * width + left_px;
        let diff = pixels[idx] != center_colour;
        pixels[idx] = edge_colour;
        hit_test_map[idx] = HIT_NONE;
        if diff {
            break;
        }
    }

    // Right hairline — upward from center
    let center_colour_right = pixels[center_y * width + right_px];
    for py in (y_start..=center_y).rev() {
        let idx = py * width + right_px;
        let diff = pixels[idx] != center_colour_right;
        pixels[idx] = edge_colour;
        hit_test_map[idx] = HIT_NONE;
        if diff {
            break;
        }
    }
    // Right hairline — downward from center+1
    for py in (center_y + 1)..(y_start + button_height) {
        let idx = py * width + right_px;
        let diff = pixels[idx] != center_colour_right;
        pixels[idx] = edge_colour;
        hit_test_map[idx] = HIT_NONE;
        if diff {
            break;
        }
    }
}

/// Verbatim port of photon's `draw_button_hover_by_pixels`. Add or subtract a packed-u32 hover delta over a precomputed pixel list (deltas chosen so RGB wraps and alpha absorbs the carry).
pub fn draw_button_hover_by_pixels(pixels: &mut [u32], pixel_list: &[usize], hover: bool, button_id: u8) {
    let hover_delta = match button_id {
        HIT_CLOSE_BUTTON => theme::CLOSE_HOVER,
        HIT_MAXIMIZE_BUTTON => theme::MAXIMIZE_HOVER,
        HIT_MINIMIZE_BUTTON => theme::MINIMIZE_HOVER,
        _ => return,
    };
    for &hit_idx in pixel_list {
        pixels[hit_idx] = if hover {
            pixels[hit_idx].wrapping_add(hover_delta)
        } else {
            pixels[hit_idx].wrapping_sub(hover_delta)
        };
    }
}

/// Build a pixel-index list for `button_id` by scanning `hit_test_map`. Photon caches these lists across frames in its `Compositor`; for v0 we recompute on hover-state change (still O(width*height) but only once per hover event, not per frame).
pub fn pixels_for_button(hit_test_map: &[u8], button_id: u8) -> Vec<usize> {
    if button_id == HIT_NONE { return Vec::new(); }
    let mut out = Vec::new();
    for (i, &id) in hit_test_map.iter().enumerate() {
        if id == button_id { out.push(i); }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pixels_for_button_returns_matching_indices() {
        let mut map = vec![HIT_NONE; 16];
        map[3] = HIT_CLOSE_BUTTON;
        map[7] = HIT_CLOSE_BUTTON;
        map[10] = HIT_MAXIMIZE_BUTTON;
        let close = pixels_for_button(&map, HIT_CLOSE_BUTTON);
        assert_eq!(close, vec![3, 7]);
        let max = pixels_for_button(&map, HIT_MAXIMIZE_BUTTON);
        assert_eq!(max, vec![10]);
    }

    #[test]
    fn hover_by_pixels_shifts_listed_pixels_only() {
        let mut pixels = vec![0xFF20_2024u32; 8];
        draw_button_hover_by_pixels(&mut pixels, &[2, 5], true, HIT_CLOSE_BUTTON);
        assert_ne!(pixels[2], 0xFF20_2024);
        assert_ne!(pixels[5], 0xFF20_2024);
        assert_eq!(pixels[0], 0xFF20_2024);
        assert_eq!(pixels[7], 0xFF20_2024);
    }

    #[test]
    fn resize_edge_corners_win() {
        assert_eq!(get_resize_edge(1920, 1080, 0.0, 0.0), ResizeEdge::TopLeft);
        assert_eq!(get_resize_edge(1920, 1080, 1919.9, 0.0), ResizeEdge::TopRight);
        assert_eq!(get_resize_edge(1920, 1080, 0.0, 1079.9), ResizeEdge::BottomLeft);
        assert_eq!(get_resize_edge(1920, 1080, 1919.9, 1079.9), ResizeEdge::BottomRight);
    }

    #[test]
    fn resize_edge_none_in_center() {
        assert_eq!(get_resize_edge(1920, 1080, 960.0, 540.0), ResizeEdge::None);
    }

    #[test]
    fn draw_window_controls_writes_hit_map() {
        let w = 800u32;
        let h = 600u32;
        let mut pixels = vec![0u32; (w * h) as usize];
        let mut hit_map = vec![HIT_NONE; (w * h) as usize];
        let _ = draw_window_controls(&mut pixels, &mut hit_map, w, h, 1.0);
        // Some pixels in the top-right strip should have button IDs.
        let close_count = hit_map.iter().filter(|&&id| id == HIT_CLOSE_BUTTON).count();
        let max_count = hit_map.iter().filter(|&&id| id == HIT_MAXIMIZE_BUTTON).count();
        let min_count = hit_map.iter().filter(|&&id| id == HIT_MINIMIZE_BUTTON).count();
        assert!(close_count > 0, "no close pixels");
        assert!(max_count > 0, "no maximize pixels");
        assert!(min_count > 0, "no minimize pixels");
    }
}
