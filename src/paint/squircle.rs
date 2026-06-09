//! Squircle pill rasterizers — the rounded-rectangle / pill silhouette every chrome perimeter, textbox, and button in fluor is built on. Two variants of every entry point: integer-`squirdleyness` (fast path via `powi`) and fractional-`squirdleyness` (`_f` suffix, `powf` for in-between shapes like Button's `1.5`).
//!
//! Layout: each public entry point dispatches into one of the shared `_with_crossings` rasterizer bodies so the only difference between fast and slow paths is which crossings generator they call. Inner kernels (`draw_squircle_pill_unclipped` / `_clipped`) are private and shared by both paths.

use super::HitId;
use crate::canvas::Canvas;
use crate::pixel::{Blend, BlendMode};

/// Hard-pixel squircle pill with AA on both the X-axis curve (sides) and Y-axis curve (cap tops/bottoms). Photon's avatar-ring strategy in one call — render twice with different sizes/colours to get a stroke ring.
///
/// Photon-faithful: precompute squircle crossings once (`(inset_px, l_aa, h_aa)` per pixel-row offset into the cap), then walk pure integer indices per corner. Each crossing produces BOTH a vertical-edge AA pixel and a horizontal-edge AA pixel via the squircle's diagonal symmetry — no separate per-col walk needed.
///
/// Every interior + AA-edge write composes the new pixel UNDERNEATH whatever's already in the buffer via [`Blend::under`] (`BlendMode::Normal`). Two stacked pills in the same buffer therefore behave like any other front-to-back composite: draw the topmost first (it lands cleanly into the empty buffer), then the underneath one (it fills only the remaining α budget the topmost left behind). No max-α tiebreaker, no inner/outer dual mode — one consistent kernel.
pub fn draw_squircle_pill(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    colour: u32,
    squirdleyness: i32,
) {
    let radius_f = pill_h as f32 * 0.5;
    let crossings = squircle_crossings(radius_f, squirdleyness);
    draw_squircle_pill_with_crossings(canvas, pill_x, pill_y, pill_w, pill_h, colour, &crossings);
}

/// Fractional-exponent variant of [`draw_squircle_pill`]. Identical rasterization (shares the inner `_with_crossings` worker) — only the crossings-table computation differs, using [`squircle_crossings_f`]'s `powf` path so non-integer `squirdleyness` traces smooth in-between shapes (e.g. `1.5` between ellipse and diamond). Slower per-call; use only when the desired shape can't be expressed with an integer exponent.
pub fn draw_squircle_pill_f(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    colour: u32,
    squirdleyness: f32,
) {
    let radius_f = pill_h as f32 * 0.5;
    let crossings = squircle_crossings_f(radius_f, squirdleyness);
    draw_squircle_pill_with_crossings(canvas, pill_x, pill_y, pill_w, pill_h, colour, &crossings);
}

/// Shared rasterizer body for [`draw_squircle_pill`] and [`draw_squircle_pill_f`]. Takes pre-computed `crossings` so the two entry points can dispatch through one painting kernel — the integer / fractional choice lives entirely in which crossings generator the caller used.
fn draw_squircle_pill_with_crossings(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    colour: u32,
    crossings: &[(u16, u8, u8)],
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if pill_w <= 0 || pill_h <= 0 {
        return;
    }
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    // Bbox-overlap early-out — pill entirely off-buffer.
    if pill_x + pill_w <= 0 || pill_y + pill_h <= 0 || pill_x >= buf_w_i || pill_y >= buf_h_i {
        return;
    }
    // Damage = pill bbox clipped to buffer. Helpers use the same range internally for their per-row clips.
    {
        let dx0 = pill_x.max(0) as usize;
        let dy0 = pill_y.max(0) as usize;
        let dx1 = (pill_x + pill_w).min(buf_w_i).max(0) as usize;
        let dy1 = (pill_y + pill_h).min(buf_h_i).max(0) as usize;
        canvas.damage.add_bounds(dx0, dy0, dx1, dy1);
    }

    let radius = (pill_h / 2) as isize;
    // α + darkness: force opaque (α=0xFF) by setting the top byte. RGB darkness intact.
    let solid = (colour & 0x00FF_FFFF) | 0xFF000000;
    let colour_rgb = colour & 0x00FF_FFFF;
    let pixels: &mut [u32] = canvas.pixels;

    // Fast/slow split. Fast path: pill bbox fully inside the buffer → no per-pixel checks. Slow path: partial overhang (scroll/resize transitions) → range clips at the corner-block boundary so each AA write has its row already proven in-buffer.
    let fully_inside =
        pill_x >= 0 && pill_y >= 0 && pill_x + pill_w <= buf_w_i && pill_y + pill_h <= buf_h_i;

    if fully_inside {
        draw_squircle_pill_unclipped(
            pixels,
            buf_w,
            pill_x as usize,
            pill_y as usize,
            pill_w as usize,
            pill_h as usize,
            radius as usize,
            crossings,
            colour_rgb,
            solid,
        );
    } else {
        draw_squircle_pill_clipped(
            pixels, buf_w, buf_h, pill_x, pill_y, pill_w, pill_h, radius, crossings, colour_rgb,
            solid,
        );
    }

    // Center rectangle between the two semicircle caps — both paths share this. Range already clips to [0, buf_w) × [0, buf_h) via .max(0).min(buf), so no per-pixel guard needed.
    let center_x_start = pill_x + radius;
    let center_x_end = pill_x + pill_w - radius;
    if center_x_start < center_x_end {
        let cy_start = pill_y.max(0) as usize;
        let cy_end = (pill_y + pill_h).min(buf_h_i).max(0) as usize;
        let cx_start = center_x_start.max(0) as usize;
        let cx_end = center_x_end.min(buf_w_i).max(0) as usize;
        for fy in cy_start..cy_end {
            let row_base = fy * buf_w;
            for fx in cx_start..cx_end {
                let idx = row_base + fx;
                pixels[idx] = pixels[idx].under(solid, BlendMode::Normal);
            }
        }
    }
}

/// Two-tone variant of [`draw_squircle_pill`] — "football seam" split. EVERY pixel in the pill (interior + AA edges) picks `light` vs `shadow` by comparing its row against a piecewise seam anchored at TR and BL corners: 45° down-left from TR through the right cap (`seam_y = pill_w − 1 − dx` for `dx ≥ pill_w − 1 − pill_h/2`), flat along the centerline through the rectangular middle (`seam_y = pill_h/2`), then 45° down-left to BL through the left cap (`seam_y = pill_h − 1 − dx` for `dx ≤ pill_h − 1 − pill_h/2`). `dy ≤ seam_y` → light, else shadow. The light region encloses TL; the shadow region encloses BR. Pixel-aligned, symmetric, never slices the squircle curve at an off-angle.
///
/// For square pills (`pill_w == pill_h`) the flat middle collapses and the seam becomes the bbox anti-diagonal (TR→BL) — natural degenerate case.
///
/// No `fill` parameter — the pill is exactly two colours.
///
/// Every pixel write composes UNDER the buffer via [`Blend::under`] (Normal) — same kernel as the single-colour path.
pub fn draw_squircle_pill_two_tone(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    light: u32,
    shadow: u32,
    squirdleyness: i32,
    hit_map: Option<&mut [HitId]>,
    hit_id: HitId,
) {
    let radius_f = pill_h as f32 * 0.5;
    let crossings = squircle_crossings(radius_f, squirdleyness);
    draw_squircle_pill_two_tone_with_crossings(
        canvas, pill_x, pill_y, pill_w, pill_h, light, shadow, &crossings, hit_map, hit_id,
    );
}

/// Fractional-exponent variant of [`draw_squircle_pill_two_tone`]. Identical seam logic and rasterizer — only the crossings-table computation swaps to [`squircle_crossings_f`] for non-integer `squirdleyness`. Used by [`crate::widgets::Button`] for its in-between-ellipse-and-diamond silhouette.
pub fn draw_squircle_pill_two_tone_f(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    light: u32,
    shadow: u32,
    squirdleyness: f32,
    hit_map: Option<&mut [HitId]>,
    hit_id: HitId,
) {
    let radius_f = pill_h as f32 * 0.5;
    let crossings = squircle_crossings_f(radius_f, squirdleyness);
    draw_squircle_pill_two_tone_with_crossings(
        canvas, pill_x, pill_y, pill_w, pill_h, light, shadow, &crossings, hit_map, hit_id,
    );
}

fn draw_squircle_pill_two_tone_with_crossings(
    canvas: &mut Canvas,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    light: u32,
    shadow: u32,
    crossings: &[(u16, u8, u8)],
    hit_map: Option<&mut [HitId]>,
    hit_id: HitId,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if pill_w <= 0 || pill_h <= 0 {
        return;
    }
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    if pill_x + pill_w <= 0 || pill_y + pill_h <= 0 || pill_x >= buf_w_i || pill_y >= buf_h_i {
        return;
    }
    {
        let dx0 = pill_x.max(0) as usize;
        let dy0 = pill_y.max(0) as usize;
        let dx1 = (pill_x + pill_w).min(buf_w_i).max(0) as usize;
        let dy1 = (pill_y + pill_h).min(buf_h_i).max(0) as usize;
        canvas.damage.add_bounds(dx0, dy0, dx1, dy1);
    }

    let radius = (pill_h / 2) as isize;
    let light_solid = (light & 0x00FF_FFFF) | 0xFF000000;
    let shadow_solid = (shadow & 0x00FF_FFFF) | 0xFF000000;
    let light_rgb = light & 0x00FF_FFFF;
    let shadow_rgb = shadow & 0x00FF_FFFF;
    let pixels: &mut [u32] = canvas.pixels;
    let mut hit_map = hit_map;

    // Football-seam pick: 45° from TR down-left into the right cap → flat across the centerline → 45° down-left to BL. dy ≤ seam_y(dx) → light, which puts TL inside the light region and BR inside the shadow region.
    let half_h = pill_h / 2;
    let a_left = pill_h - 1 - half_h; // left-cap segment exits onto the flat at this dx (seam_y = half_h)
    let a_right = pill_w - 1 - half_h; // right-cap segment enters from the flat at this dx
    let seam_y = |dx: isize| -> isize {
        if dx <= a_left {
            pill_h - 1 - dx
        } else if dx >= a_right {
            pill_w - 1 - dx
        } else {
            half_h
        }
    };
    let pick_rgb = |dx: isize, dy: isize| -> u32 {
        if dy <= seam_y(dx) {
            light_rgb
        } else {
            shadow_rgb
        }
    };
    let pick_solid = |dx: isize, dy: isize| -> u32 {
        if dy <= seam_y(dx) {
            light_solid
        } else {
            shadow_solid
        }
    };

    for (i, &(inset, _l, h)) in crossings.iter().enumerate() {
        if inset as usize > i {
            break;
        }
        let i_iso = i as isize;
        let inset_iso = inset as isize;
        let h_u32 = h as u32;
        for &(flip_x, flip_y) in &[(false, false), (true, false), (false, true), (true, true)] {
            let v_row = if flip_y {
                pill_y + pill_h - 1 - radius + i_iso
            } else {
                pill_y + radius - i_iso
            };
            let h_col = if flip_x {
                pill_x + pill_w - 1 - radius + i_iso
            } else {
                pill_x + radius - i_iso
            };

            if v_row >= 0 && v_row < buf_h_i {
                let row_base = v_row as usize * buf_w;
                let v_aa_col = if flip_x {
                    pill_x + pill_w - 1 - inset_iso
                } else {
                    pill_x + inset_iso
                };
                let diag_col = h_col;
                if v_aa_col >= 0 && v_aa_col < buf_w_i {
                    let dx = v_aa_col - pill_x;
                    let dy = v_row - pill_y;
                    write_aa(
                        pixels,
                        row_base + v_aa_col as usize,
                        pick_rgb(dx, dy),
                        h_u32,
                    );
                }
                let (fx_start, fx_end) = if flip_x {
                    (diag_col, v_aa_col)
                } else {
                    (v_aa_col + 1, diag_col + 1)
                };
                let fs = fx_start.max(0) as usize;
                let fe = fx_end.max(0).min(buf_w_i) as usize;
                let dy = v_row - pill_y;
                for fx in fs..fe {
                    let idx = row_base + fx;
                    let dx = fx as isize - pill_x;
                    pixels[idx] = pixels[idx].under(pick_solid(dx, dy), BlendMode::Normal);
                    if let Some(hm) = hit_map.as_deref_mut() {
                        hm[idx] = hit_id;
                    }
                }
            }

            if h_col >= 0 && h_col < buf_w_i {
                let col_us = h_col as usize;
                let h_aa_row = if flip_y {
                    pill_y + pill_h - 1 - inset_iso
                } else {
                    pill_y + inset_iso
                };
                let diag_row = v_row;
                if h_aa_row >= 0 && h_aa_row < buf_h_i {
                    let dx = h_col - pill_x;
                    let dy = h_aa_row - pill_y;
                    write_aa(
                        pixels,
                        h_aa_row as usize * buf_w + col_us,
                        pick_rgb(dx, dy),
                        h_u32,
                    );
                }
                let (fy_start, fy_end) = if flip_y {
                    (diag_row, h_aa_row)
                } else {
                    (h_aa_row + 1, diag_row + 1)
                };
                let fs = fy_start.max(0) as usize;
                let fe = fy_end.max(0).min(buf_h_i) as usize;
                let dx = h_col - pill_x;
                for fy in fs..fe {
                    let idx = fy * buf_w + col_us;
                    let dy = fy as isize - pill_y;
                    pixels[idx] = pixels[idx].under(pick_solid(dx, dy), BlendMode::Normal);
                    if let Some(hm) = hit_map.as_deref_mut() {
                        hm[idx] = hit_id;
                    }
                }
            }
        }
    }

    let center_x_start = pill_x + radius;
    let center_x_end = pill_x + pill_w - radius;
    if center_x_start < center_x_end {
        let cy_start = pill_y.max(0) as usize;
        let cy_end = (pill_y + pill_h).min(buf_h_i).max(0) as usize;
        let cx_start = center_x_start.max(0) as usize;
        let cx_end = center_x_end.min(buf_w_i).max(0) as usize;
        for fy in cy_start..cy_end {
            let row_base = fy * buf_w;
            let dy = fy as isize - pill_y;
            for fx in cx_start..cx_end {
                let idx = row_base + fx;
                let dx = fx as isize - pill_x;
                pixels[idx] = pixels[idx].under(pick_solid(dx, dy), BlendMode::Normal);
                if let Some(hm) = hit_map.as_deref_mut() {
                    hm[idx] = hit_id;
                }
            }
        }
    }
}

/// Fast-path squircle pill rasterizer. Bounds checks intentionally absent.
///
/// **Rule 0 — WHY/PROOF/PREVENTS:** CALLER GUARANTEES (verified at dispatch in [`draw_squircle_pill`]):
///   - `pill_x + pill_w ≤ buf_w` and `pill_y + pill_h ≤ buf_h` (cast to `usize`).
///   - `pill_w ≥ pill_h ≥ 2` (pill geometry: caps don't overlap).
///
/// PROOF — every AA / fill index is `< pixels.len() == buf_w * buf_h`:
///   - `inset ∈ [0, radius]` (from `squircle_crossings`), `i ∈ [0, radius]` (loop break at diagonal).
///   - All written cols are in `[pill_x, pill_x + pill_w)` (subset of `[0, buf_w)` by caller guarantee).
///   - All written rows are in `[pill_y, pill_y + pill_h)` (subset of `[0, buf_h)` by caller guarantee).
///   - Therefore `row * buf_w + col < buf_h * buf_w = pixels.len()`.
///
/// PREVENTS: nothing — this path runs only when the proof holds. Bounds-check elision lets the compiler emit unchecked stores.
fn draw_squircle_pill_unclipped(
    pixels: &mut [u32],
    buf_w: usize,
    pill_x: usize,
    pill_y: usize,
    pill_w: usize,
    pill_h: usize,
    radius: usize,
    crossings: &[(u16, u8, u8)],
    colour_rgb: u32,
    solid: u32,
) {
    for (i, &(inset, _l, h)) in crossings.iter().enumerate() {
        if inset as usize > i {
            break;
        }
        let inset_us = inset as usize;
        let h_u32 = h as u32;
        for &(flip_x, flip_y) in &[(false, false), (true, false), (false, true), (true, true)] {
            // --- vertical edge: AA pixel + horizontal fill to the diagonal ---
            let v_row = if flip_y {
                pill_y + pill_h - 1 - radius + i
            } else {
                pill_y + radius - i
            };
            let v_aa_col = if flip_x {
                pill_x + pill_w - 1 - inset_us
            } else {
                pill_x + inset_us
            };
            let diag_col = if flip_x {
                pill_x + pill_w - 1 - radius + i
            } else {
                pill_x + radius - i
            };
            let row_base = v_row * buf_w;
            write_aa(pixels, row_base + v_aa_col, colour_rgb, h_u32);
            let (fx_start, fx_end) = if flip_x {
                (diag_col, v_aa_col)
            } else {
                (v_aa_col + 1, diag_col + 1)
            };
            for fx in fx_start..fx_end {
                let idx = row_base + fx;
                pixels[idx] = pixels[idx].under(solid, BlendMode::Normal);
            }

            // --- horizontal edge: AA pixel + vertical fill to the diagonal ---
            let h_col = if flip_x {
                pill_x + pill_w - 1 - radius + i
            } else {
                pill_x + radius - i
            };
            let h_aa_row = if flip_y {
                pill_y + pill_h - 1 - inset_us
            } else {
                pill_y + inset_us
            };
            let diag_row = if flip_y {
                pill_y + pill_h - 1 - radius + i
            } else {
                pill_y + radius - i
            };
            write_aa(pixels, h_aa_row * buf_w + h_col, colour_rgb, h_u32);
            let (fy_start, fy_end) = if flip_y {
                (diag_row, h_aa_row)
            } else {
                (h_aa_row + 1, diag_row + 1)
            };
            for fy in fy_start..fy_end {
                let idx = fy * buf_w + h_col;
                pixels[idx] = pixels[idx].under(solid, BlendMode::Normal);
            }
        }
    }
}

/// Slow-path squircle pill rasterizer. Used when the pill partially overhangs the buffer.
///
/// **Rule 0 — WHY/PROOF/PREVENTS for the bounds checks:** CALLER ALLOWS: `pill_x` may be negative; `pill_x + pill_w` may exceed `buf_w` (same for y). Partial overhang is the design case (scroll-out, resize transitions, off-pane drag).
///
/// PROOF that no closed-form i-range clip suffices: `inset[i]` is non-linear in `i` (squircle curve), so the AA-pixel column `pill_x + inset` can't be cleanly bracketed by a single i-range when the pill straddles `x=0` or `x=buf_w`. Linear-in-`i` coords (rows and `h_col`) ARE clipped at the corner-block level — one branch per corner instead of one per pixel. The inset-dependent AA column gets one inline check.
///
/// PREVENTS: OOB pixel write / slice panic at the math↔buffer boundary when the pill's geometric corner falls outside the buffer.
fn draw_squircle_pill_clipped(
    pixels: &mut [u32],
    buf_w: usize,
    buf_h: usize,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    radius: isize,
    crossings: &[(u16, u8, u8)],
    colour_rgb: u32,
    solid: u32,
) {
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    for (i, &(inset, _l, h)) in crossings.iter().enumerate() {
        if inset as usize > i {
            break;
        }
        let i_iso = i as isize;
        let inset_iso = inset as isize;
        let h_u32 = h as u32;
        for &(flip_x, flip_y) in &[(false, false), (true, false), (false, true), (true, true)] {
            let v_row = if flip_y {
                pill_y + pill_h - 1 - radius + i_iso
            } else {
                pill_y + radius - i_iso
            };
            let h_col = if flip_x {
                pill_x + pill_w - 1 - radius + i_iso
            } else {
                pill_x + radius - i_iso
            };

            // --- Vertical edge: row constraint hoisted ---
            if v_row >= 0 && v_row < buf_h_i {
                let row_base = v_row as usize * buf_w;
                let v_aa_col = if flip_x {
                    pill_x + pill_w - 1 - inset_iso
                } else {
                    pill_x + inset_iso
                };
                let diag_col = h_col;
                if v_aa_col >= 0 && v_aa_col < buf_w_i {
                    write_aa(pixels, row_base + v_aa_col as usize, colour_rgb, h_u32);
                }
                let (fx_start, fx_end) = if flip_x {
                    (diag_col, v_aa_col)
                } else {
                    (v_aa_col + 1, diag_col + 1)
                };
                let fs = fx_start.max(0) as usize;
                let fe = fx_end.max(0).min(buf_w_i) as usize;
                for fx in fs..fe {
                    let idx = row_base + fx;
                    pixels[idx] = pixels[idx].under(solid, BlendMode::Normal);
                }
            }

            // --- Horizontal edge: column constraint hoisted ---
            if h_col >= 0 && h_col < buf_w_i {
                let col_us = h_col as usize;
                let h_aa_row = if flip_y {
                    pill_y + pill_h - 1 - inset_iso
                } else {
                    pill_y + inset_iso
                };
                let diag_row = v_row;
                if h_aa_row >= 0 && h_aa_row < buf_h_i {
                    write_aa(
                        pixels,
                        h_aa_row as usize * buf_w + col_us,
                        colour_rgb,
                        h_u32,
                    );
                }
                let (fy_start, fy_end) = if flip_y {
                    (diag_row, h_aa_row)
                } else {
                    (h_aa_row + 1, diag_row + 1)
                };
                let fs = fy_start.max(0) as usize;
                let fe = fy_end.max(0).min(buf_h_i) as usize;
                for fy in fs..fe {
                    let idx = fy * buf_w + col_us;
                    pixels[idx] = pixels[idx].under(solid, BlendMode::Normal);
                }
            }
        }
    }
}

/// Generate photon's squircle crossings: one entry per pixel-row offset from the cap edge into the diagonal. Each entry is `(inset_int, l_aa, h_aa)` where `inset_int` is the integer column offset where the curve crosses that row, and `l/h_aa = sqrt(frac(inset))*256` / `sqrt(1-frac(inset))*256` are the perceptual AA weights (low = outside fraction, high = inside fraction). Verbatim port of photon's loop in `compositing.rs::draw_textbox`.
///
/// **Integer-exponent fast path.** `powi(x, n)` for small `n` is `n−1` multiplies (`x*x*x` for `n=3`), while `powf` is `exp(y * ln(x))` — orders of magnitude slower. Chrome's perimeter at `n=24`, textbox's `n=3`, and most "pill or rounded-rect" shapes use integer values; they get the fast path here. For fractional shapes (e.g. button's `n=1.5` for an in-between ellipse / diamond), see [`squircle_crossings_f`].
pub fn squircle_crossings(radius: f32, squirdleyness: i32) -> alloc::vec::Vec<(u16, u8, u8)> {
    let mut crossings: alloc::vec::Vec<(u16, u8, u8)> = alloc::vec::Vec::new();
    let mut offset = 0f32;
    loop {
        let y_norm = offset / radius;
        let x_norm = crate::math::powf(
            1. - crate::math::powi(y_norm, squirdleyness),
            1. / squirdleyness as f32,
        );
        let cx = x_norm * radius;
        let inset = radius - cx;
        if inset >= 0. {
            let l = (crate::math::sqrt(crate::math::fract(inset)) * 256.) as u8;
            let h = (crate::math::sqrt(1. - crate::math::fract(inset)) * 256.) as u8;
            crossings.push((inset as u16, l, h));
        }
        if cx < offset {
            break;
        }
        offset += 1.0;
    }
    crossings
}

/// Fractional-exponent variant of [`squircle_crossings`]. Same math, swaps the inner `powi` for `powf` so non-integer `squirdleyness` (e.g. `1.5`, `1.41`, `2.5`) traces the in-between curves between ellipse / diamond / squared-pill. Slower per iteration; only worth using when the desired shape can't be expressed with an integer exponent. Outer root in the integer version is already `powf(x, 1/n)` so the speed gap is roughly the inner `powi` vs `powf` — ~2× total for small `n`.
pub fn squircle_crossings_f(radius: f32, squirdleyness: f32) -> alloc::vec::Vec<(u16, u8, u8)> {
    let mut crossings: alloc::vec::Vec<(u16, u8, u8)> = alloc::vec::Vec::new();
    let mut offset = 0f32;
    loop {
        let y_norm = offset / radius;
        let x_norm = crate::math::powf(
            1. - crate::math::powf(y_norm, squirdleyness),
            1. / squirdleyness,
        );
        let cx = x_norm * radius;
        let inset = radius - cx;
        if inset >= 0. {
            let l = (crate::math::sqrt(crate::math::fract(inset)) * 256.) as u8;
            let h = (crate::math::sqrt(1. - crate::math::fract(inset)) * 256.) as u8;
            crossings.push((inset as u16, l, h));
        }
        if cx < offset {
            break;
        }
        offset += 1.0;
    }
    crossings
}

/// AA write at a proven-in-buffer index. Composes a partial-α pixel (`α=h_aa`, straight darkness=`colour_rgb`) UNDERNEATH whatever's already in the buffer via [`Blend::under`] — same kernel as every other compositing op in fluor. The buffer is treated as the topmost-first composite (anything already painted there sits "above" this write). With `Normal` mode, an empty pixel (`0x00000000`) absorbs the new contribution fully; an already-opaque pixel takes the early-out and the new write is invisible.
#[inline]
fn write_aa(pixels: &mut [u32], idx: usize, colour_rgb: u32, h_aa: u32) {
    let new_pixel = (h_aa << 24) | colour_rgb;
    pixels[idx] = pixels[idx].under(new_pixel, BlendMode::Normal);
}

/// Photon's squircle inset formula — single row, parameterized by `squirdleyness`. Returns the curve's inset (distance from the bbox edge to the leftmost / rightmost inside pixel) for a row at `y_from_center` rows above or below the squircle's vertical center.
///
/// `squirdleyness = 2` → circle. `squirdleyness = 3` → photon's textbox pill default (slightly flatter than a circle). Higher = more rectangular. Both `draw_textbox_pill` (AA path) and the textbox widget's hard-pixel renderer route thru this so any tweak to the curve math flows thru both code paths.
///
/// Identical to the per-iteration formula at photon's [`compositing.rs:4567-4570`](/mnt/Octopus/Code/photon/src/ui/compositing.rs).
#[inline]
pub fn squircle_inset(y_from_center: f32, radius: f32, squirdleyness: i32) -> f32 {
    let y_norm = (y_from_center / radius).min(1.0);
    let x_norm = crate::math::powf(
        1.0 - crate::math::powi(y_norm, squirdleyness),
        1.0 / squirdleyness as f32,
    );
    radius - x_norm * radius
}
