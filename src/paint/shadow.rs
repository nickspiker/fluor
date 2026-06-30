//! Drop shadow rasterizer — 45° diagonal rays cast from chrome's silhouette in both directions (DR for the down-right drop shadow, UL for the up-left ambient occlusion). Per-cell α decay is `alpha[k+1] = (alpha[k] × factor_256) >> 8`. Each ray writes black premult α into the screen buffer, composing UNDER existing chrome content so AA edge pixels get a clean shadow boost without crunching.
//!
//! Operates directly on the `&mut [u32]` screen buffer (post-finalize, OS visible-RGB format) — no Canvas wrapper, no Blend trait. The math is purely α-byte manipulation; RGB is left alone (or directly assigned to black).
//!
//! **Dead code dropped on extraction.** The original module included `blend_aa_edge`, `cast_ray_dr`/`cast_ray_ul`, `band_fill_right_dr`/`band_fill_bottom_dr`/`band_fill_top_ul`/`band_fill_left_ul`, `cast_debug_ray`/`cast_debug_ray_ul`, and a `DEBUG_SHADOW_RGB` const — all unused (paint_shadow only calls `cast_shadow_ray` and `cast_shadow_ray_ul`). Removed at extraction time to keep this file focused; `git show 9b42319^:src/paint.rs` recovers them if needed.

/// Directional drop shadow via 45-degree diagonal rays cast from each chrome edge pixel. For each row in chrome's y-range, scan leftward from the rectangle's right edge to find that row's rightmost chrome pixel (handles squircle corners where chrome ends inside the rectangle), then cast a single diagonal ray stepping `(x+1, y+1)` per pixel with α decaying by `factor_256` each step. Same pattern for the bottom edge per column. Light source is upper-left → rays trail to the lower-right.
///
/// `factor_256` is the per-pixel decay multiplier in `[1, 255]`: 240 ≈ ×0.9375 (~60-pixel ray), 250 ≈ ×0.9766 (~150-pixel). Caller scales from `effective_span` so shadow length is RU-invariant.
///
/// Visual: each chrome edge pixel emits one diagonal ray. Adjacent rows emit adjacent diagonals → together they cover the shadow region with parallel diagonal stripes. Right-edge rays and bottom-edge rays only overlap in a thin BR corner area ("minimal double taps") and max-compose there.
///
/// AA-edge fix folded in: when the scan finds the chrome edge pixel, if its α is partial (squircle AA), force it to 0xFF in place. Chrome's premult RGB over the implicit black shadow underneath is identity, so opaque-with-premult-RGB blends correctly.
pub fn paint_shadow(
    screen: &mut [u32],
    scr_w: usize,
    factor_256: u32,
    shadow_seed: u32,
    window_rect: (i32, i32, i32, i32),
) {
    if scr_w == 0 || factor_256 == 0 || factor_256 >= 256 {
        return;
    }
    let scr_h = screen.len() / scr_w;
    let (rx, ry, rw, rh) = window_rect;
    if rw <= 0 || rh <= 0 {
        return;
    }
    let x_chrome_left = rx.max(0) as usize;
    let y_chrome_top = ry.max(0) as usize;
    let x_chrome_end = ((rx + rw).max(0) as usize).min(scr_w);
    let y_chrome_end = ((ry + rh).max(0) as usize).min(scr_h);
    if x_chrome_left >= x_chrome_end || y_chrome_top >= y_chrome_end {
        return;
    }

    // === SEEK + FAN OF DR RAYS ===
    let right_col = x_chrome_end - 1;
    if y_chrome_top + 1 >= y_chrome_end {
        let _ = scr_h;
        let _ = factor_256;
        return;
    }
    let mut x0 = right_col;
    let mut y0 = y_chrome_top + 1;
    let mut found = false;
    loop {
        let a = (screen[y0 * scr_w + x0] >> 24) & 0xFF;
        if a != 0 {
            found = true;
            break;
        }
        if x0 == x_chrome_left || y0 + 1 >= y_chrome_end {
            break;
        }
        x0 -= 1;
        y0 += 1;
    }
    if !found {
        let _ = scr_h;
        let _ = factor_256;
        return;
    }

    let tr_seed_x = x0;
    let tr_seed_y = y0;

    cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, x0, y0);

    let y_center = (y_chrome_top + y_chrome_end) / 2;
    let mut x = x0;
    let mut y = y0;
    while y < y_center && y + 1 < y_chrome_end {
        y += 1;
        while ((screen[y * scr_w + x] >> 24) & 0xFF) >= 0xFE {
            if x + 1 >= scr_w || y + 1 >= scr_h {
                break;
            }
            x += 1;
            y += 1;
        }
        cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, x, y);
    }

    let bot_row = y_chrome_end - 1;
    while y + 1 < y_chrome_end {
        y += 1;
        while ((screen[y * scr_w + x] >> 24) & 0xFF) == 0 {
            if x == 0 || y == 0 {
                break;
            }
            x -= 1;
            y -= 1;
        }
        cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, x, y);
        let dx_left = right_col.saturating_sub(x);
        let dy_up = bot_row.saturating_sub(y);
        if dx_left > dy_up {
            break;
        }
    }

    let x_center = (x_chrome_left + x_chrome_end) / 2;
    while x > x_center {
        if y + 1 <= bot_row {
            y += 1;
        }
        while ((screen[y * scr_w + x] >> 24) & 0xFF) == 0 {
            if x == 0 || y == 0 {
                break;
            }
            x -= 1;
            y -= 1;
        }
        cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, x, y);
        if y >= bot_row && x > x_center {
            if x == 0 {
                break;
            }
            x -= 1;
        }
    }

    if bot_row == 0 {
        let _ = scr_h;
        return;
    }
    let mut xf = x_chrome_left;
    let mut yf = bot_row - 1;
    let mut found_f = false;
    loop {
        let a = (screen[yf * scr_w + xf] >> 24) & 0xFF;
        if a != 0 {
            found_f = true;
            break;
        }
        if xf + 1 >= x_chrome_end || yf == y_chrome_top {
            break;
        }
        xf += 1;
        yf -= 1;
    }
    if found_f {
        let bl_seed_x = xf;
        let bl_seed_y = yf;
        cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, xf, yf);
        while xf < x_center {
            xf += 1;
            if xf >= scr_w {
                break;
            }
            while yf < bot_row && ((screen[yf * scr_w + xf] >> 24) & 0xFF) >= 0xFE {
                if xf + 1 >= scr_w || yf + 1 >= scr_h {
                    break;
                }
                xf += 1;
                yf += 1;
            }
            cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, xf, yf);
        }

        let tl_seed = shadow_seed >> 1;
        let mut xg = bl_seed_x;
        let mut yg = bl_seed_y;
        cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xg, yg);
        let y_center = (y_chrome_top + y_chrome_end) / 2;
        while yg > y_center {
            if yg == 0 {
                break;
            }
            yg -= 1;
            while ((screen[yg * scr_w + xg] >> 24) & 0xFF) >= 0xFE {
                if xg == 0 || yg == 0 {
                    break;
                }
                xg -= 1;
                yg -= 1;
            }
            cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xg, yg);
        }

        while yg > y_chrome_top {
            yg -= 1;
            while ((screen[yg * scr_w + xg] >> 24) & 0xFF) == 0 {
                if xg + 1 >= scr_w || yg + 1 >= scr_h {
                    break;
                }
                xg += 1;
                yg += 1;
            }
            cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xg, yg);
            let dx_right = xg.saturating_sub(x_chrome_left);
            let dy_down = yg.saturating_sub(y_chrome_top);
            if dx_right > dy_down {
                break;
            }
        }

        while xg < x_center {
            if yg > y_chrome_top {
                yg -= 1;
            }
            while ((screen[yg * scr_w + xg] >> 24) & 0xFF) == 0 {
                if xg + 1 >= scr_w || yg + 1 >= scr_h {
                    break;
                }
                xg += 1;
                yg += 1;
            }
            cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xg, yg);
            if yg <= y_chrome_top && xg < x_center {
                if xg + 1 >= scr_w {
                    break;
                }
                xg += 1;
            }
        }
    }

    let tl_seed = shadow_seed >> 1;
    let mut xj = tr_seed_x;
    let mut yj = tr_seed_y;
    cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xj, yj);
    while xj > x_center {
        if xj == 0 {
            break;
        }
        xj -= 1;
        while yj > y_chrome_top && ((screen[yj * scr_w + xj] >> 24) & 0xFF) >= 0xFE {
            if xj == 0 || yj == 0 {
                break;
            }
            xj -= 1;
            yj -= 1;
        }
        cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xj, yj);
    }

    let _ = scr_h;
}

/// Cast one DR shadow ray. Flat loop, single zero check on shadow_alpha or screen edge. Per cell: * α > 0 (chrome AA) → under-blend black: α += shadow_alpha * (256 - α) >> 8; chrome RGB stays (shadow's premult RGB is 0 since visible black).
///   * α == 0 (transparent) → direct-assign black premult: (shadow_alpha << 24), RGB all zero.
/// Decay shadow_alpha by factor_256 each step.
fn cast_shadow_ray(
    screen: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    factor_256: u32,
    shadow_seed: u32,
    mut x: usize,
    mut y: usize,
) {
    let mut shadow_alpha: u32 = shadow_seed;
    loop {
        let idx = y * scr_w + x;
        let p = screen[idx];
        let a = (p >> 24) & 0xFF;
        if a == 0 {
            screen[idx] = shadow_alpha << 24;
        } else {
            let cover = 256 - a;
            let boost = (shadow_alpha * cover) >> 8;
            let na = (a + boost).min(0xFF);
            screen[idx] = (p & 0x00FFFFFF) | (na << 24);
        }
        shadow_alpha = (shadow_alpha * factor_256) >> 8;
        if shadow_alpha == 0 || x + 1 >= scr_w || y + 1 >= scr_h {
            break;
        }
        x += 1;
        y += 1;
    }
}

/// Mirror of [`cast_shadow_ray`] — UL direction (-1, -1). Same alpha math; stops at zero alpha or screen origin (x == 0 || y == 0). Used by the TL ambient-occlusion pass which goes up-left from chrome's top + left boundaries.
fn cast_shadow_ray_ul(
    screen: &mut [u32],
    scr_w: usize,
    factor_256: u32,
    shadow_seed: u32,
    mut x: usize,
    mut y: usize,
) {
    let mut shadow_alpha: u32 = shadow_seed;
    loop {
        let idx = y * scr_w + x;
        let p = screen[idx];
        let a = (p >> 24) & 0xFF;
        if a == 0 {
            screen[idx] = shadow_alpha << 24;
        } else {
            let cover = 256 - a;
            let boost = (shadow_alpha * cover) >> 8;
            let na = (a + boost).min(0xFF);
            screen[idx] = (p & 0x00FFFFFF) | (na << 24);
        }
        shadow_alpha = (shadow_alpha * factor_256) >> 8;
        if shadow_alpha == 0 || x == 0 || y == 0 {
            break;
        }
        x -= 1;
        y -= 1;
    }
}
