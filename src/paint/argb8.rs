//! The 8-bit-only helpers: the colour boundary into the authored palette, the theme-delta arithmetic, and the passes that work on the OS's finalized visible pixels (the hover overlay, the damage outline, the window-control glyphs).
//! Each speaks bytes on purpose, so none of it draws at a buffer's depth ([`crate::pixel::Pixel`]): [`pack_argb`] and [`unpack_argb`] make and read the 8-bit `g2t` colours every depth's kernels take (`Pixel::from_g2t8` widens them), the `wrap_*` deltas are byte arithmetic on theme constants, and the overlay, outline and glyphs write the 8-bit screen buffer after finalize.
//! The overlay alone reads the app's own buffer, at its depth, each pixel's colour step taken where the present took it ([`super::Finalize::to_sdr_visible_at`], dither included), so a restored pixel matches its neighbours.
//! They lived in paint.rs and shape.rs until the rendering core went generic; the paths are unchanged (`crate::paint::pack_argb`, `crate::paint::glyph`, and so on, thru paint's re-export).

use super::HitId;

/// Pack four 8-bit channels into a fluor internal pixel (`0xααRRGGBB`, α + darkness convention). The public API takes visible RGB and opacity α (`a = 255` means fully opaque) so consumer code reads naturally. Inside, RGB is stored as darkness (`255 − channel`); α is stored direct. This is the canonical external→internal boundary.
#[inline]
pub fn pack_argb(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((a as u32) << 24) | (((255 - r) as u32) << 16) | (((255 - g) as u32) << 8) | ((255 - b) as u32)
}


/// Wrap-add the RGB bytes of `delta_rgb` into `pixels[i]` for every `i` where `hit_map[i] == target_id`. The α byte is preserved. Used to apply a "tint" effect (hover, focus, etc.) on top of an already-painted region whose pixels were stamped with a known hit ID — mirrors the chrome's bake_hover pattern but generic over any hit ID. `wrapping_sub` of the same delta exactly reverses the add, so callers maintain old_delta/new_delta state and unbake/bake atomically.
pub fn wrap_add_rgb_where(pixels: &mut [u32], hit_map: &[HitId], target_id: HitId, delta_rgb: u32) {
    let dr = ((delta_rgb >> 16) & 0xFF) as u8;
    let dg = ((delta_rgb >> 8) & 0xFF) as u8;
    let db = (delta_rgb & 0xFF) as u8;
    for i in 0..pixels.len() {
        if hit_map[i] == target_id {
            let p = pixels[i];
            let a = p & 0xFF00_0000;
            let r = (((p >> 16) & 0xFF) as u8).wrapping_add(dr) as u32;
            let g = (((p >> 8) & 0xFF) as u8).wrapping_add(dg) as u32;
            let b = ((p & 0xFF) as u8).wrapping_add(db) as u32;
            pixels[i] = a | (r << 16) | (g << 8) | b;
        }
    }
}

/// Generic per-hit-id overlay pass: walk `hit_test_map`, for each pixel whose hit id is currently tinted OR was tinted last frame, COPY the corresponding scratch pixel into persistent_screen — XOR'd to visible RGB, forced opaque, with an optional per-channel wrap-sub of `deltas[id]` if the id is currently tinted. The hit_test_map IS the silhouette trace of each interactive element (same flood-fill identification Photon uses), and we use it strictly to bound which pixels we touch — there's no diff math, no `last_applied_delta` accumulation, just "read scratch, maybe-adjust, write screen." Every frame's overlay is independently correct against scratch's content, so nothing the persistent_screen does between frames (fade, debug toggles, anything) can corrupt the tint.
///
/// Overlay-delta flag (bit 24, above the three channel bytes): apply the per-pixel sqrt gamma lift instead of a constant wrap-sub. The low 24 bits are zero, so a consumer that predates the flag degrades to a visible no-op write rather than a wrong colour.
pub const OVERLAY_SQRT_BRIGHTEN: u32 = 0x0100_0000;

/// `SQRT_BRIGHTEN_LUT[v] = round(sqrt(v/255)·255)` — the visible-RGB gamma-lift curve for [`OVERLAY_SQRT_BRIGHTEN`]. Endpoints are fixed points (0→0, 255→255); midtones lift hard (64→128).
static SQRT_BRIGHTEN_LUT: [u8; 256] = build_sqrt_brighten_lut();
const fn build_sqrt_brighten_lut() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut v = 0usize;
    while v < 256 {
        let n = v * 255;
        let mut r = 0usize;
        while (r + 1) * (r + 1) <= n {
            r += 1;
        }
        // Round to nearest: bump when the remainder passes the (r + 0.5)² midpoint.
        t[v] = if n - r * r > r { r + 1 } else { r } as u8;
        v += 1;
    }
    t
}

/// `deltas[id] == 0` means "this id is not currently tinted." `last_active[id] == true` means "we wrote this id last frame, so this frame we still need to restore its pixels to clean scratch baseline even if the tint is now zero." After the walk, `last_active` is updated to reflect which ids had non-zero delta this frame.
///
/// Why wrap-sub when applying: chrome's hover-colour constants and `wrap_sub_rgb(THEME_HOVER, THEME_FILL)` are darkness-space deltas. The persistent screen is post-finalize visible-RGB. Since `visible = 255 − darkness` per channel, adding `delta` in darkness = subtracting `delta` in visible. So the same delta constants the bake path used apply here with a flipped operator.
///
/// `win_ox / win_oy` are the screen-space offset of the window's top-left. `hit_test_map` and `scratch` are window-space (both length `win_w × win_h`). Pixels translating outside the screen are skipped.
///
/// Force-α=0xFF on every write is safe because pixels with non-`HIT_NONE` id are always interior to the window shape (corners and the squircle AA rim are stamped `HIT_NONE`), so no clip_mask carve gets stomped.
///
/// `deltas` and `last_active` are caller-owned slices the same length, sized to `registry.next_id` (= 1 + number of registered hit zones). With the u16 hit-id widening a fixed `[T; 256]` would either truncate the ID space or balloon to 256 KB per frame; slices keep the per-frame cost proportional to live widget count. IDs past `deltas.len()` are treated as `HIT_NONE` for safety — defensive against a stale paint stamping with an unregistered ID.
///
/// Besides the constant wrap-sub deltas, a delta with [`OVERLAY_SQRT_BRIGHTEN`] set applies a per-pixel gamma lift instead: each visible channel becomes `sqrt(v/255)·255` (LUT'd). Nonlinear, so it can't be a constant delta — it brightens midtones hard while leaving black and white alone, which reads as the whole image glowing rather than a flat tint wash. Used by the chrome app-icon orb's hover.
pub fn apply_overlay<P: super::Finalize>(
    scratch: &[P],
    persistent_screen: &mut [u32],
    scr_w: usize,
    win_ox: i32,
    win_oy: i32,
    hit_test_map: &[HitId],
    win_w: usize,
    win_h: usize,
    deltas: &[u32],
    bboxes: &[Option<crate::canvas::PixelRect>],
    last_active: &mut [bool],
) -> Option<crate::canvas::PixelRect> {
    debug_assert_eq!(
        deltas.len(),
        last_active.len(),
        "deltas / last_active must be sized identically to registry.next_id"
    );
    // Quick reject: if no id is currently tinted AND none were tinted last frame, the overlay has zero work.
    let any_current = deltas.iter().any(|&d| d != 0);
    let any_last = last_active.iter().any(|&a| a);
    if !any_current && !any_last {
        return None;
    }
    // WHY/PROOF: win_w and win_h are the OS window's, 0 while minimized, and hit_test_map is the app's, resized on its own schedule after an OS resize — the scan would index past it.
    if scr_w == 0
        || win_w == 0
        || win_h == 0
        || hit_test_map.len() < win_w * win_h
        || scratch.len() < win_w * win_h
    {
        return None;
    }
    let table_len = deltas.len();
    let scr_h = persistent_screen.len() / scr_w;
    // The full on-screen window range — the fallback scan for an id with no bbox.
    // WHY/PROOF: win_ox and win_oy are the OS window's place on the screen buffer, negative or past its edge while the window is dragged partly off-screen — the window rows and columns outside the screen would index past persistent_screen.
    let full_y_lo = ((-win_oy).max(0) as usize).min(win_h);
    let full_y_hi = win_h.min(((scr_h as i32) - win_oy).max(0) as usize);
    let full_x_lo = ((-win_ox).max(0) as usize).min(win_w);
    let full_x_hi = win_w.min(((scr_w as i32) - win_ox).max(0) as usize);
    if full_y_lo >= full_y_hi || full_x_lo >= full_x_hi {
        for a in last_active.iter_mut() {
            *a = false;
        }
        return None;
    }

    // Snapshot prev → new. With u16 IDs the previous fixed `[bool; 256]` swap can't survive, but the table is bounded by registered widget count (~tens, not 65 k) so a per-frame Vec pair is cheap.
    let prev_active: alloc::vec::Vec<bool> = last_active.to_vec();
    let mut new_active = alloc::vec![false; table_len];
    // The window pixels the scans covered, for a present that uploads only what changed.
    let mut touched: Option<crate::canvas::PixelRect> = None;

    // Per-id BOUNDED scan: for each id that's tinted now OR was last frame, walk ONLY its bbox (clamped to the on-screen range; falling back to the full window if it has none), tinting the pixels where `hit_map == id`.
    // This makes hover cost O(hovered widget), not O(screen) — the whole point of the bbox table.
    // A just-left id (delta == 0, was_active) restores its pixels to clean scratch.
    for id in 0..table_len {
        let delta = deltas[id];
        let was_active = prev_active[id];
        if delta == 0 && !was_active {
            continue;
        }
        if delta != 0 {
            new_active[id] = true;
        }
        // WHY/PROOF: bboxes is the app's table (overlay_bboxes), which may be shorter than the hit registry or carry rects past the on-screen window — get() falls back to the full window, and the clamps keep the scan inside both buffers.
        let (y0, y1, x0, x1) = match bboxes.get(id).copied().flatten() {
            Some(r) => (
                r.y0.clamp(full_y_lo, full_y_hi),
                r.y1.clamp(full_y_lo, full_y_hi),
                r.x0.clamp(full_x_lo, full_x_hi),
                r.x1.clamp(full_x_lo, full_x_hi),
            ),
            None => (full_y_lo, full_y_hi, full_x_lo, full_x_hi),
        };
        if y0 < y1 && x0 < x1 {
            touched = Some(match touched {
                Some(t) => crate::canvas::PixelRect::new(t.x0.min(x0), t.y0.min(y0), t.x1.max(x1), t.y1.max(y1)),
                None => crate::canvas::PixelRect::new(x0, y0, x1, y1),
            });
        }
        let sqrt = delta & OVERLAY_SQRT_BRIGHTEN != 0;
        let dr = ((delta >> 16) & 0xFF) as u8;
        let dg = ((delta >> 8) & 0xFF) as u8;
        let db = (delta & 0xFF) as u8;
        let id_hit = id as HitId;
        for y in y0..y1 {
            let scr_y = (y as i32 + win_oy) as usize;
            let map_row = y * win_w;
            let scr_row = scr_y * scr_w;
            for x in x0..x1 {
                let map_idx = map_row + x;
                if hit_test_map[map_idx] != id_hit {
                    continue;
                }
                let raw = scratch[map_idx];
                let scr_x = (x as i32 + win_ox) as usize;
                // ROUNDED, NOT DITHERED (Nick 2026-10-29, the 16-bit canvas: "a grid of white pixels" on the focused textbox, one per 8×8): the tint wrap-subtracts a delta built from exact 8-bit theme codes, and the ordered dither floors a near-black fill one code low at one position per cell — there the subtraction wrapped to ~255. The Bayer cell at (1, 0) holds 130, half a code, so every pixel rounds to the nearest code the delta was built against.
                let visible = raw.to_sdr_visible_at(1, 0);
                let (r, g, b) = if sqrt {
                    // Per-pixel gamma lift: v' = sqrt(v/255)·255 per channel. See the doc note above.
                    (
                        SQRT_BRIGHTEN_LUT[((visible >> 16) & 0xFF) as usize] as u32,
                        SQRT_BRIGHTEN_LUT[((visible >> 8) & 0xFF) as usize] as u32,
                        SQRT_BRIGHTEN_LUT[(visible & 0xFF) as usize] as u32,
                    )
                } else {
                    (
                        (((visible >> 16) & 0xFF) as u8).wrapping_sub(dr) as u32,
                        (((visible >> 8) & 0xFF) as u8).wrapping_sub(dg) as u32,
                        ((visible & 0xFF) as u8).wrapping_sub(db) as u32,
                    )
                };
                persistent_screen[scr_row + scr_x] = 0xFF00_0000 | (r << 16) | (g << 8) | b;
            }
        }
    }
    last_active.copy_from_slice(&new_active);
    touched
}


/// Per-byte wrap-subtract of `b` from `a`'s RGB bytes, returning a packed RGB delta suitable for [`wrap_add_rgb_where`]. The α byte is zeroed in the result. Used to derive a tint delta from two theme colours at runtime (e.g. `wrap_sub_rgb(TEXTBOX_HOVER, TEXTBOX_FILL)` → the per-channel offset that, wrap-added to a `TEXTBOX_FILL` pixel, lands at `TEXTBOX_HOVER`).
#[inline]
pub fn wrap_sub_rgb(a: u32, b: u32) -> u32 {
    let ar = (a >> 16) & 0xFF;
    let ag = (a >> 8) & 0xFF;
    let ab = a & 0xFF;
    let br = (b >> 16) & 0xFF;
    let bg = (b >> 8) & 0xFF;
    let bb = b & 0xFF;
    ((ar.wrapping_sub(br) & 0xFF) << 16)
        | ((ag.wrapping_sub(bg) & 0xFF) << 8)
        | (ab.wrapping_sub(bb) & 0xFF)
}

/// Like [`wrap_sub_rgb`] but scales the SIGNED per-channel delta `(target − base)` by `num/den` before wrapping — a fractional-strength tint.
/// The overlay's visible-space wrap-sub of the result lands a base-coloured pixel `num/den` of the way toward `target` (e.g. `(1, 4)` = quarter-opacity hover), so a vivid target hue reads as a gentle wash instead of a full-saturation flood.
/// α byte is dropped (0).
pub fn wrap_sub_rgb_scaled(target: u32, base: u32, num: i32, den: i32) -> u32 {
    let mut out = 0u32;
    for shift in [16u32, 8, 0] {
        let t = ((target >> shift) & 0xFF) as i32;
        let b = ((base >> shift) & 0xFF) as i32;
        let d = (t - b) * num / den;
        out |= ((d.rem_euclid(256)) as u32) << shift;
    }
    out
}

/// Unpack a fluor internal pixel into `(r, g, b, a)` with visible RGB and opacity α — the inverse of [`pack_argb`]. Flips darkness back to visible RGB; α passes thru.
#[inline]
pub fn unpack_argb(packed: u32) -> (u8, u8, u8, u8) {
    let a = (packed >> 24) as u8;
    let r = 255 - ((packed >> 16) as u8);
    let g = 255 - ((packed >> 8) as u8);
    let b = 255 - (packed as u8);
    (r, g, b, a)
}

/// Scale a fluor internal pixel's VISIBLE brightness by `num/den`, each channel saturating at 255 thru the float-to-u8 cast, preserving α. Derives a lighter/darker variant of a fill in the same hue — e.g. a custom-coloured button's hover/held fills off its resting fill, so any tint gets the same luminance ramp the `BUTTON_FILL/HOVER/HELD` constants encode (dark idle → ~1.5× hover → ~2.2× held) instead of a hand-picked pair per colour. `(num, den) = (1, 1)` is identity.
#[inline]
pub fn scale_brightness(packed: u32, num: u16, den: u16) -> u32 {
    let (r, g, b, a) = unpack_argb(packed);
    let s = |c: u8| (c as f64 * num as f64 / den as f64) as u8;
    pack_argb(s(r), s(g), s(b), a)
}


/// Stamp a 1-pixel magenta hairline around `bbox` directly into the screen-sized, visible-RGB back buffer. `bbox` is window-local; `offset_x` / `offset_y` are the window's top-left in screen space. Pure overwrite — no blending, no `under` path, no damage tracking. Caller invokes between `copy_from_slice(&persistent_screen)` and `buffer.present()` so the magenta lives for exactly one frame.
pub fn stamp_damage_outline_visible(
    buf: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    bbox: crate::canvas::PixelRect,
    offset_x: i32,
    offset_y: i32,
) {
    const MAGENTA: u32 = 0xFFFF_00FF;
    if bbox.is_empty() || scr_w == 0 || scr_h == 0 {
        return;
    }
    let sx0_i = offset_x + bbox.x0 as i32;
    let sy0_i = offset_y + bbox.y0 as i32;
    let sx1_i = offset_x + bbox.x1 as i32;
    let sy1_i = offset_y + bbox.y1 as i32;
    // WHY/PROOF: offset_x and offset_y are the OS window's place on the screen, negative or past its edge while the window hangs off-screen — the outline would index outside the back buffer.
    let x0 = sx0_i.max(0) as usize;
    let y0 = sy0_i.max(0) as usize;
    let x1 = (sx1_i.max(0) as usize).min(scr_w);
    let y1 = (sy1_i.max(0) as usize).min(scr_h);
    if x0 >= x1 || y0 >= y1 {
        return;
    }

    // Top row.
    let top_row = y0 * scr_w;
    for px in x0..x1 {
        buf[top_row + px] = MAGENTA;
    }
    // Bottom row — only paint if distinct from the top (i.e., rect is ≥ 2 tall); otherwise the top row already covered it.
    if y1 - 1 > y0 {
        let bot_row = (y1 - 1) * scr_w;
        for px in x0..x1 {
            buf[bot_row + px] = MAGENTA;
        }
    }
    // Left and right columns (interior rows only — corners are claimed by the top/bottom rows).
    let interior_y0 = y0 + 1;
    let interior_y1 = y1 - 1;
    if interior_y0 < interior_y1 {
        let right_col = x1 - 1;
        for py in interior_y0..interior_y1 {
            let row = py * scr_w;
            buf[row + x0] = MAGENTA;
            if right_col > x0 {
                buf[row + right_col] = MAGENTA;
            }
        }
    }
}


/// Glyph rasterizers for window controls. Ported verbatim from photon's [compositing.rs](/mnt/Harbor/Code/photon/src/ui/compositing.rs) — the squircle minus / squircle ring / capsule X — so chrome looks identical to photon.
pub mod glyph {
    /// Draw a horizontal squircle "minus" stroke centered at `(x, y)` inside a button of pixel radius `r`. Uses a 4-power squircle with widened axis to make a flat horizontal pill.
    pub fn minimize_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
    ) {
        let r = r + 1;
        let r_render = r / 4 + 1;
        let r_2 = r_render * r_render;
        let r_4 = r_2 * r_2;
        let r_3 = r_render * r_render * r_render;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;

        for h in -(r_render as isize)..=(r_render as isize) {
            for w in -(r as isize)..=(r as isize) {
                let h2 = h * h;
                let h4 = h2 * h2;
                let a = (w.abs() - (r * 3 / 4) as isize).max(0);
                let w2 = a * a;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 <= r_4 {
                    let px = (x as isize + w) as usize;
                    let py = (y as isize + h + (r / 2) as isize) as usize;
                    let idx = py * width + px;
                    let gradient = ((r_4 - dist_4) << 8) / (r_3 << 2);
                    if gradient > 255 {
                        pixels[idx] = stroke_packed;
                    } else {
                        pixels[idx] = blend_swar(pixels[idx], stroke_packed, gradient as u64);
                    }
                }
            }
        }
    }

    /// Draw a square "maximize" symbol — squircle ring with stroke + interior fill — centered at `(x, y)` with pixel radius `r`.
    pub fn maximize_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
        fill_colour: u32,
    ) {
        let r = r + 1;
        let mut r_4 = r * r;
        r_4 *= r_4;
        let r_3 = r * r * r;

        let r_inner = r * 4 / 5;
        let mut r_inner_4 = r_inner * r_inner;
        r_inner_4 *= r_inner_4;
        let r_inner_3 = r_inner * r_inner * r_inner;

        let outer_edge_threshold = r_3 << 2;
        let inner_edge_threshold = r_inner_3 << 2;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;
        let fill_packed = fill_colour & 0x00FF_FFFF;

        for h in -(r as isize)..=r as isize {
            for w in -(r as isize)..=r as isize {
                let h2 = h * h;
                let h4 = h2 * h2;
                let w2 = w * w;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 > r_4 {
                    continue;
                }
                let px = (x as isize + w) as usize;
                let py = (y as isize + h) as usize;
                let idx = py * width + px;

                let dist_from_outer = r_4 - dist_4;
                if dist_4 <= r_inner_4 {
                    let dist_from_inner = r_inner_4 - dist_4;
                    if dist_from_inner <= inner_edge_threshold {
                        let gradient = (dist_from_inner << 8) / inner_edge_threshold;
                        pixels[idx] = blend_swar(stroke_packed, fill_packed, gradient as u64);
                    } else {
                        pixels[idx] = fill_packed;
                    }
                } else if dist_from_outer <= outer_edge_threshold {
                    let gradient = (dist_from_outer << 8) / outer_edge_threshold;
                    pixels[idx] = blend_swar(pixels[idx], stroke_packed, gradient as u64);
                } else {
                    pixels[idx] = stroke_packed;
                }
            }
        }
    }

    /// Draw an antialiased "X" close symbol — two crossed capsule lines — centered at `(x, y)` with pixel radius `r`.
    pub fn close_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
    ) {
        let r = r + 1;
        let thickness = (r / 3).max(1) as f32;
        let radius = thickness / 2.0;
        let size = (r * 2) as f32;
        let cxf = x as f32;
        let cyf = y as f32;
        let end = size / 3.0;

        let x1_start = cxf - end;
        let y1_start = cyf - end;
        let x1_end = cxf + end;
        let y1_end = cyf + end;
        let x2_start = cxf + end;
        let y2_start = cyf - end;
        let x2_end = cxf - end;
        let y2_end = cyf + end;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;
        let cxi = x as i32;
        let cyi = y as i32;
        let height = (pixels.len() / width) as i32;
        // WHY/PROOF: the button sits wherever the chrome strip lands on an OS-sized viewport, which on a tiny window pushes it past an edge — the quadrant loops would index outside the pixel buffer.
        let min_x = ((x as i32) - (r as i32)).max(0);
        let max_x = ((x as i32) + (r as i32)).min(width as i32);
        let min_y = ((y as i32) - (r as i32)).max(0);
        let max_y = ((y as i32) + (r as i32)).min(height);

        // Each quadrant samples one of the two diagonals (whichever passes thru it).
        let quadrants: [(i32, i32, i32, i32, f32, f32, f32, f32); 4] = [
            (min_x, cxi, min_y, cyi, x1_start, y1_start, x1_end, y1_end), // top-left, diag1
            (cxi, max_x, min_y, cyi, x2_start, y2_start, x2_end, y2_end), // top-right, diag2
            (min_x, cxi, cyi, max_y, x2_start, y2_start, x2_end, y2_end), // bottom-left, diag2
            (cxi, max_x, cyi, max_y, x1_start, y1_start, x1_end, y1_end), // bottom-right, diag1
        ];
        for (qx0, qx1, qy0, qy1, x0, y0, x1, y1) in quadrants {
            for py in qy0..qy1 {
                for px in qx0..qx1 {
                    let dist = distance_to_capsule(
                        px as f32 + 0.5,
                        py as f32 + 0.5,
                        x0,
                        y0,
                        x1,
                        y1,
                        radius,
                    );
                    let alpha_f = if dist < -0.5 {
                        1.0
                    } else if dist < 0.5 {
                        0.5 - dist
                    } else {
                        0.0
                    };
                    if alpha_f > 0.0 {
                        let idx = py as usize * width + px as usize;
                        let alpha = (alpha_f * 256.0) as u64;
                        pixels[idx] = blend_swar(pixels[idx], stroke_packed, alpha);
                    }
                }
            }
        }
    }

    /// Distance from a point to a capsule (line segment + radius). Negative inside the capsule, positive outside, used as an SDF for AA.
    #[inline]
    fn distance_to_capsule(
        px: f32,
        py: f32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        radius: f32,
    ) -> f32 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;
        let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
        // Not a guard: clamping t to the segment IS the capsule's geometry (closest point on a segment, round caps at both ends) — a smooth stand-in would bend the stroke's ends.
        let t_clamped = t.clamp(0.0, 1.0);
        let cx = x1 + t_clamped * dx;
        let cy = y1 + t_clamped * dy;
        let ex = px - cx;
        let ey = py - cy;
        crate::math::sqrt(ex * ex + ey * ey) - radius
    }

    /// Pre-multiplied SWAR blend of `fg` over `bg` with explicit `alpha` (0..=256). Photon's exact pattern: widen each 32-bit pixel to 64 bits with each channel in its own 16-bit slot, do `bg*(256-α) + fg*α` in parallel, narrow back.
    #[inline]
    fn blend_swar(bg: u32, fg: u32, alpha: u64) -> u32 {
        let inv = 256 - alpha;
        let mut bg64 = bg as u64;
        bg64 = (bg64 | (bg64 << 16)) & 0x0000_FFFF_0000_FFFF;
        bg64 = (bg64 | (bg64 << 8)) & 0x00FF_00FF_00FF_00FF;
        let mut fg64 = fg as u64;
        fg64 = (fg64 | (fg64 << 16)) & 0x0000_FFFF_0000_FFFF;
        fg64 = (fg64 | (fg64 << 8)) & 0x00FF_00FF_00FF_00FF;
        let mut blended = bg64 * inv + fg64 * alpha;
        blended = (blended >> 8) & 0x00FF_00FF_00FF_00FF;
        blended = (blended | (blended >> 8)) & 0x0000_FFFF_0000_FFFF;
        blended = blended | (blended >> 16);
        blended as u32
    }
}


/// Photon's `blend_rgb_only` helper: weighted RGB blend of two colours with explicit per-pixel weights. Verbatim port from [compositing.rs:5821](/mnt/Harbor/Code/photon/src/ui/compositing.rs#L5821). Used by `draw_window_controls` for AA squircle edges.
pub fn blend_rgb_only(bg_colour: u32, fg_colour: u32, weight_bg: u8, weight_fg: u8) -> u32 {
    let mut bg = bg_colour as u64;
    bg = (bg | (bg << 16)) & 0x0000FFFF0000FFFF;
    bg = (bg | (bg << 8)) & 0x00FF00FF00FF00FF;

    let mut fg = fg_colour as u64;
    fg = (fg | (fg << 16)) & 0x0000FFFF0000FFFF;
    fg = (fg | (fg << 8)) & 0x00FF00FF00FF00FF;

    let mut blended = bg * weight_bg as u64 + fg * weight_fg as u64;
    blended = (blended >> 8) & 0x00FF00FF00FF00FF;
    blended = (blended | (blended >> 8)) & 0x0000FFFF0000FFFF;
    blended = (blended | (blended >> 16)) & 0x00FFFFFF;
    // α + darkness: force opaque (α=0xFF) by setting the top byte. RGB darkness is whatever the blend produced.
    (blended as u32) | 0xFF000000
}

