//! Single-line text-entry widget. Pill-shaped with AA edges, wave-animated blinkey cursor, text scrolling, focus glow, and selection.
//!
//! Patterns lifted from photon's text_editing.rs + compositing.rs — `chars + widths + cursor` model, pill shape via squircle crossings, wave blinkey with alternating top/bottom brightness, symmetric scroll margins, 4-directional glow blur, XOR selection inversion.

use crate::canvas::PixelRect;
use crate::coord::Coord;
use crate::paint::{self, Clip};
use crate::region::Region;
use crate::text::TextRenderer;
use crate::theme;
use alloc::string::String;
use alloc::vec::Vec;

pub struct Textbox {
    /// Text content as a `Vec<char>` — character-indexed cursor + width arrays.
    pub chars: Vec<char>,
    /// Insertion point in `[0, chars.len()]`.
    pub cursor: usize,
    pub focused: bool,
    /// Cursor is hovering over the textbox bbox. Drives the hover fill colour.
    pub hovered: bool,
    /// Stroke thickness in RU (multiplied by `font_size`). `0.0` → 1px minimum via the photon `+ 1`
    /// idiom in `render_content_into`. Stroke eats inward from the outer pill silhouette.
    pub stroke_ru: f32,
    /// Pixel rect (center-anchored).
    pub center_x: Coord,
    pub center_y: Coord,
    pub width: Coord,
    pub height: Coord,
    /// Font size in pixels.
    pub font_size: Coord,
    /// Per-char pixel widths cached after the last edit.
    widths: Vec<Coord>,
    font: &'static str,

    // --- Scroll ---
    /// Horizontal scroll offset in pixels. Positive = text shifted right (cursor near left edge).
    pub scroll_offset: Coord,

    // --- Blinkey ---
    /// Whether the blinkey is currently drawn (visible half of blink cycle).
    pub blinkey_visible: bool,
    /// Which wave variant is drawn: true = top-bright, false = bottom-bright.
    pub blinkey_wave_top: bool,

    // --- Selection ---
    /// If Some, the anchor index where the selection started (shift+click or shift+arrow).
    pub selection_anchor: Option<usize>,

    // --- Photon-style differential cache ---
    /// Persistent painted pill (α + darkness, post-composed from-empty). Bbox-sized. Squircle re-rasterizes only when [`cache_dirty`] is set (geometry / zoom change); hover / focus state shifts mutate this buffer in place via wrap-add/sub of a tint delta.
    cache: Vec<u32>,
    cache_w: usize,
    cache_h: usize,
    /// Viewport-space top-left where the cache should blit. Recomputed in `render_content_into` from `(center_x − width/2, center_y − height/2)` after offset.
    cache_origin_x: isize,
    cache_origin_y: isize,
    /// `true` → squircle needs full re-rasterize on the next render. Set by `set_rect` / `set_font_size`. Cleared at end of cache rasterize. The cache stores the BASE-color (TEXTBOX_FILL) squircle; hover / focus tints are applied in-flight during blit, never baked into the cache.
    cache_dirty: bool,

    // --- Blinkey screen-overlay state (lives on the host's persistent_screen buffer post-finalize) ---
    /// Was the blinkey wave painted onto persistent_screen last frame? Used by [`paint_blinkey_into_screen`] to decide whether to wrap-subtract the prior wave before wrap-adding the new one.
    last_painted_blinkey_on: bool,
    /// Screen-space position of the prior frame's blinkey (left of the ±7 spread, top of the wave band). `None` until the first paint.
    last_painted_blinkey_screen_x: i32,
    last_painted_blinkey_screen_y: i32,
    /// Wave variant baked into persistent_screen last frame (top-bright vs bottom-bright). The unbake must use the same polynomial that the bake used.
    last_painted_blinkey_wave_top: bool,
    /// Height in pixels of the wave baked last frame.
    last_painted_blinkey_height: i32,

    // --- Damage protocol (persistent-scratch differential rendering) ---
    /// Where the textbox painted into target last frame (in target pixel coords, clipped to viewport). `None` = no prior paint (first frame or just resized). Used by [`Textbox::damage_rect`] to union with the current bbox so moves are covered.
    last_painted_bbox: Option<PixelRect>,
    /// `focused` value at the time of the last paint — tracks whether the glow was painted last frame, so [`Textbox::damage_rect`] can expand to `glow_bbox` on focus on/off transitions.
    last_painted_focused: bool,
    /// `hovered` value at the time of the last paint — diff against `self.hovered` tells `damage_rect` to flag hover-only changes (`bbox`, not `glow_bbox`).
    last_painted_hovered: bool,
}

/// Photon's blinkey wave polynomial, wrap-added (or wrap-subtracted) per-channel into a visible-RGB screen buffer. The wave is `±7` pixels of horizontal spread centered on `bx`, `height` pixels tall vertically with intensity falling off via `wave >> |dx|`. `top_bright` picks the upper-half-bright vs lower-half-bright variant.
///
/// Operates in visible-RGB space (post-finalize boundary) — adding to each channel brightens the displayed color directly. Wrap arithmetic: `wrapping_add` of value `x` is exactly reversed by `wrapping_sub` of the same `x`, regardless of starting byte, so painting on then off restores the underlying pixel bit-for-bit. Out-of-screen pixels are skipped (no panic on edge cases).
fn wrap_blinkey_into_screen(
    screen: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    bx: i32,
    by: i32,
    height: i32,
    top_bright: bool,
    subtract: bool,
) {
    if height <= 0 || scr_w == 0 || scr_h == 0 {
        return;
    }
    let half = (height / 2) as f32;
    if half == 0.0 {
        return;
    }
    let cursor_brightness = crate::theme::CURSOR_BRIGHTNESS;
    for dy in 0..height {
        let py = by + dy;
        if py < 0 || (py as usize) >= scr_h {
            continue;
        }
        let row_base = (py as usize) * scr_w;
        let t = ((dy as f32) - half) / half;
        let wave = if top_bright {
            (1.0 - t * t) * (1.0 - t) * (1.0 - t) * cursor_brightness
        } else {
            (1.0 - t * t) * (1.0 + t) * (1.0 + t) * cursor_brightness
        };
        let w_base = wave as u32;
        for dx in -7i32..=7i32 {
            let px = bx + dx;
            if px < 0 || (px as usize) >= scr_w {
                continue;
            }
            let w = w_base >> (dx.unsigned_abs());
            if w == 0 {
                continue;
            }
            let pixel_delta = 0x0001_0101u32 * w;
            let idx = row_base + (px as usize);
            screen[idx] = if subtract {
                screen[idx].wrapping_sub(pixel_delta)
            } else {
                screen[idx].wrapping_add(pixel_delta)
            };
        }
    }
}

/// Convert a viewport-coord `Region` (Coord f32 rectangle, possibly negative or off-buffer) into a `PixelRect` (usize half-open rect) clamped to the viewport bounds. Used by [`Textbox::damage_rect`] / [`Textbox::record_painted`] to express widget bboxes in the host's pixel-rect language for damage union.
fn region_to_pixelrect(region: Region, viewport_w: usize, viewport_h: usize) -> PixelRect {
    let vw = viewport_w as f32;
    let vh = viewport_h as f32;
    let x0 = region.x.max(0.0).min(vw) as usize;
    let y0 = region.y.max(0.0).min(vh) as usize;
    let x1 = (region.x + region.w).max(0.0).min(vw) as usize;
    let y1 = (region.y + region.h).max(0.0).min(vh) as usize;
    PixelRect::new(x0, y0, x1, y1)
}

/// Blit the cache (pre-composed α + darkness, holding the BASE-color squircle — no tint baked in) onto `canvas` at `(origin_x, origin_y)`. The hover / focus tint `delta_rgb` is wrap-added IN-FLIGHT to each opaque source pixel during the copy, never written back into the cache — so the cache stays in its base state forever and can be reused across any tint without unbake/rebake cycles. AA-edge pixels (α<255) skip the tint to keep the silhouette transition clean.
///
/// Composition uses the `flatten_premult` formula (src is pre-attenuated, so we scale by `(256 − dst.α)` only — not by `(256 − dst.α) × src.α`).
///
/// Hit stamp: writes `hit_id` to `hit_map` at every opaque cache pixel during the same walk.
fn blit_cache_to_target(
    cache: &[u32],
    cache_w: usize,
    cache_h: usize,
    origin_x: isize,
    origin_y: isize,
    canvas: &mut crate::canvas::Canvas,
    hit_map: Option<&mut [u8]>,
    hit_id: u8,
    delta_rgb: u32,
    clip: Option<paint::Clip>,
) {
    let target_w = canvas.width;
    let target_h = canvas.height;
    if cache_w == 0 || cache_h == 0 {
        return;
    }
    let target_w_i = target_w as isize;
    let target_h_i = target_h as isize;
    let clip_rect = paint::Clip::resolve(clip, target_w, target_h);

    // Damage = blit region ∩ target bounds ∩ clip.
    let dx0 = (origin_x.max(0).min(target_w_i) as usize).max(clip_rect.x_start);
    let dy0 = (origin_y.max(0).min(target_h_i) as usize).max(clip_rect.y_start);
    let dx1 = ((origin_x + cache_w as isize).max(0).min(target_w_i) as usize).min(clip_rect.x_end);
    let dy1 = ((origin_y + cache_h as isize).max(0).min(target_h_i) as usize).min(clip_rect.y_end);
    if dx0 >= dx1 || dy0 >= dy1 {
        return;
    }
    canvas.damage.add_bounds(dx0, dy0, dx1, dy1);

    let target = &mut *canvas.pixels;
    let mut hit_map = hit_map;
    let tint_r = ((delta_rgb >> 16) & 0xFF) as u8;
    let tint_g = ((delta_rgb >> 8) & 0xFF) as u8;
    let tint_b = (delta_rgb & 0xFF) as u8;
    for cy in 0..cache_h {
        let ty = origin_y + cy as isize;
        if ty < 0 || ty >= target_h_i {
            continue;
        }
        let ty = ty as usize;
        if ty < clip_rect.y_start || ty >= clip_rect.y_end {
            continue;
        }
        let cache_row = cy * cache_w;
        let target_row = ty * target_w;
        for cx in 0..cache_w {
            let tx = origin_x + cx as isize;
            if tx < 0 || tx >= target_w_i {
                continue;
            }
            let tx = tx as usize;
            if tx < clip_rect.x_start || tx >= clip_rect.x_end {
                continue;
            }
            let cache_idx = cache_row + cx;
            let target_idx = target_row + tx;

            let d = target[target_idx];
            let s = cache[cache_idx];
            let s_a = (s >> 24) & 0xFF;
            // Wrap-add the tint into opaque cache pixels in-flight — the cache itself is never mutated, so the base FILL squircle is reusable forever across any tint state.
            let (eff_r, eff_g, eff_b) = if s_a == 0xFF && delta_rgb != 0 {
                (
                    (((s >> 16) & 0xFF) as u8).wrapping_add(tint_r) as u32,
                    (((s >> 8) & 0xFF) as u8).wrapping_add(tint_g) as u32,
                    ((s & 0xFF) as u8).wrapping_add(tint_b) as u32,
                )
            } else {
                ((s >> 16) & 0xFF, (s >> 8) & 0xFF, s & 0xFF)
            };
            // flatten_premult: scale src contribution by (256 - dst.α) only — src.dark already premultiplied.
            if d < 0xFF000000 {
                let d_a = d >> 24;
                let factor = 256 - d_a;
                let d_r = (d >> 16) & 0xFF;
                let d_g = (d >> 8) & 0xFF;
                let d_b = d & 0xFF;
                let new_a = d_a + ((factor * s_a) >> 8);
                let new_r = d_r + ((factor * eff_r) >> 8);
                let new_g = d_g + ((factor * eff_g) >> 8);
                let new_b = d_b + ((factor * eff_b) >> 8);
                target[target_idx] = (new_a << 24) | (new_r << 16) | (new_g << 8) | new_b;
            }
            if s_a == 0xFF {
                if let Some(hm) = hit_map.as_deref_mut() {
                    hm[target_idx] = hit_id;
                }
            }
        }
    }
}

impl Textbox {
    pub fn new(
        center_x: Coord,
        center_y: Coord,
        width: Coord,
        height: Coord,
        font_size: Coord,
    ) -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
            focused: false,
            hovered: false,
            stroke_ru: 0.0, // → 1 px minimum stroke via the +1 idiom in render_content_into
            center_x,
            center_y,
            width,
            height,
            font_size,
            widths: Vec::new(),
            font: "Open Sans",
            scroll_offset: 0.0,
            blinkey_visible: false,
            blinkey_wave_top: true,
            selection_anchor: None,
            cache: Vec::new(),
            cache_w: 0,
            cache_h: 0,
            cache_origin_x: 0,
            cache_origin_y: 0,
            cache_dirty: true,
            last_painted_bbox: None,
            last_painted_focused: false,
            last_painted_hovered: false,
            last_painted_blinkey_on: false,
            last_painted_blinkey_screen_x: 0,
            last_painted_blinkey_screen_y: 0,
            last_painted_blinkey_wave_top: true,
            last_painted_blinkey_height: 0,
        }
    }

    /// Paint the blinkey wave directly into the host's `persistent_screen` buffer (visible-RGB, post-finalize, post-shadow). Wrap-subtracts the prior frame's wave (if any) before wrap-adding the new one — `wrapping_sub` of a `wrapping_add` returns the original byte exactly, so the operation is reversible regardless of the underlying pixel value. The persistent_screen survives across frames, so the only work per blink tick is the diff: a couple hundred bytes of wrap-add + wrap-sub.
    ///
    /// `(window_origin_x, window_origin_y)` translates viewport-space cursor coords into screen-space.
    ///
    /// Mirrors [`paint::draw_blinkey`]'s polynomial wave + `0x00010101 × w` per-channel write but in visible-RGB space (the persistent_screen is past the darkness → visible XOR boundary), so wrap-add brightens directly.
    pub fn paint_blinkey_into_screen(
        &mut self,
        screen: &mut [u32],
        scr_w: usize,
        scr_h: usize,
        window_origin_x: i32,
        window_origin_y: i32,
    ) {
        let want_on = self.focused && !self.has_selection() && self.blinkey_visible;

        // Compute current target position in screen-space.
        let cursor_view_x = self.cursor_pixel_x() as i32;
        let cursor_view_y = (self.center_y - self.font_size * 0.5) as i32;
        let cur_x = cursor_view_x + window_origin_x;
        let cur_y = cursor_view_y + window_origin_y;
        let cur_h = self.font_size as i32;
        let cur_wave_top = self.blinkey_wave_top;

        // Unbake the prior frame's wave (if it was on) — wrap-sub at last position with last wave params.
        if self.last_painted_blinkey_on {
            wrap_blinkey_into_screen(
                screen,
                scr_w,
                scr_h,
                self.last_painted_blinkey_screen_x,
                self.last_painted_blinkey_screen_y,
                self.last_painted_blinkey_height,
                self.last_painted_blinkey_wave_top,
                /*subtract=*/ true,
            );
        }

        // Bake the current frame's wave (if it should be on) at the new position.
        if want_on {
            wrap_blinkey_into_screen(
                screen,
                scr_w,
                scr_h,
                cur_x,
                cur_y,
                cur_h,
                cur_wave_top,
                /*subtract=*/ false,
            );
        }

        // Record current state for next frame's unbake.
        self.last_painted_blinkey_on = want_on;
        self.last_painted_blinkey_screen_x = cur_x;
        self.last_painted_blinkey_screen_y = cur_y;
        self.last_painted_blinkey_wave_top = cur_wave_top;
        self.last_painted_blinkey_height = cur_h;
    }

    /// Damage region this widget contributes to the host's per-frame clip rect.
    ///
    /// Returns `None` if nothing changed since the last paint (no rasterize, no blit needed — host can persist the previous frame's pixels in scratch).
    ///
    /// Returns `Some(rect)` if any state change requires repaint. The rect is the union of:
    ///   - `last_painted_bbox` (where the textbox was last frame — must be cleared if anything moves or content changes)
    ///   - the current bbox (where it'll paint this frame)
    ///
    /// Picks the right bbox flavor: `glow_bbox` if glow is currently OR was previously painted; `bbox` for the bare-pill steady state. Hover-only changes use bare bbox; geometry / focus changes use the wider glow bbox.
    pub fn damage_rect(&self, viewport_w: usize, viewport_h: usize) -> Option<PixelRect> {
        let focus_changed = self.focused != self.last_painted_focused;
        let hover_changed = self.hovered != self.last_painted_hovered;
        let dirty = self.cache_dirty || focus_changed || hover_changed;
        if !dirty && self.last_painted_bbox.is_some() {
            return None;
        }
        let need_glow = self.focused || self.last_painted_focused;
        let current_region = if need_glow { self.glow_bbox() } else { self.bbox() };
        let current_rect = region_to_pixelrect(current_region, viewport_w, viewport_h);
        let combined = match self.last_painted_bbox {
            Some(prev) => prev.union(current_rect),
            None => current_rect,
        };
        if combined.is_empty() {
            None
        } else {
            Some(combined)
        }
    }

    /// Record the bbox we just painted into and the focus/hover state that drove it — called at the tail of [`render_content_into`] so the next frame's [`damage_rect`] knows what to union with.
    fn record_painted(&mut self, viewport_w: usize, viewport_h: usize) {
        let need_glow = self.focused;
        let region = if need_glow { self.glow_bbox() } else { self.bbox() };
        let rect = region_to_pixelrect(region, viewport_w, viewport_h);
        self.last_painted_bbox = if rect.is_empty() { None } else { Some(rect) };
        self.last_painted_focused = self.focused;
        self.last_painted_hovered = self.hovered;
    }

    /// Force a full cache rasterize on the next `render_content_into`. Call after any geometry/zoom change that affects the squircle shape.
    pub fn invalidate_cache(&mut self) {
        self.cache_dirty = true;
    }

    /// True if pixel `(x, y)` is inside the textbox's bare bbox rect (`width × height`, NO glow padding). This is a rectangular check, NOT shape-aware — square bbox corners outside the squircle return `true`. For pill-silhouette-accurate hit testing in a chrome-integrated app, prefer `chrome.hit_at(x, y) == HIT_TEXTBOX` (chrome's hit_test_map is stamped with the actual pill shape by the textbox stroke pass). This method is the fallback for chrome-less consumers and for internal click routing where coarse bbox accuracy is acceptable.
    pub fn contains(&self, x: Coord, y: Coord) -> bool {
        let half_w = self.width * 0.5;
        let half_h = self.height * 0.5;
        x >= self.center_x - half_w
            && x < self.center_x + half_w
            && y >= self.center_y - half_h
            && y < self.center_y + half_h
    }

    pub fn set_rect(&mut self, center_x: Coord, center_y: Coord, width: Coord, height: Coord) {
        if self.center_x != center_x || self.center_y != center_y || self.width != width || self.height != height {
            self.cache_dirty = true;
        }
        self.center_x = center_x;
        self.center_y = center_y;
        self.width = width;
        self.height = height;
    }

    pub fn set_font_size(&mut self, font_size: Coord, text: &mut TextRenderer) {
        if self.font_size != font_size {
            self.cache_dirty = true;
        }
        self.font_size = font_size;
        self.recalc_widths(text);
    }

    fn recalc_widths(&mut self, text: &mut TextRenderer) {
        self.widths.clear();
        self.widths.reserve(self.chars.len());
        let mut buf = [0u8; 4];
        for ch in &self.chars {
            let s = ch.encode_utf8(&mut buf);
            let w = text.measure_text_width(s, self.font_size, 400, self.font);
            self.widths.push(w);
        }
    }

    /// Total text width in pixels.
    fn text_width(&self) -> Coord {
        self.widths.iter().sum()
    }

    /// Pixel x of the leftmost glyph baseline (left inset of pill shape).
    fn text_left(&self) -> Coord {
        self.center_x - self.width * 0.5 + self.padding()
    }

    fn text_right(&self) -> Coord {
        self.center_x + self.width * 0.5 - self.padding()
    }

    fn padding(&self) -> Coord {
        self.font_size * 0.4
    }

    /// Usable text area width (pill interior minus padding on both sides).
    fn usable_width(&self) -> Coord {
        self.width - self.padding() * 2.0
    }

    // --- Editing ---

    pub fn insert_char(&mut self, c: char, text: &mut TextRenderer) {
        if c.is_control() {
            return;
        }
        self.delete_selection(text);
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
        self.recalc_widths(text);
        self.update_scroll();
    }

    pub fn backspace(&mut self, text: &mut TextRenderer) {
        if self.has_selection() {
            self.delete_selection(text);
            return;
        }
        if self.cursor == 0 {
            return;
        }
        self.cursor -= 1;
        self.chars.remove(self.cursor);
        self.recalc_widths(text);
        self.update_scroll();
    }

    pub fn delete_forward(&mut self, text: &mut TextRenderer) {
        if self.has_selection() {
            self.delete_selection(text);
            return;
        }
        if self.cursor >= self.chars.len() {
            return;
        }
        self.chars.remove(self.cursor);
        self.recalc_widths(text);
        self.update_scroll();
    }

    pub fn cursor_left(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.update_scroll();
        }
    }
    pub fn cursor_right(&mut self) {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
            self.update_scroll();
        }
    }
    pub fn cursor_home(&mut self) {
        self.cursor = 0;
        self.update_scroll();
    }
    pub fn cursor_end(&mut self) {
        self.cursor = self.chars.len();
        self.update_scroll();
    }

    // --- Selection ---

    pub fn has_selection(&self) -> bool {
        self.selection_anchor.map_or(false, |a| a != self.cursor)
    }

    /// Get sorted (start, end) of the selection range.
    pub fn selection_range(&self) -> Option<(usize, usize)> {
        self.selection_anchor.and_then(|a| {
            if a == self.cursor {
                None
            } else {
                Some((a.min(self.cursor), a.max(self.cursor)))
            }
        })
    }

    pub fn select_all(&mut self) {
        self.selection_anchor = Some(0);
        self.cursor = self.chars.len();
    }

    pub fn delete_selection(&mut self, text: &mut TextRenderer) {
        if let Some((start, end)) = self.selection_range() {
            self.chars.drain(start..end);
            self.cursor = start;
            self.selection_anchor = None;
            self.recalc_widths(text);
            self.update_scroll();
        }
    }

    /// Get selected text as a String.
    pub fn selected_text(&self) -> Option<String> {
        self.selection_range()
            .map(|(s, e)| self.chars[s..e].iter().collect())
    }

    /// Replace selection (or insert at cursor) with string.
    pub fn insert_str(&mut self, s: &str, text: &mut TextRenderer) {
        self.delete_selection(text);
        for c in s.chars() {
            if !c.is_control() {
                self.chars.insert(self.cursor, c);
                self.cursor += 1;
            }
        }
        self.recalc_widths(text);
        self.update_scroll();
    }

    // --- Scrolling ---

    /// Update scroll offset to keep the cursor visible within symmetric margins.
    fn update_scroll(&mut self) {
        let tw = self.text_width();
        let uw = self.usable_width();
        if tw <= uw {
            self.scroll_offset = 0.0;
            return;
        }
        let margin = uw * 0.05;
        let usable_half = uw * 0.5;
        let text_half = tw * 0.5;

        let max_scroll_right = usable_half - margin - text_half;
        let max_scroll_left = text_half - usable_half + margin;
        self.scroll_offset = self.scroll_offset.clamp(max_scroll_right, max_scroll_left);

        let cursor_px: Coord = self.widths[..self.cursor].iter().sum();
        let cursor_in_centered = cursor_px - text_half;
        let cursor_in_view = cursor_in_centered + self.scroll_offset;
        if cursor_in_view < -usable_half + margin {
            self.scroll_offset = -usable_half + margin - cursor_in_centered;
        } else if cursor_in_view > usable_half - margin {
            self.scroll_offset = usable_half - margin - cursor_in_centered;
        }
    }

    // --- Click ---

    pub fn cursor_index_from_x(&self, click_x: Coord) -> usize {
        let text_start = self.text_start_x();
        if click_x <= text_start {
            return 0;
        }
        let mut accum = text_start;
        for (i, &w) in self.widths.iter().enumerate() {
            let mid = accum + w * 0.5;
            if click_x < mid {
                return i;
            }
            accum += w;
        }
        self.chars.len()
    }

    pub fn handle_click(&mut self, x: Coord, y: Coord) {
        if self.contains(x, y) {
            self.focused = true;
            self.cursor = self.cursor_index_from_x(x);
            self.selection_anchor = None;
            self.blinkey_visible = true;
            self.blinkey_wave_top = true;
        } else {
            self.focused = false;
            self.blinkey_visible = false;
            self.selection_anchor = None;
        }
    }

    // --- Blinkey ---

    /// Flip the blinkey wave between top-bright and bottom-bright. Returns true if the blinkey is visible (caller should redraw).
    pub fn flip_blinkey(&mut self) -> bool {
        if !self.focused {
            return false;
        }
        self.blinkey_wave_top = !self.blinkey_wave_top;
        self.blinkey_visible = true;
        true
    }

    // --- Layout helpers ---

    /// Compute the pixel X where text rendering starts, accounting for centering and scroll.
    fn text_start_x(&self) -> Coord {
        let tw = self.text_width();
        let uw = self.usable_width();
        let usable_center = self.center_x;
        if tw <= uw {
            // Text fits — center it in the usable area.
            usable_center - tw * 0.5
        } else {
            // Text overflows — offset by scroll.
            usable_center - tw * 0.5 + self.scroll_offset
        }
    }

    /// Pixel X of the cursor bar.
    fn cursor_pixel_x(&self) -> Coord {
        self.text_start_x() + self.widths[..self.cursor].iter().sum::<Coord>()
    }

    /// Bounding rect (viewport coords) of the cursor's wave smear including the 7-pixel decay on each side. Used to size a sub-viewport `cursor_group` so blink ticks touch ~16 × font_size pixels instead of the entire viewport.
    pub fn cursor_bbox(&self) -> Region {
        let cpx = self.cursor_pixel_x();
        let x = cpx - 8.0;
        let y = self.center_y - self.font_size * 0.5;
        Region::new(x, y, 16.0, self.font_size)
    }

    /// Bare textbox bounding rect (viewport coords) — exactly the pill rect, NO glow padding. This is the cache size, the per-keystroke dirty rect, and the per-hover dirty rect — anything that touches just the textbox itself.
    pub fn bbox(&self) -> Region {
        Region::new(
            self.center_x - self.width * 0.5,
            self.center_y - self.height * 0.5,
            self.width,
            self.height,
        )
    }

    /// Larger bbox with `font_size` glow padding on every side. Use this for the focus-on / focus-off transition (glow appearing / disappearing) — the only time we need to repaint the wider halo region. Roughly 3× the area of [`bbox`] at default geometry, so keep it off the per-keystroke hot path.
    pub fn glow_bbox(&self) -> Region {
        let glow_pad = self.font_size;
        Region::new(
            self.center_x - self.width * 0.5 - glow_pad,
            self.center_y - self.height * 0.5 - glow_pad,
            self.width + 2.0 * glow_pad,
            self.height + 2.0 * glow_pad,
        )
    }

    // --- Rendering ---

    /// Render the textbox interior — two squircle pills stacked (photon avatar-ring pattern, but
    /// using AA edges instead of a separate stroke line).
    ///
    /// Outer pill: full size, painted in stroke colour with AA at the curve. Outer-pass AA writes
    /// `(alpha = h_aa, RGB = stroke)` so the layer's outer composite (AlphaOver onto chrome)
    /// blends stroke into bg for a smooth pill silhouette.
    ///
    /// Inner pill: inset by `stroke_px` on every side, painted in state-based fill colour. Inner-
    /// pass AA blends fill_RGB with the underlying outer-stroke RGB at the inner curve, keeping
    /// alpha = 255 — producing the proper `fill·h + stroke·(1-h)` transition.
    ///
    /// `stroke_px = (stroke_ru × font_size + 1.0) as isize` — photon's "+1" idiom for 1-px
    /// minimum stroke when `stroke_ru == 0`.
    ///
    /// Fill colour by state:
    ///   - focused (active):  `theme::TEXTBOX_ACTIVE`
    ///   - hovered only:      `theme::TEXTBOX_HOVER`
    ///   - default:           `theme::TEXTBOX_FILL`
    ///
    /// Squirdleyness = 3 (photon default; adjustable per [`paint::squircle_inset`]). Populates
    /// `self.mask` (255 inside outer silhouette, AA values on outer curve) for downstream glow.
    pub fn render_content_into(
        &mut self,
        canvas: &mut crate::canvas::Canvas,
        offset_x: Coord,
        offset_y: Coord,
        _text: &mut TextRenderer,
        clip: Option<Clip>,
        _mask: Option<&paint::AlphaMask>,
        hit_map: Option<&mut [u8]>,
        hit_id: u8,
    ) {
        let pill_x_target = (self.center_x - self.width * 0.5 - offset_x) as isize;
        let pill_y_target = (self.center_y - self.height * 0.5 - offset_y) as isize;
        let pill_w = self.width as isize;
        let pill_h = self.height as isize;

        if pill_w <= 0 || pill_h <= 0 {
            return;
        }

        // --- Cache rasterize (full squircle repaint), only when geometry changed ---
        if self.cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            let cw = pill_w as usize;
            let ch = pill_h as usize;
            self.cache.clear();
            self.cache.resize(cw * ch, 0);
            self.cache_w = cw;
            self.cache_h = ch;

            let squirdleyness = 3i32;
            let mut cache_damage = crate::canvas::Damage::new();
            let mut cache_canvas = crate::canvas::Canvas::new(
                &mut self.cache, cw, ch, &mut cache_damage,
            );

            // Paint into cache at LOCAL coords (origin = 0,0) so it's blit-translatable.
            let stroke_px = (self.stroke_ru * self.font_size) as isize + 1;
            let inner_x = stroke_px;
            let inner_y = stroke_px;
            let inner_w = (pill_w - 2 * stroke_px).max(0);
            let inner_h = (pill_h - 2 * stroke_px).max(0);
            if inner_w > 0 && inner_h > 0 {
                paint::draw_squircle_pill(
                    &mut cache_canvas,
                    inner_x,
                    inner_y,
                    inner_w,
                    inner_h,
                    theme::TEXTBOX_FILL,
                    squirdleyness,
                );
            }
            paint::draw_squircle_pill_two_tone(
                &mut cache_canvas,
                0,
                0,
                pill_w,
                pill_h,
                theme::TEXTBOX_LIGHT_EDGE,
                theme::TEXTBOX_SHADOW_EDGE,
                squirdleyness,
                None,
                0,
            );
            self.cache_dirty = false;
        }

        self.cache_origin_x = pill_x_target;
        self.cache_origin_y = pill_y_target;

        // --- In-flight tint: cache stays in its base FILL state forever; blit wrap-adds the delta on opaque pixels during the copy ---
        let tint_delta = if self.focused {
            paint::wrap_sub_rgb(theme::TEXTBOX_ACTIVE, theme::TEXTBOX_FILL)
        } else if self.hovered {
            paint::wrap_sub_rgb(theme::TEXTBOX_HOVER, theme::TEXTBOX_FILL)
        } else {
            0
        };

        blit_cache_to_target(
            &self.cache,
            self.cache_w,
            self.cache_h,
            pill_x_target,
            pill_y_target,
            canvas,
            hit_map,
            hit_id,
            tint_delta,
            clip,
        );

        // --- Focus glow on target (NOT cached — paints fresh each frame against current chrome) ---
        //
        // RU-invariant exponential falloff matching the chrome shadow: target_radius derived from font_size (3× font_size as the half-life-ish reach), factor_256 = 256 − 1240/target_radius clamped to [96, 254]. Same curve as paint_shadow; just emitted at 0°/180° (left/right) and 90°/270° (top/bottom) instead of 45° diagonals, and white instead of black. Vertical passes use half-density seed (0x40 vs horizontal 0x80) so the top/bottom halo reads softer.
        if self.focused {
            let target_radius = (self.font_size * 3.0).max(8.0);
            let drop = (1240.0 / target_radius) as u32;
            let factor_256 = (256u32.saturating_sub(drop)).clamp(96, 254);
            paint::apply_textbox_glow_right(
                canvas,
                pill_x_target,
                pill_y_target,
                pill_w,
                pill_h,
                theme::GLOW_DEFAULT,
                0x80,
                factor_256,
                clip,
            );
            paint::apply_textbox_glow_left(
                canvas,
                pill_x_target,
                pill_y_target,
                pill_w,
                pill_h,
                theme::GLOW_DEFAULT,
                0x80,
                factor_256,
                clip,
            );
            paint::apply_textbox_glow_top(
                canvas,
                pill_x_target,
                pill_y_target,
                pill_w,
                pill_h,
                theme::GLOW_DEFAULT,
                0x40,
                factor_256,
                clip,
            );
            paint::apply_textbox_glow_bottom(
                canvas,
                pill_x_target,
                pill_y_target,
                pill_w,
                pill_h,
                theme::GLOW_DEFAULT,
                0x40,
                factor_256,
                clip,
            );
        }

        // Record what we just painted so the next frame's damage_rect can union prev∪current bboxes (handles moves, hover-only changes, focus-on/off).
        self.record_painted(canvas.width, canvas.height);
    }

    /// Render only the blinkey wave cursor into a buffer (typically a sub-viewport `cursor_group` buffer). `(offset_x, offset_y)` is the buffer's top-left in viewport coords. The buffer should be zeroed before calling — blinkey writes non-zero pixels for additive composition.
    pub fn render_blinkey_into(
        &self,
        canvas: &mut crate::canvas::Canvas,
        offset_x: Coord,
        offset_y: Coord,
    ) {
        if !self.focused || self.has_selection() || !self.blinkey_visible {
            return;
        }
        let buf_w = canvas.width;
        let buf_h = canvas.height;
        let cpx_v = self.cursor_pixel_x();
        let blinkey_x_v = cpx_v as isize;
        let blinkey_x = (blinkey_x_v - offset_x as isize) as usize;
        let blinkey_y = ((self.center_y - self.font_size * 0.5) - offset_y) as usize;
        let blinkey_h = self.font_size as usize;
        if blinkey_x >= 7 && blinkey_x + 7 < buf_w && blinkey_y + blinkey_h <= buf_h {
            paint::draw_blinkey(
                canvas,
                blinkey_x,
                blinkey_y,
                blinkey_h,
                self.blinkey_wave_top,
            );
        }
    }
}
