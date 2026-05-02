//! Single-line text-entry widget. Tracks `chars: Vec<char>` and a `cursor: usize` insertion index between them. Renders fill + border + text + cursor; routes mouse clicks (focus + cursor positioning) and keyboard input (insert / backspace).
//!
//! Patterns lifted from photon's [text_editing.rs](/mnt/Octopus/Code/photon/src/ui/text_editing.rs) — `chars + widths + cursor` model, Vec<usize> per-char widths cached for click-to-cursor mapping. Photon's wave-animated blinkey and multi-line / selection / scrolling features are deferred; v0 textbox is single-line, solid cursor, no selection.

use crate::coord::Coord;
use crate::paint::{self, AlphaMask, Clip};
use crate::text::TextRenderer;
use crate::theme;

pub struct Textbox {
    /// Text content as a `Vec<char>` (not `String`) — character-indexed cursor + width arrays, no UTF-8 byte juggling at edit time.
    pub chars: Vec<char>,
    /// Insertion point, in `[0, chars.len()]`. `cursor == 0` is before all chars; `cursor == chars.len()` is after.
    pub cursor: usize,
    pub focused: bool,
    /// Pixel rect (center-anchored).
    pub center_x: Coord,
    pub center_y: Coord,
    pub width: Coord,
    pub height: Coord,
    /// Font size in pixels.
    pub font_size: Coord,
    /// Per-char pixel widths cached after the last edit. `widths[i]` = pixel width of `chars[i]`.
    widths: Vec<Coord>,
    font: &'static str,
}

impl Textbox {
    pub fn new(center_x: Coord, center_y: Coord, width: Coord, height: Coord, font_size: Coord) -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
            focused: false,
            center_x,
            center_y,
            width,
            height,
            font_size,
            widths: Vec::new(),
            font: "Open Sans",
        }
    }

    /// True if pixel `(x, y)` is inside the textbox rect.
    pub fn contains(&self, x: Coord, y: Coord) -> bool {
        let half_w = self.width * 0.5;
        let half_h = self.height * 0.5;
        x >= self.center_x - half_w && x < self.center_x + half_w
            && y >= self.center_y - half_h && y < self.center_y + half_h
    }

    /// Reposition + resize, e.g., on viewport resize. Doesn't invalidate text content; recalc_widths is still required after a font_size change.
    pub fn set_rect(&mut self, center_x: Coord, center_y: Coord, width: Coord, height: Coord) {
        self.center_x = center_x;
        self.center_y = center_y;
        self.width = width;
        self.height = height;
    }

    pub fn set_font_size(&mut self, font_size: Coord, text: &mut TextRenderer) {
        self.font_size = font_size;
        self.recalc_widths(text);
    }

    /// Recompute per-character pixel widths via the `TextRenderer`. Called after any edit + after font size changes.
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

    /// Insert a printable char at the cursor and advance.
    pub fn insert_char(&mut self, c: char, text: &mut TextRenderer) {
        if c.is_control() { return; }
        self.chars.insert(self.cursor, c);
        self.cursor += 1;
        self.recalc_widths(text);
    }

    /// Delete the char before the cursor (Backspace).
    pub fn backspace(&mut self, text: &mut TextRenderer) {
        if self.cursor == 0 { return; }
        self.cursor -= 1;
        self.chars.remove(self.cursor);
        self.recalc_widths(text);
    }

    /// Delete the char after the cursor (Delete).
    pub fn delete_forward(&mut self, text: &mut TextRenderer) {
        if self.cursor >= self.chars.len() { return; }
        self.chars.remove(self.cursor);
        self.recalc_widths(text);
    }

    pub fn cursor_left(&mut self) {
        if self.cursor > 0 { self.cursor -= 1; }
    }

    pub fn cursor_right(&mut self) {
        if self.cursor < self.chars.len() { self.cursor += 1; }
    }

    pub fn cursor_home(&mut self) { self.cursor = 0; }
    pub fn cursor_end(&mut self) { self.cursor = self.chars.len(); }

    /// Convert a click x-coordinate (in window pixel space) to a cursor index. Walks the cached `widths` array, picking the nearest char boundary.
    pub fn cursor_index_from_x(&self, click_x: Coord) -> usize {
        let text_left = self.text_left();
        if click_x <= text_left { return 0; }
        let mut accum = text_left;
        for (i, &w) in self.widths.iter().enumerate() {
            let mid = accum + w * 0.5;
            if click_x < mid { return i; }
            accum += w;
        }
        self.chars.len()
    }

    /// Handle a mouse click. Sets focus + cursor position if the click is inside; clears focus otherwise.
    pub fn handle_click(&mut self, x: Coord, y: Coord) {
        if self.contains(x, y) {
            self.focused = true;
            self.cursor = self.cursor_index_from_x(x);
        } else {
            self.focused = false;
        }
    }

    /// Pixel x of the leftmost glyph (left edge of the textbox + a small inset).
    fn text_left(&self) -> Coord {
        self.center_x - self.width * 0.5 + self.padding()
    }

    /// Symmetric inset between the textbox border and the text, scaled with font size.
    fn padding(&self) -> Coord {
        self.font_size * 0.4
    }

    /// Render the textbox: fill + border + text + (focused) cursor. The `clip` and `mask` are forwarded to the text path so the textbox composes inside larger clipped regions (e.g., a pane).
    pub fn render(
        &self,
        pixels: &mut [u32],
        buf_w: usize,
        buf_h: usize,
        text: &mut TextRenderer,
        clip: Option<Clip>,
        mask: Option<&AlphaMask>,
    ) {
        let half_w = self.width * 0.5;
        let half_h = self.height * 0.5;
        let x = (self.center_x - half_w) as isize;
        let y = (self.center_y - half_h) as isize;
        let w = self.width as isize;
        let h = self.height as isize;

        // 1. Fill.
        paint::fill_rect_solid(pixels, buf_w, buf_h, x, y, w, h, theme::TEXTBOX_FILL, clip);
        // 2. Border (1 px) — focus shifts the colour subtly.
        let border = if self.focused { theme::TEXTBOX_LIGHT_EDGE } else { theme::TEXTBOX_SHADOW_EDGE };
        paint::stroke_rect(pixels, buf_w, buf_h, x, y, w, h, 1, border, clip, mask);

        // 3. Text — left-aligned, vertically centered.
        let s: String = self.chars.iter().collect();
        if !s.is_empty() {
            let _ = text.draw_text_left_u32(
                pixels, buf_w, buf_h,
                &s,
                self.text_left(),
                self.center_y,
                self.font_size,
                400,
                theme::TEXT_COLOUR,
                self.font,
                clip,
                mask,
                None,
            );
        }

        // 4. Cursor — solid 1-px-or-thicker vertical bar at the cursor's pixel x. Only when focused.
        if self.focused {
            let cursor_x = self.text_left() + self.widths[..self.cursor].iter().sum::<Coord>();
            let cursor_h = self.font_size;
            let cy_top = self.center_y - cursor_h * 0.5;
            let thickness = (self.font_size * 0.06).max(1.0) as isize;
            paint::fill_rect_solid(
                pixels, buf_w, buf_h,
                cursor_x as isize, cy_top as isize,
                thickness, cursor_h as isize,
                theme::TEXT_COLOUR,
                clip,
            );
        }
    }
}
