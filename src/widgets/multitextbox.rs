//! Multi-line textbox — the chat-compose widget: wrap-aware layout, a real (line, column) caret, vertical growth to a caller-set cap, then internal scroll.
//!
//! A deliberate SIBLING of [`super::textbox::Textbox`], not a mode on it: the single-line box is a horizontal-pan machine (per-char widths, pan-spring, centre-anchored layout) and the multi-line box is a wrap machine (line table, vertical scroll, bottom-anchored growth). One struct doing both was the complexity trap the split avoids. Shared idioms are shared code: the squircle pill cache, the α-mask text clip, the three-layer under() composition, the damage protocol (`last_painted_*` diffing) and the [`crate::host::widget::Widget`] capability surface all match Textbox — a host that speaks to one speaks to both.
//!
//! What the widget does NOT decide: what Enter means. An Enter that reaches [`Key::on_key`] inserts a newline, full stop — the APP intercepts Enter before dispatch when it means "send" (the desktop convention: Enter sends, Shift+Enter reaches the widget; a soft-IME platform routes every Enter here and sends via its button). Keeping the send semantic out of the widget keeps the widget honest about being a text field.

use crate::canvas::PixelRect;
use crate::coord::Coord;
use crate::event::{ElementState, Key as FKey, KeyEvent, ModifiersState, NamedKey};
use crate::host::widget::{next_id, Click, Focus, Hover, Key, PaintCtx, Widget};
use crate::paint::HitId;
use crate::paint::{self, Clip};
use crate::region::Region;
use crate::text::{TextRenderer, TextStyle};
use crate::theme;

use super::textbox::{blit_cache_to_target, region_to_pixelrect};

pub struct MultiTextbox {
    /// Text content as a `Vec<char>` — `'\n'` lives inline (a newline is a character of the text, zero-width, terminating its visual line).
    pub chars: Vec<char>,
    /// Insertion point in `[0, chars.len()]`.
    pub cursor: usize,
    /// Selection anchor (shift+movement / drag) — selection is the `anchor..cursor` char range, either order.
    pub selection_anchor: Option<usize>,
    hit_id: HitId,
    focused: bool,
    hovered: bool,
    enabled: bool,
    pub stroke_ru: f32,

    // --- Geometry (centre-anchored rect like every widget; the APP anchors growth by recomputing center_y from a fixed bottom edge after `set_layout`) ---
    pub center_x: Coord,
    pub center_y: Coord,
    pub width: Coord,
    /// CURRENT height — computed by [`Self::set_layout`] from the wrapped line count, clamped to the caller's cap. Read it back for the surrounding layout (the list above yields to it).
    pub height: Coord,
    pub font_size: Coord,
    font: &'static str,
    /// Text-area exclusion from the pill's RIGHT edge (an overlaid send button). Applies to every line — a clean rectangle, not a last-line notch.
    right_inset: Coord,

    // --- Wrap layout (rebuilt by `relayout` on any text / width / font change) ---
    /// Per-char advance widths (`'\n'` = 0).
    widths: Vec<Coord>,
    /// Char index where each VISUAL line begins. Always ≥ 1 entry (`[0]`). Line `i` spans `line_starts[i]..line_end(i)`; a terminating `'\n'` belongs to the line it ends.
    line_starts: Vec<usize>,
    /// Vertical scroll in pixels — 0 = top of content at the top of the box. Clamped by `relayout`/`ensure_cursor_visible`; only meaningful once content exceeds the box.
    scroll_y: Coord,
    /// The pixel x the caret aims to keep across Up/Down moves (standard editor column memory). Cleared by any horizontal move or edit.
    goal_x: Option<Coord>,

    // --- Blinkey ---
    pub blinkey_visible: bool,
    pub blinkey_wave_top: bool,

    // --- Caches + damage protocol (the Textbox discipline, verbatim) ---
    pill_cache: Vec<u32>,
    pill_cache_w: usize,
    pill_cache_h: usize,
    pill_cache_dirty: bool,
    inner_pill_mask: Vec<u8>,
    text_cache: Vec<u32>,
    text_cache_w: usize,
    text_cache_h: usize,
    text_cache_dirty: bool,
    last_painted_bbox: Option<PixelRect>,
    last_painted_focused: bool,
    last_painted_selection: Option<(usize, usize)>,
    last_painted_blinkey_on: bool,
    last_painted_blinkey_bbox: Option<PixelRect>,
}

impl MultiTextbox {
    pub fn new(hit_counter: &mut HitId, font_size: Coord, font: &'static str) -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
            selection_anchor: None,
            hit_id: next_id(hit_counter),
            focused: false,
            hovered: false,
            enabled: true,
            stroke_ru: 1.0 / (1 << 5) as f32,
            center_x: 0.0,
            center_y: 0.0,
            width: 1.0,
            height: 1.0,
            font_size,
            font,
            right_inset: 0.0,
            widths: Vec::new(),
            line_starts: vec![0],
            scroll_y: 0.0,
            goal_x: None,
            blinkey_visible: true,
            blinkey_wave_top: true,
            pill_cache: Vec::new(),
            pill_cache_w: 0,
            pill_cache_h: 0,
            pill_cache_dirty: true,
            inner_pill_mask: Vec::new(),
            text_cache: Vec::new(),
            text_cache_w: 0,
            text_cache_h: 0,
            text_cache_dirty: true,
            last_painted_bbox: None,
            last_painted_focused: false,
            last_painted_selection: None,
            last_painted_blinkey_on: false,
            last_painted_blinkey_bbox: None,
        }
    }

    // ---- Metrics ----

    /// Horizontal text inset from the pill edge (both sides).
    pub fn pad_x(&self) -> Coord {
        self.font_size * 0.55
    }
    /// Vertical text inset (top and bottom).
    pub fn pad_y(&self) -> Coord {
        self.font_size * 0.45
    }
    /// One visual line's row height.
    pub fn row_h(&self) -> Coord {
        self.font_size * 1.4
    }
    /// The wrap width text lays out against.
    fn usable_w(&self) -> Coord {
        (self.width - self.pad_x() * 2.0 - self.right_inset).max(self.font_size)
    }
    /// Total laid-out text height (all lines), independent of the box height.
    pub fn content_h(&self) -> Coord {
        self.line_starts.len() as Coord * self.row_h() + self.pad_y() * 2.0
    }
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }

    // ---- Layout ----

    fn measure_char(&self, c: char, text: &mut TextRenderer) -> Coord {
        if c == '\n' {
            return 0.0;
        }
        let style = TextStyle::new(self.font_size, theme::TEXTBOX_TEXT).font(self.font);
        text.measure_text(c.encode_utf8(&mut [0u8; 4]), &style)
    }

    /// Full re-measure + rewrap — geometry/font changes only. EDITS splice `widths` incrementally and call `rewrap` alone: re-measuring the whole buffer per keystroke was O(chars) cosmic-text shapes per key, tens of ms on a real draft (the 2026-08-09 compose lag).
    fn relayout(&mut self, text: &mut TextRenderer) {
        self.widths.clear();
        self.widths.reserve(self.chars.len());
        for i in 0..self.chars.len() {
            let c = self.chars[i];
            let w = self.measure_char(c, text);
            self.widths.push(w);
        }
        self.rewrap();
    }

    /// Rebuild the line table from the EXISTING widths — pure arithmetic, no shaping.
    fn rewrap(&mut self) {
        let max_w = self.usable_w();
        self.line_starts.clear();
        self.line_starts.push(0);
        let mut line_w: Coord = 0.0;
        let mut last_space: Option<usize> = None; // index of the last breakable space in the CURRENT line
        let mut i = 0usize;
        while i < self.chars.len() {
            let c = self.chars[i];
            if c == '\n' {
                self.line_starts.push(i + 1);
                line_w = 0.0;
                last_space = None;
                i += 1;
                continue;
            }
            let w = self.widths[i];
            if line_w + w > max_w && line_w > 0.0 {
                // Overflow: break AFTER the last space when one exists in this line (word wrap), else right here (mid-word).
                let brk = match last_space {
                    Some(sp) if sp + 1 > *self.line_starts.last().unwrap() => sp + 1,
                    _ => i,
                };
                self.line_starts.push(brk);
                line_w = self.widths[brk..=i.saturating_sub(0)]
                    .iter()
                    .take(i - brk)
                    .sum::<Coord>()
                    + w;
                last_space = None;
                i += 1;
                continue;
            }
            if c == ' ' {
                last_space = Some(i);
            }
            line_w += w;
            i += 1;
        }
        self.clamp_scroll();
        self.text_cache_dirty = true;
    }

    /// Set geometry for this frame: wrap against `width`, grow to fit the content, cap at `max_h`, anchor the BOTTOM edge at `bottom_y`. The caller reads `.height` back for its own layout. Re-entrant per frame — every step self-diffs, so an unchanged frame costs two float compares.
    pub fn set_layout(
        &mut self,
        center_x: Coord,
        bottom_y: Coord,
        width: Coord,
        max_h: Coord,
        text: &mut TextRenderer,
    ) {
        let width_changed = (self.width - width).abs() > 0.5;
        self.center_x = center_x;
        if width_changed {
            self.width = width;
            self.relayout(text);
            self.pill_cache_dirty = true;
        }
        let min_h = self.row_h() + self.pad_y() * 2.0;
        let new_h = self.content_h().clamp(min_h, max_h.max(min_h));
        if (self.height - new_h).abs() > 0.5 {
            self.height = new_h;
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
        }
        let new_cy = bottom_y - self.height * 0.5;
        if (self.center_y - new_cy).abs() > 0.5 {
            self.center_y = new_cy;
        }
        self.clamp_scroll();
    }

    pub fn set_font_size(&mut self, font_size: Coord, text: &mut TextRenderer) {
        if (self.font_size - font_size).abs() > 0.01 {
            self.font_size = font_size;
            self.pill_cache_dirty = true;
            self.relayout(text);
        }
    }

    pub fn set_right_inset(&mut self, inset: Coord) {
        if (self.right_inset - inset).abs() > 0.5 {
            self.right_inset = inset;
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
        }
    }

    // ---- Line math ----

    fn line_end(&self, line: usize) -> usize {
        self.line_starts
            .get(line + 1)
            .copied()
            .unwrap_or(self.chars.len())
    }
    /// Chars drawable on `line` — the terminating `'\n'` (if any) is excluded.
    fn line_draw_range(&self, line: usize) -> (usize, usize) {
        let s = self.line_starts[line];
        let mut e = self.line_end(line);
        if e > s && self.chars.get(e - 1) == Some(&'\n') {
            e -= 1;
        }
        (s, e)
    }
    fn line_of(&self, idx: usize) -> usize {
        match self.line_starts.binary_search(&idx) {
            Ok(l) => l,
            Err(ins) => ins - 1,
        }
    }
    fn x_in_line(&self, idx: usize) -> Coord {
        let line = self.line_of(idx);
        let s = self.line_starts[line];
        self.widths[s..idx].iter().sum()
    }
    /// Char index on `line` closest to pixel `x` (line-local, 0 = line's left edge).
    fn index_at_line_x(&self, line: usize, x: Coord) -> usize {
        let (s, e) = self.line_draw_range(line);
        let mut cum = 0.0;
        for i in s..e {
            let w = self.widths[i];
            if x < cum + w * 0.5 {
                return i;
            }
            cum += w;
        }
        e
    }

    /// The box-local top of the text area (first line's top at scroll 0).
    fn inner_top(&self) -> Coord {
        self.pad_y()
    }
    fn inner_h(&self) -> Coord {
        self.height - self.pad_y() * 2.0
    }
    fn clamp_scroll(&mut self) {
        let max = (self.line_starts.len() as Coord * self.row_h() - self.inner_h()).max(0.0);
        self.scroll_y = self.scroll_y.clamp(0.0, max);
    }
    /// Keep the caret's line inside the visible band.
    fn ensure_cursor_visible(&mut self) {
        let line = self.line_of(self.cursor);
        let line_top = line as Coord * self.row_h();
        let line_bot = line_top + self.row_h();
        if line_top < self.scroll_y {
            self.scroll_y = line_top;
        } else if line_bot > self.scroll_y + self.inner_h() {
            self.scroll_y = line_bot - self.inner_h();
        }
        self.clamp_scroll();
    }

    // ---- Edits (each: mutate → relayout → cursor visible → caches dirty) ----

    pub fn insert_char(&mut self, c: char, text: &mut TextRenderer) {
        self.delete_selection_internal();
        let w = self.measure_char(c, text);
        self.chars.insert(self.cursor, c);
        self.widths.insert(self.cursor, w);
        self.cursor += 1;
        self.goal_x = None;
        self.rewrap();
        self.ensure_cursor_visible();
    }
    pub fn insert_str(&mut self, s: &str, text: &mut TextRenderer) {
        self.delete_selection_internal();
        for c in s.chars() {
            let w = self.measure_char(c, text);
            self.chars.insert(self.cursor, c);
            self.widths.insert(self.cursor, w);
            self.cursor += 1;
        }
        self.goal_x = None;
        self.rewrap();
        self.ensure_cursor_visible();
    }
    pub fn backspace(&mut self, _text: &mut TextRenderer) {
        if self.has_selection() {
            self.delete_selection_internal();
        } else if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
            self.widths.remove(self.cursor);
        }
        self.goal_x = None;
        self.rewrap();
        self.ensure_cursor_visible();
    }
    pub fn delete_forward(&mut self, _text: &mut TextRenderer) {
        if self.has_selection() {
            self.delete_selection_internal();
        } else if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
            self.widths.remove(self.cursor);
        }
        self.goal_x = None;
        self.rewrap();
        self.ensure_cursor_visible();
    }
    /// TRUE range replacement (the honest-IME write path — voice dictation rewriting earlier words). Caret lands at the end of the inserted text.
    pub fn replace_char_range(
        &mut self,
        start: usize,
        end: usize,
        s: &str,
        text: &mut TextRenderer,
    ) {
        let n = self.chars.len();
        let start = start.min(n);
        let end = end.clamp(start, n);
        self.chars.drain(start..end);
        self.widths.drain(start..end);
        let mut at = start;
        for c in s.chars() {
            let w = self.measure_char(c, text);
            self.chars.insert(at, c);
            self.widths.insert(at, w);
            at += 1;
        }
        self.cursor = at;
        self.selection_anchor = None;
        self.goal_x = None;
        self.rewrap();
        self.ensure_cursor_visible();
    }

    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.selection_anchor = None;
        self.goal_x = None;
        self.scroll_y = 0.0;
        self.line_starts.clear();
        self.line_starts.push(0);
        self.widths.clear();
        self.text_cache_dirty = true;
    }

    // ---- Selection ----

    pub fn has_selection(&self) -> bool {
        self.selection_range().is_some()
    }
    pub fn selection_range(&self) -> Option<(usize, usize)> {
        let a = self.selection_anchor?;
        if a == self.cursor {
            return None;
        }
        Some((a.min(self.cursor), a.max(self.cursor)))
    }
    pub fn select_all(&mut self) {
        self.selection_anchor = Some(0);
        self.cursor = self.chars.len();
        self.text_cache_dirty = true;
    }
    pub fn selected_text(&self) -> Option<String> {
        self.selection_range()
            .map(|(s, e)| self.chars[s..e].iter().collect())
    }
    fn delete_selection_internal(&mut self) {
        if let Some((s, e)) = self.selection_range() {
            self.chars.drain(s..e);
            self.widths.drain(s..e);
            self.cursor = s;
        }
        self.selection_anchor = None;
    }
    pub fn delete_selection(&mut self, _text: &mut TextRenderer) {
        self.delete_selection_internal();
        self.rewrap();
        self.ensure_cursor_visible();
    }

    // ---- Caret movement ----

    pub fn cursor_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
        self.goal_x = None;
        self.ensure_cursor_visible();
        self.text_cache_dirty = true;
    }
    pub fn cursor_right(&mut self) {
        self.cursor = (self.cursor + 1).min(self.chars.len());
        self.goal_x = None;
        self.ensure_cursor_visible();
        self.text_cache_dirty = true;
    }
    pub fn cursor_home(&mut self) {
        let line = self.line_of(self.cursor);
        self.cursor = self.line_starts[line];
        self.goal_x = None;
        self.ensure_cursor_visible();
        self.text_cache_dirty = true;
    }
    pub fn cursor_end(&mut self) {
        let (_, e) = self.line_draw_range(self.line_of(self.cursor));
        self.cursor = e;
        self.goal_x = None;
        self.ensure_cursor_visible();
        self.text_cache_dirty = true;
    }
    /// Up/Down with column memory: the caret aims for the SAME pixel x on the neighbour line, remembering the goal across consecutive vertical moves (standard editor behaviour — sailing past a short line doesn't lose the column).
    pub fn cursor_vertical(&mut self, down: bool) {
        let line = self.line_of(self.cursor);
        let target_line = if down {
            if line + 1 >= self.line_starts.len() {
                return;
            }
            line + 1
        } else {
            if line == 0 {
                return;
            }
            line - 1
        };
        let x = self.goal_x.unwrap_or_else(|| self.x_in_line(self.cursor));
        self.cursor = self.index_at_line_x(target_line, x);
        self.goal_x = Some(x);
        self.ensure_cursor_visible();
        self.text_cache_dirty = true;
    }

    // ---- Pointer ----

    /// Box-local coords of a viewport point.
    fn to_local(&self, x: Coord, y: Coord) -> (Coord, Coord) {
        (
            x - (self.center_x - self.width * 0.5),
            y - (self.center_y - self.height * 0.5),
        )
    }
    /// Char index nearest a viewport (x, y) — the click/drag caret placement.
    pub fn cursor_index_from_xy(&self, x: Coord, y: Coord) -> usize {
        let (lx, ly) = self.to_local(x, y);
        let line_f = (ly - self.inner_top() + self.scroll_y) / self.row_h();
        let line = (line_f.max(0.0) as usize).min(self.line_starts.len().saturating_sub(1));
        self.index_at_line_x(line, lx - self.pad_x())
    }
    /// Press: place the caret, clear the anchor (drag extends from here).
    pub fn pointer_press(&mut self, x: Coord, y: Coord) {
        self.cursor = self.cursor_index_from_xy(x, y);
        self.selection_anchor = None;
        self.goal_x = None;
        self.text_cache_dirty = true;
    }
    /// Drag: extend the selection to the pointer.
    pub fn pointer_drag(&mut self, x: Coord, y: Coord) {
        if self.selection_anchor.is_none() {
            self.selection_anchor = Some(self.cursor);
        }
        self.cursor = self.cursor_index_from_xy(x, y);
        self.text_cache_dirty = true;
    }
    /// Word-select at a char index (double-tap): extends across the probe char's class run — same rule as the single-line box.
    pub fn select_word_at(&mut self, idx: usize) {
        if self.chars.is_empty() {
            return;
        }
        let idx = idx.min(self.chars.len() - 1);
        let class = |c: char| -> u8 {
            if c.is_alphanumeric() || c == '_' {
                0
            } else if c.is_whitespace() {
                1
            } else {
                2
            }
        };
        let k = class(self.chars[idx]);
        let mut s = idx;
        while s > 0 && class(self.chars[s - 1]) == k && self.chars[s - 1] != '\n' {
            s -= 1;
        }
        let mut e = idx + 1;
        while e < self.chars.len() && class(self.chars[e]) == k && self.chars[e] != '\n' {
            e += 1;
        }
        self.selection_anchor = Some(s);
        self.cursor = e;
        self.text_cache_dirty = true;
    }
    /// Wheel/fling scroll of the internal band. Returns true when the offset moved.
    pub fn scroll_by(&mut self, dy: Coord) -> bool {
        let before = self.scroll_y;
        self.scroll_y += dy;
        self.clamp_scroll();
        if (self.scroll_y - before).abs() > 0.01 {
            self.text_cache_dirty = true;
            true
        } else {
            false
        }
    }

    // ---- Blinkey / bboxes ----

    pub fn flip_blinkey(&mut self) -> bool {
        if !self.focused {
            return false;
        }
        self.blinkey_visible = !self.blinkey_visible;
        self.blinkey_wave_top = !self.blinkey_wave_top;
        true
    }
    /// Viewport-space caret rect (the blinkey band).
    pub fn cursor_bbox(&self) -> Region {
        let line = self.line_of(self.cursor);
        let x = self.center_x - self.width * 0.5
            + self.pad_x()
            + self.x_in_line(self.cursor);
        let y = self.center_y - self.height * 0.5 + self.inner_top()
            + line as Coord * self.row_h()
            - self.scroll_y;
        Region::new(x - 8.0, y, 16.0, self.row_h())
    }
    pub fn bbox(&self) -> Region {
        Region::new(
            self.center_x - self.width * 0.5,
            self.center_y - self.height * 0.5,
            self.width,
            self.height,
        )
    }
    pub fn glow_bbox(&self) -> Region {
        let pad = self.font_size * 1.2;
        Region::new(
            self.center_x - self.width * 0.5 - pad,
            self.center_y - self.height * 0.5 - pad,
            self.width + pad * 2.0,
            self.height + pad * 2.0,
        )
    }
    pub fn contains(&self, x: Coord, y: Coord) -> bool {
        let (lx, ly) = self.to_local(x, y);
        lx >= 0.0 && lx <= self.width && ly >= 0.0 && ly <= self.height
    }
    pub fn hit_id(&self) -> HitId {
        self.hit_id
    }
    pub fn is_focused(&self) -> bool {
        self.focused
    }
    pub fn set_focused(&mut self, focused: bool) {
        if self.focused != focused {
            self.focused = focused;
            self.blinkey_visible = focused;
        }
    }
    pub fn is_hovered(&self) -> bool {
        self.hovered
    }
    pub fn set_hovered(&mut self, hovered: bool) {
        self.hovered = hovered;
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    pub fn invalidate_cache(&mut self) {
        self.pill_cache_dirty = true;
        self.text_cache_dirty = true;
    }
    pub fn reset_paint_tracking(&mut self) {
        self.last_painted_bbox = None;
        self.last_painted_blinkey_on = false;
        self.last_painted_blinkey_bbox = None;
        self.last_painted_selection = None;
        self.pill_cache_dirty = true;
        self.text_cache_dirty = true;
    }

    // ---- Damage (the Textbox protocol, adapted) ----

    pub fn damage_rect(&self, viewport_w: usize, viewport_h: usize) -> Option<PixelRect> {
        let focus_changed = self.focused != self.last_painted_focused;
        let selection_changed = self.selection_range() != self.last_painted_selection;
        let pill_dirty =
            self.pill_cache_dirty || self.text_cache_dirty || focus_changed || selection_changed;
        let blinkey_want = self.focused && !self.has_selection() && self.blinkey_visible;
        let blinkey_active = blinkey_want || self.last_painted_blinkey_on;
        if self.last_painted_bbox.is_none() {
            return None;
        }
        if !pill_dirty && !blinkey_active {
            return None;
        }
        let mut combined: Option<PixelRect> = None;
        let union_in = |slot: &mut Option<PixelRect>, r: PixelRect| {
            if r.is_empty() {
                return;
            }
            *slot = Some(match *slot {
                Some(c) => c.union(r),
                None => r,
            });
        };
        if pill_dirty {
            let need_glow = focus_changed || self.pill_cache_dirty;
            let cur = if need_glow { self.glow_bbox() } else { self.bbox() };
            union_in(
                &mut combined,
                region_to_pixelrect(cur, viewport_w, viewport_h),
            );
            if let Some(prev) = self.last_painted_bbox {
                union_in(&mut combined, prev);
            }
        }
        if blinkey_want {
            union_in(
                &mut combined,
                region_to_pixelrect(self.cursor_bbox(), viewport_w, viewport_h),
            );
        }
        if self.last_painted_blinkey_on {
            if let Some(prev) = self.last_painted_blinkey_bbox {
                union_in(&mut combined, prev);
            }
        }
        combined.filter(|r| !r.is_empty())
    }

    fn record_painted(&mut self, viewport_w: usize, viewport_h: usize) {
        let rect = region_to_pixelrect(self.bbox(), viewport_w, viewport_h);
        self.last_painted_bbox = if rect.is_empty() { None } else { Some(rect) };
        self.last_painted_focused = self.focused;
        self.last_painted_selection = self.selection_range();
        let blinkey_want = self.focused && !self.has_selection() && self.blinkey_visible;
        self.last_painted_blinkey_on = blinkey_want;
        self.last_painted_blinkey_bbox = if blinkey_want {
            let cur = region_to_pixelrect(self.cursor_bbox(), viewport_w, viewport_h);
            (!cur.is_empty()).then_some(cur)
        } else {
            None
        };
    }

    // ---- Render (pill cache → text cache with per-line draws → three-layer under() blits → blinkey wave) ----

    #[allow(clippy::too_many_arguments)]
    pub fn render_content_into(
        &mut self,
        canvas: &mut crate::canvas::Canvas,
        offset_x: Coord,
        offset_y: Coord,
        text: &mut TextRenderer,
        clip: Option<Clip>,
        hit_map: Option<&mut [HitId]>,
        hit_id: HitId,
    ) {
        let pill_x_target = (self.center_x - self.width * 0.5 - offset_x) as isize;
        let pill_y_target = (self.center_y - self.height * 0.5 - offset_y) as isize;
        let pill_w = self.width as isize;
        let pill_h = self.height as isize;
        if pill_w <= 0 || pill_h <= 0 {
            return;
        }
        let cw = pill_w as usize;
        let ch = pill_h as usize;
        let squirdleyness = 3;
        let stroke_px = (self.stroke_ru * self.font_size) as isize + 1;

        if self.pill_cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.pill_cache.clear();
            self.pill_cache.resize(cw * ch, 0);
            self.pill_cache_w = cw;
            self.pill_cache_h = ch;
            let inner_x = stroke_px;
            let inner_y = stroke_px;
            let inner_w = (pill_w - 2 * stroke_px).max(0);
            let inner_h = (pill_h - 2 * stroke_px).max(0);
            let mut cache_damage = crate::canvas::Damage::new();
            {
                let mut cache_canvas =
                    crate::canvas::Canvas::new(&mut self.pill_cache, cw, ch, &mut cache_damage);
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
            }
            self.inner_pill_mask.clear();
            self.inner_pill_mask.resize(cw * ch, 0);
            for i in 0..(cw * ch) {
                self.inner_pill_mask[i] = ((self.pill_cache[i] >> 24) & 0xFF) as u8;
            }
            if self.right_inset > 0.0 {
                let cut = (pill_w as Coord - self.right_inset).max(0.0) as usize;
                for y in 0..ch {
                    for x in cut..cw {
                        self.inner_pill_mask[y * cw + x] = 0;
                    }
                }
            }
            {
                let mut cache_canvas =
                    crate::canvas::Canvas::new(&mut self.pill_cache, cw, ch, &mut cache_damage);
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
            }
            self.pill_cache_dirty = false;
        }

        if self.selection_range() != self.last_painted_selection {
            self.text_cache_dirty = true;
        }

        if self.text_cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.text_cache.clear();
            self.text_cache.resize(cw * ch, 0);
            self.text_cache_w = cw;
            self.text_cache_h = ch;
            let style = TextStyle::new(self.font_size, theme::TEXTBOX_TEXT).font(self.font);
            let local_left = self.pad_x();
            let row_h = self.row_h();
            let sel = self.selection_range();

            // Visible line band under the current scroll — everything else is culled.
            let first_line = ((self.scroll_y / row_h).floor() as usize)
                .min(self.line_starts.len().saturating_sub(1));
            let last_line = (((self.scroll_y + self.inner_h()) / row_h).ceil() as usize)
                .min(self.line_starts.len().saturating_sub(1));

            // Per-line draw work precomputed BEFORE the cache borrow (line text, y centre, selection rect) — the borrow of text_cache below can't coexist with &self reads.
            struct LineDraw {
                s: String,
                y_center: Coord,
                sel: Option<(Coord, Coord)>,
            }
            let mut draws: Vec<LineDraw> = Vec::with_capacity(last_line + 1 - first_line);
            for line in first_line..=last_line {
                let (ds, de) = self.line_draw_range(line);
                let y_top = self.inner_top() + line as Coord * row_h - self.scroll_y;
                let y_center = y_top + row_h * 0.5;
                let sel_rect = sel.and_then(|(a, b)| {
                    let ls = a.max(ds);
                    let le = b.min(de);
                    if ls >= le {
                        // A selection crossing this line's newline still highlights a token-wide stub past the end, so a selected line break is visible.
                        if a <= de && b > de && self.line_end(line) > de {
                            let x0: Coord =
                                local_left + self.widths[ds..de].iter().sum::<Coord>();
                            return Some((x0, x0 + self.font_size * 0.35));
                        }
                        return None;
                    }
                    let x0: Coord = local_left + self.widths[ds..ls].iter().sum::<Coord>();
                    let x1: Coord = x0 + self.widths[ls..le].iter().sum::<Coord>();
                    Some((x0, x1))
                });
                draws.push(LineDraw {
                    s: self.chars[ds..de].iter().collect(),
                    y_center,
                    sel: sel_rect,
                });
            }

            let mut text_damage = crate::canvas::Damage::new();
            let mut text_canvas =
                crate::canvas::Canvas::new(&mut self.text_cache, cw, ch, &mut text_damage);
            let mask_buffer = paint::AlphaMask::new(&self.inner_pill_mask, cw, ch);
            for d in &draws {
                if !d.s.is_empty() {
                    text.draw_text_left(
                        &mut text_canvas,
                        &d.s,
                        local_left,
                        d.y_center,
                        &style,
                        None,
                        Some(&mask_buffer),
                    );
                }
            }
            for d in &draws {
                if let Some((x0, x1)) = d.sel {
                    paint::fill_rect(
                        &mut text_canvas,
                        x0 as isize,
                        (d.y_center - self.font_size * 0.5) as isize,
                        (x1 - x0) as isize,
                        self.font_size as isize,
                        theme::TEXTBOX_SELECTION_BG,
                        None,
                        Some(&mask_buffer),
                    );
                }
            }
            self.text_cache_dirty = false;
        }

        blit_cache_to_target(
            &self.text_cache,
            self.text_cache_w,
            self.text_cache_h,
            pill_x_target,
            pill_y_target,
            canvas,
            None,
            0,
            clip,
        );
        blit_cache_to_target(
            &self.pill_cache,
            self.pill_cache_w,
            self.pill_cache_h,
            pill_x_target,
            pill_y_target,
            canvas,
            hit_map,
            hit_id,
            clip,
        );

        // Blinkey — THE canonical painter (paint::draw_blinkey), at the caret's (line, x). The first cut hand-rolled a wave that read as no cursor at all (field, 2026-08-09).
        if self.focused && !self.has_selection() && self.blinkey_visible {
            let cb = self.cursor_bbox();
            let bx = (cb.x + 8.0 - offset_x) as isize;
            let by_top = (cb.y - offset_y) as isize;
            let bh = self.row_h() as usize;
            let buf_w = canvas.width;
            let buf_h = canvas.height;
            if bx >= 7 && (bx as usize) + 7 < buf_w && by_top >= 0 && by_top as usize + bh <= buf_h
            {
                paint::draw_blinkey(
                    canvas,
                    bx as usize,
                    by_top as usize,
                    bh,
                    self.blinkey_wave_top,
                );
            }
        }

        self.record_painted(canvas.width, canvas.height);
    }
}

impl Widget for MultiTextbox {
    fn id(&self) -> HitId {
        self.hit_id
    }
    fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
        // Same phase-5 note as Textbox: the app drives rendering via render_content_into.
    }
    fn click(&mut self) -> Option<&mut dyn Click> {
        self.enabled.then_some(self as &mut dyn Click)
    }
    fn key(&mut self) -> Option<&mut dyn Key> {
        self.enabled.then_some(self as &mut dyn Key)
    }
    fn focus(&mut self) -> Option<&mut dyn Focus> {
        self.enabled.then_some(self as &mut dyn Focus)
    }
    fn hover(&mut self) -> Option<&mut dyn Hover> {
        self.enabled.then_some(self as &mut dyn Hover)
    }
    fn damage_rect(&self, viewport_w: usize, viewport_h: usize) -> Option<PixelRect> {
        MultiTextbox::damage_rect(self, viewport_w, viewport_h)
    }
    fn is_text_input(&self) -> bool {
        true
    }
}

impl Click for MultiTextbox {
    fn on_click(&mut self, x: Coord, y: Coord, _mods: ModifiersState) -> crate::host::EventResponse {
        self.cursor = self.cursor_index_from_xy(x, y);
        self.selection_anchor = None;
        self.goal_x = None;
        self.text_cache_dirty = true;
        crate::host::EventResponse::Handled
    }
}

impl Focus for MultiTextbox {
    fn set_focused(&mut self, focused: bool) {
        MultiTextbox::set_focused(self, focused);
    }
    fn focus_bbox(&self) -> Option<PixelRect> {
        let x0 = (self.center_x - self.width * 0.5).max(0.0) as usize;
        let y0 = (self.center_y - self.height * 0.5).max(0.0) as usize;
        let x1 = (self.center_x + self.width * 0.5).max(0.0) as usize;
        let y1 = (self.center_y + self.height * 0.5).max(0.0) as usize;
        Some(PixelRect::new(x0, y0, x1, y1))
    }
}

impl Hover for MultiTextbox {
    fn set_hovered(&mut self, hovered: bool) {
        MultiTextbox::set_hovered(self, hovered);
    }
    fn tint_delta(&self) -> u32 {
        if self.is_focused() {
            paint::wrap_sub_rgb(theme::TEXTBOX_ACTIVE, theme::TEXTBOX_FILL)
        } else if self.hovered {
            paint::wrap_sub_rgb(theme::TEXTBOX_HOVER, theme::TEXTBOX_FILL)
        } else {
            0
        }
    }
    fn hover_bbox(&self, viewport_w: usize, viewport_h: usize) -> Option<PixelRect> {
        Some(region_to_pixelrect(self.bbox(), viewport_w, viewport_h))
    }
}

impl Key for MultiTextbox {
    fn on_key(
        &mut self,
        kev: &KeyEvent,
        mods: ModifiersState,
        text: &mut TextRenderer,
    ) -> crate::host::EventResponse {
        if kev.state != ElementState::Pressed {
            return crate::host::EventResponse::Pass;
        }
        let shift = mods.shift_key();
        let ctrl = mods.super_key() || mods.control_key();
        let mut changed = false;
        let start_selection_if_needed = |slf: &mut MultiTextbox| {
            if shift && slf.selection_anchor.is_none() {
                slf.selection_anchor = Some(slf.cursor);
            } else if !shift {
                slf.selection_anchor = None;
            }
        };
        match &kev.logical_key {
            FKey::Named(NamedKey::Backspace) => {
                self.backspace(text);
                changed = true;
            }
            FKey::Named(NamedKey::Delete) => {
                self.delete_forward(text);
                changed = true;
            }
            // Enter that REACHES the widget is a newline, always — "Enter means send" is the app's interception upstream, never this widget's opinion.
            FKey::Named(NamedKey::Enter) => {
                self.insert_char('\n', text);
                changed = true;
            }
            FKey::Named(NamedKey::ArrowLeft) => {
                if !shift {
                    if let Some((start, _)) = self.selection_range() {
                        self.selection_anchor = None;
                        self.cursor = start;
                        self.ensure_cursor_visible();
                        self.text_cache_dirty = true;
                    } else {
                        self.cursor_left();
                    }
                } else {
                    start_selection_if_needed(self);
                    self.cursor_left();
                }
                changed = true;
            }
            FKey::Named(NamedKey::ArrowRight) => {
                if !shift {
                    if let Some((_, end)) = self.selection_range() {
                        self.selection_anchor = None;
                        self.cursor = end;
                        self.ensure_cursor_visible();
                        self.text_cache_dirty = true;
                    } else {
                        self.cursor_right();
                    }
                } else {
                    start_selection_if_needed(self);
                    self.cursor_right();
                }
                changed = true;
            }
            FKey::Named(NamedKey::ArrowUp) => {
                start_selection_if_needed(self);
                self.cursor_vertical(false);
                changed = true;
            }
            FKey::Named(NamedKey::ArrowDown) => {
                start_selection_if_needed(self);
                self.cursor_vertical(true);
                changed = true;
            }
            FKey::Named(NamedKey::Home) => {
                start_selection_if_needed(self);
                self.cursor_home();
                changed = true;
            }
            FKey::Named(NamedKey::End) => {
                start_selection_if_needed(self);
                self.cursor_end();
                changed = true;
            }
            FKey::Character(c) if ctrl && (c == "a" || c == "A") => {
                self.select_all();
                changed = true;
            }
            _ => {
                if !ctrl {
                    if let Some(s) = &kev.text {
                        for c in s.chars() {
                            if !c.is_control() {
                                self.insert_char(c, text);
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
        if changed {
            crate::host::EventResponse::Handled
        } else {
            crate::host::EventResponse::Pass
        }
    }
}
