//! Dropdown (select) widget — a Button-family pill showing the current choice plus a chevron; clicking (or Enter/Space when focused) opens a popup list of options. Same squircle silhouette, two-tone AA edge, focus glow, and hover-tint-via-overlay conventions as [`super::Button`] so the family reads coherently.
//!
//! **Row dispatch.** Each option row owns its own dense [`HitId`] ([`DropdownRow`], a tiny [`Widget`] impl), mirroring how [`crate::host::chrome_widget::DefaultChrome`] owns four `ChromeButton`s. Per-row hover tints, click routing, and hit-testing all ride the existing id-indexed machinery for free. The app's [`crate::host::widget::Container::visit`] should hand out the dropdown itself (the pill) and then call [`Dropdown::visit_rows`] — rows only appear in the walk while the popup is open, so they never enter the tab cycle or overlay table when closed.
//!
//! **Front-to-back placement.** fluor composites under-blend (first writer wins), so the OPEN popup must be the first content painted into the frame: call [`Dropdown::render_popup_into`] at the top of the app's render, before other widgets and before the chrome flatten. Everything painted afterwards composes underneath it — no z-order machinery needed. The closed pill renders in normal widget order via [`Dropdown::render_content_into`].
//!
//! **Action model.** Poll-based like Button/Slider: row clicks and keyboard commits accumulate into a change counter; the app calls [`Dropdown::take_change`] after dispatch (and/or in `tick`) — `Some(index)` means the selection changed since last poll. `take_change` is also where row-click commits are folded into `selected` (the row widgets can't reach back into the parent during the visit walk).

use crate::canvas::PixelRect;
use crate::coord::Coord;
use crate::paint::{self, Clip, HitId};
use crate::region::Region;
use crate::text::TextRenderer;
use crate::theme;
use crate::widgets::textbox::{blit_cache_to_target, region_to_pixelrect};
use alloc::string::String;
use alloc::vec::Vec;

/// Gap between the pill's bottom edge and the popup's top edge, in stroke widths.
const POPUP_GAP_PX: Coord = 2.0;

/// One option row. A minimal [`crate::host::widget::Widget`]: carries its own hit id (stamped over the row band while the popup is open), hover state (drives the overlay tint), and a fired flag the parent folds into `selected` at [`Dropdown::take_change`] time.
pub struct DropdownRow {
    hit_id: HitId,
    /// Index into the parent's `options`. Fixed at construction.
    index: usize,
    hovered: bool,
    fired: bool,
}

impl DropdownRow {
    pub fn hit_id(&self) -> HitId {
        self.hit_id
    }
    pub fn index(&self) -> usize {
        self.index
    }
}

/// Discrete-choice selector. Closed: a pill with the selected option's label and a chevron. Open: a popup listing every option, one hoverable/clickable row each.
pub struct Dropdown {
    /// The pill's hit id (allocated first, then one per row, all from the same counter).
    hit_id: HitId,
    options: Vec<String>,
    rows: Vec<DropdownRow>,
    selected: usize,
    /// Keyboard highlight while open — mirrored into the rows' `hovered` flags so the overlay pass tints it exactly like a mouse hover.
    highlight: usize,
    open: bool,
    font: &'static str,

    /// Stroke thickness as a fraction of `font_size` — same convention as Button/Textbox.
    pub stroke_ru: f32,
    pub center_x: Coord,
    pub center_y: Coord,
    pub width: Coord,
    pub height: Coord,
    pub font_size: Coord,

    focused: bool,
    hovered: bool,
    enabled: bool,

    change_counter: u32,
    last_seen_change_counter: u32,

    // --- Pill cache (closed state): squircle bg + AA edges; label + chevron in text cache. Same layering as Button. ---
    pill_cache: Vec<u32>,
    pill_cache_w: usize,
    pill_cache_h: usize,
    pill_cache_dirty: bool,
    text_cache: Vec<u32>,
    text_cache_dirty: bool,
    inner_pill_mask: Vec<u8>,
    /// `true` when the caches were rasterized with the disabled (dimmed) palette — so enable/disable transitions re-rasterize.
    cache_disabled: bool,

    // --- Damage protocol ---
    last_painted_bbox: Option<PixelRect>,
    last_painted_focused: bool,
    /// Popup rect painted last frame (viewport pixels), if the popup was open. Unioned into `damage_rect` so both the open transition (paint new popup) and the close transition (clear stale popup pixels) get exactly one frame of damage.
    last_painted_popup: Option<PixelRect>,
    /// Whether anything about the popup (open state, highlight, selection) changed since the last paint.
    popup_dirty: bool,
}

impl Dropdown {
    /// `hit_counter` is the app's monotonic [`HitId`] allocator. The dropdown claims `1 + options.len()` ids: the pill first, then one per row in option order.
    pub fn new(
        hit_counter: &mut HitId,
        center_x: Coord,
        center_y: Coord,
        width: Coord,
        height: Coord,
        font_size: Coord,
        options: Vec<String>,
    ) -> Self {
        assert!(!options.is_empty(), "Dropdown requires at least one option");
        let hit_id = crate::host::widget::next_id(hit_counter);
        let rows = (0..options.len())
            .map(|index| DropdownRow {
                hit_id: crate::host::widget::next_id(hit_counter),
                index,
                hovered: false,
                fired: false,
            })
            .collect();
        Self {
            hit_id,
            options,
            rows,
            selected: 0,
            highlight: 0,
            open: false,
            font: "Open Sans",
            stroke_ru: 1.0 / (1 << 5) as f32,
            center_x,
            center_y,
            width,
            height,
            font_size,
            focused: false,
            hovered: false,
            enabled: true,
            change_counter: 0,
            last_seen_change_counter: 0,
            pill_cache: Vec::new(),
            pill_cache_w: 0,
            pill_cache_h: 0,
            pill_cache_dirty: true,
            text_cache: Vec::new(),
            text_cache_dirty: true,
            inner_pill_mask: Vec::new(),
            cache_disabled: false,
            last_painted_bbox: None,
            last_painted_focused: false,
            last_painted_popup: None,
            popup_dirty: false,
        }
    }

    pub fn hit_id(&self) -> HitId {
        self.hit_id
    }
    pub fn is_open(&self) -> bool {
        self.open
    }
    pub fn is_focused(&self) -> bool {
        self.focused
    }
    pub fn is_hovered(&self) -> bool {
        self.hovered
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    pub fn selected(&self) -> usize {
        self.selected
    }
    pub fn selected_label(&self) -> &str {
        &self.options[self.selected]
    }
    pub fn options(&self) -> &[String] {
        &self.options
    }

    /// True when `hit` is the pill or any row of this dropdown.
    pub fn owns_hit(&self, hit: HitId) -> bool {
        hit != crate::paint::HIT_NONE
            && (hit == self.hit_id || self.rows.iter().any(|r| r.hit_id == hit))
    }

    /// Programmatic selection (no change-counter bump — that's for user actions).
    pub fn set_selected(&mut self, index: usize) {
        let index = index.min(self.options.len() - 1);
        if index != self.selected {
            self.selected = index;
            self.text_cache_dirty = true;
            self.popup_dirty = true;
        }
    }

    pub fn set_focused(&mut self, focused: bool) {
        if focused != self.focused {
            self.focused = focused;
            // Focus loss closes without commit — covers Esc-at-app-level, click-elsewhere, tab-away.
            if !focused {
                self.close();
            }
        }
    }

    pub fn set_hovered(&mut self, hovered: bool) {
        if !self.enabled {
            return;
        }
        if hovered != self.hovered {
            self.hovered = hovered;
        }
    }

    /// Enable / disable. Disabled drops out of dispatch (capability accessors return `None`), closes the popup, and re-rasterizes with a dimmed label so the user can see it's inert (Slider-style greying — unlike Button, a disabled selector reads wrong if it stays full-brightness).
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled == self.enabled {
            return;
        }
        self.enabled = enabled;
        if !enabled {
            self.focused = false;
            self.hovered = false;
            self.close();
        }
        self.text_cache_dirty = true;
    }

    /// Open the popup with the keyboard highlight synced to the current selection.
    pub fn open(&mut self) {
        if self.open || !self.enabled {
            return;
        }
        self.open = true;
        self.highlight = self.selected;
        self.sync_row_highlight();
        self.popup_dirty = true;
    }

    /// Close the popup without committing. Idempotent.
    pub fn close(&mut self) {
        if !self.open {
            return;
        }
        self.open = false;
        for r in self.rows.iter_mut() {
            r.hovered = false;
            r.fired = false;
        }
        self.popup_dirty = true;
    }

    fn toggle_open(&mut self) {
        if self.open {
            self.close();
        } else {
            self.open();
        }
    }

    fn commit(&mut self, index: usize) {
        let index = index.min(self.options.len() - 1);
        if index != self.selected {
            self.selected = index;
            self.text_cache_dirty = true;
        }
        // Commit always counts as a change event, even re-selecting the same option — apps that re-run an action on select can distinguish via selected() if they care.
        self.change_counter = self.change_counter.wrapping_add(1);
        self.close();
    }

    /// Mirror `highlight` into the rows' hovered flags so the overlay pass tints the keyboard highlight identically to a mouse hover.
    fn sync_row_highlight(&mut self) {
        for r in self.rows.iter_mut() {
            r.hovered = r.index == self.highlight;
        }
    }

    /// Drive hover state for the pill AND the rows from the frame's hit result. Call from the app's CursorMoved path with whatever `hit_at`/hit-map lookup produced. Returns `true` if any hover state changed (request a redraw — row hover also feeds `highlight` so mouse and keyboard stay coherent).
    pub fn sync_hover(&mut self, hit: HitId) -> bool {
        if !self.enabled {
            return false;
        }
        let mut changed = false;
        let pill_hover = hit == self.hit_id;
        if pill_hover != self.hovered {
            self.hovered = pill_hover;
            changed = true;
        }
        if self.open {
            for i in 0..self.rows.len() {
                let want = self.rows[i].hit_id == hit;
                if self.rows[i].hovered != want {
                    self.rows[i].hovered = want;
                    changed = true;
                    if want {
                        self.highlight = self.rows[i].index;
                    }
                }
            }
            if changed {
                self.popup_dirty = true;
            }
        }
        changed
    }

    /// Fold pending row clicks into the selection, then report whether the selection changed since the last poll. Call after click dispatch and/or once per `tick`. Returns `Some(selected_index)` on a fresh change.
    pub fn take_change(&mut self) -> Option<usize> {
        if self.open {
            if let Some(idx) = self
                .rows
                .iter_mut()
                .find_map(|r| r.fired.then_some(r.index))
            {
                for r in self.rows.iter_mut() {
                    r.fired = false;
                }
                self.commit(idx);
            }
        }
        if self.change_counter != self.last_seen_change_counter {
            self.last_seen_change_counter = self.change_counter;
            Some(self.selected)
        } else {
            None
        }
    }

    /// Yield the option rows to a dispatch/overlay walk — only while open, so closed rows never enter hover tables or dispatch. Call right after handing the dropdown itself to the same callback (see module docs).
    pub fn visit_rows(&mut self, f: &mut dyn FnMut(&mut dyn crate::host::widget::Widget)) {
        if !self.open || !self.enabled {
            return;
        }
        for r in self.rows.iter_mut() {
            f(r);
        }
    }

    pub fn set_rect(&mut self, center_x: Coord, center_y: Coord, width: Coord, height: Coord) {
        if self.center_x != center_x
            || self.center_y != center_y
            || self.width != width
            || self.height != height
        {
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
            self.popup_dirty = true;
        }
        self.center_x = center_x;
        self.center_y = center_y;
        self.width = width;
        self.height = height;
    }

    pub fn set_font_size(&mut self, font_size: Coord) {
        if self.font_size != font_size {
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
            self.popup_dirty = true;
        }
        self.font_size = font_size;
    }

    pub fn bbox(&self) -> Region {
        Region::new(
            self.center_x - self.width * 0.5,
            self.center_y - self.height * 0.5,
            self.width,
            self.height,
        )
    }

    fn row_height(&self) -> Coord {
        self.font_size * 1.6
    }

    fn popup_padding(&self) -> Coord {
        self.font_size * 0.4
    }

    /// Popup rectangle in viewport pixels. Opens downward from the pill; flips upward when the viewport bottom would clip it. Deterministic from widget state + viewport height, so `damage_rect` and the painter agree without stored geometry.
    pub fn popup_region(&self, viewport_h: usize) -> Region {
        let pad = self.popup_padding();
        let h = self.rows.len() as Coord * self.row_height() + pad * 2.0;
        let w = self.width;
        let x = self.center_x - w * 0.5;
        let below_y = self.center_y + self.height * 0.5 + POPUP_GAP_PX;
        let y = if below_y + h > viewport_h as Coord {
            (self.center_y - self.height * 0.5 - POPUP_GAP_PX - h).max(0.0)
        } else {
            below_y
        };
        Region::new(x, y, w, h)
    }

    /// Damage region. `None` when nothing changed since the last paint. Unions the pill bbox (or glow envelope on focus transitions) with the popup rect on any frame where the popup is, or was, visible-and-dirty.
    pub fn damage_rect(&self, viewport_w: usize, viewport_h: usize) -> Option<PixelRect> {
        let focus_changed = self.focused != self.last_painted_focused;
        let pill_dirty = self.pill_cache_dirty || self.text_cache_dirty;
        if !pill_dirty && !focus_changed && !self.popup_dirty && self.last_painted_bbox.is_some() {
            return None;
        }
        let mut combined: Option<PixelRect> = self.last_painted_bbox;
        let current = if focus_changed {
            region_to_pixelrect(self.glow_bbox(), viewport_w, viewport_h)
        } else {
            region_to_pixelrect(self.bbox(), viewport_w, viewport_h)
        };
        combined = Some(combined.map_or(current, |c| c.union(current)));
        if self.popup_dirty {
            // Cover both where the popup was (clear on close) and where it will be (paint on open/highlight move).
            if let Some(prev) = self.last_painted_popup {
                combined = Some(combined.map_or(prev, |c| c.union(prev)));
            }
            if self.open {
                let cur = region_to_pixelrect(self.popup_region(viewport_h), viewport_w, viewport_h);
                combined = Some(combined.map_or(cur, |c| c.union(cur)));
            }
        }
        combined
    }

    /// Glow envelope — same reach math as Button so the focus glow damage matches what gets painted.
    pub fn glow_bbox(&self) -> Region {
        let horiz_factor = glow_factor_256(self.font_size, 1.5);
        let vert_factor = glow_factor_256(self.font_size, 0.75);
        let horiz_pad = paint::ray_reach_px(0x80, horiz_factor) as f32;
        let vert_pad = paint::ray_reach_px(0x40, vert_factor) as f32;
        Region::new(
            self.center_x - self.width * 0.5 - horiz_pad,
            self.center_y - self.height * 0.5 - vert_pad,
            self.width + 2.0 * horiz_pad,
            self.height + 2.0 * vert_pad,
        )
    }

    fn text_colour(&self) -> u32 {
        if self.enabled {
            theme::TEXTBOX_TEXT
        } else {
            // Greyed: keep the darkness payload, halve the α so the label visibly dims.
            (theme::TEXTBOX_TEXT & 0x00FF_FFFF) | 0x8000_0000
        }
    }

    /// Paint the OPEN popup into `canvas`, stamping each row's hit id over its band. MUST be called before any content the popup should cover (top of the app's render — see module docs). No-op when closed or disabled.
    pub fn render_popup_into(
        &mut self,
        canvas: &mut crate::canvas::Canvas,
        text: &mut TextRenderer,
        clip: Option<Clip>,
        mut hit_map: Option<&mut [HitId]>,
    ) {
        if !self.open || !self.enabled {
            // Still reconcile damage bookkeeping on the close frame.
            if self.popup_dirty && !self.open {
                self.last_painted_popup = None;
                self.popup_dirty = false;
            }
            return;
        }
        let viewport_w = canvas.width;
        let viewport_h = canvas.height;
        let region = self.popup_region(viewport_h);
        let x = region.x as isize;
        let y = region.y as isize;
        let w = region.w as isize;
        let h = region.h as isize;
        if w <= 0 || h <= 0 {
            return;
        }
        let stroke_px = (self.stroke_ru * self.font_size) as isize + 1;
        let squirdleyness = 3.0;
        let pad = self.popup_padding();
        let row_h = self.row_height();

        paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Rows first (topmost-first doctrine): marker dot + label, then hit-band stamp; the pill fill composes under them at the end.
        let text_left = region.x + pad * 2.2;
        for (i, label) in self.options.iter().enumerate() {
            let row_top = region.y + pad + i as Coord * row_h;
            let row_cy = row_top + row_h * 0.5;
            if i == self.selected {
                paint::draw_circle(
                    canvas,
                    region.x + pad * 1.2,
                    row_cy,
                    self.font_size * 0.14,
                    theme::TEXTBOX_TEXT,
                    clip,
                );
            }
            text.draw_text_left_u32(
                canvas,
                label,
                text_left,
                row_cy,
                self.font_size,
                400,
                theme::TEXTBOX_TEXT,
                self.font,
                clip,
                None,
                None,
            );
            // Stamp the row's hit band (interior of the popup only, clear of the AA edge).
            if let Some(map) = hit_map.as_deref_mut() {
                let x0 = (x + stroke_px).max(0) as usize;
                let x1 = ((x + w - stroke_px).max(0) as usize).min(viewport_w);
                let y0 = (row_top as isize).max(0) as usize;
                let y1 = ((row_top + row_h) as isize).max(0) as usize;
                let y1 = y1.min(viewport_h);
                for yy in y0..y1 {
                    let base = yy * viewport_w;
                    for xx in x0..x1 {
                        map[base + xx] = self.rows[i].hit_id;
                    }
                }
            }
        }

        // Popup pill: opaque fill + two-tone edge, drawn LAST so text/marker won the pixels they need and the fill claims the rest.
        paint::draw_squircle_pill_f(
            canvas,
            x + stroke_px,
            y + stroke_px,
            (w - 2 * stroke_px).max(0),
            (h - 2 * stroke_px).max(0),
            theme::TEXTBOX_FILL,
            squirdleyness,
        );
        paint::draw_squircle_pill_two_tone_f(
            canvas,
            x,
            y,
            w,
            h,
            theme::TEXTBOX_LIGHT_EDGE,
            theme::TEXTBOX_SHADOW_EDGE,
            squirdleyness,
            None,
            0,
        );

        self.last_painted_popup = Some(region_to_pixelrect(region, viewport_w, viewport_h));
        self.popup_dirty = false;
    }

    /// Paint the closed-state pill (selected label + chevron) in normal widget order, stamping the pill's hit id. Mirrors [`super::Button::render_content_into`].
    pub fn render_content_into(
        &mut self,
        canvas: &mut crate::canvas::Canvas,
        offset_x: Coord,
        offset_y: Coord,
        text: &mut TextRenderer,
        clip: Option<Clip>,
        hit_map: Option<&mut [HitId]>,
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
        let squirdleyness = 1.75;
        let stroke_px = (self.stroke_ru * self.font_size) as isize + 1;

        if self.cache_disabled != !self.enabled {
            self.pill_cache_dirty = true;
            self.text_cache_dirty = true;
        }

        // --- Pill cache: same fill/edge family as Button. ---
        if self.pill_cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.pill_cache.clear();
            self.pill_cache.resize(cw * ch, 0);
            self.pill_cache_w = cw;
            self.pill_cache_h = ch;
            let mut cache_damage = crate::canvas::Damage::new();
            {
                let mut cache_canvas =
                    crate::canvas::Canvas::new(&mut self.pill_cache, cw, ch, &mut cache_damage);
                let inner_w = (pill_w - 2 * stroke_px).max(0);
                let inner_h = (pill_h - 2 * stroke_px).max(0);
                if inner_w > 0 && inner_h > 0 {
                    paint::draw_squircle_pill_f(
                        &mut cache_canvas,
                        stroke_px,
                        stroke_px,
                        inner_w,
                        inner_h,
                        theme::BUTTON_FILL,
                        squirdleyness,
                    );
                }
            }
            self.inner_pill_mask.clear();
            self.inner_pill_mask.resize(cw * ch, 0);
            for i in 0..(cw * ch) {
                self.inner_pill_mask[i] = ((self.pill_cache[i] >> 24) & 0xFF) as u8;
            }
            {
                let mut cache_canvas =
                    crate::canvas::Canvas::new(&mut self.pill_cache, cw, ch, &mut cache_damage);
                paint::draw_squircle_pill_two_tone_f(
                    &mut cache_canvas,
                    0,
                    0,
                    pill_w,
                    pill_h,
                    theme::TEXTBOX_SHADOW_EDGE,
                    theme::TEXTBOX_LIGHT_EDGE,
                    squirdleyness,
                    None,
                    0,
                );
            }
            self.pill_cache_dirty = false;
        }

        // --- Text cache: selected label (left-aligned) + chevron (right side), clipped by the inner-pill mask. ---
        if self.text_cache_dirty {
            paint::RASTERIZE_OPS.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            self.text_cache.clear();
            self.text_cache.resize(cw * ch, 0);
            let colour = self.text_colour();
            let mut text_damage = crate::canvas::Damage::new();
            let mut text_canvas =
                crate::canvas::Canvas::new(&mut self.text_cache, cw, ch, &mut text_damage);
            let local_y_center = pill_h as Coord * 0.5;
            let pad = self.font_size * 0.4;
            if self.font_size > 0.0 {
                let mask_buffer = paint::AlphaMask::new(&self.inner_pill_mask, cw, ch);
                text.draw_text_left_u32(
                    &mut text_canvas,
                    &self.options[self.selected],
                    pad * 1.5,
                    local_y_center,
                    self.font_size,
                    400,
                    colour,
                    self.font,
                    None,
                    Some(&mask_buffer),
                    None,
                );
            }
            // Chevron: two 45° strokes forming a "v", right-aligned. Drawn geometrically (rotated rects) rather than as a glyph so it renders identically regardless of font coverage.
            let chev_w = self.font_size * 0.55;
            let chev_cx = pill_w as Coord - pad * 1.5 - chev_w * 0.5;
            let chev_cy = local_y_center;
            let seg_len = chev_w * 0.5 * core::f32::consts::SQRT_2;
            let seg_th = (self.stroke_ru * self.font_size + 1.5).max(1.5);
            let angle = core::f32::consts::FRAC_PI_4;
            paint::draw_rect_rotated(
                &mut text_canvas,
                chev_cx - chev_w * 0.25,
                chev_cy,
                seg_len,
                seg_th,
                angle,
                colour,
                None,
            );
            paint::draw_rect_rotated(
                &mut text_canvas,
                chev_cx + chev_w * 0.25,
                chev_cy,
                seg_len,
                seg_th,
                -angle,
                colour,
                None,
            );
            self.text_cache_dirty = false;
            self.cache_disabled = !self.enabled;
        }

        // --- Composition: text/chevron (topmost) → pill (stamps hit map). ---
        blit_cache_to_target(
            &self.text_cache,
            cw,
            ch,
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
            self.hit_id,
            clip,
        );

        // --- Focus glow, Button-identical. ---
        if self.focused {
            let horiz_factor = glow_factor_256(self.font_size, 1.5);
            let vert_factor = glow_factor_256(self.font_size, 0.75);
            paint::apply_textbox_glow_right(
                canvas, pill_x_target, pill_y_target, pill_w, pill_h,
                theme::GLOW_DARK, 0x80, horiz_factor, clip,
            );
            paint::apply_textbox_glow_left(
                canvas, pill_x_target, pill_y_target, pill_w, pill_h,
                theme::GLOW_DARK, 0x80, horiz_factor, clip,
            );
            paint::apply_textbox_glow_top(
                canvas, pill_x_target, pill_y_target, pill_w, pill_h,
                theme::GLOW_DARK, 0x40, vert_factor, clip,
            );
            paint::apply_textbox_glow_bottom(
                canvas, pill_x_target, pill_y_target, pill_w, pill_h,
                theme::GLOW_DARK, 0x40, vert_factor, clip,
            );
        }

        self.last_painted_bbox = Some(region_to_pixelrect(
            self.bbox(),
            canvas.width,
            canvas.height,
        ));
        self.last_painted_focused = self.focused;
    }
}

/// Same RU-invariant glow factor as Button's.
fn glow_factor_256(font_size: f32, radius_scale: f32) -> u32 {
    let target_radius = (font_size * radius_scale).max(8.0);
    let drop = (1240.0 / target_radius) as u32;
    (256u32.saturating_sub(drop)).clamp(96, 254)
}

mod widget_impls {
    //! Capability traits. The Dropdown itself is the pill widget (click toggles, keyboard navigates/commits); each [`DropdownRow`] is a click+hover widget that records a fired flag for the parent to fold in at [`Dropdown::take_change`].

    use super::{Dropdown, DropdownRow};
    use crate::coord::Coord;
    use crate::event::{ElementState, Key as FKey, KeyEvent, ModifiersState, NamedKey};
    use crate::host::widget::{Click, Focus, Hover, Key, PaintCtx, Widget};
    use crate::paint::HitId;
    use crate::text::TextRenderer;

    impl Widget for Dropdown {
        fn id(&self) -> HitId {
            self.hit_id()
        }
        fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
            // Consumers drive rendering via render_content_into / render_popup_into — same convention as Button/Textbox.
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
    }

    impl Click for Dropdown {
        fn on_click(
            &mut self,
            _x: Coord,
            _y: Coord,
            _mods: ModifiersState,
        ) -> crate::host::EventResponse {
            self.toggle_open();
            crate::host::EventResponse::Handled
        }
    }

    impl Key for Dropdown {
        fn on_key(
            &mut self,
            kev: &KeyEvent,
            _mods: ModifiersState,
            _text: &mut TextRenderer,
        ) -> crate::host::EventResponse {
            if kev.state != ElementState::Pressed {
                return crate::host::EventResponse::Pass;
            }
            let n = self.options().len();
            match &kev.logical_key {
                FKey::Named(NamedKey::Enter) | FKey::Named(NamedKey::Space) => {
                    if self.is_open() {
                        let idx = self.highlight;
                        self.commit(idx);
                    } else {
                        self.open();
                    }
                    crate::host::EventResponse::Handled
                }
                FKey::Named(NamedKey::ArrowDown) => {
                    if self.is_open() {
                        self.highlight = (self.highlight + 1) % n;
                        self.sync_row_highlight();
                        self.popup_dirty = true;
                    } else {
                        let idx = (self.selected + 1) % n;
                        self.commit(idx);
                    }
                    crate::host::EventResponse::Handled
                }
                FKey::Named(NamedKey::ArrowUp) => {
                    if self.is_open() {
                        self.highlight = (self.highlight + n - 1) % n;
                        self.sync_row_highlight();
                        self.popup_dirty = true;
                    } else {
                        let idx = (self.selected + n - 1) % n;
                        self.commit(idx);
                    }
                    crate::host::EventResponse::Handled
                }
                FKey::Named(NamedKey::Escape) if self.is_open() => {
                    self.close();
                    crate::host::EventResponse::Handled
                }
                _ => crate::host::EventResponse::Pass,
            }
        }
    }

    impl Focus for Dropdown {
        fn set_focused(&mut self, focused: bool) {
            Dropdown::set_focused(self, focused);
        }
        fn focus_bbox(&self) -> Option<crate::canvas::PixelRect> {
            let r = self.bbox();
            Some(crate::canvas::PixelRect::new(
                r.x.max(0.0) as usize,
                r.y.max(0.0) as usize,
                (r.x + r.w).max(0.0) as usize,
                (r.y + r.h).max(0.0) as usize,
            ))
        }
    }

    impl Hover for Dropdown {
        fn set_hovered(&mut self, hovered: bool) {
            Dropdown::set_hovered(self, hovered);
        }
        fn tint_delta(&self) -> u32 {
            if self.is_focused() {
                crate::paint::wrap_sub_rgb(crate::theme::BUTTON_ACTIVE, crate::theme::BUTTON_FILL)
            } else if self.is_hovered() {
                crate::paint::wrap_sub_rgb(crate::theme::BUTTON_HOVER, crate::theme::BUTTON_FILL)
            } else {
                0
            }
        }
    }

    impl Widget for DropdownRow {
        fn id(&self) -> HitId {
            self.hit_id
        }
        fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
            // Rows are painted collectively by Dropdown::render_popup_into.
        }
        fn click(&mut self) -> Option<&mut dyn Click> {
            Some(self)
        }
        fn hover(&mut self) -> Option<&mut dyn Hover> {
            Some(self)
        }
    }

    impl Click for DropdownRow {
        fn on_click(
            &mut self,
            _x: Coord,
            _y: Coord,
            _mods: ModifiersState,
        ) -> crate::host::EventResponse {
            self.fired = true;
            crate::host::EventResponse::Handled
        }
    }

    impl Hover for DropdownRow {
        fn set_hovered(&mut self, hovered: bool) {
            self.hovered = hovered;
        }
        fn tint_delta(&self) -> u32 {
            if self.hovered {
                crate::paint::wrap_sub_rgb(crate::theme::TEXTBOX_HOVER, crate::theme::TEXTBOX_FILL)
            } else {
                0
            }
        }
    }
}
