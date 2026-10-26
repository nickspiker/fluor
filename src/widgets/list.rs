//! A list of rows — the one shape every app's lists share (photon's contacts and conversations, mahere's layers, areas, places found and turns): a title, a detail line under it when there is one, something at the left (a box to tick, a fold mark, a colour), something at the right (a chevron, a word, a colour), a step in from the edge for a row under a group, and one of them picked. The list scrolls inside its rect with the same spring as a textbox's overscroll, and a tap names its row ([`List::take_click`]) — the app decides what the row means, as it does for a button.

use crate::canvas::{Canvas, PixelRect};
use crate::coord::Coord;
use crate::paint::{self, Clip, HitId};
use crate::region::Region;
use crate::text::{TextRenderer, TextStyle};
use crate::theme;
use alloc::string::String;
use alloc::vec::Vec;

/// What sits at a row's left.
#[derive(Clone, Debug, PartialEq)]
pub enum Lead {
    None,
    /// A box, ticked or not: the row is a setting.
    Check(bool),
    /// A fold mark: the row is a group, open or closed.
    Fold(bool),
    /// A colour the row is about.
    Swatch(u32),
}

/// What sits at a row's right.
#[derive(Clone, Debug, PartialEq)]
pub enum Trail {
    None,
    /// The row opens a page.
    Chevron,
    /// A word or a value, dim.
    Text(String),
    /// A colour the row is about.
    Swatch(u32),
}

/// One row.
#[derive(Clone, Debug, PartialEq)]
pub struct ListRow {
    pub title: String,
    pub detail: Option<String>,
    pub lead: Lead,
    pub trail: Trail,
    /// Steps in from the list's edge, a group's rows one deeper than the group.
    pub indent: u8,
    /// Greyed: said, not offered (a setting a dominating one has switched off, a build there is no getting).
    pub dim: bool,
}

impl ListRow {
    pub fn new(title: impl Into<String>) -> Self {
        ListRow { title: title.into(), detail: None, lead: Lead::None, trail: Trail::None, indent: 0, dim: false }
    }
    pub fn detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
    pub fn lead(mut self, lead: Lead) -> Self {
        self.lead = lead;
        self
    }
    pub fn trail(mut self, trail: Trail) -> Self {
        self.trail = trail;
        self
    }
    pub fn indent(mut self, indent: u8) -> Self {
        self.indent = indent;
        self
    }
    pub fn dim(mut self, dim: bool) -> Self {
        self.dim = dim;
        self
    }
}

/// The rows in a rect, scrolled, one tapped at a time.
pub struct List {
    hit_id: HitId,
    rows: Vec<ListRow>,
    /// The rect's top left, width and height.
    pub x: Coord,
    pub y: Coord,
    pub width: Coord,
    pub height: Coord,
    pub font_size: Coord,
    /// How far the rows have scrolled up past the rect's top, pixels; past the ends while a finger overscrolls, sprung back by [`Self::spring_scroll`].
    scroll: Coord,
    /// The row picked, drawn on its own band.
    selected: Option<usize>,
    /// The row a tap named, until polled.
    clicked: Option<usize>,
    hovered: bool,
    pressed: bool,
}

impl List {
    /// Claims one [`HitId`] for the whole list; a tap's row is found by its place.
    pub fn new(hit_counter: &mut HitId, x: Coord, y: Coord, width: Coord, height: Coord, font_size: Coord) -> Self {
        List { hit_id: crate::host::widget::next_id(hit_counter), rows: Vec::new(), x, y, width, height, font_size, scroll: 0.0, selected: None, clicked: None, hovered: false, pressed: false }
    }

    pub fn hit_id(&self) -> HitId {
        self.hit_id
    }

    pub fn rows(&self) -> &[ListRow] {
        &self.rows
    }

    pub fn rows_mut(&mut self) -> &mut Vec<ListRow> {
        &mut self.rows
    }

    /// The rows, afresh. The scroll stays where it was so a list repainted with the same rows does not jump; it settles to the new extent on the next spring.
    pub fn set_rows(&mut self, rows: Vec<ListRow>) {
        self.rows = rows;
        if self.selected.is_some_and(|s| s >= self.rows.len()) {
            self.selected = None;
        }
    }

    pub fn set_rect(&mut self, x: Coord, y: Coord, width: Coord, height: Coord) {
        self.x = x;
        self.y = y;
        self.width = width;
        self.height = height;
    }

    pub fn set_font_size(&mut self, font_size: Coord) {
        self.font_size = font_size;
    }

    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    pub fn set_selected(&mut self, selected: Option<usize>) {
        self.selected = selected;
    }

    /// A row's height: its title's line, and its detail's under it.
    pub fn row_height(&self, row: &ListRow) -> Coord {
        self.font_size * if row.detail.is_some() { 2.5 } else { 1.6 }
    }

    /// Every row's height together.
    pub fn extent(&self) -> Coord {
        self.rows.iter().map(|r| self.row_height(r)).sum()
    }

    pub fn scroll(&self) -> Coord {
        self.scroll
    }

    /// The scroll put somewhere, unbounded: a finger's drag goes past the ends and the spring brings it back.
    pub fn pan_scroll_to(&mut self, scroll: Coord) -> bool {
        if self.scroll == scroll {
            return false;
        }
        self.scroll = scroll;
        true
    }

    pub fn nudge_scroll(&mut self, delta: Coord) -> bool {
        self.pan_scroll_to(self.scroll + delta)
    }

    /// The furthest the rows scroll: nothing when they fit, else the overflow.
    fn bound(&self) -> Coord {
        let over = self.extent() - self.height;
        if over > 0.0 { over } else { 0.0 }
    }

    /// The scroll sprung back toward its range by one tick, `over × decay` (the app passes e^(−kΔt)), the last third of a pixel snapped so it ends. True while it moves.
    pub fn spring_scroll(&mut self, decay: Coord) -> bool {
        let bound = self.bound();
        // The spring's rest point is the nearest end of the range; the range is the target, not a guard.
        let target = if self.scroll < 0.0 { 0.0 } else if self.scroll > bound { bound } else { self.scroll };
        let over = self.scroll - target;
        if over == 0.0 {
            return false;
        }
        let next = over * decay;
        self.scroll = if next.abs() < 0.3 { target } else { target + next };
        true
    }

    /// The row under a point of the viewport, if the point is in the list and on a row.
    pub fn row_at(&self, px: Coord, py: Coord) -> Option<usize> {
        if !self.bbox().contains(px, py) {
            return None;
        }
        let mut top = self.y - self.scroll;
        for (i, r) in self.rows.iter().enumerate() {
            let h = self.row_height(r);
            if py >= top && py < top + h {
                return Some(i);
            }
            top += h;
        }
        None
    }

    /// A row's rect in the viewport as painted now (it may lie outside the list's rect when scrolled away).
    pub fn row_rect(&self, i: usize) -> Region {
        let top = self.y - self.scroll + self.rows[..i].iter().map(|r| self.row_height(r)).sum::<Coord>();
        Region::new(self.x, top, self.width, self.row_height(&self.rows[i]))
    }

    pub fn bbox(&self) -> Region {
        Region::new(self.x, self.y, self.width, self.height)
    }

    /// The row a tap named since the last poll, once — the same edge as a button's click.
    pub fn take_click(&mut self) -> Option<usize> {
        self.clicked.take()
    }

    /// A tap at a point of the viewport: the row there is named for [`Self::take_click`]. For apps that match taps to boxes themselves; the host's dispatch comes here through [`crate::host::widget::Click`].
    pub fn click_at(&mut self, px: Coord, py: Coord) -> bool {
        match self.row_at(px, py) {
            Some(i) => {
                self.clicked = Some(i);
                true
            }
            None => false,
        }
    }

    pub fn set_hovered(&mut self, hovered: bool) {
        self.hovered = hovered;
    }

    pub fn set_pressed(&mut self, pressed: bool) {
        self.pressed = pressed;
    }

    /// Paint the rows that lie in the rect, topmost first (the picked row's band last, under its text), and stamp the hit id over the rect.
    pub fn render_content_into(&mut self, canvas: &mut Canvas, text: &mut TextRenderer, clip: Option<Clip>, mut hit_map: Option<&mut [HitId]>) {
        let font = self.font_size;
        // Everything this paints stays inside the list's rect, whatever clip the caller brought.
        let (cx0, cy0, cx1, cy1) = (self.x as usize, self.y as usize, (self.x + self.width) as usize, (self.y + self.height) as usize);
        let clip = Some(match clip {
            Some(c) => Clip::new(c.x_start.max(cx0), c.y_start.max(cy0), c.x_end.min(cx1), c.y_end.min(cy1)),
            None => Clip::new(cx0, cy0, cx1, cy1),
        });
        let title = TextStyle::new(font, theme::TEXTBOX_TEXT);
        let dimmed = TextStyle::new(font, theme::LABEL_COLOUR);
        let small = TextStyle::new(font * 0.8, theme::LABEL_COLOUR);
        let indent = font * 1.2;
        let mut top = self.y - self.scroll;
        for (i, row) in self.rows.iter().enumerate() {
            let h = self.row_height(row);
            if top + h >= self.y && top <= self.y + self.height {
                let x0 = self.x + font * 0.6 + indent * row.indent as Coord;
                let mid = top + font * 0.8;
                let style = if row.dim { &dimmed } else { &title };
                // The left: a box, a fold mark or a colour, and the title after it.
                let tx = match &row.lead {
                    Lead::None => x0,
                    Lead::Check(on) => {
                        Self::paint_box(canvas, x0, mid, font * 1.1, *on, clip);
                        x0 + font * 1.6
                    }
                    Lead::Fold(open) => {
                        text.draw_text_left(canvas, if *open { "\u{25be}" } else { "\u{25b8}" }, x0, mid, style, clip, None);
                        x0 + font * 1.1
                    }
                    Lead::Swatch(colour) => {
                        paint::fill_rect(canvas, x0 as isize, (mid - font * 0.45) as isize, (font * 0.9) as isize, (font * 0.9) as isize, *colour, clip, None);
                        x0 + font * 1.4
                    }
                };
                text.draw_text_left(canvas, &row.title, tx, mid, style, clip, None);
                if let Some(d) = &row.detail {
                    text.draw_text_left(canvas, d, tx + font * 0.3, mid + font * 0.95, &small, clip, None);
                }
                // The right: a chevron, a word or a colour.
                let right = self.x + self.width - font * 0.6;
                match &row.trail {
                    Trail::None => {}
                    Trail::Chevron => {
                        text.draw_text_right(canvas, "\u{203a}", right, mid, style, clip, None);
                    }
                    Trail::Text(s) => {
                        text.draw_text_right(canvas, s, right, mid, &small, clip, None);
                    }
                    Trail::Swatch(colour) => paint::fill_rect(canvas, (right - font * 1.4) as isize, (mid - font * 0.45) as isize, (font * 1.4) as isize, (font * 0.9) as isize, *colour, clip, None),
                }
                // The picked row's band under its text.
                if self.selected == Some(i) {
                    paint::fill_rect(canvas, self.x as isize, top as isize, self.width as isize, h as isize, theme::BUTTON_ACTIVE, clip, None);
                }
            }
            top += h;
        }
        if let Some(map) = hit_map.as_deref_mut() {
            let rect = PixelRect::new(cx0, cy0, cx1, cy1);
            let bw = canvas.width;
            let bh = canvas.height;
            // WHY/PROOF: the rect is the app's layout, past the canvas when a pane scrolls off an edge — the stamp would index outside the hit map.
            let x1 = rect.x1.min(bw);
            let y1 = rect.y1.min(bh);
            for y in rect.y0..y1 {
                let base = y * bw;
                for x in rect.x0..x1 {
                    map[base + x] = self.hit_id;
                }
            }
        }
    }

    /// A row's box: the checkbox's own square, ticked or not, at `side`.
    fn paint_box(canvas: &mut Canvas, x0: Coord, mid: Coord, side: Coord, on: bool, clip: Option<Clip>) {
        let y0 = mid - side * 0.5;
        // The `+ 1` floor every widget carries: a floorless ring vanishes under the fill below a 32 px font.
        let stroke = (side / 42.0) as isize + 1;
        if on {
            let cx = x0 + side * 0.5;
            // A bare line stroke with no fill behind it keeps a pixel so it does not vanish sub-pixel.
            let seg = (side / 15.0).max(1.0);
            paint::draw_rect_rotated(canvas, cx - side * 0.14, mid + side * 0.14, side * 0.32, seg, core::f32::consts::FRAC_PI_4, theme::TEXTBOX_TEXT, clip);
            paint::draw_rect_rotated(canvas, cx + side * 0.06, mid - side * 0.02, side * 0.5, seg, -core::f32::consts::FRAC_PI_4, theme::TEXTBOX_TEXT, clip);
        }
        let fill = if on { theme::BUTTON_FILL } else { theme::TEXTBOX_FILL };
        let inner = side as isize - 2 * stroke;
        let big = side * 0.32;
        if inner > 0 {
            paint::draw_squircle_rrect_f(canvas, x0 as isize + stroke, y0 as isize + stroke, inner, inner, fill, paint::asymmetric_radii(big - stroke as f32), paint::SHAPE_POWER as f32);
        }
        paint::draw_squircle_rrect_two_tone_f(canvas, x0 as isize, y0 as isize, side as isize, side as isize, theme::TEXTBOX_SHADOW_EDGE, theme::TEXTBOX_LIGHT_EDGE, paint::asymmetric_radii(big), paint::SHAPE_POWER as f32, None, 0);
    }
}

mod widget_impls {
    //! The capability traits, so a list rides the host's dispatch and hover like a button: a click names its row.

    use super::List;
    use crate::coord::Coord;
    use crate::event::ModifiersState;
    use crate::host::widget::{Click, Hover, PaintCtx, Widget};
    use crate::paint::HitId;

    impl Widget for List {
        fn id(&self) -> HitId {
            self.hit_id()
        }
        fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
            // The host drives painting via `render_content_into`, as for every widget.
        }
        fn click(&mut self) -> Option<&mut dyn Click> {
            Some(self)
        }
        fn hover(&mut self) -> Option<&mut dyn Hover> {
            Some(self)
        }
    }

    impl Click for List {
        fn on_click(&mut self, x: Coord, y: Coord, _mods: ModifiersState) -> crate::host::EventResponse {
            if self.click_at(x, y) { crate::host::EventResponse::Handled } else { crate::host::EventResponse::Pass }
        }
        fn set_pressed(&mut self, pressed: bool) {
            List::set_pressed(self, pressed);
        }
    }

    impl Hover for List {
        fn set_hovered(&mut self, hovered: bool) {
            List::set_hovered(self, hovered);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Lead, List, ListRow};
    use crate::paint::HIT_NONE;

    #[test]
    fn rows_are_found_by_their_place_and_scroll() {
        let mut hits = HIT_NONE;
        let mut list = List::new(&mut hits, 10.0, 20.0, 200.0, 100.0, 10.0);
        list.set_rows(vec![ListRow::new("a"), ListRow::new("b").detail("more"), ListRow::new("c").lead(Lead::Check(true))]);
        // 16 + 25 + 16 high.
        assert_eq!(list.extent(), 57.0);
        assert_eq!(list.row_at(50.0, 25.0), Some(0));
        assert_eq!(list.row_at(50.0, 40.0), Some(1));
        assert_eq!(list.row_at(50.0, 70.0), Some(2));
        assert_eq!(list.row_at(50.0, 100.0), None, "below the rows");
        assert_eq!(list.row_at(5.0, 25.0), None, "outside the rect");
        assert!(list.click_at(50.0, 40.0));
        assert_eq!(list.take_click(), Some(1));
        assert_eq!(list.take_click(), None, "once");
        // Scrolled by a row, the second row is at the top.
        list.pan_scroll_to(16.0);
        assert_eq!(list.row_at(50.0, 22.0), Some(1));
        assert_eq!(list.row_rect(1).y, 20.0);
        // The rows fit the rect: the spring brings any scroll back to nothing.
        let mut steps = 0;
        while list.spring_scroll(0.5) {
            steps += 1;
        }
        assert_eq!(list.scroll(), 0.0);
        assert!(steps > 0 && steps < 20);
    }

    #[test]
    fn a_tall_list_scrolls_only_as_far_as_its_overflow() {
        let mut hits = HIT_NONE;
        let mut list = List::new(&mut hits, 0.0, 0.0, 100.0, 50.0, 10.0);
        list.set_rows((0..10).map(|i| ListRow::new(format!("row {i}"))).collect());
        list.pan_scroll_to(500.0);
        while list.spring_scroll(0.5) {}
        assert_eq!(list.scroll(), 160.0 - 50.0);
        list.pan_scroll_to(-30.0);
        while list.spring_scroll(0.5) {}
        assert_eq!(list.scroll(), 0.0);
    }
}
