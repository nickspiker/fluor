//! One of several — a row (or a column) of buttons of which one is in force and wears the held fill: a way of travel, a mode, a tool. The app reads which ([`Choice::selected`]) and the change edge ([`Choice::take_change`]), as it reads a button's click.

use crate::canvas::Canvas;
use crate::pixel::{Argb8, Pixel};
use crate::coord::Coord;
use crate::paint::{Clip, HitId};
use crate::region::Region;
use crate::text::TextRenderer;
use crate::theme;
use crate::widgets::Button;
use alloc::string::String;
use alloc::vec::Vec;

pub struct Choice<P: Pixel = Argb8> {
    buttons: Vec<Button<P>>,
    selected: usize,
    /// Down a column rather than along a row.
    vertical: bool,
    /// The buttons' rect, shared out among them with a gap of a third of the font between.
    pub x: Coord,
    pub y: Coord,
    pub width: Coord,
    pub height: Coord,
    pub font_size: Coord,
    change_counter: u32,
    last_seen_change_counter: u32,
}

impl Choice {
    /// A button per label, each with its own [`HitId`]; `selected` the one in force. 8-bit; [`Choice::with_depth`] builds one at another pixel depth.
    pub fn new(hit_counter: &mut HitId, labels: &[&str], selected: usize, font_size: Coord) -> Self {
        Self::with_depth(hit_counter, labels, selected, font_size)
    }
}

impl<P: Pixel> Choice<P> {
    /// The same control at depth `P`.
    pub fn with_depth(hit_counter: &mut HitId, labels: &[&str], selected: usize, font_size: Coord) -> Self {
        let buttons = labels.iter().map(|l| Button::with_depth(hit_counter, 0.0, 0.0, 10.0, 10.0, font_size, *l)).collect();
        Choice { buttons, selected, vertical: false, x: 0.0, y: 0.0, width: 10.0, height: 10.0, font_size, change_counter: 0, last_seen_change_counter: 0 }
    }

    pub fn set_vertical(&mut self, vertical: bool) {
        self.vertical = vertical;
    }

    pub fn len(&self) -> usize {
        self.buttons.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buttons.is_empty()
    }

    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Put in force from the app's side, without a change edge: a setting mirrored into the control must not echo back.
    pub fn set_selected(&mut self, selected: usize) {
        self.selected = selected;
    }

    pub fn labels(&self) -> Vec<String> {
        self.buttons.iter().map(|b| b.label().into()).collect()
    }

    /// New words, as many buttons as before or the pool grown or cut to fit (new ids come from `hit_counter`).
    pub fn set_labels(&mut self, hit_counter: &mut HitId, labels: &[&str]) {
        self.buttons.truncate(labels.len());
        for (i, l) in labels.iter().enumerate() {
            match self.buttons.get_mut(i) {
                Some(b) => b.set_label(*l),
                None => self.buttons.push(Button::with_depth(hit_counter, 0.0, 0.0, 10.0, 10.0, self.font_size, *l)),
            }
        }
        if self.selected >= self.buttons.len() && !self.buttons.is_empty() {
            self.selected = self.buttons.len() - 1;
        }
    }

    /// The rect the buttons share, top left, width and height.
    pub fn set_rect(&mut self, x: Coord, y: Coord, width: Coord, height: Coord) {
        self.x = x;
        self.y = y;
        self.width = width;
        self.height = height;
    }

    pub fn set_font_size(&mut self, font_size: Coord) {
        self.font_size = font_size;
        for b in &mut self.buttons {
            b.set_font_size(font_size);
        }
    }

    pub fn bbox(&self) -> Region {
        Region::new(self.x, self.y, self.width, self.height)
    }

    /// Which button a point of the viewport is on.
    pub fn index_at(&self, px: Coord, py: Coord) -> Option<usize> {
        self.buttons.iter().position(|b| b.bbox().contains(px, py))
    }

    /// A tap at a point: the button there comes into force, a change edge if it was not. For apps that match taps to boxes themselves; the host's dispatch reaches the buttons through [`crate::host::widget::Container`] and [`Self::poll`].
    pub fn click_at(&mut self, px: Coord, py: Coord) -> bool {
        match self.index_at(px, py) {
            Some(i) => {
                self.choose(i);
                true
            }
            None => false,
        }
    }

    fn choose(&mut self, i: usize) {
        if i != self.selected {
            self.selected = i;
            self.change_counter += 1;
        }
    }

    /// Under the host's dispatch the buttons fire on their own; this notices which did and puts it in force. Call each tick.
    pub fn poll(&mut self) {
        for i in 0..self.buttons.len() {
            if self.buttons[i].take_click() {
                self.choose(i);
            }
        }
    }

    /// The button newly in force since the last poll, once.
    pub fn take_change(&mut self) -> Option<usize> {
        if self.change_counter != self.last_seen_change_counter {
            self.last_seen_change_counter = self.change_counter;
            Some(self.selected)
        } else {
            None
        }
    }

    /// Lay the buttons out in the rect and paint them, the one in force on the held fill.
    pub fn render_content_into(&mut self, canvas: &mut Canvas<P>, text: &mut TextRenderer, clip: Option<Clip>, mut hit_map: Option<&mut [HitId]>) {
        let n = self.buttons.len();
        if n == 0 {
            return;
        }
        let gap = self.font_size * 0.35;
        let span = if self.vertical { self.height } else { self.width };
        let each = (span - gap * (n as Coord - 1.0)) / n as Coord;
        for (i, b) in self.buttons.iter_mut().enumerate() {
            let along = (each + gap) * i as Coord;
            let (cx, cy, w, h) = if self.vertical { (self.x + self.width * 0.5, self.y + along + each * 0.5, self.width, each) } else { (self.x + along + each * 0.5, self.y + self.height * 0.5, each, self.height) };
            b.set_rect(cx, cy, w, h);
            b.set_fill((i == self.selected).then_some(theme::BUTTON_HELD));
            let id = b.hit_id();
            b.render_content_into(canvas, 0.0, 0.0, text, clip, hit_map.as_deref_mut(), id);
        }
    }
}

impl<P: Pixel> crate::host::widget::Container for Choice<P> {
    fn visit(&mut self, f: &mut dyn FnMut(&mut dyn crate::host::widget::Widget)) {
        for b in &mut self.buttons {
            f(b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Choice;
    use crate::paint::HIT_NONE;
    use crate::text::TextRenderer;

    #[test]
    fn a_tap_puts_one_in_force_once() {
        let mut hits = HIT_NONE;
        let mut c = Choice::new(&mut hits, &["Car", "Bike", "Foot"], 0, 10.0);
        c.set_rect(0.0, 0.0, 300.0, 20.0);
        // Laid out at paint; a paint into a small canvas places the buttons.
        let mut buf = vec![0u32; 300 * 20];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 300, 20, &mut damage);
        let mut text = TextRenderer::new();
        c.render_content_into(&mut canvas, &mut text, None, None);
        assert_eq!(c.index_at(150.0, 10.0), Some(1));
        assert!(c.click_at(150.0, 10.0));
        assert_eq!(c.selected(), 1);
        assert_eq!(c.take_change(), Some(1));
        assert_eq!(c.take_change(), None, "once");
        assert!(c.click_at(150.0, 10.0), "the same again is a tap but no change");
        assert_eq!(c.take_change(), None);
        c.set_selected(2);
        assert_eq!(c.take_change(), None, "a programmatic set does not echo");
        c.set_vertical(true);
        c.set_rect(0.0, 0.0, 100.0, 90.0);
        c.render_content_into(&mut canvas, &mut text, None, None);
        assert_eq!(c.index_at(50.0, 80.0), Some(2));
    }
}
