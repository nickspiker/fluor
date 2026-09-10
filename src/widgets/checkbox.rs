//! Labelled on/off checkbox — a small squircle box (ticked when on) with a label to its right. Toggles its own visual state on click / Space / Enter and exposes a poll-based change edge ([`Checkbox::take_toggle`]), matching the Button family's `take_click` idiom. Same squircle silhouette, two-tone AA edge, and no-floor stroke as Button / Textbox so the three read as one family.

use crate::canvas::{Canvas, PixelRect};
use crate::coord::Coord;
use crate::paint::{self, Clip, HitId, HIT_NONE};
use crate::region::Region;
use crate::text::TextRenderer;
use crate::text::TextStyle;
use crate::theme;
use alloc::string::String;

/// A labelled on/off box. The box sits at the left of the widget's rect; the label is drawn to its right. Click (or Space/Enter while focused) flips `checked` and bumps a change counter the app polls via [`Self::take_toggle`].
pub struct Checkbox {
    hit_id: HitId,
    label: String,
    checked: bool,
    /// Centre of the whole widget (box + label) rect.
    pub center_x: Coord,
    pub center_y: Coord,
    /// Full widget width including the label. The box itself is a square of side `height`.
    pub width: Coord,
    pub height: Coord,
    pub font_size: Coord,
    focused: bool,
    hovered: bool,
    change_counter: u32,
    last_seen_change_counter: u32,
    /// Optional fill override for the UNCHECKED box (e.g. an app scolding the user for unticking something opinionated); `None` = the standard textbox-dark.
    empty_fill: Option<u32>,
}

impl Checkbox {
    /// Claim one [`HitId`] from the app's monotonic allocator, like every other fluor widget.
    pub fn new(
        hit_counter: &mut HitId,
        label: impl Into<String>,
        center_x: Coord,
        center_y: Coord,
        width: Coord,
        height: Coord,
        font_size: Coord,
        checked: bool,
    ) -> Self {
        Self {
            hit_id: crate::host::widget::next_id(hit_counter),
            label: label.into(),
            checked,
            center_x,
            center_y,
            width,
            height,
            font_size,
            focused: false,
            hovered: false,
            change_counter: 0,
            last_seen_change_counter: 0,
            empty_fill: None,
        }
    }

    /// Override the unchecked-state box fill (`None` restores the default). The checked fill is never overridden — on always reads as the one action-blue.
    pub fn set_empty_fill(&mut self, fill: Option<u32>) {
        self.empty_fill = fill;
    }

    pub fn hit_id(&self) -> HitId {
        self.hit_id
    }
    pub fn is_checked(&self) -> bool {
        self.checked
    }
    pub fn is_focused(&self) -> bool {
        self.focused
    }
    pub fn is_hovered(&self) -> bool {
        self.hovered
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// The box is one square of `font_size × 1.3`, never the widget height: the height is the LABEL's, which may wrap to several lines (a scaled-up font on a phone), while the box stays a box.
    pub fn box_side(&self) -> Coord {
        (self.font_size * 1.3).min(self.height.max(self.font_size))
    }

    /// The label wrapped to the width left of the box — one entry per drawn line. Photon 2026-09-09: "Auto-attest on reboot", "Be a custodian", "Vibrate on incoming" all clipped at the pane edge under a large font because the label was one unbreakable line.
    pub fn label_lines(&self, text: &mut TextRenderer) -> Vec<String> {
        if self.label.is_empty() {
            return Vec::new();
        }
        let style = TextStyle::new(self.font_size, theme::TEXTBOX_TEXT);
        let max_w = (self.width - self.box_side() - self.font_size * 0.8).max(self.font_size);
        let mut lines = Vec::new();
        let mut cur = String::new();
        for word in self.label.split_whitespace() {
            let cand = if cur.is_empty() { word.to_string() } else { format!("{cur} {word}") };
            if !cur.is_empty() && text.measure_text(&cand, &style) > max_w {
                lines.push(core::mem::take(&mut cur));
                cur = word.to_string();
            } else {
                cur = cand;
            }
        }
        if !cur.is_empty() {
            lines.push(cur);
        }
        lines
    }

    /// The height this widget needs at its current width and font: the box, or the wrapped label if that is taller. Callers size their band from this BEFORE the final `set_rect`.
    pub fn needed_height(&self, text: &mut TextRenderer) -> Coord {
        let n = self.label_lines(text).len().max(1) as Coord;
        (self.font_size * 1.3).max(n * self.font_size * 1.25)
    }

    pub fn set_label(&mut self, label: impl Into<String>) {
        self.label = label.into();
    }

    pub fn set_rect(&mut self, center_x: Coord, center_y: Coord, width: Coord, height: Coord) {
        self.center_x = center_x;
        self.center_y = center_y;
        self.width = width;
        self.height = height;
    }

    pub fn set_font_size(&mut self, font_size: Coord) {
        self.font_size = font_size;
    }

    /// Returns `true` once per toggle since the last poll — the same rising-edge contract as `Button::take_click`.
    pub fn take_toggle(&mut self) -> bool {
        if self.change_counter != self.last_seen_change_counter {
            self.last_seen_change_counter = self.change_counter;
            true
        } else {
            false
        }
    }

    /// Flip the state and bump the change counter — called from the Click / Key impls.
    fn toggle(&mut self) {
        self.checked = !self.checked;
        self.change_counter = self.change_counter.wrapping_add(1);
    }

    /// Set the state programmatically (mirroring a synced setting into the UI) WITHOUT bumping the change counter — a programmatic set must never re-fire `take_toggle` and echo back into the setting it came from.
    pub fn set_checked(&mut self, checked: bool) {
        self.checked = checked;
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    pub fn set_hovered(&mut self, hovered: bool) {
        self.hovered = hovered;
    }

    pub fn bbox(&self) -> Region {
        Region::new(
            self.center_x - self.width * 0.5,
            self.center_y - self.height * 0.5,
            self.width,
            self.height,
        )
    }

    /// Paint the box (topmost-first: label glyphs, then tick, then the box fill that stamps the hit id) plus the label. Follows the same under-blend doctrine as Button — content the box should sit under is painted before the box fill claims its pixels.
    pub fn render_content_into(
        &mut self,
        canvas: &mut Canvas,
        text: &mut TextRenderer,
        clip: Option<Clip>,
        mut hit_map: Option<&mut [HitId]>,
    ) {
        let side = self.box_side();
        let box_x0 = self.center_x - self.width * 0.5;
        let box_y0 = self.center_y - side * 0.5;
        let stroke = (self.font_size / 32.0) as isize + 1; // the `+ 1` floor every other widget carries (a floorless ring vanishes under the fill below a 32 px font)

        // Tick first (topmost) when checked, then the two-tone edge + fill so the fill claims the rest of the box.
        if self.checked {
            let inset = side * 0.28;
            // A simple check-mark from two rotated strokes.
            let cx = box_x0 + side * 0.5;
            let cy = self.center_y;
            let seg_th = (self.font_size / 12.0).max(1.0); // 1px hairline floor — a bare LINE stroke (no fill behind it) must not vanish sub-pixel
            paint::draw_rect_rotated(
                canvas,
                cx - side * 0.14,
                cy + side * 0.14,
                side * 0.32,
                seg_th,
                core::f32::consts::FRAC_PI_4,
                theme::TEXTBOX_TEXT,
                clip,
            );
            paint::draw_rect_rotated(
                canvas,
                cx + side * 0.06,
                cy - side * 0.02,
                side * 0.5,
                seg_th,
                -core::f32::consts::FRAC_PI_4,
                theme::TEXTBOX_TEXT,
                clip,
            );
            let _ = inset;
        }

        // Fill colour: ticked = action-blue, empty = textbox-dark, so on/off reads at a glance.
        let fill = if self.checked {
            theme::BUTTON_FILL
        } else {
            self.empty_fill.unwrap_or(theme::TEXTBOX_FILL)
        };
        // Asymmetric corners like the textbox, button and window perimeter (Nick 2026-09-10): TL+BR at the box's deep radius, TR+BL at half — the inner fill's corners sit one stroke inside the outer ring's.
        let inner = (side as isize - 2 * stroke).max(0);
        let big = side * 0.32;
        if inner > 0 {
            paint::draw_squircle_rrect_f(
                canvas,
                box_x0 as isize + stroke,
                box_y0 as isize + stroke,
                inner,
                inner,
                fill,
                paint::asymmetric_radii((big - stroke as f32).max(1.0)),
                2.5,
            );
        }
        paint::draw_squircle_rrect_two_tone_f(
            canvas,
            box_x0 as isize,
            box_y0 as isize,
            side as isize,
            side as isize,
            theme::TEXTBOX_SHADOW_EDGE,
            theme::TEXTBOX_LIGHT_EDGE,
            paint::asymmetric_radii(big),
            2.5,
            None,
            0,
        );

        // Label to the right of the box.
        if !self.label.is_empty() {
            // Wrapped label, vertically centred on the widget: one line sits on center_y exactly as before; more lines stack around it.
            let lines = self.label_lines(text);
            let step = self.font_size * 1.25;
            let first_y = self.center_y - step * (lines.len().saturating_sub(1) as Coord) * 0.5;
            for (i, line) in lines.iter().enumerate() {
                text.draw_text_left(
                    canvas,
                    line,
                    box_x0 + side + self.font_size * 0.5,
                    first_y + step * i as Coord,
                    &TextStyle::new(self.font_size, theme::TEXTBOX_TEXT),
                    clip,
                    None,
                );
            }
        }

        // Stamp the hit id over the whole widget rect (box + label) so the entire row is clickable.
        if let Some(map) = hit_map.as_deref_mut() {
            let rect = PixelRect::new(
                (self.center_x - self.width * 0.5).max(0.0) as usize,
                (self.center_y - self.height * 0.5).max(0.0) as usize,
                (self.center_x + self.width * 0.5).max(0.0) as usize,
                (self.center_y + self.height * 0.5).max(0.0) as usize,
            );
            let bw = canvas.width;
            let bh = canvas.height;
            let x1 = rect.x1.min(bw);
            let y1 = rect.y1.min(bh);
            for y in rect.y0..y1 {
                let base = y * bw;
                for x in rect.x0..x1 {
                    map[base + x] = self.hit_id;
                }
            }
        }
        let _ = HIT_NONE;
    }
}

mod widget_impls {
    //! fluor capability traits so the checkbox rides the same dispatch / tab / hover machinery as Button.

    use super::Checkbox;
    use crate::coord::Coord;
    use crate::event::{ElementState, Key as FKey, KeyEvent, ModifiersState, NamedKey};
    use crate::host::widget::{Click, Focus, Hover, Key, PaintCtx, Widget};
    use crate::paint::HitId;
    use crate::text::TextRenderer;
    use crate::theme;

    impl Widget for Checkbox {
        fn id(&self) -> HitId {
            self.hit_id()
        }
        fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
            // The host drives painting via `render_content_into`, same convention as Button/Textbox.
        }
        fn click(&mut self) -> Option<&mut dyn Click> {
            Some(self)
        }
        fn key(&mut self) -> Option<&mut dyn Key> {
            Some(self)
        }
        fn focus(&mut self) -> Option<&mut dyn Focus> {
            Some(self)
        }
        fn hover(&mut self) -> Option<&mut dyn Hover> {
            Some(self)
        }
    }

    impl Click for Checkbox {
        fn on_click(
            &mut self,
            _x: Coord,
            _y: Coord,
            _mods: ModifiersState,
        ) -> crate::host::EventResponse {
            self.toggle();
            crate::host::EventResponse::Handled
        }
    }

    impl Key for Checkbox {
        fn on_key(
            &mut self,
            kev: &KeyEvent,
            _mods: ModifiersState,
            _text: &mut TextRenderer,
        ) -> crate::host::EventResponse {
            if kev.state != ElementState::Pressed {
                return crate::host::EventResponse::Pass;
            }
            match &kev.logical_key {
                FKey::Named(NamedKey::Enter) | FKey::Named(NamedKey::Space) => {
                    self.toggle();
                    crate::host::EventResponse::Handled
                }
                _ => crate::host::EventResponse::Pass,
            }
        }
    }

    impl Focus for Checkbox {
        fn set_focused(&mut self, focused: bool) {
            Checkbox::set_focused(self, focused);
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

    impl Hover for Checkbox {
        fn set_hovered(&mut self, hovered: bool) {
            Checkbox::set_hovered(self, hovered);
        }
        fn tint_delta(&self) -> u32 {
            if self.is_hovered() {
                crate::paint::wrap_sub_rgb(theme::BUTTON_HOVER, theme::BUTTON_FILL)
            } else {
                0
            }
        }
    }
}
