//! Horizontal slider widget, styled after lumis's camera-control sliders: a white bar from the left edge to the handle, a black bar from the handle to the right edge, and a filled circular handle at the position.
//! Greyed (light/dark grey instead of white/black) when disabled.
//!
//! **Value model.** `value` is `0.0..=1.0`, left to right; the app maps it to whatever domain it needs (log scales, -1..1, etc.).
//! Like [`super::Button`]'s click counter, value edits accumulate into a change counter the app polls via [`Slider::take_change`] each frame — no callbacks.
//!
//! **Drag model.** A click anywhere on the grab band jumps the handle there ([`crate::host::widget::Click`] impl).
//! Continuous dragging is app-driven, matching Textbox's drag-select pattern: the app remembers the pressed slider's [`Slider::hit_id`] on mouse-down, then feeds cursor x thru [`Slider::set_value_from_x`] on every move until release.
//! Arrow keys nudge the value by 1/64 when the slider holds focus (Home/End jump to the ends).

use crate::canvas::PixelRect;
use crate::coord::Coord;
use crate::paint::{self, HitId};
use crate::region::Region;

/// Opaque darkness-convention colours, lumis palette: white(left)/black(right), grey pair when disabled.
const LEFT_FILL: u32 = 0xFF00_0000; // white (zero darkness)
const RIGHT_FILL: u32 = 0xFFFF_FFFF; // black (full darkness)
const LEFT_DISABLED: u32 = 0xFF00_0000 | (0x00FF_FFFF ^ 0x005A_5A5A); // 90-grey
const RIGHT_DISABLED: u32 = 0xFF00_0000 | (0x00FF_FFFF ^ 0x003C_3C3C); // 60-grey
const HANDLE_FILL: u32 = 0xFF00_0000; // white
const HANDLE_DISABLED: u32 = 0xFF00_0000 | (0x00FF_FFFF ^ 0x005A_5A5A);

pub struct Slider {
    /// Allocated at construction; stamped into the host hit map across the whole grab band.
    hit_id: HitId,
    pub center_x: Coord,
    pub center_y: Coord,
    /// Full grab-band width; the track spans it edge to edge.
    pub width: Coord,
    /// Grab-band height. The painted track is a thin strip thru the middle; the handle radius is `height/2`.
    pub height: Coord,

    /// Position in `0.0..=1.0`, left to right.
    value: f32,
    focused: bool,
    hovered: bool,
    enabled: bool,

    /// Monotonic edit counter, bumped on every value change. Poll via [`Self::take_change`].
    change_counter: u32,
    last_seen_change_counter: u32,

    // --- Damage protocol (same shape as Button's) ---
    last_painted_bbox: Option<PixelRect>,
    last_painted_value: f32,
    last_painted_enabled: bool,
    last_painted_focused: bool,
}

impl Slider {
    /// `hit_counter` is the app's monotonic [`HitId`] allocator; each Slider claims one ID.
    pub fn new(
        hit_counter: &mut HitId,
        center_x: Coord,
        center_y: Coord,
        width: Coord,
        height: Coord,
        value: f32,
    ) -> Self {
        Self {
            hit_id: crate::host::widget::next_id(hit_counter),
            center_x,
            center_y,
            width,
            height,
            value: value.clamp(0.0, 1.0),
            focused: false,
            hovered: false,
            enabled: true,
            change_counter: 0,
            last_seen_change_counter: 0,
            last_painted_bbox: None,
            last_painted_value: -1.0,
            last_painted_enabled: true,
            last_painted_focused: false,
        }
    }

    pub fn hit_id(&self) -> HitId {
        self.hit_id
    }
    pub fn value(&self) -> f32 {
        self.value
    }
    pub fn is_focused(&self) -> bool {
        self.focused
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Set the value directly (clamped to `0..=1`). Bumps the change counter if it actually moved.
    pub fn set_value(&mut self, value: f32) {
        let v = value.clamp(0.0, 1.0);
        if v != self.value {
            self.value = v;
            self.change_counter = self.change_counter.wrapping_add(1);
        }
    }

    /// Map a viewport x coordinate to a value and set it — the drag primitive.
    /// The app calls this on cursor moves while a press that started on this slider is held.
    pub fn set_value_from_x(&mut self, x: Coord) {
        if !self.enabled {
            return;
        }
        let left = self.center_x - self.width * 0.5;
        self.set_value(((x - left) / self.width.max(1e-6)) as f32);
    }

    /// `true` if the value changed since the last call. Coalesces like [`super::Button::take_click`].
    pub fn take_change(&mut self) -> bool {
        if self.change_counter != self.last_seen_change_counter {
            self.last_seen_change_counter = self.change_counter;
            true
        } else {
            false
        }
    }

    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    pub fn set_hovered(&mut self, hovered: bool) {
        if !self.enabled {
            return;
        }
        self.hovered = hovered;
    }

    /// Enable / disable input. Disabled drops the slider out of dispatch (Widget accessors return `None`) and paints the grey palette.
    pub fn set_enabled(&mut self, enabled: bool) {
        if enabled == self.enabled {
            return;
        }
        self.enabled = enabled;
        if !enabled {
            self.focused = false;
            self.hovered = false;
        }
    }

    pub fn set_rect(&mut self, center_x: Coord, center_y: Coord, width: Coord, height: Coord) {
        self.center_x = center_x;
        self.center_y = center_y;
        self.width = width;
        self.height = height;
    }

    pub fn bbox(&self) -> Region {
        Region::new(
            self.center_x - self.width * 0.5,
            self.center_y - self.height * 0.5,
            self.width,
            self.height,
        )
    }

    /// Damage region: `None` when nothing visible changed since the last paint, else the union of the prior and current bboxes.
    pub fn damage_rect(&self, viewport_w: usize, viewport_h: usize) -> Option<PixelRect> {
        let dirty = self.value != self.last_painted_value
            || self.enabled != self.last_painted_enabled
            || self.focused != self.last_painted_focused;
        if !dirty && self.last_painted_bbox.is_some() {
            return None;
        }
        let current =
            crate::widgets::textbox::region_to_pixelrect(self.bbox(), viewport_w, viewport_h);
        Some(self.last_painted_bbox.map_or(current, |p| p.union(current)))
    }

    /// Paint the slider into `canvas` and stamp `hit_id` into `hit_map` across the whole grab band.
    /// Draws directly (no caches) — two rects and a circle are cheap enough to rasterize per frame.
    pub fn render_content_into(
        &mut self,
        canvas: &mut crate::canvas::Canvas,
        hit_map: Option<&mut [HitId]>,
        hit_id: HitId,
    ) {
        let band_x = (self.center_x - self.width * 0.5) as isize;
        let band_y = (self.center_y - self.height * 0.5) as isize;
        let band_w = self.width as isize;
        let band_h = self.height as isize;
        if band_w <= 0 || band_h <= 0 {
            return;
        }

        let (left_fill, right_fill, handle_fill) = if self.enabled {
            (LEFT_FILL, RIGHT_FILL, HANDLE_FILL)
        } else {
            (LEFT_DISABLED, RIGHT_DISABLED, HANDLE_DISABLED)
        };

        // Track: a strip 1/4 of the band height, vertically centred. The handle circle overhangs it.
        let track_h = (band_h / 4).max(2);
        let track_y = band_y + (band_h - track_h) / 2;
        let handle_x = band_x + (self.value as Coord * self.width) as isize;
        let radius = (band_h / 2 - 1).max(2);

        // Handle first: topmost-first doctrine, the circle wins where it overlaps the track.
        paint::circle_filled(
            canvas,
            handle_x,
            band_y + band_h / 2,
            radius,
            handle_fill,
            None,
            None,
        );
        // White from the left edge to the handle, black from the handle to the right edge.
        paint::fill_rect(
            canvas,
            band_x,
            track_y,
            handle_x - band_x,
            track_h,
            left_fill,
            None,
            None,
        );
        paint::fill_rect(
            canvas,
            handle_x,
            track_y,
            band_x + band_w - handle_x,
            track_h,
            right_fill,
            None,
            None,
        );

        // Stamp the hit map across the full band so clicks and drags land anywhere on it.
        if let Some(map) = hit_map {
            let vw = canvas.width;
            let vh = canvas.height;
            let x0 = band_x.max(0) as usize;
            let y0 = band_y.max(0) as usize;
            let x1 = ((band_x + band_w).max(0) as usize).min(vw);
            let y1 = ((band_y + band_h).max(0) as usize).min(vh);
            for row in y0..y1 {
                for col in x0..x1 {
                    map[row * vw + col] = hit_id;
                }
            }
        }

        let vw = canvas.width;
        let vh = canvas.height;
        self.last_painted_bbox = Some(crate::widgets::textbox::region_to_pixelrect(
            self.bbox(),
            vw,
            vh,
        ));
        self.last_painted_value = self.value;
        self.last_painted_enabled = self.enabled;
        self.last_painted_focused = self.focused;
    }
}

mod widget_impls {
    //! [`crate::host::widget`] capability traits for [`Slider`], mirroring Button's: Click jumps the handle to the click x, Key nudges with arrows, Focus/Hover route thru the setters.

    use super::Slider;
    use crate::coord::Coord;
    use crate::event::{ElementState, Key as FKey, KeyEvent, ModifiersState, NamedKey};
    use crate::host::widget::{Click, Focus, Hover, Key, PaintCtx, Widget};
    use crate::paint::HitId;
    use crate::text::TextRenderer;

    impl Widget for Slider {
        fn id(&self) -> HitId {
            self.hit_id()
        }
        fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {
            // No-op, same reason as Button: the app drives rendering via `render_content_into`.
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

    impl Click for Slider {
        fn on_click(
            &mut self,
            x: Coord,
            _y: Coord,
            _mods: ModifiersState,
        ) -> crate::host::EventResponse {
            self.set_value_from_x(x);
            crate::host::EventResponse::Handled
        }
        // Engage on press: the press sets the value at the thumb and the ensuing drag tracks it — the slider owns its press-drag, so no drag-off-cancel.
        fn activate_on_release(&self) -> bool {
            false
        }
    }

    impl Key for Slider {
        fn on_key(
            &mut self,
            kev: &KeyEvent,
            _mods: ModifiersState,
            _text: &mut TextRenderer,
        ) -> crate::host::EventResponse {
            if kev.state != ElementState::Pressed {
                return crate::host::EventResponse::Pass;
            }
            const STEP: f32 = 1.0 / 64.0;
            match &kev.logical_key {
                FKey::Named(NamedKey::ArrowLeft) => {
                    self.set_value(self.value() - STEP);
                    crate::host::EventResponse::Handled
                }
                FKey::Named(NamedKey::ArrowRight) => {
                    self.set_value(self.value() + STEP);
                    crate::host::EventResponse::Handled
                }
                FKey::Named(NamedKey::Home) => {
                    self.set_value(0.0);
                    crate::host::EventResponse::Handled
                }
                FKey::Named(NamedKey::End) => {
                    self.set_value(1.0);
                    crate::host::EventResponse::Handled
                }
                _ => crate::host::EventResponse::Pass,
            }
        }
    }

    impl Focus for Slider {
        fn set_focused(&mut self, focused: bool) {
            Slider::set_focused(self, focused);
        }
        fn focus_bbox(&self) -> Option<crate::canvas::PixelRect> {
            let r = self.bbox();
            let x0 = r.x.max(0.0) as usize;
            let y0 = r.y.max(0.0) as usize;
            let x1 = (r.x + r.w).max(0.0) as usize;
            let y1 = (r.y + r.h).max(0.0) as usize;
            Some(crate::canvas::PixelRect::new(x0, y0, x1, y1))
        }
    }

    impl Hover for Slider {
        fn set_hovered(&mut self, hovered: bool) {
            Slider::set_hovered(self, hovered);
        }
        fn tint_delta(&self) -> u32 {
            // The slider repaints directly on value changes; no overlay tint needed.
            0
        }
    }
}
