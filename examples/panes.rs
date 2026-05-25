//! Three panes + DefaultChrome + a Textbox + a rotation-demo text — a full `FluorApp` driver.
//!
//! This example owns *all* the demo content; the host (`fluor::host::app::run_app`) only opens a window, runs the event loop, and presents the buffer. Future consumers (rhe / photon / basecalc) follow the same pattern: implement `FluorApp`, hand the impl to `run_app`.

use std::time::Instant;

use fluor::coord::Coord;
use fluor::geom::Viewport;
use fluor::group::Group;
use fluor::host::app::{Context, EventResponse, FluorApp};
use fluor::host::chrome::{
    self, HIT_CLOSE_BUTTON, HIT_MAXIMIZE_BUTTON, HIT_MINIMIZE_BUTTON, HIT_NONE, ResizeEdge,
};
use fluor::host::chrome_widget::DefaultChrome;
use fluor::paint::pack_argb;
use fluor::paint::{self, BlendMode, Transform};
use fluor::region::Region;
use fluor::stack::Op;
use fluor::theme;
use fluor::widgets::{BlinkTimer, Textbox};
use fluor::{Compositor, RuVec2};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use winit::window::CursorIcon;

struct PanesDemo {
    title: String,
    compositor: Compositor,
    chrome: DefaultChrome,
    textbox: Textbox,
    textbox_group: Group,
    cursor_group: Group,
    /// Rotation demo text — full-viewport AlphaOver group, re-rasterized only when aspect changes.
    rotation_group: Group,
    blink: BlinkTimer,
    is_dragging_select: bool,
    selection_scroll_time: Option<Instant>,
    /// True while D is physically held as part of the debug chord (was pressed with primary+shift modifiers). Cleared on D release. Pure event-driven, no timeout.
    d_chord_held: bool,
    /// Toggle for the H chord action — paints the chrome's hit_test_map as an opaque tinted overlay.
    show_hitmask: bool,
    /// 256-entry random colour table indexed by hit_test_map byte; regenerated each time H toggles on so distinct IDs get visibly-distinct colours. Photon's debug-hit pattern.
    debug_hit_colours: Vec<u32>,
}

impl PanesDemo {
    fn new(viewport: Viewport, title: impl Into<String>) -> Self {
        let mut compositor = Compositor::new(viewport);
        compositor.insert(
            RuVec2::new(-0.15, -0.08),
            RuVec2::new(0.14, 0.10),
            pack_argb(220, 90, 80, 255),
        );
        compositor.insert(
            RuVec2::new(0.05, 0.04),
            RuVec2::new(0.12, 0.12),
            pack_argb(90, 180, 100, 255),
        );
        compositor.insert(
            RuVec2::new(0.18, -0.14),
            RuVec2::new(0.09, 0.14),
            pack_argb(80, 100, 220, 255),
        );

        let title = title.into();
        let chrome = DefaultChrome::new(viewport, title.clone());

        // Placeholder textbox + groups — actual geometry computed in `init`/`on_resize`.
        // textbox_group has two layers: content (under) + glow (on top, pre-knocked by (255-mask)).
        // Stack (topmost-first): Push glow (topmost diffusion halo), Push content, Under(Normal).
        // Glow's AA edge lightly tints whatever's behind via partial-t; content (the pill body + text + selection) sits under the glow.
        let mut textbox = Textbox::new(0.0, 0.0, 1.0, 1.0, 12.0);
        textbox.stroke_ru = 0.15; // a smidge of an RU so the inner/outer pills are visibly distinct
        let placeholder_region = Region::new(0.0, 0.0, 1.0, 1.0);
        let mut textbox_group = Group::new(placeholder_region, BlendMode::Normal);
        let content_layer = textbox_group.new_layer();
        let glow_layer = textbox_group.new_layer();
        textbox_group.set_program(vec![
            Op::Push(glow_layer),
            Op::Push(content_layer),
            Op::Under(BlendMode::Normal),
        ]);
        let mut cursor_group = Group::new(placeholder_region, BlendMode::Add);
        let l = cursor_group.new_layer();
        cursor_group.set_program(vec![Op::Push(l)]);

        let viewport_region = Region::new(
            0.0,
            0.0,
            viewport.width_px as Coord,
            viewport.height_px as Coord,
        );
        let mut rotation_group = Group::new(viewport_region, BlendMode::Normal);
        let l = rotation_group.new_layer();
        rotation_group.set_program(vec![Op::Push(l)]);

        Self {
            title,
            compositor,
            chrome,
            textbox,
            textbox_group,
            cursor_group,
            rotation_group,
            blink: BlinkTimer::new(),
            is_dragging_select: false,
            selection_scroll_time: None,
            d_chord_held: false,
            show_hitmask: false,
            debug_hit_colours: Vec::new(),
        }
    }

    /// Recompute textbox geometry from the viewport span and resize the textbox + cursor groups to match.
    fn update_layout(&mut self, ctx: &mut Context) {
        let vp = ctx.viewport;
        // Use effective_span (= span * ru) so all derived sizes pick up the user's zoom — Ctrl+/Ctrl-/Ctrl+scroll grow/shrink the textbox + cursor + rotation regions together with chrome.
        let span = vp.effective_span();
        let bw = (span / 32.0).ceil();
        // Aspect-driven horizontal shift: square window centers the textbox; wider pushes it right, taller pushes it left. Magnitude scales with span so the shift is visible but bounded.
        let aspect = vp.width_px as Coord / vp.height_px as Coord;
        let center_x = vp.width_px as Coord * 0.5 + (aspect - 1.0) * span * 0.25;
        let center_y = bw * 7.0;
        let width = (vp.width_px as Coord * 0.5).max(bw * 8.0);
        let height = bw * 1.6;
        let font_size = bw * 0.55;
        self.textbox.set_rect(center_x, center_y, width, height);
        self.textbox.set_font_size(font_size, ctx.text);

        self.textbox_group.resize(self.textbox.bbox());
        self.cursor_group.resize(self.textbox.cursor_bbox());

        let viewport_region = Region::new(0.0, 0.0, vp.width_px as Coord, vp.height_px as Coord);
        self.rotation_group.resize(viewport_region);
    }
}

impl FluorApp for PanesDemo {
    fn title(&self) -> &str {
        &self.title
    }

    fn init(&mut self, ctx: &mut Context) {
        self.compositor
            .resize(ctx.viewport.width_px, ctx.viewport.height_px);
        self.chrome.resize(ctx.viewport);
        self.update_layout(ctx);
    }

    fn on_resize(&mut self, w: u32, h: u32, ctx: &mut Context) {
        self.compositor.resize(w, h);
        self.chrome.resize(ctx.viewport);
        self.update_layout(ctx);
    }

    fn on_event(&mut self, event: &WindowEvent, ctx: &mut Context) -> EventResponse {
        match event {
            WindowEvent::CursorMoved { .. } => {
                if self.is_dragging_select {
                    let tl = self.textbox.center_x - self.textbox.width * 0.5
                        + self.textbox.font_size * 0.4;
                    let tr = self.textbox.center_x + self.textbox.width * 0.5
                        - self.textbox.font_size * 0.4;
                    let clamped_x = ctx.cursor_x.clamp(tl, tr);
                    if self.textbox.selection_anchor.is_none() {
                        self.textbox.selection_anchor = Some(self.textbox.cursor);
                    }
                    self.textbox.cursor = self.textbox.cursor_index_from_x(clamped_x);
                    self.textbox_group.invalidate();
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                let chrome_changed = {
                    let new_hit = self.chrome.hit_at(ctx.cursor_x, ctx.cursor_y);
                    self.chrome.set_hover(new_hit)
                };
                let new_textbox_hover = self.textbox.contains(ctx.cursor_x, ctx.cursor_y);
                let textbox_changed = self.textbox.hovered != new_textbox_hover;
                if textbox_changed {
                    self.textbox.hovered = new_textbox_hover;
                    self.textbox_group.invalidate();
                }
                if chrome_changed || textbox_changed {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::CursorLeft { .. } => {
                let chrome_changed = self.chrome.set_hover(HIT_NONE);
                let textbox_changed = self.textbox.hovered;
                if textbox_changed {
                    self.textbox.hovered = false;
                    self.textbox_group.invalidate();
                }
                if chrome_changed || textbox_changed {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                match self.chrome.hit_at(ctx.cursor_x, ctx.cursor_y) {
                    HIT_CLOSE_BUTTON => return EventResponse::Close,
                    HIT_MINIMIZE_BUTTON => {
                        ctx.window.set_minimized(true);
                        return EventResponse::Handled;
                    }
                    HIT_MAXIMIZE_BUTTON => {
                        ctx.window.set_maximized(!ctx.window.is_maximized());
                        return EventResponse::Handled;
                    }
                    _ => {}
                }
                let edge = chrome::get_resize_edge(
                    ctx.viewport.width_px,
                    ctx.viewport.height_px,
                    ctx.cursor_x,
                    ctx.cursor_y,
                );
                if edge != ResizeEdge::None {
                    return EventResponse::StartResize(edge);
                }
                let was_focused = self.textbox.focused;
                self.textbox.handle_click(ctx.cursor_x, ctx.cursor_y);
                if self.textbox.focused {
                    self.is_dragging_select = true;
                    self.selection_scroll_time = None;
                    self.textbox_group.invalidate();
                    self.blink.start(Instant::now());
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                } else if was_focused {
                    self.blink.stop();
                    self.is_dragging_select = false;
                    self.textbox_group.invalidate();
                    ctx.window.request_redraw();
                }
                EventResponse::StartWindowDrag
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                if self.is_dragging_select {
                    self.is_dragging_select = false;
                    self.selection_scroll_time = None;
                    if self.textbox.selection_anchor == Some(self.textbox.cursor) {
                        self.textbox.selection_anchor = None;
                    }
                    self.blink.start(Instant::now());
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::KeyboardInput { event: kev, .. } => {
                let shift = ctx.modifiers.shift_key();
                let ctrl = ctx.modifiers.super_key() || ctx.modifiers.control_key();

                // --- Debug chord: Ctrl/Cmd + Shift + D held, then action key.
                // Track D held state event-driven. D press with modifiers enters chord (swallowed
                // as not-text); D release clears state. Other keys pressed while D is held with
                // modifiers fire the matching action. No timer, no arming.
                if let Key::Character(c) = &kev.logical_key {
                    if c == "d" || c == "D" {
                        if kev.state == ElementState::Pressed && ctrl && shift {
                            self.d_chord_held = true;
                            return EventResponse::Handled;
                        }
                        if kev.state == ElementState::Released && self.d_chord_held {
                            self.d_chord_held = false;
                            return EventResponse::Handled;
                        }
                    }
                }
                if self.d_chord_held && ctrl && shift && kev.state == ElementState::Pressed {
                    if let Key::Character(c) = &kev.logical_key {
                        let mut acted = true;
                        if c == "h" || c == "H" {
                            self.show_hitmask = !self.show_hitmask;
                            if self.show_hitmask {
                                // Fill the 256-entry colour table with distinct random RGBs each
                                // toggle. xorshift32 seeded from process nanos — debug-quality, no
                                // crypto needed. Each call to toggle = fresh palette.
                                let seed = (std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .map(|d| d.subsec_nanos())
                                    .unwrap_or(1))
                                    | 1;
                                let mut s = seed;
                                self.debug_hit_colours.clear();
                                self.debug_hit_colours.reserve(256);
                                for _ in 0..256 {
                                    s ^= s << 13;
                                    s ^= s >> 17;
                                    s ^= s << 5;
                                    let r = (s >> 16) & 0xFF;
                                    s ^= s << 13;
                                    s ^= s >> 17;
                                    s ^= s << 5;
                                    let g = (s >> 16) & 0xFF;
                                    s ^= s << 13;
                                    s ^= s >> 17;
                                    s ^= s << 5;
                                    let b = (s >> 16) & 0xFF;
                                    // α + darkness: α=0xFF means opaque. RGB stored as darkness — the value here is debug-overlay color, used only by the hit-id visualization which OR's it directly into a presented buffer slot, so leave it in the host-facing convention (α=0 transparent doesn't matter; this isn't a fluor pixel).
                                    self.debug_hit_colours.push((r << 16) | (g << 8) | b);
                                }
                            }
                        } else if c == "p" || c == "P" {
                            let cur = paint::DEBUG_SKIP_PREMULT
                                .load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SKIP_PREMULT
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                        } else if c == "a" || c == "A" {
                            // Cycle: off (0) → grayscale (1) → force-opaque (2) → off.
                            let cur = paint::DEBUG_SHOW_ALPHA
                                .load(std::sync::atomic::Ordering::Relaxed);
                            let next = (cur + 1) % 3;
                            paint::DEBUG_SHOW_ALPHA
                                .store(next, std::sync::atomic::Ordering::Relaxed);
                        } else if c == "c" || c == "C" {
                            let cur = paint::DEBUG_SKIP_CHROME
                                .load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SKIP_CHROME
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                            self.chrome.invalidate_chrome();
                            ctx.window.request_redraw();
                        } else if c == "x" || c == "X" {
                            let cur = paint::DEBUG_SKIP_CONTROLS
                                .load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SKIP_CONTROLS
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                            self.chrome.invalidate_chrome();
                            ctx.window.request_redraw();
                        } else if c == "r" || c == "R" {
                            self.chrome.group.invalidate();
                            self.textbox_group.invalidate();
                            self.cursor_group.invalidate();
                            self.rotation_group.invalidate();
                        } else {
                            acted = false;
                        }
                        if acted {
                            ctx.window.request_redraw();
                            return EventResponse::Handled;
                        }
                    }
                }

                if kev.state != ElementState::Pressed {
                    return EventResponse::Pass;
                }
                if !self.textbox.focused {
                    return EventResponse::Pass;
                }
                let mut changed = false;
                match &kev.logical_key {
                    Key::Named(NamedKey::Backspace) => {
                        self.textbox.backspace(ctx.text);
                        changed = true;
                    }
                    Key::Named(NamedKey::Delete) => {
                        self.textbox.delete_forward(ctx.text);
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        if shift && self.textbox.selection_anchor.is_none() {
                            self.textbox.selection_anchor = Some(self.textbox.cursor);
                        } else if !shift {
                            self.textbox.selection_anchor = None;
                        }
                        self.textbox.cursor_left();
                        changed = true;
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        if shift && self.textbox.selection_anchor.is_none() {
                            self.textbox.selection_anchor = Some(self.textbox.cursor);
                        } else if !shift {
                            self.textbox.selection_anchor = None;
                        }
                        self.textbox.cursor_right();
                        changed = true;
                    }
                    Key::Named(NamedKey::Home) => {
                        if shift && self.textbox.selection_anchor.is_none() {
                            self.textbox.selection_anchor = Some(self.textbox.cursor);
                        } else if !shift {
                            self.textbox.selection_anchor = None;
                        }
                        self.textbox.cursor_home();
                        changed = true;
                    }
                    Key::Named(NamedKey::End) => {
                        if shift && self.textbox.selection_anchor.is_none() {
                            self.textbox.selection_anchor = Some(self.textbox.cursor);
                        } else if !shift {
                            self.textbox.selection_anchor = None;
                        }
                        self.textbox.cursor_end();
                        changed = true;
                    }
                    Key::Character(c) if ctrl && (c == "a" || c == "A") => {
                        self.textbox.select_all();
                        changed = true;
                    }
                    Key::Character(c) if ctrl && (c == "c" || c == "C") => {
                        if let Some(selected) = self.textbox.selected_text() {
                            if let Ok(mut clip) = arboard::Clipboard::new() {
                                let _ = clip.set_text(selected);
                            }
                        }
                    }
                    Key::Character(c) if ctrl && (c == "x" || c == "X") => {
                        if let Some(selected) = self.textbox.selected_text() {
                            let ok = arboard::Clipboard::new()
                                .and_then(|mut clip| clip.set_text(selected))
                                .is_ok();
                            if ok {
                                self.textbox.delete_selection(ctx.text);
                                changed = true;
                            }
                        }
                    }
                    Key::Character(c) if ctrl && (c == "v" || c == "V") => {
                        if let Ok(mut clip) = arboard::Clipboard::new() {
                            if let Ok(paste) = clip.get_text() {
                                self.textbox.insert_str(&paste, ctx.text);
                                changed = true;
                            }
                        }
                    }
                    _ => {
                        if let Some(s) = &kev.text {
                            if !ctrl {
                                for c in s.chars() {
                                    if !c.is_control() {
                                        self.textbox.insert_char(c, ctx.text);
                                        changed = true;
                                    }
                                }
                            }
                        }
                    }
                }
                if changed {
                    self.textbox_group.invalidate();
                    self.blink.start(Instant::now());
                    ctx.window.request_redraw();
                }
                EventResponse::Handled
            }
            _ => EventResponse::Pass,
        }
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let buf_w = ctx.viewport.width_px as usize;
        let buf_h = ctx.viewport.height_px as usize;

        // Hairline + background + hover overlay. Chrome's bg layer gets photon's background_noise; chrome layer gets the perimeter hairline + controls (or stays empty under Ctrl+Shift+D+C); hover layer gets a partial-α tint over the currently-hovered button (or stays empty if hover_state == HIT_NONE). The chrome group's Stack program (`Push hover, Push chrome, Under(Normal), Push bg, Under(Normal)`) front-to-back-composites them, then flattens under the target. Order matters: rasterize_chrome MUST run before rasterize_hover because hover reads `hit_test_map` which chrome populates.
        self.chrome
            .rasterize_bg(|buf, w, h| paint::background_noise(buf, w, h, 0, true, 0, None));
        self.chrome.rasterize_chrome(ctx.text, ctx.clip_mask);
        self.chrome.rasterize_hover();
        self.chrome.flatten_into(target, buf_w, buf_h);

        // Debug overlay (photon-style): for every pixel, look up the hit_test_map's ID and paint
        // its opaque random colour from `debug_hit_colours`. Fully replaces the underlying image —
        // distinct hit zones are visually unmistakable. Drawn last over everything (including
        // textbox + cursor) since hit testing is per-final-pixel anyway.
        if self.show_hitmask && !self.debug_hit_colours.is_empty() {
            let map = &self.chrome.hit_test_map;
            let n = map.len().min(target.len());
            for i in 0..n {
                target[i] = self.debug_hit_colours[map[i] as usize];
            }
        }
    }

    fn cursor_for(&self, x: Coord, y: Coord, ctx: &Context) -> CursorIcon {
        let hit = self.chrome.hit_at(x, y);
        match hit {
            HIT_CLOSE_BUTTON | HIT_MINIMIZE_BUTTON | HIT_MAXIMIZE_BUTTON => {
                return CursorIcon::Pointer;
            }
            _ => {}
        }
        match chrome::get_resize_edge(ctx.viewport.width_px, ctx.viewport.height_px, x, y) {
            ResizeEdge::Top | ResizeEdge::Bottom => CursorIcon::NsResize,
            ResizeEdge::Left | ResizeEdge::Right => CursorIcon::EwResize,
            ResizeEdge::TopLeft | ResizeEdge::BottomRight => CursorIcon::NwseResize,
            ResizeEdge::TopRight | ResizeEdge::BottomLeft => CursorIcon::NeswResize,
            ResizeEdge::None => {
                if self.textbox.contains(x, y) {
                    CursorIcon::Text
                } else {
                    CursorIcon::Default
                }
            }
        }
    }

    fn wake_at(&self) -> Option<Instant> {
        self.blink.next_tick()
    }

    fn tick(&mut self, ctx: &mut Context) -> bool {
        let mut needs_redraw = false;

        // Selection drag: auto-scroll the textbox content while the cursor is held outside its bounds.
        if self.is_dragging_select {
            let tl =
                self.textbox.center_x - self.textbox.width * 0.5 + self.textbox.font_size * 0.4;
            let tr =
                self.textbox.center_x + self.textbox.width * 0.5 - self.textbox.font_size * 0.4;
            let distance_outside = if ctx.cursor_x < tl {
                tl - ctx.cursor_x
            } else if ctx.cursor_x > tr {
                ctx.cursor_x - tr
            } else {
                0.0
            };
            if distance_outside > 0.0 {
                let now = Instant::now();
                let dt = self
                    .selection_scroll_time
                    .map(|t| now.duration_since(t).as_secs_f32())
                    .unwrap_or(0.0);
                self.selection_scroll_time = Some(now);
                let uw = self.textbox.width - self.textbox.font_size * 0.8;
                let speed = 1000.0 * distance_outside / uw;
                let delta = speed * dt;
                if ctx.cursor_x < tl {
                    self.textbox.scroll_offset += delta;
                } else {
                    self.textbox.scroll_offset -= delta;
                }
                let clamped_x = ctx.cursor_x.clamp(tl, tr);
                self.textbox.cursor = self.textbox.cursor_index_from_x(clamped_x);
                self.textbox_group.invalidate();
                needs_redraw = true;
            } else {
                self.selection_scroll_time = None;
            }
        }

        // Blink timer. State flip needs a frame so the cursor's new transparency reaches the
        // present buffer; the cursor itself re-renders unconditionally in `render()`.
        if self.blink.poll(Instant::now()) {
            if self.textbox.flip_blinkey() {
                needs_redraw = true;
            }
        }

        needs_redraw
    }
}

fn main() {
    let demo = PanesDemo::new(Viewport::new(1280, 800), "fluor — panes");
    fluor::host::app::run_app(demo).expect("event loop");
}
