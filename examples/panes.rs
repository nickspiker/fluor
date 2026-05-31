//! Three panes + DefaultChrome + a Textbox + a rotation-demo text — a full `FluorApp` driver.
//!
//! This example owns *all* the demo content; the host (`fluor::host::app::run_app`) only opens a window, runs the event loop, and presents the buffer. Future consumers (rhe / photon / basecalc) follow the same pattern: implement `FluorApp`, hand the impl to `run_app`.

use std::time::Instant;

use fluor::coord::Coord;
use fluor::geom::Viewport;
use fluor::group::Group;
use fluor::host::app::{Context, EventResponse, FluorApp};
use fluor::host::chrome::{
    self, HIT_CLOSE_BUTTON, HIT_MAXIMIZE_BUTTON, HIT_MINIMIZE_BUTTON, HIT_NONE, HIT_TEXTBOX, HitId,
    ResizeEdge,
};
use fluor::host::chrome_widget::DefaultChrome;
use fluor::host::icon::Icon;
use fluor::host::os_input;
use fluor::host::widget::{self as widget, Container, TabDir};
use fluor::paint::pack_argb;
use fluor::paint::{self, BlendMode, Transform};
use fluor::region::Region;
use fluor::stack::Op;
use fluor::theme;
use fluor::widgets::{BlinkTimer, Button, Textbox};
use fluor::{Compositor, RuVec2};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, NamedKey};
use winit::window::CursorIcon;

/// Grace period after a `[` or `]` Release event before we consider the key actually released. X11 fires synthetic Release events for held keys when another key is pressed — without this grace, the chord disarms a millisecond before an action key fires and you'd never see one work. Long enough to absorb the synthetic release, short enough that real releases feel instant.
const CHORD_RELEASE_GRACE: std::time::Duration = std::time::Duration::from_millis(40);

/// Pixel proximity tolerance for multi-click sequences. A press farther than this from the previous resets the count. The temporal half of the test uses [`host::os_input::double_click_interval`] — OS-polled, not a hardcoded constant — so users who've cranked the system double-click speed up or down get the behavior they expect.
const MULTI_CLICK_TOL_PX: Coord = 4.0;

/// Debug chord bindings shown in the hint overlay while the chord is armed. Keep in sync with the action dispatch in the keyboard handler.
const CHORD_HINTS: &[(&str, &str)] = &[
    ("H", "Hit-mask overlay"),
    ("P", "Skip premultiply"),
    ("A", "Show alpha (cycle)"),
    ("C", "Skip chrome"),
    ("L", "Skip controls"),
    ("R", "Force redraw"),
    ("F", "FPS / per-stage timings strip"),
    ("W", "Damage rect outline (Where)"),
    ("D", "Screen-buffer decay (fade)"),
    ("B", "Finalize copy-pass blue tint"),
];

/// Compute the bbox the chord hint panel covers — matches `paint::draw_chord_hint`'s positioning math so `panes.damage_rect` can include it when both brackets are held.
fn chord_hint_bbox(
    viewport: fluor::geom::Viewport,
    vw: usize,
    vh: usize,
) -> fluor::canvas::PixelRect {
    let span = viewport.effective_span();
    let font_size = (span * 0.014).max(11.0);
    let line_h = font_size * 1.55;
    let pad = font_size * 1.25;
    let line_count = CHORD_HINTS.len() as f32 + 1.5;
    let panel_h = line_count * line_h + pad * 2.0;
    let panel_w = (span * 0.45).clamp(font_size * 22.0, font_size * 36.0);
    let cx = vw as f32 * 0.5;
    let cy = vh as f32 * 0.4;
    let x0 = (cx - panel_w * 0.5).max(0.0) as usize;
    let y0 = (cy - panel_h * 0.5).max(0.0) as usize;
    let x1 = ((cx + panel_w * 0.5).max(0.0) as usize).min(vw);
    let y1 = ((cy + panel_h * 0.5).max(0.0) as usize).min(vh);
    fluor::canvas::PixelRect::new(x0, y0, x1, y1)
}

/// Convert a host-provided damage `PixelRect` into the `Option<Clip>` shape every paint primitive accepts. `None` skips the conversion when the rect already covers the full viewport (a small optimization — `Clip::buffer(w, h)` would resolve identically but adds a struct copy per call). Today we always pass `Some(clip)` for explicit damage; the host's rect is the union the app declared in `damage_rect`.
fn pixelrect_to_clip(rect: fluor::canvas::PixelRect) -> Option<paint::Clip> {
    Some(paint::Clip::new(rect.x0, rect.y0, rect.x1, rect.y1))
}

struct PanesDemo {
    title: String,
    compositor: Compositor,
    chrome: DefaultChrome,
    /// All textboxes the demo owns. Adding a 20th is `self.textboxes.push(Textbox::new(&mut self.hit_counter, ...))` — single line, everything else (visit, hover dispatch, focus arbitration, render, damage union, overlay deltas, blinkey routing) iterates over this vec and does the right thing automatically. The architecture proof for "N widgets is one line": touched here once means touched everywhere.
    textboxes: Vec<Textbox>,
    /// Per-textbox group caches, indexed lockstep with [`Self::textboxes`]. Invariant: `textbox_groups.len() == textboxes.len()`. Helper methods rely on this so id-to-group lookup is `textboxes.iter().position(|tb| tb.hit_id() == id).map(|i| &mut textbox_groups[i])`.
    textbox_groups: Vec<Group>,
    /// All buttons the demo owns. Same Vec pattern as [`Self::textboxes`] — adding a button is `self.buttons.push(Button::new(&mut self.hit_counter, ..., "label"))`.
    buttons: Vec<Button>,
    button_groups: Vec<Group>,
    cursor_group: Group,
    /// Currently focused widget id, or `None` for "nothing focused" (background click, Esc, no prior focus). Source of truth for keyboard delivery and Tab cycling — widgets' internal `focused` flags are derived state set by `apply_focus_change` after this field updates.
    current_focus: Option<HitId>,
    /// Monotonic dense-ID counter shared across chrome + textboxes. Chrome claims 1..=4 at construction; textbox_a gets 5, textbox_b gets 6. Stored on the demo so future runtime widget creation (e.g. a popup) can keep allocating without re-threading the counter through constructors.
    hit_counter: HitId,
    /// Rotation demo text — full-viewport AlphaOver group, re-rasterized only when aspect changes.
    rotation_group: Group,
    blink: BlinkTimer,
    is_dragging_select: bool,
    selection_scroll_time: Option<Instant>,
    /// Most-recent mouse-press timestamp. Used to detect double/triple-click sequences for word- / line-select. Reset when the gap exceeds `MULTI_CLICK_WINDOW` OR the cursor moves more than `MULTI_CLICK_TOL_PX` between presses.
    last_click_time: Option<Instant>,
    /// Cursor position at the last mouse press, for the proximity test on multi-clicks.
    last_click_pos: (Coord, Coord),
    /// Consecutive-click count: 1 = single, 2 = double (word-select), 3 = triple (line/whole-text select). Increments on a continuation; resets to 1 otherwise.
    click_count: u32,
    /// Last `[` Press timestamp; refreshed by auto-repeat events too. `None` until first press.
    chord_lb_press: Option<Instant>,
    /// Last `[` Release timestamp. Combined with [`Self::chord_lb_press`] this determines whether `[` is currently held: held iff `press > release` OR the release was within [`CHORD_RELEASE_GRACE`].
    chord_lb_release: Option<Instant>,
    /// Mirror of [`Self::chord_lb_press`] for `]`.
    chord_rb_press: Option<Instant>,
    /// Mirror of [`Self::chord_lb_release`] for `]`.
    chord_rb_release: Option<Instant>,
    /// Toggle for the H chord action — paints the chrome's hit_test_map as an opaque tinted overlay.
    show_hitmask: bool,
    /// 256-entry random colour table indexed by hit_test_map byte; regenerated each time H toggles on so distinct IDs get visibly-distinct colours. Photon's debug-hit pattern.
    debug_hit_colours: Vec<u32>,
    /// Chord-hint "was held last frame" toggle tracker so the panel area gets one frame of damage to clear stale pixels when both brackets release.
    last_chord_held: bool,
    /// Demo rotation angle for the rotating rect — advanced by mouse-wheel scroll, used by [`paint::draw_rect_rotated`]. Shows off the rect primitive + RU-scaling: dimensions derive from `viewport.effective_span()` so the rect grows/shrinks with Ctrl+/Ctrl-/Ctrl+scroll.
    rect_angle: Coord,
    /// Vertical scroll offset for the noise background — advanced by the same mouse-wheel events that rotate the rect. Passed straight to [`paint::background_noise`].
    bg_scroll: isize,
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
        // Decode the bundled app-icon orb (256×256 uncompressed VSF, hp+hb hashes). Bake-in via include_bytes! so the example is a single-file artefact at runtime — no on-disk asset lookup.
        let orb_bytes = include_bytes!("assets/example_orb.vsf");
        let app_icon = Icon::from_vsf_bytes(orb_bytes).ok();
        // Dense hit-id counter — chrome claims 1..=4 (min, max, close, app icon) because it constructs first; widget constructors that follow (textboxes etc.) get 5..N.
        let mut hit_counter: HitId = HIT_NONE;
        let chrome = DefaultChrome::new(
            viewport,
            title.clone(),
            app_icon,
            Some("ready".to_string()),
            &mut hit_counter,
        );

        // Placeholder textboxes + groups — actual geometry computed in `init`/`on_resize`. The helper `make_textbox_group` does the per-textbox group setup (content layer + a placeholder glow slot the future rasterize loop expects) so each new textbox is two lines: one Textbox::new, one matching group push. Repeat N times for N textboxes.
        let placeholder_region = Region::new(0.0, 0.0, 1.0, 1.0);
        let make_textbox_group = || {
            let mut g = Group::new(placeholder_region, BlendMode::Normal);
            let content_layer = g.new_layer();
            let _glow_layer = g.new_layer();
            g.set_program(vec![Op::Push(content_layer)]);
            g
        };
        let mut textboxes: Vec<Textbox> = Vec::new();
        let mut textbox_groups: Vec<Group> = Vec::new();
        for _ in 0..2 {
            let mut tb = Textbox::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 12.0);
            tb.stroke_ru = 1. / 12.;
            textboxes.push(tb);
            textbox_groups.push(make_textbox_group());
        }
        // Demo button — proves the Button widget participates in the same dispatch (click / focus / hover / keyboard activate) and tab cycle as Textbox. Same group pattern (content + placeholder glow layer); rasterize-into-target via `render_content_into`. Tick polls `take_click` and prints when fired.
        let mut buttons: Vec<Button> = Vec::new();
        let mut button_groups: Vec<Group> = Vec::new();
        for (i, label) in [("Submit"), ("Clear")].iter().enumerate() {
            let mut b = Button::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 12.0, *label);
            b.stroke_ru = 1. / 12.;
            let _ = i;
            buttons.push(b);
            button_groups.push(make_textbox_group());
        }
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
            textboxes,
            textbox_groups,
            buttons,
            button_groups,
            cursor_group,
            current_focus: None,
            hit_counter,
            rotation_group,
            blink: BlinkTimer::new(),
            is_dragging_select: false,
            selection_scroll_time: None,
            last_click_time: None,
            last_click_pos: (0.0, 0.0),
            click_count: 0,
            chord_lb_press: None,
            chord_lb_release: None,
            chord_rb_press: None,
            chord_rb_release: None,
            last_chord_held: false,
            show_hitmask: false,
            debug_hit_colours: Vec::new(),
            rect_angle: 0.0,
            bg_scroll: 0,
        }
    }

    /// Recompute textbox geometry from the viewport span and resize the textbox + cursor groups to match.
    fn update_layout(&mut self, ctx: &mut Context) {
        let vp = ctx.viewport;
        // Use effective_span (= span * ru) so all derived sizes pick up the user's zoom — Ctrl+/Ctrl-/Ctrl+scroll grow/shrink the textbox + cursor + rotation regions together with chrome.
        let span = vp.effective_span();
        let bw = span / 32.0;
        // Aspect-driven horizontal shift: square window centers the textbox; wider pushes it right, taller pushes it left. Magnitude scales with span so the shift is visible but bounded.
        let aspect = vp.width_px as Coord / vp.height_px as Coord;
        let center_x = vp.width_px as Coord * 0.5 + (aspect - 1.0) * span * 0.25;
        let center_y = bw * 7.0;
        let width = (vp.width_px as Coord * 0.5).max(bw * 8.0);
        let height = bw * 2.;
        let font_size = bw;
        // Stack textboxes vertically, one button-width apart starting at `center_y`. Works for N — just push more textboxes in `new()` and they fall into place. The cursor_group caches off the first textbox's bbox today; future work generalises that too.
        for (i, tb) in self.textboxes.iter_mut().enumerate() {
            let cy = center_y + (i as Coord) * (height + bw);
            tb.set_rect(center_x, cy, width, height);
            tb.set_font_size(font_size, ctx.text);
        }
        for (tb, g) in self.textboxes.iter().zip(self.textbox_groups.iter_mut()) {
            g.resize(tb.bbox());
        }
        if let Some(first) = self.textboxes.first() {
            self.cursor_group.resize(first.cursor_bbox());
        }
        // Buttons row below the textbox stack: half-width each, sitting side-by-side with a button-width gap, centred on the same column. Same RU-driven sizing — they grow / shrink with the window like the textboxes.
        let buttons_row_y = center_y + (self.textboxes.len() as Coord) * (height + bw);
        let button_w = (width * 0.5 - bw * 0.5).max(bw * 4.0);
        let button_count = self.buttons.len() as Coord;
        for (i, btn) in self.buttons.iter_mut().enumerate() {
            let offset = (i as Coord - (button_count - 1.0) * 0.5) * (button_w + bw);
            btn.set_rect(center_x + offset, buttons_row_y, button_w, height);
            btn.set_font_size(font_size);
        }
        for (btn, g) in self.buttons.iter().zip(self.button_groups.iter_mut()) {
            g.resize(btn.bbox());
        }

        let viewport_region = Region::new(0.0, 0.0, vp.width_px as Coord, vp.height_px as Coord);
        self.rotation_group.resize(viewport_region);
    }

    /// True iff both `[` and `]` are currently held. A bracket is "held" if its press timestamp is more recent than its release timestamp, OR the release was within [`CHORD_RELEASE_GRACE`] — that grace absorbs X11's habit of firing a synthetic Release for a held key the instant another key is pressed.
    fn brackets_held(&self, now: Instant) -> bool {
        fn key_held(
            press: Option<Instant>,
            release: Option<Instant>,
            now: Instant,
            grace: std::time::Duration,
        ) -> bool {
            match (press, release) {
                (Some(p), Some(r)) => p > r || now.duration_since(r) < grace,
                (Some(_), None) => true,
                _ => false,
            }
        }
        key_held(
            self.chord_lb_press,
            self.chord_lb_release,
            now,
            CHORD_RELEASE_GRACE,
        ) && key_held(
            self.chord_rb_press,
            self.chord_rb_release,
            now,
            CHORD_RELEASE_GRACE,
        )
    }

    /// Convenience: borrow whichever textbox currently has focus, or `None` for "focus is on a non-textbox widget or nothing." Iterates [`Self::textboxes`] so adding a new textbox doesn't touch this method. Used by the Ctrl+C / X / V clipboard interception path.
    fn focused_textbox_mut(&mut self) -> Option<&mut Textbox> {
        let focus = self.current_focus?;
        self.textboxes.iter_mut().find(|tb| tb.hit_id() == focus)
    }

    /// Index of the textbox with `id`, or `None`. Helper for `invalidate_group_by_id` so textbox + group stay in lockstep via index, not duplicated id-to-name match arms.
    fn textbox_index_by_id(&self, id: HitId) -> Option<usize> {
        self.textboxes.iter().position(|tb| tb.hit_id() == id)
    }

    /// Index of the button with `id`, or `None`. Same shape as `textbox_index_by_id`.
    fn button_index_by_id(&self, id: HitId) -> Option<usize> {
        self.buttons.iter().position(|b| b.hit_id() == id)
    }

    /// Invalidate the [`Group`] cache associated with `id`. Used by `change_focus` to mark the prior + new focused widget's group dirty so the next paint reflects the focus-glow on/off transition. No-op for ids that don't have a group (chrome buttons paint into chrome's monolithic layer; chrome handles its own dirty marking via `set_focused`).
    fn invalidate_group_by_id(&mut self, id: HitId) {
        if let Some(i) = self.textbox_index_by_id(id) {
            self.textbox_groups[i].invalidate();
        } else if let Some(i) = self.button_index_by_id(id) {
            self.button_groups[i].invalidate();
        }
    }

    /// Apply a focus change: drive `apply_focus_change` over the widget tree (which calls `set_focused` on the old + new targets), invalidate the involved groups so the focus-glow transition lands on the next paint, and start / stop the blink timer. Returns `true` if anything changed (no-op when `new == self.current_focus`).
    fn change_focus(&mut self, new_focus: Option<HitId>, ctx: &mut Context) -> bool {
        if new_focus == self.current_focus {
            return false;
        }
        let prior = self.current_focus;
        widget::apply_focus_change(self as &mut dyn Container, prior, new_focus);
        self.current_focus = new_focus;
        if let Some(id) = prior {
            self.invalidate_group_by_id(id);
        }
        if let Some(id) = new_focus {
            self.invalidate_group_by_id(id);
            self.blink.start(Instant::now());
        } else {
            self.blink.stop();
        }
        ctx.window.request_redraw();
        true
    }

    /// `click_count` capped at 3. Multi-click escalation only goes up to triple-click (select-all); past that the user is just clicking, treat as "still triple."
    fn click_count_capped(&self) -> u32 {
        self.click_count.min(3)
    }
}

/// Walk all textboxes + the chrome's four buttons in tab order. Textboxes first (content), then chrome buttons (window controls) — matches macOS / GNOME convention where Tab traverses form fields before window-frame controls. `linear_tab_next` reads this order off the visit walk to compute the next focusable id. Iterates [`PanesDemo::textboxes`] so the cycle expands naturally as more textboxes are pushed.
impl Container for PanesDemo {
    fn visit(&mut self, f: &mut dyn FnMut(&mut dyn fluor::host::widget::Widget)) {
        for tb in self.textboxes.iter_mut() {
            f(tb);
        }
        for btn in self.buttons.iter_mut() {
            f(btn);
        }
        self.chrome.visit(f);
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
        // Push the orb into the OS taskbar / window-list / alt-tab so the in-window chrome icon and the OS-level app icon are the same artifact. One-shot at startup — winit holds the icon for the surface's lifetime.
        if let Some(orb) = self.chrome.app_icon.as_ref() {
            if let Some(winit_icon) = orb.to_winit_icon() {
                ctx.window.set_window_icon(Some(winit_icon));
            }
        }
        self.update_layout(ctx);
    }

    fn on_resize(&mut self, w: u32, h: u32, ctx: &mut Context) {
        self.compositor.resize(w, h);
        self.chrome.resize(ctx.viewport);
        // Sync chrome's full-edge mode with the host's maximized state — they only diverge across a ToggleMaximized, which always triggers an on_resize (size always changes between user-sized and screen-sized).
        self.chrome.set_full_edge(ctx.is_maximized);
        self.update_layout(ctx);
    }

    fn on_event(&mut self, event: &WindowEvent, ctx: &mut Context) -> EventResponse {
        match event {
            WindowEvent::CursorMoved { .. } => {
                if self.is_dragging_select {
                    let focus_id = self.current_focus;
                    if let Some(tb) = self.focused_textbox_mut() {
                        let tl = tb.text_left();
                        let tr = tb.text_right();
                        let clamped_x = ctx.cursor_x.clamp(tl, tr);
                        if tb.selection_anchor.is_none() {
                            tb.selection_anchor = Some(tb.cursor);
                        }
                        tb.cursor = tb.cursor_index_from_x(clamped_x);
                    }
                    if let Some(id) = focus_id {
                        self.invalidate_group_by_id(id);
                    }
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                let new_hit = self.chrome.hit_at(ctx.cursor_x, ctx.cursor_y);
                let chrome_changed = self.chrome.set_hover(new_hit);
                // Iterate textboxes so adding more doesn't touch this. `any_textbox_changed` is the union-redraw signal.
                let mut any_widget_changed = false;
                for (tb, g) in self
                    .textboxes
                    .iter_mut()
                    .zip(self.textbox_groups.iter_mut())
                {
                    let want = new_hit == tb.hit_id();
                    if tb.is_hovered() != want {
                        tb.set_hovered(want);
                        g.invalidate();
                        any_widget_changed = true;
                    }
                }
                for (btn, g) in self.buttons.iter_mut().zip(self.button_groups.iter_mut()) {
                    let want = new_hit == btn.hit_id();
                    if btn.is_hovered() != want {
                        btn.set_hovered(want);
                        g.invalidate();
                        any_widget_changed = true;
                    }
                }
                if chrome_changed || any_widget_changed {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::CursorLeft { .. } => {
                let chrome_changed = self.chrome.set_hover(HIT_NONE);
                let mut any_widget_changed = false;
                for (tb, g) in self
                    .textboxes
                    .iter_mut()
                    .zip(self.textbox_groups.iter_mut())
                {
                    if tb.is_hovered() {
                        tb.set_hovered(false);
                        g.invalidate();
                        any_widget_changed = true;
                    }
                }
                for (btn, g) in self.buttons.iter_mut().zip(self.button_groups.iter_mut()) {
                    if btn.is_hovered() {
                        btn.set_hovered(false);
                        g.invalidate();
                        any_widget_changed = true;
                    }
                }
                if chrome_changed || any_widget_changed {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // Widget hit-test FIRST. The chrome buttons live inside the resize-edge bands at the top corners (close = top-right, min/max = top), so checking resize edge first would short-circuit chrome-button dispatch. Widget IDs take precedence; resize edge is the fallback when no widget owns the pixel.
                let hit_id = self.chrome.hit_at(ctx.cursor_x, ctx.cursor_y);
                if hit_id == HIT_NONE {
                    // Resize-edge — only consulted when no widget claimed the pixel.
                    let edge = chrome::get_resize_edge(
                        ctx.viewport.width_px,
                        ctx.viewport.height_px,
                        ctx.cursor_x,
                        ctx.cursor_y,
                    );
                    if edge != ResizeEdge::None {
                        // Defensively clear any in-progress selection-drag so cursor moves during resize can't bleed into textbox selection extension. Host suppresses CursorMoved dispatch during resize, but a prior interaction that didn't release cleanly could leave stale state.
                        self.is_dragging_select = false;
                        self.selection_scroll_time = None;
                        return EventResponse::StartResize(edge);
                    }
                    // Background click → clear focus + start window drag.
                    let had_focus = self.current_focus.is_some();
                    if had_focus {
                        self.change_focus(None, ctx);
                        self.is_dragging_select = false;
                    }
                    return EventResponse::StartWindowDrag;
                }
                // 3. Dispatch click + capture focus target via Container walk. The closure stays small — captures the event-response and focus-target slots; the actual focus side-effect is applied below where we can re-borrow self mutably.
                let x = ctx.cursor_x;
                let y = ctx.cursor_y;
                let mods = ctx.modifiers;
                let mut response = EventResponse::Pass;
                let mut focus_target: Option<HitId> = None;
                self.visit(&mut |w| {
                    if w.id() == hit_id {
                        if let Some(c) = w.click() {
                            response = c.on_click(x, y, mods);
                        }
                        if w.focus().is_some() {
                            focus_target = Some(hit_id);
                        }
                    }
                });
                // 4. Apply focus change (drives set_focused on old/new widgets + group invalidations + blink timer).
                if self.current_focus != focus_target {
                    self.change_focus(focus_target, ctx);
                }
                // 5. Textbox-specific post-click: multi-click escalation + selection-drag setup. Chrome buttons return their action response above and we propagate it unchanged.
                let is_textbox_focus = focus_target
                    .and_then(|id| self.textbox_index_by_id(id))
                    .is_some();
                if is_textbox_focus {
                    let now = Instant::now();
                    let is_continuation = match self.last_click_time {
                        Some(prev) => {
                            let dx = ctx.cursor_x - self.last_click_pos.0;
                            let dy = ctx.cursor_y - self.last_click_pos.1;
                            let dist_sq = dx * dx + dy * dy;
                            now.duration_since(prev) <= os_input::double_click_interval()
                                && dist_sq <= MULTI_CLICK_TOL_PX * MULTI_CLICK_TOL_PX
                        }
                        None => false,
                    };
                    self.click_count = if is_continuation {
                        self.click_count + 1
                    } else {
                        1
                    };
                    self.last_click_time = Some(now);
                    self.last_click_pos = (ctx.cursor_x, ctx.cursor_y);
                    let capped = self.click_count_capped();
                    if let Some(tb) = self.focused_textbox_mut() {
                        match capped {
                            2 => tb.select_word_at(tb.cursor),
                            3 => tb.select_all(),
                            _ => {}
                        }
                    }
                    // Cap the counter so a 4th-click doesn't try to "escalate" past select-all — treat the 4th click as another triple-cycle anchor.
                    if self.click_count >= 3 {
                        self.click_count = 3;
                    }
                    // Single click → arm drag-select (so mouse-drag extends the selection). Double / triple click → DON'T arm: the multi-click escalation just set a word / line selection, and any mouse jitter before the user releases would otherwise be interpreted by `CursorMoved` as "extend selection from anchor to cursor", clobbering the word / line selection with a character-position pair. Drag-extending a multi-click selection (word-by-word / line-by-line) is a future refinement.
                    self.is_dragging_select = self.click_count < 2;
                    self.selection_scroll_time = None;
                    if let Some(id) = focus_target {
                        self.invalidate_group_by_id(id);
                    }
                    self.blink.start(now);
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                response
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                if self.is_dragging_select {
                    self.is_dragging_select = false;
                    self.selection_scroll_time = None;
                    if let Some(tb) = self.focused_textbox_mut() {
                        if tb.selection_anchor == Some(tb.cursor) {
                            tb.selection_anchor = None;
                        }
                    }
                    self.blink.start(Instant::now());
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::KeyboardInput { event: kev, .. } => {
                let shift = ctx.modifiers.shift_key();
                let ctrl = ctx.modifiers.super_key() || ctx.modifiers.control_key();

                // --- Debug chord: `[` AND `]` held simultaneously + action key. Track Press/Release per bracket and treat the bracket as currently held iff its press timestamp is more recent than its release timestamp OR the release was within [`CHORD_RELEASE_GRACE`] (absorbs X11 synthetic-release-on-other-keypress so an action key during the chord doesn't spuriously disarm). Bracket presses themselves type into focused text as normal — we don't swallow them, since the cost of using a typeable key as the chord arm is that it types. Auto-repeat is suppressed for action keys so holding F doesn't toggle the FPS strip on every repeat tick.
                let mut action_char: Option<char> = None;
                if let Key::Character(c) = &kev.logical_key {
                    let cs = c.as_str();
                    let now = Instant::now();
                    match (cs, kev.state) {
                        ("[", ElementState::Pressed) => {
                            self.chord_lb_press = Some(now);
                        }
                        ("[", ElementState::Released) => {
                            self.chord_lb_release = Some(now);
                        }
                        ("]", ElementState::Pressed) => {
                            self.chord_rb_press = Some(now);
                        }
                        ("]", ElementState::Released) => {
                            self.chord_rb_release = Some(now);
                        }
                        (_, ElementState::Pressed) if !kev.repeat => {
                            // Only fire on the user's actual press, not on auto-repeat ticks. brackets_held is the only gate — the dispatch chain below decides what each letter does, and unknown letters fall through to `else { acted = false; }` as a no-op. No whitelist here: a second gating layer would just mean every new chord has to be added in two places, which silently breaks bindings when one is missed.
                            if self.brackets_held(now) {
                                action_char = c.to_ascii_lowercase().chars().next();
                            }
                        }
                        _ => {}
                    }
                    let _ = (shift, ctrl);
                }
                // Request a redraw whenever bracket state changes so the hint panel appears/disappears in lockstep with the user's grip.
                if matches!(&kev.logical_key, Key::Character(c) if c.as_str() == "[" || c.as_str() == "]")
                {
                    ctx.window.request_redraw();
                }
                if let Some(ac) = action_char {
                    eprintln!("[panes] chord fired: [+]+{}", ac.to_ascii_uppercase());
                    {
                        let mut acted = true;
                        if ac == 'h' {
                            self.show_hitmask = !self.show_hitmask;
                            // Sync the global atomic so finalize switches to the FORCE_OPAQUE debug branch and the host gates paint_shadow off while the hitmask viz is up. Promotes the next frame to a full repaint via the host's transition detector.
                            paint::DEBUG_SHOW_HITMASK
                                .store(self.show_hitmask, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("[]h hitmask = {}", self.show_hitmask);
                            if self.show_hitmask {
                                // Fill the 256-entry colour table with distinct random RGBs each toggle. xorshift32 seeded from process nanos — debug-quality, no crypto needed. Each call to toggle = fresh palette.
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
                                    let visible = (r << 16) | (g << 8) | b;
                                    let dark = visible ^ 0x00FFFFFF;
                                    self.debug_hit_colours.push(0xFF000000 | dark);
                                }
                            }
                        } else if ac == 'p' {
                            let cur = paint::DEBUG_SKIP_PREMULT
                                .load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SKIP_PREMULT
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("[]p skip-premult = {}", !cur);
                        } else if ac == 'a' {
                            // Cycle: off (0) → grayscale (1) → force-opaque (2) → off.
                            let cur =
                                paint::DEBUG_SHOW_ALPHA.load(std::sync::atomic::Ordering::Relaxed);
                            let next = (cur + 1) % 3;
                            paint::DEBUG_SHOW_ALPHA
                                .store(next, std::sync::atomic::Ordering::Relaxed);
                            let label = match next {
                                0 => "off",
                                1 => "grayscale",
                                _ => "force-opaque",
                            };
                            eprintln!("[]a show-alpha = {} ({})", next, label);
                        } else if ac == 'c' {
                            let cur =
                                paint::DEBUG_SKIP_CHROME.load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SKIP_CHROME
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                            self.chrome.invalidate_chrome();
                            ctx.window.request_redraw();
                            eprintln!("[]c skip-chrome = {}", !cur);
                        } else if ac == 'l' {
                            let cur = paint::DEBUG_SKIP_CONTROLS
                                .load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SKIP_CONTROLS
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                            self.chrome.invalidate_chrome();
                            ctx.window.request_redraw();
                            eprintln!("[]l skip-controls = {}", !cur);
                        } else if ac == 'r' {
                            self.chrome.group.invalidate();
                            for g in self.textbox_groups.iter_mut() {
                                g.invalidate();
                            }
                            self.cursor_group.invalidate();
                            self.rotation_group.invalidate();
                            eprintln!("[]r force-redraw");
                        } else if ac == 'f' {
                            // Toggle the host's bottom-of-window diagnostic strip.
                            let cur =
                                paint::DEBUG_SHOW_FPS.load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SHOW_FPS.store(!cur, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("[]f fps-strip = {}", !cur);
                        } else if ac == 'w' {
                            // Toggle the host's per-frame damage outline overlay (Where).
                            let cur =
                                paint::DEBUG_SHOW_DAMAGE.load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SHOW_DAMAGE
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("[]w damage-outline = {}", !cur);
                        } else if ac == 'd' {
                            // Toggle the host's screen-buffer decay. Each frame the host saturating-subtracts 8 from every persistent_screen RGB byte, so unrefreshed pixels visibly decay toward black while fresh writes from finalize / overlay stay bright. Diagnoses whether the incremental opaque-scan finalize is actually covering everything it should.
                            let cur =
                                paint::DEBUG_SHOW_FADE.load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SHOW_FADE
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("[]d screen-decay = {}", !cur);
                        } else if ac == 'b' {
                            // Toggle the finalize's opaque-scan blue-tint visualization. Each finalize-written pixel (clip_mask == 255) gets +16 to its blue byte (saturating). On []b transition the host promotes the next frame to a full_repaint, so toggling visibly washes the entire silhouette interior in one shot.
                            let cur = paint::DEBUG_SHOW_OPAQUE_SCAN
                                .load(std::sync::atomic::Ordering::Relaxed);
                            paint::DEBUG_SHOW_OPAQUE_SCAN
                                .store(!cur, std::sync::atomic::Ordering::Relaxed);
                            eprintln!("[]b opaque-scan tint = {}", !cur);
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
                // Tab / Shift+Tab → focus cycle. Runs BEFORE clipboard interception and BEFORE delivery to the focused widget so a textbox can't swallow Tab as a "\t" insertion. `linear_tab_next` walks the Container in registration order; with our visit order (textbox_a → textbox_b → chrome buttons) Tab advances textbox_a → textbox_b → app_icon → min → max → close → wrap.
                if matches!(kev.logical_key, Key::Named(NamedKey::Tab)) {
                    let dir = if shift {
                        TabDir::Backward
                    } else {
                        TabDir::Forward
                    };
                    let current = self.current_focus;
                    let next = widget::linear_tab_next(self as &mut dyn Container, current, dir);
                    self.change_focus(next, ctx);
                    return EventResponse::Handled;
                }
                // Escape → clear focus.
                if matches!(kev.logical_key, Key::Named(NamedKey::Escape)) {
                    if self.current_focus.is_some() {
                        self.change_focus(None, ctx);
                    }
                    return EventResponse::Handled;
                }
                let Some(focus_id) = self.current_focus else {
                    return EventResponse::Pass;
                };
                // Enter / Space on a chrome button activates it. The widget's Key::on_key returns the action's EventResponse; we propagate it so the host fires Close / Minimize / ToggleMaximized as if the button were clicked.
                // Clipboard interception (textbox-focused only). Ctrl+C / Ctrl+X / Ctrl+V need the OS clipboard adapter (arboard) which is a single global resource — threading it through every widget that might want clipboard access would be premature abstraction at one consumer. Apps handle the chord before delivering to the focused widget; the widget never sees Ctrl+C/X/V. Ctrl+A is widget-internal (select-all) and Textbox::on_key handles it.
                let focused_is_textbox = self.textbox_index_by_id(focus_id).is_some();
                if focused_is_textbox {
                    if let Key::Character(c) = &kev.logical_key {
                        if ctrl {
                            let lower = c.to_ascii_lowercase();
                            // Hoist `ctx.text` reborrow out of the match arms so the focused-textbox borrow (which extends until end-of-arm via `tb`) doesn't conflict with the renderer borrow. `text` here is a fresh `&mut TextRenderer` per match arm; rust's NLL closes it cleanly when each arm returns.
                            match lower.as_str() {
                                "c" => {
                                    if let Some(tb) = self.focused_textbox_mut() {
                                        if let Some(selected) = tb.selected_text() {
                                            if let Ok(mut clip) = arboard::Clipboard::new() {
                                                let _ = clip.set_text(selected);
                                            }
                                        }
                                    }
                                    return EventResponse::Handled;
                                }
                                "x" => {
                                    let mut did_change = false;
                                    if let Some(tb) = self.focused_textbox_mut() {
                                        if let Some(selected) = tb.selected_text() {
                                            if arboard::Clipboard::new()
                                                .and_then(|mut clip| clip.set_text(selected))
                                                .is_ok()
                                            {
                                                tb.delete_selection(&mut *ctx.text);
                                                did_change = true;
                                            }
                                        }
                                    }
                                    if did_change {
                                        self.invalidate_group_by_id(focus_id);
                                        self.blink.start(Instant::now());
                                        ctx.window.request_redraw();
                                    }
                                    return EventResponse::Handled;
                                }
                                "v" => {
                                    let mut did_change = false;
                                    if let Some(tb) = self.focused_textbox_mut() {
                                        if let Ok(mut clip) = arboard::Clipboard::new() {
                                            if let Ok(paste) = clip.get_text() {
                                                tb.insert_str(&paste, &mut *ctx.text);
                                                did_change = true;
                                            }
                                        }
                                    }
                                    if did_change {
                                        self.invalidate_group_by_id(focus_id);
                                        self.blink.start(Instant::now());
                                        ctx.window.request_redraw();
                                    }
                                    return EventResponse::Handled;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                // Deliver remaining keys to the focused widget via the Key trait. Walk the tree, match on id, call on_key. The closure body holds the only `&mut TextRenderer` it needs (via the explicit reborrow); `response` is captured as `&mut EventResponse` and mutated on the match.
                let modifiers = ctx.modifiers;
                let text = &mut *ctx.text;
                let mut response = EventResponse::Pass;
                self.visit(&mut |w| {
                    if w.id() == focus_id {
                        if let Some(k) = w.key() {
                            response = k.on_key(kev, modifiers, text);
                        }
                    }
                });
                if matches!(response, EventResponse::Handled) && focused_is_textbox {
                    self.invalidate_group_by_id(focus_id);
                    self.blink.start(Instant::now());
                    ctx.window.request_redraw();
                }
                response
            }
            WindowEvent::Focused(focused) => {
                if self.chrome.set_focused(*focused) {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Scroll-driven demo: each notch rotates the rect by ~6° (dimensionless, size-independent) AND shifts the noise background by `1/128th` of `effective_span` (size-independent — same visual amount on tiny and 4K windows). Trackpad pixel deltas accumulate at 32 raw-px per notch. Both denominators are powers of two so the f32 divides are mantissa-exact and the optimiser can collapse them to exponent adjusts.
                let steps: Coord = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as Coord) / 32.0,
                };
                if steps != 0.0 {
                    self.rect_angle += steps * 0.125;
                    self.bg_scroll += (steps * ctx.viewport.effective_span() / 128.0) as isize;
                    self.chrome.invalidate_bg();
                    ctx.window.request_redraw();
                    return EventResponse::Handled;
                }
                EventResponse::Pass
            }
            _ => EventResponse::Pass,
        }
    }

    fn damage_rect(&self, viewport: fluor::geom::Viewport) -> Option<fluor::canvas::PixelRect> {
        let vw = viewport.width_px as usize;
        let vh = viewport.height_px as usize;
        let mut combined: Option<fluor::canvas::PixelRect> = None;
        let mut union_in = |r: Option<fluor::canvas::PixelRect>| {
            if let Some(r) = r {
                combined = Some(combined.map_or(r, |c| c.union(r)));
            }
        };
        union_in(self.chrome.damage_rect());
        for tb in self.textboxes.iter() {
            union_in(tb.damage_rect(vw, vh));
        }
        for btn in self.buttons.iter() {
            union_in(btn.damage_rect(vw, vh));
        }
        // Chord hint overlay — when both `[` and `]` are held the hint panel paints into the viewport center. Union its bbox so the host's damage_clip + finalize cover it. Also covers the "was held last frame" case (toggle off) by checking last_chord_held — clears stale hint pixels in one frame.
        let held_now = self.brackets_held(Instant::now());
        if held_now || self.last_chord_held {
            union_in(Some(chord_hint_bbox(viewport, vw, vh)));
        }
        combined
    }

    fn hit_test_map(&self) -> Option<(&[HitId], usize, usize)> {
        let (w, h) = self.chrome.dims();
        Some((self.chrome.hit_test_map(), w, h))
    }

    fn overlay_deltas(&self) -> Vec<u32> {
        // Per-hit-id tint deltas applied to persistent_screen by the host's overlay pass. Slice sized to the live hit-id count (hit_counter + 1 since IDs are 1-indexed and HIT_NONE = 0 takes slot 0).
        let mut t = vec![0u32; self.hit_counter as usize + 1];
        if let Some(c) = fluor::host::chrome_widget::hover_colour_for(self.chrome.hover_state) {
            t[self.chrome.hover_state as usize] = c;
        }
        // Same focus / hover → tint formula applied to each textbox at its own dense hit id. Generalises to a Container walk in Phase 5 once we have a way for widgets to surface their tint contribution through the trait.
        let tb_tint = |tb: &Textbox| -> u32 {
            if tb.is_focused() {
                paint::wrap_sub_rgb(fluor::theme::TEXTBOX_ACTIVE, fluor::theme::TEXTBOX_FILL)
            } else if tb.is_hovered() {
                paint::wrap_sub_rgb(fluor::theme::TEXTBOX_HOVER, fluor::theme::TEXTBOX_FILL)
            } else {
                0
            }
        };
        for tb in self.textboxes.iter() {
            t[tb.hit_id() as usize] = tb_tint(tb);
        }
        // Buttons share the same fill / hover / active palette in the demo. Both widgets read from the same theme constants so a Button next to a Textbox reads as the same family.
        let btn_tint = |b: &Button| -> u32 {
            if b.is_focused() {
                paint::wrap_sub_rgb(fluor::theme::TEXTBOX_ACTIVE, fluor::theme::TEXTBOX_FILL)
            } else if b.is_hovered() {
                paint::wrap_sub_rgb(fluor::theme::TEXTBOX_HOVER, fluor::theme::TEXTBOX_FILL)
            } else {
                0
            }
        };
        for btn in self.buttons.iter() {
            t[btn.hit_id() as usize] = btn_tint(btn);
        }
        t
    }

    fn render(&mut self, target: &mut [u32], ctx: &mut Context) {
        let buf_w = ctx.viewport.width_px as usize;
        let buf_h = ctx.viewport.height_px as usize;

        // Hairline + background + hover overlay. Chrome's bg layer gets photon's background_noise; chrome layer gets the perimeter hairline + controls (or stays empty under `[]c`); hover layer gets a partial-α tint over the currently-hovered button (or stays empty if hover_state == HIT_NONE). The chrome group's Stack program (`Push hover, Push chrome, Under(Normal), Push bg, Under(Normal)`) front-to-back-composites them, then flattens under the target. Order matters: rasterize_chrome MUST run before rasterize_hover because hover reads `hit_test_map` which chrome populates.
        //
        // Shape demos: each paint primitive gets a partial-transparency instance, all painted FIRST
        // (topmost-first doctrine), then noise composes behind via `under()` — scrolled vertically
        // by `self.bg_scroll`. All sizes/positions derive from `viewport.effective_span()` so they
        // stay RU-coherent across window sizes.
        //
        // Rects: 50% cyan rotating + 25% orange aligned overlapping it.
        // Ellipses: aligned magenta matching the window's aspect ratio, plus a 2:1 yellow rotated
        // ellipse spinning OPPOSITE the rect at 1/3 speed.
        // Circle: 50% pink, fixed.
        let span = ctx.viewport.effective_span();
        let view_w = ctx.viewport.width_px as Coord;
        let view_h = ctx.viewport.height_px as Coord;
        let aspect = view_w / view_h;
        let cx = view_w * 0.5;
        let cy = view_h * 0.7;
        let rect_w = span / 8.0;
        let rect_h = span / 24.0;
        let rect_colour = pack_argb(80, 220, 220, 0x80);
        let static_w = span / 10.0;
        let static_h = span / 16.0;
        let static_cx = cx + rect_w * 0.35;
        let static_cy = cy - rect_h * 0.6;
        let static_colour = pack_argb(255, 180, 80, 0x40);
        // Circle.
        let circle_cx = view_w * 0.25;
        let circle_cy = view_h * 0.3;
        let circle_r = span / 20.0;
        let circle_colour = pack_argb(255, 120, 200, 0x80);
        // Aligned ellipse — aspect matches the window.
        let ellipse_cx = view_w * 0.5;
        let ellipse_cy = view_h * 0.3;
        let ellipse_ry = span / 24.0;
        let ellipse_rx = ellipse_ry * aspect;
        let ellipse_colour = pack_argb(200, 120, 255, 0x80);
        // Rotated ellipse — 2:1, opposite direction at 1/3 speed.
        let rot_ellipse_cx = view_w * 0.75;
        let rot_ellipse_cy = view_h * 0.3;
        let rot_ellipse_rx = span / 14.0;
        let rot_ellipse_ry = rot_ellipse_rx * 0.5;
        let rot_ellipse_colour = pack_argb(255, 230, 100, 0x80);
        let angle = self.rect_angle;
        let ellipse_angle = -self.rect_angle / 3.0;
        let bg_scroll = self.bg_scroll;
        self.chrome.rasterize_bg(ctx.damage, move |canvas| {
            paint::draw_rect_rotated(canvas, cx, cy, rect_w, rect_h, angle, rect_colour, None);
            paint::draw_rect(
                canvas,
                static_cx,
                static_cy,
                static_w,
                static_h,
                static_colour,
                None,
            );
            paint::draw_circle(canvas, circle_cx, circle_cy, circle_r, circle_colour, None);
            paint::draw_ellipse(
                canvas,
                ellipse_cx,
                ellipse_cy,
                ellipse_rx,
                ellipse_ry,
                ellipse_colour,
                None,
            );
            paint::draw_ellipse_rotated(
                canvas,
                rot_ellipse_cx,
                rot_ellipse_cy,
                rot_ellipse_rx,
                rot_ellipse_ry,
                ellipse_angle,
                rot_ellipse_colour,
                None,
            );
            paint::background_noise(canvas, 0, true, bg_scroll, None);
        });
        self.chrome
            .rasterize_chrome(ctx.damage, ctx.text, ctx.clip_mask);
        self.chrome.rasterize_hover();

        // Blinkey is no longer rasterized into a scratch-side cursor_group. It now lives entirely on the host's persistent_screen buffer, painted post-finalize via `FluorApp::paint_screen_overlay` → `Textbox::paint_blinkey_into_screen`. Wrap-add on / wrap-sub off, ~hundreds of bytes touched per blink — no scratch fill, no flatten, no chrome re-composite.

        // Chord-hint overlay — visible while both brackets are held. Painted into `target` BEFORE every flatten so the hint glyphs sit at the TOP of the under-blend chain; everything else composes UNDER them. Track held state for next frame's `damage_rect` (need a one-frame clear when released).
        let held_now = self.brackets_held(Instant::now());
        self.last_chord_held = held_now;
        if held_now {
            let span = ctx.viewport.effective_span();
            let mut canvas = fluor::canvas::Canvas::new(target, buf_w, buf_h, ctx.damage);
            paint::draw_chord_hint(&mut canvas, ctx.text, CHORD_HINTS, span);
        }

        // Topmost-first chain. The textbox is painted DIRECTLY into target (no intermediate layer) so the squircle's per-pixel `under()` writes compose against the final under-chain accumulator — one premult, not two. Order: textbox squircle (topmost) → chrome (under). Blinkey lives entirely on the host's persistent_screen post-finalize; the cursor_group flatten is gone.
        //
        // Every step is clipped to `ctx.damage_clip` so only the damaged region gets touched — outside the clip, scratch persists from the previous frame.
        let clip = pixelrect_to_clip(ctx.damage_clip);
        {
            let mut canvas = fluor::canvas::Canvas::new(target, buf_w, buf_h, ctx.damage);
            // Each textbox stamps its own pill silhouette into the shared hit_test_map at its own dense hit id — click + hover dispatch routes by `hit_map[idx]` and finds the matching widget via Container::visit. Iterate so N textboxes is N stamps.
            for tb in self.textboxes.iter_mut() {
                let id = tb.hit_id();
                tb.render_content_into(
                    &mut canvas,
                    0.0,
                    0.0,
                    ctx.text,
                    clip,
                    None,
                    Some(&mut self.chrome.hit_test_map),
                    id,
                );
            }
            for btn in self.buttons.iter_mut() {
                let id = btn.hit_id();
                btn.render_content_into(
                    &mut canvas,
                    0.0,
                    0.0,
                    ctx.text,
                    clip,
                    Some(&mut self.chrome.hit_test_map),
                    id,
                );
            }
        }
        // Textbox tint is now baked into its own cache by `render_content_into` (Photon-style differential) — no per-frame walk over `hit_test_map` needed here.
        self.chrome.flatten_into(target, buf_w, buf_h, clip);

        // Debug overlay (photon-style): for every pixel, look up the hit_test_map's ID and paint its opaque random colour from `debug_hit_colours`. Fully replaces the underlying image — distinct hit zones are visually unmistakable. Drawn last over everything (including textbox + cursor) since hit testing is per-final-pixel anyway. Bounds check on the colour-table index keeps the post-u16 widening safe: real widget IDs in this demo stay well under 256, but a future stale stamp at an unregistered high id would panic without the `.get`.
        if self.show_hitmask && !self.debug_hit_colours.is_empty() {
            let map = &self.chrome.hit_test_map;
            let n = map.len().min(target.len());
            for i in 0..n {
                target[i] = self
                    .debug_hit_colours
                    .get(map[i] as usize)
                    .copied()
                    .unwrap_or(0);
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
                if hit == HIT_TEXTBOX {
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

        // Selection drag: auto-scroll the focused textbox's content while the cursor is held outside its bounds.
        if self.is_dragging_select {
            let cursor_x = ctx.cursor_x;
            let focus_id = self.current_focus;
            let mut did_scroll = false;
            let last_scroll = self.selection_scroll_time;
            let now = Instant::now();
            if let Some(tb) = self.focused_textbox_mut() {
                let tl = tb.text_left();
                let tr = tb.text_right();
                let distance_outside = if cursor_x < tl {
                    tl - cursor_x
                } else if cursor_x > tr {
                    cursor_x - tr
                } else {
                    0.0
                };
                if distance_outside > 0.0 {
                    let dt = last_scroll
                        .map(|t| now.duration_since(t).as_secs_f32())
                        .unwrap_or(0.0);
                    let uw = tb.usable_width();
                    // Drag-scroll speed (pixels/second) ≈ 1024 × (distance-past-edge / usable_width). Power of two so f32 multiply is exact (mantissa-exact, no rounding) and the compiler can fold to an exponent adjust where it sees fit. Visually equivalent to the previous 1000.0 (within 2.4 % — imperceptible at human reaction times).
                    let speed = 1024.0 * distance_outside / uw;
                    let delta = speed * dt;
                    if cursor_x < tl {
                        tb.nudge_scroll_offset(delta);
                    } else {
                        tb.nudge_scroll_offset(-delta);
                    }
                    let clamped_x = cursor_x.clamp(tl, tr);
                    tb.cursor = tb.cursor_index_from_x(clamped_x);
                    did_scroll = true;
                }
            }
            if did_scroll {
                self.selection_scroll_time = Some(now);
                if let Some(id) = focus_id {
                    self.invalidate_group_by_id(id);
                }
                needs_redraw = true;
            } else {
                self.selection_scroll_time = None;
            }
        }

        // Blink timer. flip whichever textbox is currently focused so both A and B's blinkeys animate. flip_blinkey is a no-op on an unfocused textbox.
        if self.blink.poll(Instant::now()) {
            if let Some(tb) = self.focused_textbox_mut() {
                if tb.flip_blinkey() {
                    needs_redraw = true;
                }
            }
        }

        // Button poll — each button's internal click counter is consumed by `take_click`; we react with a stderr print so it's obvious the dispatch path is alive. App-defined behaviour (clearing a textbox, submitting a form, etc.) hangs off the same boolean per button.
        for btn in self.buttons.iter_mut() {
            if btn.take_click() {
                eprintln!("[button] '{}' clicked", btn.label());
            }
        }

        let _ = ctx;
        needs_redraw
    }
}

fn main() {
    let demo = PanesDemo::new(Viewport::new(1280, 800), "fluor — panes");
    fluor::host::app::run_app(demo).expect("event loop");
}
