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
use fluor::paint::pack_argb;
use fluor::paint::{self, BlendMode, Transform};
use fluor::region::Region;
use fluor::stack::Op;
use fluor::theme;
use fluor::widgets::{BlinkTimer, Textbox};
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
    textbox: Textbox,
    textbox_group: Group,
    cursor_group: Group,
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

        // Placeholder textbox + groups — actual geometry computed in `init`/`on_resize`.
        // Solid-fill-only iteration: textbox_group has just the content layer in its program. We still allocate a glow layer slot so the existing rasterize loop can clear it without panicking, but we don't fold it into the composite — that avoids double premultiplication (an empty glow on top of content via under() premultiplies content once, then `flatten_into` to target premultiplies it again, brightening every AA edge). When we wire glow back in we'll choose a compose that does the math correctly.
        let mut textbox = Textbox::new(&mut hit_counter, 0.0, 0.0, 1.0, 1.0, 12.0);
        textbox.stroke_ru = 1. / 12.; // thin-but-non-zero so the diagonal two-tone reads as a visible band, not just a hairline
        let placeholder_region = Region::new(0.0, 0.0, 1.0, 1.0);
        let mut textbox_group = Group::new(placeholder_region, BlendMode::Normal);
        let content_layer = textbox_group.new_layer();
        let _glow_layer = textbox_group.new_layer();
        textbox_group.set_program(vec![Op::Push(content_layer)]);
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
        self.textbox.set_rect(center_x, center_y, width, height);
        self.textbox.set_font_size(font_size, ctx.text);

        self.textbox_group.resize(self.textbox.bbox());
        self.cursor_group.resize(self.textbox.cursor_bbox());

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
        // Sync chrome's full-edge mode with the host's maximized state — they only diverge across a ToggleMaximized, which always triggers an on_resize (size always changes between user-sized and screen-sized).
        self.chrome.set_full_edge(ctx.is_maximized);
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
                let new_hit = self.chrome.hit_at(ctx.cursor_x, ctx.cursor_y);
                let chrome_changed = self.chrome.set_hover(new_hit);
                let new_textbox_hover = new_hit == HIT_TEXTBOX;
                let textbox_changed = self.textbox.is_hovered() != new_textbox_hover;
                if textbox_changed {
                    self.textbox.set_hovered(new_textbox_hover);
                    self.textbox_group.invalidate();
                }
                if chrome_changed || textbox_changed {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::CursorLeft { .. } => {
                let chrome_changed = self.chrome.set_hover(HIT_NONE);
                let textbox_changed = self.textbox.is_hovered();
                if textbox_changed {
                    self.textbox.set_hovered(false);
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
                    HIT_MAXIMIZE_BUTTON => return EventResponse::ToggleMaximized,
                    _ => {}
                }
                let edge = chrome::get_resize_edge(
                    ctx.viewport.width_px,
                    ctx.viewport.height_px,
                    ctx.cursor_x,
                    ctx.cursor_y,
                );
                if edge != ResizeEdge::None {
                    // Starting a resize-drag — defensively clear any in-progress selection-drag state so cursor moves during the resize can't bleed into the textbox's "extend selection" path. Host suppresses CursorMoved dispatch during resize, but if `is_dragging_select` was set from a prior interaction that didn't release cleanly, the post-resize state would still have stale drag flags.
                    self.is_dragging_select = false;
                    self.selection_scroll_time = None;
                    return EventResponse::StartResize(edge);
                }
                let was_focused = self.textbox.is_focused();
                self.textbox.handle_click(ctx.cursor_x, ctx.cursor_y);
                if self.textbox.is_focused() {
                    // Multi-click sequence: continuation iff the gap to the previous press is within the OS double-click interval AND the cursor hasn't moved more than `MULTI_CLICK_TOL_PX`. count==2 → word-select around current cursor; count==3 → select-all. Reset to 1 on a non-continuation. We let `handle_click` set the cursor first so the word-select probe targets the actual clicked char.
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
                    self.click_count = if is_continuation { self.click_count + 1 } else { 1 };
                    self.last_click_time = Some(now);
                    self.last_click_pos = (ctx.cursor_x, ctx.cursor_y);
                    match self.click_count {
                        2 => self.textbox.select_word_at(self.textbox.cursor),
                        n if n >= 3 => {
                            self.textbox.select_all();
                            // Cap the counter so a 4th-click doesn't try to "escalate" past select-all — at that point the user is just clicking, treat the 4th click as another triple-cycle anchor.
                            self.click_count = 3;
                        }
                        _ => {}
                    }
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
                            self.textbox_group.invalidate();
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
                if !self.textbox.is_focused() {
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
            WindowEvent::Focused(focused) => {
                if self.chrome.set_focused(*focused) {
                    ctx.window.request_redraw();
                }
                EventResponse::Pass
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Scroll-driven demo: each notch rotates the rect by ~6° (dimensionless, size-independent) AND shifts the noise background by 1/100th of `effective_span` (size-independent — same visual amount on tiny and 4K windows). Trackpad pixel deltas accumulate at ~30 raw-px per notch to match the zoom shortcut's feel.
                let steps: Coord = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as Coord) / 30.0,
                };
                if steps != 0.0 {
                    self.rect_angle += steps * 0.1;
                    self.bg_scroll += (steps * ctx.viewport.effective_span() / 100.0) as isize;
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
        union_in(self.textbox.damage_rect(vw, vh));
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
        // Per-hit-id tint deltas applied to persistent_screen by the host's overlay pass. Slice sized to the highest live hit id + 1 (HIT_TEXTBOX is the max here). Most apps will allocate this from `registry.next_id` once it exists; for now the demo uses the compat constants.
        let mut t = vec![0u32; HIT_TEXTBOX as usize + 1];
        if let Some(c) = fluor::host::chrome_widget::hover_color_for(self.chrome.hover_state) {
            t[self.chrome.hover_state as usize] = c;
        }
        t[HIT_TEXTBOX as usize] = if self.textbox.is_focused() {
            paint::wrap_sub_rgb(fluor::theme::TEXTBOX_ACTIVE, fluor::theme::TEXTBOX_FILL)
        } else if self.textbox.is_hovered() {
            paint::wrap_sub_rgb(fluor::theme::TEXTBOX_HOVER, fluor::theme::TEXTBOX_FILL)
        } else {
            0
        };
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
        let rect_color = pack_argb(80, 220, 220, 0x80);
        let static_w = span / 10.0;
        let static_h = span / 16.0;
        let static_cx = cx + rect_w * 0.35;
        let static_cy = cy - rect_h * 0.6;
        let static_color = pack_argb(255, 180, 80, 0x40);
        // Circle.
        let circle_cx = view_w * 0.25;
        let circle_cy = view_h * 0.3;
        let circle_r = span / 20.0;
        let circle_color = pack_argb(255, 120, 200, 0x80);
        // Aligned ellipse — aspect matches the window.
        let ellipse_cx = view_w * 0.5;
        let ellipse_cy = view_h * 0.3;
        let ellipse_ry = span / 24.0;
        let ellipse_rx = ellipse_ry * aspect;
        let ellipse_color = pack_argb(200, 120, 255, 0x80);
        // Rotated ellipse — 2:1, opposite direction at 1/3 speed.
        let rot_ellipse_cx = view_w * 0.75;
        let rot_ellipse_cy = view_h * 0.3;
        let rot_ellipse_rx = span / 14.0;
        let rot_ellipse_ry = rot_ellipse_rx * 0.5;
        let rot_ellipse_color = pack_argb(255, 230, 100, 0x80);
        let angle = self.rect_angle;
        let ellipse_angle = -self.rect_angle / 3.0;
        let bg_scroll = self.bg_scroll;
        self.chrome.rasterize_bg(ctx.damage, move |canvas| {
            paint::draw_rect_rotated(canvas, cx, cy, rect_w, rect_h, angle, rect_color, None);
            paint::draw_rect(
                canvas,
                static_cx,
                static_cy,
                static_w,
                static_h,
                static_color,
                None,
            );
            paint::draw_circle(canvas, circle_cx, circle_cy, circle_r, circle_color, None);
            paint::draw_ellipse(
                canvas,
                ellipse_cx,
                ellipse_cy,
                ellipse_rx,
                ellipse_ry,
                ellipse_color,
                None,
            );
            paint::draw_ellipse_rotated(
                canvas,
                rot_ellipse_cx,
                rot_ellipse_cy,
                rot_ellipse_rx,
                rot_ellipse_ry,
                ellipse_angle,
                rot_ellipse_color,
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
            self.textbox.render_content_into(
                &mut canvas,
                0.0,
                0.0,
                ctx.text,
                clip,
                None,
                Some(&mut self.chrome.hit_test_map),
                HIT_TEXTBOX,
            );
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
                    self.textbox.nudge_scroll_offset(delta);
                } else {
                    self.textbox.nudge_scroll_offset(-delta);
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

        let _ = ctx;
        needs_redraw
    }
}

fn main() {
    let demo = PanesDemo::new(Viewport::new(1280, 800), "fluor — panes");
    fluor::host::app::run_app(demo).expect("event loop");
}
