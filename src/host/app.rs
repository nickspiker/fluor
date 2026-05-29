//! `FluorApp` trait + entry point for consumer-driven desktop apps.
//!
//! Consumers implement [`FluorApp`] and pass the impl to [`run_app`]. The host opens a window, runs the event loop, presents the buffer, and dispatches events through the trait. All visible content (chrome, widgets, panes) is the consumer's responsibility — the host owns no domain state.
//!
//! Compose [`super::chrome_widget::DefaultChrome`] for the borderless window frame, [`crate::widgets::Textbox`] / [`crate::widgets::BlinkTimer`] for the textbox + blinking-cursor pattern, [`crate::Group`] for sub-viewport composite caching. The [`Context`] struct exposes the host's shared resources (viewport, text renderer, window handle, modifier state) to the consumer for the duration of each callback.
//!
//! The current `desktop::run(compositor, title)` is a transitional shim that wraps the legacy demo into a `FluorApp`. New code should use [`run_app`] directly.

use super::chrome::ResizeEdge;
use crate::coord::Coord;
use crate::geom::Viewport;
use crate::text::TextRenderer;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{CursorIcon, Window, WindowAttributes, WindowId};

/// X11-only: issue a SINGLE `XConfigureWindow` with all four of `(x, y, width, height)` so the WM applies position AND size atomically — eliminates the visible "first-size-then-position" seam you get when winit's separate `set_outer_position` / `request_inner_size` calls each generate their own ConfigureRequest. Returns `true` if the atomic call succeeded (window is X11 and the request was sent); `false` if the window is Wayland or the X11 connection failed → caller falls back to winit's separate calls (which is the correct path on Wayland anyway, since `set_outer_position` is a no-op there).
#[cfg(target_os = "linux")]
mod x11_atomic {
    use std::sync::OnceLock;
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use x11rb::connection::Connection;
    use x11rb::protocol::xproto::{ConfigureWindowAux, ConnectionExt};
    use x11rb::rust_connection::RustConnection;

    /// Lazily-opened XCB connection, shared across all atomic-geometry calls. Independent of the connection winit holds internally (which we can't access) — the X server doesn't care which client sends the ConfigureRequest as long as we name the right window ID.
    fn conn() -> Option<&'static RustConnection> {
        static CONN: OnceLock<Option<RustConnection>> = OnceLock::new();
        CONN.get_or_init(|| x11rb::connect(None).ok().map(|(c, _screen)| c))
            .as_ref()
    }

    pub fn set_geometry(window: &winit::window::Window, x: i32, y: i32, w: u32, h: u32) -> bool {
        let Ok(handle) = window.window_handle() else {
            return false;
        };
        let xid = match handle.as_raw() {
            RawWindowHandle::Xcb(h) => h.window.get(),
            RawWindowHandle::Xlib(h) => h.window as u32,
            _ => return false, // Wayland or other non-X11 — caller falls back to winit
        };
        let Some(conn) = conn() else {
            return false;
        };
        let aux = ConfigureWindowAux::new()
            .x(Some(x))
            .y(Some(y))
            .width(Some(w))
            .height(Some(h));
        if conn.configure_window(xid, &aux).is_err() {
            return false;
        }
        let _ = conn.flush();
        true
    }

    /// Restrict the window's INPUT region to the given screen-space rectangle. Clicks outside this rect pass through to whatever window is behind us. Used by the fullscreen-compositor architecture: our OS surface covers the whole screen but the visible window is just a sub-rect, so we tell X11 "I'm only hittable inside that sub-rect" — the rest is mouse-transparent. Call once per `window_rect` change (initial creation, drag-to-move, resize-drag, monitor change).
    ///
    /// The rect is in window-relative coordinates (= screen coords when the OS window is fullscreen at the screen origin). Negative offsets get clamped to 0 since XShape rectangles must be unsigned. Returns `true` if the call was sent successfully, `false` if the window isn't X11 or the connection failed.
    pub fn set_input_region(
        window: &winit::window::Window,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
    ) -> bool {
        use x11rb::protocol::shape::{ConnectionExt as _, SK, SO};
        use x11rb::protocol::xproto::{ClipOrdering, Rectangle};

        let Ok(handle) = window.window_handle() else {
            return false;
        };
        let xid = match handle.as_raw() {
            RawWindowHandle::Xcb(h) => h.window.get(),
            RawWindowHandle::Xlib(h) => h.window as u32,
            _ => return false,
        };
        let Some(conn) = conn() else {
            return false;
        };
        let rect = Rectangle {
            x: x.max(0).min(i16::MAX as i32) as i16,
            y: y.max(0).min(i16::MAX as i32) as i16,
            width: w.min(u16::MAX as u32) as u16,
            height: h.min(u16::MAX as u32) as u16,
        };
        if conn
            .shape_rectangles(SO::SET, SK::INPUT, ClipOrdering::UNSORTED, xid, 0, 0, &[rect])
            .is_err()
        {
            return false;
        }
        let _ = conn.flush();
        true
    }
}

/// Per-callback access to host-owned shared resources. Re-borrowed for each call into the trait — the consumer can keep references for the duration of the call but not across calls.
pub struct Context<'a> {
    /// Current viewport in physical pixels.
    pub viewport: Viewport,
    /// Shared font system + glyph caches. Initialized lazily by the host on first window creation; passed by mutable reference because cache insertion and font loading require mutation.
    pub text: &'a mut TextRenderer,
    /// Window-shape clip mask, one byte per pixel, same dimensions as the present buffer. Default fill is `255` (fully visible — finalize_for_os multiplies by 255 ≈ no change). Consumers with a rounded window (e.g. `DefaultChrome`) carve the corner cutouts here once per resize; the boundary's [`crate::paint::finalize_for_os`] multiplies it into each pixel's α to trim the OS handoff. Decoupled from the present-buffer RGB so internal layer compositing stays opaque-or-empty and never deals with partial-α drift.
    pub clip_mask: &'a mut [u8],
    /// Frame-level damage accumulator owned by the host. Consumers paint into Canvas instances backed by this accumulator (`Canvas::new(target, w, h, ctx.damage)`); every rasterizer reports its painted bbox into it. The host reads it after `app.render` to know exactly what changed this frame — drives the optional damage-rect outline overlay today and (eventually) damage-clipped composite + present.
    pub damage: &'a mut crate::canvas::Damage,
    /// The damage clip the host computed for THIS frame, derived from `app.damage_rect(...)` before render. Consumers should thread this through every flatten / blit / glow call as the `clip` parameter so they only touch pixels inside the dirty region. Defaults to the full viewport (legacy apps that don't override `FluorApp::damage_rect` get the current full-redraw behavior).
    pub damage_clip: crate::canvas::PixelRect,
    /// The host's winit window handle. Use for `set_cursor`, `set_minimized`, `set_maximized`, `request_redraw`, `drag_window`, `set_title`, `outer_position`.
    pub window: &'a Window,
    /// Latest tracked modifier state (shift / ctrl / alt / super).
    pub modifiers: ModifiersState,
    /// Last known cursor position in viewport pixels (host-tracked across all events).
    pub cursor_x: Coord,
    pub cursor_y: Coord,
}

/// What the consumer wants the host to do after [`FluorApp::on_event`]. Pass-through behaviour lets the consumer ignore events they don't care about; the explicit variants override the host's default for that event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventResponse {
    /// Consumer handled the event; host does nothing more.
    Handled,
    /// Consumer didn't handle it; host applies its default (mouse-down on no widget = drag the window; close button = exit).
    Pass,
    /// Consumer wants the host to begin a window-move drag (typical: mouse-down on chrome strip with no button hit).
    StartWindowDrag,
    /// Consumer wants the host to begin a window-resize drag in the given edge direction.
    StartResize(ResizeEdge),
    /// Consumer requests window close. Host calls `std::process::exit(0)` (Killswitch-compliant).
    Close,
}

/// What a consumer implements to drive the desktop host.
pub trait FluorApp {
    /// Initial window title. Default is empty; override or call `ctx.window.set_title(...)` from `init` if you want it set later.
    fn title(&self) -> &str {
        ""
    }

    /// One-shot setup after the window exists. Allocate Groups, widgets, initial geometry. The viewport in `ctx` is the actual physical size the host opened.
    fn init(&mut self, ctx: &mut Context);

    /// The window resized. Resize internal Groups / widget bboxes to match.
    fn on_resize(&mut self, width: u32, height: u32, ctx: &mut Context);

    /// Window event from winit. Consumer returns an [`EventResponse`] telling the host what to do next.
    fn on_event(&mut self, event: &WindowEvent, ctx: &mut Context) -> EventResponse;

    /// Damage region this app will repaint this frame. Returns `None` if no widget state changed since the last frame — host can persist scratch as-is and skip render entirely. Returns `Some(rect)` to declare the union of all dirty widget bboxes (each widget's `prev ∪ current` from `widget.damage_rect(...)`); host clears scratch in that rect and threads it through `ctx.damage_clip` so the consumer's render call clips every flatten / blit to it.
    ///
    /// Default impl returns `Some(full viewport)` — safe fallback that preserves today's full-redraw behavior. Apps opt into differential rendering by overriding this to union their widget damage rects.
    ///
    /// Takes `Viewport` directly (not `Context`) so the host can call it without holding the text-renderer borrow that `Context` carries.
    fn damage_rect(&self, viewport: Viewport) -> Option<crate::canvas::PixelRect> {
        let w = viewport.width_px as usize;
        let h = viewport.height_px as usize;
        Some(crate::canvas::PixelRect::new(0, 0, w, h))
    }

    /// Per-frame paint into the host's CPU present buffer. Flatten owned Groups onto `target`. The damage clip computed pre-render is in `ctx.damage_clip`; thread it through every flatten / blit / glow call to skip pixels outside the dirty region.
    fn render(&mut self, target: &mut [u32], ctx: &mut Context);

    /// Post-finalize, post-shadow screen overlay pass. Runs AFTER the host has composited `scratch` into the persistent screen buffer and painted the drop shadow. Use for non-destructive overlays that should mutate just a few pixels each frame WITHOUT going through scratch / finalize / shadow re-rasterization — the textbox blinkey is the canonical case (wrap-add the wave on, wrap-subtract off).
    ///
    /// `screen` is the host's persistent screen-sized buffer in **OS visible-RGB** format (already past the α + darkness → visible XOR boundary). `(window_origin_x, window_origin_y)` is where the window content starts inside `screen`, so a consumer translates viewport-space widget coords by adding these to land in the right pixels.
    ///
    /// Default impl: no-op.
    fn paint_screen_overlay(
        &mut self,
        screen: &mut [u32],
        scr_w: usize,
        scr_h: usize,
        window_origin_x: i32,
        window_origin_y: i32,
    ) {
        let _ = (screen, scr_w, scr_h, window_origin_x, window_origin_y);
    }

    /// Cursor icon at `(x, y)` in viewport pixel coords. Called whenever the cursor moves.
    fn cursor_for(&self, x: Coord, y: Coord, ctx: &Context) -> CursorIcon;

    /// When to wake up next (animation timers, blinks). `None` = wait for input only. The host calls this once per `about_to_wait` cycle and feeds it into `ControlFlow::WaitUntil`.
    fn wake_at(&self) -> Option<Instant> {
        None
    }

    /// Called once per `about_to_wait` cycle (after the host's own platform polling). Drive time-based state here — blink timers, animation tweens, drag-scroll. Return `true` if state changed and a redraw is needed; the host will call `request_redraw` for you.
    fn tick(&mut self, ctx: &mut Context) -> bool {
        let _ = ctx;
        false
    }
}

/// Run the desktop host until the window closes.
pub fn run_app<A: FluorApp + 'static>(app: A) -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new()?;
    let mut shell = DesktopShell::new(app);
    event_loop.run_app(&mut shell)
}

/// Visible-window placement inside the fullscreen screen buffer. fluor now runs as a fullscreen transparent OS window owning the whole display — the "window" the consumer paints into is a sub-rect of that screen buffer at `(x, y)` with `(w, h)` pixels. `(x, y, w, h)` are screen-space pixel coordinates. `(0, 0)` is the top-left of the display. WindowRect is mutated by drag-to-move (changes `x, y`) and resize-drag (changes `w, h`); both are in-buffer operations that don't touch the OS window geometry.
#[derive(Clone, Copy, Debug)]
struct WindowRect {
    x: i32,
    y: i32,
    w: u32,
    h: u32,
}

/// Damage-clipped fill(0) — wipes only the `rect` sub-region of `scratch` (viewport-flat slice, row-major width `win_w`). Replaces a full-buffer `fill(0)` so pixels outside the damage rect persist between frames. Each row inside the rect uses the SIMD-friendly slice `fill(0)` so the per-row cost is the same as the full-buffer call, just over fewer rows.
fn clear_scratch_rect(scratch: &mut [u32], win_w: usize, rect: crate::canvas::PixelRect) {
    if rect.is_empty() {
        return;
    }
    let win_h = scratch.len() / win_w.max(1);
    let y0 = rect.y0.min(win_h);
    let y1 = rect.y1.min(win_h);
    let x0 = rect.x0.min(win_w);
    let x1 = rect.x1.min(win_w);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    for y in y0..y1 {
        let row_base = y * win_w;
        scratch[row_base + x0..row_base + x1].fill(0);
    }
}

/// The host's adapter — owns platform handles + the consumer's `App`, dispatches events through the trait. Not user-facing; constructed by [`run_app`].
///
/// **Compositor architecture.** The OS window is fullscreen borderless transparent — fluor owns the entire screen buffer. The consumer paints into a window-sized scratch buffer (sized to `viewport` = `window_rect.w × window_rect.h`); the host then blits that scratch into the screen buffer at the `window_rect` offset. Pixels outside the window stay α=0 so the OS compositor shows whatever's behind us. Click-through is via a per-resize input-region call (set later, see step 2 of the fullscreen-compositor pivot) so clicks outside `window_rect` route to whatever's underneath.
struct DesktopShell<A: FluorApp> {
    app: A,
    window: Option<Arc<Window>>,
    /// Consumer-visible viewport — sized to `window_rect.w × window_rect.h`, NOT the screen. Consumers paint and lay out as if their window is `viewport.width_px × viewport.height_px`; the host handles placing that paint inside the larger screen buffer.
    viewport: Viewport,
    /// Display size in pixels (= OS window size in fullscreen mode). The OS surface buffer matches this.
    screen_size: (u32, u32),
    /// Where the visible window lives inside the screen buffer. Driven by drag-to-move + resize-drag (later steps); initialized centered with a 3/4-of-monitor-short initial size.
    window_rect: WindowRect,
    /// Window-sized scratch buffer. The consumer renders into this (at viewport dimensions); the host runs `finalize_for_os` on it with the window-space clip mask, then blits row-by-row into the screen buffer at `window_rect.x, window_rect.y`. Resized on `window_rect` size change.
    scratch: Vec<u32>,

    // --- Renderer ---
    #[cfg(target_os = "macos")]
    renderer: Option<super::renderer_wgpu::Renderer>,
    #[cfg(not(target_os = "macos"))]
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,

    // --- Shared resources ---
    text: Option<TextRenderer>,
    /// Window-shape clip mask, one byte α per pixel. Sized to `viewport` (= window-space, NOT screen-space). The consumer carves shape into it (rounded corner cutouts etc.); `finalize_for_os` multiplies it into each pixel's α at the scratch-buffer boundary before the host blits to screen. Default `255` (fully visible) means a consumer that doesn't touch it gets a rectangular window.
    clip_mask: Vec<u8>,
    cursor_x: Coord,
    cursor_y: Coord,
    modifiers: ModifiersState,

    // --- Self-driven resize tracking. fluor owns the input side of the resize-drag loop on every platform: on edge-press we capture start geometry; on cursor-move we compute the new target geometry and push it to the OS via request_inner_size + set_outer_position. The OS confirms via Resized events which trigger the actual surface resize + paint — keeping buffer size == window size always, eliminating X11 PutImage mismatch smear. Replaces the WM-driven drag_resize_window path AND the macOS NSEvent polling hack with one unified flow.
    is_dragging_resize: bool,
    resize_edge: ResizeEdge,
    drag_start_size: (u32, u32),
    drag_start_window_pos: (i32, i32),
    drag_start_cursor_screen_pos: (i32, i32),

    // --- Drag-to-move tracking. In the fullscreen-compositor architecture the OS window is fullscreen and `window.drag_window()` doesn't move anything — we move our internal `window_rect` inside the screen buffer instead. On press we capture the cursor's screen position + window_rect origin; on cursor-move we update window_rect.x/y by the delta. The actual screen-buffer shift happens at vsync (RedrawRequested) via paint::shift_screen_wrap — skipping consumer render + finalize + shadow entirely during the drag. On drag release, a request_redraw kicks off a clean full re-render that overwrites the wrap artefacts.
    is_dragging_move: bool,
    drag_move_anchor_screen: (i32, i32),
    drag_move_rect_start: (i32, i32),
    /// Last window_rect (x, y, w, h) that was actually painted into the screen buffer. Set after every render_frame; consulted at drag-move vsync ticks to compute the (dx, dy) delta to feed into `shift_screen_wrap`. Without this we'd have no way to know "how much did the window move since the last frame" because the cursor anchor describes total drag distance, not per-frame increment.
    last_painted_rect: WindowRect,

    /// `false` until the first `WindowEvent::Resized` arrives confirming the OS surface size. Most WMs open a default-sized window (800×600 or similar) and then animate / configure it to fullscreen — Resized fires when the actual surface is ready. Until then, painting positions chrome against a stale `window_rect` (sized for the monitor we expected) inside a buffer that's smaller than expected, producing a brief "chrome in the top-left of a tiny window" flash as the WM grows the surface. Defer all rendering until this flips true.
    surface_ready: bool,

    /// Tracks `WindowEvent::Focused` so the drop shadow can dim when the window is inactive — focused windows cast a stronger shadow (`SHADOW_SEED_FOCUSED`), unfocused ones use a quarter-strength shadow (`SHADOW_SEED_UNFOCUSED`).
    is_focused: bool,

    /// Live render-pipeline counters. Updated every `render_frame` call (composite-time EMA +
    /// frame counter); rendered to a bottom-of-window debug strip when [`paint::DEBUG_SHOW_FPS`]
    /// is set via the `Ctrl/Cmd + Shift + D + F` chord.
    debug_stats: crate::paint::DebugStats,

    /// Frame-level damage accumulator. Reset at the top of each `render_frame`; passed to the consumer via [`Context::damage`]; read back after consumer render to drive damage-clipped composite and the [`paint::DEBUG_SHOW_DAMAGE`] outline overlay.
    pending_damage: crate::canvas::Damage,
    /// FPS strip active state from the previous frame. When it toggles `true → false`, this frame's damage_clip must include the strip bbox so the just-vanished strip pixels get cleared from scratch (and propagated into persistent_screen via finalize). Tracked instead of a generic `prev_damage_clip` union to avoid sticky viewport-sized damage on hover frames after any prior full repaint.
    last_strip_active: bool,
    /// Damage outline overlay state from the previous frame; same toggle-off-clearing logic as `last_strip_active`. When the outline was on last frame and is off this frame, union its bbox into damage_clip so the magenta border gets wiped.
    last_outline_active: bool,
    /// The bbox the damage outline was drawn around last frame (pending_damage.bbox at the time it painted). Used by the toggle-off case so we know which pixels to clear.
    last_outline_bbox: crate::canvas::PixelRect,
    /// Persistent screen-sized pixel buffer, owned by us. Survives across frames so post-finalize overlays (the textbox blinkey via [`FluorApp::paint_screen_overlay`]) can mutate just a few pixels with wrap-add / wrap-sub semantics instead of re-running finalize for the whole window. The platform's softbuffer / wgpu back buffer may rotate or arrive stale; we always memcpy `persistent_screen` over it just before `present()` so the platform buffer's prior state doesn't matter. Resized when `screen_size` changes.
    persistent_screen: Vec<u32>,
    /// Set on drag release: tells the next [`render_frame`] to wipe `persistent_screen` and force `damage_clip = viewport` for one frame. `apply_move_drag_shift` intentionally shifts the rotating platform back buffer (NOT persistent_screen) during drag so wrap artefacts don't accumulate frame-to-frame — but that means persistent_screen is stale post-drag and needs one full rebuild to match the new window position.
    pending_full_invalidate: bool,
}

impl<A: FluorApp> DesktopShell<A> {
    fn new(app: A) -> Self {
        Self {
            app,
            window: None,
            viewport: Viewport::new(1, 1),
            screen_size: (1, 1),
            window_rect: WindowRect { x: 0, y: 0, w: 1, h: 1 },
            scratch: Vec::new(),
            #[cfg(target_os = "macos")]
            renderer: None,
            #[cfg(not(target_os = "macos"))]
            surface: None,
            text: None,
            clip_mask: Vec::new(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            modifiers: ModifiersState::empty(),
            is_dragging_resize: false,
            resize_edge: ResizeEdge::None,
            drag_start_size: (0, 0),
            drag_start_window_pos: (0, 0),
            drag_start_cursor_screen_pos: (0, 0),
            is_dragging_move: false,
            drag_move_anchor_screen: (0, 0),
            drag_move_rect_start: (0, 0),
            last_painted_rect: WindowRect { x: 0, y: 0, w: 1, h: 1 },
            surface_ready: false,
            is_focused: true,
            debug_stats: crate::paint::DebugStats::default(),
            pending_damage: crate::canvas::Damage::new(),
            last_strip_active: false,
            last_outline_active: false,
            last_outline_bbox: crate::canvas::PixelRect::empty(),
            persistent_screen: Vec::new(),
            pending_full_invalidate: false,
        }
    }

    fn render_frame(&mut self) {
        if !self.surface_ready {
            return;
        }
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };
        let win_w = self.viewport.width_px as usize;
        let win_h = self.viewport.height_px as usize;
        let win_px = win_w * win_h;
        // Keep scratch + clip_mask in sync with the consumer-visible viewport (= window_rect dims). Resize-drag (later step) changes these; we re-allocate when the size shifts.
        if self.scratch.len() != win_px {
            self.scratch = vec![0u32; win_px];
        }
        if self.clip_mask.len() != win_px {
            self.clip_mask = vec![255u8; win_px];
        }

        // Reset the frame's damage accumulator before the consumer paints. Every Canvas the consumer constructs against `ctx.damage` will union into this; after `app.render` returns we have the bounding rect of everything touched this frame.
        self.pending_damage.clear();

        // Pre-render damage query. `None` from app.damage_rect = no widget needs scratch work — the screen overlay path (hover tint, blinkey) handles the visible change entirely. Empty `damage_clip` then skips the entire scratch / finalize / shadow chain below; only overlay + memcpy + present run. FPS strip bbox is unioned in when active so the strip's stats refresh each frame. Drag-release forces a full invalidate: persistent_screen is wiped and damage_clip is bumped to the whole viewport so finalize fully repaints at the new position.
        let viewport_rect = crate::canvas::PixelRect::new(0, 0, win_w, win_h);
        let mut damage_clip = if self.pending_full_invalidate {
            self.pending_full_invalidate = false;
            self.persistent_screen.fill(0);
            viewport_rect
        } else {
            self.app.damage_rect(self.viewport).unwrap_or(crate::canvas::PixelRect::empty())
        };
        #[cfg(feature = "text")]
        let strip_active = crate::paint::DEBUG_SHOW_FPS.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(feature = "text"))]
        let strip_active = false;
        let strip_y_start = (win_h * 11) / 12;
        let strip_rect = crate::canvas::PixelRect::new(0, strip_y_start, win_w, win_h);
        // Strip needs the bbox union when it's currently active OR was active last frame (so toggle-off pixels get cleared from scratch + finalized away). Avoids saving a sticky viewport-prev that would force every subsequent hover frame into a full repaint.
        if strip_active || self.last_strip_active {
            damage_clip = damage_clip.union(strip_rect);
        }
        self.last_strip_active = strip_active;

        // Damage outline overlay: same toggle-off-cleanup pattern. The outline is drawn around the actual pending_damage post-render — but for clearing on toggle-off we use the bbox we recorded last frame.
        let outline_active = crate::paint::DEBUG_SHOW_DAMAGE.load(std::sync::atomic::Ordering::Relaxed);
        if !outline_active && self.last_outline_active {
            damage_clip = damage_clip.union(self.last_outline_bbox);
        }

        clear_scratch_rect(&mut self.scratch, win_w, damage_clip);

        // Paint the FPS strip first (topmost-first inside `damage_clip`). Uses the previous frame's stats — this frame's time isn't known until after render+finalize+shadow below.
        #[cfg(feature = "text")]
        if strip_active {
            if let Some(text) = self.text.as_mut() {
                let mut strip_damage = crate::canvas::Damage::new();
                let mut canvas = crate::canvas::Canvas::new(
                    &mut self.scratch,
                    win_w,
                    win_h,
                    &mut strip_damage,
                );
                crate::paint::draw_debug_strip(&mut canvas, text, &self.debug_stats);
            }
        }

        let Some(text) = self.text.as_mut() else {
            return;
        };

        let mut ctx = Context {
            viewport: self.viewport,
            text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &window,
            modifiers: self.modifiers,
            cursor_x: self.cursor_x - self.window_rect.x as Coord,
            cursor_y: self.cursor_y - self.window_rect.y as Coord,
            damage_clip,
        };

        // Per-stage stopwatches. Each Instant brackets one pipeline stage; the strip displays each as FPS so toggling SIMD/Rayon shows which stage actually moves. `buffer.present()` is excluded everywhere because it blocks for vsync, which would pin every reading to the display refresh rate.
        let mut app_dt = 0.0f32;
        let mut fill_dt = 0.0f32;
        let mut finalize_dt = 0.0f32;
        let mut shadow_dt = 0.0f32;

        let app_start = Instant::now();
        self.app.render(&mut self.scratch, &mut ctx);
        drop(ctx);
        app_dt = app_start.elapsed().as_secs_f32();

        // Outline state for next frame's toggle-off check. Record only when active so a toggle-off can union the rect.
        self.last_outline_active = outline_active;
        if outline_active {
            self.last_outline_bbox = self.pending_damage.bbox();
        }

        // Optional damage outline overlay (Ctrl+Shift+D+W). Paints a 2-px magenta hairline around the bounding rect of everything that reported damage this frame. Painted into scratch BEFORE finalize so it flows through the boundary like any other content. Outline drawing reports its own damage back into `pending_damage`, which is harmless because we've already snapshot the bbox.
        if crate::paint::DEBUG_SHOW_DAMAGE.load(std::sync::atomic::Ordering::Relaxed) {
            let bbox = self.pending_damage.bbox();
            if !bbox.is_empty() {
                let mut overlay_damage = crate::canvas::Damage::new();
                let mut canvas = crate::canvas::Canvas::new(
                    &mut self.scratch,
                    win_w,
                    win_h,
                    &mut overlay_damage,
                );
                crate::paint::draw_damage_outline(&mut canvas, bbox);
            }
        }

        let scr_w = self.screen_size.0 as usize;
        let scr_h = self.screen_size.1 as usize;
        let rect_x = self.window_rect.x;
        let rect_y = self.window_rect.y;

        // Persistent screen lives across frames so the post-finalize overlay (blinkey) can mutate just a few pixels each frame without re-running finalize. Resize on screen-size change; new pixels start at 0 which is fine — finalize on the next render will populate them.
        let scr_px = scr_w * scr_h;
        let t = Instant::now();
        if self.persistent_screen.len() != scr_px {
            self.persistent_screen.resize(scr_px, 0);
        }
        fill_dt = t.elapsed().as_secs_f32();

        // Finalize + shadow ONLY when there's real scratch damage to push through. On overlay-only frames (hover, focus tint change, blinkey tick) `damage_clip` is empty and the persistent_screen + shadow band from prior frames already hold the correct content — the overlay does the entire visible change. The shadow specifically only re-paints on a full-viewport repaint (chrome / focus / drag-release / resize), since damage that's interior to the window never affects edge-band shadow pixels.
        let t = Instant::now();
        if !damage_clip.is_empty() {
            crate::paint::finalize_into_screen(
                &self.scratch,
                &self.clip_mask,
                win_w,
                win_h,
                &mut self.persistent_screen,
                scr_w,
                rect_x,
                rect_y,
            );
        }
        finalize_dt = t.elapsed().as_secs_f32();

        let t = Instant::now();
        if damage_clip == viewport_rect {
            // Drop shadow: 45-degree diagonal rays cast from each chrome right/bottom edge pixel. factor_256 derived from `effective_span` so ray length is RU-invariant: target ≈ effective_span / 16; `factor_256 ≈ 256 − 1240/r`. Seed is the starting shadow α: 0x80 when focused, 0x40 (quarter strength) when unfocused.
            let span = self.viewport.effective_span();
            let target_radius = (span / 16.0).max(8.0);
            let drop = (1240.0 / target_radius) as u32;
            let factor_256 = (256u32.saturating_sub(drop)).clamp(96, 254);
            let shadow_seed: u32 = if self.is_focused { 0x80 } else { 0x40 };
            let rect_for_shadow = (
                self.window_rect.x,
                self.window_rect.y,
                self.window_rect.w as i32,
                self.window_rect.h as i32,
            );
            crate::paint::paint_shadow(
                &mut self.persistent_screen,
                scr_w,
                factor_256,
                shadow_seed,
                rect_for_shadow,
            );
        }
        shadow_dt = t.elapsed().as_secs_f32();

        // Post-finalize, post-shadow overlay pass — runs against persistent_screen so it can wrap-add/wrap-sub a few pixels (textbox blinkey) without going through the whole scratch / finalize / shadow chain.
        self.app.paint_screen_overlay(
            &mut self.persistent_screen,
            scr_w,
            scr_h,
            self.window_rect.x,
            self.window_rect.y,
        );

        // Copy persistent_screen → platform back buffer (whichever softbuffer/wgpu hands us this frame; it may be stale or rotated, but we always overwrite the whole thing from our owned persistent_screen).
        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            let mut buffer = renderer.lock_buffer();
            buffer.copy_from_slice(&self.persistent_screen);
            let _ = buffer.present();
        }
        #[cfg(not(target_os = "macos"))]
        {
            let Some(surface) = self.surface.as_mut() else {
                return;
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            buffer.copy_from_slice(&self.persistent_screen);
            buffer.present().expect("softbuffer buffer.present");
        }
        // Record what we just painted so the next drag-tick can compute its delta.
        self.last_painted_rect = self.window_rect;

        // Differential stats: F (frame) bumps every present; R (rasterize) only when a primitive actually did geometric paint work this frame (via the RASTERIZE_OPS atomic). On hover-only updates the atomic stays at 0 and R sticks. `damage_pct` reflects how much of the viewport this frame actually touched — drops to a small fraction on bbox-only updates.
        let ras_ops = crate::paint::RASTERIZE_OPS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let viewport_area = (win_w * win_h) as f32;
        let damage_area = (damage_clip.width() * damage_clip.height()) as f32;
        let damage_pct = if viewport_area > 0.0 { damage_area / viewport_area } else { 0.0 };
        if ras_ops > 0 {
            self.debug_stats.record_rasterize(app_dt, fill_dt, finalize_dt, shadow_dt, damage_pct);
        }
        self.debug_stats.record_present(damage_pct);
    }

    /// Drag-tick fast path: shift the screen buffer in place by the delta since the last paint, push the input region update, and present. Skips consumer render, scratch fill, finalize, and shadow rasterization entirely — the existing chrome pixels just slide through the screen buffer, with anything that falls off any edge wrapping to the opposite side. On drag release, a normal `render_frame` overwrites the wrap artefacts in one clean frame.
    fn apply_move_drag_shift(&mut self) {
        let dx = self.window_rect.x - self.last_painted_rect.x;
        let dy = self.window_rect.y - self.last_painted_rect.y;
        if dx == 0 && dy == 0 {
            return;
        }
        let scr_w = self.screen_size.0 as usize;
        let scr_h = self.screen_size.1 as usize;
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };
        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            let mut buffer = renderer.lock_buffer();
            crate::paint::shift_screen_wrap(&mut buffer, scr_w, scr_h, dx, dy);
            let _ = buffer.present();
        }
        #[cfg(not(target_os = "macos"))]
        {
            let Some(surface) = self.surface.as_mut() else {
                return;
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            crate::paint::shift_screen_wrap(&mut buffer, scr_w, scr_h, dx, dy);
            buffer.present().expect("softbuffer buffer.present");
        }
        #[cfg(target_os = "linux")]
        x11_atomic::set_input_region(
            &window,
            self.window_rect.x,
            self.window_rect.y,
            self.window_rect.w,
            self.window_rect.h,
        );
        self.last_painted_rect = self.window_rect;
    }

    /// Apply an [`EventResponse`] returned from `app.on_event`. Returns `true` if the response was `Close` (caller should terminate).
    fn apply_response(&mut self, response: EventResponse) -> bool {
        let Some(window) = self.window.as_ref().cloned() else {
            return false;
        };
        match response {
            EventResponse::Handled | EventResponse::Pass => false,
            EventResponse::StartWindowDrag => {
                // Fullscreen-compositor model: OS window.drag_window() would do nothing (OS window is fullscreen). Drag is internal — capture the anchor here and move window_rect on CursorMoved.
                self.is_dragging_move = true;
                self.drag_move_anchor_screen = (self.cursor_x as i32, self.cursor_y as i32);
                self.drag_move_rect_start = (self.window_rect.x, self.window_rect.y);
                false
            }
            EventResponse::StartResize(edge) => {
                self.start_resize(edge, &window);
                false
            }
            EventResponse::Close => {
                std::process::exit(0);
            }
        }
    }

    /// Begin a self-driven resize drag. In the fullscreen-compositor model we resize `window_rect` inside our own screen buffer instead of asking the OS to resize the OS window (which is fullscreen). Captures the start geometry (window_rect size + position) and the screen-space cursor anchor; subsequent cursor moves compute the new (w, h, x, y) by delta from these starting values.
    fn start_resize(&mut self, edge: ResizeEdge, _window: &Window) {
        self.is_dragging_resize = true;
        self.resize_edge = edge;
        self.drag_start_size = (self.window_rect.w, self.window_rect.h);
        self.drag_start_window_pos = (self.window_rect.x, self.window_rect.y);
        // cursor_x/y are screen-space (raw from winit, OS window = screen in fullscreen) so no translation needed for the anchor.
        self.drag_start_cursor_screen_pos = (self.cursor_x as i32, self.cursor_y as i32);
    }

    /// Apply one tick of the self-driven resize drag — in-buffer. Called from `RedrawRequested` when `is_dragging_resize` (throttled to vsync). Updates `window_rect` directly (no OS round-trip — the OS window is fullscreen and request_inner_size / set_outer_position are no-ops). When the size changed, resizes `scratch` + `clip_mask` to the new dimensions and calls the consumer's `on_resize` so they can reflow. Always pushes a new XShape input region so click-through follows the visible window. The subsequent `render_frame` paints at the new geometry into the screen buffer.
    fn apply_resize_drag(&mut self) {
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };

        // Screen-relative cursor delta from the drag-start anchor. cursor_x/y is already screen-space (raw winit / OS = screen in fullscreen) so no per-frame translation needed.
        let dx = (self.cursor_x as i32 - self.drag_start_cursor_screen_pos.0) as Coord;
        let dy = (self.cursor_y as i32 - self.drag_start_cursor_screen_pos.1) as Coord;

        // Min size keeps the squircle math from degenerating. 128 px matches the pre-pivot limit.
        let min_size: Coord = 128.0;

        let (new_w, new_h, new_x, new_y) = match self.resize_edge {
            ResizeEdge::Right => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                (w, self.drag_start_size.1, self.drag_start_window_pos.0, self.drag_start_window_pos.1)
            }
            ResizeEdge::Left => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let dw = self.drag_start_size.0 as i32 - w as i32;
                (w, self.drag_start_size.1, self.drag_start_window_pos.0 + dw, self.drag_start_window_pos.1)
            }
            ResizeEdge::Bottom => {
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                (self.drag_start_size.0, h, self.drag_start_window_pos.0, self.drag_start_window_pos.1)
            }
            ResizeEdge::Top => {
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let dh = self.drag_start_size.1 as i32 - h as i32;
                (self.drag_start_size.0, h, self.drag_start_window_pos.0, self.drag_start_window_pos.1 + dh)
            }
            ResizeEdge::TopRight => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let dh = self.drag_start_size.1 as i32 - h as i32;
                (w, h, self.drag_start_window_pos.0, self.drag_start_window_pos.1 + dh)
            }
            ResizeEdge::TopLeft => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let dw = self.drag_start_size.0 as i32 - w as i32;
                let dh = self.drag_start_size.1 as i32 - h as i32;
                (w, h, self.drag_start_window_pos.0 + dw, self.drag_start_window_pos.1 + dh)
            }
            ResizeEdge::BottomRight => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                (w, h, self.drag_start_window_pos.0, self.drag_start_window_pos.1)
            }
            ResizeEdge::BottomLeft => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                let dw = self.drag_start_size.0 as i32 - w as i32;
                (w, h, self.drag_start_window_pos.0 + dw, self.drag_start_window_pos.1)
            }
            ResizeEdge::None => return,
        };

        let size_changed = new_w != self.window_rect.w || new_h != self.window_rect.h;
        let pos_changed = new_x != self.window_rect.x || new_y != self.window_rect.y;
        if !size_changed && !pos_changed {
            return;
        }

        self.window_rect = WindowRect {
            x: new_x,
            y: new_y,
            w: new_w,
            h: new_h,
        };

        if size_changed {
            // Carry the user's zoom (ru) across the resize so Ctrl+/Ctrl-/Ctrl+scroll state survives.
            self.viewport = Viewport::new(new_w, new_h).with_ru(self.viewport.ru);
            let win_px = (new_w as usize) * (new_h as usize);
            self.scratch = vec![0u32; win_px];
            self.clip_mask = vec![255u8; win_px];

            // Let the consumer reflow — they may relayout panes, recompute glyph metrics, etc.
            if let Some(text) = self.text.as_mut() {
                let mut ctx = Context {
                    viewport: self.viewport,
                    text,
                    clip_mask: &mut self.clip_mask,
                    damage: &mut self.pending_damage,
                    window: &window,
                    modifiers: self.modifiers,
                    cursor_x: self.cursor_x - self.window_rect.x as Coord,
                    cursor_y: self.cursor_y - self.window_rect.y as Coord,
                    damage_clip: crate::canvas::PixelRect::new(0, 0, self.viewport.width_px as usize, self.viewport.height_px as usize),
                };
                self.app.on_resize(new_w, new_h, &mut ctx);
            }
        }

        // Update click-through region so the OS routes clicks based on the new rect.
        #[cfg(target_os = "linux")]
        x11_atomic::set_input_region(&window, new_x, new_y, new_w, new_h);
    }
}

impl<A: FluorApp + 'static> ApplicationHandler for DesktopShell<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Probe the monitor BEFORE creating the window so we can request an OS surface of exactly the right size + position. We deliberately avoid `with_fullscreen` — it triggers the WM's animated transition (default-window-size → grow → fullscreen) which makes the chrome appear to scale up from the top-left. Instead, ask for a plain borderless transparent window covering the monitor: the WM creates it at the requested geometry directly, no animation.
        let (mon_w, mon_h) = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
            .map(|m| (m.size().width.max(1), m.size().height.max(1)))
            .unwrap_or((1920, 1080));
        self.screen_size = (mon_w, mon_h);

        let attrs = WindowAttributes::default()
            .with_title(self.app.title())
            .with_inner_size(winit::dpi::PhysicalSize::new(mon_w, mon_h))
            .with_position(winit::dpi::PhysicalPosition::new(0i32, 0i32))
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(false);
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));

        #[cfg(target_os = "macos")]
        {
            use winit::platform::macos::WindowExtMacOS;
            window.set_has_shadow(false);
        }

        // Initial visible-window size: half the screen in each axis, centred. Matches the desktop convention of "open at a reasonable fraction of the display, centred." User can drag/resize from there.
        let initial_w = (mon_w / 2).max(1);
        let initial_h = (mon_h / 2).max(1);
        let win_x = ((mon_w as i32) - (initial_w as i32)) / 2;
        let win_y = ((mon_h as i32) - (initial_h as i32)) / 2;
        self.window_rect = WindowRect {
            x: win_x,
            y: win_y,
            w: initial_w,
            h: initial_h,
        };
        self.viewport = Viewport::new(initial_w, initial_h);

        #[cfg(target_os = "macos")]
        {
            self.renderer = Some(super::renderer_wgpu::Renderer::new(
                &window,
                self.screen_size.0,
                self.screen_size.1,
            ));
        }
        #[cfg(not(target_os = "macos"))]
        {
            use std::num::NonZeroU32;
            let context =
                softbuffer::Context::new(window.clone()).expect("softbuffer Context::new");
            let mut surface = softbuffer::Surface::new(&context, window.clone())
                .expect("softbuffer Surface::new");
            surface
                .resize(
                    NonZeroU32::new(self.screen_size.0).expect("nonzero screen width"),
                    NonZeroU32::new(self.screen_size.1).expect("nonzero screen height"),
                )
                .expect("softbuffer Surface::resize");
            self.surface = Some(surface);
        }

        if self.text.is_none() {
            self.text = Some(TextRenderer::new());
        }

        // Scratch + clip_mask are sized to the visible window (= viewport), NOT the screen. The host blits scratch into the screen buffer at window_rect offset; pixels outside the window stay at the screen buffer's α=0 init.
        let win_px = (self.window_rect.w as usize) * (self.window_rect.h as usize);
        self.scratch = vec![0u32; win_px];
        self.clip_mask = vec![255u8; win_px];

        // Hand control to the consumer's init.
        {
            let text = self.text.as_mut().expect("text renderer initialized");
            let mut ctx = Context {
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
                damage_clip: crate::canvas::PixelRect::new(0, 0, self.viewport.width_px as usize, self.viewport.height_px as usize),
            };
            self.app.init(&mut ctx);
        }

        self.window = Some(window.clone());
        // Surface is created at the requested monitor size — we can paint immediately. The Resized handler still flips this flag if it sees a different first size, but with the non-fullscreen approach we expect the surface to come up at the right size on the first frame.
        self.surface_ready = true;

        // Click-through: tell X11 our hittable area is just `window_rect`. Clicks outside the rect pass through to whatever app is beneath us. Drag-to-move + resize-drag steps will re-call this on every rect change. No-op on non-X11 platforms; macOS/Windows passthrough handling lands in their own backend modules later.
        #[cfg(target_os = "linux")]
        x11_atomic::set_input_region(
            &window,
            self.window_rect.x,
            self.window_rect.y,
            self.window_rect.w,
            self.window_rect.h,
        );

        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let needs_redraw = if let (Some(window), Some(text)) =
            (self.window.as_ref().cloned(), self.text.as_mut())
        {
            let mut ctx = Context {
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
                damage_clip: crate::canvas::PixelRect::new(0, 0, self.viewport.width_px as usize, self.viewport.height_px as usize),
            };
            self.app.tick(&mut ctx)
        } else {
            false
        };
        if needs_redraw {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }

        if let Some(when) = self.app.wake_at() {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(when));
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => {
                std::process::exit(0);
            }
            WindowEvent::Resized(size) => {
                // In the fullscreen-compositor architecture, the OS surface is the whole screen — `size` is the SCREEN size, not the consumer-visible viewport. WMs commonly fire Resized multiple times during fullscreen activation (default-window-size → animating → final fullscreen); each tick we resize the surface to match, re-centre the visible window inside the new bounds, and re-issue the input region. Suppresses the "chrome appears in the top-left of a growing window" artefact during WM fullscreen animations.
                if size.width == 0 || size.height == 0 {
                    return;
                }
                if size.width == self.screen_size.0
                    && size.height == self.screen_size.1
                    && self.surface_ready
                {
                    return;
                }

                self.screen_size = (size.width, size.height);

                #[cfg(target_os = "macos")]
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
                #[cfg(not(target_os = "macos"))]
                if let Some(surface) = self.surface.as_mut() {
                    use std::num::NonZeroU32;
                    if let (Some(w), Some(h)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        surface.resize(w, h).expect("softbuffer Surface::resize");
                    }
                }

                // Re-centre + clamp window_rect to the current screen. Keeps the visible window at half-screen size, centred, on every screen size change (initial fullscreen, monitor switch, etc.). Skip during an active drag — the user is steering the rect themselves.
                if !self.is_dragging_resize && !self.is_dragging_move {
                    let new_w = (size.width / 2).max(1).min(size.width);
                    let new_h = (size.height / 2).max(1).min(size.height);
                    let new_x = ((size.width as i32) - (new_w as i32)) / 2;
                    let new_y = ((size.height as i32) - (new_h as i32)) / 2;
                    let rect_changed = new_w != self.window_rect.w
                        || new_h != self.window_rect.h
                        || new_x != self.window_rect.x
                        || new_y != self.window_rect.y;
                    self.window_rect = WindowRect {
                        x: new_x,
                        y: new_y,
                        w: new_w,
                        h: new_h,
                    };
                    if rect_changed {
                        self.viewport = Viewport::new(new_w, new_h).with_ru(self.viewport.ru);
                        let win_px = (new_w as usize) * (new_h as usize);
                        self.scratch = vec![0u32; win_px];
                        self.clip_mask = vec![255u8; win_px];
                        if let (Some(window), Some(text)) =
                            (self.window.as_ref().cloned(), self.text.as_mut())
                        {
                            let mut ctx = Context {
                                viewport: self.viewport,
                                text,
                                clip_mask: &mut self.clip_mask,
                                damage: &mut self.pending_damage,
                                window: &window,
                                modifiers: self.modifiers,
                                cursor_x: self.cursor_x - new_x as Coord,
                                cursor_y: self.cursor_y - new_y as Coord,
                                damage_clip: crate::canvas::PixelRect::new(0, 0, self.viewport.width_px as usize, self.viewport.height_px as usize),
                            };
                            self.app.on_resize(new_w, new_h, &mut ctx);
                        }
                        #[cfg(target_os = "linux")]
                        if let Some(window) = self.window.as_ref() {
                            x11_atomic::set_input_region(window, new_x, new_y, new_w, new_h);
                        }
                    }
                }

                // First Resized confirms the OS surface is actually allocated — safe to start painting.
                self.surface_ready = true;
                self.render_frame();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x as Coord;
                self.cursor_y = position.y as Coord;

                // During a self-driven resize drag, CursorMoved fires at hundreds of Hz (raw input rate) AND we synthesize more via set_outer_position (window-relative cursor pos changes when the window moves). Doing a full resize+paint+OS-update per event floods X11 (`XIO: fatal IO error 11`) and creates a multi-second backlog of stale requests that play back after release. Coalesce: just stash the new cursor pos and request a redraw — winit caps RedrawRequested to vsync (~60-144 Hz), and the actual drag tick runs there. Skips consumer event dispatch too (consumer doesn't need to see resize-drag cursor moves).
                if self.is_dragging_resize {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                }

                // In-buffer drag-to-move: update window_rect.x/y by the cursor delta from the drag anchor. The actual screen-buffer shift + input-region update + present happens at vsync in `apply_move_drag_shift` (called from RedrawRequested), naturally coalescing the 200+ Hz raw input rate down to the display refresh rate. Skip consumer dispatch — they don't need cursor moves during the drag.
                if self.is_dragging_move {
                    let dx = (self.cursor_x as i32) - self.drag_move_anchor_screen.0;
                    let dy = (self.cursor_y as i32) - self.drag_move_anchor_screen.1;
                    self.window_rect.x = self.drag_move_rect_start.0 + dx;
                    self.window_rect.y = self.drag_move_rect_start.1 + dy;
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                }

                if let (Some(window), Some(text)) =
                    (self.window.as_ref().cloned(), self.text.as_mut())
                {
                    let mut ctx = Context {
                        viewport: self.viewport,
                        text,
                        clip_mask: &mut self.clip_mask,
                        damage: &mut self.pending_damage,
                        window: &window,
                        modifiers: self.modifiers,
                        cursor_x: self.cursor_x - self.window_rect.x as Coord,
                        cursor_y: self.cursor_y - self.window_rect.y as Coord,
                        damage_clip: crate::canvas::PixelRect::new(0, 0, self.viewport.width_px as usize, self.viewport.height_px as usize),
                    };
                    let response = self.app.on_event(&event, &mut ctx);
                    // Cursor coords must be window-relative — same translation as Context's cursor_x/y — so the consumer's hit_at sees the chrome at origin (0,0). Raw screen-space coords would miss every button when the window_rect isn't at (0,0).
                    let icon = self.app.cursor_for(ctx.cursor_x, ctx.cursor_y, &ctx);
                    drop(ctx);
                    window.set_cursor(icon);
                    self.apply_response(response);
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
                self.dispatch_event(event);
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } if (self.modifiers.control_key() || self.modifiers.super_key())
                && key_event.state == ElementState::Pressed =>
            {
                // Ctrl/Cmd + =/+/-/0 → zoom. Match `logical_key.to_text()` (the produced character) rather than `physical_key` so non-US layouts (Colemak/Dvorak/etc.) work — the user pressing the key labelled `=` should zoom in regardless of which physical-position that key occupies. `+` covers Shift+= and the numpad `+`; `=` covers the plain key on US. `-` covers minus and the numpad `-`. `0` covers digit and numpad 0.
                if let Some(text) = key_event.logical_key.to_text() {
                    match text {
                        "=" | "+" => {
                            self.apply_zoom_change(Some(1.0));
                            return;
                        }
                        "-" => {
                            self.apply_zoom_change(Some(-1.0));
                            return;
                        }
                        "0" => {
                            self.apply_zoom_change(None);
                            return;
                        }
                        _ => {}
                    }
                }
                self.dispatch_event(event);
            }
            WindowEvent::MouseWheel { delta, .. }
                if self.modifiers.control_key() || self.modifiers.super_key() =>
            {
                // Ctrl/Cmd + scroll → zoom. 1 step per scroll notch (LineDelta). Trackpad PixelDelta accumulates many small events; ~31 px per zoom-in step, ~32 px per zoom-out step matches typical trackpad density. The 31/32 split mirrors `adjust_zoom`'s 32/31 vs 32/33 asymmetry — in/out aren't exact inverses, so repeated in→out scrolling drifts by a small fraction per pair, giving subpixel positioning instead of clamping the user to a discrete lattice.
                let steps: f32 = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => {
                        let py = p.y as f32;
                        let divisor = if py >= 0.0 { 31.0 } else { 32.0 };
                        py / divisor
                    }
                };
                if steps != 0.0 {
                    self.apply_zoom_change(Some(steps));
                    return;
                }
                self.dispatch_event(event);
            }
            WindowEvent::Focused(focused) => {
                let focused = *focused;
                self.is_focused = focused;
                // Cancel any in-progress resize drag if we lose focus mid-drag (the user alt-tabbed or the WM stole focus). Keeps state consistent.
                if !focused && self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                }
                self.dispatch_event(event);
                // Repaint so the drop shadow dims/brightens immediately.
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                // End of resize drag — release ownership of the loop. The buffer is already in the final state from the last drag tick; no extra repaint needed.
                if self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                }
                // End of in-buffer drag-to-move. Drop the flag and request one full redraw — the wrap-shift fast-path leaves wrap artefacts at whichever edges the window slid across AND persistent_screen has stale content from before the drag. Set `pending_full_invalidate` so the next render_frame wipes persistent_screen and forces a full repaint at the new window position.
                if self.is_dragging_move {
                    self.is_dragging_move = false;
                    self.pending_full_invalidate = true;
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
                self.dispatch_event(event);
            }
            WindowEvent::RedrawRequested => {
                // Drag-to-move fast path: shift the existing screen pixels by the per-tick delta instead of re-rendering anything. Skips consumer.render(), scratch fill, finalize, and shadow rasterization.
                if self.is_dragging_move {
                    self.apply_move_drag_shift();
                    return;
                }
                // Resize drag: apply the new geometry in-buffer, then paint at the new size.
                if self.is_dragging_resize {
                    self.apply_resize_drag();
                }
                self.render_frame();
            }
            _ => {
                self.dispatch_event(event);
            }
        }
    }
}

impl<A: FluorApp + 'static> DesktopShell<A> {
    /// Apply a zoom change to `self.viewport.ru` and propagate to the consumer. `steps = Some(s)` adjusts by `s` photon-asymmetric log steps (positive in, negative out); `steps = None` resets to 1.0 (Ctrl+0 binding). Calls `app.on_resize` with unchanged pixel dimensions so the consumer's existing resize path picks up the new `ctx.viewport.ru`, marks chrome / widget layers dirty (via their internal Group resize), and re-rasterizes at the new effective span. No separate `on_zoom` callback needed — the consumer's on_resize is the single "viewport changed" entry point.
    fn apply_zoom_change(&mut self, steps: Option<f32>) {
        match steps {
            Some(s) => self.viewport.adjust_zoom(s),
            None => self.viewport.reset_zoom(),
        }
        if let (Some(window), Some(text)) = (self.window.as_ref().cloned(), self.text.as_mut()) {
            let mut ctx = Context {
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
                damage_clip: crate::canvas::PixelRect::new(0, 0, self.viewport.width_px as usize, self.viewport.height_px as usize),
            };
            self.app
                .on_resize(self.viewport.width_px, self.viewport.height_px, &mut ctx);
            drop(ctx);
            window.request_redraw();
        }
    }

    /// Helper: dispatch a generic event to `app.on_event`, applying any returned [`EventResponse`].
    fn dispatch_event(&mut self, event: WindowEvent) {
        if let (Some(window), Some(text)) = (self.window.as_ref().cloned(), self.text.as_mut()) {
            let mut ctx = Context {
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
                damage_clip: crate::canvas::PixelRect::new(0, 0, self.viewport.width_px as usize, self.viewport.height_px as usize),
            };
            let response = self.app.on_event(&event, &mut ctx);
            drop(ctx);
            self.apply_response(response);
        }
    }
}

