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

    /// Per-frame paint into the host's CPU present buffer. Flatten owned Groups onto `target`.
    fn render(&mut self, target: &mut [u32], ctx: &mut Context);

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

    // --- Drag-to-move tracking. In the fullscreen-compositor architecture the OS window is fullscreen and `window.drag_window()` doesn't move anything — we move our internal `window_rect` inside the screen buffer instead. On press we capture the cursor's screen position + window_rect origin; on cursor-move we update window_rect.x/y by the delta and re-set the XShape input region so click-through follows the visible window.
    is_dragging_move: bool,
    drag_move_anchor_screen: (i32, i32),
    drag_move_rect_start: (i32, i32),

    /// `false` until the first `WindowEvent::Resized` arrives confirming the OS surface size. Most WMs open a default-sized window (800×600 or similar) and then animate / configure it to fullscreen — Resized fires when the actual surface is ready. Until then, painting positions chrome against a stale `window_rect` (sized for the monitor we expected) inside a buffer that's smaller than expected, producing a brief "chrome in the top-left of a tiny window" flash as the WM grows the surface. Defer all rendering until this flips true.
    surface_ready: bool,
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
            surface_ready: false,
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
        let Some(text) = self.text.as_mut() else {
            return;
        };

        let mut ctx = Context {
            viewport: self.viewport,
            text,
            clip_mask: &mut self.clip_mask,
            window: &window,
            modifiers: self.modifiers,
            cursor_x: self.cursor_x - self.window_rect.x as Coord,
            cursor_y: self.cursor_y - self.window_rect.y as Coord,
        };

        // Step 1: render consumer into the window-sized scratch (α + darkness convention, with clip_mask carving — UNCHANGED from the pre-fullscreen pipeline). Step 2: clear the screen buffer (so pixels outside window_rect are α=0 and the OS compositor sees through us). Step 3: finalize_into_screen reads scratch + clip_mask once per pixel and writes OS-ready ARGB into the screen buffer at window_rect offset — one pass over pixels, same per-pixel cost as the old in-place finalize_for_os.
        self.scratch.fill(0);
        self.app.render(&mut self.scratch, &mut ctx);
        drop(ctx);

        let scr_w = self.screen_size.0 as usize;
        let rect_x = self.window_rect.x;
        let rect_y = self.window_rect.y;
        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            let mut buffer = renderer.lock_buffer();
            buffer.fill(0);
            crate::paint::finalize_into_screen(
                &self.scratch,
                &self.clip_mask,
                win_w,
                win_h,
                &mut buffer,
                scr_w,
                rect_x,
                rect_y,
            );
            let _ = buffer.present();
        }
        #[cfg(not(target_os = "macos"))]
        {
            let Some(surface) = self.surface.as_mut() else {
                return;
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            buffer.fill(0);
            crate::paint::finalize_into_screen(
                &self.scratch,
                &self.clip_mask,
                win_w,
                win_h,
                &mut buffer,
                scr_w,
                rect_x,
                rect_y,
            );
            buffer.present().expect("softbuffer buffer.present");
        }
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
                    window: &window,
                    modifiers: self.modifiers,
                    cursor_x: self.cursor_x - self.window_rect.x as Coord,
                    cursor_y: self.cursor_y - self.window_rect.y as Coord,
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
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
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
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
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
                                window: &window,
                                modifiers: self.modifiers,
                                cursor_x: self.cursor_x - new_x as Coord,
                                cursor_y: self.cursor_y - new_y as Coord,
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

                // In-buffer drag-to-move: update window_rect.x/y by the cursor delta from the drag anchor. Also push a new XShape input region so click-through follows the visible window in real time (otherwise clicks land in the OLD rect for a frame). Skip consumer dispatch — they don't need cursor moves during the drag.
                if self.is_dragging_move {
                    let dx = (self.cursor_x as i32) - self.drag_move_anchor_screen.0;
                    let dy = (self.cursor_y as i32) - self.drag_move_anchor_screen.1;
                    self.window_rect.x = self.drag_move_rect_start.0 + dx;
                    self.window_rect.y = self.drag_move_rect_start.1 + dy;
                    if let Some(window) = self.window.as_ref().cloned() {
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
                    return;
                }

                if let (Some(window), Some(text)) =
                    (self.window.as_ref().cloned(), self.text.as_mut())
                {
                    let mut ctx = Context {
                        viewport: self.viewport,
                        text,
                        clip_mask: &mut self.clip_mask,
                        window: &window,
                        modifiers: self.modifiers,
                        cursor_x: self.cursor_x - self.window_rect.x as Coord,
                        cursor_y: self.cursor_y - self.window_rect.y as Coord,
                    };
                    let response = self.app.on_event(&event, &mut ctx);
                    let icon = self.app.cursor_for(self.cursor_x, self.cursor_y, &ctx);
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
                // Ctrl/Cmd + scroll → zoom. 1 step per scroll notch (LineDelta). Trackpad PixelDelta accumulates many small events; ~30px per step matches typical trackpad density.
                let steps: f32 = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 30.0,
                };
                if steps != 0.0 {
                    self.apply_zoom_change(Some(steps));
                    return;
                }
                self.dispatch_event(event);
            }
            WindowEvent::Focused(false) => {
                // Cancel any in-progress resize drag if we lose focus mid-drag (the user alt-tabbed or the WM stole focus). Keeps state consistent.
                if self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                }
                self.dispatch_event(event);
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
                // End of in-buffer drag-to-move. window_rect is already at its final position; input region was updated each tick. Just drop the flag.
                if self.is_dragging_move {
                    self.is_dragging_move = false;
                }
                self.dispatch_event(event);
            }
            WindowEvent::RedrawRequested => {
                // During a self-driven resize drag, push the new target geometry to the OS first. This is async: the OS will confirm via a later Resized event which triggers its own paint at the confirmed size.
                if self.is_dragging_resize {
                    self.apply_resize_drag();
                }
                // ALWAYS paint per vsync — even during a drag. Otherwise the screen "sticks" for however many vsyncs the OS takes to confirm our request_inner_size. Painting at the current viewport (= last OS-confirmed size) is correct: buffer size matches window size, no smear; when OS finally confirms a new size, Resized handler resizes + paints again at the new size, and we converge.
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
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
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
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
            };
            let response = self.app.on_event(&event, &mut ctx);
            drop(ctx);
            self.apply_response(response);
        }
    }
}

