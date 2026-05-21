//! `FluorApp` trait + entry point for consumer-driven desktop apps.
//!
//! Consumers implement [`FluorApp`] and pass the impl to [`run_app`]. The host opens a window, runs the event loop, presents the buffer, and dispatches events through the trait. All visible content (chrome, widgets, panes) is the consumer's responsibility — the host owns no domain state.
//!
//! Compose [`super::chrome_widget::DefaultChrome`] for the borderless window frame, [`crate::widgets::Textbox`] / [`crate::widgets::BlinkTimer`] for the textbox + blinking-cursor pattern, [`crate::Group`] for sub-viewport composite caching. The [`Context`] struct exposes the host's shared resources (viewport, text renderer, window handle, modifier state) to the consumer for the duration of each callback.
//!
//! The current `desktop::run(compositor, title)` is a transitional shim that wraps the legacy demo into a `FluorApp`. New code should use [`run_app`] directly.

use super::chrome::{self, ResizeEdge};
use crate::coord::Coord;
use crate::geom::Viewport;
use crate::text::TextRenderer;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{CursorIcon, Window, WindowAttributes, WindowId};

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

/// The host's adapter — owns platform handles + the consumer's `App`, dispatches events through the trait. Not user-facing; constructed by [`run_app`].
struct DesktopShell<A: FluorApp> {
    app: A,
    window: Option<Arc<Window>>,
    viewport: Viewport,

    // --- Renderer ---
    #[cfg(target_os = "macos")]
    renderer: Option<super::renderer_wgpu::Renderer>,
    #[cfg(not(target_os = "macos"))]
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,

    // --- Shared resources ---
    text: Option<TextRenderer>,
    /// Window-shape clip mask, one byte α per pixel. Same dimensions as the present buffer. Consumers write to it via `ctx.clip_mask` once per resize (e.g. `DefaultChrome` carves the rounded-corner cutouts); the boundary step multiplies it into each pixel's α before handing the buffer to the OS. Default `255` (fully visible) means a consumer that doesn't touch it gets a rectangular window with no clipping.
    clip_mask: Vec<u8>,
    cursor_x: Coord,
    cursor_y: Coord,
    modifiers: ModifiersState,

    // --- macOS manual resize tracking (winit stops delivering CursorMoved during edge-drag on macOS) ---
    #[cfg(target_os = "macos")]
    is_dragging_resize: bool,
    #[cfg(target_os = "macos")]
    resize_edge: ResizeEdge,
    #[cfg(target_os = "macos")]
    drag_start_size: (u32, u32),
    #[cfg(target_os = "macos")]
    drag_start_window_pos: (i32, i32),
    #[cfg(target_os = "macos")]
    drag_start_cursor_screen_pos: (f64, f64),
    #[cfg(target_os = "macos")]
    screen_height: u32,
}

impl<A: FluorApp> DesktopShell<A> {
    fn new(app: A) -> Self {
        Self {
            app,
            window: None,
            viewport: Viewport::new(1, 1),
            #[cfg(target_os = "macos")]
            renderer: None,
            #[cfg(not(target_os = "macos"))]
            surface: None,
            text: None,
            clip_mask: Vec::new(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            modifiers: ModifiersState::empty(),
            #[cfg(target_os = "macos")]
            is_dragging_resize: false,
            #[cfg(target_os = "macos")]
            resize_edge: ResizeEdge::None,
            #[cfg(target_os = "macos")]
            drag_start_size: (0, 0),
            #[cfg(target_os = "macos")]
            drag_start_window_pos: (0, 0),
            #[cfg(target_os = "macos")]
            drag_start_cursor_screen_pos: (0.0, 0.0),
            #[cfg(target_os = "macos")]
            screen_height: 0,
        }
    }

    fn render_frame(&mut self) {
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };
        let buf_w = self.viewport.width_px as usize;
        let buf_h = self.viewport.height_px as usize;
        // Make sure the clip mask is sized to the current viewport. Default fill is 255 = fully visible (consumer is opting in to clip-shape carving).
        let needed = buf_w * buf_h;
        if self.clip_mask.len() != needed {
            self.clip_mask.resize(needed, 255);
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
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
        };

        // Boundary: present buffer init to canonical empty (`0xFFFFFFFF` — byte-uniform memset, t=255 transparent), consumer renders into it (writing t-convention pixels + carving `ctx.clip_mask`), then `finalize_for_os` does the t→α flip + clip-mask multiply + Linux premult + pack in one pass. `ctx` drops first so we can re-borrow `self.clip_mask` immutably for the boundary call.
        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            let mut buffer = renderer.lock_buffer();
            buffer.fill(0xFFFFFFFF);
            self.app.render(&mut buffer, &mut ctx);
            drop(ctx);
            crate::paint::finalize_for_os(&mut buffer, &self.clip_mask);
            let _ = buffer.present();
        }
        #[cfg(not(target_os = "macos"))]
        {
            let Some(surface) = self.surface.as_mut() else {
                return;
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            buffer.fill(0xFFFFFFFF);
            self.app.render(&mut buffer, &mut ctx);
            drop(ctx);
            crate::paint::finalize_for_os(&mut buffer, &self.clip_mask);
            buffer.present().expect("softbuffer buffer.present");
        }
        // Drop `ctx` (releases &mut text borrow) and `buf_w`/`buf_h` shadow used for nothing — placeholder to avoid warnings if cfg drops both arms.
        let _ = (buf_w, buf_h);
    }

    /// Apply an [`EventResponse`] returned from `app.on_event`. Returns `true` if the response was `Close` (caller should terminate).
    fn apply_response(&mut self, response: EventResponse) -> bool {
        let Some(window) = self.window.as_ref().cloned() else {
            return false;
        };
        match response {
            EventResponse::Handled | EventResponse::Pass => false,
            EventResponse::StartWindowDrag => {
                let _ = window.drag_window();
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

    #[cfg(target_os = "linux")]
    fn start_resize(&mut self, edge: ResizeEdge, window: &Window) {
        if let Some(dir) = resize_direction(edge) {
            let _ = window.drag_resize_window(dir);
        }
    }

    #[cfg(target_os = "windows")]
    fn start_resize(&mut self, edge: ResizeEdge, window: &Window) {
        if let Some(dir) = resize_direction(edge) {
            let _ = window.drag_resize_window(dir);
        }
    }

    #[cfg(target_os = "macos")]
    fn start_resize(&mut self, edge: ResizeEdge, window: &Window) {
        // macOS: manual resize tracking via NSEvent polling — winit stops delivering CursorMoved once the cursor leaves during a resize drag on macOS.
        self.is_dragging_resize = true;
        self.resize_edge = edge;
        self.drag_start_size = (self.viewport.width_px, self.viewport.height_px);
        if let Ok(window_pos) = window.outer_position() {
            self.drag_start_window_pos = (window_pos.x, window_pos.y);
            self.drag_start_cursor_screen_pos = (
                window_pos.x as f64 + self.cursor_x as f64,
                window_pos.y as f64 + self.cursor_y as f64,
            );
        }
    }

    // --- macOS manual resize ---

    #[cfg(target_os = "macos")]
    fn poll_macos_resize(&mut self) -> bool {
        let Some(window) = self.window.as_ref() else {
            return false;
        };
        use std::ffi::{c_char, c_void};

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct NSPoint {
            x: f64,
            y: f64,
        }

        unsafe extern "C" {
            fn objc_msgSend(receiver: *const c_void, sel: *const c_void) -> usize;
            fn sel_registerName(name: *const c_char) -> *const c_void;
            fn objc_getClass(name: *const c_char) -> *const c_void;
        }

        unsafe {
            let cls = objc_getClass(b"NSEvent\0".as_ptr() as *const c_char);

            let sel_loc = sel_registerName(b"mouseLocation\0".as_ptr() as *const c_char);
            let mouse_location: extern "C" fn(*const c_void, *const c_void) -> NSPoint =
                std::mem::transmute(objc_msgSend as *const ());
            let ns_point = mouse_location(cls, sel_loc);

            let sel_btn = sel_registerName(b"pressedMouseButtons\0".as_ptr() as *const c_char);
            let buttons = objc_msgSend(cls, sel_btn);
            let left_held = buttons & 1 != 0;

            let scale = window.scale_factor();
            let phys_x = ns_point.x * scale;
            let phys_y = self.screen_height as f64 - ns_point.y * scale;

            if let Ok(window_pos) = window.outer_position() {
                self.cursor_x = (phys_x - window_pos.x as f64) as Coord;
                self.cursor_y = (phys_y - window_pos.y as f64) as Coord;
            }

            if !left_held {
                self.is_dragging_resize = false;
                self.resize_edge = ResizeEdge::None;
                return true;
            }

            self.apply_macos_resize();
        }
        false
    }

    #[cfg(target_os = "macos")]
    fn apply_macos_resize(&self) {
        let Some(window) = self.window.as_ref() else {
            return;
        };
        let Ok(window_pos) = window.outer_position() else {
            return;
        };

        let current_screen_x = window_pos.x as f64 + self.cursor_x as f64;
        let current_screen_y = window_pos.y as f64 + self.cursor_y as f64;

        let dx = (current_screen_x - self.drag_start_cursor_screen_pos.0) as Coord;
        let dy = (current_screen_y - self.drag_start_cursor_screen_pos.1) as Coord;

        let min_size: Coord = 128.0;

        let (new_width, new_height, should_move, new_x, new_y) = match self.resize_edge {
            ResizeEdge::Right => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                (w, self.drag_start_size.1, false, 0, 0)
            }
            ResizeEdge::Left => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let width_change = self.drag_start_size.0 as i32 - w as i32;
                (
                    w,
                    self.drag_start_size.1,
                    true,
                    self.drag_start_window_pos.0 + width_change,
                    self.drag_start_window_pos.1,
                )
            }
            ResizeEdge::Bottom => {
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                (self.drag_start_size.0, h, false, 0, 0)
            }
            ResizeEdge::Top => {
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let height_change = self.drag_start_size.1 as i32 - h as i32;
                (
                    self.drag_start_size.0,
                    h,
                    true,
                    self.drag_start_window_pos.0,
                    self.drag_start_window_pos.1 + height_change,
                )
            }
            ResizeEdge::TopRight => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let height_change = self.drag_start_size.1 as i32 - h as i32;
                (
                    w,
                    h,
                    true,
                    self.drag_start_window_pos.0,
                    self.drag_start_window_pos.1 + height_change,
                )
            }
            ResizeEdge::TopLeft => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let width_change = self.drag_start_size.0 as i32 - w as i32;
                let height_change = self.drag_start_size.1 as i32 - h as i32;
                (
                    w,
                    h,
                    true,
                    self.drag_start_window_pos.0 + width_change,
                    self.drag_start_window_pos.1 + height_change,
                )
            }
            ResizeEdge::BottomRight => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                (w, h, false, 0, 0)
            }
            ResizeEdge::BottomLeft => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                let width_change = self.drag_start_size.0 as i32 - w as i32;
                (
                    w,
                    h,
                    true,
                    self.drag_start_window_pos.0 + width_change,
                    self.drag_start_window_pos.1,
                )
            }
            ResizeEdge::None => return,
        };

        if should_move {
            let _ = window.set_outer_position(winit::dpi::PhysicalPosition::new(new_x, new_y));
        }
        let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(new_width, new_height));
    }
}

impl<A: FluorApp + 'static> ApplicationHandler for DesktopShell<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let monitor = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next());

        let initial = if let Some(ref mon) = monitor {
            let size = mon.size();
            let short = size.width.min(size.height);
            let h = short * 3 / 4;
            let w = h * 4 / 3;
            winit::dpi::PhysicalSize::new(w, h)
        } else {
            winit::dpi::PhysicalSize::new(1280, 800)
        };

        #[cfg(target_os = "macos")]
        if let Some(ref mon) = monitor {
            self.screen_height = mon.size().height;
        }

        let attrs = WindowAttributes::default()
            .with_title(self.app.title())
            .with_inner_size(initial)
            .with_min_inner_size(winit::dpi::PhysicalSize::new(24u32, 8u32))
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(cfg!(not(target_os = "macos")));
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));

        #[cfg(target_os = "macos")]
        {
            use winit::platform::macos::WindowExtMacOS;
            window.set_has_shadow(false);
        }

        self.viewport = Viewport::new(initial.width, initial.height);

        #[cfg(target_os = "macos")]
        {
            self.renderer = Some(super::renderer_wgpu::Renderer::new(
                &window,
                initial.width,
                initial.height,
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
                    NonZeroU32::new(initial.width).expect("nonzero width"),
                    NonZeroU32::new(initial.height).expect("nonzero height"),
                )
                .expect("softbuffer Surface::resize");
            self.surface = Some(surface);
        }

        if self.text.is_none() {
            self.text = Some(TextRenderer::new());
        }

        // Allocate the clip-mask buffer matched to the initial viewport (default 255 = fully visible — consumer carves shape via DefaultChrome or similar).
        let needed = self.viewport.width_px as usize * self.viewport.height_px as usize;
        if self.clip_mask.len() != needed {
            self.clip_mask.resize(needed, 255);
        }

        // Hand control to the consumer's init.
        {
            let text = self.text.as_mut().expect("text renderer initialized");
            let mut ctx = Context {
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x,
                cursor_y: self.cursor_y,
            };
            self.app.init(&mut ctx);
        }

        self.window = Some(window.clone());
        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        #[cfg(target_os = "macos")]
        if self.is_dragging_resize {
            if self.poll_macos_resize() {
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                std::time::Instant::now() + std::time::Duration::from_millis(16),
            ));
            return;
        }

        let needs_redraw = if let (Some(window), Some(text)) =
            (self.window.as_ref().cloned(), self.text.as_mut())
        {
            let mut ctx = Context {
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x,
                cursor_y: self.cursor_y,
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
                if size.width == self.viewport.width_px && size.height == self.viewport.height_px {
                    return;
                }
                if size.width == 0 || size.height == 0 {
                    return;
                }

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

                self.viewport = Viewport::new(size.width, size.height);
                let needed = size.width as usize * size.height as usize;
                if self.clip_mask.len() != needed {
                    self.clip_mask.resize(needed, 255);
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
                        cursor_x: self.cursor_x,
                        cursor_y: self.cursor_y,
                    };
                    self.app.on_resize(size.width, size.height, &mut ctx);
                    window.request_redraw();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x as Coord;
                self.cursor_y = position.y as Coord;

                if let (Some(window), Some(text)) =
                    (self.window.as_ref().cloned(), self.text.as_mut())
                {
                    let mut ctx = Context {
                        viewport: self.viewport,
                        text,
                        clip_mask: &mut self.clip_mask,
                        window: &window,
                        modifiers: self.modifiers,
                        cursor_x: self.cursor_x,
                        cursor_y: self.cursor_y,
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
            WindowEvent::Focused(false) => {
                #[cfg(target_os = "macos")]
                {
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
                #[cfg(target_os = "macos")]
                if self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }
                self.dispatch_event(event);
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
            }
            _ => {
                self.dispatch_event(event);
            }
        }
    }
}

impl<A: FluorApp + 'static> DesktopShell<A> {
    /// Helper: dispatch a generic event to `app.on_event`, applying any returned [`EventResponse`].
    fn dispatch_event(&mut self, event: WindowEvent) {
        if let (Some(window), Some(text)) = (self.window.as_ref().cloned(), self.text.as_mut()) {
            let mut ctx = Context {
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                window: &window,
                modifiers: self.modifiers,
                cursor_x: self.cursor_x,
                cursor_y: self.cursor_y,
            };
            let response = self.app.on_event(&event, &mut ctx);
            drop(ctx);
            self.apply_response(response);
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn resize_direction(edge: ResizeEdge) -> Option<winit::window::ResizeDirection> {
    use winit::window::ResizeDirection;
    Some(match edge {
        ResizeEdge::Top => ResizeDirection::North,
        ResizeEdge::Bottom => ResizeDirection::South,
        ResizeEdge::Left => ResizeDirection::West,
        ResizeEdge::Right => ResizeDirection::East,
        ResizeEdge::TopLeft => ResizeDirection::NorthWest,
        ResizeEdge::TopRight => ResizeDirection::NorthEast,
        ResizeEdge::BottomLeft => ResizeDirection::SouthWest,
        ResizeEdge::BottomRight => ResizeDirection::SouthEast,
        ResizeEdge::None => return None,
    })
}
