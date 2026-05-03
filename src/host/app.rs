//! `FluorApp` trait + entry point for consumer-driven desktop apps.
//!
//! Consumers implement [`FluorApp`] and pass the impl to [`run_app`]. The host opens a window, runs the event loop, presents the buffer, and dispatches events through the trait. All visible content (chrome, widgets, panes) is the consumer's responsibility — the host owns no domain state.
//!
//! Compose [`super::chrome_widget::DefaultChrome`] for the borderless window frame, [`crate::widgets::Textbox`] / [`crate::widgets::BlinkTimer`] for the textbox + blinking-cursor pattern, [`crate::Group`] for sub-viewport composite caching. The [`Context`] struct exposes the host's shared resources (viewport, text renderer, window handle, modifier state) to the consumer for the duration of each callback.
//!
//! The current `desktop::run(compositor, title)` is a transitional shim that wraps the legacy demo into a `FluorApp`. New code should use [`run_app`] directly.

use crate::coord::Coord;
use crate::geom::Viewport;
use crate::text::TextRenderer;
use super::chrome::ResizeEdge;
use std::time::Instant;
use winit::error::EventLoopError;
use winit::event::WindowEvent;
use winit::keyboard::ModifiersState;
use winit::window::{CursorIcon, Window};

/// Per-callback access to host-owned shared resources. Re-borrowed for each call into the trait — the consumer can keep references for the duration of the call but not across calls.
pub struct Context<'a> {
    /// Current viewport in physical pixels.
    pub viewport: Viewport,
    /// Shared font system + glyph caches. Initialized lazily by the host on first window creation; passed by mutable reference because cache insertion and font loading require mutation.
    pub text: &'a mut TextRenderer,
    /// The host's winit window handle. Use for `set_cursor`, `set_minimized`, `set_maximized`, `request_redraw`, `drag_window`, `set_title`, `outer_position`.
    pub window: &'a Window,
    /// Latest tracked modifier state (shift / ctrl / alt / super).
    pub modifiers: ModifiersState,
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
    fn title(&self) -> &str { "" }

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
    fn wake_at(&self) -> Option<Instant> { None }
}

/// Run the desktop host until the window closes.
///
/// **Phase C scaffold:** the full platform plumbing implementation lands in Phase D when the new `examples/panes.rs` exercises every code path. For now, callers should continue using [`super::desktop::run`] (which still drives the legacy demo through the existing `DesktopApp`).
#[allow(unused_variables)]
pub fn run_app<A: FluorApp + 'static>(app: A) -> Result<(), EventLoopError> {
    unimplemented!("run_app lands in Phase D — use desktop::run for now");
}
