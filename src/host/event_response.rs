//! [`EventResponse`] — fluor-native return type for widget capability traits and `FluorApp::on_event`.
//!
//! Lives outside `host::app` because widget capability traits (which return `EventResponse`) compile without `host-winit`. Same enum, just relocated so the dependency graph works on every host.

/// Resize-edge classification (see `chrome::get_resize_edge` for the classifier). Lives here
/// rather than in `chrome` so `EventResponse::StartResize` compiles without the `icon` feature
/// (chrome is icon-gated; this enum is pure geometry). Re-exported from `chrome` for old paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeEdge {
    None,
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// What the consumer wants the host to do after a widget click / key / `FluorApp::on_event`. Pass-thru behaviour lets the consumer ignore events they don't care about; the explicit variants override the host's default for that event.
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
    /// Toggle the internal `window_rect` between user-sized and screen-sized. The fullscreen-compositor architecture means the OS surface is always at monitor size; `winit::Window::set_maximized` is a no-op on a borderless fullscreen window. The host owns the actual toggle state — on first invocation it saves the current `window_rect` and resizes to the full screen; on the next, it restores the saved rect. Triggers `on_resize`, full-repaint, and an X11 input-region update under the hood.
    ToggleMaximized,
    /// Minimize the window. On desktop calls `winit::Window::set_minimized(true)`; on Android a no-op (the OS owns lifecycle). Exists as a distinct variant (rather than the consumer calling `ctx.window.set_minimized` directly) so chrome's `Minimize`-button widget can return a window-handle-free response.
    Minimize,
}
