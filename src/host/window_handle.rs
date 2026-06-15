//! [`WindowHandle`] — minimal app-facing window controls that work on every host.
//!
//! Apps reach this thru [`crate::host::app::Context::window`]. Today only `request_redraw` lives here because that's the only method real apps (photon, panes) actually call on `ctx.window`. Everything else — title changes, maximize / minimize toggles, drag-window — flows thru [`crate::host::EventResponse`] variants so the host's window state stays the single source of truth.
//!
//! Adding methods: only if a real app needs them and the operation has a sensible implementation on every host. Things that only make sense on desktop (e.g. setting an OS-level cursor) should stay private to the host's internal code, not exposed thru `WindowHandle`.

/// App-facing window controls. Whichever host is driving the event loop (host-winit on desktop, host-android on Android) provides a concrete type implementing this trait via [`crate::host::app::Context::window`].
pub trait WindowHandle {
    /// Request that the host repaint the window at its earliest convenience. Idempotent within a frame — many calls in one tick coalesce into one render. Apps call this any time they've mutated state that affects the next paint (cursor moves, focus changes, network events arriving on the UI thread, animations driven by tick).
    fn request_redraw(&self);
}

#[cfg(feature = "host-winit")]
impl WindowHandle for winit::window::Window {
    fn request_redraw(&self) {
        winit::window::Window::request_redraw(self);
    }
}
