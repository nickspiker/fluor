//! [`WindowHandle`] — minimal app-facing window controls that work on every host.
//!
//! Apps reach this thru [`crate::host::app::Context::window`]. Today only `request_redraw` lives here because that's the only method real apps (photon, panes) actually call on `ctx.window`. Everything else — title changes, maximize / minimize toggles, drag-window — flows thru [`crate::host::EventResponse`] variants so the host's window state stays the single source of truth.
//!
//! Adding methods: only if a real app needs them and the operation has a sensible implementation on every host. Things that only make sense on desktop (e.g. setting an OS-level cursor) should stay private to the host's internal code, not exposed thru `WindowHandle`. Geometry READS are deliberately absent: under the fullscreen-compositor model the OS window is the whole screen, so `outer_position`/`inner_size` answer for the surface, not the visible window — exactly the trap that produced the 2026-08-16 vanish + dead-click restore. The visible rect flows to apps thru `FluorApp::on_window_rect_changed`.

/// App-facing window controls. Whichever host is driving the event loop (host-winit on desktop, host-android on Android) provides a concrete type implementing this trait via [`crate::host::app::Context::window`].
pub trait WindowHandle {
    /// Request that the host repaint the window at its earliest convenience. Idempotent within a frame — many calls in one tick coalesce into one render. Apps call this any time they've mutated state that affects the next paint (cursor moves, focus changes, network events arriving on the UI thread, animations driven by tick).
    fn request_redraw(&self);

    /// Set the OS window title — what the task switcher, the window menu, Mission Control and the Dock read. Distinct from any title the app DRAWS: a client-decorated app paints its own bar, and the OS never sees that string unless it is also set here, so a title that tracks the open document has to come through this. Default is a no-op, for hosts with no OS-level title to set (Android's surface IS the window).
    fn set_title(&self, _title: &str) {}

    /// Put text on the system clipboard — the one call every app's copy button makes (Nick 2026-10-26). On a desktop it goes to the OS clipboard through one handle kept for the life of the process (an X11 or Wayland clipboard is served by its owner, and a handle made for each copy and dropped lost most copies, photon 2026-09-02); on Android the shell holds it for the activity's clipboard manager ([`crate::host::android::AndroidShell::take_clipboard`]). Default is a no-op, for a host with no clipboard.
    fn set_clipboard(&self, _text: &str) {}

    /// Swap the OS window icon at runtime — the taskbar button, alt-tab and title bar, where the OS reads it from the window: straight (not premultiplied) RGBA rows, `width * height * 4` bytes, or `None` to hand the icon back to the OS default (the executable's, the `.desktop` file's).
    /// Reach is winit's `set_window_icon`: Windows and X11; on Wayland and macOS the OS reads the icon from packaging, not the window, so this does nothing there. Default is a no-op, for hosts with no OS window icon.
    fn set_icon(&self, _icon: Option<(&[u8], u32, u32)>) {}
}

#[cfg(feature = "host-winit")]
impl WindowHandle for winit::window::Window {
    fn request_redraw(&self) {
        winit::window::Window::request_redraw(self);
    }

    fn set_title(&self, title: &str) {
        winit::window::Window::set_title(self, title);
    }

    fn set_clipboard(&self, text: &str) {
        crate::host::clipboard::set_text(text);
    }

    fn set_icon(&self, icon: Option<(&[u8], u32, u32)>) {
        let icon = match icon {
            None => None,
            // winit refuses only a byte count that disagrees with the size, the caller's bug; the window keeps the icon it had.
            Some((rgba, w, h)) => match winit::window::Icon::from_rgba(rgba.to_vec(), w, h) {
                Ok(icon) => Some(icon),
                Err(_) => return,
            },
        };
        winit::window::Window::set_window_icon(self, icon);
    }
}
