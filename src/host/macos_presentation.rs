//! macOS presentation options: let a fullscreen-compositor app cover the system menu bar.
//!
//! # Why this exists
//!
//! fluor's desktop model is a borderless, transparent, monitor-sized OS window pinned to each
//! monitor's origin ([`super::app`]). On Windows and X11 that genuinely covers the screen. On macOS
//! it does not: the system menu bar is drawn at a window level above ordinary windows, so it sits on
//! top of the surface no matter where the window is placed or how it is decorated. An app that fills
//! the screen therefore loses a strip at the top — and, for a remote-desktop viewer, loses it out of
//! the pixels it streams, so the guest is rendered at less than the monitor's real height.
//!
//! Raising the window level is the obvious-looking fix and is wrong: winit's
//! `WindowLevel::AlwaysOnTop` maps to `NSFloatingWindowLevel` (3), which is *below*
//! `NSMainMenuWindowLevel` (24). Levels above the menu bar exist but fight the OS for input routing
//! and break Cmd-Tab. The supported route is `NSApplication`'s presentation options, which is what
//! full-screen video and editing apps use.
//!
//! # The pairing rule
//!
//! Neither menu-bar option stands alone — AppKit raises `NSInvalidArgumentException` unless it is
//! combined with a Dock option, and the legal pairings are asymmetric:
//!
//! | menu bar          | requires                    |
//! |-------------------|-----------------------------|
//! | `AutoHideMenuBar` | `AutoHideDock` or `HideDock`|
//! | `HideMenuBar`     | `HideDock` (only)           |
//!
//! We take `HideMenuBar | HideDock`: the bar is GONE, not auto-hidden. Auto-hide was the first cut
//! and it is wrong for a remote viewer — the reveal gesture is "put the pointer at the top edge",
//! which is exactly what the viewer forwards to the guest, so the bar kept flickering in over the
//! stream when the user was reaching for the guest's own menu bar. Nothing is made unreachable:
//! Cmd-Tab, Cmd-Q and force-quit are untouched, and both bar and Dock return the moment the app
//! stops being frontmost (see the activity note below). An app that hides the menu bar this way
//! should surface its own controls somewhere the OS chrome isn't — [`super::macos_dock_menu`].
//!
//! # Only while active — and why that needs a frame re-assert
//!
//! Presentation options are process-global and apply ONLY while the app is frontmost, so this is a
//! no-op in the background and needs no teardown on focus loss. That property has a sharp edge: at
//! window-creation time the app is typically NOT yet active, so the menu bar is still up, the
//! screen's `visibleFrame` still excludes it, and AppKit constrains each new window to that shorter
//! rect. Hiding the bar afterwards does not give the strip back on its own — the window stays short
//! and the strip becomes dead space. [`super::app::DesktopShell`] therefore re-asserts each
//! surface's full monitor rect on the activation edge; see `reassert_monitor_frames`.

use objc2_app_kit::{NSApplication, NSApplicationPresentationOptions};
use objc2_foundation::MainThreadMarker;

/// Hide (or restore) the system menu bar and Dock for this process.
///
/// Fully hidden, not auto-hidden: no mouse-to-edge reveal. Both return when the app is no longer
/// frontmost. Safe to call repeatedly with the same value. Does nothing when called off the main
/// thread, which is where AppKit requires it.
pub(crate) fn set_menu_bar_hidden(hidden: bool) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    let options = if hidden {
        // HideMenuBar is only legal alongside HideDock — see the pairing rule above.
        NSApplicationPresentationOptions::HideMenuBar | NSApplicationPresentationOptions::HideDock
    } else {
        NSApplicationPresentationOptions::Default
    };
    app.setPresentationOptions(options);
}
