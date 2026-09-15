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
//! `AutoHideMenuBar` on its own raises `NSInvalidArgumentException` — AppKit requires it to be
//! combined with `AutoHideDock` or `HideDock`. We pair it with `AutoHideDock` so both come back on a
//! mouse-to-edge gesture; nothing is made permanently unreachable, and no Cmd-Tab or force-quit
//! behaviour is disabled.
//!
//! Presentation options are process-global and only apply while the app is active, so this is a
//! no-op in the background and needs no teardown on focus loss.

use objc2_app_kit::{NSApplication, NSApplicationPresentationOptions};
use objc2_foundation::MainThreadMarker;

/// Hide (or restore) the system menu bar and Dock for this process.
///
/// Both reappear when the pointer is moved to the screen edge. Safe to call repeatedly with the same
/// value. Does nothing when called off the main thread, which is where AppKit requires it.
pub(crate) fn set_menu_bar_hidden(hidden: bool) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    let app = NSApplication::sharedApplication(mtm);
    let options = if hidden {
        // AutoHideMenuBar is only legal alongside a Dock option — see the pairing rule above.
        NSApplicationPresentationOptions::AutoHideMenuBar
            | NSApplicationPresentationOptions::AutoHideDock
    } else {
        NSApplicationPresentationOptions::Default
    };
    app.setPresentationOptions(options);
}
