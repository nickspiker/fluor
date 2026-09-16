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
//! # Only while active — and why the work area must not be read from AppKit
//!
//! Presentation options are process-global and apply ONLY while the app is frontmost, so this is a
//! no-op in the background and needs no teardown on focus loss. That property has a sharp edge:
//! `NSScreen.visibleFrame` — which is how [`super::app`] derives every monitor's work area — reports
//! the menu bar and Dock as reserved whenever they happen to be showing, which includes the entire
//! startup window before the app first becomes active.
//!
//! Reading it at the wrong instant poisons geometry permanently. Measured live: the monitor list was
//! enumerated a few milliseconds BEFORE the surfaces, so it captured `3840x2130` (30pt reserved for
//! a bar this very app was about to hide) while the surfaces, enumerated after, captured the true
//! `3840x2160`. "Move window to this monitor" reads the monitor list, so the visible window came out
//! 30pt short inside a full-size surface — a dead strip on screen, and a resolution-following viewer
//! asking its guest for the short size.
//!
//! So the work area is not asked of AppKit at all once we have hidden the bar: [`menu_bar_is_hidden`]
//! short-circuits it to the full monitor rect. That is the truth by construction — we hid the bar AND
//! the Dock, so nothing is reserved — and it holds no matter when any caller happens to ask.

use core::sync::atomic::{AtomicBool, Ordering};

use objc2_app_kit::{NSApplication, NSApplicationPresentationOptions};
use objc2_foundation::MainThreadMarker;

/// Set once we have asked AppKit to hide the menu bar and Dock. Read by the work-area derivation,
/// which must NOT consult `visibleFrame` afterwards — see the module docs.
static HIDDEN: AtomicBool = AtomicBool::new(false);

/// Have we hidden the menu bar (and, per the pairing rule, the Dock)? When true there is no reserved
/// strip on any screen and a monitor's work area is simply its full rect.
pub(crate) fn menu_bar_is_hidden() -> bool {
    HIDDEN.load(Ordering::Relaxed)
}

/// Hide (or restore) the system menu bar and Dock for this process.
///
/// Fully hidden, not auto-hidden: no mouse-to-edge reveal. Both return when the app is no longer
/// frontmost. Safe to call repeatedly with the same value. Does nothing when called off the main
/// thread, which is where AppKit requires it.
///
/// Call this BEFORE enumerating monitors — the flag it sets is what keeps the work-area derivation
/// away from a `visibleFrame` that still counts the strip we are in the act of claiming.
pub(crate) fn set_menu_bar_hidden(hidden: bool) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    // Set before the AppKit call, not after: the flag describes our INTENT, and every work-area read
    // from here on must already agree with it — including any that races the option taking effect.
    HIDDEN.store(hidden, Ordering::Relaxed);
    let app = NSApplication::sharedApplication(mtm);
    let options = if hidden {
        // HideMenuBar is only legal alongside HideDock — see the pairing rule above.
        NSApplicationPresentationOptions::HideMenuBar | NSApplicationPresentationOptions::HideDock
    } else {
        NSApplicationPresentationOptions::Default
    };
    app.setPresentationOptions(options);
}
