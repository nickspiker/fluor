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
//! That asymmetry decides this for us. `HideMenuBar` was tried and it FORCES `HideDock` — there is
//! no combination that removes the bar outright and leaves the Dock reachable. The result was an app
//! with no menu bar, no Dock, and therefore no way to reach its own controls without Cmd-Tabbing off
//! it first. Removing both pieces of chrome is only viable for an app that draws its own.
//!
//! So: `AutoHideMenuBar | AutoHideDock`. Both slide away, both come back on a pointer-to-edge
//! gesture, and the app's menus stay reachable in the bar where a macOS user expects them. The cost
//! is that a remote viewer forwards that same top-edge gesture to its guest, so reaching for the
//! guest's own menu bar can pull ours in over the stream — an annoyance, against no access at all.
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
//! So the work area is not asked of AppKit at all once we have taken the chrome out of the reserved
//! area: [`nothing_reserved`] short-circuits it to the full monitor rect. That is the truth by
//! construction — an auto-hiding bar and Dock reserve nothing, which is the whole point of asking —
//! and it holds no matter when any caller happens to ask.

use core::sync::atomic::{AtomicBool, Ordering};

use objc2_app_kit::{NSApplication, NSApplicationPresentationOptions};
use objc2_foundation::MainThreadMarker;

/// Set once we have asked AppKit to auto-hide the menu bar and Dock. Read by the work-area
/// derivation, which must NOT consult `visibleFrame` afterwards — see the module docs.
static UNRESERVED: AtomicBool = AtomicBool::new(false);

/// Have we taken the menu bar and Dock out of the screen's reserved area? When true, no screen
/// reserves a strip and a monitor's work area is simply its full rect.
pub(crate) fn nothing_reserved() -> bool {
    UNRESERVED.load(Ordering::Relaxed)
}

/// Hide (or restore) the system menu bar and Dock for this process.
///
/// Auto-hide (or restore) the system menu bar and Dock for this process.
///
/// Both slide away and both come back on a pointer-to-edge gesture, so the app's own menus stay
/// reachable in the bar. Safe to call repeatedly with the same value. Does nothing when called off
/// the main thread, which is where AppKit requires it.
///
/// Call this BEFORE enumerating monitors — the flag it sets is what keeps the work-area derivation
/// away from a `visibleFrame` that still counts the strip we are in the act of claiming.
pub(crate) fn set_menu_bar_hidden(hidden: bool) {
    let Some(mtm) = MainThreadMarker::new() else {
        return;
    };
    // Set before the AppKit call, not after: the flag describes our INTENT, and every work-area read
    // from here on must already agree with it — including any that races the option taking effect.
    UNRESERVED.store(hidden, Ordering::Relaxed);
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
