//! macOS: the DISPLAY going dark reads as the app losing focus (Nick 2026-10-15: "should be the same on all platforms. screen off should be treated as defocus").
//!
//! A window keeps its key status while the display sleeps, the screensaver locks it, or another user's session takes the console, so winit reports no focus change and an app's focus edges (saving work, sleeping the caret, telling its other devices it stopped looking) never ran. NSWorkspace says it directly: `ScreensDidSleep` / `ScreensDidWake`, and `SessionDidResignActive` / `SessionDidBecomeActive` for fast user switching. The observer only records the state; the host drains it in `about_to_wait` (the notifications are posted on the main run loop, which runs that step right after) and folds it into the app's focus.

use std::sync::atomic::{AtomicBool, Ordering};

use objc2::runtime::AnyObject;
use objc2::{class, msg_send};
use objc2_foundation::NSString;

/// The display is asleep (or the session is not the active one).
static ASLEEP: AtomicBool = AtomicBool::new(false);
/// The state moved since the host last drained it.
static CHANGED: AtomicBool = AtomicBool::new(false);
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Watch the workspace's screen and session notifications. Idempotent; the observers live for the process.
pub fn install() {
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }
    // SAFETY: plain AppKit messages on the main thread (called from the host's `resumed`); the blocks are leaked deliberately — the observers must outlive every notification.
    unsafe {
        let ws: *mut AnyObject = msg_send![class!(NSWorkspace), sharedWorkspace];
        if ws.is_null() {
            return;
        }
        let nc: *mut AnyObject = msg_send![ws, notificationCenter];
        if nc.is_null() {
            return;
        }
        for (name, asleep) in [
            ("NSWorkspaceScreensDidSleepNotification", true),
            ("NSWorkspaceScreensDidWakeNotification", false),
            ("NSWorkspaceSessionDidResignActiveNotification", true),
            ("NSWorkspaceSessionDidBecomeActiveNotification", false),
        ] {
            let n = NSString::from_str(name);
            let block = block2::RcBlock::new(move |_note: *mut AnyObject| {
                ASLEEP.store(asleep, Ordering::SeqCst);
                CHANGED.store(true, Ordering::SeqCst);
            });
            let _observer: *mut AnyObject = msg_send![nc, addObserverForName: &*n, object: std::ptr::null::<AnyObject>(), queue: std::ptr::null::<AnyObject>(), usingBlock: &*block];
            std::mem::forget(block);
        }
    }
}

/// The new state if it moved since the last call: `Some(true)` = the display went dark.
pub fn take_change() -> Option<bool> {
    CHANGED.swap(false, Ordering::SeqCst).then(|| ASLEEP.load(Ordering::SeqCst))
}
