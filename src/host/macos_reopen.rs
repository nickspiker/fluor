//! macOS Dock-reopen support: teach winit's application delegate one extra selector.
//!
//! # Why this exists
//!
//! macOS separates the APPLICATION from its windows. A Dock icon represents the app, so clicking it is not a window operation and produces no window event. AppKit instead asks the application's delegate
//! `applicationShouldHandleReopen:hasVisibleWindows:`. Nothing answered it, so a Dock click on a minimized
//! Photon was silently dropped while cmd-Tab and Mission Control worked (those drive WINDOWS, which do produce events).
//!
//! # Why not the other approaches
//!
//! - **Observe `NSApplicationDidBecomeActiveNotification`** — tried, and it does not fire: minimizing does
//!   not DEACTIVATE the app, so there is no activation transition on a Dock click. Confirmed in a live log:
//!   exactly one activation at launch, none on the Dock click. That is precisely why AppKit has a dedicated
//!   reopen message instead of reusing activation.
//! - **Install our own `NSApplicationDelegate`** — tried, and it aborts inside `-[NSApplication run]`
//!   (SIGABRT, no Rust panic). winit 0.30 DOES install its own delegate (`WinitApplicationDelegate`, see
//!   its `platform_impl/macos/event_loop.rs` `app.setDelegate(...)`), and its whole event pump hangs off
//!   that object. Replacing it tears out `applicationDidFinishLaunching:` and everything after.
//!   (winit's own macOS module docs claim it registers no delegate; that is not true of this version.)
//! - **Fork winit** — diverges a dependency forever against a file upstream actively changes.
//!
//! # What this does instead
//!
//! Adds the selector to winit's EXISTING delegate class with `class_addMethod`. Purely additive: winit's own methods are untouched, nothing is replaced, and the app keeps exactly one delegate. It no-ops safely if the class is ever renamed or already answers the selector, so a winit upgrade degrades to today's behaviour rather than crashing.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use objc2::runtime::{AnyClass, AnyObject, Bool, Sel};
use objc2::sel;

/// Set when the Dock icon is clicked; drained by the host loop.
static REOPEN: AtomicBool = AtomicBool::new(false);
/// Count of reopen messages seen — lets a log distinguish "never delivered" from "delivered, restore failed".
static REOPEN_COUNT: AtomicU32 = AtomicU32::new(0);
/// Guards against installing twice.
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// The added method. Signature must match AppKit's exactly:
/// `- (BOOL)applicationShouldHandleReopen:(NSApplication *)app hasVisibleWindows:(BOOL)flag`
///
/// `extern "C"` with no unwinding across the boundary: this is called BY AppKit, so a panic here would abort the process. It does nothing but set a flag, which cannot panic.
extern "C-unwind" fn should_handle_reopen(
    _this: *mut AnyObject,
    _cmd: Sel,
    _app: *mut AnyObject,
    _has_visible: Bool,
) -> Bool {
    REOPEN_COUNT.fetch_add(1, Ordering::Relaxed);
    REOPEN.store(true, Ordering::Relaxed);
    // true: let AppKit also run its default handling (unhide, etc.).
    Bool::YES
}

/// Attach the selector to winit's delegate class. Idempotent and safe to call every resume.
pub(super) fn install() {
    if INSTALLED.swap(true, Ordering::Relaxed) {
        return;
    }
    // SAFETY: we look the class up by name and only ADD a method that is not already present. The implementation matches the selector's documented signature, and it cannot unwind.
    unsafe {
        let Some(cls) = AnyClass::get(c"WinitApplicationDelegate") else {
            log::warn!("FLUOR-REOPEN: WinitApplicationDelegate not found — Dock reopen unavailable (winit renamed its delegate?)");
            return;
        };
        let sel = sel!(applicationShouldHandleReopen:hasVisibleWindows:);
        if cls.instance_method(sel).is_some() {
            log::info!("FLUOR-REOPEN: delegate already answers reopen — leaving it alone");
            return;
        }
        // "B@:@B" — returns BOOL, takes (self, _cmd, id, BOOL).
        let types = c"B@:@B";
        let imp: extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject, Bool) -> Bool =
            should_handle_reopen;
        // `Imp` is a bare `unsafe extern "C-unwind" fn()`; the cast erases our real signature, which is why the `types` string above must describe it exactly ("B@:@B" = BOOL result; self, _cmd, id, BOOL).
        let added = objc2::ffi::class_addMethod(
            cls as *const AnyClass as *mut AnyClass,
            sel,
            std::mem::transmute::<_, objc2::runtime::Imp>(imp),
            types.as_ptr(),
        );
        if added.is_true() {
            log::info!("FLUOR-REOPEN: Dock reopen wired onto WinitApplicationDelegate");
        } else {
            log::warn!("FLUOR-REOPEN: class_addMethod refused — Dock reopen unavailable");
        }
    }
}

/// True once per Dock click; clears on read.
pub(super) fn take_reopen() -> bool {
    REOPEN.swap(false, Ordering::Relaxed)
}

/// How many reopen messages have arrived — diagnostic.
pub(super) fn reopen_count() -> u32 {
    REOPEN_COUNT.load(Ordering::Relaxed)
}
