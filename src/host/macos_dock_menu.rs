//! macOS Dock-icon menu: teach winit's application delegate one more selector.
//!
//! # Why this exists
//!
//! An app that hides the system menu bar ([`super::macos_presentation`]) hides its OWN menus with
//! it — on macOS an app's menu IS the menu bar. A fullscreen compositor surface therefore has
//! nowhere to put "switch monitor" / "toggle HUD" once it covers the screen. The Dock icon's
//! context menu is the one piece of app chrome that survives: AppKit asks the application delegate
//! `applicationDockMenu:` and stacks whatever it returns ABOVE the stock items, so the app's own
//! entries sit directly over Options / Show All Windows / Hide / Quit.
//!
//! # Reaching it while the menu bar is hidden
//!
//! `HideMenuBar` forces `HideDock` (AppKit's pairing rule), so the Dock is not on screen while the
//! app is frontmost — Cmd-Tab to another app and the Dock, and this menu, come back. That is the
//! cost of a fully-hidden bar; the alternative was a menu bar that reveals on a pointer-to-top-edge
//! gesture the viewer forwards to its guest.
//!
//! # How it attaches
//!
//! Exactly like [`super::macos_reopen`], and for the same reason: winit 0.30 installs its own
//! `WinitApplicationDelegate` and its whole event pump hangs off that object, so a second delegate
//! aborts inside `-[NSApplication run]`. We add the selector to winit's EXISTING class with
//! `class_addMethod` — purely additive, nothing replaced, one delegate. It no-ops safely if the
//! class is renamed or already answers the selector, so a winit upgrade degrades to "no Dock menu"
//! rather than crashing.

use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

use objc2::runtime::{AnyClass, AnyObject, Sel};
use objc2::sel;

/// The `NSMenu` handed back to AppKit, or null before one is built. Deliberately a raw pointer to a
/// LEAKED menu: it lives for the process, and `Retained<NSMenu>` is main-thread-only so it cannot
/// sit in a static. Written once on the main thread during setup, read on the main thread by AppKit.
static DOCK_MENU: AtomicPtr<AnyObject> = AtomicPtr::new(ptr::null_mut());
/// Guards against installing twice.
static INSTALLED: AtomicBool = AtomicBool::new(false);

/// Publish the menu AppKit should show for the Dock icon. Takes ownership of a `+1` pointer (the
/// caller leaks a `Retained`), which is correct precisely once per process.
pub(super) fn set_menu(menu: *mut AnyObject) {
    DOCK_MENU.store(menu, Ordering::Relaxed);
}

/// The added method. Signature must match AppKit's exactly:
/// `- (NSMenu *)applicationDockMenu:(NSApplication *)sender`
///
/// `extern "C-unwind"` with no unwinding across the boundary: this is called BY AppKit, so a panic
/// here would abort the process. It does nothing but load a pointer, which cannot panic. Returning
/// null is legal and simply means "no custom items" — that is the state before the app's menu spec
/// has been read.
extern "C-unwind" fn application_dock_menu(
    _this: *mut AnyObject,
    _cmd: Sel,
    _app: *mut AnyObject,
) -> *mut AnyObject {
    DOCK_MENU.load(Ordering::Relaxed)
}

/// Attach the selector to winit's delegate class. Idempotent and safe to call every resume.
pub(super) fn install() {
    if INSTALLED.swap(true, Ordering::Relaxed) {
        return;
    }
    // SAFETY: we look the class up by name and only ADD a method that is not already present. The
    // implementation matches the selector's documented signature, and it cannot unwind.
    unsafe {
        let Some(cls) = AnyClass::get(c"WinitApplicationDelegate") else {
            log::warn!("FLUOR-DOCKMENU: WinitApplicationDelegate not found — Dock menu unavailable (winit renamed its delegate?)");
            return;
        };
        let sel = sel!(applicationDockMenu:);
        if cls.instance_method(sel).is_some() {
            log::info!("FLUOR-DOCKMENU: delegate already answers applicationDockMenu: — leaving it alone");
            return;
        }
        // "@@:@" — returns id, takes (self, _cmd, id).
        let types = c"@@:@";
        let imp: extern "C-unwind" fn(*mut AnyObject, Sel, *mut AnyObject) -> *mut AnyObject =
            application_dock_menu;
        // `Imp` is a bare `unsafe extern "C-unwind" fn()`; the cast erases our real signature, which
        // is why the `types` string above must describe it exactly.
        let added = objc2::ffi::class_addMethod(
            cls as *const AnyClass as *mut AnyClass,
            sel,
            std::mem::transmute::<_, objc2::runtime::Imp>(imp),
            types.as_ptr(),
        );
        if added.is_true() {
            log::info!("FLUOR-DOCKMENU: Dock menu wired onto WinitApplicationDelegate");
        } else {
            log::warn!("FLUOR-DOCKMENU: class_addMethod refused — Dock menu unavailable");
        }
    }
}
