//! Native menu bar, hand-rolled — no `muda`, no cross-platform menu crate. On macOS we build a
//! real `NSMenu` from the app's [`super::menu::MenuItem`] spec using the same `objc2-app-kit`
//! vocabulary fluor already speaks for click-thru. Everywhere else the two hooks are no-ops.
//!
//! Routing: an `NSMenuItem`'s target/action fires on the **main thread**, synchronously, inside
//! the AppKit run loop that winit is already pumping — so there's no cross-thread wake to arrange.
//! The action stashes the item's `tag` (= the app's `id`) in a small queue; the host drains it in
//! `about_to_wait` and dispatches each as [`crate::event::Event::MenuItem`]. One target object
//! serves every item (it reads `sender.tag()`), kept alive for the process in a static.

extern crate alloc;
use alloc::vec::Vec;

// ─────────────────────────────────────────── non-macOS: no menu bar yet ───────────────────────

#[cfg(not(target_os = "macos"))]
pub(crate) fn install(_items: &[super::menu::MenuItem]) {}

#[cfg(not(target_os = "macos"))]
pub(crate) fn drain() -> Vec<u32> {
    Vec::new()
}

// ─────────────────────────────────────────── macOS: real NSMenu ───────────────────────────────

#[cfg(target_os = "macos")]
pub(crate) use imp::{drain, install};

#[cfg(target_os = "macos")]
mod imp {
    use super::super::menu::MenuItem;
    use alloc::vec::Vec;
    use objc2::rc::Retained;
    use objc2::runtime::NSObject;
    use objc2::{define_class, msg_send, sel};
    use objc2_app_kit::{NSApplication, NSMenu, NSMenuItem};
    use objc2_foundation::{MainThreadMarker, NSString};
    use std::sync::Mutex;

    /// Menu-item ids chosen since the last drain (pushed on the main thread by `fired:`).
    static QUEUE: Mutex<Vec<u32>> = Mutex::new(Vec::new());
    /// The single target object every item points at — `NSMenuItem.target` is unretained, so we
    /// must keep it alive ourselves for the life of the menu (i.e., the process).
    static TARGET: Mutex<Option<Retained<Target>>> = Mutex::new(None);

    pub(crate) fn drain() -> Vec<u32> {
        core::mem::take(&mut *QUEUE.lock().unwrap())
    }

    define_class!(
        #[unsafe(super(NSObject))]
        #[name = "FluorMenuTarget"]
        struct Target;

        impl Target {
            /// The action for every fluor menu item. `sender.tag()` carries the app-assigned id.
            #[unsafe(method(fluorMenuFired:))]
            fn fired(&self, sender: &NSMenuItem) {
                let tag = unsafe { sender.tag() };
                if tag >= 0 {
                    QUEUE.lock().unwrap().push(tag as u32);
                }
            }
        }
    );

    fn new_target() -> Retained<Target> {
        unsafe { msg_send![Target::alloc(), init] }
    }

    fn make_action(
        mtm: MainThreadMarker,
        target: &Target,
        label: &str,
        id: u32,
    ) -> Retained<NSMenuItem> {
        let it = NSMenuItem::new(mtm);
        unsafe {
            it.setTitle(&NSString::from_str(label));
            it.setTag(id as isize);
            it.setTarget(Some(target));
            it.setAction(Some(sel!(fluorMenuFired:)));
        }
        it
    }

    fn add(mtm: MainThreadMarker, target: &Target, parent: &NSMenu, spec: &MenuItem) {
        match spec {
            MenuItem::Action { id, label } => {
                let it = make_action(mtm, target, label, *id);
                unsafe { parent.addItem(&it) };
            }
            MenuItem::Sub { label, items } => {
                // A menu-bar submenu is an NSMenuItem whose submenu is a titled NSMenu.
                let holder = NSMenuItem::new(mtm);
                let sub = NSMenu::new(mtm);
                unsafe {
                    holder.setTitle(&NSString::from_str(label));
                    sub.setTitle(&NSString::from_str(label));
                }
                for child in items {
                    add(mtm, target, &sub, child);
                }
                unsafe {
                    holder.setSubmenu(Some(&sub));
                    parent.addItem(&holder);
                }
            }
            MenuItem::Separator => {
                let sep = NSMenuItem::separatorItem(mtm);
                unsafe { parent.addItem(&sep) };
            }
        }
    }

    pub(crate) fn install(items: &[MenuItem]) {
        if items.is_empty() {
            return;
        }
        let Some(mtm) = MainThreadMarker::new() else { return };
        let app = NSApplication::sharedApplication(mtm);
        let target = new_target();
        // Augment the existing main menu (winit sets up an app menu with Quit) rather than replace
        // it, so we don't lose ⌘Q. Our items are top-level submenus appended to the bar.
        let main = match app.mainMenu() {
            Some(m) => m,
            None => {
                let m = NSMenu::new(mtm);
                app.setMainMenu(Some(&m));
                m
            }
        };
        for spec in items {
            add(mtm, &target, &main, spec);
        }
        *TARGET.lock().unwrap() = Some(target);
    }
}
