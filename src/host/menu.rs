//! Native menu-bar item spec.
//!
//! An app declares its menu by returning a tree of these from [`crate::host::app::FluorApp::menu`].
//! The host builds a real OS menu from it — today that's an `NSMenu` on macOS (hand-rolled on
//! objc2, no external menu crate); other platforms don't surface a menu yet and simply ignore it.
//! When the user chooses an [`MenuItem::Action`], the host delivers its `id` back to the app as
//! [`crate::event::Event::MenuItem`]. The app owns all state — the menu is a fire-an-id surface,
//! not a stateful widget; toggles just re-fire their id and the app flips its own flag.
//!
//! IDs are the app's namespace: any `u32` the app cares to route on. Keep them stable across a
//! rebuild so a menu refresh doesn't renumber live items.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

/// One entry in the native menu tree. Grow deliberately — every arm has to be understood by each
/// platform's builder.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuItem {
    /// A clickable item. Choosing it delivers `Event::MenuItem(id)`.
    Action {
        id: u32,
        label: String,
    },
    /// A submenu grouping `items` under `label` (one level of nesting is all the host builds today).
    Sub {
        label: String,
        items: Vec<MenuItem>,
    },
    /// A horizontal separator line.
    Separator,
}
