//! Three static panes in a host window. Smallest possible end-to-end demo: opens a desktop window, renders a fluor `Compositor` with three opaque panes, handles resize. No mouse drag, no persistence — those land in follow-up examples.

use fluor::{Compositor, RuVec2, Viewport};
use fluor::paint::pack_argb;

fn main() {
    let mut compositor = Compositor::new(Viewport::new(1280, 800));

    // Three overlapping panes. Center-origin RU coords: (0,0) is window center, +y is down.
    compositor.insert(
        RuVec2::new(-0.15, -0.08),
        RuVec2::new(0.14, 0.10),
        pack_argb(220, 90, 80, 255),
    );
    compositor.insert(
        RuVec2::new(0.05, 0.04),
        RuVec2::new(0.12, 0.12),
        pack_argb(90, 180, 100, 255),
    );
    compositor.insert(
        RuVec2::new(0.18, -0.14),
        RuVec2::new(0.09, 0.14),
        pack_argb(80, 100, 220, 255),
    );

    fluor::host::desktop::run(compositor, "fluor — panes").expect("event loop");
}
