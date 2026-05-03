//! Widget primitives — single-line textbox today, more to come (button, scroll, etc.). Each widget owns its visual + interaction state and renders into a pixel buffer via the shared paint + text infrastructure.
//!
//! Widgets are positioned in **pixel coordinates** for now. Once consumers are migrating, this will move to RU coords matching the rest of the layout.

pub mod blink;
pub mod textbox;
pub use blink::BlinkTimer;
pub use textbox::Textbox;
