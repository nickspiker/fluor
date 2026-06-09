#![cfg_attr(not(feature = "std"), no_std)]
//! Fluor — first-principles GUI compositor library.
//!
//! Named for fluorite (the mineral that gave us "fluorescence" — glows on a band nothing else occupies) and *fluere* (Latin: to flow — "liquid stone"). Hard substrate of pane geometry + chrome shared across consumers, fluid center-origin coordinates that scale with the viewport.
//!
//! Every mainstream layout system is pixel-anchored: Android `dp`, CSS `px`, WPF DIPs, iOS `pt`, Flutter `dp` — all scale factors on a fixed physical reference. Even CSS `vmin` (the closest "viewport-relative" unit) uses `min(w, h)` and inherits a discontinuity at the diagonal. Fluor uses **harmonic-mean span** `2wh/(w+h)` as its scaling base, with the origin at the viewport center and `+y` down. That combination — center-origin + harmonic-mean unit + default convention (not opt-in) — appears to be unoccupied territory among compositors and toolkits.
//!
//! Built to deduplicate the chrome / paint / pane code currently duplicated across photon, rhe, and mandelbrot-exploder, and to be the eventual compositor for ferros.

extern crate alloc;

pub mod canvas;
pub mod coord;
pub mod event;
pub mod geom;
pub mod group;
pub mod host;
pub(crate) mod math;
pub mod paint;
pub mod pane;
pub(crate) mod par;
pub mod pixel;
pub mod region;
#[cfg(feature = "simd")]
pub(crate) mod simd;
pub mod stack;
pub mod theme;

#[cfg(feature = "text")]
pub mod text;

// Widgets (Textbox, Button) speak fluor-native event types ([`crate::event`]) and so compile on every host with `text` enabled. Whichever host is driving (host-winit on desktop, host-android on Android) translates platform input to fluor events at the boundary.
#[cfg(feature = "text")]
pub mod widgets;

pub use coord::{Coord, RuVec2};
pub use geom::Viewport;
pub use group::Group;
pub use paint::BlendMode;
pub use pane::{Compositor, Pane, PaneId};
pub use pixel::Argb8;
pub use region::Region;
