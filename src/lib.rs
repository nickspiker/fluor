#![cfg_attr(not(feature = "std"), no_std)]
//! Fluor — first-principles GUI compositor library.
//!
//! Center-origin, harmonic-mean RU coordinates with +y down and Spirix `ScalarF4E4` storage. CPU pixel buffer with SIMD blits. Layout state serializes via VSF. Targets: aarch64 (production, including ferros bare-metal), x86_64 (development).

extern crate alloc;

pub mod coord;
pub mod geom;
pub mod host;
pub mod paint;
pub mod pane;
pub mod theme;

#[cfg(feature = "text")]
pub mod text;

pub use coord::{Coord, RuVec2};
pub use geom::Viewport;
pub use pane::{Compositor, Pane, PaneId};
