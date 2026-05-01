#![cfg_attr(not(feature = "std"), no_std)]
//! Fluor — first-principles GUI compositor library.
//!
//! Named for fluorite (the mineral that gave us "fluorescence" — glows on a band nothing else occupies) and *fluere* (Latin: to flow — "liquid stone"). Hard substrate of pane geometry + chrome shared across consumers, fluid center-origin coordinates that scale with the viewport.
//!
//! Every mainstream layout system is pixel-anchored: Android `dp`, CSS `px`, WPF DIPs, iOS `pt`, Flutter `dp` — all scale factors on a fixed physical reference. Even CSS `vmin` (the closest "viewport-relative" unit) uses `min(w, h)` and inherits a discontinuity at the diagonal. Fluor uses **harmonic-mean span** `2wh/(w+h)` as its scaling base, with the origin at the viewport center and `+y` down. That combination — center-origin + harmonic-mean unit + default convention (not opt-in) — appears to be unoccupied territory among compositors and toolkits.
//!
//! Built to deduplicate the chrome / paint / pane code currently duplicated across photon, rhe, and mandelbrot-exploder, and to be the eventual compositor for ferros.

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
