//! Center-origin, +y-down 2D coordinates in `f32`.
//!
//! `Coord` is the RU scalar. RU is an authoring unit: layout is computed in it and converted to pixels with [`Viewport::ru_to_px_x`](crate::Viewport::ru_to_px_x) and friends. The conversion is one-way on purpose — the runtime unit is pixels, and a pointer lands on a widget by indexing the hit map, not by comparing geometry, so nothing needs to come back from pixels to RU.
//!
//! Storage is `f32` (see `Coord` below). Hardware-native addition and multiplication on every relevant target — same speed as the bespoke compositors fluor replaces. Spirix is welcome in precision-critical app code or via a future `spirix-coord` feature flag, but it is not the default for windowing.


/// Single scalar used for all RU coordinate components. `f32` by default — hardware-native on aarch64 NEON and x86 AVX/SSE.
pub type Coord = f32;
