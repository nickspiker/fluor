# fluor

**First-principles GUI compositor library — center-origin coordinates, harmonic-mean relative units, CPU softbuffer rendering, ARM-first.**

Named for **fluorite** — the mineral that gave us "fluorescence" — and ***fluere***, Latin for "to flow." Liquid stone. Hard substrate of pane geometry and chrome shared across consumers, fluid coordinates that flow with the viewport so the same layout reads correctly and identically on a watch, a 8K monitor, or any size limited only by hardware.

The name is also the differentiator. Every mainstream layout system is pixel-anchored: Android `dp`, CSS `px`, WPF DIPs, iOS `pt`, Flutter `dp` — every one of them is just a scale factor on a fixed physical reference. Even CSS `vmin` (the closest "viewport-relative" unit) uses `min(w, h)` and inherits a discontinuity at the diagonal. Fluor uses **harmonic-mean span** `2wh/(w+h)` as its scaling base, with the origin at the viewport center and `+y` down. As far as I can tell, that combination — center-origin + harmonic-mean unit + dominant convention (not opt-in) — is unoccupied territory among compositors and toolkits. Fluorite glows on a band nothing else emits in; fluor occupies a layout band nothing else occupies.

`fluor` exists to deduplicate the bespoke compositor code currently sitting inside [photon](https://github.com/nickspiker/photon), [rhe](https://github.com/nickspiker/rhe), and [mandelbrot-exploder](https://github.com/nickspiker/mandelbrot-exploder), and to be the eventual compositor for [ferros](https://github.com/nickspiker/ferros) — a Killswitch-ready Rust OS targeting ARM-only with no GPU drivers. Today it is a thin layer of shared chrome + paint primitives; consumer migrations begin once text rendering lands.

## Status

**v0 — pre-alpha.** Window chrome, pane composition, and transform-aware text rendering all work end-to-end. Textboxes, widgets, and layout persistence are not yet built. Expect breaking changes at every layer until the first consumer migration validates the API.

| Layer | State |
|---|---|
| Center-origin coords (`RuVec2`, `Viewport`) | ✓ f32 storage, harmonic-mean span/perimeter/diagonal_sq |
| Pane tree (`Compositor`) | ✓ insert / remove / get / hit-test / focus / z-order / render |
| Paint primitives | ✓ ARGB blend, fill_rect (solid + blend), stroke_rect, circle_filled, glyph rasterizers, background noise; `Clip` + `AlphaMask` + `Transform` types; `quantize_rotation` / `snap_rotation` helpers |
| Window chrome | ✓ controls strip, edges-and-mask, hairlines, hover overlay (lifted verbatim from photon); always-visible at minimum window size via `MIN_BUTTON_HEIGHT_PX + ceil(span/32)` formula |
| Drag / resize | ✓ drag-to-move + 8-region edge resize via winit; WM-enforced `min_inner_size = (24, 8)` |
| Text rendering | ✓ cosmic-text + swash; Open Sans bundled; **transform-aware** (arbitrary rotation / skew / scale via `swash::scale`, glyph contour transformed in font space for proper hinting + AA); per-glyph LRU image cache keyed on `(font, glyph, size, transform)` |
| Killswitch close | ✓ `std::process::exit(0)` on close button + `CloseRequested` — no Drop chain, kernel reclaims everything in microseconds; same semantics as ferros's hardware power cutoff |
| Textbox / widgets | ✗ planned |
| Layout persistence (VSF) | ✗ planned — 1 Hz / release debounce |
| `host-bare` (ferros, no_std framebuffer) | ✗ planned |
| SIMD blit kernels (NEON / SSE2) | ✗ deferred — scalar path already hits ~500 fps fullscreen 4K with a normal layout |

## Quick example

```rust
use fluor::{Compositor, RuVec2, Viewport};
use fluor::paint::pack_argb;

fn main() {
    // 1280×800 viewport — center is (0, 0), +x right, +y down, units are RU (relative).
    let mut compositor = Compositor::new(Viewport::new(1280, 800));

    compositor.insert(
        RuVec2::new(-0.15, -0.08),       // center in RU
        RuVec2::new(0.14, 0.10),         // half-extent in RU (so width = 2 * 0.14)
        pack_argb(220, 90, 80, 255),     // ARGB background
    );

    fluor::host::desktop::run(compositor, "fluor — panes").expect("event loop");
}
```

Run the bundled demo with `cargo run --example panes`.

## Why center-origin coordinates

Origin at `(0, 0)` is the viewport center, +x right, +y down. The y-down convention matches text engines, image scanlines, and pixel storage (zero flip points below the layout layer); apps that genuinely want y-up math negate y in their content boundary.

The rationale for center-origin vs. the conventional top-left:
- **Symmetric transforms.** Zoom and rotate around origin require no offset bookkeeping.
- **Resize is the natural case.** A pane pinned at `(0, 0)` stays centered when the host window resizes; no recomputation.
- **Sign carries meaning.** A glance at `(-3, +2)` tells you "upper-left of center."
- **Polar layouts trivial.** Radial menus, gauges, knobs, anything circular is implicitly in (r, θ).

## Why RU (relative units) instead of pixels

`1 RU = span_pixels * ru_multiplier` where `span = 2wh / (w + h)` is the harmonic mean of viewport width and height. The same RU layout looks right on an 11" laptop and a 32" monitor without DPI awareness.

Photon's `AGENT.md` "Universal Scaling Units" section is the canonical reasoning: harmonic mean is the unique scaling base with smooth derivative at `w == h`, finite slope at the axes, slope exactly 1 along the diagonal, and a bias toward the smaller dimension on narrow displays.

## Why f32 storage (not Spirix)

Layout coordinates and viewport dimensions are `f32`. Hardware has native `f32` add/mul on every relevant target (NEON `fadd`/`fmul`, AVX/SSE `addps`/`mulps`); Spirix is software-emulated. Photon and other consumers run as fast as their pre-fluor code only when the layout layer is f32. Spirix support is welcome where it specifically matters (precision-critical rasterizer paths, deterministic-zoom apps, future ferros builds via a `spirix-coord` feature flag) but is not the default for windowing.

## Why CPU softbuffer (not wgpu)

Because the CPU path is already fast enough that the GPU isn't needed: ~500 fps fullscreen 4K with a normal layout. The perf headroom over the 60–144 Hz consumer ceiling is so large that adding wgpu would buy nothing for the common case while adding a heavy dependency, a driver-stack failure mode, and a second renderer to keep in sync. GPU support may be added if a future workload genuinely requires it (high-density vector animation, very large textures); until then the CPU rasterizer is the only path.

A pleasant side effect: the same rendering code runs on bare-metal targets like ferros that have no GPU drivers at all — no fallback path required, the production path *is* the bare-metal path.

## Architecture

```
fluor (lib)
├── coord       — RuVec2, Coord (= f32)
├── geom        — Viewport with span/perimeter/diagonal_sq + RU↔pixel
├── paint       — blend, fill_rect, stroke_rect, circle_filled, glyph::*, scale_alpha, blend_rgb_only, background_noise; Clip / AlphaMask / Transform; quantize_rotation + snap_rotation
├── pane        — Pane, PaneId, Compositor (tree + hit-test + focus + z-order + render)
├── text        — TextRenderer (cosmic-text + swash); transform-aware glyph rasterization via swash::scale; LRU glyph cache keyed on (font, glyph, size, transform's linear part)
├── theme       — color constants (Android byte-swap behind cfg)
└── host/
    ├── chrome  — draw_window_controls, draw_window_edges_and_mask, draw_button_hairlines, draw_button_hover_by_pixels, get_resize_edge, hit_test_map (verbatim photon port; MIN_BUTTON_HEIGHT_PX floor for guaranteed visibility at any legal window size)
    └── desktop — winit + softbuffer host (feature `host-winit`, default; std::process::exit(0) on close for Killswitch compliance)
```

Future: `host-bare` (no_std framebuffer for ferros), textbox + widget kit, SIMD kernels, layout VSF persistence.

## Features

- `default = ["std", "host-winit", "simd"]`
- `host-winit` — winit + softbuffer desktop host (default; transitively requires `text`)
- `host-bare` — bare-metal `&mut [u32]` framebuffer host (planned, gated for `no_std`)
- `text` — cosmic-text + swash text rendering with transform support
- `simd` — runtime-dispatched NEON / SSE2 / AVX2 blit kernels (planned)

## Building

```sh
./build-development.sh   # canonical dev build
cargo run --example panes
```

`./build-development.sh` is preferred over `cargo build --release` per `AGENT.md`; release builds only when explicitly requested.

## Coding rules

`AGENT.md` (verbatim from photon) governs this codebase. Notable rules: no bounds checks / clamps / saturating arithmetic without proven justification (Rule 0); decimal indexing forbidden; VSF type-marker matching, never positional; no fixed-pixel values (use `span` / `perimeter` / `diagonal_sq`); persistence cadence on streaming UI events is ≤1 Hz with flush-on-release; public API stable, internal renderer hot-swappable via enum / feature / runtime detect.

## License

MIT OR Apache-2.0, at your option.

## Author

Nick Spiker — `<fractaldecoder@proton.me>`
