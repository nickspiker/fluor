# fluor

**First-principles GUI compositor library — center-origin coordinates, harmonic-mean relative units, CPU softbuffer rendering, ARM-first.**

Named for **fluorite** — the mineral that gave us "fluorescence" — and ***fluere***, Latin for "to flow." Liquid stone. Hard substrate of pane geometry and chrome shared across consumers, fluid coordinates that flow with the viewport so the same layout reads correctly and identically on a watch, an 8K monitor, or any size limited only by hardware.

---

## What fluor does differently

Four independent design choices, each with a concrete reason. They compose into a system that is simultaneously more correct, more portable, and faster than conventional approaches.

### 1. Center-origin coordinates

Every mainstream layout system puts the origin at the top-left corner of the viewport. fluor puts it at the center.

The consequence is mostly invisible until you do transforms. Zoom and rotate around origin require no offset bookkeeping — the math just works. A pane placed at `(0, 0)` stays centered when the window resizes; the center of the screen is always `(0, 0)`. Sign carries spatial meaning: `(-0.3, +0.1)` is left of center, slightly below — readable at a glance. Polar layouts (gauges, radial menus, knobs) require no coordinate conversion. The y-axis is down, matching text engines, image scanlines, and pixel memory order, so there are no flip points below the layout layer.

Top-left origin is a historical artifact of raster scanlines. For a layout system it is a wrong default.

### 2. Harmonic-mean relative units

Every mainstream layout system defines a "device-independent pixel" that is a scaled version of a physical pixel: Android `dp`, CSS `px`, WPF DIPs, iOS `pt`, Flutter `dp`. Even CSS `vmin` — the closest thing to a viewport-relative unit in the mainstream — uses `min(w, h)`, which has a discontinuity: on a portrait display one axis drives the unit; rotate 90° and the other axis drives it, causing a jump.

fluor uses the harmonic mean of width and height: `span = 2wh / (w + h)`.

The harmonic mean is the unique scaling base with:
- Smooth derivative at `w == h` (no discontinuity at the diagonal)
- Finite slope at the axes (no singularities on degenerate layouts)
- Slope exactly 1 along the diagonal (unit scales 1:1 with both dimensions on a square display)
- Natural bias toward the smaller dimension on non-square displays (the layout is constrained by what's scarce, not what's abundant)

A layout specified in RU (relative units) looks correct on an 11" laptop, a 32" monitor, and an embedded display without any DPI awareness code. This combination — center-origin plus harmonic-mean unit as the dominant convention rather than an opt-in — is, as far as can be determined, unoccupied territory among compositors and GUI toolkits.

### 3. Front-to-back compositing with a transparency accumulator

Conventional compositing renders back-to-front (painter's algorithm). Each layer blends onto the accumulated result beneath it:

```
// Standard Porter-Duff "over", bottom-up, float path
let a = (src >> 24) as f32 / 255.0;   // float convert + divide
let r = (src >> 16 & 0xFF) as f32;
let dr = (dst >> 16 & 0xFF) as f32;
let out_r = dr * (1.0 - a) + r * a;   // sub + mul + mul + add
// repeat for g, b
// repack dst
// repeat for every layer, every pixel, always
```

The float division `/ 255.0` and the repeated `1.0 - a` are not free. More importantly: there is no early-out. In a UI where the frontmost layers are overwhelmingly opaque — buttons, chrome, panels — you are doing full blend work on pixels that will be completely covered. The work is pure waste.

fluor renders front-to-back with a `u8` transparency accumulator. Transparency convention: `0 = fully opaque, 255 = fully transparent`. The accumulator tracks how much of the pixel's contribution budget remains unclaimed:

```rust
// Front-to-back, u8 transparency, no floats

let mut r_acc: u16 = 0;
let mut g_acc: u16 = 0;
let mut b_acc: u16 = 0;
let mut remaining: u8 = 255; // full budget: pixel is entirely unclaimed

for &(pixel, t) in layers_front_to_back {
    if remaining == 0 { break; }           // budget exhausted — skip everything behind

    let a: u16 = (255 - t) as u16;        // opacity: one subtraction
    let contrib = (remaining as u16 * a) >> 8; // budget × opacity: one mul, one shift

    r_acc += contrib * ((pixel >> 16 & 0xFF) as u16) >> 8;
    g_acc += contrib * ((pixel >>  8 & 0xFF) as u16) >> 8;
    b_acc += contrib * ((pixel       & 0xFF) as u16) >> 8;

    remaining = (remaining as u16 * t as u16 >> 8) as u8; // attenuate budget: one mul, one shift
}

// one repack at the end
let result = 0xFF_00_00_00
    | (r_acc as u32) << 16
    | (g_acc as u32) <<  8
    |  b_acc as u32;
```

#### A concrete example

Four layers: tooltip over button over panel over background.

```
layer 0  tooltip     t=180  remaining=255  contrib=75   remaining→179
layer 1  button      t=0    remaining=179  contrib=179  remaining→0
layer 2  panel       t=60   remaining==0   BREAK ← never touched
layer 3  background  t=0                   never touched
```

The button is fully opaque. After layer 1 the transparency accumulator is zero. Layers 2 and 3 are never entered — not skipped at the SIMD level, not zeroed out, simply never executed.

#### Cost comparison per pixel per layer

| | Bottom-up float | Front-to-back u8 |
|---|---|---|
| Float converts | 4 (one per channel + alpha) | 0 |
| `1.0 - α` subtractions | 1 per layer | 0 |
| Multiplications | 8 floats per layer | 3 u16 per layer |
| Layers executed | always N | stops at first opaque |
| Repacks | 1 per layer | 1 total |
| Early-out | impossible | `remaining == 0` |

In a real UI the frontmost opaque surface is typically encountered at layer 1 or 2. Chrome, buttons, and panels are overwhelmingly opaque. The common case pays 3 u16 multiplications for the semi-transparent layers above the first opaque one, then stops.

### 4. Why transparency-convention alpha (`0 = opaque`)

Every existing API uses opacity convention: `α = 255` means fully opaque. fluor inverts this for its internal representation.

The reason is exactness, and it matters in integer arithmetic.

With opacity convention and a `u8` accumulator, the ceiling is exact: `opacity = 255` contributes 100% exactly. But the floor is not: `opacity = 0` contributes `0/256`, which rounds to zero but is not mathematically zero. Invisible layers are slightly inexact.

With transparency convention and a `u8` accumulator, the floor is exact: `transparency = 0` sets `remaining = 0` exactly — `(remaining * 0) >> 8 == 0`, no rounding. The accumulator hits zero cleanly and the early-out fires precisely.

The ceiling is slightly inexact: `transparency = 255` attenuates the budget by `255/256 ≈ 0.996` rather than exactly 1.0. But a layer with `transparency = 255` is invisible — it would be culled before entering the blend pass. The imprecision never executes.

The background layer closes the guarantee. The background is always fully opaque (`t = 0`), so `remaining` hits zero exactly on every pixel. There is no path to a pixel that escapes without being fully accounted for.

The external boundary — PNG load, image decode, glyph coverage from cosmic-text — performs a one-time `255 - a` flip on import. The cost is paid once per asset load, not per pixel per frame.

Internally, all variables are named `t` or `transparency`, never `alpha` or `a`. The convention is stated explicitly wherever it appears in the codebase.

---

## Why CPU softbuffer instead of GPU

The CPU rasterizer sustains ~500 fps fullscreen at 4K with a typical pane layout. The consumer refresh rate ceiling is 60–144 Hz. The headroom is so large that adding a GPU path would buy nothing for the common case while adding a driver-stack failure mode, a second renderer to maintain, and a heavy dependency.

The more important reason: the same CPU rasterizer runs on bare-metal targets with no GPU drivers — specifically ferros, a no_std ARM OS. There is no fallback path because there is no fallback needed. The production path is the bare-metal path.

GPU support may be added if a specific workload genuinely requires it (high-density vector animation, very large textures). Until then adding it would be adding complexity, not capability.

---

## Why float coordinates but u32 pixels

Layout coordinates (`RuVec2`, `Viewport`) are `float`. Every relevant hardware target has native `float` arithmetic: NEON `fadd`/`fmul`, AVX/SSE `addps`/`mulps`. The precision is more than sufficient for layout geometry at any viewport size.

The compositing pass — the inner loop that actually writes pixels — is entirely `u32` packed ARGB and `u16`/`u8` integer arithmetic. No floats cross into the blend path. This is intentional and load-bearing: float conversion and float multiply in the inner loop are the dominant cost in conventional compositors. fluor eliminates them entirely.

Spirix (fluor's companion floating-point arithmetic system) is welcome on precision-critical rasterizer paths and deterministic-zoom applications, and will be the default for the ferros build via a `spirix-coord` feature flag. It is not the default for windowing because the software-emulation overhead would eliminate the performance advantage of CPU rendering.

---

## Status

**v0 — pre-alpha.** Window chrome, pane composition, and transform-aware text rendering all work end-to-end. Textboxes, widgets, and layout persistence are not yet built. Expect breaking changes at every layer until the first consumer migration validates the API.

| Layer | State |
|---|---|
| Center-origin coords (`RuVec2`, `Viewport`) | ✓ float storage, harmonic-mean span/perimeter/diagonal_sq |
| Pane tree (`Compositor`) | ✓ insert / remove / get / hit-test / focus / z-order / render |
| Paint primitives | ✓ ARGB blend, fill_rect (solid + blend), stroke_rect, circle_filled, glyph rasterizers, background noise; `Clip` + `AlphaMask` + `Transform` types; `quantize_rotation` / `snap_rotation` helpers |
| Window chrome | ✓ controls strip, edges-and-mask, hairlines, hover overlay; always-visible at minimum window size via `MIN_BUTTON_HEIGHT_PX + ceil(span/32)` formula |
| Drag / resize | ✓ drag-to-move + 8-region edge resize via winit; WM-enforced `min_inner_size = (24, 8)` |
| Text rendering | ✓ cosmic-text + swash; Open Sans bundled; transform-aware (arbitrary rotation / skew / scale via `swash::scale`); per-glyph LRU cache keyed on `(font, glyph, size, transform)` |
| Killswitch close | ✓ `std::process::exit(0)` on close + `CloseRequested` — no Drop chain, kernel reclaims everything |
| Textbox / widgets | ✗ planned |
| Layout persistence (VSF) | ✗ planned — 1 Hz / release debounce |
| `host-bare` (ferros, no_std framebuffer) | ✗ planned |
| SIMD blit kernels (NEON / SSE2) | ✗ deferred — scalar path already hits ~500 fps fullscreen 4K |

---

## Quick example

```rust
use fluor::{Compositor, RuVec2, Viewport};
use fluor::paint::pack_argb;

fn main() {
    // 1280×800 viewport — center is (0, 0), +x right, +y down, units in RU.
    let mut compositor = Compositor::new(Viewport::new(1280, 800));

    compositor.insert(
        RuVec2::new(-0.15, -0.08),       // center in RU
        RuVec2::new(0.14, 0.10),         // half-extent in RU (width = 2 × 0.14)
        pack_argb(220, 90, 80, 255),     // ARGB background
    );

    fluor::host::desktop::run(compositor, "fluor — panes").expect("event loop");
}
```

Run the bundled demo: `cargo run --example panes`

---

## Architecture

```
fluor (lib)
├── coord       — RuVec2, Coord (= f32)
├── geom        — Viewport with span/perimeter/diagonal_sq + RU↔pixel
├── paint       — front-to-back blend, fill_rect, stroke_rect, circle_filled,
│                 glyph::*, scale_alpha, blend_rgb_only, background_noise;
│                 Clip / AlphaMask / Transform;
│                 quantize_rotation + snap_rotation
│                 (transparency convention: t=0 opaque, t=255 transparent)
├── pane        — Pane, PaneId, Compositor (tree + hit-test + focus + z-order + render)
│                 render walk is front-to-back; early-out when remaining==0
├── text        — TextRenderer (cosmic-text + swash); transform-aware glyph
│                 rasterization via swash::scale; per-glyph LRU image cache
├── theme       — color constants (Android byte-swap behind cfg)
└── host/
    ├── chrome  — draw_window_controls, draw_window_edges_and_mask,
    │             draw_button_hairlines, draw_button_hover_by_pixels,
    │             get_resize_edge, hit_test_map
    └── desktop — winit + softbuffer host (feature `host-winit`, default;
                  std::process::exit(0) on close for Killswitch compliance)
```

Future: `host-bare` (no_std framebuffer for ferros), textbox + widget kit, SIMD kernels, layout VSF persistence.

---

## Features

- `default = ["std", "host-winit", "simd"]`
- `host-winit` — winit + softbuffer desktop host (default; transitively requires `text`)
- `host-bare` — bare-metal `&mut [u32]` framebuffer host (planned, gated for `no_std`)
- `text` — cosmic-text + swash text rendering with transform support
- `simd` — runtime-dispatched NEON / SSE2 / AVX2 blit kernels (planned)
- `spirix-coord` — Spirix arithmetic for precision-critical paths (planned)

---

## Building

```sh
./build-development.sh   # canonical dev build
cargo run --example panes
```

`./build-development.sh` is preferred over `cargo build --release` per `AGENT.md`; release builds only when explicitly requested.

---

## Coding rules

`AGENT.md` governs this codebase. Notable rules: no bounds checks / clamps / saturating arithmetic without proven justification (Rule 0); decimal indexing forbidden; VSF type-marker matching, never positional; no fixed-pixel values (use `span` / `perimeter` / `diagonal_sq`); persistence cadence on streaming UI events is ≤1 Hz with flush-on-release; public API stable, internal renderer hot-swappable via enum / feature / runtime detect.

Alpha convention: variables named `t` or `transparency` use transparency convention (`0 = opaque`). Never rename these to `alpha` or `a` — the conventions are opposite and the blend math depends on which is in use.

---

## License

MIT OR Apache-2.0, at your option.

## Author

Nick Spiker — `<fractaldecoder@proton.me>`