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

### 3. Front-to-back compositing — buffer-as-accumulator, single `under` kernel

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

fluor inverts the direction and uses the buffer itself as the transparency accumulator. The buffer's t-byte (top byte) tracks remaining transparency budget per pixel. Layers paint topmost-first via one binary operation:

```rust
// t-convention: t=0 opaque, t=255 transparent. Argb8 is a type alias for u32.
trait Blend {
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8;
}

impl Blend for Argb8 {
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8 {
        if self < 0x01000000 { return self; }       // dst opaque (t==0): single CMP early-out
        let top_t = self >> 24;
        let bot_t = bottom >> 24;
        let top_opacity = 256 - top_t;
        let contrib     = (top_t * (256 - bot_t)) >> 8;

        let (mr, mg, mb) = mode.kernel(top_rgb, bot_rgb);   // pure channel function; no transparency math here

        let nr = (tr * top_opacity + mr * contrib) >> 8;    // top_opacity + contrib ≤ 256 → nr ≤ 255 exactly
        let ng = (tg * top_opacity + mg * contrib) >> 8;
        let nb = (tb * top_opacity + mb * contrib) >> 8;
        ((top_t * bot_t) >> 8) << 24 | (nr << 16) | (ng << 8) | nb
    }
}
```

No floats. No `/ 255`. All `>> 8`. The invariant `top_opacity + contrib ≤ 256` keeps the per-channel result in `[0, 255]` without explicit saturation. Every blend mode (`Normal`, `Multiply`, `Screen`, `Add`, `Subtract`, `Overlay`, `Darken`, `Lighten`) goes through this same outer shape — only `(mr, mg, mb)` changes.

#### How a frame composes

The present buffer is initialized to `0xFFFFFFFF` (t=255 transparent, RGB=255 — invisible at full transparency, byte-uniform so the fill is a single `memset`). Groups flatten into it topmost-first. Each `flatten_into` walks the bbox and calls `dst[i].under(src[i], mode)` per pixel. When `dst.t` reaches 0 at a pixel, every subsequent Group short-circuits on that pixel via the `dst < 0x01000000` early-out — *one u32 compare against an immediate*, no shift, no mask.

#### A concrete example

Four layers: tooltip over button over panel over background.

```
layer 0  tooltip     t=180  buffer.t=255 (empty)   → contrib=53   buffer.t→179
layer 1  button      t=0    buffer.t=179           → contrib=179  buffer.t→0   (now opaque)
layer 2  panel       t=60   buffer.t==0            → EARLY-OUT, src not read
layer 3  background  t=0    buffer.t==0            → EARLY-OUT, src not read
```

After layer 1 the buffer's t-byte is 0 at this pixel. Layers 2 and 3 are short-circuited — the lower bbox kernel calls visit those pixels but the very first instruction (`if self < 0x01000000`) returns immediately. Not skipped at the SIMD level, not zeroed out, simply never decoded past the early-out branch.

#### Cost comparison per pixel per layer

| | Bottom-up float | Front-to-back u8 (`under`) |
|---|---|---|
| Float converts | 4 (one per channel + alpha) | 0 |
| `1.0 - α` subtractions | 1 per layer | 0 |
| Multiplications | 8 floats per layer | 5 u32 per layer (3 RGB + opacity + new_t) |
| Layers executed | always N | stops at first opaque dst |
| Repacks | 1 per layer | 0 (buffer carries packed state) |
| Early-out | impossible | `if dst < 0x01000000` — one CMP |

In a real UI the frontmost opaque surface is typically encountered at layer 1 or 2. Chrome, buttons, and panels are overwhelmingly opaque. The common case pays a few u32 multiplications for the semi-transparent layers above the first opaque one, then stops — and not for the entire layer, just for the pixels that haven't already become opaque from any topmost paint.

### 4. Why transparency-convention alpha (`0 = opaque`)

Every existing API uses opacity convention: `α = 255` means fully opaque. fluor inverts this for its internal representation.

The reason is exactness, and it matters in integer arithmetic.

With opacity convention and a `u8` accumulator, the ceiling is exact: `opacity = 255` contributes 100% exactly. But the floor is not: `opacity = 0` contributes `0/256`, which rounds to zero but is not mathematically zero. Invisible layers are slightly inexact.

With transparency convention and a `u8` accumulator, the floor is exact: `transparency = 0` sets `new_t = (dst_t * 0) >> 8 == 0` exactly — no rounding. The buffer's t-byte hits zero cleanly and the early-out fires precisely.

The ceiling is slightly inexact: a partial layer with `t = 255` attenuates the budget by `255/256 ≈ 0.996` rather than exactly 1.0. A layer with `t = 255` everywhere is invisible and culled before entering the blend pass; the imprecision never executes on a meaningful path.

A second exactness win comes from the empty-buffer value. The canonical empty pixel is `0xFFFFFFFF` (`t=255`, `RGB=255`): a single byte pattern that fills via one `memset(0xFF)` instruction, *and* the white RGB compensates the `>>8` truncation at the transparent endpoint. Painting opaque `mr=255` content into an `0xFFFFFFFF` empty buffer:

```
nr = (255 * top_opacity + 255 * contrib) >> 8
   = (255 * 1 + 255 * 255) >> 8
   = 65280 >> 8 = 255   // exact
```

If the empty buffer were `0xFF000000` (`RGB=0`), the same paint lands at `253` — a 1-2 LSB drift per channel at every transparent pixel touched once. The white-empty trick costs nothing (transparent pixels never display their RGB) and removes the drift.

The background layer closes the guarantee. The background is always fully opaque (`t = 0`), so the buffer's t-byte hits zero exactly on every pixel. There is no path to a pixel that escapes without being fully accounted for.

The external boundary — PNG load, image decode, glyph coverage from cosmic-text — performs a one-time `255 - a` flip on import. The present pass flips `t → α` once before submitting to wgpu / softbuffer. The cost is paid at the edges, never per pixel per frame inside the blend kernel.

Internally, all variables are named `t` or `transparency`, never `alpha` or `a`. The convention is stated explicitly wherever it appears in the codebase, and the `Blend::under` trait is the only path that composites layers — there is no painter's-algorithm fallback.

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
| Paint primitives | ✓ Every primitive routes through `Blend::under` (no painter's algorithm anywhere); fill_rect (solid + blend), stroke_rect, circle_filled, glyph rasterizers, background noise; `Clip` + `AlphaMask` + `Transform` types; `quantize_rotation` / `snap_rotation` helpers |
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
├── pixel       — Argb8 (= u32, 0xttRRGGBB t-convention); Blend trait with the single
│                 `under(self, bottom, mode)` kernel; BlendMode { Normal, Multiply, Screen,
│                  Add, Subtract, Overlay, Darken, Lighten }
├── stack       — StackCompositor + Op::{Push, Constant, Under(BlendMode)}
│                 (Stack Notation evaluator with snapshot-based partial re-eval)
├── group       — Group { region, rpn: StackCompositor, blend: BlendMode, hitmask, text_clip };
│                 flatten_into walks the bbox calling dst.under(src, blend) per pixel
├── paint       — flatten(dst, src, mode); fill_rect_solid, fill_rect_blend, stroke_rect,
│                 circle_filled, glyph::*, background_noise; Clip / AlphaMask / Transform;
│                 quantize_rotation + snap_rotation. ALL primitives route through Blend::under —
│                 no painter's algorithm path exists.
├── pane        — Pane, PaneId, Compositor (tree + hit-test + focus + z-order + render);
│                 render iterates topmost-first; under-chain handles z-order via dst-opaque early-out
├── text        — TextRenderer (cosmic-text + swash); transform-aware glyph
│                 rasterization via swash::scale; per-glyph LRU image cache
├── theme       — color constants (Android byte-swap behind cfg)
└── host/
    ├── chrome  — draw_window_controls, draw_window_edges_and_mask,
    │             draw_button_hairlines, draw_button_hover_by_pixels,
    │             get_resize_edge, hit_test_map
    └── desktop — winit + softbuffer host (feature `host-winit`, default;
                  present buffer initialized to 0xFFFFFFFF each frame; Groups flattened
                  topmost-first; std::process::exit(0) on close for Killswitch compliance)
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