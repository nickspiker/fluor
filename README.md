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

### 3. Front-to-back compositing — buffer-as-additive-accumulator, single `under` kernel

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

fluor inverts the direction and uses the buffer itself as a dual additive accumulator. The buffer's α-byte (top byte) accumulates opacity from 0 toward 0xFF. The RGB bytes accumulate darkness from 0 toward 0xFF (where darkness is the bitwise complement of visible RGB). Both halves of the pixel start at 0 and add up. Layers paint topmost-first via one binary operation:

```rust
// α + darkness convention: α=0 transparent, α=0xFF opaque; RGB stores darkness (0=white, 255=black).
// Empty pixel = 0x00000000. Argb8 is a type alias for u32.
trait Blend {
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8;
}

impl Blend for Argb8 {
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8 {
        if self >= 0xFF000000 { return self; }      // dst opaque (α==0xFF): single CMP early-out
        let top_a = self >> 24;
        let bot_a = bottom >> 24;
        let consumed = ((256 - top_a) * bot_a) >> 8;

        let (mr, mg, mb) = mode.kernel(top_dark, bot_dark);   // pure channel function, in darkness space

        let nr = top_dark_r + ((mr * consumed) >> 8);   // additive, bounded ≤ 255 by floor proof
        let ng = top_dark_g + ((mg * consumed) >> 8);
        let nb = top_dark_b + ((mb * consumed) >> 8);
        let na = top_a + consumed;                       // additive, bounded ≤ 255 by floor proof
        (na << 24) | (nr << 16) | (ng << 8) | nb
    }
}
```

No floats. No `/ 255`. All `>> 8`. Plain `+`, no `saturating_add` — the invariant `dark ≤ α ≤ 255` is preserved inductively by integer floor (`floor(k × 255 / 256) ≤ k − 1` strictly), so neither half can ever overflow a u8. Every blend mode (`Normal`, `Multiply`, `Screen`, `Add`, `Subtract`, `Overlay`, `Darken`, `Lighten`) goes through this same outer shape — only `(mr, mg, mb)` changes.

The kernel saves three subtractions per pixel per Under call versus a visible-RGB convention: there is no `255 − bot_R` anywhere in the apply step. The dominant blend mode (`Normal`, which is what 99% of compositing calls use in a UI) gets one mul + one shift + one add per channel.

#### Why store darkness instead of visible RGB

Storing darkness (the bitwise complement of visible RGB) is what makes the Under accumulator purely additive. In visible-RGB storage, layering content darkens the running pixel — you'd subtract. In darkness storage, layering content adds darkness to the running total. Both halves of the pixel work the same way: start at 0, accumulate up, saturate at the top.

At the OS boundary, a single `pixel ^= 0x00FFFFFF` flips the RGB bytes back to visible. α passes through (already opacity-direction in storage). That XOR is one instruction; it folds into the existing clip-mask multiply + Linux premultiply step at no measurable cost.

#### Why empty = `0x00000000`

The canonical empty pixel is all zeros. `vec![0u32; n]` uses `calloc` → zero pages → genuinely free initialization on every modern OS. `buf.fill(0)` is the fastest possible re-clear. Empty is the natural zero element of the Under accumulator: top_α=0 + bot contribution = bot contribution, top_dark=0 + contrib = contrib. No special-case math needed.

#### How a frame composes

The present buffer is initialized to `0` (calloc-free, every pixel = α=0, darkness=0). Groups flatten into it topmost-first. Each `flatten_into` walks the bbox and calls `dst[i].under(src[i], mode)` per pixel. When `dst.α` saturates at `0xFF` at a pixel, every subsequent Group short-circuits on that pixel via the `dst >= 0xFF000000` early-out — *one u32 compare against an immediate*, no shift, no mask.

#### A concrete example

Four layers: tooltip over button over panel over background. Tooltip α=75 (≈30% opaque), button fully opaque, panel α=195, background fully opaque.

```
layer 0  tooltip     α=75   buffer.α=0   (empty)  → consumed=75   buffer.α→75
layer 1  button      α=255  buffer.α=75           → consumed=180  buffer.α→255  (now opaque)
layer 2  panel       α=195  buffer.α==255         → EARLY-OUT, src not read
layer 3  background  α=255  buffer.α==255         → EARLY-OUT, src not read
```

After layer 1 the buffer's α-byte saturates at 255 at this pixel. Layers 2 and 3 are short-circuited — the lower bbox kernel calls visit those pixels but the very first instruction (`if self >= 0xFF000000`) returns immediately. Not skipped at the SIMD level, not zeroed out, simply never decoded past the early-out branch.

#### Cost comparison per pixel per layer

| | Bottom-up float | Front-to-back u8 (`under`) |
|---|---|---|
| Float converts | 4 (one per channel + alpha) | 0 |
| `1.0 − α` subtractions | 1 per layer | 1 per layer (`256 − top_α` only) |
| Apply-step subtractions (per channel) | 1 (mul by `1 − α`) | 0 (pure add) |
| Multiplications | 8 floats per layer | 5 u32 per layer (3 RGB + opacity + α) |
| Layers executed | always N | stops at first opaque dst |
| Repacks | 1 per layer | 0 (buffer carries packed state) |
| Buffer initialization | `memset` or per-frame clear | `calloc` — free zero pages |
| OS-boundary cost | per-pixel pack + premult | one XOR + clip + premult, single pass |
| Early-out | impossible | `if dst >= 0xFF000000` — one CMP |

In a real UI the frontmost opaque surface is typically encountered at layer 1 or 2. Chrome, buttons, and panels are overwhelmingly opaque. The common case pays a few u32 multiplications for the semi-transparent layers above the first opaque one, then stops — and not for the entire layer, just for the pixels that haven't already saturated from any topmost paint.

### 4. Why the convention saturates at all zeros and all ones

Every existing API uses one of two endpoint conventions in isolation: α with opacity (255=opaque, 0=transparent) or "premultiplied" α (RGB pre-scaled by α). fluor uses α with opacity AND stores RGB as darkness. The combined empty pixel is `0x00000000` and the combined fully-opaque-black pixel is `0xFFFFFFFF`. Both endpoints are bitwise-uniform; the convention has a clean zero element AND a clean saturation point.

The reason is mathematical symmetry. Both halves of the pixel accumulate in the same direction:

| Byte | Meaning | Empty | Saturated | Direction |
|---|---|---|---|---|
| α (top) | opacity | 0 | 0xFF | adds up |
| R/G/B | darkness | 0 | 0xFF | adds up |

The Under kernel does one thing: add the new layer's contribution to the running total. There is no asymmetric handling between α and RGB — the same `top + consumed`-style update applies to all four bytes. Initialization is free (`calloc` gives zero pages without ever touching them). Saturation is detected by a single u32 compare (`dst >= 0xFF000000`). The OS boundary is a single bit flip (`pixel ^= 0x00FFFFFF`).

This combination — α-direction matches industry standard, RGB-direction is darkness so the apply step is pure addition, empty marker is zero-init-friendly, saturation marker is a single immediate-compare — is the unique convention that makes all five properties true simultaneously. Visible-RGB storage forces subtractive math. Transparency-direction α (255=transparent) forces a non-zero empty marker. Premultiplied RGB breaks per-mode blend semantics for partial-α content.

#### Overflow safety without saturation

The Under formula uses plain integer `+` without `saturating_add` or `.min(255)`. The safety comes from integer floor arithmetic:

With `top_α ∈ [0, 254]` on the math path (255 hits the early-out), let `k = 256 − top_α ∈ [2, 256]`. Then `consumed = floor(k × bot_α / 256) ≤ floor(k × 255 / 256) = k − 1 = 255 − top_α`. So `new_α = top_α + consumed ≤ 255`. Never overflows.

The invariant `dark ≤ α` is preserved inductively: empty pixel `(0, 0)` satisfies it; the inductive step shows `new_dark − new_α ≤ contrib − consumed ≤ 0` by the same floor argument. Therefore `new_dark ≤ new_α ≤ 255` strictly — both halves bounded, plain `+` is safe.

#### Internal naming

Inside the codebase, the α-byte is named `α` or `alpha` (industry-standard direction). The RGB bytes are named `dark_r` / `dark_g` / `dark_b` where the distinction matters; elsewhere they're just `r`/`g`/`b` since the storage convention is documented at the type level. Theme colour constants are written as human-readable visible RGB (`0x00_44_41_37` for a warm gray); a compile-time `dark()` helper inverts to stored darkness and sets α=0xFF, so call sites read the colour they expect. The `Blend::under` trait is the only path that composites layers — there is no painter's-algorithm fallback anywhere in the rendering pipeline.

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
| Window chrome | ✓ controls strip, edges-and-mask, hairlines, hover overlay; always-visible at minimum window size via `ceil(span/32)` span-relative formula |
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
├── pixel       — Argb8 (= u32, 0xααRRGGBB α + darkness convention); Blend trait with the
│                 single `under(self, bottom, mode)` kernel; BlendMode { Normal, Multiply,
│                  Screen, Add, Subtract, Overlay, Darken, Lighten }
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
├── theme       — colour constants (visible-RGB literals; `dark()` helper inverts to stored
│                 darkness + α=0xFF at compile time; Android byte-swap behind cfg)
└── host/
    ├── chrome  — draw_window_edges_and_mask, draw_strip_curves, draw_strip_hairlines,
    │             draw_strip_bg, draw_minimize_symbol, draw_maximize_symbol,
    │             draw_close_symbol, get_resize_edge, hit_test_map
    └── desktop — winit + softbuffer host (feature `host-winit`, default;
                  present buffer initialized to 0x00000000 each frame (calloc-free);
                  Groups flattened topmost-first; finalize_for_os does darkness→visible
                  XOR + clip + premult in a single pass at the OS boundary;
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

Pixel convention: `0xααRRGGBB` — α-byte is opacity (industry-standard direction, `α=0xFF` opaque), RGB bytes are darkness (bitwise complement of visible RGB). Empty pixel = `0x00000000`. The OS boundary applies a single `pixel ^= 0x00FFFFFF` to flip darkness back to visible RGB. Theme colour constants are written as human-readable visible RGB (the compile-time `dark()` helper inverts them at definition time). The `Blend::under` trait is the only path that composites layers; no painter's-algorithm fallback exists anywhere.

---

## License

MIT OR Apache-2.0, at your option.

## Author

Nick Spiker — `<fractaldecoder@proton.me>`