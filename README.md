# fluor

**CPU softbuffer GUI compositor. Front to back rendering. α + darkness pixel format. Harmonic mean relative units. ARM viable down to no_std bare metal.**

![Liquid stone](liquid-stone.webp)

fluor draws windows in the natural order. Front to back, like stacking sheets of glass. The top sheet (a button, a menu, a tooltip) lands on the buffer first. When it covers a pixel completely, the buffer's job at that pixel is finished. Nothing behind it gets drawn.

Every other GUI compositor paints back to front. Background, then content, then chrome, then overlays. Most of that paint gets covered up by the next pass. It works. It's also mostly busywork. On a desktop you don't notice. On a watch, a phone, or anything battery bound, it's the difference between fluid and stuttering.

The reason nobody else renders top down comes down to encoding. Compositing is filter stacking. Stack tinted glass, stack slide-film dye layers, stack window surfaces — densities add, opacities accumulate, nothing subtracts. The Porter-Duff `out = src · α + dst · (1 − α)` formula in every textbook is not what compositing physically does; the `1 − α` term is a coordinate-change artifact, the cost of running an additive operator thru brightness storage. Store the complement and the operator collapses to addition. fluor calls this **α + darkness** storage. Opacity in the top byte, industry standard direction, 255 means opaque. Darkness in the bottom three, the bitwise complement of visible RGB. Both halves accumulate from zero. Empty pixel is `0x00000000`. Saturated black is `0xFFFFFFFF`. The blend kernel is pure addition. The done check is `if dst >= 0xFF000000`, one CPU compare against an immediate. Brightness is recovered once at the panel boundary as a single bit flip. One encoding choice, the whole pipeline simplifies. No GPU needed. The scalar CPU path hits roughly 500 fps fullscreen at 4K.

The coordinate system is the second move. Layout uses harmonic mean relative units (`2wh / (w + h)`) centered on the viewport origin. No pixel value appears anywhere in user code. The same layout code is provably correct on a watch and an 8K monitor. Frame independent the way physics equations became frame independent after relativity. You don't write responsive code. There is nothing to respond to.

The four sections below trace those moves and the two consequences (CPU softbuffer rendering, multi app system compositor scaling). Same engine drives a desktop demo today, an ARM bare metal display next, and, at scale, a system compositor with privacy enforced by the protocol itself rather than by trust.

---

## What fluor does differently

Four design choices. Each has a concrete reason. They compose into a single engine.

### 1. Center origin coordinates

Every mainstream layout system puts the origin at the top left corner of the viewport. fluor puts it at the center.

The consequence stays invisible until you do transforms. Zoom and rotate around origin require no offset bookkeeping. The math just works. A pane placed at `(0, 0)` stays centered when the window resizes. The center of the screen is always `(0, 0)`. Sign carries spatial meaning. `(-0.3, +0.1)` reads as "left of center, slightly below" at a glance. Polar layouts (gauges, radial menus, knobs) need no coordinate conversion. The y axis points down, matching text engines, image scanlines, and pixel memory order, so no flip points exist below the layout layer.

Top left origin is a historical artifact of raster scanlines. For a layout system it is a wrong default.

### 2. Harmonic mean relative units

Every mainstream layout system defines a "device independent pixel" as a scaled physical pixel. Android `dp`, CSS `px`, WPF DIPs, iOS `pt`, Flutter `dp`. Even CSS `vmin`, the closest mainstream thing to a viewport relative unit, uses `min(w, h)`, which has a discontinuity. On a portrait display one axis drives the unit. Rotate 90° and the other axis drives it, causing a jump.

fluor uses the harmonic mean of width and height. `span = 2wh / (w + h)`.

The harmonic mean is the unique scaling base with:
- Smooth derivative at `w == h`, no discontinuity at the diagonal.
- Finite slope at the axes, no singularities on degenerate layouts.
- Slope exactly 1 along the diagonal, so the unit scales 1:1 with both dimensions on a square display.
- Natural bias toward the smaller dimension on non square displays. The layout responds to what's scarce, not what's abundant.

A layout specified in RU (relative units) looks correct on an 11" laptop, a 32" monitor, and an embedded display, with no DPI awareness code anywhere. Center origin plus harmonic mean unit as the dominant convention, rather than as an opt in escape hatch, sits in territory no other compositor or GUI toolkit currently occupies.

### 3. Front to back compositing. Buffer as additive accumulator. Single `under` kernel.

Conventional compositing renders back to front (painter's algorithm). Each layer blends onto the accumulated result beneath it:

```rust
// Standard Porter-Duff "over", bottom up, float path.
let a = (src >> 24) as f32 / 255.0;   // float convert plus divide
let r = (src >> 16 & 0xFF) as f32;
let dr = (dst >> 16 & 0xFF) as f32;
let out_r = dr * (1.0 - a) + r * a;   // sub, mul, mul, add
// repeat for g, b
// repack dst
// repeat for every layer, every pixel, always
```

Two losses sit inside that formula. The `1.0 - a` and `/ 255.0` are not compositing work; they are coordinate-change overhead for running additive filter-stacking math in brightness coordinates. And bottom-up traversal forecloses any early out: in a UI where the frontmost layers are overwhelmingly opaque (buttons, chrome, panels), full blend work runs on pixels that will get completely covered. Pure waste.

fluor flips both choices. Direction goes front to back so an opaque destination short-circuits every layer behind it. Storage moves to the complement for colour — darkness instead of brightness — so the kernel runs the operator the math actually wants, which is addition. The buffer is a dual additive accumulator: α byte and RGB bytes both start at 0 and walk toward 0xFF. Layers paint topmost first via one binary operation:

```rust
// α + darkness convention. α=0 transparent, α=0xFF opaque.
// RGB stores darkness (0 means white, 255 means black).
// Empty pixel = 0x00000000. Argb8 is a type alias for u32.
trait Blend {
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8;
}

impl Blend for Argb8 {
    fn under(self, bottom: Argb8, mode: BlendMode) -> Argb8 {
        if self >= 0xFF000000 { return self; }      // dst opaque (α==0xFF): single CMP early out
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

No floats. No `/ 255`. All `>> 8`. Plain `+`, no `saturating_add`. The invariant `dark ≤ α ≤ 255` survives inductively under integer floor (`floor(k × 255 / 256) ≤ k − 1` strictly), so neither half overflows a u8. Every blend mode (`Normal`, `Multiply`, `Screen`, `Add`, `Subtract`, `Overlay`, `Darken`, `Lighten`) shares this outer shape. Only `(mr, mg, mb)` changes between modes.

The kernel saves three subtractions per pixel per Under call against a visible RGB convention. No `255 − bot_R` appears anywhere in the apply step. The dominant blend mode (`Normal`, what 99% of compositing calls use in a UI) costs one mul, one shift, one add per channel.

#### Why store darkness instead of visible RGB

Compositing accumulates absorption. Each layer in the stack absorbs some of the light coming up from behind and lets the rest thru. The running total at a pixel is the accumulated absorption — how dark the stack has gotten so far. That total is what the operator manipulates at every blend.

Brightness storage is a representation mismatch with that operator. The accumulator wants to grow toward darkness, but the bytes hold the brightness complement, so every layer's contribution gets converted (`255 − x` for colour, `1 − α` for opacity) before it can be added. The familiar `out = src · α + dst · (1 − α)` over formula is exactly that conversion sitting in the inner loop: the `1 − α` term is the price of running an absorption-domain accumulator thru brightness-domain storage. Every `255 − bot_R` in a textbook blend mode is the same tax on the colour channels.

Store darkness directly and the conversion disappears. The bytes already hold what the operator needs. Layering becomes a pure add, both halves of the pixel work the same way (start at 0, accumulate up, saturate at 0xFF), and the kernel runs one operator across all four bytes with no asymmetric handling between α and RGB. Brightness is recovered exactly once at the panel boundary as a single bit flip, folded into the existing clip mask multiply plus Linux premultiply step at no measurable cost. The kernel never sees brightness.

#### Slide film does the same thing

Each emulsion layer in a slide stores dye density. Yellow, magenta, cyan, additive in the log domain. Empty film is clear. Maximum density is opaque black. The projector lamp passes thru the stacked dyes and the result is the visible image. fluor's α + darkness convention is the digital version of the same encoding, with the panel-boundary bit flip standing in for the projector lamp. The compositing early out exists in slide film too, enforced by physics. Once dye density saturates, no light reaches the layers behind, so those dyes contribute nothing regardless of how much is there. fluor's `dst >= 0xFF000000` short circuit is that physics turned into a CPU compare. Negative plus paper printing and painter's algorithm compositing both lose this property for the same reason: both invert the encoding somewhere along the way and the additive accumulator goes with it.

#### Why empty equals `0x00000000`

The canonical empty pixel is all zeros. `vec![0u32; n]` uses `calloc` and gets zero pages, genuinely free initialization on every modern OS. `buf.fill(0)` is the fastest possible re-clear. Empty is the natural zero element of the Under accumulator. `top_α=0` plus `bot` contribution equals `bot` contribution. `top_dark=0` plus contrib equals contrib. No special case math.

#### How a frame composes

The present buffer starts at `0` every frame (calloc free, every pixel is `α=0`, darkness=0). Groups flatten into it topmost first. Each `flatten_into` walks the bbox and calls `dst[i].under(src[i], mode)` per pixel. When `dst.α` saturates at `0xFF` at a pixel, every subsequent Group short circuits on that pixel via the `dst >= 0xFF000000` early out. One u32 compare against an immediate. No shift, no mask.

#### A concrete example

Four layers: tooltip over button over panel over background. Tooltip α=75 (about 30% opaque), button fully opaque, panel α=195, background fully opaque.

```
layer 0  tooltip     α=75   buffer.α=0   (empty)  → consumed=75   buffer.α→75
layer 1  button      α=255  buffer.α=75           → consumed=180  buffer.α→255  (now opaque)
layer 2  panel       α=195  buffer.α==255         → EARLY OUT, src not read
layer 3  background  α=255  buffer.α==255         → EARLY OUT, src not read
```

After layer 1 the buffer's α byte saturates at 255 at this pixel. Layers 2 and 3 short circuit. The lower bbox kernel calls visit those pixels, but the very first instruction (`if self >= 0xFF000000`) returns immediately. Not skipped at the SIMD level, not zeroed out, simply never decoded past the early out branch.

#### Cost comparison per pixel per layer

| | Bottom up float | Front to back u8 (`under`) |
|---|---|---|
| Float converts | 4 (one per channel plus alpha) | 0 |
| `1.0 − α` subtractions | 1 per layer | 1 per layer (`256 − top_α` only) |
| Apply step subtractions (per channel) | 1 (mul by `1 − α`) | 0 (pure add) |
| Multiplications | 8 floats per layer | 5 u32 per layer (3 RGB plus opacity plus α) |
| Layers executed | always N | stops at first opaque dst |
| Repacks | 1 per layer | 0 (buffer carries packed state) |
| Buffer initialization | `memset` or per frame clear | `calloc`, free zero pages |
| OS boundary cost | per pixel pack plus premult | one XOR plus clip plus premult, single pass |
| Early out | impossible | `if dst >= 0xFF000000`, one CMP |

In a real UI the frontmost opaque surface usually shows up at layer 1 or 2. Chrome, buttons, panels stay overwhelmingly opaque. The common case pays a few u32 multiplications for the semi transparent layers above the first opaque one, then stops. And not for the entire layer, just for the pixels that haven't already saturated from any topmost paint.

### 4. Why the convention saturates at all zeros and all ones

Every existing API uses one of two endpoint conventions in isolation. α with opacity (255 means opaque, 0 means transparent), or "premultiplied" α (RGB pre scaled by α). fluor uses α with opacity AND stores RGB as darkness. The combined empty pixel is `0x00000000`. The combined fully opaque black pixel is `0xFFFFFFFF`. Both endpoints are bitwise uniform. The convention has a clean zero element AND a clean saturation point.

The reason is mathematical symmetry. Both halves of the pixel accumulate in the same direction:

| Byte | Meaning | Empty | Saturated | Direction |
|---|---|---|---|---|
| α (top) | opacity | 0 | 0xFF | adds up |
| R/G/B | darkness | 0 | 0xFF | adds up |

The Under kernel does one thing. Add the new layer's contribution to the running total. No asymmetric handling between α and RGB. The same `top + consumed` style update applies to all four bytes. Initialization is free (`calloc` gives zero pages without ever touching them). Saturation detection is a single u32 compare (`dst >= 0xFF000000`). The OS boundary is a single bit flip.

This combination (α direction matches industry standard, RGB direction is darkness so the apply step is pure addition, empty marker is zero init friendly, saturation marker is a single immediate compare) is the unique convention that makes all five properties true at once. Visible RGB storage forces subtractive math. Transparency direction α (255 means transparent) forces a non-zero empty marker. Premultiplied RGB breaks per mode blend semantics for partial α content.

#### Overflow safety without saturation

The Under formula uses plain integer `+` without `saturating_add` or `.min(255)`. The safety comes from integer floor arithmetic.

With `top_α ∈ [0, 254]` on the math path (255 hits the early out), let `k = 256 − top_α ∈ [2, 256]`. Then `consumed = floor(k × bot_α / 256) ≤ floor(k × 255 / 256) = k − 1 = 255 − top_α`. So `new_α = top_α + consumed ≤ 255`. No overflow possible.

The invariant `dark ≤ α` survives inductively. Empty pixel `(0, 0)` satisfies it. The inductive step shows `new_dark − new_α ≤ contrib − consumed ≤ 0` by the same floor argument. So `new_dark ≤ new_α ≤ 255` strictly. Both halves bounded. Plain `+` is safe.

#### Internal naming

Inside the codebase the α byte is named `α` or `alpha` (industry standard direction). The RGB bytes are named `dark_r`, `dark_g`, `dark_b` where the distinction matters. Elsewhere they're just `r`, `g`, `b` since the storage convention is documented at the type level. Theme colour constants are written as human readable visible RGB (`0x00_44_41_37` for a warm gray). A compile time `dark()` helper inverts to stored darkness and sets α=0xFF, so call sites read the colour they expect. The `Blend::under` trait is the only path that composites layers. No painter's algorithm fallback exists anywhere in the rendering pipeline.

### 5. Continuity everywhere, because humans feel it

Every curve in fluor — squircle corner, AA hairline, glow falloff, opacity gradient — is derived from an analytical function that's `C¹` continuous (or smoother) at every join, at every precision. Layout interpolations use tanh, logistic, or algebraic sigmoid (`x / √(1 + x²)`), never `clamp(_, 0, 1)` followed by a polynomial that's only valid inside the clamped domain. Easings have matching first derivatives at their endpoints. No piecewise approximations, no circular arcs masquerading as squircles, no gaussian blurs faking analytical falloffs.

The reason isn't mathematical purity. It's that humans *feel* derivative discontinuities. Conscious vision tops out near 8-bit pixel accuracy, but the visual cortex processes motion and gradients as continuous fields. A kink in a curve registers as a micro-jolt — the same way a pothole the suspension absorbed still hits your spine. You don't have to consciously see the discontinuity to recoil from it. Smooth surfaces feel like a massage; discontinuous ones feel like a slap.

Most compositors substitute approximations because they're easier to implement and look "fine". They're not visibly wrong at typical viewing distance — they just feel bad. Add up a hundred such papercuts across a UI and the whole product feels cheap, with nobody able to articulate why. Fluor pays the math cost up front so micro-jolts never accumulate. The harmonic mean unit (section 2) is the same principle applied to the scaling base; this extends it to every component possible.

---

## Why CPU softbuffer rather than GPU

The CPU rasterizer sustains roughly 500 fps fullscreen at 4K with a typical pane layout. The consumer refresh rate ceiling is 60 to 144 Hz. The headroom is so large that adding a GPU path would buy nothing for the common case while adding a driver stack failure mode, a second renderer to maintain, and a heavy dependency.

The more important reason: the same CPU rasterizer runs on bare metal targets with no GPU drivers. Specifically ferros, a no_std ARM OS. No fallback path exists because no fallback is needed. The production path is the bare metal path.

GPU support may show up if a specific workload genuinely requires it (high density vector animation, very large textures). Until then it adds complexity, not capability.

---

## Why float coordinates but u32 pixels

Layout coordinates (`RuVec2`, `Viewport`) are float. Every relevant hardware target has native float arithmetic. NEON `fadd` and `fmul`, AVX/SSE `addps` and `mulps`. The precision is more than sufficient for layout geometry at any viewport size.

The compositing pass, the inner loop that actually writes pixels, runs entirely in `u32` packed ARGB and `u16` / `u8` integer arithmetic. No floats cross into the blend path. This is intentional and load bearing. Float conversion and float multiply in the inner loop are the dominant cost in conventional compositors. fluor cuts them out.

Spirix (fluor's companion floating point arithmetic system) belongs on precision critical rasterizer paths and deterministic zoom applications, and ships as the default for the ferros build via a `spirix-coord` feature flag. It stays off by default for windowing because the software emulation overhead would erase the performance advantage of CPU rendering.

---

## The same kernel scales to a multi app system compositor

fluor's `Group` (a buffer with a dirty flag, an internal layer stack, and a cached composite) is also the right primitive for a multi app system compositor. Each app's content becomes one or several Groups in a system wide z stack. When app A changes, only A's Groups re rasterize. Every other app's cached composite stays reused. The top level Under chain stitches them together front to back with the same `dst >= 0xFF000000` early out from section 3. Opaque content in higher Groups stops the descent for those pixels, so apps below don't get walked where they can't be seen. Single kernel mechanism scaled out one level. Every property fluor relies on inside a single app holds at multi app scale.

That front to back ordering becomes a security primitive when enforced thru the protocol itself rather than thru policy or trust. A lower app cannot overpaint a higher one. The compositor silently drops paints into pixels already claimed by an opaque higher Group. Compare to X11 (no compositor enforces anything, any app can XPutImage anywhere) or Wayland (apps rigidly clipped to per window surfaces, no cross window effects possible without bolted on protocol extensions). The Group model gets both behaviours. Apps declare paint regions anywhere on screen (absolute mode, in screen RU) or within their own window (relative mode, in window local RU), and the compositor decides what's actually visible. Privacy lives in the protocol. Apps submit Groups but never receive composited results, so they never see other apps' pixels unless the user explicitly grants a read capability (screen capture, accessibility, recording).

"Windows" stop being a kernel primitive and become a shell convention. The shell might give apps a window sized Group decorated with chrome by default, but a tooltip popping past the window's edge is just a second Group at the right z slice. System tray icons, notification popovers, modal dim overlays, drag ghost images that follow the cursor across windows, all just Groups. No per case protocol extensions of the kind Wayland accumulates. Everything the legacy window primitive struggles to express because the primitive is the wrong abstraction at that level.

**This generalization is not specific to any one compositor.** Any system (wlroots based, a custom DRM/KMS scanout, a userspace replacement for KWin or Mutter, ferros's own) could adopt this model. fluor's primitives (`Group` for sub viewport caching, RU for scale invariance, front to back for the early out, one `Under` kernel for one code path) each got chosen for a different local reason. The system level architecture falls out of them. The generalization is cheap to reach from where fluor already sits. The architectural argument stays independent. Any compositor built around these primitives gets the same scaling, security, and clarity benefits regardless of what sits underneath it.

---

## Status

**v0, pre alpha.** Window chrome, pane composition, and transform aware text rendering all work end to end. Textboxes, widgets, and layout persistence still need building. Expect breaking changes at every layer until the first consumer migration validates the API.

| Layer | State |
|---|---|
| Center origin coords (`RuVec2`, `Viewport`) | ✓ float storage, harmonic mean span/perimeter/diagonal_sq |
| Pane tree (`Compositor`) | ✓ insert, remove, get, hit test, focus, z order, render |
| Paint primitives | ✓ every primitive routes thru `Blend::under` (no painter's algorithm anywhere). fill_rect (solid and blend), stroke_rect, circle_filled, glyph rasterizers, background noise. `Clip`, `AlphaMask`, `Transform` types. `quantize_rotation` and `snap_rotation` helpers |
| Window chrome | ✓ controls strip, edges and mask, hairlines, hover overlay. Always visible at minimum window size via `ceil(span/32)` span relative formula |
| Drag and resize | ✓ self driven loop on every platform. fluor owns input, computes target geometry, pushes via `request_inner_size` and `set_outer_position`. Paints every vsync at the OS confirmed size. Linux X11 uses one atomic `XConfigureWindow` (via `x11rb`) when both size and position change so the WM applies them together. Unified across Linux, Windows, macOS. No WM `drag_resize_window` path, no macOS NSEvent polling hack. WM enforced `min_inner_size = (24, 8)` |
| Text rendering | ✓ cosmic-text plus swash. Open Sans bundled. Transform aware (arbitrary rotation, skew, scale via `swash::scale`). Per glyph LRU cache keyed on `(font, glyph, size, transform)` |
| Killswitch close | ✓ `std::process::exit(0)` on close and `CloseRequested`. No Drop chain, kernel reclaims everything |
| Textbox and widgets | ✗ planned |
| Layout persistence (VSF) | ✗ planned. 1 Hz, release debounce |
| `host-bare` (ferros, no_std framebuffer) | ✗ planned |
| SIMD blit kernels (NEON, SSE2) | ✗ deferred. Scalar path already hits ~500 fps fullscreen at 4K |

---

## Quick example

```rust
use fluor::coord::Coord;
use fluor::host::app::{Context, FluorApp, run_app};
use winit::window::CursorIcon;

struct Hello;

impl FluorApp for Hello {
    fn render(&mut self, _target: &mut [u32], _ctx: &mut Context) {
        // Your paint code lives here — fluor hands you a viewport-sized α + darkness scratch
        // and a Context with the viewport, font cache, clip mask, and damage accumulator.
        // The host runs finalize → shadow → OS handoff after this returns.
    }
    fn cursor_for(&self, _x: Coord, _y: Coord, _ctx: &Context) -> CursorIcon {
        CursorIcon::Default
    }
}

fn main() {
    run_app(Hello).expect("event loop");
}
```

That's the floor — an empty window with chrome the host owns. For the full pattern (chrome wiring, multi-widget Container, focus arbitration, Tab cycling, overlay tints) read [`examples/panes.rs`](examples/panes.rs) and run it with `cargo run --example panes`.

---

## Architecture

```
fluor (lib)
├── coord       : RuVec2, Coord (= f32)
├── geom        : Viewport with span/perimeter/diagonal_sq + RU↔pixel
├── pixel       : Argb8 (= u32, 0xααRRGGBB α + darkness convention). Blend trait with the
│                 single `under(self, bottom, mode)` kernel. BlendMode { Normal, Multiply,
│                 Screen, Add, Subtract, Overlay, Darken, Lighten }
├── stack       : StackCompositor + Op::{Push, Constant, Under(BlendMode)}
│                 (Stack Notation evaluator with snapshot based partial re-eval)
├── group       : Group { region, rpn: StackCompositor, blend: BlendMode, hitmask, text_clip }.
│                 flatten_into walks the bbox calling dst.under(src, blend) per pixel
├── paint       : flatten(dst, src, mode). fill_rect_solid, fill_rect_blend, stroke_rect,
│                 circle_filled, glyph::*, background_noise. Clip / AlphaMask / Transform.
│                 quantize_rotation + snap_rotation. ALL primitives route thru Blend::under.
│                 No painter's algorithm path exists.
├── pane        : Pane, PaneId, Compositor (tree + hit-test + focus + z-order + render).
│                 Render iterates topmost first. Under chain handles z-order via dst-opaque early out
├── text        : TextRenderer (cosmic-text + swash). Transform aware glyph
│                 rasterization via swash::scale. Per glyph LRU image cache
├── theme       : colour constants (visible-RGB literals. `dark()` helper inverts to stored
│                 darkness + α=0xFF at compile time. Android byte swap behind cfg)
└── host/
    ├── chrome         : draw_window_edges_and_mask, draw_strip_curves, draw_strip_hairlines,
    │                    draw_strip_bg, draw_minimize_symbol, draw_maximize_symbol,
    │                    draw_close_symbol, get_resize_edge — low-level chrome primitives
    ├── chrome_widget  : DefaultChrome { Group, hit_test_map, four ChromeButton widgets }.
    │                    Container impl walks the buttons; hover_colour_for / owns_hit query
    │                    the live ids without exposing constants
    ├── widget         : Widget + Click / Key / Focus / Hover capability traits. HitId u16
    │                    dense allocator (`next_id`). linear_tab_next + apply_focus_change
    │                    + build_overlay_deltas helpers. PaintCtx wraps the per-frame canvas
    ├── icon           : Icon (vsfimg-decoded raster source for chrome's app-icon orb)
    ├── os_input       : XSettings polling for OS double-click interval (Linux/X11)
    └── app            : FluorApp trait + run_app entry point (feature `host-winit`).
                         winit + softbuffer event loop. finalize_for_os does darkness→visible
                         XOR + clip + premult in a single pass at the OS boundary. Self-driven
                         resize / drag-move / maximize. Linux X11 input-region shape via x11rb.
                         std::process::exit(0) on close for Killswitch compliance.
```

Future: `host-bare` (no_std framebuffer for ferros), textbox + widget kit, SIMD kernels, layout VSF persistence.

---

## Features

- `default = ["std", "host-winit", "simd"]`
- `host-winit`: winit + softbuffer desktop host (default. Transitively requires `text`)
- `host-bare`: bare metal `&mut [u32]` framebuffer host (planned, gated for `no_std`)
- `text`: cosmic-text + swash text rendering with transform support
- `simd`: runtime dispatched NEON / SSE2 / AVX2 blit kernels (planned)
- `spirix-coord`: Spirix arithmetic for precision critical paths (planned)

---

## Building

```sh
./build-development.sh   # canonical dev build
cargo run --example panes
```

`./build-development.sh` is preferred over `cargo build --release` per `AGENT.md`. Release builds only when explicitly requested.

---

## Coding rules

`AGENT.md` governs this codebase. Notable rules: no bounds checks, clamps, or saturating arithmetic without proven justification (Rule 0). Decimal indexing forbidden. VSF type marker matching, never positional. No fixed pixel values (use `span`, `perimeter`, or `diagonal_sq`). Persistence cadence on streaming UI events is ≤1 Hz with flush on release. Public API stable. Internal renderer hot swappable via enum / feature / runtime detect.

Pixel convention: `0xααRRGGBB`. α byte is opacity (industry standard direction, `α=0xFF` opaque). RGB bytes are darkness (bitwise complement of visible RGB). Empty pixel = `0x00000000`. The OS boundary applies a single `pixel ^= 0x00FFFFFF` to flip darkness back to visible RGB. Theme colour constants are written as human readable visible RGB (the compile time `dark()` helper inverts them at definition time). The `Blend::under` trait is the only path that composites layers. No painter's algorithm fallback exists anywhere.

---

## Terminology

- **α + darkness** — fluor's pixel format: opacity in the top byte, the bitwise complement of visible RGB ("darkness") in the low three. Both halves accumulate from zero, so the blend kernel is pure addition and the done-check is one compare.
- **under kernel** — the single front-to-back compositing operation: a new layer lands *under* what has already accumulated.
- **harmonic mean unit** — the relative layout unit `2wh / (w + h)`, centred on the viewport origin, so layout is resolution-independent.

fluor carries no identity vocabulary of its own; the rest of the stack's terms live in the cross-stack glossary: `GLOSSARY.md` in the ferros repo.

---

## License

MIT OR Apache-2.0, at your option.

## Author

Nick Spiker, `<fractaldecoder@proton.me>`