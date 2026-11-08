//! Pixel-buffer paint primitives. Packed layout is `0xααRRGGBB` (α-byte high, blue low) — top byte is **α (opacity)**, industry-standard direction (`α = 0` transparent, `α = 0xFF` opaque). RGB bytes store darkness (`0 = white`, `255 = black`). See [`crate::pixel`] for the locked convention. All inputs are pixel-space, not RU — convert via [`Viewport::ru_to_px`](crate::Viewport::ru_to_px) before calling.
//!
//! Internal to fluor's render pipeline. Per `## API / Implementation Separation` in AGENT.md, these are not part of the consumer-facing API: future SIMD kernels (NEON, SSE2) will dispatch thru the same entry points without changing call sites in `pane` or `Compositor`.
//!
//! Blend model is α + darkness front-to-back: `dst` is the partial composite already accumulated above (its α-byte = accumulated opacity, RGB = accumulated darkness), `src` is the new layer going behind. Per-pixel early-out fires when `dst >= 0xFF000000` (dst α saturated = opaque) via a single u32 compare. Math thruout is `>> 8` with the `(256 − top_α)` trick — never `/ 255`, never floats in the inner loop. Multi-layer composition is additive on BOTH halves (α adds, darkness adds); the buffer carries the accumulator state across Group boundaries so the early-out chain survives between flatten passes.
//!
//! Every blending primitive accepts an optional [`Clip`] (defaults to full buffer when `None`) and an optional [`AlphaMask`] (full-frame, multiplies into per-pixel alpha for soft clipping — rounded textboxes, squircle pane corners, scroll fades). The clip is resolved once at entry into `(x_min, y_min, x_max, y_max)` loop bounds, so the inner loops carry **zero per-pixel bounds checks** — the math at the entry is the proof. AlphaMask dimensions must equal the buffer's `(buf_w, buf_h)`; mismatches panic per AGENT.md "fail loud."

use alloc::vec::Vec;
use crate::canvas::Canvas;
#[cfg(feature = "text")]
use crate::text::TextStyle;
use crate::coord::Coord;

// Submodules — extracted from this file as part of the paint-organisation pass. Each is re-exported flat so existing call sites (`crate::paint::draw_squircle_pill`, etc.) continue to resolve without any caller changes.
pub mod argb8;
pub mod finalize;
pub mod glow;
pub mod shadow;
pub mod edge;
pub mod shape;
pub mod squircle;
pub use argb8::*;
pub use edge::*;
pub use finalize::*;
pub use glow::*;
pub use shadow::*;
pub use shape::*;
pub use squircle::*;

/// Clipping rectangle in buffer pixel coordinates. `x_end` and `y_end` are exclusive (matches Rust ranges). Construct via [`Clip::new`] or [`Clip::buffer`] for a full-buffer clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clip {
    pub x_start: usize,
    pub y_start: usize,
    pub x_end: usize,
    pub y_end: usize,
}

impl Clip {
    pub const fn new(x_start: usize, y_start: usize, x_end: usize, y_end: usize) -> Self {
        Self {
            x_start,
            y_start,
            x_end,
            y_end,
        }
    }

    /// Full-buffer clip. Equivalent to passing `None` to a primitive.
    pub const fn buffer(buf_w: usize, buf_h: usize) -> Self {
        Self {
            x_start: 0,
            y_start: 0,
            x_end: buf_w,
            y_end: buf_h,
        }
    }

    /// Resolve an optional clip — `None` defaults to the full buffer, and a given clip is cut to the buffer here, once, so no primitive needs its own `.min(buf_w)` / `.min(buf_h)`. Used by every primitive at entry so the rest of the function reads from a single concrete `Clip`.
    #[inline]
    pub fn resolve(opt: Option<Clip>, buf_w: usize, buf_h: usize) -> Self {
        match opt {
            // WHY/PROOF: a caller's clip is the app's damage rect, which can overhang the buffer (a widget straddling the window edge, a stale rect after a shrink) — every primitive indexes the buffer up to x_end and y_end.
            Some(c) => Self { x_start: c.x_start.min(buf_w), y_start: c.y_start.min(buf_h), x_end: c.x_end.min(buf_w), y_end: c.y_end.min(buf_h) },
            None => Self::buffer(buf_w, buf_h),
        }
    }

    /// Intersect a primitive's `i32` bbox with `opt` (resolved against the buffer extent), returning integer pixel bounds suitable for `for` loops. Returns `None` if the intersection is empty (whole primitive is offscreen or fully clipped). Used by every rasterizer's entry path so the clip story is one call: pass `clip` thru, get back either `(x_start, y_start, x_end, y_end)` to iterate or an early-return signal.
    #[inline]
    pub fn intersect_bbox(
        opt: Option<Clip>,
        buf_w: usize,
        buf_h: usize,
        x_min: i32,
        x_max: i32,
        y_min: i32,
        y_max: i32,
    ) -> Option<(usize, usize, usize, usize)> {
        let c = Self::resolve(opt, buf_w, buf_h);
        // WHY/PROOF: the bbox is the primitive's caller-placed geometry, negative or past the buffer whenever it straddles an edge — max(0) before the cast keeps a negative from wrapping to a huge usize, and the clip bounds keep the loops inside the buffer.
        let x_start = (x_min.max(0) as usize).max(c.x_start);
        let y_start = (y_min.max(0) as usize).max(c.y_start);
        let x_end = (x_max.max(0) as usize).min(c.x_end);
        let y_end = (y_max.max(0) as usize).min(c.y_end);
        if x_start >= x_end || y_start >= y_end {
            None
        } else {
            Some((x_start, y_start, x_end, y_end))
        }
    }
}

/// 2D affine transform — a 2×3 matrix laid out as `[a c tx; b d ty]`. Applied to a point `(x, y)` as `(a*x + c*y + tx, b*x + d*y + ty)`. Composes via [`then`](Self::then) (`a.then(b)` = "do `a` first, then `b`"). Used by the text path so glyph contours rotate / scale / skew *before* swash rasterizes them — proper hinting and AA on rotated glyphs, not a post-rotation pixel-shuffle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub a: Coord,
    pub b: Coord,
    pub c: Coord,
    pub d: Coord,
    pub tx: Coord,
    pub ty: Coord,
}

impl Transform {
    pub const IDENTITY: Transform = Transform {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        tx: 0.0,
        ty: 0.0,
    };

    #[inline]
    pub const fn new(a: Coord, b: Coord, c: Coord, d: Coord, tx: Coord, ty: Coord) -> Self {
        Self { a, b, c, d, tx, ty }
    }

    #[inline]
    pub fn rotate(radians: Coord) -> Self {
        let (s, co) = crate::math::sin_cos(radians);
        Self {
            a: co,
            b: s,
            c: -s,
            d: co,
            tx: 0.0,
            ty: 0.0,
        }
    }

    #[inline]
    pub fn scale(sx: Coord, sy: Coord) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: sy,
            tx: 0.0,
            ty: 0.0,
        }
    }

    #[inline]
    pub fn skew(kx: Coord, ky: Coord) -> Self {
        Self {
            a: 1.0,
            b: ky,
            c: kx,
            d: 1.0,
            tx: 0.0,
            ty: 0.0,
        }
    }

    #[inline]
    pub fn translate(tx: Coord, ty: Coord) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
            tx,
            ty,
        }
    }

    /// Compose `self` then `other` (i.e. `other ∘ self` in math notation). The result transform applies `self` to the point first, then `other` to the result.
    #[inline]
    pub fn then(self, other: Self) -> Self {
        Self {
            a: other.a * self.a + other.c * self.b,
            b: other.b * self.a + other.d * self.b,
            c: other.a * self.c + other.c * self.d,
            d: other.b * self.c + other.d * self.d,
            tx: other.a * self.tx + other.c * self.ty + other.tx,
            ty: other.b * self.tx + other.d * self.ty + other.ty,
        }
    }

    /// Apply to a point.
    #[inline]
    pub fn apply(self, x: Coord, y: Coord) -> (Coord, Coord) {
        (
            self.a * x + self.c * y + self.tx,
            self.b * x + self.d * y + self.ty,
        )
    }

    /// Bit-exact identity check. For "approximately identity" use a tolerance compare on the field deltas.
    #[inline]
    pub fn is_identity(self) -> bool {
        self.a == 1.0
            && self.b == 0.0
            && self.c == 0.0
            && self.d == 1.0
            && self.tx == 0.0
            && self.ty == 0.0
    }

    /// Axis-aligned bounding box of the transformed rectangle `[0, w] × [0, h]`. Returns `(min_x, min_y, max_x, max_y)` in transformed coordinates. Used by text rasterizers to compute the clip-clamp range for a transformed glyph.
    pub fn aabb_of_rect(self, w: Coord, h: Coord) -> (Coord, Coord, Coord, Coord) {
        let p0 = self.apply(0.0, 0.0);
        let p1 = self.apply(w, 0.0);
        let p2 = self.apply(0.0, h);
        let p3 = self.apply(w, h);
        let min_x = p0.0.min(p1.0).min(p2.0).min(p3.0);
        let max_x = p0.0.max(p1.0).max(p2.0).max(p3.0);
        let min_y = p0.1.min(p1.1).min(p2.1).min(p3.1);
        let max_y = p0.1.max(p1.1).max(p2.1).max(p3.1);
        (min_x, min_y, max_x, max_y)
    }
}

/// Quantize a continuous rotation angle (radians) into a bin index `0..N` where `N = ceil_to_multiple_of(K, ceil(2π × radius))`. The "1-pixel-arc rule": `radius = font_size_px / 2` makes one bin = one pixel of arc travel at the glyph's outer edge — the perceptual minimum step. Ceiling to a multiple of `K` (typically 8) lands cardinal + octant angles exactly on bin boundaries.
///
/// Use `.floor()` (not `.round()`) for monotonic bin assignment — `rem_euclid` handles negative angles cleanly and `.floor()` is single-uop on aarch64 NEON / x86 SSE4 ROUNDSS, so the legacy "subtract epsilon, use round" workaround is unnecessary.
pub fn quantize_rotation(radians: f32, font_size_px: f32, k: u32) -> u16 {
    let radius = font_size_px * 0.5;
    let raw_divs = crate::math::ceil(core::f32::consts::TAU * radius) as u32;
    let divs = ((raw_divs + k - 1) / k) * k;
    let theta = crate::math::rem_euclid(radians, core::f32::consts::TAU);
    let bin = crate::math::floor(theta / core::f32::consts::TAU * divs as f32) as u32;
    (bin % divs.max(1)) as u16
}

/// Snap a continuous rotation angle to the nearest quantization bin and return the bin's representative angle (in radians, range `[0, 2π)`). Same quantization grid as [`quantize_rotation`]. Use this when constructing a [`Transform`] for text that should benefit from the rasterized-glyph cache: animated rotation that varies continuously per frame would miss the cache every frame; pre-snapping to bins makes consecutive frames within the same bin cache-hit.
pub fn snap_rotation(radians: f32, font_size_px: f32, k: u32) -> f32 {
    let radius = font_size_px * 0.5;
    let raw_divs = crate::math::ceil(core::f32::consts::TAU * radius) as u32;
    let divs = ((raw_divs + k - 1) / k) * k;
    if divs == 0 {
        return 0.0;
    }
    let step = core::f32::consts::TAU / divs as f32;
    let theta = crate::math::rem_euclid(radians, core::f32::consts::TAU);
    crate::math::floor(theta / step) * step
}

/// Per-pixel alpha mask sized to the framebuffer. Multiplies into rendered alpha for soft clipping (textbox shapes, squircle pane corners, scroll fades). Carries its dimensions so primitives can panic on mismatch (per AGENT.md: init bugs fail loud, not silently render garbage).
pub struct AlphaMask<'a> {
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
}

impl<'a> AlphaMask<'a> {
    /// Construct an alpha mask. Panics if `pixels.len() != width * height`.
    pub fn new(pixels: &'a [u8], width: usize, height: usize) -> Self {
        assert_eq!(
            pixels.len(),
            width * height,
            "AlphaMask: pixels.len() ({}) != width * height ({} * {} = {})",
            pixels.len(),
            width,
            height,
            width * height,
        );
        Self {
            pixels,
            width,
            height,
        }
    }
}

/// Assert mask dimensions match buffer; panic with a descriptive message if not. Per AGENT.md a mask attached to the wrong buffer is an initialization bug — fail loud.
#[inline]
pub(crate) fn assert_mask_matches_buffer(mask: &AlphaMask, buf_w: usize, buf_h: usize) {
    assert!(
        mask.width == buf_w && mask.height == buf_h,
        "AlphaMask dimensions {}×{} don't match buffer {}×{}",
        mask.width,
        mask.height,
        buf_w,
        buf_h,
    );
}

/// Per-pixel hit-test ID type. `u16` gives 65 535 registerable interactive zones — enough for "every list row is a hit zone" patterns without revisiting. Cost vs `u8` at 4 K is +8 MB on a ~100 MB+ render-state footprint, which is well inside the engineering budget for "never have to think about it again." Lives in `paint` because every hit-stamping rasterizer is here; [`crate::host::chrome`] and [`crate::host::app::HitRegistry`] re-export and consume.
pub type HitId = u16;

/// Reserved "no widget here" ID. Always 0 — registrations start at 1 so the value cannot collide.
pub const HIT_NONE: HitId = 0;

/// Compose `src` underneath `dst` where `src` is in *pre-composed* α + darkness form (the result of a chain of `under()` writes starting from empty — so `src.dark` is already attenuated by `src.α`). Unlike [`flatten`] with `BlendMode::Normal`, this kernel scales src's contribution by `(256 − dst.α)` only, NOT by `(256 − dst.α) × src.α`, avoiding the second premult that would dim every AA pixel.
///
/// Use case: blit a cached widget layer (built by repeated `under()` from empty) onto a target buffer — every Stack layer, Group composite and chrome layer is one.
pub fn flatten_premult<P: Pixel>(dst: &mut [P], src: &[P]) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    const CHUNK: usize = 4096;
    let src = &src[..n];
    crate::par::par_chunks(&mut dst[..n], CHUNK, |off, chunk| flatten_premult_chunk(chunk, &src[off..off + chunk.len()]));
}

/// One run of [`flatten_premult`], sequential: the widgets' cache blit composes a row at a time thru it.
#[inline]
pub(crate) fn flatten_premult_chunk<P: Pixel>(dst: &mut [P], src: &[P]) {
    let n = dst.len();
    for i in 0..n {
        let d = dst[i];
        let d_a = d.alpha();
        if d_a == P::ONE {
            continue;
        }
        let s = src[i];
        let factor = P::ONE + 1 - d_a;
        let [s_r, s_g, s_b] = s.channels();
        let [d_r, d_g, d_b] = d.channels();
        let new_a = d_a + ((factor * s.alpha()) >> P::DEPTH);
        let new_r = d_r + ((factor * s_r) >> P::DEPTH);
        let new_g = d_g + ((factor * s_g) >> P::DEPTH);
        let new_b = d_b + ((factor * s_b) >> P::DEPTH);
        dst[i] = P::from_channels(new_a, [new_r, new_g, new_b]);
    }
}

/// [`flatten_premult`] in any blend mode: `src` a composite (its darkness premultiplied by its α) composed under `dst` thru [`Blend::under_premult`](crate::pixel::Blend::under_premult), the same operator as [`flatten`]'s with the bottom's α kept inside its darkness. Normal takes [`flatten_premult`]'s kernel (the same arithmetic); the other modes run per pixel with Rayon chunking, as [`flatten`]'s do.
pub fn flatten_premult_mode<P: Pixel>(dst: &mut [P], src: &[P], mode: BlendMode) {
    if mode == BlendMode::Normal {
        return flatten_premult(dst, src);
    }
    let n = dst.len();
    if n == 0 {
        return;
    }
    const CHUNK: usize = 4096;
    let src = &src[..n];
    crate::par::par_chunks(&mut dst[..n], CHUNK, |off, chunk| {
        let src_chunk = &src[off..off + chunk.len()];
        for i in 0..chunk.len() {
            chunk[i] = chunk[i].under_premult(src_chunk[i], mode);
        }
    });
}

pub use crate::pixel::BlendMode;
use crate::pixel::{Argb8, Pixel};

/// Flatten `src` underneath `dst` across the whole slice, pixel-by-pixel, via [`Blend::under`](crate::pixel::Blend::under). `dst` is the partial composite already accumulated above (its α-byte = accumulated opacity); `src` is the new layer going behind. Per-pixel early-out fires when `dst.α == 0xFF`. Both slices must be the same length (a shorter `src` panics: a size mismatch is a bug upstream, never something to paper over).
///
/// For `BlendMode::Normal` (the 99% case) the kernel runs the depth's chunk hook ([`Pixel::under_chunk_normal`]: 8-wide SIMD for Argb8) + Rayon parallelism. Other modes fall back to scalar per-pixel with Rayon row-chunking still applied — the math just stays in the scalar [`Blend::under`](crate::pixel::Blend::under) kernel.
#[inline]
pub fn flatten<P: Pixel>(dst: &mut [P], src: &[P], mode: BlendMode) {
    let n = dst.len();
    if n == 0 {
        return;
    }
    let dst = &mut dst[..n];
    let src = &src[..n];
    const CHUNK: usize = 4096;
    if mode == BlendMode::Normal {
        crate::par::par_chunks(dst, CHUNK, |off, chunk| {
            let src_chunk = &src[off..off + chunk.len()];
            P::under_chunk_normal(chunk, src_chunk);
        });
    } else {
        crate::par::par_chunks(dst, CHUNK, |off, chunk| {
            let src_chunk = &src[off..off + chunk.len()];
            for i in 0..chunk.len() {
                chunk[i] = chunk[i].under(src_chunk[i], mode);
            }
        });
    }
}

/// Intersect a caller-supplied `(x, y, w, h)` rect (in pixels, top-left origin, may be negative or extend off-clip) with a `Clip`. Returns `(x_min, y_min, x_max, y_max)` in `usize`, all guaranteed in-bounds for `pixels[y * buf_w + x]` indexing **as long as the supplied `Clip` is itself within the buffer**. Returns an empty range (x_min >= x_max or y_min >= y_max) if the rect lies entirely outside the clip.
///
/// **Rule 0 — WHY/PROOF/PREVENTS:** rect coords are external inputs (caller can pass a pane dragged off the window edge). WHY: compositor semantics demand "draw the intersection with the clip." PROOF without it: a negative `x as usize` wraps to a huge value, indexing past the pixel slice panics. PREVENTS: panic on partial-offscreen rects, which is a normal use case. The clip happens once per rect; inner loops trust the math.
#[inline]
fn clip_rect(
    clip: Clip,
    x: isize,
    y: isize,
    rect_w: isize,
    rect_h: isize,
) -> (usize, usize, usize, usize) {
    // Negative isize → huge usize after cast; .min(clip.x_end) clamps it down. .max(clip.x_start) ensures we never index before the clip's left edge.
    let x_end = x + rect_w;
    let y_end = y + rect_h;
    let x_min = if x < 0 {
        clip.x_start
    } else {
        (x as usize).clamp(clip.x_start, clip.x_end)
    };
    let y_min = if y < 0 {
        clip.y_start
    } else {
        (y as usize).clamp(clip.y_start, clip.y_end)
    };
    let x_max = if x_end < 0 {
        clip.x_start
    } else {
        (x_end as usize).clamp(clip.x_start, clip.x_end)
    };
    let y_max = if y_end < 0 {
        clip.y_start
    } else {
        (y_end as usize).clamp(clip.y_start, clip.y_end)
    };
    (x_min, y_min, x_max, y_max)
}

/// Fill a rectangle by under-blending `colour` into the buffer — the single rect paint primitive.
///
/// Per-pixel: `pixels[idx] = pixels[idx].under(effective_colour, BlendMode::Normal)`. The dst-opaque early-out fires automatically where a topmost paint has already claimed the pixel (`dst.α == 0xFF`), so the topmost-first doctrine is honored without callers having to think about z-order.
///
/// **Hairline convention:** a `0` width or height means "1 pixel thick" along that axis — pass `rect_h = 0` for a horizontal hairline, `rect_w = 0` for a vertical one, both `0` for a single-pixel dot. This lets callers draw rules/dividers without inventing a thickness; for a thicker line pass the real pixel count. (Negative dimensions stay a no-op.)
///
/// `mask: Some(&AlphaMask)` multiplies each pixel's mask alpha into `colour`'s opacity (`effective_opacity = colour_opacity * mask_alpha >> 8`) — used for shaped textbox interiors, scroll fades, etc. `mask: None` skips that math entirely.
pub fn fill_rect<P: Pixel>(
    canvas: &mut Canvas<P>,
    x: isize,
    y: isize,
    rect_w: isize,
    rect_h: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    // 0 → 1-pixel hairline along that axis (both 0 = single-pixel dot). Coerce before clipping so the rect always covers at least one pixel line.
    let rect_w = if rect_w == 0 { 1 } else { rect_w };
    let rect_h = if rect_h == 0 { 1 } else { rect_h };
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    let clip = Clip::resolve(clip, buf_w, buf_h);
    if let Some(m) = mask {
        assert_mask_matches_buffer(m, buf_w, buf_h);
    }
    let (x_min, y_min, x_max, y_max) = clip_rect(clip, x, y, rect_w, rect_h);
    canvas.damage.add_bounds(x_min, y_min, x_max, y_max);
    let colour = P::from_g2t8(colour);
    let pixels: &mut [P] = canvas.pixels;
    match mask {
        None => {
            for row in y_min..y_max {
                let base = row * buf_w;
                for col in x_min..x_max {
                    let idx = base + col;
                    pixels[idx] = pixels[idx].under(colour, BlendMode::Normal);
                }
            }
        }
        Some(m) => {
            // Mask convention matches `TextRenderer::render_buffer_u32`: `effective_α = colour_α × mask_α / 255` (opacity multiplies). Mask=0 → pixel fully clipped; mask=255 → colour passes thru at full α. The previous implementation here used `colour_opacity = 255 - α`, which inverted the semantics — for an opaque colour (α=0xFF) the mask was multiplied by zero and had no effect, so masks were silently ineffective on solid fills.
            let colour_alpha = colour.alpha();
            let colour_rgb = colour.rgb();
            for row in y_min..y_max {
                let base = row * buf_w;
                for col in x_min..x_max {
                    let idx = base + col;
                    let mask_a = m.pixels[idx] as u32;
                    // Floor `>> 8` with the +1 weight bump instead of `/ 255` (house convention, paint.rs module doc): mask 255 → +1 → ×256 passes colour_α thru EXACTLY, mask 0 → ×1 floors to 0, everything between lands within 1 LSB of the exact 255-division — and the shift is far cheaper than the division in this per-pixel loop.
                    // The `>> 8` is the mask's own depth (an AlphaMask is bytes at every pixel depth), so α keeps the pixel's.
                    let effective_alpha = (colour_alpha * (mask_a + 1)) >> 8;
                    if effective_alpha == 0 {
                        continue;
                    }
                    let masked = colour_rgb.with_alpha(effective_alpha);
                    pixels[idx] = pixels[idx].under(masked, BlendMode::Normal);
                }
            }
        }
    }
}

/// Stroke (outline) an axis-aligned rectangle. Draws four filled rect strips along the edges via [`fill_rect`]; corners are not joined separately because at 90° angles the strips meet cleanly.
pub fn stroke_rect<P: Pixel>(
    canvas: &mut Canvas<P>,
    x: isize,
    y: isize,
    rect_w: isize,
    rect_h: isize,
    stroke: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    if stroke <= 0 || rect_w <= 0 || rect_h <= 0 {
        return;
    }
    let inner_h = rect_h - 2 * stroke;
    let edges: [(isize, isize, isize, isize); 4] = [
        (x, y, rect_w, stroke),                             // top
        (x, y + rect_h - stroke, rect_w, stroke),           // bottom
        (x, y + stroke, stroke, inner_h),                   // left
        (x + rect_w - stroke, y + stroke, stroke, inner_h), // right
    ];
    // Damage accumulates via the four fill_rect calls, each reporting its own clipped bbox.
    for &(ex, ey, ew, eh) in &edges {
        fill_rect(canvas, ex, ey, ew, eh, colour, clip, mask);
    }
}

/// Fill the buffer with photon's signature procedural background — symmetric organic noise. Rows are RNG-independent (each row reseeds from `logical_row`), so the outer loop parallelizes cleanly via [`crate::par::par_rows`]. Mirrored left/right halves like photon. Set `fullscreen=true` to fill the whole buffer; `false` leaves a 1px border for the window edge stroke. `shimmer` is mixed into each row's starting colour so animating it cycles the colour bias across rows without changing the noise topology; `scroll_offset` shifts the texture vertically (for content scroll integration). The speckle gate (the rare bright/dim flash branch inside each row) fires at a constant 1/256 rate independent of `shimmer`.
///
/// SIMD inside a row is intentionally not done — the per-pixel RNG (`rng ^= rng.rotate_left(13).wrapping_add(const)`) is a serial dependency chain; vectorizing it would require N independent RNG streams per lane and would change photon's visual pattern. If profiling shows the per-row scalar work still dominating after Rayon, that's the next lever.
///
/// Clip restricts the row range. Mask isn't supported here (background is bg — masking it would mean "draw nothing where mask is zero" which is the same as just clearing afterward; if you need that, do it explicitly).
pub fn background_noise<P: Pixel>(
    canvas: &mut Canvas<P>,
    shimmer: usize,
    fullscreen: bool,
    scroll_offset: isize,
    clip: Option<Clip>,
    base: Option<u32>,
) {
    background_noise_split(canvas, shimmer, fullscreen, scroll_offset, None, None, clip, base)
}

/// [`background_noise`] with an explicit texture ORIGIN and independent per-half scroll.
/// The noise mirrors left/right around `split_x` (the "texture origin"); `None` = the buffer centre `width/2` (what plain [`background_noise`] uses).
/// `left_scroll = None` scrolls both halves together by `scroll_offset` (the chat / single-pane case); `Some(v)` scrolls the LEFT half by `v` while the right half keeps `scroll_offset` — so a split pane (e.g. the settings rail vs content) scrolls each half with its own pane, with the origin sitting on the divider between them.
#[allow(clippy::too_many_arguments)]
pub fn background_noise_split<P: Pixel>(
    canvas: &mut Canvas<P>,
    shimmer: usize,
    fullscreen: bool,
    scroll_offset: isize,
    split_x: Option<usize>,
    left_scroll: Option<isize>,
    clip: Option<Clip>,
    base: Option<u32>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if buf_w < 2 || buf_h < 2 {
        return;
    }
    // Texture origin: the left/right mirror axis + each half's inner edge. Default = buffer centre; `background_row` clamps it into the paint range.
    let split_x = split_x.unwrap_or(buf_w / 2);
    // Right half keeps `scroll_offset`; the left half tracks its own pane when given, else moves with the right (a single unified scroll → the two halves stay mirror-locked).
    let left_scroll = left_scroll.unwrap_or(scroll_offset);
    let clip = Clip::resolve(clip, buf_w, buf_h);
    // Clip first, then `fullscreen` further insets by 1 px for the edge hairline.
    let (row_start, row_end, x_start, x_end) = if fullscreen {
        (clip.y_start, clip.y_end, clip.x_start, clip.x_end)
    } else {
        // The 1 px inset crosses itself on a clip under 3 px tall or wide (and would underflow at 0), so it runs signed and an empty result returns here.
        let (ys, ye) = (clip.y_start as isize + 1, clip.y_end as isize - 1);
        let (xs, xe) = (clip.x_start as isize + 1, clip.x_end as isize - 1);
        if ys >= ye || xs >= xe {
            return;
        }
        (ys as usize, ye as usize, xs as usize, xe as usize)
    };
    if row_start >= row_end || x_start >= x_end {
        return;
    }
    canvas.damage.add_bounds(x_start, row_start, x_end, row_end);
    let pixels: &mut [P] = canvas.pixels;
    let resolved_base = base.unwrap_or(crate::theme::BG_BASE);
    crate::par::par_rows(pixels, buf_w, row_start, row_end, |row_idx, row_pixels| {
        background_row(
            row_pixels,
            buf_w,
            row_idx as isize,
            buf_h,
            x_start,
            x_end,
            split_x,
            left_scroll,
            scroll_offset,
            shimmer,
            resolved_base,
        );
    });
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn background_row<P: Pixel>(
    row_pixels: &mut [P],
    _width: usize,
    row: isize,
    height: usize,
    x_start: usize,
    x_end: usize,
    split_x: usize,
    left_scroll: isize,
    right_scroll: isize,
    shimmer: usize,
    base: u32,
) {
    use crate::theme::{BG_MASK, BG_SPECKLE};
    // Noise math runs in visible-RGB space (matching photon's reference), on the 8-bit authored palette (BG_BASE, BG_MASK, BG_SPECKLE are bytes). At the store site we flip the visible result to stored darkness via XOR, then OR α=0xFF for opaque, and widen the 8-bit pixel to the canvas's depth. Mask off the top byte first to strip any carry from `wrapping_add`.
    const VISIBLE_TO_DARK_FLIP: u32 = Argb8::RGB_MASK;
    const RGB_MASK: u32 = Argb8::RGB_MASK;
    const OPAQUE_ALPHA: u32 = Argb8::OPAQUE;
    // Hybrid 2-pass: pass 1 fills `noise_buf` with the row's chunk of noise values via the serial RNG/colour chain (branches stay scalar — predicating the speckle gate would cost as much as it saves). Pass 2 hands the chunk to the depth's under-blend chunk kernel (`Pixel::under_chunk_normal`, 8-wide SIMD for Argb8), which composites it into row_pixels at ~1 cycle/pixel amortized. Output is bit-identical to the old straight-scalar version.
    const CHUNK: usize = 64;
    let mut noise_buf = [P::EMPTY; CHUNK];
    let ones = 0x0001_0101u32;
    // RNG chain is explicitly u64 (not usize) so the noise pattern is bit-identical on 32-bit targets (wasm32 browser renderer) and 64-bit desktops — same seed, same wrap points.
    // Each half's seed is keyed by its OWN scrolled row, so equal scroll → identical seed → the two halves stay mirror-locked (the chat case); different scroll → each half's texture tracks its pane (the split settings case).
    let seed_for = |logical_row: isize| -> u64 {
        0xDEAD_BEEF_0123_4567u64
            ^ (logical_row as i64 as u64)
                .wrapping_sub(height as u64 / 2)
                .wrapping_mul(0x9E37_79B9_4517_B397)
    };
    // The mirror axis: both halves grow outward from here, clamped into the paint range.
    // WHY/PROOF: split_x is the app's pane split, which the damage clip can leave outside [x_start, x_end) — the halves would index the row outside the paint range.
    let origin = split_x.clamp(x_start, x_end);

    // Right half — origin to right edge, ADD step, scrolled by `right_scroll`. Noise composes UNDER existing content (topmost-first): an empty pixel gets the noise; a non-empty pixel (e.g. a topmost rect already painted) has the noise blended behind it.
    let mut rng = seed_for(row - right_scroll);
    let mut colour = rng.wrapping_add(shimmer as u64) as u32 & BG_MASK;
    let mut x = origin;
    while x < x_end {
        let chunk_len = (x_end - x).min(CHUNK);
        for i in 0..chunk_len {
            rng ^= rng.rotate_left(13).wrapping_add(12_345_678_942);
            let adder = rng as u32 & ones;
            if rng < u64::MAX / 256 {
                colour = (rng as u32 >> 8) & BG_SPECKLE;
            } else {
                colour = colour.wrapping_add(adder) & BG_MASK;
                let subtractor = (rng >> 5) as u32 & ones;
                colour = colour.wrapping_sub(subtractor) & BG_MASK;
            }
            noise_buf[i] =
                P::from_g2t8(((colour.wrapping_add(base) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA);
        }
        P::under_chunk_normal(&mut row_pixels[x..x + chunk_len], &noise_buf[..chunk_len]);
        x += chunk_len;
    }

    // Left half — origin leftward, SUB instead of ADD on the rng step, scrolled by `left_scroll` (its own seed). Within each chunk the RNG iterates rightmost-pixel-first; we store into `noise_buf` in left-to-right order (`i = chunk_len-1` down to 0) so the chunk dispatch can scan the buffer sequentially.
    rng = seed_for(row - left_scroll);
    let mut colour = rng.wrapping_add(shimmer as u64) as u32 & BG_MASK;
    let mut x_hi = origin;
    while x_hi > x_start {
        let chunk_len = (x_hi - x_start).min(CHUNK);
        let chunk_lo = x_hi - chunk_len;
        for i in (0..chunk_len).rev() {
            rng ^= rng.rotate_left(13).wrapping_sub(12_345_678_942);
            let adder = rng as u32 & ones;
            if rng < u64::MAX / 256 {
                colour = (rng as u32 >> 8) & BG_SPECKLE;
            } else {
                colour = colour.wrapping_add(adder) & BG_MASK;
                let subtractor = (rng >> 5) as u32 & ones;
                colour = colour.wrapping_sub(subtractor) & BG_MASK;
            }
            noise_buf[i] =
                P::from_g2t8(((colour.wrapping_add(base) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA);
        }
        P::under_chunk_normal(&mut row_pixels[chunk_lo..x_hi], &noise_buf[..chunk_len]);
        x_hi = chunk_lo;
    }
}

/// The background noise generated once (at 8 bits, whatever the frame's depth: 4 bytes a pixel kept, widened as it is laid down) and rolled (owner, 2026-11-08: "keep a background buffer that you generate once ... and just rotate it vertically"): [`background_noise_split`]'s texture kept as two strips, generated for a viewport size and base colour and then never again, and laid under a frame at any scroll by reading its rows at an offset, wrapping at the bottom. A scroll costs nothing but the compose, which is one pass of [`Pixel::under_chunk_normal`] over the frame.
/// The strips hold what each half of the mirror grows from its axis: `right` the pixels going right from the origin (the ADD stream), `left` those going left (the SUB stream), stored right-aligned so a frame's left half reads forward. Any origin up to the strips' reach shows the same texture [`background_noise_split`] draws there, so a split screen (settings: the axis on the rail divider, each half with its own pane's scroll) and a centred one share the one field.
/// Rows are independent (each seeds from its own row number), so the field tiles vertically with no seam: row `t` of a `h`-row field is the row the texture has at logical row `t`, and a scroll `s` shows row `(y − s) mod h` at `y`. Where `0 ≤ y − s < h` that is exactly [`background_noise_split`]'s pixel at scroll `s` with no shimmer; the shimmer (a colour bias the old path cycled while scrolling) is gone: the field moves, it does not change.
pub struct NoiseField {
    width: usize,
    height: usize,
    base: u32,
    right_w: usize,
    left_w: usize,
    right: Vec<Argb8>,
    left: Vec<Argb8>,
}

impl Default for NoiseField {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseField {
    /// An empty field; the first [`Self::ensure`] generates it.
    pub const fn new() -> Self {
        Self { width: 0, height: 0, base: 0, right_w: 0, left_w: 0, right: Vec::new(), left: Vec::new() }
    }

    /// Make the field cover a `width × height` frame with its mirror axis at `origin` (≤ `width`) in colour `base` (`None` = [`crate::theme::BG_BASE`]). Generates only when the size or the base changed, or the axis needs a strip longer than the one kept (the reach only grows, so a screen that moves the axis back and forth generates once per size). True when it generated.
    pub fn ensure(&mut self, width: usize, height: usize, origin: usize, base: Option<u32>) -> bool {
        let base = base.unwrap_or(crate::theme::BG_BASE);
        let same = self.width == width && self.height == height && self.base == base;
        // The strips each half needs at this axis; a kept strip at least that long serves.
        let (need_r, need_l) = (width - origin, origin);
        if same && need_r <= self.right_w && need_l <= self.left_w {
            return false;
        }
        // The new reach: what this axis needs, and what the old field already reached for this same frame (so returning to the other axis does not generate again).
        let (right_w, left_w) = if same { (need_r.max(self.right_w), need_l.max(self.left_w)) } else { (need_r, need_l) };
        self.width = width;
        self.height = height;
        self.base = base;
        self.right_w = right_w;
        self.left_w = left_w;
        self.right.clear();
        self.right.resize(right_w * height, 0);
        self.left.clear();
        self.left.resize(left_w * height, 0);
        if right_w > 0 {
            crate::par::par_rows(&mut self.right, right_w, 0, height, |t, row| noise_strip(row, t, height, base, false));
        }
        if left_w > 0 {
            crate::par::par_rows(&mut self.left, left_w, 0, height, |t, row| noise_strip(row, t, height, base, true));
        }
        true
    }

    /// Lay the field under `canvas` (the frame [`Self::ensure`] was given): mirror axis at `origin`, the left half scrolled by `left_scroll` and the right by `right_scroll` (equal for a centred screen, each half's pane on a split one), the whole frame or (`fullscreen` false) all but its 1-pixel edge, inside `clip`. One pass over the rows in one dispatch to the workers.
    #[allow(clippy::too_many_arguments)]
    pub fn under_into<P: Pixel>(&self, canvas: &mut Canvas<P>, origin: usize, left_scroll: isize, right_scroll: isize, fullscreen: bool, clip: Option<Clip>) {
        // A frame of another size than the field's is a bug in the caller (ensure was skipped): fail here, not in a slice somewhere.
        assert!(canvas.width == self.width && canvas.height == self.height, "the noise field is {} x {}, the frame {} x {}", self.width, self.height, canvas.width, canvas.height);
        assert!(self.width - origin <= self.right_w && origin <= self.left_w, "the noise field does not reach the axis at {origin}: ensure it first");
        let (w, h) = (self.width, self.height);
        // Under 2 × 2 there is no inside to the 1-pixel edge ([`background_noise_split`] draws nothing there either).
        if w < 2 || h < 2 {
            return;
        }
        let clip = Clip::resolve(clip, w, h);
        let inset = if fullscreen { 0 } else { 1 };
        // The 1-pixel edge left for the window's hairline: the clip cut to the inside of it (a clip thinner than the edge comes out empty below).
        let (ys, ye) = (clip.y_start.max(inset), clip.y_end.min(h - inset));
        let (xs, xe) = (clip.x_start.max(inset), clip.x_end.min(w - inset));
        if ys >= ye || xs >= xe {
            return;
        }
        canvas.damage.add_bounds(xs, ys, xe, ye);
        let (rw, lw) = (self.right_w, self.left_w);
        let (right, left) = (&self.right, &self.left);
        crate::par::par_rows(canvas.pixels, w, ys, ye, |y, row| {
            // The texture rows this frame row shows, each half by its own scroll, wrapping at the field's height.
            let tr = (y as isize - right_scroll).rem_euclid(h as isize) as usize;
            let tl = (y as isize - left_scroll).rem_euclid(h as isize) as usize;
            // The row splits at the axis: [max(xs, origin), xe) is the right half's part of the clip, [xs, min(xe, origin)) the left's. These are the clip's two pieces, not a clamp; an axis outside the clip leaves one piece empty.
            if xe > origin {
                let a = xs.max(origin);
                under_widened(&mut row[a..xe], &right[tr * rw + (a - origin)..tr * rw + (xe - origin)]);
            }
            if xs < origin {
                let b = xe.min(origin);
                under_widened(&mut row[xs..b], &left[tl * lw + (lw - origin + xs)..tl * lw + (lw - origin + b)]);
            }
        });
    }
}

/// `src` (8-bit pixels) under `dst` at `dst`'s depth: a run of 64 widened on the stack, then the depth's chunk kernel, as [`background_row`] composes its noise.
#[inline]
fn under_widened<P: Pixel>(dst: &mut [P], src: &[Argb8]) {
    const CHUNK: usize = 64;
    let mut buf = [P::EMPTY; CHUNK];
    for (d, s) in dst.chunks_mut(CHUNK).zip(src.chunks(CHUNK)) {
        for (b, &c) in buf.iter_mut().zip(s) {
            *b = P::from_g2t8(c);
        }
        P::under_chunk_normal(d, &buf[..d.len()]);
    }
}

/// One strip row of [`NoiseField`]: row `t` of the texture, going right from the axis (`left` false: `out[k]` is the k-th pixel right of it) or left (`left` true: right-aligned, `out[len − 1 − k]` the k-th pixel left of it). The same seed, steps and speckle gate as [`background_row`], so the pixels are the same.
fn noise_strip(out: &mut [Argb8], t: usize, height: usize, base: u32, left: bool) {
    use crate::theme::{BG_MASK, BG_SPECKLE};
    const VISIBLE_TO_DARK_FLIP: u32 = Argb8::RGB_MASK;
    const RGB_MASK: u32 = Argb8::RGB_MASK;
    const OPAQUE_ALPHA: u32 = Argb8::OPAQUE;
    let ones = 0x0001_0101u32;
    let mut rng = 0xDEAD_BEEF_0123_4567u64 ^ (t as u64).wrapping_sub(height as u64 / 2).wrapping_mul(0x9E37_79B9_4517_B397);
    let mut colour = rng as u32 & BG_MASK;
    let n = out.len();
    for k in 0..n {
        rng ^= if left { rng.rotate_left(13).wrapping_sub(12_345_678_942) } else { rng.rotate_left(13).wrapping_add(12_345_678_942) };
        let adder = rng as u32 & ones;
        if rng < u64::MAX / 256 {
            colour = (rng as u32 >> 8) & BG_SPECKLE;
        } else {
            colour = colour.wrapping_add(adder) & BG_MASK;
            let subtractor = (rng >> 5) as u32 & ones;
            colour = colour.wrapping_sub(subtractor) & BG_MASK;
        }
        out[if left { n - 1 - k } else { k }] = ((colour.wrapping_add(base) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA;
    }
}


/// Debug toggle that lets the `[]p` chord skip the boundary premultiply at runtime — A/B the Linux premult fix without recompiling. Stays `false` by default.
pub static DEBUG_SKIP_PREMULT: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Debug cycle bound to the `[]a` chord. Three states (rotate each press): `0` = off (normal boundary conversion), `1` = α-as-grayscale (replace each pixel with `(final_α, final_α, final_α, 0xFF)` — inspect alpha distribution), `2` = force-opaque (force every pixel's α to 255 and pass the visible RGB thru unmodified — inspect what the kernel produced BEFORE the clip mask + premultiply trimmed it).
pub static DEBUG_SHOW_ALPHA: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
pub const DEBUG_SHOW_ALPHA_OFF: u8 = 0;
pub const DEBUG_SHOW_ALPHA_GRAYSCALE: u8 = 1;
pub const DEBUG_SHOW_ALPHA_FORCE_OPAQUE: u8 = 2;

/// Debug toggle bound to the `[]h` chord. When set, `finalize_into_screen` routes thru the FORCE_OPAQUE debug branch (XOR darkness → visible RGB, ignore clip_mask trim, force α=0xFF, skip premult) so the per-id colours the consumer paints into scratch land in `persistent_screen` exactly as written — no AA edges, no corner cutouts, no shadow boost on the perimeter. The host additionally skips `paint_shadow` while this is on so the band outside the window doesn't disturb the hit-id view at the chrome edge.
pub static DEBUG_SHOW_HITMASK: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Debug toggle bound to the `[]d` chord (Decay). When set, the host saturating-subtracts 1 from every persistent_screen pixel's RGB at the top of each frame (before finalize) and self-requests a continuous redraw chain. Fresh pixels written by finalize / overlay land at full brightness; pixels that aren't repainted fade visibly toward black. Used to verify the incremental left/right opaque-scan finalize is actually copying the regions it should — anything that doesn't get touched on a given frame visibly decays.
pub static DEBUG_SHOW_FADE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Debug toggle bound to the `[]b` chord. When set, the **incremental** finalize path saturating-adds 16 to the blue byte of every pixel inside the per-row `[l, r)` opaque-scan range, AFTER the chunk dispatch writes the post-XOR visible-RGB pixel. Makes exactly the pixels the incremental scan touches each frame glow blue — full_repaint frames are deliberately NOT tinted because a uniform interior wash on every focus change / resize / drag release would drown out the actual diagnostic signal (which is "what does the incremental scan land on?"). Pairs naturally with `DEBUG_SHOW_FADE` (the blue stack reaches equilibrium where incremental finalize hits often, decays elsewhere).
pub static DEBUG_SHOW_OPAQUE_SCAN: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// OPAQUE BACKDROP (Nick 2026-09-09: "on fullscreen/android, blank with no border… whatever the buffer colour is, black"). When set, finalize hands the OS an α=255 frame: the premultiplied RGB it already computed IS the colour composited over black, so every partly-transparent pixel — the background field, the AA rims, anything the app left thin — lands on black instead of whatever the compositor keeps behind the surface. Flipped by the chrome's full-edge mode (maximized / Android), where there is no desktop to see thru to. Not a debug toggle.
pub static OPAQUE_BACKDROP: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Debug toggle that suppresses chrome layer rasterization (perimeter hairline + future controls + title) so consumers can see the background / panes / textbox underneath without chrome on top. Bound to the `[]c` chord. The clip_mask is still carved at the boundary, so the window-shape trim remains visible. Stays `false` by default.
pub static DEBUG_SKIP_CHROME: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Debug toggle that suppresses ONLY the controls strip (curves + hairlines + glyphs + dividers + strip-bg fill) while keeping the window perimeter intact. Bound to the `[]l` chord (controLs). Useful for isolating perimeter rendering from controls rendering. Stays `false` by default.
pub static DEBUG_SKIP_CONTROLS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Debug toggle that overlays a one-line diagnostic strip across the bottom of the window showing live render-pipeline stats: composite-FPS (= `1.0 / composite_time`, NOT the vsync-capped frame rate) and the cumulative frame counter. Bound to the `[]f` chord. The composite-FPS is the actual headroom — a 144 Hz display showing "1240 FPS" means each composite took ~0.8 ms, leaving 6.1 ms of slack against vsync. `false` by default. Counter bumped by primitives that perform genuine *rasterize* work (geometric paint, glyph shaping, etc.) — NOT by blits/copies/tint applications. The host reads-and-resets this with `.swap(0, ...)` after `app.render` to decide whether to call `DebugStats::record_rasterize` or only `record_present`. Lets the F (frame) counter climb on hover-only frames while R (rasterize) stays put.
pub static RASTERIZE_OPS: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

pub static DEBUG_SHOW_FPS: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Debug toggle that overlays a 1-px magenta hairline around the damage rect the host repaints this frame. Bound to the `[]w` chord ("Where"). Drawn directly into the platform back buffer AFTER `persistent_screen` has been copied in, BEFORE `present()`. The outline never enters `persistent_screen`, never flows thru finalize, and never survives more than one frame — so toggling it on/off needs no full-repaint promotion and there is no stale-bbox state to carry between frames. `false` by default.
pub static DEBUG_SHOW_DAMAGE: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

/// Overlay a chord-hint panel listing the consumer's debug shortcuts. Painted while the consumer's debug chord is armed so users discover bindings without consulting docs. `hints` is `(chord_label, description)` pairs. `span` is the viewport's effective span (`2wh/(w+h)`) — all sizes derive from it so the panel scales with the user's zoom.
///
/// Topmost-first: text glyphs paint first; the semi-opaque panel background fills the gaps behind them.
#[cfg(feature = "text")]
pub fn draw_chord_hint<P: Pixel>(
    canvas: &mut Canvas<P>,
    text: &mut crate::text::TextRenderer,
    hints: &[(&str, &str)],
    span: Coord,
) {
    if hints.is_empty() || canvas.width < 200 || canvas.height < 120 {
        return;
    }
    // RU-coherent typography. `font_size = span × 0.014` lands at ~14 px at span ≈ 1000, ~28 px at 2000, etc. Vertical rhythm + padding follow as ratios of the font size so the panel scales as one unit. No pixel floor — span already scales with the viewport, and a hardcoded minimum would break resolution independence (and desync from photon's matching bbox math).
    let font_size = span * 0.014;
    let header_size = font_size * 1.18;
    let line_h = font_size * 1.55;
    let pad = font_size * 1.25;

    let line_count = hints.len() as f32 + 1.5;
    let panel_h = line_count * line_h + pad * 2.0;
    // A span fraction — the old `.clamp(font_size*22, font_size*36)` was a dead no-op (span*0.45 always sits in [0.308span, 0.504span] since font_size = span*0.014). No clamp, no pixels; photon's matching bbox math uses the identical expression.
    let panel_w = span * 0.45;

    let cx = canvas.width as f32 * 0.5;
    let cy = canvas.height as f32 * 0.4;
    let panel_y = cy - panel_h * 0.5;

    let title_colour = pack_argb(255, 255, 255, 0xFF);
    let body_colour = pack_argb(220, 220, 220, 0xFF);

    text.draw_text_center(canvas, "Debug chord — [ + ] then …", cx, panel_y + pad + header_size * 0.5, &TextStyle::new(header_size, title_colour).weight(500), None, None);

    for (i, (chord, desc)) in hints.iter().enumerate() {
        let line_y = panel_y + pad + header_size + line_h * (i as f32 + 0.5) + font_size * 0.5;
        let line = alloc::format!("{}  —  {}", chord, desc);
        text.draw_text_center(canvas, &line, cx, line_y, &TextStyle::new(font_size, body_colour), None, None);
    }

    // Panel background fills behind the glyphs (text rows already occupied; bar's under() only lands on the gaps).
    let bg = pack_argb(0, 0, 0, 0xD8);
    draw_rect(canvas, cx, cy, panel_w, panel_h, bg, None);
}

/// Live diagnostic counters owned by the host's render loop and read by [`draw_debug_strip`]. All fields are simple POD; the host updates them every frame when [`DEBUG_SHOW_FPS`] is on and the helper renders them as a single line of text into the bottom-of-window scratch region before the boundary pass runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugStats {
    /// Raw work time of the most recent frame, per stage, in seconds. NOT smoothed — each value is exactly the last measurement so SIMD/Rayon toggles produce immediately legible swings. FPS shown in the strip is `1.0 / stage_secs` per stage and `1.0 / sum` for total.
    pub app_secs: f32,
    pub fill_secs: f32,
    pub finalize_secs: f32,
    pub shadow_secs: f32,
    /// Frames presented to the OS since start — includes "frame-only" updates (tint diff, cursor blink) that did no rasterization. Wraps at `u64::MAX`.
    pub frame_count: u64,
    /// Frames where actual rasterization work happened. Diverges from `frame_count` once dirty-tracking lands and lets the host skip rasterize on hover-only / tint-only updates.
    pub rasterize_count: u64,
    /// Fraction (`0.0..=1.0`) of the rasterizable area touched by the most recent rasterize. `1.0` = full repaint; smaller values appear once damage-clipped partial rasterization is wired. Held verbatim from the last `record_rasterize` call.
    pub last_rasterize_pct: f32,
    /// Fraction (`0.0..=1.0`) of the viewport area covered by this frame's `damage_clip` — what the host actually cleared + asked the app to repaint. Updated EVERY frame (on present), so it reflects the live per-frame damage even when `last_rasterize_pct` is sticky from the last full rasterize. Drops on hover/focus frames where damage is just the textbox bbox.
    pub last_damage_pct: f32,
}

impl DebugStats {
    /// Bump `frame_count` only — for frames that present without rasterizing (cursor blink off-frame, tint-diff-only hover update once dirty tracking lands). No stage timings updated; the strip continues to show whatever the last rasterize recorded. `damage_pct` updates every present so the live damage area shows even when no rasterize fired.
    #[inline]
    pub fn record_present(&mut self, damage_pct: f32) {
        self.frame_count += 1;
        self.last_damage_pct = damage_pct;
    }

    /// Store this rasterize's stage times, bump `rasterize_count`, and record the fraction of the rasterizable area that was touched. Pass `pct = 1.0` for a full repaint; once damage-clipped rasterization is wired, pass `damage_area / full_area`.
    #[inline]
    pub fn record_rasterize(&mut self, app: f32, fill: f32, finalize: f32, shadow: f32, pct: f32) {
        self.app_secs = app;
        self.fill_secs = fill;
        self.finalize_secs = finalize;
        self.shadow_secs = shadow;
        self.rasterize_count += 1;
        self.last_rasterize_pct = pct;
    }

    /// Shorthand for the current "every frame is a full rasterize" world — bumps both counters and stores timings with `pct = 1.0`. Replace per-call with `record_rasterize` + `record_present` once dirty tracking can drop the rasterize on tint-only updates.
    #[inline]
    pub fn record_frame(&mut self, app: f32, fill: f32, finalize: f32, shadow: f32) {
        self.record_rasterize(app, fill, finalize, shadow, 1.0);
        self.record_present(1.0);
    }

    #[inline]
    pub fn total_secs(&self) -> f32 {
        self.app_secs + self.fill_secs + self.finalize_secs + self.shadow_secs
    }
}

/// Overlay a one-line diagnostic strip across the bottom of `pixels` showing the live stats in [`DebugStats`]. Gated by [`DEBUG_SHOW_FPS`] — the host should check that flag before calling. Paints into the α + darkness scratch buffer BEFORE the boundary pass so the strip flows thru `finalize_*` like any other content (no special handling needed downstream).
///
/// The strip is `~24` pixels tall, semi-opaque black background, bright green monospace text (terminal-style for readability against any underlying content). Positioned at the very bottom of `pixels`; clipped to the buffer if the window is too short to fit the strip (returns early in that case — diagnostic, not load-bearing).
///
/// `strip_y` is the canvas-relative row where the strip's top edge lands. Callers can pass any value that fits inside the canvas (`strip_y + DEBUG_STRIP_H ≤ canvas.height`); the host typically points the canvas at a dedicated `DEBUG_STRIP_H`-tall staging buffer with `strip_y = 0` so the strip never touches the app's scratch.
#[cfg(feature = "text")]
pub fn draw_debug_strip<P: Pixel>(
    canvas: &mut Canvas<P>,
    text: &mut crate::text::TextRenderer,
    stats: &DebugStats,
    strip_y: usize,
) {
    const FONT_SIZE: f32 = 13.0;
    let width = canvas.width;
    let height = canvas.height;
    if width == 0 || height < strip_y + DEBUG_STRIP_H {
        return;
    }

    let app_ms = stats.app_secs * 1000.0;
    let fill_ms = stats.fill_secs * 1000.0;
    let fin_ms = stats.finalize_secs * 1000.0;
    let shadow_ms = stats.shadow_secs * 1000.0;
    let total_ms = stats.total_secs() * 1000.0;

    let damage_pct = stats.last_damage_pct * 100.0;
    let stats_line = alloc::format!(
        "app {app_ms:>6.3} ms  fill {fill_ms:>6.3} ms  fin {fin_ms:>6.3} ms  shdw {shadow_ms:>6.3} ms    tot {total_ms:>6.3} ms    F {:>7} R {:>7} ({damage_pct:>5.1}%)",
        stats.frame_count,
        stats.rasterize_count,
    );

    // Topmost-first ordering: text glyphs paint FIRST so the bar's under() writes are rejected by the glyph pixels, leaving the green characters visible against the black. If the bar were painted first it would fill all strip pixels opaque and every glyph would be eaten.
    let fg = pack_argb(80, 255, 120, 0xFF);
    let text_cy = strip_y as f32 + DEBUG_STRIP_H as f32 * 0.5;
    text.draw_text_center(canvas, &stats_line, width as f32 * 0.5, text_cy, &TextStyle::new(FONT_SIZE, fg).font("monospace"), None, None);

    // Bar fills behind the glyphs (topmost-first: text rows already occupied, bar's under() only lands on the gaps).
    let bg = pack_argb(0, 0, 0, 0xE0);
    draw_rect(
        canvas,
        width as Coord * 0.5,
        strip_y as Coord + DEBUG_STRIP_H as Coord * 0.5,
        width as Coord,
        DEBUG_STRIP_H as Coord,
        bg,
        None,
    );
}

/// Height in pixels of the [`draw_debug_strip`] band — the consumer / host uses it to size the staging buffer and pick the screen-space y position.
#[cfg(feature = "text")]
pub const DEBUG_STRIP_H: usize = 24;



/// The blinkey (the text caret): a vertical wave of white light at column `bx`, `height` tall from row `by`, brightest toward its top (`top_bright`) or its bottom, smeared seven pixels either side at half the light per step.
/// Bounds contract: `bx ≥ 7`, `bx + 7 < canvas.width` and `by + height ≤ canvas.height`, which every caller checks; the smear indexes the buffer directly.
pub fn draw_blinkey<P: Pixel>(canvas: &mut Canvas<P>, bx: usize, by: usize, height: usize, top_bright: bool) {
    draw_blinkey_tinted(canvas, bx, by, height, top_bright, (255, 255, 255));
}

// PROOF that the caret's α stays within ONE: its peak light, (8/9)(16/9) × CURSOR_BRIGHTNESS of 255, is under the code of white's brightest channel, and α is that light ÷ that code.
const _: () = {
    let w = crate::theme::WHITE;
    let (r, g, b) = ((w >> 16) & 0xFF, (w >> 8) & 0xFF, w & 0xFF);
    let least = if r < g && r < b { r } else if g < b { g } else { b };
    assert!(crate::theme::CURSOR_BRIGHTNESS * (128. / 81.) < (255 - least) as f32);
};

/// The blinkey in a COLOUR: the same brightness wave, scaled per channel by `tint` (255 = the full wave, 0 = that channel stays dark). White tint = the plain blinkey; a link-purple caret is (160, 80, 255). Same bounds contract as [`draw_blinkey`].
///
/// The caret is a layer of light in FRONT of what is drawn: a glow of white's colour (the theme's WHITE, channel by channel) at a falling α, `α = light ÷ the code of white's brightest channel` (so over black that channel shows exactly `light`, and over white the caret shows white), each channel's light `α × its white code ÷ ONE × tint ÷ 255`.
/// It is drawn after the text it sits in (the callers paint the field, then the caret), so it cannot go into the accumulator first; it goes in front of the finished pixels instead, `caret.under_premult(pixel)`, which is the same composite as the caret laid first and the field under it, since `under` is associative over premultiplied operands.
/// Until 2026-11-07 the caret subtracted its light from the finished darkness (drawn last, painter's order), floored at white: a pixel already brighter than white kept its light, and on a translucent pixel the caret added no coverage of its own.
pub fn draw_blinkey_tinted<P: Pixel>(canvas: &mut Canvas<P>, bx: usize, by: usize, height: usize, top_bright: bool, tint: (u8, u8, u8)) {
    let buf_w = canvas.width;
    // Damage: ±7 horizontal spread × `height` vertical band. Caller's bounds invariant (bx ≥ 7, bx < buf_w-7) guarantees this stays in-buffer.
    canvas.damage.add_bounds(bx - 7, by, bx + 8, by + height);
    let pixels: &mut [P] = canvas.pixels;
    let half = height / 2;
    // White's codes (light) in this depth, channel by channel: the caret's colour at full light; its α is measured on the brightest.
    let white_code = P::from_g2t8(crate::theme::WHITE).channels().map(|d| P::ONE - d);
    let brightest = white_code[0].max(white_code[1]).max(white_code[2]);
    // The wave is authored on the 8-bit scale (CURSOR_BRIGHTNESS); ONE / 255 is exactly 1 at 8 bits and carries the wave's fraction to 16. Each row's light is rounded to the nearest code (truncated, the 8-bit caret came out up to two steps darker than the 16-bit one once its α floored too).
    let to_depth = P::ONE as f32 / 255.;
    for y in by..by + height {
        let idx = y * buf_w + bx;
        let t = (y - by - half) as isize as f32 / half as f32;
        let wave = if top_bright {
            (1.0 - t * t) * (1.0 - t) * (1.0 - t) * crate::theme::CURSOR_BRIGHTNESS
        } else {
            (1.0 - t * t) * (1.0 + t) * (1.0 + t) * crate::theme::CURSOR_BRIGHTNESS
        };
        let w = (wave * to_depth + 0.5) as u32;
        for dx in -7i32..=7 {
            let k = w >> dx.unsigned_abs();
            // α = light ÷ the brightest white code, rounded, at most ONE by the const PROOF above; each channel's light is α's share of its white code × tint, rounded once (u64: the product passes 2³² at 16 bits), at most α, so its darkness α − light stays in 0..=α.
            let a = (k * P::ONE + (brightest >> 1)) / brightest;
            let tints = [tint.0, tint.1, tint.2];
            let light: [u32; 3] = core::array::from_fn(|c| ((a as u64 * white_code[c] as u64 * tints[c] as u64 + (P::ONE as u64 * 255 >> 1)) / (P::ONE as u64 * 255)) as u32);
            let caret = P::from_channels(a, light.map(|l| a - l));
            let p = &mut pixels[(idx as isize + dx as isize) as usize];
            *p = caret.under_premult(*p, BlendMode::Normal);
        }
    }
}


/// Fill a circle with a 1-pixel-wide AA edge ring. Center `(cx, cy)` and `radius` are in pixels; `colour` is straight-alpha ARGB (the AA coverage modulates the supplied alpha, so a translucent fill stays translucent at the edge). Optional `mask` multiplies into the per-pixel alpha for soft clipping.
///
/// AA via gradient-magnitude (no sqrt): for a pixel at squared distance `d²`, coverage is `(r_outer² - d²) / (r_outer² - r_inner²)` where `r_outer = radius` and `r_inner = radius - 1`. Gives a smooth 0→1 ramp across one pixel of edge.
pub fn circle_filled<P: Pixel>(
    canvas: &mut Canvas<P>,
    cx: isize,
    cy: isize,
    radius: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if radius <= 0 {
        return;
    }
    // Damage = bbox of the circle clipped to buffer + caller clip.
    {
        let c = Clip::resolve(clip, buf_w, buf_h);
        // WHY/PROOF: the centre is the caller's layout position, off the buffer whenever the circle straddles an edge — a negative bound would wrap to a huge usize in the damage rect.
        let x0 = (cx - radius).max(c.x_start as isize) as usize;
        let y0 = (cy - radius).max(c.y_start as isize) as usize;
        let x1 = (cx + radius + 1).min(c.x_end as isize).max(0) as usize;
        let y1 = (cy + radius + 1).min(c.y_end as isize).max(0) as usize;
        canvas.damage.add_bounds(x0, y0, x1, y1);
    }
    let pixels: &mut [P] = canvas.pixels;
    let clip = Clip::resolve(clip, buf_w, buf_h);
    if let Some(m) = mask {
        assert_mask_matches_buffer(m, buf_w, buf_h);
    }
    let r_outer = radius;
    let r_outer2 = r_outer * r_outer;
    let r_inner = radius - 1;
    let r_inner2 = r_inner * r_inner;
    let edge_range = r_outer2 - r_inner2;

    // Circle's bounding box, clipped. Side length is 2r + 1 (inclusive on both ends).
    let (x_min, y_min, x_max, y_max) = clip_rect(
        clip,
        cx - r_outer,
        cy - r_outer,
        2 * r_outer + 1,
        2 * r_outer + 1,
    );

    // α + darkness: the α field IS opacity; use directly. RGB darkness intact.
    let colour = P::from_g2t8(colour);
    let fg_opacity = colour.alpha();
    let colour_rgb = colour.rgb();

    for py in y_min..y_max {
        let dy = py as isize - cy;
        let dy2 = dy * dy;
        let base = py * buf_w;
        for px in x_min..x_max {
            let dx = px as isize - cx;
            let dist2 = dx * dx + dy2;
            if dist2 > r_outer2 {
                continue;
            }
            // Coverage in the depth's own fixed point, `1 << DEPTH` = whole pixel (256 at 8 bits).
            let coverage: u32 = if dist2 <= r_inner2 {
                1 << P::DEPTH
            } else {
                (((r_outer2 - dist2) << P::DEPTH) / edge_range) as u32
            };
            let scaled_opacity = (fg_opacity * coverage) >> P::DEPTH;
            let idx = base + px;
            // The mask is bytes at every depth, so its `>> 8` stays.
            let final_opacity = match mask {
                Some(m) => (scaled_opacity * m.pixels[idx] as u32) >> 8,
                None => scaled_opacity,
            };
            let scaled_colour = colour_rgb.with_alpha(final_opacity);
            pixels[idx] = pixels[idx].under(scaled_colour, BlendMode::Normal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::Blend;

    #[test]
    fn pack_unpack_round_trip() {
        let cases = [
            (0, 0, 0, 0),
            (255, 255, 255, 255),
            (12, 34, 56, 78),
            (200, 100, 50, 200),
        ];
        for &(r, g, b, a) in &cases {
            let p = pack_argb(r, g, b, a);
            assert_eq!(unpack_argb(p), (r, g, b, a));
        }
    }

    #[test]
    fn pack_layout_stores_alpha_in_top_byte() {
        // α + darkness storage: α = 0x12 in the top byte, RGB stored as darkness (255 − channel). pack_argb(0xAB, 0xCD, 0xEF, 0x12) → α=0x12, dark=(0x54, 0x32, 0x10).
        assert_eq!(pack_argb(0xAB, 0xCD, 0xEF, 0x12), 0x12_54_32_10);
    }

    #[test]
    fn under_fully_transparent_top_yields_bottom_within_one_lsb() {
        // Canonical empty top (α=0, dark=0 → 0x00000000) over opaque bottom → result visible ≈ bottom with ≤1 LSB drift from the >>8 truncation.
        let top = pack_argb(255, 255, 255, 0);
        let bottom = pack_argb(100, 150, 200, 255);
        let result = top.under(bottom, BlendMode::Normal);
        let (r, g, b, _) = unpack_argb(result);
        assert!((r as i32 - 100).abs() <= 1, "r got {}", r);
        assert!((g as i32 - 150).abs() <= 1, "g got {}", g);
        assert!((b as i32 - 200).abs() <= 1, "b got {}", b);
    }

    #[test]
    fn under_opaque_top_returns_top_exact() {
        // top with α=0xFF (opaque) — early-out returns top unchanged regardless of bottom.
        let top = pack_argb(50, 80, 110, 255);
        let bottom = pack_argb(100, 150, 200, 255);
        let result = top.under(bottom, BlendMode::Normal);
        assert_eq!(result, top);
        let (r, g, b, _) = unpack_argb(result);
        assert_eq!((r, g, b), (50, 80, 110));
    }

    /// The blinkey is light in front of what is drawn, at both depths: over opaque black it brightens and stays opaque, over white it stays white (to a code), and over an empty buffer it lands as a layer of its own, white at its α. Until 2026-11-07 it subtracted from the darkness already there: an empty buffer stayed empty.
    #[test]
    fn blinkey_is_light_in_front() {
        fn run<P: Pixel>() {
            let (w, h) = (24usize, 16usize);
            let white = P::from_g2t8(crate::theme::WHITE);
            let white_dark = white.channels();
            for (fill, name) in [(P::from_channels(P::ONE, [P::ONE; 3]), "black"), (white, "white"), (P::EMPTY, "empty")] {
                let mut buf = vec![fill; w * h];
                let mut damage = crate::canvas::Damage::new();
                draw_blinkey(&mut crate::canvas::Canvas::new(&mut buf, w, h, &mut damage), 12, 0, h, true);
                assert!(buf.iter().any(|&p| p != fill), "{name}: the caret drew nothing");
                for p in buf {
                    let (a, d) = (p.alpha(), p.channels());
                    assert!(d.iter().all(|&c| c <= a), "{name}: {p:?} darkness past α");
                    match name {
                        "black" => assert_eq!(a, P::ONE, "{name}: {p:?}"),
                        "white" => assert!(a == P::ONE && (0..3).all(|c| d[c].abs_diff(white_dark[c]) <= 1), "{name}: {p:?}"),
                        // The caret's own colour, white, premultiplied: darkness = α × white's darkness ÷ ONE, to a code.
                        _ => assert!((0..3).all(|c| d[c].abs_diff(a * white_dark[c] / P::ONE) <= 1), "{name}: {p:?}"),
                    }
                }
            }
        }
        run::<Argb8>();
        run::<crate::pixel::G2t16>();
    }

    #[test]
    fn stroke_rect_only_touches_edges() {
        // Buffer starts fully transparent (α=0, dark=0 → 0x00000000) — the canonical empty state for under-blend painting.
        let mut buf = vec![0u32; 10 * 10];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 10, 10, &mut damage);
        stroke_rect(
            &mut canvas,
            2,
            2,
            6,
            6,
            1,
            pack_argb(255, 0, 0, 255),
            None,
            None,
        );
        // Interior of the rect — never touched, stays empty.
        assert_eq!(buf[5 * 10 + 5], 0);
        let (r, _, _, _) = unpack_argb(buf[2 * 10 + 2]);
        assert!(r > 240, "top-left stroke pixel r={}", r);
        // Outside the rect — untouched, stays empty.
        assert_eq!(buf[1 * 10 + 1], 0);
    }

    #[test]
    fn circle_filled_center_is_colour() {
        let mut buf = vec![0u32; 16 * 16];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 16, 16, &mut damage);
        circle_filled(&mut canvas, 8, 8, 5, pack_argb(255, 0, 0, 255), None, None);
        let (r, g, b, _) = unpack_argb(buf[8 * 16 + 8]);
        assert!(
            r > 240 && g < 16 && b < 16,
            "center = ({}, {}, {})",
            r,
            g,
            b
        );
        // Corner outside the circle — untouched, stays empty.
        assert_eq!(buf[0], 0);
    }

    #[test]
    fn circle_filled_clips_partial_offscreen() {
        let mut buf = vec![0u32; 8 * 8];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 8, 8, &mut damage);
        circle_filled(
            &mut canvas,
            -2,
            -2,
            4,
            pack_argb(255, 255, 255, 255),
            None,
            None,
        );
        let (r, _, _, _) = unpack_argb(buf[0]);
        assert!(r > 200, "buf[0] r={}", r);
    }

    #[test]
    fn under_half_top_blends_with_opaque_bottom() {
        // Half-α white top (α=128, visible white) over opaque black bottom. Porter-Duff over: 1 − (1−0.5)×1 = 1.0 α, visible ≈ 0.5×white + 0.5×black = mid-gray (~128 per channel).
        let top = pack_argb(255, 255, 255, 128);
        let bottom = pack_argb(0, 0, 0, 255);
        let result = top.under(bottom, BlendMode::Normal);
        let (r, g, b, _) = unpack_argb(result);
        assert!((r as i32 - 128).abs() <= 2, "r got {}", r);
        assert!((g as i32 - 128).abs() <= 2, "g got {}", g);
        assert!((b as i32 - 128).abs() <= 2, "b got {}", b);
    }

    #[test]
    fn fill_rect_full_buffer() {
        // Buffer starts EMPTY (α=0, dark=0). Paint opaque visible RGB(0x11,0x22,0x33) under it. Result: ~opaque colour (1-LSB drift per channel from the >>8 normalization).
        let mut buf = vec![0u32; 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        // Visible (0x11, 0x22, 0x33) with α=0xFF → α + darkness = α=0xFF, dark=(0xEE, 0xDD, 0xCC).
        fill_rect(
            &mut canvas,
            0,
            0,
            4,
            4,
            pack_argb(0x11, 0x22, 0x33, 0xFF),
            None,
            None,
        );
        for &p in &buf {
            assert_eq!(p >> 24, 0xFF, "pixel α should be 0xFF (opaque): {p:#x}");
            // Darkness of visible 0x11 is 0xEE; minus 1-LSB drift.
            assert!(((p >> 16) & 0xFF) >= 0xEC);
        }
    }

    #[test]
    fn fill_rect_partial() {
        let mut buf = vec![0u32; 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        fill_rect(
            &mut canvas,
            1,
            1,
            2,
            2,
            pack_argb(0xAA, 0xBB, 0xCC, 0xFF),
            None,
            None,
        );
        for y in 0..4usize {
            for x in 0..4usize {
                let p = buf[y * 4 + x];
                let inside = x >= 1 && x < 3 && y >= 1 && y < 3;
                if inside {
                    assert_eq!(
                        p >> 24,
                        0xFF,
                        "inside pixel ({x},{y}) should be opaque: {p:#x}"
                    );
                } else {
                    assert_eq!(p, 0, "outside pixel ({x},{y}) untouched, got {p:#x}");
                }
            }
        }
    }

    #[test]
    fn fill_rect_clips_negative_origin() {
        let mut buf = vec![0u32; 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        fill_rect(
            &mut canvas,
            -2,
            -2,
            4,
            4,
            pack_argb(0xFF, 0xFF, 0xFE, 0xFF),
            None,
            None,
        );
        for y in 0..4usize {
            for x in 0..4usize {
                let p = buf[y * 4 + x];
                let inside = x < 2 && y < 2;
                if inside {
                    assert_eq!(
                        p >> 24,
                        0xFF,
                        "inside pixel ({x},{y}) should be opaque: {p:#x}"
                    );
                } else {
                    assert_eq!(p, 0, "outside pixel ({x},{y}) untouched");
                }
            }
        }
    }

    #[test]
    fn fill_rect_fully_offscreen_is_noop() {
        let mut buf = vec![0u32; 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        fill_rect(
            &mut canvas,
            100,
            100,
            5,
            5,
            pack_argb(0, 0, 0, 0xFF),
            None,
            None,
        );
        assert!(buf.iter().all(|&p| p == 0));
        let mut damage2 = crate::canvas::Damage::new();
        let mut canvas2 = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage2);
        fill_rect(
            &mut canvas2,
            -10,
            -10,
            5,
            5,
            pack_argb(0, 0, 0, 0xFF),
            None,
            None,
        );
        assert!(buf.iter().all(|&p| p == 0));
    }

    #[test]
    fn fill_rect_transparent_src_into_opaque_dst_is_noop() {
        // Dst is opaque (α=0xFF). Under doctrine: dst-opaque early-out fires for every pixel. Src never read.
        let mut buf = vec![pack_argb(50, 60, 70, 255); 4 * 4];
        let mut damage = crate::canvas::Damage::new();
        let mut canvas = crate::canvas::Canvas::new(&mut buf, 4, 4, &mut damage);
        fill_rect(&mut canvas, 0, 0, 4, 4, pack_argb(255, 0, 0, 0), None, None);
        // Buffer unchanged — every pixel was opaque dst, every under call early-out'd.
        assert!(buf.iter().all(|&p| p == pack_argb(50, 60, 70, 255)));
    }

    #[test]
    fn transform_identity_round_trip() {
        let p = Transform::IDENTITY.apply(3.0, 5.0);
        assert_eq!(p, (3.0, 5.0));
        assert!(Transform::IDENTITY.is_identity());
        let composed = Transform::IDENTITY.then(Transform::IDENTITY);
        assert!(composed.is_identity());
    }

    #[test]
    fn transform_rotate_quarter_turn_maps_x_to_y() {
        // 90° rotation: (1, 0) → (0, 1).
        let r = Transform::rotate(core::f32::consts::FRAC_PI_2);
        let (x, y) = r.apply(1.0, 0.0);
        assert!(x.abs() < 1e-6, "x = {}", x);
        assert!((y - 1.0).abs() < 1e-6, "y = {}", y);
    }

    #[test]
    fn transform_compose_translate_then_rotate() {
        // translate(1,0) then rotate(90°): point (0,0) → (1,0) → (0,1).
        let t =
            Transform::translate(1.0, 0.0).then(Transform::rotate(core::f32::consts::FRAC_PI_2));
        let (x, y) = t.apply(0.0, 0.0);
        assert!(x.abs() < 1e-6 && (y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn transform_aabb_of_rotated_unit_square() {
        // Unit square rotated 45° → bbox is a 1.414×1.414 square centered at origin's diagonal.
        let r = Transform::rotate(core::f32::consts::FRAC_PI_4);
        let (min_x, min_y, max_x, max_y) = r.aabb_of_rect(1.0, 1.0);
        let w = max_x - min_x;
        let h = max_y - min_y;
        assert!((w - core::f32::consts::SQRT_2).abs() < 1e-5, "w = {}", w);
        assert!((h - core::f32::consts::SQRT_2).abs() < 1e-5, "h = {}", h);
    }

    #[test]
    fn quantize_rotation_landings_for_k8() {
        // 10 px font: ceil(π × 10) = 32 (already a multiple of 8). Cardinal angles must land on bins 0, 8, 16, 24.
        let n = 32u16;
        assert_eq!(quantize_rotation(0.0, 10.0, 8), 0);
        assert_eq!(quantize_rotation(core::f32::consts::FRAC_PI_2, 10.0, 8), 8);
        assert_eq!(quantize_rotation(core::f32::consts::PI, 10.0, 8), 16);
        assert_eq!(quantize_rotation(core::f32::consts::PI * 1.5, 10.0, 8), 24);
        // Octant: 45° → bin 4, 135° → 12, etc.
        assert_eq!(quantize_rotation(core::f32::consts::FRAC_PI_4, 10.0, 8), 4);
        // Wrap-around: 2π == 0.
        assert_eq!(quantize_rotation(core::f32::consts::TAU, 10.0, 8), 0);
        // Negative angles wrap correctly via rem_euclid.
        assert_eq!(
            quantize_rotation(-core::f32::consts::FRAC_PI_2, 10.0, 8),
            n - 8
        );
    }

    #[test]
    fn quantize_rotation_scales_with_font_size() {
        // Bigger font → more divisions.
        let small = {
            let raw = crate::math::ceil(core::f32::consts::TAU * 10.0 * 0.5) as u32;
            ((raw + 7) / 8) * 8
        };
        let big = {
            let raw = crate::math::ceil(core::f32::consts::TAU * 100.0 * 0.5) as u32;
            ((raw + 7) / 8) * 8
        };
        assert!(big > small);
        // 100 px font: ceil(π × 100) = 315, ceil to multiple of 8 = 320.
        assert_eq!(big, 320);
        // 10 px font: ceil(π × 10) = 32.
        assert_eq!(small, 32);
    }
}

#[cfg(test)]
mod noise_field_tests {
    use super::*;
    use crate::canvas::Damage;

    fn old(w: usize, h: usize, split: usize, left_scroll: isize, right_scroll: isize, fullscreen: bool) -> Vec<Argb8> {
        let mut px = vec![0u32; w * h];
        let mut d = Damage::new();
        let mut c = Canvas::new(&mut px, w, h, &mut d);
        background_noise_split(&mut c, 0, fullscreen, right_scroll, Some(split), Some(left_scroll), None, None);
        px
    }

    fn new(field: &mut NoiseField, w: usize, h: usize, split: usize, left_scroll: isize, right_scroll: isize, fullscreen: bool) -> Vec<Argb8> {
        field.ensure(w, h, split, None);
        let mut px = vec![0u32; w * h];
        let mut d = Damage::new();
        let mut c = Canvas::new(&mut px, w, h, &mut d);
        field.under_into(&mut c, split, left_scroll, right_scroll, fullscreen, None);
        px
    }

    /// The field is the old texture: centred and split, with the edge and without, and at any scroll on every row the old path drew from inside the field's height.
    #[test]
    fn the_field_is_the_old_noise() {
        let (w, h) = (96, 64);
        let mut field = NoiseField::new();
        for (split, ls, rs, full) in [(w / 2, 0, 0, true), (w / 3, 0, 0, true), (w / 2, 0, 0, false), (w / 3, 5, 17, true), (w / 2, 9, 9, false)] {
            let (a, b) = (old(w, h, split, ls, rs, full), new(&mut field, w, h, split, ls, rs, full));
            for y in 0..h {
                for x in 0..w {
                    let s = if x < split { ls } else { rs };
                    if (y as isize) >= s {
                        assert_eq!(a[y * w + x], b[y * w + x], "split {split} scroll {ls}/{rs} full {full} at {x},{y}");
                    }
                }
            }
        }
        // Back at the centre after the split: no new generation, and the same pixels.
        assert!(!field.ensure(w, h, w / 2, None));
    }

    /// At 16 bits the widened field is the old noise drawn at 16 bits.
    #[test]
    fn the_field_widens_like_the_old_noise() {
        use crate::pixel::G2t16;
        let (w, h) = (72, 40);
        let mut a = vec![0u64; w * h];
        let mut d = Damage::new();
        background_noise_split(&mut Canvas::<G2t16>::new(&mut a, w, h, &mut d), 0, true, 0, Some(w / 3), None, None, None);
        let mut field = NoiseField::new();
        field.ensure(w, h, w / 3, None);
        let mut b = vec![0u64; w * h];
        field.under_into(&mut Canvas::<G2t16>::new(&mut b, w, h, &mut d), w / 3, 0, 0, true, None);
        assert_eq!(a, b);
    }

    /// Rolled by its own height the field is where it started, and every scroll shows every texture row once.
    #[test]
    fn the_field_wraps() {
        let (w, h) = (40, 24);
        let mut field = NoiseField::new();
        let a = new(&mut field, w, h, w / 2, 0, 0, true);
        let b = new(&mut field, w, h, w / 2, h as isize, h as isize, true);
        assert_eq!(a, b);
        let c = new(&mut field, w, h, w / 2, -7, -7, true);
        for y in 0..h {
            assert_eq!(c[y * w..(y + 1) * w], a[((y + 7) % h) * w..((y + 7) % h + 1) * w]);
        }
    }
}
