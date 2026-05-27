//! Pixel-buffer paint primitives. Packed layout is `0xααRRGGBB` (α-byte high, blue low) — top byte is **α (opacity)**, industry-standard direction (`α = 0` transparent, `α = 0xFF` opaque). RGB bytes store darkness (`0 = white`, `255 = black`). See [`crate::pixel`] for the locked convention. All inputs are pixel-space, not RU — convert via [`Viewport::ru_to_px`](crate::Viewport::ru_to_px) before calling.
//!
//! Internal to fluor's render pipeline. Per `## API / Implementation Separation` in AGENT.md, these are not part of the consumer-facing API: future SIMD kernels (NEON, SSE2) will dispatch thru the same entry points without changing call sites in `pane` or `Compositor`.
//!
//! Blend model is α + darkness front-to-back: `dst` is the partial composite already accumulated above (its α-byte = accumulated opacity, RGB = accumulated darkness), `src` is the new layer going behind. Per-pixel early-out fires when `dst >= 0xFF000000` (dst α saturated = opaque) via a single u32 compare. Math throughout is `>> 8` with the `(256 − top_α)` trick — never `/ 255`, never floats in the inner loop. Multi-layer composition is additive on BOTH halves (α adds, darkness adds); the buffer carries the accumulator state across Group boundaries so the early-out chain survives between flatten passes.
//!
//! Every blending primitive accepts an optional [`Clip`] (defaults to full buffer when `None`) and an optional [`AlphaMask`] (full-frame, multiplies into per-pixel alpha for soft clipping — rounded textboxes, squircle pane corners, scroll fades). The clip is resolved once at entry into `(x_min, y_min, x_max, y_max)` loop bounds, so the inner loops carry **zero per-pixel bounds checks** — the math at the entry is the proof. AlphaMask dimensions must equal the buffer's `(buf_w, buf_h)`; mismatches panic per AGENT.md "fail loud."

use crate::canvas::Canvas;
use crate::coord::Coord;

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

    /// Resolve an optional clip — `None` defaults to the full buffer. Used by every primitive at entry so the rest of the function reads from a single concrete `Clip`.
    #[inline]
    pub fn resolve(opt: Option<Clip>, buf_w: usize, buf_h: usize) -> Self {
        match opt {
            Some(c) => c,
            None => Self::buffer(buf_w, buf_h),
        }
    }

    /// Intersect a primitive's `i32` bbox with `opt` (resolved against the buffer extent),
    /// returning integer pixel bounds suitable for `for` loops. Returns `None` if the
    /// intersection is empty (whole primitive is offscreen or fully clipped). Used by every
    /// rasterizer's entry path so the clip story is one call: pass `clip` through, get back
    /// either `(x_start, y_start, x_end, y_end)` to iterate or an early-return signal.
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
        let x_start = (x_min.max(0) as usize).max(c.x_start);
        let y_start = (y_min.max(0) as usize).max(c.y_start);
        let x_end = (x_max.max(0) as usize).min(buf_w).min(c.x_end);
        let y_end = (y_max.max(0) as usize).min(buf_h).min(c.y_end);
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

/// Pack four 8-bit channels into a fluor internal pixel (`0xααRRGGBB`, α + darkness convention). The public API takes visible RGB and opacity α (`a = 255` means fully opaque) so consumer code reads naturally. Inside, RGB is stored as darkness (`255 − channel`); α is stored direct. This is the canonical external→internal boundary.
#[inline]
pub fn pack_argb(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((a as u32) << 24) | (((255 - r) as u32) << 16) | (((255 - g) as u32) << 8) | ((255 - b) as u32)
}

/// Unpack a fluor internal pixel into `(r, g, b, a)` with visible RGB and opacity α — the inverse of [`pack_argb`]. Flips darkness back to visible RGB; α passes through.
#[inline]
pub fn unpack_argb(packed: u32) -> (u8, u8, u8, u8) {
    let a = (packed >> 24) as u8;
    let r = 255 - ((packed >> 16) as u8);
    let g = 255 - ((packed >> 8) as u8);
    let b = 255 - (packed as u8);
    (r, g, b, a)
}

use crate::pixel::Blend;
pub use crate::pixel::BlendMode;

/// Flatten `src` underneath `dst` across the whole slice, pixel-by-pixel, via [`Blend::under`]. `dst` is the partial composite already accumulated above (its α-byte = accumulated opacity); `src` is the new layer going behind. Per-pixel early-out fires when `dst.α == 0xFF`. Both slices must be the same length.
///
/// For `BlendMode::Normal` (the 99% case) the kernel uses 8-wide SIMD + Rayon parallelism.
/// Other modes fall back to scalar per-pixel with Rayon row-chunking still applied — the math
/// just stays in the scalar [`Blend::under`] kernel.
#[inline]
pub fn flatten(dst: &mut [u32], src: &[u32], mode: BlendMode) {
    let n = dst.len().min(src.len());
    if n == 0 {
        return;
    }
    let dst = &mut dst[..n];
    let src = &src[..n];
    const CHUNK: usize = 4096;
    if mode == BlendMode::Normal {
        crate::par::par_chunks(dst, CHUNK, |off, chunk| {
            let src_chunk = &src[off..off + chunk.len()];
            under_chunk_normal_dispatch(chunk, src_chunk);
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

/// Per-chunk Normal-under dispatcher: SIMD with the `simd` feature, scalar fallback otherwise.
/// Output is bit-identical between paths.
#[inline]
pub(crate) fn under_chunk_normal_dispatch(dst: &mut [u32], src: &[u32]) {
    #[cfg(feature = "simd")]
    {
        under_chunk_normal_simd(dst, src);
    }
    #[cfg(not(feature = "simd"))]
    {
        under_chunk_normal_scalar(dst, src);
    }
}

/// 8-wide SIMD chunk kernel for Normal-mode under. Tail (0..7 leftover pixels) runs scalar.
#[cfg(feature = "simd")]
fn under_chunk_normal_simd(dst: &mut [u32], src: &[u32]) {
    use crate::simd::{LANES, u32x8};
    let n = dst.len();
    let mut i = 0;
    while i + LANES <= n {
        let d_arr: [u32; 8] = dst[i..i + LANES].try_into().unwrap();
        let s_arr: [u32; 8] = src[i..i + LANES].try_into().unwrap();
        let out = crate::pixel::under_x8_normal(u32x8::from(d_arr), u32x8::from(s_arr));
        dst[i..i + LANES].copy_from_slice(out.as_array_ref());
        i += LANES;
    }
    while i < n {
        dst[i] = dst[i].under(src[i], BlendMode::Normal);
        i += 1;
    }
}

/// Scalar Normal-under chunk kernel for `--no-default-features` (no `simd`) builds.
#[cfg(not(feature = "simd"))]
fn under_chunk_normal_scalar(dst: &mut [u32], src: &[u32]) {
    for i in 0..dst.len() {
        dst[i] = dst[i].under(src[i], BlendMode::Normal);
    }
}

/// Per-chunk Normal-under dispatcher for a CONSTANT src pixel — what the rasterizer interior
/// fast paths need: every dst pixel composes the same `full_pixel` underneath. Saves the cost
/// of materializing an 8-pixel src array when all lanes are the same value (the SIMD path uses
/// `u32x8::splat` instead).
#[inline]
pub(crate) fn under_chunk_const_dispatch(dst: &mut [u32], src_const: u32) {
    #[cfg(feature = "simd")]
    {
        under_chunk_const_simd(dst, src_const);
    }
    #[cfg(not(feature = "simd"))]
    {
        under_chunk_const_scalar(dst, src_const);
    }
}

/// 8-wide SIMD constant-src under kernel. `src_const` is broadcast to all 8 lanes once outside
/// the inner loop. Tail scalar.
#[cfg(feature = "simd")]
fn under_chunk_const_simd(dst: &mut [u32], src_const: u32) {
    use crate::simd::{LANES, u32x8};
    let src = u32x8::splat(src_const);
    let n = dst.len();
    let mut i = 0;
    while i + LANES <= n {
        let d_arr: [u32; 8] = dst[i..i + LANES].try_into().unwrap();
        let out = crate::pixel::under_x8_normal(u32x8::from(d_arr), src);
        dst[i..i + LANES].copy_from_slice(out.as_array_ref());
        i += LANES;
    }
    while i < n {
        dst[i] = dst[i].under(src_const, BlendMode::Normal);
        i += 1;
    }
}

/// Scalar constant-src under for `--no-default-features` (no `simd`) builds.
#[cfg(not(feature = "simd"))]
fn under_chunk_const_scalar(dst: &mut [u32], src_const: u32) {
    for i in 0..dst.len() {
        dst[i] = dst[i].under(src_const, BlendMode::Normal);
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
/// `mask: Some(&AlphaMask)` multiplies each pixel's mask alpha into `colour`'s opacity (`effective_opacity = colour_opacity * mask_alpha >> 8`) — used for shaped textbox interiors, scroll fades, etc. `mask: None` skips that math entirely.
pub fn fill_rect(
    canvas: &mut Canvas,
    x: isize,
    y: isize,
    rect_w: isize,
    rect_h: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    let clip = Clip::resolve(clip, buf_w, buf_h);
    if let Some(m) = mask {
        assert_mask_matches_buffer(m, buf_w, buf_h);
    }
    let (x_min, y_min, x_max, y_max) = clip_rect(clip, x, y, rect_w, rect_h);
    canvas.damage.add_bounds(x_min, y_min, x_max, y_max);
    let pixels: &mut [u32] = canvas.pixels;
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
            let colour_opacity = 255 - ((colour >> 24) & 0xFF);
            let colour_rgb = colour & 0x00FF_FFFF;
            for row in y_min..y_max {
                let base = row * buf_w;
                for col in x_min..x_max {
                    let idx = base + col;
                    let mask_a = m.pixels[idx] as u32;
                    let effective_opacity = (colour_opacity * mask_a) >> 8;
                    let masked = colour_rgb | ((255 - effective_opacity) << 24);
                    pixels[idx] = pixels[idx].under(masked, BlendMode::Normal);
                }
            }
        }
    }
}

/// Stroke (outline) an axis-aligned rectangle. Draws four filled rect strips along the edges via [`fill_rect`]; corners are not joined separately because at 90° angles the strips meet cleanly.
pub fn stroke_rect(
    canvas: &mut Canvas,
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

/// Fill the buffer with photon's signature procedural background — symmetric organic noise plus speckle. Rows are RNG-independent (each row reseeds from `logical_row`), so the outer loop parallelizes cleanly via [`crate::par::par_rows`]. Mirrored left/right halves like photon. Set `fullscreen=true` to fill the whole buffer; `false` leaves a 1px border for the window edge stroke. `speckle` is an animation counter (constant 0 for static); `scroll_offset` shifts the texture vertically (for content scroll integration).
///
/// SIMD inside a row is intentionally not done — the per-pixel RNG (`rng ^= rng.rotate_left(13).wrapping_add(const)`) is a serial dependency chain; vectorizing it would require N independent RNG streams per lane and would change photon's visual pattern. If profiling shows the per-row scalar work still dominating after Rayon, that's the next lever.
///
/// Clip restricts the row range. Mask isn't supported here (background is bg — masking it would mean "draw nothing where mask is zero" which is the same as just clearing afterward; if you need that, do it explicitly).
pub fn background_noise(
    canvas: &mut Canvas,
    speckle: usize,
    fullscreen: bool,
    scroll_offset: isize,
    clip: Option<Clip>,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if buf_w < 2 || buf_h < 2 {
        return;
    }
    let clip = Clip::resolve(clip, buf_w, buf_h);
    // Clip first, then `fullscreen` further insets by 1 px for the edge hairline.
    let (row_start, row_end, x_start, x_end) = if fullscreen {
        (clip.y_start, clip.y_end, clip.x_start, clip.x_end)
    } else {
        (
            (clip.y_start + 1).min(clip.y_end),
            clip.y_end.saturating_sub(1).max(clip.y_start),
            (clip.x_start + 1).min(clip.x_end),
            clip.x_end.saturating_sub(1).max(clip.x_start),
        )
    };
    if row_start >= row_end || x_start >= x_end {
        return;
    }
    canvas.damage.add_bounds(x_start, row_start, x_end, row_end);
    let pixels: &mut [u32] = canvas.pixels;
    crate::par::par_rows(pixels, buf_w, row_start, row_end, |row_idx, row_pixels| {
        let logical_row = row_idx as isize - scroll_offset;
        background_row(
            row_pixels,
            buf_w,
            logical_row,
            buf_h,
            x_start,
            x_end,
            speckle,
        );
    });
}

#[inline]
fn background_row(
    row_pixels: &mut [u32],
    width: usize,
    logical_row: isize,
    height: usize,
    x_start: usize,
    x_end: usize,
    speckle: usize,
) {
    use crate::theme::{BG_BASE, BG_MASK, BG_SPECKLE};
    // Noise math runs in visible-RGB space (matching photon's reference). At the store site we flip the visible result to stored darkness via XOR, then OR α=0xFF for opaque. Mask off the top byte first to strip any carry from `wrapping_add`.
    const VISIBLE_TO_DARK_FLIP: u32 = 0x00FFFFFF;
    const RGB_MASK: u32 = 0x00FFFFFF;
    const OPAQUE_ALPHA: u32 = 0xFF000000;
    // Hybrid 2-pass: pass 1 fills `noise_buf` with the row's chunk of noise values via the serial RNG/colour chain (branches stay scalar — predicating speckle would cost as much as it saves). Pass 2 hands the chunk to the 8-wide SIMD under-blend kernel (`under_chunk_normal_dispatch`), which composites it into row_pixels at ~1 cycle/pixel amortized. Output is bit-identical to the old straight-scalar version.
    const CHUNK: usize = 64;
    let mut noise_buf = [0u32; CHUNK];
    let ones = 0x0001_0101u32;
    let seed: usize = 0xDEAD_BEEF_0123_4567usize
        ^ (logical_row as usize)
            .wrapping_sub(height / 2)
            .wrapping_mul(0x9E37_79B9_4517_B397);

    // Right half — left to right. Noise composes UNDER existing content (topmost-first): an empty pixel gets the noise; a non-empty pixel (e.g. a topmost rect already painted) has the noise blended behind it.
    let mut rng = seed;
    let mut colour = rng as u32 & BG_MASK;
    let mut x = width / 2;
    while x < x_end {
        let chunk_len = (x_end - x).min(CHUNK);
        for i in 0..chunk_len {
            rng ^= rng.rotate_left(13).wrapping_add(12_345_678_942);
            let adder = rng as u32 & ones;
            if rng.wrapping_add(speckle) < usize::MAX / 256 {
                colour = (rng as u32 >> 8) & BG_SPECKLE;
            } else {
                colour = colour.wrapping_add(adder) & BG_MASK;
                let subtractor = (rng >> 5) as u32 & ones;
                colour = colour.wrapping_sub(subtractor) & BG_MASK;
            }
            noise_buf[i] =
                ((colour.wrapping_add(BG_BASE) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA;
        }
        under_chunk_normal_dispatch(
            &mut row_pixels[x..x + chunk_len],
            &noise_buf[..chunk_len],
        );
        x += chunk_len;
    }

    // Left half — right to left, same RNG seed (mirror), SUB instead of ADD on the rng step. Within each chunk the RNG iterates rightmost-pixel-first; we store into `noise_buf` in left-to-right order (`i = chunk_len-1` down to 0) so the chunk dispatch can scan the buffer sequentially.
    rng = seed;
    colour = rng as u32 & BG_MASK;
    let mut x_hi = width / 2;
    while x_hi > x_start {
        let chunk_lo = x_hi.saturating_sub(CHUNK).max(x_start);
        let chunk_len = x_hi - chunk_lo;
        for i in (0..chunk_len).rev() {
            rng ^= rng.rotate_left(13).wrapping_sub(12_345_678_942);
            let adder = rng as u32 & ones;
            if rng.wrapping_add(speckle) < usize::MAX / 256 {
                colour = (rng as u32 >> 8) & BG_SPECKLE;
            } else {
                colour = colour.wrapping_add(adder) & BG_MASK;
                let subtractor = (rng >> 5) as u32 & ones;
                colour = colour.wrapping_sub(subtractor) & BG_MASK;
            }
            noise_buf[i] =
                ((colour.wrapping_add(BG_BASE) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA;
        }
        under_chunk_normal_dispatch(
            &mut row_pixels[chunk_lo..x_hi],
            &noise_buf[..chunk_len],
        );
        x_hi = chunk_lo;
    }
}

/// Debug toggle that lets the chord `Ctrl/Cmd+Shift+D+P` skip the boundary premultiply at runtime — A/B the Linux premult fix without recompiling. Stays `false` by default.
pub static DEBUG_SKIP_PREMULT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug cycle bound to the `Ctrl/Cmd + Shift + D + A` chord. Three states (rotate each press): `0` = off (normal boundary conversion), `1` = α-as-grayscale (replace each pixel with `(final_α, final_α, final_α, 0xFF)` — inspect alpha distribution), `2` = force-opaque (force every pixel's α to 255 and pass the visible RGB through unmodified — inspect what the kernel produced BEFORE the clip mask + premultiply trimmed it).
pub static DEBUG_SHOW_ALPHA: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
pub const DEBUG_SHOW_ALPHA_OFF: u8 = 0;
pub const DEBUG_SHOW_ALPHA_GRAYSCALE: u8 = 1;
pub const DEBUG_SHOW_ALPHA_FORCE_OPAQUE: u8 = 2;

/// Debug toggle that suppresses chrome layer rasterization (perimeter hairline + future controls + title) so consumers can see the background / panes / textbox underneath without chrome on top. Bound to the `Ctrl/Cmd + Shift + D + C` chord. The clip_mask is still carved at the boundary, so the window-shape trim remains visible. Stays `false` by default.
pub static DEBUG_SKIP_CHROME: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug toggle that suppresses ONLY the controls strip (curves + hairlines + glyphs + dividers + strip-bg fill) while keeping the window perimeter intact. Bound to the `Ctrl/Cmd + Shift + D + X` chord. Useful for isolating perimeter rendering from controls rendering. Stays `false` by default.
pub static DEBUG_SKIP_CONTROLS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Debug toggle that overlays a one-line diagnostic strip across the bottom of the window
/// showing live render-pipeline stats: composite-FPS (= `1.0 / composite_time`, NOT the vsync-
/// capped frame rate) and the cumulative frame counter. Bound to the `Ctrl/Cmd + Shift + D + Q`
/// chord (F was the original choice but Linux window managers eat Ctrl+Shift+F as a system
/// shortcut). The composite-FPS is the actual headroom — a 144 Hz display showing "1240 FPS"
/// means each composite took ~0.8 ms, leaving 6.1 ms of slack against vsync. `false` by default.
pub static DEBUG_SHOW_FPS: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Live diagnostic counters owned by the host's render loop and read by [`draw_debug_strip`].
/// All fields are simple POD; the host updates them every frame when [`DEBUG_SHOW_FPS`] is on
/// and the helper renders them as a single line of text into the bottom-of-window scratch
/// region before the boundary pass runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct DebugStats {
    /// Raw work time of the most recent frame, per stage, in seconds. NOT smoothed — each value is exactly the last measurement so SIMD/Rayon toggles produce immediately legible swings. FPS shown in the strip is `1.0 / stage_secs` per stage and `1.0 / sum` for total.
    pub app_secs: f32,
    pub fill_secs: f32,
    pub finalize_secs: f32,
    pub shadow_secs: f32,
    /// Monotonic count of full frames rendered since the host started. Wraps at `u64::MAX`, effectively never.
    pub frames_rendered: u64,
}

impl DebugStats {
    /// Store this frame's per-stage times and bump the counter. No smoothing — displayed values are exactly what was just measured.
    #[inline]
    pub fn record_frame(&mut self, app: f32, fill: f32, finalize: f32, shadow: f32) {
        self.app_secs = app;
        self.fill_secs = fill;
        self.finalize_secs = finalize;
        self.shadow_secs = shadow;
        self.frames_rendered = self.frames_rendered.wrapping_add(1);
    }

    #[inline]
    pub fn total_secs(&self) -> f32 {
        self.app_secs + self.fill_secs + self.finalize_secs + self.shadow_secs
    }
}

/// Overlay a one-line diagnostic strip across the bottom of `pixels` showing the live stats
/// in [`DebugStats`]. Gated by [`DEBUG_SHOW_FPS`] — the host should check that flag before
/// calling. Paints into the α + darkness scratch buffer BEFORE the boundary pass so the strip
/// flows through `finalize_*` like any other content (no special handling needed downstream).
///
/// The strip is `~24` pixels tall, semi-opaque black background, bright green monospace text
/// (terminal-style for readability against any underlying content). Positioned at the very
/// bottom of `pixels`; clipped to the buffer if the window is too short to fit the strip
/// (returns early in that case — diagnostic, not load-bearing).
#[cfg(feature = "text")]
pub fn draw_debug_strip(
    canvas: &mut Canvas,
    text: &mut crate::text::TextRenderer,
    stats: &DebugStats,
) {
    const STRIP_H: usize = 24;
    const FONT_SIZE: f32 = 13.0;
    let width = canvas.width;
    let height = canvas.height;
    if width == 0 || height < STRIP_H * 2 {
        return;
    }
    // Position the strip centered inside the bottom 1/12th of the window — flush-bottom collided with chrome's bottom bar and the rounded-corner clip carved into the edges.
    let band_top = (height * 11) / 12;
    let strip_y = band_top + ((height - band_top).saturating_sub(STRIP_H)) / 2;

    let app_ms = stats.app_secs * 1000.0;
    let fill_ms = stats.fill_secs * 1000.0;
    let fin_ms = stats.finalize_secs * 1000.0;
    let shadow_ms = stats.shadow_secs * 1000.0;
    let total_ms = stats.total_secs() * 1000.0;

    let stats_line = alloc::format!(
        "app {app_ms:>6.3} ms  fill {fill_ms:>6.3} ms  fin {fin_ms:>6.3} ms  shdw {shadow_ms:>6.3} ms    tot {total_ms:>6.3} ms    Frames {:>7}",
        stats.frames_rendered,
    );

    // Topmost-first ordering: text glyphs paint FIRST so the bar's under() writes are rejected by the glyph pixels, leaving the green characters visible against the black. If the bar were painted first it would fill all strip pixels opaque and every glyph would be eaten.
    let fg = pack_argb(80, 255, 120, 0xFF);
    let text_cy = strip_y as f32 + STRIP_H as f32 * 0.5;
    text.draw_text_center_u32(
        canvas,
        &stats_line,
        width as f32 * 0.5,
        text_cy,
        FONT_SIZE,
        400,
        fg,
        "monospace",
        None,
        None,
        None,
    );

    // Bar fills behind the glyphs (topmost-first: text rows already occupied, bar's under() only lands on the gaps).
    let bg = pack_argb(0, 0, 0, 0xE0);
    draw_rect(
        canvas,
        width as Coord * 0.5,
        strip_y as Coord + STRIP_H as Coord * 0.5,
        width as Coord,
        STRIP_H as Coord,
        bg,
        None,
    );
}

/// Boundary step that finalizes the present buffer for the OS in a **single pass** per pixel — folds the darkness→visible flip, the window-shape clip-mask multiply, the Linux RGB premultiply, and the pack into one go. Walks `pixels` and `clip_mask` in lockstep; both slices must be the same length.
///
/// Per pixel:
/// 1. `v = pixel ^ 0x00FFFFFF` — single XOR flips RGB darkness to visible (255 − dark). α stays as α (already opacity-direction in storage).
/// 2. `final_α = (α × clip_mask_α) >> 8` — multiply with the window-shape clip; trims to the window's actual shape while preserving any partial α the under-chain produced.
/// 3. **Linux only**: premultiply RGB by `final_α / 256`. macOS / other platforms get straight-α output and the OS does its own multiply at composite time.
/// 4. Pack back into `0xααRRGGBB`.
///
/// Debug toggles:
/// * `DEBUG_SHOW_ALPHA` (Ctrl+Shift+D+A): replace each pixel with `(final_α, final_α, final_α, 0xFF)` — grayscale α visualization, opaque so the OS shows it.
/// * `DEBUG_SKIP_PREMULT` (Ctrl+Shift+D+P): skip the Linux RGB×α step.
pub fn finalize_for_os(pixels: &mut [u32], clip_mask: &[u8]) {
    let alpha_mode = DEBUG_SHOW_ALPHA.load(std::sync::atomic::Ordering::Relaxed);
    let n = pixels.len().min(clip_mask.len());
    if n == 0 {
        return;
    }

    // Debug-visualization paths stay scalar — they're rare (toggle-only) and not worth SIMD-izing.
    if alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE || alpha_mode == DEBUG_SHOW_ALPHA_FORCE_OPAQUE {
        finalize_scalar_debug_inplace(&mut pixels[..n], &clip_mask[..n], alpha_mode);
        return;
    }

    // On non-Linux the premult step is a no-op (s = 256 = identity multiply); we route through
    // the same SIMD kernel with `skip_premult = true` so the hot loop is uniform across platforms.
    #[cfg(target_os = "linux")]
    let skip_premult = DEBUG_SKIP_PREMULT.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(target_os = "linux"))]
    let skip_premult = true;

    let pixels = &mut pixels[..n];
    let clip = &clip_mask[..n];

    // Chunk size of 4096 pixels (16 KiB of u32). Large enough to amortize Rayon's ~1 µs
    // task-dispatch overhead against ~10 µs of SIMD work per chunk; small enough that a typical
    // 8-core system pulls hundreds of tasks from a 4K (8M pixel) finalize and load-balances
    // cleanly. On non-Rayon builds, this is just a sequential walk in 4096-pixel windows.
    const CHUNK: usize = 4096;

    crate::par::par_chunks(pixels, CHUNK, |off, chunk| {
        let clip_chunk = &clip[off..off + chunk.len()];
        finalize_chunk_dispatch(chunk, clip_chunk, skip_premult);
    });
}

/// Per-chunk dispatcher: SIMD path when the `simd` feature is enabled, scalar fallback otherwise.
/// Both branches produce bit-identical output (the SIMD path is a straight lane-wise translation
/// of the scalar math, not an approximation).
#[inline]
fn finalize_chunk_dispatch(chunk: &mut [u32], clip: &[u8], skip_premult: bool) {
    #[cfg(feature = "simd")]
    {
        finalize_chunk_simd(chunk, clip, skip_premult);
    }
    #[cfg(not(feature = "simd"))]
    {
        finalize_chunk_scalar(chunk, clip, skip_premult);
    }
}

/// SIMD finalize kernel: 8 pixels per inner iter (u32x8), scalar tail for the leftover 0..7.
/// Same math as [`finalize_chunk_scalar`] lane-by-lane.
#[cfg(feature = "simd")]
fn finalize_chunk_simd(chunk: &mut [u32], clip: &[u8], skip_premult: bool) {
    use crate::simd::{LANES, u32x8};
    let n = chunk.len();
    let mask_ff = u32x8::splat(0xFF);
    let xor_flip = u32x8::splat(0x00FFFFFF);
    let const_256 = u32x8::splat(256);

    let mut i = 0;
    while i + LANES <= n {
        let raw: [u32; 8] = chunk[i..i + LANES].try_into().unwrap();
        let v = u32x8::from(raw) ^ xor_flip;
        let m = u32x8::from([
            clip[i] as u32,
            clip[i + 1] as u32,
            clip[i + 2] as u32,
            clip[i + 3] as u32,
            clip[i + 4] as u32,
            clip[i + 5] as u32,
            clip[i + 6] as u32,
            clip[i + 7] as u32,
        ]);
        let inner_a = (v >> 24) & mask_ff;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { const_256 } else { final_a };
        let r = (((v >> 16) & mask_ff) * s) >> 8;
        let g = (((v >> 8) & mask_ff) * s) >> 8;
        let b = ((v & mask_ff) * s) >> 8;
        let out: u32x8 = (final_a << 24) | (r << 16) | (g << 8) | b;
        chunk[i..i + LANES].copy_from_slice(out.as_array_ref());
        i += LANES;
    }
    // Scalar tail (0..LANES-1 pixels).
    while i < n {
        let v = chunk[i] ^ 0x00FFFFFF;
        let m = clip[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { 256u32 } else { final_a };
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        chunk[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
        i += 1;
    }
}

/// Scalar finalize kernel for `--no-default-features` (no `simd`) builds. Identical math.
#[cfg(not(feature = "simd"))]
fn finalize_chunk_scalar(chunk: &mut [u32], clip: &[u8], skip_premult: bool) {
    for i in 0..chunk.len() {
        let v = chunk[i] ^ 0x00FFFFFF;
        let m = clip[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { 256u32 } else { final_a };
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        chunk[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
    }
}

/// Debug-visualization fallback: GRAYSCALE replaces each pixel with `(final_α, final_α, final_α, 0xFF)`;
/// FORCE_OPAQUE keeps the kernel's visible RGB exactly and sets α=255 (lets you see what the
/// kernel produced BEFORE the clip mask + premult trimmed it). Both stay scalar — they're
/// debug-toggle paths, not on the hot path.
fn finalize_scalar_debug_inplace(pixels: &mut [u32], clip_mask: &[u8], alpha_mode: u8) {
    for i in 0..pixels.len() {
        let v = pixels[i] ^ 0x00FFFFFF;
        let m = clip_mask[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        if alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE {
            pixels[i] = 0xFF000000 | (final_a << 16) | (final_a << 8) | final_a;
        } else {
            // DEBUG_SHOW_ALPHA_FORCE_OPAQUE
            pixels[i] = 0xFF000000 | (v & 0x00FFFFFF);
        }
    }
}

/// 2D wrap-shift the screen buffer in place. Two passes — one per axis. The X pass walks each row, memmoves the row by `dx` columns, and pastes the wrap segment (the pixels that fall off one edge) at the opposite edge. The Y pass treats the whole buffer as a stack of rows, memmoves the rows by `dy` rows, and pastes the wrap row-block at the opposite end.
///
/// Used during in-buffer drag-to-move to skip the chrome / panes / shadow re-rasterization entirely — the window just slides through the screen buffer with its existing pixels, and pixels that fall off any edge wrap around to the opposite end. On drag release the host does one full re-render to clear the wrap artefacts.
///
/// Per-pixel cost: one read + one write, via `slice::copy_within` which lowers to platform `memmove`. No branches inside the inner copy loops — the direction (right/left, up/down) selects between two precomputed `(src_range, dst_offset, wrap_src_range, wrap_dst_range)` tuples, then the copies execute unconditionally.
///
/// `dx` / `dy` are typically bounded by per-frame cursor motion (≪ screen dimensions). For oversized deltas (cursor teleport, stalled frame, multi-monitor span) the signed remainder is used: a shift by exactly `scr_w` is a full wrap = no-op, so `dx = scr_w + 100` is equivalent to `dx = 100` (same wrapped result). Keeps the function panic-free for any input.
pub fn shift_screen_wrap(screen: &mut [u32], scr_w: usize, scr_h: usize, dx: i32, dy: i32) {
    // Normalize to the (-scr_w, +scr_w) range via signed remainder. Direction (sign) is preserved.
    let signed_dx = if scr_w == 0 { 0 } else { dx % (scr_w as i32) };
    let signed_dy = if scr_h == 0 { 0 } else { dy % (scr_h as i32) };
    let nx = signed_dx.unsigned_abs() as usize;
    let ny = signed_dy.unsigned_abs() as usize;

    // X pass — per row. Direction picks the (wrap-source, body-source, body-destination, wrap-destination) tuple once; the per-row work is three unconditional copies. For signed_dx == 0 the function never enters either branch and the X pass is skipped entirely.
    if signed_dx != 0 {
        let mut tmp_x = alloc::vec![0u32; nx];
        let (wrap_src, body_src, body_dst, wrap_dst) = if signed_dx > 0 {
            (scr_w - nx..scr_w, 0..scr_w - nx, nx, 0..nx)
        } else {
            (0..nx, nx..scr_w, 0, scr_w - nx..scr_w)
        };
        for y in 0..scr_h {
            let row_start = y * scr_w;
            let row = &mut screen[row_start..row_start + scr_w];
            tmp_x.copy_from_slice(&row[wrap_src.clone()]);
            row.copy_within(body_src.clone(), body_dst);
            row[wrap_dst.clone()].copy_from_slice(&tmp_x);
        }
    }

    // Y pass — whole rows as a block. Same precomputed-direction pattern, applied to the buffer as a flat row-major slice (one body memmove of contiguous bytes, no per-row loop).
    if signed_dy != 0 {
        let mut tmp_y = alloc::vec![0u32; ny * scr_w];
        let split = (scr_h - ny) * scr_w;
        let (wrap_src, body_src, body_dst, wrap_dst) = if signed_dy > 0 {
            (split..scr_h * scr_w, 0..split, ny * scr_w, 0..ny * scr_w)
        } else {
            (
                0..ny * scr_w,
                ny * scr_w..scr_h * scr_w,
                0,
                split..scr_h * scr_w,
            )
        };
        tmp_y.copy_from_slice(&screen[wrap_src]);
        screen.copy_within(body_src, body_dst);
        screen[wrap_dst].copy_from_slice(&tmp_y);
    }
}

/// Combined finalize + blit: same per-pixel math as [`finalize_for_os`] (XOR darkness→visible, multiply clip_mask into α, Linux RGB×α premultiply) but reads from a `(win_w × win_h)` scratch buffer and writes into a `(scr_w × scr_h)` screen buffer at the offset `(rect_x, rect_y)`. Used by the fullscreen-compositor host path: the consumer renders into the scratch (window-space, contiguous), this function reads scratch + clip_mask once per pixel and writes the OS-ready ARGB into the screen buffer's sub-rect. The scratch buffer is **not mutated** — its α + darkness convention is preserved so a future incremental-rendering path can reuse it across frames without forcing a full re-render.
///
/// Pre-conditions: `rect_x + win_w ≤ scr_w` and `rect_y + win_h ≤ scr_h` (rect fits inside the screen). Caller is responsible for clearing the destination region (typically the whole screen buffer cleared to `0` so pixels outside `rect` stay α=0 and the OS compositor shows whatever's behind us).
///
/// One pass over pixels — same cost per pixel as `finalize_for_os` plus one address calculation for the screen-buffer offset.
pub fn finalize_into_screen(
    scratch: &[u32],
    clip_mask: &[u8],
    win_w: usize,
    win_h: usize,
    screen: &mut [u32],
    scr_w: usize,
    rect_x: i32,
    rect_y: i32,
) {
    let alpha_mode = DEBUG_SHOW_ALPHA.load(std::sync::atomic::Ordering::Relaxed);
    if scr_w == 0 {
        return;
    }
    let scr_h = screen.len() / scr_w;

    // Clip iteration to the rect ∩ screen. WHY: rect_x/y are screen-space coords that can be negative (window partially off-screen left/top) or push past the right/bottom edge when the surface is smaller than expected (initial size mismatch before the first Resized event, monitor change, etc.). PROOF: sy_min/sx_min skip rows/cols above/left of the screen origin; sy_max/sx_max stop at the screen edge. PREVENTS: i32 → usize wrap on negative offsets + writes past `screen.len()`.
    let sy_min = (-rect_y).max(0) as usize;
    let sx_min = (-rect_x).max(0) as usize;
    let sy_max = win_h.min(((scr_h as i32) - rect_y).max(0) as usize);
    let sx_max = win_w.min(((scr_w as i32) - rect_x).max(0) as usize);
    if sy_min >= sy_max || sx_min >= sx_max {
        return;
    }

    // Screen-space dst-row range for the Rayon row-iter wrapper. dst_y_min/max are guaranteed
    // ≥ 0 and < scr_h by the clipping above.
    let dst_y_min = (rect_y + sy_min as i32) as usize;
    let dst_y_max = (rect_y + sy_max as i32) as usize;
    let dst_x_min = (rect_x + sx_min as i32) as usize;
    let row_len = sx_max - sx_min;

    // Debug-visualization paths stay scalar.
    if alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE || alpha_mode == DEBUG_SHOW_ALPHA_FORCE_OPAQUE {
        finalize_into_scalar_debug(
            scratch, clip_mask, screen, scr_w, win_w, alpha_mode, sy_min, sy_max, sx_min, sx_max,
            rect_x, rect_y,
        );
        return;
    }

    #[cfg(target_os = "linux")]
    let skip_premult = DEBUG_SKIP_PREMULT.load(std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(target_os = "linux"))]
    let skip_premult = true;

    crate::par::par_rows(screen, scr_w, dst_y_min, dst_y_max, |dst_y, screen_row| {
        let sy = (dst_y as i32 - rect_y) as usize;
        let scratch_off = sy * win_w + sx_min;
        let src_chunk = &scratch[scratch_off..scratch_off + row_len];
        let clip_chunk = &clip_mask[scratch_off..scratch_off + row_len];
        let dst_chunk = &mut screen_row[dst_x_min..dst_x_min + row_len];
        finalize_into_chunk_dispatch(src_chunk, clip_chunk, dst_chunk, skip_premult);
    });
}

/// Per-row src→dst dispatcher — same shape as [`finalize_chunk_dispatch`] but reading from a
/// separate src buffer rather than in-place.
#[inline]
fn finalize_into_chunk_dispatch(src: &[u32], clip: &[u8], dst: &mut [u32], skip_premult: bool) {
    #[cfg(feature = "simd")]
    {
        finalize_into_chunk_simd(src, clip, dst, skip_premult);
    }
    #[cfg(not(feature = "simd"))]
    {
        finalize_into_chunk_scalar(src, clip, dst, skip_premult);
    }
}

/// SIMD finalize+blit kernel: reads from `src` + `clip`, writes to `dst`. Same math as the
/// in-place [`finalize_chunk_simd`]; the only difference is the read/write split.
#[cfg(feature = "simd")]
fn finalize_into_chunk_simd(src: &[u32], clip: &[u8], dst: &mut [u32], skip_premult: bool) {
    use crate::simd::{LANES, u32x8};
    let n = src.len();
    let mask_ff = u32x8::splat(0xFF);
    let xor_flip = u32x8::splat(0x00FFFFFF);
    let const_256 = u32x8::splat(256);

    let mut i = 0;
    while i + LANES <= n {
        let raw: [u32; 8] = src[i..i + LANES].try_into().unwrap();
        let v = u32x8::from(raw) ^ xor_flip;
        let m = u32x8::from([
            clip[i] as u32,
            clip[i + 1] as u32,
            clip[i + 2] as u32,
            clip[i + 3] as u32,
            clip[i + 4] as u32,
            clip[i + 5] as u32,
            clip[i + 6] as u32,
            clip[i + 7] as u32,
        ]);
        let inner_a = (v >> 24) & mask_ff;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { const_256 } else { final_a };
        let r = (((v >> 16) & mask_ff) * s) >> 8;
        let g = (((v >> 8) & mask_ff) * s) >> 8;
        let b = ((v & mask_ff) * s) >> 8;
        let out: u32x8 = (final_a << 24) | (r << 16) | (g << 8) | b;
        dst[i..i + LANES].copy_from_slice(out.as_array_ref());
        i += LANES;
    }
    while i < n {
        let v = src[i] ^ 0x00FFFFFF;
        let m = clip[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { 256u32 } else { final_a };
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        dst[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
        i += 1;
    }
}

/// Scalar finalize+blit kernel for `--no-default-features` (no `simd`) builds.
#[cfg(not(feature = "simd"))]
fn finalize_into_chunk_scalar(src: &[u32], clip: &[u8], dst: &mut [u32], skip_premult: bool) {
    for i in 0..src.len() {
        let v = src[i] ^ 0x00FFFFFF;
        let m = clip[i] as u32;
        let inner_a = (v >> 24) & 0xFF;
        let final_a = (inner_a * m) >> 8;
        let s = if skip_premult { 256u32 } else { final_a };
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        dst[i] = (final_a << 24) | (r << 16) | (g << 8) | b;
    }
}

/// Debug-visualization fallback for [`finalize_into_screen`]: GRAYSCALE / FORCE_OPAQUE paths
/// stay scalar (debug-toggle, off the hot path). Sequential — Rayon adds no value here.
#[allow(clippy::too_many_arguments)]
fn finalize_into_scalar_debug(
    scratch: &[u32],
    clip_mask: &[u8],
    screen: &mut [u32],
    scr_w: usize,
    win_w: usize,
    alpha_mode: u8,
    sy_min: usize,
    sy_max: usize,
    sx_min: usize,
    sx_max: usize,
    rect_x: i32,
    rect_y: i32,
) {
    for sy in sy_min..sy_max {
        let dst_y = (rect_y + sy as i32) as usize;
        let dst_row = dst_y * scr_w;
        let src_row = sy * win_w;
        for sx in sx_min..sx_max {
            let scratch_idx = src_row + sx;
            if scratch_idx >= scratch.len() || scratch_idx >= clip_mask.len() {
                break;
            }
            let dst_idx = dst_row + (rect_x + sx as i32) as usize;
            let v = scratch[scratch_idx] ^ 0x00FFFFFF;
            let m = clip_mask[scratch_idx] as u32;
            let inner_a = (v >> 24) & 0xFF;
            let final_a = (inner_a * m) >> 8;
            if alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE {
                screen[dst_idx] = 0xFF000000 | (final_a << 16) | (final_a << 8) | final_a;
            } else {
                screen[dst_idx] = 0xFF000000 | (v & 0x00FFFFFF);
            }
        }
    }
}

/// Directional drop shadow via 45-degree diagonal rays cast from each chrome edge pixel. For each row in chrome's y-range, scan leftward from the rectangle's right edge to find that row's rightmost chrome pixel (handles squircle corners where chrome ends inside the rectangle), then cast a single diagonal ray stepping `(x+1, y+1)` per pixel with α decaying by `factor_256` each step. Same pattern for the bottom edge per column. Light source is upper-left → rays trail to the lower-right.
///
/// `factor_256` is the per-pixel decay multiplier in `[1, 255]`: 240 ≈ ×0.9375 (~60-pixel ray), 250 ≈ ×0.9766 (~150-pixel). Caller scales from `effective_span` so shadow length is RU-invariant.
///
/// Visual: each chrome edge pixel emits one diagonal ray. Adjacent rows emit adjacent diagonals → together they cover the shadow region with parallel diagonal stripes. Right-edge rays and bottom-edge rays only overlap in a thin BR corner area ("minimal double taps") and max-compose there.
///
/// AA-edge fix folded in: when the scan finds the chrome edge pixel, if its α is partial (squircle AA), force it to 0xFF in place. Chrome's premult RGB over the implicit black shadow underneath is identity, so opaque-with-premult-RGB blends correctly.
///
/// Per chrome edge pixel: ~1-N scan reads (1 for straight middle rows; up to the squircle inset for corner rows) + up to `log_factor(1/255)` ray writes. Total work ≈ chrome perimeter × shadow-extent.
/// Compose shadow underneath a chrome AA edge pixel in place. Premultiplied-alpha "Under" math: `α_out = α_chrome + α_shadow × (1 − α_chrome / 255)`; premult RGB stays unchanged (shadow's straight RGB is black, so its premult contribution to RGB is zero — chrome's dim premult RGB IS already chrome compositing over black). Result: AA edge α gets boosted by the shadow's local strength, but stays partial when the shadow itself is partial (e.g., the TL halo at 0x20 seed). On bright desktops the AA still blends naturally instead of going crunchy from a force-opaque hack.
///
/// `shadow_seed` is the shadow's full-strength α at the chrome edge (`0x40` for the BR pass, `0x20` for the TL pass). The integer formula uses `(256 − α_chrome)` so the divide by 256 lowers to a `>> 8` shift instead of an actual `/ 255` (which would emit an IDIV on the hot path). Overestimates the true blend by ~1/255 — negligible for AA pixels.
#[inline]
fn blend_aa_edge(screen: &mut [u32], idx: usize, shadow_seed: u32) {
    let p = screen[idx];
    let chrome_a = (p >> 24) & 0xFF;
    // Bounds: new_a = chrome_a × (1 − seed/256) + seed. For seed ≤ 0x40 and chrome_a ≤ 0xFF, max new_a = 0xFF (at chrome_a = 0xFF: 0xFF + 0 = 0xFF; at chrome_a = 0: seed ≤ 0x40). No clamp needed.
    let boost = (shadow_seed * (256 - chrome_a)) >> 8;
    let new_a = chrome_a + boost;
    screen[idx] = (p & 0x00FFFFFF) | (new_a << 24);
}

/// Debug: shadow cells get visible green (R=0, G=255, B=0). The screen buffer is already in OS visible-RGB format at this point (post-finalize), so this lands as bright green directly. Lets every shadow write be eyeballed against the chrome.
const DEBUG_SHADOW_RGB: u32 = 0x0000FF00;

/// Cast a (+1, +1) shadow ray starting from `(start_x + 1, start_y + 1)` with α seeded at `seed` and decaying by `factor_256` per step. Stops at screen edge or α == 0. Writes skipped when the ray pixel is inside `chrome_bbox` — corner rays start at the chrome curve and the first few steps land on chrome interior cells; the chrome owns those cells, so we step the α down (advancing along the diagonal) without writing.
#[inline]
fn cast_ray_dr(
    screen: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    factor_256: u32,
    start_x: usize,
    start_y: usize,
    seed: u32,
    chrome_bbox: (usize, usize, usize, usize),
) {
    let (cx0, cy0, cx1, cy1) = chrome_bbox;
    let mut alpha = seed;
    let mut x = start_x + 1;
    let mut y = start_y + 1;
    while x < scr_w && y < scr_h {
        alpha = (alpha * factor_256) >> 8;
        if alpha == 0 {
            break;
        }
        if !(x >= cx0 && x < cx1 && y >= cy0 && y < cy1) {
            let idx = y * scr_w + x;
            screen[idx] = (alpha << 24) | DEBUG_SHADOW_RGB;
        }
        x += 1;
        y += 1;
    }
}

/// Mirror of [`cast_ray_dr`]: casts a (-1, -1) ray from `(start_x - 1, start_y - 1)`. Bails immediately at the screen origin. Same bbox-guard semantics — writes skipped when the ray pixel is inside `chrome_bbox`.
#[inline]
fn cast_ray_ul(
    screen: &mut [u32],
    scr_w: usize,
    factor_256: u32,
    start_x: usize,
    start_y: usize,
    seed: u32,
    chrome_bbox: (usize, usize, usize, usize),
) {
    if start_x == 0 || start_y == 0 {
        return;
    }
    let (cx0, cy0, cx1, cy1) = chrome_bbox;
    let mut alpha = seed;
    let mut x = start_x - 1;
    let mut y = start_y - 1;
    loop {
        alpha = (alpha * factor_256) >> 8;
        if alpha == 0 {
            break;
        }
        if !(x >= cx0 && x < cx1 && y >= cy0 && y < cy1) {
            let idx = y * scr_w + x;
            screen[idx] = (alpha << 24) | DEBUG_SHADOW_RGB;
        }
        if x == 0 || y == 0 {
            break;
        }
        x -= 1;
        y -= 1;
    }
}

/// Precompute the per-step α decay table for a ray. `alphas[0] = seed`; `alphas[k] = (alphas[k-1] * factor_256) >> 8`. Returns the populated prefix length (where `alphas[len-1] > 0`). Replaces the per-step mul on the hot path with an array lookup. 1024 entries cover the worst case (seed 0x40, factor 254 → ~528 non-zero steps).
#[inline]
fn compute_alphas(seed: u32, factor_256: u32) -> ([u32; 1024], usize) {
    let mut alphas = [0u32; 1024];
    alphas[0] = seed;
    let mut a = seed;
    let mut len = 1;
    while len < 1024 {
        a = (a * factor_256) >> 8;
        if a == 0 {
            break;
        }
        alphas[len] = a;
        len += 1;
    }
    (alphas, len)
}

/// Band fill for a straight right edge: source col = `source_col`, source rows `[y_start..y_end)`, rays going (+1,+1). Each output row r > y_start gets a horizontal run starting at column `source_col + k_min` going right. k = column offset from source_col; alpha = `alphas[k]`. Iterates output-row-major for cache locality.
fn band_fill_right_dr(
    screen: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    source_col: usize,
    y_start: usize,
    y_end: usize,
    alphas: &[u32; 1024],
    alphas_len: usize,
) {
    if y_end <= y_start || alphas_len <= 1 {
        return;
    }
    let max_k = alphas_len - 1;
    let r_end = (y_end + max_k).min(scr_h);
    for r in (y_start + 1)..r_end {
        let k_min = (r + 1).saturating_sub(y_end).max(1);
        let k_max = (r - y_start).min(max_k);
        if k_min > k_max {
            continue;
        }
        let row = r * scr_w;
        let c_start = source_col + k_min;
        let c_end = (source_col + k_max + 1).min(scr_w);
        for c in c_start..c_end {
            let k = c - source_col;
            let a = alphas[k];
            let idx = row + c;
            screen[idx] = (a << 24) | DEBUG_SHADOW_RGB;
        }
    }
}

/// Band fill for a straight bottom edge: source row = `source_row`, source cols `[x_start..x_end)`, rays going (+1,+1). At output row r = `source_row + k`, alpha is constant `alphas[k]` across cols `[x_start + k..x_end + k)`. Sequential writes within each row.
fn band_fill_bottom_dr(
    screen: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    source_row: usize,
    x_start: usize,
    x_end: usize,
    alphas: &[u32; 1024],
    alphas_len: usize,
) {
    if x_end <= x_start || alphas_len <= 1 {
        return;
    }
    let max_k = alphas_len - 1;
    let r_end = (source_row + max_k + 1).min(scr_h);
    for r in (source_row + 1)..r_end {
        let k = r - source_row;
        let a = alphas[k];
        let row = r * scr_w;
        let c_start = x_start + k;
        let c_end = (x_end + k).min(scr_w);
        if c_start >= c_end {
            continue;
        }
        for c in c_start..c_end {
            let idx = row + c;
            screen[idx] = (a << 24) | DEBUG_SHADOW_RGB;
        }
    }
}

/// Band fill for a straight top edge: source row = `source_row`, source cols `[x_start..x_end)`, rays going (-1,-1). At output row r = `source_row - k`, alpha is constant `alphas[k]` across cols `[x_start - k..x_end - k)` (clipped at 0).
fn band_fill_top_ul(
    screen: &mut [u32],
    scr_w: usize,
    source_row: usize,
    x_start: usize,
    x_end: usize,
    alphas: &[u32; 1024],
    alphas_len: usize,
) {
    if x_end <= x_start || alphas_len <= 1 {
        return;
    }
    let max_k = (alphas_len - 1).min(source_row);
    for k in 1..=max_k {
        if x_end <= k {
            continue;
        }
        let a = alphas[k];
        let r = source_row - k;
        let row = r * scr_w;
        let c_start = x_start.saturating_sub(k);
        let c_end = (x_end - k).min(scr_w);
        if c_start >= c_end {
            continue;
        }
        for c in c_start..c_end {
            let idx = row + c;
            screen[idx] = (a << 24) | DEBUG_SHADOW_RGB;
        }
    }
}

/// Band fill for a straight left edge: source col = `source_col`, source rows `[y_start..y_end)`, rays going (-1,-1). Each output row r in `[y_start - max_k..y_end - 1)` gets a horizontal run; column = `source_col - k`. Output rows are written via per-step k loop (column-major within k) but each step's writes are still sequential within their row.
fn band_fill_left_ul(
    screen: &mut [u32],
    scr_w: usize,
    source_col: usize,
    y_start: usize,
    y_end: usize,
    alphas: &[u32; 1024],
    alphas_len: usize,
) {
    if y_end <= y_start || alphas_len <= 1 {
        return;
    }
    let max_k = (alphas_len - 1).min(source_col);
    for k in 1..=max_k {
        if y_end <= k {
            continue;
        }
        let a = alphas[k];
        let c = source_col - k;
        let r_start = y_start.saturating_sub(k);
        let r_end = y_end - k;
        for r in r_start..r_end {
            let idx = r * scr_w + c;
            screen[idx] = (a << 24) | DEBUG_SHADOW_RGB;
        }
    }
}

pub fn paint_shadow(
    screen: &mut [u32],
    scr_w: usize,
    factor_256: u32,
    shadow_seed: u32,
    window_rect: (i32, i32, i32, i32),
) {
    if scr_w == 0 || factor_256 == 0 || factor_256 >= 256 {
        return;
    }
    let scr_h = screen.len() / scr_w;
    let (rx, ry, rw, rh) = window_rect;
    if rw <= 0 || rh <= 0 {
        return;
    }
    let x_chrome_left = rx.max(0) as usize;
    let y_chrome_top = ry.max(0) as usize;
    let x_chrome_end = ((rx + rw).max(0) as usize).min(scr_w);
    let y_chrome_end = ((ry + rh).max(0) as usize).min(scr_h);
    if x_chrome_left >= x_chrome_end || y_chrome_top >= y_chrome_end {
        return;
    }

    // === SEEK + FAN OF DR RAYS ===
    // Phase A: walk (-1, +1) from (right_col, y_chrome_top + 1) to find the first non-zero cell on the TR diagonal. Save (x0, y0).
    // Phase B: cast Ray 0 from (x0, y0): walk (+1, +1), under-blend BLUE while α > 0, direct-assign RED once we cross into transparent.
    // Phase C: for each subsequent ray with start (x0, y_start) where y_start = y0+1, y0+2, ..., y_center: walk (+1, +1):
    //   - opaque (α == 0xFF) ⇒ direct-assign YELLOW (we drifted into chrome interior).
    //   - first partial AA ⇒ flip to GREEN under-blend (stays until transparent).
    //   - first transparent after that ⇒ flip to MAGENTA direct-assign (until taper runs out).
    //   - decay shadow_alpha each step.
    let right_col = x_chrome_end - 1;
    if y_chrome_top + 1 >= y_chrome_end {
        let _ = scr_h;
        let _ = factor_256;
        return;
    }
    let mut x0 = right_col;
    let mut y0 = y_chrome_top + 1;
    let mut found = false;
    loop {
        let a = (screen[y0 * scr_w + x0] >> 24) & 0xFF;
        if a != 0 {
            found = true;
            break;
        }
        if x0 == x_chrome_left || y0 + 1 >= y_chrome_end {
            break;
        }
        x0 -= 1;
        y0 += 1;
    }
    if !found {
        let _ = scr_h;
        let _ = factor_256;
        return;
    }

    // Save TR seed — Phase J (TL shadow's TR-side trace) reuses it.
    let tr_seed_x = x0;
    let tr_seed_y = y0;

    // Cast Ray 0 from the seed cell.
    cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, x0, y0);

    // Phase C (TR half): production black shadow. Each subsequent ray: y += 1, then advance through any opaque cells (chrome interior absorbed by the curve since the last ray). Cache (x, y) — the chrome curve doesn't cut inward in the TR quadrant, so x only stays or grows. Cast the ray from the first non-opaque landing. Stop when y reaches the chrome's vertical center.
    let y_center = (y_chrome_top + y_chrome_end) / 2;
    let mut x = x0;
    let mut y = y0;
    while y < y_center && y + 1 < y_chrome_end {
        y += 1;
        // Fully-opaque chrome cells land at α=0xFE after finalize (premult: (0xFF*0xFF)>>8 = 0xFE).
        while ((screen[y * scr_w + x] >> 24) & 0xFF) >= 0xFE {
            if x + 1 >= scr_w || y + 1 >= scr_h {
                break;
            }
            x += 1;
            y += 1;
        }
        cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, x, y);
    }

    // Phase D (BR half — production black, gated): y += 1, then advance UP-LEFT through any transparent cells silently (no paint — the orange debug paint is now off). Each step of the walk is (-1, -1): one left, one up — the chrome BR curve cuts cells away from both right and bottom as we descend. Stop when our position has moved more LEFT of the BR corner than UP of it — `(right_col - x) > (bot_row - y)`. That's the geometric midpoint of the BR corner arc — halfway done with the BR shadow.
    let bot_row = y_chrome_end - 1;
    while y + 1 < y_chrome_end {
        y += 1;
        while ((screen[y * scr_w + x] >> 24) & 0xFF) == 0 {
            if x == 0 || y == 0 {
                break;
            }
            x -= 1;
            y -= 1;
        }
        cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, x, y);
        let dx_left = right_col.saturating_sub(x);
        let dy_up = bot_row.saturating_sub(y);
        if dx_left > dy_up {
            break;
        }
    }

    // Phase E (continuation past BR midpoint — production black, gated): same algorithm structure as Phase D (outer y += 1, walk UL through transparent), silent walks. Loop bound is `x > x_center` so the trace can continue past `y == bot_row`. Y is clamped to bot_row once reached; if the walk isn't firing at bot_row (straight-bottom AA), advance x manually so the loop progresses.
    let x_center = (x_chrome_left + x_chrome_end) / 2;
    while x > x_center {
        if y + 1 <= bot_row {
            y += 1;
        }
        while ((screen[y * scr_w + x] >> 24) & 0xFF) == 0 {
            if x == 0 || y == 0 {
                break;
            }
            x -= 1;
            y -= 1;
        }
        cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, x, y);
        if y >= bot_row && x > x_center {
            if x == 0 {
                break;
            }
            x -= 1;
        }
    }

    // Phase F (BL mirror — production black, gated): start a fresh trace from the chrome's BL corner going RIGHT to x_center. Mirror of Phase A+B+C: seed via diagonal walk from `(x_chrome_left, bot_row - 1)` (one row up from the bbox BL corner, which is always transparent because the curve cuts it — mirrors Phase A's `(right_col, y_chrome_top + 1)` shift) stepping UR (+1, -1) until first non-zero pixel. Then outer x += 1; when the new cell is opaque chrome interior (>= 0xFE — BL curve descends as x increases), walk DR (+1, +1) silently until we land on a non-opaque cell. Cast shadow ray from each landing. Stop at the horizontal midpoint, meeting Phase E.
    if bot_row == 0 {
        let _ = scr_h;
        return;
    }
    let mut xf = x_chrome_left;
    let mut yf = bot_row - 1;
    let mut found_f = false;
    loop {
        let a = (screen[yf * scr_w + xf] >> 24) & 0xFF;
        if a != 0 {
            found_f = true;
            break;
        }
        if xf + 1 >= x_chrome_end || yf == y_chrome_top {
            break;
        }
        xf += 1;
        yf -= 1;
    }
    if found_f {
        // Save the BL seed — Phase G (TL shadow) starts from the same cell.
        let bl_seed_x = xf;
        let bl_seed_y = yf;
        cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, xf, yf);
        while xf < x_center {
            xf += 1;
            if xf >= scr_w {
                break;
            }
            // Walk DR only while we're still above bot_row — past it we're in straight bottom AA territory where Phase E and Phase F should cast symmetrically at (x, bot_row), not one step past. Without the yf bound, the chrome bottom edge's solid α=0xFE cells would trigger the walk and shift Phase F's casts one cell DR, creating a 1-alpha-step diagonal seam at the Phase E/F meeting point.
            while yf < bot_row && ((screen[yf * scr_w + xf] >> 24) & 0xFF) >= 0xFE {
                if xf + 1 >= scr_w || yf + 1 >= scr_h {
                    break;
                }
                xf += 1;
                yf += 1;
            }
            cast_shadow_ray(screen, scr_w, scr_h, factor_256, shadow_seed, xf, yf);
        }

        // Phase G (TL shadow — bottom half of left edge, production black at HALF the BR seed): same BL seed as Phase F (no fresh diagonal walk), but rays go UL (-1, -1) instead of DR (+1, +1) and start at half strength — the TL ambient occlusion is subtler than the BR drop shadow. Outer y -= 1; when the new cell is opaque chrome interior (BL curve indenting leftward as y decreases), walk UL silently. Cast UL shadow ray with `tl_seed`. Stop at the vertical midpoint.
        let tl_seed = shadow_seed >> 1;
        let mut xg = bl_seed_x;
        let mut yg = bl_seed_y;
        cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xg, yg);
        let y_center = (y_chrome_top + y_chrome_end) / 2;
        while yg > y_center {
            if yg == 0 {
                break;
            }
            yg -= 1;
            while ((screen[yg * scr_w + xg] >> 24) & 0xFF) >= 0xFE {
                if xg == 0 || yg == 0 {
                    break;
                }
                xg -= 1;
                yg -= 1;
            }
            cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xg, yg);
        }

        // Phase H (TL shadow — top half of left edge, production black at tl_seed): continue from Phase G's end going UP past y_center toward the TL corner. Outer y -= 1; when the new cell is transparent (TL curve indents right as we ascend), walk DR (+1, +1) silently — symmetric mirror of Phase E's walk-UL-on-transparent (each step cancels the outer y -= 1 and advances xg right). Cast UL shadow ray, then check stop (cast-before-stop, like Phase D, to avoid skipping the stop-iter's diagonal). Stop at the TL corner's 45° midpoint: `(xg - x_chrome_left) > (yg - y_chrome_top)`.
        while yg > y_chrome_top {
            yg -= 1;
            while ((screen[yg * scr_w + xg] >> 24) & 0xFF) == 0 {
                if xg + 1 >= scr_w || yg + 1 >= scr_h {
                    break;
                }
                xg += 1;
                yg += 1;
            }
            cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xg, yg);
            let dx_right = xg.saturating_sub(x_chrome_left);
            let dy_down = yg.saturating_sub(y_chrome_top);
            if dx_right > dy_down {
                break;
            }
        }

        // Phase I (TL shadow — top edge to x_center, production black at tl_seed): mirror of Phase E. Outer y -= 1 clamped at y_chrome_top; walk DR (+1, +1) on transparent (TL curve cells with chrome above-and-right); if walk doesn't fire at y_chrome_top (straight-top AA), advance x manually. Stop at xg >= x_center.
        while xg < x_center {
            if yg > y_chrome_top {
                yg -= 1;
            }
            while ((screen[yg * scr_w + xg] >> 24) & 0xFF) == 0 {
                if xg + 1 >= scr_w || yg + 1 >= scr_h {
                    break;
                }
                xg += 1;
                yg += 1;
            }
            cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xg, yg);
            if yg <= y_chrome_top && xg < x_center {
                if xg + 1 >= scr_w {
                    break;
                }
                xg += 1;
            }
        }
    }

    // Phase J (TL shadow — TR mirror trace, production black at tl_seed): start at the TR seed (saved from Phase A — no fresh diagonal walk needed, it's the same cell) and trace LEFT to x_center via TR curve + top edge. Mirror of Phase F: outer xi -= 1; when the new cell is opaque chrome interior (TR curve descending leftward = topmost row decreasing), walk UL (-1, -1) — but only while we're still below y_chrome_top (analogous to Phase F's `yf < bot_row` guard) so we don't shift past the chrome top edge at the straight-top AA cells. Cast UL shadow ray from each landing. Stop at the horizontal midpoint, meeting Phase I.
    let tl_seed = shadow_seed >> 1;
    let mut xj = tr_seed_x;
    let mut yj = tr_seed_y;
    cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xj, yj);
    while xj > x_center {
        if xj == 0 {
            break;
        }
        xj -= 1;
        while yj > y_chrome_top && ((screen[yj * scr_w + xj] >> 24) & 0xFF) >= 0xFE {
            if xj == 0 || yj == 0 {
                break;
            }
            xj -= 1;
            yj -= 1;
        }
        cast_shadow_ray_ul(screen, scr_w, factor_256, tl_seed, xj, yj);
    }

    let _ = scr_h;
}

/// Debug-color ray cast. Same shape as [`cast_shadow_ray`] but per cell:
///   * α > 0 → under-blend BLUE (ray 0) or GREEN (rays 1+) — α boost + corresponding channel boost.
///   * α == 0 → direct-assign RED (ray 0) or MAGENTA (rays 1+) premult.
/// Used in development phases to validate trace geometry; swap for `cast_shadow_ray` once the phase is correct.
fn cast_debug_ray(
    screen: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    factor_256: u32,
    mut x: usize,
    mut y: usize,
    is_ray_zero: bool,
) {
    let mut shadow_alpha: u32 = 0xFF;
    loop {
        let idx = y * scr_w + x;
        let p = screen[idx];
        let a = (p >> 24) & 0xFF;
        if a == 0 {
            screen[idx] = if is_ray_zero {
                (shadow_alpha << 24) | (shadow_alpha << 16)
            } else {
                (shadow_alpha << 24) | (shadow_alpha << 16) | shadow_alpha
            };
        } else {
            let cr = (p >> 16) & 0xFF;
            let cg = (p >> 8) & 0xFF;
            let cb = p & 0xFF;
            let cover = 256 - a;
            let boost = (shadow_alpha * cover) >> 8;
            let na = (a + boost).min(0xFF);
            screen[idx] = if is_ray_zero {
                let nb = (cb + boost).min(0xFF);
                (na << 24) | (cr << 16) | (cg << 8) | nb
            } else {
                let ng = (cg + boost).min(0xFF);
                (na << 24) | (cr << 16) | (ng << 8) | cb
            };
        }
        shadow_alpha = (shadow_alpha * factor_256) >> 8;
        if shadow_alpha == 0 || x + 1 >= scr_w || y + 1 >= scr_h {
            break;
        }
        x += 1;
        y += 1;
    }
}

/// Mirror of [`cast_debug_ray`] — UL ray direction. Per cell:
///   * α > 0 → under-blend BLUE (ray 0) or GREEN (rays 1+).
///   * α == 0 → direct-assign RED (ray 0) or MAGENTA (rays 1+) premult.
/// Decay shadow_alpha by factor_256 each step; stops at zero alpha or screen origin (x == 0 || y == 0).
fn cast_debug_ray_ul(
    screen: &mut [u32],
    scr_w: usize,
    factor_256: u32,
    mut x: usize,
    mut y: usize,
    is_ray_zero: bool,
) {
    let mut shadow_alpha: u32 = 0xFF;
    loop {
        let idx = y * scr_w + x;
        let p = screen[idx];
        let a = (p >> 24) & 0xFF;
        if a == 0 {
            screen[idx] = if is_ray_zero {
                (shadow_alpha << 24) | (shadow_alpha << 16)
            } else {
                (shadow_alpha << 24) | (shadow_alpha << 16) | shadow_alpha
            };
        } else {
            let cr = (p >> 16) & 0xFF;
            let cg = (p >> 8) & 0xFF;
            let cb = p & 0xFF;
            let cover = 256 - a;
            let boost = (shadow_alpha * cover) >> 8;
            let na = (a + boost).min(0xFF);
            screen[idx] = if is_ray_zero {
                let nb = (cb + boost).min(0xFF);
                (na << 24) | (cr << 16) | (cg << 8) | nb
            } else {
                let ng = (cg + boost).min(0xFF);
                (na << 24) | (cr << 16) | (ng << 8) | cb
            };
        }
        shadow_alpha = (shadow_alpha * factor_256) >> 8;
        if shadow_alpha == 0 || x == 0 || y == 0 {
            break;
        }
        x -= 1;
        y -= 1;
    }
}

/// Mirror of [`cast_shadow_ray`] — UL direction (-1, -1). Same alpha math; stops at zero alpha or screen origin (x == 0 || y == 0). Used by the TL ambient-occlusion pass which goes up-left from chrome's top + left boundaries.
fn cast_shadow_ray_ul(
    screen: &mut [u32],
    scr_w: usize,
    factor_256: u32,
    shadow_seed: u32,
    mut x: usize,
    mut y: usize,
) {
    let mut shadow_alpha: u32 = shadow_seed;
    loop {
        let idx = y * scr_w + x;
        let p = screen[idx];
        let a = (p >> 24) & 0xFF;
        if a == 0 {
            screen[idx] = shadow_alpha << 24;
        } else {
            let cover = 256 - a;
            let boost = (shadow_alpha * cover) >> 8;
            let na = (a + boost).min(0xFF);
            screen[idx] = (p & 0x00FFFFFF) | (na << 24);
        }
        shadow_alpha = (shadow_alpha * factor_256) >> 8;
        if shadow_alpha == 0 || x == 0 || y == 0 {
            break;
        }
        x -= 1;
        y -= 1;
    }
}

/// Cast one DR shadow ray. Flat loop, single zero check on shadow_alpha or screen edge. Per cell:
///   * α > 0 (chrome AA) → under-blend black: α += shadow_alpha * (256 - α) >> 8; chrome RGB stays (shadow's premult RGB is 0 since visible black).
///   * α == 0 (transparent) → direct-assign black premult: (shadow_alpha << 24), RGB all zero.
/// Decay shadow_alpha by factor_256 each step.
fn cast_shadow_ray(
    screen: &mut [u32],
    scr_w: usize,
    scr_h: usize,
    factor_256: u32,
    shadow_seed: u32,
    mut x: usize,
    mut y: usize,
) {
    let mut shadow_alpha: u32 = shadow_seed;
    loop {
        let idx = y * scr_w + x;
        let p = screen[idx];
        let a = (p >> 24) & 0xFF;
        if a == 0 {
            screen[idx] = shadow_alpha << 24;
        } else {
            let cover = 256 - a;
            let boost = (shadow_alpha * cover) >> 8;
            let na = (a + boost).min(0xFF);
            screen[idx] = (p & 0x00FFFFFF) | (na << 24);
        }
        shadow_alpha = (shadow_alpha * factor_256) >> 8;
        if shadow_alpha == 0 || x + 1 >= scr_w || y + 1 >= scr_h {
            break;
        }
        x += 1;
        y += 1;
    }
}

/// Photon's `blend_rgb_only` helper: weighted RGB blend of two colours with explicit per-pixel weights. Verbatim port from [compositing.rs:5821](/mnt/Octopus/Code/photon/src/ui/compositing.rs#L5821). Used by `draw_window_controls` for AA squircle edges.
pub fn blend_rgb_only(bg_colour: u32, fg_colour: u32, weight_bg: u8, weight_fg: u8) -> u32 {
    let mut bg = bg_colour as u64;
    bg = (bg | (bg << 16)) & 0x0000FFFF0000FFFF;
    bg = (bg | (bg << 8)) & 0x00FF00FF00FF00FF;

    let mut fg = fg_colour as u64;
    fg = (fg | (fg << 16)) & 0x0000FFFF0000FFFF;
    fg = (fg | (fg << 8)) & 0x00FF00FF00FF00FF;

    let mut blended = bg * weight_bg as u64 + fg * weight_fg as u64;
    blended = (blended >> 8) & 0x00FF00FF00FF00FF;
    blended = (blended | (blended >> 8)) & 0x0000FFFF0000FFFF;
    blended = (blended | (blended >> 16)) & 0x00FFFFFF;
    // α + darkness: force opaque (α=0xFF) by setting the top byte. RGB darkness is whatever the blend produced.
    (blended as u32) | 0xFF000000
}

/// Filled rectangle, anti-aliased, axis-aligned. Centered at `(cx, cy)` with fractional dimensions `(rect_w, rect_h)` — sub-pixel position + size both honoured. Color is α + darkness packed (build with [`pack_argb`]); the rect blends UNDER any existing pixel content in the buffer via [`Blend::under`].
///
/// AA: each pixel's coverage = `clamp(0.5 + distance_inside, 0, 1)` against the nearest rect edge. Interior pixels saturate to coverage = 1.0 (full color). Edge pixels get a fraction of the source α. Off-buffer pixels are clipped by the iteration bounds.
///
/// Rule 0: iteration bounds clamp `x_min/y_min ≥ 0` and `x_max/y_max ≤ buffer dim` because the rect can be partially or fully off-screen (caller passes arbitrary cx/cy); without the clamps an i32→usize cast on negative values wraps to a huge number → OOB panic. Coverage clamps to `[0, 1]` because pixels inside the rect have unbounded `d_inside`; without the cap, the `α × coverage` multiply would exceed the 0..255 byte range.
pub fn draw_rect(
    canvas: &mut Canvas,
    cx: Coord,
    cy: Coord,
    rect_w: Coord,
    rect_h: Coord,
    color: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if rect_w <= 0.0 || rect_h <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let hw = rect_w * 0.5;
    let hh = rect_h * 0.5;
    // Bbox via raw f32 → i32 cast (truncate-toward-zero). For positive floats this matches `floor`; for negative floats the `.max(0)` clamp inside `intersect_bbox` erases any off-by-one. The `+1.5` on the upper bounds replaces `ceil(x + 0.5)` — one extra-margin pixel at exact integer values is harmless (its coverage is 0). Avoids the GOT-indirect `libm::floorf/ceilf` calls that LLVM emits when the crate::math wrappers aren't inlined.
    let x_min = (cx - hw - 0.5) as i32;
    let x_max = (cx + hw + 1.5) as i32;
    let y_min = (cy - hh - 0.5) as i32;
    let y_max = (cy + hh + 1.5) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [u32] = canvas.pixels;

    let color_alpha = ((color >> 24) & 0xFF) as Coord;
    let color_dark = color & 0x00FFFFFF;

    // Row-parallel via Rayon when enabled; sequential walk otherwise. Each row is fully
    // independent (no row-to-row data hazards), so this scales linearly with cores up to the
    // memory-bandwidth ceiling. For tiny shapes the Rayon overhead of creating one task is
    // still cheap (sub-microsecond on modern hardware), so no threshold guard needed.
    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy_abs = ((py as Coord + 0.5) - cy).abs();
        let dy_inside = hh - dy_abs;
        for px in x_start..x_end {
            let dx_abs = ((px as Coord + 0.5) - cx).abs();
            let dx_inside = hw - dx_abs;
            let d_inside = dx_inside.min(dy_inside);
            let coverage = (0.5 + d_inside).clamp(0.0, 1.0);
            if coverage <= 0.0 {
                continue;
            }
            let new_alpha = (color_alpha * coverage) as u32;
            if new_alpha == 0 {
                continue;
            }
            let rect_pixel = (new_alpha << 24) | color_dark;
            row[px] = row[px].under(rect_pixel, BlendMode::Normal);
        }
    });
}

/// Filled rectangle, anti-aliased, rotated by `angle` radians around `(cx, cy)`. Positive angle rotates counter-clockwise (standard math convention). Other semantics match [`draw_rect`] — α + darkness colour, AA edges, UNDER-blend onto existing pixel content.
///
/// Scanline + per-pixel edge AA. Per row we analytically solve the px range where each pixel's centre lies in the rect's local-coord interior (`|lx| ≤ hw − ½`, `|ly| ≤ hh − ½`, full coverage) vs the wider "any coverage" band (`|lx| ≤ hw + ½`, `|ly| ≤ hh + ½`, AA needed). The interior loop is a tight UNDER-blend with a precomputed `(α<<24)|RGB` value — no per-pixel rotation, no coverage math, no abs/clamp. Only the two AA strips at the row endpoints (and full-AA rows near top/bottom corners) pay for the 4-edge product coverage. Bounding box is the tight rotated extent `(hw|cos| + hh|sin|, hw|sin| + hh|cos|)`.
///
/// Coverage at AA pixels: product of four `clamp(0.5 + signed_dist_inside, 0, 1)` terms, one per edge. This is the standard analytical AA approximation — exact for axis-aligned, slight over/under-estimate near corners at extreme angles, but visually clean and branch-free.
pub fn draw_rect_rotated(
    canvas: &mut crate::canvas::Canvas,
    cx: Coord,
    cy: Coord,
    rect_w: Coord,
    rect_h: Coord,
    angle: Coord,
    color: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if rect_w <= 0.0 || rect_h <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let hw = rect_w * 0.5;
    let hh = rect_h * 0.5;
    let (sin_a, cos_a) = crate::math::sin_cos(angle);
    let abs_cos = cos_a.abs();
    let abs_sin = sin_a.abs();
    let bbox_hw = hw * abs_cos + hh * abs_sin;
    let bbox_hh = hw * abs_sin + hh * abs_cos;
    // See [`draw_rect`] for the cast-trick rationale.
    let x_min = (cx - bbox_hw - 0.5) as i32;
    let x_max = (cx + bbox_hw + 1.5) as i32;
    let y_min = (cy - bbox_hh - 0.5) as i32;
    let y_max = (cy + bbox_hh + 1.5) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };

    // Damage report: the rasterizer never touches pixels outside this bbox, so this is the tight rect the host needs to compose / present. Reported before the par_rows borrow because damage and pixels both live on canvas; sequencing avoids a split-borrow dance.
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [u32] = canvas.pixels;

    let color_alpha = ((color >> 24) & 0xFF) as Coord;
    let color_dark = color & 0x00FFFFFF;
    let full_alpha = (color >> 24) & 0xFF;
    let full_top = (full_alpha << 24) | color_dark;

    // Per-pixel local-coord deltas (px += 1 → screen dx += 1).
    let dlx = cos_a;
    let dly = -sin_a;

    // Any-coverage band: cov > 0 iff |lx| < hw+½ AND |ly| < hh+½.
    let lx_outer = hw + 0.5;
    let ly_outer = hh + 0.5;
    // Full-coverage interior: cov == 1 iff |lx| ≤ hw−½ AND |ly| ≤ hh−½. Empty if sub-pixel.
    let lx_inner = hw - 0.5;
    let ly_inner = hh - 0.5;
    let has_inner = lx_inner >= 0.0 && ly_inner >= 0.0;

    let x_start_f = x_start as Coord + 0.5;
    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy_row = (py as Coord + 0.5) - cy;
        let dx0 = x_start_f - cx;
        let lx0 = dx0 * cos_a + dy_row * sin_a;
        let ly0 = -dx0 * sin_a + dy_row * cos_a;

        // Outer band (pixels with any coverage at all).
        let (ox_lo, ox_hi) = px_range(lx0, dlx, -lx_outer, lx_outer, x_start, x_end);
        let (oy_lo, oy_hi) = px_range(ly0, dly, -ly_outer, ly_outer, x_start, x_end);
        let outer_lo = ox_lo.max(oy_lo);
        let outer_hi = ox_hi.min(oy_hi);
        if outer_lo >= outer_hi {
            return; // exits this row's closure; next row's task continues.
        }

        // Interior (full-coverage fast-path range). px_range must be called with the same
        // (x_start, x_end) used for lx0/ly0 — i is computed relative to x_start, so passing
        // outer_lo here would shift the interior off by (outer_lo − x_start) pixels.
        let (inner_lo, inner_hi) = if has_inner {
            let (ix_lo, ix_hi) = px_range(lx0, dlx, -lx_inner, lx_inner, x_start, x_end);
            let (iy_lo, iy_hi) = px_range(ly0, dly, -ly_inner, ly_inner, x_start, x_end);
            let lo = ix_lo.max(iy_lo).max(outer_lo);
            let hi = ix_hi.min(iy_hi).min(outer_hi);
            if lo >= hi {
                (outer_hi, outer_hi)
            } else {
                (lo, hi)
            }
        } else {
            (outer_hi, outer_hi)
        };

        // Left AA strip. Topmost-first: rect pixel goes as BOTTOM; the under formula scales
        // it by `consumed = α` into the (initially-empty) buffer, satisfying the dark ≤ α
        // invariant. Subsequent layers (noise, etc.) compose behind via the same call form.
        let mut lx = lx0 + (outer_lo as Coord - x_start as Coord) * dlx;
        let mut ly = ly0 + (outer_lo as Coord - x_start as Coord) * dly;
        for px in outer_lo..inner_lo {
            let cov_r = (0.5 + (hw - lx)).clamp(0.0, 1.0);
            let cov_l = (0.5 + (lx + hw)).clamp(0.0, 1.0);
            let cov_t = (0.5 + (hh - ly)).clamp(0.0, 1.0);
            let cov_b = (0.5 + (ly + hh)).clamp(0.0, 1.0);
            let coverage = cov_r * cov_l * cov_t * cov_b;
            let na = (color_alpha * coverage) as u32;
            if na > 0 {
                let rect_pixel = (na << 24) | color_dark;
                row[px] = row[px].under(rect_pixel, BlendMode::Normal);
            }
            lx += dlx;
            ly += dly;
        }

        // Interior fast path: full alpha, no coverage math. SIMD-blits the row segment via
        // 8-wide under() with `full_top` splatted across all lanes — one `vpsrld + vpmulld +
        // vpaddd` ymm pipeline per 8 pixels, dropping the per-pixel `under()` call overhead.
        under_chunk_const_dispatch(&mut row[inner_lo..inner_hi], full_top);

        // Right AA strip.
        let mut lx = lx0 + (inner_hi as Coord - x_start as Coord) * dlx;
        let mut ly = ly0 + (inner_hi as Coord - x_start as Coord) * dly;
        for px in inner_hi..outer_hi {
            let cov_r = (0.5 + (hw - lx)).clamp(0.0, 1.0);
            let cov_l = (0.5 + (lx + hw)).clamp(0.0, 1.0);
            let cov_t = (0.5 + (hh - ly)).clamp(0.0, 1.0);
            let cov_b = (0.5 + (ly + hh)).clamp(0.0, 1.0);
            let coverage = cov_r * cov_l * cov_t * cov_b;
            let na = (color_alpha * coverage) as u32;
            if na > 0 {
                let rect_pixel = (na << 24) | color_dark;
                row[px] = row[px].under(rect_pixel, BlendMode::Normal);
            }
            lx += dlx;
            ly += dly;
        }
    });
}

/// Pixel-range solver for `lo ≤ v0 + (px − x_start)·dv ≤ hi`. Returns half-open `[px_lo, px_hi)` clamped to `[x_start, x_end]`. Empty range is returned as `(x_end, x_end)`.
fn px_range(
    v0: Coord,
    dv: Coord,
    lo: Coord,
    hi: Coord,
    x_start: usize,
    x_end: usize,
) -> (usize, usize) {
    if dv.abs() < 1e-6 {
        if v0 >= lo && v0 <= hi {
            return (x_start, x_end);
        }
        return (x_end, x_end);
    }
    let i_a = (lo - v0) / dv;
    let i_b = (hi - v0) / dv;
    let i_lo = i_a.min(i_b);
    let i_hi = i_a.max(i_b);
    let i_min = crate::math::ceil(i_lo) as i64;
    let i_max = crate::math::floor(i_hi) as i64;
    let start_i = x_start as i64;
    let end_i = x_end as i64;
    let lo_px = (start_i + i_min).clamp(start_i, end_i);
    let hi_px = (start_i + i_max + 1).clamp(start_i, end_i);
    if lo_px >= hi_px {
        (x_end, x_end)
    } else {
        (lo_px as usize, hi_px as usize)
    }
}

/// Hardware fast reciprocal square root — `1/sqrt(x)` as ONE operation when the target supports it. LLVM won't lower `1.0/sqrtf(x)` to `rsqrtss` without `-ffast-math` (verified in asm: it emits `SQRTSS` + `DIVSS`, ~25 cycles), so we reach for the platform intrinsic explicitly. RSQRTSS gives ~12 bits of precision in ~5 cycles — far beyond what the 8-bit AA coverage byte needs. On non-x86_64 targets, fall back to the portable `1/sqrt` path; LLVM still picks the platform-best sqrt instruction underneath.
///
/// Why "fused": dividing by a square root is so common in graphics (vector normalize, distance ops, lighting) that hardware vendors gave it dedicated silicon — same philosophy as FMA. The unit uses a lookup-table seed + Newton iteration internally; the divide never happens explicitly.
///
/// Precondition: `x > 0`. Inputs that hit `x == 0` should be screened by the caller (we never call this on the AA path for an ellipse center pixel because the cheap `4·f² ≥ |∇f|²` classifier promotes it to the interior fast path first).
#[inline(always)]
fn fast_inv_sqrt(x: Coord) -> Coord {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let v = _mm_set_ss(x);
        let r = _mm_rsqrt_ss(v);
        _mm_cvtss_f32(r)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0 / libm::sqrtf(x)
    }
}

/// Filled circle, anti-aliased. Centered at `(cx, cy)` with fractional radius `r`. Colour is α + darkness packed (build with [`pack_argb`]); composes topmost-first via `pixels[idx].under(circle_pixel, Normal)`.
///
/// Stays in squared-distance space — no `sqrt` anywhere. Pre-squares the inner and outer AA thresholds (`(r−½)²`, `(r+½)²`); per pixel computes `dist² = dx² + dy²` (one mul + one add + one fma) and classifies:
/// * `dist² ≤ r_in²` — full coverage, fast under-blend.
/// * `dist² < r_out²` — AA: coverage `t = (r_out² − dist²) / (r_out² − r_in²)`, a linear ratio in dist² space that approximates the perpendicular-pixel-distance AA `clamp(0.5 + (r − dist), 0, 1)` to ≤ 1 LSB for `r ≥ 2`. Justification: Taylor expansion `dist² − r² ≈ 2r·(dist − r)` so "linear in dist²" ≡ "linear in dist, scaled by 2r" and the divide-by-`diff ≈ 2r` cancels the scaling.
/// * `dist² ≥ r_out²` — outside, skip.
///
/// Sub-pixel radii (`r < ½`) clamp `r_in² = 0`, so the entire circle falls in the AA band — graceful degradation, no special case. Mirrors [`draw_app_icon`]'s orb AA topology.
pub fn draw_circle(
    canvas: &mut Canvas,
    cx: Coord,
    cy: Coord,
    r: Coord,
    color: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if r <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let r_in = (r - 0.5).max(0.0);
    let r_out = r + 0.5;
    let r_in2 = r_in * r_in;
    let r_out2 = r_out * r_out;
    // Per-pixel AA coverage needs `(r_out² − dist²) / diff` — divisor is constant per call, so we precompute the reciprocal once and multiply per pixel instead of dividing per pixel. FDIVSS ≈ 10–14 cycles, MULSS ≈ 3–4 — saves ~10 cycles per AA pixel.
    let inv_diff = 1.0 / (r_out2 - r_in2);
    // See [`draw_rect`] for the cast-trick rationale. `r_out` already includes the half-pixel AA margin so the lower bound is `cx - r_out` (not `... - 0.5`); upper bound adds 1.0 to convert truncation to ceil.
    let x_min = (cx - r_out) as i32;
    let x_max = (cx + r_out + 1.0) as i32;
    let y_min = (cy - r_out) as i32;
    let y_max = (cy + r_out + 1.0) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [u32] = canvas.pixels;

    let color_alpha = ((color >> 24) & 0xFF) as Coord;
    let color_dark = color & 0x00FFFFFF;
    let full_alpha = (color >> 24) & 0xFF;
    let full_pixel = (full_alpha << 24) | color_dark;

    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy = (py as Coord + 0.5) - cy;
        let dy2 = dy * dy;
        for px in x_start..x_end {
            let dx = (px as Coord + 0.5) - cx;
            let dist2 = dx * dx + dy2;
            if dist2 <= r_in2 {
                row[px] = row[px].under(full_pixel, BlendMode::Normal);
            } else if dist2 < r_out2 {
                let t = (r_out2 - dist2) * inv_diff;
                let na = (color_alpha * t) as u32;
                if na > 0 {
                    let circle_pixel = (na << 24) | color_dark;
                    row[px] = row[px].under(circle_pixel, BlendMode::Normal);
                }
            }
        }
    });
}

/// Filled axis-aligned ellipse, anti-aliased. Centered at `(cx, cy)` with fractional radii `(rx, ry)`. Colour is α + darkness packed (build with [`pack_argb`]); composes topmost-first via `pixels[idx].under(ellipse_pixel, Normal)`.
///
/// Per-pixel implicit form `f = (dx/rx)² + (dy/ry)² − 1` (interior `f < 0`, boundary `f = 0`, exterior `f > 0`) plus its gradient-magnitude-squared `|∇f|² = 4·((dx/rx²)² + (dy/ry²)²)`. First-order perpendicular distance ≈ `−f / |∇f|`, so AA coverage = `clamp(½ − f / sqrt(|∇f|²), 0, 1)`.
///
/// The cheap classifier `4·f² > |∇f|²` (no sqrt) tells us whether the pixel is at least ½ unit from the boundary: `f < 0` → deep interior (full coverage, skip the sqrt); `f > 0` → fully outside (skip). The AA band pays one `sqrt` + one `div` per pixel — typically <5% of the bbox total. Pre-computed `1/rx²` and `1/ry²` keep the per-pixel work to four muls + two adds for `f` and another two muls + one add for `|∇f|²`.
///
/// Sub-pixel ellipses (`rx < ½` or `ry < ½`) work without special-casing — every pixel falls in the AA band and gets the per-pixel coverage formula.
pub fn draw_ellipse(
    canvas: &mut Canvas,
    cx: Coord,
    cy: Coord,
    rx: Coord,
    ry: Coord,
    color: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if rx <= 0.0 || ry <= 0.0 || width == 0 || height == 0 {
        return;
    }
    let inv_rx2 = 1.0 / (rx * rx);
    let inv_ry2 = 1.0 / (ry * ry);
    // See [`draw_rect`] for the cast-trick rationale.
    let x_min = (cx - rx - 0.5) as i32;
    let x_max = (cx + rx + 1.5) as i32;
    let y_min = (cy - ry - 0.5) as i32;
    let y_max = (cy + ry + 1.5) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [u32] = canvas.pixels;

    let color_alpha = ((color >> 24) & 0xFF) as Coord;
    let color_dark = color & 0x00FFFFFF;
    let full_alpha = (color >> 24) & 0xFF;
    let full_pixel = (full_alpha << 24) | color_dark;

    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy = (py as Coord + 0.5) - cy;
        let dy2 = dy * dy;
        let dy2_b = dy2 * inv_ry2;
        for px in x_start..x_end {
            let dx = (px as Coord + 0.5) - cx;
            let dx2 = dx * dx;
            let dx2_a = dx2 * inv_rx2;
            let f = dx2_a + dy2_b - 1.0;
            let grad2 = 4.0 * (dx2_a * inv_rx2 + dy2_b * inv_ry2);
            let f_sq_4 = 4.0 * f * f;
            if f <= 0.0 && f_sq_4 >= grad2 {
                row[px] = row[px].under(full_pixel, BlendMode::Normal);
            } else if f < 0.0 || f_sq_4 < grad2 {
                // AA: coverage = clamp(½ − f / sqrt(grad2), 0, 1). Sqrt-and-divide fuse into a
                // single hardware reciprocal-sqrt (RSQRTSS on x86_64) via `fast_inv_sqrt` —
                // ~5 cycles vs ~25 for separate SQRTSS+DIVSS. ~12-bit precision is well over
                // the 8-bit coverage tolerance.
                let inv_g = fast_inv_sqrt(grad2);
                let coverage = (0.5 - f * inv_g).clamp(0.0, 1.0);
                let na = (color_alpha * coverage) as u32;
                if na > 0 {
                    let ellipse_pixel = (na << 24) | color_dark;
                    row[px] = row[px].under(ellipse_pixel, BlendMode::Normal);
                }
            }
        }
    });
}

/// Filled ellipse, anti-aliased, rotated by `angle` radians around `(cx, cy)`. Positive angle rotates counter-clockwise (standard math convention). Other semantics match [`draw_ellipse`] — α + darkness colour, AA edges, UNDER-blend onto existing pixel content.
///
/// Inverse-rotates each pixel's screen-delta into ellipse-local coords (`(lx, ly)`), then runs the same gradient-normalized implicit-form AA as [`draw_ellipse`]. Per-row incremental rotation: precompute `lx, ly` at the row's left pixel, then `lx += cos_a; ly -= sin_a` per pixel — only 2 adds per pixel for the rotation, no per-pixel trig.
///
/// Bounding box uses the tight rotated extent `(rx|cos| + ry|sin|, rx|sin| + ry|cos|)` — same formula as [`draw_rect_rotated`].
pub fn draw_ellipse_rotated(
    canvas: &mut Canvas,
    cx: Coord,
    cy: Coord,
    rx: Coord,
    ry: Coord,
    angle: Coord,
    color: u32,
    clip: Option<Clip>,
) {
    let width = canvas.width;
    let height = canvas.height;
    if rx <= 0. || ry <= 0. || width == 0 || height == 0 {
        return;
    }
    let inv_rx2 = 1. / (rx * rx);
    let inv_ry2 = 1. / (ry * ry);
    let (sin_a, cos_a) = crate::math::sin_cos(angle);
    let abs_cos = cos_a.abs();
    let abs_sin = sin_a.abs();
    let bbox_hw = rx * abs_cos + ry * abs_sin;
    let bbox_hh = rx * abs_sin + ry * abs_cos;
    // See [`draw_rect`] for the cast-trick rationale.
    let x_min = (cx - bbox_hw - 0.5) as i32;
    let x_max = (cx + bbox_hw + 1.5) as i32;
    let y_min = (cy - bbox_hh - 0.5) as i32;
    let y_max = (cy + bbox_hh + 1.5) as i32;
    let Some((x_start, y_start, x_end, y_end)) =
        Clip::intersect_bbox(clip, width, height, x_min, x_max, y_min, y_max)
    else {
        return;
    };
    canvas.damage.add_bounds(x_start, y_start, x_end, y_end);
    let pixels: &mut [u32] = canvas.pixels;

    let color_alpha = ((color >> 24) & 0xFF) as Coord;
    let color_dark = color & 0x00FFFFFF;
    let full_alpha = (color >> 24) & 0xFF;
    let full_pixel = (full_alpha << 24) | color_dark;

    let x_start_f = x_start as Coord + 0.5;
    crate::par::par_rows(pixels, width, y_start, y_end, |py, row| {
        let dy_screen = (py as Coord + 0.5) - cy;
        let dx_screen0 = x_start_f - cx;
        let mut lx = dx_screen0 * cos_a + dy_screen * sin_a;
        let mut ly = -dx_screen0 * sin_a + dy_screen * cos_a;
        for px in x_start..x_end {
            let lx2_a = lx * lx * inv_rx2;
            let ly2_b = ly * ly * inv_ry2;
            let f = lx2_a + ly2_b - 1.;
            let grad2 = 4. * (lx2_a * inv_rx2 + ly2_b * inv_ry2);
            let f_sq_4 = 4. * f * f;
            if f <= 0. && f_sq_4 >= grad2 {
                row[px] = row[px].under(full_pixel, BlendMode::Normal);
            } else if f < 0. || f_sq_4 < grad2 {
                // AA via fused reciprocal-sqrt — see [`draw_ellipse`] for the rationale.
                let inv_g = fast_inv_sqrt(grad2);
                let coverage = (0.5 - f * inv_g).clamp(0., 1.);
                let na = (color_alpha * coverage) as u32;
                if na > 0 {
                    let ellipse_pixel = (na << 24) | color_dark;
                    row[px] = row[px].under(ellipse_pixel, BlendMode::Normal);
                }
            }
            lx += cos_a;
            ly -= sin_a;
        }
    });
}

/// Hard-pixel squircle pill with AA on both the X-axis curve (sides) and Y-axis curve (cap tops/bottoms). Photon's avatar-ring strategy in one call — render twice with different sizes/colors to get a stroke ring.
///
/// Photon-faithful: precompute squircle crossings once (`(inset_px, l_aa, h_aa)` per pixel-row offset into the cap), then walk pure integer indices per corner. Each crossing produces BOTH a vertical-edge AA pixel and a horizontal-edge AA pixel via the squircle's diagonal symmetry — no separate per-col walk needed. Photon's `compositing.rs` `draw_textbox` is the reference; this is the single-color silhouette adaptation (`draw_textbox_pill` keeps the two-tone hairline version photon uses for textboxes).
///
/// `blend_aa_with_existing = false` (outer pass): AA pixels write `(alpha = h_aa, RGB = color)`. Conflicting writes at the diagonal pixel pick MAX h_aa.
///
/// `blend_aa_with_existing = true` (inner pass): AA pixels blend `color_rgb` into the current pixel's RGB by `h_aa`, keeping alpha=255 — produces the proper `fill·h + outside·(1-h)` transition when painted on top of an outer-pass stroke result.
pub fn draw_squircle_pill(
    canvas: &mut Canvas,
    mask: &mut [u8],
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    color: u32,
    squirdleyness: i32,
    blend_aa_with_existing: bool,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    if pill_w <= 0 || pill_h <= 0 {
        return;
    }
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    // Bbox-overlap early-out — pill entirely off-buffer.
    if pill_x + pill_w <= 0 || pill_y + pill_h <= 0 || pill_x >= buf_w_i || pill_y >= buf_h_i {
        return;
    }
    // Damage = pill bbox clipped to buffer. Helpers use the same range internally for their per-row clips.
    {
        let dx0 = pill_x.max(0) as usize;
        let dy0 = pill_y.max(0) as usize;
        let dx1 = (pill_x + pill_w).min(buf_w_i).max(0) as usize;
        let dy1 = (pill_y + pill_h).min(buf_h_i).max(0) as usize;
        canvas.damage.add_bounds(dx0, dy0, dx1, dy1);
    }

    let radius_f = pill_h as f32 * 0.5;
    let radius = (pill_h / 2) as isize;
    // α + darkness: force opaque (α=0xFF) by setting the top byte. RGB darkness intact.
    let solid = (color & 0x00FF_FFFF) | 0xFF000000;
    let color_rgb = color & 0x00FF_FFFF;
    let crossings = squircle_crossings(radius_f, squirdleyness);
    let pixels: &mut [u32] = canvas.pixels;

    // Fast/slow split. Fast path: pill bbox fully inside the buffer → no per-pixel checks. Slow path: partial overhang (scroll/resize transitions) → range clips at the corner-block boundary so each AA write has its row already proven in-buffer.
    let fully_inside =
        pill_x >= 0 && pill_y >= 0 && pill_x + pill_w <= buf_w_i && pill_y + pill_h <= buf_h_i;

    if fully_inside {
        draw_squircle_pill_unclipped(
            pixels,
            mask,
            buf_w,
            pill_x as usize,
            pill_y as usize,
            pill_w as usize,
            pill_h as usize,
            radius as usize,
            &crossings,
            color_rgb,
            solid,
            blend_aa_with_existing,
        );
    } else {
        draw_squircle_pill_clipped(
            pixels,
            mask,
            buf_w,
            buf_h,
            pill_x,
            pill_y,
            pill_w,
            pill_h,
            radius,
            &crossings,
            color_rgb,
            solid,
            blend_aa_with_existing,
        );
    }

    // Center rectangle between the two semicircle caps — both paths share this. Range already
    // clips to [0, buf_w) × [0, buf_h) via .max(0).min(buf), so no per-pixel guard needed.
    let center_x_start = pill_x + radius;
    let center_x_end = pill_x + pill_w - radius;
    if center_x_start < center_x_end {
        let cy_start = pill_y.max(0) as usize;
        let cy_end = (pill_y + pill_h).min(buf_h_i).max(0) as usize;
        let cx_start = center_x_start.max(0) as usize;
        let cx_end = center_x_end.min(buf_w_i).max(0) as usize;
        for fy in cy_start..cy_end {
            let row_base = fy * buf_w;
            for fx in cx_start..cx_end {
                let idx = row_base + fx;
                pixels[idx] = solid;
                mask[idx] = 255;
            }
        }
    }
}

/// Fast-path squircle pill rasterizer. Bounds checks intentionally absent.
///
/// **Rule 0 — WHY/PROOF/PREVENTS:**
/// CALLER GUARANTEES (verified at dispatch in [`draw_squircle_pill`]):
///   - `pill_x + pill_w ≤ buf_w` and `pill_y + pill_h ≤ buf_h` (cast to `usize`).
///   - `pill_w ≥ pill_h ≥ 2` (pill geometry: caps don't overlap).
///
/// PROOF — every AA / fill index is `< pixels.len() == buf_w * buf_h`:
///   - `inset ∈ [0, radius]` (from `squircle_crossings`), `i ∈ [0, radius]` (loop break at diagonal).
///   - All written cols are in `[pill_x, pill_x + pill_w)` (subset of `[0, buf_w)` by caller guarantee).
///   - All written rows are in `[pill_y, pill_y + pill_h)` (subset of `[0, buf_h)` by caller guarantee).
///   - Therefore `row * buf_w + col < buf_h * buf_w = pixels.len()`.
///
/// PREVENTS: nothing — this path runs only when the proof holds. Bounds-check elision lets the compiler emit unchecked stores.
fn draw_squircle_pill_unclipped(
    pixels: &mut [u32],
    mask: &mut [u8],
    buf_w: usize,
    pill_x: usize,
    pill_y: usize,
    pill_w: usize,
    pill_h: usize,
    radius: usize,
    crossings: &[(u16, u8, u8)],
    color_rgb: u32,
    solid: u32,
    blend_aa_with_existing: bool,
) {
    for (i, &(inset, _l, h)) in crossings.iter().enumerate() {
        if inset as usize > i {
            break;
        }
        let inset_us = inset as usize;
        let h_u32 = h as u32;
        for &(flip_x, flip_y) in &[(false, false), (true, false), (false, true), (true, true)] {
            // --- vertical edge: AA pixel + horizontal fill to the diagonal ---
            let v_row = if flip_y {
                pill_y + pill_h - 1 - radius + i
            } else {
                pill_y + radius - i
            };
            let v_aa_col = if flip_x {
                pill_x + pill_w - 1 - inset_us
            } else {
                pill_x + inset_us
            };
            let diag_col = if flip_x {
                pill_x + pill_w - 1 - radius + i
            } else {
                pill_x + radius - i
            };
            let row_base = v_row * buf_w;
            write_aa(
                pixels,
                mask,
                row_base + v_aa_col,
                color_rgb,
                h_u32,
                blend_aa_with_existing,
            );
            let (fx_start, fx_end) = if flip_x {
                (diag_col, v_aa_col)
            } else {
                (v_aa_col + 1, diag_col + 1)
            };
            for fx in fx_start..fx_end {
                let idx = row_base + fx;
                pixels[idx] = solid;
                mask[idx] = 255;
            }

            // --- horizontal edge: AA pixel + vertical fill to the diagonal ---
            let h_col = if flip_x {
                pill_x + pill_w - 1 - radius + i
            } else {
                pill_x + radius - i
            };
            let h_aa_row = if flip_y {
                pill_y + pill_h - 1 - inset_us
            } else {
                pill_y + inset_us
            };
            let diag_row = if flip_y {
                pill_y + pill_h - 1 - radius + i
            } else {
                pill_y + radius - i
            };
            write_aa(
                pixels,
                mask,
                h_aa_row * buf_w + h_col,
                color_rgb,
                h_u32,
                blend_aa_with_existing,
            );
            let (fy_start, fy_end) = if flip_y {
                (diag_row, h_aa_row)
            } else {
                (h_aa_row + 1, diag_row + 1)
            };
            for fy in fy_start..fy_end {
                let idx = fy * buf_w + h_col;
                pixels[idx] = solid;
                mask[idx] = 255;
            }
        }
    }
}

/// Slow-path squircle pill rasterizer. Used when the pill partially overhangs the buffer.
///
/// **Rule 0 — WHY/PROOF/PREVENTS for the bounds checks:**
/// CALLER ALLOWS: `pill_x` may be negative; `pill_x + pill_w` may exceed `buf_w` (same for y). Partial overhang is the design case (scroll-out, resize transitions, off-pane drag).
///
/// PROOF that no closed-form i-range clip suffices: `inset[i]` is non-linear in `i` (squircle curve), so the AA-pixel column `pill_x + inset` can't be cleanly bracketed by a single i-range when the pill straddles `x=0` or `x=buf_w`. Linear-in-`i` coords (rows and `h_col`) ARE clipped at the corner-block level — one branch per corner instead of one per pixel. The inset-dependent AA column gets one inline check.
///
/// PREVENTS: OOB pixel write / slice panic at the math↔buffer boundary when the pill's geometric corner falls outside the buffer.
fn draw_squircle_pill_clipped(
    pixels: &mut [u32],
    mask: &mut [u8],
    buf_w: usize,
    buf_h: usize,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    radius: isize,
    crossings: &[(u16, u8, u8)],
    color_rgb: u32,
    solid: u32,
    blend_aa_with_existing: bool,
) {
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    for (i, &(inset, _l, h)) in crossings.iter().enumerate() {
        if inset as usize > i {
            break;
        }
        let i_iso = i as isize;
        let inset_iso = inset as isize;
        let h_u32 = h as u32;
        for &(flip_x, flip_y) in &[(false, false), (true, false), (false, true), (true, true)] {
            let v_row = if flip_y {
                pill_y + pill_h - 1 - radius + i_iso
            } else {
                pill_y + radius - i_iso
            };
            let h_col = if flip_x {
                pill_x + pill_w - 1 - radius + i_iso
            } else {
                pill_x + radius - i_iso
            };

            // --- Vertical edge: row constraint hoisted ---
            if v_row >= 0 && v_row < buf_h_i {
                let row_base = v_row as usize * buf_w;
                let v_aa_col = if flip_x {
                    pill_x + pill_w - 1 - inset_iso
                } else {
                    pill_x + inset_iso
                };
                let diag_col = h_col;
                // AA pixel: inline col check (inset is nonlinear in i; can't pre-clip).
                if v_aa_col >= 0 && v_aa_col < buf_w_i {
                    write_aa(
                        pixels,
                        mask,
                        row_base + v_aa_col as usize,
                        color_rgb,
                        h_u32,
                        blend_aa_with_existing,
                    );
                }
                // Fill: range self-clips to [0, buf_w).
                let (fx_start, fx_end) = if flip_x {
                    (diag_col, v_aa_col)
                } else {
                    (v_aa_col + 1, diag_col + 1)
                };
                let fs = fx_start.max(0) as usize;
                let fe = fx_end.max(0).min(buf_w_i) as usize;
                for fx in fs..fe {
                    let idx = row_base + fx;
                    pixels[idx] = solid;
                    mask[idx] = 255;
                }
            }

            // --- Horizontal edge: column constraint hoisted ---
            if h_col >= 0 && h_col < buf_w_i {
                let col_us = h_col as usize;
                let h_aa_row = if flip_y {
                    pill_y + pill_h - 1 - inset_iso
                } else {
                    pill_y + inset_iso
                };
                let diag_row = v_row;
                if h_aa_row >= 0 && h_aa_row < buf_h_i {
                    write_aa(
                        pixels,
                        mask,
                        h_aa_row as usize * buf_w + col_us,
                        color_rgb,
                        h_u32,
                        blend_aa_with_existing,
                    );
                }
                let (fy_start, fy_end) = if flip_y {
                    (diag_row, h_aa_row)
                } else {
                    (h_aa_row + 1, diag_row + 1)
                };
                let fs = fy_start.max(0) as usize;
                let fe = fy_end.max(0).min(buf_h_i) as usize;
                for fy in fs..fe {
                    let idx = fy * buf_w + col_us;
                    pixels[idx] = solid;
                    mask[idx] = 255;
                }
            }
        }
    }
}

/// Generate photon's squircle crossings: one entry per pixel-row offset from the cap edge into the diagonal. Each entry is `(inset_int, l_aa, h_aa)` where `inset_int` is the integer column offset where the curve crosses that row, and `l/h_aa = sqrt(frac(inset))*256` / `sqrt(1-frac(inset))*256` are the perceptual AA weights (low = outside fraction, high = inside fraction). Verbatim port of photon's loop in `compositing.rs::draw_textbox`.
pub fn squircle_crossings(radius: f32, squirdleyness: i32) -> alloc::vec::Vec<(u16, u8, u8)> {
    let mut crossings: alloc::vec::Vec<(u16, u8, u8)> = alloc::vec::Vec::new();
    let mut offset = 0f32;
    loop {
        let y_norm = offset / radius;
        let x_norm = crate::math::powf(
            1. - crate::math::powi(y_norm, squirdleyness),
            1. / squirdleyness as f32,
        );
        let cx = x_norm * radius;
        let inset = radius - cx;
        if inset >= 0. {
            let l = (crate::math::sqrt(crate::math::fract(inset)) * 256.) as u8;
            let h = (crate::math::sqrt(1. - crate::math::fract(inset)) * 256.) as u8;
            crossings.push((inset as u16, l, h));
        }
        if cx < offset {
            break;
        }
        offset += 1.0;
    }
    crossings
}

/// AA write at a proven-in-buffer index. Caller proves `idx < pixels.len() == mask.len()`; this function does no bounds checks. Outer pass MAX-combines alpha so the vertical-edge and horizontal-edge AA writes don't fight at the diagonal pixel. Inner pass blends RGB into the existing pixel.
#[inline]
fn write_aa(
    pixels: &mut [u32],
    mask: &mut [u8],
    idx: usize,
    color_rgb: u32,
    h_aa: u32,
    blend_aa_with_existing: bool,
) {
    if blend_aa_with_existing {
        let curr = pixels[idx];
        let curr_r = (curr >> 16) & 0xFF;
        let curr_g = (curr >> 8) & 0xFF;
        let curr_b = curr & 0xFF;
        let new_r = (color_rgb >> 16) & 0xFF;
        let new_g = (color_rgb >> 8) & 0xFF;
        let new_b = color_rgb & 0xFF;
        let inv = 256 - h_aa;
        let br = (curr_r * inv + new_r * h_aa) >> 8;
        let bg = (curr_g * inv + new_g * h_aa) >> 8;
        let bb = (curr_b * inv + new_b * h_aa) >> 8;
        pixels[idx] = (curr & 0xFF00_0000) | (br << 16) | (bg << 8) | bb;
        mask[idx] = 255;
    } else {
        // α + darkness: top byte is α (opacity). MORE OPAQUE write wins (higher α). h_aa is AA coverage (0..255, higher = more covered) which IS α in this convention.
        let new_a = h_aa;
        let existing_a = (pixels[idx] >> 24) & 0xFF;
        if new_a > existing_a {
            pixels[idx] = (new_a << 24) | color_rgb;
            mask[idx] = h_aa as u8;
        }
    }
}

/// Photon's squircle inset formula — single row, parameterized by `squirdleyness`. Returns the curve's inset (distance from the bbox edge to the leftmost / rightmost inside pixel) for a row at `y_from_center` rows above or below the squircle's vertical center.
///
/// `squirdleyness = 2` → circle. `squirdleyness = 3` → photon's textbox pill default (slightly flatter than a circle). Higher = more rectangular. Both `draw_textbox_pill` (AA path) and the textbox widget's hard-pixel renderer route thru this so any tweak to the curve math flows thru both code paths.
///
/// Identical to the per-iteration formula at photon's [`compositing.rs:4567-4570`](/mnt/Octopus/Code/photon/src/ui/compositing.rs).
#[inline]
pub fn squircle_inset(y_from_center: f32, radius: f32, squirdleyness: i32) -> f32 {
    let y_norm = (y_from_center / radius).min(1.0);
    let x_norm = crate::math::powf(
        1.0 - crate::math::powi(y_norm, squirdleyness),
        1.0 / squirdleyness as f32,
    );
    radius - x_norm * radius
}

/// Draw a pill-shaped textbox (semicircular ends, squirdleyness=3) with two-tone AA edges and generate an alpha mask. Ported from photon's `draw_textbox` in compositing.rs.
///
/// Writes into `pixels` (fill + AA edges) and `mask` (0 outside, 255 interior, AA values on edges). The mask is used downstream by the glow effect, text clipping, and blinkey. `center_x`/`center_y` are pixel coordinates; `box_width`/`box_height` in pixels.
///
/// **Rule 0 — WHY/PROOF/PREVENTS:** `center_y` is signed because scroll can push the textbox off-screen (negative Y). The function clips to `[0, buf_h)` using signed arithmetic. PREVENTS: out-of-bounds pixel access on partial visibility.
pub fn draw_textbox_pill(
    canvas: &mut Canvas,
    mask: &mut [u8],
    center_x: usize,
    center_y: isize,
    box_width: usize,
    box_height: usize,
) {
    use crate::theme;

    let buf_w = canvas.width;
    let buf_h = canvas.height;
    let height = buf_h;
    let height_signed = height as isize;

    let x = center_x.wrapping_sub(box_width / 2);
    let y_signed = center_y - (box_height as isize / 2);
    let y = if y_signed >= 0 {
        y_signed as usize
    } else {
        0usize.wrapping_sub((-y_signed) as usize)
    };

    // Early out if entirely off-screen.
    let box_top = y_signed;
    let box_bottom = y_signed + box_height as isize;
    if box_bottom <= 0 || box_top >= height_signed {
        return;
    }
    // Damage = textbox pill bbox clipped to buffer.
    {
        let buf_w_i = buf_w as isize;
        let x_signed = center_x as isize - (box_width as isize / 2);
        let dx0 = x_signed.max(0).min(buf_w_i) as usize;
        let dy0 = box_top.max(0).min(height_signed) as usize;
        let dx1 = (x_signed + box_width as isize).max(0).min(buf_w_i) as usize;
        let dy1 = box_bottom.max(0).min(height_signed) as usize;
        canvas.damage.add_bounds(dx0, dy0, dx1, dy1);
    }
    let pixels: &mut [u32] = canvas.pixels;

    let light = theme::TEXTBOX_LIGHT_EDGE;
    let shadow = theme::TEXTBOX_SHADOW_EDGE;
    let fill = theme::TEXTBOX_FILL;
    let radius = box_height as f32 / 2.0;
    let squirdleyness = 3i32;

    // Generate squircle crossings from edge toward diagonal.
    let mut crossings: alloc::vec::Vec<(u16, u8, u8)> = alloc::vec::Vec::new();
    let mut offset = 0f32;
    loop {
        let y_norm = offset / radius;
        let x_norm = crate::math::powf(
            1.0 - crate::math::powi(y_norm, squirdleyness),
            1.0 / squirdleyness as f32,
        );
        let cx = x_norm * radius;
        let inset = radius - cx;
        if inset >= 0.0 {
            let l = (crate::math::sqrt(crate::math::fract(inset)) * 256.0) as u8;
            let h = (crate::math::sqrt(1.0 - crate::math::fract(inset)) * 256.0) as u8;
            crossings.push((inset as u16, l, h));
        }
        if cx < offset {
            break;
        }
        offset += 1.0;
    }

    // Helper: draw one corner's crossings.
    // flip_x: false=left, true=right. flip_y: false=top, true=bottom.
    // Each crossing generates: vertical edge pixel, horizontal edge pixel, diagonal fill between them.
    for (i, &(inset, l, h)) in crossings.iter().enumerate() {
        if inset as usize > i {
            break;
        }

        // --- Top-left corner ---
        {
            let py = y.wrapping_add(radius as usize).wrapping_sub(i);
            let px = x.wrapping_add(inset as usize);
            if py < height && px < buf_w {
                let idx = py * buf_w + px;
                pixels[idx] = blend_rgb_only(pixels[idx], light, l, h);
            }
            let px1 = px.wrapping_add(1);
            if py < height && px1 < buf_w {
                let idx = py * buf_w + px1;
                pixels[idx] = blend_rgb_only(light, fill, l, h);
                mask[idx] = h;
            }
            if py < height {
                let diag_x = x.wrapping_add(radius as usize).wrapping_sub(i).min(buf_w);
                for fill_x in px.wrapping_add(2)..=diag_x {
                    if fill_x >= buf_w {
                        continue;
                    }
                    let idx = py * buf_w + fill_x;
                    pixels[idx] = fill;
                    mask[idx] = 255;
                }
            }
            let hx = x.wrapping_add(radius as usize).wrapping_sub(i);
            let hy = y.wrapping_add(inset as usize);
            if hy < height && hx < buf_w {
                let idx = hy * buf_w + hx;
                pixels[idx] = blend_rgb_only(pixels[idx], light, l, h);
            }
            let hy1 = hy.wrapping_add(1);
            if hy1 < height && hx < buf_w {
                let idx = hy1 * buf_w + hx;
                pixels[idx] = blend_rgb_only(light, fill, l, h);
                mask[idx] = h;
            }
            let hy_s = y_signed + inset as isize;
            let diag_y_s = y_signed + radius as isize - i as isize;
            let fs = (hy_s + 2).max(0).min(height_signed) as usize;
            let fe = diag_y_s.max(0).min(height_signed) as usize;
            if hx < buf_w && fs < fe {
                for fy in fs..fe {
                    let idx = fy * buf_w + hx;
                    pixels[idx] = fill;
                    mask[idx] = 255;
                }
            }
        }

        // --- Top-right corner ---
        {
            let py = y.wrapping_add(radius as usize).wrapping_sub(i);
            let px = x
                .wrapping_add(box_width)
                .wrapping_sub(1)
                .wrapping_sub(inset as usize);
            if py < height && px < buf_w {
                let idx = py * buf_w + px;
                pixels[idx] = blend_rgb_only(pixels[idx], shadow, l, h);
            }
            let px1 = px.wrapping_sub(1);
            if py < height && px1 < buf_w {
                let idx = py * buf_w + px1;
                pixels[idx] = blend_rgb_only(shadow, fill, l, h);
                mask[idx] = h;
            }
            if py < height {
                let diag_x = x
                    .wrapping_add(box_width)
                    .wrapping_sub(1)
                    .wrapping_sub(radius as usize)
                    .wrapping_add(i);
                for fill_x in diag_x..px.wrapping_sub(1) {
                    if fill_x >= buf_w {
                        continue;
                    }
                    let idx = py * buf_w + fill_x;
                    pixels[idx] = fill;
                    mask[idx] = 255;
                }
            }
            let hx = x
                .wrapping_add(box_width)
                .wrapping_sub(1)
                .wrapping_sub(radius as usize)
                .wrapping_add(i);
            let hy = y.wrapping_add(inset as usize);
            if hy < height && hx < buf_w {
                let idx = hy * buf_w + hx;
                pixels[idx] = blend_rgb_only(pixels[idx], light, l, h);
            }
            let hy1 = hy.wrapping_add(1);
            if hy1 < height && hx < buf_w {
                let idx = hy1 * buf_w + hx;
                pixels[idx] = blend_rgb_only(light, fill, l, h);
                mask[idx] = h;
            }
            let hy_s = y_signed + inset as isize;
            let diag_y_s = y_signed + radius as isize - i as isize;
            let fs = (hy_s + 2).max(0).min(height_signed) as usize;
            let fe = diag_y_s.max(0).min(height_signed) as usize;
            if hx < buf_w && fs < fe {
                for fy in fs..fe {
                    let idx = fy * buf_w + hx;
                    pixels[idx] = fill;
                    mask[idx] = 255;
                }
            }
        }

        // --- Bottom-left corner ---
        {
            let py = y
                .wrapping_add(box_height)
                .wrapping_sub(radius as usize)
                .wrapping_add(i);
            let px = x.wrapping_add(inset as usize);
            if py < height && px < buf_w {
                let idx = py * buf_w + px;
                pixels[idx] = blend_rgb_only(pixels[idx], light, l, h);
            }
            let px1 = px.wrapping_add(1);
            if py < height && px1 < buf_w {
                let idx = py * buf_w + px1;
                pixels[idx] = blend_rgb_only(light, fill, l, h);
                mask[idx] = h;
            }
            if py < height {
                let diag_x = x.wrapping_add(radius as usize).wrapping_sub(i).min(buf_w);
                for fill_x in px.wrapping_add(2)..=diag_x {
                    if fill_x >= buf_w {
                        continue;
                    }
                    let idx = py * buf_w + fill_x;
                    pixels[idx] = fill;
                    mask[idx] = 255;
                }
            }
            let hx = x.wrapping_add(radius as usize).wrapping_sub(i);
            let hy = y.wrapping_add(box_height).wrapping_sub(inset as usize);
            if hy < height && hx < buf_w {
                let idx = hy * buf_w + hx;
                pixels[idx] = blend_rgb_only(pixels[idx], shadow, l, h);
            }
            let hy1 = hy.wrapping_sub(1);
            if hy1 < height && hx < buf_w {
                let idx = hy1 * buf_w + hx;
                pixels[idx] = blend_rgb_only(shadow, fill, l, h);
                mask[idx] = h;
            }
            let diag_y_s = y_signed + box_height as isize - radius as isize + i as isize;
            let hy_s = y_signed + box_height as isize - inset as isize;
            let fs = (diag_y_s + 1).max(0).min(height_signed) as usize;
            let fe = (hy_s - 1).max(0).min(height_signed) as usize;
            if hx < buf_w && fs < fe {
                for fy in fs..fe {
                    let idx = fy * buf_w + hx;
                    pixels[idx] = fill;
                    mask[idx] = 255;
                }
            }
        }

        // --- Bottom-right corner ---
        {
            let py = y
                .wrapping_add(box_height)
                .wrapping_sub(radius as usize)
                .wrapping_add(i);
            let px = x
                .wrapping_add(box_width)
                .wrapping_sub(1)
                .wrapping_sub(inset as usize);
            if py < height && px < buf_w {
                let idx = py * buf_w + px;
                pixels[idx] = blend_rgb_only(pixels[idx], shadow, l, h);
            }
            let px1 = px.wrapping_sub(1);
            if py < height && px1 < buf_w {
                let idx = py * buf_w + px1;
                pixels[idx] = blend_rgb_only(shadow, fill, l, h);
                mask[idx] = h;
            }
            if py < height {
                let diag_x = x
                    .wrapping_add(box_width)
                    .wrapping_sub(1)
                    .wrapping_sub(radius as usize)
                    .wrapping_add(i);
                for fill_x in diag_x..px.wrapping_sub(1) {
                    if fill_x >= buf_w {
                        continue;
                    }
                    let idx = py * buf_w + fill_x;
                    pixels[idx] = fill;
                    mask[idx] = 255;
                }
            }
            let hx = x
                .wrapping_add(box_width)
                .wrapping_sub(1)
                .wrapping_sub(radius as usize)
                .wrapping_add(i);
            let hy = y.wrapping_add(box_height).wrapping_sub(inset as usize);
            if hy < height && hx < buf_w {
                let idx = hy * buf_w + hx;
                pixels[idx] = blend_rgb_only(pixels[idx], shadow, l, h);
            }
            let hy1 = hy.wrapping_sub(1);
            if hy1 < height && hx < buf_w {
                let idx = hy1 * buf_w + hx;
                pixels[idx] = blend_rgb_only(shadow, fill, l, h);
                mask[idx] = h;
            }
            let diag_y_s = y_signed + box_height as isize - radius as isize + i as isize;
            let hy_s = y_signed + box_height as isize - inset as isize;
            let fs = (diag_y_s + 1).max(0).min(height_signed) as usize;
            let fe = (hy_s - 1).max(0).min(height_signed) as usize;
            if hx < buf_w && fs < fe {
                for fy in fs..fe {
                    let idx = fy * buf_w + hx;
                    pixels[idx] = fill;
                    mask[idx] = 255;
                }
            }
        }
    }

    // Fill center and straight edges.
    let radius_int = radius as isize;
    if box_width > box_height {
        // Fat box: top/bottom straight edges + center fill.
        let left_edge = x.wrapping_add(radius as usize);
        let right_edge = x.wrapping_add(box_width).wrapping_sub(radius as usize);
        // Top edge hairline.
        if y_signed >= 0 && y_signed < height_signed {
            let top_y = y_signed as usize;
            for px in left_edge..right_edge {
                if px >= buf_w {
                    continue;
                }
                pixels[top_y * buf_w + px] = light;
            }
        }
        // Bottom edge hairline.
        let bot_y_s = y_signed + box_height as isize;
        if bot_y_s >= 0 && bot_y_s < height_signed {
            let bot_y = bot_y_s as usize;
            for px in left_edge..right_edge {
                if px >= buf_w {
                    continue;
                }
                pixels[bot_y * buf_w + px] = shadow;
            }
        }
        // Center fill.
        let fill_top = (y_signed + 1).max(0).min(height_signed) as usize;
        let fill_bot = (y_signed + box_height as isize).max(0).min(height_signed) as usize;
        for py in fill_top..fill_bot {
            for px in left_edge..right_edge {
                if px >= buf_w {
                    continue;
                }
                let idx = py * buf_w + px;
                pixels[idx] = fill;
                mask[idx] = 255;
            }
        }
    } else {
        // Skinny box: left/right straight edges + center fill.
        let top_edge = (y_signed + radius_int).max(0).min(height_signed) as usize;
        let bot_edge = (y_signed + box_height as isize - radius_int)
            .max(0)
            .min(height_signed) as usize;
        if x < buf_w {
            for py in top_edge..bot_edge {
                pixels[py * buf_w + x] = light;
            }
        }
        let right_x = x.wrapping_add(box_width);
        if right_x < buf_w {
            for py in top_edge..bot_edge {
                pixels[py * buf_w + right_x] = shadow;
            }
        }
        for py in top_edge..bot_edge {
            for px in x.wrapping_add(1)..x.wrapping_add(box_width).wrapping_sub(1) {
                if px >= buf_w {
                    continue;
                }
                let idx = py * buf_w + px;
                pixels[idx] = fill;
                mask[idx] = 255;
            }
        }
    }
}

/// Add or remove a wave-shaped cursor (blinkey) at `(bx, by)` with `height` pixels tall.
/// `top_bright`: true = intensity concentrated at top, false = bottom.
/// `add`: true = additive blend, false = subtractive (undo).
///
/// Wave polynomial: `(1 - t²)(1 ∓ t)²` where t ∈ [-1, 1] maps over the cursor height.
/// Spreads ±7 pixels horizontally with intensity falling off by bit-shift: `wave >> |x|`.
///
/// **Rule 0 — WHY/PROOF/PREVENTS:** `bx` must be ≥ 7 and < `buf_w - 7` for the ±7 spread.
/// Caller must ensure the blinkey is within textbox bounds (which are inset from window edges).
/// PREVENTS: out-of-bounds writes on the horizontal spread.
pub fn draw_blinkey(
    canvas: &mut Canvas,
    bx: usize,
    by: usize,
    height: usize,
    top_bright: bool,
) {
    let buf_w = canvas.width;
    // Damage: ±7 horizontal spread × `height` vertical band. Caller's bounds invariant (bx ≥ 7, bx < buf_w-7) guarantees this stays in-buffer.
    canvas
        .damage
        .add_bounds(bx - 7, by, bx + 8, by + height);
    let pixels: &mut [u32] = canvas.pixels;
    let half = height / 2;
    for y in by..by + height {
        let idx = y * buf_w + bx;
        let t = (y - by - half) as isize as f32 / half as f32;
        let wave = if top_bright {
            (1.0 - t * t) * (1.0 - t) * (1.0 - t) * crate::theme::CURSOR_BRIGHTNESS
        } else {
            (1.0 - t * t) * (1.0 + t) * (1.0 + t) * crate::theme::CURSOR_BRIGHTNESS
        };
        let w = wave as u32;
        for dx in -7i32..=7 {
            let pixel = 0x00010101u32 * (w >> dx.unsigned_abs());
            pixels[(idx as isize + dx as isize) as usize] += pixel;
        }
    }
}

/// Paint a 4-directional blur glow around a textbox pill silhouette.
///
/// Loop structure + adder/intensity math are ported verbatim from photon's `apply_textbox_glow` (`compositing.rs:4479`). The pixel-write step diverges in three ways for fluor's layered model:
///
/// 1. **Constant glow_colour RGB + per-pass alpha** (instead of photon's intensity-scaled RGB additive blend onto opaque bg). Each pass writes `(intensity << 24) | (glow_colour & 0x00FFFFFF)` and saturating-adds onto the current pixel. The downstream textbox_group's `AlphaOver` flatten then produces `glow_colour × α/256 + chrome × (1 - α/256)` — exactly equivalent to photon's perceived appearance on opaque backgrounds, but also correct on bright or transparent ones (the layered-model equivalent of photon's additive paint).
/// 2. **Saturating per-byte add** instead of wrapping `+=` — the textbox_group's layer starts at zero, so multi-direction accumulation would byte-wrap and produce dark stripes where the glow should be brightest. Each byte caps at 0xFF.
/// 3. **Gated on `intensity > 0`** — photon's `pixels[idx] += 0 = no-op` was implicit; here we'd otherwise saturating-add `glow_rgb` (with alpha=0) into the pill interior every pass, staining it with the glow color. Explicit skip preserves photon's behaviour.
///
/// The `add: bool` / "remove glow" path photon used is dropped — fluor's Group model fully re-rasterizes the layer each dirty cycle, so there's nothing to remove.
pub fn apply_textbox_glow(
    canvas: &mut Canvas,
    mask: &[u8],
    center_y: isize,
    box_width: usize,
    box_height: usize,
    glow_colour: u32,
) {
    let buf_w = canvas.width;
    let buf_h = canvas.height;
    let blur_h = 32usize;
    let blur_v = 16usize;

    let half_h = (box_height / 2) as isize;
    if (center_y - half_h) as usize >= buf_h || (center_y + half_h) as usize >= buf_h {
        return;
    }
    let cy = center_y as usize;

    let y_top = cy - box_height / 2;
    let y_bot = cy + box_height / 2;
    // Damage = glow halo bbox (box + lateral blur padding, vertical box + vertical blur).
    {
        let center_x = buf_w / 2;
        let x0 = center_x.saturating_sub(box_width / 2 + blur_h);
        let x1 = (center_x + box_width / 2 + blur_h).min(buf_w);
        let y0 = y_top.saturating_sub(blur_v);
        let y1 = (y_bot + blur_v).min(buf_h);
        canvas.damage.add_bounds(x0, y0, x1, y1);
    }
    let pixels: &mut [u32] = canvas.pixels;

    // Find horizontal bounds by scanning mask at center row.
    let center_x = buf_w / 2;
    let mut x_left = center_x;
    let mut x_right = center_x;
    let scan = cy * buf_w;
    for lx in (0..center_x).rev() {
        if mask[scan + lx] > 0 {
            x_left = lx;
        } else {
            break;
        }
    }
    for rx in center_x..buf_w {
        if mask[scan + rx] > 0 {
            x_right = rx;
        } else {
            break;
        }
    }

    let corner_r = 2 * box_width * box_height / (box_width + box_height);
    let xvs = x_left + corner_r;
    let xve = x_right - corner_r;
    let yhs = y_top + corner_r;
    let yhe = y_bot - corner_r;

    let glow_rgb = glow_colour & 0x00FF_FFFF;

    /// Glow accumulation under fluor's α + darkness convention. The 4-direction passes accumulate opacity (α) and darkness (RGB) into a pixel that starts at empty (`0x00000000` — α=0, dark=0). Top byte: saturating-add intensity (drives α toward 0xFF/opaque). RGB: saturating-add per-channel darkness from `glow_dark`. Corner pixels touched by multiple directions get extra accumulated darkness, matching photon's brighter-at-corners effect. Caller must initialize the pixel buffer to `0x00000000` (calloc-free) before invoking.
    #[inline]
    fn glow_accumulate(dst: u32, intensity: u32, glow_dark: u32) -> u32 {
        let a = ((dst >> 24) & 0xFF).saturating_add(intensity);
        let dr = (dst >> 16) & 0xFF;
        let dg = (dst >> 8) & 0xFF;
        let db = dst & 0xFF;
        let sr = (glow_dark >> 16) & 0xFF;
        let sg = (glow_dark >> 8) & 0xFF;
        let sb = glow_dark & 0xFF;
        let nr = (dr + sr).min(0xFF);
        let ng = (dg + sg).min(0xFF);
        let nb = (db + sb).min(0xFF);
        (a << 24) | (nr << 16) | (ng << 8) | nb
    }

    // Right blur.
    for y in y_top..y_bot {
        let mut adder = 0u32;
        let start = x_right
            - (yhs as isize - y as isize).max(0) as usize
            - (y as isize - yhe as isize).max(0) as usize;
        for bx in start..x_right + blur_h {
            if bx >= buf_w {
                break;
            }
            let idx = y * buf_w + bx;
            if bx > 0 && mask[idx] < mask[idx - 1] {
                adder += (mask[idx - 1] - mask[idx]) as u32;
            }
            adder = (adder * 15 >> 4).min(71);
            let intensity = (adder * (255 - mask[idx]) as u32) >> 8;
            if intensity > 0 {
                pixels[idx] = glow_accumulate(pixels[idx], intensity, glow_rgb);
            }
        }
    }
    // Left blur.
    for y in y_top..y_bot {
        let mut adder = 0u32;
        let end = x_left
            + (yhs as isize - y as isize).max(0) as usize
            + (y as isize - yhe as isize).max(0) as usize;
        for bx in (x_left.saturating_sub(blur_h)..=end).rev() {
            let idx = y * buf_w + bx;
            if bx + 1 < buf_w && mask[idx] < mask[idx + 1] {
                adder += (mask[idx + 1] - mask[idx]) as u32;
            }
            adder = (adder * 15 >> 4).min(71);
            let intensity = (adder * (255 - mask[idx]) as u32) >> 8;
            if intensity > 0 {
                pixels[idx] = glow_accumulate(pixels[idx], intensity, glow_rgb);
            }
        }
    }
    // Down blur.
    for bx in x_left..x_right {
        let mut adder = 0u32;
        let start = y_bot
            - (xvs as isize - bx as isize).max(0) as usize
            - (bx as isize - xve as isize).max(0) as usize;
        for by in start..y_bot + blur_v {
            if by >= buf_h {
                break;
            }
            let idx = by * buf_w + bx;
            if by > 0 {
                let ia = (by - 1) * buf_w + bx;
                if mask[idx] < mask[ia] {
                    adder += (mask[ia] - mask[idx]) as u32;
                }
            }
            adder = (adder * 3 >> 2).min(70);
            let intensity = (adder * (255 - mask[idx]) as u32) >> 8;
            if intensity > 0 {
                pixels[idx] = glow_accumulate(pixels[idx], intensity, glow_rgb);
            }
        }
    }
    // Up blur.
    for bx in x_left..x_right {
        let mut adder = 0u32;
        let end = y_top
            + (xvs as isize - bx as isize).max(0) as usize
            + (bx as isize - xve as isize).max(0) as usize;
        for by in (0..=end).rev() {
            if by + blur_v < y_top {
                break;
            }
            if by >= buf_h {
                continue;
            }
            let idx = by * buf_w + bx;
            if by + 1 < buf_h {
                let ib = (by + 1) * buf_w + bx;
                if mask[idx] < mask[ib] {
                    adder += (mask[ib] - mask[idx]) as u32;
                }
            }
            adder = (adder * 3 >> 2).min(70);
            let intensity = (adder * (255 - mask[idx]) as u32) >> 8;
            if intensity > 0 {
                pixels[idx] = glow_accumulate(pixels[idx], intensity, glow_rgb);
            }
        }
    }
}

/// Glyph rasterizers for window controls. Ported verbatim from photon's [compositing.rs](/mnt/Octopus/Code/photon/src/ui/compositing.rs) — the squircle minus / squircle ring / capsule X — so chrome looks identical to photon.
pub mod glyph {
    /// Draw a horizontal squircle "minus" stroke centered at `(x, y)` inside a button of pixel radius `r`. Uses a 4-power squircle with widened axis to make a flat horizontal pill.
    pub fn minimize_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
    ) {
        let r = r + 1;
        let r_render = r / 4 + 1;
        let r_2 = r_render * r_render;
        let r_4 = r_2 * r_2;
        let r_3 = r_render * r_render * r_render;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;

        for h in -(r_render as isize)..=(r_render as isize) {
            for w in -(r as isize)..=(r as isize) {
                let h2 = h * h;
                let h4 = h2 * h2;
                let a = (w.abs() - (r * 3 / 4) as isize).max(0);
                let w2 = a * a;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 <= r_4 {
                    let px = (x as isize + w) as usize;
                    let py = (y as isize + h + (r / 2) as isize) as usize;
                    let idx = py * width + px;
                    let gradient = ((r_4 - dist_4) << 8) / (r_3 << 2);
                    if gradient > 255 {
                        pixels[idx] = stroke_packed;
                    } else {
                        pixels[idx] = blend_swar(pixels[idx], stroke_packed, gradient as u64);
                    }
                }
            }
        }
    }

    /// Draw a square "maximize" symbol — squircle ring with stroke + interior fill — centered at `(x, y)` with pixel radius `r`.
    pub fn maximize_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
        fill_colour: u32,
    ) {
        let r = r + 1;
        let mut r_4 = r * r;
        r_4 *= r_4;
        let r_3 = r * r * r;

        let r_inner = r * 4 / 5;
        let mut r_inner_4 = r_inner * r_inner;
        r_inner_4 *= r_inner_4;
        let r_inner_3 = r_inner * r_inner * r_inner;

        let outer_edge_threshold = r_3 << 2;
        let inner_edge_threshold = r_inner_3 << 2;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;
        let fill_packed = fill_colour & 0x00FF_FFFF;

        for h in -(r as isize)..=r as isize {
            for w in -(r as isize)..=r as isize {
                let h2 = h * h;
                let h4 = h2 * h2;
                let w2 = w * w;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 > r_4 {
                    continue;
                }
                let px = (x as isize + w) as usize;
                let py = (y as isize + h) as usize;
                let idx = py * width + px;

                let dist_from_outer = r_4 - dist_4;
                if dist_4 <= r_inner_4 {
                    let dist_from_inner = r_inner_4 - dist_4;
                    if dist_from_inner <= inner_edge_threshold {
                        let gradient = (dist_from_inner << 8) / inner_edge_threshold;
                        pixels[idx] = blend_swar(stroke_packed, fill_packed, gradient as u64);
                    } else {
                        pixels[idx] = fill_packed;
                    }
                } else if dist_from_outer <= outer_edge_threshold {
                    let gradient = (dist_from_outer << 8) / outer_edge_threshold;
                    pixels[idx] = blend_swar(pixels[idx], stroke_packed, gradient as u64);
                } else {
                    pixels[idx] = stroke_packed;
                }
            }
        }
    }

    /// Draw an antialiased "X" close symbol — two crossed capsule lines — centered at `(x, y)` with pixel radius `r`.
    pub fn close_symbol(
        pixels: &mut [u32],
        width: usize,
        x: usize,
        y: usize,
        r: usize,
        stroke_colour: u32,
    ) {
        let r = r + 1;
        let thickness = (r / 3).max(1) as f32;
        let radius = thickness / 2.0;
        let size = (r * 2) as f32;
        let cxf = x as f32;
        let cyf = y as f32;
        let end = size / 3.0;

        let x1_start = cxf - end;
        let y1_start = cyf - end;
        let x1_end = cxf + end;
        let y1_end = cyf + end;
        let x2_start = cxf + end;
        let y2_start = cyf - end;
        let x2_end = cxf - end;
        let y2_end = cyf + end;

        let stroke_packed = stroke_colour & 0x00FF_FFFF;
        let cxi = x as i32;
        let cyi = y as i32;
        let height = (pixels.len() / width) as i32;
        let min_x = ((x as i32) - (r as i32)).max(0);
        let max_x = ((x as i32) + (r as i32)).min(width as i32);
        let min_y = ((y as i32) - (r as i32)).max(0);
        let max_y = ((y as i32) + (r as i32)).min(height);

        // Each quadrant samples one of the two diagonals (whichever passes thru it).
        let quadrants: [(i32, i32, i32, i32, f32, f32, f32, f32); 4] = [
            (min_x, cxi, min_y, cyi, x1_start, y1_start, x1_end, y1_end), // top-left, diag1
            (cxi, max_x, min_y, cyi, x2_start, y2_start, x2_end, y2_end), // top-right, diag2
            (min_x, cxi, cyi, max_y, x2_start, y2_start, x2_end, y2_end), // bottom-left, diag2
            (cxi, max_x, cyi, max_y, x1_start, y1_start, x1_end, y1_end), // bottom-right, diag1
        ];
        for (qx0, qx1, qy0, qy1, x0, y0, x1, y1) in quadrants {
            for py in qy0..qy1 {
                for px in qx0..qx1 {
                    let dist = distance_to_capsule(
                        px as f32 + 0.5,
                        py as f32 + 0.5,
                        x0,
                        y0,
                        x1,
                        y1,
                        radius,
                    );
                    let alpha_f = if dist < -0.5 {
                        1.0
                    } else if dist < 0.5 {
                        0.5 - dist
                    } else {
                        0.0
                    };
                    if alpha_f > 0.0 {
                        let idx = py as usize * width + px as usize;
                        let alpha = (alpha_f * 256.0) as u64;
                        pixels[idx] = blend_swar(pixels[idx], stroke_packed, alpha);
                    }
                }
            }
        }
    }

    /// Distance from a point to a capsule (line segment + radius). Negative inside the capsule, positive outside, used as an SDF for AA.
    #[inline]
    fn distance_to_capsule(
        px: f32,
        py: f32,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        radius: f32,
    ) -> f32 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let len_sq = dx * dx + dy * dy;
        let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
        let t_clamped = t.clamp(0.0, 1.0);
        let cx = x1 + t_clamped * dx;
        let cy = y1 + t_clamped * dy;
        let ex = px - cx;
        let ey = py - cy;
        crate::math::sqrt(ex * ex + ey * ey) - radius
    }

    /// Pre-multiplied SWAR blend of `fg` over `bg` with explicit `alpha` (0..=256). Photon's exact pattern: widen each 32-bit pixel to 64 bits with each channel in its own 16-bit slot, do `bg*(256-α) + fg*α` in parallel, narrow back.
    #[inline]
    fn blend_swar(bg: u32, fg: u32, alpha: u64) -> u32 {
        let inv = 256 - alpha;
        let mut bg64 = bg as u64;
        bg64 = (bg64 | (bg64 << 16)) & 0x0000_FFFF_0000_FFFF;
        bg64 = (bg64 | (bg64 << 8)) & 0x00FF_00FF_00FF_00FF;
        let mut fg64 = fg as u64;
        fg64 = (fg64 | (fg64 << 16)) & 0x0000_FFFF_0000_FFFF;
        fg64 = (fg64 | (fg64 << 8)) & 0x00FF_00FF_00FF_00FF;
        let mut blended = bg64 * inv + fg64 * alpha;
        blended = (blended >> 8) & 0x00FF_00FF_00FF_00FF;
        blended = (blended | (blended >> 8)) & 0x0000_FFFF_0000_FFFF;
        blended = blended | (blended >> 16);
        blended as u32
    }
}

/// Fill a circle with a 1-pixel-wide AA edge ring. Center `(cx, cy)` and `radius` are in pixels; `colour` is straight-alpha ARGB (the AA coverage modulates the supplied alpha, so a translucent fill stays translucent at the edge). Optional `mask` multiplies into the per-pixel alpha for soft clipping.
///
/// AA via gradient-magnitude (no sqrt): for a pixel at squared distance `d²`, coverage is `(r_outer² - d²) / (r_outer² - r_inner²)` where `r_outer = radius` and `r_inner = radius - 1`. Gives a smooth 0→1 ramp across one pixel of edge.
pub fn circle_filled(
    canvas: &mut Canvas,
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
        let x0 = (cx - radius).max(c.x_start as isize).max(0) as usize;
        let y0 = (cy - radius).max(c.y_start as isize).max(0) as usize;
        let x1 = (cx + radius + 1)
            .min(c.x_end as isize)
            .min(buf_w as isize)
            .max(0) as usize;
        let y1 = (cy + radius + 1)
            .min(c.y_end as isize)
            .min(buf_h as isize)
            .max(0) as usize;
        canvas.damage.add_bounds(x0, y0, x1, y1);
    }
    let pixels: &mut [u32] = canvas.pixels;
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

    // α + darkness: top byte IS opacity; use directly. RGB darkness intact.
    let fg_opacity = (colour >> 24) & 0xFF;
    let colour_rgb = colour & 0x00FF_FFFF;

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
            let coverage: u32 = if dist2 <= r_inner2 {
                256
            } else {
                (((r_outer2 - dist2) << 8) / edge_range) as u32
            };
            let scaled_opacity = (fg_opacity * coverage) >> 8;
            let idx = base + px;
            let final_opacity = match mask {
                Some(m) => (scaled_opacity * m.pixels[idx] as u32) >> 8,
                None => scaled_opacity,
            };
            let scaled_colour = colour_rgb | (final_opacity << 24);
            pixels[idx] = pixels[idx].under(scaled_colour, BlendMode::Normal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // α + darkness storage: α = 0x12 in the top byte, RGB stored as darkness (255 − channel).
        // pack_argb(0xAB, 0xCD, 0xEF, 0x12) → α=0x12, dark=(0x54, 0x32, 0x10).
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
        circle_filled(
            &mut canvas,
            8,
            8,
            5,
            pack_argb(255, 0, 0, 255),
            None,
            None,
        );
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
        // Buffer starts EMPTY (α=0, dark=0). Paint opaque visible RGB(0x11,0x22,0x33) under it.
        // Result: ~opaque colour (1-LSB drift per channel from the >>8 normalization).
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
        fill_rect(
            &mut canvas,
            0,
            0,
            4,
            4,
            pack_argb(255, 0, 0, 0),
            None,
            None,
        );
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
