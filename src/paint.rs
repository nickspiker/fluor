//! Pixel-buffer paint primitives. Packed layout is `0xααRRGGBB` (α-byte high, blue low) — top byte is **α (opacity)**, industry-standard direction (`α = 0` transparent, `α = 0xFF` opaque). RGB bytes store darkness (`0 = white`, `255 = black`). See [`crate::pixel`] for the locked convention. All inputs are pixel-space, not RU — convert via [`Viewport::ru_to_px`](crate::Viewport::ru_to_px) before calling.
//!
//! Internal to fluor's render pipeline. Per `## API / Implementation Separation` in AGENT.md, these are not part of the consumer-facing API: future SIMD kernels (NEON, SSE2) will dispatch thru the same entry points without changing call sites in `pane` or `Compositor`.
//!
//! Blend model is α + darkness front-to-back: `dst` is the partial composite already accumulated above (its α-byte = accumulated opacity, RGB = accumulated darkness), `src` is the new layer going behind. Per-pixel early-out fires when `dst >= 0xFF000000` (dst α saturated = opaque) via a single u32 compare. Math throughout is `>> 8` with the `(256 − top_α)` trick — never `/ 255`, never floats in the inner loop. Multi-layer composition is additive on BOTH halves (α adds, darkness adds); the buffer carries the accumulator state across Group boundaries so the early-out chain survives between flatten passes.
//!
//! Every blending primitive accepts an optional [`Clip`] (defaults to full buffer when `None`) and an optional [`AlphaMask`] (full-frame, multiplies into per-pixel alpha for soft clipping — rounded textboxes, squircle pane corners, scroll fades). The clip is resolved once at entry into `(x_min, y_min, x_max, y_max)` loop bounds, so the inner loops carry **zero per-pixel bounds checks** — the math at the entry is the proof. AlphaMask dimensions must equal the buffer's `(buf_w, buf_h)`; mismatches panic per AGENT.md "fail loud."

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
    ((a as u32) << 24)
        | (((255 - r) as u32) << 16)
        | (((255 - g) as u32) << 8)
        | ((255 - b) as u32)
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
#[inline]
pub fn flatten(dst: &mut [u32], src: &[u32], mode: BlendMode) {
    for i in 0..dst.len() {
        dst[i] = dst[i].under(src[i], mode);
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
    pixels: &mut [u32],
    buf_w: usize,
    buf_h: usize,
    x: isize,
    y: isize,
    rect_w: isize,
    rect_h: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    let clip = Clip::resolve(clip, buf_w, buf_h);
    if let Some(m) = mask {
        assert_mask_matches_buffer(m, buf_w, buf_h);
    }
    let (x_min, y_min, x_max, y_max) = clip_rect(clip, x, y, rect_w, rect_h);
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
    pixels: &mut [u32],
    buf_w: usize,
    buf_h: usize,
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
    for &(ex, ey, ew, eh) in &edges {
        fill_rect(pixels, buf_w, buf_h, ex, ey, ew, eh, colour, clip, mask);
    }
}

/// Fill the buffer with photon's signature procedural background — symmetric organic noise plus speckle. Sequential (no rayon dep at this layer) but mirrored left/right halves like photon. Set `fullscreen=true` to fill the whole buffer; `false` leaves a 1px border for the window edge stroke. `speckle` is an animation counter (constant 0 for static); `scroll_offset` shifts the texture vertically (for content scroll integration).
///
/// Clip restricts the row range. Mask isn't supported here (background is bg — masking it would mean "draw nothing where mask is zero" which is the same as just clearing afterward; if you need that, do it explicitly).
pub fn background_noise(
    pixels: &mut [u32],
    buf_w: usize,
    buf_h: usize,
    speckle: usize,
    fullscreen: bool,
    scroll_offset: isize,
    clip: Option<Clip>,
) {
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
    for row_idx in row_start..row_end {
        let logical_row = row_idx as isize - scroll_offset;
        let row_pixels = &mut pixels[row_idx * buf_w..(row_idx + 1) * buf_w];
        background_row(
            row_pixels,
            buf_w,
            logical_row,
            buf_h,
            x_start,
            x_end,
            speckle,
        );
    }
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
    let mut rng: usize = (0xDEAD_BEEF_0123_4567)
        ^ ((logical_row as usize)
            .wrapping_sub(height / 2)
            .wrapping_mul(0x9E37_79B9_4517_B397));
    let ones = 0x0001_0101u32;
    let mut colour = rng as u32 & BG_MASK;

    // Right half — left to right.
    for x in (width / 2)..x_end {
        rng ^= rng.rotate_left(13).wrapping_add(12_345_678_942);
        let adder = rng as u32 & ones;
        if rng.wrapping_add(speckle) < usize::MAX / 256 {
            colour = (rng as u32 >> 8) & BG_SPECKLE;
        } else {
            colour = colour.wrapping_add(adder) & BG_MASK;
            let subtractor = (rng >> 5) as u32 & ones;
            colour = colour.wrapping_sub(subtractor) & BG_MASK;
        }
        row_pixels[x] =
            ((colour.wrapping_add(BG_BASE) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA;
    }

    // Left half — right to left, same RNG seed (mirror).
    rng = 0xDEAD_BEEF_0123_4567
        ^ ((logical_row as usize)
            .wrapping_sub(height / 2)
            .wrapping_mul(0x9E37_79B9_4517_B397));
    colour = rng as u32 & BG_MASK;
    for x in (x_start..(width / 2)).rev() {
        rng ^= rng.rotate_left(13).wrapping_sub(12_345_678_942);
        let adder = rng as u32 & ones;
        if rng.wrapping_add(speckle) < usize::MAX / 256 {
            colour = (rng as u32 >> 8) & BG_SPECKLE;
        } else {
            colour = colour.wrapping_add(adder) & BG_MASK;
            let subtractor = (rng >> 5) as u32 & ones;
            colour = colour.wrapping_sub(subtractor) & BG_MASK;
        }
        row_pixels[x] =
            ((colour.wrapping_add(BG_BASE) & RGB_MASK) ^ VISIBLE_TO_DARK_FLIP) | OPAQUE_ALPHA;
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
    for i in 0..n {
        let v = pixels[i] ^ 0x00FFFFFF;
        let m = clip_mask[i] as u32;
        let inner_alpha = (v >> 24) & 0xFF;
        let final_alpha = (inner_alpha * m) >> 8;
        if alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE {
            pixels[i] = 0xFF000000 | (final_alpha << 16) | (final_alpha << 8) | final_alpha;
            continue;
        }
        if alpha_mode == DEBUG_SHOW_ALPHA_FORCE_OPAQUE {
            // Force-opaque: keep the kernel's visible RGB exactly (no premultiply, no clip-mask trim), set α=255 so the OS displays it. Lets you see what landed in the buffer BEFORE the boundary trimmed/multiplied anything.
            pixels[i] = 0xFF000000 | (v & 0x00FFFFFF);
            continue;
        }
        #[cfg(target_os = "linux")]
        let s = if DEBUG_SKIP_PREMULT.load(std::sync::atomic::Ordering::Relaxed) {
            256u32
        } else {
            final_alpha
        };
        #[cfg(not(target_os = "linux"))]
        let s = 256u32;
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        pixels[i] = (final_alpha << 24) | (r << 16) | (g << 8) | b;
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
    rect_x: usize,
    rect_y: usize,
) {
    let alpha_mode = DEBUG_SHOW_ALPHA.load(std::sync::atomic::Ordering::Relaxed);
    let n = (win_w * win_h).min(scratch.len()).min(clip_mask.len());
    for i in 0..n {
        let sy = i / win_w;
        let sx = i - sy * win_w;
        let dst_idx = (rect_y + sy) * scr_w + (rect_x + sx);

        let v = scratch[i] ^ 0x00FFFFFF;
        let m = clip_mask[i] as u32;
        let inner_alpha = (v >> 24) & 0xFF;
        let final_alpha = (inner_alpha * m) >> 8;
        if alpha_mode == DEBUG_SHOW_ALPHA_GRAYSCALE {
            screen[dst_idx] =
                0xFF000000 | (final_alpha << 16) | (final_alpha << 8) | final_alpha;
            continue;
        }
        if alpha_mode == DEBUG_SHOW_ALPHA_FORCE_OPAQUE {
            screen[dst_idx] = 0xFF000000 | (v & 0x00FFFFFF);
            continue;
        }
        #[cfg(target_os = "linux")]
        let s = if DEBUG_SKIP_PREMULT.load(std::sync::atomic::Ordering::Relaxed) {
            256u32
        } else {
            final_alpha
        };
        #[cfg(not(target_os = "linux"))]
        let s = 256u32;
        let r = (((v >> 16) & 0xFF) * s) >> 8;
        let g = (((v >> 8) & 0xFF) * s) >> 8;
        let b = ((v & 0xFF) * s) >> 8;
        screen[dst_idx] = (final_alpha << 24) | (r << 16) | (g << 8) | b;
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

/// Hard-pixel squircle pill with AA on both the X-axis curve (sides) and Y-axis curve (cap tops/bottoms). Photon's avatar-ring strategy in one call — render twice with different sizes/colors to get a stroke ring.
///
/// Photon-faithful: precompute squircle crossings once (`(inset_px, l_aa, h_aa)` per pixel-row offset into the cap), then walk pure integer indices per corner. Each crossing produces BOTH a vertical-edge AA pixel and a horizontal-edge AA pixel via the squircle's diagonal symmetry — no separate per-col walk needed. Photon's `compositing.rs` `draw_textbox` is the reference; this is the single-color silhouette adaptation (`draw_textbox_pill` keeps the two-tone hairline version photon uses for textboxes).
///
/// `blend_aa_with_existing = false` (outer pass): AA pixels write `(alpha = h_aa, RGB = color)`. Conflicting writes at the diagonal pixel pick MAX h_aa.
///
/// `blend_aa_with_existing = true` (inner pass): AA pixels blend `color_rgb` into the current pixel's RGB by `h_aa`, keeping alpha=255 — produces the proper `fill·h + outside·(1-h)` transition when painted on top of an outer-pass stroke result.
pub fn draw_squircle_pill(
    pixels: &mut [u32],
    mask: &mut [u8],
    buf_w: usize,
    buf_h: usize,
    pill_x: isize,
    pill_y: isize,
    pill_w: isize,
    pill_h: isize,
    color: u32,
    squirdleyness: i32,
    blend_aa_with_existing: bool,
) {
    if pill_w <= 0 || pill_h <= 0 {
        return;
    }
    let buf_w_i = buf_w as isize;
    let buf_h_i = buf_h as isize;
    // Bbox-overlap early-out — pill entirely off-buffer.
    if pill_x + pill_w <= 0 || pill_y + pill_h <= 0 || pill_x >= buf_w_i || pill_y >= buf_h_i {
        return;
    }

    let radius_f = pill_h as f32 * 0.5;
    let radius = (pill_h / 2) as isize;
    // α + darkness: force opaque (α=0xFF) by setting the top byte. RGB darkness intact.
    let solid = (color & 0x00FF_FFFF) | 0xFF000000;
    let color_rgb = color & 0x00FF_FFFF;
    let crossings = squircle_crossings(radius_f, squirdleyness);

    // Fast/slow split. Fast path: pill bbox fully inside the buffer → no per-pixel checks.
    // Slow path: partial overhang (scroll/resize transitions) → range clips at the corner-block
    // boundary so each AA write has its row already proven in-buffer.
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
    pixels: &mut [u32],
    mask: &mut [u8],
    buf_w: usize,
    buf_h: usize,
    center_x: usize,
    center_y: isize,
    box_width: usize,
    box_height: usize,
) {
    use crate::theme;

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
    pixels: &mut [u32],
    buf_w: usize,
    bx: usize,
    by: usize,
    height: usize,
    top_bright: bool,
) {
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
    pixels: &mut [u32],
    mask: &[u8],
    buf_w: usize,
    buf_h: usize,
    center_y: isize,
    box_width: usize,
    box_height: usize,
    glow_colour: u32,
) {
    let blur_h = 32usize;
    let blur_v = 16usize;

    let half_h = (box_height / 2) as isize;
    if (center_y - half_h) as usize >= buf_h || (center_y + half_h) as usize >= buf_h {
        return;
    }
    let cy = center_y as usize;

    let y_top = cy - box_height / 2;
    let y_bot = cy + box_height / 2;

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
    pixels: &mut [u32],
    buf_w: usize,
    buf_h: usize,
    cx: isize,
    cy: isize,
    radius: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    if radius <= 0 {
        return;
    }
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
        stroke_rect(
            &mut buf,
            10,
            10,
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
        circle_filled(
            &mut buf,
            16,
            16,
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
        circle_filled(
            &mut buf,
            8,
            8,
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
        // Visible (0x11, 0x22, 0x33) with α=0xFF → α + darkness = α=0xFF, dark=(0xEE, 0xDD, 0xCC).
        fill_rect(&mut buf, 4, 4, 0, 0, 4, 4, pack_argb(0x11, 0x22, 0x33, 0xFF), None, None);
        for &p in &buf {
            assert_eq!(p >> 24, 0xFF, "pixel α should be 0xFF (opaque): {p:#x}");
            // Darkness of visible 0x11 is 0xEE; minus 1-LSB drift.
            assert!(((p >> 16) & 0xFF) >= 0xEC);
        }
    }

    #[test]
    fn fill_rect_partial() {
        let mut buf = vec![0u32; 4 * 4];
        fill_rect(&mut buf, 4, 4, 1, 1, 2, 2, pack_argb(0xAA, 0xBB, 0xCC, 0xFF), None, None);
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
                    assert_eq!(
                        p, 0,
                        "outside pixel ({x},{y}) untouched, got {p:#x}"
                    );
                }
            }
        }
    }

    #[test]
    fn fill_rect_clips_negative_origin() {
        let mut buf = vec![0u32; 4 * 4];
        fill_rect(&mut buf, 4, 4, -2, -2, 4, 4, pack_argb(0xFF, 0xFF, 0xFE, 0xFF), None, None);
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
        fill_rect(&mut buf, 4, 4, 100, 100, 5, 5, pack_argb(0, 0, 0, 0xFF), None, None);
        assert!(buf.iter().all(|&p| p == 0));
        fill_rect(&mut buf, 4, 4, -10, -10, 5, 5, pack_argb(0, 0, 0, 0xFF), None, None);
        assert!(buf.iter().all(|&p| p == 0));
    }

    #[test]
    fn fill_rect_transparent_src_into_opaque_dst_is_noop() {
        // Dst is opaque (α=0xFF). Under doctrine: dst-opaque early-out fires for every pixel. Src never read.
        let mut buf = vec![pack_argb(50, 60, 70, 255); 4 * 4];
        fill_rect(
            &mut buf,
            4,
            4,
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
