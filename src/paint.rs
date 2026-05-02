//! Pixel-buffer paint primitives. ARGB layout is `0xAARRGGBB` (alpha high byte, blue low). All inputs are pixel-space, not RU — convert via [`Viewport::ru_to_px`](crate::Viewport::ru_to_px) before calling.
//!
//! Internal to fluor's render pipeline. Per `## API / Implementation Separation` in AGENT.md, these are not part of the consumer-facing API: future SIMD kernels (NEON, SSE2) will dispatch through the same entry points without changing call sites in `pane` or `Compositor`.
//!
//! Blend model is straight (non-premultiplied) alpha lerp: `result = bg * (1 - α) + fg * α`. For an opaque target framebuffer (the common case — the host window's backbuffer) the alpha channel of the result is don't-care; for layered translucency a Porter-Duff over would be needed and is not provided here.
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
        Self { x_start, y_start, x_end, y_end }
    }

    /// Full-buffer clip. Equivalent to passing `None` to a primitive.
    pub const fn buffer(buf_w: usize, buf_h: usize) -> Self {
        Self { x_start: 0, y_start: 0, x_end: buf_w, y_end: buf_h }
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
    pub a: Coord, pub b: Coord,
    pub c: Coord, pub d: Coord,
    pub tx: Coord, pub ty: Coord,
}

impl Transform {
    pub const IDENTITY: Transform = Transform { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx: 0.0, ty: 0.0 };

    #[inline]
    pub const fn new(a: Coord, b: Coord, c: Coord, d: Coord, tx: Coord, ty: Coord) -> Self {
        Self { a, b, c, d, tx, ty }
    }

    #[inline]
    pub fn rotate(radians: Coord) -> Self {
        let (s, co) = crate::math::sin_cos(radians);
        Self { a: co, b: s, c: -s, d: co, tx: 0.0, ty: 0.0 }
    }

    #[inline]
    pub fn scale(sx: Coord, sy: Coord) -> Self {
        Self { a: sx, b: 0.0, c: 0.0, d: sy, tx: 0.0, ty: 0.0 }
    }

    #[inline]
    pub fn skew(kx: Coord, ky: Coord) -> Self {
        Self { a: 1.0, b: ky, c: kx, d: 1.0, tx: 0.0, ty: 0.0 }
    }

    #[inline]
    pub fn translate(tx: Coord, ty: Coord) -> Self {
        Self { a: 1.0, b: 0.0, c: 0.0, d: 1.0, tx, ty }
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
        (self.a * x + self.c * y + self.tx, self.b * x + self.d * y + self.ty)
    }

    /// Bit-exact identity check. For "approximately identity" use a tolerance compare on the field deltas.
    #[inline]
    pub fn is_identity(self) -> bool {
        self.a == 1.0 && self.b == 0.0 && self.c == 0.0 && self.d == 1.0 && self.tx == 0.0 && self.ty == 0.0
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
    if divs == 0 { return 0.0; }
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
            pixels.len(), width, height, width * height,
        );
        Self { pixels, width, height }
    }
}

/// Assert mask dimensions match buffer; panic with a descriptive message if not. Per AGENT.md a mask attached to the wrong buffer is an initialization bug — fail loud.
#[inline]
pub(crate) fn assert_mask_matches_buffer(mask: &AlphaMask, buf_w: usize, buf_h: usize) {
    assert!(
        mask.width == buf_w && mask.height == buf_h,
        "AlphaMask dimensions {}×{} don't match buffer {}×{}",
        mask.width, mask.height, buf_w, buf_h,
    );
}

/// Pack four 8-bit channels into a single 32-bit ARGB value (`0xAARRGGBB`).
#[inline]
pub fn pack_argb(r: u8, g: u8, b: u8, a: u8) -> u32 {
    ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
}

/// Unpack a 32-bit ARGB value into `(r, g, b, a)`.
#[inline]
pub fn unpack_argb(packed: u32) -> (u8, u8, u8, u8) {
    let a = (packed >> 24) as u8;
    let r = (packed >> 16) as u8;
    let g = (packed >> 8) as u8;
    let b = packed as u8;
    (r, g, b, a)
}

/// Straight-alpha lerp of `fg` onto `bg`. SWAR pattern: widen each 32-bit ARGB pixel to 64 bits with each 8-bit channel in its own 16-bit slot, do four channel multiplies in parallel via u64 arithmetic, narrow back. The `>>8` divisor is 256 (not 255) — the canonical fast-blend approximation; per-channel error is below 1/256 and imperceptible.
#[inline]
pub fn blend(bg: u32, fg: u32) -> u32 {
    let alpha = ((fg >> 24) & 0xFF) as u64;
    let inv_alpha = 256 - alpha;

    let mut bg64 = bg as u64;
    bg64 = (bg64 | (bg64 << 16)) & 0x0000_FFFF_0000_FFFF;
    bg64 = (bg64 | (bg64 << 8)) & 0x00FF_00FF_00FF_00FF;

    let mut fg64 = fg as u64;
    fg64 = (fg64 | (fg64 << 16)) & 0x0000_FFFF_0000_FFFF;
    fg64 = (fg64 | (fg64 << 8)) & 0x00FF_00FF_00FF_00FF;

    let mut blended = bg64 * inv_alpha + fg64 * alpha;

    blended = (blended >> 8) & 0x00FF_00FF_00FF_00FF;
    blended = (blended | (blended >> 8)) & 0x0000_FFFF_0000_FFFF;
    blended = blended | (blended >> 16);
    blended as u32
}

/// Intersect a caller-supplied `(x, y, w, h)` rect (in pixels, top-left origin, may be negative or extend off-clip) with a `Clip`. Returns `(x_min, y_min, x_max, y_max)` in `usize`, all guaranteed in-bounds for `pixels[y * buf_w + x]` indexing **as long as the supplied `Clip` is itself within the buffer**. Returns an empty range (x_min >= x_max or y_min >= y_max) if the rect lies entirely outside the clip.
///
/// **Rule 0 — WHY/PROOF/PREVENTS:** rect coords are external inputs (caller can pass a pane dragged off the window edge). WHY: compositor semantics demand "draw the intersection with the clip." PROOF without it: a negative `x as usize` wraps to a huge value, indexing past the pixel slice panics. PREVENTS: panic on partial-offscreen rects, which is a normal use case. The clip happens once per rect; inner loops trust the math.
#[inline]
fn clip_rect(clip: Clip, x: isize, y: isize, rect_w: isize, rect_h: isize) -> (usize, usize, usize, usize) {
    // Negative isize → huge usize after cast; .min(clip.x_end) clamps it down. .max(clip.x_start) ensures we never index before the clip's left edge.
    let x_end = x + rect_w;
    let y_end = y + rect_h;
    let x_min = if x < 0 { clip.x_start } else { (x as usize).clamp(clip.x_start, clip.x_end) };
    let y_min = if y < 0 { clip.y_start } else { (y as usize).clamp(clip.y_start, clip.y_end) };
    let x_max = if x_end < 0 { clip.x_start } else { (x_end as usize).clamp(clip.x_start, clip.x_end) };
    let y_max = if y_end < 0 { clip.y_start } else { (y_end as usize).clamp(clip.y_start, clip.y_end) };
    (x_min, y_min, x_max, y_max)
}

/// Fill a rectangle with a solid (opaque-replace) ARGB colour. Solid means no alpha math — the source colour overwrites the destination directly. If you want alpha blending or alpha masking, use [`fill_rect_blend`].
pub fn fill_rect_solid(
    pixels: &mut [u32], buf_w: usize, buf_h: usize,
    x: isize, y: isize, rect_w: isize, rect_h: isize,
    colour: u32,
    clip: Option<Clip>,
) {
    let clip = Clip::resolve(clip, buf_w, buf_h);
    let (x_min, y_min, x_max, y_max) = clip_rect(clip, x, y, rect_w, rect_h);
    for row in y_min..y_max {
        let base = row * buf_w;
        for col in x_min..x_max {
            pixels[base + col] = colour;
        }
    }
}

/// Fill a rectangle by alpha-blending `colour` over the existing buffer contents. With `mask = Some(&AlphaMask)`, the per-pixel mask alpha multiplies into `colour`'s alpha (soft clipping for shaped textboxes, scroll fades, etc.) — `effective_alpha = colour_alpha * mask_alpha / 256`.
pub fn fill_rect_blend(
    pixels: &mut [u32], buf_w: usize, buf_h: usize,
    x: isize, y: isize, rect_w: isize, rect_h: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    let clip = Clip::resolve(clip, buf_w, buf_h);
    if let Some(m) = mask { assert_mask_matches_buffer(m, buf_w, buf_h); }
    let (x_min, y_min, x_max, y_max) = clip_rect(clip, x, y, rect_w, rect_h);
    let colour_a = (colour >> 24) & 0xFF;
    let colour_rgb = colour & 0x00FF_FFFF;
    match mask {
        None => {
            for row in y_min..y_max {
                let base = row * buf_w;
                for col in x_min..x_max {
                    let idx = base + col;
                    pixels[idx] = blend(pixels[idx], colour);
                }
            }
        }
        Some(m) => {
            for row in y_min..y_max {
                let base = row * buf_w;
                for col in x_min..x_max {
                    let idx = base + col;
                    let mask_a = m.pixels[idx] as u32;
                    let effective_a = (colour_a * mask_a) >> 8;
                    let masked = colour_rgb | (effective_a << 24);
                    pixels[idx] = blend(pixels[idx], masked);
                }
            }
        }
    }
}

/// Stroke (outline) an axis-aligned rectangle. Draws four filled rect strips along the edges; corners are not joined separately because at 90° angles the strips meet cleanly. If `colour` is fully opaque (alpha = 0xFF) and `mask` is `None`, takes the fast `fill_rect_solid` path; otherwise routes each strip through `fill_rect_blend`.
pub fn stroke_rect(
    pixels: &mut [u32], buf_w: usize, buf_h: usize,
    x: isize, y: isize, rect_w: isize, rect_h: isize,
    stroke: isize, colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    if stroke <= 0 || rect_w <= 0 || rect_h <= 0 { return; }
    let solid = (colour >> 24) == 0xFF && mask.is_none();
    let inner_h = rect_h - 2 * stroke;
    let edges: [(isize, isize, isize, isize); 4] = [
        (x, y, rect_w, stroke),                                       // top
        (x, y + rect_h - stroke, rect_w, stroke),                     // bottom
        (x, y + stroke, stroke, inner_h),                             // left
        (x + rect_w - stroke, y + stroke, stroke, inner_h),           // right
    ];
    for &(ex, ey, ew, eh) in &edges {
        if solid {
            fill_rect_solid(pixels, buf_w, buf_h, ex, ey, ew, eh, colour, clip);
        } else {
            fill_rect_blend(pixels, buf_w, buf_h, ex, ey, ew, eh, colour, clip, mask);
        }
    }
}

/// Fill the buffer with photon's signature procedural background — symmetric organic noise plus speckle. Sequential (no rayon dep at this layer) but mirrored left/right halves like photon. Set `fullscreen=true` to fill the whole buffer; `false` leaves a 1px border for the window edge stroke. `speckle` is an animation counter (constant 0 for static); `scroll_offset` shifts the texture vertically (for content scroll integration).
///
/// Clip restricts the row range. Mask isn't supported here (background is bg — masking it would mean "draw nothing where mask is zero" which is the same as just clearing afterward; if you need that, do it explicitly).
pub fn background_noise(
    pixels: &mut [u32], buf_w: usize, buf_h: usize,
    speckle: usize, fullscreen: bool, scroll_offset: isize,
    clip: Option<Clip>,
) {
    if buf_w < 2 || buf_h < 2 { return; }
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
    if row_start >= row_end || x_start >= x_end { return; }
    for row_idx in row_start..row_end {
        let logical_row = row_idx as isize - scroll_offset;
        let row_pixels = &mut pixels[row_idx * buf_w..(row_idx + 1) * buf_w];
        background_row(row_pixels, buf_w, logical_row, buf_h, x_start, x_end, speckle);
    }
}

#[inline]
fn background_row(row_pixels: &mut [u32], width: usize, logical_row: isize, height: usize, x_start: usize, x_end: usize, speckle: usize) {
    use crate::theme::{BG_ALPHA, BG_BASE, BG_MASK, BG_SPECKLE};
    let mut rng: usize = (0xDEAD_BEEF_0123_4567)
        ^ ((logical_row as usize).wrapping_sub(height / 2).wrapping_mul(0x9E37_79B9_4517_B397));
    let ones = 0x0001_0101u32;
    let mut colour = rng as u32 & BG_MASK | BG_ALPHA;

    // Right half — left to right.
    for x in (width / 2)..x_end {
        rng ^= rng.rotate_left(13).wrapping_add(12_345_678_942);
        let adder = rng as u32 & ones;
        if rng.wrapping_add(speckle) < usize::MAX / 256 {
            colour = (rng as u32 >> 8) & BG_SPECKLE | BG_ALPHA;
        } else {
            colour = colour.wrapping_add(adder) & BG_MASK;
            let subtractor = (rng >> 5) as u32 & ones;
            colour = colour.wrapping_sub(subtractor) & BG_MASK;
        }
        row_pixels[x] = colour.wrapping_add(BG_BASE) | BG_ALPHA;
    }

    // Left half — right to left, same RNG seed (mirror).
    rng = 0xDEAD_BEEF_0123_4567
        ^ ((logical_row as usize).wrapping_sub(height / 2).wrapping_mul(0x9E37_79B9_4517_B397));
    colour = rng as u32 & BG_MASK | BG_ALPHA;
    for x in (x_start..(width / 2)).rev() {
        rng ^= rng.rotate_left(13).wrapping_sub(12_345_678_942);
        let adder = rng as u32 & ones;
        if rng.wrapping_add(speckle) < usize::MAX / 256 {
            colour = (rng as u32 >> 8) & BG_SPECKLE | BG_ALPHA;
        } else {
            colour = colour.wrapping_add(adder) & BG_MASK;
            let subtractor = (rng >> 5) as u32 & ones;
            colour = colour.wrapping_sub(subtractor) & BG_MASK;
        }
        row_pixels[x] = colour.wrapping_add(BG_BASE) | BG_ALPHA;
    }
}

/// Photon's `PREMULTIPLIED` cfg flag: when true (Linux/Windows/macOS targets) the framebuffer expects premultiplied alpha; transparent edge pixels need their RGB scaled by alpha. False elsewhere (Android, etc.) — straight ARGB.
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
pub const PREMULTIPLIED: bool = true;
#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
pub const PREMULTIPLIED: bool = false;

/// Photon's `scale_alpha` helper. Verbatim port from [compositing.rs:5809](/mnt/Octopus/Code/photon/src/ui/compositing.rs#L5809). Multiplies all four channels of `colour` by `alpha/256` using SWAR — premultiplies RGB so a fully transparent pixel reads as `0x00000000`.
pub fn scale_alpha(colour: u32, alpha: u8) -> u32 {
    let mut c = colour as u64;
    c = (c | (c << 16)) & 0x0000FFFF0000FFFF;
    c = (c | (c << 8)) & 0x00FF00FF00FF00FF;
    let mut scaled = c * alpha as u64;
    scaled = (scaled >> 8) & 0x00FF00FF00FF00FF;
    scaled = (scaled | (scaled >> 8)) & 0x0000FFFF0000FFFF;
    scaled = scaled | (scaled >> 16);
    scaled as u32
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
    blended = blended | (blended >> 16) | 0xFF000000;

    blended as u32
}

/// Glyph rasterizers for window controls. Ported verbatim from photon's [compositing.rs](/mnt/Octopus/Code/photon/src/ui/compositing.rs) — the squircle minus / squircle ring / capsule X — so chrome looks identical to photon.
pub mod glyph {
    /// Draw a horizontal squircle "minus" stroke centered at `(x, y)` inside a button of pixel radius `r`. Uses a 4-power squircle with widened axis to make a flat horizontal pill.
    pub fn minimize_symbol(pixels: &mut [u32], width: usize, x: usize, y: usize, r: usize, stroke_colour: u32) {
        let r = r + 1;
        let r_render = r / 4 + 1;
        let r_2 = r_render * r_render;
        let r_4 = r_2 * r_2;
        let r_3 = r_render * r_render * r_render;

        let stroke_packed = stroke_colour | 0xFF00_0000;

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
    pub fn maximize_symbol(pixels: &mut [u32], width: usize, x: usize, y: usize, r: usize, stroke_colour: u32, fill_colour: u32) {
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

        let stroke_packed = stroke_colour | 0xFF00_0000;
        let fill_packed = fill_colour | 0xFF00_0000;

        for h in -(r as isize)..=r as isize {
            for w in -(r as isize)..=r as isize {
                let h2 = h * h;
                let h4 = h2 * h2;
                let w2 = w * w;
                let w4 = w2 * w2;
                let dist_4 = (h4 + w4) as usize;
                if dist_4 > r_4 { continue; }
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
    pub fn close_symbol(pixels: &mut [u32], width: usize, x: usize, y: usize, r: usize, stroke_colour: u32) {
        let r = r + 1;
        let thickness = (r / 3).max(1) as f32;
        let radius = thickness / 2.0;
        let size = (r * 2) as f32;
        let cxf = x as f32;
        let cyf = y as f32;
        let end = size / 3.0;

        let x1_start = cxf - end; let y1_start = cyf - end;
        let x1_end   = cxf + end; let y1_end   = cyf + end;
        let x2_start = cxf + end; let y2_start = cyf - end;
        let x2_end   = cxf - end; let y2_end   = cyf + end;

        let stroke_packed = stroke_colour | 0xFF00_0000;
        let cxi = x as i32;
        let cyi = y as i32;
        let height = (pixels.len() / width) as i32;
        let min_x = ((x as i32) - (r as i32)).max(0);
        let max_x = ((x as i32) + (r as i32)).min(width as i32);
        let min_y = ((y as i32) - (r as i32)).max(0);
        let max_y = ((y as i32) + (r as i32)).min(height);

        // Each quadrant samples one of the two diagonals (whichever passes through it).
        let quadrants: [(i32, i32, i32, i32, f32, f32, f32, f32); 4] = [
            (min_x, cxi, min_y, cyi, x1_start, y1_start, x1_end, y1_end),  // top-left, diag1
            (cxi,   max_x, min_y, cyi, x2_start, y2_start, x2_end, y2_end),// top-right, diag2
            (min_x, cxi, cyi,   max_y, x2_start, y2_start, x2_end, y2_end),// bottom-left, diag2
            (cxi,   max_x, cyi, max_y, x1_start, y1_start, x1_end, y1_end),// bottom-right, diag1
        ];
        for (qx0, qx1, qy0, qy1, x0, y0, x1, y1) in quadrants {
            for py in qy0..qy1 {
                for px in qx0..qx1 {
                    let dist = distance_to_capsule(
                        px as f32 + 0.5, py as f32 + 0.5,
                        x0, y0, x1, y1, radius,
                    );
                    let alpha_f = if dist < -0.5 { 1.0 } else if dist < 0.5 { 0.5 - dist } else { 0.0 };
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
    fn distance_to_capsule(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32, radius: f32) -> f32 {
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
    pixels: &mut [u32], buf_w: usize, buf_h: usize,
    cx: isize, cy: isize, radius: isize,
    colour: u32,
    clip: Option<Clip>,
    mask: Option<&AlphaMask>,
) {
    if radius <= 0 { return; }
    let clip = Clip::resolve(clip, buf_w, buf_h);
    if let Some(m) = mask { assert_mask_matches_buffer(m, buf_w, buf_h); }
    let r_outer = radius;
    let r_outer2 = r_outer * r_outer;
    let r_inner = radius - 1;
    let r_inner2 = r_inner * r_inner;
    let edge_range = r_outer2 - r_inner2;

    // Circle's bounding box, clipped. Side length is 2r + 1 (inclusive on both ends).
    let (x_min, y_min, x_max, y_max) = clip_rect(clip, cx - r_outer, cy - r_outer, 2 * r_outer + 1, 2 * r_outer + 1);

    let fg_alpha = (colour >> 24) & 0xFF;
    let colour_rgb = colour & 0x00FF_FFFF;

    for py in y_min..y_max {
        let dy = py as isize - cy;
        let dy2 = dy * dy;
        let base = py * buf_w;
        for px in x_min..x_max {
            let dx = px as isize - cx;
            let dist2 = dx * dx + dy2;
            if dist2 > r_outer2 { continue; }
            let coverage: u32 = if dist2 <= r_inner2 {
                256
            } else {
                (((r_outer2 - dist2) << 8) / edge_range) as u32
            };
            let scaled_alpha = (fg_alpha * coverage) >> 8;
            let idx = base + px;
            let final_alpha = match mask {
                Some(m) => (scaled_alpha * m.pixels[idx] as u32) >> 8,
                None => scaled_alpha,
            };
            let scaled_colour = colour_rgb | (final_alpha << 24);
            pixels[idx] = blend(pixels[idx], scaled_colour);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        let cases = [(0, 0, 0, 0), (255, 255, 255, 255), (12, 34, 56, 78), (200, 100, 50, 200)];
        for &(r, g, b, a) in &cases {
            let p = pack_argb(r, g, b, a);
            assert_eq!(unpack_argb(p), (r, g, b, a));
        }
    }

    #[test]
    fn pack_layout_is_argb() {
        assert_eq!(pack_argb(0xAB, 0xCD, 0xEF, 0x12), 0x12AB_CDEF);
    }

    #[test]
    fn blend_alpha_zero_preserves_bg() {
        let bg = pack_argb(100, 150, 200, 255);
        let fg = pack_argb(255, 0, 0, 0);
        let result = blend(bg, fg);
        let (r, g, b, _) = unpack_argb(result);
        assert_eq!((r, g, b), (100, 150, 200));
    }

    #[test]
    fn blend_alpha_full_replaces_rgb() {
        let bg = pack_argb(100, 150, 200, 255);
        let fg = pack_argb(50, 80, 110, 255);
        let result = blend(bg, fg);
        let (r, g, b, _) = unpack_argb(result);
        // alpha=255, inv_alpha=1: result.r = (100*1 + 50*255) / 256 = (100 + 12750)/256 ≈ 50
        // Off by at most 1 from fg per channel.
        assert!((r as i32 - 50).abs() <= 1, "r got {}", r);
        assert!((g as i32 - 80).abs() <= 1, "g got {}", g);
        assert!((b as i32 - 110).abs() <= 1, "b got {}", b);
    }

    #[test]
    fn stroke_rect_only_touches_edges() {
        let mut buf = vec![0u32; 10 * 10];
        stroke_rect(&mut buf, 10, 10, 2, 2, 6, 6, 1, pack_argb(255, 0, 0, 255), None, None);
        assert_eq!(buf[5 * 10 + 5], 0);
        let (r, _, _, _) = unpack_argb(buf[2 * 10 + 2]);
        assert!(r > 240, "top-left stroke pixel r={}", r);
        assert_eq!(buf[1 * 10 + 1], 0);
    }

    #[test]
    fn circle_filled_center_is_colour() {
        let mut buf = vec![0u32; 16 * 16];
        circle_filled(&mut buf, 16, 16, 8, 8, 5, pack_argb(255, 0, 0, 255), None, None);
        let (r, g, b, _) = unpack_argb(buf[8 * 16 + 8]);
        assert!(r > 240 && g < 16 && b < 16, "center = ({}, {}, {})", r, g, b);
        assert_eq!(buf[0], 0);
    }

    #[test]
    fn circle_filled_clips_partial_offscreen() {
        let mut buf = vec![0u32; 8 * 8];
        circle_filled(&mut buf, 8, 8, -2, -2, 4, pack_argb(255, 255, 255, 255), None, None);
        let (r, _, _, _) = unpack_argb(buf[0]);
        assert!(r > 200, "buf[0] r={}", r);
    }

    #[test]
    fn blend_alpha_half_is_midpoint() {
        let bg = pack_argb(0, 0, 0, 255);
        let fg = pack_argb(200, 200, 200, 128);
        let result = blend(bg, fg);
        let (r, g, b, _) = unpack_argb(result);
        // (0 * 128 + 200 * 128) / 256 = 100
        assert!((r as i32 - 100).abs() <= 1, "r got {}", r);
        assert!((g as i32 - 100).abs() <= 1, "g got {}", g);
        assert!((b as i32 - 100).abs() <= 1, "b got {}", b);
    }

    #[test]
    fn fill_rect_solid_full_buffer() {
        let mut buf = vec![0u32; 4 * 4];
        fill_rect_solid(&mut buf, 4, 4, 0, 0, 4, 4, 0xFF112233, None);
        assert!(buf.iter().all(|&p| p == 0xFF112233));
    }

    #[test]
    fn fill_rect_solid_partial() {
        let mut buf = vec![0u32; 4 * 4];
        fill_rect_solid(&mut buf, 4, 4, 1, 1, 2, 2, 0xFFAABBCC, None);
        let expected: [u32; 16] = [
            0, 0, 0, 0,
            0, 0xFFAABBCC, 0xFFAABBCC, 0,
            0, 0xFFAABBCC, 0xFFAABBCC, 0,
            0, 0, 0, 0,
        ];
        assert_eq!(buf.as_slice(), &expected);
    }

    #[test]
    fn fill_rect_solid_clips_negative_origin() {
        let mut buf = vec![0u32; 4 * 4];
        // Rect from (-2, -2) of size 4x4: only (0,0)..(2,2) intersects the buffer.
        fill_rect_solid(&mut buf, 4, 4, -2, -2, 4, 4, 0xFF000001, None);
        let expected: [u32; 16] = [
            0xFF000001, 0xFF000001, 0, 0,
            0xFF000001, 0xFF000001, 0, 0,
            0,          0,          0, 0,
            0,          0,          0, 0,
        ];
        assert_eq!(buf.as_slice(), &expected);
    }

    #[test]
    fn fill_rect_solid_fully_offscreen_is_noop() {
        let mut buf = vec![0u32; 4 * 4];
        fill_rect_solid(&mut buf, 4, 4, 100, 100, 5, 5, 0xFFFFFFFF, None);
        assert!(buf.iter().all(|&p| p == 0));
        fill_rect_solid(&mut buf, 4, 4, -10, -10, 5, 5, 0xFFFFFFFF, None);
        assert!(buf.iter().all(|&p| p == 0));
    }

    #[test]
    fn fill_rect_blend_alpha_zero_no_change() {
        let mut buf = vec![pack_argb(50, 60, 70, 255); 4 * 4];
        fill_rect_blend(&mut buf, 4, 4, 0, 0, 4, 4, pack_argb(255, 0, 0, 0), None, None);
        assert!(buf.iter().all(|&p| {
            let (r, g, b, _) = unpack_argb(p);
            (r, g, b) == (50, 60, 70)
        }));
    }

    #[test]
    fn fill_rect_blend_clips_partial() {
        let mut buf = vec![pack_argb(0, 0, 0, 255); 4 * 4];
        fill_rect_blend(&mut buf, 4, 4, 2, 2, 10, 10, pack_argb(200, 200, 200, 128), None, None);
        // Pixels at (2,2), (3,2), (2,3), (3,3) should be ~(100, 100, 100); rest unchanged.
        for y in 0..4usize {
            for x in 0..4usize {
                let (r, g, b, _) = unpack_argb(buf[y * 4 + x]);
                if x >= 2 && y >= 2 {
                    assert!((r as i32 - 100).abs() <= 1);
                    assert!((g as i32 - 100).abs() <= 1);
                    assert!((b as i32 - 100).abs() <= 1);
                } else {
                    assert_eq!((r, g, b), (0, 0, 0));
                }
            }
        }
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
        let t = Transform::translate(1.0, 0.0).then(Transform::rotate(core::f32::consts::FRAC_PI_2));
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
        assert_eq!(quantize_rotation(-core::f32::consts::FRAC_PI_2, 10.0, 8), n - 8);
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
