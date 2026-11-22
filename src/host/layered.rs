//! The layered show (ferros docs/GPU-COMPOSE.md 3.1 and 6.4; the display's payloads/display/src/layered.rs, forge 5dc381dd): a window handed to the display as its layers instead of one flattened surface, so the noise is parameters the display generates under the window and the chrome a layer of its own in front of the app's.
//!
//! The form the display reads, every field little-endian: op 22's 50-byte form (its ring none: a layer carries its own), then `[n u8]` and n layers, frontmost first:
//! - a pixel layer, 29 bytes: `[0][source u64][format u8: 2, BAKED][flags u8: bit 0 opaque][x u16][y u16][w u16][h u16][ring's source y u16][its height u16][origin u16][tiles u32]`, a surface of the caller's covering the screen rect; a ring is the layer's own rows (source y = y, height = h, origin under h), a height of 0 none; tiles the place of its coverage bytes in its surface ([`coverage::AT`]), 0 none.
//! - the background, 27 bytes, last: `[1][x u16][y u16][w u16][h u16][axis u16][scroll i32][base u32, G#RRGGBB][shimmer u16][full u8][frame u32][generator u8]`, fluor's noise ([`crate::paint::NoiseLayer`]) under what the pixel layers leave clear, inside its clip.
//!
//! No op of its own: the doc's "new op beside op 22" became op 22's tail when the display was built (GPU-COMPOSE 6.4), and op 5's feature byte (14) bit 1 says a display reads it ([`FEATURE_LAYERS`]), as bit 0 says it reads a ring ([`FEATURE_RING`]).
//! The display takes a tail that does not parse (a source not the caller's, a rect past the screen, a ring that is not its layer's own rows, more than [`PIX_MOST`] pixel layers) as the plain show its first 50 bytes are.
//!
//! Kept apart from the ferros host's syscalls, so the bytes are built and tested on every host.

use crate::canvas::PixelRect;
use crate::paint::NoiseLayer;
use crate::paint::ring::Ring;

/// The show's op (ferros WINDOWS.md), whose 50-byte form the layers follow.
pub const OP_SHOW: u8 = 22;
/// Op 22's 50-byte form: the head every layered show starts with.
pub const HEAD: usize = 50;
/// Op 5's reply byte that holds the display's features.
pub const FEATURES_AT: usize = 14;
/// Op 5's feature bit 0: the display reads a ring in op 22's 50-byte form.
pub const FEATURE_RING: u8 = 1;
/// Op 5's feature bit 1 (forge 2026-11-21): the display reads the layered form after op 22's 50 bytes.
pub const FEATURE_LAYERS: u8 = 1 << 1;
/// The most pixel layers a window carries (the display's `PIX`: chrome and content, double-buffered or not).
pub const PIX_MOST: usize = 1 << 2;
const PIX_BYTES: usize = 29;
const NOISE_BYTES: usize = 27;
const KIND_PIX: u8 = 0;
const KIND_NOISE: u8 = 1;
/// A pixel layer's format: the panel's codes (`panel::BAKED`), the only one the display composes as a layer.
const FORMAT: u8 = crate::host::panel::BAKED;
const FLAG_OPAQUE: u8 = 1;
/// The background's generator byte: 1, the parallel form (lowbias32 hashes and a segmented scan) on the GPU; the display's NEON patches use fluor's `noise_under` with the same parameters.
pub const GENERATOR_PARALLEL: u8 = 1;
/// The longest layered show: the head, the count, four pixel layers and the background.
pub const SHOW_MOST: usize = HEAD + 1 + PIX_MOST * PIX_BYTES + NOISE_BYTES;

/// A layer's tile coverage (ferros_gpuq `coverage`, COMPOSITOR.md rule 2): a byte a tile of 2^[`TILE_LOG2`](coverage::TILE_LOG2) pixels square, row by row across the surface, in the layer's surface [`AT`](coverage::AT) bytes from its slot's start, past the pixels.
/// The display carries the place and reads none of it yet (GPU-COMPOSE knob 10); the numbers are ferros's, carried here, since fluor does not depend on ferros.
pub mod coverage {
    pub const TILE_LOG2: u32 = 6;
    /// From the surface slot's start (ferros cachemap: a 2^25-byte slot, the pixels from its porch on).
    pub const AT: usize = (32 << 20) - (64 << 10);
    /// The bytes of a surface's slot before its pixels: a surface's `dma` base is its slot's start plus this (ferros cachemap `SURFACE_PORCH`).
    pub const PORCH: usize = 1 << 6;
    pub const EMPTY: u8 = 0;
    pub const PARTIAL: u8 = 1;
    pub const OPAQUE: u8 = 2;
    /// Set on a tile the show wrote.
    pub const CHANGED: u8 = 1 << 7;

    /// The tiles across a surface `width` pixels wide (a partial tile at the right edge counts).
    pub const fn tiles_across(width: usize) -> usize {
        (width + (1 << TILE_LOG2) - 1) >> TILE_LOG2
    }

    /// A tile's class from its pixels that hold anything (`any`) and that are opaque (`opaque`) of the `pixels` it covers.
    pub const fn class(any: u32, opaque: u32, pixels: u32) -> u8 {
        if any == 0 {
            EMPTY
        } else if opaque == pixels {
            OPAQUE
        } else {
            PARTIAL
        }
    }

    /// Each tile that `rects` (surface pixels) touch classed again from the surface's baked words and marked [`CHANGED`]; every other tile keeps its class with [`CHANGED`] cleared.
    /// A word with the see-through bit (63) holds nothing; one whose transparency (bits 49-62) is at most its lowest step is opaque (`PanelPixel::to_baked`: a whole α bakes to 1 at 16 bits, to 0 at 8).
    /// `tiles` holds `tiles_across(stride)` bytes a row of tiles, as many rows as cover the surface's `surface.len() / stride` rows.
    pub fn classify(tiles: &mut [u8], surface: &[u64], stride: usize, rects: &[crate::canvas::PixelRect]) {
        let (across, height) = (tiles_across(stride), surface.len() / stride);
        let down = tiles_across(height);
        crate::par::par_rows(tiles, across, 0, down, |ty, row| {
            // WHY/PROOF: the last row and column of tiles may be partial (2992 rows is 46 tiles and 48 rows); the surface ends there.
            let (y0, y1) = (ty << TILE_LOG2, ((ty + 1) << TILE_LOG2).min(height));
            for (tx, b) in row.iter_mut().enumerate() {
                let (x0, x1) = (tx << TILE_LOG2, ((tx + 1) << TILE_LOG2).min(stride));
                if !rects.iter().any(|r| r.x0 < x1 && x0 < r.x1 && r.y0 < y1 && y0 < r.y1) {
                    *b &= !CHANGED;
                    continue;
                }
                let (mut any, mut opaque) = (0u32, 0u32);
                for y in y0..y1 {
                    for &v in &surface[y * stride + x0..y * stride + x1] {
                        let held = v >> 63 == 0;
                        any += held as u32;
                        opaque += (held && (v >> 49) & 0x3FFF <= 1) as u32;
                    }
                }
                *b = class(any, opaque, ((y1 - y0) * (x1 - x0)) as u32) | CHANGED;
            }
        });
    }
}

/// A pixel layer of the show: a surface of the caller's, `BAKED`, covering a rect of the screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixLayer {
    /// The surface (or a mailbox's buffer) it reads: the caller's own, by its `dma` base.
    pub source: u64,
    /// Nothing behind it shows thru anywhere in its rect: the display stops there.
    pub opaque: bool,
    /// The screen rect it covers.
    pub rect: PixelRect,
    /// Its rows a ring (`paint::ring`), the ring the layer's own rows: the origin, under the rect's height; None, linear.
    pub ring: Option<usize>,
    /// Where its coverage bytes lie in its surface, from its slot's start ([`coverage::AT`]), 0 none.
    pub tiles: u32,
}

/// The background layer of the show: the noise in screen coordinates, as the display generates it under the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Background {
    /// Its clip on the screen.
    pub clip: PixelRect,
    /// The mirror column on the screen.
    pub axis: usize,
    /// The texture's row offset on the screen.
    pub scroll: i32,
    /// The colour, G#RRGGBB.
    pub base: u32,
    pub shimmer: u16,
    pub full: bool,
    pub frame: u32,
}

impl Background {
    /// A frame's noise ([`NoiseLayer`], frame pixels) for a `w × h` frame at (`fx`, `fy`) on the screen: its clip (the whole frame where it names none) and its axis moved onto the screen, its scroll by the frame's top so each frame row keeps its texture row, and its base resolved (None: the theme's [`crate::theme::BG_BASE`]).
    /// The display's texture is the screen's height, and its 1-pixel edge (`full` false) the screen's: a frame that is the whole screen gets the frame's own noise, a smaller one the same statistics (GPU-COMPOSE "Decided": the noise needs no match).
    /// None when the clip holds nothing of the frame: the display refuses a tail with an empty rect, and there is no noise to lay.
    pub fn of(n: &NoiseLayer, fx: usize, fy: usize, w: usize, h: usize) -> Option<Self> {
        let c = crate::paint::Clip::resolve(n.clip, w, h);
        if c.x_start >= c.x_end || c.y_start >= c.y_end {
            return None;
        }
        Some(Self {
            clip: PixelRect::new(fx + c.x_start, fy + c.y_start, fx + c.x_end, fy + c.y_end),
            axis: fx + n.axis,
            scroll: (n.scroll + fy as isize) as i32,
            base: n.base.unwrap_or(crate::theme::BG_BASE) & 0xFF_FFFF,
            shimmer: n.shimmer,
            full: n.full,
            frame: n.frame,
        })
    }
}

/// Whether a ring over `pane` can be said as a layer's own rows within a `frame` (both screen rects): a pane as wide as the frame, so the rows above and below it are layers of their own ([`content`]).
pub fn ring_fits(frame: PixelRect, pane: PixelRect) -> bool {
    pane.x0 == frame.x0 && pane.x1 == frame.x1 && pane.y0 >= frame.y0 && pane.y1 <= frame.y1
}

/// The pixel layers of a frame's own surface `source` (screen rects): the frame whole, or with a ring over a pane as wide as the frame ([`ring_fits`]), the rows above the pane, the pane on its ring and the rows below it, each a layer (the display reads a ring only over a layer's own rows).
/// The layers into `out`, how many returned; a ring that does not fit is a bug in the host, which homes such a ring and moves the pane's pixels instead.
pub fn content(source: u64, frame: PixelRect, ring: Option<Ring>, tiles: u32, out: &mut [PixLayer; 3]) -> usize {
    let layer = |rect: PixelRect, ring: Option<usize>| PixLayer { source, opaque: false, rect, ring, tiles };
    let Some(g) = ring else {
        out[0] = layer(frame, None);
        return 1;
    };
    assert!(ring_fits(frame, g.rect), "a ring over a pane narrower than the frame cannot be said as a layer");
    let mut n = 0;
    for (rect, ring) in [(PixelRect::new(frame.x0, frame.y0, frame.x1, g.rect.y0), None), (g.rect, Some(g.origin)), (PixelRect::new(frame.x0, g.rect.y1, frame.x1, frame.y1), None)] {
        if !rect.is_empty() {
            out[n] = layer(rect, ring);
            n += 1;
        }
    }
    n
}

/// The layers after a show's 50-byte head already in `m` (the head's own ring none): `[n]`, the pixel layers frontmost first, then the background when there is one. The show's length returned.
/// More than [`PIX_MOST`] pixel layers is a bug in the host, and panics.
pub fn tail(m: &mut [u8; SHOW_MOST], pix: &[PixLayer], noise: Option<&Background>) -> usize {
    assert!(pix.len() <= PIX_MOST, "more pixel layers than the display composes");
    let mut at = HEAD;
    let mut put = |bytes: &[u8]| {
        m[at..at + bytes.len()].copy_from_slice(bytes);
        at += bytes.len();
    };
    put(&[(pix.len() + noise.is_some() as usize) as u8]);
    let rect = |r: PixelRect| -> [u8; 8] {
        let mut b = [0u8; 8];
        for (k, v) in [r.x0, r.y0, r.width(), r.height()].into_iter().enumerate() {
            b[2 * k..2 * k + 2].copy_from_slice(&(v as u16).to_le_bytes());
        }
        b
    };
    for p in pix {
        put(&[KIND_PIX]);
        put(&p.source.to_le_bytes());
        put(&[FORMAT, if p.opaque { FLAG_OPAQUE } else { 0 }]);
        put(&rect(p.rect));
        let (sy, sh, o) = match p.ring {
            Some(o) => (p.rect.y0, p.rect.height(), o),
            None => (0, 0, 0),
        };
        for v in [sy, sh, o] {
            put(&(v as u16).to_le_bytes());
        }
        put(&p.tiles.to_le_bytes());
    }
    if let Some(b) = noise {
        put(&[KIND_NOISE]);
        put(&rect(b.clip));
        put(&(b.axis as u16).to_le_bytes());
        put(&b.scroll.to_le_bytes());
        put(&b.base.to_le_bytes());
        put(&b.shimmer.to_le_bytes());
        put(&[b.full as u8]);
        put(&b.frame.to_le_bytes());
        put(&[GENERATOR_PARALLEL]);
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A known frame's show, byte for byte as the display's layered.rs parses it: the chrome on a surface of its own, the content split round its ring, the noise last.
    #[test]
    fn a_known_frame_shows_these_bytes() {
        let frame = PixelRect::new(0, 0, 1344, 2992);
        let (content_at, chrome_at) = (0x0012_3400_0040u64, 0x0056_7800_0040u64);
        let mut layers = [PixLayer { source: chrome_at, opaque: false, rect: frame, ring: None, tiles: coverage::AT as u32 }; PIX_MOST];
        let mut c = [layers[0]; 3];
        let ring = Ring { rect: PixelRect::new(0, 200, 1344, 2800), origin: 37 };
        let n = content(content_at, frame, Some(ring), coverage::AT as u32, &mut c);
        assert_eq!(n, 3);
        layers[1..4].copy_from_slice(&c);
        let noise = NoiseLayer { axis: 672, scroll: -5, base: Some(0x0C_14_0E), shimmer: 3, full: true, clip: None, frame: 0 };
        let bg = Background::of(&noise, 0, 0, 1344, 2992).unwrap();
        let mut m = [0u8; SHOW_MOST];
        m[0] = OP_SHOW;
        let len = tail(&mut m, &layers, Some(&bg));
        assert_eq!(len, SHOW_MOST);
        let mut want = alloc::vec![4 + 1];
        let pix = |want: &mut alloc::vec::Vec<u8>, src: u64, r: [u16; 4], ring: [u16; 3]| {
            want.push(0);
            want.extend(src.to_le_bytes());
            want.extend([2, 0]);
            for v in r.into_iter().chain(ring) {
                want.extend(v.to_le_bytes());
            }
            want.extend((coverage::AT as u32).to_le_bytes());
        };
        pix(&mut want, chrome_at, [0, 0, 1344, 2992], [0, 0, 0]);
        pix(&mut want, content_at, [0, 0, 1344, 200], [0, 0, 0]);
        pix(&mut want, content_at, [0, 200, 1344, 2600], [200, 2600, 37]);
        pix(&mut want, content_at, [0, 2800, 1344, 192], [0, 0, 0]);
        want.push(1);
        for v in [0u16, 0, 1344, 2992, 672] {
            want.extend(v.to_le_bytes());
        }
        want.extend((-5i32).to_le_bytes());
        want.extend(0x000C_140Eu32.to_le_bytes());
        want.extend(3u16.to_le_bytes());
        want.push(1);
        want.extend(0u32.to_le_bytes());
        want.push(GENERATOR_PARALLEL);
        assert_eq!(&m[HEAD..len], &want[..]);
        // The display's own reading of the same bytes (layered.rs `parse`): the background's base at its byte 15, its frame at 22, a pixel layer's ring at 19.
        let at = HEAD + 1 + 2 * PIX_BYTES;
        assert_eq!(u16::from_le_bytes([m[at + 19], m[at + 20]]), 200);
        assert_eq!(u16::from_le_bytes([m[at + 21], m[at + 22]]), 2600);
        assert_eq!(u16::from_le_bytes([m[at + 23], m[at + 24]]), 37);
        let nz = HEAD + 1 + 4 * PIX_BYTES;
        assert_eq!(u32::from_le_bytes(m[nz + 15..nz + 19].try_into().unwrap()), 0x0C_140E);
        assert_eq!(m[nz + 21], 1);
        assert_eq!(u32::from_le_bytes(m[nz + 22..nz + 26].try_into().unwrap()), 0);
    }

    /// A frame smaller than the screen: the noise's clip, axis and scroll move onto the screen, its clip is cut at the frame, and a clip that misses the frame is no background at all; with no ring the content is the frame whole.
    #[test]
    fn a_frame_off_the_origin_moves_its_noise_onto_the_screen() {
        let noise = NoiseLayer { axis: 100, scroll: 7, base: None, shimmer: 0, full: false, clip: Some(crate::paint::Clip::new(10, 20, 900, 900)), frame: 0 };
        let bg = Background::of(&noise, 40, 60, 200, 300).unwrap();
        assert_eq!(bg.clip, PixelRect::new(50, 80, 240, 360));
        assert_eq!((bg.axis, bg.scroll, bg.base), (140, 67, crate::theme::BG_BASE & 0xFF_FFFF));
        assert!(Background::of(&NoiseLayer { clip: Some(crate::paint::Clip::new(300, 0, 400, 10)), ..noise }, 40, 60, 200, 300).is_none());
        let mut c = [PixLayer { source: 0, opaque: false, rect: PixelRect::empty(), ring: None, tiles: 0 }; 3];
        assert_eq!(content(9, PixelRect::new(40, 60, 240, 360), None, 0, &mut c), 1);
        assert_eq!(c[0].rect, PixelRect::new(40, 60, 240, 360));
        assert!(!ring_fits(PixelRect::new(40, 60, 240, 360), PixelRect::new(41, 70, 240, 200)));
        assert!(ring_fits(PixelRect::new(40, 60, 240, 360), PixelRect::new(40, 60, 240, 200)));
        // A pane from the frame's top has no rows above it.
        assert_eq!(content(9, PixelRect::new(40, 60, 240, 360), Some(Ring { rect: PixelRect::new(40, 60, 240, 200), origin: 0 }), 0, &mut c), 2);
        let mut m = [0u8; SHOW_MOST];
        assert_eq!(tail(&mut m, &c[..2], None), HEAD + 1 + 2 * PIX_BYTES);
        assert_eq!(m[HEAD], 2);
    }

    /// The tiles a show wrote are classed from the baked words and marked changed; the rest keep their class and lose the mark.
    #[test]
    fn coverage_classes_the_tiles_written() {
        use coverage::*;
        let (w, h) = (130, 70);
        let see = 1u64 << 63;
        let mut s = alloc::vec![see; w * h];
        // Tile (0, 0) opaque at 16 bits (transparency 1), tile (1, 0) one partial pixel, tile (2, 0) (two columns wide) opaque at 8 bits (transparency 0).
        for y in 0..64 {
            for x in 0..64 {
                s[y * w + x] = 1 << 49 | 0x1234;
            }
            s[y * w + 128] = 0x55;
            s[y * w + 129] = 0x55;
        }
        s[5 * w + 70] = 900 << 49;
        let (across, down) = (tiles_across(w), tiles_across(h));
        assert_eq!((across, down), (3, 2));
        let mut t = alloc::vec![PARTIAL | CHANGED; across * down];
        classify(&mut t, &s, w, &[PixelRect::new(0, 0, w, 64)]);
        assert_eq!(t, [OPAQUE | CHANGED, PARTIAL | CHANGED, OPAQUE | CHANGED, PARTIAL, PARTIAL, PARTIAL]);
        classify(&mut t, &s, w, &[PixelRect::new(60, 64, 70, 70)]);
        assert_eq!(t, [OPAQUE, PARTIAL, OPAQUE, EMPTY | CHANGED, EMPTY | CHANGED, PARTIAL]);
    }
}
