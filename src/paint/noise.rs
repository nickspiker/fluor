//! The background noise: Photon's horizontal-streak texture, a random walk per row mirrored about an axis, generated where it is drawn.
//!
//! The walk runs in the four 16-bit lanes of a `u64` (α, R, G, B), and its output is taken as a finished [`G2t16`] pixel (owner, 2026-11-21: "true sixteen and just assume it's g2t so it's a straight thru"): there is no 8-bit picture widened to 16.
//! The 8-bit picture is the 16-bit one's top bytes ([`Pixel::from_g2t16`]).
//!
//! Each row seeds from its texture row alone, so rows are independent and the texture tiles vertically with no seam: a frame scrolled by `s` shows texture row `(y − s) mod height` at row `y`.
//! A row's walk starts at the axis and runs outward each way, the right half stepping its generator with an add and the left with a subtract, so the texture at a pixel never depends on the clip it was drawn in.
//!
//! WHY IT IS GENERATED, NOT CACHED (measured 2026-11-21 at 1344 × 2992, 16 bits; the Pro with its clocks fixed: Tensor, A715 2367 MHz, A510 1704 MHz, MIF 3744):
//!
//! | | generated, 16-bit, 4 rows interleaved | cached 16-bit field | the 8-bit generator before it |
//! |---|---|---|---|
//! | Pro, one A715 | 10.73 ms | 11.65 ms | 26.43 ms |
//! | Pro, four A715s | 5.68 ms | 7.78 ms | 9.45 ms |
//! | Pro, one A510 | 19.37 ms | 20.79 ms | 71.32 ms |
//! | Desktop (Ryzen 5950X, 64 MB L3, so the field is in cache), 4 threads | 1.76 ms | 3.70 ms | — |
//!
//! Generating is faster on every core type, holds no 32 MB field, reads no 32 MB a frame from memory (bandwidth the camera and the GPU share, and DRAM reads cost battery), and leaves nothing to split or roll out of step with the frame.
//!
//! Each worker takes four rows at a time and interleaves their walks, so each row's serial chain of dependent steps overlaps the other three's.

use super::{BlendMode, Clip};
use crate::canvas::Canvas;
use crate::pixel::{G2t16, Pixel};

/// Rows a worker walks together, interleaved.
const GROUP: usize = 1 << 2;
/// Pixels a row's walk writes to the stack before they go into the frame.
const CHUNK: usize = 1 << 6;
/// The generator's step constant (the texture's own, unchanged since the 8-bit walk).
const STIR: u64 = 12_345_678_942;
/// A step whose generator lands below this (its top byte zero, one step in 2⁸) is a speckle.
const SPECKLE_BELOW: u64 = u64::MAX >> 8;
/// The three colour lanes, α's lane excluded.
const LANES: u64 = G2t16::RGB_MASK;
/// The walk's step in each colour lane: up by one draw of this many bits and down by another, each uniform over `0 .. 2⁹`.
/// Bit 8 of each draw is the 8-bit walk's step exactly (one code, 2⁸ at 16 bits, at even odds), and the low byte under it is the 16-bit grain: today's look, with 2⁸ steps inside each code.
/// Measured over a 1344 × 2992 frame, the mean change of each channel across 16, 64 and 256 pixels is the 8-bit walk's to within 13 % (R and G within 4 %), the look kept; one draw of 2⁸ (grain alone, no code step) moved 20 to 40 % less across 16 pixels, and its streaks ran longer.
/// The 8-bit walk's variance is only 3/4 of this one's on paper, but each of its borrows between lanes moved the next channel a whole code, where a 16-bit lane's borrow moves it 1/2⁸ of one.
const STEP: u64 = 0x0000_01FF_01FF_01FF;
/// Where the down draw is taken from: the 9 bits above the up draw's in each lane.
/// B's down draw reaches 2 bits into G's lane and G's into R's (the up draws' two lowest bits there); 3 lanes of two 9-bit draws need 54 of the 64 bits, and with α's lane free the only overlap is those low bits, a correlation under 1/2⁸ between neighbouring channels.
const DOWN: u32 = 9;

/// A theme mask (a byte a channel) as the walk's 16-bit lanes: each channel's byte the lane's top byte and its low byte all ones, so a lane wraps at the same codes as the 8-bit walk with 2⁸ steps inside each.
fn widen_mask(m8: u32) -> u64 {
    let lane = |s: u32| ((((m8 >> s) & 0xFF) as u64) << 8) | 0xFF;
    (lane(16) << 32) | (lane(8) << 16) | lane(0)
}

/// A base colour (a byte a channel, visible) as the walk's 16-bit lanes: 8-bit code `c` is 16-bit code `c × 2⁸`, the same light ([`Pixel::from_g2t8`]).
fn widen_base(b8: u32) -> u64 {
    let lane = |s: u32| (((b8 >> s) & 0xFF) as u64) << 8;
    (lane(16) << 32) | (lane(8) << 16) | lane(0)
}

/// The walk's constants for one frame's base colour.
struct Walk {
    mask: u64,
    speckle: u64,
    base: u64,
    /// Added to each row's starting colour ([`background_noise`]'s `shimmer`, in 8-bit codes of blue).
    shimmer: u64,
}

impl Walk {
    fn new(base: u32, shimmer: usize) -> Self {
        Self { mask: widen_mask(crate::theme::BG_MASK), speckle: widen_mask(crate::theme::BG_SPECKLE), base: widen_base(base), shimmer: (shimmer as u64) << 8 }
    }

    /// One step of a row's walk: the generator stirred (an add going right, a subtract going left), then the colour either a speckle or moved by the step.
    /// The lanes wrap: a lane's step carries into or borrows from the next lane's low bits and the mask takes it back off, which is the walk's wrap at its mask and the 8-bit walk's own packed arithmetic.
    #[inline(always)]
    fn step<const LEFT: bool>(&self, rng: &mut u64, colour: &mut u64) {
        let r = rng.rotate_left(13);
        *rng ^= if LEFT { r.wrapping_sub(STIR) } else { r.wrapping_add(STIR) };
        *colour = if *rng < SPECKLE_BELOW {
            (*rng >> 8) & self.speckle
        } else {
            (colour.wrapping_add(*rng & STEP) & self.mask).wrapping_sub((*rng >> DOWN) & STEP) & self.mask
        };
    }

    /// The colour as a finished opaque pixel: the base added (each lane's sum stays inside it while each channel's mask and base bytes sum under 2⁸, as the 8-bit walk needed), and the visible value turned to darkness.
    #[inline(always)]
    fn pixel(&self, colour: u64) -> G2t16 {
        ((colour.wrapping_add(self.base) & LANES) ^ LANES) | G2t16::OPAQUE
    }
}

/// Texture row `t`'s seed in a texture `height` rows tall.
#[inline(always)]
fn seed(t: usize, height: usize) -> u64 {
    0xDEAD_BEEF_0123_4567u64 ^ (t as u64).wrapping_sub(height as u64 / 2).wrapping_mul(0x9E37_79B9_4517_B397)
}

/// A run of noise into the frame: a store where the frame is empty (front to back, nothing is in front of it there), else `under` what is in front.
/// The store is exact; `under` into an empty pixel would floor each darkness by one, so an empty pixel is never handed to it.
/// `empty` is a run of empty pixels as long as `dst`: comparing the two slices is one memory compare, where a pixel-by-pixel test cost a fifth of the whole pass.
#[inline(always)]
fn put<P: Pixel>(dst: &mut [P], src: &[P], empty: &[P]) {
    if dst == empty {
        dst.copy_from_slice(src);
        return;
    }
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = if *d == P::EMPTY { s } else { d.under(s, BlendMode::Normal) };
    }
}

/// One half of a group's rows: the walks of `t`'s texture rows from the axis outward (rightward from `axis`, or leftward from `axis − 1`), stepped unstored up to the clip and put into `[lo, hi)`.
/// `group` is the group's rows (`rows` of them, each `w` long); a group short of four rows still walks four chains and puts only its own.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn half<P: Pixel, const LEFT: bool>(group: &mut [P], w: usize, rows: usize, t: [usize; GROUP], height: usize, axis: usize, lo: usize, hi: usize, walk: &Walk) {
    let mut rng = t.map(|t| seed(t, height));
    let mut colour = rng.map(|r| r.wrapping_add(walk.shimmer) & walk.mask);
    // The pixels between the axis and the clip are walked, not drawn, so the texture is the same whatever the clip.
    let skip = if LEFT { axis - hi } else { lo - axis };
    for _ in 0..skip {
        for j in 0..GROUP {
            walk.step::<LEFT>(&mut rng[j], &mut colour[j]);
        }
    }
    let mut buf = [[P::EMPTY; CHUNK]; GROUP];
    let empty = [P::EMPTY; CHUNK];
    let span = hi - lo;
    let mut done = 0;
    while done < span {
        // WHY/PROOF: the span's last chunk is what is left of it, shorter than CHUNK; every other is CHUNK whole.
        let len = (span - done).min(CHUNK);
        for i in 0..len {
            // Going left the walk meets the chunk's pixels right to left.
            let at = if LEFT { len - 1 - i } else { i };
            for j in 0..GROUP {
                walk.step::<LEFT>(&mut rng[j], &mut colour[j]);
                buf[j][at] = P::from_g2t16(walk.pixel(colour[j]));
            }
        }
        let x = if LEFT { hi - done - len } else { lo + done };
        for j in 0..rows {
            put(&mut group[j * w + x..j * w + x + len], &buf[j][..len], &empty[..len]);
        }
        done += len;
    }
}

/// Lay the noise under `canvas` inside `clip`, mirrored about the column `axis`, scrolled down by `scroll` rows, in colour `base` (`None` = [`crate::theme::BG_BASE`]); `fullscreen` false leaves the frame's 1-pixel edge for the window's hairline.
/// Rows seed from their texture row (`(y − scroll) mod height`), so any clip of a frame gets exactly the pixels a whole frame has there, and a frame moved by `d` rows shows the texture the frame at `scroll + d` does.
pub fn noise_under<P: Pixel>(canvas: &mut Canvas<P>, axis: usize, scroll: isize, fullscreen: bool, clip: Option<Clip>, base: Option<u32>) {
    lay(canvas, axis, scroll, 0, fullscreen, clip, base);
}

/// [`noise_under`] mirrored about the frame's centre column, with `shimmer` (8-bit codes of blue) added to each row's starting colour: a colour bias that leaves the walk's shape alone.
pub fn background_noise<P: Pixel>(canvas: &mut Canvas<P>, shimmer: usize, fullscreen: bool, scroll_offset: isize, clip: Option<Clip>, base: Option<u32>) {
    let axis = canvas.width / 2;
    lay(canvas, axis, scroll_offset, shimmer, fullscreen, clip, base);
}

fn lay<P: Pixel>(canvas: &mut Canvas<P>, axis: usize, scroll: isize, shimmer: usize, fullscreen: bool, clip: Option<Clip>, base: Option<u32>) {
    let (w, h) = (canvas.width, canvas.height);
    // Under 2 × 2 there is no inside to the 1-pixel edge.
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
    let walk = Walk::new(base.unwrap_or(crate::theme::BG_BASE), shimmer);
    crate::par::par_chunks(&mut canvas.pixels[ys * w..ye * w], GROUP * w, |at, group| {
        let y0 = ys + at / w;
        let t: [usize; GROUP] = core::array::from_fn(|j| (y0 as isize + j as isize - scroll).rem_euclid(h as isize) as usize);
        let rows = group.len() / w;
        // The row splits at the axis: [max(xs, axis), xe) is the right half's part of the clip and [xs, min(xe, axis)) the left's, the clip's two pieces (not a clamp); an axis outside the clip leaves one of them empty.
        if xe > axis {
            half::<P, false>(group, w, rows, t, h, axis, xs.max(axis), xe, &walk);
        }
        if xs < axis {
            half::<P, true>(group, w, rows, t, h, axis, xs, xe.min(axis), &walk);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::Damage;
    use crate::pixel::{Argb8, Blend};
    use alloc::vec;
    use alloc::vec::Vec;

    fn frame<P: Pixel>(w: usize, h: usize, axis: usize, scroll: isize, clip: Option<Clip>) -> Vec<P> {
        let mut px = vec![P::EMPTY; w * h];
        let mut d = Damage::new();
        noise_under(&mut Canvas::new(&mut px, w, h, &mut d), axis, scroll, true, clip, None);
        px
    }

    /// The same frame twice is the same pixels, and a clip of it (any rows, any columns, either side of the axis or across it) is exactly the whole frame's pixels there and nothing outside.
    #[test]
    fn rows_are_deterministic() {
        let (w, h) = (90, 37);
        let a = frame::<G2t16>(w, h, w / 2, 5, None);
        assert_eq!(a, frame::<G2t16>(w, h, w / 2, 5, None));
        assert!(a.iter().all(|&p| p.alpha() == G2t16::ONE));
        for (x0, y0, x1, y1) in [(0, 0, w, 1), (3, 6, 20, 30), (50, 1, 89, 36), (40, 9, 51, 10), (0, 35, w, h)] {
            let b = frame::<G2t16>(w, h, w / 2, 5, Some(Clip::new(x0, y0, x1, y1)));
            for y in 0..h {
                for x in 0..w {
                    let inside = (x0..x1).contains(&x) && (y0..y1).contains(&y);
                    assert_eq!(b[y * w + x], if inside { a[y * w + x] } else { G2t16::EMPTY }, "clip {x0},{y0}..{x1},{y1} at {x},{y}");
                }
            }
        }
    }

    /// Two scrolls show the same texture shifted by their difference, wrapping at the frame's height (so it tiles with no seam), and a scroll of the whole height is no scroll.
    #[test]
    fn a_scroll_shifts_the_texture() {
        let (w, h) = (64, 29);
        let a = frame::<G2t16>(w, h, w / 3, 0, None);
        assert_eq!(a, frame::<G2t16>(w, h, w / 3, h as isize, None));
        for s in [1isize, 7, -11, 40] {
            let b = frame::<G2t16>(w, h, w / 3, s, None);
            for y in 0..h {
                let t = (y as isize - s).rem_euclid(h as isize) as usize;
                assert_eq!(b[y * w..(y + 1) * w], a[t * w..(t + 1) * w], "scroll {s} row {y}");
            }
        }
    }

    /// The 8-bit picture is the 16-bit one's top byte in every field.
    #[test]
    fn eight_bits_are_the_top_byte() {
        let (w, h) = (75, 23);
        let (a, b) = (frame::<G2t16>(w, h, 30, 3, None), frame::<Argb8>(w, h, 30, 3, None));
        for (&p16, &p8) in a.iter().zip(&b) {
            for k in 0..4 {
                assert_eq!((p8 >> (8 * k)) & 0xFF, ((p16 >> (16 * k + 8)) & 0xFF) as u32);
            }
        }
    }

    /// The noise goes behind what is already there: an opaque pixel keeps itself, a partly covered one takes the noise under it, an empty one takes it as it is.
    #[test]
    fn it_goes_under() {
        let (w, h) = (40, 8);
        let bare = frame::<G2t16>(w, h, w / 2, 0, None);
        let mut px = vec![G2t16::EMPTY; w * h];
        let (opaque, half_on) = (G2t16::OPAQUE | 0x1234_5678_9ABC, (G2t16::ONE as u64 / 2) << 48 | 0x0400_0500_0600);
        px[3] = opaque;
        px[w + 30] = half_on;
        let mut d = Damage::new();
        noise_under(&mut Canvas::new(&mut px, w, h, &mut d), w / 2, 0, true, None, None);
        assert_eq!(px[3], opaque);
        assert_eq!(px[w + 30], half_on.under(bare[w + 30], BlendMode::Normal));
        assert_eq!(px[4], bare[4]);
    }
}
