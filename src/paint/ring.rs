//! A scrolling pane as a ring of rows (ferros docs/DISPLAY-SCROLL.md section 2, option B): nothing moves on a scroll, the ring's origin does.
//!
//! For the ring's rect (columns x0..x1, rows y0..y1, H = y1 − y0) and origin o, reader row y inside the rect lives in buffer row `y0 + (y − y0 + o) mod H`, at the rect's columns only; every other pixel of the buffer is where it always was.
//! A scroll by `dy` (positive: the content moved down) is `o ← (o − dy) mod H`, which is exactly what a memmove by `dy` makes a reader see, except that the band a memmove leaves stale holds the rows that left at the other edge instead.
//! Both are inside the frame's damage by `scroll_hint`'s contract (the exposed band is always repainted), so a reader that maps its rows thru the ring sees the memmove's pixels.
//! `y − y0 + o` is under 2H, so the mod is one compare and one subtraction a row: nothing per pixel.
//!
//! The writer (fluor's ferros host) writes each damage rect thru [`Ring::pieces`]; the reader (the ferros display, op 22's ring form) maps each row the same way.
//! When the pane changes under a live ring, [`Ring::home`] rotates the rect once in place, so its rows are linear again (origin 0).

use crate::canvas::PixelRect;
use alloc::vec::Vec;

/// A rect of a row-major buffer whose rows are a ring, in the buffer's own coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring {
    pub rect: PixelRect,
    /// Under the rect's height: the buffer row (from the rect's top) that reader row `rect.y0` lives in.
    pub origin: usize,
}

/// At most this many pieces from [`Ring::pieces`]: the columns left and right of the ring, the rows above and below it, and the ring's own rows in two runs (where they wrap).
pub const RING_PIECES: usize = 6;

impl Ring {
    /// A ring over `rect` at origin 0: every row where a linear reader looks for it.
    #[inline]
    pub fn new(rect: PixelRect) -> Self {
        Self { rect, origin: 0 }
    }

    /// The buffer row of reader row `y`, which is inside the rect's rows.
    #[inline]
    pub fn row(&self, y: usize) -> usize {
        let h = self.rect.y1 - self.rect.y0;
        let r = y - self.rect.y0 + self.origin;
        self.rect.y0 + if r >= h { r - h } else { r }
    }

    /// The ring after its content moved by `dy` rows (positive down), as a memmove by `dy` moves it for a linear reader: the origin is `o − dy` mod H.
    /// `|dy|` is under the rect's height (fluor's ferros host moves a pane only then; a move past it converts the whole pane).
    #[inline]
    pub fn scrolled(self, dy: i32) -> Self {
        let h = (self.rect.y1 - self.rect.y0) as i64;
        let o = (self.origin as i64 - dy as i64).rem_euclid(h);
        Self { rect: self.rect, origin: o as usize }
    }

    /// Rect `r` (reader coordinates, inside the buffer) cut where it meets the ring: each piece a rect in reader coordinates whose rows are contiguous in the buffer, with the buffer row its first row lives in.
    /// Outside the ring's columns or rows a piece is where it is (its own first row); inside, the ring's rows come in at most two runs, split where they wrap.
    pub fn pieces(&self, r: PixelRect) -> ([(PixelRect, usize); RING_PIECES], usize) {
        let mut out = [(PixelRect::empty(), 0usize); RING_PIECES];
        let mut n = 0;
        fn put(out: &mut [(PixelRect, usize); RING_PIECES], n: &mut usize, p: PixelRect, at: usize) {
            if !p.is_empty() {
                out[*n] = (p, at);
                *n += 1;
            }
        }
        let g = self.rect;
        // The ring's columns inside r (the two rects' intersection, not a clamp), and the columns of r left and right of them; a rect that misses the ring's columns or rows is all one piece.
        let (cx0, cx1) = (r.x0.max(g.x0), r.x1.min(g.x1));
        if cx0 >= cx1 || r.y1 <= g.y0 || r.y0 >= g.y1 {
            put(&mut out, &mut n, r, r.y0);
            return (out, n);
        }
        put(&mut out, &mut n, PixelRect::new(r.x0, r.y0, cx0, r.y1), r.y0);
        put(&mut out, &mut n, PixelRect::new(cx1, r.y0, r.x1, r.y1), r.y0);
        // The ring's columns: the rows above and below the ring as they are, the ring's own rows thru the map.
        let (ry0, ry1) = (r.y0.max(g.y0), r.y1.min(g.y1));
        put(&mut out, &mut n, PixelRect::new(cx0, r.y0, cx1, ry0), r.y0);
        put(&mut out, &mut n, PixelRect::new(cx0, ry1, cx1, r.y1), ry1);
        // The reader row that lives in the ring's last buffer row: rows from ry0 run on in the buffer until it, then wrap to the ring's first.
        let h = g.y1 - g.y0;
        let wrap = g.y0 + (h - self.origin);
        if ry0 < wrap && wrap < ry1 {
            put(&mut out, &mut n, PixelRect::new(cx0, ry0, cx1, wrap), self.row(ry0));
            put(&mut out, &mut n, PixelRect::new(cx0, wrap, cx1, ry1), g.y0);
        } else {
            put(&mut out, &mut n, PixelRect::new(cx0, ry0, cx1, ry1), self.row(ry0));
        }
        (out, n)
    }

    /// The rect's rows rotated in place so the ring's origin is 0 (a linear reader then sees what the ring's reader saw): one move of the rect, and the rows that wrap held aside meanwhile, the shorter side of the two.
    /// `buf` is `stride` pixels a row and holds the rect.
    pub fn home<T: Copy + Send + Sync>(&mut self, buf: &mut [T], stride: usize) {
        let o = self.origin;
        if o == 0 {
            return;
        }
        let g = self.rect;
        let (h, w) = (g.y1 - g.y0, g.x1 - g.x0);
        let rows = buf.len() / stride;
        // Reader row y0 + i lives in buffer row y0 + (i + o) mod H, and must end in y0 + i.
        if o <= h - o {
            // The first o buffer rows (reader rows H − o..H) aside, the rest up by o, then the o rows at the bottom.
            let mut aside: Vec<T> = Vec::with_capacity(o * w);
            for y in g.y0..g.y0 + o {
                aside.extend_from_slice(&buf[y * stride + g.x0..y * stride + g.x1]);
            }
            crate::paint::scroll_copy_rect(buf, stride, rows, g.x0, g.y0, g.x1, g.y1, -(o as i32));
            for (i, y) in (g.y1 - o..g.y1).enumerate() {
                buf[y * stride + g.x0..y * stride + g.x1].copy_from_slice(&aside[i * w..(i + 1) * w]);
            }
        } else {
            // The last k = H − o buffer rows (reader rows 0..k) aside, the rest down by k, then the k rows at the top.
            let k = h - o;
            let mut aside: Vec<T> = Vec::with_capacity(k * w);
            for y in g.y1 - k..g.y1 {
                aside.extend_from_slice(&buf[y * stride + g.x0..y * stride + g.x1]);
            }
            crate::paint::scroll_copy_rect(buf, stride, rows, g.x0, g.y0, g.x1, g.y1, k as i32);
            for (i, y) in (g.y0..g.y0 + k).enumerate() {
                buf[y * stride + g.x0..y * stride + g.x1].copy_from_slice(&aside[i * w..(i + 1) * w]);
            }
        }
        self.origin = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 7;
    const H: usize = 23;

    /// A buffer of distinct values (row, column) so any misplaced pixel shows.
    fn fresh() -> Vec<u32> {
        (0..W * H).map(|i| i as u32).collect()
    }

    /// What a reader of `buf` sees thru `ring`: every pixel at its reader position.
    fn read(buf: &[u32], ring: &Ring) -> Vec<u32> {
        let g = ring.rect;
        let mut out = vec![0; W * H];
        for y in 0..H {
            for x in 0..W {
                let sy = if x >= g.x0 && x < g.x1 && y >= g.y0 && y < g.y1 { ring.row(y) } else { y };
                out[y * W + x] = buf[sy * W + x];
            }
        }
        out
    }

    /// `r` written thru the ring's pieces with each reader pixel's value `f(x, y)`.
    fn write(buf: &mut [u32], ring: &Ring, r: PixelRect, f: impl Fn(usize, usize) -> u32) {
        let (ps, n) = ring.pieces(r);
        for &(p, at) in &ps[..n] {
            for (i, y) in (p.y0..p.y1).enumerate() {
                for x in p.x0..p.x1 {
                    buf[(at + i) * W + x] = f(x, y);
                }
            }
        }
    }

    /// The same rect written linearly.
    fn write_lin(buf: &mut [u32], r: PixelRect, f: impl Fn(usize, usize) -> u32) {
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                buf[y * W + x] = f(x, y);
            }
        }
    }

    /// The pane at every origin and every scroll that keeps part of it reads back identical to the moved copy, the band and a damage rect across the ring's edge written thru the pieces (and linearly in the copy), then identical again after `home`.
    #[test]
    fn a_ring_reads_back_as_the_moved_copy_at_every_origin() {
        // A pane narrower than the buffer, with rows above and below it, so the pieces outside the ring are exercised too.
        let pane = PixelRect::new(1, 3, 6, 19);
        let ph = pane.y1 - pane.y0;
        for start in 0..ph {
            for dy in -(ph as i32 - 1)..ph as i32 {
                let mut lin = fresh();
                // The ring at `start`: each reader row of the linear copy put in the buffer row the ring keeps it in.
                let mut ring = Ring { rect: pane, origin: start };
                let mut buf = fresh();
                for y in pane.y0..pane.y1 {
                    let at = ring.row(y);
                    buf[at * W + pane.x0..at * W + pane.x1].copy_from_slice(&lin[y * W + pane.x0..y * W + pane.x1]);
                }
                assert_eq!(read(&buf, &ring), lin, "origin {start} before a move");
                // The memmove in the copy, the origin in the ring.
                crate::paint::scroll_copy_rect(&mut lin, W, H, pane.x0, pane.y0, pane.x1, pane.y1, dy);
                ring = ring.scrolled(dy);
                // The band the move exposed, and a damage rect that crosses the pane's columns and rows, written in both.
                let n = dy.unsigned_abs() as usize;
                let band = if dy > 0 { PixelRect::new(pane.x0, pane.y0, pane.x1, pane.y0 + n) } else { PixelRect::new(pane.x0, pane.y1 - n, pane.x1, pane.y1) };
                let dmg = PixelRect::new(0, 2, W, 9);
                let paint = |x: usize, y: usize| (1000 + y * W + x) as u32 + (start as u32) * 7919;
                for r in [band, dmg] {
                    write_lin(&mut lin, r, paint);
                    write(&mut buf, &ring, r, paint);
                }
                assert_eq!(read(&buf, &ring), lin, "origin {start}, dy {dy}");
                ring.home(&mut buf, W);
                assert_eq!(ring.origin, 0);
                assert_eq!(buf, lin, "origin {start}, dy {dy}, after home");
            }
        }
    }

    /// Every piece is inside the rect it was cut from and they cover it exactly once.
    #[test]
    fn pieces_cover_the_rect_once() {
        let pane = PixelRect::new(2, 4, 5, 17);
        for o in 0..pane.y1 - pane.y0 {
            let ring = Ring { rect: pane, origin: o };
            for r in [PixelRect::new(0, 0, W, H), PixelRect::new(3, 10, 7, 20), PixelRect::new(0, 0, 2, 4), PixelRect::new(2, 4, 5, 17), PixelRect::new(4, 16, 5, 17)] {
                let mut seen = vec![0u8; W * H];
                let (ps, n) = ring.pieces(r);
                for &(p, _) in &ps[..n] {
                    assert!(p.x0 >= r.x0 && p.x1 <= r.x1 && p.y0 >= r.y0 && p.y1 <= r.y1);
                    for y in p.y0..p.y1 {
                        for x in p.x0..p.x1 {
                            seen[y * W + x] += 1;
                        }
                    }
                }
                for y in 0..H {
                    for x in 0..W {
                        let inside = x >= r.x0 && x < r.x1 && y >= r.y0 && y < r.y1;
                        assert_eq!(seen[y * W + x], inside as u8, "origin {o}, rect {r:?}, pixel ({x}, {y})");
                    }
                }
            }
        }
    }
}
