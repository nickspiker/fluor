//! The layer stack's geometry: bands and spans (ferros WINDOWS.md, "The layer stack", 2026-11-02), shared by ferros's display server and by any host that stacks windows.
//! The layers' top and bottom edges cut the screen into bands of rows; within a band every row is laid out alike, so the layers' left and right edges cut it into spans. A span's chain is the layers covering it, frontmost first, cut after the first opaque one: nothing behind an opaque layer is ever read. The layout is rebuilt when the layers change, never per frame; composing a row is walking its band's spans.
//! No allocation: the caller owns the bands (`N` layers at most, `M` ≥ 2N + 1 spans a band and bands in all), so a bare-metal compositor keeps them in statics.

/// A layer's place, in pixels, and whether it promises no see-through pixel anywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layer {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    pub opaque: bool,
}

/// A run of columns laid out alike within a band: the layers covering it, frontmost first, as indices into the layout's layers, cut after the first opaque one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span<const N: usize> {
    pub x0: u16,
    pub x1: u16,
    pub n: u8,
    pub chain: [u8; N],
}
impl<const N: usize> Span<N> {
    pub const NONE: Self = Self { x0: 0, x1: 0, n: 0, chain: [0; N] };
    pub fn chain(&self) -> &[u8] {
        &self.chain[..self.n as usize]
    }
}

/// A run of rows laid out alike: its spans, left to right, covering the whole width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Band<const N: usize, const M: usize> {
    pub y0: u16,
    pub y1: u16,
    pub n: u8,
    pub spans: [Span<N>; M],
}
impl<const N: usize, const M: usize> Band<N, M> {
    pub const NONE: Self = Self { y0: 0, y1: 0, n: 0, spans: [Span::NONE; M] };
    pub fn spans(&self) -> &[Span<N>] {
        &self.spans[..self.n as usize]
    }
}

/// The sorted, distinct edges along one axis that fall strictly inside `0..end`, with 0 first and `end` last: at most 2N + 2 of them.
fn edges<const E: usize>(vals: impl Iterator<Item = i32>, end: usize) -> ([u16; E], usize) {
    let mut e = [0u16; E];
    let mut m = 1;
    for v in vals {
        if v > 0 && (v as usize) < end {
            let v = v as u16;
            let mut at = m;
            while at > 1 && e[at - 1] > v {
                e[at] = e[at - 1];
                at -= 1;
            }
            if at > 1 && e[at - 1] == v {
                // Already there: the shifted tail moves back.
                for i in at..m {
                    e[i] = e[i + 1];
                }
                continue;
            }
            e[at] = v;
            m += 1;
        }
    }
    e[m] = end as u16;
    (e, m + 1)
}

/// The bands of a `width × height` screen under `layers` (frontmost first, at most N of them, each clipped to the screen by the caller or not: a layer past the screen only contributes the edges inside it), written into `out`, which holds at least 2N + 1 bands of M ≥ 2N + 1 spans. Returns how many bands there are. A span nothing covers has an empty chain.
pub fn layout<const N: usize, const M: usize>(layers: &[Layer], width: usize, height: usize, out: &mut [Band<N, M>]) -> usize {
    assert!(layers.len() <= N && M >= 2 * N + 1 && out.len() >= 2 * N + 1, "layout: more layers than the bands were sized for");
    // E = 2N + 2 edges along an axis; the arrays are sized by M, which is at least 2N + 1, plus one.
    let (ys, ny) = edges::<M>(layers.iter().flat_map(|l| [l.y0, l.y1]), height);
    let mut nb = 0;
    for b in 0..ny - 1 {
        let (y0, y1) = (ys[b], ys[b + 1]);
        let rows = || layers.iter().enumerate().filter(move |(_, l)| l.y0 <= y0 as i32 && l.y1 >= y1 as i32);
        let (xs, nx) = edges::<M>(rows().flat_map(|(_, l)| [l.x0, l.x1]), width);
        let mut band = Band { y0, y1, n: 0, spans: [Span::NONE; M] };
        for s in 0..nx - 1 {
            let (x0, x1) = (xs[s], xs[s + 1]);
            let mut span = Span { x0, x1, n: 0, chain: [0; N] };
            for (i, l) in rows() {
                if l.x0 <= x0 as i32 && l.x1 >= x1 as i32 {
                    span.chain[span.n as usize] = i as u8;
                    span.n += 1;
                    if l.opaque {
                        break;
                    }
                }
            }
            band.spans[band.n as usize] = span;
            band.n += 1;
        }
        out[nb] = band;
        nb += 1;
    }
    nb
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A keyboard over the lower half of an app that fills the screen: three bands (above, beside, below the keyboard's rows), and in the keyboard's band three spans whose middle one chains keyboard then app; the app alone elsewhere. With the keyboard opaque the middle chain stops at it.
    #[test]
    fn keyboard_over_app() {
        let app = Layer { x0: 0, y0: 0, x1: 100, y1: 200, opaque: false };
        let kb = Layer { x0: 10, y0: 120, x1: 90, y1: 190, opaque: false };
        let mut out = [Band::<2, 5>::NONE; 5];
        let n = layout(&[kb, app], 100, 200, &mut out);
        assert_eq!(n, 3);
        assert_eq!((out[0].y0, out[0].y1, out[0].n), (0, 120, 1));
        assert_eq!(out[0].spans[0].chain(), &[1]);
        assert_eq!((out[1].y0, out[1].y1, out[1].n), (120, 190, 3));
        assert_eq!((out[1].spans[1].x0, out[1].spans[1].x1), (10, 90));
        assert_eq!(out[1].spans[1].chain(), &[0, 1]);
        assert_eq!(out[1].spans[0].chain(), &[1]);
        assert_eq!((out[2].y0, out[2].y1, out[2].n), (190, 200, 1));
        let kb = Layer { opaque: true, ..kb };
        layout(&[kb, app], 100, 200, &mut out);
        assert_eq!(out[1].spans[1].chain(), &[0]);
    }

    /// Nothing shown: one band, one span, an empty chain. Shared edges fold into one.
    #[test]
    fn empty_and_shared_edges() {
        let mut out = [Band::<2, 5>::NONE; 5];
        assert_eq!(layout(&[], 10, 10, &mut out), 1);
        assert_eq!((out[0].y0, out[0].y1, out[0].n, out[0].spans[0].n), (0, 10, 1, 0));
        let a = Layer { x0: 0, y0: 0, x1: 5, y1: 10, opaque: false };
        let b = Layer { x0: 5, y0: 0, x1: 10, y1: 10, opaque: false };
        assert_eq!(layout(&[a, b], 10, 10, &mut out), 1);
        assert_eq!(out[0].n, 2);
        assert_eq!(out[0].spans[0].chain(), &[0]);
        assert_eq!(out[0].spans[1].chain(), &[1]);
    }
}
