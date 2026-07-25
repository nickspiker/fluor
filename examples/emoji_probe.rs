// Probe: rasterize the red-circle emoji and report per-channel darkness sums — settles RGBA-vs-BGRA from swash empirically.
use fluor::text::{TextRenderer, TextStyle};
fn main() {
    let mut tr = TextRenderer::new();
    let w = 64usize;
    let h = 64usize;
    let mut buf = vec![0u32; w * h];
    let mut damage = fluor::canvas::Damage::new();
    let mut canvas = fluor::canvas::Canvas::new(&mut buf, w, h, &mut damage);
    let style = TextStyle::new(40.0, 0xFF_00_00_00);
    tr.draw_text_left(&mut canvas, "\u{1F534}", 4.0, 32.0, &style, None, None);
    let (mut sr, mut sg, mut sb, mut n) = (0u64, 0u64, 0u64, 0u64);
    for &p in buf.iter() {
        let a = (p >> 24) & 0xFF;
        if a > 128 {
            // darkness bytes; visible = 255 - dark
            sr += (255 - ((p >> 16) & 0xFF)) as u64;
            sg += (255 - ((p >> 8) & 0xFF)) as u64;
            sb += (255 - (p & 0xFF)) as u64;
            n += 1;
        }
    }
    if n == 0 { println!("NO PIXELS — emoji did not rasterize"); return; }
    println!("pixels={} avg visible R={} G={} B={}", n, sr / n, sg / n, sb / n);
    println!("verdict: {}", if sr / n > sb / n * 2 { "RED dominant — desktop packing CORRECT (bits 16-23 = R)" } else if sb / n > sr / n * 2 { "BLUE dominant — channels SWAPPED at the pack" } else { "ambiguous" });
}
