//! Bench: u32 ARGB + parallel u16 hit_id  vs  u64 packed (ARGB + hit_id in same word).
//!
//! Synthetic scene: full-viewport chrome bg + 4 textbox-sized overlays + 1 cursor-sized overlay,
//! all composited front-to-back per pixel with hit_id stamping (first opaque layer wins).
//! Boundary step: flip top byte (t→α) + extract to u32 staging buffer (u64 only).
//!
//! Run: `cargo run --release --example bench_composite`. Release is mandatory — debug renders
//! these kernels ~20x slower and skews the comparison.
//!
//! Output: per-frame wall time for composite, boundary, total. Two variants, two resolutions.

use std::time::Instant;

const ITERS: usize = 100;
const NUM_OVERLAYS: usize = 5; // 4 textbox-ish + 1 cursor-ish

/// One layer: u32 ARGB pixels + a single hit_id stamped at every non-transparent pixel.
struct Layer {
    pixels: Vec<u32>,
    bbox: (usize, usize, usize, usize), // x, y, w, h
    hit_id: u16,
}

fn make_scene(width: usize, height: usize) -> Vec<Layer> {
    let mut layers = Vec::with_capacity(NUM_OVERLAYS + 1);

    // Chrome bg — full viewport, fully opaque (t=0) noise-like pattern.
    let mut bg = vec![0u32; width * height];
    for (i, p) in bg.iter_mut().enumerate() {
        let r = (i & 0x1F) as u32;
        let g = ((i >> 5) & 0x1F) as u32;
        let b = ((i >> 10) & 0x1F) as u32;
        *p = (r << 16) | (g << 8) | b; // t=0, RGB=noise
    }
    layers.push(Layer { pixels: bg, bbox: (0, 0, width, height), hit_id: 1 });

    // Four textbox-sized overlays scattered around — half-opaque (t=128).
    let textbox_w = width / 4;
    let textbox_h = height / 8;
    for k in 0..4 {
        let x = (width / 8) + (k % 2) * (width / 2);
        let y = (height / 8) + (k / 2) * (height / 2);
        let mut buf = vec![0xFF000000u32; textbox_w * textbox_h]; // transparent init
        for i in 0..buf.len() {
            buf[i] = (0x80u32 << 24) | 0x008080; // t=128 (half-opaque), greyish RGB
        }
        layers.push(Layer { pixels: buf, bbox: (x, y, textbox_w, textbox_h), hit_id: (10 + k) as u16 });
    }

    // Cursor — fully opaque small dot.
    let cursor_w = 16;
    let cursor_h = 32;
    let buf = vec![0x00FF_FFFFu32; cursor_w * cursor_h]; // t=0, RGB=white
    layers.push(Layer { pixels: buf, bbox: (width / 2 - 8, height / 2 - 16, cursor_w, cursor_h), hit_id: 100 });

    layers
}

#[inline(always)]
fn alpha_over_u32(dst: u32, src: u32) -> u32 {
    let src_t = ((src >> 24) & 0xFF) as u64;
    if src_t == 0 { return src; }
    let dst_t = ((dst >> 24) & 0xFF) as u64;
    let inv = 256 - src_t;

    let mut bg = dst as u64;
    bg = (bg | (bg << 16)) & 0x0000_FFFF_0000_FFFF;
    bg = (bg | (bg << 8)) & 0x00FF_00FF_00FF_00FF;
    let mut fg = src as u64;
    fg = (fg | (fg << 16)) & 0x0000_FFFF_0000_FFFF;
    fg = (fg | (fg << 8)) & 0x00FF_00FF_00FF_00FF;
    let mut r = fg * inv + bg * src_t;
    r = (r >> 8) & 0x00FF_00FF_00FF_00FF;
    r = (r | (r >> 8)) & 0x0000_FFFF_0000_FFFF;
    r = r | (r >> 16);
    let rgb_only = (r as u32) & 0x00FF_FFFF;
    let result_t = ((src_t * dst_t) >> 8) as u32;
    rgb_only | (result_t << 24)
}

/// u32 variant: target = Vec<u32>, hit_id = Vec<u16> parallel. Compositor stamps hit_id at every
/// non-transparent layer pixel (last write wins under bottom-up = top-of-stack).
fn render_u32(layers: &[Layer], width: usize, height: usize) -> (std::time::Duration, std::time::Duration) {
    let mut target = vec![0u32; width * height];
    let mut hit_buf = vec![0u16; width * height];

    let composite_start = Instant::now();
    for layer in layers {
        let (lx, ly, lw, lh) = layer.bbox;
        for row in 0..lh {
            let ty = ly + row;
            if ty >= height { break; }
            let dst_row = ty * width;
            let src_row = row * lw;
            for col in 0..lw {
                let tx = lx + col;
                if tx >= width { break; }
                let src_px = layer.pixels[src_row + col];
                if src_px == 0xFF000000 { continue; }
                target[dst_row + tx] = alpha_over_u32(target[dst_row + tx], src_px);
                hit_buf[dst_row + tx] = layer.hit_id;
            }
        }
    }
    let composite = composite_start.elapsed();

    let boundary_start = Instant::now();
    for p in target.iter_mut() {
        *p ^= 0xFF000000; // flip t→α
    }
    let boundary = boundary_start.elapsed();

    // Anti-DCE: read back hit_buf so the compiler doesn't elide it.
    std::hint::black_box(&hit_buf);
    std::hint::black_box(&target);
    (composite, boundary)
}

/// u64 variant: target = Vec<u64>, ARGB in low 32 + hit_id in bits 32-47. Single write per pixel
/// covers both. Boundary step extracts u32 staging buffer (flip + low-half mask).
fn render_u64(layers: &[Layer], width: usize, height: usize) -> (std::time::Duration, std::time::Duration) {
    let mut target = vec![0u64; width * height];

    let composite_start = Instant::now();
    for layer in layers {
        let (lx, ly, lw, lh) = layer.bbox;
        let hit_id_packed = (layer.hit_id as u64) << 32;
        for row in 0..lh {
            let ty = ly + row;
            if ty >= height { break; }
            let dst_row = ty * width;
            let src_row = row * lw;
            for col in 0..lw {
                let tx = lx + col;
                if tx >= width { break; }
                let src_px = layer.pixels[src_row + col];
                if src_px == 0xFF000000 { continue; }
                let dst_idx = dst_row + tx;
                let dst_low = (target[dst_idx] & 0xFFFF_FFFF) as u32;
                let new_low = alpha_over_u32(dst_low, src_px);
                target[dst_idx] = (new_low as u64) | hit_id_packed;
            }
        }
    }
    let composite = composite_start.elapsed();

    let boundary_start = Instant::now();
    let mut staging = vec![0u32; width * height];
    for (i, p) in target.iter().enumerate() {
        staging[i] = ((*p) as u32) ^ 0xFF000000; // extract low + flip
    }
    let boundary = boundary_start.elapsed();

    std::hint::black_box(&staging);
    std::hint::black_box(&target);
    (composite, boundary)
}

fn run(label: &str, width: usize, height: usize) {
    let layers = make_scene(width, height);

    // Warm-up
    for _ in 0..3 { let _ = render_u32(&layers, width, height); }
    for _ in 0..3 { let _ = render_u64(&layers, width, height); }

    let mut u32_composite_ns = vec![];
    let mut u32_boundary_ns = vec![];
    let mut u64_composite_ns = vec![];
    let mut u64_boundary_ns = vec![];

    for _ in 0..ITERS {
        let (c, b) = render_u32(&layers, width, height);
        u32_composite_ns.push(c.as_nanos() as u64);
        u32_boundary_ns.push(b.as_nanos() as u64);
    }
    for _ in 0..ITERS {
        let (c, b) = render_u64(&layers, width, height);
        u64_composite_ns.push(c.as_nanos() as u64);
        u64_boundary_ns.push(b.as_nanos() as u64);
    }

    let med = |mut v: Vec<u64>| { v.sort(); v[v.len() / 2] };
    let p99 = |mut v: Vec<u64>| { v.sort(); v[v.len() * 99 / 100] };

    let u32c_med = med(u32_composite_ns.clone());
    let u32c_p99 = p99(u32_composite_ns.clone());
    let u32b_med = med(u32_boundary_ns.clone());
    let u32b_p99 = p99(u32_boundary_ns.clone());
    let u64c_med = med(u64_composite_ns.clone());
    let u64c_p99 = p99(u64_composite_ns.clone());
    let u64b_med = med(u64_boundary_ns.clone());
    let u64b_p99 = p99(u64_boundary_ns.clone());

    let u32_total_med = u32c_med + u32b_med;
    let u64_total_med = u64c_med + u64b_med;

    println!("\n=== {label} ({width}×{height}, {ITERS} iters, {} layers) ===", layers.len());
    println!("                          composite (med / p99)      boundary (med / p99)       total (med)");
    println!("  u32 + parallel u16:     {:>7.2} ms / {:>7.2} ms     {:>7.2} ms / {:>7.2} ms     {:>7.2} ms",
        u32c_med as f64 / 1_000_000.0, u32c_p99 as f64 / 1_000_000.0,
        u32b_med as f64 / 1_000_000.0, u32b_p99 as f64 / 1_000_000.0,
        u32_total_med as f64 / 1_000_000.0);
    println!("  u64 packed:             {:>7.2} ms / {:>7.2} ms     {:>7.2} ms / {:>7.2} ms     {:>7.2} ms",
        u64c_med as f64 / 1_000_000.0, u64c_p99 as f64 / 1_000_000.0,
        u64b_med as f64 / 1_000_000.0, u64b_p99 as f64 / 1_000_000.0,
        u64_total_med as f64 / 1_000_000.0);
    let ratio = u64_total_med as f64 / u32_total_med as f64;
    println!("  ratio (u64/u32):        {:>7.3}×", ratio);
}

fn main() {
    println!("Compositor bench: u32 ARGB + parallel u16 hit_id   vs   u64 packed (ARGB + hit_id)");
    println!("Single-threaded scalar. Median + p99 over {ITERS} iterations after warm-up.");

    run("1080p", 1920, 1080);
    run("1440p", 2560, 1440);
    run("4K",    3840, 2160);
}
