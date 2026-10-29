//! The source gate: the 8-bit pixel layout spelled out by hand stays where 8 bits is the point.
//! A kernel that writes the α mask, the RGB mask, the packed-channel mask or a 24-bit shift itself only works at 8 bits, so outside the files below every pixel field goes thru [`crate::pixel::Pixel`] (`alpha`, `rgb`, `with_alpha`, `channels`, `ONE`, `DEPTH`).
//! The scan reads `src/` at test time, ignores comments (prose may name the layout) and stops at a file's `#[cfg(test)]` (tests build their pixels by hand on purpose).
//! A listed file that no longer spells the layout fails too, so the list only shrinks: take its line out when phase 4 lands it.

use alloc::string::String;
use alloc::vec::Vec;

/// Files and directories that keep the 8-bit layout on purpose, each with why.
/// A path ending in `/` covers everything under it.
const EIGHT_BIT: &[(&str, &str)] = &[
    ("pixel.rs", "the depths' own layouts: the trait's Argb8 and G2t16 impls, the 8-bit SIMD kernel"),
    ("theme.rs", "the authored 8-bit g2t palette and the present tables built from it"),
    ("paint/finalize.rs", "writes the OS's 8-bit (and F16 / 10-bit) surface pixels; a G2t16 present is its own pass"),
    ("paint/shadow.rs", "runs on the finalized OS screen (visible premultiplied 8-bit), after the depth's own buffer is gone"),
    ("paint/argb8.rs", "the 8-bit colour boundary (pack_argb), the theme-delta byte arithmetic, and the overlay, damage outline and control glyphs on the finalized screen"),
    ("text/argb8.rs", "the legacy byte-buffer draws and the additive / subtractive wrapping blits"),
    ("host/", "the hosts' present and chrome code; phase 5 takes ferros to G2t16"),
    ("widgets/button.rs", "phase 4: pill and text caches are Vec<u32>"),
    ("widgets/dropdown.rs", "phase 4: pill and text caches are Vec<u32>"),
    ("widgets/multitextbox.rs", "phase 4: pill and text caches are Vec<u32>"),
    ("widgets/textbox.rs", "phase 4: pill and text caches are Vec<u32>, and its premultiplied cache blit"),
    ("golden.rs", "test scenes author 8-bit colours"),
    ("gate.rs", "this file names the patterns it refuses"),
];

/// The layout as written by hand, in a line already lowercased and stripped of `_`.
fn spells_the_layout(code: &str) -> bool {
    if ["0xff000000", "0x00ffffff", "0x00ff00ff"].iter().any(|m| code.contains(m)) {
        return true;
    }
    // A shift by exactly 24: `>> 24` or `<< 24`, with or without the space, not followed by another digit.
    for op in [">>", "<<"] {
        let mut rest = code;
        while let Some(at) = rest.find(op) {
            let after = rest[at + 2..].trim_start();
            if after.starts_with("24") && !after[2..].starts_with(|c: char| c.is_ascii_digit()) {
                return true;
            }
            rest = &rest[at + 2..];
        }
    }
    false
}

fn rust_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("src/ is readable at test time") {
        let path = entry.expect("a directory entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn the_8_bit_layout_stays_in_its_files() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_files(&src, &mut files);
    files.sort();
    let mut refused: Vec<String> = Vec::new();
    let mut listed_hit = alloc::vec![false; EIGHT_BIT.len()];
    for path in &files {
        let rel = path.strip_prefix(&src).unwrap().to_string_lossy().replace('\\', "/");
        let listed = EIGHT_BIT.iter().position(|(p, _)| if p.ends_with('/') { rel.starts_with(p) } else { rel == *p });
        let text = std::fs::read_to_string(path).unwrap();
        for (n, line) in text.lines().enumerate() {
            if line.starts_with("#[cfg(test)]") {
                break;
            }
            let code = line.split("//").next().unwrap().to_lowercase().replace('_', "");
            if !spells_the_layout(&code) {
                continue;
            }
            match listed {
                Some(i) => listed_hit[i] = true,
                None => refused.push(alloc::format!("src/{rel}:{}: {}", n + 1, line.trim())),
            }
        }
    }
    for (i, (p, _)) in EIGHT_BIT.iter().enumerate() {
        if !p.ends_with('/') && *p != "gate.rs" && !listed_hit[i] {
            refused.push(alloc::format!("src/{p} no longer spells the 8-bit layout: take it off EIGHT_BIT"));
        }
    }
    assert!(refused.is_empty(), "the 8-bit pixel layout written by hand outside its files (go thru crate::pixel::Pixel):\n{}", refused.join("\n"));
}

#[test]
fn the_gate_sees_what_it_refuses() {
    assert!(spells_the_layout("let a = p >> 24;"));
    assert!(spells_the_layout("(na<<24) | x"));
    assert!(spells_the_layout(&"p & 0x00FF_FFFF".to_lowercase().replace('_', "")));
    assert!(spells_the_layout(&"0xFF00_0000 | x".to_lowercase().replace('_', "")));
    assert!(spells_the_layout(&"(c & 0x00_FF_00_FF)".to_lowercase().replace('_', "")));
    assert!(!spells_the_layout("x >> 240"));
    assert!(!spells_the_layout("x << 2"));
    assert!(!spells_the_layout("p.with_alpha(a)"));
}
