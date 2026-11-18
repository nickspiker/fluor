//! The hit map goes front to back with the pixels: the frontmost widget claims its pixels first, a pixel already claimed is never written again, and so whatever is frontmost on screen takes the click.
//! Each test draws an overlap in front-to-back order and checks every overlapped pixel: the click goes to the frontmost, and the pixel there is the frontmost's own.
//! Until 2026-11-17 every hit write overwrote what was there, so the widget drawn LAST (the one behind) took the clicks of everything it overlapped.

use crate::canvas::{Canvas, Damage};
use crate::paint::{self, HitId, HIT_BLOCK, HIT_NONE};
use crate::pixel::{Argb8, Blend, BlendMode, Pixel};
use crate::text::TextRenderer;
use crate::widgets::{Button, Dropdown, Textbox};
use alloc::vec;
use alloc::vec::Vec;

const W: usize = 320;
const H: usize = 240;

/// A fresh frame: empty pixels, an empty hit map.
fn frame() -> (Vec<Argb8>, Vec<HitId>, Damage) {
    (vec![0; W * H], vec![HIT_NONE; W * H], Damage::new())
}

/// An open dropdown's popup drawn first, then a button drawn after it whose pill lies under the popup: at every pixel of the popup the click goes to a row or to the popup's frame, never to the button, and the button keeps the rest of its pill.
#[test]
fn an_open_dropdown_keeps_its_clicks_from_a_widget_drawn_after_it() {
    let mut text = TextRenderer::new();
    let mut ids: HitId = HIT_NONE;
    let options = vec!["Zil".into(), "Zila".into(), "Zilor".into(), "Ter".into()];
    let mut d = Dropdown::new(&mut ids, 160., 30., 160., 36., 18., options);
    d.open();
    let popup = d.popup_region(H);
    // A wide button across the popup's middle, reaching past both its sides.
    let mut b = Button::new(&mut ids, 160., popup.y + popup.h * 0.5, 280., 60., 20., "Behind");
    let (mut px, mut hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        d.render_popup_into(&mut c, &mut text, None, Some(&mut hit));
        d.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut hit));
        let id = b.hit_id();
        b.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut hit), id);
    }
    let mut rows: Vec<HitId> = Vec::new();
    d.visit_rows(&mut |w| rows.push(w.id()));
    let (x0, y0, x1, y1) = (popup.x as usize, popup.y as usize, (popup.x + popup.w) as usize, (popup.y + popup.h) as usize);
    let mut overlapped = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            let h = hit[y * W + x];
            assert!(h == HIT_BLOCK || rows.contains(&h), "({x}, {y}) in the popup went to {h}, not the popup (rows {rows:?})");
            overlapped += 1;
        }
    }
    assert!(overlapped > 0);
    // Each row takes the middle of its own band.
    for (i, &row) in rows.iter().enumerate() {
        let band = popup.h / rows.len() as f32;
        let (x, y) = ((popup.x + popup.w * 0.5) as usize, (popup.y + band * (i as f32 + 0.5)) as usize);
        assert_eq!(hit[y * W + x], row, "row {i} at ({x}, {y})");
    }
    // The button still takes its pill outside the popup.
    let (bx, by) = ((b.center_x - b.width * 0.5 + 8.) as usize, b.center_y as usize);
    assert!(bx < x0, "the button reaches past the popup's left side");
    assert_eq!(hit[by * W + bx], b.hit_id());
    // The popup's pixels are its own where they are opaque: the button drawn after went under them.
    let (mut alone, _, mut dmg2) = frame();
    {
        let mut c = Canvas::new(&mut alone, W, H, &mut dmg2);
        d.render_popup_into(&mut c, &mut text, None, None);
    }
    for y in y0..y1 {
        for x in x0..x1 {
            if alone[y * W + x].alpha() == Argb8::ONE {
                assert_eq!(px[y * W + x], alone[y * W + x], "({x}, {y}): the button came out in front of the popup");
            }
        }
    }
}

/// Two buttons that overlap, the front one drawn first: at every pixel the front one claimed the back one never takes the click, and where the front pill is opaque the pixel is the front pill's.
/// `stamp_hit_into` at the front button's place in the order claims its anti-aliased rim too; at the back button's place, after the front, it takes nothing of the front's.
#[test]
fn overlapping_buttons_give_the_click_to_the_front_one() {
    let mut text = TextRenderer::new();
    let mut ids: HitId = HIT_NONE;
    let mut front = Button::new(&mut ids, 130., 100., 140., 48., 20., "Front");
    let mut back = Button::new(&mut ids, 190., 120., 140., 48., 20., "Back");
    let (fid, bid) = (front.hit_id(), back.hit_id());
    // Each alone: where each claims, and the front's pixels.
    let (mut front_px, mut front_hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut front_px, W, H, &mut dmg);
        front.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut front_hit), fid);
    }
    let (mut back_px, mut back_hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut back_px, W, H, &mut dmg);
        back.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut back_hit), bid);
    }
    let (mut px, mut hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        front.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut hit), fid);
        front.stamp_hit_into(&mut hit, W, H, fid);
        back.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut hit), bid);
        back.stamp_hit_into(&mut hit, W, H, bid);
    }
    let mut overlapped = 0;
    for i in 0..W * H {
        let in_front = front_hit[i] == fid;
        if in_front && back_hit[i] == bid {
            overlapped += 1;
        }
        if in_front {
            assert_eq!(hit[i], fid, "({}, {}): the back button took a pixel the front one claimed", i % W, i / W);
            assert_eq!(px[i], front_px[i], "({}, {}): the back button's pixels came out in front", i % W, i / W);
        } else if back_hit[i] == bid && front_px[i] == 0 {
            assert_eq!(hit[i], bid, "({}, {}): the back button lost a pixel nothing is in front of", i % W, i / W);
        }
    }
    assert!(overlapped > 64, "the buttons barely overlap ({overlapped} pixels)");
    // The front pill's anti-aliased rim, wherever it touches the back pill, is the front's.
    let rim = front_px.iter().zip(&back_hit).filter(|&(&p, &b)| p.alpha() != 0 && p.alpha() != Argb8::ONE && b == bid).count();
    assert!(rim > 0, "the rims do not cross");
    for i in 0..W * H {
        if front_px[i].alpha() != 0 && back_hit[i] == bid {
            assert_eq!(hit[i], fid, "({}, {}): the front rim went to the back button", i % W, i / W);
        }
    }
}

/// The caret goes in front of the text it sits in and behind whatever is in front of its textbox: under a front widget's opaque pixels it is hidden and the click is the front widget's; elsewhere every caret pixel is the textbox's to click, its light in front of the field.
#[test]
fn the_caret_is_in_front_of_its_text_and_behind_what_is_in_front_of_the_box() {
    let mut text = TextRenderer::new();
    let mut ids: HitId = HIT_NONE;
    let mut t = Textbox::new(&mut ids, 160., 120., 240., 48., 22.);
    t.set_text("caret golden", &mut text);
    t.set_focused(true);
    t.cursor_end();
    t.blinkey_visible = true;
    let tid = t.hit_id();
    let front_id = paint::HIT_BLOCK - 1;
    let render = |t: &mut Textbox, text: &mut TextRenderer, front: bool| {
        let (mut px, mut hit, mut dmg) = frame();
        {
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            if front {
                // A front widget over the whole right half of the box, the caret's end included: an opaque rect, claimed first.
                paint::fill_rect(&mut c, 160, 90, 120, 60, crate::theme::BUTTON_FILL, None, None);
                for y in 90..150 {
                    paint::claim_hits(&mut hit[y * W + 160..y * W + 280], front_id);
                }
            }
            t.render_content_into(&mut c, 0., 0., text, None, None, Some(&mut hit), tid);
        }
        (px, hit)
    };
    let (with_caret, hit) = render(&mut t, &mut text, false);
    // The caret alone, a layer of its own in an empty buffer.
    let (mut caret, _, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut caret, W, H, &mut dmg);
        t.render_blinkey_into(&mut c, 0., 0.);
    }
    // The same box with its caret hidden: the field alone.
    t.blinkey_visible = false;
    let (field, _) = render(&mut t, &mut text, false);
    t.blinkey_visible = true;
    let mut lit = 0;
    for i in 0..W * H {
        if caret[i] == 0 {
            assert_eq!(with_caret[i], field[i], "({}, {}): the caret touched a pixel it does not reach", i % W, i / W);
            continue;
        }
        lit += 1;
        assert_eq!(hit[i], tid, "({}, {}): a caret pixel is not the textbox's to click", i % W, i / W);
        // The caret in front of the field, to two codes: the frame rounds at each of its three composites (caret, then the text under it, then the pill), the expectation at its one (the caret over the finished field), and `under` is associative only up to that rounding.
        // Measured 2026-11-17 on this scene: of 232 caret pixels 212 exact, 19 one code off, 1 two codes off.
        let want = caret[i].under_premult(field[i], BlendMode::Normal);
        let (got, want) = (with_caret[i], want);
        assert!(got.alpha().abs_diff(want.alpha()) <= 2 && (0..3).all(|c| got.channels()[c].abs_diff(want.channels()[c]) <= 2), "({}, {}): {got:08x} is not the caret in front of the field {want:08x}", i % W, i / W);
    }
    assert!(lit > 0, "the caret drew nothing");
    assert_ne!(with_caret, field, "the caret changed nothing");
    // With a front widget over the caret: there the front's pixel and the front's click, the caret hidden behind it.
    let (covered, covered_hit) = render(&mut t, &mut text, true);
    let (mut front_alone, _, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut front_alone, W, H, &mut dmg);
        paint::fill_rect(&mut c, 160, 90, 120, 60, crate::theme::BUTTON_FILL, None, None);
    }
    let mut hidden = 0;
    for y in 90..150 {
        for x in 160..280 {
            let i = y * W + x;
            assert_eq!(covered_hit[i], front_id, "({x}, {y}) went behind the front widget");
            assert_eq!(covered[i], front_alone[i], "({x}, {y}): the caret or the box came out in front of the front widget");
            if caret[i] != 0 {
                hidden += 1;
            }
        }
    }
    assert!(hidden > 0, "the front widget does not cover the caret");
}
