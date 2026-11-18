//! The hit map goes front to back with the pixels: at each pixel the frontmost layer that SHOWS there (its own coverage above zero, translucent or not) claims it, a pixel already claimed is never written again, and where the accumulator is already opaque a layer behind neither composes nor claims.
//! A layer with no id claims nothing, so whatever shows thru it takes the click; a widget's fully transparent pixels claim nothing, and its anti-aliased rim and its own glow claim for it.
//! The owner (2026-11-17): "if it's showing on screen, it NEEDS to be hitting the hitmap, if it's already opaque and won't show? SKIP!", and of a translucent front: "the frontmost is locking it up".
//! Until 2026-11-17 every hit write overwrote what was there, so the widget drawn LAST (the one behind) took the clicks of everything it overlapped.

use crate::canvas::{Canvas, Damage, PixelRect};
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

/// A button rendered alone into a fresh frame: its pixels, so a test knows where it shows.
fn alone(b: &mut Button, text: &mut TextRenderer) -> Vec<Argb8> {
    let (mut px, mut hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        let id = b.hit_id();
        b.render_content_into(&mut c, 0., 0., text, None, Some(&mut hit), id);
    }
    px
}

/// A rect over the whole frame's middle, drawn first, at `colour`, claiming `id` wherever it shows (HIT_NONE: a decoration, claiming nothing); then a button behind it.
/// Returns the frame's pixels and hits, and the button's id.
fn layer_over_button(colour: u32, id: HitId, text: &mut TextRenderer) -> (Vec<Argb8>, Vec<HitId>, HitId) {
    let mut ids: HitId = HIT_NONE;
    let mut b = Button::new(&mut ids, 160., 120., 200., 60., 20., "Behind");
    let (mut px, mut hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        let mut shown = paint::Shown::default();
        shown.before(c.pixels, W, H, PixelRect::new(100, 80, 220, 160));
        paint::fill_rect(&mut c, 100, 80, 120, 80, colour, None, None);
        shown.settle(c.pixels);
        if id != HIT_NONE {
            shown.claim(&mut hit, id);
        }
        let bid = b.hit_id();
        b.render_content_into(&mut c, 0., 0., text, None, Some(&mut hit), bid);
    }
    (px, hit, b.hit_id())
}

/// A translucent wash over a button: with an id (or HIT_BLOCK) the wash is the frontmost that shows and takes the click; with no id it is a decoration and the button behind it takes it.
#[test]
fn a_translucent_wash_takes_the_click_only_with_an_id() {
    let mut text = TextRenderer::new();
    // A dim blue wash at 3/8 opacity.
    let wash = paint::pack_argb(40, 60, 120, 0x60);
    let mut b = Button::new(&mut 0, 160., 120., 200., 60., 20., "Behind");
    let behind = alone(&mut b, &mut text);
    let inside = |i: usize| (100..220).contains(&(i % W)) && (80..160).contains(&(i / W));
    for (id, name) in [(HIT_BLOCK - 1, "an id"), (HIT_BLOCK, "HIT_BLOCK"), (HIT_NONE, "no id")] {
        let (_, hit, bid) = layer_over_button(wash, id, &mut text);
        let mut overlapped = 0;
        for i in (0..W * H).filter(|&i| inside(i) && behind[i].alpha() != 0) {
            let want = if id == HIT_NONE { bid } else { id };
            assert_eq!(hit[i], want, "wash with {name}: ({}, {})", i % W, i / W);
            overlapped += 1;
        }
        assert!(overlapped > 0, "the wash does not cover the button");
        // Inside the wash where the button does not show, the wash claims only with an id.
        let i = (82 * W) + 102;
        assert_eq!(behind[i].alpha(), 0);
        assert_eq!(hit[i], id, "wash with {name}: its own pixel off the button");
    }
}

/// An opaque decorative block (no id) over a button: the button does not show behind it, so it claims nothing there and the click is blocked, while the rest of the button keeps its clicks.
#[test]
fn an_opaque_block_blocks_the_click() {
    let mut text = TextRenderer::new();
    let mut b = Button::new(&mut 0, 160., 120., 200., 60., 20., "Behind");
    let behind = alone(&mut b, &mut text);
    let (_, hit, bid) = layer_over_button(crate::theme::BUTTON_FILL, HIT_NONE, &mut text);
    let mut blocked = 0;
    let mut kept = 0;
    for i in (0..W * H).filter(|&i| behind[i].alpha() != 0) {
        if (100..220).contains(&(i % W)) && (80..160).contains(&(i / W)) {
            assert_eq!(hit[i], HIT_NONE, "({}, {}) went thru the opaque block", i % W, i / W);
            blocked += 1;
        } else {
            assert_eq!(hit[i], bid, "({}, {}): the button lost a pixel nothing covers", i % W, i / W);
            kept += 1;
        }
    }
    assert!(blocked > 0 && kept > 0, "{blocked} blocked, {kept} kept");
}

/// A widget's transparent corner (outside its squircle, inside its box) claims nothing: a click there falls thru to what is behind.
/// Both pill paths: a retained Button, and the immediate pill, which until 2026-11-17 claimed its whole box.
#[test]
fn a_transparent_corner_falls_through() {
    let mut text = TextRenderer::new();
    let back_id = HIT_BLOCK - 1;
    let region = crate::region::Region::new(60., 90., 200., 60.);
    for immediate in [false, true] {
        let mut ids: HitId = HIT_NONE;
        let mut front = Button::new(&mut ids, 160., 120., 200., 60., 20., "Front");
        let fid = front.hit_id();
        let paint_front = |front: &mut Button, c: &mut Canvas, text: &mut TextRenderer, hit: &mut [HitId]| {
            if immediate {
                Button::draw_pill_immediate(c, text, Some(hit), W, H, region, "Front", "Open Sans", fid, false, false, true, None);
            } else {
                front.render_content_into(c, 0., 0., text, None, Some(hit), fid);
            }
        };
        let (mut front_px, mut front_hit, mut dmg) = frame();
        {
            let mut c = Canvas::new(&mut front_px, W, H, &mut dmg);
            paint_front(&mut front, &mut c, &mut text, &mut front_hit);
        }
        let (mut px, mut hit, mut dmg) = frame();
        {
            let mut c = Canvas::new(&mut px, W, H, &mut dmg);
            paint_front(&mut front, &mut c, &mut text, &mut hit);
            // An opaque back layer with an id over the whole button box and beyond.
            let mut shown = paint::Shown::default();
            shown.before(c.pixels, W, H, PixelRect::new(40, 70, 280, 170));
            paint::fill_rect(&mut c, 40, 70, 240, 100, crate::theme::TEXTBOX_FILL, None, None);
            shown.settle(c.pixels);
            shown.claim(&mut hit, back_id);
        }
        let mut corners = 0;
        for y in 90..150 {
            for x in 60..260 {
                let i = y * W + x;
                if front_px[i].alpha() == 0 {
                    assert_eq!(hit[i], back_id, "immediate {immediate}: ({x}, {y}): the button's transparent pixel kept the click");
                    corners += 1;
                } else {
                    assert_eq!(hit[i], fid, "immediate {immediate}: ({x}, {y}): a pixel the button shows on went behind it");
                }
            }
        }
        assert!(corners > 0, "immediate {immediate}: the button's box has no transparent corner");
    }
}

/// An open dropdown's popup drawn first, then a button drawn after it whose pill lies under the popup: wherever the popup shows the click goes to a row or to the popup's frame (HIT_BLOCK, translucent edge included), never to the button; where the popup does not show (its rounded corners) the button behind it takes the click.
#[test]
fn an_open_dropdown_keeps_its_clicks_from_a_widget_drawn_after_it() {
    let mut text = TextRenderer::new();
    let mut ids: HitId = HIT_NONE;
    let options = vec!["Zil".into(), "Zila".into(), "Zilor".into(), "Ter".into()];
    let mut d = Dropdown::new(&mut ids, 160., 30., 160., 36., 18., options);
    d.open();
    let popup = d.popup_region(H);
    // A wide button across the popup's top corners, reaching past both its sides.
    let mut b = Button::new(&mut ids, 160., popup.y + 10., 280., 60., 20., "Behind");
    let behind = alone(&mut b, &mut text);
    let (mut px, mut hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        d.render_popup_into(&mut c, &mut text, None, Some(&mut hit));
        d.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut hit));
        let id = b.hit_id();
        b.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut hit), id);
    }
    let (mut popup_px, mut popup_hit, mut dmg2) = frame();
    {
        let mut c = Canvas::new(&mut popup_px, W, H, &mut dmg2);
        d.render_popup_into(&mut c, &mut text, None, Some(&mut popup_hit));
    }
    let mut rows: Vec<HitId> = Vec::new();
    d.visit_rows(&mut |w| rows.push(w.id()));
    let (x0, y0, x1, y1) = (popup.x as usize, popup.y as usize, (popup.x + popup.w) as usize, (popup.y + popup.h) as usize);
    let (mut shows, mut translucent, mut through) = (0, 0, 0);
    for y in y0..y1 {
        for x in x0..x1 {
            let i = y * W + x;
            let a = popup_px[i].alpha();
            if a != 0 {
                assert!(hit[i] == HIT_BLOCK || rows.contains(&hit[i]), "({x}, {y}) shows the popup but went to {}, not the popup (rows {rows:?})", hit[i]);
                shows += 1;
                translucent += (a != Argb8::ONE) as usize;
                if a == Argb8::ONE {
                    assert_eq!(px[i], popup_px[i], "({x}, {y}): the button came out in front of the popup");
                }
            } else if behind[i].alpha() != 0 {
                assert_eq!(hit[i], b.hit_id(), "({x}, {y}): the popup does not show here, so the button behind takes it");
                through += 1;
            }
        }
    }
    assert!(shows > 0 && translucent > 0 && through > 0, "{shows} popup pixels, {translucent} translucent, {through} thru its corners");
    // Each row takes the middle of its own band.
    for (i, &row) in rows.iter().enumerate() {
        let band = popup.h / rows.len() as f32;
        let (x, y) = ((popup.x + popup.w * 0.5) as usize, (popup.y + band * (i as f32 + 0.5)) as usize);
        assert_eq!(hit[y * W + x], row, "row {i} at ({x}, {y})");
    }
}

/// Two overlapping buttons, the front one drawn first: at every pixel the front one shows on (its fill, its anti-aliased rim, translucent or not) the front takes the click and the back never does; the back takes the pixels it shows on that the front does not; where the front is opaque the pixel is the front's own.
/// `stamp_hit_into` at each button's place in the order changes nothing: it claims from the same mask its paint settled.
#[test]
fn overlapping_buttons_give_the_click_to_the_front_one() {
    let mut text = TextRenderer::new();
    let mut ids: HitId = HIT_NONE;
    let mut front = Button::new(&mut ids, 130., 100., 140., 48., 20., "Front");
    let mut back = Button::new(&mut ids, 190., 120., 140., 48., 20., "Back");
    let (fid, bid) = (front.hit_id(), back.hit_id());
    let front_px = alone(&mut front, &mut text);
    let back_px = alone(&mut back, &mut text);
    let (mut px, mut hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        front.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut hit), fid);
        front.stamp_hit_into(&mut hit, W, H, fid);
        back.render_content_into(&mut c, 0., 0., &mut text, None, Some(&mut hit), bid);
        back.stamp_hit_into(&mut hit, W, H, bid);
    }
    let (mut overlapped, mut rim) = (0, 0);
    for i in 0..W * H {
        let (fa, ba) = (front_px[i].alpha(), back_px[i].alpha());
        let want = if fa != 0 { fid } else if ba != 0 { bid } else { HIT_NONE };
        assert_eq!(hit[i], want, "({}, {})", i % W, i / W);
        if fa != 0 && ba != 0 {
            overlapped += 1;
            rim += (fa != Argb8::ONE) as usize;
        }
        if fa == Argb8::ONE {
            assert_eq!(px[i], front_px[i], "({}, {}): the back button's pixels came out in front", i % W, i / W);
        }
    }
    assert!(overlapped > 64 && rim > 0, "the buttons barely overlap ({overlapped} pixels, {rim} on the front's rim)");
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

/// A checkbox's tap shape is its box and its label's box: a tap between the label's letters, or inside an "o", is the checkbox's, though nothing is drawn there; past the label's end nothing claims.
#[test]
fn a_tap_between_label_letters_or_inside_an_o_hits_the_checkbox() {
    let mut text = TextRenderer::new();
    let mut ids: HitId = HIT_NONE;
    let mut cb = crate::widgets::Checkbox::new(&mut ids, "Good dog", 160., 120., 280., 40., 24., false);
    let cid = cb.hit_id();
    let (mut px, mut hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        cb.render_content_into(&mut c, &mut text, None, Some(&mut hit));
    }
    // The label's box as the checkbox lays it out: right of the box, one line step tall round the centre.
    let lx = cb.center_x - cb.width * 0.5 + cb.box_side() + cb.font_size * 0.5;
    let lw = text.measure_text("Good dog", &crate::text::TextStyle::new(cb.font_size, crate::theme::TEXTBOX_TEXT));
    let half = cb.font_size * 1.25 * 0.5;
    let (x0, x1, y0, y1) = (lx.ceil() as usize, (lx + lw).floor() as usize, (cb.center_y - half).ceil() as usize, (cb.center_y + half).floor() as usize);
    let clear = |x: usize, y: usize| px[y * W + x].alpha() == 0;
    let (mut gaps, mut counters) = (0, 0);
    for y in y0..y1 {
        for x in x0..x1 {
            if !clear(x, y) {
                continue;
            }
            assert_eq!(hit[y * W + x], cid, "({x}, {y}) in the label's box went elsewhere");
            // Between letters: the whole column of the label's box is clear.
            gaps += (y0..y1).all(|yy| clear(x, yy)) as usize;
            // Inside a counter (an "o"): ink on both sides of the row and above and below in the column, within the label's box.
            let ink = |mut it: core::ops::Range<usize>, f: &dyn Fn(usize) -> bool| it.any(|v| !f(v));
            let (row, col) = (|v: usize| clear(v, y), |v: usize| clear(x, v));
            counters += (ink(x0..x, &row) && ink(x + 1..x1, &row) && ink(y0..y, &col) && ink(y + 1..y1, &col)) as usize;
        }
    }
    assert!(gaps > 0, "no tap landed between letters");
    assert!(counters > 0, "no tap landed inside a letter's counter");
    // Past the label's end, on its centre row: nothing drawn and outside the shape, so nothing claims.
    let y = cb.center_y as usize;
    let past = (lx + lw).ceil() as usize + 4;
    assert!(clear(past, y));
    assert_eq!(hit[y * W + past], HIT_NONE, "past the label's end");
}

/// A slider's tap shape is its track widened to a band span / 16 tall: a tap beside the track (above or below it, away from the handle) within the band hits the slider; outside the band, nothing.
#[test]
fn a_tap_beside_the_slider_track_within_the_band_hits_the_slider() {
    let mut ids: HitId = HIT_NONE;
    let mut s = crate::widgets::Slider::new(&mut ids, 160., 120., 200., 20., 0.5);
    let sid = s.hit_id();
    let (mut px, mut hit, mut dmg) = frame();
    {
        let mut c = Canvas::new(&mut px, W, H, &mut dmg);
        s.render_content_into(&mut c, Some(&mut hit), sid);
    }
    // The band the slider declares: span / 16 round the track's centre (a 20-tall layout band, its track a quarter of it from row 117: centre 119.5).
    let span = 2. * W as f32 * H as f32 / (W + H) as f32;
    let (cy, half) = (119.5f32, span / 32.);
    let (mut beside, mut outside) = (0, 0);
    // Columns well clear of the handle (radius 10 round x = 160).
    for x in (64..140).chain(180..256) {
        for y in 96..144 {
            let i = y * W + x;
            let in_band = ((y as f32 + 0.5) - cy).abs() < half;
            if in_band {
                assert_eq!(hit[i], sid, "({x}, {y}) inside the slider's band");
                beside += (px[i].alpha() == 0) as usize;
            } else {
                assert_eq!(px[i].alpha(), 0, "({x}, {y}): the slider drew outside its band");
                assert_eq!(hit[i], HIT_NONE, "({x}, {y}) outside the slider's band");
                outside += 1;
            }
        }
    }
    assert!(beside > 0 && outside > 0, "{beside} taps beside the track, {outside} outside the band");
}
