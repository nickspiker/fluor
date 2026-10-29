//! A window at either pixel depth, to see the 16-bit present's dither against 8-bit banding: dark ramps across the window, the chrome around them.
//! `cargo run --example depth --features host-winit` runs at 16 bits (G2t16); `-- 8` runs the same app at 8 (Argb8).
//! The ramps are written at the depth's own precision (darkness interpolated in the canvas's fields, not authored 8-bit colours), so at 8 bits they band, one band per `g2t` code, and at 16 bits they reach the screen as an ordered dither finer than one 8-bit step.

use fluor::canvas::Canvas;
use fluor::coord::Coord;
use fluor::event::{CursorIcon, ElementState, Event, MouseButton, ModifiersState};
use fluor::host::app::{run_app, Context, EventResponse, FluorApp};
use fluor::host::chrome::{get_resize_edge, ResizeEdge};
use fluor::host::chrome_widget::DefaultChrome;
use fluor::host::widget::{build_overlay_deltas, dispatch_release};
use fluor::paint::{Finalize, HitId, HIT_NONE};
use fluor::pixel::{Argb8, G2t16, Pixel};

struct Demo<P: Pixel> {
    chrome: DefaultChrome<P>,
    hits: HitId,
}

impl<P: Pixel> Demo<P> {
    fn new() -> Self {
        let mut hits = HIT_NONE;
        let chrome = DefaultChrome::with_depth(fluor::geom::Viewport::new(1, 1), format!("fluor at {} bits", P::DEPTH), None, None, &mut hits);
        Demo { chrome, hits }
    }
}

/// Four horizontal ramps, each a band of the window: darkness falls from `top` to `top − span` codes (of 256) left to right, interpolated in the canvas's own fields.
/// Codes near black and a blue one: the dark end of `g2t` is where an 8-bit step is widest on an SDR screen.
fn ramps<P: Pixel>(c: &mut Canvas<P>) {
    let (w, h) = (c.width, c.height);
    let bands: [([u32; 3], u32); 4] = [([255, 255, 255], 48), ([224, 224, 224], 32), ([255, 240, 200], 64), ([200, 200, 200], 16)];
    for y in 0..h {
        let (top, span) = bands[y * bands.len() / h];
        for x in 0..w {
            // Darkness in this depth's scale: ONE × (top − span × x / w) / 255, one multiply and divide in u64 so 16 bits keep every step.
            let d = top.map(|t| ((P::ONE as u64 * (t as u64 * w as u64 - span as u64 * x as u64)) / (255 * w as u64)) as u32);
            c.pixels[y * w + x] = P::from_channels(P::ONE, d);
        }
    }
    c.damage.add_bounds(0, 0, w, h);
}

impl<P: Finalize> FluorApp<P> for Demo<P> {
    type UserEvent = ();

    fn title(&self) -> &str {
        &self.chrome.title
    }

    fn init(&mut self, ctx: &mut Context) {
        self.chrome.resize(ctx.viewport);
    }

    fn on_resize(&mut self, _width: u32, _height: u32, ctx: &mut Context) {
        self.chrome.resize(ctx.viewport);
    }

    fn on_event(&mut self, event: &Event, ctx: &mut Context) -> EventResponse {
        match event {
            Event::CursorMoved { x, y } => {
                if self.chrome.set_hover(self.chrome.hit_at(*x, *y)) {
                    ctx.window.request_redraw();
                }
                EventResponse::Handled
            }
            Event::MouseInput { state: ElementState::Pressed, button: MouseButton::Left } => {
                let edge = get_resize_edge(ctx.viewport, ctx.cursor_x, ctx.cursor_y);
                if edge != ResizeEdge::None {
                    EventResponse::StartResize(edge)
                } else if self.chrome.hit_at(ctx.cursor_x, ctx.cursor_y) == HIT_NONE {
                    EventResponse::StartWindowDrag
                } else {
                    EventResponse::Handled
                }
            }
            _ => EventResponse::Pass,
        }
    }

    fn on_activate(&mut self, hit_id: HitId, x: Coord, y: Coord, mods: ModifiersState, _ctx: &mut Context) -> EventResponse {
        dispatch_release(&mut self.chrome, hit_id, x, y, mods)
    }

    fn render(&mut self, target: &mut [P], ctx: &mut Context) {
        let (w, h) = (ctx.viewport.width_px as usize, ctx.viewport.height_px as usize);
        self.chrome.rasterize_perimeter(target, w, h, ctx.clip_mask);
        self.chrome.rasterize_chrome(ctx.damage, ctx.text, ctx.clip_mask);
        self.chrome.rasterize_bg(ctx.damage, |c| ramps(c));
        self.chrome.flatten_into(target, w, h, None);
    }

    fn cursor_for(&self, _x: Coord, _y: Coord, _ctx: &Context) -> CursorIcon {
        CursorIcon::Default
    }

    fn hit_test_map(&self) -> Option<(&[HitId], usize, usize)> {
        let (w, h) = self.chrome.dims();
        Some((self.chrome.hit_test_map(), w, h))
    }

    fn overlay_deltas(&mut self) -> Vec<u32> {
        build_overlay_deltas(&mut self.chrome, self.hits as usize + 1)
    }
}

fn main() {
    let eight = std::env::args().nth(1).is_some_and(|a| a == "8");
    let res = if eight { run_app(Demo::<Argb8>::new()) } else { run_app(Demo::<G2t16>::new()) };
    res.expect("event loop failed");
}
