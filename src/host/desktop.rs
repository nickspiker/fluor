//! Desktop host: winit window + softbuffer CPU framebuffer. Available under feature `host-winit` (on by default).
//!
//! Borderless window. Chrome is rendered by photon's `draw_window_controls` (verbatim, see [`super::chrome`]). Click routing uses photon's `hit_test_map` (per-pixel button ID) + `get_resize_edge`. Window setup matches photon's main.rs: `with_decorations(false)`, `with_transparent(true)`, `with_resizable(...)`, monitor-relative initial size, macOS drop shadow off.

use super::chrome::{self, ResizeEdge, HIT_CLOSE_BUTTON, HIT_MAXIMIZE_BUTTON, HIT_MINIMIZE_BUTTON, HIT_NONE};
use crate::paint;
use crate::paint::{snap_rotation, Transform};
use crate::text::TextRenderer;
use crate::theme;
use crate::Compositor;
use std::num::NonZeroU32;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{CursorIcon, ResizeDirection, Window, WindowAttributes, WindowId};

/// Run the desktop host until the window closes.
pub fn run(compositor: Compositor, title: &str) -> Result<(), EventLoopError> {
    let event_loop = EventLoop::new()?;
    let mut app = DesktopApp::new(compositor, title.to_string());
    event_loop.run_app(&mut app)
}

struct DesktopApp {
    compositor: Compositor,
    title: String,
    window: Option<Arc<Window>>,
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    /// Per-pixel button-id map written by `draw_window_controls` and read on click.
    hit_test_map: Vec<u8>,
    cursor_x: f32,
    cursor_y: f32,
    /// Currently hovered chrome button id (HIT_NONE if none). Drives the hover overlay.
    hover_state: u8,
    /// Cached pixel list for the currently hovered button — recomputed on hover-state change.
    hover_pixel_list: Vec<usize>,
    /// Font system + glyph cache, lazily initialized on first `resumed`.
    text: Option<TextRenderer>,
}

impl DesktopApp {
    fn new(compositor: Compositor, title: String) -> Self {
        Self {
            compositor,
            title,
            window: None,
            surface: None,
            hit_test_map: Vec::new(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            hover_state: chrome::HIT_NONE,
            hover_pixel_list: Vec::new(),
            text: None,
        }
    }

    fn render_frame(&mut self) {
        let Some(surface) = self.surface.as_mut() else { return; };
        let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
        let vp = self.compositor.viewport();
        let buf_w = vp.width_px as usize;
        let buf_h = vp.height_px as usize;
        let needed = buf_w * buf_h;
        if self.hit_test_map.len() != needed {
            self.hit_test_map.resize(needed, HIT_NONE);
        } else {
            self.hit_test_map.fill(HIT_NONE);
        }

        // 1. Background noise — full buffer fill
        paint::background_noise(&mut buffer, buf_w, buf_h, 0, true, 0, None);
        // 2. Panes on top of background
        self.compositor.render(&mut buffer, buf_w, buf_h);
        // 3. Chrome controls strip (writes hit_test_map for click routing).
        let (start, crossings, button_x_start, button_height) = chrome::draw_window_controls(
            &mut buffer,
            &mut self.hit_test_map,
            vp.width_px,
            vp.height_px,
            1.0,
        );
        // 4. Two-tone window edges + squircle corner mask (verbatim photon)
        chrome::draw_window_edges_and_mask(
            &mut buffer,
            &mut self.hit_test_map,
            vp.width_px,
            vp.height_px,
            start,
            &crossings,
        );
        // 5. Vertical hairlines between control buttons (verbatim photon)
        chrome::draw_button_hairlines(
            &mut buffer,
            &mut self.hit_test_map,
            vp.width_px,
            vp.height_px,
            button_x_start,
            button_height,
            start,
            &crossings,
        );
        // 6. Title text in the top-left of the chrome strip — left-aligned, vertically centered in the button row, sized relative to button height. `bw` matches chrome's button_height formula (MIN + scaled) so text aligns with the buttons.
        // Title text is skipped below a readable size — controls (which always render) are the load-bearing UI; text is visual decoration that shouldn't smear into illegibility.
        let span = 2.0 * vp.width_px as f32 * vp.height_px as f32 / (vp.width_px as f32 + vp.height_px as f32);
        let bw = chrome::MIN_BUTTON_HEIGHT_PX as f32 + (span / 32.0).ceil();
        let title_size = bw * 0.55;
        if title_size >= 6.0 {
        if let Some(text) = self.text.as_mut() {
            let pad = bw * 0.5;
            let baseline_y = bw * 0.5;
            let _ = text.draw_text_left_u32(
                &mut buffer,
                buf_w,
                buf_h,
                &self.title,
                pad,
                baseline_y,
                title_size,
                400,
                theme::TEXT_COLOUR,
                "Open Sans",
                None,
                None,
                None,
            );

            // Aspect-ratio-driven rotation demo. Rotation = (aspect - 1) × π, so a square window (1:1) is upright, a 2:1 landscape window is upside-down, a 1:2 portrait window tilts the other way. Snapped to the per-font-size quantization grid (1-pixel-arc-at-radius rule, K=8) so consecutive frames at the same aspect hit the rasterized-glyph cache in `text::TextRenderer`. Pivots on the text's own midpoint via `draw_text_center_u32` + a rotate-about-anchor transform — the run spins like a clock hand, not a flag flapping from one end.
            let aspect = vp.width_px as f32 / vp.height_px as f32;
            let demo_size = title_size * 0.85;
            let theta = snap_rotation((aspect - 1.0) * core::f32::consts::PI, demo_size, 8);
            let demo_anchor_x = vp.width_px as f32 * 0.5;
            let demo_anchor_y = bw * 4.0;
            let demo_transform = Transform::translate(-demo_anchor_x, -demo_anchor_y)
                .then(Transform::rotate(theta))
                .then(Transform::translate(demo_anchor_x, demo_anchor_y));
            let _ = text.draw_text_center_u32(
                &mut buffer,
                buf_w,
                buf_h,
                "rotation tracks viewport aspect ratio (proper AA via swash::scale)",
                demo_anchor_x,
                demo_anchor_y,
                demo_size,
                400,
                theme::TEXT_COLOUR,
                "Open Sans",
                None,
                None,
                Some(demo_transform),
            );
        }
        }

        // 7. Hover overlay on whichever button is hovered.
        self.hover_pixel_list = chrome::pixels_for_button(&self.hit_test_map, self.hover_state);
        chrome::draw_button_hover_by_pixels(&mut buffer, &self.hover_pixel_list, true, self.hover_state);
        buffer.present().expect("softbuffer buffer.present");
    }

    /// Look up the chrome hit-id under the current cursor. Cursor coordinates are external input — winit reports positions outside the window during drag-resize and during the moment cursor leaves.
    ///
    /// **Rule 0 — WHY/PROOF/PREVENTS:** WHY: a negative `mx` cast to `usize` wraps to a huge value; without the check, `hit_test_map[idx]` panics. PROOF: indexing past the slice panics. PREVENTS: panic on cursor outside window.
    fn hit_at_cursor(&self) -> u8 {
        let vp = self.compositor.viewport();
        let mx = self.cursor_x as i32;
        let my = self.cursor_y as i32;
        // Cast-and-compare: negative i32 casts to a huge usize, fails the `< width` check naturally.
        if (mx as usize) < (vp.width_px as usize) && (my as usize) < (vp.height_px as usize) {
            self.hit_test_map[(my as usize) * (vp.width_px as usize) + (mx as usize)]
        } else {
            HIT_NONE
        }
    }
}

impl ApplicationHandler for DesktopApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        // Default initial size: 4:3 window scaled to the monitor's short edge. `short = min(w,h)` is orientation-independent (works the same on portrait, landscape, square monitors). Window height = 3/4 of the short edge leaves a quarter-edge of breathing room for OS chrome / dock / taskbars; width = 4h/3 = the full short edge gives a classic 4:3 aspect ratio. Numbers are arbitrary defaults — consumers override by passing their preferred Viewport size and we'll honor it next session when the API is wired through. Falls back to the compositor's existing viewport size when no monitor info is available.
        let initial = if let Some(monitor) = event_loop.primary_monitor() {
            let size = monitor.size();
            let short = size.width.min(size.height);
            let h = short * 3 / 4;
            let w = h * 4 / 3;
            winit::dpi::PhysicalSize::new(w, h)
        } else {
            let vp = self.compositor.viewport();
            winit::dpi::PhysicalSize::new(vp.width_px, vp.height_px)
        };

        let attrs = WindowAttributes::default()
            .with_title(&self.title)
            .with_inner_size(initial)
            // 24 × 8 minimum: chosen so the controls strip is **always visible** at any non-degenerate window size. With chrome's `button_height = MIN_BUTTON_HEIGHT_PX + ceil(span/32)` formula and `MIN_BUTTON_HEIGHT_PX = 4`, the maximum button_height as height → ∞ at width=W is `4 + ceil(2W/32) = 4 + ceil(W/16)`, giving total_width = `7*(4 + ceil(W/16))/2`. Setting `width >= 24` guarantees `total_width <= 21 <= 24` at all heights; `height >= 8` guarantees the strip fits vertically (button_height <= 6 at min).
            .with_min_inner_size(winit::dpi::PhysicalSize::new(24u32, 8u32))
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(cfg!(not(target_os = "macos")));
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));

        #[cfg(target_os = "macos")]
        {
            use winit::platform::macos::WindowExtMacOS;
            window.set_has_shadow(false);
        }

        let context = softbuffer::Context::new(window.clone()).expect("softbuffer Context::new");
        let mut surface = softbuffer::Surface::new(&context, window.clone()).expect("softbuffer Surface::new");
        surface
            .resize(
                NonZeroU32::new(initial.width).expect("nonzero width"),
                NonZeroU32::new(initial.height).expect("nonzero height"),
            )
            .expect("softbuffer Surface::resize");

        // Compositor must match the actual surface size.
        self.compositor.resize(initial.width, initial.height);
        // Rule 0: `width * height` as u32 can overflow on absurd sizes (4G+ pixels). `checked_mul` returns None and we cap at 0 (rendering produces no hits). PREVENTS: silent u32 wrap → wrong allocation size → either OOM on a tiny vec or panicky out-of-bounds reads.
        let map_size = (initial.width as usize)
            .checked_mul(initial.height as usize)
            .unwrap_or(0);
        self.hit_test_map = vec![HIT_NONE; map_size];

        // Lazily build the text renderer (FontSystem creation parses bundled TTFs).
        if self.text.is_none() {
            self.text = Some(TextRenderer::new());
        }

        self.window = Some(window.clone());
        self.surface = Some(surface);
        window.request_redraw();
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                // Killswitch-compliant exit: hand the process back to the kernel directly. No Drop chain, no surface teardown, no font-cache release — all of that is process-scoped state the kernel reclaims in microseconds anyway. Per AGENT.md's persistence cadence rule, every state change is already on disk within 1 s of the change, so there's nothing to flush. "Ring always valid; any state is a valid checkpoint." Same path as ferros's hardware power cutoff, just at the OS-process level.
                std::process::exit(0);
            }
            WindowEvent::Resized(size) => {
                // Coalesce no-op resizes. Some compositors (Wayland especially) send a flood of same-size Resized events while the user drags past the other edge with min_inner_size clamping in effect — each one would request_redraw and re-thrash the glyph cache, pegging CPU. Bail if nothing actually changed.
                let current_vp = self.compositor.viewport();
                if size.width == current_vp.width_px && size.height == current_vp.height_px {
                    return;
                }
                if let (Some(surface), Some(width), Some(height)) = (
                    self.surface.as_mut(),
                    NonZeroU32::new(size.width),
                    NonZeroU32::new(size.height),
                ) {
                    surface.resize(width, height).expect("softbuffer Surface::resize");
                    self.compositor.resize(size.width, size.height);
                    let map_size = (size.width as usize)
                        .checked_mul(size.height as usize)
                        .unwrap_or(0);
                    self.hit_test_map.resize(map_size, HIT_NONE);
                    if let Some(window) = self.window.as_ref() { window.request_redraw(); }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x as f32;
                self.cursor_y = position.y as f32;
                let new_hit = self.hit_at_cursor();
                if let Some(window) = self.window.as_ref() {
                    window.set_cursor(cursor_for_state(new_hit, self.cursor_x, self.cursor_y, &self.compositor));
                }
                if new_hit != self.hover_state {
                    self.hover_state = new_hit;
                    if let Some(window) = self.window.as_ref() { window.request_redraw(); }
                }
            }
            WindowEvent::CursorLeft { .. } => {
                if self.hover_state != chrome::HIT_NONE {
                    self.hover_state = chrome::HIT_NONE;
                    if let Some(window) = self.window.as_ref() {
                        window.set_cursor(CursorIcon::Default);
                        window.request_redraw();
                    }
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                let Some(window) = self.window.as_ref() else { return; };
                // Photon's priority: window controls > resize edges > drag.
                match self.hit_at_cursor() {
                    HIT_CLOSE_BUTTON => { std::process::exit(0); }
                    HIT_MINIMIZE_BUTTON => { window.set_minimized(true); return; }
                    HIT_MAXIMIZE_BUTTON => { window.set_maximized(!window.is_maximized()); return; }
                    _ => {}
                }
                let vp = self.compositor.viewport();
                let edge = chrome::get_resize_edge(vp.width_px, vp.height_px, self.cursor_x, self.cursor_y);
                if let Some(dir) = resize_direction(edge) {
                    let _ = window.drag_resize_window(dir);
                    return;
                }
                let _ = window.drag_window();
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
            }
            _ => {}
        }
    }
}

fn resize_direction(edge: ResizeEdge) -> Option<ResizeDirection> {
    Some(match edge {
        ResizeEdge::Top => ResizeDirection::North,
        ResizeEdge::Bottom => ResizeDirection::South,
        ResizeEdge::Left => ResizeDirection::West,
        ResizeEdge::Right => ResizeDirection::East,
        ResizeEdge::TopLeft => ResizeDirection::NorthWest,
        ResizeEdge::TopRight => ResizeDirection::NorthEast,
        ResizeEdge::BottomLeft => ResizeDirection::SouthWest,
        ResizeEdge::BottomRight => ResizeDirection::SouthEast,
        ResizeEdge::None => return None,
    })
}

fn cursor_for_state(hit: u8, x: f32, y: f32, compositor: &Compositor) -> CursorIcon {
    match hit {
        HIT_CLOSE_BUTTON | HIT_MINIMIZE_BUTTON | HIT_MAXIMIZE_BUTTON => return CursorIcon::Pointer,
        _ => {}
    }
    let vp = compositor.viewport();
    match chrome::get_resize_edge(vp.width_px, vp.height_px, x, y) {
        ResizeEdge::Top | ResizeEdge::Bottom => CursorIcon::NsResize,
        ResizeEdge::Left | ResizeEdge::Right => CursorIcon::EwResize,
        ResizeEdge::TopLeft | ResizeEdge::BottomRight => CursorIcon::NwseResize,
        ResizeEdge::TopRight | ResizeEdge::BottomLeft => CursorIcon::NeswResize,
        ResizeEdge::None => CursorIcon::Default,
    }
}
