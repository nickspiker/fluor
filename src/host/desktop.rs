//! Desktop host: winit window + softbuffer CPU framebuffer. Available under feature `host-winit` (on by default).
//!
//! Borderless window. Chrome is rendered by photon's `draw_window_controls` (verbatim, see [`super::chrome`]). Click routing uses photon's `hit_test_map` (per-pixel button ID) + `get_resize_edge`. Window setup matches photon's main.rs: `with_decorations(false)`, `with_transparent(true)`, `with_resizable(...)`, monitor-relative initial size, macOS drop shadow off.

use super::chrome::{self, ResizeEdge, HIT_CLOSE_BUTTON, HIT_MAXIMIZE_BUTTON, HIT_MINIMIZE_BUTTON, HIT_NONE};
use crate::paint;
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
        paint::background_noise(&mut buffer, buf_w, buf_h, 0, true, 0);
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
        // 6. Hover overlay on whichever button is hovered.
        self.hover_pixel_list = chrome::pixels_for_button(&self.hit_test_map, self.hover_state);
        chrome::draw_button_hover_by_pixels(&mut buffer, &self.hover_pixel_list, true, self.hover_state);
        buffer.present().expect("softbuffer buffer.present");
    }

    fn hit_at_cursor(&self) -> u8 {
        let vp = self.compositor.viewport();
        let mx = self.cursor_x as i32;
        let my = self.cursor_y as i32;
        if mx < 0 || my < 0 || mx >= vp.width_px as i32 || my >= vp.height_px as i32 {
            return HIT_NONE;
        }
        let idx = (my as usize) * (vp.width_px as usize) + (mx as usize);
        if idx < self.hit_test_map.len() { self.hit_test_map[idx] } else { HIT_NONE }
    }
}

impl ApplicationHandler for DesktopApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        // Photon's recipe: window dimensions derived from the monitor so HiDPI scales correctly. Falls back to the compositor's viewport size if no monitor info is available.
        let initial = if let Some(monitor) = event_loop.primary_monitor() {
            let size = monitor.size();
            let short = size.width.min(size.height);
            let h = (short * 3 / 4).max(480);
            let w = (h * 4 / 3).min(size.width).max(640);
            winit::dpi::PhysicalSize::new(w, h)
        } else {
            let vp = self.compositor.viewport();
            winit::dpi::PhysicalSize::new(vp.width_px, vp.height_px)
        };

        let attrs = WindowAttributes::default()
            .with_title(&self.title)
            .with_inner_size(initial)
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
        self.hit_test_map = vec![HIT_NONE; (initial.width * initial.height) as usize];

        self.window = Some(window.clone());
        self.surface = Some(surface);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let (Some(surface), Some(width), Some(height)) = (
                    self.surface.as_mut(),
                    NonZeroU32::new(size.width),
                    NonZeroU32::new(size.height),
                ) {
                    surface.resize(width, height).expect("softbuffer Surface::resize");
                    self.compositor.resize(size.width, size.height);
                    self.hit_test_map.resize((size.width * size.height) as usize, HIT_NONE);
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
                    HIT_CLOSE_BUTTON => { event_loop.exit(); return; }
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
