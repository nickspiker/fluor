//! Desktop host: winit window + platform-appropriate framebuffer.
//!
//! macOS: wgpu/Metal renderer with PostMultiplied alpha (transparent squircle corners).
//! Linux/Windows: softbuffer CPU framebuffer.
//!
//! Borderless window. Chrome is rendered by photon's `draw_window_controls` (verbatim, see [`super::chrome`]). Click routing uses photon's `hit_test_map` (per-pixel button ID) + `get_resize_edge`. Window setup matches photon's main.rs: `with_decorations(false)`, `with_transparent(true)`, `with_resizable(...)`, monitor-relative initial size, macOS drop shadow off.
//!
//! macOS resize uses manual mouse tracking via direct NSEvent polling (photon's approach) because winit stops delivering CursorMoved events once the cursor leaves the window during a resize drag. Linux uses native `drag_resize_window`.

use super::chrome::{self, ResizeEdge, HIT_CLOSE_BUTTON, HIT_MAXIMIZE_BUTTON, HIT_MINIMIZE_BUTTON, HIT_NONE};
use crate::coord::Coord;
use crate::paint;
use crate::paint::{snap_rotation, Transform};
use crate::text::TextRenderer;
use crate::theme;
use crate::widgets::Textbox;
use crate::Compositor;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::error::EventLoopError;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{CursorIcon, Window, WindowAttributes, WindowId};

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

    // --- Renderer ---
    // macOS: wgpu/Metal renderer (PostMultiplied alpha for transparent corners).
    #[cfg(target_os = "macos")]
    renderer: Option<super::renderer_wgpu::Renderer>,
    // Linux/Windows: softbuffer CPU framebuffer.
    #[cfg(not(target_os = "macos"))]
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,

    /// Per-pixel button-id map written by `draw_window_controls` and read on click.
    hit_test_map: Vec<u8>,
    cursor_x: Coord,
    cursor_y: Coord,
    /// Currently hovered chrome button id (HIT_NONE if none). Drives the hover overlay.
    hover_state: u8,
    /// Cached pixel list for the currently hovered button — recomputed on hover-state change.
    hover_pixel_list: Vec<usize>,
    /// Font system + glyph cache, lazily initialized on first `resumed`.
    text: Option<TextRenderer>,
    /// Demo textbox living in the panes example. Edit it with the keyboard once you click in.
    demo_textbox: Option<Textbox>,

    // --- macOS manual resize tracking ---
    // winit stops delivering CursorMoved when the cursor leaves the window during a
    // resize drag on macOS. We poll NSEvent.mouseLocation directly via objc_msgSend
    // and apply resizes manually via request_inner_size + set_outer_position.
    #[cfg(target_os = "macos")]
    is_dragging_resize: bool,
    #[cfg(target_os = "macos")]
    resize_edge: ResizeEdge,
    #[cfg(target_os = "macos")]
    drag_start_size: (u32, u32),
    #[cfg(target_os = "macos")]
    drag_start_window_pos: (i32, i32),
    #[cfg(target_os = "macos")]
    drag_start_cursor_screen_pos: (f64, f64),
    #[cfg(target_os = "macos")]
    screen_height: u32,
}

impl DesktopApp {
    fn new(compositor: Compositor, title: String) -> Self {
        Self {
            compositor,
            title,
            window: None,
            #[cfg(target_os = "macos")]
            renderer: None,
            #[cfg(not(target_os = "macos"))]
            surface: None,
            hit_test_map: Vec::new(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            hover_state: chrome::HIT_NONE,
            hover_pixel_list: Vec::new(),
            text: None,
            demo_textbox: None,
            #[cfg(target_os = "macos")]
            is_dragging_resize: false,
            #[cfg(target_os = "macos")]
            resize_edge: ResizeEdge::None,
            #[cfg(target_os = "macos")]
            drag_start_size: (0, 0),
            #[cfg(target_os = "macos")]
            drag_start_window_pos: (0, 0),
            #[cfg(target_os = "macos")]
            drag_start_cursor_screen_pos: (0.0, 0.0),
            #[cfg(target_os = "macos")]
            screen_height: 0,
        }
    }

    /// Reposition + resize the demo textbox to match the current viewport. Called after window create + on resize so the textbox tracks the viewport span.
    fn update_textbox_layout(&mut self) {
        let vp = self.compositor.viewport();
        let span = 2.0 * vp.width_px as Coord * vp.height_px as Coord / (vp.width_px as Coord + vp.height_px as Coord);
        let bw = chrome::MIN_BUTTON_HEIGHT_PX as Coord + crate::math::ceil(span / 32.0);
        let center_x = vp.width_px as Coord * 0.5;
        let center_y = bw * 7.0;
        let width = (vp.width_px as Coord * 0.5).max(bw * 8.0);
        let height = bw * 1.6;
        let font_size = bw * 0.55;
        let text = match self.text.as_mut() { Some(t) => t, None => return };
        if let Some(tb) = self.demo_textbox.as_mut() {
            tb.set_rect(center_x, center_y, width, height);
            tb.set_font_size(font_size, text);
        } else {
            let mut tb = Textbox::new(center_x, center_y, width, height, font_size);
            tb.set_font_size(font_size, text);
            self.demo_textbox = Some(tb);
        }
    }

    fn render_frame(&mut self) {
        let vp = self.compositor.viewport();
        let buf_w = vp.width_px as usize;
        let buf_h = vp.height_px as usize;
        let vp_w = vp.width_px;
        let vp_h = vp.height_px;
        let needed = buf_w * buf_h;
        if self.hit_test_map.len() != needed {
            self.hit_test_map.resize(needed, HIT_NONE);
        } else {
            self.hit_test_map.fill(HIT_NONE);
        }

        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = self.renderer.as_mut() else { return; };
            let mut buffer = renderer.lock_buffer();
            Self::render_into(&mut self.compositor, &mut self.hit_test_map, &self.title, self.text.as_mut(), self.demo_textbox.as_ref(), &mut self.hover_pixel_list, self.hover_state, &mut buffer, buf_w, buf_h, vp_w, vp_h);
            let _ = buffer.present();
        }
        #[cfg(not(target_os = "macos"))]
        {
            let Some(surface) = self.surface.as_mut() else { return; };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            Self::render_into(&mut self.compositor, &mut self.hit_test_map, &self.title, self.text.as_mut(), self.demo_textbox.as_ref(), &mut self.hover_pixel_list, self.hover_state, &mut buffer, buf_w, buf_h, vp_w, vp_h);
            buffer.present().expect("softbuffer buffer.present");
        }
    }

    /// Paint everything into the given pixel buffer. Shared between wgpu and softbuffer paths.
    fn render_into(compositor: &mut Compositor, hit_test_map: &mut Vec<u8>, title: &str, mut text: Option<&mut TextRenderer>, demo_textbox: Option<&Textbox>, hover_pixel_list: &mut Vec<usize>, hover_state: u8, buffer: &mut [u32], buf_w: usize, buf_h: usize, vp_w: u32, vp_h: u32) {
        // 1. Background noise — full buffer fill
        paint::background_noise(buffer, buf_w, buf_h, 0, true, 0, None);
        // 2. Panes on top of background
        compositor.render(buffer, buf_w, buf_h);
        // 3. Chrome controls strip (writes hit_test_map for click routing).
        let (start, crossings, button_x_start, button_height) = chrome::draw_window_controls(
            buffer,
            hit_test_map,
            vp_w,
            vp_h,
            1.0,
        );
        // 4. Two-tone window edges + squircle corner mask (verbatim photon)
        chrome::draw_window_edges_and_mask(
            buffer,
            hit_test_map,
            vp_w,
            vp_h,
            start,
            &crossings,
        );
        // 5. Vertical hairlines between control buttons (verbatim photon)
        chrome::draw_button_hairlines(
            buffer,
            hit_test_map,
            vp_w,
            vp_h,
            button_x_start,
            button_height,
            start,
            &crossings,
        );
        // 6. Title text — left-aligned, vertically centered in the button row, sized relative to button height.
        // Skipped below a readable size — controls are the load-bearing UI; text shouldn't smear into illegibility.
        let span = 2.0 * vp_w as Coord * vp_h as Coord / (vp_w as Coord + vp_h as Coord);
        let bw = chrome::MIN_BUTTON_HEIGHT_PX as Coord + crate::math::ceil(span / 32.0);
        let title_size = bw * 0.55;
        if title_size >= 6.0 {
        if let Some(ref mut text) = text {
            let pad = bw * 0.5;
            let baseline_y = bw * 0.5;
            let _ = text.draw_text_left_u32(
                buffer,
                buf_w,
                buf_h,
                title,
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

            // Aspect-ratio-driven rotation demo.
            let aspect = vp_w as Coord / vp_h as Coord;
            let demo_size = title_size * 0.85;
            let theta = snap_rotation((aspect - 1.0) * core::f32::consts::PI, demo_size, 8);
            let demo_anchor_x = vp_w as Coord * 0.5;
            let demo_anchor_y = bw * 4.0;
            let demo_transform = Transform::translate(-demo_anchor_x, -demo_anchor_y)
                .then(Transform::rotate(theta))
                .then(Transform::translate(demo_anchor_x, demo_anchor_y));
            let _ = text.draw_text_center_u32(
                buffer,
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

        // 7. Demo textbox — sits below the rotation demo, accepts focus + keyboard input.
        if let Some(tb) = demo_textbox {
            if let Some(ref mut text) = text {
                tb.render(buffer, buf_w, buf_h, text, None, None);
            }
        }

        // 8. Hover overlay on whichever button is hovered.
        *hover_pixel_list = chrome::pixels_for_button(hit_test_map, hover_state);
        chrome::draw_button_hover_by_pixels(buffer, hover_pixel_list, true, hover_state);
    }

    /// Look up the chrome hit-id under the current cursor. Cursor coordinates are external input — winit reports positions outside the window during drag-resize and during the moment cursor leaves.
    ///
    /// **Rule 0 — WHY/PROOF/PREVENTS:** WHY: a negative `mx` cast to `usize` wraps to a huge value; without the check, `hit_test_map[idx]` panics. PROOF: indexing past the slice panics. PREVENTS: panic on cursor outside window.
    fn hit_at_cursor(&self) -> u8 {
        let vp = self.compositor.viewport();
        let mx = self.cursor_x as i32;
        let my = self.cursor_y as i32;
        if (mx as usize) < (vp.width_px as usize) && (my as usize) < (vp.height_px as usize) {
            self.hit_test_map[(my as usize) * (vp.width_px as usize) + (mx as usize)]
        } else {
            HIT_NONE
        }
    }

    // --- macOS manual resize ---

    /// macOS: poll mouse position and button state directly from AppKit.
    /// Called from `about_to_wait()` to track the cursor even when it's outside the window
    /// (winit stops delivering CursorMoved once the cursor leaves during resize).
    #[cfg(target_os = "macos")]
    fn poll_macos_resize(&mut self) -> bool {
        let Some(window) = self.window.as_ref() else { return false; };
        use std::ffi::{c_char, c_void};

        #[repr(C)]
        #[derive(Clone, Copy)]
        struct NSPoint { x: f64, y: f64 }

        unsafe extern "C" {
            fn objc_msgSend(receiver: *const c_void, sel: *const c_void) -> usize;
            fn sel_registerName(name: *const c_char) -> *const c_void;
            fn objc_getClass(name: *const c_char) -> *const c_void;
        }

        unsafe {
            let cls = objc_getClass(b"NSEvent\0".as_ptr() as *const c_char);

            let sel_loc = sel_registerName(b"mouseLocation\0".as_ptr() as *const c_char);
            let mouse_location: extern "C" fn(*const c_void, *const c_void) -> NSPoint =
                std::mem::transmute(objc_msgSend as *const ());
            let ns_point = mouse_location(cls, sel_loc);

            let sel_btn = sel_registerName(b"pressedMouseButtons\0".as_ptr() as *const c_char);
            let buttons = objc_msgSend(cls, sel_btn);
            let left_held = buttons & 1 != 0;

            // Convert AppKit coords (logical, bottom-left) to winit coords (physical, top-left)
            let scale = window.scale_factor();
            let phys_x = ns_point.x * scale;
            let phys_y = self.screen_height as f64 - ns_point.y * scale;

            if let Ok(window_pos) = window.outer_position() {
                self.cursor_x = (phys_x - window_pos.x as f64) as Coord;
                self.cursor_y = (phys_y - window_pos.y as f64) as Coord;
            }

            if !left_held {
                self.is_dragging_resize = false;
                self.resize_edge = ResizeEdge::None;
                return true;
            }

            self.apply_resize();
        }
        false
    }

    /// macOS: compute new window size from current mouse delta and apply via `request_inner_size` + `set_outer_position`.
    #[cfg(target_os = "macos")]
    fn apply_resize(&self) {
        let Some(window) = self.window.as_ref() else { return; };
        let Ok(window_pos) = window.outer_position() else { return; };

        let current_screen_x = window_pos.x as f64 + self.cursor_x as f64;
        let current_screen_y = window_pos.y as f64 + self.cursor_y as f64;

        let dx = (current_screen_x - self.drag_start_cursor_screen_pos.0) as Coord;
        let dy = (current_screen_y - self.drag_start_cursor_screen_pos.1) as Coord;

        let min_size: Coord = 128.0;

        let (new_width, new_height, should_move, new_x, new_y) = match self.resize_edge {
            ResizeEdge::Right => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                (w, self.drag_start_size.1, false, 0, 0)
            }
            ResizeEdge::Left => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let width_change = self.drag_start_size.0 as i32 - w as i32;
                (w, self.drag_start_size.1, true, self.drag_start_window_pos.0 + width_change, self.drag_start_window_pos.1)
            }
            ResizeEdge::Bottom => {
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                (self.drag_start_size.0, h, false, 0, 0)
            }
            ResizeEdge::Top => {
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let height_change = self.drag_start_size.1 as i32 - h as i32;
                (self.drag_start_size.0, h, true, self.drag_start_window_pos.0, self.drag_start_window_pos.1 + height_change)
            }
            ResizeEdge::TopRight => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let height_change = self.drag_start_size.1 as i32 - h as i32;
                (w, h, true, self.drag_start_window_pos.0, self.drag_start_window_pos.1 + height_change)
            }
            ResizeEdge::TopLeft => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let width_change = self.drag_start_size.0 as i32 - w as i32;
                let height_change = self.drag_start_size.1 as i32 - h as i32;
                (w, h, true, self.drag_start_window_pos.0 + width_change, self.drag_start_window_pos.1 + height_change)
            }
            ResizeEdge::BottomRight => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                (w, h, false, 0, 0)
            }
            ResizeEdge::BottomLeft => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                let width_change = self.drag_start_size.0 as i32 - w as i32;
                (w, h, true, self.drag_start_window_pos.0 + width_change, self.drag_start_window_pos.1)
            }
            ResizeEdge::None => return,
        };

        if should_move {
            let _ = window.set_outer_position(winit::dpi::PhysicalPosition::new(new_x, new_y));
        }
        let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(new_width, new_height));
    }
}

impl ApplicationHandler for DesktopApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }

        let monitor = event_loop.primary_monitor()
            .or_else(|| event_loop.available_monitors().next());

        // Default initial size: 4:3 window scaled to the monitor's short edge.
        let initial = if let Some(ref mon) = monitor {
            let size = mon.size();
            let short = size.width.min(size.height);
            let h = short * 3 / 4;
            let w = h * 4 / 3;
            winit::dpi::PhysicalSize::new(w, h)
        } else {
            let vp = self.compositor.viewport();
            winit::dpi::PhysicalSize::new(vp.width_px, vp.height_px)
        };

        // Store screen height for macOS NSEvent coordinate conversion (AppKit uses bottom-left origin).
        #[cfg(target_os = "macos")]
        if let Some(ref mon) = monitor {
            self.screen_height = mon.size().height;
        }

        let attrs = WindowAttributes::default()
            .with_title(&self.title)
            .with_inner_size(initial)
            .with_min_inner_size(winit::dpi::PhysicalSize::new(24u32, 8u32))
            .with_decorations(false)
            .with_transparent(true)
            // macOS: start non-resizable so AppKit doesn't override our cursor near edges.
            // We track resize manually via NSEvent polling.
            .with_resizable(cfg!(not(target_os = "macos")));
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));

        #[cfg(target_os = "macos")]
        {
            use winit::platform::macos::WindowExtMacOS;
            window.set_has_shadow(false);
        }

        // Compositor must match the actual surface size.
        self.compositor.resize(initial.width, initial.height);
        let map_size = (initial.width as usize)
            .checked_mul(initial.height as usize)
            .unwrap_or(0);
        self.hit_test_map = vec![HIT_NONE; map_size];

        // --- Platform renderer init ---
        #[cfg(target_os = "macos")]
        {
            self.renderer = Some(super::renderer_wgpu::Renderer::new(&window, initial.width, initial.height));
        }
        #[cfg(not(target_os = "macos"))]
        {
            use std::num::NonZeroU32;
            let context = softbuffer::Context::new(window.clone()).expect("softbuffer Context::new");
            let mut surface = softbuffer::Surface::new(&context, window.clone()).expect("softbuffer Surface::new");
            surface
                .resize(
                    NonZeroU32::new(initial.width).expect("nonzero width"),
                    NonZeroU32::new(initial.height).expect("nonzero height"),
                )
                .expect("softbuffer Surface::resize");
            self.surface = Some(surface);
        }

        // Lazily build the text renderer (FontSystem creation parses bundled TTFs).
        if self.text.is_none() {
            self.text = Some(TextRenderer::new());
        }
        self.update_textbox_layout();

        self.window = Some(window.clone());
        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // macOS resize polling: when dragging a resize edge, winit stops delivering
        // CursorMoved once the cursor leaves the window. We poll NSEvent.mouseLocation
        // directly and apply resizes each tick until the button is released.
        #[cfg(target_os = "macos")]
        if self.is_dragging_resize {
            if self.poll_macos_resize() {
                if let Some(window) = self.window.as_ref() { window.request_redraw(); }
            }
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(
                std::time::Instant::now() + std::time::Duration::from_millis(16),
            ));
            return;
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                std::process::exit(0);
            }
            WindowEvent::Resized(size) => {
                let current_vp = self.compositor.viewport();
                if size.width == current_vp.width_px && size.height == current_vp.height_px {
                    return;
                }
                if size.width == 0 || size.height == 0 { return; }

                #[cfg(target_os = "macos")]
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
                #[cfg(not(target_os = "macos"))]
                if let Some(surface) = self.surface.as_mut() {
                    use std::num::NonZeroU32;
                    if let (Some(w), Some(h)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height)) {
                        surface.resize(w, h).expect("softbuffer Surface::resize");
                    }
                }

                self.compositor.resize(size.width, size.height);
                let map_size = (size.width as usize)
                    .checked_mul(size.height as usize)
                    .unwrap_or(0);
                self.hit_test_map.resize(map_size, HIT_NONE);
                self.update_textbox_layout();
                if let Some(window) = self.window.as_ref() { window.request_redraw(); }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x as Coord;
                self.cursor_y = position.y as Coord;
                let new_hit = self.hit_at_cursor();
                let over_textbox = self
                    .demo_textbox
                    .as_ref()
                    .map_or(false, |tb| tb.contains(self.cursor_x, self.cursor_y));
                let icon = if new_hit == HIT_NONE && over_textbox {
                    CursorIcon::Text
                } else {
                    cursor_for_state(new_hit, self.cursor_x, self.cursor_y, &self.compositor)
                };
                if let Some(window) = self.window.as_ref() {
                    window.set_cursor(icon);
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
            WindowEvent::Focused(false) => {
                // Cancel any in-progress resize drag when the window loses focus —
                // we won't receive a MouseInput Released event in that case.
                #[cfg(target_os = "macos")]
                {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                }
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                let Some(window) = self.window.as_ref() else { return; };
                // Priority: window controls > resize edges > textbox > drag.
                match self.hit_at_cursor() {
                    HIT_CLOSE_BUTTON => { std::process::exit(0); }
                    HIT_MINIMIZE_BUTTON => { window.set_minimized(true); return; }
                    HIT_MAXIMIZE_BUTTON => { window.set_maximized(!window.is_maximized()); return; }
                    _ => {}
                }
                let vp = self.compositor.viewport();
                let edge = chrome::get_resize_edge(vp.width_px, vp.height_px, self.cursor_x, self.cursor_y);
                if edge != ResizeEdge::None {
                    // Linux: native resize protocol.
                    #[cfg(target_os = "linux")]
                    {
                        use winit::window::ResizeDirection;
                        if let Some(dir) = resize_direction(edge) {
                            let _ = window.drag_resize_window(dir);
                        }
                        return;
                    }
                    // macOS/Windows: manual resize tracking.
                    #[cfg(not(target_os = "linux"))]
                    {
                        #[cfg(target_os = "macos")]
                        {
                            self.is_dragging_resize = true;
                            self.resize_edge = edge;
                            self.drag_start_size = (vp.width_px, vp.height_px);
                            if let Ok(window_pos) = window.outer_position() {
                                self.drag_start_window_pos = (window_pos.x, window_pos.y);
                                self.drag_start_cursor_screen_pos = (
                                    window_pos.x as f64 + self.cursor_x as f64,
                                    window_pos.y as f64 + self.cursor_y as f64,
                                );
                            }
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            use winit::window::ResizeDirection;
                            if let Some(dir) = resize_direction(edge) {
                                let _ = window.drag_resize_window(dir);
                            }
                        }
                        return;
                    }
                }
                // Textbox click — focus + cursor positioning if inside, defocus otherwise.
                if let Some(tb) = self.demo_textbox.as_mut() {
                    let was_focused = tb.focused;
                    tb.handle_click(self.cursor_x, self.cursor_y);
                    if tb.focused || was_focused {
                        window.request_redraw();
                        if tb.focused { return; }
                    }
                }
                let _ = window.drag_window();
            }
            #[cfg(target_os = "macos")]
            WindowEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left, .. } => {
                if self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                    if let Some(window) = self.window.as_ref() { window.request_redraw(); }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state != ElementState::Pressed { return; }
                let Some(tb) = self.demo_textbox.as_mut() else { return; };
                if !tb.focused { return; }
                let Some(text) = self.text.as_mut() else { return; };
                let mut changed = false;
                match &event.logical_key {
                    Key::Named(NamedKey::Backspace) => { tb.backspace(text); changed = true; }
                    Key::Named(NamedKey::Delete) => { tb.delete_forward(text); changed = true; }
                    Key::Named(NamedKey::ArrowLeft) => { tb.cursor_left(); changed = true; }
                    Key::Named(NamedKey::ArrowRight) => { tb.cursor_right(); changed = true; }
                    Key::Named(NamedKey::Home) => { tb.cursor_home(); changed = true; }
                    Key::Named(NamedKey::End) => { tb.cursor_end(); changed = true; }
                    _ => {
                        if let Some(s) = &event.text {
                            for c in s.chars() {
                                if !c.is_control() {
                                    tb.insert_char(c, text);
                                    changed = true;
                                }
                            }
                        }
                    }
                }
                if changed {
                    if let Some(window) = self.window.as_ref() { window.request_redraw(); }
                }
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
            }
            _ => {}
        }
    }
}

/// Map resize edge to winit's `ResizeDirection` (used on Linux/Windows where native resize works).
#[cfg(not(target_os = "macos"))]
fn resize_direction(edge: ResizeEdge) -> Option<winit::window::ResizeDirection> {
    use winit::window::ResizeDirection;
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

fn cursor_for_state(hit: u8, x: Coord, y: Coord, compositor: &Compositor) -> CursorIcon {
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
