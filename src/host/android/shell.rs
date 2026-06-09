//! [`AndroidShell`] — the Android-side equivalent of host-winit's `DesktopShell`.
//!
//! Owns a [`FluorApp`] instance + the rendering pipeline + the Android surface. Photon's
//! `jni_android.rs` constructs one of these in `nativeInit` and ferries the pointer back to
//! Kotlin as a `jlong`. Each subsequent JNI entry point downcasts the pointer back to
//! `&mut AndroidShell<A>` and invokes the matching method.
//!
//! Pipeline mirrors `DesktopShell::render_frame` but stripped to the bare essentials Android
//! needs:
//! - No drop shadow (the surface IS the full screen, no band to cast onto).
//! - No persistent_screen distinction (window_rect = full surface = same buffer).
//! - No clip_mask carving (no rounded window corners on Android).
//! - Choreographer-driven scheduling, not winit's event loop.
//! - Skip render entirely when the AndroidWindow dirty flag is false (saves the ANativeWindow
//!   lock/copy cycle on idle frames, though Choreographer still advances).

use alloc::string::String;
use alloc::vec::Vec;
use core::time::Duration;
use std::time::Instant;

use ndk::native_window::NativeWindow;

use super::events;
use super::surface::Surface;
use super::window::AndroidWindow;
use crate::canvas::{Damage, PixelRect};
use crate::coord::Coord;
use crate::event::{Event as FEvent, ModifiersState as FModifiersState};
use crate::geom::Viewport;
use crate::host::app::{Context, FluorApp};
use crate::host::EventResponse;
use crate::text::TextRenderer;

/// The Android equivalent of `DesktopShell`. Wraps a [`FluorApp`] with the surface +
/// pipeline + input translation needed to drive it from Android's Choreographer + JNI.
pub struct AndroidShell<A: FluorApp> {
    app: A,
    surface: Surface,
    text: Option<TextRenderer>,
    window: AndroidWindow,
    viewport: Viewport,
    cursor_x: Coord,
    cursor_y: Coord,
    modifiers: FModifiersState,
    /// Scratch buffer in α + darkness format that the app paints into via `app.render`.
    scratch: Vec<u32>,
    /// Window-shape clip mask. Android has no rounded corners so this stays all-255 (fully
    /// visible). Resized to match scratch.
    clip_mask: Vec<u8>,
    /// Damage accumulator for the current frame.
    pending_damage: Damage,
    /// Last-known good last_tick used by `tick`-style apps.
    #[allow(dead_code)]
    last_tick: Option<Instant>,
}

impl<A: FluorApp> AndroidShell<A> {
    /// Construct the shell. Caller provides the surface dimensions Android opened the
    /// SurfaceView at (typically full-screen). The app's `init` is invoked here once the
    /// shell has its viewport + text renderer ready.
    pub fn new(mut app: A, width: u32, height: u32) -> Self {
        let viewport = Viewport::new(width, height);
        let mut shell = Self {
            app,
            surface: Surface::new(width, height),
            text: Some(TextRenderer::new()),
            window: AndroidWindow::new(),
            viewport,
            cursor_x: 0.0,
            cursor_y: 0.0,
            modifiers: FModifiersState::empty(),
            scratch: alloc::vec![0u32; (width as usize) * (height as usize)],
            clip_mask: alloc::vec![255u8; (width as usize) * (height as usize)],
            pending_damage: Damage::new(),
            last_tick: None,
        };
        shell.with_context(|app, ctx| app.init(ctx));
        shell
    }

    /// Resize the surface + viewport + scratch + clip_mask. Called from `nativeResize`.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface.resize(width, height);
        self.viewport = Viewport::new(width, height).with_ru(self.viewport.ru);
        let px = (width as usize) * (height as usize);
        self.scratch.resize(px, 0);
        self.clip_mask.resize(px, 255);
        self.window.mark_dirty();
        self.with_context(|app, ctx| app.on_resize(width, height, ctx));
    }

    /// Render one frame. Returns `true` if anything was actually copied to the
    /// ANativeWindow surface (the magic-pixel cache can short-circuit on cached buffers).
    pub fn draw(&mut self, window: &NativeWindow) -> bool {
        // Drive any per-tick animation state first; apps that return `true` mean "I changed
        // something, please paint." Mark window dirty so the pipeline below runs.
        let now = Instant::now();
        self.last_tick = Some(now);
        let tick_dirty = self.with_context(|app, ctx| app.tick(ctx));
        if tick_dirty {
            self.window.mark_dirty();
        }

        let was_dirty = self.window.take_dirty();
        if !was_dirty {
            // No content change; still post the surface so Choreographer's frame timing
            // continues, but skip the render+copy entirely. surface.present(window, false)
            // tells the magic-pixel cache "no new content" and copies only if the current
            // Android buffer happens to hold stale pixels.
            return self.surface.present(window, false);
        }

        // Compute damage clip — same shape as DesktopShell::render_frame's path.
        let viewport_rect = PixelRect::new(
            0,
            0,
            self.viewport.width_px as usize,
            self.viewport.height_px as usize,
        );
        let damage_clip = self
            .app
            .damage_rect(self.viewport)
            .unwrap_or(viewport_rect);
        if damage_clip.is_empty() {
            return self.surface.present(window, false);
        }

        // Wipe the scratch region the app is about to paint over.
        clear_scratch_rect(
            &mut self.scratch,
            self.viewport.width_px as usize,
            damage_clip,
        );

        self.pending_damage.clear();

        // App renders into scratch (α + darkness format).
        self.with_context_render(damage_clip, |app, scratch, ctx| {
            app.render(scratch, ctx);
        });

        // Finalize scratch (α + darkness) → surface buffer (visible RGB ARGB) at offset
        // (0, 0). On Android the window IS the full surface so there's no window_rect
        // offset, no shadow band, no persistent_screen distinction.
        let win_w = self.viewport.width_px as usize;
        let win_h = self.viewport.height_px as usize;
        crate::paint::finalize_into_screen(
            &self.scratch,
            &self.clip_mask,
            win_w,
            win_h,
            self.surface.buffer_mut(),
            win_w,
            0,
            0,
            damage_clip,
            true, // Treat every frame as a full repaint for now — incremental damage on Android can come later once we benchmark.
        );

        // Overlay deltas: walk hit_test_map and apply per-id hover/focus tints. apply_overlay
        // diffs against `last_active` so a stale-active id wraps its delta off; we keep a
        // session-long Vec for that so per-frame allocation stays small. Future work: hoist
        // last_active onto `self` so it survives across draws — for now it's zeroed each
        // frame, which means no flicker (current deltas overwrite cleanly) but each tint
        // toggle re-paints every pixel the first frame it lands.
        let overlay = self.app.overlay_deltas();
        if let Some((map, hw, hh)) = self.app.hit_test_map() {
            let mut last_active = alloc::vec![false; overlay.len()];
            let buf = self.surface.buffer_mut();
            crate::paint::apply_overlay(
                &self.scratch,
                buf,
                self.viewport.width_px as usize,
                0,
                0,
                map,
                hw,
                hh,
                &overlay,
                &mut last_active,
            );
        }

        self.surface.present(window, true)
    }

    /// Touch dispatch from `nativeOnTouch`. Translates Android action codes into one or two
    /// fluor events, dispatches each through `app.on_event`. Tracks cursor position on
    /// CursorMoved so Context.cursor_x/y stays accurate.
    pub fn on_touch(&mut self, action: i32, x: f32, y: f32) {
        let (count, evs) = events::translate_touch(action, x as Coord, y as Coord);
        for ev in &evs[..count] {
            // Track cursor position so subsequent draws/hit-tests see the latest touch.
            if let FEvent::CursorMoved { x: cx, y: cy } = ev {
                self.cursor_x = *cx;
                self.cursor_y = *cy;
            }
            let _ = self.dispatch(ev);
        }
        // Touch always potentially affects visuals (hover state, focus). Mark dirty so the
        // next draw runs the pipeline.
        self.window.mark_dirty();
    }

    /// Key event from `nativeOnKeyEvent`. Returns true if the host handled it (app's response
    /// was `Handled`). Untranslated keys (Key::Unidentified) return false so Android's
    /// default behavior runs.
    pub fn on_key_event(&mut self, key_code: i32) -> bool {
        let Some(ev) = events::key_press_from_keycode(key_code) else {
            return false;
        };
        let handled = matches!(self.dispatch(&ev), EventResponse::Handled);
        self.window.mark_dirty();
        handled
    }

    /// Text input from `nativeOnTextInput` (Java `String` committed by the soft IME). Builds
    /// an `Event::Ime(Commit(text))` and dispatches.
    pub fn on_text_input(&mut self, text: String) {
        let ev = events::ime_commit(text);
        let _ = self.dispatch(&ev);
        self.window.mark_dirty();
    }

    /// Android back button. Today we route it as `Escape` (cancels focus / closes overlays).
    /// Apps that want explicit back-handling can intercept `Event::KeyboardInput` with
    /// `NamedKey::Escape` and respond accordingly.
    pub fn on_back_pressed(&mut self) -> bool {
        let Some(ev) = events::key_press_from_keycode(111 /* KEYCODE_ESCAPE */) else {
            return false;
        };
        let handled = matches!(self.dispatch(&ev), EventResponse::Handled);
        self.window.mark_dirty();
        handled
    }

    /// Pinch-to-zoom. Multiplies the viewport's `ru` by the scale factor and triggers
    /// `on_resize` so layout code re-runs against the new effective span. Matches the
    /// desktop Ctrl++/Ctrl-+ semantic.
    pub fn on_scale(&mut self, scale_factor: f32) {
        if scale_factor <= 0.0 || !scale_factor.is_finite() {
            return;
        }
        let new_ru = self.viewport.ru * scale_factor;
        self.viewport = self.viewport.with_ru(new_ru);
        self.window.mark_dirty();
        let (w, h) = (self.viewport.width_px, self.viewport.height_px);
        self.with_context(|app, ctx| app.on_resize(w, h, ctx));
    }

    /// Borrow the underlying app. Lets photon's JNI shim wire app-specific functionality
    /// that doesn't fit fluor's compositor surface (avatar picker, FCM peer updates).
    pub fn app(&mut self) -> &mut A {
        &mut self.app
    }

    // ------------------------------------------------------------------------
    // Internal Context builders

    fn with_context<R>(&mut self, f: impl FnOnce(&mut A, &mut Context) -> R) -> R {
        let text = self
            .text
            .as_mut()
            .expect("TextRenderer must be initialized in AndroidShell::new");
        let mut ctx = Context {
            viewport: self.viewport,
            text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &self.window,
            modifiers: self.modifiers,
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
            is_maximized: false,
            damage_clip: PixelRect::new(
                0,
                0,
                self.viewport.width_px as usize,
                self.viewport.height_px as usize,
            ),
        };
        f(&mut self.app, &mut ctx)
    }

    fn with_context_render(
        &mut self,
        damage_clip: PixelRect,
        f: impl FnOnce(&mut A, &mut [u32], &mut Context),
    ) {
        let text = self
            .text
            .as_mut()
            .expect("TextRenderer must be initialized in AndroidShell::new");
        let mut ctx = Context {
            viewport: self.viewport,
            text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &self.window,
            modifiers: self.modifiers,
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
            is_maximized: false,
            damage_clip,
        };
        f(&mut self.app, &mut self.scratch, &mut ctx);
    }

    fn dispatch(&mut self, ev: &FEvent) -> EventResponse {
        self.with_context(|app, ctx| app.on_event(ev, ctx))
    }
}

/// Clear `scratch[y0..y1][x0..x1]` to α=0 + zero-darkness. Same shape as
/// `DesktopShell::clear_scratch_rect`.
fn clear_scratch_rect(scratch: &mut [u32], width: usize, rect: PixelRect) {
    if rect.is_empty() || width == 0 {
        return;
    }
    let x0 = rect.x0;
    let x1 = rect.x1.min(width);
    if x0 >= x1 {
        return;
    }
    let row_len = x1 - x0;
    for y in rect.y0..rect.y1 {
        let base = y * width + x0;
        if base + row_len > scratch.len() {
            break;
        }
        for cell in &mut scratch[base..base + row_len] {
            *cell = 0;
        }
    }
}
