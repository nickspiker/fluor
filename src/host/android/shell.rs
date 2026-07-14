//! [`AndroidShell`] — the Android-side equivalent of host-winit's `DesktopShell`.
//!
//! Owns a [`FluorApp`] instance + the rendering pipeline + the Android surface. Photon's `jni_android.rs` constructs one of these in `nativeInit` and ferries the pointer back to Kotlin as a `jlong`. Each subsequent JNI entry point downcasts the pointer back to `&mut AndroidShell<A>` and invokes the matching method.
//!
//! Pipeline mirrors `DesktopShell::render_frame` but stripped to the bare essentials Android needs:
//! - No drop shadow (the surface IS the full screen, no band to cast onto).
//! - No persistent_screen distinction (window_rect = full surface = same buffer).
//! - No clip_mask carving (no rounded window corners on Android).
//! - Choreographer-driven scheduling, not winit's event loop.
//! - Skip render entirely when the AndroidWindow dirty flag is false (saves the ANativeWindow lock/copy cycle on idle frames, tho Choreographer still advances).

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
use crate::host::wake::NoopWakeSender;
use crate::host::EventResponse;
use crate::paint::{HitId, HIT_NONE};
use crate::text::TextRenderer;
use alloc::sync::Arc;

/// The Android equivalent of `DesktopShell`. Wraps a [`FluorApp`] with the surface + pipeline + input translation needed to drive it from Android's Choreographer + JNI.
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
    /// Window-shape clip mask. Android has no rounded corners so this stays all-255 (fully visible). Resized to match scratch.
    clip_mask: Vec<u8>,
    /// Damage accumulator for the current frame.
    pending_damage: Damage,
    /// Press-hold-release + drag-off-cancel arbiter (shared with the desktop host). Fed the hit id under the finger at each touch DOWN / MOVE / UP; gates action dispatch to a validated release and surfaces the held id for the "held" colour. See [`crate::host::pointer`].
    pointer: crate::host::pointer::PointerArbiter,
    /// A finger is currently down (between ACTION_DOWN and ACTION_UP/CANCEL). Gates touch-drag → scroll: a MOVE while down emits a synthetic `MouseWheel` so the app's existing wheel handling scrolls (contacts, conversation, settings) — desktop has a wheel, touch didn't.
    touch_down: bool,
    /// The finger's y at the last touch event, for the per-move scroll delta.
    touch_last_y: Coord,
    /// Last-known good last_tick used by `tick`-style apps.
    #[allow(dead_code)]
    last_tick: Option<Instant>,
}

impl<A: FluorApp> AndroidShell<A> {
    /// Construct the shell. Caller provides the surface dimensions Android opened the SurfaceView at (typically full-screen). The app's `set_event_proxy` is invoked with a [`NoopWakeSender`] (Android background tasks talk to the UI thread thru JNI callbacks, not the proxy) and then `init` runs once the shell has its viewport + text renderer ready — same host contract as `DesktopShell`.
    pub fn new(mut app: A, width: u32, height: u32) -> Self {
        // Android writes finalize output directly into an ANativeWindow_lock'd buffer that the SurfaceFlinger compositor consumes after unlockAndPost. Worker-thread writes to that buffer need to be visible to the compositor by the time unlockAndPost runs; rayon's join is a CPU memory barrier but not a guaranteed device-coherent flush across all worker cores. Forcing par_rows / par_chunks sequential keeps writes on the calling thread so unlockAndPost's cache flush covers everything in one shot — eliminates the "horizontal white band at a random row" tear we hit with parallel finalize.
        crate::par::FORCE_SEQUENTIAL.store(true, core::sync::atomic::Ordering::Relaxed);
        let viewport = Viewport::new(width, height);
        // Host contract: set_event_proxy fires BEFORE init. On desktop run_app wraps winit's EventLoopProxy; here we hand the app a no-op sender. Apps that override `on_user_event` to react to background-task pings won't see any (background tasks should use JNI callbacks to wake the Activity on Android instead).
        let wake: Arc<dyn crate::host::wake::WakeSender<A::UserEvent>> = Arc::new(NoopWakeSender);
        app.set_event_proxy(wake);
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
            pointer: crate::host::pointer::PointerArbiter::new(),
            touch_down: false,
            touch_last_y: 0.0,
            last_tick: None,
        };
        shell.with_context(|app, ctx| app.init(ctx));
        shell
    }

    /// Resize the surface + viewport + scratch + clip_mask. Called from `nativeResize` on surfaceChanged (IME show/hide, screen orientation change). The Activity drives orientation itself via `setRequestedOrientation`, so a tilt arrives here as a regular resize with swapped (w, h) — photon's existing resize codepath reflows the layout automatically.
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

    /// Render one frame. Returns `true` if pixels were actually written to the ANativeWindow surface (the magic-pixel cache can short-circuit on cached buffers).
    pub fn draw(&mut self, window: &NativeWindow) -> bool {
        let now = Instant::now();
        self.last_tick = Some(now);
        let tick_dirty = self.with_context(|app, ctx| app.tick(ctx));
        if tick_dirty {
            self.window.mark_dirty();
        }

        let win_w = self.viewport.width_px as usize;
        let win_h = self.viewport.height_px as usize;
        let viewport_rect = PixelRect::new(0, 0, win_w, win_h);
        let was_dirty = self.window.take_dirty();

        if was_dirty {
            let damage_clip = self
                .app
                .damage_rect(self.viewport)
                .unwrap_or(viewport_rect);
            if !damage_clip.is_empty() {
                clear_scratch_rect(&mut self.scratch, win_w, damage_clip);
                self.pending_damage.clear();
                self.with_context_render(damage_clip, |app, scratch, ctx| {
                    app.render(scratch, ctx);
                });
            }
        }

        // Single-pass present: finalize scratch (α + darkness) directly into the locked ANativeWindow bits at the buffer's stride, skipping the intermediate Vec<u32>. The full viewport is passed as the finalize clip — on Android we treat every paint as a full repaint, so on cache-miss frames the locked buffer gets refreshed from scratch's current state regardless of which damage rect drove this frame.
        self.surface.present(
            window,
            &self.scratch,
            &self.clip_mask,
            win_w,
            win_h,
            viewport_rect,
            was_dirty,
        )
    }

    /// Touch dispatch from `nativeOnTouch`. Translates Android action codes into one or two fluor events, dispatches each thru `app.on_event`. Tracks cursor position on CursorMoved so Context.cursor_x/y stays accurate.
    ///
    /// Return value is the Android `nativeOnTouch` ABI: `1` = host should show the soft keyboard (focus moved into a text input), `-1` = hide, `0` = no change. Wraps `FluorApp::wants_keyboard`; the JNI shim passes it straight back to Java.
    pub fn on_touch(&mut self, action: i32, x: f32, y: f32) -> i32 {
        use crate::event::{ElementState, MouseButton};
        let (count, evs) = events::translate_touch(action, x as Coord, y as Coord);
        for ev in &evs[..count] {
            if let FEvent::CursorMoved { x: cx, y: cy } = ev {
                self.cursor_x = *cx;
                self.cursor_y = *cy;
            }
            // Feed the press-hold-release arbiter (identical semantics to the desktop host). Down arms the element under the finger; a drag off toggles the held colour; a release over the same element is the validated activation (`on_activate`); ACTION_CANCEL (→ CursorLeft) disarms. Arbiter runs BEFORE `dispatch(ev)` so a release's activation is delivered ahead of the raw Released, matching desktop ordering.
            match ev {
                FEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                    self.pointer.on_down(self.hit_under_cursor());
                    // Arm touch-drag scroll from the press position (the DOWN's CursorMoved already updated cursor_y just above).
                    self.touch_down = true;
                    self.touch_last_y = self.cursor_y;
                }
                FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left, .. } => {
                    if let Some(id) = self.pointer.on_up(self.hit_under_cursor()) {
                        self.dispatch_activate(id);
                    }
                    self.touch_down = false;
                }
                FEvent::CursorMoved { .. } => {
                    if self.pointer.on_move(self.hit_under_cursor()) {
                        self.window.mark_dirty();
                    }
                }
                FEvent::CursorLeft => {
                    self.pointer.on_cancel();
                    self.touch_down = false;
                }
                _ => {}
            }
            let _ = self.dispatch(ev);
            // Touch-drag → scroll: a MOVE while the finger is down emits a synthetic MouseWheel so the app's wheel handling scrolls (contacts / conversation / settings). Same sign convention as a trackpad flick — drag up (y decreases) yields a negative pixel delta, which the app's wheel arm turns into "reveal lower". Dispatched AFTER the CursorMoved so the arbiter's drag-off (tap cancel) is already processed. The DOWN's own CursorMoved arrives before `touch_down` is armed, so it never scrolls.
            if let FEvent::CursorMoved { y: cy, .. } = ev {
                if self.touch_down {
                    let dy = *cy - self.touch_last_y;
                    self.touch_last_y = *cy;
                    if dy != 0.0 {
                        let _ = self.dispatch(&FEvent::MouseWheel {
                            delta: crate::event::MouseScrollDelta::Pixels(0.0, dy),
                        });
                    }
                }
            }
        }
        self.window.mark_dirty();
        self.poll_keyboard()
    }

    /// Hit id under the finger right now, from the app's [`FluorApp::hit_test_map`] at the surface-local cursor (Android's surface IS the window, so cursor coords need no origin offset). `HIT_NONE` when there is no map or the point is out of bounds. Feeds the [`crate::host::pointer::PointerArbiter`].
    fn hit_under_cursor(&self) -> HitId {
        let x = self.cursor_x as i32;
        let y = self.cursor_y as i32;
        if x < 0 || y < 0 {
            return HIT_NONE;
        }
        match self.app.hit_test_map() {
            Some((map, w, h)) if (x as usize) < w && (y as usize) < h => {
                map[(y as usize) * w + (x as usize)]
            }
            _ => HIT_NONE,
        }
    }

    /// Deliver a validated activation ([`FluorApp::on_activate`]) — finger up over the same element it went down on. Android mirror of the desktop `dispatch_activate`.
    fn dispatch_activate(&mut self, hit_id: HitId) {
        let (x, y, mods) = (self.cursor_x, self.cursor_y, self.modifiers);
        self.with_context(|app, ctx| {
            let _ = app.on_activate(hit_id, x, y, mods, ctx);
        });
    }

    /// Poll `FluorApp::wants_keyboard` and map its one-shot Option to the Android IME-action ABI (`1` = show, `-1` = hide, `0` = no change). Called from both `on_touch` (focus changes driven by user taps) and the JNI shim's per-frame `nativePollKeyboard` hook so app-driven focus changes (e.g. `change_focus(None)` from `submit_handle` while the user is just watching the "Attesting…" spinner) propagate to the Activity without waiting for the next touch.
    pub fn poll_keyboard(&mut self) -> i32 {
        match self.app.wants_keyboard() {
            Some(true) => 1,
            Some(false) => -1,
            None => 0,
        }
    }

    /// Poll `FluorApp::wants_input_reset` → `1` (the Activity should `InputMethodManager.restartInput` to clear a stale IME composing buffer after a programmatic text clear) or `0`. One-shot, drained here.
    pub fn poll_input_reset(&mut self) -> i32 {
        self.app.wants_input_reset() as i32
    }

    /// Key event from `nativeOnKeyEvent`. Returns true if the host handled it (app's response was `Handled`). Untranslated keys (Key::Unidentified) return false so Android's default behavior runs.
    pub fn on_key_event(&mut self, key_code: i32) -> bool {
        let Some(ev) = events::key_press_from_keycode(key_code) else {
            return false;
        };
        let handled = matches!(self.dispatch(&ev), EventResponse::Handled);
        self.window.mark_dirty();
        handled
    }

    /// Text input from `nativeOnTextInput` (Java `String` committed by the soft IME). Builds an `Event::Ime(Commit(text))` and dispatches.
    pub fn on_text_input(&mut self, text: String) {
        let ev = events::ime_commit(text);
        let _ = self.dispatch(&ev);
        self.window.mark_dirty();
    }

    /// Android back button. Today we route it as `Escape` (cancels focus / closes overlays). Apps that want explicit back-handling can intercept `Event::KeyboardInput` with `NamedKey::Escape` and respond accordingly.
    pub fn on_back_pressed(&mut self) -> bool {
        let Some(ev) = events::key_press_from_keycode(111 /* KEYCODE_ESCAPE */) else {
            return false;
        };
        let handled = matches!(self.dispatch(&ev), EventResponse::Handled);
        self.window.mark_dirty();
        handled
    }

    /// Pinch-to-zoom. Multiplies the viewport's `ru` by the scale factor and triggers `on_resize` so layout code re-runs against the new effective span. Matches the desktop Ctrl++/Ctrl-+ semantic.
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

    /// Borrow the underlying app. Lets photon's JNI shim wire app-specific functionality that doesn't fit fluor's compositor surface (avatar picker, FCM peer updates).
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
            pressed_hit: self.pointer.held_id(),
            viewport: self.viewport,
            text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &self.window,
            modifiers: self.modifiers,
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
            is_maximized: false,
            window_origin: (0, 0),
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
            pressed_hit: self.pointer.held_id(),
            viewport: self.viewport,
            text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &self.window,
            modifiers: self.modifiers,
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
            is_maximized: false,
            window_origin: (0, 0),
            damage_clip,
        };
        f(&mut self.app, &mut self.scratch, &mut ctx);
    }

    fn dispatch(&mut self, ev: &FEvent) -> EventResponse {
        self.with_context(|app, ctx| app.on_event(ev, ctx))
    }
}

/// Clear `scratch[y0..y1][x0..x1]` to α=0 + zero-darkness. Same shape as `DesktopShell::clear_scratch_rect`.
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
