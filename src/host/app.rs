//! `FluorApp` trait + entry point for consumer-driven desktop apps.
//!
//! Consumers implement [`FluorApp`] and pass the impl to [`run_app`]. The host opens a window, runs the event loop, presents the buffer, and dispatches events thru the trait. All visible content (chrome, widgets, panes) is the consumer's responsibility — the host owns no domain state.
//!
//! Compose [`super::chrome_widget::DefaultChrome`] for the borderless window frame, [`crate::widgets::Textbox`] / [`crate::widgets::BlinkTimer`] for the textbox + blinking-cursor pattern, [`crate::Group`] for sub-viewport composite caching. The [`Context`] struct exposes the host's shared resources (viewport, text renderer, window handle, modifier state) to the consumer for the duration of each callback.
//!
//! The current `desktop::run(compositor, title)` is a transitional shim that wraps the legacy demo into a `FluorApp`. New code should use [`run_app`] directly.

use super::WindowHandle;
use crate::coord::Coord;
use crate::event::{CursorIcon as FCursorIcon, Event as FEvent, ModifiersState as FModifiersState};
use crate::geom::Viewport;
use crate::text::TextRenderer;
use std::time::Instant;
// FluorApp::set_event_proxy takes a fluor-native `Arc<dyn WakeSender<Self::UserEvent>>`. Concrete winit machinery (ApplicationHandler, EventLoop, EventLoopProxy, WindowAttributes, etc.) only enters via the desktop_shell sub-module below, behind the host-winit feature gate.

#[cfg(feature = "host-winit")]
use super::chrome::ResizeEdge;
#[cfg(feature = "host-winit")]
use super::winit_compat;
#[cfg(feature = "host-winit")]
use std::sync::Arc;
#[cfg(feature = "host-winit")]
use winit::application::ApplicationHandler;
#[cfg(feature = "host-winit")]
use winit::error::EventLoopError;
#[cfg(feature = "host-winit")]
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
#[cfg(feature = "host-winit")]
use winit::event_loop::{ActiveEventLoop, EventLoop};
#[cfg(feature = "host-winit")]
use winit::keyboard::ModifiersState;
#[cfg(feature = "host-winit")]
use winit::window::{Window, WindowAttributes, WindowId};

/// X11-only XShape helpers — direct XCB calls that winit doesn't expose. Currently houses [`x11_atomic::set_input_region`] (window-shape input clipping); historically also held an atomic-geometry helper that's gone now. The `x11_atomic` name is retained because the (single) remaining helper still operates on an XCB connection independent of winit's, which is the property the name actually tracks.
#[cfg(all(feature = "host-winit", target_os = "linux"))]
mod x11_atomic {
    use std::sync::OnceLock;
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use x11rb::connection::Connection;
    use x11rb::rust_connection::RustConnection;

    /// Lazily-opened XCB connection, shared across all atomic-geometry calls. Independent of the connection winit holds internally (which we can't access) — the X server doesn't care which client sends the ConfigureRequest as long as we name the right window ID.
    fn conn() -> Option<&'static RustConnection> {
        static CONN: OnceLock<Option<RustConnection>> = OnceLock::new();
        CONN.get_or_init(|| x11rb::connect(None).ok().map(|(c, _screen)| c))
            .as_ref()
    }

    /// Restrict the window's INPUT region to the given screen-space rectangle. Clicks outside this rect pass thru to whatever window is behind us. Used by the fullscreen-compositor architecture: our OS surface covers the whole screen but the visible window is just a sub-rect, so we tell X11 "I'm only hittable inside that sub-rect" — the rest is mouse-transparent. Call once per `window_rect` change (initial creation, drag-to-move, resize-drag, monitor change).
    ///
    /// The rect is in window-relative coordinates (= screen coords when the OS window is fullscreen at the screen origin). Negative offsets get clamped to 0 since XShape rectangles must be unsigned. Returns `true` if the call was sent successfully, `false` if the window isn't X11 or the connection failed.
    pub fn set_input_region(
        window: &winit::window::Window,
        x: i32,
        y: i32,
        w: u32,
        h: u32,
    ) -> bool {
        use x11rb::protocol::shape::{ConnectionExt as _, SK, SO};
        use x11rb::protocol::xproto::{ClipOrdering, Rectangle};

        let Ok(handle) = window.window_handle() else {
            return false;
        };
        let xid = match handle.as_raw() {
            RawWindowHandle::Xcb(h) => h.window.get(),
            RawWindowHandle::Xlib(h) => h.window as u32,
            _ => return false,
        };
        let Some(conn) = conn() else {
            return false;
        };
        let rect = Rectangle {
            x: x.max(0).min(i16::MAX as i32) as i16,
            y: y.max(0).min(i16::MAX as i32) as i16,
            width: w.min(u16::MAX as u32) as u16,
            height: h.min(u16::MAX as u32) as u16,
        };
        if conn
            .shape_rectangles(
                SO::SET,
                SK::INPUT,
                ClipOrdering::UNSORTED,
                xid,
                0,
                0,
                &[rect],
            )
            .is_err()
        {
            return false;
        }
        let _ = conn.flush();
        true
    }

    /// The desktop work area `(x, y, w, h)` — the monitor minus space reserved by panels /
    /// taskbars — read from the root window's EWMH `_NET_WORKAREA` property. Used to place
    /// the visible window so its bottom edge (the chrome status band) doesn't slide under a
    /// taskbar. `_NET_WORKAREA` holds `[x, y, w, h]` per virtual desktop; we take the first
    /// (current/default desktop). Returns `None` if not X11, the atom is unset (no EWMH WM),
    /// or the read fails — caller falls back to the full monitor.
    pub fn work_area() -> Option<(i32, i32, u32, u32)> {
        use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};
        let conn = conn()?;
        let root = conn.setup().roots.first()?.root;
        let atom = conn
            .intern_atom(false, b"_NET_WORKAREA")
            .ok()?
            .reply()
            .ok()?
            .atom;
        if atom == 0 {
            return None;
        }
        let reply = conn
            .get_property(false, root, atom, AtomEnum::CARDINAL, 0, 4)
            .ok()?
            .reply()
            .ok()?;
        let mut vals = reply.value32()?;
        let x = vals.next()? as i32;
        let y = vals.next()? as i32;
        let w = vals.next()?;
        let h = vals.next()?;
        if w == 0 || h == 0 {
            None
        } else {
            Some((x, y, w, h))
        }
    }
}

/// Windows work-area query — `SystemParametersInfo(SPI_GETWORKAREA)` gives the primary
/// monitor's work rect (full screen minus the taskbar), in virtual-screen pixels.
#[cfg(all(feature = "host-winit", target_os = "windows"))]
fn work_area_windows() -> Option<(i32, i32, u32, u32)> {
    use windows::Win32::Foundation::RECT;
    use windows::Win32::UI::WindowsAndMessaging::{
        SystemParametersInfoW, SPI_GETWORKAREA, SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS,
    };
    let mut r = RECT::default();
    let ok = unsafe {
        SystemParametersInfoW(
            SPI_GETWORKAREA,
            0,
            Some(&mut r as *mut RECT as *mut core::ffi::c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        )
    };
    if ok.is_err() {
        return None;
    }
    let w = (r.right - r.left).max(0) as u32;
    let h = (r.bottom - r.top).max(0) as u32;
    if w == 0 || h == 0 {
        None
    } else {
        Some((r.left, r.top, w, h))
    }
}

/// macOS work-area query — `NSScreen.mainScreen.visibleFrame` (full frame minus the menu
/// bar + Dock). macOS frames are in points with a bottom-left origin; we derive the insets
/// as fractions of the full frame and apply them to the physical monitor pixels (`mon_w` ×
/// `mon_h`), which is scale-factor-independent and yields a top-left-origin pixel rect.
#[cfg(all(feature = "host-winit", target_os = "macos"))]
fn work_area_macos(mon_w: u32, mon_h: u32) -> Option<(i32, i32, u32, u32)> {
    use objc2_app_kit::NSScreen;
    let mtm = objc2::MainThreadMarker::new()?;
    let screen = NSScreen::mainScreen(mtm)?;
    let full = screen.frame();
    let vf = screen.visibleFrame();
    let (fw, fh) = (full.size.width, full.size.height);
    if fw <= 0.0 || fh <= 0.0 {
        return None;
    }
    let left_f = ((vf.origin.x - full.origin.x) / fw).max(0.0);
    let bottom_f = ((vf.origin.y - full.origin.y) / fh).max(0.0);
    let right_f = (((full.origin.x + fw) - (vf.origin.x + vf.size.width)) / fw).max(0.0);
    let top_f = (((full.origin.y + fh) - (vf.origin.y + vf.size.height)) / fh).max(0.0);
    let (mw, mh) = (mon_w as f64, mon_h as f64);
    let wa_w = (mw * (1.0 - left_f - right_f)).max(0.0) as u32;
    let wa_h = (mh * (1.0 - top_f - bottom_f)).max(0.0) as u32;
    if wa_w == 0 || wa_h == 0 {
        None
    } else {
        Some(((left_f * mw) as i32, (top_f * mh) as i32, wa_w, wa_h))
    }
}

/// Desktop work area `(x, y, w, h)` in physical pixels (top-left origin) — the monitor
/// minus space reserved by panels / taskbars / the menu bar + Dock. Dispatches to the
/// platform query (X11 `_NET_WORKAREA`, Windows `SPI_GETWORKAREA`, macOS `visibleFrame`);
/// falls back to the full monitor on Wayland (no client-side work-area query) and anywhere
/// the query is unavailable.
#[cfg(feature = "host-winit")]
fn monitor_work_area(mon_w: u32, mon_h: u32) -> (i32, i32, u32, u32) {
    #[cfg(target_os = "linux")]
    {
        // Wayland has no EWMH root window; `work_area()` returns None there and we fall back.
        return x11_atomic::work_area().unwrap_or((0, 0, mon_w, mon_h));
    }
    #[cfg(target_os = "windows")]
    {
        return work_area_windows().unwrap_or((0, 0, mon_w, mon_h));
    }
    #[cfg(target_os = "macos")]
    {
        return work_area_macos(mon_w, mon_h).unwrap_or((0, 0, mon_w, mon_h));
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        (0, 0, mon_w, mon_h)
    }
}

/// Per-callback access to host-owned shared resources. Re-borrowed for each call into the trait — the consumer can keep references for the duration of the call but not across calls.
pub struct Context<'a> {
    /// Current viewport in physical pixels.
    pub viewport: Viewport,
    /// Shared font system + glyph caches. Initialized lazily by the host on first window creation; passed by mutable reference because cache insertion and font loading require mutation.
    pub text: &'a mut TextRenderer,
    /// Window-shape clip mask, one byte per pixel, same dimensions as the present buffer. Default fill is `255` (fully visible — finalize_for_os multiplies by 255 ≈ no change). Consumers with a rounded window (e.g. `DefaultChrome`) carve the corner cutouts here once per resize; the boundary's [`crate::paint::finalize_for_os`] multiplies it into each pixel's α to trim the OS handoff. Decoupled from the present-buffer RGB so internal layer compositing stays opaque-or-empty and never deals with partial-α drift.
    pub clip_mask: &'a mut [u8],
    /// Frame-level damage accumulator owned by the host. Consumers paint into Canvas instances backed by this accumulator (`Canvas::new(target, w, h, ctx.damage)`); every rasterizer reports its painted bbox into it. The host reads it after `app.render` to know exactly what changed this frame — drives the optional damage-rect outline overlay today and (eventually) damage-clipped composite + present.
    pub damage: &'a mut crate::canvas::Damage,
    /// The damage clip the host computed for THIS frame, derived from `app.damage_rect(...)` before render. Consumers should thread this thru every flatten / blit / glow call as the `clip` parameter so they only touch pixels inside the dirty region. Defaults to the full viewport (legacy apps that don't override `FluorApp::damage_rect` get the current full-redraw behavior).
    pub damage_clip: crate::canvas::PixelRect,
    /// App-facing window handle. `WindowHandle` is intentionally minimal — only `request_redraw` lives there because it's the only window operation real apps invoke from the trait surface. Cursor / drag / maximize / minimize flow thru [`EventResponse`] variants instead, so the host's window state stays the single source of truth.
    pub window: &'a dyn WindowHandle,
    /// Latest tracked modifier state (shift / ctrl / alt / super) in fluor-native form. Hosts translate from platform input (winit `ModifiersState`, Android JNI mod-key flags) before constructing this.
    pub modifiers: FModifiersState,
    /// Last known cursor position in viewport pixels (host-tracked across all events).
    pub cursor_x: Coord,
    pub cursor_y: Coord,
    /// `true` if the host's `window_rect` is currently in the screen-sized "maximized" state set by [`EventResponse::ToggleMaximized`]. Consumers consult this so chrome can switch to full-edge mode (no corner cutouts, no perimeter hairline, no drop shadow) — the shadow/hairline are screen edges anyway, the WM can't show them, and AA on a corner that's flush with the screen is wasted work.
    pub is_maximized: bool,
    /// The visible window's top-left corner in screen coordinates (the fullscreen-compositor `window_rect` origin). Lets consumers screen-anchor content across origin-moving operations — a left/top edge resize shifts the origin, and content that should stay put on screen (an image canvas, a document) compensates by the origin delta. Bottom/right resizes and pure renders leave it unchanged. Android: always (0, 0) — the surface IS the window.
    pub window_origin: (i32, i32),
    /// The hit id currently held DOWN under the pointer and eligible to fire on release — the host's [`crate::host::pointer::PointerArbiter`] state, surfaced so the app can paint that element in its "held" colour (see [`crate::theme::BUTTON_HELD`]). `HIT_NONE` when nothing is pressed, or when a press has been dragged off its target. Consult it in `render` for custom hit-stamped elements; widget trees get it applied automatically via [`crate::host::widget::apply_pressed`].
    pub pressed_hit: crate::paint::HitId,
}

pub use super::event_response::EventResponse;

/// What a consumer implements to drive the desktop host.
pub trait FluorApp {
    /// Custom user-event payload for cross-thread wake-up. Background tasks (network, file I/O, async ceremonies) clone the `Arc<dyn WakeSender<Self::UserEvent>>` from [`Self::set_event_proxy`] and call `proxy.send(payload)` to wake the host; the host dispatches the payload back thru [`Self::on_user_event`] on the UI thread. Apps that don't need cross-thread wake-up declare `type UserEvent = ();` and skip the two methods.
    type UserEvent: 'static + Send;

    /// Initial window title. Default is empty; override or call `ctx.window.set_title(...)` from `init` if you want it set later.
    fn title(&self) -> &str {
        ""
    }

    /// The app-identity icon for the OS window (taskbar / alt-tab / title bar). The host
    /// applies it at window creation so the OS-level icon matches the in-chrome orb — apps
    /// that hold a [`DefaultChrome`] typically return `self.chrome.app_icon.as_ref()`.
    ///
    /// **Platform reach.** This drives winit's `set_window_icon`, which only takes effect on
    /// **Windows and X11**. On **Wayland** the icon is sourced from a `.desktop` file matched
    /// by `app_id`, and on **macOS** from the `.app` bundle's `.icns` — both are build-time
    /// packaging, not a runtime call, so this hook is a no-op there. Returns `None` by
    /// default (no OS icon set).
    fn window_icon(&self) -> Option<&crate::host::icon::Icon> {
        None
    }

    /// Hand off the host's wake-sender ONCE, before [`Self::init`], so the app can clone it for background threads. host-winit wraps `winit::event_loop::EventLoopProxy`; host-android wraps a JNI callback (or a [`super::NoopWakeSender`] when the app doesn't use cross-thread wake-ups). A typical implementer stashes the `Arc` in its own field and clone-and-ships it to spawned tasks. Default no-op for apps that don't need cross-thread wake-up.
    fn set_event_proxy(&mut self, proxy: alloc::sync::Arc<dyn super::WakeSender<Self::UserEvent>>) {
        let _ = proxy;
    }

    /// One-shot setup after the window exists. Allocate Groups, widgets, initial geometry. The viewport in `ctx` is the actual physical size the host opened.
    fn init(&mut self, ctx: &mut Context);

    /// The window resized. Resize internal Groups / widget bboxes to match.
    fn on_resize(&mut self, width: u32, height: u32, ctx: &mut Context);

    /// Window event from the host. Consumer returns an [`EventResponse`] telling the host what to do next. Events are fluor-native [`crate::event::Event`] values — hosts translate platform input at the boundary.
    fn on_event(&mut self, event: &FEvent, ctx: &mut Context) -> EventResponse;

    /// A clickable element was ACTIVATED — the pointer went down on `hit_id` and released over the same `hit_id`, with no drag-off in between (the press-hold-release model, arbitrated by [`crate::host::pointer::PointerArbiter`]). This is where apps fire the *action* for their custom hit-stamped elements, and dispatch release-activated widgets via [`crate::host::widget::dispatch_release`]. Raw press/release still arrive via [`Self::on_event`] for press-time concerns (focus, cursor placement, drag-select, window drag); actions belong here so a mis-touch dragged off before release fires nothing. `(x, y)` is the release position in viewport pixels. Default no-op ([`EventResponse::Pass`]) — apps opt in; those that don't keep whatever they do in `on_event` unchanged.
    fn on_activate(
        &mut self,
        hit_id: crate::paint::HitId,
        x: Coord,
        y: Coord,
        mods: FModifiersState,
        ctx: &mut Context,
    ) -> EventResponse {
        let _ = (hit_id, x, y, mods, ctx);
        EventResponse::Pass
    }

    /// Damage region this app will repaint this frame. Returns `None` if no widget state changed since the last frame — host can persist scratch as-is and skip render entirely. Returns `Some(rect)` to declare the union of all dirty widget bboxes (each widget's `prev ∪ current` from `widget.damage_rect(...)`); host clears scratch in that rect and threads it thru `ctx.damage_clip` so the consumer's render call clips every flatten / blit to it.
    ///
    /// Default impl returns `Some(full viewport)` — safe fallback that preserves today's full-redraw behavior. Apps opt into differential rendering by overriding this to union their widget damage rects.
    ///
    /// Takes `Viewport` directly (not `Context`) so the host can call it without holding the text-renderer borrow that `Context` carries.
    ///
    /// `&mut self` so an app can union widget damage by walking its own widget tree (which yields `&mut dyn Widget`) — the walk only reads each widget's `damage_rect`, but the tree-walk currency is `&mut`. Nothing is mutated.
    fn damage_rect(&mut self, viewport: Viewport) -> Option<crate::canvas::PixelRect> {
        let w = viewport.width_px as usize;
        let h = viewport.height_px as usize;
        Some(crate::canvas::PixelRect::new(0, 0, w, h))
    }

    /// Per-frame paint into the host's CPU present buffer. Flatten owned Groups onto `target`. The damage clip computed pre-render is in `ctx.damage_clip`; thread it thru every flatten / blit / glow call to skip pixels outside the dirty region.
    fn render(&mut self, target: &mut [u32], ctx: &mut Context);

    /// Per-hit-id overlay delta table for THIS frame. The host runs one walk over `hit_test_map()` after finalize+shadow; for each pixel `i`, if `current[id] != last_applied[id]`, it wrap-adds the prior delta back and wrap-subs the current delta in `persistent_screen` (visible-RGB space). Apps return a slice where entry `[id]` is the visible-RGB delta to apply to pixels marked with that hit id this frame (e.g. the hover tint when a button is hovered, zero otherwise). Length must equal `registry.next_id` (= 1 + number of registered hit zones); IDs past the slice are treated as zero-delta. Default impl: empty slice (no overlay tints, no allocations).
    ///
    /// Takes `&mut self` so apps can build the table by walking their [`crate::host::widget::Container`] (which threads `&mut dyn Widget` thru `visit`) — see [`crate::host::widget::build_overlay_deltas`] for the canonical one-liner implementation.
    fn overlay_deltas(&mut self) -> Vec<u32> {
        Vec::new()
    }

    /// Per-hit-id bbox table for THIS frame, PARALLEL to [`Self::overlay_deltas`] — entry `[id]` is the pixel bbox of that widget, or `None`.
    /// Lets the host bound the overlay tint scan to each hovered widget's rect instead of scanning the whole window every frame (the tint only touches pixels where `hit_map == id` inside the rect).
    /// `None` entries (and the default empty slice) fall back to a full-window scan for those ids — correct, just slower.
    /// Build via [`crate::host::widget::build_overlay_bboxes`].
    fn overlay_bboxes(
        &mut self,
        _viewport_w: usize,
        _viewport_h: usize,
    ) -> Vec<Option<crate::canvas::PixelRect>> {
        Vec::new()
    }

    /// Read-only handle to the consumer's hit-test map so the host's overlay diff pass can walk it. Returns `Some((&map, win_w, win_h))` where `map.len() >= win_w * win_h` (one [`crate::paint::HitId`] per pixel — `u16` since the v0.0 widening). Default `None` = no overlay walk, no hover support.
    fn hit_test_map(&self) -> Option<(&[crate::paint::HitId], usize, usize)> {
        None
    }

    /// Cursor icon at `(x, y)` in viewport pixel coords. Called whenever the cursor moves. Returns a fluor-native [`crate::event::CursorIcon`]; the host translates to its platform's cursor type before calling `set_cursor` on the OS window.
    fn cursor_for(&self, x: Coord, y: Coord, ctx: &Context) -> FCursorIcon;

    /// When to wake up next (animation timers, blinks). `None` = wait for input only. The host calls this once per `about_to_wait` cycle and feeds it into `ControlFlow::WaitUntil`.
    fn wake_at(&self) -> Option<Instant> {
        None
    }

    /// Called once per `about_to_wait` cycle (after the host's own platform polling). Drive time-based state here — blink timers, animation tweens, drag-scroll. Return `true` if state changed and a redraw is needed; the host will call `request_redraw` for you.
    fn tick(&mut self, ctx: &mut Context) -> bool {
        let _ = ctx;
        false
    }

    /// User-event payload arrived from a background thread via [`EventLoopProxy::send_event`]. Typical use: a network task completed, an avatar download finished, a key-ceremony hit a milestone — the task sends the appropriate variant; this method routes it to the right state-machine handler and (usually) calls `ctx.window.request_redraw()` to repaint with the new state. Default no-op for apps that declared `type UserEvent = ()`.
    fn on_user_event(&mut self, event: Self::UserEvent, ctx: &mut Context) {
        let _ = (event, ctx);
    }

    /// Initial visible-window size when the app first opens, given the monitor dimensions in pixels. Default returns half the monitor in each axis (the conventional "open at a reasonable fraction of the display, centred" desktop convention) — apps with strong aspect-ratio opinions (Photon's portrait launch window, fixed-aspect editors) override. Return `(width, height)`; the host clamps each to ≥ 1 and centres the window on the monitor.
    fn initial_size(&self, monitor: (u32, u32)) -> (u32, u32) {
        (monitor.0 / 2, monitor.1 / 2)
    }

    /// Whether the currently-focused widget wants the soft keyboard up. Polled by the Android host after each input event so the Activity can raise/dismiss the IME. `Some(true)` = the host should show the keyboard, `Some(false)` = hide, `None` = no change. Default `None` for apps that don't have text input. Desktop hosts ignore this — IME shows whenever a text field is focused on most desktop platforms anyway.
    ///
    /// `&mut self` so apps can implement "show on transition" via a one-shot pending flag that this call clears — repeated polls without a focus change return `None` and the Activity doesn't churn the IME.
    fn wants_keyboard(&mut self) -> Option<bool> {
        None
    }

    /// One-shot: the app cleared its text field programmatically (e.g. sent a message), so the Android host should `InputMethodManager.restartInput` to reset the IME's stale composing buffer — otherwise a predictive keyboard re-materialises the just-sent text on the next keystroke. Default `false` (no-op); drained per poll like [`FluorApp::wants_keyboard`].
    fn wants_input_reset(&mut self) -> bool {
        false
    }
}

/// Run the desktop host until the window closes. Builds an `EventLoop` typed on `A::UserEvent` so background-thread wake-ups via the WakeSender route thru [`FluorApp::on_user_event`]. The proxy is created up-front, wrapped in a [`winit_compat::WinitWakeSender`], and handed to the app via [`FluorApp::set_event_proxy`] BEFORE the event loop starts, so apps can clone-and-ship the Arc to background tasks during their own constructor or [`FluorApp::init`].
#[cfg(feature = "host-winit")]
pub fn run_app<A: FluorApp + 'static>(mut app: A) -> Result<(), EventLoopError> {
    let event_loop = EventLoop::<A::UserEvent>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();
    let wake: alloc::sync::Arc<dyn super::WakeSender<A::UserEvent>> =
        alloc::sync::Arc::new(winit_compat::WinitWakeSender::new(proxy));
    app.set_event_proxy(wake);
    let mut shell = DesktopShell::new(app);
    event_loop.run_app(&mut shell)
}

// ============================================================================ Everything below this point is `host-winit`-only — DesktopShell + winit event loop. AndroidShell lives at [`crate::host::android::shell`]. ============================================================================

/// Visible-window placement inside the fullscreen screen buffer. fluor now runs as a fullscreen transparent OS window owning the whole display — the "window" the consumer paints into is a sub-rect of that screen buffer at `(x, y)` with `(w, h)` pixels. `(x, y, w, h)` are screen-space pixel coordinates. `(0, 0)` is the top-left of the display. WindowRect is mutated by drag-to-move (changes `x, y`) and resize-drag (changes `w, h`); both are in-buffer operations that don't touch the OS window geometry.
#[derive(Clone, Copy, Debug)]
#[cfg(feature = "host-winit")]
struct WindowRect {
    x: i32,
    y: i32,
    w: u32,
    h: u32,
}

/// Damage-clipped fill(0) — wipes only the `rect` sub-region of `scratch` (viewport-flat slice, row-major width `win_w`). Replaces a full-buffer `fill(0)` so pixels outside the damage rect persist between frames. Each row inside the rect uses the SIMD-friendly slice `fill(0)` so the per-row cost is the same as the full-buffer call, just over fewer rows.
#[cfg(feature = "host-winit")]
fn clear_scratch_rect(scratch: &mut [u32], win_w: usize, rect: crate::canvas::PixelRect) {
    if rect.is_empty() {
        return;
    }
    let win_h = scratch.len() / win_w.max(1);
    let y0 = rect.y0.min(win_h);
    let y1 = rect.y1.min(win_h);
    let x0 = rect.x0.min(win_w);
    let x1 = rect.x1.min(win_w);
    if x0 >= x1 || y0 >= y1 {
        return;
    }
    for y in y0..y1 {
        let row_base = y * win_w;
        scratch[row_base + x0..row_base + x1].fill(0);
    }
}

/// The host's adapter — owns platform handles + the consumer's `App`, dispatches events thru the trait. Not user-facing; constructed by [`run_app`].
///
/// **Compositor architecture.** The OS window is fullscreen borderless transparent — fluor owns the entire screen buffer. The consumer paints into a window-sized scratch buffer (sized to `viewport` = `window_rect.w × window_rect.h`); the host then blits that scratch into the screen buffer at the `window_rect` offset. Pixels outside the window stay α=0 so the OS compositor shows whatever's behind us. Click-thru is via a per-resize input-region call (set later, see step 2 of the fullscreen-compositor pivot) so clicks outside `window_rect` route to whatever's underneath.
#[cfg(feature = "host-winit")]
struct DesktopShell<A: FluorApp> {
    app: A,
    window: Option<Arc<Window>>,
    /// Consumer-visible viewport — sized to `window_rect.w × window_rect.h`, NOT the screen. Consumers paint and lay out as if their window is `viewport.width_px × viewport.height_px`; the host handles placing that paint inside the larger screen buffer.
    viewport: Viewport,
    /// Display size in pixels (= OS window size in fullscreen mode). The OS surface buffer matches this.
    screen_size: (u32, u32),
    /// Desktop work area `(x, y, w, h)` — monitor minus panels/taskbars (X11 `_NET_WORKAREA`,
    /// else the full monitor). Initial placement and the "maximized" rect target this instead
    /// of the raw screen so the window never lands under a taskbar. Set in `resumed`.
    work_area: (i32, i32, u32, u32),
    /// Where the visible window lives inside the screen buffer. Driven by drag-to-move + resize-drag (later steps); initialized centered with a 3/4-of-monitor-short initial size.
    window_rect: WindowRect,
    /// Window-sized scratch buffer. The consumer renders into this (at viewport dimensions); the host runs `finalize_for_os` on it with the window-space clip mask, then blits row-by-row into the screen buffer at `window_rect.x, window_rect.y`. Resized on `window_rect` size change.
    scratch: Vec<u32>,

    // --- Renderer ---
    #[cfg(target_os = "macos")]
    renderer: Option<super::renderer_wgpu::Renderer>,
    #[cfg(not(target_os = "macos"))]
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,

    // --- Shared resources ---
    text: Option<TextRenderer>,
    /// Window-shape clip mask, one byte α per pixel. Sized to `viewport` (= window-space, NOT screen-space). The consumer carves shape into it (rounded corner cutouts etc.); `finalize_for_os` multiplies it into each pixel's α at the scratch-buffer boundary before the host blits to screen. Default `255` (fully visible) means a consumer that doesn't touch it gets a rectangular window.
    clip_mask: Vec<u8>,
    cursor_x: Coord,
    cursor_y: Coord,
    modifiers: ModifiersState,

    // --- Self-driven resize tracking. fluor owns the input side of the resize-drag loop on every platform: on edge-press we capture start geometry; on cursor-move we compute the new target geometry and push it to the OS via request_inner_size + set_outer_position. The OS confirms via Resized events which trigger the actual surface resize + paint — keeping buffer size == window size always, eliminating X11 PutImage mismatch smear. Replaces the WM-driven drag_resize_window path AND the macOS NSEvent polling hack with one unified flow.
    is_dragging_resize: bool,
    resize_edge: ResizeEdge,
    drag_start_size: (u32, u32),
    drag_start_window_pos: (i32, i32),
    drag_start_cursor_screen_pos: (i32, i32),

    // --- Drag-to-move tracking. In the fullscreen-compositor architecture the OS window is fullscreen and `window.drag_window()` doesn't move anything — we move our internal `window_rect` inside the screen buffer instead. On press we capture the cursor's screen position + window_rect origin; on cursor-move we update window_rect.x/y by the delta. The actual screen-buffer shift happens at vsync (RedrawRequested) via paint::shift_screen_wrap — skipping consumer render + finalize + shadow entirely during the drag. On drag release, a request_redraw kicks off a clean full re-render that overwrites the wrap artefacts.
    is_dragging_move: bool,
    /// Press-hold-release + drag-off-cancel arbiter (shared with the Android host). Fed the hit id under the cursor at each mouse down / move / up; gates action dispatch to a validated release and surfaces the currently-held id for the "held" colour. See [`crate::host::pointer`].
    pointer: crate::host::pointer::PointerArbiter,
    /// Click hit a drag-eligible area; the NEXT CursorMoved commits the move-drag (no dead zone — 1:1 tracking from the first pixel of motion). Set on `EventResponse::StartWindowDrag`; cleared on mouse release. A click with zero motion never commits because the commit lives in the CursorMoved arm, so click-without-drag stays free of wrap-shift artefacts.
    move_drag_armed: bool,
    drag_move_anchor_screen: (i32, i32),
    drag_move_rect_start: (i32, i32),
    /// Last window_rect (x, y, w, h) that was actually painted into the screen buffer. Set after every render_frame; consulted at drag-move vsync ticks to compute the (dx, dy) delta to feed into `shift_screen_wrap`. Without this we'd have no way to know "how much did the window move since the last frame" because the cursor anchor describes total drag distance, not per-frame increment.
    last_painted_rect: WindowRect,
    /// Saved `window_rect` from BEFORE the last `EventResponse::ToggleMaximized` set us screen-sized. `Some` ⇒ we're currently in the maximized state and the next toggle restores from here; `None` ⇒ we're at user-sized and the next toggle saves+grows. Drag-to-move while maximized currently drags the screen-sized rect (weird but harmless); a future iteration could auto-unmaximize on drag like most WMs.
    saved_rect_for_maximize: Option<WindowRect>,

    /// `false` until the first `WindowEvent::Resized` arrives confirming the OS surface size. Most WMs open a default-sized window (800×600 or similar) and then animate / configure it to fullscreen — Resized fires when the actual surface is ready. Until then, painting positions chrome against a stale `window_rect` (sized for the monitor we expected) inside a buffer that's smaller than expected, producing a brief "chrome in the top-left of a tiny window" flash as the WM grows the surface. Defer all rendering until this flips true.
    surface_ready: bool,

    /// Tracks `WindowEvent::Focused` so the drop shadow can dim when the window is inactive — focused windows cast a stronger shadow (`SHADOW_SEED_FOCUSED`), unfocused ones use a quarter-strength shadow (`SHADOW_SEED_UNFOCUSED`).
    is_focused: bool,

    /// Live render-pipeline counters. Updated every `render_frame` call (composite-time EMA + frame counter); rendered to a bottom-of-window debug strip when [`paint::DEBUG_SHOW_FPS`] is set via the `[]f` chord.
    debug_stats: crate::paint::DebugStats,

    /// Frame-level damage accumulator. Reset at the top of each `render_frame`; passed to the consumer via [`Context::damage`]; read back after consumer render to drive damage-clipped composite and the [`paint::DEBUG_SHOW_DAMAGE`] outline overlay.
    pending_damage: crate::canvas::Damage,
    /// FPS strip active state from the previous frame. When it toggles `true → false`, this frame's damage_clip must include the strip bbox so the just-vanished strip pixels get cleared from scratch (and propagated into persistent_screen via finalize). Tracked instead of a generic `prev_damage_clip` union to avoid sticky viewport-sized damage on hover frames after any prior full repaint.
    last_strip_active: bool,
    /// Persistent screen-sized pixel buffer, owned by us. Survives across frames so post-finalize overlays (hover/focus tint diffs, blinkey) can mutate just a few pixels each frame without re-running finalize for the whole window. The platform's softbuffer / wgpu back buffer may rotate or arrive stale; we always memcpy `persistent_screen` over it just before `present()` so the platform buffer's prior state doesn't matter. Resized when `screen_size` changes.
    persistent_screen: Vec<u32>,
    /// Set by any event that destroys the chrome perimeter + shadow band content in [`Self::persistent_screen`]: drag release, resize, zoom, focus change. Consumed once per `render_frame` to switch from incremental mode to full-repaint mode (wipe `persistent_screen`, finalize copies every pixel, paint_shadow runs once into the fresh band). Replaces every prior geometric-equality check on `damage_clip`.
    pending_full_repaint: bool,
    /// Which hit-ids the overlay wrote to persistent_screen LAST frame. Used so a transition (an id that was tinted, no longer is) still gets its pixels rewritten from scratch this frame to clear the prior tint. No tint magnitude is kept — the overlay just reads scratch and conditionally subtracts the current frame's delta. Re-sized to match the consumer's `overlay_deltas().len()` each frame (extended with `false` if the app registered new IDs since last frame; shrunk only on a full repaint). Cleared whenever `persistent_screen` is wiped.
    last_overlay_active: Vec<bool>,
    /// Last-seen value of `paint::DEBUG_SHOW_HITMASK`. When this differs from the current atomic value at the top of `render_frame`, we promote to a full repaint so the new finalize behavior (FORCE_OPAQUE-style scalar debug path / no shadow) lands across the whole window in one frame.
    last_hitmask: bool,
    /// Last-seen value of `paint::DEBUG_SHOW_ALPHA`. Same transition logic as `last_hitmask` — toggling alpha-viz changes finalize's debug branch, which requires a full repaint to refresh persistent_screen.
    last_alpha_mode: u8,
    /// Last-seen value of `paint::DEBUG_SHOW_OPAQUE_SCAN`. Same transition logic as `last_hitmask` — toggling the opaque-scan tint changes what finalize stamps into persistent_screen (every interior pixel gains +16 blue while on; goes back to clean copy while off), so the next frame must be a full_repaint to wash the entire silhouette interior in one shot rather than only the next incidentally-damaged sub-rect.
    last_opaque_scan: bool,
    /// Dedicated staging buffer for the FPS strip (debug). Sized to `win_w × DEBUG_STRIP_H` lazily on first use; the strip rasterizes here in α + darkness and then gets converted + clobbered into persistent_screen. Kept entirely separate from the app's scratch so the strip never contaminates the consumer's render path.
    strip_buf: Vec<u32>,
    /// macOS click-thru: true when we've told the OS to ignore mouse events for this window (cursor is over a transparent area). A global NSEvent monitor polls cursor position to detect re-entry.
    #[cfg(target_os = "macos")]
    hittest_off: bool,
    #[cfg(target_os = "macos")]
    hittest_monitor: Option<super::macos_hittest::HittestMonitor>,
}

#[cfg(feature = "host-winit")]
impl<A: FluorApp> DesktopShell<A> {
    fn new(app: A) -> Self {
        Self {
            app,
            window: None,
            viewport: Viewport::new(1, 1),
            screen_size: (1, 1),
            work_area: (0, 0, 1, 1),
            window_rect: WindowRect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
            scratch: Vec::new(),
            #[cfg(target_os = "macos")]
            renderer: None,
            #[cfg(not(target_os = "macos"))]
            surface: None,
            text: None,
            clip_mask: Vec::new(),
            cursor_x: 0.0,
            cursor_y: 0.0,
            modifiers: ModifiersState::empty(),
            is_dragging_resize: false,
            resize_edge: ResizeEdge::None,
            drag_start_size: (0, 0),
            drag_start_window_pos: (0, 0),
            drag_start_cursor_screen_pos: (0, 0),
            is_dragging_move: false,
            pointer: crate::host::pointer::PointerArbiter::new(),
            move_drag_armed: false,
            drag_move_anchor_screen: (0, 0),
            drag_move_rect_start: (0, 0),
            last_painted_rect: WindowRect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
            surface_ready: false,
            is_focused: true,
            debug_stats: crate::paint::DebugStats::default(),
            pending_damage: crate::canvas::Damage::new(),
            last_strip_active: false,
            persistent_screen: Vec::new(),
            pending_full_repaint: true,
            last_hitmask: false,
            last_alpha_mode: 0,
            last_opaque_scan: false,
            strip_buf: Vec::new(),
            last_overlay_active: Vec::new(),
            saved_rect_for_maximize: None,
            #[cfg(target_os = "macos")]
            hittest_off: false,
            #[cfg(target_os = "macos")]
            hittest_monitor: None,
        }
    }

    /// macOS click-thru: only disable hittest when the cursor is outside the window rect.
    /// Inside the window rect we always accept events — checking alpha per-pixel there is too fragile (transparent UI elements, frame transitions, etc. cause false negatives that drop clicks to the app behind us).
    #[cfg(target_os = "macos")]
    fn update_macos_hittest(&mut self) {
        let cx = self.cursor_x as i32;
        let cy = self.cursor_y as i32;
        let r = &self.window_rect;
        let inside = cx >= r.x && cx < r.x + r.w as i32
                  && cy >= r.y && cy < r.y + r.h as i32;
        // NEVER re-engage click-thru mid-drag. A resize-grow (or a move) pushes the cursor to or past the CURRENT rect edge before `apply_resize_drag` catches the rect up; if we flipped hittest off there, macOS would stop delivering the drag and the window could shrink but never grow. Hold hittest ON for the whole drag; the next cursor-move after release recomputes normally.
        let should_ignore = !inside && !self.is_dragging_resize && !self.is_dragging_move;
        if should_ignore != self.hittest_off {
            if let Some(window) = self.window.as_ref() {
                if should_ignore {
                    window.set_cursor(winit::window::CursorIcon::Default);
                }
                let _ = window.set_cursor_hittest(!should_ignore);
                self.hittest_off = should_ignore;
            }
        }
    }

    fn render_frame(&mut self) {
        if !self.surface_ready {
            return;
        }
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };
        let win_w = self.viewport.width_px as usize;
        let win_h = self.viewport.height_px as usize;
        let win_px = win_w * win_h;
        // Keep scratch + clip_mask in sync with the consumer-visible viewport (= window_rect dims). Resize-drag (later step) changes these; we re-allocate when the size shifts.
        if self.scratch.len() != win_px {
            self.scratch = vec![0u32; win_px];
        }
        if self.clip_mask.len() != win_px {
            self.clip_mask = vec![255u8; win_px];
        }

        // Reset the frame's damage accumulator before the consumer paints. Every Canvas the consumer constructs against `ctx.damage` will union into this; after `app.render` returns we have the bounding rect of everything touched this frame.
        self.pending_damage.clear();

        // Two render modes, chosen by an explicit host flag (NOT by comparing damage_clip's geometry to viewport_rect). `pending_full_repaint` is set by events that destroy the chrome perimeter + shadow band in persistent_screen — drag release, resize, zoom, focus change. Debug-toggle transitions (hitmask / alpha mode / FPS strip) also promote to a full repaint here because those flags change either finalize's branch or what's overlaid post-finalize, and need a clean window to flow thru. On those frames we wipe persistent_screen, reset overlay state, set damage_clip = viewport, and finalize copies every pixel (including AA edges); paint_shadow then casts ONCE into the freshly-zero band (and only when hitmask is off). On every other frame, damage_clip is whatever app.damage_rect returns (typically a small interior region or empty); finalize is narrowed AND skips non-opaque source pixels so the AA hairline pixels at the window perimeter stay untouched, and paint_shadow is NOT called so it never compounds.
        let hitmask_now =
            crate::paint::DEBUG_SHOW_HITMASK.load(std::sync::atomic::Ordering::Relaxed);
        let alpha_mode_now =
            crate::paint::DEBUG_SHOW_ALPHA.load(std::sync::atomic::Ordering::Relaxed);
        let opaque_scan_now =
            crate::paint::DEBUG_SHOW_OPAQUE_SCAN.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "text")]
        let strip_active = crate::paint::DEBUG_SHOW_FPS.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(feature = "text"))]
        let strip_active = false;
        if hitmask_now != self.last_hitmask
            || alpha_mode_now != self.last_alpha_mode
            || strip_active != self.last_strip_active
            || opaque_scan_now != self.last_opaque_scan
        {
            self.pending_full_repaint = true;
            self.last_hitmask = hitmask_now;
            self.last_alpha_mode = alpha_mode_now;
            self.last_strip_active = strip_active;
            self.last_opaque_scan = opaque_scan_now;
        }
        let viewport_rect = crate::canvas::PixelRect::new(0, 0, win_w, win_h);
        let full_repaint = self.pending_full_repaint;
        if full_repaint {
            self.pending_full_repaint = false;
            self.persistent_screen.fill(0);
            for a in self.last_overlay_active.iter_mut() {
                *a = false;
            }
        }
        let damage_clip = if full_repaint {
            viewport_rect
        } else {
            self.app
                .damage_rect(self.viewport)
                .unwrap_or(crate::canvas::PixelRect::empty())
        };
        // Strip is painted in a clobber pass AFTER finalize + overlay — it does NOT contribute to damage_clip and does NOT bump damage_pct.

        // Damage outline overlay (`[]w`). Sampled once here so the post-finalize stamp uses a stable value for this frame. The outline is stamped DIRECTLY into the platform back buffer between the persistent_screen copy and `present()`, so it never enters persistent_screen, never flows thru finalize, and never carries state between frames.
        let outline_active =
            crate::paint::DEBUG_SHOW_DAMAGE.load(std::sync::atomic::Ordering::Relaxed);

        clear_scratch_rect(&mut self.scratch, win_w, damage_clip);

        let Some(text) = self.text.as_mut() else {
            return;
        };

        let mut ctx = Context {
            pressed_hit: self.pointer.held_id(),
            viewport: self.viewport,
            text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &*window,
            modifiers: winit_compat::from_winit_mods(self.modifiers),
            cursor_x: self.cursor_x - self.window_rect.x as Coord,
            cursor_y: self.cursor_y - self.window_rect.y as Coord,
            is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
            damage_clip,
        };

        // Per-stage stopwatches. Each Instant brackets one pipeline stage; the strip displays each as FPS so toggling SIMD/Rayon shows which stage actually moves. `buffer.present()` is excluded everywhere because it blocks for vsync, which would pin every reading to the display refresh rate. Each `let` is uninitialized and assigned exactly once below — drops the dead `= 0.0f32` placeholder the compiler used to flag.
        let app_dt;
        let fill_dt;
        let finalize_dt;
        let shadow_dt;

        let app_start = Instant::now();
        self.app.render(&mut self.scratch, &mut ctx);
        drop(ctx);
        app_dt = app_start.elapsed().as_secs_f32();

        let scr_w = self.screen_size.0 as usize;
        let scr_h = self.screen_size.1 as usize;
        let rect_x = self.window_rect.x;
        let rect_y = self.window_rect.y;

        // Persistent screen lives across frames so the post-finalize overlay (blinkey) can mutate just a few pixels each frame without re-running finalize. Resize on screen-size change; new pixels start at 0 which is fine — finalize on the next render will populate them.
        let scr_px = scr_w * scr_h;
        let t = Instant::now();
        if self.persistent_screen.len() != scr_px {
            self.persistent_screen.resize(scr_px, 0);
        }
        fill_dt = t.elapsed().as_secs_f32();

        // Debug fade: saturating-subtract `FADE_STEP` from every persistent_screen RGB byte. Runs BEFORE finalize so pixels that finalize / overlay / strip overwrite this frame land at full brightness while pixels that nobody touches visibly decay toward black — diagnoses whether the incremental opaque-scan finalize is actually copying the regions it should. Skipped on full_repaint since persistent_screen is being wiped anyway.
        let fade_active = crate::paint::DEBUG_SHOW_FADE.load(std::sync::atomic::Ordering::Relaxed);
        if fade_active && !full_repaint {
            const FADE_STEP: u8 = 4;
            for px in self.persistent_screen.iter_mut() {
                let a = *px & 0xFF00_0000;
                let r = (((*px >> 16) & 0xFF) as u8).saturating_sub(FADE_STEP) as u32;
                let g = (((*px >> 8) & 0xFF) as u8).saturating_sub(FADE_STEP) as u32;
                let b = ((*px & 0xFF) as u8).saturating_sub(FADE_STEP) as u32;
                *px = a | (r << 16) | (g << 8) | b;
            }
        }

        // Finalize: on a full repaint we copy every pixel from scratch (AA + opaque alike). On an incremental frame we narrow to damage_clip AND skip non-opaque source pixels — the chrome perimeter AA pixels in persistent_screen already carry their finalized RGB and the shadow boost from the last full repaint, and overwriting them would (a) drop the shadow integration and (b) require re-running paint_shadow. The opaque-only path uses left/right scans per row to find the bounded copy range, then does a contiguous finalize on that range (no per-pixel if-gating).
        let t = Instant::now();
        if !damage_clip.is_empty() {
            crate::paint::finalize_into_screen(
                &self.scratch,
                &self.clip_mask,
                win_w,
                win_h,
                &mut self.persistent_screen,
                scr_w,
                rect_x,
                rect_y,
                damage_clip,
                full_repaint,
            );
        }
        finalize_dt = t.elapsed().as_secs_f32();

        // Drop shadow runs ONCE per full repaint, into a known-cleared band (persistent_screen.fill(0) above). Never runs on incremental frames — the perimeter AA pixels with their shadow contribution were preserved by the opaque-only finalize, and the shadow band cells outside the window were not touched either, so the shadow visible from the last full repaint is still correct. Skipped when hitmask debug is on so the band doesn't disturb the raw hit-id view at the chrome edge, and skipped when maximized because there's nothing outside the window to cast onto — the OS surface already covers the screen.
        let t = Instant::now();
        if full_repaint && !hitmask_now && self.saved_rect_for_maximize.is_none() {
            let span = self.viewport.effective_span();
            let target_radius = (span / 16.0).max(8.0);
            let drop = (1240.0 / target_radius) as u32;
            let factor_256 = (256u32.saturating_sub(drop)).clamp(96, 254);
            let shadow_seed: u32 = if self.is_focused { 0x80 } else { 0x40 };
            let rect_for_shadow = (
                self.window_rect.x,
                self.window_rect.y,
                self.window_rect.w as i32,
                self.window_rect.h as i32,
            );
            crate::paint::paint_shadow(
                &mut self.persistent_screen,
                scr_w,
                factor_256,
                shadow_seed,
                rect_for_shadow,
            );
        }
        shadow_dt = t.elapsed().as_secs_f32();

        // Post-finalize, post-shadow overlay pass. For each pixel whose hit id is currently tinted OR was tinted last frame, copy the scratch pixel → XOR to visible → optionally wrap-sub the per-id delta → write to persistent_screen. Restores the scratch baseline on unhover and applies the tint on hover — no diff math, no accumulation, just "copy and conditionally adjust." Runs every frame regardless of damage_clip so hover tints follow the cursor even when nothing else dirtied scratch.
        //
        // Order matters: [`FluorApp::overlay_deltas`] takes `&mut self` (so the app can walk its widget tree), so we build the table first and release the borrow before grabbing the shared `hit_test_map` borrow used by `apply_overlay`.
        let current = self.app.overlay_deltas();
        // Parallel bbox table so the overlay scan is bounded to each hovered widget's rect, not the whole window.
        // Built before the hit_test_map borrow (both take &mut / &self respectively).
        let bboxes = self.app.overlay_bboxes(win_w, win_h);
        if let Some((map, hw, hh)) = self.app.hit_test_map() {
            // Match last_overlay_active length to deltas length. Grow with `false` if the app registered new IDs since last frame; shrink if it (rare) collapsed. apply_overlay debug-asserts equal lengths.
            if self.last_overlay_active.len() != current.len() {
                self.last_overlay_active.resize(current.len(), false);
            }
            crate::paint::apply_overlay(
                &self.scratch,
                &mut self.persistent_screen,
                scr_w,
                self.window_rect.x,
                self.window_rect.y,
                map,
                hw,
                hh,
                &current,
                &bboxes,
                &mut self.last_overlay_active,
            );
        }

        // FPS strip: drawn LAST, clobber-style, into a DEDICATED staging buffer (`self.strip_buf`) — never touches the app's scratch or clip_mask. After rasterizing α + darkness into the staging buffer we XOR → visible RGB, force α=0xFF, and clobber-write into persistent_screen at the strip rect. The whole pass is bracketed by a snapshot+restore of `RASTERIZE_OPS` so the strip's text/rect rasterizers don't bump the R counter or pollute `damage_pct`. Does NOT contribute to `damage_clip`, does NOT trigger paint_shadow. Toggle on/off promotes to full repaint via the transition detector above so the strip-rect underlying pixels get correctly restored when it disappears.
        #[cfg(feature = "text")]
        if strip_active {
            let strip_h = crate::paint::DEBUG_STRIP_H;
            // Centre the strip in the bottom 1/12th of the window (mirrors the old computation; flush-bottom would collide with the squircle corner cutouts).
            let band_top = (win_h * 11) / 12;
            let strip_y_in_window = band_top + ((win_h - band_top).saturating_sub(strip_h)) / 2;

            let strip_px = win_w.saturating_mul(strip_h);
            if self.strip_buf.len() != strip_px {
                self.strip_buf = vec![0u32; strip_px];
            } else {
                self.strip_buf.fill(0);
            }
            let saved_ops = crate::paint::RASTERIZE_OPS.load(std::sync::atomic::Ordering::Relaxed);
            if let Some(text) = self.text.as_mut() {
                let mut strip_damage = crate::canvas::Damage::new();
                let mut canvas = crate::canvas::Canvas::new(
                    &mut self.strip_buf,
                    win_w,
                    strip_h,
                    &mut strip_damage,
                );
                crate::paint::draw_debug_strip(&mut canvas, text, &self.debug_stats, 0);
            }
            crate::paint::RASTERIZE_OPS.store(saved_ops, std::sync::atomic::Ordering::Relaxed);

            // Clobber `strip_buf` rows into persistent_screen at `(window.x, window.y + strip_y_in_window)`. Per-pixel: XOR α + darkness → visible RGB, force α=0xFF.
            let rect_x = self.window_rect.x;
            let rect_y_top = self.window_rect.y + strip_y_in_window as i32;
            for y in 0..strip_h {
                let scr_y = rect_y_top + y as i32;
                if scr_y < 0 || (scr_y as usize) >= scr_h {
                    continue;
                }
                let scr_y = scr_y as usize;
                let sb_row = y * win_w;
                let ps_row = scr_y * scr_w;
                for x in 0..win_w {
                    let scr_x = rect_x + x as i32;
                    if scr_x < 0 || (scr_x as usize) >= scr_w {
                        continue;
                    }
                    let scr_x = scr_x as usize;
                    let v = self.strip_buf[sb_row + x] ^ 0x00FF_FFFF;
                    self.persistent_screen[ps_row + scr_x] = 0xFF00_0000 | (v & 0x00FF_FFFF);
                }
            }
        }

        // Copy persistent_screen → platform back buffer (whichever softbuffer/wgpu hands us this frame; it may be stale or rotated, but we always overwrite the whole thing from our owned persistent_screen). The damage outline overlay is stamped AFTER this copy and BEFORE present so it lives for exactly one frame and never touches persistent_screen.
        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            let mut buffer = renderer.lock_buffer();
            buffer.copy_from_slice(&self.persistent_screen);
            if outline_active && !damage_clip.is_empty() {
                crate::paint::stamp_damage_outline_visible(
                    &mut buffer,
                    scr_w,
                    scr_h,
                    damage_clip,
                    rect_x,
                    rect_y,
                );
            }
            let _ = buffer.present();
            // Update the global mouse monitor's window rect for re-entry detection.
            if let Some(ref monitor) = self.hittest_monitor {
                let r = &self.window_rect;
                monitor.update_rect(r.x, r.y, r.w, r.h);
            }
        }
        // Windows: present the owned screen buffer thru the layered window (per-pixel alpha + click-thru on α=0). The damage outline (a dev overlay) is stamped into a scratch copy first so it lives one frame and never touches persistent_screen, matching the softbuffer path.
        #[cfg(target_os = "windows")]
        {
            let (sw, sh) = self.screen_size;
            if let Some(window) = self.window.as_ref() {
                if outline_active && !damage_clip.is_empty() {
                    let mut scratch_screen = self.persistent_screen.clone();
                    crate::paint::stamp_damage_outline_visible(
                        &mut scratch_screen,
                        scr_w,
                        scr_h,
                        damage_clip,
                        rect_x,
                        rect_y,
                    );
                    super::windows_layered::present(window, &scratch_screen, sw, sh);
                } else {
                    super::windows_layered::present(window, &self.persistent_screen, sw, sh);
                }
            }
        }
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            let Some(surface) = self.surface.as_mut() else {
                return;
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            buffer.copy_from_slice(&self.persistent_screen);
            if outline_active && !damage_clip.is_empty() {
                crate::paint::stamp_damage_outline_visible(
                    &mut buffer,
                    scr_w,
                    scr_h,
                    damage_clip,
                    rect_x,
                    rect_y,
                );
            }
            buffer.present().expect("softbuffer buffer.present");
        }
        // Record what we just painted so the next drag-tick can compute its delta.
        self.last_painted_rect = self.window_rect;

        // Differential stats: F (frame) bumps every present; R (rasterize) only when a primitive actually did geometric paint work this frame (via the RASTERIZE_OPS atomic). On hover-only updates the atomic stays at 0 and R sticks. `damage_pct` reflects how much of the viewport this frame actually touched — drops to a small fraction on bbox-only updates.
        let ras_ops = crate::paint::RASTERIZE_OPS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let viewport_area = (win_w * win_h) as f32;
        let damage_area = (damage_clip.width() * damage_clip.height()) as f32;
        let damage_pct = if viewport_area > 0.0 {
            damage_area / viewport_area
        } else {
            0.0
        };
        if ras_ops > 0 {
            self.debug_stats
                .record_rasterize(app_dt, fill_dt, finalize_dt, shadow_dt, damage_pct);
        }
        self.debug_stats.record_present(damage_pct);
    }

    /// Drag-tick fast path: shift the screen buffer in place by the delta since the last paint, push the input region update, and present. Skips consumer render, scratch fill, finalize, and shadow rasterization entirely — the existing chrome pixels just slide thru the screen buffer, with anything that falls off any edge wrapping to the opposite side. On drag release, a normal `render_frame` overwrites the wrap artefacts in one clean frame.
    fn apply_move_drag_shift(&mut self) {
        let dx = self.window_rect.x - self.last_painted_rect.x;
        let dy = self.window_rect.y - self.last_painted_rect.y;
        if dx == 0 && dy == 0 {
            return;
        }
        let scr_w = self.screen_size.0 as usize;
        let scr_h = self.screen_size.1 as usize;
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };
        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = self.renderer.as_mut() else {
                return;
            };
            let mut buffer = renderer.lock_buffer();
            crate::paint::shift_screen_wrap(&mut buffer, scr_w, scr_h, dx, dy);
            let _ = buffer.present();
        }
        // Windows: no softbuffer surface — shift our owned persistent_screen and re-present it thru the layered window. (The layered window already moves with window_rect via the α channel, so there's no OS input-region call to push like X11 does below.)
        #[cfg(target_os = "windows")]
        {
            crate::paint::shift_screen_wrap(&mut self.persistent_screen, scr_w, scr_h, dx, dy);
            let (sw, sh) = self.screen_size;
            super::windows_layered::present(&window, &self.persistent_screen, sw, sh);
        }
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            let Some(surface) = self.surface.as_mut() else {
                return;
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            crate::paint::shift_screen_wrap(&mut buffer, scr_w, scr_h, dx, dy);
            buffer.present().expect("softbuffer buffer.present");
        }
        #[cfg(target_os = "linux")]
        x11_atomic::set_input_region(
            &window,
            self.window_rect.x,
            self.window_rect.y,
            self.window_rect.w,
            self.window_rect.h,
        );
        self.last_painted_rect = self.window_rect;
    }

    /// Apply an [`EventResponse`] returned from `app.on_event`. Returns `true` if the response was `Close` (caller should terminate).
    fn apply_response(&mut self, response: EventResponse) -> bool {
        let Some(window) = self.window.as_ref().cloned() else {
            return false;
        };
        match response {
            EventResponse::Handled | EventResponse::Pass => false,
            EventResponse::StartWindowDrag => {
                // Fullscreen-compositor model: OS window.drag_window() would do nothing (OS window is fullscreen). Drag is internal — capture the anchor here and move window_rect on CursorMoved. We ARM the drag without committing; the first CursorMoved commits it (no dead zone). Click-without-motion never commits, so no wrap-shift fast path runs, no persistent_screen wrap artefacts, and the textbox's small `glow_bbox` damage_rect drives the only repaint.
                //
                // Maximized state suppresses drag entirely. Most WMs handle this with "drag a maximized window → unmaximize and resume drag at cursor"; that's the right ergonomic but more involved (need to compute the unmaximized origin relative to cursor, then begin the drag). For v0 the simpler rule is "ignore the drag request" — title-bar clicks while maximized do nothing instead of producing nonsense (the drag would translate the fullscreen-sized rect into negative offsets and clip_through the input region). Revisit when we add the unmaximize-then-drag flow.
                if self.saved_rect_for_maximize.is_none() {
                    self.move_drag_armed = true;
                    self.drag_move_anchor_screen = (self.cursor_x as i32, self.cursor_y as i32);
                    self.drag_move_rect_start = (self.window_rect.x, self.window_rect.y);
                }
                false
            }
            EventResponse::StartResize(edge) => {
                self.start_resize(edge, &window);
                false
            }
            EventResponse::Close => {
                std::process::exit(0);
            }
            EventResponse::ToggleMaximized => {
                self.toggle_maximized(&window);
                false
            }
            EventResponse::Minimize => {
                window.set_minimized(true);
                false
            }
        }
    }

    /// Flip `window_rect` between the user-sized rect (saved in `saved_rect_for_maximize`) and the full screen. Mirrors the geometry-change tail of `resize_drag_update`: resize scratch + clip_mask, reflow viewport, notify the consumer via `on_resize`, mark full-repaint, and update the X11 input region. No-op if `screen_size` is still the (1,1) placeholder — first `Resized` event hasn't landed yet, no real geometry to swap to.
    fn toggle_maximized(&mut self, window: &Window) {
        let (scr_w, scr_h) = self.screen_size;
        if scr_w <= 1 || scr_h <= 1 {
            return;
        }
        let new_rect = match self.saved_rect_for_maximize.take() {
            Some(prev) => prev,
            None => {
                self.saved_rect_for_maximize = Some(self.window_rect);
                // Maximize to the work area (monitor minus panels), not the raw screen, so
                // the maximized window's bottom chrome stays clear of the taskbar. Falls
                // back to full screen if the work area was never resolved.
                let (wx, wy, ww, wh) = self.work_area;
                if ww > 1 && wh > 1 {
                    WindowRect { x: wx, y: wy, w: ww, h: wh }
                } else {
                    WindowRect { x: 0, y: 0, w: scr_w, h: scr_h }
                }
            }
        };

        if new_rect.w == self.window_rect.w
            && new_rect.h == self.window_rect.h
            && new_rect.x == self.window_rect.x
            && new_rect.y == self.window_rect.y
        {
            return;
        }

        let size_changed = new_rect.w != self.window_rect.w || new_rect.h != self.window_rect.h;
        self.window_rect = new_rect;

        if size_changed {
            self.viewport = Viewport::new(new_rect.w, new_rect.h).with_ru(self.viewport.ru);
            let win_px = (new_rect.w as usize) * (new_rect.h as usize);
            self.scratch = vec![0u32; win_px];
            self.clip_mask = vec![255u8; win_px];
            self.pending_full_repaint = true;

            if let Some(text) = self.text.as_mut() {
                let mut ctx = Context {
                    pressed_hit: self.pointer.held_id(),
                    viewport: self.viewport,
                    text,
                    clip_mask: &mut self.clip_mask,
                    damage: &mut self.pending_damage,
                    window,
                    modifiers: winit_compat::from_winit_mods(self.modifiers),
                    cursor_x: self.cursor_x - self.window_rect.x as Coord,
                    cursor_y: self.cursor_y - self.window_rect.y as Coord,
                    is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                    damage_clip: crate::canvas::PixelRect::new(
                        0,
                        0,
                        self.viewport.width_px as usize,
                        self.viewport.height_px as usize,
                    ),
                };
                self.app.on_resize(new_rect.w, new_rect.h, &mut ctx);
            }
        } else {
            // Position-only change still needs a full repaint — the old window_rect's pixels in persistent_screen are stale.
            self.pending_full_repaint = true;
        }

        #[cfg(target_os = "linux")]
        x11_atomic::set_input_region(window, new_rect.x, new_rect.y, new_rect.w, new_rect.h);

        window.request_redraw();
    }

    /// Begin a self-driven resize drag. In the fullscreen-compositor model we resize `window_rect` inside our own screen buffer instead of asking the OS to resize the OS window (which is fullscreen). Captures the start geometry (window_rect size + position) and the screen-space cursor anchor; subsequent cursor moves compute the new (w, h, x, y) by delta from these starting values.
    fn start_resize(&mut self, edge: ResizeEdge, _window: &Window) {
        self.is_dragging_resize = true;
        self.resize_edge = edge;
        self.drag_start_size = (self.window_rect.w, self.window_rect.h);
        self.drag_start_window_pos = (self.window_rect.x, self.window_rect.y);
        // cursor_x/y are screen-space (raw from winit, OS window = screen in fullscreen) so no translation needed for the anchor.
        self.drag_start_cursor_screen_pos = (self.cursor_x as i32, self.cursor_y as i32);
    }

    /// Apply one tick of the self-driven resize drag — in-buffer. Called from `RedrawRequested` when `is_dragging_resize` (throttled to vsync). Updates `window_rect` directly (no OS round-trip — the OS window is fullscreen and request_inner_size / set_outer_position are no-ops). When the size changed, resizes `scratch` + `clip_mask` to the new dimensions and calls the consumer's `on_resize` so they can reflow. Always pushes a new XShape input region so click-thru follows the visible window. The subsequent `render_frame` paints at the new geometry into the screen buffer.
    fn apply_resize_drag(&mut self) {
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };

        // Screen-relative cursor delta from the drag-start anchor. cursor_x/y is already screen-space (raw winit / OS = screen in fullscreen) so no per-frame translation needed.
        let dx = (self.cursor_x as i32 - self.drag_start_cursor_screen_pos.0) as Coord;
        let dy = (self.cursor_y as i32 - self.drag_start_cursor_screen_pos.1) as Coord;

        // Min size keeps the squircle math from degenerating. 128 px matches the pre-pivot limit.
        let min_size: Coord = 128.0;

        let (new_w, new_h, new_x, new_y) = match self.resize_edge {
            ResizeEdge::Right => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                (
                    w,
                    self.drag_start_size.1,
                    self.drag_start_window_pos.0,
                    self.drag_start_window_pos.1,
                )
            }
            ResizeEdge::Left => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let dw = self.drag_start_size.0 as i32 - w as i32;
                (
                    w,
                    self.drag_start_size.1,
                    self.drag_start_window_pos.0 + dw,
                    self.drag_start_window_pos.1,
                )
            }
            ResizeEdge::Bottom => {
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                (
                    self.drag_start_size.0,
                    h,
                    self.drag_start_window_pos.0,
                    self.drag_start_window_pos.1,
                )
            }
            ResizeEdge::Top => {
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let dh = self.drag_start_size.1 as i32 - h as i32;
                (
                    self.drag_start_size.0,
                    h,
                    self.drag_start_window_pos.0,
                    self.drag_start_window_pos.1 + dh,
                )
            }
            ResizeEdge::TopRight => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let dh = self.drag_start_size.1 as i32 - h as i32;
                (
                    w,
                    h,
                    self.drag_start_window_pos.0,
                    self.drag_start_window_pos.1 + dh,
                )
            }
            ResizeEdge::TopLeft => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord - dy).max(min_size) as u32;
                let dw = self.drag_start_size.0 as i32 - w as i32;
                let dh = self.drag_start_size.1 as i32 - h as i32;
                (
                    w,
                    h,
                    self.drag_start_window_pos.0 + dw,
                    self.drag_start_window_pos.1 + dh,
                )
            }
            ResizeEdge::BottomRight => {
                let w = (self.drag_start_size.0 as Coord + dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                (
                    w,
                    h,
                    self.drag_start_window_pos.0,
                    self.drag_start_window_pos.1,
                )
            }
            ResizeEdge::BottomLeft => {
                let w = (self.drag_start_size.0 as Coord - dx).max(min_size) as u32;
                let h = (self.drag_start_size.1 as Coord + dy).max(min_size) as u32;
                let dw = self.drag_start_size.0 as i32 - w as i32;
                (
                    w,
                    h,
                    self.drag_start_window_pos.0 + dw,
                    self.drag_start_window_pos.1,
                )
            }
            ResizeEdge::None => return,
        };

        let size_changed = new_w != self.window_rect.w || new_h != self.window_rect.h;
        let pos_changed = new_x != self.window_rect.x || new_y != self.window_rect.y;
        if !size_changed && !pos_changed {
            return;
        }

        // Manual resize invalidates any saved-for-maximize rect: the user has picked a new "natural" size and that's what the next un-maximize should restore to. Clearing here means the next ToggleMaximized will save the post-resize rect, not the pre-toggle one.
        if size_changed {
            self.saved_rect_for_maximize = None;
        }

        self.window_rect = WindowRect {
            x: new_x,
            y: new_y,
            w: new_w,
            h: new_h,
        };

        if size_changed {
            // Carry the user's zoom (ru) across the resize so Ctrl+/Ctrl-/Ctrl+scroll state survives.
            self.viewport = Viewport::new(new_w, new_h).with_ru(self.viewport.ru);
            let win_px = (new_w as usize) * (new_h as usize);
            self.scratch = vec![0u32; win_px];
            self.clip_mask = vec![255u8; win_px];
            // Window dims changed → perimeter, AA edges, shadow rays all need a fresh single-pass repaint.
            self.pending_full_repaint = true;

            // Let the consumer reflow — they may relayout panes, recompute glyph metrics, etc.
            if let Some(text) = self.text.as_mut() {
                let mut ctx = Context {
                    pressed_hit: self.pointer.held_id(),
                    viewport: self.viewport,
                    text,
                    clip_mask: &mut self.clip_mask,
                    damage: &mut self.pending_damage,
                    window: &*window,
                    modifiers: winit_compat::from_winit_mods(self.modifiers),
                    cursor_x: self.cursor_x - self.window_rect.x as Coord,
                    cursor_y: self.cursor_y - self.window_rect.y as Coord,
                    is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                    damage_clip: crate::canvas::PixelRect::new(
                        0,
                        0,
                        self.viewport.width_px as usize,
                        self.viewport.height_px as usize,
                    ),
                };
                self.app.on_resize(new_w, new_h, &mut ctx);
            }
        }

        // Update click-thru region so the OS routes clicks based on the new rect.
        #[cfg(target_os = "linux")]
        x11_atomic::set_input_region(&window, new_x, new_y, new_w, new_h);
    }
}

#[cfg(feature = "host-winit")]
impl<A: FluorApp + 'static> ApplicationHandler<A::UserEvent> for DesktopShell<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Probe the monitor BEFORE creating the window so we can request an OS surface of exactly the right size + position. We deliberately avoid `with_fullscreen` — it triggers the WM's animated transition (default-window-size → grow → fullscreen) which makes the chrome appear to scale up from the top-left. Instead, ask for a plain borderless transparent window covering the monitor: the WM creates it at the requested geometry directly, no animation.
        let (mon_w, mon_h) = event_loop
            .primary_monitor()
            .or_else(|| event_loop.available_monitors().next())
            .map(|m| (m.size().width.max(1), m.size().height.max(1)))
            .unwrap_or((1920, 1080));
        self.screen_size = (mon_w, mon_h);

        let attrs = WindowAttributes::default()
            .with_title(self.app.title())
            .with_inner_size(winit::dpi::PhysicalSize::new(mon_w, mon_h))
            .with_position(winit::dpi::PhysicalPosition::new(0i32, 0i32))
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(false);
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));

        // Match the OS window icon (taskbar / alt-tab / title bar) to the app's orb. winit
        // honours this on Windows + X11; it's a no-op on Wayland (icon from .desktop app_id)
        // and macOS (icon from the .app bundle), which source the icon at packaging time.
        if let Some(icon) = self.app.window_icon() {
            if let Some(winit_icon) = icon.to_winit_icon() {
                window.set_window_icon(Some(winit_icon));
            }
        }

        #[cfg(target_os = "macos")]
        {
            use winit::platform::macos::WindowExtMacOS;
            window.set_has_shadow(false);
            self.hittest_monitor =
                super::macos_hittest::HittestMonitor::install(mon_h);
        }

        // Windows: make the OS window LAYERED so UpdateLayeredWindow can present per-pixel alpha (and route clicks thru the α=0 region). winit's `with_transparent(true)` alone gives an opaque softbuffer surface on Windows — the layered style is what the fullscreen compositor needs.
        #[cfg(target_os = "windows")]
        super::windows_layered::make_layered(&window);

        // Desktop work area (monitor minus panels/taskbars/menu-bar+Dock) so the visible
        // window — and especially its bottom chrome status band — doesn't end up under a
        // taskbar. Per-OS query (X11 `_NET_WORKAREA`, Windows `SPI_GETWORKAREA`, macOS
        // `visibleFrame`); falls back to the full monitor on Wayland and where unavailable.
        let (wa_x, wa_y, wa_w, wa_h) = monitor_work_area(mon_w, mon_h);
        self.work_area = (wa_x, wa_y, wa_w, wa_h);

        // Initial visible-window size: app-supplied (defaults to half the screen in each axis), clamped to the work area and centred within it. Apps with aspect-ratio opinions override [`FluorApp::initial_size`].
        let (req_w, req_h) = self.app.initial_size((wa_w, wa_h));
        let initial_w = req_w.max(1).min(wa_w);
        let initial_h = req_h.max(1).min(wa_h);
        let win_x = wa_x + ((wa_w as i32) - (initial_w as i32)) / 2;
        let win_y = wa_y + ((wa_h as i32) - (initial_h as i32)) / 2;
        self.window_rect = WindowRect {
            x: win_x,
            y: win_y,
            w: initial_w,
            h: initial_h,
        };
        self.viewport = Viewport::new(initial_w, initial_h);

        #[cfg(target_os = "macos")]
        {
            self.renderer = Some(super::renderer_wgpu::Renderer::new(
                &window,
                self.screen_size.0,
                self.screen_size.1,
            ));
        }
        // Windows presents via UpdateLayeredWindow from `persistent_screen` directly (softbuffer's BitBlt present is opaque), so it needs no softbuffer surface. Every other non-macOS target (Linux/X11, Redox/Orbital) uses softbuffer.
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            use std::num::NonZeroU32;
            let context =
                softbuffer::Context::new(window.clone()).expect("softbuffer Context::new");
            let mut surface = softbuffer::Surface::new(&context, window.clone())
                .expect("softbuffer Surface::new");
            surface
                .resize(
                    NonZeroU32::new(self.screen_size.0).expect("nonzero screen width"),
                    NonZeroU32::new(self.screen_size.1).expect("nonzero screen height"),
                )
                .expect("softbuffer Surface::resize");
            self.surface = Some(surface);
        }

        if self.text.is_none() {
            self.text = Some(TextRenderer::new());
        }

        // Scratch + clip_mask are sized to the visible window (= viewport), NOT the screen. The host blits scratch into the screen buffer at window_rect offset; pixels outside the window stay at the screen buffer's α=0 init.
        let win_px = (self.window_rect.w as usize) * (self.window_rect.h as usize);
        self.scratch = vec![0u32; win_px];
        self.clip_mask = vec![255u8; win_px];

        // Hand control to the consumer's init.
        {
            let text = self.text.as_mut().expect("text renderer initialized");
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                damage_clip: crate::canvas::PixelRect::new(
                    0,
                    0,
                    self.viewport.width_px as usize,
                    self.viewport.height_px as usize,
                ),
            };
            self.app.init(&mut ctx);
        }

        self.window = Some(window.clone());
        // Surface is created at the requested monitor size — we can paint immediately. The Resized handler still flips this flag if it sees a different first size, but with the non-fullscreen approach we expect the surface to come up at the right size on the first frame.
        self.surface_ready = true;

        // Click-thru: tell X11 our hittable area is just `window_rect`. Clicks outside the rect pass thru to whatever app is beneath us. Drag-to-move + resize-drag steps will re-call this on every rect change. No-op on non-X11 platforms; macOS/Windows passthrough handling lands in their own backend modules later.
        #[cfg(target_os = "linux")]
        x11_atomic::set_input_region(
            &window,
            self.window_rect.x,
            self.window_rect.y,
            self.window_rect.w,
            self.window_rect.h,
        );

        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let needs_redraw = if let (Some(window), Some(text)) =
            (self.window.as_ref().cloned(), self.text.as_mut())
        {
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                damage_clip: crate::canvas::PixelRect::new(
                    0,
                    0,
                    self.viewport.width_px as usize,
                    self.viewport.height_px as usize,
                ),
            };
            self.app.tick(&mut ctx)
        } else {
            false
        };
        if needs_redraw {
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }

        if let Some(when) = self.app.wake_at() {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(when));
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match &event {
            WindowEvent::CloseRequested => {
                std::process::exit(0);
            }
            WindowEvent::Resized(size) => {
                // In the fullscreen-compositor architecture, the OS surface is the whole screen — `size` is the SCREEN size, not the consumer-visible viewport. WMs commonly fire Resized multiple times during fullscreen activation (default-window-size → animating → final fullscreen); each tick we resize the surface to match, re-centre the visible window inside the new bounds, and re-issue the input region. Suppresses the "chrome appears in the top-left of a growing window" artefact during WM fullscreen animations.
                if size.width == 0 || size.height == 0 {
                    return;
                }
                if size.width == self.screen_size.0
                    && size.height == self.screen_size.1
                    && self.surface_ready
                {
                    return;
                }

                self.screen_size = (size.width, size.height);

                #[cfg(target_os = "macos")]
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
                #[cfg(not(target_os = "macos"))]
                if let Some(surface) = self.surface.as_mut() {
                    use std::num::NonZeroU32;
                    if let (Some(w), Some(h)) =
                        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
                    {
                        surface.resize(w, h).expect("softbuffer Surface::resize");
                    }
                }

                // Re-centre + clamp window_rect to the current screen on every screen-size change (initial fullscreen, monitor switch, etc.). Skip during an active drag — the user is steering the rect themselves.
                //
                // SIZE comes from the app: on the FIRST real surface (before surface_ready) we (re)apply `FluorApp::initial_size` now that the true screen size is known — `resumed` set it against the monitor we *expected*, and Windows in particular reports a different size here (DPI virtualization), so deriving it again keeps the app's aspect (e.g. Photon's tall portrait window) instead of the old hardcoded screen/2 that made the window "supa fat". On LATER resizes we PRESERVE the current window size (the user may have resized it) and only re-centre + clamp.
                if !self.is_dragging_resize && !self.is_dragging_move {
                    let (new_w, new_h) = if !self.surface_ready {
                        let (rw, rh) = self.app.initial_size((size.width, size.height));
                        (rw.max(1).min(size.width), rh.max(1).min(size.height))
                    } else {
                        (
                            self.window_rect.w.max(1).min(size.width),
                            self.window_rect.h.max(1).min(size.height),
                        )
                    };
                    let new_x = ((size.width as i32) - (new_w as i32)) / 2;
                    let new_y = ((size.height as i32) - (new_h as i32)) / 2;
                    let rect_changed = new_w != self.window_rect.w
                        || new_h != self.window_rect.h
                        || new_x != self.window_rect.x
                        || new_y != self.window_rect.y;
                    self.window_rect = WindowRect {
                        x: new_x,
                        y: new_y,
                        w: new_w,
                        h: new_h,
                    };
                    if rect_changed {
                        self.viewport = Viewport::new(new_w, new_h).with_ru(self.viewport.ru);
                        let win_px = (new_w as usize) * (new_h as usize);
                        self.scratch = vec![0u32; win_px];
                        self.clip_mask = vec![255u8; win_px];
                        // Surface-driven resize → window geometry changed → full repaint required.
                        self.pending_full_repaint = true;
                        if let (Some(window), Some(text)) =
                            (self.window.as_ref().cloned(), self.text.as_mut())
                        {
                            let mut ctx = Context {
                                pressed_hit: self.pointer.held_id(),
                                viewport: self.viewport,
                                text,
                                clip_mask: &mut self.clip_mask,
                                damage: &mut self.pending_damage,
                                window: &*window,
                                modifiers: winit_compat::from_winit_mods(self.modifiers),
                                cursor_x: self.cursor_x - new_x as Coord,
                                cursor_y: self.cursor_y - new_y as Coord,
                                damage_clip: crate::canvas::PixelRect::new(
                                    0,
                                    0,
                                    self.viewport.width_px as usize,
                                    self.viewport.height_px as usize,
                                ),
                                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                            };
                            self.app.on_resize(new_w, new_h, &mut ctx);
                        }
                        #[cfg(target_os = "linux")]
                        if let Some(window) = self.window.as_ref() {
                            x11_atomic::set_input_region(window, new_x, new_y, new_w, new_h);
                        }
                    }
                }

                // First Resized confirms the OS surface is actually allocated — safe to start painting.
                self.surface_ready = true;
                self.render_frame();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_x = position.x as Coord;
                self.cursor_y = position.y as Coord;

                #[cfg(target_os = "macos")]
                {
                    self.update_macos_hittest();
                    if self.hittest_off {
                        return;
                    }
                }

                // During a self-driven resize drag, CursorMoved fires at hundreds of Hz (raw input rate) AND we synthesize more via set_outer_position (window-relative cursor pos changes when the window moves). Doing a full resize+paint+OS-update per event floods X11 (`XIO: fatal IO error 11`) and creates a multi-second backlog of stale requests that play back after release. Coalesce: just stash the new cursor pos and request a redraw — winit caps RedrawRequested to vsync (~60-144 Hz), and the actual drag tick runs there. Skips consumer event dispatch too (consumer doesn't need to see resize-drag cursor moves).
                if self.is_dragging_resize {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                }

                // In-buffer drag-to-move: update window_rect.x/y by the cursor delta from the drag anchor. The actual screen-buffer shift + input-region update + present happens at vsync in `apply_move_drag_shift` (called from RedrawRequested), naturally coalescing the 200+ Hz raw input rate down to the display refresh rate. Skip consumer dispatch — they don't need cursor moves during the drag. No dead zone: the drag commits on the first cursor move after the press — 1:1 tracking from the first pixel (the old 4px threshold was a feel papercut; a click-without-motion still never commits because this arm only runs on CursorMoved).
                if self.move_drag_armed {
                    let dx = (self.cursor_x as i32) - self.drag_move_anchor_screen.0;
                    let dy = (self.cursor_y as i32) - self.drag_move_anchor_screen.1;
                    if !self.is_dragging_move {
                        self.is_dragging_move = true;
                        if let Some(window) = self.window.as_ref() {
                            window.set_cursor(winit::window::CursorIcon::Grabbing);
                        }
                    }
                    self.window_rect.x = self.drag_move_rect_start.0 + dx;
                    self.window_rect.y = self.drag_move_rect_start.1 + dy;
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                    return;
                }

                // Press-hold-release: while a press is in flight, track whether the pointer is still over the armed target. A drag off (or back on) toggles the held colour — request a redraw so it appears/clears. Runs before the app dispatch so ctx.pressed_hit below reflects this move.
                if self.pointer.on_move(self.hit_under_cursor()) {
                    if let Some(window) = self.window.as_ref() {
                        window.request_redraw();
                    }
                }

                if let (Some(window), Some(text)) =
                    (self.window.as_ref().cloned(), self.text.as_mut())
                {
                    let mut ctx = Context {
                        pressed_hit: self.pointer.held_id(),
                        viewport: self.viewport,
                        text,
                        clip_mask: &mut self.clip_mask,
                        damage: &mut self.pending_damage,
                        window: &*window,
                        modifiers: winit_compat::from_winit_mods(self.modifiers),
                        cursor_x: self.cursor_x - self.window_rect.x as Coord,
                        cursor_y: self.cursor_y - self.window_rect.y as Coord,
                        is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                        damage_clip: crate::canvas::PixelRect::new(
                            0,
                            0,
                            self.viewport.width_px as usize,
                            self.viewport.height_px as usize,
                        ),
                    };
                    // Translate winit → fluor at the boundary. Events that don't map (decorator/raw-input/etc.) skip app.on_event entirely; the host continues handling them internally below as needed.
                    let response = match winit_compat::from_winit_event(&event) {
                        Some(fevent) => self.app.on_event(&fevent, &mut ctx),
                        None => EventResponse::Pass,
                    };
                    // Cursor coords must be window-relative — same translation as Context's cursor_x/y — so the consumer's hit_at sees the chrome at origin (0,0). Raw screen-space coords would miss every button when the window_rect isn't at (0,0).
                    let icon = self.app.cursor_for(ctx.cursor_x, ctx.cursor_y, &ctx);
                    drop(ctx);
                    window.set_cursor(winit_compat::to_winit_cursor(icon));
                    self.apply_response(response);
                }
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
                self.dispatch_event(event);
            }
            WindowEvent::KeyboardInput {
                event: key_event, ..
            } if (self.modifiers.control_key() || self.modifiers.super_key())
                && key_event.state == ElementState::Pressed =>
            {
                // Ctrl/Cmd + =/+/-/0 → zoom. Match `logical_key.to_text()` (the produced character) rather than `physical_key` so non-US layouts (Colemak/Dvorak/etc.) work — the user pressing the key labelled `=` should zoom in regardless of which physical-position that key occupies. `+` covers Shift+= and the numpad `+`; `=` covers the plain key on US. `-` covers minus and the numpad `-`. `0` covers digit and numpad 0.
                if let Some(text) = key_event.logical_key.to_text() {
                    match text {
                        "=" | "+" => {
                            self.apply_zoom_change(Some(1.0));
                            return;
                        }
                        "-" => {
                            self.apply_zoom_change(Some(-1.0));
                            return;
                        }
                        "0" => {
                            self.apply_zoom_change(None);
                            return;
                        }
                        _ => {}
                    }
                }
                self.dispatch_event(event);
            }
            WindowEvent::MouseWheel { delta, .. }
                if self.modifiers.control_key() || self.modifiers.super_key() =>
            {
                // Ctrl/Cmd + scroll → zoom. 1 step per scroll notch (LineDelta). Trackpad PixelDelta accumulates many small events; a step's worth of travel is span/(1<<6) — ≈21 px on a 1920×1080 window (the legacy photon "20 px" notch feel), derived from the display instead of hardcoded (no fixed pixels). Bare span, not effective_span: feed sensitivity must not compound with the ru being adjusted. Direction-independent — the dense-reachability design lives in `zoom_step_factor`'s in/out ratios, not the feed (the old 31/32-px split was fixed pixels AND redundant asymmetry).
                let steps: f32 = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / (self.viewport.span / (1 << 6) as f32),
                };
                if steps != 0.0 {
                    self.apply_zoom_change(Some(steps));
                    return;
                }
                self.dispatch_event(event);
            }
            // Plain (non-zoom) wheel — the consumer scrolls its content. Scrolling MOVES content under a stationary cursor, so it's a content-moving event exactly like resize / zoom / drag-release: the incremental opaque-only finalize would leave stale AA pixels (avatar rims, glyph edges, dividers) at the pre-scroll positions, and the post-finalize hover overlay would then tint the current hit-map over that stale content (the "hover fill in the wrong spot" on scroll). Promote to a full repaint so the whole window re-finalizes at the new positions, AA pixels included, and the overlay reads coherent content. Dispatch to the consumer first so it updates its scroll offset, then repaint.
            WindowEvent::MouseWheel { .. } => {
                self.dispatch_event(event);
                self.pending_full_repaint = true;
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::Focused(focused) => {
                let focused = *focused;
                self.is_focused = focused;
                // Cancel any in-progress resize drag if we lose focus mid-drag (the user alt-tabbed or the WM stole focus). Keeps state consistent.
                if !focused && self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                }
                // Shadow seed depends on focus (full strength vs quarter strength) → re-cast shadow over a fresh band, which only happens on the full-repaint path.
                self.pending_full_repaint = true;
                self.dispatch_event(event);
                // Repaint so the drop shadow dims/brightens immediately.
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // Press-hold-release: arm the element under the pointer. The action does NOT fire here — it waits for a release over the same element (drag-off cancels). The raw press is still forwarded so the app can do its press-time work (focus, textbox cursor, drag-select arm, window-drag / resize). Redraw so the "held" colour appears.
                self.pointer.on_down(self.hit_under_cursor());
                self.dispatch_event(event);
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                // Press-hold-release: a release over the SAME element the press armed is a validated activation; a release after a drag-off fires nothing. Emit the activation BEFORE forwarding the raw release so the app's release-time bookkeeping sees a consistent world.
                let activate = self.pointer.on_up(self.hit_under_cursor());
                // End of resize drag — release ownership of the loop. The buffer is already in the final state from the last drag tick; no extra repaint needed.
                if self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                }
                // End of in-buffer drag-to-move. Two release paths: (a) armed but never committed (zero cursor motion during the press) — no shifts happened, persistent_screen is intact, consumer's damage_rect drives the next paint. (b) Committed (`is_dragging_move = true`) — the wrap-shift fast path moved persistent_screen contents, leaving wrap artefacts at whichever edges the window slid across, so a full repaint is required to clean them up. Always request_redraw on either path so consumer-side invalidations queued during the press window (e.g. textbox defocus → glow_bbox damage) get a fresh render_frame to flush.
                if self.move_drag_armed {
                    self.move_drag_armed = false;
                    if self.is_dragging_move {
                        self.is_dragging_move = false;
                        self.pending_full_repaint = true;
                    }
                    if let Some(window) = self.window.as_ref() {
                        window.set_cursor(winit::window::CursorIcon::Default);
                        window.request_redraw();
                    }
                }
                if let Some(id) = activate {
                    self.dispatch_activate(id);
                }
                self.dispatch_event(event);
                // Clear the "held" colour now that the press ended (whether it fired or cancelled).
                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // macOS click-thru: if the global monitor detected the cursor re-entering an opaque region while hittest was off, flip it back on. While hittest is off we keep requesting redraws to poll the monitor flag at vsync rate.
                #[cfg(target_os = "macos")]
                if self.hittest_off {
                    if let Some(ref monitor) = self.hittest_monitor {
                        if monitor.check_reenter() {
                            if let Some(window) = self.window.as_ref() {
                                let _ = window.set_cursor_hittest(true);
                                self.hittest_off = false;
                            }
                        } else if let Some(window) = self.window.as_ref() {
                            // Keep polling — next vsync will check again.
                            window.request_redraw();
                        }
                    }
                }
                // Drag-to-move fast path: shift the existing screen pixels by the per-tick delta instead of re-rendering anything. Skips consumer.render(), scratch fill, finalize, and shadow rasterization.
                if self.is_dragging_move {
                    self.apply_move_drag_shift();
                    return;
                }
                // Resize drag: apply the new geometry in-buffer, then paint at the new size.
                if self.is_dragging_resize {
                    self.apply_resize_drag();
                }
                self.render_frame();
            }
            _ => {
                self.dispatch_event(event);
            }
        }
    }

    /// Cross-thread user-event payload from [`EventLoopProxy::send_event`]. Builds a [`Context`] over the host's shared resources and hands the typed event to [`FluorApp::on_user_event`]. The consumer typically reads/mutates app state and calls `ctx.window.request_redraw()` if the state change should repaint; if it doesn't request_redraw the next tick still runs normally.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: A::UserEvent) {
        let Some(window) = self.window.as_ref().cloned() else {
            return;
        };
        let Some(text) = self.text.as_mut() else {
            return;
        };
        let mut ctx = Context {
            pressed_hit: self.pointer.held_id(),
            viewport: self.viewport,
            text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &*window,
            modifiers: winit_compat::from_winit_mods(self.modifiers),
            cursor_x: self.cursor_x - self.window_rect.x as Coord,
            cursor_y: self.cursor_y - self.window_rect.y as Coord,
            is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
            damage_clip: crate::canvas::PixelRect::new(
                0,
                0,
                self.viewport.width_px as usize,
                self.viewport.height_px as usize,
            ),
        };
        self.app.on_user_event(event, &mut ctx);
    }
}

#[cfg(feature = "host-winit")]
impl<A: FluorApp + 'static> DesktopShell<A> {
    /// Apply a zoom change to `self.viewport.ru` and propagate to the consumer. `steps = Some(s)` adjusts by `s` photon-asymmetric log steps (positive in, negative out); `steps = None` resets to 1.0 (Ctrl+0 binding). Calls `app.on_resize` with unchanged pixel dimensions so the consumer's existing resize path picks up the new `ctx.viewport.ru`, marks chrome / widget layers dirty (via their internal Group resize), and re-rasterizes at the new effective span. No separate `on_zoom` callback needed — the consumer's on_resize is the single "viewport changed" entry point.
    fn apply_zoom_change(&mut self, steps: Option<f32>) {
        match steps {
            Some(s) => self.viewport.adjust_zoom(s),
            None => self.viewport.reset_zoom(),
        }
        // Zoom changes effective_span → chrome perimeter, AA edges, glyphs, shadow ray length all scale. Full repaint required.
        self.pending_full_repaint = true;
        if let (Some(window), Some(text)) = (self.window.as_ref().cloned(), self.text.as_mut()) {
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                damage_clip: crate::canvas::PixelRect::new(
                    0,
                    0,
                    self.viewport.width_px as usize,
                    self.viewport.height_px as usize,
                ),
            };
            self.app
                .on_resize(self.viewport.width_px, self.viewport.height_px, &mut ctx);
            drop(ctx);
            window.request_redraw();
        }
    }

    /// Helper: dispatch a generic event to `app.on_event`, applying any returned [`EventResponse`].
    /// Hit id under the cursor right now, read from the app's [`FluorApp::hit_test_map`] at the window-local cursor position — the same map + indexing the overlay pass uses. `HIT_NONE` when the app exposes no map, or the cursor is out of bounds. Feeds the [`crate::host::pointer::PointerArbiter`] on every down / move / up.
    fn hit_under_cursor(&self) -> crate::paint::HitId {
        let x = (self.cursor_x - self.window_rect.x as Coord) as i32;
        let y = (self.cursor_y - self.window_rect.y as Coord) as i32;
        if x < 0 || y < 0 {
            return crate::paint::HIT_NONE;
        }
        match self.app.hit_test_map() {
            Some((map, w, h)) if (x as usize) < w && (y as usize) < h => {
                map[(y as usize) * w + (x as usize)]
            }
            _ => crate::paint::HIT_NONE,
        }
    }

    /// Deliver a validated activation ([`FluorApp::on_activate`]) — pointer up over the same element it went down on, no drag-off. Mirrors [`Self::dispatch_event`]'s Context build; called from the mouse-release arm before the raw Released is forwarded.
    fn dispatch_activate(&mut self, hit_id: crate::paint::HitId) {
        if let (Some(window), Some(text)) = (self.window.as_ref().cloned(), self.text.as_mut()) {
            let x = self.cursor_x - self.window_rect.x as Coord;
            let y = self.cursor_y - self.window_rect.y as Coord;
            let mods = winit_compat::from_winit_mods(self.modifiers);
            let mut ctx = Context {
                pressed_hit: crate::paint::HIT_NONE,
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: mods,
                cursor_x: x,
                cursor_y: y,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                damage_clip: crate::canvas::PixelRect::new(
                    0,
                    0,
                    self.viewport.width_px as usize,
                    self.viewport.height_px as usize,
                ),
            };
            let response = self.app.on_activate(hit_id, x, y, mods, &mut ctx);
            drop(ctx);
            self.apply_response(response);
        }
    }

    fn dispatch_event(&mut self, event: WindowEvent) {
        if let (Some(window), Some(text)) = (self.window.as_ref().cloned(), self.text.as_mut()) {
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: self.cursor_x - self.window_rect.x as Coord,
                cursor_y: self.cursor_y - self.window_rect.y as Coord,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: (self.window_rect.x, self.window_rect.y),
                damage_clip: crate::canvas::PixelRect::new(
                    0,
                    0,
                    self.viewport.width_px as usize,
                    self.viewport.height_px as usize,
                ),
            };
            let response = match winit_compat::from_winit_event(&event) {
                Some(fevent) => self.app.on_event(&fevent, &mut ctx),
                None => EventResponse::Pass,
            };
            drop(ctx);
            self.apply_response(response);
        }
    }
}
