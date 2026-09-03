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
    /// The rect is in window-relative coordinates (= surface-local coords when the OS window is fullscreen at its monitor's origin). Negative offsets get clamped to 0 since XShape rectangles must be unsigned. A zero `w` or `h` sends an EMPTY rectangle list, which makes the window fully click-thru — X11's SET with no rectangles yields an empty input region. Returns `true` if the call was sent successfully, `false` if the window isn't X11 or the connection failed.
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
        // Empty rect → empty rectangle list → the whole window is mouse-transparent (used when the visible window doesn't intersect this surface at all).
        let rect = Rectangle {
            x: x.max(0).min(i16::MAX as i32) as i16,
            y: y.max(0).min(i16::MAX as i32) as i16,
            width: w.min(u16::MAX as u32) as u16,
            height: h.min(u16::MAX as u32) as u16,
        };
        let rects: &[Rectangle] = if w == 0 || h == 0 {
            &[]
        } else {
            std::slice::from_ref(&rect)
        };
        if conn
            .shape_rectangles(SO::SET, SK::INPUT, ClipOrdering::UNSORTED, xid, 0, 0, rects)
            .is_err()
        {
            return false;
        }
        let _ = conn.flush();
        true
    }

    /// Tell the compositor to NEVER unredirect this window (`_NET_WM_BYPASS_COMPOSITOR = 2`). Mutter/KWin fast-path a window that exactly covers a monitor by bypassing compositing — and an unredirected window loses its alpha channel, so every transparent pixel of our monitor-sized surface renders OPAQUE BLACK (observed: the taskbar strip going black "on occasion"). The hint is advisory but honoured by both major compositors; set once per surface at creation.
    pub fn set_never_bypass_compositor(window: &winit::window::Window) -> bool {
        use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _, PropMode};
        use x11rb::wrapper::ConnectionExt as _;

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
        let Some(atom) = conn
            .intern_atom(false, b"_NET_WM_BYPASS_COMPOSITOR")
            .ok()
            .and_then(|c| c.reply().ok())
            .map(|r| r.atom)
        else {
            return false;
        };
        // EWMH values: 0 = no preference, 1 = request bypass, 2 = request NO bypass — we need 2.
        let ok = conn
            .change_property32(PropMode::REPLACE, xid, atom, AtomEnum::CARDINAL, &[2u32])
            .is_ok();
        let _ = conn.flush();
        ok
    }

    /// Ask the WM to keep this window OUT of the taskbar and pager via an EWMH `_NET_WM_STATE` client message carrying `_NET_WM_STATE_SKIP_TASKBAR` + `_NET_WM_STATE_SKIP_PAGER` (action add when `skip`, remove otherwise). Sent on every non-anchor monitor surface so alt-tab and the taskbar show ONE entry for the app instead of one per monitor. Returns `true` if the message was sent; `false` if the window isn't X11 or the connection failed (a WM that ignores the hint just shows extra entries — cosmetic, not fatal).
    pub fn set_skip_taskbar(window: &winit::window::Window, skip: bool) -> bool {
        use x11rb::protocol::xproto::{ClientMessageEvent, ConnectionExt as _, EventMask};

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
        let Some(root) = conn.setup().roots.first().map(|s| s.root) else {
            return false;
        };
        let intern = |name: &[u8]| {
            conn.intern_atom(false, name)
                .ok()
                .and_then(|c| c.reply().ok())
                .map(|r| r.atom)
        };
        let (Some(state), Some(skip_taskbar), Some(skip_pager)) = (
            intern(b"_NET_WM_STATE"),
            intern(b"_NET_WM_STATE_SKIP_TASKBAR"),
            intern(b"_NET_WM_STATE_SKIP_PAGER"),
        ) else {
            return false;
        };
        // _NET_WM_STATE action codes: 0 = remove, 1 = add. data.l[3] = 1 marks the source as a normal application per EWMH.
        let action: u32 = if skip { 1 } else { 0 };
        let ev = ClientMessageEvent::new(32, xid, state, [action, skip_taskbar, skip_pager, 1, 0]);
        if conn
            .send_event(
                false,
                root,
                EventMask::SUBSTRUCTURE_REDIRECT | EventMask::SUBSTRUCTURE_NOTIFY,
                ev,
            )
            .is_err()
        {
            return false;
        }
        let _ = conn.flush();
        true
    }

    /// The desktop work area `(x, y, w, h)` — the monitor minus space reserved by panels /
    /// taskbars — read from the root window's EWMH `_NET_WORKAREA` property. Used to place the visible window so its bottom edge (the chrome status band) doesn't slide under a taskbar. `_NET_WORKAREA` holds `[x, y, w, h]` per virtual desktop; we take the first
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

/// Windows work-area query — `SystemParametersInfo(SPI_GETWORKAREA)` gives the primary monitor's work rect (full screen minus the taskbar), in virtual-screen pixels.
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

/// macOS work-area query — the matching `NSScreen`'s `visibleFrame` (full frame minus the menu bar + Dock), per monitor.
/// `origin`/`size` are the monitor's GLOBAL point rect in winit's top-left-origin convention; NSScreen frames are global POINTS with a bottom-left origin whose flip reference is the primary screen (`screens[0]`) height, so we flip each candidate frame and match it against the requested rect within a small epsilon (frames are integral in practice; the epsilon absorbs f64 noise).
/// Returns the visible frame as a GLOBAL top-left-origin point rect, or `None` when no screen matches (caller falls back to the full monitor rect).
#[cfg(all(feature = "host-winit", target_os = "macos"))]
fn work_area_macos(origin: (i32, i32), size: (u32, u32)) -> Option<(i32, i32, u32, u32)> {
    use objc2_app_kit::NSScreen;
    let mtm = objc2::MainThreadMarker::new()?;
    let screens = NSScreen::screens(mtm);
    let primary_h = screens.iter().next()?.frame().size.height;
    const EPS: f64 = 1.5;
    for screen in screens.iter() {
        let f = screen.frame();
        let top_y = primary_h - (f.origin.y + f.size.height);
        if (f.origin.x - origin.0 as f64).abs() > EPS
            || (top_y - origin.1 as f64).abs() > EPS
            || (f.size.width - size.0 as f64).abs() > EPS
            || (f.size.height - size.1 as f64).abs() > EPS
        {
            continue;
        }
        let vf = screen.visibleFrame();
        let wa_x = vf.origin.x.round() as i32;
        let wa_y = (primary_h - (vf.origin.y + vf.size.height)).round() as i32;
        let wa_w = vf.size.width.round().max(0.0) as u32;
        let wa_h = vf.size.height.round().max(0.0) as u32;
        if wa_w == 0 || wa_h == 0 {
            return None;
        }
        return Some((wa_x, wa_y, wa_w, wa_h));
    }
    None
}

/// Intersect two `(x, y, w, h)` rects in global desktop units; `None` when they don't overlap.
#[cfg(feature = "host-winit")]
fn intersect_rect(
    a: (i32, i32, u32, u32),
    b: (i32, i32, u32, u32),
) -> Option<(i32, i32, u32, u32)> {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = (a.0 + a.2 as i32).min(b.0 + b.2 as i32);
    let y1 = (a.1 + a.3 as i32).min(b.1 + b.3 as i32);
    if x0 < x1 && y0 < y1 {
        Some((x0, y0, (x1 - x0) as u32, (y1 - y0) as u32))
    } else {
        None
    }
}

/// Per-monitor work area `(x, y, w, h)` in GLOBAL desktop units — the monitor rect at `origin`/`size` minus space reserved by panels / taskbars / the menu bar + Dock. The platform queries are GLOBAL (X11 `_NET_WORKAREA` is per-virtual-desktop, Windows `SPI_GETWORKAREA` is the primary monitor in virtual-screen coords), so we intersect the global work area with this monitor's rect; per-strut per-monitor refinement is a phase-D note. Falls back to the full monitor rect on Wayland (no client-side work-area query), when the rects don't overlap, and anywhere the query is unavailable.
#[cfg(feature = "host-winit")]
fn monitor_work_area(origin: (i32, i32), size: (u32, u32)) -> (i32, i32, u32, u32) {
    let mon_rect = (origin.0, origin.1, size.0, size.1);
    #[cfg(target_os = "linux")]
    {
        // Wayland has no EWMH root window; `work_area()` returns None there and we fall back.
        return x11_atomic::work_area()
            .and_then(|wa| intersect_rect(wa, mon_rect))
            .unwrap_or(mon_rect);
    }
    #[cfg(target_os = "windows")]
    {
        return work_area_windows()
            .and_then(|wa| intersect_rect(wa, mon_rect))
            .unwrap_or(mon_rect);
    }
    #[cfg(target_os = "macos")]
    {
        // The macOS query matches the NSScreen by GLOBAL point rect and returns a GLOBAL point rect already; intersect defensively like the other platforms.
        return work_area_macos(origin, size)
            .and_then(|wa| intersect_rect(wa, mon_rect))
            .unwrap_or(mon_rect);
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
    {
        mon_rect
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

    /// The app's native menu-bar spec, read ONCE at window creation and built into a real OS menu
    /// (macOS `NSMenu`; other platforms ignore it for now). Choosing an [`super::menu::MenuItem::Action`]
    /// delivers its `id` back via [`crate::event::Event::MenuItem`]. Default: no menu. Keep it
    /// static for now — the host builds it once and doesn't yet rebuild on the fly.
    fn menu(&self) -> alloc::vec::Vec<super::menu::MenuItem> {
        alloc::vec::Vec::new()
    }

    /// The app-identity icon for the OS window (taskbar / alt-tab / title bar). The host applies it at window creation so the OS-level icon matches the in-chrome orb — apps that hold a [`DefaultChrome`] typically return `self.chrome.app_icon.as_ref()`.
    ///
    /// **Platform reach.** This drives winit's `set_window_icon`, which only takes effect on
    /// **Windows and X11**. On **Wayland** the icon is sourced from a `.desktop` file matched by `app_id`, and on **macOS** from the `.app` bundle's `.icns` — both are build-time packaging, not a runtime call, so this hook is a no-op there. Returns `None` by default (no OS icon set).
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

    /// The app's folded OS focus flipped (focused = any surface focused). Fires once per change — the natural durability edge for "the user looked away". Default: ignored.
    fn on_focus_changed(&mut self, focused: bool) {
        let _ = focused;
    }

    /// The visible window's rect SETTLED after a user gesture — drag-move release or resize-drag end, never while maximized (a maximized rect is a mode, not a placement). GLOBAL desktop units: the exact currency [`take_window_geometry_request`](Self::take_window_geometry_request) restores. The geometry-persistence edge. Default: ignored.
    fn on_window_rect_changed(&mut self, x: i32, y: i32, w: u32, h: u32) {
        let _ = (x, y, w, h);
    }

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

    /// Optional scroll-copy hint for THIS frame: `(rect, dy)` declaring that the contents of `rect` (viewport pixels) translated RIGIDLY by `dy` viewport pixels since the last frame — positive `dy` = moved down. When present and this is NOT a full repaint, the host `memmove`s that region of `persistent_screen` by `dy` (scaled to backing pixels) BEFORE finalize, so the only pixels that get redrawn are the freshly-exposed band the app reports through [`Self::damage_rect`] — a scroll becomes a copy plus a sliver instead of a whole-pane redraw.
    ///
    /// Contract the app must honour for this to be correct: (1) `rect` contains ONLY content that translated rigidly — pin nothing inside it that stayed put or animated independently (put those outside `rect`, or list them in `damage_rect` so they repaint); (2) `damage_rect` this frame covers the `|dy|`-tall exposed band inside `rect` (plus anything else that changed); (3) `dy` is an integer count of viewport pixels that matches how far the content actually moved — the host snaps to backing pixels, so drive the content from an integer-quantised scroll offset to avoid sub-pixel drift accumulating in the un-repainted bulk.
    ///
    /// Default `None`: no scroll-copy; `damage_rect` alone drives a normal (possibly full) repaint.
    fn scroll_hint(&mut self, _viewport: Viewport) -> Option<(crate::canvas::PixelRect, i32)> {
        None
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
    /// One-shot ABSOLUTE zoom request — e.g. a restored per-device zoom setting. Polled by the host each idle pass BEFORE tick; `Some(ru)` applies exactly like a user zoom (clamped, full repaint, `on_resize` propagation) and the app must return it at most once (take semantics). Default: never.
    fn take_zoom_request(&mut self) -> Option<f32> {
        None
    }

    /// One-shot window-geometry restore — `(outer x, outer y, inner w, inner h)` in physical pixels, the same convention [`WindowHandle::outer_position`]/[`inner_size`](WindowHandle::inner_size) report for saving. Polled beside the zoom request; the host applies the size always but the POSITION only when the point still lands on a live monitor (a saved spot on an unplugged screen must not strand the window off-desktop). Take semantics like the zoom.
    fn take_window_geometry_request(&mut self) -> Option<(i32, i32, u32, u32)> {
        None
    }

    fn tick(&mut self, ctx: &mut Context) -> bool {
        let _ = ctx;
        false
    }

    /// User-event payload arrived from a background thread via [`EventLoopProxy::send_event`]. Typical use: a network task completed, an avatar download finished, a key-ceremony hit a milestone — the task sends the appropriate variant; this method routes it to the right state-machine handler and (usually) calls `ctx.window.request_redraw()` to repaint with the new state. The returned [`EventResponse`] goes thru the same host dispatch as `on_event`'s — so a background trigger can drive window-state changes (chiefly `ShowWindow` for a second-launch handoff). Default `Pass` for apps that declared `type UserEvent = ()`.
    fn on_user_event(&mut self, event: Self::UserEvent, ctx: &mut Context) -> EventResponse {
        let _ = (event, ctx);
        EventResponse::Pass
    }

    /// Whether the window should be created INVISIBLE — the resident/background launch (`--background`-style flags): the app boots, network comes up, but no surface shows until an explicit [`EventResponse::ShowWindow`]. Consulted once at window creation. Default `false` (normal visible launch).
    fn start_hidden(&self) -> bool {
        false
    }

    /// The window is being closed — by the OS (`WindowEvent::CloseRequested`: Alt-F4, taskbar close) or by the app's own chrome close button ([`EventResponse::Close`]). Return `true` to stay RESIDENT: the host hides the window (`set_visible(false)`) and the process keeps running — network, timers, everything — until an [`EventResponse::ShowWindow`] surfaces it again. Return `false` (the default) for the normal exit. The app is the policy owner: track your own hidden state here (the host won't call anything else).
    fn on_close_requested(&mut self) -> bool {
        false
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

    /// HONEST IME EDITOR, read half: the focused textbox's full text + cursor (in CHARS — the host converts to UTF-16). The Android host mirrors this to the InputConnection so `getTextBeforeCursor`-style queries answer TRUTHFULLY — Google voice typing reads the field back continuously and aborts mid-sentence against an editor that claims to be empty. Default: no editor (`None`).
    fn ime_editor_state(&mut self) -> Option<(alloc::string::String, usize)> {
        None
    }

    /// HONEST IME EDITOR, write half: replace char range `[start, end)` of the focused textbox with `s` (cursor lands after the insertion). This is how commitText / setComposingText / setComposingRegion / deleteSurroundingText express themselves as TRUE range edits instead of backspace-replay at the cursor. Default: no-op.
    fn ime_replace_chars(&mut self, _start: usize, _end: usize, _s: &str, _text: &mut TextRenderer) {}

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

/// Visible-window placement inside the fullscreen compositor surfaces. fluor runs fullscreen transparent OS windows owning each display — the "window" the consumer paints into is a sub-rect at `(x, y)` with `(w, h)` pixels. `(x, y, w, h)` are GLOBAL virtual-desktop units (winit's monitor-position space); each surface blits the window at `(x, y) − surface.origin`, so on a single monitor at origin (0, 0) these are numerically the old screen-space coords. WindowRect is mutated by drag-to-move (changes `x, y`) and resize-drag (changes `w, h`); both are in-buffer operations that don't touch the OS window geometry.
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

/// One per-monitor OS surface in the fullscreen-compositor model — a fullscreen borderless transparent window pinned to its monitor, plus the per-surface presentation state.
/// Phase A holds exactly ONE of these (the primary monitor); the multi-monitor phases spawn one per output and route events by `WindowId`.
#[cfg(feature = "host-winit")]
struct MonitorSurface {
    /// The fullscreen borderless transparent OS window covering this monitor.
    window: Arc<Window>,
    /// The winit monitor this surface is pinned to; geometry refreshes (rotation, hotplug) re-read from here.
    #[allow(dead_code)]
    monitor: winit::monitor::MonitorHandle,
    /// This monitor's top-left corner in virtual-desktop units — physical px on X11/Windows, POINTS on macOS (recovered as winit's physical position ÷ scale, which is exact because winit derives the physical values FROM the native point layout by multiplying).
    origin: (i32, i32),
    /// This monitor's size in desktop units (= the OS surface buffer size on X11/Windows; points on macOS, backing buffer = `backing()`).
    size: (u32, u32),
    /// The monitor scale factor as reported by winit.
    scale: f64,
    /// Surface backing pixels per desktop unit — 1.0 on X11/Windows (physical-pixel desktop), the monitor scale on macOS (point desktop).
    pixel_ratio: f64,
    /// This monitor's work area `(x, y, w, h)` in GLOBAL desktop units — the monitor minus panels/taskbars, via [`monitor_work_area`].
    work_area: (i32, i32, u32, u32),
    /// wgpu renderer for this surface (macOS present path).
    #[cfg(target_os = "macos")]
    renderer: Option<super::renderer_wgpu::Renderer>,
    /// softbuffer surface for this window (Linux/X11, Redox); `None` on Windows which presents via UpdateLayeredWindow directly.
    #[cfg(not(target_os = "macos"))]
    surface: Option<softbuffer::Surface<Arc<Window>, Arc<Window>>>,
    /// Persistent surface-sized pixel buffer, owned by us. Survives across frames so post-finalize overlays (hover/focus tint diffs, blinkey) can mutate just a few pixels each frame without re-running finalize for the whole window. The platform's softbuffer / wgpu back buffer may rotate or arrive stale; we always memcpy this over it just before `present()` so the platform buffer's prior state doesn't matter. Resized when the surface size changes.
    persistent_screen: Vec<u32>,
    /// `false` until the first `WindowEvent::Resized` (or creation-at-known-size) confirms the OS surface is allocated at real geometry — painting before that positions chrome against a stale rect.
    surface_ready: bool,
    /// The window doesn't currently intersect this surface — present nothing, accept no input.
    dormant: bool,
    /// This surface's OS window has keyboard focus; the shell's `is_focused` is the any() fold over all surfaces.
    focused: bool,
    /// One-shot per-surface full-repaint request — set when the window enters this surface (phase B) so its first composite refills the buffer; cleared by `composite_and_present`.
    needs_full_blit: bool,
    /// Last input region pushed to the OS for this surface, in surface-local desktop units, `(0, 0, 0, 0)` meaning fully click-thru. Used by [`DesktopShell::push_input_region`] to dedupe the XShape call (and the macOS `set_cursor_hittest` equivalent).
    last_input_region: (i32, i32, u32, u32),
}

#[cfg(feature = "host-winit")]
impl MonitorSurface {
    /// Backing-pixel dimensions of this surface's present buffers: desktop units × `pixel_ratio`, rounded. On X11/Windows `pixel_ratio` is 1.0 so this IS `size`; on macOS `size` is points and this is the Retina backing buffer. Rounding convention for ALL unit→backing conversions in this file: multiply by the ratio, then `.round()` (ties away from zero).
    fn backing(&self) -> (usize, usize) {
        (
            ((self.size.0 as f64) * self.pixel_ratio).round().max(1.0) as usize,
            ((self.size.1 as f64) * self.pixel_ratio).round().max(1.0) as usize,
        )
    }

    /// This monitor's rect in GLOBAL desktop units, in the tuple shape [`intersect_rect`] speaks.
    fn rect(&self) -> (i32, i32, u32, u32) {
        (self.origin.0, self.origin.1, self.size.0, self.size.1)
    }
}

/// Dual-raster gate for mixed-DPI straddles (phase C). `true` = while the window straddles a surface whose scale differs from `window_scale`, the app is re-rendered once per extra scale (full repaint at that density) so BOTH halves are native-crisp. `false` = the documented perf escape hatch: every surface composites from the home pass — still pixel-correct on X11/Windows (every pass shares the window dims there, the far half just rasterizes at home density), but on macOS pass dims differ per scale so the far half would blit at the wrong size; flip this off only for X11 perf triage.
#[cfg(feature = "host-winit")]
const STRADDLE_DUAL_RASTER: bool = true;

/// One rasterization pass at a specific monitor scale (phase C). Pass 0 is the shell's own `viewport` + `scratch` + `clip_mask` trio at `window_scale`; a `RasterPass` exists ONLY for an involved surface whose scale differs mid-straddle (deduped by scale, dropped when the straddle ends).
/// Pass geometry: on X11/Windows desktop units are physical px, so every pass shares the window dims and the density difference enters purely thru `viewport.ru` (= user ru × pass scale ÷ `window_scale`). On macOS desktop units are POINTS, so pass dims = points × pass scale and `ru` stays the bare user zoom — fluor's layout is span-relative, so the scaled dims already carry the density and the two passes lay out identically at different pixel counts.
#[cfg(feature = "host-winit")]
struct RasterPass {
    scale: f64,
    viewport: Viewport,
    scratch: Vec<u32>,
    clip_mask: Vec<u8>,
}

/// The host's adapter — owns platform handles + the consumer's `App`, dispatches events thru the trait. Not user-facing; constructed by [`run_app`].
///
/// **Compositor architecture.** The OS window is fullscreen borderless transparent — fluor owns the entire screen buffer. The consumer paints into a window-sized scratch buffer (sized to `viewport` = `window_rect.w × window_rect.h`); the host then blits that scratch into the screen buffer at the `window_rect` offset. Pixels outside the window stay α=0 so the OS compositor shows whatever's behind us. Click-thru is via a per-resize input-region call (set later, see step 2 of the fullscreen-compositor pivot) so clicks outside `window_rect` route to whatever's underneath.
#[cfg(feature = "host-winit")]
struct DesktopShell<A: FluorApp> {
    app: A,
    /// Per-monitor OS surfaces. Phase A: exactly one entry (the primary monitor), created in `resumed`; empty until then.
    surfaces: Vec<MonitorSurface>,
    /// Index of the surface that owns tick/render/maximize — the surface the window (mostly) lives on; re-elected by [`Self::update_home`] on every `window_rect` mutation.
    home: usize,
    /// Index of the taskbar-visible surface (the one that keeps title + icon + alt-tab presence); every other surface is skip-taskbar'd at creation.
    anchor: usize,
    /// The monitor scale the window geometry was last rebased to (phase-C DPI model). Set to the home surface's scale at creation; re-anchored by [`Self::settle_rebase`] at settle points (drag release, maximize, home ScaleFactorChanged).
    window_scale: f64,
    /// PASS-0 consumer-visible viewport — sized to the window's pixel dims at `window_scale` density (= `window_rect.w × h` on X11/Windows, `window_rect` points × `window_scale` on macOS), NOT the screen. Together with `scratch` + `clip_mask` this trio IS raster pass 0; `extra_passes` holds the straddle-only passes at other scales.
    viewport: Viewport,
    /// Where the visible window lives, in GLOBAL virtual-desktop units (winit's monitor-position space; physical px on X11/Windows, points on macOS). Driven by drag-to-move + resize-drag; on a single monitor at origin (0, 0) these are numerically the old screen-buffer coords.
    window_rect: WindowRect,
    /// Pass-0 window-sized scratch buffer. The consumer renders into this (at viewport dimensions); the host runs `finalize_for_os` on it with the window-space clip mask, then blits row-by-row into each involved surface's buffer at `(window_rect.xy − surface.origin) × pixel_ratio`. Resized on `window_rect` size change.
    scratch: Vec<u32>,
    /// Straddle-only raster passes at scales other than `window_scale` (phase C) — see [`RasterPass`]. Empty except while the window straddles a differing-scale surface with [`STRADDLE_DUAL_RASTER`] on.
    extra_passes: Vec<RasterPass>,

    // --- Shared resources ---
    text: Option<TextRenderer>,
    /// Window-shape clip mask, one byte α per pixel. Sized to `viewport` (= window-space, NOT screen-space). The consumer carves shape into it (rounded corner cutouts etc.); `finalize_for_os` multiplies it into each pixel's α at the scratch-buffer boundary before the host blits to screen. Default `255` (fully visible) means a consumer that doesn't touch it gets a rectangular window.
    clip_mask: Vec<u8>,
    /// Last known cursor position in GLOBAL virtual-desktop units — surface-local position + that surface's origin, updated on every `CursorMoved`. On a single monitor at origin (0, 0) these equal the old screen-space values.
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
    /// Last window_rect (x, y, w, h) that was actually painted, in GLOBAL virtual-desktop units like `window_rect` itself. Set after every render_frame; consulted at drag-move vsync ticks to compute the (dx, dy) delta to feed into `shift_screen_wrap`. Without this we'd have no way to know "how much did the window move since the last frame" because the cursor anchor describes total drag distance, not per-frame increment.
    last_painted_rect: WindowRect,
    /// Saved `window_rect` (GLOBAL virtual-desktop units) from BEFORE the last `EventResponse::ToggleMaximized` set us work-area-sized. `Some` ⇒ we're currently in the maximized state and the next toggle restores from here; `None` ⇒ we're at user-sized and the next toggle saves+grows. Drag-to-move while maximized currently drags the screen-sized rect (weird but harmless); a future iteration could auto-unmaximize on drag like most WMs.
    saved_rect_for_maximize: Option<WindowRect>,

    /// Tracks `WindowEvent::Focused` folded over every surface (any() of `MonitorSurface::focused`) so the drop shadow can dim when the window is inactive — focused windows cast a stronger shadow (`SHADOW_SEED_FOCUSED`), unfocused ones use a quarter-strength shadow (`SHADOW_SEED_UNFOCUSED`).
    is_focused: bool,

    /// Live render-pipeline counters. Updated every `render_frame` call (composite-time EMA + frame counter); rendered to a bottom-of-window debug strip when [`paint::DEBUG_SHOW_FPS`] is set via the `[]f` chord.
    debug_stats: crate::paint::DebugStats,

    /// Frame-level damage accumulator. Reset at the top of each `render_frame`; passed to the consumer via [`Context::damage`]; read back after consumer render to drive damage-clipped composite and the [`paint::DEBUG_SHOW_DAMAGE`] outline overlay.
    pending_damage: crate::canvas::Damage,
    /// FPS strip active state from the previous frame. When it toggles `true → false`, this frame's damage_clip must include the strip bbox so the just-vanished strip pixels get cleared from scratch (and propagated into persistent_screen via finalize). Tracked instead of a generic `prev_damage_clip` union to avoid sticky viewport-sized damage on hover frames after any prior full repaint.
    last_strip_active: bool,
    /// Set by any event that destroys the chrome perimeter + shadow band content in the surfaces' `persistent_screen` buffers: drag release, resize, zoom, focus change. Consumed once per `render_frame` to switch from incremental mode to full-repaint mode (wipe the involved surfaces' buffers, finalize copies every pixel, paint_shadow runs once into the fresh band). Replaces every prior geometric-equality check on `damage_clip`.
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
            surfaces: Vec::new(),
            home: 0,
            anchor: 0,
            window_scale: 1.0,
            viewport: Viewport::new(1, 1),
            window_rect: WindowRect {
                x: 0,
                y: 0,
                w: 1,
                h: 1,
            },
            scratch: Vec::new(),
            extra_passes: Vec::new(),
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
            is_focused: true,
            debug_stats: crate::paint::DebugStats::default(),
            pending_damage: crate::canvas::Damage::new(),
            last_strip_active: false,
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

    /// The home surface's window handle, cloned — the drop-in replacement for the old single `self.window` Option (empty surfaces vec pre-`resumed` maps to `None`).
    fn home_window(&self) -> Option<Arc<Window>> {
        self.surfaces.get(self.home).map(|s| s.window.clone())
    }

    /// Which surface owns this `WindowId` — linear scan, the vec is at most one-per-monitor small.
    fn surface_for_window(&self, id: WindowId) -> Option<usize> {
        self.surfaces.iter().position(|s| s.window.id() == id)
    }

    /// Pass-0 pixels per desktop unit — `window_scale` on macOS (desktop units are points, the app rasterizes at points × scale), 1.0 everywhere else (desktop units ARE physical pixels).
    fn unit_to_px(&self) -> Coord {
        #[cfg(target_os = "macos")]
        {
            self.window_scale as Coord
        }
        #[cfg(not(target_os = "macos"))]
        {
            1.0
        }
    }

    /// The cursor in WINDOW-RELATIVE pass-0 pixels — the coordinate space every `Context` and hit-map lookup speaks: (global desktop-unit cursor − window_rect origin) × [`Self::unit_to_px`]. On X11/Windows the multiplier is 1 so this is literally the old subtraction.
    fn win_cursor_px(&self) -> (Coord, Coord) {
        let k = self.unit_to_px();
        (
            (self.cursor_x - self.window_rect.x as Coord) * k,
            (self.cursor_y - self.window_rect.y as Coord) * k,
        )
    }

    /// `Context::window_origin` — the window's top-left in pass-0 PIXEL space, measured RELATIVE TO
    /// THE HOME SURFACE (apps compensate content by origin deltas in pixels, so on macOS the point
    /// origin is scaled up by `window_scale` under the standard rounding convention).
    ///
    /// Surface-relative, NOT global desktop space: `FEvent::CursorMoved` carries winit's raw
    /// surface-LOCAL position (0-based within the reporting surface), so a consumer computing
    /// `raw_position − window_origin` to get a window-relative coordinate needs the origin in that
    /// same surface-local space. On the anchor surface (origin 0,0) the two are identical; on a
    /// monitor at a non-zero origin — e.g. a 4K panel placed at desktop (−434,−2160) — using the
    /// global origin put every click off by the surface origin (the "move to the other monitor →
    /// clicks way off" field report: ~+2130px in y). Subtracting the home surface's origin lands it.
    fn ctx_window_origin(&self) -> (i32, i32) {
        let k = self.unit_to_px() as f64;
        let (sox, soy) = self.surfaces.get(self.home).map_or((0, 0), |s| s.origin);
        (
            (((self.window_rect.x - sox) as f64) * k).round() as i32,
            (((self.window_rect.y - soy) as f64) * k).round() as i32,
        )
    }

    /// Build the pass-0 viewport for a window of `w × h` desktop units: identical dims on X11/Windows, points × `window_scale` on macOS (the app lays out span-relative, so the scaled dims carry the density and `ru` stays the user's zoom multiplier untouched).
    fn pass0_viewport(&self, w: u32, h: u32, ru: Coord) -> Viewport {
        let k = self.unit_to_px() as f64;
        let pw = (((w as f64) * k).round() as u32).max(1);
        let ph = (((h as f64) * k).round() as u32).max(1);
        Viewport::new(pw, ph).with_ru(ru)
    }

    /// Indices of surfaces whose monitor rect intersects `window_rect` — the set that composites this frame (phase B).
    fn involved(&self) -> Vec<usize> {
        let r = (
            self.window_rect.x,
            self.window_rect.y,
            self.window_rect.w,
            self.window_rect.h,
        );
        self.surfaces
            .iter()
            .enumerate()
            .filter(|(_, s)| intersect_rect(r, s.rect()).is_some())
            .map(|(i, _)| i)
            .collect()
    }

    /// Re-elect the home surface = argmax window-overlap area, ties keep the current home. Called after EVERY `window_rect` mutation (drag tick, resize tick, maximize, surface-resize clamp). A home change invalidates the composited chrome state (shadow band, AA edges were built against the old home), so it promotes to a full repaint. `window_scale` is NOT touched here — the DPI rebase waits for a settle point ([`Self::settle_rebase`]).
    fn update_home(&mut self) {
        let r = (
            self.window_rect.x,
            self.window_rect.y,
            self.window_rect.w,
            self.window_rect.h,
        );
        let mut best = self.home;
        let mut best_area: u64 = 0;
        if let Some(s) = self.surfaces.get(self.home) {
            if let Some((_, _, w, h)) = intersect_rect(r, s.rect()) {
                best_area = (w as u64) * (h as u64);
            }
        }
        for (i, s) in self.surfaces.iter().enumerate() {
            if i == self.home {
                continue;
            }
            if let Some((_, _, w, h)) = intersect_rect(r, s.rect()) {
                let a = (w as u64) * (h as u64);
                if a > best_area {
                    best = i;
                    best_area = a;
                }
            }
        }
        if best != self.home {
            log::info!("FLUOR-MON: home {} → {} (window=({},{}) {}x{})", self.home, best, r.0, r.1, r.2, r.3);
            self.home = best;
            self.pending_full_repaint = true;
        }
    }

    /// Flip surfaces between active and dormant as the window enters/leaves them (phase B). Entering: wake, arm `needs_full_blit` so the first composite refills the buffer, push the real input region. Leaving: zero the buffer, present the zeros once (the vacated monitor goes visually empty immediately), go dormant, push the now-empty input region (fully click-thru). The home surface is by definition involved; the extra guard keeps it live even in the degenerate zero-overlap case mid-drag.
    fn refresh_involvement(&mut self) {
        let r = (
            self.window_rect.x,
            self.window_rect.y,
            self.window_rect.w,
            self.window_rect.h,
        );
        for si in 0..self.surfaces.len() {
            let inv = intersect_rect(r, self.surfaces[si].rect()).is_some();
            if inv && self.surfaces[si].dormant {
                log::info!("FLUOR-MON: surface {} WAKES (window=({},{}) {}x{} ∩ surface={:?})", si, r.0, r.1, r.2, r.3, self.surfaces[si].rect());
                self.surfaces[si].dormant = false;
                self.surfaces[si].needs_full_blit = true;
                self.push_input_region(si);
            } else if !inv && !self.surfaces[si].dormant && si != self.home {
                log::info!("FLUOR-MON: surface {} EVACUATES (window left it)", si);
                self.surfaces[si].persistent_screen.fill(0);
                self.present_surface_raw(si);
                self.surfaces[si].dormant = true;
                self.push_input_region(si);
            }
        }
    }

    /// Push surface `si`'s persistent buffer to the OS as-is — no finalize, no overlay, no outline. Serves non-home `RedrawRequested` (OS expose), the dormant first present (all-zero buffer = fully transparent, so a fresh window never flashes back-buffer garbage), and the evacuation present that clears a just-vacated monitor.
    fn present_surface_raw(&mut self, si: usize) {
        let Some(s) = self.surfaces.get_mut(si) else {
            return;
        };
        let (scr_w, scr_h) = s.backing();
        let scr_px = scr_w * scr_h;
        if s.persistent_screen.len() != scr_px {
            s.persistent_screen.resize(scr_px, 0);
        }
        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = s.renderer.as_mut() else {
                return;
            };
            let mut buffer = renderer.lock_buffer();
            // A transient mismatch (scale change announced before the matching Resized) skips one present rather than panicking in copy_from_slice.
            if buffer.len() != s.persistent_screen.len() {
                return;
            }
            buffer.copy_from_slice(&s.persistent_screen);
            let _ = buffer.present();
        }
        #[cfg(target_os = "windows")]
        {
            super::windows_layered::present(&s.window, &s.persistent_screen, scr_w as u32, scr_h as u32);
        }
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            let Some(surface) = s.surface.as_mut() else {
                return;
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            if buffer.len() != s.persistent_screen.len() {
                return;
            }
            buffer.copy_from_slice(&s.persistent_screen);
            buffer.present().expect("softbuffer buffer.present");
        }
    }

    /// The surface that FULLY contains `r`, if any. The drag-move wrap-shift fast path is only legal when the whole window lived on ONE surface at the last paint AND still lives on that same surface now — the shift slides pixels within a single buffer and can't cross a seam.
    fn surface_fully_containing(&self, r: WindowRect) -> Option<usize> {
        self.surfaces.iter().position(|s| {
            r.x >= s.origin.0
                && r.y >= s.origin.1
                && r.x + r.w as i32 <= s.origin.0 + s.size.0 as i32
                && r.y + r.h as i32 <= s.origin.1 + s.size.1 as i32
        })
    }

    /// Clamp a restore-from-maximize rect into the union of surface rects: if it still intersects any surface it stands as saved; otherwise (its monitor shrank or vanished) it snaps into the NEAREST surface's work area — size clamped to fit, position clamped inside.
    fn clamp_rect_to_surfaces(&self, r: WindowRect) -> WindowRect {
        let rr = (r.x, r.y, r.w, r.h);
        if self
            .surfaces
            .iter()
            .any(|s| intersect_rect(rr, s.rect()).is_some())
        {
            return r;
        }
        let cx = r.x as i64 + (r.w as i64) / 2;
        let cy = r.y as i64 + (r.h as i64) / 2;
        let mut best = 0usize;
        let mut best_d = i64::MAX;
        for (i, s) in self.surfaces.iter().enumerate() {
            let sx = s.origin.0 as i64 + (s.size.0 as i64) / 2;
            let sy = s.origin.1 as i64 + (s.size.1 as i64) / 2;
            let d = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy);
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        let Some(s) = self.surfaces.get(best) else {
            return r;
        };
        let (wx, wy, ww, wh) = s.work_area;
        let w = r.w.min(ww.max(1));
        let h = r.h.min(wh.max(1));
        let x = r.x.max(wx).min(wx + ww as i32 - w as i32);
        let y = r.y.max(wy).min(wy + wh as i32 - h as i32);
        WindowRect { x, y, w, h }
    }

    /// DPI settle rebase (phase C): once the window SETTLES on a home surface whose scale differs from `window_scale` — drag release, resize release, restore-from-maximize, a home `ScaleFactorChanged` — re-anchor the geometry to the new scale so apparent size stays constant and the straddle passes collapse back to one.
    /// macOS: `window_rect` is in POINTS, so w/h don't change — apparent size is constant by construction and the rebase is just `window_scale` adoption + a pass-0 rebuild at the new density. X11/Windows: desktop units are physical px, so w/h scale by `home.scale ÷ window_scale` about the center, then the position clamps into the home work area. User zoom (`viewport.ru`) is untouched — it stays a separate multiplier by design.
    fn settle_rebase(&mut self) {
        let Some(hs) = self.surfaces.get(self.home).map(|s| s.scale) else {
            return;
        };
        if (hs - self.window_scale).abs() < 1e-6 {
            return;
        }
        #[cfg(not(target_os = "macos"))]
        {
            let r = hs / self.window_scale;
            let new_w = (((self.window_rect.w as f64) * r).round() as u32).max(1);
            let new_h = (((self.window_rect.h as f64) * r).round() as u32).max(1);
            self.window_rect.x += (self.window_rect.w as i32 - new_w as i32) / 2;
            self.window_rect.y += (self.window_rect.h as i32 - new_h as i32) / 2;
            self.window_rect.w = new_w;
            self.window_rect.h = new_h;
            // Position-only clamp into the home work area — the size was just chosen to preserve apparent size, so only the placement gets corrected.
            let (wx, wy, ww, wh) = self.surfaces[self.home].work_area;
            if ww > 1 && wh > 1 {
                self.window_rect.x = self
                    .window_rect
                    .x
                    .min(wx + ww as i32 - self.window_rect.w as i32)
                    .max(wx);
                self.window_rect.y = self
                    .window_rect
                    .y
                    .min(wy + wh as i32 - self.window_rect.h as i32)
                    .max(wy);
            }
        }
        self.window_scale = hs;
        self.rebuild_pass0_geometry();
        if let Some(window) = self.home_window() {
            window.request_redraw();
        }
    }

    /// Rebuild pass 0 (viewport + scratch + clip_mask) for the current `window_rect` under the current `window_scale`, notify the app via `on_resize`, mark the full repaint, and re-push the home input region. Shared tail of [`Self::settle_rebase`] and `toggle_maximized`'s scale sync; user zoom rides thru unchanged.
    fn rebuild_pass0_geometry(&mut self) {
        self.viewport = self.pass0_viewport(self.window_rect.w, self.window_rect.h, self.viewport.ru);
        let win_px = (self.viewport.width_px as usize) * (self.viewport.height_px as usize);
        self.scratch = vec![0u32; win_px];
        self.clip_mask = vec![255u8; win_px];
        self.pending_full_repaint = true;
        let (vw, vh) = (self.viewport.width_px, self.viewport.height_px);
        let (ccx, ccy) = self.win_cursor_px();
        let wo = self.ctx_window_origin();
        if let (Some(window), Some(text)) = (self.home_window(), self.text.as_mut()) {
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: ccx,
                cursor_y: ccy,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: wo,
                damage_clip: crate::canvas::PixelRect::new(0, 0, vw as usize, vh as usize),
            };
            self.app.on_resize(vw, vh, &mut ctx);
        }
        self.push_input_region(self.home);
    }

    /// The extra raster pass matching surface `si`'s scale, or `None` for pass 0 — the matching-scale case, the [`STRADDLE_DUAL_RASTER`]-off escape hatch, and the not-yet-built fallback all composite from pass 0.
    fn pass_for_surface(&self, si: usize) -> Option<usize> {
        if !STRADDLE_DUAL_RASTER {
            return None;
        }
        let sc = self.surfaces[si].scale;
        if (sc - self.window_scale).abs() < 1e-6 {
            return None;
        }
        self.extra_passes
            .iter()
            .position(|p| (p.scale - sc).abs() < 1e-6)
    }

    /// Render one FULL pass per distinct involved-surface scale that differs from `window_scale` (mixed-DPI straddle only; a no-op the rest of the time). Mirrors `apply_zoom_change`'s sequence per pass: `on_resize` the app into the pass viewport, render the whole frame into the pass scratch, and after all passes `on_resize` back to pass 0 so the app ends the frame at pass-0 geometry. Known cost: app-side cache thrash from the viewport swaps — that's what [`STRADDLE_DUAL_RASTER`] gates.
    fn render_extra_passes(&mut self) {
        if !STRADDLE_DUAL_RASTER {
            self.extra_passes.clear();
            return;
        }
        // Distinct non-pass-0 scales among the involved surfaces.
        let mut scales: Vec<f64> = Vec::new();
        for si in self.involved() {
            let sc = self.surfaces[si].scale;
            if (sc - self.window_scale).abs() > 1e-6 && !scales.iter().any(|s| (s - sc).abs() < 1e-6) {
                scales.push(sc);
            }
        }
        // Drop passes whose straddle ended.
        self.extra_passes
            .retain(|p| scales.iter().any(|s| (s - p.scale).abs() < 1e-6));
        if scales.is_empty() {
            return;
        }
        let user_ru = self.viewport.ru;
        let Some(window) = self.home_window() else {
            return;
        };
        for sc in scales {
            // Pass geometry — see [`RasterPass`]: same dims + ru multiplier on X11/Windows, points × scale dims + bare user ru on macOS.
            #[cfg(target_os = "macos")]
            let (pw, ph, pru) = (
                (((self.window_rect.w as f64) * sc).round() as u32).max(1),
                (((self.window_rect.h as f64) * sc).round() as u32).max(1),
                user_ru,
            );
            #[cfg(not(target_os = "macos"))]
            let (pw, ph, pru) = (
                self.window_rect.w,
                self.window_rect.h,
                user_ru * (sc / self.window_scale) as Coord,
            );
            let vp = Viewport::new(pw, ph).with_ru(pru);
            let px = (pw as usize) * (ph as usize);
            let pi = match self.extra_passes.iter().position(|p| (p.scale - sc).abs() < 1e-6) {
                Some(pi) => pi,
                None => {
                    self.extra_passes.push(RasterPass {
                        scale: sc,
                        viewport: vp,
                        scratch: Vec::new(),
                        clip_mask: Vec::new(),
                    });
                    self.extra_passes.len() - 1
                }
            };
            self.extra_passes[pi].viewport = vp;
            if self.extra_passes[pi].scratch.len() != px {
                self.extra_passes[pi].scratch = vec![0u32; px];
            } else {
                self.extra_passes[pi].scratch.fill(0);
            }
            // Reset to fully-visible before the swap-in `on_resize` — the app re-carves its window shape (corner cutouts) in its resize path.
            self.extra_passes[pi].clip_mask.clear();
            self.extra_passes[pi].clip_mask.resize(px, 255);
            // Cursor + origin in THIS pass's pixel density.
            #[cfg(target_os = "macos")]
            let pk = sc as Coord;
            #[cfg(not(target_os = "macos"))]
            let pk: Coord = 1.0;
            let ccx = (self.cursor_x - self.window_rect.x as Coord) * pk;
            let ccy = (self.cursor_y - self.window_rect.y as Coord) * pk;
            let wo = (
                ((self.window_rect.x as f64) * pk as f64).round() as i32,
                ((self.window_rect.y as f64) * pk as f64).round() as i32,
            );
            let full = crate::canvas::PixelRect::new(0, 0, pw as usize, ph as usize);
            let Some(text) = self.text.as_mut() else {
                return;
            };
            let RasterPass {
                scratch, clip_mask, ..
            } = &mut self.extra_passes[pi];
            // Swap-in: the app reflows to the pass viewport exactly like a zoom change (the apply_zoom_change precedent), then paints the whole frame at the pass density.
            {
                let mut ctx = Context {
                    pressed_hit: self.pointer.held_id(),
                    viewport: vp,
                    text: &mut *text,
                    clip_mask: clip_mask.as_mut_slice(),
                    damage: &mut self.pending_damage,
                    window: &*window,
                    modifiers: winit_compat::from_winit_mods(self.modifiers),
                    cursor_x: ccx,
                    cursor_y: ccy,
                    is_maximized: self.saved_rect_for_maximize.is_some(),
                    window_origin: wo,
                    damage_clip: full,
                };
                self.app.on_resize(pw, ph, &mut ctx);
            }
            {
                let mut ctx = Context {
                    pressed_hit: self.pointer.held_id(),
                    viewport: vp,
                    text: &mut *text,
                    clip_mask: clip_mask.as_mut_slice(),
                    damage: &mut self.pending_damage,
                    window: &*window,
                    modifiers: winit_compat::from_winit_mods(self.modifiers),
                    cursor_x: ccx,
                    cursor_y: ccy,
                    is_maximized: self.saved_rect_for_maximize.is_some(),
                    window_origin: wo,
                    damage_clip: full,
                };
                self.app.render(scratch.as_mut_slice(), &mut ctx);
            }
        }
        // Swap back: the composite reads pass 0 for matching-scale surfaces and the app must end the frame laid out at pass-0 geometry (its widget metadata feeds the pass-0 overlay + hit map).
        let (vw, vh) = (self.viewport.width_px, self.viewport.height_px);
        let (ccx, ccy) = self.win_cursor_px();
        let wo = self.ctx_window_origin();
        if let Some(text) = self.text.as_mut() {
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: ccx,
                cursor_y: ccy,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: wo,
                damage_clip: crate::canvas::PixelRect::new(0, 0, vw as usize, vh as usize),
            };
            self.app.on_resize(vw, vh, &mut ctx);
        }
    }

    /// Re-assert per-surface `ignoresMouseEvents` from current shell state: a surface accepts events only when the shell isn't in cursor-outside click-thru AND its last-pushed window∩surface input region is non-empty (dormant surfaces are always click-thru; the creation sentinel counts as empty).
    #[cfg(target_os = "macos")]
    fn apply_macos_hittest(&mut self) {
        let off = self.hittest_off;
        for s in self.surfaces.iter() {
            let empty = s.last_input_region.2 == 0 || s.last_input_region.3 == 0;
            let _ = s.window.set_cursor_hittest(!(off || empty));
        }
    }

    /// Push the click-thru input region for surface `si`: the window ∩ surface intersection in surface-local desktop units, `(0, 0, 0, 0)` (fully click-thru) when they don't overlap. Deduped against `last_input_region` so repeated pushes with unchanged geometry cost nothing. Replaces every direct `x11_atomic::set_input_region` call site. On macOS the equivalent is per-window `ignoresMouseEvents` (there's no sub-window shape): an empty region turns the surface fully click-thru, a non-empty one accepts events unless the shell's cursor-outside state (`hittest_off`) is engaged. No-op on Windows (the layered window's α channel routes clicks already).
    fn push_input_region(&mut self, si: usize) {
        #[cfg(target_os = "macos")]
        let hittest_off = self.hittest_off;
        let Some(s) = self.surfaces.get_mut(si) else {
            return;
        };
        let r = &self.window_rect;
        // Inflate the hittable region by the resize band so the border extends INVISIBLY past the window edge — the CSD convention. With the region exactly the window rect, a cursor approaching an edge from OUTSIDE parked on pixels that received no events at all (they pass thru to the window below), so resize could only ever be grabbed from inside ("moving mouse in from outside never lets you resize"). get_resize_edge already classifies just-outside (negative window-relative) coords as the matching edge, and hit_at bounds-checks, so the only missing piece was the region. Same band as the classifier: strip_height/4.
        let band = (super::chrome::strip_height(self.viewport) / 4.0).ceil() as i32;
        let ix0 = (r.x - band).max(s.origin.0);
        let iy0 = (r.y - band).max(s.origin.1);
        let ix1 = (r.x + r.w as i32 + band).min(s.origin.0 + s.size.0 as i32);
        let iy1 = (r.y + r.h as i32 + band).min(s.origin.1 + s.size.1 as i32);
        let region = if ix0 < ix1 && iy0 < iy1 {
            (
                ix0 - s.origin.0,
                iy0 - s.origin.1,
                (ix1 - ix0) as u32,
                (iy1 - iy0) as u32,
            )
        } else {
            (0, 0, 0, 0)
        };
        if region == s.last_input_region {
            return;
        }
        s.last_input_region = region;
        #[cfg(target_os = "linux")]
        x11_atomic::set_input_region(&s.window, region.0, region.1, region.2, region.3);
        #[cfg(target_os = "macos")]
        {
            let empty = region.2 == 0 || region.3 == 0;
            let _ = s.window.set_cursor_hittest(!(hittest_off || empty));
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = region;
    }

    /// Create one fullscreen transparent OS surface pinned to `monitor` — window attributes as the old single-window `resumed` PLUS the monitor-origin position, softbuffer/wgpu at monitor size, work area from the (origin, size) signature. We deliberately avoid `with_fullscreen` — it triggers the WM's animated transition (default-window-size → grow → fullscreen) which makes the chrome appear to scale up from the top-left. Instead, ask for a plain borderless transparent window covering the monitor: the WM creates it at the requested geometry directly, no animation.
    fn create_monitor_surface(
        &mut self,
        event_loop: &ActiveEventLoop,
        monitor: winit::monitor::MonitorHandle,
        is_anchor: bool,
    ) -> MonitorSurface {
        let scale = monitor.scale_factor();
        // Desktop units: winit reports monitor position/size in PHYSICAL px on every platform. On macOS the native space is POINTS and winit derives the physical values FROM the point layout by multiplying by scale, so dividing back is an exact recovery of the point rect. Everywhere else desktop units ARE physical px.
        #[cfg(target_os = "macos")]
        let (origin, size) = (
            (
                ((monitor.position().x as f64) / scale).round() as i32,
                ((monitor.position().y as f64) / scale).round() as i32,
            ),
            (
                ((monitor.size().width.max(1) as f64) / scale).round().max(1.0) as u32,
                ((monitor.size().height.max(1) as f64) / scale).round().max(1.0) as u32,
            ),
        );
        #[cfg(not(target_os = "macos"))]
        let (origin, mut size) = (
            (monitor.position().x, monitor.position().y),
            (monitor.size().width.max(1), monitor.size().height.max(1)),
        );
        // Linux/X11: undersize the surface by ONE pixel row. An undecorated window EXACTLY monitor-sized gets auto-promoted to legacy FULLSCREEN by Mutter-family WMs (Muffin 6.4 verified live 2026-07-25: _NET_WM_STATE grew FULLSCREEN unrequested) — fullscreen-layer stacking sits ABOVE the panel, burying the taskbar (the "black taskbar" lockup; _NET_WM_BYPASS_COMPOSITOR=2 and the unredirect gsetting were both red herrings). h−1 defeats the exact-size match; the missing row is imperceptible — maximize targets the work area, and the bottom row usually sits over the panel anyway.
        #[cfg(target_os = "linux")]
        {
            size.1 = size.1.saturating_sub(1).max(1);
        }
        #[cfg(target_os = "macos")]
        let pixel_ratio = scale;
        #[cfg(not(target_os = "macos"))]
        let pixel_ratio = 1.0f64;

        // Title (and later the icon) only on the anchor surface — the taskbar/alt-tab/cmd-tab shows ONE entry for the app, not one per monitor.
        let title = if is_anchor {
            self.app.title().to_string()
        } else {
            String::new()
        };
        let attrs = WindowAttributes::default()
            .with_title(title)
            .with_decorations(false)
            .with_transparent(true)
            .with_visible(!self.app.start_hidden())
            .with_resizable(false);
        // Geometry in the platform's request currency: macOS takes LOGICAL (point) values so the window lands exactly on its NSScreen regardless of per-monitor scale; X11/Windows take physical px.
        #[cfg(target_os = "macos")]
        let attrs = attrs
            .with_inner_size(winit::dpi::LogicalSize::new(size.0 as f64, size.1 as f64))
            .with_position(winit::dpi::LogicalPosition::new(
                origin.0 as f64,
                origin.1 as f64,
            ));
        #[cfg(not(target_os = "macos"))]
        let attrs = attrs
            .with_inner_size(winit::dpi::PhysicalSize::new(size.0, size.1))
            .with_position(winit::dpi::PhysicalPosition::new(origin.0, origin.1));
        let window = Arc::new(event_loop.create_window(attrs).expect("create_window"));
        // Pin the surface to its monitor's origin post-create — some WMs apply their own placement to the pre-map with_position request, and the compositor model requires the surface to sit exactly on its monitor.
        #[cfg(target_os = "macos")]
        window.set_outer_position(winit::dpi::LogicalPosition::new(
            origin.0 as f64,
            origin.1 as f64,
        ));
        #[cfg(not(target_os = "macos"))]
        window.set_outer_position(winit::dpi::PhysicalPosition::new(origin.0, origin.1));

        // A monitor-sized window is exactly what compositors unredirect for performance — and unredirected means no alpha, so our transparent pixels (including the strip over the taskbar) render opaque black. Opt out permanently.
        #[cfg(target_os = "linux")]
        x11_atomic::set_never_bypass_compositor(&window);

        #[cfg(target_os = "macos")]
        {
            use winit::platform::macos::WindowExtMacOS;
            window.set_has_shadow(false);
        }

        // Windows: make the OS window LAYERED so UpdateLayeredWindow can present per-pixel alpha (and route clicks thru the α=0 region). winit's `with_transparent(true)` alone gives an opaque softbuffer surface on Windows — the layered style is what the fullscreen compositor needs.
        #[cfg(target_os = "windows")]
        super::windows_layered::make_layered(&window);

        // Per-monitor work area (monitor minus panels/taskbars/menu-bar+Dock), in GLOBAL desktop units, so the visible window — and especially its bottom chrome status band — doesn't end up under a taskbar.
        let work_area = monitor_work_area(origin, size);

        // Renderer sized to BACKING pixels (points × scale = the raw physical size winit reported). It owns an Arc of the window now — the old `&'static` transmute is gone.
        #[cfg(target_os = "macos")]
        let renderer = Some(super::renderer_wgpu::Renderer::new(
            window.clone(),
            ((size.0 as f64) * scale).round().max(1.0) as u32,
            ((size.1 as f64) * scale).round().max(1.0) as u32,
        ));
        // Windows presents via UpdateLayeredWindow from `persistent_screen` directly (softbuffer's BitBlt present is opaque), so it needs no softbuffer surface. Every other non-macOS target (Linux/X11, Redox/Orbital) uses softbuffer.
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        let surface = {
            use std::num::NonZeroU32;
            let context =
                softbuffer::Context::new(window.clone()).expect("softbuffer Context::new");
            let mut surface = softbuffer::Surface::new(&context, window.clone())
                .expect("softbuffer Surface::new");
            surface
                .resize(
                    NonZeroU32::new(size.0).expect("nonzero screen width"),
                    NonZeroU32::new(size.1).expect("nonzero screen height"),
                )
                .expect("softbuffer Surface::resize");
            Some(surface)
        };
        #[cfg(target_os = "windows")]
        let surface = None;

        // Persistent buffer at backing pixels (== desktop units on X11/Windows, points × scale on macOS).
        let bw = ((size.0 as f64) * pixel_ratio).round().max(1.0) as usize;
        let bh = ((size.1 as f64) * pixel_ratio).round().max(1.0) as usize;
        // Multi-monitor diagnostic (rides the consumer's log bridge into the pullable app log): the raw winit monitor geometry AND the derived desktop-unit rect. The macOS cross-monitor vanish investigation hinges on whether these origins land in ONE coherent point space when the two monitors carry different scales — a wrong origin makes involved() miss the second surface forever.
        log::info!(
            "FLUOR-MON: surface for {:?} anchor={} raw_pos=({},{}) raw_size={}x{} scale={} → origin=({},{}) size={}x{} ratio={} work_area={:?}",
            monitor.name(), is_anchor, monitor.position().x, monitor.position().y,
            monitor.size().width, monitor.size().height, scale,
            origin.0, origin.1, size.0, size.1, pixel_ratio, work_area
        );
        MonitorSurface {
            window,
            monitor,
            origin,
            size,
            scale,
            pixel_ratio,
            work_area,
            #[cfg(target_os = "macos")]
            renderer,
            #[cfg(not(target_os = "macos"))]
            surface,
            persistent_screen: vec![0u32; bw * bh],
            surface_ready: false,
            dormant: false,
            focused: false,
            needs_full_blit: false,
            // Sentinel that can never equal a computed region (x is never negative there), so the first push always reaches the OS.
            last_input_region: (i32::MIN, i32::MIN, 0, 0),
        }
    }

    /// Surface `si`'s OS window resized. In the fullscreen-compositor architecture the OS surface is the whole monitor — `size` is that surface's size, not the consumer-visible viewport. WMs commonly fire Resized multiple times during fullscreen activation (default-window-size → animating → final fullscreen); each tick we resize the surface buffers to match, re-centre the visible window inside the new bounds (when it lives on this surface), and re-issue the input region. Suppresses the "chrome appears in the top-left of a growing window" artefact during WM fullscreen animations.
    fn handle_surface_resized(&mut self, si: usize, size: winit::dpi::PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        // Windows reports minimize as a Resized with the caption-stub geometry (~160×24), NOT 0×0 — adopting it clamps window_rect down to the stub, and the restore's Resized then "preserves the user's current size" at that clamped stub: the restore-from-minimize super-tiny-window bug. A minimized window has no visible surface to size against; ignore the event wholesale (is_minimized is None where the platform can't say, which safely falls thru).
        if self.surfaces[si].window.is_minimized() == Some(true) {
            log::info!("FLUOR: Resized {}x{} while minimized — ignored", size.width, size.height);
            return;
        }
        if size.width == self.surfaces[si].size.0
            && size.height == self.surfaces[si].size.1
            && self.surfaces[si].surface_ready
        {
            return;
        }
        // Convert the OS physical size to desktop units — points on macOS (physical ÷ scale is exact: winit derives physical FROM the point layout, so the division inverts it), physical px everywhere else.
        #[cfg(target_os = "macos")]
        let unit_size = {
            let pr = self.surfaces[si].pixel_ratio.max(0.5);
            (
                ((size.width as f64) / pr).round().max(1.0) as u32,
                ((size.height as f64) / pr).round().max(1.0) as u32,
            )
        };
        #[cfg(not(target_os = "macos"))]
        let unit_size = (size.width, size.height);
        // On macOS a scale change can keep the POINT size identical while the backing size doubles/halves — the unit-size equality alone must not skip the renderer resize below.
        #[cfg(target_os = "macos")]
        let backing_matches = {
            let (bw, bh) = self.surfaces[si].backing();
            bw == size.width as usize && bh == size.height as usize
        };
        #[cfg(not(target_os = "macos"))]
        let backing_matches = true;
        if unit_size == self.surfaces[si].size && backing_matches && self.surfaces[si].surface_ready
        {
            return;
        }
        // Sample readiness BEFORE mutating — the pre-ready branch below re-derives the initial size against the first real geometry.
        let was_ready = self.surfaces[si].surface_ready;

        self.surfaces[si].size = unit_size;

        // Platform present buffers stay at BACKING pixels — the raw physical size the OS just reported.
        #[cfg(target_os = "macos")]
        if let Some(renderer) = self.surfaces[si].renderer.as_mut() {
            renderer.resize(size.width, size.height);
        }
        #[cfg(not(target_os = "macos"))]
        if let Some(surface) = self.surfaces[si].surface.as_mut() {
            use std::num::NonZeroU32;
            if let (Some(w), Some(h)) =
                (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
            {
                surface.resize(w, h).expect("softbuffer Surface::resize");
            }
        }
        // Match the persistent buffer to the new backing geometry; new pixels start at 0 which is fine — the next finalize populates them (composite's lazy length check backstops this).
        let (bw, bh) = self.surfaces[si].backing();
        let scr_px = bw * bh;
        if self.surfaces[si].persistent_screen.len() != scr_px {
            self.surfaces[si].persistent_screen.resize(scr_px, 0);
        }
        // Re-query the work area against the refreshed monitor rect — panels/struts may have moved with the mode change.
        self.surfaces[si].work_area =
            monitor_work_area(self.surfaces[si].origin, self.surfaces[si].size);

        // Dormant surface (phase B): confirm geometry, present one all-zero frame so the fresh transparent window never shows back-buffer garbage, and stop — no window re-clamp, no render tick.
        if self.surfaces[si].dormant {
            self.surfaces[si].surface_ready = true;
            self.surfaces[si].persistent_screen.fill(0);
            self.present_surface_raw(si);
            return;
        }

        // Re-centre + clamp window_rect within this surface's GLOBAL rect on every surface-size change (initial fullscreen, monitor switch, etc.) — but only when the window actually lives on this surface (phase A: it always does, there's one surface). Skip during an active drag — the user is steering the rect themselves.
        //
        // SIZE comes from the app: on the FIRST real surface (before surface_ready) we (re)apply `FluorApp::initial_size` now that the true screen size is known — `resumed` set it against the monitor we *expected*, and Windows in particular reports a different size here (DPI virtualization), so deriving it again keeps the app's aspect (e.g. Photon's tall portrait window) instead of the old hardcoded screen/2 that made the window "supa fat". On LATER resizes we PRESERVE the current window size (the user may have resized it) and only re-centre + clamp.
        let origin = self.surfaces[si].origin;
        let (uw, uh) = unit_size;
        let intersects = {
            let r = &self.window_rect;
            r.x < origin.0 + uw as i32
                && r.x + r.w as i32 > origin.0
                && r.y < origin.1 + uh as i32
                && r.y + r.h as i32 > origin.1
        };
        // Re-clamp only when the HOME surface changed under the window — a mode change on a non-home surface must not yank a straddling window off its home (phase-D rotation handles cross-surface geometry refresh properly). A never-ready home hasn't confirmed geometry yet, so the intersect test against the stale rect doesn't gate the initial placement.
        if !self.is_dragging_resize
            && !self.is_dragging_move
            && si == self.home
            && (intersects || !was_ready)
        {
            let (new_w, new_h) = if !was_ready {
                let (rw, rh) = self.app.initial_size((uw, uh));
                (rw.max(1).min(uw), rh.max(1).min(uh))
            } else {
                (
                    self.window_rect.w.max(1).min(uw),
                    self.window_rect.h.max(1).min(uh),
                )
            };
            let new_x = origin.0 + ((uw as i32) - (new_w as i32)) / 2;
            let new_y = origin.1 + ((uh as i32) - (new_h as i32)) / 2;
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
                self.update_home();
                self.viewport = self.pass0_viewport(new_w, new_h, self.viewport.ru);
                let win_px = (self.viewport.width_px as usize) * (self.viewport.height_px as usize);
                self.scratch = vec![0u32; win_px];
                self.clip_mask = vec![255u8; win_px];
                // Surface-driven resize → window geometry changed → full repaint required.
                self.pending_full_repaint = true;
                let (vw, vh) = (self.viewport.width_px, self.viewport.height_px);
                let (ccx, ccy) = self.win_cursor_px();
                let wo = self.ctx_window_origin();
                if let (Some(window), Some(text)) = (self.home_window(), self.text.as_mut()) {
                    let mut ctx = Context {
                        pressed_hit: self.pointer.held_id(),
                        viewport: self.viewport,
                        text,
                        clip_mask: &mut self.clip_mask,
                        damage: &mut self.pending_damage,
                        window: &*window,
                        modifiers: winit_compat::from_winit_mods(self.modifiers),
                        cursor_x: ccx,
                        cursor_y: ccy,
                        damage_clip: crate::canvas::PixelRect::new(0, 0, vw as usize, vh as usize),
                        is_maximized: self.saved_rect_for_maximize.is_some(),
                        window_origin: wo,
                    };
                    self.app.on_resize(vw, vh, &mut ctx);
                }
                self.push_input_region(si);
            }
        }

        // First Resized confirms the OS surface is actually allocated — safe to start painting.
        self.surfaces[si].surface_ready = true;
        self.render_frame();
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
            if should_ignore {
                if let Some(window) = self.home_window() {
                    window.set_cursor(winit::window::CursorIcon::Default);
                }
            }
            self.hittest_off = should_ignore;
            // Fan the flip out to every surface — each combines the shell-level cursor-outside state with its own region emptiness.
            self.apply_macos_hittest();
        }
    }

    fn render_frame(&mut self) {
        if !self.surfaces.get(self.home).is_some_and(|s| s.surface_ready) {
            return;
        }
        let Some(window) = self.home_window() else {
            return;
        };
        // Phase B: settle surface involvement BEFORE painting — wake newly-entered surfaces (their needs_full_blit forces a fresh composite) and evacuate just-left ones (zeroed, presented once, click-thru).
        self.refresh_involvement();
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
            // The per-surface buffer wipe happens in `composite_and_present` (each surface owns its persistent_screen now); the overlay bookkeeping is shell-level and resets here.
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

        // Scroll-copy hint for this frame: the app declares a rigidly-translated region so `composite_and_present` can memmove persistent_screen instead of forcing a full repaint. Only meaningful on an INCREMENTAL frame — a full repaint wipes and redraws everything, so there's nothing to copy forward. Queried here (before the text borrow) so the per-surface composite loop below can apply it.
        let scroll_hint = if full_repaint {
            None
        } else {
            self.app.scroll_hint(self.viewport)
        };

        clear_scratch_rect(&mut self.scratch, win_w, damage_clip);

        let (ccx, ccy) = self.win_cursor_px();
        let wo = self.ctx_window_origin();
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
            cursor_x: ccx,
            cursor_y: ccy,
            is_maximized: self.saved_rect_for_maximize.is_some(),
            window_origin: wo,
            damage_clip,
        };

        // Per-stage stopwatches. Each Instant brackets one pipeline stage; the strip displays each as FPS so toggling SIMD/Rayon shows which stage actually moves. `buffer.present()` is excluded everywhere because it blocks for vsync, which would pin every reading to the display refresh rate. The composite stages time themselves inside `composite_and_present` and return their readings.
        let app_start = Instant::now();
        self.app.render(&mut self.scratch, &mut ctx);
        drop(ctx);
        let app_dt = app_start.elapsed().as_secs_f32();

        // Phase C: mixed-DPI straddle — one extra full pass per involved scale that differs from window_scale, app swapped back to pass-0 geometry afterwards.
        self.render_extra_passes();

        // Composite + present EVERY involved surface (phase B); each picks the raster pass matching its scale (phase C). Stage timings sum across surfaces for the debug strip.
        let mut fill_dt = 0.0f32;
        let mut finalize_dt = 0.0f32;
        let mut shadow_dt = 0.0f32;
        for si in self.involved() {
            let (f1, f2, f3) =
                self.composite_and_present(si, damage_clip, full_repaint, outline_active, scroll_hint);
            fill_dt += f1;
            finalize_dt += f2;
            shadow_dt += f3;
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

    /// Composite the finished window scratch onto surface `si` and present it: finalize into the surface's persistent buffer at the per-surface blit origin (`window_rect.xy − origin`), cast the drop shadow, run the overlay + FPS-strip passes with the same translated coords, then push the buffer thru the platform present path. Returns the (fill, finalize, shadow) stage timings for the debug stats. Phase A calls this once with the home surface; phase B calls it per involved surface.
    fn composite_and_present(
        &mut self,
        si: usize,
        damage_clip: crate::canvas::PixelRect,
        full_repaint: bool,
        outline_active: bool,
        scroll_hint: Option<(crate::canvas::PixelRect, i32)>,
    ) -> (f32, f32, f32) {
        // Pass selection (phase C): composite from the raster pass matching this surface's scale; pass 0 (the shell's viewport/scratch/clip_mask trio) when the scale matches `window_scale`, when dual-raster is off, or when the pass isn't built — the documented fallback.
        let pass = self.pass_for_surface(si);
        let src_vp = match pass {
            Some(pi) => self.extra_passes[pi].viewport,
            None => self.viewport,
        };
        let win_w = src_vp.width_px as usize;
        let win_h = src_vp.height_px as usize;
        // Overlay + FPS-strip run only when this surface composites at true pass-0 density — the hit map and strip staging live at pass-0 dims, so a mismatched-scale surface (extra pass, or the escape-hatch fallback) skips them.
        let pass_primary =
            pass.is_none() && (self.surfaces[si].scale - self.window_scale).abs() < 1e-6;
        // Extra passes are full re-renders with no incremental damage bookkeeping: composite them full-blit over their whole pass rect.
        let (damage_clip, full_repaint) = if pass.is_some() {
            (crate::canvas::PixelRect::new(0, 0, win_w, win_h), true)
        } else {
            (damage_clip, full_repaint)
        };
        let hitmask_now =
            crate::paint::DEBUG_SHOW_HITMASK.load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(feature = "text")]
        let strip_active = crate::paint::DEBUG_SHOW_FPS.load(std::sync::atomic::Ordering::Relaxed);
        // Per-surface full repaint: the frame-level flag OR this surface's one-shot needs_full_blit (set when the window enters a surface in phase B); the one-shot is consumed here.
        let first_after_wake = self.surfaces[si].needs_full_blit;
        let full_repaint = full_repaint || first_after_wake;
        self.surfaces[si].needs_full_blit = false;

        let (scr_w, scr_h) = self.surfaces[si].backing();
        let pr = self.surfaces[si].pixel_ratio;
        // The wake composite is the make-or-break moment of a cross-monitor drag — log its geometry once so a remote repro (macbook) shows whether the blit landed on-buffer.
        if first_after_wake {
            log::info!("FLUOR-MON: surface {} first composite after wake: blit_origin=({},{}) win_px={}x{} backing={}x{} ratio={}", si, ((self.window_rect.x - self.surfaces[si].origin.0) as f64 * pr).round() as i32, ((self.window_rect.y - self.surfaces[si].origin.1) as f64 * pr).round() as i32, ((self.window_rect.w as f64) * pr).round() as i32, ((self.window_rect.h as f64) * pr).round() as i32, scr_w, scr_h, pr);
        }
        // Per-surface blit origin in surface BACKING pixels: the window's GLOBAL desktop-unit rect translated into this surface's local space, then × pixel_ratio under the ONE unit→backing rounding convention (multiply, then `.round()`, ties away from zero) — every conversion in this function uses it.
        let rect_x = (((self.window_rect.x - self.surfaces[si].origin.0) as f64) * pr).round() as i32;
        let rect_y = (((self.window_rect.y - self.surfaces[si].origin.1) as f64) * pr).round() as i32;
        // The window's backing-pixel footprint on this surface (shadow geometry).
        let px_w = ((self.window_rect.w as f64) * pr).round() as i32;
        let px_h = ((self.window_rect.h as f64) * pr).round() as i32;
        // Source pass buffers — plain field-path borrows so the `&mut self.surfaces[..]` borrows below stay disjoint.
        let (src_scratch, src_clip): (&[u32], &[u8]) = match pass {
            Some(pi) => (
                &self.extra_passes[pi].scratch,
                &self.extra_passes[pi].clip_mask,
            ),
            None => (&self.scratch, &self.clip_mask),
        };

        // Persistent screen lives across frames so the post-finalize overlay (blinkey) can mutate just a few pixels each frame without re-running finalize. Resize on surface-size change; new pixels start at 0 which is fine — finalize on the next render will populate them.
        let scr_px = scr_w * scr_h;
        let t = Instant::now();
        if self.surfaces[si].persistent_screen.len() != scr_px {
            self.surfaces[si].persistent_screen.resize(scr_px, 0);
        }
        // A full repaint wipes this surface's buffer so finalize copies every pixel and paint_shadow casts into a known-zero band (the wipe lived at the top of render_frame when the buffer was shell-owned).
        if full_repaint {
            self.surfaces[si].persistent_screen.fill(0);
        }
        let fill_dt = t.elapsed().as_secs_f32();

        // Scroll-copy: before finalize, memmove the rigidly-translated region of persistent_screen so only the exposed band (in `damage_clip`) needs finalizing. Gated on `!full_repaint` — a full repaint (frame-level, first_after_wake, or an extra pass) wipes and redraws the surface, so there's nothing to carry forward. The hint is in viewport pixels; translate the rect corners and the delta into this surface's BACKING pixels with the same `(rect_x, rect_y)` blit origin + `pr` scale finalize uses, so the copy lands exactly where the finalized pixels will.
        if !full_repaint {
            if let Some((rect, dy)) = scroll_hint {
                let bx0 = (rect_x + (rect.x0 as f64 * pr).round() as i32).max(0) as usize;
                let by0 = (rect_y + (rect.y0 as f64 * pr).round() as i32).max(0) as usize;
                let bx1 = (rect_x + (rect.x1 as f64 * pr).round() as i32).max(0) as usize;
                let by1 = (rect_y + (rect.y1 as f64 * pr).round() as i32).max(0) as usize;
                let dy_backing = (dy as f64 * pr).round() as i32;
                crate::paint::scroll_copy_rect(
                    &mut self.surfaces[si].persistent_screen,
                    scr_w,
                    scr_h,
                    bx0,
                    by0,
                    bx1,
                    by1,
                    dy_backing,
                );
            }
        }

        // Debug fade: saturating-subtract `FADE_STEP` from every persistent_screen RGB byte. Runs BEFORE finalize so pixels that finalize / overlay / strip overwrite this frame land at full brightness while pixels that nobody touches visibly decay toward black — diagnoses whether the incremental opaque-scan finalize is actually copying the regions it should. Skipped on full_repaint since persistent_screen is being wiped anyway.
        let fade_active = crate::paint::DEBUG_SHOW_FADE.load(std::sync::atomic::Ordering::Relaxed);
        if fade_active && !full_repaint {
            const FADE_STEP: u8 = 4;
            for px in self.surfaces[si].persistent_screen.iter_mut() {
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
                src_scratch,
                src_clip,
                win_w,
                win_h,
                &mut self.surfaces[si].persistent_screen,
                scr_w,
                rect_x,
                rect_y,
                damage_clip,
                full_repaint,
            );
        }
        let finalize_dt = t.elapsed().as_secs_f32();

        // Drop shadow runs ONCE per full repaint, into a known-cleared band (persistent_screen.fill(0) above). Never runs on incremental frames — the perimeter AA pixels with their shadow contribution were preserved by the opaque-only finalize, and the shadow band cells outside the window were not touched either, so the shadow visible from the last full repaint is still correct. Skipped when hitmask debug is on so the band doesn't disturb the raw hit-id view at the chrome edge, and skipped when maximized because there's nothing outside the window to cast onto — the OS surface already covers the screen.
        let t = Instant::now();
        if full_repaint && !hitmask_now && self.saved_rect_for_maximize.is_none() {
            // Span of the pass this surface composites from, so the shadow radius matches the density it casts over.
            let span = src_vp.effective_span();
            let target_radius = span / 16.0;
            let drop = (1240.0 / target_radius) as u32;
            let factor_256 = (256u32.saturating_sub(drop)).clamp(96, 254);
            let shadow_seed: u32 = if self.is_focused { 0x80 } else { 0x40 };
            // Shadow casts in surface-local BACKING pixels — the same translated blit origin finalize used, with the window footprint scaled by pixel_ratio.
            let rect_for_shadow = (rect_x, rect_y, px_w, px_h);
            crate::paint::paint_shadow(
                &mut self.surfaces[si].persistent_screen,
                scr_w,
                factor_256,
                shadow_seed,
                rect_for_shadow,
            );
        }
        let shadow_dt = t.elapsed().as_secs_f32();

        // Post-finalize, post-shadow overlay pass. For each pixel whose hit id is currently tinted OR was tinted last frame, copy the scratch pixel → XOR to visible → optionally wrap-sub the per-id delta → write to persistent_screen. Restores the scratch baseline on unhover and applies the tint on hover — no diff math, no accumulation, just "copy and conditionally adjust." Runs every frame regardless of damage_clip so hover tints follow the cursor even when nothing else dirtied scratch.
        //
        // Order matters: [`FluorApp::overlay_deltas`] takes `&mut self` (so the app can walk its widget tree), so we build the table first and release the borrow before grabbing the shared `hit_test_map` borrow used by `apply_overlay`.
        // The whole overlay walk is pass-0-only (`pass_primary`) — the hit map is stamped at pass-0 dims, so running it against an extra pass would misindex. Mid-straddle the far-density half simply skips hover tints for those frames (it's re-rendered fully each frame anyway).
        if pass_primary {
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
                    src_scratch,
                    &mut self.surfaces[si].persistent_screen,
                    scr_w,
                    rect_x,
                    rect_y,
                    map,
                    hw,
                    hh,
                    &current,
                    &bboxes,
                    &mut self.last_overlay_active,
                );
            }
        }

        // FPS strip: drawn LAST, clobber-style, into a DEDICATED staging buffer (`self.strip_buf`) — never touches the app's scratch or clip_mask. After rasterizing α + darkness into the staging buffer we XOR → visible RGB, force α=0xFF, and clobber-write into persistent_screen at the strip rect. The whole pass is bracketed by a snapshot+restore of `RASTERIZE_OPS` so the strip's text/rect rasterizers don't bump the R counter or pollute `damage_pct`. Does NOT contribute to `damage_clip`, does NOT trigger paint_shadow. Toggle on/off promotes to full repaint via the transition detector above so the strip-rect underlying pixels get correctly restored when it disappears.
        #[cfg(feature = "text")]
        if strip_active && pass_primary {
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

            // Clobber `strip_buf` rows into this surface's persistent_screen at the surface-local `(rect_x, rect_y + strip_y_in_window)`. Per-pixel: XOR α + darkness → visible RGB, force α=0xFF.
            let rect_y_top = rect_y + strip_y_in_window as i32;
            let ps = &mut self.surfaces[si].persistent_screen;
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
                    ps[ps_row + scr_x] = 0xFF00_0000 | (v & 0x00FF_FFFF);
                }
            }
        }

        // Copy persistent_screen → platform back buffer (whichever softbuffer/wgpu hands us this frame; it may be stale or rotated, but we always overwrite the whole thing from our owned persistent_screen). The damage outline overlay is stamped AFTER this copy and BEFORE present so it lives for exactly one frame and never touches persistent_screen.
        #[cfg(target_os = "macos")]
        {
            let s = &mut self.surfaces[si];
            let Some(renderer) = s.renderer.as_mut() else {
                return (fill_dt, finalize_dt, shadow_dt);
            };
            // First composite after a wake (cross-monitor move / window entering a dormant surface):
            // the renderer's wgpu surface was `configure`d back at creation time, while this window
            // was dormant/off-screen, so its CAMetalLayer drawable is stale — `get_current_texture`
            // hands back a black drawable and every later present inherits it (root cause of the
            // "move to the other monitor → solid black, stays black" field report). Re-configure the
            // surface to its current backing here so the layer re-establishes a live drawable bound
            // to the monitor it's actually showing on. One-shot: `first_after_wake` was consumed above.
            if first_after_wake {
                renderer.resize(scr_w as u32, scr_h as u32);
            }
            let mut buffer = renderer.lock_buffer();
            // A transient mismatch (scale change announced before the matching Resized lands) skips one present rather than panicking in copy_from_slice.
            if buffer.len() != s.persistent_screen.len() {
                return (fill_dt, finalize_dt, shadow_dt);
            }
            buffer.copy_from_slice(&s.persistent_screen);
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
            // Update the global mouse monitor's window rect for re-entry detection — GLOBAL desktop points now (the monitor flips NSEvent's bottom-left mouseLocation against the primary screen height, landing in the same global top-left point space window_rect lives in).
            if let Some(ref monitor) = self.hittest_monitor {
                let r = &self.window_rect;
                monitor.update_rect(r.x, r.y, r.w, r.h);
            }
        }
        // Windows: present the owned screen buffer thru the layered window (per-pixel alpha + click-thru on α=0). The damage outline (a dev overlay) is stamped into a scratch copy first so it lives one frame and never touches persistent_screen, matching the softbuffer path.
        #[cfg(target_os = "windows")]
        {
            let s = &self.surfaces[si];
            let (sw, sh) = (scr_w as u32, scr_h as u32);
            if outline_active && !damage_clip.is_empty() {
                let mut scratch_screen = s.persistent_screen.clone();
                crate::paint::stamp_damage_outline_visible(
                    &mut scratch_screen,
                    scr_w,
                    scr_h,
                    damage_clip,
                    rect_x,
                    rect_y,
                );
                super::windows_layered::present(&s.window, &scratch_screen, sw, sh);
            } else {
                super::windows_layered::present(&s.window, &s.persistent_screen, sw, sh);
            }
        }
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            let s = &mut self.surfaces[si];
            let Some(surface) = s.surface.as_mut() else {
                return (fill_dt, finalize_dt, shadow_dt);
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            buffer.copy_from_slice(&s.persistent_screen);
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

        (fill_dt, finalize_dt, shadow_dt)
    }

    /// Drag-tick fast path: shift the screen buffer in place by the delta since the last paint, push the input region update, and present. Skips consumer render, scratch fill, finalize, and shadow rasterization entirely — the existing chrome pixels just slide thru the screen buffer, with anything that falls off any edge wrapping to the opposite side. On drag release, a normal `render_frame` overwrites the wrap artefacts in one clean frame.
    fn apply_move_drag_shift(&mut self) {
        let dx = self.window_rect.x - self.last_painted_rect.x;
        let dy = self.window_rect.y - self.last_painted_rect.y;
        if dx == 0 && dy == 0 {
            return;
        }
        let si = self.home;
        let Some(s) = self.surfaces.get_mut(si) else {
            return;
        };
        let (scr_w, scr_h) = s.backing();
        // Unit delta → backing-pixel delta under the standard rounding convention; on X11/Windows pixel_ratio is 1 so this is the old integer delta.
        let dxp = ((dx as f64) * s.pixel_ratio).round() as i32;
        let dyp = ((dy as f64) * s.pixel_ratio).round() as i32;
        #[cfg(target_os = "macos")]
        {
            let Some(renderer) = s.renderer.as_mut() else {
                return;
            };
            let mut buffer = renderer.lock_buffer();
            if buffer.len() != scr_w * scr_h {
                return;
            }
            crate::paint::shift_screen_wrap(&mut buffer, scr_w, scr_h, dxp, dyp);
            let _ = buffer.present();
        }
        // Windows: no softbuffer surface — shift the surface's owned persistent_screen and re-present it thru the layered window. (The layered window already moves with window_rect via the α channel, so there's no OS input-region call to push like X11 does below.)
        #[cfg(target_os = "windows")]
        {
            crate::paint::shift_screen_wrap(&mut s.persistent_screen, scr_w, scr_h, dxp, dyp);
            super::windows_layered::present(
                &s.window,
                &s.persistent_screen,
                scr_w as u32,
                scr_h as u32,
            );
        }
        #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
        {
            let Some(surface) = s.surface.as_mut() else {
                return;
            };
            let mut buffer = surface.buffer_mut().expect("softbuffer buffer_mut");
            crate::paint::shift_screen_wrap(&mut buffer, scr_w, scr_h, dxp, dyp);
            buffer.present().expect("softbuffer buffer.present");
        }
        self.push_input_region(si);
        self.last_painted_rect = self.window_rect;
    }

    /// Apply an [`EventResponse`] returned from `app.on_event`. Returns `true` if the response was `Close` (caller should terminate).
    fn apply_response(&mut self, response: EventResponse) -> bool {
        let Some(window) = self.home_window() else {
            return false;
        };
        match response {
            EventResponse::Handled | EventResponse::Pass => false,
            EventResponse::StartWindowDrag => {
                // Fullscreen-compositor model: OS window.drag_window() would do nothing (OS window is fullscreen). Drag is internal — capture the anchor here and move window_rect on CursorMoved. We ARM the drag without committing; the first CursorMoved commits it (no dead zone). Click-without-motion never commits, so no wrap-shift fast path runs, no persistent_screen wrap artefacts, and the textbox's small `glow_bbox` damage_rect drives the only repaint.
                //
                // Maximized state suppresses drag entirely. Most WMs handle this with "drag a maximized window → unmaximize and resume drag at cursor"; that's the right ergonomic but more involved (need to compute the unmaximized origin relative to cursor, then begin the drag). For v0 the simpler rule is "ignore the drag request" — title-bar clicks while maximized do nothing instead of producing nonsense (the drag would translate the fullscreen-sized rect into negative offsets and clip_through the input region). Revisit when we add the unmaximize-then-drag flow.
                if self.saved_rect_for_maximize.is_none() {
                    // Same capture rule as start_resize: the move gesture must see its own release wherever it lands.
                    #[cfg(target_os = "windows")]
                    if let Some(w) = self.home_window() {
                        super::windows_layered::capture_mouse(&w);
                    }
                    self.move_drag_armed = true;
                    self.drag_move_anchor_screen = (self.cursor_x as i32, self.cursor_y as i32);
                    self.drag_move_rect_start = (self.window_rect.x, self.window_rect.y);
                }
                false
            }
            EventResponse::StartResize(edge) => {
                self.start_resize(edge);
                false
            }
            EventResponse::Close => {
                // Same policy gate as the OS CloseRequested path: a resident app hides instead of dying.
                if self.app.on_close_requested() {
                    window.set_visible(false);
                    false
                } else {
                    std::process::exit(0);
                }
            }
            EventResponse::ShowWindow => {
                // Surface a hidden resident window: un-hide EVERY monitor surface (hide-on-close hid them all), focus the home one, and repaint everything — the surface content is stale (or never-painted, on a start_hidden boot). Dormant surfaces stay all-zero, so showing them is invisible.
                log::info!("FLUOR: ShowWindow — surfacing (home ready={})", self.surfaces.get(self.home).is_some_and(|s| s.surface_ready));
                for s in self.surfaces.iter() {
                    // Windows: a layered window whose style bit got dropped anywhere in the hide/show shuffle presents into the void — re-asserting is a no-op when it's still set.
                    #[cfg(target_os = "windows")]
                    super::windows_layered::make_layered(&s.window);
                    s.window.set_visible(true);
                    // Un-minimize too. A minimized window is VISIBLE-but-minimized, so set_visible(true)
                    // and focus_window() are both no-ops against it and it stays parked in the Dock/taskbar.
                    // This is why "Show" from the tray menu did nothing after a minimize.
                    s.window.set_minimized(false);
                }
                // Muffin (and X11 WMs generally) DENY focus_window from a background app as focus stealing, so the un-hidden window surfaced BEHIND the stack ("shows very briefly, goes behind" — field, 2026-08-16). A window-LEVEL pulse raises unconditionally — restack requests are not focus requests — then drops back to normal; focus_window afterwards has a fighting chance, and the worst case is raised-but-unfocused instead of buried.
                window.set_window_level(winit::window::WindowLevel::AlwaysOnTop);
                window.set_window_level(winit::window::WindowLevel::Normal);
                window.focus_window();
                self.pending_full_repaint = true;
                window.request_redraw();
                false
            }
            EventResponse::ToggleMaximized => {
                self.toggle_maximized();
                false
            }
            EventResponse::Minimize => {
                log::info!("FLUOR: minimize requested");
                window.set_minimized(true);
                false
            }
            EventResponse::MoveToMonitor(delta) => {
                self.move_to_monitor(delta);
                false
            }
        }
    }

    /// Move the visible window to another connected monitor, cycling `home` by `delta` (wrapping)
    /// and filling that surface's work area through the full `apply_window_rect` machinery. No-op
    /// with a single surface. Clears any saved-for-maximize rect so a later maximize toggle
    /// re-captures against the new monitor.
    fn move_to_monitor(&mut self, delta: i32) {
        let n = self.surfaces.len() as i32;
        if n <= 1 {
            return;
        }
        let target = (self.home as i32 + delta).rem_euclid(n) as usize;
        let s = &self.surfaces[target];
        let (wx, wy, ww, wh) = s.work_area;
        let rect = if ww > 1 && wh > 1 {
            WindowRect { x: wx, y: wy, w: ww, h: wh }
        } else {
            let o = s.origin;
            WindowRect { x: o.0, y: o.1, w: s.size.0, h: s.size.1 }
        };
        log::info!("FLUOR-MON: move window to monitor {target}/{n} rect={rect:?}");
        self.saved_rect_for_maximize = None;
        self.apply_window_rect(rect);
    }

    /// Flip `window_rect` between the user-sized rect (saved in `saved_rect_for_maximize`) and the home surface's work area. Mirrors the geometry-change tail of `resize_drag_update`: resize scratch + clip_mask, reflow viewport, notify the consumer via `on_resize`, mark full-repaint, and update the X11 input region. No-op if the home surface's size is still a placeholder — first `Resized` event hasn't landed yet, no real geometry to swap to.
    fn toggle_maximized(&mut self) {
        if self.home_window().is_none() {
            return;
        }
        let (scr_w, scr_h) = self.surfaces[self.home].size;
        if scr_w <= 1 || scr_h <= 1 {
            return;
        }
        let new_rect = match self.saved_rect_for_maximize.take() {
            // Restore clamps into the union of surface rects — if the saved rect's monitor shrank or vanished, it snaps into the nearest surviving surface's work area.
            Some(prev) => self.clamp_rect_to_surfaces(prev),
            None => {
                self.saved_rect_for_maximize = Some(self.window_rect);
                // Maximize to the home surface's work area (monitor minus panels), not the raw screen, so the maximized window's bottom chrome stays clear of the taskbar. Falls back to the full surface if the work area was never resolved. Both are GLOBAL desktop-unit rects.
                let (wx, wy, ww, wh) = self.surfaces[self.home].work_area;
                if ww > 1 && wh > 1 {
                    WindowRect { x: wx, y: wy, w: ww, h: wh }
                } else {
                    let o = self.surfaces[self.home].origin;
                    WindowRect { x: o.0, y: o.1, w: scr_w, h: scr_h }
                }
            }
        };

        self.apply_window_rect(new_rect);
    }

    /// Adopt an explicit desktop-unit `window_rect` thru the FULL machinery — home re-election, scale settle, viewport/scratch/clip rebuild, consumer `on_resize`, input region, repaint. The one path every programmatic placement shares (maximize/restore, the geometry-restore setting); raw OS-window calls are meaningless here (the OS window is the fullscreen surface — moving IT was the 2026-08-16 vanish + dead-click field failure).
    fn apply_window_rect(&mut self, new_rect: WindowRect) {
        let Some(window) = self.home_window() else {
            return;
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
        // Maximize targets the (pre-toggle) home; restore may land the rect on a different surface — re-elect before the scale sync below.
        self.update_home();
        // Waking the target surface is NOT optional when the rect crosses monitors: a bare home
        // re-election leaves the destination surface dormant, so it keeps presenting its all-zero
        // (black) frame. This is the fix for "Move to Next Monitor briefly shows content, then goes
        // black" — relocating home must also wake its surface (and evacuate the one we left).
        self.refresh_involvement();
        // macOS click-thru: a programmatic move (esp. from the menu bar) happens with the cursor
        // OUTSIDE the window, so `hittest_off` is stale-true and the just-woken destination surface
        // would keep ignoring clicks — worst on a secondary monitor, where cursor re-entry may not
        // re-toggle it. Point the global hittest monitor at the new rect and re-enable mouse
        // acceptance on the new home now; the normal cursor-outside logic re-engages on next move.
        #[cfg(target_os = "macos")]
        {
            if let Some(m) = self.hittest_monitor.as_ref() {
                let r = &self.window_rect;
                m.update_rect(r.x, r.y, r.w, r.h);
            }
            self.hittest_off = false;
            self.apply_macos_hittest();
        }
        // Maximize/restore hands us an explicit desktop-unit rect, so no geometric w/h rebase applies — just adopt the (possibly new) home scale before the pass-0 rebuild so the viewport lands at the right density (phase C settle semantics).
        let scale_changed = self
            .surfaces
            .get(self.home)
            .is_some_and(|s| (s.scale - self.window_scale).abs() > 1e-6);
        if scale_changed {
            self.window_scale = self.surfaces[self.home].scale;
        }

        if size_changed || scale_changed {
            self.viewport = self.pass0_viewport(new_rect.w, new_rect.h, self.viewport.ru);
            let win_px = (self.viewport.width_px as usize) * (self.viewport.height_px as usize);
            self.scratch = vec![0u32; win_px];
            self.clip_mask = vec![255u8; win_px];
            self.pending_full_repaint = true;

            let (vw, vh) = (self.viewport.width_px, self.viewport.height_px);
            let (ccx, ccy) = self.win_cursor_px();
            let wo = self.ctx_window_origin();
            if let Some(text) = self.text.as_mut() {
                let mut ctx = Context {
                    pressed_hit: self.pointer.held_id(),
                    viewport: self.viewport,
                    text,
                    clip_mask: &mut self.clip_mask,
                    damage: &mut self.pending_damage,
                    window: &*window,
                    modifiers: winit_compat::from_winit_mods(self.modifiers),
                    cursor_x: ccx,
                    cursor_y: ccy,
                    is_maximized: self.saved_rect_for_maximize.is_some(),
                    window_origin: wo,
                    damage_clip: crate::canvas::PixelRect::new(0, 0, vw as usize, vh as usize),
                };
                self.app.on_resize(vw, vh, &mut ctx);
            }
        } else {
            // Position-only change still needs a full repaint — the old window_rect's pixels in persistent_screen are stale.
            self.pending_full_repaint = true;
        }

        self.push_input_region(self.home);

        // Redraw the NEW home window, not the stale `window` captured before `update_home` may have
        // moved us to another surface — otherwise the destination monitor never repaints (black).
        if let Some(w) = self.home_window() {
            w.request_redraw();
        }
    }

    /// Begin a self-driven resize drag. In the fullscreen-compositor model we resize `window_rect` inside our own screen buffer instead of asking the OS to resize the OS window (which is fullscreen). Captures the start geometry (window_rect size + position) and the desktop-unit cursor anchor; subsequent cursor moves compute the new (w, h, x, y) by delta from these starting values.
    fn start_resize(&mut self, edge: ResizeEdge) {
        // Windows: own the mouse for the gesture — a release over an α=0 click-thru pixel otherwise goes to the window underneath and the drag never ends (see windows_layered::capture_mouse).
        #[cfg(target_os = "windows")]
        if let Some(w) = self.home_window() {
            super::windows_layered::capture_mouse(&w);
        }
        self.is_dragging_resize = true;
        self.resize_edge = edge;
        self.drag_start_size = (self.window_rect.w, self.window_rect.h);
        self.drag_start_window_pos = (self.window_rect.x, self.window_rect.y);
        // cursor_x/y are global desktop units (surface-local + surface origin) so no translation needed for the anchor.
        self.drag_start_cursor_screen_pos = (self.cursor_x as i32, self.cursor_y as i32);
    }

    /// Apply one tick of the self-driven resize drag — in-buffer. Called from `RedrawRequested` when `is_dragging_resize` (throttled to vsync). Updates `window_rect` directly (no OS round-trip — the OS window is fullscreen and request_inner_size / set_outer_position are no-ops). When the size changed, resizes `scratch` + `clip_mask` to the new dimensions and calls the consumer's `on_resize` so they can reflow. Always pushes a new XShape input region so click-thru follows the visible window. The subsequent `render_frame` paints at the new geometry into the screen buffer.
    fn apply_resize_drag(&mut self) {
        let Some(window) = self.home_window() else {
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
        // Resize-drag is a window_rect mutation like any other — an edge dragged deep into a neighbour can shift the overlap majority (resize tick, per the update_home contract).
        self.update_home();

        if size_changed {
            // Carry the user's zoom (ru) across the resize so Ctrl+/Ctrl-/Ctrl+scroll state survives.
            self.viewport = self.pass0_viewport(new_w, new_h, self.viewport.ru);
            let win_px = (self.viewport.width_px as usize) * (self.viewport.height_px as usize);
            self.scratch = vec![0u32; win_px];
            self.clip_mask = vec![255u8; win_px];
            // Window dims changed → perimeter, AA edges, shadow rays all need a fresh single-pass repaint.
            self.pending_full_repaint = true;

            // Let the consumer reflow — they may relayout panes, recompute glyph metrics, etc.
            let (vw, vh) = (self.viewport.width_px, self.viewport.height_px);
            let (ccx, ccy) = self.win_cursor_px();
            let wo = self.ctx_window_origin();
            if let Some(text) = self.text.as_mut() {
                let mut ctx = Context {
                    pressed_hit: self.pointer.held_id(),
                    viewport: self.viewport,
                    text,
                    clip_mask: &mut self.clip_mask,
                    damage: &mut self.pending_damage,
                    window: &*window,
                    modifiers: winit_compat::from_winit_mods(self.modifiers),
                    cursor_x: ccx,
                    cursor_y: ccy,
                    is_maximized: self.saved_rect_for_maximize.is_some(),
                    window_origin: wo,
                    damage_clip: crate::canvas::PixelRect::new(0, 0, vw as usize, vh as usize),
                };
                self.app.on_resize(vw, vh, &mut ctx);
            }
        }

        // Update click-thru region so the OS routes clicks based on the new rect.
        self.push_input_region(self.home);
    }
}

#[cfg(feature = "host-winit")]
impl<A: FluorApp + 'static> ApplicationHandler<A::UserEvent> for DesktopShell<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Watch app activation so a Dock click can reach the window layer at all — see macos_reopen.
        #[cfg(target_os = "macos")]
        super::macos_reopen::install();

        if !self.surfaces.is_empty() {
            return;
        }

        // Probe the monitors BEFORE creating windows so we can request OS surfaces of exactly the right size + position — see `create_monitor_surface` for the no-`with_fullscreen` rationale. Phase B eager-spawns one surface per output, PRIMARY FIRST: index 0 is both `anchor` (title + icon + taskbar identity) and the initial `home`; the fallback when the platform reports no primary is whatever monitor enumerates first.
        let mut monitors: Vec<winit::monitor::MonitorHandle> = Vec::new();
        if let Some(p) = event_loop.primary_monitor() {
            monitors.push(p);
        }
        for m in event_loop.available_monitors() {
            if !monitors.iter().any(|q| *q == m) {
                monitors.push(m);
            }
        }
        if monitors.is_empty() {
            // No monitor to pin a surface to (headless X server mid-teardown); nothing to create.
            return;
        }
        for (i, monitor) in monitors.into_iter().enumerate() {
            let mut surface = self.create_monitor_surface(event_loop, monitor, i == 0);
            if i != 0 {
                // Non-anchor surfaces start DORMANT: no input region, no taskbar/pager entry, and their first Resized presents one all-zero (fully transparent) frame instead of ticking the render loop.
                surface.dormant = true;
                #[cfg(target_os = "linux")]
                x11_atomic::set_skip_taskbar(&surface.window, true);
                // macOS needs no skip-taskbar equivalent — auxiliary NSWindows don't get their own Dock/cmd-tab entries.
            }
            self.surfaces.push(surface);
        }
        let window = self.surfaces[0].window.clone();
        self.home = 0;
        self.anchor = 0;
        self.window_scale = self.surfaces[0].scale;

        // Explicitly activate the home surface on first launch. A borderless, monitor-sized, transparent surface is exactly what Linux WMs (with focus-stealing prevention) decline to auto-raise/focus on map — so the app opened un-topmost, sometimes below the previously-active window. winit's focus_window() issues _NET_ACTIVE_WINDOW (X11) / the platform activate, matching the proven
        // ShowWindow + macOS-reopen paths. Skipped when starting hidden: a resident boot must NOT grab focus.
        if !self.app.start_hidden() {
            window.focus_window();
        }

        // Match the OS window icon (taskbar / alt-tab / title bar) to the app's orb. winit honours this on Windows + X11; it's a no-op on Wayland (icon from .desktop app_id)
        // and macOS (icon from the .app bundle), which source the icon at packaging time.
        if let Some(icon) = self.app.window_icon() {
            if let Some(winit_icon) = icon.to_winit_icon() {
                window.set_window_icon(Some(winit_icon));
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Install with the PRIMARY screen's height in POINTS (surfaces[0].size is points now) — that's the flip reference NSEvent's bottom-left-origin global mouseLocation needs to land in the same top-left global point space as window_rect.
            self.hittest_monitor =
                super::macos_hittest::HittestMonitor::install(self.surfaces[0].size.1);
        }

        // Build the app's native menu bar once (macOS NSMenu; no-op on other platforms). Clicks
        // come back via the queue drained in `about_to_wait` and dispatched as Event::MenuItem.
        super::macos_menu::install(&self.app.menu());

        // Initial visible-window size: app-supplied (defaults to half the screen in each axis), clamped to the surface's work area and centred within it — the work area is already in GLOBAL desktop units, so the centering math places the window correctly even when the primary monitor isn't at (0, 0). Apps with aspect-ratio opinions override [`FluorApp::initial_size`].
        let (wa_x, wa_y, wa_w, wa_h) = self.surfaces[0].work_area;
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
        // Pass-0 viewport at the home density (identical dims on X11/Windows; points × scale on macOS).
        self.viewport = self.pass0_viewport(initial_w, initial_h, 1.0);

        if self.text.is_none() {
            self.text = Some(TextRenderer::new());
        }

        // Scratch + clip_mask are sized to the pass-0 viewport, NOT the screen. The host blits scratch into each involved surface's buffer at the window_rect offset; pixels outside the window stay at the screen buffer's α=0 init.
        let win_px = (self.viewport.width_px as usize) * (self.viewport.height_px as usize);
        self.scratch = vec![0u32; win_px];
        self.clip_mask = vec![255u8; win_px];

        // Hand control to the consumer's init.
        {
            let (ccx, ccy) = self.win_cursor_px();
            let wo = self.ctx_window_origin();
            let text = self.text.as_mut().expect("text renderer initialized");
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: ccx,
                cursor_y: ccy,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: wo,
                damage_clip: crate::canvas::PixelRect::new(
                    0,
                    0,
                    self.viewport.width_px as usize,
                    self.viewport.height_px as usize,
                ),
            };
            self.app.init(&mut ctx);
        }

        // Surfaces are created at their requested monitor sizes — we can paint immediately. The Resized handler still re-derives if a different first size arrives (X11 always sends the initial ConfigureNotify; Windows resizes during creation under DPI virtualization). Marking ready applies to EVERY surface, not just the anchor: macOS never delivers an initial Resized for a window created at its final size, so a non-anchor surface waiting on one stayed !surface_ready forever — and the moment a cross-monitor drag flipped `home` onto it, render_frame's ready-gate bailed while dormancy had already evacuated the old surface: the window VANISHED (macbook repro 2026-07-25, "Safari moves fine, photon disappears").
        for s in self.surfaces.iter_mut() {
            s.surface_ready = true;
        }

        // Click-thru: tell the OS each surface's hittable area is just its window_rect intersection — the anchor gets the real region, every dormant surface gets the empty (fully click-thru) one. Drag-to-move + resize-drag re-push these on every rect change.
        for si in 0..self.surfaces.len() {
            self.push_input_region(si);
        }

        window.request_redraw();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Dock icon clicked. macOS asks the APPLICATION delegate, never the window, which is why this cannot be handled in `window_event` and why it was silently dropped before (see macos_reopen).
        #[cfg(target_os = "macos")]
        if super::macos_reopen::take_reopen() {
            log::info!("FLUOR-REOPEN: Dock reopen (#{})", super::macos_reopen::reopen_count());
            for s in self.surfaces.iter() {
                s.window.set_minimized(false);
                s.window.set_visible(true);
            }
            if let Some(window) = self.home_window() {
                window.focus_window();
                window.request_redraw();
            }
            self.pending_full_repaint = true;
        }

        self.consume_app_placement_requests();

        // Native menu clicks queued (on the main thread) since the last loop: dispatch each as
        // Event::MenuItem, apply any window-level response (e.g. a menu Fullscreen → ToggleMaximized),
        // then repaint. Empty (and free) on platforms without a menu bar.
        let menu_ids = super::macos_menu::drain();
        if !menu_ids.is_empty() {
            let (mcx, mcy) = self.win_cursor_px();
            let mwo = self.ctx_window_origin();
            let mut responses: Vec<EventResponse> = Vec::new();
            if let (Some(window), Some(text)) = (self.home_window(), self.text.as_mut()) {
                let mut ctx = Context {
                    pressed_hit: self.pointer.held_id(),
                    viewport: self.viewport,
                    text,
                    clip_mask: &mut self.clip_mask,
                    damage: &mut self.pending_damage,
                    window: &*window,
                    modifiers: winit_compat::from_winit_mods(self.modifiers),
                    cursor_x: mcx,
                    cursor_y: mcy,
                    is_maximized: self.saved_rect_for_maximize.is_some(),
                    window_origin: mwo,
                    damage_clip: crate::canvas::PixelRect::new(
                        0,
                        0,
                        self.viewport.width_px as usize,
                        self.viewport.height_px as usize,
                    ),
                };
                for id in menu_ids {
                    responses.push(self.app.on_event(&FEvent::MenuItem(id), &mut ctx));
                }
            }
            for r in responses {
                self.apply_response(r);
            }
            if let Some(window) = self.home_window() {
                window.request_redraw();
            }
        }

        let (ccx, ccy) = self.win_cursor_px();
        let wo = self.ctx_window_origin();
        let needs_redraw = if let (Some(window), Some(text)) =
            (self.home_window(), self.text.as_mut())
        {
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: ccx,
                cursor_y: ccy,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: wo,
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
            if let Some(window) = self.home_window() {
                window.request_redraw();
            }
        }

        if let Some(when) = self.app.wake_at() {
            event_loop.set_control_flow(winit::event_loop::ControlFlow::WaitUntil(when));
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        // Route by WindowId first — every event names the surface it arrived on; events for windows we don't own (or surfaces already dropped) are ignored wholesale.
        let Some(si) = self.surface_for_window(id) else {
            return;
        };
        match &event {
            WindowEvent::CloseRequested => {
                // Resident apps hide on close and keep running (see FluorApp::on_close_requested); everyone else exits, as ever. Hiding hides every surface — the app is one entity across monitors.
                if self.app.on_close_requested() {
                    for s in self.surfaces.iter() {
                        s.window.set_visible(false);
                    }
                } else {
                    std::process::exit(0);
                }
            }
            WindowEvent::Resized(size) => {
                self.handle_surface_resized(si, *size);
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Promote surface-local cursor coords to GLOBAL desktop units by the reporting surface's origin — everything downstream (window-relative subtraction, drag anchors) already speaks the global space. winit delivers the local position in surface BACKING pixels; ÷ pixel_ratio recovers desktop units (points on macOS — exact, winit derives physical FROM the point layout; a no-op ÷1 on X11/Windows). Cross-monitor drags ride X11's implicit press-grab: the pressed surface keeps reporting out-of-bounds locals and this same math lands them on the neighbour — no special case.
                let pr = self.surfaces[si].pixel_ratio as Coord;
                self.cursor_x = position.x as Coord / pr + self.surfaces[si].origin.0 as Coord;
                self.cursor_y = position.y as Coord / pr + self.surfaces[si].origin.1 as Coord;

                #[cfg(target_os = "macos")]
                {
                    self.update_macos_hittest();
                    if self.hittest_off {
                        return;
                    }
                }

                // During a self-driven resize drag, CursorMoved fires at hundreds of Hz (raw input rate) AND we synthesize more via set_outer_position (window-relative cursor pos changes when the window moves). Doing a full resize+paint+OS-update per event floods X11 (`XIO: fatal IO error 11`) and creates a multi-second backlog of stale requests that play back after release. Coalesce: just stash the new cursor pos and request a redraw — winit caps RedrawRequested to vsync (~60-144 Hz), and the actual drag tick runs there. Skips consumer event dispatch too (consumer doesn't need to see resize-drag cursor moves).
                if self.is_dragging_resize {
                    if let Some(window) = self.home_window() {
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
                        if let Some(window) = self.home_window() {
                            window.set_cursor(winit::window::CursorIcon::Grabbing);
                        }
                    }
                    self.window_rect.x = self.drag_move_rect_start.0 + dx;
                    self.window_rect.y = self.drag_move_rect_start.1 + dy;
                    // Drag tick = window_rect mutation → re-elect home so ticks/renders follow the overlap majority mid-drag (the DPI rebase itself waits for release).
                    self.update_home();
                    if let Some(window) = self.home_window() {
                        window.request_redraw();
                    }
                    return;
                }

                // Press-hold-release: while a press is in flight, track whether the pointer is still over the armed target. A drag off (or back on) toggles the held colour — request a redraw so it appears/clears. Runs before the app dispatch so ctx.pressed_hit below reflects this move.
                if self.pointer.on_move(self.hit_under_cursor()) {
                    if let Some(window) = self.home_window() {
                        window.request_redraw();
                    }
                }

                let (ccx, ccy) = self.win_cursor_px();
                let wo = self.ctx_window_origin();
                if let (Some(window), Some(text)) =
                    (self.home_window(), self.text.as_mut())
                {
                    let mut ctx = Context {
                        pressed_hit: self.pointer.held_id(),
                        viewport: self.viewport,
                        text,
                        clip_mask: &mut self.clip_mask,
                        damage: &mut self.pending_damage,
                        window: &*window,
                        modifiers: winit_compat::from_winit_mods(self.modifiers),
                        cursor_x: ccx,
                        cursor_y: ccy,
                        is_maximized: self.saved_rect_for_maximize.is_some(),
                        window_origin: wo,
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
                if let Some(window) = self.home_window() {
                    window.request_redraw();
                }
            }
            WindowEvent::Focused(focused) => {
                let focused = *focused;
                // Per-surface focus, folded into the shell-level flag — the app is focused when ANY of its surfaces is.
                self.surfaces[si].focused = focused;
                let was_focused = self.is_focused;
                self.is_focused = self.surfaces.iter().any(|s| s.focused);
                // The app hears the FOLDED edge only — per-surface flicker while focus hops between spans is not a focus change.
                if self.is_focused != was_focused {
                    self.app.on_focus_changed(self.is_focused);
                }
                // Cancel any in-progress resize drag if we lose focus mid-drag (the user alt-tabbed or the WM stole focus). Keeps state consistent.
                if !focused && self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                }
                // STALE MODIFIERS on the focus edge: Windows in particular releases keys while another window holds focus and never tells us — a Shift held at alt-tab stayed latched forever, and the next plain click on the close button read as the Shift+close deliberate-quit chord (Emma's Windows log 2026-09-01: 'deliberate quit' on an ordinary red-X press, app fully exited instead of hiding to tray). Clear on BOTH edges: keys released while we were away are unknowable, and winit re-sends ModifiersChanged after focus for keys still genuinely down.
                self.modifiers = ModifiersState::empty();
                // Shadow seed depends on focus (full strength vs quarter strength) → re-cast shadow over a fresh band, which only happens on the full-repaint path.
                self.pending_full_repaint = true;
                self.dispatch_event(event);
                // Repaint so the drop shadow dims/brightens immediately.
                if let Some(window) = self.home_window() {
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
                // Raise-on-click: a press that reached our surface means the pointer is over the visible app, so bring it to the front if it isn't already. Without this, clicking the app on a fullscreen-surface model didn't raise the OS window — it could stay behind another window the WM had stacked above.
                // Gated on `!is_focused` so an already-front window never re-requests activation (no WM flicker/fight).
                if !self.is_focused {
                    if let Some(window) = self.home_window() {
                        window.focus_window();
                    }
                }
                self.dispatch_event(event);
                if let Some(window) = self.home_window() {
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
                // Captured BEFORE the end-blocks clear the flags: a committed move or an active resize ending on THIS release is the geometry-persistence edge.
                let gesture_ended = self.is_dragging_resize || self.is_dragging_move;
                // Windows: hand the mouse back the moment any gesture's press ends (no-op when nothing was captured).
                #[cfg(target_os = "windows")]
                if self.is_dragging_resize || self.move_drag_armed {
                    super::windows_layered::release_mouse();
                }
                // End of resize drag — release ownership of the loop. The buffer is already in the final state from the last drag tick; no extra repaint needed.
                if self.is_dragging_resize {
                    self.is_dragging_resize = false;
                    self.resize_edge = ResizeEdge::None;
                    // A resize can carry the window's bulk onto another monitor too — same settle semantics as a move release (no-op when scales match).
                    self.update_home();
                    self.settle_rebase();
                }
                // End of in-buffer drag-to-move. Two release paths: (a) armed but never committed (zero cursor motion during the press) — no shifts happened, persistent_screen is intact, consumer's damage_rect drives the next paint. (b) Committed (`is_dragging_move = true`) — the wrap-shift fast path moved persistent_screen contents, leaving wrap artefacts at whichever edges the window slid across, so a full repaint is required to clean them up. Always request_redraw on either path so consumer-side invalidations queued during the press window (e.g. textbox defocus → glow_bbox damage) get a fresh render_frame to flush.
                if self.move_drag_armed {
                    self.move_drag_armed = false;
                    if self.is_dragging_move {
                        self.is_dragging_move = false;
                        self.pending_full_repaint = true;
                        // Settle point: the drag ended — if it landed on a different-scale home, re-anchor now (phase-C rebase; no-op on matching scales).
                        self.update_home();
                        self.settle_rebase();
                    }
                    if let Some(window) = self.home_window() {
                        window.set_cursor(winit::window::CursorIcon::Default);
                        window.request_redraw();
                    }
                }
                // Geometry-persistence edge: the rect the user just settled on, once per gesture. Maximized is a mode, not a placement — its rect never persists.
                if gesture_ended && self.saved_rect_for_maximize.is_none() {
                    let r = self.window_rect;
                    self.app.on_window_rect_changed(r.x, r.y, r.w, r.h);
                }
                if let Some(id) = activate {
                    self.dispatch_activate(id);
                }
                self.dispatch_event(event);
                // Clear the "held" colour now that the press ended (whether it fired or cancelled).
                if let Some(window) = self.home_window() {
                    window.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                // Non-home surfaces never tick or drag — an OS expose just re-presents whatever the surface holds (zeros while dormant: the transparent first present that keeps a fresh window from flashing back-buffer garbage; the last composite otherwise).
                if si != self.home {
                    self.present_surface_raw(si);
                    return;
                }
                // macOS click-thru: if the global monitor detected the cursor re-entering an opaque region while hittest was off, flip it back on (fanning the state out to every surface). While hittest is off we keep requesting redraws to poll the monitor flag at vsync rate.
                #[cfg(target_os = "macos")]
                if self.hittest_off {
                    if let Some(ref monitor) = self.hittest_monitor {
                        if monitor.check_reenter() {
                            self.hittest_off = false;
                            self.apply_macos_hittest();
                        } else if let Some(window) = self.home_window() {
                            // Keep polling — next vsync will check again.
                            window.request_redraw();
                        }
                    }
                }
                // Drag-to-move fast path: shift the existing screen pixels by the per-tick delta instead of re-rendering anything. Skips consumer.render(), scratch fill, finalize, and shadow rasterization. ONLY legal when the window was fully inside one surface at the last paint AND is fully inside that SAME surface now — the shift slides pixels within a single buffer and can't cross a seam. Straddle ticks fall thru to the normal render path (finalize's clipping does the partial blits), which must be a full repaint because the window moved and the composited pixels sit at stale offsets.
                if self.is_dragging_move {
                    let prev = self.surface_fully_containing(self.last_painted_rect);
                    let now = self.surface_fully_containing(self.window_rect);
                    if prev.is_some() && prev == now {
                        self.apply_move_drag_shift();
                        return;
                    }
                    self.pending_full_repaint = true;
                }
                // Resize drag: apply the new geometry in-buffer, then paint at the new size.
                if self.is_dragging_resize {
                    self.apply_resize_drag();
                }
                // Adopt any restored zoom/geometry BEFORE pixels are presented — the about_to_wait consume runs AFTER the event batch, so frame one would paint at the default placement and flash (field report, 2026-08-16). Ordering: read state, place window, THEN render.
                self.consume_app_placement_requests();
                self.render_frame();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                // A monitor's scale changed under a surface (display-settings change, or the OS reporting the true per-monitor scale post-create). Adopt it on the surface; if it's the home surface the window re-anchors immediately — there's no drag mid-flight when the OS emits this, so it's a settle point. The matching Resized (new backing size) follows separately and flows thru handle_surface_resized.
                let sc = *scale_factor;
                self.surfaces[si].scale = sc;
                #[cfg(target_os = "macos")]
                {
                    self.surfaces[si].pixel_ratio = sc;
                }
                if si == self.home {
                    self.settle_rebase();
                }
            }
            _ => {
                self.dispatch_event(event);
            }
        }
    }

    /// Cross-thread user-event payload from [`EventLoopProxy::send_event`]. Builds a [`Context`] over the host's shared resources and hands the typed event to [`FluorApp::on_user_event`]. The consumer typically reads/mutates app state and calls `ctx.window.request_redraw()` if the state change should repaint; if it doesn't request_redraw the next tick still runs normally.
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: A::UserEvent) {
        let Some(window) = self.home_window() else {
            return;
        };
        let (ccx, ccy) = self.win_cursor_px();
        let wo = self.ctx_window_origin();
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
            cursor_x: ccx,
            cursor_y: ccy,
            is_maximized: self.saved_rect_for_maximize.is_some(),
            window_origin: wo,
            damage_clip: crate::canvas::PixelRect::new(
                0,
                0,
                self.viewport.width_px as usize,
                self.viewport.height_px as usize,
            ),
        };
        let response = self.app.on_user_event(event, &mut ctx);
        self.apply_response(response);
    }
}

#[cfg(feature = "host-winit")]
impl<A: FluorApp + 'static> DesktopShell<A> {
    /// Apply a zoom change to `self.viewport.ru` and propagate to the consumer. `steps = Some(s)` adjusts by `s` photon-asymmetric log steps (positive in, negative out); `steps = None` resets to 1.0 (Ctrl+0 binding). Calls `app.on_resize` with unchanged pixel dimensions so the consumer's existing resize path picks up the new `ctx.viewport.ru`, marks chrome / widget layers dirty (via their internal Group resize), and re-rasterizes at the new effective span. No separate `on_zoom` callback needed — the consumer's on_resize is the single "viewport changed" entry point.
    /// Drain the app's one-shot placement requests — restored zoom, then restored window geometry. Called from TWO sites: the top of the home surface's `RedrawRequested` (so frame one adopts the restored placement BEFORE any pixels are presented — the launch-flash fix) and `about_to_wait` (requests armed outside a redraw, e.g. a later settings load). Geometry is taken only once the home surface has real dimensions, so an early pass can't burn the one-shot against placeholders.
    fn consume_app_placement_requests(&mut self) {
        // Restored zoom: set the ru, then run the standard zoom propagation with a zero-step change (factor 1.0 — the set already happened) so chrome/widgets re-rasterize exactly like a user zoom.
        if let Some(ru) = self.app.take_zoom_request() {
            self.viewport.set_zoom(ru);
            self.apply_zoom_change(Some(0.0));
        }
        // Restored window geometry: a window_rect in GLOBAL desktop units, applied thru the same machinery maximize uses — clamped into the surviving surfaces so an unplugged monitor's rect snaps on-screen.
        if self
            .surfaces
            .get(self.home)
            .is_some_and(|s| s.size.0 > 1 && s.size.1 > 1)
        {
            if let Some((x, y, w, h)) = self.app.take_window_geometry_request() {
                if w > 0 && h > 0 {
                    let target = self.clamp_rect_to_surfaces(WindowRect { x, y, w, h });
                    self.apply_window_rect(target);
                }
            }
        }
    }

    fn apply_zoom_change(&mut self, steps: Option<f32>) {
        match steps {
            Some(s) => self.viewport.adjust_zoom(s),
            None => self.viewport.reset_zoom(),
        }
        // Zoom changes effective_span → chrome perimeter, AA edges, glyphs, shadow ray length all scale. Full repaint required.
        self.pending_full_repaint = true;
        let (ccx, ccy) = self.win_cursor_px();
        let wo = self.ctx_window_origin();
        if let (Some(window), Some(text)) = (self.home_window(), self.text.as_mut()) {
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: ccx,
                cursor_y: ccy,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: wo,
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
        // Window-relative pass-0 PIXELS (the hit map's space) — on macOS that's points × window_scale, elsewhere the plain subtraction.
        let (cx, cy) = self.win_cursor_px();
        let x = cx as i32;
        let y = cy as i32;
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
        let (px, py) = self.win_cursor_px();
        let wo = self.ctx_window_origin();
        if let (Some(window), Some(text)) = (self.home_window(), self.text.as_mut()) {
            let x = px;
            let y = py;
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
                window_origin: wo,
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
        let (ccx, ccy) = self.win_cursor_px();
        let wo = self.ctx_window_origin();
        if let (Some(window), Some(text)) = (self.home_window(), self.text.as_mut()) {
            let mut ctx = Context {
                pressed_hit: self.pointer.held_id(),
                viewport: self.viewport,
                text,
                clip_mask: &mut self.clip_mask,
                damage: &mut self.pending_damage,
                window: &*window,
                modifiers: winit_compat::from_winit_mods(self.modifiers),
                cursor_x: ccx,
                cursor_y: ccy,
                is_maximized: self.saved_rect_for_maximize.is_some(),
                window_origin: wo,
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
