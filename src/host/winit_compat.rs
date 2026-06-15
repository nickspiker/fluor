//! Winit ↔ fluor translation helpers. Only compiled with `host-winit`.
//!
//! Hosts and apps both touch the winit/fluor boundary — apps that still match on `winit::WindowEvent` in their `on_event` need to convert to fluor's [`crate::event`] vocabulary before calling into widget dispatch (which speaks fluor). These helpers are the canonical conversion functions; using anything else risks drift if the mapping ever needs to evolve.
//!
//! Direction notes:
//! - winit → fluor: lossy in places (e.g. `winit::keyboard::Key::Character` is `SmolStr`, we copy to `String`). winit arms we don't model are skipped (`from_winit_event` returns `None`).
//! - fluor → winit: only `CursorIcon` for now — the host calls `window.set_cursor` with a winit type after `app.cursor_for` returns a fluor type.

use crate::event::{
    CursorIcon, ElementState, Event, Ime, Key, KeyEvent, ModifiersState, MouseButton,
    MouseScrollDelta, NamedKey,
};
use crate::host::wake::{WakeError, WakeSender};

/// Wraps a `winit::event_loop::EventLoopProxy<E>` as a fluor [`WakeSender`]. Constructed by `run_app` and handed to the app via `FluorApp::set_event_proxy`; apps clone the `Arc` and ship to background threads, calling `wake.send(payload)` to route a `Self::UserEvent` back thru `on_user_event` on the UI thread.
pub struct WinitWakeSender<E: 'static + Send> {
    proxy: winit::event_loop::EventLoopProxy<E>,
}

impl<E: 'static + Send> WinitWakeSender<E> {
    pub fn new(proxy: winit::event_loop::EventLoopProxy<E>) -> Self {
        Self { proxy }
    }
}

impl<E: 'static + Send> WakeSender<E> for WinitWakeSender<E> {
    fn send(&self, event: E) -> Result<(), WakeError> {
        self.proxy.send_event(event).map_err(|_| WakeError {
            event_type: core::any::type_name::<E>(),
        })
    }
}

// ============================================================================

// winit → fluor ============================================================================
/// Convert a winit `ModifiersState` to fluor's. Bit-by-bit equivalent.
pub fn from_winit_mods(m: winit::keyboard::ModifiersState) -> ModifiersState {
    ModifiersState {
        shift: m.shift_key(),
        ctrl: m.control_key(),
        alt: m.alt_key(),
        meta: m.super_key(),
    }
}

/// Convert a winit `ElementState` to fluor's.
pub fn from_winit_element_state(s: winit::event::ElementState) -> ElementState {
    match s {
        winit::event::ElementState::Pressed => ElementState::Pressed,
        winit::event::ElementState::Released => ElementState::Released,
    }
}

/// Convert a winit `MouseButton` to fluor's. Side buttons collapse to `Other(0)` since the winit variants don't carry an index.
pub fn from_winit_mouse_button(b: winit::event::MouseButton) -> MouseButton {
    match b {
        winit::event::MouseButton::Left => MouseButton::Left,
        winit::event::MouseButton::Right => MouseButton::Right,
        winit::event::MouseButton::Middle => MouseButton::Middle,
        winit::event::MouseButton::Back => MouseButton::Other(0),
        winit::event::MouseButton::Forward => MouseButton::Other(1),
        winit::event::MouseButton::Other(n) => MouseButton::Other(n),
    }
}

/// Convert a winit `MouseScrollDelta` to fluor's. Line deltas are pre-normalised by winit; pixel deltas come from the underlying `PhysicalPosition<f64>` and we cast to `f32`.
pub fn from_winit_scroll_delta(d: winit::event::MouseScrollDelta) -> MouseScrollDelta {
    match d {
        winit::event::MouseScrollDelta::LineDelta(x, y) => MouseScrollDelta::Lines(x, y),
        winit::event::MouseScrollDelta::PixelDelta(p) => {
            MouseScrollDelta::Pixels(p.x as f32, p.y as f32)
        }
    }
}

/// Convert a winit `Key` to fluor's. `Character` arms copy the SmolStr to a fresh `String`; named keys map only those arms we model and collapse the rest to `Unidentified`.
pub fn from_winit_key(k: &winit::keyboard::Key) -> Key {
    use winit::keyboard::{Key as WKey, NamedKey as WNamed};
    match k {
        WKey::Named(n) => {
            let mapped = match n {
                WNamed::Enter => Some(NamedKey::Enter),
                WNamed::Escape => Some(NamedKey::Escape),
                WNamed::Backspace => Some(NamedKey::Backspace),
                WNamed::Tab => Some(NamedKey::Tab),
                WNamed::Delete => Some(NamedKey::Delete),
                WNamed::ArrowLeft => Some(NamedKey::ArrowLeft),
                WNamed::ArrowRight => Some(NamedKey::ArrowRight),
                WNamed::ArrowUp => Some(NamedKey::ArrowUp),
                WNamed::ArrowDown => Some(NamedKey::ArrowDown),
                WNamed::Home => Some(NamedKey::Home),
                WNamed::End => Some(NamedKey::End),
                WNamed::PageUp => Some(NamedKey::PageUp),
                WNamed::PageDown => Some(NamedKey::PageDown),
                WNamed::Space => Some(NamedKey::Space),
                WNamed::Shift => Some(NamedKey::Shift),
                WNamed::Control => Some(NamedKey::Control),
                WNamed::Alt => Some(NamedKey::Alt),
                WNamed::Super | WNamed::Meta => Some(NamedKey::Super),
                _ => None,
            };
            match mapped {
                Some(named) => Key::Named(named),
                None => Key::Unidentified,
            }
        }
        WKey::Character(s) => Key::Character(s.as_str().to_string()),
        _ => Key::Unidentified,
    }
}

/// Convert a winit `KeyEvent` to fluor's. `text` carries the printable payload if any.
pub fn from_winit_key_event(kev: &winit::event::KeyEvent) -> KeyEvent {
    KeyEvent {
        logical_key: from_winit_key(&kev.logical_key),
        state: from_winit_element_state(kev.state),
        repeat: kev.repeat,
        text: kev.text.as_ref().map(|s| s.as_str().to_string()),
    }
}

/// Convert a winit `Ime` to fluor's `Ime`. Only `Commit` is translated today — Preedit, Enabled, Disabled return `None`.
pub fn from_winit_ime(i: &winit::event::Ime) -> Option<Ime> {
    match i {
        winit::event::Ime::Commit(s) => Some(Ime::Commit(s.clone())),
        _ => None,
    }
}

/// Convert a winit `WindowEvent` to fluor's `Event`. Returns `None` for events we don't currently model — host should keep handling those internally (drag-to-move, redraw requests, OS-level lifecycle events) rather than forwarding to the app.
pub fn from_winit_event(event: &winit::event::WindowEvent) -> Option<Event> {
    use winit::event::WindowEvent as W;
    match event {
        W::CloseRequested => Some(Event::CloseRequested),
        W::Resized(size) => Some(Event::Resized {
            width: size.width,
            height: size.height,
        }),
        W::CursorMoved { position, .. } => Some(Event::CursorMoved {
            x: position.x as crate::coord::Coord,
            y: position.y as crate::coord::Coord,
        }),
        W::CursorLeft { .. } => Some(Event::CursorLeft),
        W::MouseInput { state, button, .. } => Some(Event::MouseInput {
            state: from_winit_element_state(*state),
            button: from_winit_mouse_button(*button),
        }),
        W::MouseWheel { delta, .. } => Some(Event::MouseWheel {
            delta: from_winit_scroll_delta(*delta),
        }),
        W::KeyboardInput { event: kev, .. } => Some(Event::KeyboardInput {
            event: from_winit_key_event(kev),
        }),
        W::ModifiersChanged(m) => Some(Event::ModifiersChanged(from_winit_mods(m.state()))),
        W::Focused(f) => Some(Event::Focused(*f)),
        W::Ime(i) => from_winit_ime(i).map(Event::Ime),
        W::DroppedFile(path) => Some(Event::DroppedFile(path.to_string_lossy().into_owned())),
        _ => None,
    }
}

// ============================================================================

// fluor → winit ============================================================================
/// Convert a fluor `CursorIcon` to winit's. Host calls this before `window.set_cursor`.
pub fn to_winit_cursor(c: CursorIcon) -> winit::window::CursorIcon {
    match c {
        CursorIcon::Default => winit::window::CursorIcon::Default,
        CursorIcon::Pointer => winit::window::CursorIcon::Pointer,
        CursorIcon::Text => winit::window::CursorIcon::Text,
        CursorIcon::NsResize => winit::window::CursorIcon::NsResize,
        CursorIcon::EwResize => winit::window::CursorIcon::EwResize,
        CursorIcon::NwseResize => winit::window::CursorIcon::NwseResize,
        CursorIcon::NeswResize => winit::window::CursorIcon::NeswResize,
    }
}
