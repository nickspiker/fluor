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
        physical_key: winit_physical_to_scancode(&kev.physical_key),
    }
}

/// winit's layout-neutral `KeyCode` → standard PC Set-1 scancode (== Windows `position_code`).
/// Extended keys carry the `0xE0` high byte (arrows, right-hand modifiers, nav cluster). `0` for
/// anything we don't map. This is the one place fluor knows winit's key names, so the mapping
/// lives here; every consumer just reads the neutral `physical_key` u16.
fn winit_physical_to_scancode(pk: &winit::keyboard::PhysicalKey) -> u16 {
    use winit::keyboard::{KeyCode as K, PhysicalKey};
    let PhysicalKey::Code(c) = pk else { return 0 };
    match c {
        // ── letters ──
        K::KeyA => 0x1E, K::KeyB => 0x30, K::KeyC => 0x2E, K::KeyD => 0x20, K::KeyE => 0x12,
        K::KeyF => 0x21, K::KeyG => 0x22, K::KeyH => 0x23, K::KeyI => 0x17, K::KeyJ => 0x24,
        K::KeyK => 0x25, K::KeyL => 0x26, K::KeyM => 0x32, K::KeyN => 0x31, K::KeyO => 0x18,
        K::KeyP => 0x19, K::KeyQ => 0x10, K::KeyR => 0x13, K::KeyS => 0x1F, K::KeyT => 0x14,
        K::KeyU => 0x16, K::KeyV => 0x2F, K::KeyW => 0x11, K::KeyX => 0x2D, K::KeyY => 0x15,
        K::KeyZ => 0x2C,
        // ── digit row ──
        K::Digit1 => 0x02, K::Digit2 => 0x03, K::Digit3 => 0x04, K::Digit4 => 0x05,
        K::Digit5 => 0x06, K::Digit6 => 0x07, K::Digit7 => 0x08, K::Digit8 => 0x09,
        K::Digit9 => 0x0A, K::Digit0 => 0x0B,
        // ── symbols ──
        K::Minus => 0x0C, K::Equal => 0x0D, K::BracketLeft => 0x1A, K::BracketRight => 0x1B,
        K::Backslash => 0x2B, K::Semicolon => 0x27, K::Quote => 0x28, K::Backquote => 0x29,
        K::Comma => 0x33, K::Period => 0x34, K::Slash => 0x35,
        // ── whitespace / edit ──
        K::Space => 0x39, K::Enter => 0x1C, K::Tab => 0x0F, K::Backspace => 0x0E,
        K::Escape => 0x01, K::CapsLock => 0x3A,
        // ── modifiers (right-hand ones are 0xE0-extended) ──
        K::ShiftLeft => 0x2A, K::ShiftRight => 0x36, K::ControlLeft => 0x1D,
        K::ControlRight => 0xE01D, K::AltLeft => 0x38, K::AltRight => 0xE038,
        K::SuperLeft => 0xE05B, K::SuperRight => 0xE05C, K::ContextMenu => 0xE05D,
        // ── nav cluster (extended) ──
        K::Insert => 0xE052, K::Delete => 0xE053, K::Home => 0xE047, K::End => 0xE04F,
        K::PageUp => 0xE049, K::PageDown => 0xE051, K::ArrowUp => 0xE048, K::ArrowDown => 0xE050,
        K::ArrowLeft => 0xE04B, K::ArrowRight => 0xE04D,
        // ── function row ──
        K::F1 => 0x3B, K::F2 => 0x3C, K::F3 => 0x3D, K::F4 => 0x3E, K::F5 => 0x3F, K::F6 => 0x40,
        K::F7 => 0x41, K::F8 => 0x42, K::F9 => 0x43, K::F10 => 0x44, K::F11 => 0x57, K::F12 => 0x58,
        // ── numpad ──
        K::Numpad0 => 0x52, K::Numpad1 => 0x4F, K::Numpad2 => 0x50, K::Numpad3 => 0x51,
        K::Numpad4 => 0x4B, K::Numpad5 => 0x4C, K::Numpad6 => 0x4D, K::Numpad7 => 0x47,
        K::Numpad8 => 0x48, K::Numpad9 => 0x49, K::NumpadAdd => 0x4E, K::NumpadSubtract => 0x4A,
        K::NumpadMultiply => 0x37, K::NumpadDivide => 0xE035, K::NumpadDecimal => 0x53,
        K::NumpadEnter => 0xE01C, K::NumLock => 0x45,
        _ => 0,
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
