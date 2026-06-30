//! Android input → fluor::event translation.
//!
//! Touch events arrive from `nativeOnTouch` as `(action: i32, x: f32, y: f32)`. Action codes are Android's `MotionEvent.ACTION_*`:
//! - 0 DOWN — primary touch began. Translate to `CursorMoved` (so widgets get the focus-tracking coordinate update) followed by `MouseInput { Pressed, Left }`.
//! - 1 UP — primary touch ended. `MouseInput { Released, Left }` only — the cursor stays where it last moved, matching desktop convention (mouse buttons release; cursor doesn't teleport).
//! - 2 MOVE — drag. `CursorMoved`.
//! - 3 CANCEL — gesture cancelled by the system. `CursorLeft` + `MouseInput { Released, Left }` so any in-flight drag-select / button-press cleans up.
//!
//! Key events arrive from `nativeOnKeyEvent` as Android `KeyEvent` keycodes. Only the codes photon currently routes are translated; everything else maps to `Key::Unidentified` so the JNI layer can short-circuit and return `false` (unhandled).
//!
//! Text input arrives from `nativeOnTextInput` as a Java `String`. Translates to `Event::Ime` with the committed text; the focused textbox's `Key` impl already ignores the keystroke when an IME commit is active — apps just see the IME variant.

use crate::event::{Event, Ime, Key, NamedKey};

/// Android `MotionEvent.ACTION_DOWN`.
pub const ACTION_DOWN: i32 = 0;
/// Android `MotionEvent.ACTION_UP`.
pub const ACTION_UP: i32 = 1;
/// Android `MotionEvent.ACTION_MOVE`.
pub const ACTION_MOVE: i32 = 2;
/// Android `MotionEvent.ACTION_CANCEL`.
pub const ACTION_CANCEL: i32 = 3;

/// Translate an Android touch (`action`, x, y) into the fluor event stream the host loop should dispatch in order. Returns a stack-allocated array of up to two events (the longest translation is DOWN → CursorMoved + MouseInput, and CANCEL → CursorLeft + MouseInput).
///
/// `count` of the returned tuple is `0`, `1`, or `2` — callers iterate `events[..count]`.
pub fn translate_touch(action: i32, x: f32, y: f32) -> (usize, [Event; 2]) {
    use crate::event::{ElementState, MouseButton};
    let empty = [Event::CursorLeft, Event::CursorLeft];
    let press_at = |x: f32, y: f32| Event::CursorMoved { x, y };
    let press_button = Event::MouseInput {
        state: ElementState::Pressed,
        button: MouseButton::Left,
    };
    let release_button = Event::MouseInput {
        state: ElementState::Released,
        button: MouseButton::Left,
    };
    match action {
        ACTION_DOWN => (2, [press_at(x, y), press_button]),
        ACTION_MOVE => (1, [press_at(x, y), empty[1].clone()]),
        ACTION_UP => (1, [release_button, empty[1].clone()]),
        ACTION_CANCEL => (2, [Event::CursorLeft, release_button]),
        _ => (0, empty),
    }
}

// ----------------------------------------------------------------------------

// Key codes — Android `KeyEvent.KEYCODE_*`. Values from <android/keycodes.h>. Only what photon uses is mapped; extending the set is safe and additive.
const KEYCODE_DPAD_UP: i32 = 19;
const KEYCODE_DPAD_DOWN: i32 = 20;
const KEYCODE_DPAD_LEFT: i32 = 21;
const KEYCODE_DPAD_RIGHT: i32 = 22;
const KEYCODE_DEL: i32 = 67; // Backspace on Android
const KEYCODE_ENTER: i32 = 66;
const KEYCODE_TAB: i32 = 61;
const KEYCODE_ESCAPE: i32 = 111;
const KEYCODE_MOVE_HOME: i32 = 122;
const KEYCODE_MOVE_END: i32 = 123;
const KEYCODE_FORWARD_DEL: i32 = 112; // Delete (right of Backspace)
const KEYCODE_SPACE: i32 = 62;

/// Translate an Android `KeyEvent` keycode to a fluor [`Key`]. `Unidentified` for anything we don't model — the JNI shim should treat that as "host doesn't handle this key" and return false from `nativeOnKeyEvent` so Android's default behaviour kicks in.
pub fn translate_keycode(key_code: i32) -> Key {
    match key_code {
        KEYCODE_DEL => Key::Named(NamedKey::Backspace),
        KEYCODE_ENTER => Key::Named(NamedKey::Enter),
        KEYCODE_TAB => Key::Named(NamedKey::Tab),
        KEYCODE_ESCAPE => Key::Named(NamedKey::Escape),
        KEYCODE_DPAD_LEFT => Key::Named(NamedKey::ArrowLeft),
        KEYCODE_DPAD_RIGHT => Key::Named(NamedKey::ArrowRight),
        KEYCODE_DPAD_UP => Key::Named(NamedKey::ArrowUp),
        KEYCODE_DPAD_DOWN => Key::Named(NamedKey::ArrowDown),
        KEYCODE_MOVE_HOME => Key::Named(NamedKey::Home),
        KEYCODE_MOVE_END => Key::Named(NamedKey::End),
        KEYCODE_FORWARD_DEL => Key::Named(NamedKey::Delete),
        KEYCODE_SPACE => Key::Named(NamedKey::Space),
        _ => Key::Unidentified,
    }
}

/// Convenience: build a `KeyboardInput` press event from an Android keycode. Returns `None` for unmapped keys (caller treats as unhandled).
pub fn key_press_from_keycode(key_code: i32) -> Option<Event> {
    use crate::event::{ElementState, KeyEvent as FKeyEvent};
    let logical_key = translate_keycode(key_code);
    if matches!(logical_key, Key::Unidentified) {
        return None;
    }
    Some(Event::KeyboardInput {
        event: FKeyEvent {
            logical_key,
            state: ElementState::Pressed,
            repeat: false,
            text: None,
        },
    })
}

/// Build an IME commit event from a committed text string.
pub fn ime_commit(text: alloc::string::String) -> Event {
    Event::Ime(Ime::Commit(text))
}
