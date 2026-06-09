//! Platform-neutral event vocabulary.
//!
//! Fluor's hosts (`host-winit` on desktop, `host-android` on Android, future `host-bare` on ferros) all translate their platform input into these types at the boundary, then dispatch the unified values into [`FluorApp::on_event`] and widget capability traits. Apps and widgets never see winit or Android types — they speak fluor.
//!
//! Designed so the translation is mechanical: winit → fluor on desktop is a small match table inside `host::desktop` (or wherever `DesktopShell` lives); Android JNI → fluor on Android is the equivalent in `host::android::events`.
//!
//! Naming mirrors winit's where the concepts overlap (KeyEvent / ModifiersState / ElementState / MouseButton) so the porting cost from winit-based code is low. New concepts that winit doesn't model (multi-touch points, gesture deltas, pen pressure) belong here directly and aren't constrained to a winit-shaped hole.

extern crate alloc;
use alloc::string::String;

use crate::coord::Coord;

// ============================================================================
// Top-level event ============================================================

/// A platform-neutral window event delivered to [`crate::host::app::FluorApp::on_event`].
///
/// Hosts translate platform input (winit `WindowEvent`, Android JNI input) into these arms at the boundary. Variants cover only the cases consumers actually match on today — adding new arms requires updating every host's translation table, so we keep the enum tight and grow it deliberately.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// User requested window close (close button, Alt+F4, swipe-up gesture). Host's default response is to call `std::process::exit(0)` unless the app intercepts.
    CloseRequested,
    /// Window/surface size changed in physical pixels. Includes initial sizing on desktop and Android surfaceChanged callbacks.
    Resized {
        width: u32,
        height: u32,
    },
    /// Cursor / primary touch point moved. Coords are in viewport pixels (top-left origin, +y down).
    CursorMoved {
        x: Coord,
        y: Coord,
    },
    /// Cursor / primary touch point left the window. Android: lifted finger. Desktop: cursor moved outside the OS surface.
    CursorLeft,
    /// Mouse button pressed or released. Touch-down / touch-up on Android maps to `MouseInput { button: Left, ... }` at the primary touch point. Multi-touch arrives via the `Touch` arm instead (when we add it).
    MouseInput {
        state: ElementState,
        button: MouseButton,
    },
    /// Scroll wheel / two-finger pan delta.
    MouseWheel {
        delta: MouseScrollDelta,
    },
    /// Keyboard input — a key was pressed or released. Includes IME-composed text via `key_text` for character-producing presses.
    KeyboardInput {
        event: KeyEvent,
    },
    /// Modifier state changed (shift / ctrl / alt / super). Hosts emit this whenever any of the four modifiers transitions; the new full state is in the event.
    ModifiersChanged(ModifiersState),
    /// Window focus gained or lost. `true` = focused.
    Focused(bool),
    /// IME event — composition preedit or commit. Android `InputConnection` commit text and desktop IME both flow through here.
    Ime(Ime),
}

// ============================================================================
// Element state / mouse button ===============================================

/// Press / release transition for buttons and keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ElementState {
    Pressed,
    Released,
}

/// Mouse button. Touch on Android arrives as `Left` at the primary touch point; non-`Left` mouse buttons are desktop-only today.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    /// Side / extra mouse button — vendor-specific. `u16` carries the platform's button index for apps that want to disambiguate.
    Other(u16),
}

// ============================================================================
// Scroll delta ===============================================================

/// Direction + magnitude of a scroll event. Hosts pick whichever unit their platform delivers; consumers typically multiply through the same scaling factor regardless.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MouseScrollDelta {
    /// Discrete steps (mouse wheel notches). `(x, y)` — positive `y` scrolls content up.
    Lines(f32, f32),
    /// Continuous pixel deltas (trackpad swipe, touch flick). Same sign convention as `Lines`.
    Pixels(f32, f32),
}

// ============================================================================
// Modifier state =============================================================

/// Live modifier-key state. All four bits set simultaneously is legal (shift+ctrl+alt+super).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct ModifiersState {
    pub shift: bool,
    pub ctrl: bool,
    pub alt: bool,
    /// "Super" / Windows logo / Command key. Named `meta` (not `super`) because `super` is a reserved keyword and `r#super` is rejected as a field identifier. Method accessor is [`Self::super_key`] for winit-API parity.
    pub meta: bool,
}

impl ModifiersState {
    pub const fn empty() -> Self {
        Self {
            shift: false,
            ctrl: false,
            alt: false,
            meta: false,
        }
    }

    #[inline]
    pub const fn shift_key(self) -> bool {
        self.shift
    }
    #[inline]
    pub const fn control_key(self) -> bool {
        self.ctrl
    }
    #[inline]
    pub const fn alt_key(self) -> bool {
        self.alt
    }
    #[inline]
    pub const fn super_key(self) -> bool {
        self.meta
    }
}

// ============================================================================
// Keys =======================================================================

/// Logical key value, post-keymap. `Character(c)` carries the character a printable key would produce (one Unicode scalar — IME composed text arrives via [`Event::Ime`], not here). `Named` covers non-printable keys.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Key {
    Named(NamedKey),
    Character(String),
    Unidentified,
}

/// Non-printable keys we route on. Variant set covers what fluor's widgets and chrome handle today; extending the enum requires updating every host's translation table, so we grow it deliberately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NamedKey {
    Enter,
    Escape,
    Backspace,
    Tab,
    Delete,
    ArrowLeft,
    ArrowRight,
    ArrowUp,
    ArrowDown,
    Home,
    End,
    PageUp,
    PageDown,
    Space,
    Shift,
    Control,
    Alt,
    Super,
}

/// Key press / release. `state` distinguishes press vs release; `repeat = true` means an OS-generated repeat (held-key auto-fire). `text` carries the character-producing keystroke (e.g. `Some("a")` for an unshifted A) when the press would type something — widgets typically read `text` to insert characters and read `logical_key` (Named arms) for control keys.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyEvent {
    pub logical_key: Key,
    pub state: ElementState,
    pub repeat: bool,
    /// Text the keystroke would produce if accepted (printable character). `None` on releases, non-printable keys, modifier presses, and IME-composed text (IME flows through [`Event::Ime`]).
    pub text: Option<String>,
}

// ============================================================================
// IME ========================================================================

/// IME composition events. Today we only translate `Commit` (the cross-platform "user accepted this string, type it"); preedit + state changes can land here when we wire IME on desktop or build a fancier Android InputConnection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ime {
    /// IME committed a string — host inserts it as if typed.
    Commit(String),
}

// ============================================================================
// Cursor icon ================================================================

/// Cursor shape the host should display. Variants are limited to what fluor's chrome + widgets request today; Android stubs all variants to no-ops (no pointer cursor on touchscreens).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CursorIcon {
    Default,
    Pointer,
    Text,
    NsResize,
    EwResize,
    NwseResize,
    NeswResize,
}

impl Default for CursorIcon {
    fn default() -> Self {
        CursorIcon::Default
    }
}
