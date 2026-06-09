//! Android host — fluor's compositor wired to ANativeWindow + Choreographer + JNI input.
//!
//! Architecture mirrors `host-winit` (which translates `winit::WindowEvent` to fluor events and presents via softbuffer): on Android the surface backend is `ANativeWindow_lock` / `unlockAndPost`, the event loop is the Activity's `Choreographer.postFrameCallback` driving JNI `nativeDraw` calls, and input arrives through JNI entry points (`nativeOnTouch`, `nativeOnKeyEvent`, `nativeOnTextInput`).
//!
//! Consumer model: app implements `FluorApp` once, then on Android creates a `AndroidShell<A>` and ferries it across the JNI boundary as an opaque `jlong`. PhotonActivity.kt holds that pointer and calls fluor's JNI entry points; fluor takes the lock/draw/dispatch path and returns to Java.
//!
//! Submodules:
//! - [`surface`] — ANativeWindow lock/post wrapper, magic-pixel triple-buffer optimization, Samsung-compositor workaround.
//! - [`events`] — Android touch / key / IME → fluor::event translation.
//! - [`jni`] — `#[no_mangle] pub extern "C" fn Java_*` entry points matching PhotonActivity.kt's contract.
//! - [`lifecycle`] — pause/resume/destroy hooks the Activity invokes during lifecycle transitions.
//!
//! Status: Phase 2.3 — events.rs + window.rs + shell.rs all wired. `AndroidShell` is the entry point; consumers' JNI thin-shims construct one in `nativeInit` and call its `draw` / `resize` / `on_touch` / `on_text_input` / `on_key_event` / `on_back_pressed` / `on_scale` methods from the matching `nativeXxx` JNI entry points.

pub mod events;
pub mod shell;
pub mod surface;
pub mod window;

pub use shell::AndroidShell;
pub use window::AndroidWindow;
