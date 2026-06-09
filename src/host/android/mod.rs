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
//! Status: SCAFFOLD. Phase 2.1 of the host-android plan — module structure + Cargo feature in place; surface code ported next; JNI + event loop after that. `AndroidShell` is the entry point that ties everything together once all pieces exist.

pub mod surface;
