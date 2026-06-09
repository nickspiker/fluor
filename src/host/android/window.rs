//! Android [`crate::host::WindowHandle`] implementation.
//!
//! Photon's apps call `ctx.window.request_redraw()` after mutating state that affects the next
//! paint. On winit/desktop that schedules a redraw with the platform's event loop. On Android
//! Choreographer fires every vsync regardless of any "request" — so `request_redraw` just
//! sets a dirty flag the shell checks before each frame, skipping the surface copy entirely
//! when the flag is false.
//!
//! The flag uses interior mutability via `AtomicBool` so the trait can take `&self` (which
//! widgets and apps assume).

use core::sync::atomic::{AtomicBool, Ordering};

use crate::host::WindowHandle;

/// Atomic dirty-flag holder. App side calls `request_redraw` to set; shell reads + clears
/// before each frame so the next `nativeDraw` is allowed to do the full pipeline. Constructed
/// dirty so the first `nativeDraw` after `nativeInit` is guaranteed to paint.
pub struct AndroidWindow {
    dirty: AtomicBool,
}

impl AndroidWindow {
    pub fn new() -> Self {
        Self {
            dirty: AtomicBool::new(true),
        }
    }

    /// True if the app has marked the window dirty since the last `clear_dirty`. The shell
    /// reads this each frame to decide whether to run the full render pipeline.
    pub fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::AcqRel)
    }

    /// Force the next frame to render unconditionally. Called from the shell on resize or
    /// when external signals (peer update, tick) demand a paint.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

impl Default for AndroidWindow {
    fn default() -> Self {
        Self::new()
    }
}

impl WindowHandle for AndroidWindow {
    fn request_redraw(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}
