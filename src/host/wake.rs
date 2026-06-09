//! [`WakeSender`] — host-agnostic cross-thread wake-up.
//!
//! Apps stash an `Arc<dyn WakeSender<UserEvent>>` and clone it across background threads (network workers, IO tasks, async ceremonies). When a background task wants the UI to repaint with a result, it calls `wake.send(payload)`; the concrete impl routes the payload back through `FluorApp::on_user_event` on the UI thread.
//!
//! Hosts provide concrete impls: host-winit wraps `winit::event_loop::EventLoopProxy` (in [`super::winit_compat`]); host-android wires JNI callbacks (or a no-op proxy if the app doesn't use cross-thread wake-ups — the Activity polls via Choreographer).
//!
//! This decouples `FluorApp` from winit's `EventLoopProxy` type, which is what made winit a transitive dep on Android even though we never run winit's event loop there.

use core::any::type_name;

/// Errors returned by [`WakeSender::send`]. The runtime is responsible for deciding whether a send failure should be silently logged, retried, or panicked on; the trait surface stays minimal.
#[derive(Debug)]
pub struct WakeError {
    /// Best-effort name of the user-event type the send was attempted with — surfaces in panic / log messages.
    pub event_type: &'static str,
}

impl core::fmt::Display for WakeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "WakeSender::send failed for event type {}",
            self.event_type
        )
    }
}

/// Cross-thread wake-up channel from app background tasks back to the UI thread. Host provides a concrete impl when constructing the shell.
pub trait WakeSender<E>: Send + Sync {
    /// Deliver `event` to the UI thread. Returns `Err` if the host's event loop has closed (typical: app is exiting) — most callers ignore the error since their work is moot if the loop is gone.
    fn send(&self, event: E) -> Result<(), WakeError>;
}

/// No-op wake sender. Used when the host can't provide a working channel (e.g. host-android before Choreographer + JNI callbacks are wired) but the type must still satisfy the trait. Every `send` returns `Err` so callers' best-effort send loops don't busy-spin assuming success.
pub struct NoopWakeSender;

impl<E> WakeSender<E> for NoopWakeSender {
    fn send(&self, _event: E) -> Result<(), WakeError> {
        Err(WakeError {
            event_type: type_name::<E>(),
        })
    }
}
