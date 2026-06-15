//! Host backends — adapters between fluor's [`Compositor`](crate::Compositor) and the platform's window/framebuffer system.
//!
//! Per `## API / Implementation Separation` in AGENT.md, hosts are interchangeable: the same `Compositor` API drives the desktop winit+softbuffer host, the future ferros bare-metal framebuffer host, and anything else. Each host is gated by a Cargo feature so consumers compile in only what they need.

pub mod chrome;
pub mod event_response;
pub mod icon;
pub mod wake;
pub mod window_handle;

pub use event_response::EventResponse;
pub use wake::{NoopWakeSender, WakeError, WakeSender};
pub use window_handle::WindowHandle;

// chrome_widget (DefaultChrome + ChromeButton) speaks fluor-native event types ([`crate::event`]) in its capability-trait impls. Still gated on `text` since the chrome paints title glyphs. Available on every supported host now (host-winit, host-android) once the host translates platform input to fluor events at the boundary.
#[cfg(feature = "text")]
pub mod chrome_widget;

// `app` contains the `FluorApp` trait + `Context` + (gated below) `DesktopShell`. The trait and Context compile on any host with `text` + `winit` (data types only — see Cargo.toml's host-android comment); DesktopShell + run_app stay host-winit-only.
#[cfg(all(feature = "text", any(feature = "host-winit", feature = "host-android")))]
pub mod app;

#[cfg(all(feature = "host-android", target_os = "android"))]
pub mod android;

#[cfg(feature = "host-winit")]
pub mod os_input;

// Widget abstraction (Container, Widget, Click, Key, Focus, Hover capability traits). Available on any host with `text` because the capability traits speak fluor-native events now — apps build their widget tree against this regardless of which host (winit/android/future) is driving the event loop.
#[cfg(feature = "text")]
pub mod widget;

/// macOS renderer — wgpu/Metal with PostMultiplied alpha for transparent corners.
#[cfg(all(feature = "host-winit", target_os = "macos"))]
pub mod renderer_wgpu;

/// Winit ↔ fluor event translation helpers. Used by host-winit's `DesktopShell` to translate at the event-loop boundary, and by consumers that still receive winit-shaped events from `FluorApp::on_event` while the trait migration is in flight.
#[cfg(feature = "host-winit")]
pub mod winit_compat;

/// macOS click-thru: global NSEvent monitor for re-entry detection.
#[cfg(all(feature = "host-winit", target_os = "macos"))]
pub(crate) mod macos_hittest;
