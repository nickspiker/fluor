//! Host backends — adapters between fluor's [`Compositor`](crate::Compositor) and the platform's window/framebuffer system.
//!
//! Per `## API / Implementation Separation` in AGENT.md, hosts are interchangeable: the same `Compositor` API drives the desktop winit+softbuffer host, the future ferros bare-metal framebuffer host, and anything else. Each host is gated by a Cargo feature so consumers compile in only what they need.

pub mod chrome;

#[cfg(feature = "host-winit")]
pub mod desktop;
