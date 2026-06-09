//! ANativeWindow surface management.
//!
//! Single-pass pipeline per frame: app renders α + darkness into `scratch`; `Surface::present` locks the next ANativeWindow back buffer, runs `finalize_into_screen` directly into the locked bits at the buffer's reported `stride`, stamps the magic pixel, unlocks + posts.
//!
//! Magic-pixel triple-buffer optimization: Android rotates ~three back buffers. Each gets the current `content_version` stamped into the top-right pixel after finalize. On the next frame, if the locked buffer's magic pixel already matches the latest `content_version`, the buffer is already up-to-date and we skip the finalize entirely. Samsung-mode falls back to unconditional finalize because their compositor mutates the magic pixel between lock/unlock cycles.
//!
//! Idle frames (`dirty = false` and magic-pixel matches) cost one lock + one read + one unlock+post — no per-pixel work. Stale-but-idle frames re-finalize from `scratch` (which retains the last full α + darkness frame), so we converge to the cached state across the buffer rotation without keeping an intermediate `Vec<u32>` cache.

use ndk::native_window::NativeWindow;
use ndk_sys::{ANativeWindow_Buffer, ANativeWindow_lock, ANativeWindow_unlockAndPost};

use crate::canvas::PixelRect;

/// Samsung-device flag.
static mut SAMSUNG_MODE: bool = false;

pub fn set_samsung_mode(is_samsung: bool) {
    unsafe {
        SAMSUNG_MODE = is_samsung;
    }
}

#[inline]
fn is_samsung() -> bool {
    unsafe { SAMSUNG_MODE }
}

/// Lightweight surface state: tracks the surface dimensions and the monotonic content-version counter that drives the magic-pixel triple-buffer cache. No intermediate pixel buffer — finalize writes directly into ANativeWindow's locked bits.
pub struct Surface {
    width: u32,
    height: u32,
    content_version: u32,
}

impl Surface {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            content_version: 1,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        self.content_version = self.content_version.wrapping_add(1);
        if self.content_version == 0 {
            self.content_version = 1;
        }
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Lock the next Android triple-buffer, finalize `scratch` (α + darkness) + `clip_mask` directly into the locked bits at the buffer's stride (skipping a magic-pixel-cache hit), stamp magic, unlock+post.
    ///
    /// `dirty = true` means the caller painted new content into `scratch` this frame — we bump `content_version` so any other Android buffer still holding the previous content reads as stale and gets re-finalized when it next rotates in. `dirty = false` means the caller didn't repaint; we still run the lock/unlock cycle to drive Choreographer timing, but skip the finalize entirely when the magic pixel already matches.
    ///
    /// Returns `true` if pixels were actually written this frame, `false` if the cache hit and we short-circuited.
    pub fn present(
        &mut self,
        window: &NativeWindow,
        scratch: &[u32],
        clip_mask: &[u8],
        win_w: usize,
        win_h: usize,
        damage_clip: PixelRect,
        dirty: bool,
    ) -> bool {
        unsafe {
            let mut android_buffer: ANativeWindow_Buffer = core::mem::zeroed();
            if ANativeWindow_lock(
                window.ptr().as_ptr(),
                &mut android_buffer,
                core::ptr::null_mut(),
            ) < 0
            {
                log::error!("fluor::host::android::surface: ANativeWindow_lock failed");
                return false;
            }

            let stride = android_buffer.stride as usize;
            let dst_height = android_buffer.height as usize;
            let dst_width = android_buffer.width as usize;
            let dst_pixels: &mut [u32] = core::slice::from_raw_parts_mut(
                android_buffer.bits as *mut u32,
                stride.saturating_mul(dst_height),
            );

            if dirty {
                self.content_version = self.content_version.wrapping_add(1);
                if self.content_version == 0 {
                    self.content_version = 1;
                }
            }

            let magic_idx = dst_width.saturating_sub(1);
            let buffer_is_current = !is_samsung()
                && magic_idx < stride
                && dst_pixels[magic_idx] == self.content_version;

            let wrote = if buffer_is_current {
                false
            } else {
                let copy_width = dst_width.min(win_w);
                let copy_height = dst_height.min(win_h);
                let clip = PixelRect::new(
                    damage_clip.x0.min(copy_width),
                    damage_clip.y0.min(copy_height),
                    damage_clip.x1.min(copy_width),
                    damage_clip.y1.min(copy_height),
                );
                crate::paint::finalize_into_screen(
                    scratch,
                    clip_mask,
                    win_w,
                    win_h,
                    dst_pixels,
                    stride,
                    0,
                    0,
                    clip,
                    true,
                );
                if magic_idx < stride {
                    dst_pixels[magic_idx] = self.content_version;
                }
                true
            };

            ANativeWindow_unlockAndPost(window.ptr().as_ptr());
            wrote
        }
    }
}
