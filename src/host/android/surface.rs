//! ANativeWindow surface management.
//!
//! Path per frame: app renders into `scratch` (α + darkness format); shell calls `fluor::paint::finalize_into_screen` to write scratch → `Surface.buffer` (visible-RGB ARGB); `Surface::present` locks the next ANativeWindow back buffer, memcpys our buffer onto it (or skips via the magic-pixel cache), stamps the magic pixel, unlocks + posts.
//!
//! Magic-pixel triple-buffer optimization: Android rotates three back buffers. Each gets the current `content_version` stamped into the top-right pixel after copy. On the next frame, if the locked buffer's magic pixel already matches the latest `content_version`, the buffer is already up-to-date and we skip the rowwise memcpy. Samsung-mode falls back to unconditional copy because their compositor mutates the magic pixel between lock/unlock cycles.

use ndk::native_window::NativeWindow;
use ndk_sys::{ANativeWindow_Buffer, ANativeWindow_lock, ANativeWindow_unlockAndPost};

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

/// CPU surface backed by a `Vec<u32>` in visible-RGB ARGB format. Shell calls `buffer_mut` to get a `&mut [u32]` to finalize into; then `present` blits that onto the ANativeWindow surface and posts it.
pub struct Surface {
    width: u32,
    height: u32,
    buffer: Vec<u32>,
    content_version: u32,
}

impl Surface {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: alloc::vec![0; (width as usize).saturating_mul(height as usize)],
            content_version: 1,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        self.buffer
            .resize((width as usize).saturating_mul(height as usize), 0);
        self.content_version = self.content_version.wrapping_add(1);
        if self.content_version == 0 {
            self.content_version = 1;
        }
    }

    pub fn buffer_mut(&mut self) -> &mut [u32] {
        self.buffer.as_mut_slice()
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Lock the next Android triple-buffer, copy our render buffer onto it (or skip via magic-pixel cache), stamp magic, unlock+post.
    pub fn present(&mut self, window: &NativeWindow, dirty: bool) -> bool {
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

            let src_width = self.width as usize;
            let copy_height = dst_height.min(self.height as usize);
            let copy_width = dst_width.min(src_width);

            if dirty {
                self.content_version = self.content_version.wrapping_add(1);
                if self.content_version == 0 {
                    self.content_version = 1;
                }
            }

            let copied = if is_samsung() {
                copy_rows(&self.buffer, src_width, dst_pixels, stride, copy_width, copy_height);
                true
            } else {
                let magic_idx = dst_width.saturating_sub(1);
                let buffer_is_current =
                    magic_idx < stride && dst_pixels[magic_idx] == self.content_version;
                if buffer_is_current {
                    false
                } else {
                    copy_rows(&self.buffer, src_width, dst_pixels, stride, copy_width, copy_height);
                    if magic_idx < stride {
                        dst_pixels[magic_idx] = self.content_version;
                    }
                    true
                }
            };

            ANativeWindow_unlockAndPost(window.ptr().as_ptr());
            copied
        }
    }
}

fn copy_rows(
    src: &[u32],
    src_width: usize,
    dst: &mut [u32],
    dst_stride: usize,
    copy_width: usize,
    copy_height: usize,
) {
    for y in 0..copy_height {
        let s = y.saturating_mul(src_width);
        let d = y.saturating_mul(dst_stride);
        dst[d..d + copy_width].copy_from_slice(&src[s..s + copy_width]);
    }
}
