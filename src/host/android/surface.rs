//! ANativeWindow surface management — ports `photon/src/ui/renderer_android.rs` into fluor.
//!
//! The pipeline: app renders into a `Vec<u32>` (visible-RGB ARGB after fluor's finalize step converts α + darkness → α + visible-RGB), then `present()` locks the next Android triple-buffered surface, optionally checks a magic pixel to short-circuit redundant copies, memcpy rows, writes the new magic value, and `unlockAndPost`s.
//!
//! Pixel format: `ANativeWindow_setBuffersGeometry` (called from the JNI surface-creation path) sets `WINDOW_FORMAT_RGBA_8888`. The buffer surface is 4 bytes per pixel laid out as `[R, G, B, A]` byte-wise in little-endian memory — which read as `u32` is `0xAABBGGRR`. fluor's internal format is `0xAARRGGBB` (α high, RGB visible after finalize). The present-time row copy is byte-for-byte without channel swizzle; Android samples the bytes in `R, G, B, A` order regardless of `u32` endianness.
//!
//! Threading: present is called from the UI thread inside `nativeDraw`, which runs on the Activity thread driven by Choreographer. The ANativeWindow's lock/unlock cycle is the only synchronization needed.
//!
//! Magic-pixel triple-buffer optimization: Android rotates three back buffers. Each gets the current content_version stamped into the top-right pixel after copy. On the next frame, if the locked buffer's magic pixel already matches the latest content_version, the buffer is already up-to-date and we skip the rowwise memcpy. Reverts to unconditional copy on Samsung devices where their compositor mutates the magic pixel.

use ndk::native_window::NativeWindow;
use ndk_sys::{ANativeWindow_Buffer, ANativeWindow_lock, ANativeWindow_unlockAndPost};

/// Samsung-device flag. Their compositor mutates pixels between lock/unlock cycles in ways that break the magic-pixel cache; we fall back to unconditional row copy when this is set. Caller sets once at app startup via [`set_samsung_mode`] from the JNI init shim, reading `Build.MANUFACTURER` Java-side.
static mut SAMSUNG_MODE: bool = false;

/// Set Samsung-device mode. Call once from JNI init before any rendering; thereafter the surface uses unconditional row copy (slower but correct on Samsung's compositor).
pub fn set_samsung_mode(is_samsung: bool) {
    unsafe {
        SAMSUNG_MODE = is_samsung;
    }
}

#[inline]
fn is_samsung() -> bool {
    unsafe { SAMSUNG_MODE }
}

/// CPU surface backed by a `Vec<u32>`. App renders into the buffer (via fluor's compositor and finalize pipeline), then [`Surface::present`] blits it onto the Android NativeWindow surface and posts it for display.
///
/// Lifetime model: the Surface owns the pixel buffer; the NativeWindow is borrowed for each `present()` call. The JNI shim holds the NativeWindow (acquired from the surface-creation callback) and threads it through every render call alongside this Surface.
pub struct Surface {
    width: u32,
    height: u32,
    /// CPU pixel buffer. Apps obtain `&mut [u32]` via [`Surface::buffer_mut`] to render into.
    buffer: Vec<u32>,
    /// Monotonic content-version counter. Incremented every time the buffer is presented with new content; written to the top-right pixel as the magic-pixel cache key so the next frame can detect whether the Android triple-buffer it just locked already holds the latest content.
    content_version: u32,
}

impl Surface {
    /// Create a CPU surface with the given pixel dimensions. The buffer is zero-initialised (fully-transparent, since fluor's α=0 means transparent post-finalize).
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: alloc::vec![0; (width as usize).saturating_mul(height as usize)],
            // Start at 1 so 0 (uninitialised Android buffer) never matches.
            content_version: 1,
        }
    }

    /// Resize the buffer. Caller invokes this on Android `surfaceChanged` (window dimensions changed) before the next render. Bumps content_version so the next present is a full copy even if it happens to lock a buffer that previously matched.
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

    /// Borrow the pixel buffer for rendering. App writes directly into the slice; size is `width * height` `u32`s, row-major, no padding.
    pub fn buffer_mut(&mut self) -> &mut [u32] {
        self.buffer.as_mut_slice()
    }

    /// Current surface dimensions in pixels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Lock the next Android triple-buffer, copy our render buffer onto it (or skip when the magic-pixel cache says it's already current), stamp the magic pixel, unlock+post.
    ///
    /// `dirty = true` means the app rendered new content this frame — we bump content_version to invalidate any other Android buffers that might still hold the previous frame. `dirty = false` means the app reports no visible change — we still call the lock/unlock cycle to drive Choreographer timing, but skip the rowwise copy if the locked buffer's magic pixel already matches our content_version.
    ///
    /// Returns `true` if rows were actually copied this frame, `false` if the magic-pixel cache hit and we short-circuited.
    ///
    /// # Safety
    /// `window` must be a valid live ANativeWindow handle, not freed between this call's `lock` and `unlockAndPost`. The Android lifecycle guarantees this as long as the Surface is presented only between `surfaceCreated` and `surfaceDestroyed` callbacks.
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

            // RGBA_8888: 4 bytes per pixel, interpret as u32 (visible bytes match fluor's post-finalize format byte-for-byte — see module docs).
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
                // Samsung: always copy (their compositor mutates the magic pixel).
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

            // Always post — Choreographer needs the buffer flip to drive its frame timing.
            ANativeWindow_unlockAndPost(window.ptr().as_ptr());
            copied
        }
    }
}

/// Per-row memcpy from `src` (`src_width`-wide rows) to `dst` (`dst_stride`-wide rows). Copies `copy_width` × `copy_height` pixels. Caller already ensured both buffers are sized to fit.
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
