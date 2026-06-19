//! ANativeWindow surface management.
//!
//! Single-pass pipeline per frame: app renders α + darkness into `scratch`; `Surface::present` locks the next ANativeWindow back buffer, runs `finalize_into_screen` directly into the locked bits at the buffer's reported `stride`, stamps the magic pixel, unlocks + posts.
//!
//! Magic-pixel triple-buffer optimization: Android rotates ~three back buffers. Each gets the current `content_version` stamped into the top-right pixel after finalize. On the next frame, if the locked buffer's magic pixel already matches the latest `content_version`, the buffer is already up-to-date and we skip the finalize entirely. Samsung-mode falls back to unconditional finalize because their compositor mutates the magic pixel between lock/unlock cycles.
//!
//! Idle frames (`dirty = false` and magic-pixel matches) cost one lock + one read + one unlock+post — no per-pixel work. Stale-but-idle frames re-finalize from `scratch` (which retains the last full α + darkness frame), so we converge to the cached state across the buffer rotation without keeping an intermediate `Vec<u32>` cache.

use ndk::native_window::NativeWindow;
use ndk_sys::{ADataSpace, ANativeWindow_Buffer, ANativeWindow_lock, ANativeWindow_unlockAndPost};

use crate::canvas::PixelRect;

/// `ANativeWindow_setBuffersDataSpace` resolved at runtime via `dlsym`. Required because the symbol was added in NDK API 28 — linking it as a strong import would prevent the .so from loading at all on devices below 28 (the dynamic linker resolves all imports eagerly at `dlopen` time). Resolving via `dlsym` keeps the binary compatible with our `api-level-26` floor: on 26-27 the symbol comes back null and we silently skip the dataspace tag (the buffer stays at the compositor's default — sRGB — which is the correct fallback). On 28+ we get the proper BT.2020 tagging.
type SetBuffersDataSpaceFn =
    unsafe extern "C" fn(*mut ndk_sys::ANativeWindow, i32) -> i32;

fn lookup_set_buffers_data_space() -> Option<SetBuffersDataSpaceFn> {
    use core::sync::atomic::{AtomicPtr, Ordering};
    static CACHED: AtomicPtr<core::ffi::c_void> = AtomicPtr::new(core::ptr::null_mut());
    // Sentinel: we use 1 as "lookup done, symbol not present" so a non-zero non-1 value means "valid function pointer". This keeps the load-acquire path branch-free in the common case.
    const NOT_PRESENT: *mut core::ffi::c_void = 1 as *mut _;
    let cached = CACHED.load(Ordering::Acquire);
    if cached == NOT_PRESENT {
        return None;
    }
    if !cached.is_null() {
        return Some(unsafe { core::mem::transmute::<*mut core::ffi::c_void, SetBuffersDataSpaceFn>(cached) });
    }
    // RTLD_DEFAULT doesn't find post-link-time NDK symbols on Android (the dynamic linker namespaces hide libandroid.so from the default search). Explicitly dlopen libandroid.so first, then dlsym off that handle.
    unsafe {
        let libname = c"libandroid.so";
        let lib = libc::dlopen(libname.as_ptr(), libc::RTLD_NOW | libc::RTLD_NOLOAD);
        let lib = if lib.is_null() {
            // RTLD_NOLOAD returned null — try a real load.
            libc::dlopen(libname.as_ptr(), libc::RTLD_NOW)
        } else {
            lib
        };
        if lib.is_null() {
            CACHED.store(NOT_PRESENT, Ordering::Release);
            return None;
        }
        let name = c"ANativeWindow_setBuffersDataSpace";
        let handle = libc::dlsym(lib, name.as_ptr());
        let to_store = if handle.is_null() { NOT_PRESENT } else { handle };
        CACHED.store(to_store, Ordering::Release);
        if handle.is_null() {
            None
        } else {
            Some(core::mem::transmute::<*mut core::ffi::c_void, SetBuffersDataSpaceFn>(handle))
        }
    }
}

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

/// Lightweight surface state: tracks the surface dimensions, the monotonic content-version counter that drives the magic-pixel triple-buffer cache, and a one-shot flag that marks whether the ANativeWindow buffer dataspace has been pushed to BT.2020 yet. No intermediate pixel buffer — finalize writes directly into ANativeWindow's locked bits.
pub struct Surface {
    width: u32,
    height: u32,
    content_version: u32,
    /// True once `ANativeWindow_setBuffersDataSpace(BT2020 | GAMMA2_2 | FULL)` has been called for the current `NativeWindow`. Combined with the Activity's `colorMode = WIDE_COLOR_GAMUT` + `preferMinimalPostProcessing`, this gives the consumer pipeline a display-native target: the bytes we write are taken as BT.2020 RGB and land on the panel without an sRGB clamp or vendor saturation pass. Photon does its own colour-management later on the theme constants + chromatic wave, so any OS-side clamp would be actively destructive. Reset to `false` on resize because Android may re-create the buffer queue under a new geometry and lose the dataspace setting.
    dataspace_set: bool,
}

impl Surface {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            content_version: 1,
            dataspace_set: false,
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
        // Force re-push of the BT.2020 dataspace next frame — surfaceChanged on Android can recreate the back-buffer queue, and a fresh queue defaults back to the implicit sRGB dataspace.
        self.dataspace_set = false;
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
            // One-shot per buffer-queue lifetime: declare our pixels are in BT.2020, not sRGB. Without this, the compositor treats the bytes we write as sRGB and runs them thru an sRGB→panel-native colour transform — exactly the desaturation/wash the photon pipeline is going to fight by doing its own colour management on theme + spectrum colours later. Resolved via dlsym (see [`lookup_set_buffers_data_space`]) so the binary stays loadable on pre-API-28 devices that don't ship the symbol.
            if !self.dataspace_set {
                match lookup_set_buffers_data_space() {
                    Some(set_ds) => {
                        // PHOTON COLOUR PIPELINE ON ANDROID. Photon writes γ=2.0 BT.2020 RGB into the buffer. We tag the buffer as γ=2.2 BT.2020 because γ=2.2 is the closest named transfer Android offers — there is no `TRANSFER_GAMMA2_0` constant and no way to pass a custom transfer function to SurfaceFlinger. The mismatch (we say γ=2.2, we send γ=2.0) means the panel renders slightly darker than authored because SF interprets each stored value as `code^2.2` when in fact we stored `code^2.0`. This trade-off is intentional and load-bearing: photon's pipeline uses square / sqrt for encode / decode at 2-4 CPU cycles per pixel; replacing them with `powf(x, 1.0/2.2)` or worse the sRGB piecewise transfer would cost 50-100 cycles per pixel running on every rasterizer, every glyph, every overlay — flatly unacceptable, and fractional gammas are numerically miserable to invert and compose. We pick γ=2.0 once at architecture time and stick with it. Users who want exact-correct rendering need to be on ferros (our own host, which we build to honour γ=2.0 end-to-end) or a non-Android platform — the slight darkness on Android is the documented cost of shipping on a platform whose colour-management API has no γ=2.0 entry; if a user complains, file a feature request against Android for `TRANSFER_GAMMA2_0`. BT.2020 primaries because photon's spectral pipeline can synthesize colours wider than P3 and we'd rather tag them honestly (and let SF tonemap into the display gamut) than clip into a narrower gamut ourselves. Constructed by OR since there's no matching `ADATASPACE_*` named constant for BT2020 + GAMMA2_2: layout 6<<16 | 4<<22 | 1<<27 = 151388160.
                        const STANDARD_BT2020: i32 = 6 << 16;
                        const TRANSFER_GAMMA2_2: i32 = 4 << 22;
                        const RANGE_FULL: i32 = 1 << 27;
                        let custom_dataspace = STANDARD_BT2020 | TRANSFER_GAMMA2_2 | RANGE_FULL;
                        let rc = set_ds(window.ptr().as_ptr(), custom_dataspace);
                        if rc == 0 {
                            log::info!(
                                "fluor::host::android::surface: setBuffersDataSpace(BT2020 | GAMMA2_2 | FULL = {}) ok",
                                custom_dataspace
                            );
                        } else {
                            log::warn!(
                                "fluor::host::android::surface: setBuffersDataSpace(BT2020 | GAMMA2_2 | FULL) returned {} — staying on compositor default",
                                rc
                            );
                        }
                    }
                    None => log::warn!(
                        "fluor::host::android::surface: ANativeWindow_setBuffersDataSpace symbol not found (API < 28); BT.2020 tag not applied"
                    ),
                }
                // Set the flag whether or not the call succeeded so we don't retry every frame.
                self.dataspace_set = true;
            }

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
