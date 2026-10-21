
//! The EXTENDED-BRIGHTNESS (HDR) Android surface (HDR plan, Android first; Nick 2026-10-20).
//!
//! The app's pixels go to our own `ASurfaceControl` child of the window, never thru `ANativeWindow_lock`: three CPU-written RGBA_1010102 `AHardwareBuffer`s, each frame one transaction carrying the buffer, its dataspace (BT.2020, γ2.2, extended range) and `setExtendedRangeBrightness(current, desired)` — so the light in the buffer and the ratio it was made for always travel together.
//! Why 10-bit and not F16 (measured on a Pixel 8 Pro, AOSP read): SurfaceFlinger classifies an F16 BT.2020-linear layer as GENERIC_HDR (tone-mapped like video), the display processor cannot scan F16 out (`MPPUnspported`), and the GPU pass composites it into an 8-bit sRGB client target that squeezes SDR into its bottom codes while boosted. An extended-range layer of any other format with a ratio above 1.01 "self-promotes" to DISPLAY_HDR, and 1010102 stays on the display processor.
//! On that path 1023 means SDR white × `current` (SurfaceFlinger: layerBrightnessNits = sdrWhitePointNits × currentHdrSdrRatio), so `current` is what the buffer was encoded for: the frame's peak light, capped at what the display grants ([`set_display_headroom`], fed from `Display.getHdrSdrRatio` and its listener) — a declaration past the grant would dim the whole layer, white with it.
//! `desired` is the frame's own brightest light, never a constant: an SDR frame declares and asks 1, and its table never changes.
//! The 8-bit `g2t` buffer and its blending are untouched: finalize maps it thru [`crate::theme::Hdr10Table`] (one read per channel), rebuilt only when the declared ratio moves.
//! A buffer is rewritten only after SurfaceFlinger released it (the transaction's OnComplete hands back the release fence of the buffer it replaced, and the CPU lock waits on that fence); a frame with no free buffer is kept pending and shown on the next vsync.
//! Everything resolves thru `dlsym` (API-26 floor); any missing piece or failure leaves the host on the 8-bit `ANativeWindow` path.

use core::ffi::{c_char, c_void};
use core::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;

use crate::theme::Hdr10Table;

/// The app opted in ([`enable`]); off, the host never builds an HDR surface.
static ENABLED: AtomicBool = AtomicBool::new(false);
/// The display's granted HDR/SDR ratio, Q16 (65536 = 1×: no headroom, the safe start).
static GRANTED_Q16: AtomicU32 = AtomicU32::new(65536);

/// Opt the window into the extended-range surface (the app calls this once at startup, before the first frame).
pub fn enable() {
    ENABLED.store(true, Ordering::Relaxed);
}

/// The headroom the display grants right now (×SDR white): `Display.getHdrSdrRatio` at start and from its change listener. The next present rebuilds the table and repaints when it moved.
pub fn set_display_headroom(ratio: f32) {
    let r = if ratio.is_finite() && ratio > 1.0 { ratio } else { 1.0 };
    GRANTED_Q16.store((r * 65536.0) as u32, Ordering::Relaxed);
}

/// Whether the app opted in.
pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

fn granted() -> f32 {
    GRANTED_Q16.load(Ordering::Relaxed) as f32 / 65536.0
}

// ADataSpace bits (android/data_space.h): BT.2020 primaries, γ2.2 transfer, extended range — 1023 is SDR white × the declared ratio.
const DATASPACE_BT2020_GAMMA22_EXTENDED: i32 = (6 << 16) | (4 << 22) | (3 << 27);
const TRANSPARENCY_OPAQUE: i8 = 2;
const VISIBILITY_SHOW: i8 = 1;

type CreateFromWindow = unsafe extern "C" fn(*mut ndk_sys::ANativeWindow, *const c_char) -> *mut c_void;
type ScRelease = unsafe extern "C" fn(*mut c_void);
type TxCreate = unsafe extern "C" fn() -> *mut c_void;
type TxDelete = unsafe extern "C" fn(*mut c_void);
type TxApply = unsafe extern "C" fn(*mut c_void);
type TxSetBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut ndk_sys::AHardwareBuffer, i32);
type TxSetDataSpace = unsafe extern "C" fn(*mut c_void, *mut c_void, i32);
type TxSetExtended = unsafe extern "C" fn(*mut c_void, *mut c_void, f32, f32);
type TxSetTransparency = unsafe extern "C" fn(*mut c_void, *mut c_void, i8);
type TxSetVisibility = unsafe extern "C" fn(*mut c_void, *mut c_void, i8);
type TxReparent = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void);
type OnComplete = unsafe extern "C" fn(*mut c_void, *mut c_void);
type TxSetOnComplete = unsafe extern "C" fn(*mut c_void, *mut c_void, OnComplete);
type StatsPrevReleaseFence = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;

#[derive(Clone, Copy)]
struct Fns {
    create: CreateFromWindow,
    sc_release: ScRelease,
    tx_create: TxCreate,
    tx_delete: TxDelete,
    apply: TxApply,
    set_buffer: TxSetBuffer,
    set_dataspace: TxSetDataSpace,
    set_extended: TxSetExtended,
    set_transparency: TxSetTransparency,
    set_visibility: TxSetVisibility,
    reparent: TxReparent,
    set_on_complete: TxSetOnComplete,
    prev_release_fence: StatsPrevReleaseFence,
}

/// Resolved once; `None` when this Android lacks any piece (setExtendedRangeBrightness is API 34).
fn fns() -> Option<Fns> {
    static ONCE: std::sync::OnceLock<Option<Fns>> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| unsafe {
        let lib = libc::dlopen(c"libandroid.so".as_ptr(), libc::RTLD_NOW);
        if lib.is_null() {
            return None;
        }
        macro_rules! sym {
            ($name:literal, $ty:ty) => {{
                let p = libc::dlsym(lib, concat!($name, "\0").as_ptr() as *const c_char);
                if p.is_null() {
                    log::info!("HDR: {} not on this Android — staying on the 8-bit surface", $name);
                    return None;
                }
                core::mem::transmute::<*mut c_void, $ty>(p)
            }};
        }
        Some(Fns {
            create: sym!("ASurfaceControl_createFromWindow", CreateFromWindow),
            sc_release: sym!("ASurfaceControl_release", ScRelease),
            tx_create: sym!("ASurfaceTransaction_create", TxCreate),
            tx_delete: sym!("ASurfaceTransaction_delete", TxDelete),
            apply: sym!("ASurfaceTransaction_apply", TxApply),
            set_buffer: sym!("ASurfaceTransaction_setBuffer", TxSetBuffer),
            set_dataspace: sym!("ASurfaceTransaction_setBufferDataSpace", TxSetDataSpace),
            set_extended: sym!("ASurfaceTransaction_setExtendedRangeBrightness", TxSetExtended),
            set_transparency: sym!("ASurfaceTransaction_setBufferTransparency", TxSetTransparency),
            set_visibility: sym!("ASurfaceTransaction_setVisibility", TxSetVisibility),
            reparent: sym!("ASurfaceTransaction_reparent", TxReparent),
            set_on_complete: sym!("ASurfaceTransaction_setOnComplete", TxSetOnComplete),
            prev_release_fence: sym!("ASurfaceTransactionStats_getPreviousReleaseFenceFd", StatsPrevReleaseFence),
        })
    })
}

/// A buffer's release state, shared with SurfaceFlinger's OnComplete callback (a binder thread).
struct SlotState {
    /// SurfaceFlinger no longer holds it (it was never shown, or the frame after it has landed).
    released: AtomicBool,
    /// The release fence the lock must wait on before writing (−1 = none).
    fence: AtomicI32,
}

struct Slot {
    hb: *mut ndk_sys::AHardwareBuffer,
    stride: usize,
    state: Arc<SlotState>,
}

/// What the OnComplete callback needs: the slot the transaction replaced, and the control to read its fence for.
struct Completion {
    prev: Option<Arc<SlotState>>,
    sc: *mut c_void,
    prev_release_fence: StatsPrevReleaseFence,
}

unsafe extern "C" fn on_complete(ctx: *mut c_void, stats: *mut c_void) {
    let c = Box::from_raw(ctx as *mut Completion);
    if let Some(prev) = c.prev {
        let fd = (c.prev_release_fence)(stats, c.sc);
        let old = prev.fence.swap(fd, Ordering::AcqRel);
        if old >= 0 {
            libc::close(old);
        }
        prev.released.store(true, Ordering::Release);
    }
}

/// The extended-range surface for one window at one size.
pub struct HdrSurface {
    fns: Fns,
    sc: *mut c_void,
    window: *mut ndk_sys::ANativeWindow,
    width: usize,
    height: usize,
    slots: Vec<Slot>,
    shown: Option<usize>,
    table: Hdr10Table,
    /// The last frame's brightest `g2t` code: what the next frame declares and asks for.
    peak_code: u8,
    /// A dirty frame found no free buffer: the next present shows it even if nothing new was painted.
    pending: bool,
}

/// Why a present asked for another frame: the table it used no longer matches the frame's peak or the grant.
pub struct HdrPresent {
    pub wrote: bool,
    pub again: bool,
}

impl HdrSurface {
    /// Build the child surface and its buffers, or `None` (not enabled, an API missing, an allocation refused) — the caller stays on the 8-bit path.
    pub fn new(window: *mut ndk_sys::ANativeWindow, width: usize, height: usize) -> Option<Self> {
        if !ENABLED.load(Ordering::Relaxed) || width == 0 || height == 0 {
            return None;
        }
        let fns = fns()?;
        unsafe {
            let sc = (fns.create)(window, c"fluor_hdr".as_ptr());
            if sc.is_null() {
                log::warn!("HDR: createFromWindow refused — staying on the 8-bit surface");
                return None;
            }
            let mut slots = Vec::with_capacity(3);
            for _ in 0..3 {
                match alloc(width, height) {
                    Some((hb, stride)) => slots.push(Slot { hb, stride, state: Arc::new(SlotState { released: AtomicBool::new(true), fence: AtomicI32::new(-1) }) }),
                    None => {
                        log::warn!("HDR: 10-bit buffer {}×{} refused — staying on the 8-bit surface", width, height);
                        for s in &slots {
                            ndk_sys::AHardwareBuffer_release(s.hb);
                        }
                        (fns.sc_release)(sc);
                        return None;
                    }
                }
            }
            log::info!("HDR: extended-range surface {}×{} (RGBA_1010102, BT.2020 γ2.2 extended), granted {:.2}×", width, height, granted());
            Some(HdrSurface { fns, sc, window, width, height, slots, shown: None, table: Hdr10Table::new(1.0), peak_code: 194, pending: false })
        }
    }

    pub fn fits(&self, window: *mut ndk_sys::ANativeWindow, width: usize, height: usize) -> bool {
        self.window == window && self.width == width && self.height == height
    }

    /// Finalize `scratch` into a free buffer and show it with its dataspace and ratios. `dirty = false` and an unchanged table post nothing: SurfaceFlinger keeps showing the last buffer.
    pub fn present(&mut self, scratch: &[u32], clip_mask: &[u8], dirty: bool) -> HdrPresent {
        let dirty = dirty || core::mem::take(&mut self.pending);
        // DECLARE what the buffer spans: the content's peak, but never more than the display grants (a declaration past the grant makes SurfaceFlinger dim the whole layer, white with it). An SDR frame declares 1 and its table never changes.
        let light = |code: u8| (crate::theme::G2T_LIGHT_Q16[code as usize] as f32 / 65536.0).max(1.0);
        let declared = light(self.peak_code).min(granted().max(1.0));
        if (self.table.declared_q16 as i64 - (declared * 65536.0) as i64).abs() > 64 {
            self.table = Hdr10Table::new(declared);
        } else if !dirty && self.shown.is_some() {
            return HdrPresent { wrote: false, again: false };
        }
        let Some(i) = (0..self.slots.len()).find(|&i| Some(i) != self.shown && self.slots[i].state.released.load(Ordering::Acquire)) else {
            self.pending = true;
            return HdrPresent { wrote: false, again: true }; // every buffer still on screen or in flight: next vsync
        };
        let f = self.fns;
        unsafe {
            let slot = &self.slots[i];
            let fence = slot.state.fence.swap(-1, Ordering::AcqRel);
            let mut ptr: *mut c_void = core::ptr::null_mut();
            let usage = ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0;
            if ndk_sys::AHardwareBuffer_lock(slot.hb, usage, fence, core::ptr::null(), &mut ptr) != 0 || ptr.is_null() {
                log::warn!("HDR: buffer lock failed");
                self.pending = true;
                return HdrPresent { wrote: false, again: true };
            }
            let dst = core::slice::from_raw_parts_mut(ptr as *mut u32, slot.stride * self.height);
            let peak = crate::paint::finalize_into_1010102(scratch, clip_mask, self.width, self.height, dst, slot.stride, &self.table);
            let mut acquire: i32 = -1;
            ndk_sys::AHardwareBuffer_unlock(slot.hb, &mut acquire);
            slot.state.released.store(false, Ordering::Release);

            // The buffer spans what it declared (current); it asks for its own brightest light (desired), so the display rises only as far as the content does.
            let desired = light(peak);
            let current = self.table.declared();
            let tx = (f.tx_create)();
            (f.set_buffer)(tx, self.sc, slot.hb, acquire);
            (f.set_dataspace)(tx, self.sc, DATASPACE_BT2020_GAMMA22_EXTENDED);
            (f.set_extended)(tx, self.sc, current, desired);
            if crate::paint::OPAQUE_BACKDROP.load(Ordering::Relaxed) {
                (f.set_transparency)(tx, self.sc, TRANSPARENCY_OPAQUE);
            }
            if self.shown.is_none() {
                (f.set_visibility)(tx, self.sc, VISIBILITY_SHOW);
            }
            let done = Box::new(Completion { prev: self.shown.map(|p| self.slots[p].state.clone()), sc: self.sc, prev_release_fence: f.prev_release_fence });
            (f.set_on_complete)(tx, Box::into_raw(done) as *mut c_void, on_complete);
            (f.apply)(tx);
            (f.tx_delete)(tx);
            self.shown = Some(i);
            let again = peak != self.peak_code;
            // Above white is a choice, never an accident: when the frame's peak moves past white, name the first pixel that carries it (one scan on the edge, not per frame).
            if again && peak > 200 {
                let want = 255 - peak as u32;
                if let Some(at) = scratch.iter().position(|&p| p >> 24 != 0 && ((p & 0xFF) == want || ((p >> 8) & 0xFF) == want || ((p >> 16) & 0xFF) == want)) {
                    log::info!("HDR: frame peak code {} ({:.2}× white) first at x {} y {} — stored 0x{:08X}", peak, desired, at % self.width, at / self.width, scratch[at]);
                }
            }
            self.peak_code = peak;
            HdrPresent { wrote: true, again }
        }
    }
}

impl Drop for HdrSurface {
    fn drop(&mut self) {
        let f = self.fns;
        unsafe {
            let tx = (f.tx_create)();
            (f.reparent)(tx, self.sc, core::ptr::null_mut());
            (f.apply)(tx);
            (f.tx_delete)(tx);
            (f.sc_release)(self.sc);
            for s in &self.slots {
                let fd = s.state.fence.swap(-1, Ordering::AcqRel);
                if fd >= 0 {
                    libc::close(fd);
                }
                ndk_sys::AHardwareBuffer_release(s.hb); // SurfaceFlinger keeps its own reference to what it still shows
            }
        }
    }
}

/// One RGBA 2:10:10:10 buffer the CPU writes and the compositor reads; returns it with its row stride in pixels.
unsafe fn alloc(width: usize, height: usize) -> Option<(*mut ndk_sys::AHardwareBuffer, usize)> {
    let desc = ndk_sys::AHardwareBuffer_Desc {
        width: width as u32,
        height: height as u32,
        layers: 1,
        format: ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_R10G10B10A2_UNORM.0,
        usage: ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0
            | ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY.0
            | ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE.0,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    let mut hb: *mut ndk_sys::AHardwareBuffer = core::ptr::null_mut();
    if ndk_sys::AHardwareBuffer_allocate(&desc, &mut hb) != 0 || hb.is_null() {
        return None;
    }
    let mut got: ndk_sys::AHardwareBuffer_Desc = core::mem::zeroed();
    ndk_sys::AHardwareBuffer_describe(hb, &mut got);
    Some((hb, got.stride as usize))
}
