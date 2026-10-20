//! DEV PROBE — HDR plan, open test 2 (2026-10-20): does an extended-range surface take BT.2020 primaries, or only scRGB (BT.709)?
//!
//! Armed by a development build ([`arm`]); on the first present it hangs two 256×256 F16 child surfaces off the app window, each holding linear 2.0 (twice diffuse white), with `setExtendedRangeBrightness(2, 2)`: one tagged BT.2020 | LINEAR | EXTENDED, one scRGB (BT.709 | LINEAR | EXTENDED).
//! The answer is read outside the app, from `dumpsys SurfaceFlinger` (each layer's dataspace and composition type, by its debug name) — the probe only builds the layers and logs what it did.
//! Every surface-control entry point is resolved thru `dlsym`, like `ANativeWindow_setBuffersDataSpace`, so the API-26 floor still loads.

use core::ffi::{c_char, c_void};
use core::sync::atomic::{AtomicBool, Ordering};

static ARMED: AtomicBool = AtomicBool::new(false);
static RAN: AtomicBool = AtomicBool::new(false);

/// Arm the probe (a development build, at startup): it runs once, on the next present.
pub fn arm() {
    ARMED.store(true, Ordering::Relaxed);
}

// ADataSpace bits (android/data_space.h).
const STANDARD_BT709: i32 = 1 << 16;
const STANDARD_BT2020: i32 = 6 << 16;
const TRANSFER_LINEAR: i32 = 1 << 22;
const RANGE_EXTENDED: i32 = 3 << 27;

/// Probe square side: big enough that the display's brightness controller counts it as HDR content (a 16 px square was seen as an HDR layer but never boosted).
const SIDE: u32 = 256;

type CreateFromWindow = unsafe extern "C" fn(*mut ndk_sys::ANativeWindow, *const c_char) -> *mut c_void;
type TxCreate = unsafe extern "C" fn() -> *mut c_void;
type TxSetBuffer = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut ndk_sys::AHardwareBuffer, i32);
type TxSetDataSpace = unsafe extern "C" fn(*mut c_void, *mut c_void, i32);
type TxSetExtended = unsafe extern "C" fn(*mut c_void, *mut c_void, f32, f32);
type TxSetPosition = unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32);
type TxSetZ = unsafe extern "C" fn(*mut c_void, *mut c_void, i32);
type TxSetVisibility = unsafe extern "C" fn(*mut c_void, *mut c_void, i8);
type TxApply = unsafe extern "C" fn(*mut c_void);
type TxDelete = unsafe extern "C" fn(*mut c_void);

/// Run the probe against the app window if armed and not yet run.
pub fn run_once(window: *mut ndk_sys::ANativeWindow) {
    if !ARMED.load(Ordering::Relaxed) || RAN.swap(true, Ordering::Relaxed) {
        return;
    }
    unsafe { probe(window) }
}

unsafe fn probe(window: *mut ndk_sys::ANativeWindow) {
    let lib = libc::dlopen(c"libandroid.so".as_ptr(), libc::RTLD_NOW);
    if lib.is_null() {
        log::warn!("HDR PROBE: libandroid.so did not open");
        return;
    }
    macro_rules! sym {
        ($name:literal, $ty:ty) => {{
            let p = libc::dlsym(lib, concat!($name, "\0").as_ptr() as *const c_char);
            if p.is_null() {
                log::warn!("HDR PROBE: {} missing — probe abandoned", $name);
                return;
            }
            core::mem::transmute::<*mut c_void, $ty>(p)
        }};
    }
    let create = sym!("ASurfaceControl_createFromWindow", CreateFromWindow);
    let tx_create = sym!("ASurfaceTransaction_create", TxCreate);
    let set_buffer = sym!("ASurfaceTransaction_setBuffer", TxSetBuffer);
    let set_ds = sym!("ASurfaceTransaction_setBufferDataSpace", TxSetDataSpace);
    let set_ext = sym!("ASurfaceTransaction_setExtendedRangeBrightness", TxSetExtended);
    let set_pos = sym!("ASurfaceTransaction_setPosition", TxSetPosition);
    let set_z = sym!("ASurfaceTransaction_setZOrder", TxSetZ);
    let set_vis = sym!("ASurfaceTransaction_setVisibility", TxSetVisibility);
    let apply = sym!("ASurfaceTransaction_apply", TxApply);
    let tx_delete = sym!("ASurfaceTransaction_delete", TxDelete);

    let probes: [(&core::ffi::CStr, i32); 2] = [
        (c"fluor_hdr_probe_bt2020_linear_ext", STANDARD_BT2020 | TRANSFER_LINEAR | RANGE_EXTENDED),
        (c"fluor_hdr_probe_scrgb_linear", STANDARD_BT709 | TRANSFER_LINEAR | RANGE_EXTENDED),
    ];
    for (i, (name, dataspace)) in probes.iter().enumerate() {
        let sc = create(window, name.as_ptr());
        if sc.is_null() {
            log::warn!("HDR PROBE: {:?} — createFromWindow returned null", name);
            continue;
        }
        let Some(buf) = f16_square(2.0) else {
            log::warn!("HDR PROBE: {:?} — F16 buffer allocation or lock failed", name);
            continue;
        };
        let tx = tx_create();
        set_buffer(tx, sc, buf, -1);
        set_ds(tx, sc, *dataspace);
        set_ext(tx, sc, 2.0, 2.0);
        set_pos(tx, sc, 16 + (SIDE as i32 + 16) * i as i32, 240);
        set_z(tx, sc, 1000);
        set_vis(tx, sc, 1);
        apply(tx);
        tx_delete(tx);
        ndk_sys::AHardwareBuffer_release(buf); // the transaction holds its own reference
        log::info!("HDR PROBE: {:?} applied — dataspace 0x{:08X}, linear 2.0, extended brightness (2, 2)", name, dataspace);
    }
}

/// A SIDE×SIDE RGBA F16 buffer, CPU-filled with linear `v` in every colour channel and α 1.
unsafe fn f16_square(v: f32) -> Option<*mut ndk_sys::AHardwareBuffer> {
    let desc = ndk_sys::AHardwareBuffer_Desc {
        width: SIDE,
        height: SIDE,
        layers: 1,
        format: ndk_sys::AHardwareBuffer_Format::AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT.0,
        usage: ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0
            | ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_COMPOSER_OVERLAY.0
            | ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE.0,
        stride: 0,
        rfu0: 0,
        rfu1: 0,
    };
    let mut buf: *mut ndk_sys::AHardwareBuffer = core::ptr::null_mut();
    if ndk_sys::AHardwareBuffer_allocate(&desc, &mut buf) != 0 || buf.is_null() {
        return None;
    }
    let mut got: ndk_sys::AHardwareBuffer_Desc = core::mem::zeroed();
    ndk_sys::AHardwareBuffer_describe(buf, &mut got);
    let mut ptr: *mut c_void = core::ptr::null_mut();
    let usage = ndk_sys::AHardwareBuffer_UsageFlags::AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN.0;
    if ndk_sys::AHardwareBuffer_lock(buf, usage, -1, core::ptr::null(), &mut ptr) != 0 || ptr.is_null() {
        ndk_sys::AHardwareBuffer_release(buf);
        return None;
    }
    let half = half_bits(v);
    let one = half_bits(1.0);
    for y in 0..SIDE as usize {
        let row = (ptr as *mut u16).add(y * got.stride as usize * 4);
        for x in 0..SIDE as usize {
            let px = row.add(x * 4);
            *px = half;
            *px.add(1) = half;
            *px.add(2) = half;
            *px.add(3) = one;
        }
    }
    ndk_sys::AHardwareBuffer_unlock(buf, core::ptr::null_mut());
    Some(buf)
}

/// IEEE half bits for the small positive values the probe writes (no subnormals, no rounding beyond truncation).
fn half_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    ((exp as u16) << 10) | ((b >> 13) & 0x3FF) as u16
}
