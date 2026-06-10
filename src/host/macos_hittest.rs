//! macOS global mouse monitor for click-through re-entry detection.
//!
//! When `ignoresMouseEvents = true`, macOS stops delivering CursorMoved to our window.
//! We install a global NSEvent monitor that fires on mouseMoved/mouseEntered globally,
//! checks the cursor position, and flips hittest back on when the cursor re-enters
//! an opaque region of our persistent_screen buffer.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// Shared state between the global monitor callback and the host event loop.
pub(super) struct HittestMonitor {
    /// Set by the monitor callback when the cursor is over an opaque pixel.
    pub reenter_flag: Arc<AtomicBool>,
    /// Screen dimensions for index calculation.
    pub screen_w: Arc<AtomicU32>,
    pub screen_h: Arc<AtomicU32>,
    /// Pointer to the persistent_screen buffer (raw ptr for cross-thread access).
    /// Updated each frame by the host. The monitor reads it to check alpha.
    pub screen_ptr: Arc<AtomicU64>,
    pub screen_len: Arc<AtomicU64>,
    _monitor: *mut objc2::runtime::AnyObject,
}

// The monitor handle is an ObjC object we release on drop.
unsafe impl Send for HittestMonitor {}

use std::sync::atomic::AtomicU64;

impl HittestMonitor {
    /// Install a global mouse-moved monitor. Returns None if the API call fails.
    pub fn install(screen_w: u32, screen_h: u32) -> Option<Self> {
        use objc2::rc::Retained;
        use objc2::runtime::AnyObject;
        use objc2_app_kit::NSEvent;
        use objc2_app_kit::NSEventMask;
        use objc2_foundation::NSPoint;

        let reenter_flag = Arc::new(AtomicBool::new(false));
        let sw = Arc::new(AtomicU32::new(screen_w));
        let sh = Arc::new(AtomicU32::new(screen_h));
        let sptr = Arc::new(AtomicU64::new(0));
        let slen = Arc::new(AtomicU64::new(0));

        let flag = reenter_flag.clone();
        let sw2 = sw.clone();
        let sh2 = sh.clone();
        let sptr2 = sptr.clone();
        let slen2 = slen.clone();

        let mask = NSEventMask::MouseMoved
            | NSEventMask::LeftMouseDragged
            | NSEventMask::RightMouseDragged;

        let block = block2::RcBlock::new(move |_event: std::ptr::NonNull<NSEvent>| {
            let loc: NSPoint = NSEvent::mouseLocation();
            // NSEvent mouseLocation is in screen coords, origin bottom-left.
            let w = sw2.load(Ordering::Relaxed) as f64;
            let h = sh2.load(Ordering::Relaxed) as f64;
            let cx = loc.x as usize;
            let cy = (h - loc.y) as usize; // flip Y
            let scr_w = w as usize;
            if cx < scr_w && cy < (h as usize) {
                let idx = cy * scr_w + cx;
                let ptr = sptr2.load(Ordering::Relaxed) as *const u32;
                let len = slen2.load(Ordering::Relaxed) as usize;
                if !ptr.is_null() && idx < len {
                    let pixel = unsafe { *ptr.add(idx) };
                    let alpha = ((pixel >> 24) & 0xFF) as u8;
                    if alpha >= 10 {
                        flag.store(true, Ordering::Relaxed);
                    }
                }
            }
        });

        let monitor: Option<Retained<AnyObject>> =
            NSEvent::addGlobalMonitorForEventsMatchingMask_handler(mask, &block);

        monitor.map(|m| {
            let raw = Retained::into_raw(m);
            HittestMonitor {
                reenter_flag,
                screen_w: sw,
                screen_h: sh,
                screen_ptr: sptr,
                screen_len: slen,
                _monitor: raw as *mut AnyObject,
            }
        })
    }

    /// Update the screen buffer pointer (call each frame after finalize).
    pub fn update_screen(&self, buf: &[u32], w: u32, h: u32) {
        self.screen_ptr.store(buf.as_ptr() as u64, Ordering::Relaxed);
        self.screen_len.store(buf.len() as u64, Ordering::Relaxed);
        self.screen_w.store(w, Ordering::Relaxed);
        self.screen_h.store(h, Ordering::Relaxed);
    }

    /// Check and clear the re-entry flag.
    pub fn check_reenter(&self) -> bool {
        self.reenter_flag.swap(false, Ordering::Relaxed)
    }
}

impl Drop for HittestMonitor {
    fn drop(&mut self) {
        if !self._monitor.is_null() {
            use objc2_app_kit::NSEvent;
            unsafe {
                let obj = objc2::rc::Retained::from_raw(self._monitor).unwrap();
                NSEvent::removeMonitor(&obj);
            }
        }
    }
}
