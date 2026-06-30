//! macOS global mouse monitor for click-thru re-entry detection.
//!
//! When `ignoresMouseEvents = true`, macOS stops delivering CursorMoved to our window.
//! We install a global NSEvent monitor that fires on mouseMoved globally, checks the cursor position against the window rect, and flags re-entry when the cursor moves back inside.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::Arc;

/// Shared state between the global monitor callback and the host event loop.
pub(super) struct HittestMonitor {
    /// Set by the monitor callback when the cursor is inside the window rect.
    pub reenter_flag: Arc<AtomicBool>,
    /// Window rect in screen coords (top-left origin, matching winit convention).
    pub win_x: Arc<AtomicI32>,
    pub win_y: Arc<AtomicI32>,
    pub win_w: Arc<AtomicU32>,
    pub win_h: Arc<AtomicU32>,
    /// Screen height for Y-flip (NSEvent uses bottom-left origin).
    pub screen_h: Arc<AtomicU32>,
    _monitor: *mut objc2::runtime::AnyObject,
}

unsafe impl Send for HittestMonitor {}

impl HittestMonitor {
    /// Install a global mouse-moved monitor.
    pub fn install(screen_h: u32) -> Option<Self> {
        use objc2::rc::Retained;
        use objc2::runtime::AnyObject;
        use objc2_app_kit::NSEvent;
        use objc2_app_kit::NSEventMask;
        use objc2_foundation::NSPoint;

        let reenter_flag = Arc::new(AtomicBool::new(false));
        let wx = Arc::new(AtomicI32::new(0));
        let wy = Arc::new(AtomicI32::new(0));
        let ww = Arc::new(AtomicU32::new(0));
        let wh = Arc::new(AtomicU32::new(0));
        let sh = Arc::new(AtomicU32::new(screen_h));

        let flag = reenter_flag.clone();
        let wx2 = wx.clone();
        let wy2 = wy.clone();
        let ww2 = ww.clone();
        let wh2 = wh.clone();
        let sh2 = sh.clone();

        let mask = NSEventMask::MouseMoved
            | NSEventMask::LeftMouseDragged
            | NSEventMask::RightMouseDragged;

        let block = block2::RcBlock::new(move |_event: std::ptr::NonNull<NSEvent>| {
            let loc: NSPoint = NSEvent::mouseLocation();
            // NSEvent mouseLocation: bottom-left origin. Flip Y to top-left.
            let screen_h = sh2.load(Ordering::Relaxed) as f64;
            let cx = loc.x as i32;
            let cy = (screen_h - loc.y) as i32;

            let rx = wx2.load(Ordering::Relaxed);
            let ry = wy2.load(Ordering::Relaxed);
            let rw = ww2.load(Ordering::Relaxed) as i32;
            let rh = wh2.load(Ordering::Relaxed) as i32;

            if cx >= rx && cx < rx + rw && cy >= ry && cy < ry + rh {
                flag.store(true, Ordering::Relaxed);
            }
        });

        let monitor: Option<Retained<AnyObject>> =
            NSEvent::addGlobalMonitorForEventsMatchingMask_handler(mask, &block);

        monitor.map(|m| {
            let raw = Retained::into_raw(m);
            HittestMonitor {
                reenter_flag,
                win_x: wx,
                win_y: wy,
                win_w: ww,
                win_h: wh,
                screen_h: sh,
                _monitor: raw as *mut AnyObject,
            }
        })
    }

    /// Update the window rect (call after move/resize).
    pub fn update_rect(&self, x: i32, y: i32, w: u32, h: u32) {
        self.win_x.store(x, Ordering::Relaxed);
        self.win_y.store(y, Ordering::Relaxed);
        self.win_w.store(w, Ordering::Relaxed);
        self.win_h.store(h, Ordering::Relaxed);
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
