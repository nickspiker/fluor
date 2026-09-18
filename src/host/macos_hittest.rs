//! macOS global mouse monitor for click-thru re-entry detection.
//!
//! When `ignoresMouseEvents = true`, macOS stops delivering CursorMoved to our window.
//! We install a global NSEvent monitor that fires on mouseMoved globally, checks the cursor position against the HITTABLE rect (the window rect inflated by the resize band — the same rect the shell's click-thru decision and the input region use, so the three never disagree), and on the entry EDGE flags re-entry and wakes the event loop with one `request_redraw` on the armed home window. The host reads the flag in RedrawRequested and flips hittest back on. One wake per entry, no polling: the flag stays set while the cursor is inside and the host clears it when it next flips hittest off.
//!
//! Why an edge and not a poll (2026-09-18): the old design only checked the flag inside RedrawRequested and kept itself alive by requesting another redraw each vsync — but nothing requested the FIRST redraw when hittest flipped off. If the cursor left the window across a hovered widget, the un-hover repaint started the poll; if it left across plain background, no repaint, no poll, and the window stayed click-thru and cursor-blind until some incidental redraw. That was the "resize arrows sometimes don't come back" half of the field report, and while it did run the poll burned a frame per vsync with the cursor parked outside.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use winit::window::Window;

/// Shared state between the global monitor callback and the host event loop.
pub(super) struct HittestMonitor {
    /// Set by the monitor callback when the cursor is inside the window rect.
    pub reenter_flag: Arc<AtomicBool>,
    /// Window rect in GLOBAL desktop POINTS (top-left origin, primary screen's top edge at y = 0 — the same space the shell's `window_rect` lives in). `NSEvent::mouseLocation` is already global points, so multi-monitor needs no per-surface translation here: the ONE window rect covers the cursor test on every screen.
    pub win_x: Arc<AtomicI32>,
    pub win_y: Arc<AtomicI32>,
    pub win_w: Arc<AtomicU32>,
    pub win_h: Arc<AtomicU32>,
    /// PRIMARY screen height in POINTS for the Y-flip (NSEvent uses a bottom-left origin whose reference is the primary screen frame).
    pub screen_h: Arc<AtomicU32>,
    /// The home window to wake on the entry edge — armed by the host when it flips hittest off (the home can change across monitors, so it is re-armed every time). `try_lock` in the callback: a contended lock just defers the wake to the next global move, it never blocks the run loop.
    wake: Arc<Mutex<Option<Arc<Window>>>>,
    _monitor: *mut objc2::runtime::AnyObject,
}

unsafe impl Send for HittestMonitor {}

impl HittestMonitor {
    /// Install a global mouse-moved monitor. `screen_h` = the PRIMARY screen's height in POINTS (the mouseLocation flip reference).
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
        let wake: Arc<Mutex<Option<Arc<Window>>>> = Arc::new(Mutex::new(None));

        let flag = reenter_flag.clone();
        let wx2 = wx.clone();
        let wy2 = wy.clone();
        let ww2 = ww.clone();
        let wh2 = wh.clone();
        let sh2 = sh.clone();
        let wake2 = wake.clone();

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
                // The ENTRY edge: the first move inside since the host last cleared the flag wakes the loop once. Later moves inside see the flag already set and do nothing.
                if !flag.swap(true, Ordering::Relaxed) {
                    if let Ok(w) = wake2.try_lock() {
                        if let Some(w) = w.as_ref() {
                            w.request_redraw();
                        }
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
                win_x: wx,
                win_y: wy,
                win_w: ww,
                win_h: wh,
                screen_h: sh,
                wake,
                _monitor: raw as *mut AnyObject,
            }
        })
    }

    /// Update the HITTABLE rect (call after move/resize) — the window rect inflated by the resize band, GLOBAL desktop points, same space as the shell's `window_rect`. Must be the same rect the shell's click-thru decision uses (`hittable_rect`), or the two disagree in the band.
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

    /// Arm the entry-edge wake on `window` and clear any stale flag — called by the host at the moment it flips hittest OFF, so the next global move inside the hittable rect wakes exactly once.
    pub fn arm(&self, window: Arc<Window>) {
        if let Ok(mut g) = self.wake.lock() {
            *g = Some(window);
        }
        self.reenter_flag.store(false, Ordering::Relaxed);
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
