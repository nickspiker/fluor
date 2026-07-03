//! Windows present path for the fullscreen-transparent compositor.
//!
//! fluor owns a fullscreen borderless OS window and paints the visible "window" as a sub-rect, with every pixel outside it left at α=0 so the desktop shows through, and clicks outside it pass thru to whatever's underneath. On X11 that's an XShape input region + a transparent visual; on macOS it's a transparent NSWindow + a global hit-test monitor. On Windows neither of those exists, and softbuffer's present is an opaque `BitBlt` — so a plain softbuffer window is OPAQUE and screen-sized, which is the "screen/2 opaque box, no click-thru" bug.
//!
//! The Windows-native answer is a **layered window**: `WS_EX_LAYERED` + `UpdateLayeredWindow` blends a 32-bit premultiplied-BGRA bitmap onto the desktop per-pixel. Two things fall out of that for free:
//!   1. Per-pixel alpha — the α=0 pixels outside the visible window are fully transparent (desktop shows).
//!   2. Click-through — Windows routes mouse input through fully-transparent (α=0) pixels of a layered window automatically, so no separate input-region call is needed (the analog of XShape here).
//!
//! So this single present mechanism fixes BOTH Windows symptoms. The window is created `WS_EX_LAYERED` in `resumed`; this module does the per-frame present from fluor's owned `persistent_screen` buffer.

use std::sync::Arc;
use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

use windows::Win32::Foundation::{HWND, POINT, SIZE};
use windows::Win32::Graphics::Gdi::{
    CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, GetDC, ReleaseDC, SelectObject,
    BITMAPINFO, BITMAPINFOHEADER, BI_RGB, DIB_RGB_COLORS, HBITMAP, HDC, HGDIOBJ,
};
use windows::Win32::UI::WindowsAndMessaging::{
    GetWindowLongPtrW, SetWindowLongPtrW, UpdateLayeredWindow, GWL_EXSTYLE, ULW_ALPHA, WS_EX_LAYERED,
};

/// Pull the Win32 `HWND` out of a winit window. Returns `None` if the window isn't a Win32 window (shouldn't happen on this target, but the present path no-ops rather than panicking if so).
fn hwnd(window: &Window) -> Option<HWND> {
    match window.window_handle().ok()?.as_raw() {
        RawWindowHandle::Win32(h) => Some(HWND(h.hwnd.get() as *mut _)),
        _ => None,
    }
}

/// Ensure the window has the `WS_EX_LAYERED` extended style so `UpdateLayeredWindow` is valid. Called once after window creation. Idempotent (OR-ing an already-set bit is a no-op).
pub fn make_layered(window: &Arc<Window>) {
    let Some(hwnd) = hwnd(window) else { return };
    unsafe {
        let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED.0 as isize);
    }
}

/// Present `persistent_screen` (fluor's owned `0xAARRGGBB`-per-pixel screen buffer, `screen_w × screen_h`) to the layered window via `UpdateLayeredWindow`.
///
/// `UpdateLayeredWindow` requires a 32-bit top-down DIB in **premultiplied BGRA**. fluor's buffer is `0xAARRGGBB` (the same packing softbuffer/wgpu consume) NOT premultiplied, so we convert per-pixel
/// into a freshly-created DIB section, then blit. The whole screen-sized surface is updated each frame;
/// the cost is one screen-sized copy+premultiply, matching the existing `persistent_screen → back buffer` copy the softbuffer path already pays.
pub fn present(window: &Arc<Window>, persistent_screen: &[u32], screen_w: u32, screen_h: u32) {
    let Some(hwnd) = hwnd(window) else { return };
    let w = screen_w as i32;
    let h = screen_h as i32;
    if w <= 0 || h <= 0 || persistent_screen.len() < (screen_w as usize * screen_h as usize) {
        return;
    }

    unsafe {
        // Screen DC (for the layered blend source) + a memory DC holding our DIB.
        let screen_dc: HDC = GetDC(HWND(std::ptr::null_mut()));
        if screen_dc.is_invalid() {
            return;
        }
        let mem_dc: HDC = CreateCompatibleDC(screen_dc);
        if mem_dc.is_invalid() {
            ReleaseDC(HWND(std::ptr::null_mut()), screen_dc);
            return;
        }

        // Top-down 32bpp BGRA DIB (negative height = top-down so row 0 is the top, matching our buffer).
        let mut bmi = BITMAPINFO::default();
        bmi.bmiHeader = BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: w,
            biHeight: -h,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        };
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        let dib: HBITMAP =
            match CreateDIBSection(mem_dc, &bmi, DIB_RGB_COLORS, &mut bits, None, 0) {
                Ok(b) if !b.is_invalid() && !bits.is_null() => b,
                _ => {
                    let _ = DeleteDC(mem_dc);
                    ReleaseDC(HWND(std::ptr::null_mut()), screen_dc);
                    return;
                }
            };

        // Convert 0xAARRGGBB (straight alpha) → premultiplied BGRA, straight into the DIB.
        let dst = std::slice::from_raw_parts_mut(bits as *mut u32, (w * h) as usize);
        for (d, &src) in dst.iter_mut().zip(persistent_screen.iter()) {
            let a = (src >> 24) & 0xFF;
            let r = (src >> 16) & 0xFF;
            let g = (src >> 8) & 0xFF;
            let b = src & 0xFF;
            // Premultiply each channel by alpha (UpdateLayeredWindow with ULW_ALPHA expects it), and pack BGRA (DIB byte order is B,G,R,A in memory = 0xAARRGGBB little-endian — same as src once premultiplied, so we repack with the premultiplied channels).
            // Floor `>> 8` with the `a + (a >> 7)` weight bump instead of `/ 255`: α=255 passes the channel thru exactly, α=0 floors to 0, interior within 1 LSB — and premul ≤ α holds for every (channel, α) pair (verified exhaustively), which ULW_ALPHA requires. Three divisions per pixel per present was the most expensive `/ 255` in the tree.
            let ae = a + (a >> 7);
            let pr = ((r * ae) >> 8) & 0xFF;
            let pg = ((g * ae) >> 8) & 0xFF;
            let pb = ((b * ae) >> 8) & 0xFF;
            *d = (a << 24) | (pr << 16) | (pg << 8) | pb;
        }

        let old: HGDIOBJ = SelectObject(mem_dc, dib);

        let mut src_pos = POINT { x: 0, y: 0 };
        let mut size = SIZE { cx: w, cy: h };
        // dst position: the window is fullscreen at (0,0), so the layered surface maps 1:1 to the screen.
        let mut dst_pos = POINT { x: 0, y: 0 };
        let blend = windows::Win32::Graphics::Gdi::BLENDFUNCTION {
            BlendOp: windows::Win32::Graphics::Gdi::AC_SRC_OVER as u8,
            BlendFlags: 0,
            SourceConstantAlpha: 255,
            AlphaFormat: windows::Win32::Graphics::Gdi::AC_SRC_ALPHA as u8,
        };

        let _ = UpdateLayeredWindow(
            hwnd,
            screen_dc,
            Some(&mut dst_pos),
            Some(&mut size),
            mem_dc,
            Some(&mut src_pos),
            windows::Win32::Foundation::COLORREF(0),
            Some(&blend),
            ULW_ALPHA,
        );

        // Tear down GDI objects (must restore the old bitmap before deleting the DIB).
        SelectObject(mem_dc, old);
        let _ = DeleteObject(dib);
        let _ = DeleteDC(mem_dc);
        ReleaseDC(HWND(std::ptr::null_mut()), screen_dc);
    }
}
