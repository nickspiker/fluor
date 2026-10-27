//! The ferros host: a `FluorApp` on the phone as a ferros process (ferros STD-TARGET.md). No window system — the display server owns the panel, and an app in front draws 16-bit pixels into the `app-surface` region and asks the display to show them (op 22); touches come from the `touch` server's endpoint as contact records. This shell is the Android shell's twin with those two in place of a `NativeWindow` and the JNI touch callbacks: the same press-hold-release arbiter, the same drag-to-scroll, the same `Context`.
//!
//! The process holds three capabilities the caller names by grant slot (`Caps`): the display endpoint, the touch endpoint, the surface region. The syscalls are inlined here (ferros_sys is not a published crate): `call`, `dma`, `sleep`, the counter.
//!
//! Pixels: fluor's `0xααRRGGBB` darkness buffer becomes the display's working values — gamma 2 encoded, white = 16384 (Q14), R, G, B in bits 0-15, 16-31, 32-47 of a u64 — through one 256-entry table (fluor's bytes are gamma 2.2, so the display wants `(c/255)^1.1`). Opacity composites over black. The whole frame is converted and shown each time the app drew; the damage rect narrows that later.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::canvas::{Damage, PixelRect};
use crate::coord::Coord;
use crate::event::{ElementState, Event as FEvent, ModifiersState as FModifiersState, MouseButton, MouseScrollDelta};
use crate::geom::Viewport;
use crate::host::app::{Context, FluorApp};
use crate::host::wake::{WakeError, WakeSender};
use crate::host::{EventResponse, WindowHandle};
use crate::paint::{HitId, HIT_NONE};
use crate::text::TextRenderer;

/// The kernel, as a ferros process reaches it (KERNEL.md, process ABI).
mod sys {
    use core::arch::asm;

    pub const ERR: u64 = u64::MAX;
    const CALL: u64 = 1;
    const DMA: u64 = 7;
    const SLEEP: u64 = 9;

    #[inline(always)]
    pub fn svc(n: u64, a0: u64, a1: u64, a2: u64, a3: u64, a4: u64) -> u64 {
        let r: u64;
        unsafe { asm!("svc #0", inlateout("x0") a0 => r, in("x1") a1, in("x2") a2, in("x3") a3, in("x4") a4, in("x8") n, options(nostack)) };
        r
    }

    #[inline(always)]
    pub fn arc_base() -> u64 {
        let b: u64;
        unsafe { asm!("mov {}, x28", out(reg) b, options(nomem, nostack, preserves_flags)) };
        b
    }

    #[inline(always)]
    pub fn off<T: ?Sized>(p: *const T) -> u64 {
        (p as *const u8 as u64).wrapping_sub(arc_base())
    }

    /// The grants page is arc offset 0: a u32 count, then 32-byte tokens; slot 0 is the console.
    pub fn cap(slot: u64) -> u64 {
        4 + 32 * slot
    }

    pub fn call(cap: u64, msg: &[u8], reply: &mut [u8]) -> u64 {
        svc(CALL, cap, off(msg.as_ptr()), msg.len() as u64, off(reply.as_mut_ptr()), reply.len() as u64)
    }

    pub fn dma(cap: u64) -> u64 {
        svc(DMA, cap, 0, 0, 0, 0)
    }

    pub fn now() -> u64 {
        let t: u64;
        unsafe { asm!("isb", "mrs {}, cntvct_el0", out(reg) t, options(nomem, nostack, preserves_flags)) };
        t
    }

    pub fn freq() -> u64 {
        let f: u64;
        unsafe { asm!("mrs {}, cntfrq_el0", out(reg) f, options(nomem, nostack, preserves_flags)) };
        f
    }

    pub fn sleep_until(deadline: u64) {
        svc(SLEEP, deadline, 0, 0, 0, 0);
    }
}

/// The grant slots a fluor app on ferros is spawned with: `--use display --use touch --dma app-surface`, in whatever order the spawner chose.
#[derive(Clone, Copy, Debug)]
pub struct Caps {
    pub display: u64,
    pub touch: u64,
    pub surface: u64,
}

/// The dirty flag and the clipboard: all a window is here.
pub struct FerrosWindow {
    dirty: AtomicBool,
    clipboard: Mutex<Option<String>>,
}

impl FerrosWindow {
    fn new() -> Self {
        Self { dirty: AtomicBool::new(true), clipboard: Mutex::new(None) }
    }

    fn take_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::AcqRel)
    }

    fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

impl WindowHandle for FerrosWindow {
    fn request_redraw(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    fn set_clipboard(&self, text: &str) {
        if let Ok(mut c) = self.clipboard.lock() {
            *c = Some(String::from(text));
        }
    }
}

/// The wake sender an app's background threads hold on ferros: events queue here and the host drains them at the top of each frame, waking the app — there is no event loop to post to, the frame loop is the loop.
pub struct FerrosWakeSender<E> {
    queue: Mutex<Vec<E>>,
}

impl<E: Send + 'static> WakeSender<E> for FerrosWakeSender<E> {
    fn send(&self, event: E) -> Result<(), WakeError> {
        self.queue.lock().unwrap().push(event);
        Ok(())
    }
}

/// A contact record from the touch server (12 bytes): id, phase (0 enter, 1 move, 2 leave), kind, pressure, x, y, and the rest unused here.
pub struct Contact {
    pub id: u8,
    pub phase: u8,
    pub x: u16,
    pub y: u16,
}

/// The app and everything the host keeps for it between frames.
pub struct FerrosShell<A: FluorApp> {
    app: A,
    text: TextRenderer,
    window: FerrosWindow,
    viewport: Viewport,
    cursor_x: Coord,
    cursor_y: Coord,
    modifiers: FModifiersState,
    scratch: Vec<u32>,
    clip_mask: Vec<u8>,
    pending_damage: Damage,
    pointer: crate::host::pointer::PointerArbiter,
    touch_down: bool,
    touch_last_x: Coord,
    touch_last_y: Coord,
    touch_travel: Coord,
    /// The contact the pointer follows: the first finger down, until it leaves.
    finger: Option<u8>,
    close: bool,
    /// Frames drawn, for the first-frame rule and the console line.
    pub frames: u64,
    wake: Arc<FerrosWakeSender<A::UserEvent>>,
}

impl<A: FluorApp> FerrosShell<A> {
    pub fn new(mut app: A, width: u32, height: u32, ru: Coord) -> Self {
        let viewport = Viewport::new(width, height).with_ru(ru);
        let wake = Arc::new(FerrosWakeSender { queue: Mutex::new(Vec::new()) });
        app.set_event_proxy(wake.clone());
        let px = (width as usize) * (height as usize);
        let mut shell = Self {
            app,
            text: TextRenderer::new(),
            window: FerrosWindow::new(),
            viewport,
            cursor_x: 0.0,
            cursor_y: 0.0,
            modifiers: FModifiersState::empty(),
            scratch: alloc::vec![0u32; px],
            clip_mask: alloc::vec![255u8; px],
            pending_damage: Damage::new(),
            pointer: crate::host::pointer::PointerArbiter::new(),
            touch_down: false,
            touch_last_x: 0.0,
            touch_last_y: 0.0,
            touch_travel: 0.0,
            finger: None,
            close: false,
            frames: 0,
            wake,
        };
        shell.with_context(|app, ctx| app.init(ctx));
        shell.with_context(|app, ctx| app.on_resize(width, height, ctx));
        shell
    }

    /// Tick the app and, when anything is dirty, render into the scratch buffer. True when a frame was drawn.
    pub fn draw(&mut self) -> bool {
        // What the background threads sent since the last frame, delivered first.
        let events: Vec<A::UserEvent> = core::mem::take(&mut *self.wake.queue.lock().unwrap());
        for e in events {
            if self.with_context(|app, ctx| app.on_user_event(e, ctx)) == EventResponse::Close {
                self.close = true;
            }
            self.window.mark_dirty();
        }
        if let Some(ru) = self.app.take_zoom_request() {
            if ru.is_finite() && ru > 0.0 {
                self.set_zoom_and_relayout(ru);
            }
        }
        if self.with_context(|app, ctx| app.tick(ctx)) {
            self.window.mark_dirty();
        }
        if let Some(at) = self.app.wake_at() {
            if at <= Instant::now() {
                self.window.mark_dirty();
            }
        }
        if !self.window.take_dirty() {
            return false;
        }
        let (win_w, win_h) = (self.viewport.width_px as usize, self.viewport.height_px as usize);
        let viewport_rect = PixelRect::new(0, 0, win_w, win_h);
        // The first frame is the whole frame, whatever the app says is dirty: nothing has been shown yet.
        let damage_clip = if self.frames == 0 { viewport_rect } else { self.app.damage_rect(self.viewport).unwrap_or(viewport_rect) };
        if damage_clip.is_empty() {
            return false;
        }
        self.frames += 1;
        if damage_clip != viewport_rect {
            if let Some((rect, dy)) = self.app.scroll_hint(self.viewport) {
                crate::paint::scroll_copy_rect(&mut self.scratch, win_w, win_h, rect.x0, rect.y0, rect.x1, rect.y1, dy);
            }
        }
        clear_scratch_rect(&mut self.scratch, win_w, damage_clip);
        self.pending_damage.clear();
        self.with_context_render(damage_clip, |app, scratch, ctx| app.render(scratch, ctx));
        true
    }

    /// The scratch buffer as the display's working values, into the surface (row-major, `w × h` u64).
    pub fn present(&self, surface: &mut [u64]) {
        let lut = gamma_table();
        // The surface and the scratch are both the frame, w x h; zip pairs them.
        for (dst, &px) in surface.iter_mut().zip(self.scratch.iter()) {
            let a = px >> 24;
            let (r, g, b) = (255 - ((px >> 16) & 255), 255 - ((px >> 8) & 255), 255 - (px & 255));
            let (r, g, b) = ((r * a + 127) / 255, (g * a + 127) / 255, (b * a + 127) / 255);
            *dst = lut[r as usize] as u64 | (lut[g as usize] as u64) << 16 | (lut[b as usize] as u64) << 32;
        }
    }

    /// Contact records in, pointer events out: the first finger down is the pointer; a second finger is ignored for now (pinch comes with the input work).
    pub fn on_contacts(&mut self, contacts: &[Contact]) {
        for c in contacts {
            match (c.phase, self.finger) {
                (0, None) => {
                    self.finger = Some(c.id);
                    self.on_pointer(&[FEvent::CursorMoved { x: c.x as Coord, y: c.y as Coord }, FEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left }]);
                }
                (1, Some(id)) if id == c.id => self.on_pointer(&[FEvent::CursorMoved { x: c.x as Coord, y: c.y as Coord }]),
                (2, Some(id)) if id == c.id => {
                    self.finger = None;
                    self.on_pointer(&[FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left }]);
                }
                _ => {}
            }
        }
    }

    fn on_pointer(&mut self, evs: &[FEvent]) {
        for ev in evs {
            if let FEvent::CursorMoved { x, y } = ev {
                self.cursor_x = *x;
                self.cursor_y = *y;
            }
            match ev {
                FEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                    self.pointer.on_down(self.hit_under_cursor());
                    self.touch_down = true;
                    self.touch_last_x = self.cursor_x;
                    self.touch_last_y = self.cursor_y;
                    self.touch_travel = 0.0;
                }
                FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left, .. } => {
                    if let Some(id) = self.pointer.on_up(self.hit_under_cursor()) {
                        self.dispatch_activate(id);
                    }
                    self.touch_down = false;
                }
                FEvent::CursorMoved { .. } => {
                    if self.pointer.on_move(self.hit_under_cursor()) {
                        self.window.mark_dirty();
                    }
                }
                FEvent::CursorLeft => {
                    self.pointer.on_cancel();
                    self.touch_down = false;
                }
                _ => {}
            }
            if self.dispatch(ev) == EventResponse::Close {
                self.close = true;
            }
            if let FEvent::CursorMoved { x: cx, y: cy } = ev {
                if self.touch_down {
                    let dx = *cx - self.touch_last_x;
                    let dy = *cy - self.touch_last_y;
                    self.touch_last_x = *cx;
                    self.touch_last_y = *cy;
                    self.touch_travel += dx.abs() + dy.abs();
                    if self.touch_travel > self.viewport.effective_span() / 64.0 && self.pointer.held_id() != HIT_NONE {
                        self.pointer.on_cancel();
                        self.window.mark_dirty();
                    }
                    if dx != 0.0 || dy != 0.0 {
                        let _ = self.dispatch(&FEvent::MouseWheel { delta: MouseScrollDelta::Pixels(dx, dy) });
                    }
                }
            }
        }
        self.window.mark_dirty();
    }

    fn hit_under_cursor(&self) -> HitId {
        let (x, y) = (self.cursor_x as i32, self.cursor_y as i32);
        if x < 0 || y < 0 {
            return HIT_NONE;
        }
        match self.app.hit_test_map() {
            Some((map, w, h)) if (x as usize) < w && (y as usize) < h => map[(y as usize) * w + (x as usize)],
            _ => HIT_NONE,
        }
    }

    fn dispatch_activate(&mut self, hit_id: HitId) {
        let (x, y, mods) = (self.cursor_x, self.cursor_y, self.modifiers);
        self.with_context(|app, ctx| {
            let _ = app.on_activate(hit_id, x, y, mods, ctx);
        });
    }

    fn dispatch(&mut self, ev: &FEvent) -> EventResponse {
        self.with_context(|app, ctx| app.on_event(ev, ctx))
    }

    fn set_zoom_and_relayout(&mut self, new_ru: f32) {
        let ru_before = self.viewport.ru;
        self.viewport.set_zoom(new_ru);
        self.window.mark_dirty();
        let (w, h) = (self.viewport.width_px, self.viewport.height_px);
        let factor = self.viewport.ru / ru_before;
        let (ax, ay) = (w as Coord / 2.0, h as Coord / 2.0);
        self.with_context(|app, ctx| {
            if factor != 1.0 {
                app.on_zoom(factor, ax, ay, ctx);
            }
            app.on_resize(w, h, ctx)
        });
    }

    fn with_context<R>(&mut self, f: impl FnOnce(&mut A, &mut Context) -> R) -> R {
        let mut ctx = Context {
            pressed_hit: self.pointer.held_id(),
            viewport: self.viewport,
            text: &mut self.text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &self.window,
            modifiers: self.modifiers,
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
            is_maximized: true,
            window_origin: (0, 0),
            damage_clip: PixelRect::new(0, 0, self.viewport.width_px as usize, self.viewport.height_px as usize),
        };
        f(&mut self.app, &mut ctx)
    }

    fn with_context_render(&mut self, damage_clip: PixelRect, f: impl FnOnce(&mut A, &mut [u32], &mut Context)) {
        let mut ctx = Context {
            pressed_hit: self.pointer.held_id(),
            viewport: self.viewport,
            text: &mut self.text,
            clip_mask: &mut self.clip_mask,
            damage: &mut self.pending_damage,
            window: &self.window,
            modifiers: self.modifiers,
            cursor_x: self.cursor_x,
            cursor_y: self.cursor_y,
            is_maximized: true,
            window_origin: (0, 0),
            damage_clip,
        };
        f(&mut self.app, &mut self.scratch, &mut ctx)
    }
}

/// Clear `scratch[y0..y1][x0..x1]` to α=0 + zero darkness (the desktop and Android shells do the same).
fn clear_scratch_rect(scratch: &mut [u32], width: usize, rect: PixelRect) {
    if rect.is_empty() || width == 0 {
        return;
    }
    let x1 = rect.x1.min(width);
    if rect.x0 >= x1 {
        return;
    }
    let row_len = x1 - rect.x0;
    for y in rect.y0..rect.y1 {
        let base = y * width + rect.x0;
        if base + row_len > scratch.len() {
            break;
        }
        for cell in &mut scratch[base..base + row_len] {
            *cell = 0;
        }
    }
}

/// fluor's gamma-2.2 byte to the display's gamma-2 Q14 working value: `16384 × (c / 255)^1.1`.
fn gamma_table() -> &'static [u16; 256] {
    static TABLE: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [0u16; 256];
        for (c, v) in t.iter_mut().enumerate() {
            *v = (16384.0 * libm::powf(c as f32 / 255.0, 1.1) + 0.5) as u16;
        }
        t
    })
}

/// Run `app` on the panel until it asks to close (an `EventResponse::Close`): frames at most 120 a second, touches polled between them. `ru` is the resolution unit for the viewport (what the device's density would set on Android; 1.0 is fluor's native scale). `size` is the app's frame, centred on the panel, or the whole panel — a 1344 × 2992 frame is 16 MiB of scratch alone, more than a small arc holds.
pub fn run_app<A: FluorApp + 'static>(app: A, caps: Caps, ru: Coord, size: Option<(u32, u32)>) -> Result<(), String> {
    let display = sys::cap(caps.display);
    let touch = sys::cap(caps.touch);
    let mut r = [0u8; 256];
    if sys::call(display, &[5], &mut r) < 4 {
        return Err(String::from("no display server behind the display grant"));
    }
    let (pw, ph) = (u16::from_le_bytes([r[0], r[1]]) as usize, u16::from_le_bytes([r[2], r[3]]) as usize);
    // A frame larger than the panel is a bug in the caller, not something to trim.
    let (w, h) = size.map_or((pw, ph), |(w, h)| (w as usize, h as usize));
    assert!(w <= pw && h <= ph, "the app frame does not fit the panel");
    let (fx, fy) = ((pw - w) / 2, (ph - h) / 2);
    let base = sys::dma(sys::cap(caps.surface));
    if base == 0 || base == sys::ERR || w * h * 8 > 32 << 20 {
        return Err(String::from("no app surface behind the surface grant"));
    }
    let surface = unsafe { core::slice::from_raw_parts_mut(base as *mut u64, w * h) };
    // The touch panel's coordinate range, scaled to the panel's pixels.
    let (sx, sy) = if sys::call(touch, &[2], &mut r) >= 4 {
        let (xm, ym) = (u16::from_le_bytes([r[0], r[1]]) as f32, u16::from_le_bytes([r[2], r[3]]) as f32);
        (if xm > 0.0 { (pw as f32 - 1.0) / xm } else { 1.0 }, if ym > 0.0 { (ph as f32 - 1.0) / ym } else { 1.0 })
    } else {
        (1.0, 1.0)
    };
    println!("fluor: ferros host up, a {w} x {h} frame at ({fx}, {fy}) on a {pw} x {ph} panel");
    let mut shell = FerrosShell::new(app, w as u32, h as u32, ru);
    let frame = sys::freq() / 120;
    let mut contacts: Vec<Contact> = Vec::with_capacity(16);
    let (mut shown, mut touched, mut last_line) = (0u64, 0u64, sys::now());
    loop {
        // A line every ten seconds while there is no window system to look at: frames shown, contacts seen.
        if sys::now().wrapping_sub(last_line) > sys::freq() * 10 {
            last_line = sys::now();
            println!("fluor: {shown} frames shown, {} drawn, {touched} contacts", shell.frames);
        }
        let n = sys::call(touch, &[0], &mut r) as usize;
        contacts.clear();
        if n >= 1 && n != sys::ERR as usize {
            // The touch server says how many records it sent; a reply shorter than its count is a bug there, and the slice panics on it.
            let count = r[0] as usize;
            for k in 0..count {
                let rec = &r[1 + k * 12..1 + (k + 1) * 12];
                let x = (u16::from_le_bytes([rec[4], rec[5]]) as f32 * sx) as i32 - fx as i32;
                let y = (u16::from_le_bytes([rec[6], rec[7]]) as f32 * sy) as i32 - fy as i32;
                // A finger outside the app's frame is not the app's; it is dropped, not pulled to the edge.
                if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                    continue;
                }
                contacts.push(Contact { id: rec[0], phase: rec[1], x: x as u16, y: y as u16 });
            }
        }
        if !contacts.is_empty() {
            touched += contacts.len() as u64;
            shell.on_contacts(&contacts);
        }
        if shell.close {
            let mut m = [0u8; 4];
            sys::call(display, &[23], &mut m);
            return Ok(());
        }
        let t = sys::now();
        if shell.draw() {
            shell.present(surface);
            let mut m = [0u8; 9];
            m[0] = 22;
            m[1..3].copy_from_slice(&(fx as u16).to_le_bytes());
            m[3..5].copy_from_slice(&(fy as u16).to_le_bytes());
            m[5..7].copy_from_slice(&(w as u16).to_le_bytes());
            m[7..9].copy_from_slice(&(h as u16).to_le_bytes());
            let mut rr = [0u8; 4];
            if sys::call(display, &m, &mut rr) == sys::ERR {
                return Err(String::from("the display refused the frame"));
            }
            shown += 1;
            if shown == 1 {
                println!("fluor: the first frame is on the panel");
            }
        }
        sys::sleep_until(t + frame);
    }
}
