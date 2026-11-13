//! The ferros host: a `FluorApp` on the phone as a ferros process (ferros STD-TARGET.md). No window system — the display server owns the panel, and an app draws 16-bit pixels into its own full-screen surface and asks the display to show its window (op 22; ferros WINDOWS.md); touches come from the `touch` server's endpoint as contact records. This shell is the Android shell's twin with those two in place of a `NativeWindow` and the JNI touch callbacks: the same press-hold-release arbiter, the same drag-to-scroll, the same `Context`.
//!
//! The process holds three capabilities the caller names by grant slot (`Caps`): the display endpoint, the touch endpoint, its own surface (a full screen of 64-bit pixels; ferros WINDOWS.md). The syscalls are inlined here (ferros_sys is not a published crate): `call`, `dma`, `sleep`, the counter.
//!
//! Front, behind and keys (ferros WINDOWS.md "The helm", owner 2026-11-11: "Home backgrounds the kist so depending on how it was designed, like Photon, it doesn't die but it stops rendering everything and stops playing anything except for notifications", "back should signal escape to the app and go up one page"): a thread of the shell's own holds the display's op 62, answered when the display puts this window behind, brings it to the front or hands it a key. Behind, the frame loop draws nothing and reads no touches: it sleeps until the display or the app's own threads ring (`Bell`), taking in the app's user events and its tick as they come, and the app hears it lost focus (`Event::Focused(false)`, `on_focus_changed(false)`). In front again: focus back, `on_window_shown`, and the next frame drawn whole. Back's Escape arrives as a key press and release, as from a keyboard.
//!
//! The keyboard (ferros WINDOWS.md "The keyboard"): when the app's focused widget wants the soft keyboard (`FluorApp::wants_keyboard`, as on Android) the host asks the display for it (op 68), and lets it go when focus leaves the field. The system keyboard then types into this window thru the display: its text arrives on the same held op 62 as an IME commit (`Event::Ime(Ime::Commit)`), its enter and delete as key presses and releases, in the order they were typed, and the rows it covers as `FluorApp::on_keyboard_inset`, so the app keeps its field above them. The text itself never goes to the console.
//!
//! Pixels: the shell renders at the app's depth (`FluorApp<P>`: 8-bit unless the app names `G2t16`) and [`crate::host::panel`] turns the scratch into the display's working values. At 8 bits: fluor's `0xααRRGGBB` darkness buffer becomes the display's working values — gamma 2 encoded, white = 16384 (Q14), R, G, B in bits 0-15, 16-31, 32-47 of a u64 — through one 256-entry table: fluor's bytes are VSF `g2t` codes (theme.rs, since 2026-10-20: 194 diffuse white, 255 9.7× white), whose light is atanh(c/256)², so the display's gamma 2 (√light) is atanh(c/256), 255 landing at 3.1 of the surface's 4.0. Opacity composites over black. The whole frame is converted and shown each time the app drew; the damage rect narrows that later.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Instant;

use crate::canvas::{Damage, PixelRect};
use crate::coord::Coord;
use crate::event::{ElementState, Event as FEvent, Ime, Key, KeyEvent, ModifiersState as FModifiersState, MouseButton, MouseScrollDelta, NamedKey};
use crate::geom::Viewport;
use crate::host::app::{Context, FluorApp};
use crate::host::wake::{WakeError, WakeSender};
use crate::host::{EventResponse, WindowHandle};
use crate::host::panel::PanelPixel;
use crate::paint::{HitId, HIT_NONE};
use crate::pixel::Argb8;
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

    /// The grants page is arc offset 0: the handle table in its first 512 bytes, then a u32 count at G#200 and the 32-byte tokens after it (ferros SANDBOX.md, "Handles", since 2026-11-03); slot 0 is the console.
    pub fn cap(slot: u64) -> u64 {
        0x204 + 32 * slot
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

/// The grant slots a fluor app on ferros is spawned with: `--use display --use touch --dma surface`, in whatever order the spawner chose.
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

/// The wake sender an app's background threads hold on ferros: events queue here and the host drains them at the top of each frame, waking the app — there is no event loop to post to, the frame loop is the loop. Each send rings the window's bell, so a loop asleep behind wakes to take it in.
pub struct FerrosWakeSender<E> {
    queue: Mutex<Vec<E>>,
    bell: Arc<Bell>,
}

impl<E: Send + 'static> WakeSender<E> for FerrosWakeSender<E> {
    fn send(&self, event: E) -> Result<(), WakeError> {
        self.queue.lock().unwrap().push(event);
        self.bell.ring(|_| {});
        Ok(())
    }
}

/// The ferros display's op 62 and the events in its reply (ferros_sys::window): how many it holds a window, and the kinds.
const EVENTS: usize = 1 << 4;
const EV_KEY: u8 = 3;
/// The display's bake changed (a brightness change): the window is made again with the new one, every pixel.
const EV_BAKE: u8 = 4;
/// Text the system keyboard typed for this window (ferros WINDOWS.md "The keyboard"): UTF-8, its length in the low byte, its bytes in the reply's payloads.
const EV_TEXT: u8 = 6;
/// The keyboard's place: its payload two bytes, little-endian, the panel's rows it covers from the bottom up (0: gone).
const EV_INSET: u8 = 7;
/// Kinds from this one up carry a payload, its length in the low byte; the payloads follow the reply's events, in their order.
const PAYLOAD_KINDS: u8 = 6;
/// The most bytes one payload holds, and so the longest reply: the head, the events, every payload at its longest.
const PIECE: usize = 1 << 6;
const EVENTS_REPLY: usize = 7 + 2 * EVENTS + EVENTS * PIECE;

/// What the keyboard typed, in the order it was typed: a key by its USB HID usage (back's Escape too), or text.
enum Typed {
    Key(u8),
    Text(String),
}

/// The display's bake now (op 65; ferros COMPOSITOR.md "Baked"), for the frames an app makes itself (a GPU kernel's, `FluorApp::own_baked`): the parameters its last frame should be made with, None from a display that does not bake. Set by the frame loop at start and at every `EV_BAKE`.
static BAKE: Mutex<Option<Arc<crate::host::panel::Bake>>> = Mutex::new(None);

/// The display's bake now, as the frame loop last asked it (`BAKE`).
pub fn bake() -> Option<Arc<crate::host::panel::Bake>> {
    BAKE.lock().ok().and_then(|b| b.clone())
}

/// Ask the display for its bake (op 65) and keep it (`BAKE`): None from a display without the op, which answers with a status of its own.
fn ask_bake(display: u64) -> Option<Arc<crate::host::panel::Bake>> {
    let mut r = [0u8; 49];
    let b = (sys::call(display, &[65], &mut r) == 49).then(|| crate::host::panel::Bake::from_reply(&r)).flatten().map(Arc::new);
    if let Ok(mut g) = BAKE.lock() {
        *g = b.clone();
    }
    b
}
/// Keys by their USB HID usages (keyboard page): back's tap is Escape, the system keyboard sends Enter and Backspace.
const KEY_ENTER: u8 = 0x28;
const KEY_ESCAPE: u8 = 0x29;
const KEY_BACKSPACE: u8 = 0x2A;
const KEY_TAB: u8 = 0x2B;
const KEY_DELETE: u8 = 0x4C;
const KEY_RIGHT: u8 = 0x4F;
const KEY_LEFT: u8 = 0x50;
const KEY_DOWN: u8 = 0x51;
const KEY_UP: u8 = 0x52;
/// The most records one reply of the touch server's op 0 carries (ferros payloads/touch, CHANGES); it holds up to 64 waiting.
const TOUCH_BATCH: usize = 1 << 4;

/// The window's bell: what the display last said of this window (behind or in front, the keys it handed over) and whether anything rang since the frame loop last looked, the app's own threads' sends included. The frame loop sleeps on it while behind; nothing polls.
pub struct Bell {
    state: Mutex<BellState>,
    cv: Condvar,
}

#[derive(Default)]
struct BellState {
    behind: bool,
    /// Keys and text the display handed this window (back's Escape, the system keyboard's strokes), oldest first.
    typed: Vec<Typed>,
    /// The keyboard's place, the panel's rows it covers from the bottom, when it changed since the last take (`EV_INSET`).
    inset: Option<u16>,
    /// The display's bake changed since the last take (`EV_BAKE`).
    bake: bool,
    rung: bool,
}

/// What rang since the frame loop last looked (`Bell::take`).
struct Rung {
    behind: bool,
    typed: Vec<Typed>,
    inset: Option<u16>,
    bake: bool,
}

impl Bell {
    fn new() -> Self {
        Self { state: Mutex::new(BellState::default()), cv: Condvar::new() }
    }

    fn ring(&self, f: impl FnOnce(&mut BellState)) {
        let mut s = self.state.lock().unwrap();
        f(&mut s);
        s.rung = true;
        self.cv.notify_all();
    }

    /// Whether the window is behind, the keys and text since the last take, the keyboard's place if it moved since, and whether the display's bake changed since.
    fn take(&self) -> Rung {
        let mut s = self.state.lock().unwrap();
        s.rung = false;
        Rung { behind: s.behind, typed: core::mem::take(&mut s.typed), inset: s.inset.take(), bake: core::mem::take(&mut s.bake) }
    }

    /// Asleep until anything rings since the last take, or until `at` (the app's own deadline, when it has one ahead).
    fn sleep(&self, at: Option<Instant>) {
        let mut s = self.state.lock().unwrap();
        while !s.rung {
            match at {
                Some(t) => {
                    let now = Instant::now();
                    if t <= now {
                        return;
                    }
                    s = self.cv.wait_timeout(s, t - now).unwrap().0;
                }
                None => s = self.cv.wait(s).unwrap(),
            }
        }
    }
}

/// The window's events (ferros display op 62, a held call: answered only when the display puts the window behind, brings it to the front or hands it a key), rung into `bell`. On a thread of its own: the call waits. Ends when the display has no such op (an older display answers at once with one byte), or when two calls in a row fail (a token that is no display's); a display that ended holding the call fails it once, and the next call waits for the display that comes after it.
fn watch(display: u64, base: u64, bell: Arc<Bell>) {
    let mut seen = 0u32;
    let mut failed = false;
    let mut r = [0u8; EVENTS_REPLY];
    loop {
        let mut m = [0u8; 13];
        m[0] = 62;
        m[1..9].copy_from_slice(&base.to_le_bytes());
        m[9..13].copy_from_slice(&seen.to_le_bytes());
        let n = sys::call(display, &m, &mut r);
        if n == sys::ERR {
            if failed {
                return;
            }
            failed = true;
            continue;
        }
        failed = false;
        if n < 7 || r[0] != 0 {
            return;
        }
        seen = u32::from_le_bytes([r[1], r[2], r[3], r[4]]);
        let behind = r[5] & 1 == 1;
        // WHY the bounds: the count byte and the length are the display's (another process's word), and a reply longer than the buffer is cut at it; only whole events that landed in the buffer are read, and of the payloads after them only bytes that landed.
        let n = (n as usize).min(r.len());
        let k = (r[6] as usize).min((n - 7) / 2);
        let mut p = 7 + 2 * k;
        bell.ring(|s| {
            s.behind = behind;
            for i in 0..k {
                let (low, kind) = (r[7 + 2 * i], r[7 + 2 * i + 1]);
                // A payload's bytes, stepped over whatever the kind (one this host does not know is passed over by its length); None for one the reply was cut inside.
                let payload = if kind >= PAYLOAD_KINDS {
                    let from = p;
                    p += low as usize;
                    r.get(from..p.min(n)).filter(|b| b.len() == low as usize)
                } else {
                    None
                };
                match (kind, payload) {
                    (EV_KEY, _) => s.typed.push(Typed::Key(low)),
                    (EV_BAKE, _) => s.bake = true,
                    // Text that is not UTF-8 is no text: dropped, never repaired into something the keyboard did not type.
                    (EV_TEXT, Some(b)) => {
                        if let Ok(t) = core::str::from_utf8(b) {
                            s.typed.push(Typed::Text(String::from(t)));
                        }
                    }
                    (EV_INSET, Some(&[a, b])) => s.inset = Some(u16::from_le_bytes([a, b])),
                    _ => {}
                }
            }
        });
    }
}

/// A contact record from the touch server (12 bytes): id, phase (0 enter, 1 move, 2 leave), kind, pressure, x, y, and the rest unused here.
pub struct Contact {
    pub id: u8,
    pub phase: u8,
    pub x: u16,
    pub y: u16,
}

/// The app and everything the host keeps for it between frames, at the app's pixel depth `P` (8-bit unless the app implements `FluorApp<G2t16>`).
pub struct FerrosShell<A: FluorApp<P>, P: PanelPixel = Argb8> {
    app: A,
    text: TextRenderer,
    window: FerrosWindow,
    viewport: Viewport,
    cursor_x: Coord,
    cursor_y: Coord,
    modifiers: FModifiersState,
    scratch: Vec<P>,
    clip_mask: Vec<u8>,
    pending_damage: Damage,
    /// What the last draw changed in the scratch: the damage it repainted and, on a memmove frame, the rect the memmove moved. The present converts only this.
    changed: PixelRect,
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
    /// The window's bell (`Bell`): the display's events and the app's own sends.
    bell: Arc<Bell>,
    /// Back in front: the next frame is drawn whole, whatever the app says is dirty.
    full_next: bool,
    /// The last draw was the app's own frame (`FluorApp::render_own`), not the scratch.
    own: bool,
}

impl<A: FluorApp<P>, P: PanelPixel> FerrosShell<A, P> {
    pub fn new(mut app: A, width: u32, height: u32, ru: Coord) -> Self {
        let viewport = Viewport::new(width, height).with_ru(ru);
        let bell = Arc::new(Bell::new());
        let wake = Arc::new(FerrosWakeSender { queue: Mutex::new(Vec::new()), bell: bell.clone() });
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
            scratch: alloc::vec![P::EMPTY; px],
            clip_mask: alloc::vec![255u8; px],
            pending_damage: Damage::new(),
            changed: PixelRect::new(0, 0, width as usize, height as usize),
            pointer: crate::host::pointer::PointerArbiter::new(),
            touch_down: false,
            touch_last_x: 0.0,
            touch_last_y: 0.0,
            touch_travel: 0.0,
            finger: None,
            close: false,
            frames: 0,
            wake,
            bell,
            full_next: false,
            own: false,
        };
        shell.with_context(|app, ctx| app.init(ctx));
        shell.with_context(|app, ctx| app.on_resize(width, height, ctx));
        shell
    }

    /// The app, for a caller that reads input the shell does not (an input hook of [`run_app_with`]).
    pub fn app(&mut self) -> &mut A {
        &mut self.app
    }

    /// Draw again at the next pass (what an input hook changed in the app shows).
    pub fn request_redraw(&self) {
        self.window.mark_dirty();
    }

    /// What the background threads sent since the last frame, delivered; a zoom the app asked for applied; the app ticked.
    fn take_events(&mut self) {
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
    }

    /// Behind: the app's own threads' sends and its tick taken in, nothing drawn (a message still arrives and is kept, a notification still raised); what changed is drawn when the window is in front again.
    pub fn events_only(&mut self) {
        self.take_events();
    }

    /// Behind or in front again (the display's op 62). Behind: the finger let go (no touch reaches this window until it is back), and the app told it lost focus (`Focused(false)`, `on_focus_changed(false)`: draw nothing, play nothing but notifications). In front: focus back, `on_window_shown`, and the next frame drawn whole.
    pub fn set_behind(&mut self, behind: bool) {
        if behind && self.finger.take().is_some() {
            self.on_pointer(&[FEvent::CursorLeft]);
        }
        if self.dispatch(&FEvent::Focused(!behind)) == EventResponse::Close {
            self.close = true;
        }
        self.app.on_focus_changed(!behind);
        if !behind {
            self.app.on_window_shown();
            self.full_next = true;
            self.window.mark_dirty();
        }
    }

    /// A key the display handed this window, by its USB HID usage (ferros display ops 61 and 70): Escape (back's tap, "up one page"), the system keyboard's Enter and Backspace, and the other editing keys, each as a press and its release, as from a keyboard. A usage with no key here is passed over.
    pub fn on_key(&mut self, usage: u8) {
        let logical_key = match usage {
            KEY_ESCAPE => Key::Named(NamedKey::Escape),
            KEY_ENTER => Key::Named(NamedKey::Enter),
            KEY_BACKSPACE => Key::Named(NamedKey::Backspace),
            KEY_TAB => Key::Named(NamedKey::Tab),
            KEY_DELETE => Key::Named(NamedKey::Delete),
            KEY_RIGHT => Key::Named(NamedKey::ArrowRight),
            KEY_LEFT => Key::Named(NamedKey::ArrowLeft),
            KEY_DOWN => Key::Named(NamedKey::ArrowDown),
            KEY_UP => Key::Named(NamedKey::ArrowUp),
            _ => return,
        };
        for state in [ElementState::Pressed, ElementState::Released] {
            let ev = FEvent::KeyboardInput { event: KeyEvent { logical_key: logical_key.clone(), state, repeat: false, text: None, physical_key: usage as u16 } };
            if self.dispatch(&ev) == EventResponse::Close {
                self.close = true;
            }
        }
        self.window.mark_dirty();
    }

    /// Text the system keyboard typed for this window (ferros display op 70, `EV_TEXT`): an IME commit, as Android's soft keyboard commits text (`Event::Ime(Ime::Commit)`), to whatever field has focus.
    pub fn on_text(&mut self, text: String) {
        if self.dispatch(&FEvent::Ime(Ime::Commit(text))) == EventResponse::Close {
            self.close = true;
        }
        self.window.mark_dirty();
    }

    /// The keyboard's place (`EV_INSET`): `rows` of the panel from its bottom, as rows of the app's frame from the frame's bottom (the frame sits `fy` rows down a panel `ph` tall), told to the app (`FluorApp::on_keyboard_inset`).
    pub fn on_keyboard_inset(&mut self, rows: u16, fy: usize, ph: usize) {
        let h = self.viewport.height_px as i64;
        // WHY the bounds: `rows` is the display's word (another process's), and the frame may sit anywhere on the panel: the keyboard's top, as a row of the frame, is above the frame's top (the whole frame covered) or below its bottom (none of it).
        let top = (ph as i64 - rows as i64 - fy as i64).clamp(0, h);
        self.app.on_keyboard_inset((h - top) as u32);
        self.window.mark_dirty();
    }

    /// The app's keyboard wish since the last ask (`FluorApp::wants_keyboard`: a text field gained or lost focus), when it has one.
    fn keyboard_wish(&mut self) -> Option<bool> {
        self.app.wants_keyboard()
    }

    /// Tick the app and, when anything is dirty, render into the scratch buffer. True when a frame was drawn.
    pub fn draw(&mut self) -> bool {
        // What the background threads sent since the last frame, delivered first.
        self.take_events();
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
        // The first frame is the whole frame, whatever the app says is dirty: nothing has been shown yet. So is the first back in front: what changed while behind was never drawn.
        let damage_clip = if self.frames == 0 || core::mem::take(&mut self.full_next) { viewport_rect } else { self.app.damage_rect(self.viewport).unwrap_or(viewport_rect) };
        if damage_clip.is_empty() {
            return false;
        }
        self.frames += 1;
        self.changed = damage_clip;
        if damage_clip != viewport_rect {
            if let Some((rect, dy)) = self.app.scroll_hint(self.viewport) {
                crate::paint::scroll_copy_rect(&mut self.scratch, win_w, win_h, rect.x0, rect.y0, rect.x1, rect.y1, dy);
                self.changed = damage_clip.union(rect);
            }
        }
        self.pending_damage.clear();
        // A frame the app made itself (its GPU's layer): the scratch is neither cleared nor drawn, and the present takes the app's frame whole.
        let mut own = false;
        self.with_context_render(damage_clip, |app, _, ctx| own = app.render_own(ctx));
        self.own = own;
        if own {
            self.changed = viewport_rect;
            return true;
        }
        clear_scratch_rect(&mut self.scratch, win_w, damage_clip);
        self.with_context_render(damage_clip, |app, scratch, ctx| app.render(scratch, ctx));
        true
    }

    /// The scratch buffer as the display's pixels, into the surface: a full screen at the panel's width `stride`, the frame at (`fx`, `fy`) on it (ferros WINDOWS.md: every app's surface is the whole screen). With the display's `bake` (op 65) the pixels are baked here ([`PanelPixel::to_baked`]: the display then only dithers them, ferros COMPOSITOR.md "Baked"), else converted as the depth's own format ([`PanelPixel::to_panel`]).
    /// Only what the last draw changed is converted (the surface keeps the rest from the frames before), its rows split across the worker cores in one dispatch.
    pub fn present(&self, surface: &mut [u64], stride: usize, fx: usize, fy: usize, bake: Option<&crate::host::panel::Bake>) {
        use rayon::prelude::*;
        let (w, h) = (self.viewport.width_px as usize, self.viewport.height_px as usize);
        // WHY/PROOF: `changed` is the app's damage and scroll rects, which an app may hand over overhanging its frame (fluor's own primitives cut theirs at the buffer the same way, `Clip::resolve`); the slices below index the scratch and the surface up to its far edges.
        let c = self.changed;
        let (x0, y0, x1, y1) = (c.x0.min(w), c.y0.min(h), c.x1.min(w), c.y1.min(h));
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        // The app's own frame when it made one (`render_own`), else the scratch; nothing when it put its frame in the surface itself.
        let src = match (self.own, self.app.own_frame()) {
            (true, None) => return,
            (true, Some(f)) => f,
            (false, _) => &self.scratch,
        };
        let dst = &mut surface[(fy + y0) * stride..(fy + y1) * stride];
        match bake {
            Some(b) => dst.par_chunks_mut(stride).zip(src[y0 * w..y1 * w].par_chunks(w)).for_each(|(d, s)| P::to_baked(&s[x0..x1], &mut d[fx + x0..fx + x1], b)),
            None => dst.par_chunks_mut(stride).zip(src[y0 * w..y1 * w].par_chunks(w)).for_each(|(d, s)| P::to_panel(&s[x0..x1], &mut d[fx + x0..fx + x1])),
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

    fn with_context_render(&mut self, damage_clip: PixelRect, f: impl FnOnce(&mut A, &mut [P], &mut Context)) {
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
fn clear_scratch_rect<P: PanelPixel>(scratch: &mut [P], width: usize, rect: PixelRect) {
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
            *cell = P::EMPTY;
        }
    }
}

/// Every touch record the touch server holds for this app, read and dropped: until a reply carries less than a full batch (one read takes at most `TOUCH_BATCH` of the up to 64 the server holds, and the rest would then be taken as this app's). A failed call (no touch server) ends it too: its count byte would be the last reply's, and a full one would never end.
fn drain_touches(touch: u64, r: &mut [u8]) {
    loop {
        let n = sys::call(touch, &[0], r);
        if n == 0 || n == sys::ERR || (r[0] as usize) < TOUCH_BATCH {
            break;
        }
    }
}

/// Whether a screen point is this window's (ferros display op 27): the frontmost window covering it, inside the circle an app lives in.
fn mine(display: u64, base: u64, x: u16, y: u16) -> bool {
    let mut m = [0u8; 13];
    m[0] = 27;
    m[1..9].copy_from_slice(&base.to_le_bytes());
    m[9..11].copy_from_slice(&x.to_le_bytes());
    m[11..13].copy_from_slice(&y.to_le_bytes());
    let mut r = [0u8; 4];
    sys::call(display, &m, &mut r) >= 1 && r[0] == 1
}

/// Run `app` on the panel until it asks to close (an `EventResponse::Close`): frames at most 120 a second, touches polled between them while the window is in front; behind (the person went home or flipped to another kist), asleep until the display or the app's own threads ring (`Bell`). `ru` is the resolution unit for the viewport (what the device's density would set on Android; 1.0 is fluor's native scale). `size` is the app's frame, centred on the panel, or the whole panel — a 1344 × 2992 frame is 16 MiB of scratch alone, more than a small arc holds.
/// The depth is the app's: an app implementing `FluorApp` (8-bit) runs on the 8-bit shell as before, one implementing `FluorApp<G2t16>` on the 16-bit one, with the see-through bit wherever it drew nothing.
pub fn run_app<P: PanelPixel, A: FluorApp<P> + 'static>(app: A, caps: Caps, ru: Coord, size: Option<(u32, u32)>) -> Result<(), String> {
    run_app_with(app, caps, ru, size, |shell: &mut FerrosShell<A, P>, contacts: &[Contact]| {
        if !contacts.is_empty() {
            shell.on_contacts(contacts);
        }
    })
}

/// [`run_app`] with the input in the caller's hands: `input` runs once every pass of the loop, before the frame, with the contacts that pass brought which are the app's (inside its frame, not landed on another window; in frame pixels), often none. An app with gestures of its own (two fingers on a map) reads them all, and hands the shell the ones that are its pointer ([`FerrosShell::on_contacts`]); a pass with none is the place to take in what the app's own threads brought (a fix from the GPS).
pub fn run_app_with<P: PanelPixel, A: FluorApp<P> + 'static>(app: A, caps: Caps, ru: Coord, size: Option<(u32, u32)>, mut input: impl FnMut(&mut FerrosShell<A, P>, &[Contact])) -> Result<(), String> {
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
    if base == 0 || base == sys::ERR || pw * ph * 8 > 32 << 20 {
        return Err(String::from("no app surface behind the surface grant"));
    }
    // The whole screen: an app's surface is a full panel, and the window covers the frame's part of it.
    let surface = unsafe { core::slice::from_raw_parts_mut(base as *mut u64, pw * ph) };
    // The touch panel's coordinate range, scaled to the panel's pixels.
    let (sx, sy) = if sys::call(touch, &[2], &mut r) >= 4 {
        let (xm, ym) = (u16::from_le_bytes([r[0], r[1]]) as f32, u16::from_le_bytes([r[2], r[3]]) as f32);
        (if xm > 0.0 { (pw as f32 - 1.0) / xm } else { 1.0 }, if ym > 0.0 { (ph as f32 - 1.0) / ym } else { 1.0 })
    } else {
        (1.0, 1.0)
    };
    println!("fluor: ferros host up, a {w} x {h} frame at ({fx}, {fy}) on a {pw} x {ph} panel");
    let mut shell = FerrosShell::new(app, w as u32, h as u32, ru);
    // The window's events (behind, in front, keys) on a thread of their own, the display's op 62 held there. None spawned: the window never hears them, and runs as before.
    let bell = shell.bell.clone();
    {
        let bell = bell.clone();
        let _ = std::thread::Builder::new().name(String::from("fluor-window")).stack_size(1 << 16).spawn(move || watch(display, base, bell));
    }
    // The touches queued before this window was made were another window's: the tap on the map that launched this app among them, which the app took as its own first touch (ferros, 2026-11-12: Mahere opened from the map with an info card where its circle was tapped).
    drain_touches(touch, &mut r);
    let mut behind = false;
    // The app's keyboard wish not yet sent to the display (`keyboard_wish`, op 68).
    let mut wish: Option<bool> = None;
    // The display's bake (op 65): with it every frame is baked here and shown `BAKED`; a display without it gets the depth's own format, as before.
    let mut bake = ask_bake(display);
    match &bake {
        Some(b) => println!("fluor: the display bakes (generation {}): frames baked here, shown as the panel's codes", b.generation),
        None => println!("fluor: the display does not bake: frames in this depth's own format"),
    }
    let frame = sys::freq() / 120;
    let mut contacts: Vec<Contact> = Vec::with_capacity(16);
    // Contacts that entered on another window or a corner: not this app's until they leave (a bit per contact id).
    let mut foreign = 0u16;
    let (mut shown, mut touched, mut last_line) = (0u64, 0u64, sys::now());
    // Where the frames of the last ten seconds went, in counter ticks: the draw (the app's tick and render), the present (the conversion into the surface), the show (op 22): sums and the longest, and how many frames.
    let (mut t_draw, mut t_present, mut t_show, mut t_max, mut n_frames) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let us = |ticks: u64| ticks * 1_000_000 / sys::freq();
    loop {
        // A line every ten seconds while there is no window system to look at: frames shown, contacts seen, and the frames' time by stage (mean µs, then the longest whole frame).
        if sys::now().wrapping_sub(last_line) > sys::freq() * 10 {
            last_line = sys::now();
            let mean = |t: u64| if n_frames == 0 { 0 } else { us(t) / n_frames };
            println!("fluor: {shown} frames shown, {} drawn, {touched} contacts; the last {n_frames} frames: draw {} us, present {} us, show {} us a frame, longest {} us", shell.frames, mean(t_draw), mean(t_present), mean(t_show), us(t_max));
            (t_draw, t_present, t_show, t_max, n_frames) = (0, 0, 0, 0, 0);
        }
        // What the display said of this window since the last pass (`watch`): behind or in front, the keys and text it handed over, the keyboard's place, and whether its bake changed (a brightness change: the bake asked again, and the next frame drawn and baked whole).
        let Rung { behind: now_behind, typed, inset, bake: rebake } = bell.take();
        // The keyboard's place, behind or not: going behind takes the keyboard away, and the app hears it then.
        if let Some(rows) = inset {
            println!("fluor: the keyboard covers {rows} rows of the panel");
            shell.on_keyboard_inset(rows, fy, ph);
        }
        if rebake {
            bake = ask_bake(display);
            shell.full_next = true;
            shell.window.mark_dirty();
        }
        if now_behind != behind {
            behind = now_behind;
            println!("fluor: {}", if behind { "behind (home, or another kist in front): nothing drawn until it is back" } else { "in front again" });
            shell.set_behind(behind);
            if !behind {
                // The touches queued while it was behind were made on another window (the map, another kist): dropped, never replayed into this one.
                drain_touches(touch, &mut r);
                foreign = 0;
            }
        }
        if behind {
            // Behind: no touch read, no frame drawn or shown; the app's own sends and its tick taken in as they come, then asleep until the display or the app's threads ring, or the app's own deadline ahead (one already past is a frame it wants, which waits until it is in front).
            shell.events_only();
            if shell.close {
                return Ok(());
            }
            bell.sleep(shell.app.wake_at().filter(|at| *at > Instant::now()));
            continue;
        }
        for t in typed {
            match t {
                Typed::Key(k) => {
                    // Said, so back's tap and the keyboard's keys can be followed on the console whatever the app does with them.
                    println!("fluor: key G#{k:02X} from the display");
                    shell.on_key(k);
                }
                // The text itself is not said: it is the person's (a handle, a message), and the console is not theirs.
                Typed::Text(s) => {
                    println!("fluor: {} bytes typed from the display", s.len());
                    shell.on_text(s);
                }
            }
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
                // A finger outside the app's frame is not the app's; it is dropped, not pulled to the edge. Nor is one that landed where another window (or a corner of the circle) is in front: the display says, once per contact as it enters.
                if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                    continue;
                }
                if rec[1] == 0 && !mine(display, base, (x + fx as i32) as u16, (y + fy as i32) as u16) {
                    foreign |= 1 << (rec[0] & 15);
                }
                if foreign >> (rec[0] & 15) & 1 == 1 {
                    if rec[1] == 2 {
                        foreign &= !(1 << (rec[0] & 15));
                    }
                    continue;
                }
                contacts.push(Contact { id: rec[0], phase: rec[1], x: x as u16, y: y as u16 });
            }
        }
        touched += contacts.len() as u64;
        input(&mut shell, &contacts);
        // Closed: back to the app, which ends; nothing is closed on the way out (ferros, owner 2026-11-08: every end is one shot dead), the display drops the window of a process that ended.
        if shell.close {
            return Ok(());
        }
        let t = sys::now();
        if shell.draw() {
            let t_drawn = sys::now();
            shell.present(surface, pw, fx, fy, bake.as_deref());
            let t_presented = sys::now();
            // The window (ferros WINDOWS.md): the frame's part of the screen, this surface, the app level, and what its pixels are (`PanelPixel::FORMAT`: at 16 bits fluor's own, which the display decodes).
            // An opaque app (`FluorApp::opaque`) says so in the 28-byte form, with the frame as what changed: the display then never reads what the window covers (ferros_sys `window::show_full`, mode bit 3).
            let mut m = [0u8; 36];
            m[0] = 22;
            m[1..3].copy_from_slice(&(fx as u16).to_le_bytes());
            m[3..5].copy_from_slice(&(fy as u16).to_le_bytes());
            m[5..7].copy_from_slice(&(w as u16).to_le_bytes());
            m[7..9].copy_from_slice(&(h as u16).to_le_bytes());
            m[9..17].copy_from_slice(&base.to_le_bytes());
            // Baked when this host baked the frame (`present` with the display's bake), or the app made its own frame in the surface baked itself (`FluorApp::own_baked`); else the depth's own format.
            let own_in_surface = shell.own && shell.app.own_frame().is_none();
            m[18] = if (bake.is_some() && !own_in_surface) || (own_in_surface && shell.app.own_baked()) { crate::host::panel::BAKED } else { P::FORMAT };
            let opaque = shell.app.opaque();
            // An app that drew into a buffer of its own (`FluorApp::own_surface`) says which, in the 36-byte form: the display shows that buffer and answers once it has taken it (ferros GPU-KISTS.md 1.4, the mailbox).
            let own = if shell.own { shell.app.own_surface() } else { None };
            if opaque || own.is_some() {
                m[19..21].copy_from_slice(&(fx as u16).to_le_bytes());
                m[21..23].copy_from_slice(&(fy as u16).to_le_bytes());
                m[23..25].copy_from_slice(&(w as u16).to_le_bytes());
                m[25..27].copy_from_slice(&(h as u16).to_le_bytes());
                m[27] = if opaque { 8 } else { 0 };
            }
            if let Some(b) = own {
                m[28..36].copy_from_slice(&b.to_le_bytes());
            }
            let len = if own.is_some() { 36 } else if opaque { 28 } else { 19 };
            let mut rr = [0u8; 4];
            if sys::call(display, &m[..len], &mut rr) == sys::ERR {
                return Err(String::from("the display refused the frame"));
            }
            let t_shown = sys::now();
            t_draw += t_drawn - t;
            t_present += t_presented - t_drawn;
            t_show += t_shown - t_presented;
            t_max = t_max.max(t_shown - t);
            n_frames += 1;
            shown += 1;
            if shown == 1 {
                println!("fluor: the first frame is on the panel");
            }
        }
        // The keyboard (ferros WINDOWS.md "The keyboard"): a text field that gained focus asks the display for it, one that lost it lets it go (op 68). Asked after the frame's show, so a field focused on the first frame asks once the window is the app in front (the display gives the keyboard to that window alone); a wish made before any frame waits for the first.
        if let Some(on) = shell.keyboard_wish() {
            wish = Some(on);
        }
        if let (Some(on), true) = (wish, shown > 0) {
            wish = None;
            let mut m = [0u8; 10];
            m[0] = 68;
            m[1..9].copy_from_slice(&base.to_le_bytes());
            m[9] = on as u8;
            let mut rr = [0u8; 4];
            let n = sys::call(display, &m, &mut rr);
            let took = n != sys::ERR && n >= 1 && rr[0] == 0;
            println!("fluor: the keyboard {} ({})", if on { "asked for" } else { "let go" }, if took { "the display took it" } else { "the display did not: this window is not the app in front, or the display has no keyboard" });
        }
        sys::sleep_until(t + frame);
    }
}
