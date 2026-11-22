//! The ferros host: a `FluorApp` on the phone as a ferros process (ferros STD-TARGET.md). No window system — the display server owns the panel, and an app draws 16-bit pixels into its own full-screen surface and asks the display to show its window (op 22; ferros WINDOWS.md); touches come from the `touch` server's endpoint as contact records. This shell is the Android shell's twin with those two in place of a `NativeWindow` and the JNI touch callbacks: the same press-hold-release arbiter, the same drag-to-scroll, the same `Context`.
//!
//! The process holds three capabilities the caller names by grant slot (`Caps`): the display endpoint, the touch endpoint, its own surface (a full screen of 64-bit pixels; ferros WINDOWS.md). The syscalls are inlined here (ferros_sys is not a published crate): `call`, `dma`, `sleep`, the counter.
//!
//! Front, behind and keys (ferros WINDOWS.md "The helm", owner 2026-11-11: "Home backgrounds the kist so depending on how it was designed, like Photon, it doesn't die but it stops rendering everything and stops playing anything except for notifications", "back should signal escape to the app and go up one page"): a thread of the shell's own holds the display's op 62, answered when the display puts this window behind, brings it to the front or hands it a key. Behind, the frame loop draws nothing and reads no touches: it sleeps until the display or the app's own threads ring (`Bell`), taking in the app's user events and its tick as they come, and the app hears it lost focus (`Event::Focused(false)`, `on_focus_changed(false)`). In front again: focus back, `on_window_shown`, and the next frame drawn whole. Back's Escape arrives as a key press and release, as from a keyboard.
//!
//! The keyboard (ferros WINDOWS.md "The keyboard"): when the app's focused widget wants the soft keyboard (`FluorApp::wants_keyboard`, as on Android) the host asks the display for it (op 68), and lets it go when focus leaves the field. The system keyboard then types into this window thru the display: its text arrives on the same held op 62 as an IME commit (`Event::Ime(Ime::Commit)`), its enter and delete as key presses and releases, in the order they were typed, and the rows it covers as `FluorApp::on_keyboard_inset`, so the app keeps its field above them. The text itself never goes to the console.
//!
//! The safe area (ferros WINDOWS.md "Drawn under the controls", owner 2026-11-12: "Should we draw under the controls?", then "Agreed. Build it!"): the window fills the glass and the system's controls float in front of its corners, so the display tells the window where they are not (`EV_SAFE`, on the same held op 62): the app's circle and which halves outside it the system holds, at the window's first show and as its full screen turns. The host hands it to the app as `FluorApp::on_safe_area` ([`SafeArea`]), so it keeps its own buttons and text inside.
//!
//! Pixels: the shell renders at the app's depth (`FluorApp<P>`: 8-bit unless the app names `G2t16`) and [`crate::host::panel`] turns the scratch into the display's working values. At 8 bits: fluor's `0xααRRGGBB` darkness buffer becomes the display's working values — gamma 2 encoded, white = 16384 (Q14), R, G, B in bits 0-15, 16-31, 32-47 of a u64 — through one 256-entry table: fluor's bytes are VSF `g2t` codes (theme.rs, since 2026-10-20: 194 diffuse white, 255 9.7× white), whose light is atanh(c/256)², so the display's gamma 2 (√light) is atanh(c/256), 255 landing at 3.1 of the surface's 4.0. Opacity composites over black. The whole frame is converted and shown each time the app drew; the damage rect narrows that later.
//!
//! Layers (ferros docs/GPU-COMPOSE.md 3, 2026-11-22): to a display whose op 5 feature byte says it composes layers (bit 1, [`layered::FEATURE_LAYERS`]), that bakes, from a 16-bit app (its pixels see through), the window is shown as its layers rather than one flattened frame.
//! The app's noise call (`Context::lay_noise`) is taken as parameters and its chrome call (`Context::lay_chrome`) flattens the chrome into a frame of its own ([`crate::host::app::Layers`]), baked into a second surface of the process's (`surface_ask`) where the frame changed; the scratch holds the content alone.
//! Each show is op 22's 50-byte form and its layers after it ([`layered::tail`]): the chrome's surface over its box, the window's own surface (the rows above its ring, the ring, the rows below), the noise last; and each surface's coverage bytes, a byte a 64 × 64 tile past its pixels, are classed again where the present wrote it.
//! To any other display, and on the desktop and Android, the two calls draw into the frame exactly as `noise_under` and `flatten_chrome_into` over each rect did.
//! A ring here is only a pane as wide as the frame (the display reads a ring over a layer's own rows); a narrower pane is moved as without a ring.
//!
//! WHAT THE STRIP NEEDS (GPU-COMPOSE 3.2 and migration step 4's other half, not built): the content rendered ahead of the viewport into a ringed strip taller than the pane, so a scroll inside what is rendered is a new origin and no render.
//! The display first: a pixel layer's ring that is not its own rows (a source y and height of its own past the layer's rect, the strip, read at `(y − y0 + origin) mod H`); today layered.rs refuses any other.
//! Here: a strip surface of its own (the content's surface is the screen's size; a process holds four surfaces at most, its own among them, so the window's, the chrome's and the strip's leave one for a mailbox), the app's pane rendered in strip coordinates (a `FluorApp` hook naming the strip's rows and the rows it owes, so a scroll inside them asks no render), its rows baked thru the strip's ring, and the show naming the strip's origin for the pane's offset.
//! And the frame's other parts (the header, the compose bar) stay a layer of the screen's rows, as now.

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
use crate::host::app::{Context, FluorApp, SafeArea};
use crate::host::wake::{WakeError, WakeSender};
use crate::host::{EventResponse, WindowHandle};
use crate::host::layered;
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

    /// Another surface of the process's own (SYS_DMA x1 = 10 on a surface it holds, ferros_sys `surface_ask`; at most four a process): its token written into `token`, its base past the porch answered, or None.
    pub fn surface_ask(held: u64, token: &mut [u8; 32]) -> Option<u64> {
        let n = svc(DMA, held, 10, off(token.as_mut_ptr()), 0, 0);
        (n != ERR && n != 0).then_some(n)
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
/// The window's safe area (ferros WINDOWS.md "Drawn under the controls"): its payload `SAFE_BYTES`, the app's circle in the window's own pixels (centre x and y as i32, the radius as u32, each in 256ths of a pixel, little-endian), then a byte: bit 0 the glass outside the circle above its centre is the system's, bit 1 below it.
const EV_SAFE: u8 = 8;
const SAFE_BYTES: usize = 13;
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
/// Again (HID G#79): Back's swipe set to go forward, the redo of the last step back (`NamedKey::Forward`).
const KEY_AGAIN: u8 = 0x79;
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
    /// The window's safe area, when the display told it since the last take (`EV_SAFE`).
    safe: Option<SafeArea>,
    /// The display's bake changed since the last take (`EV_BAKE`).
    bake: bool,
    /// A show failed: the display that took it ended (the kernel fails the calls a server that ended had taken).
    /// Set by the frame loop, cleared by the watch's next answer, which comes from the display after it (ferros WINDOWS.md "A display started again"); meanwhile the window is held behind.
    /// Not taken: it stands until a display answers.
    lost: bool,
    /// The watch has ended (`watch`): no display answers this window any more (two failed calls in a row, or a display without op 62), or there never was a watch.
    /// A lost display is then gone for good.
    gone: bool,
    rung: bool,
}

/// What rang since the frame loop last looked (`Bell::take`).
struct Rung {
    behind: bool,
    typed: Vec<Typed>,
    inset: Option<u16>,
    safe: Option<SafeArea>,
    bake: bool,
    lost: bool,
    gone: bool,
}

/// `EV_SAFE`'s payload as a [`SafeArea`] of the window's own pixels (the window is the app's frame).
fn safe_area(p: &[u8; SAFE_BYTES]) -> SafeArea {
    let word = |i: usize| [p[i], p[i + 1], p[i + 2], p[i + 3]];
    let q8 = 1. / (1 << 8) as Coord;
    SafeArea {
        cx: i32::from_le_bytes(word(0)) as Coord * q8,
        cy: i32::from_le_bytes(word(4)) as Coord * q8,
        r: u32::from_le_bytes(word(8)) as Coord * q8,
        top: p[12] & 1 != 0,
        bottom: p[12] & 2 != 0,
    }
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

    /// Whether the window is behind, the keys and text since the last take, the keyboard's place if it moved since, whether the display's bake changed since, and whether a display that took a show ended (`lost`) or none answers any more (`gone`).
    fn take(&self) -> Rung {
        let mut s = self.state.lock().unwrap();
        s.rung = false;
        Rung { behind: s.behind, typed: core::mem::take(&mut s.typed), inset: s.inset.take(), safe: s.safe.take(), bake: core::mem::take(&mut s.bake), lost: s.lost, gone: s.gone }
    }

    /// Whether the window is behind now, as the display last said (or held behind on a lost display), without taking anything: the one gate the frame's draw reads (`FerrosShell::draw`).
    fn behind(&self) -> bool {
        let s = self.state.lock().unwrap();
        s.behind || s.lost
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
/// Every answer clears `lost` (a display answers again); its end rings `gone`, so a frame loop waiting on a lost display ends rather than waits for one that is not coming.
fn watch(display: u64, base: u64, bell: Arc<Bell>) {
    watch_events(display, base, &bell);
    bell.ring(|s| s.gone = true);
}

fn watch_events(display: u64, base: u64, bell: &Bell) {
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
            s.lost = false;
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
                    (EV_SAFE, Some(b)) => {
                        if let Ok(p) = <&[u8; SAFE_BYTES]>::try_from(b) {
                            s.safe = Some(safe_area(p));
                        }
                    }
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

/// A frame's stages summed over the frames since the last console line, in counter ticks (2026-11-17, the owner: "61ms to shift memory and draw 1/16 the screen?"): the scratch moved by the scroll hint, the damage cleared, the app's render, the surface moved, the damage converted (baked) into it, and op 22's call; with how many frames and the pixels they drew and converted.
#[derive(Clone, Copy, Default, Debug)]
pub struct Stages {
    pub frames: u64,
    pub move_scratch: u64,
    pub clear: u64,
    pub render: u64,
    pub move_surface: u64,
    pub convert: u64,
    pub show: u64,
    pub drawn_px: u64,
    pub converted_px: u64,
}

impl Stages {
    /// The ten-second line's words for these frames: each stage's mean µs a frame, and the share of the frame drawn and converted (`px` its pixels).
    fn line(&self, what: &str, px: u64) -> String {
        let n = self.frames;
        let us = |t: u64| t * 1_000_000 / sys::freq() / n;
        alloc::format!(
            "fluor: A#{n} {what}: move the scratch A#{} us, clear A#{} us, render A#{} us, move the surface A#{} us, convert A#{} us, show A#{} us a frame; drawn A#{} %, converted A#{} % of the frame",
            us(self.move_scratch),
            us(self.clear),
            us(self.render),
            us(self.move_surface),
            us(self.convert),
            us(self.show),
            self.drawn_px * 100 / (px * n),
            self.converted_px * 100 / (px * n)
        )
    }
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
    /// What the last draw changed in the scratch: the damage's rects it repainted and, on a memmove frame whose surface is not moved with it, the rect the memmove moved (with the damage as one box). The present converts only these.
    changed: crate::canvas::DamageRects,
    pointer: crate::host::pointer::PointerArbiter,
    touch_down: bool,
    touch_last_x: Coord,
    touch_last_y: Coord,
    touch_travel: Coord,
    /// The contact the pointer follows: the first finger down, until it leaves.
    finger: Option<u8>,
    /// The two fingers of a pinch (id, x, y), the first the pointer's finger; a slot is None once its finger lifted. Set when a second finger lands while the first is down.
    pinch: [Option<(u8, Coord, Coord)>; 2],
    /// The pointer is silent until every finger has lifted: a second finger cancelled it (as Android's host does on a second finger's CANCEL), so the pinch's end never reads as a tap or a drag.
    pinch_muted: bool,
    /// The pinch fingers' distance at the last move, the denominator of the next move's scale factor.
    pinch_d: Coord,
    /// This pinch scaled at least once: its end is told to the app (`FluorApp::on_zoom_end`) when its first finger lifts.
    pinch_scaled: bool,
    close: bool,
    /// Frames drawn, for the first-frame rule and the console line.
    pub frames: u64,
    /// Frames drawn since the last console line that moved the scratch by the app's scroll hint (`scroll_hint`, a memmove and the band drawn), and the pixels every frame since then drew (its damage) and converted at present (`changed`): what a scroll frame costs, said in the ten-second line.
    pub hinted: u64,
    /// Frames since the last console line drawn whole by the host's own choice (the first, back in front, a new bake), and those the app's damage named the whole frame: a full frame refuses the scroll hint (photon render.rs drops it).
    pub forced_whole: u64,
    pub app_whole: u64,
    /// The last draw's scroll (`scroll_hint`'s rect and rows) for the present to move the surface by too, so only the damage is converted (2026-11-17, the Pro: a scroll frame converted the whole moved pane, 15 ms of the bake a frame); None when the frame is converted where it changed.
    shift: Option<(PixelRect, i32)>,
    pub drawn_px: u64,
    pub changed_px: u64,
    /// Where the frames since the last console line went, stage by stage (`Stages`): `[0]` the frames a scroll hint moved, `[1]` every other frame drawn.
    pub stages: [Stages; 2],
    /// Which of `stages` the frame being made goes to.
    bucket: usize,
    /// When the present's move of the surface ended (counter ticks): the convert's start.
    t_moved: u64,
    /// The part of the frame the last present wrote into the surface (frame pixels): the converted rect, and on a scroll frame the moved rect with it. What op 22 names as changed, so the display publishes only that.
    pub shown: PixelRect,
    wake: Arc<FerrosWakeSender<A::UserEvent>>,
    /// The window's bell (`Bell`): the display's events and the app's own sends.
    bell: Arc<Bell>,
    /// Back in front: the next frame is drawn whole, whatever the app says is dirty.
    full_next: bool,
    /// Behind (home, or another kist in front; `set_behind`): no frame is drawn, whatever marks the window dirty, and a scale set meanwhile is only kept (`zoom_owed`).
    behind: bool,
    /// The app scale set while behind (`set_zoom`): the viewport takes each new ru at once, and the app hears the change once, in front again (`set_behind`), with one relayout.
    zoom_owed: crate::host::panel::OwedZoom,
    /// The last draw was the app's own frame (`FluorApp::render_own`), not the scratch.
    own: bool,
    /// The display bakes (op 65 answered): set by the frame loop before each draw, so `draw` knows whether the app may bake its own frame.
    pub display_bakes: bool,
    /// This frame the app bakes its changed rects into the surface itself (`FluorApp::bakes_own`): the scratch was neither cleared nor rendered.
    app_bakes: bool,
    /// The last frame drawn was baked by the app (`app_bakes`): the next one that is not is drawn whole (`panel::whole_frame`).
    baked_last: bool,
    /// The surface's scrolling pane as a ring of rows (`paint::ring`, ferros DISPLAY-SCROLL.md option B), in the surface's (the screen's) coordinates: a scroll frame moves its origin instead of the pane's pixels, and op 22 names it with every show.
    /// None: the surface is linear.
    pub ring: Option<crate::paint::ring::Ring>,
    /// The display reads a ring (op 5's feature bits): without it a scroll frame moves the surface's pane as before.
    pub ring_ok: bool,
    /// The display composes this window's layers (ferros docs/GPU-COMPOSE.md 3.2; op 5's feature bit 1, a display that bakes, a depth whose pixels see through, a surface of its own for the chrome): the app's noise and chrome are taken here ([`Context::lay_noise`], [`Context::lay_chrome`]), not drawn into the scratch, and each show names the layers. None: the frame is flattened as before.
    pub layers: Option<crate::host::app::Layers>,
    /// The two surfaces of a layered window, by their `dma` bases: its own (the content) and the chrome's (0 while `layers` is None).
    surfaces: (u64, u64),
    /// The rects of each surface (content, chrome) the last present wrote, in its own pixels: where their coverage tiles are classed again.
    written: (Vec<PixelRect>, Vec<PixelRect>),
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
            changed: crate::canvas::DamageRects::one(PixelRect::new(0, 0, width as usize, height as usize)),
            pointer: crate::host::pointer::PointerArbiter::new(),
            touch_down: false,
            touch_last_x: 0.0,
            touch_last_y: 0.0,
            touch_travel: 0.0,
            finger: None,
            pinch: [None, None],
            pinch_muted: false,
            pinch_d: 0.0,
            pinch_scaled: false,
            close: false,
            frames: 0,
            hinted: 0,
            forced_whole: 0,
            app_whole: 0,
            shift: None,
            drawn_px: 0,
            changed_px: 0,
            stages: [Stages::default(); 2],
            bucket: 1,
            t_moved: 0,
            shown: PixelRect::new(0, 0, width as usize, height as usize),
            wake,
            bell,
            full_next: false,
            behind: false,
            zoom_owed: crate::host::panel::OwedZoom::default(),
            own: false,
            display_bakes: false,
            app_bakes: false,
            baked_last: false,
            ring: None,
            ring_ok: false,
            layers: None,
            surfaces: (0, 0),
            written: (Vec::new(), Vec::new()),
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

    /// The RU multiplier in effect: what the app started at, times any pinch since.
    pub fn zoom(&self) -> Coord {
        self.viewport.ru
    }

    /// The RU multiplier set from outside the app (an input hook following the person's app scale, ferros preference 14), and the layout run again: the app hears it as a set, not a gesture (`FluorApp::on_zoom_set`, whose default is a pinch step about the centre), then its resize.
    /// Behind, the new ru is only kept: the app is not told, nothing is laid out or drawn, and the change reaches it once when it is in front again (`set_behind`).
    /// A pinch in another kist steps the global scale many times a second, and a window behind that relaid and drew on each step took 90 to 130 ms a step from the one in front (2026-11-21, Photon behind Lumis on the Pro: 11 to 27 frames a second shown).
    pub fn set_zoom(&mut self, ru: Coord) {
        self.follow_bell();
        let ru_before = self.viewport.ru;
        if self.behind {
            self.zoom_owed.hold(ru_before);
            self.viewport.set_zoom(ru);
            return;
        }
        self.viewport.set_zoom(ru);
        self.window.mark_dirty();
        let (w, h) = (self.viewport.width_px, self.viewport.height_px);
        let factor = self.viewport.ru / ru_before;
        self.with_context(|app, ctx| {
            if factor != 1.0 {
                app.on_zoom_set(factor, ctx);
            }
            app.on_resize(w, h, ctx)
        });
    }

    /// Whether a pinch is under way (two fingers down and moving): a hook that writes a pinch's scale somewhere waits for its end.
    pub fn pinching(&self) -> bool {
        self.pinch_scaled
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
        // The app's own zoom is left with it while behind: taken, laid out and drawn at the first frame in front.
        if !self.behind {
            if let Some(ru) = self.app.take_zoom_request() {
                if ru.is_finite() && ru > 0.0 {
                    let (w, h) = (self.viewport.width_px as Coord, self.viewport.height_px as Coord);
                    self.set_zoom_and_relayout(ru, w / 2.0, h / 2.0);
                }
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
    /// A scale set while behind (`set_zoom`) reaches the app here, in front again: one `on_zoom_set` for the whole change and one relayout, before the whole frame.
    /// Behind or in front as the bell says now (`Bell::behind`), the change said to the app once (`set_behind`): read by every path that would lay out or draw, whatever loop calls it.
    fn follow_bell(&mut self) {
        let held = self.bell.behind();
        if held != self.behind {
            self.set_behind(held);
        }
    }

    /// Said once a change: the frame loop and the draw (`draw`, from the bell) may both bring the same change, and the second is nothing.
    pub fn set_behind(&mut self, behind: bool) {
        if behind == self.behind {
            return;
        }
        self.behind = behind;
        if !behind {
            if let Some(factor) = self.zoom_owed.take(self.viewport.ru) {
                let (w, h) = (self.viewport.width_px, self.viewport.height_px);
                self.with_context(|app, ctx| {
                    if factor != 1.0 {
                        app.on_zoom_set(factor, ctx);
                    }
                    app.on_resize(w, h, ctx)
                });
            }
        }
        if behind && self.finger.take().is_some() && !self.pinch_muted {
            self.on_pointer(&[FEvent::CursorLeft]);
        }
        if behind {
            self.pinch = [None, None];
            self.pinch_muted = false;
            self.pinch_scaled = false;
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
    /// Whether the app used the key (its press answered other than `Pass`): an Escape it did not use means it is at its top already, which the display is told (op 76, `escape_unused`), so the deck can take Back's next tap out of the kist when the person chose that (ferros PREF_HELM, owner 2026-11-18).
    pub fn on_key(&mut self, usage: u8) -> bool {
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
            KEY_AGAIN => Key::Named(NamedKey::Forward),
            _ => return true,
        };
        let mut used = true;
        for state in [ElementState::Pressed, ElementState::Released] {
            let ev = FEvent::KeyboardInput { event: KeyEvent { logical_key: logical_key.clone(), state, repeat: false, text: None, physical_key: usage as u16 } };
            let r = self.dispatch(&ev);
            if r == EventResponse::Close {
                self.close = true;
            }
            if state == ElementState::Pressed {
                used = r != EventResponse::Pass;
            }
        }
        self.window.mark_dirty();
        used
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

    /// The window's safe area (`EV_SAFE`), in the window's own pixels, which are the frame's: told to the app (`FluorApp::on_safe_area`), and the frame drawn again.
    pub fn on_safe_area(&mut self, area: SafeArea) {
        self.app.on_safe_area(area);
        self.window.mark_dirty();
    }

    /// The app's keyboard wish since the last ask (`FluorApp::wants_keyboard`: a text field gained or lost focus), when it has one.
    fn keyboard_wish(&mut self) -> Option<bool> {
        self.app.wants_keyboard()
    }

    /// Tick the app and, when anything is dirty, render into the scratch buffer. True when a frame was drawn.
    pub fn draw(&mut self) -> bool {
        // Behind or in front as the display says now, before anything is laid out (`follow_bell`).
        self.follow_bell();
        // What the background threads sent since the last frame, delivered first.
        self.take_events();
        if let Some(at) = self.app.wake_at() {
            if at <= Instant::now() {
                self.window.mark_dirty();
            }
        }
        // Behind, nothing is drawn whatever marked the window dirty (a scale step, a resize, a safe area, a bake, a message): no render work at all, the dirt kept, and the first frame in front drawn whole (`set_behind`).
        // The bell is read at the top as well as by the frame loop, so whatever loop calls the draw, the display's word (behind, the screen off, a lost display) is the one gate.
        if self.behind {
            return false;
        }
        if !self.window.take_dirty() {
            return false;
        }
        let (win_w, win_h) = (self.viewport.width_px as usize, self.viewport.height_px as usize);
        let viewport_rect = PixelRect::new(0, 0, win_w, win_h);
        // The first frame is the whole frame, whatever the app says is dirty: nothing has been shown yet. So is the first back in front: what changed while behind was never drawn. So is the first after the app stopped baking its own frame (`whole_frame`).
        let bakes_now = self.display_bakes && self.app.bakes_own();
        let forced = crate::host::panel::whole_frame(self.frames == 0, core::mem::take(&mut self.full_next), self.baked_last, bakes_now);
        // The app's damage as the rects it named, cut at the frame; their bounding box is the render's clip.
        // None from the app is nothing changed, so nothing is drawn (`panel::frame_rects`): a wake that changed nothing on screen is not a frame.
        let whole = crate::canvas::DamageRects::one(viewport_rect);
        let app_rects = if forced { None } else { crate::host::app::app_damage(&mut self.app, self.viewport) };
        let rects = crate::host::panel::frame_rects(forced, app_rects, win_w, win_h);
        let damage_clip = rects.bbox();
        if forced {
            self.forced_whole += 1;
        } else if damage_clip == viewport_rect {
            self.app_whole += 1;
        }
        if damage_clip.is_empty() {
            return false;
        }
        self.frames += 1;
        self.changed = rects;
        // The surface holds the last frame converted from the scratch unless that frame was the app's own (`render_own`): only then can it be moved as the scratch is.
        let surface_is_scratch = !self.own;
        self.shift = None;
        self.app_bakes = false;
        self.bucket = 1;
        let t0 = sys::now();
        // The app's scroll hint, taken unless the frame is drawn whole as one rect: a hint's frame may name rects whose box is the whole frame (a scrolling pane with fixed bands above and below it drawn again, photon 2026-11-21), and the pane between them is still only moved.
        // Until 2026-11-21 it was taken only when the box was not the whole frame, so such a frame moved nothing while its rects left the pane unconverted (the Pro: no conversation scroll frame took it).
        let mut scroll = None;
        if rects.as_slice() != [viewport_rect] {
            if let Some((rect, dy)) = self.app.scroll_hint(self.viewport) {
                scroll = Some((rect, dy));
                self.bucket = 0;
                // A scroll that moves the whole pane out (nothing of it survives the move) is converted whole, as before; otherwise the surface is moved the same rows and only the damage converted.
                if surface_is_scratch && (dy.unsigned_abs() as usize) < rect.height() {
                    self.shift = Some((rect, dy));
                    // The scratch is moved too unless the app never reads it outside the damage (`FluorApp::scratch_unread_outside_damage`): then only the surface, which the present converts the damage into, needs the moved pane.
                    if !self.app.scratch_unread_outside_damage() {
                        crate::paint::scroll_copy_rect(&mut self.scratch, win_w, win_h, rect.x0, rect.y0, rect.x1, rect.y1, dy);
                    }
                } else {
                    crate::paint::scroll_copy_rect(&mut self.scratch, win_w, win_h, rect.x0, rect.y0, rect.x1, rect.y1, dy);
                    self.changed = crate::canvas::DamageRects::one(damage_clip.union(rect));
                }
                self.hinted += 1;
            }
        }
        let t_moved = sys::now();
        self.pending_damage.clear();
        // A frame the app made itself (its GPU's layer): the scratch is neither cleared nor drawn, and the present takes the app's frame whole.
        let mut own = false;
        self.with_context_render(rects, scroll, |app, _, ctx| own = app.render_own(ctx));
        self.own = own;
        if own {
            self.baked_last = false;
            self.changed = whole;
            self.shift = None;
            return true;
        }
        // The app bakes its own frame into the surface (`FluorApp::bakes_own`): nothing to clear or render here, the present hands it the changed rects. A scroll hint's move of the scratch above was the scratch path's; this path redraws the whole damage.
        self.app_bakes = bakes_now;
        self.baked_last = bakes_now;
        if self.app_bakes {
            self.shift = None;
            return true;
        }
        for &r in rects.as_slice() {
            clear_scratch_rect(&mut self.scratch, win_w, r);
        }
        // A layered window's chrome layer is cleared where the frame draws, as the scratch is, so a frame that lays no chrome leaves none there; outside its box it has always been empty.
        if let Some(l) = self.layers.as_mut() {
            let b = l.chrome_box;
            for &r in rects.as_slice() {
                clear_scratch_rect(l.chrome::<P>(), win_w, meet(r, b));
            }
        }
        let t_cleared = sys::now();
        self.with_context_render(rects, scroll, |app, scratch, ctx| app.render(scratch, ctx));
        let t_rendered = sys::now();
        let drawn = rects.area() as u64;
        let converted = self.changed.cut(win_w, win_h).area() as u64;
        self.drawn_px += drawn;
        self.changed_px += converted;
        let s = &mut self.stages[self.bucket];
        s.frames += 1;
        s.move_scratch += t_moved - t0;
        s.clear += t_cleared - t_moved;
        s.render += t_rendered - t_cleared;
        s.drawn_px += drawn;
        s.converted_px += converted;
        true
    }

    /// The scratch buffer as the display's pixels, into the surface: a full screen at the panel's width `stride`, the frame at (`fx`, `fy`) on it (ferros WINDOWS.md: every app's surface is the whole screen). With the display's `bake` (op 65) the pixels are baked here ([`PanelPixel::to_baked`]: the display then only dithers them, ferros COMPOSITOR.md "Baked"), else converted as the depth's own format ([`PanelPixel::to_panel`]).
    /// Only what the last draw changed is converted (the surface keeps the rest from the frames before), its rows split across the worker cores in one dispatch.
    pub fn present(&mut self, surface: &mut [u64], stride: usize, fx: usize, fy: usize, bake: Option<&crate::host::panel::Bake>) {
        let t0 = sys::now();
        self.present_rows(surface, stride, fx, fy, bake);
        if let (true, false, false, Some(b)) = (self.layers.is_some(), self.own, self.app_bakes, bake) {
            self.present_layers(surface, stride, fx, fy, b);
        }
        // The stages of a scratch frame (an own frame is neither drawn nor counted in them).
        if !self.own {
            let s = &mut self.stages[self.bucket];
            s.move_surface += self.t_moved - t0;
            s.convert += sys::now() - self.t_moved;
        }
    }

    /// A layered window's own part of the present (ferros docs/GPU-COMPOSE.md 3.2): the chrome layer baked into its own surface where the frame changed, inside the box it may hold anything in (outside it that surface is see-through from the start), and each surface's coverage tiles classed again where this present wrote it.
    fn present_layers(&mut self, surface: &mut [u64], stride: usize, fx: usize, fy: usize, bake: &crate::host::panel::Bake) {
        use rayon::prelude::*;
        let (w, h) = (self.viewport.width_px as usize, self.viewport.height_px as usize);
        let rows = surface.len() / stride;
        let (content, chrome_at) = self.surfaces;
        // SAFETY: the chrome's surface is this process's own (`surface_ask`), a full screen at the panel's stride as `surface` is, and no other slice of it lives while this one does.
        let chrome = unsafe { core::slice::from_raw_parts_mut(chrome_at as *mut u64, stride * rows) };
        let changed = self.changed.cut(w, h);
        let Some(l) = self.layers.as_mut() else { return };
        let b = l.chrome_box;
        let src = l.chrome::<P>();
        self.written.1.clear();
        for c in changed.as_slice() {
            let r = meet(*c, b);
            if r.is_empty() {
                continue;
            }
            let dst = &mut chrome[(fy + r.y0) * stride..(fy + r.y1) * stride];
            dst.par_chunks_mut(stride).zip(src[r.y0 * w..r.y1 * w].par_chunks(w)).for_each(|(d, s)| P::to_baked(&s[r.x0..r.x1], &mut d[fx + r.x0..fx + r.x1], bake));
            self.written.1.push(PixelRect::new(fx + r.x0, fy + r.y0, fx + r.x1, fy + r.y1));
        }
        // SAFETY: each surface's tiles lie in its own slot past its pixels (`tiles_of`), apart from both pixel slices.
        crate::host::layered::coverage::classify(unsafe { tiles_of(content, stride, rows) }, surface, stride, &self.written.0);
        crate::host::layered::coverage::classify(unsafe { tiles_of(chrome_at, stride, rows) }, chrome, stride, &self.written.1);
    }

    /// The layers after a show's 50-byte head (ferros docs/GPU-COMPOSE.md 3.1, `layered::tail`): the chrome's surface in front over the box it may hold anything in, the window's own surface (round its ring, `layered::content`), the noise last. The show's length.
    fn layered_tail(&self, m: &mut [u8; crate::host::layered::SHOW_MOST], fx: usize, fy: usize) -> usize {
        use crate::host::layered::{self, coverage, Background, PixLayer, PIX_MOST};
        let (w, h) = (self.viewport.width_px as usize, self.viewport.height_px as usize);
        let Some(l) = self.layers.as_ref() else { return layered::HEAD };
        let (content, chrome) = self.surfaces;
        let tiles = coverage::AT as u32;
        let mut pix = [PixLayer { source: chrome, opaque: false, rect: PixelRect::empty(), ring: None, tiles }; PIX_MOST];
        let mut n = 0;
        let b = l.chrome_box;
        if !b.is_empty() {
            pix[0].rect = PixelRect::new(fx + b.x0, fy + b.y0, fx + b.x1, fy + b.y1);
            n = 1;
        }
        let mut c = [pix[0]; 3];
        let k = layered::content(content, PixelRect::new(fx, fy, fx + w, fy + h), self.ring, tiles, &mut c);
        pix[n..n + k].copy_from_slice(&c[..k]);
        let bg = l.noise.as_ref().and_then(|z| Background::of(z, fx, fy, w, h));
        layered::tail(m, &pix[..n + k], bg.as_ref())
    }

    fn present_rows(&mut self, surface: &mut [u64], stride: usize, fx: usize, fy: usize, bake: Option<&crate::host::panel::Bake>) {
        use rayon::prelude::*;
        let (w, h) = (self.viewport.width_px as usize, self.viewport.height_px as usize);
        self.shown = PixelRect::new(0, 0, 0, 0);
        // WHY/PROOF: `changed` is the app's damage and scroll rects, which an app may hand over overhanging its frame (fluor's own primitives cut theirs at the buffer the same way, `Clip::resolve`); the slices below index the scratch and the surface up to its far edges.
        // A scroll frame: the surface's copy of the pane moved by the same rows as the scratch was (`draw`), so what is converted is the damage alone. The rect cut at the frame first, as below, so the move never reaches another part of the screen.
        // With a display that reads a ring (`ring_ok`), nothing moves: the pane's ring takes the scroll as a new origin (ferros DISPLAY-SCROLL.md option B, 2026-11-17: the move was 2.5 to 2.7 ms of the four A715s a scroll frame on the Pro).
        // A pane other than the ring's: the old ring's rows rotated home once (origin 0), then a new ring over this pane.
        // The surface rects this present writes, for a layered window's coverage tiles (`present_layers`).
        self.written.0.clear();
        if let Some((r, dy)) = self.shift {
            let (sx0, sy0, sx1, sy1) = (r.x0.min(w), r.y0.min(h), r.x1.min(w), r.y1.min(h));
            let pane = PixelRect::new(fx + sx0, fy + sy0, fx + sx1, fy + sy1);
            // A layered window's ring is a layer's own rows (ferros display layered.rs), so only a pane as wide as the frame rings there; a narrower one is moved as without a ring.
            if self.ring_ok && (self.layers.is_none() || crate::host::layered::ring_fits(PixelRect::new(fx, fy, fx + w, fy + h), pane)) {
                let ring = match self.ring.take() {
                    Some(g) if g.rect == pane => g,
                    Some(mut g) => {
                        g.home(surface, stride);
                        self.written.0.push(g.rect);
                        crate::paint::ring::Ring::new(pane)
                    }
                    None => crate::paint::ring::Ring::new(pane),
                };
                self.ring = Some(ring.scrolled(dy));
            } else {
                // A ring over another pane goes home first (none without a display that reads one).
                if let Some(mut g) = self.ring.take() {
                    g.home(surface, stride);
                    self.written.0.push(g.rect);
                }
                crate::paint::scroll_copy_rect(surface, stride, surface.len() / stride, fx + sx0, fy + sy0, fx + sx1, fy + sy1, dy);
                self.written.0.push(pane);
            }
            self.shown = PixelRect::new(sx0, sy0, sx1, sy1);
        }
        self.t_moved = sys::now();
        // Each changed rect cut at the frame (`DamageRects::cut`), converted in a dispatch of its rows to the workers.
        let changed = self.changed.cut(w, h);
        // The app put its frame in the surface itself: every row where a linear reader looks, so no ring.
        if self.own && self.app.own_frame().is_none() {
            self.ring = None;
        }
        if changed.as_slice().is_empty() {
            return;
        }
        self.shown = self.shown.union(changed.bbox());
        // The app bakes its changed rects itself (`FluorApp::present_baked`), straight from its own frame.
        // Its rows are written where they stand, so a ring goes home first and is let go.
        if let (true, Some(b)) = (self.app_bakes, bake) {
            if let Some(mut g) = self.ring.take() {
                g.home(surface, stride);
            }
            for &r in changed.as_slice() {
                self.app.present_baked(r, &mut surface[(fy + r.y0) * stride..(fy + r.y1) * stride], stride, fx, b);
            }
            return;
        }
        // A frame that converts the ring's whole rect writes every one of its rows anyway: written linearly, the ring is let go at no cost (the first frame, back in front, a new bake, the app's own whole frame).
        if let Some(g) = self.ring {
            let whole = changed.as_slice().iter().any(|c| fx + c.x0 <= g.rect.x0 && fy + c.y0 <= g.rect.y0 && fx + c.x1 >= g.rect.x1 && fy + c.y1 >= g.rect.y1);
            if whole {
                self.ring = None;
            }
        }
        // The app's own frame when it made one (`render_own`), else the scratch; nothing when it put its frame in the surface itself.
        let src = match (self.own, self.app.own_frame()) {
            (true, None) => return,
            (true, Some(f)) => f,
            (false, _) => &self.scratch,
        };
        // Each changed rect in surface coordinates, cut where it meets the ring (`Ring::pieces`): each piece's rows run on in the surface from the row it names.
        // Without a ring, the rect whole, where it is.
        for c in changed.as_slice() {
            let s = PixelRect::new(fx + c.x0, fy + c.y0, fx + c.x1, fy + c.y1);
            let (pieces, n) = match self.ring {
                Some(g) => g.pieces(s),
                None => {
                    let mut one = [(PixelRect::empty(), 0usize); crate::paint::ring::RING_PIECES];
                    one[0] = (s, s.y0);
                    (one, 1)
                }
            };
            for &(p, at) in &pieces[..n] {
                // The piece in frame coordinates (the scratch's), and its rows' first row in the surface.
                let (x0, y0, x1, y1) = (p.x0 - fx, p.y0 - fy, p.x1 - fx, p.y1 - fy);
                let dst = &mut surface[at * stride..(at + (y1 - y0)) * stride];
                self.written.0.push(PixelRect::new(p.x0, at, p.x1, at + (y1 - y0)));
                match bake {
                    Some(b) => dst.par_chunks_mut(stride).zip(src[y0 * w..y1 * w].par_chunks(w)).for_each(|(d, s)| P::to_baked(&s[x0..x1], &mut d[fx + x0..fx + x1], b)),
                    None => dst.par_chunks_mut(stride).zip(src[y0 * w..y1 * w].par_chunks(w)).for_each(|(d, s)| P::to_panel(&s[x0..x1], &mut d[fx + x0..fx + x1])),
                }
            }
        }
    }

    /// Contact records in, pointer events out: the first finger down is the pointer.
    /// A second finger while the first is down is a pinch (ferros, 2026-11-16, the owner: pinch did nothing in Photon): the pointer is cancelled (`CursorLeft`, as Android's host gets a CANCEL on a second finger) and stays silent until every finger has lifted, and each move scales by the ratio of the two fingers' distance to its last, anchored at their midpoint (`on_scale`, Android's `on_scale`). A third finger is not followed.
    pub fn on_contacts(&mut self, contacts: &[Contact]) {
        for c in contacts {
            let (x, y) = (c.x as Coord, c.y as Coord);
            match (c.phase, self.finger) {
                (0, None) if !self.pinch_muted => {
                    self.finger = Some(c.id);
                    self.pinch[0] = Some((c.id, x, y));
                    self.on_pointer(&[FEvent::CursorMoved { x, y }, FEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left }]);
                }
                (0, Some(_)) if !self.pinch_muted => {
                    self.pinch[1] = Some((c.id, x, y));
                    self.pinch_muted = true;
                    self.on_pointer(&[FEvent::CursorLeft]);
                    self.pinch_d = self.pinch_distance();
                }
                (1, _) if self.pinch_muted => {
                    for (id, px, py) in self.pinch.iter_mut().flatten() {
                        if *id == c.id {
                            (*px, *py) = (x, y);
                        }
                    }
                    let d = self.pinch_distance();
                    // WHY: the contacts are the touch controller's report, external input: two fingers reported at one point give a distance of 0, and a ratio against it is infinite or NaN. A pinch with a finger lifted gives 0 too (`pinch_distance`), and scales nothing.
                    if d > 0.0 && self.pinch_d > 0.0 {
                        let (ax, ay) = self.pinch_mid();
                        self.on_scale(d / self.pinch_d, ax, ay);
                        self.pinch_scaled = true;
                    }
                    self.pinch_d = d;
                }
                (1, Some(id)) if id == c.id => {
                    self.pinch[0] = Some((c.id, x, y));
                    self.on_pointer(&[FEvent::CursorMoved { x, y }]);
                }
                (2, _) if self.pinch_muted => {
                    for slot in self.pinch.iter_mut() {
                        if slot.is_some_and(|(id, _, _)| id == c.id) {
                            *slot = None;
                        }
                    }
                    if self.finger == Some(c.id) {
                        self.finger = None;
                    }
                    self.pinch_d = 0.0;
                    if self.pinch_scaled {
                        self.pinch_scaled = false;
                        self.with_context(|app, ctx| app.on_zoom_end(ctx));
                    }
                    if self.pinch == [None, None] {
                        self.pinch_muted = false;
                        self.finger = None;
                    }
                }
                (2, Some(id)) if id == c.id => {
                    self.finger = None;
                    self.pinch[0] = None;
                    self.on_pointer(&[FEvent::MouseInput { state: ElementState::Released, button: MouseButton::Left }]);
                }
                _ => {}
            }
        }
    }

    /// The pinch fingers' distance in frame pixels; 0 while either has lifted.
    fn pinch_distance(&self) -> Coord {
        match self.pinch {
            [Some((_, ax, ay)), Some((_, bx, by))] => ((bx - ax) * (bx - ax) + (by - ay) * (by - ay)).sqrt(),
            _ => 0.0,
        }
    }

    /// The pinch fingers' midpoint, the zoom's anchor.
    fn pinch_mid(&self) -> (Coord, Coord) {
        match self.pinch {
            [Some((_, ax, ay)), Some((_, bx, by))] => ((ax + bx) / 2.0, (ay + by) / 2.0),
            _ => (self.viewport.width_px as Coord / 2.0, self.viewport.height_px as Coord / 2.0),
        }
    }

    /// One step of a pinch, as the Android host's `on_scale`: an app that owns the gesture (a picture viewer) gets the factor and the anchor, the viewport untouched; otherwise the viewport's `ru` is scaled and the layout run again.
    fn on_scale(&mut self, factor: f32, ax: Coord, ay: Coord) {
        if self.app.owns_zoom_gesture() {
            self.window.mark_dirty();
            self.with_context(|app, ctx| app.on_zoom(factor, ax, ay, ctx));
            return;
        }
        self.set_zoom_and_relayout(self.viewport.ru * factor, ax, ay);
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

    /// The viewport's `ru` set (thru `set_zoom`, the production clamp) and the layout run again; `on_zoom` told the applied ratio at the anchor (a pinch's midpoint, else the frame's centre).
    fn set_zoom_and_relayout(&mut self, new_ru: f32, ax: Coord, ay: Coord) {
        let ru_before = self.viewport.ru;
        self.viewport.set_zoom(new_ru);
        self.window.mark_dirty();
        let (w, h) = (self.viewport.width_px, self.viewport.height_px);
        let factor = self.viewport.ru / ru_before;
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
            layers: None,
            damage_rects: crate::canvas::DamageRects::EMPTY,
            scroll: None,
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

    fn with_context_render(&mut self, damage_rects: crate::canvas::DamageRects, scroll: Option<(PixelRect, i32)>, f: impl FnOnce(&mut A, &mut [P], &mut Context)) {
        let damage_clip = damage_rects.bbox();
        let mut ctx = Context {
            pressed_hit: self.pointer.held_id(),
            layers: self.layers.as_mut(),
            damage_rects,
            scroll,
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
    // The rows the scratch holds of the rect (a rect past its bottom is cut there, as before), split across the workers: a whole frame's clear is 32 MB at 16 bits.
    let (x0, y1) = (rect.x0, rect.y1.min(scratch.len() / width));
    crate::par::par_rows(scratch, width, rect.y0.min(y1), y1, |_, row| row[x0..x1].fill(P::EMPTY));
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
    // Op 5's byte 14, the display's features (absent from a display before them; byte 13 is the Escapes count, forge 2026-11-20): bit 0, it reads a ring named in op 22's 50-byte form; bit 1, the layers after it (`layered`).
    let features = if sys::call(display, &[5], &mut r) > layered::FEATURES_AT as u64 { r[layered::FEATURES_AT] } else { 0 };
    let ring_ok = features & layered::FEATURE_RING != 0;
    println!("fluor: ferros host up, a {w} x {h} frame at ({fx}, {fy}) on a {pw} x {ph} panel; {}", if ring_ok { "a scroll moves the pane's ring, not its pixels" } else { "the display reads no ring: a scroll moves the pane" });
    let mut shell = FerrosShell::new(app, w as u32, h as u32, ru);
    shell.ring_ok = ring_ok;
    // The window's events (behind, in front, keys) on a thread of their own, the display's op 62 held there. None spawned: the window never hears them, and runs as before.
    let bell = shell.bell.clone();
    {
        let bell = bell.clone();
        let watching = bell.clone();
        if std::thread::Builder::new().name(String::from("fluor-window")).stack_size(1 << 16).spawn(move || watch(display, base, watching)).is_err() {
            // No watch: nothing would ever say a display answers again, so a lost display is gone (a failed show ends the app, as before).
            bell.ring(|s| s.gone = true);
        }
    }
    let mut behind = false;
    // The app's keyboard wish not yet sent to the display (`keyboard_wish`, op 68).
    let mut wish: Option<bool> = None;
    // The display's bake (op 65): with it every frame is baked here and shown `BAKED`; a display without it gets the depth's own format, as before.
    let mut bake = ask_bake(display);
    match &bake {
        Some(b) => println!("fluor: the display bakes (generation {}): frames baked here, shown as the panel's codes", b.generation),
        None => println!("fluor: the display does not bake: frames in this depth's own format"),
    }
    // A display that composes layers (ferros docs/GPU-COMPOSE.md 3.2), baked ones only, from a depth whose pixels see through (at 8 bits a layer would hide the ones behind it): the chrome gets a surface of its own and the noise is handed over as parameters, so neither is drawn into the frame.
    if features & layered::FEATURE_LAYERS != 0 && bake.is_some() && P::SEES_THROUGH {
        let mut token = [0u8; 32];
        match sys::surface_ask(sys::cap(caps.surface), &mut token) {
            Some(chrome) => {
                // The tiles lie past the pixels: a panel whose pixels reach them is a ferros the coverage was not laid out for.
                assert!(pw * ph * 8 <= layered::coverage::AT - layered::coverage::PORCH, "the panel's pixels reach the coverage tiles");
                // SAFETY: the surface the kernel just gave this process, a full screen of the panel's pixels, read by nothing here yet.
                let cs = unsafe { core::slice::from_raw_parts_mut(chrome as *mut u64, pw * ph) };
                // See-through over the window: a surface comes from the pool wiped to 0, which is a baked opaque black.
                for y in fy..fy + h {
                    cs[y * pw + fx..y * pw + fx + w].fill(crate::host::panel::SEE_THROUGH);
                }
                // SAFETY: the chrome surface's own tiles, past its pixels (checked above).
                layered::coverage::classify(unsafe { tiles_of(chrome, pw, ph) }, cs, pw, &[PixelRect::new(fx, fy, fx + w, fy + h)]);
                shell.layers = Some(crate::host::app::Layers::new::<P>(w, h));
                shell.surfaces = (base, chrome);
                println!("fluor: the display composes layers: the noise handed over as parameters, the chrome on a surface of its own");
            }
            None => println!("fluor: the display composes layers, but no surface was given for the chrome: the frame is flattened as before"),
        }
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
            // WHY: a ten-second window with no frame drawn would divide by zero; it has nothing to say.
            if n_frames != 0 {
                let px = (w as u64) * (h as u64);
                println!("fluor: of those frames A#{} moved by a scroll hint, A#{} drawn whole by the host and A#{} by the app's damage; drawn A#{} % of the frame a frame, converted at present A#{} %", shell.hinted, shell.forced_whole, shell.app_whole, shell.drawn_px * 100 / (px * n_frames), shell.changed_px * 100 / (px * n_frames));
                // Each kind's stages, for the kinds that had frames (a kind with none has no mean).
                for (s, what) in shell.stages.iter().zip(["scroll frames", "other frames"]) {
                    if s.frames != 0 {
                        println!("{}", s.line(what, px));
                    }
                }
            }
            (shell.hinted, shell.forced_whole, shell.app_whole, shell.drawn_px, shell.changed_px) = (0, 0, 0, 0, 0);
            shell.stages = [Stages::default(); 2];
            (t_draw, t_present, t_show, t_max, n_frames) = (0, 0, 0, 0, 0);
        }
        // What the display said of this window since the last pass (`watch`): behind or in front, the keys and text it handed over, the keyboard's place, and whether its bake changed (a brightness change: the bake asked again, and the next frame drawn and baked whole).
        let Rung { behind: now_behind, typed, inset, safe, bake: rebake, lost, gone } = bell.take();
        // A display that took a show ended, and none answers this window any more (the watch has ended): nothing will ever show it again.
        if lost && gone {
            return Err(String::from("the display ended, and none answers this window"));
        }
        // A display that took a show ended: the window is held behind (no touch read, nothing drawn or shown) until the watch hears the display after it, which says where the window stands there (behind, until the deck brings it to the front).
        let now_behind = now_behind || lost;
        // The safe area (the display's word at the window's first show and as its full screen turns): where the app keeps its own controls, the system's in front of the corners.
        if let Some(a) = safe {
            println!("fluor: the safe area is the circle at ({}, {}) radius {}, the system holding {}", a.cx, a.cy, a.r, match (a.top, a.bottom) { (true, true) => "both halves outside it", (false, true) => "the bottom half outside it", (true, false) => "the top half outside it", (false, false) => "nothing" });
            shell.on_safe_area(a);
        }
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
                // The touches queued while it was behind were made on another window (the map, another kist): dropped, never replayed into this one. Read until a reply carries less than a full batch: one read took at most `TOUCH_BATCH` of the up to 64 the server holds, and the rest were then taken as this app's. A failed call (no touch server) ends it too: its count byte would be the last reply's, and a full one would never end.
                loop {
                    let n = sys::call(touch, &[0], &mut r);
                    if n == 0 || n == sys::ERR || (r[0] as usize) < TOUCH_BATCH {
                        break;
                    }
                }
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
                    // Back's Escape unused: the app is at its top (`on_key`); the display counts it for the deck (op 76 `[76][surface base u64]`, its answer not waited on beyond the call).
                    if !shell.on_key(k) && k == KEY_ESCAPE {
                        let mut m = [0u8; 9];
                        m[0] = 76;
                        m[1..9].copy_from_slice(&base.to_le_bytes());
                        let mut r = [0u8; 1];
                        sys::call(display, &m, &mut r);
                    }
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
        shell.display_bakes = bake.is_some();
        if shell.draw() {
            let t_drawn = sys::now();
            shell.present(surface, pw, fx, fy, bake.as_deref());
            let t_presented = sys::now();
            // The window (ferros WINDOWS.md): the frame's part of the screen, this surface, the app level, and what its pixels are (`PanelPixel::FORMAT`: at 16 bits fluor's own, which the display decodes).
            // An opaque app (`FluorApp::opaque`) says so in the mode byte: the display then never reads what the window covers (ferros_sys `window::show_full`, mode bit 3).
            let mut m = [0u8; layered::SHOW_MOST];
            m[0] = layered::OP_SHOW;
            m[1..3].copy_from_slice(&(fx as u16).to_le_bytes());
            m[3..5].copy_from_slice(&(fy as u16).to_le_bytes());
            m[5..7].copy_from_slice(&(w as u16).to_le_bytes());
            m[7..9].copy_from_slice(&(h as u16).to_le_bytes());
            m[9..17].copy_from_slice(&base.to_le_bytes());
            // Baked when this host baked the frame (`present` with the display's bake), or the app made its own frame in the surface baked itself (`FluorApp::own_baked`); else the depth's own format.
            let own_in_surface = shell.own && shell.app.own_frame().is_none();
            m[18] = if (bake.is_some() && !own_in_surface) || (own_in_surface && shell.app.own_baked()) { crate::host::panel::BAKED } else { P::FORMAT };
            // Every show names what changed (ferros_sys `show_changed`): the part of the surface this present wrote (`FerrosShell::shown`: the converted damage, and on a scroll frame the moved pane or its ring), or the whole frame for the app's own frame.
            // The display publishes only that and owes it to its other phases (2026-11-17: until then a caret's blink or a hover was a whole screen's publish).
            let opaque = shell.app.opaque();
            // An app that drew into a buffer of its own (`FluorApp::own_surface`) says which, in the 36-byte form: the display shows that buffer and answers once it has taken it (ferros GPU-KISTS.md 1.4, the mailbox).
            let own = if shell.own { shell.app.own_surface() } else { None };
            let c = if shell.own { PixelRect::new(0, 0, w, h) } else { shell.shown };
            m[19..21].copy_from_slice(&((fx + c.x0) as u16).to_le_bytes());
            m[21..23].copy_from_slice(&((fy + c.y0) as u16).to_le_bytes());
            m[23..25].copy_from_slice(&(c.width() as u16).to_le_bytes());
            m[25..27].copy_from_slice(&(c.height() as u16).to_le_bytes());
            m[27] = if opaque { 8 } else { 0 };
            if let Some(b) = own {
                m[28..36].copy_from_slice(&b.to_le_bytes());
            }
            // A layered window's frame drawn from the scratch and baked here names its layers after the 50-byte form (`layered_tail`); an app's own frame is the plain show it always was.
            let layers_now = shell.layers.is_some() && bake.is_some() && !shell.own && !shell.app_bakes;
            // To a display that reads a ring, every show is the 50-byte form: the 40-byte one (the buffer, 0 for the surface itself; the frame's own time, its draw and present) and the ring, [x u16][y u16][w u16][h u16][origin u16], a height of 0 for none (a layered show's ring is its layer's).
            // Stated with every show, so the display holds nothing of it the host did not just say.
            let len = if ring_ok || layers_now {
                m[36..40].copy_from_slice(&(us(t_presented - t) as u32).to_le_bytes());
                if let (Some(g), false) = (shell.ring, layers_now) {
                    let g16 = [g.rect.x0, g.rect.y0, g.rect.width(), g.rect.height(), g.origin];
                    for (i, v) in g16.iter().enumerate() {
                        m[40 + 2 * i..42 + 2 * i].copy_from_slice(&(*v as u16).to_le_bytes());
                    }
                }
                if layers_now { shell.layered_tail(&mut m, fx, fy) } else { layered::HEAD }
            } else if own.is_some() {
                36
            } else {
                28
            };
            let mut rr = [0u8; 4];
            // A show answered ERR is a display that ended holding it (the kernel fails the calls a server that ended had taken): the frame is lost and the app runs on, its window held behind (`lost`) until the watch hears the display after it, which says where the window stands there (ferros WINDOWS.md "A display started again": behind, until the deck brings it to the front).
            // The app ends only when the watch has ended too (`gone`: no display answers any more), never by counting failures (2026-11-20: a display restarted under an app ended it).
            // The display after it starts in tens of milliseconds at least, so this is set long before its first answer, which clears it.
            if sys::call(display, &m[..len], &mut rr) == sys::ERR {
                println!("fluor: the display ended holding a frame; held behind until a display answers again");
                bell.ring(|s| s.lost = true);
            } else {
                let t_shown = sys::now();
                if !shell.own {
                    shell.stages[shell.bucket].show += t_shown - t_presented;
                }
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

/// Where two rects overlap (empty where they do not): the two rects' intersection, not a clamp.
fn meet(a: PixelRect, b: PixelRect) -> PixelRect {
    PixelRect::new(a.x0.max(b.x0), a.y0.max(b.y0), a.x1.min(b.x1), a.y1.min(b.y1))
}

/// A surface's coverage tiles (`layered::coverage`): a byte a tile, in its slot past its pixels, `coverage::AT` bytes from the slot's start, which is `PORCH` bytes before the surface's `dma` base.
/// SAFETY: `base` is a surface of this process's (a full screen of `stride × rows` pixels, which end before the tiles: checked when the window turns layered), and no other slice of its tiles lives while this one does.
unsafe fn tiles_of<'a>(base: u64, stride: usize, rows: usize) -> &'a mut [u8] {
    use crate::host::layered::coverage;
    let n = coverage::tiles_across(stride) * coverage::tiles_across(rows);
    unsafe { core::slice::from_raw_parts_mut((base as usize + coverage::AT - coverage::PORCH) as *mut u8, n) }
}
