//! OS-polled input-timing settings. Cross-platform shim around each platform's "what does the user consider a double-click?" setting so widgets don't have to hardcode a guess.
//!
//! The current API exposes [`double_click_interval`]; future additions will cover key-repeat rate, scroll-wheel acceleration, etc. Results are cached per-process — these settings change rarely enough that a once-per-session read is correct, and the cost of re-reading on every press would be significant (XSettings round-trip / `gsettings` subprocess).
//!
//! Fallback ladder per platform: * **Linux X11** — read the XSettings `Net/DoubleClickTime` property from the XSettings selection owner. This is what GTK, Qt5, and most toolkits honor on X11. Returns the value verbatim if present.
//! * **Linux Wayland (or X11 with no XSettings manager)** — shell out to `gsettings get org.gnome.desktop.peripherals.mouse double-click`. Works on GNOME and derivatives; on KDE/sway/etc. without GSettings installed this returns `None` and we fall thru to the default.
//! * **macOS** — TODO: `NSEvent.doubleClickInterval` (seconds, f64). Needs an objc2 dep; not yet wired since fluor's macOS host is wgpu-only and macOS-specific input plumbing is still ahead.
//! * **Windows** — TODO: `GetDoubleClickTime()` from user32. Needs a windows-sys dep; same status as macOS.
//! * **Default** — 400 ms. The middle of the typical OS-default range (250–500 ms) and what GTK ships when no user override is set.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

const DEFAULT_DOUBLE_CLICK_MS: u32 = 400;

/// Host-injected override, milliseconds (0 = unset). Platforms whose timing query lives OUTSIDE fluor push the OS value here — Android's `ViewConfiguration.getDoubleTapTimeout()` comes thru the app's JNI layer at init (fluor has no JNI plumbing of its own). Read before the query ladder, so a set value wins everywhere.
static OVERRIDE_MS: AtomicU32 = AtomicU32::new(0);

/// Inject the platform's double-click/tap interval from outside fluor (e.g. Android `ViewConfiguration.getDoubleTapTimeout()` via the app's JNI init). Call before the first [`double_click_interval`] read; later calls still win because the override is checked on every read.
pub fn set_double_click_interval(ms: u32) {
    OVERRIDE_MS.store(ms, Ordering::Relaxed);
}

/// Maximum time between two presses that still counts as a "double" (or third, etc.) for the same multi-click sequence. Honors the OS / DE setting where available — a host-injected override first (Android), then X11 XSettings / gsettings on Linux, with a 400 ms fallback for unsupported platforms or environments without a configured manager. The query ladder is cached for the life of the process; the override is live.
pub fn double_click_interval() -> Duration {
    let over = OVERRIDE_MS.load(Ordering::Relaxed);
    if over != 0 {
        return Duration::from_millis(over as u64);
    }
    static CACHE: OnceLock<Duration> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let ms = query_double_click_ms().unwrap_or(DEFAULT_DOUBLE_CLICK_MS);
        Duration::from_millis(ms as u64)
    })
}

#[cfg(target_os = "linux")]
fn query_double_click_ms() -> Option<u32> {
    if let Some(ms) = linux::xsettings_double_click_ms() {
        return Some(ms);
    }
    linux::gsettings_double_click_ms()
}

#[cfg(not(target_os = "linux"))]
fn query_double_click_ms() -> Option<u32> {
    // TODO: macOS via objc2 (NSEvent.doubleClickInterval — seconds f64) and Windows via windows-sys (GetDoubleClickTime — milliseconds u32). Neither dep is in fluor yet; until the macOS/Windows hosts grow native input plumbing, fall thru to DEFAULT_DOUBLE_CLICK_MS.
    None
}

#[cfg(target_os = "linux")]
mod linux {
    use std::process::Command;
    use std::sync::OnceLock;
    use x11rb::protocol::xproto::{AtomEnum, ConnectionExt};
    use x11rb::rust_connection::RustConnection;

    /// Reusable X11 connection for XSettings reads. Independent of the one in `app::x11_atomic` — keeping them separate avoids cross-module coupling on a OnceLock that may be initialized in either order. The XSettings read is one round-trip and runs once per process, so the duplicate connection has no practical cost.
    fn conn() -> Option<(&'static RustConnection, usize)> {
        static CONN: OnceLock<Option<(RustConnection, usize)>> = OnceLock::new();
        CONN.get_or_init(|| x11rb::connect(None).ok().map(|(c, s)| (c, s)))
            .as_ref()
            .map(|(c, s)| (c, *s))
    }

    /// Read `Net/DoubleClickTime` from the XSettings property on the XSettings selection owner. Returns `None` if no XSettings manager is running (common on Wayland / minimal X sessions), the property is missing, the setting isn't present in the property, or any XCB call fails. The XSettings protocol format is documented at https://specifications.freedesktop.org/xsettings-spec/xsettings-spec-0.5.html — we parse just enough to find one int-typed setting by name.
    pub fn xsettings_double_click_ms() -> Option<u32> {
        let (conn, screen_num) = conn()?;
        let selection_name = format!("_XSETTINGS_S{}", screen_num);
        let sel_atom = conn
            .intern_atom(false, selection_name.as_bytes())
            .ok()?
            .reply()
            .ok()?
            .atom;
        let owner = conn.get_selection_owner(sel_atom).ok()?.reply().ok()?.owner;
        if owner == 0 {
            return None;
        }
        let prop_atom = conn
            .intern_atom(false, b"_XSETTINGS_SETTINGS")
            .ok()?
            .reply()
            .ok()?
            .atom;
        // Pull the whole property in one call. 8192 longs = 32768 bytes is well past any real XSettings payload (GNOME's is typically <2KB).
        let reply = conn
            .get_property(false, owner, prop_atom, AtomEnum::ANY, 0, 8192)
            .ok()?
            .reply()
            .ok()?;
        parse_xsettings_int(&reply.value, "Net/DoubleClickTime")
    }

    /// Parse XSettings property bytes and return the int value for `name`. Returns `None` on truncation, unknown byte-order, mismatched type, or name-not-found. Truncation is treated as silent failure (don't pretend to honor a setting we couldn't read).
    fn parse_xsettings_int(data: &[u8], name: &str) -> Option<u32> {
        if data.len() < 12 {
            return None;
        }
        let big_endian = match data[0] {
            0 => true,
            1 => false,
            _ => return None,
        };
        let read_u16 = |off: usize| -> Option<u16> {
            let s = data.get(off..off + 2)?;
            Some(if big_endian {
                u16::from_be_bytes([s[0], s[1]])
            } else {
                u16::from_le_bytes([s[0], s[1]])
            })
        };
        let read_u32 = |off: usize| -> Option<u32> {
            let s = data.get(off..off + 4)?;
            Some(if big_endian {
                u32::from_be_bytes([s[0], s[1], s[2], s[3]])
            } else {
                u32::from_le_bytes([s[0], s[1], s[2], s[3]])
            })
        };
        // Header is byte_order(1) + unused(3) + serial(4) = 8 bytes; n_settings(4) at offset 8.
        let n_settings = read_u32(8)?;
        let mut off = 12;
        for _ in 0..n_settings {
            // Per-setting header: type(1) + unused(1) + name_len(2)
            let ty = *data.get(off)?;
            let name_len = read_u16(off + 2)? as usize;
            off += 4;
            let raw_name = data.get(off..off + name_len)?;
            let setting_name = core::str::from_utf8(raw_name).ok()?;
            // Names are padded to a 4-byte boundary.
            let name_pad = (4 - (name_len % 4)) % 4;
            off += name_len + name_pad;
            // last_changed_serial (4)
            off += 4;
            let value: Option<u32> = match ty {
                0 => {
                    // Int: 4 bytes
                    let v = read_u32(off)?;
                    off += 4;
                    Some(v)
                }
                1 => {
                    // String: 4 byte len + bytes + pad
                    let slen = read_u32(off)? as usize;
                    off += 4 + slen;
                    let spad = (4 - (slen % 4)) % 4;
                    off += spad;
                    None
                }
                2 => {
                    // Colour: 4 × u16 = 8 bytes
                    off += 8;
                    None
                }
                _ => return None,
            };
            if setting_name == name {
                return value;
            }
        }
        None
    }

    /// `gsettings get org.gnome.desktop.peripherals.mouse double-click` returns an integer (ms) as plain text. Works on GNOME, Cinnamon, MATE, Pantheon, Budgie. Returns `None` on KDE/sway/etc. without GSettings, or if the binary is missing. Subprocess runs once per process via the OnceLock cache in the caller.
    pub fn gsettings_double_click_ms() -> Option<u32> {
        let out = Command::new("gsettings")
            .args(["get", "org.gnome.desktop.peripherals.mouse", "double-click"])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let s = core::str::from_utf8(&out.stdout).ok()?.trim();
        s.parse::<u32>().ok()
    }
}
