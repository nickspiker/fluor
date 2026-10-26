//! The desktop clipboard: one `arboard` handle for the life of the process. An X11 or Wayland clipboard is OWNED by the app that set it and served from there until another takes it, so a handle made for each copy and dropped at once loses the text (photon 2026-09-02: forty-eight of fifty copies gone). The handle is made the first time text is set and made again once if a set fails.

use std::sync::Mutex;

static CLIPBOARD: Mutex<Option<arboard::Clipboard>> = Mutex::new(None);

/// Put `text` on the clipboard; false when the platform has none to give.
pub fn set_text(text: &str) -> bool {
    let Ok(mut slot) = CLIPBOARD.lock() else { return false };
    for _ in 0..2 {
        if slot.is_none() {
            *slot = arboard::Clipboard::new().ok();
        }
        match slot.as_mut() {
            Some(c) => {
                if c.set_text(text.to_string()).is_ok() {
                    return true;
                }
                *slot = None;
            }
            None => return false,
        }
    }
    false
}
