//! Shared pointer-activation arbiter — the press→arm→release/cancel state machine behind fluor's "press arms, release-on-target fires, drag-off cancels" interaction model.
//!
//! Both host backends (the desktop winit loop in [`crate::host::app`] and the Android touch path in [`crate::host::android::shell`]) own one of these and feed it the hit id under the pointer at each down / move / up, keyed off the app's [`crate::host::app::FluorApp::hit_test_map`]. The arbiter is pure logic — no rendering, no platform deps — so the two hosts share one definition of "what counts as a valid activation" instead of each re-deriving it. That is what makes the behaviour identical on every OS.
//!
//! The model, per pointer gesture:
//! - **down** over hit `H` → arm `H` (`held_id()` now reports `H`, so the app paints it in its "held" colour).
//! - **move** → the FIRST time the pointer leaves `H`'s pixels the press is cancelled for good: the held colour clears and it stays cleared. Sliding back onto `H` does NOT re-arm — this is a true drag-off-cancel (you have to lift and press again). A move that never leaves `H` keeps it armed.
//! - **up** → fires `H` iff the press was never dragged off (release-on-target); a release after any drag-off, or on a different element, fires nothing.
//! - **cancel** (Android `ACTION_CANCEL`, or the cursor leaving the window mid-press) → drops the arm, fires nothing.
//!
//! Widgets that instead *use* the drag (a textbox placing + extending a selection, a slider tracking the thumb) opt out via [`crate::host::widget::Click::activate_on_release`] returning `false`; the host engages those on press and never routes them through this arbiter's release gate.

use crate::paint::{HitId, HIT_NONE};

/// Press-hold-release + drag-off-cancel state for a single pointer. See the module docs.
#[derive(Debug, Default, Clone, Copy)]
pub struct PointerArbiter {
    /// The hit id captured at pointer-down, held for the lifetime of the press. `HIT_NONE` when no press is in flight, or once the press has been cancelled by a drag-off.
    pressed_hit: HitId,
}

impl PointerArbiter {
    pub const fn new() -> Self {
        Self { pressed_hit: HIT_NONE }
    }

    /// Pointer-down over `hit` — arm it. `HIT_NONE` (a press on empty space) arms nothing, so the later release is a no-op.
    pub fn on_down(&mut self, hit: HitId) {
        self.pressed_hit = hit;
    }

    /// Pointer moved while the button/finger is down; `hit` is what's under it now. If it has left the armed target, the press is CANCELLED — `pressed_hit` is dropped, so no later slide-back-on can re-arm it and the release fires nothing (true drag-off-cancel). Returns `true` iff this move cancelled an in-flight press (the host should redraw so the held colour clears). A move that stays on target, or one with no press in flight, is a no-op returning `false`.
    pub fn on_move(&mut self, hit: HitId) -> bool {
        if self.pressed_hit == HIT_NONE {
            return false; // no press in flight, or already cancelled
        }
        if hit != self.pressed_hit {
            self.pressed_hit = HIT_NONE; // dragged off → cancel for good
            return true;
        }
        false // still on target
    }

    /// Pointer-up; `hit` is what's under it at release. Returns `Some(id)` — the validated activation — iff the press was never dragged off AND the release is still over that same element; `None` otherwise. Always disarms.
    pub fn on_up(&mut self, hit: HitId) -> Option<HitId> {
        let fire = (self.pressed_hit != HIT_NONE && hit == self.pressed_hit)
            .then_some(self.pressed_hit);
        self.pressed_hit = HIT_NONE;
        fire
    }

    /// Gesture cancelled by the system (Android `ACTION_CANCEL`) or the cursor left the window mid-press. Disarm; fire nothing.
    pub fn on_cancel(&mut self) {
        self.pressed_hit = HIT_NONE;
    }

    /// The hit id the app should currently paint in its "held" colour, or `HIT_NONE` when nothing is armed (idle, or the press was dragged off / cancelled). Fed into [`crate::host::app::Context`]`.pressed_hit` every frame.
    pub fn held_id(&self) -> HitId {
        self.pressed_hit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: HitId = 7;
    const B: HitId = 9;

    #[test]
    fn press_then_release_on_same_target_fires() {
        let mut p = PointerArbiter::new();
        p.on_down(A);
        assert_eq!(p.held_id(), A, "armed target is held");
        assert_eq!(p.on_up(A), Some(A), "release over the armed target fires it");
    }

    #[test]
    fn drag_off_before_release_cancels() {
        let mut p = PointerArbiter::new();
        p.on_down(A);
        assert!(p.on_move(B), "moving off the target changes held state");
        assert_eq!(p.held_id(), HIT_NONE, "dragged off → nothing held");
        assert_eq!(p.on_up(B), None, "release off the armed target fires nothing");
    }

    #[test]
    fn drag_off_then_back_on_stays_cancelled() {
        let mut p = PointerArbiter::new();
        p.on_down(A);
        assert!(p.on_move(B), "off A → cancelled");
        assert!(!p.on_move(A), "back onto A does NOT re-arm (true drag-off-cancel)");
        assert_eq!(p.held_id(), HIT_NONE, "still nothing held after slide-back");
        assert_eq!(p.on_up(A), None, "release on A after a drag-off fires nothing");
    }

    #[test]
    fn release_on_different_target_fires_nothing() {
        let mut p = PointerArbiter::new();
        p.on_down(A);
        // No move event, but the up lands on B (e.g. layout shifted): still a non-match → no fire.
        assert_eq!(p.on_up(B), None);
    }

    #[test]
    fn press_on_empty_space_never_fires() {
        let mut p = PointerArbiter::new();
        p.on_down(HIT_NONE);
        assert_eq!(p.held_id(), HIT_NONE);
        assert_eq!(p.on_up(HIT_NONE), None, "HIT_NONE is never an activation");
    }

    #[test]
    fn cancel_disarms() {
        let mut p = PointerArbiter::new();
        p.on_down(A);
        p.on_cancel();
        assert_eq!(p.held_id(), HIT_NONE);
        assert_eq!(p.on_up(A), None, "a cancelled press fires nothing on the later up");
    }

    #[test]
    fn move_with_no_press_is_inert() {
        let mut p = PointerArbiter::new();
        assert!(!p.on_move(A), "hover with no press in flight changes nothing");
        assert_eq!(p.held_id(), HIT_NONE);
    }

    #[test]
    fn redundant_move_reports_no_change() {
        let mut p = PointerArbiter::new();
        p.on_down(A);
        assert!(!p.on_move(A), "still over A — no state change, no redraw");
        assert!(p.on_move(B), "off A — cancelled, changed");
        assert!(!p.on_move(B), "already cancelled — no further change");
        assert!(!p.on_move(A), "back on A after cancel — still no change (stays dead)");
    }
}
