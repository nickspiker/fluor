//! Stochastic blink timer for cursor / focus indicators.
//!
//! Schedules the next "tick" at `now + Δ` where `Δ` is a uniformly-random duration in `[0, 300ms]` via xorshift32. Photon's pattern: the visible irregularity reads as "alive" rather than mechanical, and matches the cadence of natural blinking better than a fixed rate.
//!
//! Owned by the consumer (typically alongside a [`Textbox`](crate::widgets::Textbox)). The desktop host calls [`poll`](BlinkTimer::poll) each `about_to_wait` cycle to learn whether the cursor needs to flip + repaint.

use std::time::Instant;

/// Stochastic blink timer. Holds the xorshift32 state + the next firing instant.
pub struct BlinkTimer {
    rng: u32,
    next: Option<Instant>,
}

impl BlinkTimer {
    /// New timer with a fixed-but-arbitrary seed (`0xDEAD_BEEF`). Stopped (no next tick).
    pub fn new() -> Self {
        Self { rng: 0xDEAD_BEEF, next: None }
    }

    /// Start the timer: schedule the first tick at `now + random(0..=300ms)`.
    pub fn start(&mut self, now: Instant) {
        let interval = self.advance();
        self.next = Some(now + interval);
    }

    /// Cancel the timer; [`poll`](Self::poll) will always return `false` until [`start`](Self::start) is called again.
    pub fn stop(&mut self) {
        self.next = None;
    }

    /// True if `now` has reached or passed the scheduled tick. On `true`, immediately schedules the next tick (`now + random(0..=300ms)`).
    pub fn poll(&mut self, now: Instant) -> bool {
        match self.next {
            Some(when) if now >= when => {
                let interval = self.advance();
                self.next = Some(now + interval);
                true
            }
            _ => false,
        }
    }

    /// The currently-scheduled tick instant, or `None` if stopped. Use this to feed `event_loop.set_control_flow(ControlFlow::WaitUntil(...))`.
    pub fn next_tick(&self) -> Option<Instant> {
        self.next
    }

    /// xorshift32 advance, returning the next interval in `[0, 300ms]`.
    fn advance(&mut self) -> std::time::Duration {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        std::time::Duration::from_millis((self.rng % 301) as u64)
    }
}

impl Default for BlinkTimer {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn new_is_stopped() {
        let t = BlinkTimer::new();
        assert!(t.next_tick().is_none());
    }

    #[test]
    fn poll_returns_false_when_stopped() {
        let mut t = BlinkTimer::new();
        assert!(!t.poll(Instant::now()));
    }

    #[test]
    fn start_schedules_within_300ms() {
        let mut t = BlinkTimer::new();
        let now = Instant::now();
        t.start(now);
        let scheduled = t.next_tick().unwrap();
        let delta = scheduled - now;
        assert!(delta <= Duration::from_millis(300));
    }

    #[test]
    fn poll_fires_at_or_past_scheduled() {
        let mut t = BlinkTimer::new();
        let now = Instant::now();
        t.start(now);
        let scheduled = t.next_tick().unwrap();
        // Just before the scheduled instant: no fire.
        assert!(!t.poll(scheduled - Duration::from_millis(1)));
        // At the scheduled instant: fires + reschedules.
        assert!(t.poll(scheduled));
        // After firing, next_tick is updated to a fresh future time.
        let new_scheduled = t.next_tick().unwrap();
        assert!(new_scheduled >= scheduled);
    }

    #[test]
    fn stop_clears_schedule() {
        let mut t = BlinkTimer::new();
        t.start(Instant::now());
        assert!(t.next_tick().is_some());
        t.stop();
        assert!(t.next_tick().is_none());
    }

    #[test]
    fn intervals_are_not_constant() {
        // xorshift32 should produce varied intervals across many advances.
        let mut t = BlinkTimer::new();
        let mut intervals = std::collections::HashSet::new();
        for _ in 0..20 { intervals.insert(t.advance().as_millis()); }
        assert!(intervals.len() > 5, "xorshift32 should produce >5 distinct values in 20 draws");
    }
}
