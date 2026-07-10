//! Widget abstraction — the type-system contract every interactive thing in a fluor app conforms to. First-principles split into capability traits so the type system enforces "you can't deliver a key to something that doesn't impl [`Key`]" rather than discovering the no-op at runtime. See the design plan at `~/.claude/plans/buzzing-puzzling-yao.md` for the full Sweeney critique that drove this shape.
//!
//! **Layering.** Capability traits speak [`crate::event::KeyEvent`] / [`crate::event::ModifiersState`] — fluor-native input types. Hosts translate platform input (winit on desktop, JNI on Android) into those at the boundary, so widgets compile and run on every supported target with the same code.
//!
//! **Dense IDs.** [`HitId`] (re-exported from [`crate::paint`]) is allocated by threading a single `&mut HitId` counter thru widget constructors at startup. `0` stays reserved as [`HIT_NONE`]; registrations start at `1` and increment sequentially. The denseness is an invariant of the allocation pattern — dispatch can index directly by `id - 1` if it wants (the per-frame walk in the v0 demo is O(N), but an optimised path is one slice-build away once profiling justifies it).
//!
//! **Dispatch.** An app implements [`Container`] (the recursive `visit` walk) so click / key / hover / focus events can be routed by walking the tree once and asking each widget for the matching capability. No match arms, no back-references, no lifetimes outlasting the frame. See [`linear_tab_next`] for the canonical "registration-order tab cycle" helper apps that want spatial or modal-stack-aware navigation write their own helper against the same `Container` shape.

use crate::canvas::{Canvas, PixelRect};
use crate::coord::Coord;
use crate::event::{KeyEvent, ModifiersState};
use crate::paint::{Clip, HitId};
use crate::text::TextRenderer;

pub use crate::paint::HIT_NONE;

/// Allocate the next dense hit ID. Threaded thru widget constructors at app startup — `let mut counter: HitId = HIT_NONE; let id = next_id(&mut counter);`. Increments first, returns the post-increment value, so the first call yields `1` (never `HIT_NONE`). 65 535-call ceiling at the [`HitId`] type's `u16` width; panics on overflow because exceeding 65 535 interactive zones in one app is a design error, not something to silently wrap and corrupt dispatch.
pub fn next_id(counter: &mut HitId) -> HitId {
    *counter = counter
        .checked_add(1)
        .expect("HitId allocator overflowed u16 — more than 65 535 widgets registered, which is almost certainly a leak in the allocation pattern, not a real need");
    *counter
}

/// Per-paint scratchpad bundling everything a widget's [`Widget::paint`] method needs: a mutable canvas for drawing, the shared font + glyph cache for text widgets, the shared per-pixel hit map for stamping the widget's silhouette, and an optional clip rect for damage-narrowed paints. Built fresh by the host (or app) once per render call; widgets borrow it immutably-by-shape for the duration of a single `paint`.
///
/// Lifetimes: the inner borrows all share `'ctx` (the duration of the per-frame paint call). Canvas's own buffer borrow `'buf` is separate so a long-lived canvas can be reused across paints without re-binding the lifetime.
pub struct PaintCtx<'ctx, 'buf> {
    pub canvas: &'ctx mut Canvas<'buf>,
    pub text: &'ctx mut TextRenderer,
    pub hit_map: &'ctx mut [HitId],
    pub clip: Option<Clip>,
}

/// Universal widget contract. Every interactive thing has a stable [`HitId`] and a `paint` method that stamps both pixels and (via `ctx.hit_map`) its hit-test silhouette. Capability accessors below default to `None` — implementers opt into [`Click`] / [`Key`] / [`Focus`] / [`Hover`] only for the behaviours they want. The Option-returning shape is what makes the type system load-bearing: a widget without a [`Key`] impl literally cannot receive keyboard events because there's no `&mut dyn Key` to deliver them to.
///
/// Object-safe by construction (all methods take `&self` or `&mut self`, no generic methods, no associated types). `&mut dyn Widget` is the common currency of dispatch.
pub trait Widget {
    /// The widget's hit ID. Allocated once at construction via [`next_id`]; stored on the widget; returned verbatim here every paint and every dispatch.
    fn id(&self) -> HitId;
    /// Rasterize self into `ctx.canvas` and stamp `id()` into `ctx.hit_map` at every opaque pixel the widget owns. Stamping is what makes `ctx.hit_map[y * w + x]` the canonical source of truth for "what's under the cursor."
    fn paint(&mut self, ctx: &mut PaintCtx<'_, '_>);
    /// Click capability. Default `None` ⇒ this widget ignores clicks (decoration-only). Implementers return `Some(self)`.
    fn click(&mut self) -> Option<&mut dyn Click> {
        None
    }
    /// Keyboard capability. Default `None`. Only delivered to the currently-focused widget; arbiter is the app, not the widget.
    fn key(&mut self) -> Option<&mut dyn Key> {
        None
    }
    /// Focus capability. Default `None` ⇒ widget is not in the tab cycle and can't receive keyboard events. Returning `Some` opts the widget into both the cycle and the keyboard-delivery target set.
    fn focus(&mut self) -> Option<&mut dyn Focus> {
        None
    }
    /// Hover capability. Default `None` ⇒ widget has no hover visual. Implementers wire `set_hovered` to mark their cache dirty so the next paint reflects the cursor entering / leaving them.
    fn hover(&mut self) -> Option<&mut dyn Hover> {
        None
    }
}

/// Click handler. Coordinates are in viewport-local pixels (top-left origin). `mods` is the live modifier state at press time; widgets that want shift-click / ctrl-click semantics read it here.
pub trait Click {
    fn on_click(
        &mut self,
        x: Coord,
        y: Coord,
        mods: ModifiersState,
    ) -> crate::host::EventResponse;
}

/// Keyboard handler. Receives the fluor-native [`KeyEvent`] (with both the logical key and any text-mode text payload), the live modifier state, and a mutable [`TextRenderer`] for widgets that need to recompute glyph widths after an edit (textbox inserts a character → widths must be re-measured before the next paint can position the cursor). Widgets that don't care about text (chrome buttons) ignore the `text` parameter; widgets that don't care about a key return [`crate::host::EventResponse::Pass`] so the host knows the event went unconsumed.
pub trait Key {
    fn on_key(
        &mut self,
        kev: &KeyEvent,
        mods: ModifiersState,
        text: &mut TextRenderer,
    ) -> crate::host::EventResponse;
}

/// Focus state delivery + spatial geometry. `set_focused(true)` is called once when this widget becomes the focused target; `set_focused(false)` once when it loses focus. Widgets typically use this to start / stop a blinkey, toggle a focus ring, or mark their cache layer dirty for a re-paint.
///
/// `focus_bbox` lets future spatial-tab-order helpers compute next-by-position rather than next-by-registration; returning `None` opts out (the linear helper ignores it).
pub trait Focus {
    fn set_focused(&mut self, focused: bool);
    fn focus_bbox(&self) -> Option<PixelRect> {
        None
    }
}

/// Hover state delivery. Mirrors [`Focus`] in shape but without the bbox / tab-order side; hover is purely "cursor entered / left me."
///
/// `tint_delta` returns the wrap-add visible-RGB delta that the host's overlay pass should apply to every pixel marked with this widget's [`HitId`] in the hit-test map. `0` = no tint (default). Implementers fold their own focused / hovered state into the answer — focus and hover historically used distinct visuals, but the v0 convention is "the focused widget renders the hover tint" so keyboard-only users see which widget owns input.
pub trait Hover {
    fn set_hovered(&mut self, hovered: bool);
    fn tint_delta(&self) -> u32 {
        0
    }
    /// Pixel bbox of this widget's hit footprint, for BOUNDING the host's overlay tint scan to just this widget instead of the whole window.
    /// `None` (default) → the host falls back to a full-window scan for this id (correct, just slower).
    /// Return the widget's bbox so hover tinting is O(widget), not O(screen).
    /// See [`build_overlay_bboxes`] + [`crate::paint::apply_overlay`].
    fn hover_bbox(&self, _viewport_w: usize, _viewport_h: usize) -> Option<PixelRect> {
        None
    }
}

/// Tree node. The app root is a [`Container`], chrome is a [`Container`], future panes / dialogs are [`Container`]s. The single `visit` method does depth-first traversal handing each leaf widget to the callback. Recursion handles arbitrary nesting depth; the dispatch loop in the app stays one walk regardless of N.
///
/// `f` is `&mut dyn FnMut` (not generic) so the trait stays object-safe — `&mut dyn Container` is a usable currency, letting helpers like [`linear_tab_next`] take a generic root without monomorphising per app type.
pub trait Container {
    fn visit(&mut self, f: &mut dyn FnMut(&mut dyn Widget));
}

/// Direction for [`linear_tab_next`]. `Forward` = Tab; `Backward` = Shift+Tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TabDir {
    Forward,
    Backward,
}

/// Pick the next focusable widget in registration order. Walks `root` collecting the IDs of every widget whose [`Widget::focus`] returns `Some`, then returns the neighbour of `current` in the requested direction. Wraps modularly (Tab from the last focusable → the first; Shift+Tab from the first → the last). Returns `None` only if there are zero focusables in the tree.
///
/// `current = None` (nothing currently focused) jumps to the first focusable on Forward, or the last on Backward — matches the OS convention of "Tab into the form starts at the top, Shift+Tab into the form starts at the bottom."
///
/// **Cost.** One full `visit` walk per call. At fluor's expected widget counts (tens, not thousands) this is microseconds; if it ever shows up in a profile, cache the focusable-IDs vector at paint time and re-use across the frame.
pub fn linear_tab_next(
    root: &mut dyn Container,
    current: Option<HitId>,
    dir: TabDir,
) -> Option<HitId> {
    let mut focusables: alloc::vec::Vec<HitId> = alloc::vec::Vec::new();
    root.visit(&mut |w| {
        if w.focus().is_some() {
            focusables.push(w.id());
        }
    });
    if focusables.is_empty() {
        return None;
    }
    let n = focusables.len();
    let idx_of_current = current.and_then(|cur| focusables.iter().position(|&id| id == cur));
    let next_idx = match (idx_of_current, dir) {
        (None, TabDir::Forward) => 0,
        (None, TabDir::Backward) => n - 1,
        (Some(i), TabDir::Forward) => (i + 1) % n,
        (Some(i), TabDir::Backward) => (i + n - 1) % n,
    };
    Some(focusables[next_idx])
}

/// Build the per-hit-id overlay delta table by walking the widget tree once and asking every [`Hover`]-capable widget for its [`Hover::tint_delta`]. The returned `Vec<u32>` is sized to `count` (typically `hit_counter + 1` since IDs are 1-indexed with `HIT_NONE = 0` at slot 0). Widgets whose [`HitId`] is `>= count` are silently skipped — keeps the helper safe when an app resizes its registry between frames. Drop-in replacement for the hand-rolled overlay_deltas that panes' demo used pre-walk.
pub fn build_overlay_deltas(root: &mut dyn Container, count: usize) -> alloc::vec::Vec<u32> {
    let mut t = alloc::vec![0u32; count];
    root.visit(&mut |w| {
        let id = w.id() as usize;
        if id < t.len() {
            if let Some(h) = w.hover() {
                t[id] = h.tint_delta();
            }
        }
    });
    t
}

/// Build the per-hit-id overlay BBOX table (parallel to [`build_overlay_deltas`]): each Hover-capable widget's [`Hover::hover_bbox`] by id, so the host can bound the overlay tint scan to each widget instead of the whole window.
/// Entry is `None` for ids with no widget / no bbox → the host full-window scans those (safe fallback).
pub fn build_overlay_bboxes(
    root: &mut dyn Container,
    count: usize,
    viewport_w: usize,
    viewport_h: usize,
) -> alloc::vec::Vec<Option<PixelRect>> {
    let mut t = alloc::vec![None; count];
    root.visit(&mut |w| {
        let id = w.id() as usize;
        if id < t.len() {
            if let Some(h) = w.hover() {
                t[id] = h.hover_bbox(viewport_w, viewport_h);
            }
        }
    });
    t
}

/// Deliver a click to the widget with `target_id`, if any. Walks `root` once, finds the widget whose [`Widget::id`] matches, asks for its [`Click`] capability, and invokes [`Click::on_click`] with the given `(x, y, mods)`. Returns the widget's [`crate::host::EventResponse`], or [`crate::host::EventResponse::Pass`] if no matching widget exists or the matching widget has no [`Click`] capability — same convention as a missing handler so apps can `?`-chain into chrome-button dispatch without special-casing the no-widget arm.
pub fn dispatch_click(
    root: &mut dyn Container,
    target_id: HitId,
    x: Coord,
    y: Coord,
    mods: ModifiersState,
) -> crate::host::EventResponse {
    let mut response = crate::host::EventResponse::Pass;
    root.visit(&mut |w| {
        if w.id() == target_id {
            if let Some(c) = w.click() {
                response = c.on_click(x, y, mods);
            }
        }
    });
    response
}

/// Deliver a key event to the widget with `target_id`, if any. Mirror of [`dispatch_click`] for keyboard input — caller picks the target (typically the currently-focused widget tracked by the app) and this routes the fluor-native [`KeyEvent`] + [`ModifiersState`] + [`TextRenderer`] handle thru to the widget's [`Key::on_key`] impl. Returns [`crate::host::EventResponse::Pass`] if `target_id` doesn't match any widget or the matching widget doesn't impl [`Key`].
pub fn dispatch_key(
    root: &mut dyn Container,
    target_id: HitId,
    kev: &KeyEvent,
    mods: ModifiersState,
    text: &mut TextRenderer,
) -> crate::host::EventResponse {
    let mut response = crate::host::EventResponse::Pass;
    root.visit(&mut |w| {
        if w.id() == target_id {
            if let Some(k) = w.key() {
                response = k.on_key(kev, mods, text);
            }
        }
    });
    response
}

/// Apply a focus change: call `set_focused(false)` on the old target (if any) and `set_focused(true)` on the new target (if any). Idempotent when `old == new`. Walks `root` once per non-null target so widgets that change visual state on focus transition can mark themselves dirty in the same frame.
pub fn apply_focus_change(root: &mut dyn Container, old: Option<HitId>, new: Option<HitId>) {
    if old == new {
        return;
    }
    if let Some(old_id) = old {
        root.visit(&mut |w| {
            if w.id() == old_id {
                if let Some(f) = w.focus() {
                    f.set_focused(false);
                }
            }
        });
    }
    if let Some(new_id) = new {
        root.visit(&mut |w| {
            if w.id() == new_id {
                if let Some(f) = w.focus() {
                    f.set_focused(true);
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paint::HIT_NONE;

    struct TestWidget {
        id: HitId,
        focusable: bool,
        focused: bool,
        enabled: bool,
        clicks: u32,
    }

    impl Widget for TestWidget {
        fn id(&self) -> HitId {
            self.id
        }
        fn paint(&mut self, _ctx: &mut PaintCtx<'_, '_>) {}
        // Mirrors the real widgets: a disabled widget vanishes from every capability accessor, so the dispatch + tab-cycle helpers skip it without per-handler checks.
        fn focus(&mut self) -> Option<&mut dyn Focus> {
            (self.focusable && self.enabled).then_some(self as &mut dyn Focus)
        }
        fn click(&mut self) -> Option<&mut dyn Click> {
            self.enabled.then_some(self as &mut dyn Click)
        }
    }

    impl Focus for TestWidget {
        fn set_focused(&mut self, focused: bool) {
            self.focused = focused;
        }
    }

    impl Click for TestWidget {
        fn on_click(
            &mut self,
            _x: Coord,
            _y: Coord,
            _mods: ModifiersState,
        ) -> crate::host::EventResponse {
            self.clicks += 1;
            crate::host::EventResponse::Handled
        }
    }

    struct TestRoot {
        widgets: alloc::vec::Vec<TestWidget>,
    }

    impl Container for TestRoot {
        fn visit(&mut self, f: &mut dyn FnMut(&mut dyn Widget)) {
            for w in self.widgets.iter_mut() {
                f(w);
            }
        }
    }

    #[test]
    fn next_id_starts_at_one() {
        let mut counter: HitId = HIT_NONE;
        assert_eq!(next_id(&mut counter), 1);
        assert_eq!(next_id(&mut counter), 2);
        assert_eq!(next_id(&mut counter), 3);
    }

    #[test]
    fn linear_tab_skips_non_focusable() {
        let mut root = TestRoot {
            widgets: alloc::vec![
                TestWidget {
                    id: 1,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
                TestWidget {
                    id: 2,
                    focusable: false,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
                TestWidget {
                    id: 3,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
            ],
        };
        let next = linear_tab_next(&mut root, Some(1), TabDir::Forward);
        assert_eq!(next, Some(3));
    }

    #[test]
    fn linear_tab_wraps_forward() {
        let mut root = TestRoot {
            widgets: alloc::vec![
                TestWidget {
                    id: 1,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
                TestWidget {
                    id: 2,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
            ],
        };
        let next = linear_tab_next(&mut root, Some(2), TabDir::Forward);
        assert_eq!(next, Some(1));
    }

    #[test]
    fn linear_tab_wraps_backward() {
        let mut root = TestRoot {
            widgets: alloc::vec![
                TestWidget {
                    id: 1,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
                TestWidget {
                    id: 2,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
            ],
        };
        let next = linear_tab_next(&mut root, Some(1), TabDir::Backward);
        assert_eq!(next, Some(2));
    }

    #[test]
    fn linear_tab_none_goes_to_first_forward() {
        let mut root = TestRoot {
            widgets: alloc::vec![
                TestWidget {
                    id: 5,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
                TestWidget {
                    id: 7,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
            ],
        };
        let next = linear_tab_next(&mut root, None, TabDir::Forward);
        assert_eq!(next, Some(5));
    }

    #[test]
    fn linear_tab_none_goes_to_last_backward() {
        let mut root = TestRoot {
            widgets: alloc::vec![
                TestWidget {
                    id: 5,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
                TestWidget {
                    id: 7,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
            ],
        };
        let next = linear_tab_next(&mut root, None, TabDir::Backward);
        assert_eq!(next, Some(7));
    }

    #[test]
    fn linear_tab_empty_returns_none() {
        let mut root = TestRoot {
            widgets: alloc::vec![],
        };
        assert_eq!(linear_tab_next(&mut root, None, TabDir::Forward), None);
    }

    #[test]
    fn apply_focus_change_toggles_old_and_new() {
        let mut root = TestRoot {
            widgets: alloc::vec![
                TestWidget {
                    id: 1,
                    focusable: true,
                    focused: true,
                    enabled: true,
                    clicks: 0
                },
                TestWidget {
                    id: 2,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
            ],
        };
        apply_focus_change(&mut root, Some(1), Some(2));
        assert!(!root.widgets[0].focused);
        assert!(root.widgets[1].focused);
    }

    #[test]
    fn apply_focus_change_noop_when_same() {
        let mut root = TestRoot {
            widgets: alloc::vec![TestWidget {
                id: 1,
                focusable: true,
                focused: true,
                enabled: true,
                clicks: 0
            }],
        };
        apply_focus_change(&mut root, Some(1), Some(1));
        assert!(root.widgets[0].focused);
    }

    // --- Disabled widgets drop out of dispatch + tab cycle via their `None` capability accessors ---

    #[test]
    fn linear_tab_skips_disabled() {
        let mut root = TestRoot {
            widgets: alloc::vec![
                TestWidget {
                    id: 1,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
                TestWidget {
                    id: 2,
                    focusable: true,
                    focused: false,
                    enabled: false, // disabled → focus() returns None → skipped by the cycle
                    clicks: 0
                },
                TestWidget {
                    id: 3,
                    focusable: true,
                    focused: false,
                    enabled: true,
                    clicks: 0
                },
            ],
        };
        // Tab from 1 lands on 3, not the disabled 2.
        assert_eq!(linear_tab_next(&mut root, Some(1), TabDir::Forward), Some(3));
    }

    #[test]
    fn dispatch_click_skips_disabled() {
        let mut root = TestRoot {
            widgets: alloc::vec![TestWidget {
                id: 1,
                focusable: true,
                focused: false,
                enabled: false, // disabled → click() returns None
                clicks: 0
            }],
        };
        let resp = dispatch_click(&mut root, 1, 0.0, 0.0, ModifiersState::default());
        // No Click capability is exposed, so the click goes unconsumed and never fires.
        assert_eq!(resp, crate::host::EventResponse::Pass);
        assert_eq!(root.widgets[0].clicks, 0);
    }
}
