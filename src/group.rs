//! Group — the unit of hit-routable, text-clippable composite.
//!
//! A `Group` wraps a [`Region`] (pixel-space bbox in target-buffer coordinates), a [`StackCompositor`] for internal RGB compositing, and optional side channels: a binary hit mask (`Option<Vec<u8>>`, one byte per pixel, 0 or 1) and a per-pixel text-clip alpha mask (`Option<Vec<u8>>`). All buffers are sized to the group's bbox — *not* the viewport — so memory scales with content area, not with viewport size.
//!
//! Groups are leaf-ish: there is no `children` field. The tree IS the consumer's code, exactly like [`Region`]'s "no parent pointers" doctrine. Consumers compose by holding a `Vec<Group>` (or any structure they prefer) and calling [`Group::flatten_into`] in order; hit testing iterates the same vec in reverse.
//!
//! Coordinate translation: paint primitives take buffer-relative pixel coordinates. When rasterizing into a group's Stack layer, the buffer is sized to `(region.w, region.h)`, so consumers must subtract `region.x, region.y` from any viewport-relative coordinate.
//!
//! Re-rasterization gate: per-layer `dirty: bool` flags inside [`StackCompositor`] gate expensive rasterization (text shaping, glyph rendering, squircle math). Consumers mark a layer dirty when its state changes; the layer re-rasterizes once. The Stack itself caches its final composite — when no layer is dirty, `evaluate()` returns the cache directly without re-running the program.
//!
//! "Where does this group contribute?" answer at flatten time: per-pixel `pixel >= 0xFF000000` (fully transparent) shortcut in the blend kernels skips contribution-less pixels for free. No sub-region damage tracking — the transparency map IS the dirty rect at per-pixel granularity. If a Group's bbox becomes too large for full-bbox flatten to be acceptable (chrome at 4K), the answer is to decompose into multiple smaller Groups (per-button, per-region), not to maintain a `Vec<Region>` damage list.

use crate::coord::Coord;
use crate::paint;
use crate::pixel::BlendMode;
use crate::region::Region;
use crate::stack::{Op, StackCompositor};
use alloc::vec::Vec;

pub struct Group {
    /// Pixel-space bbox in target-buffer coordinates. Buffers (Stack layers, hitmask, text_clip) are sized to `(region.w, region.h)`.
    pub region: Region,
    /// Internal RGB compositing — bbox-sized layer buffers + Stack program.
    pub rpn: StackCompositor,
    /// Binary hit mask (group-local, one byte per pixel, 0 or 1). `None` = decorative; clicks pass through.
    pub hitmask: Option<Vec<u8>>,
    /// Per-pixel alpha mask consumed by text rasterizers drawing into this group's RGB layers (group-local). `None` = no soft clip.
    pub text_clip: Option<Vec<u8>>,
    /// How this group's flatten composites onto the target buffer.
    pub blend: BlendMode,
    /// Default hit id this Group contributes when it is the topmost non-transparent contributor at a queried pixel and either no hitmask is enabled or the hitmask byte at this pixel is 0. A group with `id == 0` and no hitmask contributes pixels but is "decorative" for hit dispatch (clicks fall through). Consumers assign meaningful ids per Group (e.g., chrome=1, textbox=3, cursor=4).
    pub id: u16,
}

impl Group {
    /// New group with bbox = `region`, no side channels, Stack with no layers and an empty program. Allocate side channels separately via [`enable_hitmask`](Self::enable_hitmask) / [`enable_text_clip`](Self::enable_text_clip); add layers via [`new_layer`](Self::new_layer); set the compositing program via [`set_program`](Self::set_program).
    pub fn new(region: Region, blend: BlendMode) -> Self {
        let w = region.w as usize;
        let h = region.h as usize;
        Self {
            region,
            rpn: StackCompositor::new(w, h),
            hitmask: None,
            text_clip: None,
            blend,
            id: 0,
        }
    }

    /// Bbox dimensions in pixels. Convenience for sizing local buffers / clip rects.
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.region.w as usize, self.region.h as usize)
    }

    /// Add an Stack layer; returns its index for use in `Op::Push(idx)`. The layer's pixel buffer is bbox-sized.
    pub fn new_layer(&mut self) -> usize {
        self.rpn.new_layer()
    }

    /// Replace the Stack compositing program.
    pub fn set_program(&mut self, program: Vec<Op>) {
        self.rpn.set_program(program);
    }

    /// Allocate the binary hit mask (one byte per pixel, 0 or 1) sized to the group's bbox. Initialized to all zeros (passthrough). Consumers stamp hit-active pixels by writing 1 into [`hitmask_mut`](Self::hitmask_mut).
    pub fn enable_hitmask(&mut self) {
        let (w, h) = self.dims();
        self.hitmask = Some(alloc::vec![0u8; w * h]);
    }

    /// Allocate the per-pixel text-clip alpha mask sized to the group's bbox. Initialized to all zeros (fully clipped). Consumers fill the mask to define where text is allowed to draw inside this group.
    pub fn enable_text_clip(&mut self) {
        let (w, h) = self.dims();
        self.text_clip = Some(alloc::vec![0u8; w * h]);
    }

    /// Mutable access to the hitmask (None if decorative). Consumers write 0/1 bytes here.
    pub fn hitmask_mut(&mut self) -> Option<&mut [u8]> {
        self.hitmask.as_deref_mut()
    }

    /// Mutable access to the text-clip mask (None if not enabled). Consumers write alpha bytes here.
    pub fn text_clip_mut(&mut self) -> Option<&mut [u8]> {
        self.text_clip.as_deref_mut()
    }

    /// Resize bbox + reallocate internal buffers. All Stack layers are marked dirty (re-rasterization is the consumer's responsibility); hit + text-clip masks are zeroed.
    pub fn resize(&mut self, region: Region) {
        let (w, h) = (region.w as usize, region.h as usize);
        self.region = region;
        self.rpn.resize(w, h);
        if let Some(m) = self.hitmask.as_mut() {
            m.resize(w * h, 0);
            m.fill(0);
        }
        if let Some(m) = self.text_clip.as_mut() {
            m.resize(w * h, 0);
            m.fill(0);
        }
    }

    /// Reposition (and optionally resize) the bbox. If dimensions are unchanged, only `region.x/y` update — buffers preserved, but every layer is marked dirty so the next `flatten_into` re-blits at the new target offset (the Stack cache is per-content; the *position on target* is a separate concern). If dimensions change, behaves like [`resize`](Self::resize).
    pub fn set_region(&mut self, region: Region) {
        let same_dims = (region.w as usize) == (self.region.w as usize)
            && (region.h as usize) == (self.region.h as usize);
        if same_dims {
            self.region = region;
            self.invalidate();
        } else {
            self.resize(region);
        }
    }

    /// Group-local hit query. `(x, y)` is in target-buffer pixel space; subtracts `region.x/y` and reads the hitmask byte. Returns false for decorative groups (`hitmask = None`) or out-of-bbox points.
    pub fn hit(&self, x: Coord, y: Coord) -> bool {
        let Some(mask) = self.hitmask.as_ref() else {
            return false;
        };
        if !self.region.contains(x, y) {
            return false;
        }
        let lx = (x - self.region.x) as usize;
        let ly = (y - self.region.y) as usize;
        let w = self.region.w as usize;
        mask[ly * w + lx] != 0
    }

    /// Mark every layer dirty (forces re-rasterization on the next render). Use after viewport resize, focus/hover state change, or any other state that affects what the layers should paint.
    pub fn invalidate(&mut self) {
        for layer in &mut self.rpn.layers {
            layer.dirty = true;
        }
    }

    /// Flatten the internal Stack onto `target` at `region.x, region.y` using `self.blend`. Always blits — the host's present buffer may be double-buffered, and downstream groups composited above this one may have overwritten our pixels last frame. The internal `StackCompositor::evaluate` cheaply returns a cached composite when no layer is dirty, so the per-frame cost when nothing changed is just the blit (one pass over the bbox area, not over the viewport).
    ///
    /// Per-pixel transparent shortcut (`pixel >= 0xFF000000`) inside each blend kernel skips contribution-less pixels for free — no sub-region damage tracking needed at this layer.
    ///
    /// Pixels that would land outside the target's bounds are clipped row-by-row before the per-row blend kernel runs; the blend kernel itself sees only in-bounds slices.
    pub fn flatten_into(&mut self, target: &mut [u32], target_w: usize, target_h: usize) {
        let composite = self.rpn.evaluate();
        if composite.is_empty() {
            return;
        }

        let (gw, gh) = (self.region.w as usize, self.region.h as usize);
        let gx = self.region.x as isize;
        let gy = self.region.y as isize;

        for row in 0..gh {
            let ty = gy + row as isize;
            if ty < 0 || (ty as usize) >= target_h {
                continue;
            }
            let src_row_start = row * gw;
            let dst_row_start = (ty as usize) * target_w;

            // Horizontal clip: intersect [gx, gx+gw) with [0, target_w).
            let dst_x_start = gx.max(0) as usize;
            let dst_x_end_isize = (gx + gw as isize).min(target_w as isize);
            if dst_x_end_isize <= dst_x_start as isize {
                continue;
            }
            let dst_x_end = dst_x_end_isize as usize;
            let src_clip_left = (dst_x_start as isize - gx) as usize;
            let count = dst_x_end - dst_x_start;

            let src_slice =
                &composite[src_row_start + src_clip_left..src_row_start + src_clip_left + count];
            let dst_slice = &mut target[dst_row_start + dst_x_start..dst_row_start + dst_x_end];
            paint::flatten(dst_slice, src_slice, self.blend);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an opaque pixel (t=0) with the given RGB.
    fn opaque(rgb: u32) -> u32 {
        rgb & 0x00FFFFFF
    }
    /// The canonical transparent pixel — t=255, RGB=0. Target buffers must be initialized to this before `flatten_into` so the under-blend has a clean accumulator state.
    const TRANSPARENT: u32 = 0xFFFFFFFF;

    #[test]
    fn new_group_has_correct_bbox_buffers() {
        let g = Group::new(Region::new(10.0, 20.0, 4.0, 3.0), BlendMode::Normal);
        assert_eq!(g.dims(), (4, 3));
        assert!(g.hitmask.is_none());
        assert!(g.text_clip.is_none());
    }

    #[test]
    fn enable_hitmask_allocates_zeroed_bbox_sized_buffer() {
        let mut g = Group::new(Region::new(0.0, 0.0, 3.0, 2.0), BlendMode::Normal);
        g.enable_hitmask();
        let m = g.hitmask.as_ref().unwrap();
        assert_eq!(m.len(), 6);
        assert!(m.iter().all(|&b| b == 0));
    }

    #[test]
    fn flatten_under_normal_blits_opaque_layer_at_offset() {
        // Group at (1, 1), 2x2, Normal under-blend, single layer = solid opaque red.
        // Target is pre-initialized to TRANSPARENT (t=255) — the caller-side contract for any
        // buffer participating in the under-chain. Opaque red bottom blended underneath
        // transparent target → 1-LSB-off red lands at the group's pixels.
        let mut g = Group::new(Region::new(1.0, 1.0, 2.0, 2.0), BlendMode::Normal);
        let l = g.new_layer();
        g.rpn.layers[l].pixels = alloc::vec![opaque(0xFF0000); 4];
        g.set_program(alloc::vec![Op::Push(l)]);

        let mut target = alloc::vec![TRANSPARENT; 4 * 4];
        g.flatten_into(&mut target, 4, 4);

        // Group's pixels (1,1)..(2,2) land near opaque red (off-by-1-LSB from the (contrib+1) trick going through Blend::under once).
        for &(x, y) in &[(1, 1), (2, 1), (1, 2), (2, 2)] {
            let p = target[y * 4 + x];
            let r = (p >> 16) & 0xFF;
            let t = p >> 24;
            assert_eq!(t, 0, "pixel ({x},{y}) t expected 0, got {t:#x}");
            assert!(r >= 0xFD, "pixel ({x},{y}) r expected ~0xFE, got {r:#x}");
        }
        // Pixels outside the group's bbox stay at TRANSPARENT.
        assert_eq!(target[0 * 4 + 0], TRANSPARENT);
        assert_eq!(target[3 * 4 + 3], TRANSPARENT);
    }

    #[test]
    fn flatten_clips_to_target_bounds() {
        // Group at (-1, -1), 4x4 (extends past target's left and top edges into the buffer).
        let mut g = Group::new(Region::new(-1.0, -1.0, 4.0, 4.0), BlendMode::Normal);
        let l = g.new_layer();
        g.rpn.layers[l].pixels = alloc::vec![opaque(0x00FF00); 16];
        g.set_program(alloc::vec![Op::Push(l)]);

        let mut target = alloc::vec![TRANSPARENT; 4 * 4];
        g.flatten_into(&mut target, 4, 4);

        // Group's (1, 1) lands at target (0, 0); its (3, 3) lands at target (2, 2). target (3, 3) untouched.
        let g_pixel = target[0 * 4 + 0];
        assert_eq!(
            g_pixel >> 24,
            0,
            "blitted pixel should be opaque after under-blend"
        );
        assert!(((g_pixel >> 8) & 0xFF) >= 0xFD, "G channel ~0xFE");
        assert_eq!(target[3 * 4 + 3], TRANSPARENT);
    }

    #[test]
    fn flatten_always_blits_for_double_buffering_safety() {
        // The host's present buffer may be double-buffered; flatten must always write the
        // group's content into target even when nothing internal is dirty.
        let mut g = Group::new(Region::new(0.0, 0.0, 2.0, 1.0), BlendMode::Normal);
        let l = g.new_layer();
        g.rpn.layers[l].pixels = alloc::vec![opaque(0xAABBCC); 2];
        g.set_program(alloc::vec![Op::Push(l)]);

        let mut target = alloc::vec![TRANSPARENT; 4];
        g.flatten_into(&mut target, 2, 2);
        let first = target[0];
        assert_eq!(first >> 24, 0, "first blit should make pixel opaque");

        // Simulate a back-buffer swap: target is now TRANSPARENT again (the OTHER frame buffer).
        target.fill(TRANSPARENT);
        // Flatten must still write our content even though no layer is dirty.
        g.flatten_into(&mut target, 2, 2);
        let second = target[0];
        assert_eq!(
            second >> 24,
            0,
            "second blit must reproduce content (double-buffer safety)"
        );
        assert_eq!(first, second);
    }

    #[test]
    fn hit_returns_false_for_decorative_group() {
        let g = Group::new(Region::new(0.0, 0.0, 10.0, 10.0), BlendMode::Normal);
        assert!(!g.hit(5.0, 5.0));
    }

    #[test]
    fn hit_returns_true_for_active_pixel() {
        let mut g = Group::new(Region::new(2.0, 3.0, 4.0, 4.0), BlendMode::Normal);
        g.enable_hitmask();
        // Stamp a 1 at group-local (1, 2) → target (3, 5).
        g.hitmask_mut().unwrap()[2 * 4 + 1] = 1;
        assert!(g.hit(3.0, 5.0));
        assert!(!g.hit(3.0, 6.0));
    }

    #[test]
    fn hit_returns_false_outside_bbox() {
        let mut g = Group::new(Region::new(2.0, 3.0, 4.0, 4.0), BlendMode::Normal);
        g.enable_hitmask();
        g.hitmask_mut().unwrap().fill(1);
        assert!(!g.hit(1.0, 5.0)); // left of bbox
        assert!(!g.hit(7.0, 5.0)); // right of bbox (exclusive)
        assert!(!g.hit(3.0, 2.0)); // above bbox
        assert!(!g.hit(3.0, 8.0)); // below bbox (exclusive)
    }
}
