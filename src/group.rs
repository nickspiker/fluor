//! Group — the unit of cached, hit-routable, text-clippable composite.
//!
//! A `Group` wraps a [`Region`] (pixel-space bbox in target-buffer coordinates), an [`RpnCompositor`] for internal RGB compositing, and optional side channels: a binary hit mask (`Option<Vec<u8>>`, one byte per pixel, 0 or 1) and a per-pixel text-clip alpha mask (`Option<Vec<u8>>`). All buffers are sized to the group's bbox — *not* the viewport — so memory scales with content area, not with viewport size.
//!
//! Groups are leaf-ish: there is no `children` field. The tree IS the consumer's code, exactly like [`Region`]'s "no parent pointers" doctrine. Consumers compose by holding a `Vec<Group>` (or any structure they prefer) and calling [`Group::flatten_into`] in order; hit testing iterates the same vec in reverse.
//!
//! Coordinate translation: paint primitives take buffer-relative pixel coordinates. When rasterizing into a group's RPN layer, the buffer is sized to `(region.w, region.h)`, so consumers must subtract `region.x, region.y` from any viewport-relative coordinate. There is deliberately no wrapper "GroupCanvas" type — per AGENT.md "no design for hypothetical futures," we add one only when a real consumer hurts.
//!
//! Damage tracking: [`mark_damage`](Group::mark_damage) coalesces overlapping rects on insert. [`is_dirty`](Group::is_dirty) lets consumers gate their rasterization. [`flatten_into`](Group::flatten_into) re-runs the whole RPN program when *anything* is dirty (per-pixel partial RPN re-evaluation is a future optimization). The savings come from gating consumer-side rasterization (text shaping, glyph rasterization, squircle math) — that's where the cycles live.

use alloc::vec::Vec;
use crate::coord::Coord;
use crate::paint::BlendMode;
use crate::region::Region;
use crate::rpn::{Op, RpnCompositor};

pub struct Group {
    /// Pixel-space bbox in target-buffer coordinates. Buffers (RPN layers, hitmask, text_clip) are sized to `(region.w, region.h)`.
    pub region: Region,
    /// Internal RGB compositing — bbox-sized layer buffers + RPN program.
    pub rpn: RpnCompositor,
    /// Binary hit mask (group-local, one byte per pixel, 0 or 1). `None` = decorative; clicks pass through.
    pub hitmask: Option<Vec<u8>>,
    /// Per-pixel alpha mask consumed by text rasterizers drawing into this group's RGB layers (group-local). `None` = no soft clip.
    pub text_clip: Option<Vec<u8>>,
    /// Damage rects in group-local coordinates. Empty = clean. Coalesced on insert.
    pub damage: Vec<Region>,
    /// How this group's flatten composites onto the target buffer.
    pub blend: BlendMode,
}

impl Group {
    /// New group with bbox = `region`, no side channels, RPN with no layers and an empty program. Allocate side channels separately via [`enable_hitmask`](Self::enable_hitmask) / [`enable_text_clip`](Self::enable_text_clip); add layers via [`add_layer`](Self::add_layer); set the compositing program via [`set_program`](Self::set_program).
    pub fn new(region: Region, blend: BlendMode) -> Self {
        let w = region.w as usize;
        let h = region.h as usize;
        Self {
            region,
            rpn: RpnCompositor::new(w, h),
            hitmask: None,
            text_clip: None,
            damage: Vec::new(),
            blend,
        }
    }

    /// Bbox dimensions in pixels. Convenience for sizing local buffers / clip rects.
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.region.w as usize, self.region.h as usize)
    }

    /// Add an RPN layer; returns its index for use in `Op::Push(idx)`. The layer's pixel buffer is bbox-sized.
    pub fn add_layer(&mut self) -> usize {
        self.rpn.add_layer()
    }

    /// Replace the RPN compositing program.
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

    /// Resize bbox + reallocate internal buffers. All RPN layers are marked dirty (re-rasterization is the consumer's responsibility); hit + text-clip masks are zeroed; damage list is cleared.
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
        self.damage.clear();
    }

    /// Group-local hit query. `(x, y)` is in target-buffer pixel space; subtracts `region.x/y` and reads the hitmask byte. Returns false for decorative groups (`hitmask = None`) or out-of-bbox points.
    pub fn hit(&self, x: Coord, y: Coord) -> bool {
        let Some(mask) = self.hitmask.as_ref() else { return false; };
        if !self.region.contains(x, y) { return false; }
        let lx = (x - self.region.x) as usize;
        let ly = (y - self.region.y) as usize;
        let w = self.region.w as usize;
        mask[ly * w + lx] != 0
    }

    /// Coalescing damage insert. If `rect` overlaps any existing damage region, replace that region with their union; otherwise append. Degenerate rects (`w <= 0` or `h <= 0`) are dropped. O(N) per insert in the number of currently-tracked damage rects.
    pub fn mark_damage(&mut self, rect: Region) {
        if rect.w <= 0.0 || rect.h <= 0.0 { return; }
        for existing in self.damage.iter_mut() {
            if existing.intersects(&rect) {
                *existing = existing.union(&rect);
                return;
            }
        }
        self.damage.push(rect);
    }

    /// True if any damage region intersects `rect` (group-local coordinates). Consumers gate their rasterization on this query.
    pub fn is_dirty(&self, rect: &Region) -> bool {
        self.damage.iter().any(|d| d.intersects(rect))
    }

    /// Read-only view of the current damage list.
    pub fn dirty_rects(&self) -> &[Region] {
        &self.damage
    }

    /// Mark every layer dirty (forces a full re-flatten on the next call). Use after viewport resize, target-buffer clear, or any time the contract "target unchanged since last flatten" no longer holds.
    pub fn invalidate(&mut self) {
        for layer in &mut self.rpn.layers {
            layer.dirty = true;
        }
    }

    /// Flatten the internal RPN onto `target` at `region.x, region.y` using `self.blend`. Skips entirely when no RPN layer is dirty *and* the damage list is empty — consumers must call [`invalidate`](Self::invalidate) when the target is overwritten externally (e.g., the host clears the frame buffer between paints).
    ///
    /// Pixels that would land outside the target's bounds are clipped row-by-row before the per-row blend kernel runs; the blend kernel itself sees only in-bounds slices.
    pub fn flatten_into(&mut self, target: &mut [u32], target_w: usize, target_h: usize) {
        let needs_flatten = self.rpn.layers.iter().any(|l| l.dirty) || !self.damage.is_empty();
        if !needs_flatten { return; }

        let composite = self.rpn.evaluate();
        if composite.is_empty() { self.damage.clear(); return; }

        let (gw, gh) = (self.region.w as usize, self.region.h as usize);
        let gx = self.region.x as isize;
        let gy = self.region.y as isize;

        for row in 0..gh {
            let ty = gy + row as isize;
            if ty < 0 || (ty as usize) >= target_h { continue; }
            let src_row_start = row * gw;
            let dst_row_start = (ty as usize) * target_w;

            // Horizontal clip: intersect [gx, gx+gw) with [0, target_w).
            let dst_x_start = gx.max(0) as usize;
            let dst_x_end_isize = (gx + gw as isize).min(target_w as isize);
            if dst_x_end_isize <= dst_x_start as isize { continue; }
            let dst_x_end = dst_x_end_isize as usize;
            let src_clip_left = (dst_x_start as isize - gx) as usize;
            let count = dst_x_end - dst_x_start;

            let src_slice = &composite[src_row_start + src_clip_left .. src_row_start + src_clip_left + count];
            let dst_slice = &mut target[dst_row_start + dst_x_start .. dst_row_start + dst_x_end];
            self.blend.flatten(dst_slice, src_slice);
        }

        self.damage.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opaque(c: u32) -> u32 { 0xFF000000 | (c & 0x00FFFFFF) }

    #[test]
    fn new_group_has_correct_bbox_buffers() {
        let g = Group::new(Region::new(10.0, 20.0, 4.0, 3.0), BlendMode::Replace);
        assert_eq!(g.dims(), (4, 3));
        assert!(g.hitmask.is_none());
        assert!(g.text_clip.is_none());
        assert!(g.damage.is_empty());
    }

    #[test]
    fn enable_hitmask_allocates_zeroed_bbox_sized_buffer() {
        let mut g = Group::new(Region::new(0.0, 0.0, 3.0, 2.0), BlendMode::Replace);
        g.enable_hitmask();
        let m = g.hitmask.as_ref().unwrap();
        assert_eq!(m.len(), 6);
        assert!(m.iter().all(|&b| b == 0));
    }

    #[test]
    fn flatten_replace_blits_at_offset() {
        // Group at (1, 1), 2x2, Replace blend, single layer = solid red.
        let mut g = Group::new(Region::new(1.0, 1.0, 2.0, 2.0), BlendMode::Replace);
        let l = g.add_layer();
        g.rpn.layers[l].pixels = alloc::vec![opaque(0xFF0000); 4];
        g.set_program(alloc::vec![Op::Push(l)]);

        let mut target = alloc::vec![0u32; 4 * 4];
        g.flatten_into(&mut target, 4, 4);

        // Expect pixels at (1,1), (2,1), (1,2), (2,2) to be red; rest zero.
        assert_eq!(target[1 * 4 + 1], opaque(0xFF0000));
        assert_eq!(target[1 * 4 + 2], opaque(0xFF0000));
        assert_eq!(target[2 * 4 + 1], opaque(0xFF0000));
        assert_eq!(target[2 * 4 + 2], opaque(0xFF0000));
        assert_eq!(target[0 * 4 + 0], 0);
        assert_eq!(target[3 * 4 + 3], 0);
    }

    #[test]
    fn flatten_clips_to_target_bounds() {
        // Group at (-1, -1), 4x4 (extends past target's left and top edges into the buffer).
        let mut g = Group::new(Region::new(-1.0, -1.0, 4.0, 4.0), BlendMode::Replace);
        let l = g.add_layer();
        g.rpn.layers[l].pixels = alloc::vec![opaque(0x00FF00); 16];
        g.set_program(alloc::vec![Op::Push(l)]);

        let mut target = alloc::vec![0u32; 4 * 4];
        g.flatten_into(&mut target, 4, 4);

        // Group's (1, 1) lands at target (0, 0); its (3, 3) lands at target (2, 2). target (3, 3) untouched.
        assert_eq!(target[0 * 4 + 0], opaque(0x00FF00));
        assert_eq!(target[2 * 4 + 2], opaque(0x00FF00));
        assert_eq!(target[3 * 4 + 3], 0);
    }

    #[test]
    fn flatten_skips_when_clean() {
        // First flatten dirties + paints; second flatten with no changes should skip (target unchanged).
        let mut g = Group::new(Region::new(0.0, 0.0, 2.0, 1.0), BlendMode::Replace);
        let l = g.add_layer();
        g.rpn.layers[l].pixels = alloc::vec![opaque(0xAABBCC); 2];
        g.set_program(alloc::vec![Op::Push(l)]);

        let mut target = alloc::vec![0u32; 4];
        g.flatten_into(&mut target, 2, 2);
        assert_eq!(target[0], opaque(0xAABBCC));

        // Externally clear target; without invalidate, flatten should skip and leave target zero.
        target.fill(0);
        g.flatten_into(&mut target, 2, 2);
        assert_eq!(target[0], 0, "flatten should skip when nothing dirty");

        // After invalidate, flatten re-blits.
        g.invalidate();
        g.flatten_into(&mut target, 2, 2);
        assert_eq!(target[0], opaque(0xAABBCC));
    }

    #[test]
    fn hit_returns_false_for_decorative_group() {
        let g = Group::new(Region::new(0.0, 0.0, 10.0, 10.0), BlendMode::Replace);
        assert!(!g.hit(5.0, 5.0));
    }

    #[test]
    fn hit_returns_true_for_active_pixel() {
        let mut g = Group::new(Region::new(2.0, 3.0, 4.0, 4.0), BlendMode::Replace);
        g.enable_hitmask();
        // Stamp a 1 at group-local (1, 2) → target (3, 5).
        g.hitmask_mut().unwrap()[2 * 4 + 1] = 1;
        assert!(g.hit(3.0, 5.0));
        assert!(!g.hit(3.0, 6.0));
    }

    #[test]
    fn hit_returns_false_outside_bbox() {
        let mut g = Group::new(Region::new(2.0, 3.0, 4.0, 4.0), BlendMode::Replace);
        g.enable_hitmask();
        g.hitmask_mut().unwrap().fill(1);
        assert!(!g.hit(1.0, 5.0));   // left of bbox
        assert!(!g.hit(7.0, 5.0));   // right of bbox (exclusive)
        assert!(!g.hit(3.0, 2.0));   // above bbox
        assert!(!g.hit(3.0, 8.0));   // below bbox (exclusive)
    }

    #[test]
    fn mark_damage_coalesces_overlapping_rects() {
        let mut g = Group::new(Region::new(0.0, 0.0, 100.0, 100.0), BlendMode::Replace);
        g.mark_damage(Region::new(0.0, 0.0, 10.0, 10.0));
        g.mark_damage(Region::new(5.0, 5.0, 10.0, 10.0));
        assert_eq!(g.dirty_rects().len(), 1);
        let r = g.dirty_rects()[0];
        assert_eq!((r.x, r.y, r.w, r.h), (0.0, 0.0, 15.0, 15.0));
    }

    #[test]
    fn mark_damage_appends_disjoint_rects() {
        let mut g = Group::new(Region::new(0.0, 0.0, 100.0, 100.0), BlendMode::Replace);
        g.mark_damage(Region::new(0.0, 0.0, 5.0, 5.0));
        g.mark_damage(Region::new(50.0, 50.0, 5.0, 5.0));
        assert_eq!(g.dirty_rects().len(), 2);
    }

    #[test]
    fn mark_damage_drops_degenerate_rects() {
        let mut g = Group::new(Region::new(0.0, 0.0, 100.0, 100.0), BlendMode::Replace);
        g.mark_damage(Region::new(0.0, 0.0, 0.0, 5.0));
        g.mark_damage(Region::new(0.0, 0.0, 5.0, 0.0));
        assert!(g.dirty_rects().is_empty());
    }

    #[test]
    fn is_dirty_finds_intersection() {
        let mut g = Group::new(Region::new(0.0, 0.0, 100.0, 100.0), BlendMode::Replace);
        g.mark_damage(Region::new(10.0, 10.0, 20.0, 20.0));
        assert!(g.is_dirty(&Region::new(15.0, 15.0, 5.0, 5.0)));
        assert!(g.is_dirty(&Region::new(25.0, 25.0, 20.0, 20.0)));
        assert!(!g.is_dirty(&Region::new(50.0, 50.0, 5.0, 5.0)));
    }

    #[test]
    fn flatten_clears_damage() {
        let mut g = Group::new(Region::new(0.0, 0.0, 2.0, 1.0), BlendMode::Replace);
        let l = g.add_layer();
        g.rpn.layers[l].pixels = alloc::vec![opaque(0x112233); 2];
        g.set_program(alloc::vec![Op::Push(l)]);
        g.mark_damage(Region::new(0.0, 0.0, 1.0, 1.0));
        let mut target = alloc::vec![0u32; 2];
        g.flatten_into(&mut target, 2, 1);
        assert!(g.dirty_rects().is_empty());
    }
}
