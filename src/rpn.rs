//! RPN compositing engine — layers as operands, blend ops as operators.
//!
//! The compositor is a reverse-polish-notation calculator. `Push` loads a layer's pixel
//! buffer onto the evaluation stack; operators pop their inputs and push results. The same
//! evaluator handles simple ordered stacks (`Push 0, Push 1, AlphaOver`) and complex
//! expressions (`Push 0, Push 1, Mul, Push 2, Add`).
//!
//! Dirty tracking: hot layers (blinkey, text) go at the end of the program. When a layer
//! changes, re-evaluation starts from the earliest dirty `Push` — the tail of the program.
//! A blink tick re-runs one `Add` pass. A keystroke re-runs two or three. Only a resize
//! re-evaluates the full program.

use alloc::vec::Vec;
use crate::pixel::Argb8;

/// RPN instruction — either a data push or a channel operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Push layer `idx` onto the evaluation stack.
    Push(usize),
    /// Push a solid-colour buffer (all pixels same value).
    Constant(u32),

    // --- Binary (pop 2, push 1: second-from-top OP top) ---
    /// Per-channel add: `a + b` (wrapping).
    Add,
    /// Per-channel subtract: `a - b` (wrapping).
    Sub,
    /// Per-channel multiply: `(a * b) >> 8`.
    Mul,
    /// Porter-Duff source-over: `src * α + dst * (1 - α)`.
    AlphaOver,
    /// Screen: `inv(mul(inv(a), inv(b)))`.
    Screen,
    /// XOR RGB, preserve alpha from `a`.
    Xor,

    // --- Unary (pop 1, push 1) ---
    /// Per-channel invert: `255 - x`.
    Inv,
}

/// A named pixel buffer that the consumer rasterizes into.
pub struct RpnLayer {
    pub pixels: Vec<u32>,
    pub dirty: bool,
}

impl RpnLayer {
    pub fn new(size: usize) -> Self {
        Self { pixels: alloc::vec![0u32; size], dirty: true }
    }

    pub fn clear(&mut self) {
        self.pixels.fill(0);
    }

    pub fn resize(&mut self, size: usize) {
        self.pixels.resize(size, 0);
        self.pixels.fill(0);
        self.dirty = true;
    }
}

/// RPN compositing engine.
pub struct RpnCompositor {
    /// Named pixel buffers, indexed by `Op::Push(idx)`.
    pub layers: Vec<RpnLayer>,
    /// The compositing program.
    pub program: Vec<Op>,
    /// Evaluation stack (reused across evaluations to avoid re-alloc).
    stack: Vec<Vec<u32>>,
    /// Pool of reusable temporary buffers (avoids alloc churn on repeated evaluations).
    pool: Vec<Vec<u32>>,
    /// Cached snapshot of the stack at each instruction boundary.
    /// `snapshots[i]` = state of top-of-stack after executing `program[i]`.
    /// Used for partial re-evaluation from the earliest dirty instruction.
    snapshots: Vec<Vec<u32>>,
    /// The final composite after the last `evaluate()`.
    composite: Vec<u32>,
    /// Pixels per layer.
    size: usize,
}

impl RpnCompositor {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            layers: Vec::new(),
            program: Vec::new(),
            stack: Vec::new(),
            pool: Vec::new(),
            snapshots: Vec::new(),
            composite: Vec::new(),
            size: width * height,
        }
    }

    /// Add a layer. Returns its index for use in `Op::Push`.
    pub fn add_layer(&mut self) -> usize {
        let idx = self.layers.len();
        self.layers.push(RpnLayer::new(self.size));
        idx
    }

    /// Set the compositing program.
    pub fn set_program(&mut self, program: Vec<Op>) {
        self.snapshots.clear();
        self.program = program;
    }

    /// Resize all layers and invalidate caches.
    pub fn resize(&mut self, width: usize, height: usize) {
        self.size = width * height;
        for layer in &mut self.layers {
            layer.resize(self.size);
        }
        self.snapshots.clear();
        self.composite.resize(self.size, 0);
    }

    /// Acquire a temporary buffer from the pool or allocate a new one.
    fn acquire_buf(&mut self) -> Vec<u32> {
        if let Some(mut buf) = self.pool.pop() {
            buf.resize(self.size, 0);
            buf.fill(0);
            buf
        } else {
            alloc::vec![0u32; self.size]
        }
    }

    /// Return a temporary buffer to the pool.
    fn release_buf(&mut self, buf: Vec<u32>) {
        self.pool.push(buf);
    }

    /// Evaluate the program. Returns a reference to the final composite.
    ///
    /// Finds the earliest instruction that references a dirty layer and re-evaluates
    /// from there. If snapshots exist for clean prefix instructions, restores the
    /// stack from the snapshot before the first dirty instruction.
    pub fn evaluate(&mut self) -> &[u32] {
        if self.program.is_empty() {
            return &[];
        }

        // Find the earliest program index that touches a dirty layer.
        let first_dirty_pc = self.program.iter().enumerate()
            .position(|(_, op)| match op {
                Op::Push(idx) => self.layers.get(*idx).map_or(false, |l| l.dirty),
                _ => false,
            })
            .unwrap_or(self.program.len());

        // If nothing is dirty and we have a cached composite, return it.
        if first_dirty_pc >= self.program.len() && !self.composite.is_empty() {
            return &self.composite;
        }

        // Determine where to start evaluation.
        // If we have a snapshot for the instruction before first_dirty_pc, restore from it.
        let start_pc = if first_dirty_pc > 0 && first_dirty_pc - 1 < self.snapshots.len() {
            // Restore stack: single item = the snapshot of top-of-stack at (first_dirty_pc - 1)
            let restored = self.snapshots[first_dirty_pc - 1].clone();
            // Return any existing stack buffers to pool
            while let Some(buf) = self.stack.pop() {
                self.release_buf(buf);
            }
            self.stack.push(restored);
            first_dirty_pc
        } else {
            // Full re-evaluation from the beginning
            while let Some(buf) = self.stack.pop() {
                self.release_buf(buf);
            }
            0
        };

        // Execute from start_pc to end.
        for pc in start_pc..self.program.len() {
            match self.program[pc] {
                Op::Push(idx) => {
                    let mut buf = self.acquire_buf();
                    buf.copy_from_slice(&self.layers[idx].pixels);
                    self.stack.push(buf);
                }
                Op::Constant(colour) => {
                    let mut buf = self.acquire_buf();
                    buf.fill(colour);
                    self.stack.push(buf);
                }
                Op::Add => self.binary_op(Argb8::add),
                Op::Sub => self.binary_op(Argb8::sub),
                Op::Mul => self.binary_op(Argb8::mul),
                Op::AlphaOver => self.binary_op(Argb8::alpha_over),
                Op::Screen => self.binary_op(Argb8::screen),
                Op::Xor => self.binary_op(Argb8::xor),
                Op::Inv => self.unary_op(Argb8::inv),
            }

            // Cache the top-of-stack as a snapshot for this instruction.
            if let Some(top) = self.stack.last() {
                if pc < self.snapshots.len() {
                    self.snapshots[pc].clear();
                    self.snapshots[pc].extend_from_slice(top);
                } else {
                    while self.snapshots.len() < pc {
                        self.snapshots.push(Vec::new());
                    }
                    self.snapshots.push(top.clone());
                }
            }
        }

        // Clear dirty flags.
        for layer in &mut self.layers {
            layer.dirty = false;
        }

        // Pop final result.
        if let Some(result) = self.stack.pop() {
            self.composite = result;
        }
        // Return remaining stack buffers to pool.
        while let Some(buf) = self.stack.pop() {
            self.release_buf(buf);
        }

        &self.composite
    }

    /// Apply a binary op: pop top two, apply f(second, top) in-place on second, push result.
    fn binary_op(&mut self, f: fn(Argb8, Argb8) -> Argb8) {
        let b = self.stack.pop().expect("RPN stack underflow on binary op");
        let a = self.stack.last_mut().expect("RPN stack underflow on binary op");
        for i in 0..a.len() {
            a[i] = f(Argb8(a[i]), Argb8(b[i])).0;
        }
        self.release_buf(b);
    }

    /// Apply a unary op: pop top, apply f(x) in-place, push result.
    fn unary_op(&mut self, f: fn(Argb8) -> Argb8) {
        let a = self.stack.last_mut().expect("RPN stack underflow on unary op");
        for i in 0..a.len() {
            a[i] = f(Argb8(a[i])).0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rpn(size: usize) -> RpnCompositor {
        RpnCompositor::new(size, 1)
    }

    #[test]
    fn single_push_returns_layer() {
        let mut rpn = make_rpn(4);
        let l0 = rpn.add_layer();
        rpn.layers[l0].pixels = alloc::vec![0xFF_11_22_33; 4];
        rpn.set_program(alloc::vec![Op::Push(0)]);
        let result = rpn.evaluate();
        assert_eq!(result, &[0xFF_11_22_33; 4]);
    }

    #[test]
    fn add_two_layers() {
        let mut rpn = make_rpn(2);
        let l0 = rpn.add_layer();
        let l1 = rpn.add_layer();
        rpn.layers[l0].pixels = alloc::vec![0xFF_10_00_00; 2];
        rpn.layers[l1].pixels = alloc::vec![0x00_00_00_10; 2];
        rpn.set_program(alloc::vec![Op::Push(l0), Op::Push(l1), Op::Add]);
        let result = rpn.evaluate();
        assert_eq!(result, &[0xFF_10_00_10; 2]);
    }

    #[test]
    fn mul_expression() {
        // (a + b) * c via RPN: Push a, Push b, Add, Push c, Mul
        let mut rpn = make_rpn(1);
        let a = rpn.add_layer();
        let b = rpn.add_layer();
        let c = rpn.add_layer();
        rpn.layers[a].pixels = alloc::vec![0xFF_40_00_00];
        rpn.layers[b].pixels = alloc::vec![0x00_40_00_00];
        rpn.layers[c].pixels = alloc::vec![0xFF_80_80_80]; // multiply by ~0.5
        rpn.set_program(alloc::vec![Op::Push(a), Op::Push(b), Op::Add, Op::Push(c), Op::Mul]);
        let result = rpn.evaluate();
        // (0x40 + 0x40) * 0x80 >> 8 = 0x80 * 0x80 >> 8 = 0x40
        let r = (result[0] >> 16) & 0xFF;
        assert!(r >= 0x3F && r <= 0x41, "expected ~0x40, got {:#x}", r);
    }

    #[test]
    fn inv_then_mul_is_screen() {
        // screen(a, b) = inv(mul(inv(a), inv(b)))
        // Verify our Screen op matches this identity.
        let mut rpn = make_rpn(1);
        let a = rpn.add_layer();
        let b = rpn.add_layer();
        rpn.layers[a].pixels = alloc::vec![0xFF_80_40_C0];
        rpn.layers[b].pixels = alloc::vec![0xFF_40_80_20];

        // Manual: inv(a), inv(b), mul, inv
        rpn.set_program(alloc::vec![
            Op::Push(a), Op::Inv, Op::Push(b), Op::Inv, Op::Mul, Op::Inv,
        ]);
        let manual = rpn.evaluate()[0];

        // Via Screen op
        rpn.set_program(alloc::vec![Op::Push(a), Op::Push(b), Op::Screen]);
        // Force full re-eval
        for l in &mut rpn.layers { l.dirty = true; }
        rpn.snapshots.clear();
        let via_op = rpn.evaluate()[0];

        assert_eq!(manual, via_op);
    }

    #[test]
    fn dirty_tracking_partial_reeval() {
        let mut rpn = make_rpn(2);
        let l0 = rpn.add_layer();
        let l1 = rpn.add_layer();
        rpn.layers[l0].pixels = alloc::vec![0xFF_10_00_00; 2];
        rpn.layers[l1].pixels = alloc::vec![0x00_00_00_10; 2];
        rpn.set_program(alloc::vec![Op::Push(l0), Op::Push(l1), Op::Add]);

        // First eval: both dirty.
        let _ = rpn.evaluate();

        // Modify only layer 1.
        rpn.layers[l1].pixels = alloc::vec![0x00_00_00_20; 2];
        rpn.layers[l1].dirty = true;
        let result = rpn.evaluate();
        assert_eq!(result, &[0xFF_10_00_20; 2]);
    }

    #[test]
    fn constant_op() {
        let mut rpn = make_rpn(3);
        let l0 = rpn.add_layer();
        rpn.layers[l0].pixels = alloc::vec![0xFF_00_00_00; 3];
        rpn.set_program(alloc::vec![Op::Push(l0), Op::Constant(0x00_10_10_10), Op::Add]);
        let result = rpn.evaluate();
        assert_eq!(result, &[0xFF_10_10_10; 3]);
    }

    #[test]
    fn xor_in_program() {
        let mut rpn = make_rpn(1);
        let l0 = rpn.add_layer();
        let l1 = rpn.add_layer();
        rpn.layers[l0].pixels = alloc::vec![0xFF_FF_FF_FF];
        rpn.layers[l1].pixels = alloc::vec![0x00_FF_00_FF];
        rpn.set_program(alloc::vec![Op::Push(l0), Op::Push(l1), Op::Xor]);
        let result = rpn.evaluate();
        assert_eq!(result, &[0xFF_00_FF_00]);
    }
}
