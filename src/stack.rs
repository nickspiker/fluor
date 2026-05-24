//! Stack Notation compositing engine — layers as operands, blend ops as operators.
//!
//! "Stack Notation" describes the mechanism directly: data flows through a stack in execution order, operators are stack transformations. No precedence tables, no ambiguity. (Often called "RPN" in calculator history; that name carries decades of irrelevant baggage and the geographic branding of its inventor — neither describes what it actually does. Contrast with "Infix Notation" — the broken sibling whose arbitrary operator placement creates all the precedence/associativity problems Stack Notation sidesteps.)
//!
//! `Push` loads a layer's pixel buffer onto the evaluation stack; `Under(mode)` pops two operands (the second-from-top is the partial composite from layers above, the top is the new layer going behind) and folds them via [`crate::pixel::Blend::under`] with the chosen [`BlendMode`]. The same evaluator handles simple ordered stacks (`Push top, Push next_behind, Under(Normal)`) and complex expressions (`Push 0, Push 1, Under(Multiply), Push 2, Under(Add)`).
//!
//! Re-execution policy: a per-layer `dirty` flag gates expensive rasterization upstream of the Stack (text shaping, glyph rendering, squircle math). The Stack itself caches the *final* composite from the last `evaluate()`; if no layer is dirty, that cache returns directly without re-running the program. If anything is dirty, the entire program re-runs — no per-instruction snapshot cache. Under-blend's per-pixel early-out (`dst < 0x01000000`) makes the per-pixel work cheap, and the typical program is short enough (3-5 ops) that running it whole is faster than maintaining intermediate state.

use crate::pixel::{Argb8, Blend, BlendMode};
use alloc::vec::Vec;

/// Stack Notation instruction — pushes a buffer onto the eval stack, or folds two buffers via `under`-with-mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Push layer `idx` onto the evaluation stack.
    Push(usize),
    /// Push a solid-colour buffer (all pixels same value).
    Constant(Argb8),
    /// Pop two operands; the second-from-top is `top` (partial composite from above), the top-of-stack is `bottom` (new layer going behind). Apply `top.under(bottom, mode)` per pixel and push the result. The single compositing primitive — every blend mode flows through here.
    Under(BlendMode),
}

/// A named pixel buffer that the consumer rasterizes into.
pub struct StackLayer {
    pub pixels: Vec<Argb8>,
    pub dirty: bool,
}

impl StackLayer {
    pub fn new(size: usize) -> Self {
        Self {
            pixels: alloc::vec![0xFFFFFFFF; size],
            dirty: true,
        }
    }

    /// Reset every pixel to the canonical empty value: `0xFFFFFFFF` — `t = 255` (fully transparent), RGB = 255 (white). White is invisible at full transparency (the OS never displays a transparent pixel's RGB), and starting RGB at 255 compensates the `>>8` truncation at the transparent endpoint so opaque paints into an empty buffer land exact instead of 1-2 LSB low. Pattern is byte-uniform (`0xFF`) so the fill compiles to a single `memset`.
    pub fn clear(&mut self) {
        self.pixels.fill(0xFFFFFFFF);
    }

    pub fn resize(&mut self, size: usize) {
        self.pixels.resize(size, 0xFFFFFFFF);
        self.pixels.fill(0xFFFFFFFF);
        self.dirty = true;
    }
}

/// Stack compositing engine.
pub struct StackCompositor {
    /// Named pixel buffers, indexed by `Op::Push(idx)`.
    pub layers: Vec<StackLayer>,
    /// The compositing program.
    pub program: Vec<Op>,
    /// Evaluation stack (reused across evaluations to avoid re-alloc).
    stack: Vec<Vec<Argb8>>,
    /// Pool of reusable temporary buffers (avoids alloc churn on repeated evaluations).
    pool: Vec<Vec<Argb8>>,
    /// The final composite from the last `evaluate()`. Returned directly when no layer is dirty.
    composite: Vec<Argb8>,
    /// Pixels per layer.
    size: usize,
}

impl StackCompositor {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            layers: Vec::new(),
            program: Vec::new(),
            stack: Vec::new(),
            pool: Vec::new(),
            composite: Vec::new(),
            size: width * height,
        }
    }

    /// Add a layer. Returns its index for use in `Op::Push`.
    pub fn new_layer(&mut self) -> usize {
        let idx = self.layers.len();
        self.layers.push(StackLayer::new(self.size));
        idx
    }

    /// Set the compositing program. Clears the composite cache (next `evaluate()` re-runs the program).
    pub fn set_program(&mut self, program: Vec<Op>) {
        self.composite.clear();
        self.program = program;
    }

    /// Read-only access to the cached composite from the last `evaluate()`. Returns an empty slice if `evaluate()` has never run on a non-empty program.
    #[inline]
    pub fn composite(&self) -> &[Argb8] {
        &self.composite
    }

    /// Resize all layers and invalidate caches.
    pub fn resize(&mut self, width: usize, height: usize) {
        self.size = width * height;
        for layer in &mut self.layers {
            layer.resize(self.size);
        }
        self.composite.clear();
    }

    /// Acquire a temporary buffer from the pool or allocate a new one. New buffers start fully-transparent.
    fn acquire_buf(&mut self) -> Vec<Argb8> {
        if let Some(mut buf) = self.pool.pop() {
            buf.resize(self.size, 0xFFFFFFFF);
            buf.fill(0xFFFFFFFF);
            buf
        } else {
            alloc::vec![0xFFFFFFFF; self.size]
        }
    }

    /// Return a temporary buffer to the pool.
    fn release_buf(&mut self, buf: Vec<Argb8>) {
        self.pool.push(buf);
    }

    /// Evaluate the program. Returns a reference to the final composite.
    ///
    /// If no layer is dirty AND a cached composite exists, returns the cache directly. Otherwise runs the entire program: walks each `Op` in order, allocating temporary buffers from the pool. The per-pixel work inside `Op::Under` is dominated by `Blend::under`'s `dst < 0x01000000` early-out, so re-running the whole program is cheap even when only one layer changed — no per-instruction snapshot machinery needed.
    pub fn evaluate(&mut self) -> &[Argb8] {
        if self.program.is_empty() {
            return &[];
        }

        let any_dirty = self.program.iter().any(|op| match op {
            Op::Push(idx) => self.layers.get(*idx).map_or(false, |l| l.dirty),
            _ => false,
        });

        if !any_dirty && !self.composite.is_empty() {
            return &self.composite;
        }

        // Drain any leftover stack buffers from a previous interrupted run.
        while let Some(buf) = self.stack.pop() {
            self.release_buf(buf);
        }

        for pc in 0..self.program.len() {
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
                Op::Under(mode) => {
                    let b = self.stack.pop().expect("Stack underflow on Under");
                    let a = self.stack.last_mut().expect("Stack underflow on Under");
                    for i in 0..a.len() {
                        a[i] = a[i].under(b[i], mode);
                    }
                    self.release_buf(b);
                }
            }
        }

        for layer in &mut self.layers {
            layer.dirty = false;
        }

        if let Some(result) = self.stack.pop() {
            // Recycle the prior cached composite buffer into the pool before adopting the new one.
            let old = core::mem::replace(&mut self.composite, result);
            if !old.is_empty() {
                self.release_buf(old);
            }
        }
        while let Some(buf) = self.stack.pop() {
            self.release_buf(buf);
        }

        &self.composite
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stack(size: usize) -> StackCompositor {
        StackCompositor::new(size, 1)
    }

    #[test]
    fn single_push_returns_layer() {
        let mut stk = make_stack(4);
        let l0 = stk.new_layer();
        stk.layers[l0].pixels = alloc::vec![0x00_11_22_33; 4];
        stk.set_program(alloc::vec![Op::Push(l0)]);
        let result = stk.evaluate();
        assert_eq!(result, &[0x00_11_22_33; 4]);
    }

    #[test]
    fn under_normal_opaque_top_returns_top() {
        // Topmost (opaque red) above bottom (opaque blue) — top wins via early-out.
        let mut stk = make_stack(2);
        let top = stk.new_layer();
        let bot = stk.new_layer();
        stk.layers[top].pixels = alloc::vec![0x00_FF_00_00; 2];
        stk.layers[bot].pixels = alloc::vec![0x00_00_00_FF; 2];
        stk.set_program(alloc::vec![
            Op::Push(top),
            Op::Push(bot),
            Op::Under(BlendMode::Normal)
        ]);
        let result = stk.evaluate();
        assert_eq!(result, &[0x00_FF_00_00; 2]);
    }

    #[test]
    fn under_normal_translucent_top_attenuates_budget() {
        let mut stk = make_stack(1);
        let top = stk.new_layer();
        let bot = stk.new_layer();
        stk.layers[top].pixels = alloc::vec![0x80_00_00_00];
        stk.layers[bot].pixels = alloc::vec![0x80_FF_00_00];
        stk.set_program(alloc::vec![
            Op::Push(top),
            Op::Push(bot),
            Op::Under(BlendMode::Normal)
        ]);
        let result = stk.evaluate()[0];
        assert_eq!(result >> 24, 64, "new_t should be 64 (128*128 >> 8)");
    }

    #[test]
    fn constant_op_fills() {
        let mut stk = make_stack(3);
        stk.set_program(alloc::vec![Op::Constant(0x00_10_10_10)]);
        let result = stk.evaluate();
        assert_eq!(result, &[0x00_10_10_10; 3]);
    }

    #[test]
    fn dirty_layer_triggers_reevaluate() {
        let mut stk = make_stack(2);
        let top = stk.new_layer();
        let bot = stk.new_layer();
        stk.layers[top].pixels = alloc::vec![0xFFFFFFFF; 2]; // canonical empty top
        stk.layers[bot].pixels = alloc::vec![0x00_00_FF_00; 2]; // opaque green bottom
        stk.set_program(alloc::vec![
            Op::Push(top),
            Op::Push(bot),
            Op::Under(BlendMode::Normal)
        ]);
        let _ = stk.evaluate();
        // Modify only the bottom layer and mark it dirty. The whole program re-runs (no partial reeval) and produces a result ≈ new bottom (within 1-LSB drift from >>8 endpoint).
        stk.layers[bot].pixels = alloc::vec![0x00_00_00_FF; 2];
        stk.layers[bot].dirty = true;
        let result = stk.evaluate()[0];
        assert!(result & 0xFF >= 0xFE, "result blue channel = {:#x}", result & 0xFF);
    }

    #[test]
    fn clean_evaluation_returns_cached_composite() {
        let mut stk = make_stack(4);
        let l0 = stk.new_layer();
        stk.layers[l0].pixels = alloc::vec![0x00_11_22_33; 4];
        stk.set_program(alloc::vec![Op::Push(l0)]);
        let first = stk.evaluate().to_vec();
        // No dirty layer — second evaluate hits the cached composite, same bytes.
        let second = stk.evaluate();
        assert_eq!(&first[..], second);
    }
}
