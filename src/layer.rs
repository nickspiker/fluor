//! Layer stack — cached rasterization layers with per-layer blend modes.
//!
//! Each `Layer` owns a pixel buffer and a [`BlendMode`]. The `LayerStack` flattens them bottom-up into a composite, caching intermediate results so only dirty layers trigger recomputation.
//!
//! Consumers push layers in bottom-to-top order, rasterize into individual layer buffers, mark dirty flags, and call [`flatten`](LayerStack::flatten) to produce the final composite.

use alloc::vec::Vec;
use crate::paint::BlendMode;

/// A single compositing layer — an ARGB pixel buffer with a blend mode and dirty flag.
pub struct Layer {
    /// ARGB pixel buffer, same dimensions as the layer stack's viewport.
    pub pixels: Vec<u32>,
    /// How this layer composites onto the layers below it.
    pub blend: BlendMode,
    /// If true, the layer's content needs re-rasterization before the next flatten.
    pub dirty: bool,
}

impl Layer {
    /// Create a new layer with the given blend mode, sized to `width * height` pixels.
    pub fn new(blend: BlendMode, width: usize, height: usize) -> Self {
        Self {
            pixels: alloc::vec![0u32; width * height],
            blend,
            dirty: true,
        }
    }

    /// Clear the pixel buffer to fully transparent black.
    pub fn clear(&mut self) {
        self.pixels.fill(0);
    }

    /// Resize the pixel buffer. Marks dirty. Does not preserve content.
    pub fn resize(&mut self, width: usize, height: usize) {
        let needed = width * height;
        self.pixels.resize(needed, 0);
        self.pixels.fill(0);
        self.dirty = true;
    }
}

/// Ordered stack of compositing layers with cached intermediate flattening.
///
/// `caches[i]` holds the composite of layers `0..=i`. When layer `N` is the lowest dirty layer, flatten recomputes from `caches[N-1]` forward instead of from scratch. The final composite (`caches[last]`) is the output.
pub struct LayerStack {
    layers: Vec<Layer>,
    /// `caches[i]` = composite of layers `0..=i`. One extra buffer per layer.
    caches: Vec<Vec<u32>>,
    width: usize,
    height: usize,
}

impl LayerStack {
    /// Create an empty layer stack for a viewport of `width × height` pixels.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            layers: Vec::new(),
            caches: Vec::new(),
            width,
            height,
        }
    }

    /// Add a layer to the top of the stack with the given blend mode.
    /// Returns the layer index (0-based, bottom-to-top).
    pub fn push(&mut self, blend: BlendMode) -> usize {
        let idx = self.layers.len();
        self.layers.push(Layer::new(blend, self.width, self.height));
        self.caches.push(alloc::vec![0u32; self.width * self.height]);
        idx
    }

    /// Number of layers in the stack.
    pub fn len(&self) -> usize { self.layers.len() }
    pub fn is_empty(&self) -> bool { self.layers.is_empty() }

    /// Get a mutable reference to a layer's pixel buffer for rasterization.
    pub fn layer_mut(&mut self, index: usize) -> &mut Layer {
        &mut self.layers[index]
    }

    /// Get an immutable reference to a layer.
    pub fn layer(&self, index: usize) -> &Layer {
        &self.layers[index]
    }

    /// Mark a layer as dirty (needs re-rasterization).
    pub fn mark_dirty(&mut self, index: usize) {
        self.layers[index].dirty = true;
    }

    /// Mark all layers as dirty (e.g., on resize).
    pub fn mark_all_dirty(&mut self) {
        for layer in &mut self.layers {
            layer.dirty = true;
        }
    }

    /// Resize all layers and caches. Marks everything dirty.
    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        for layer in &mut self.layers {
            layer.resize(width, height);
        }
        let needed = width * height;
        for cache in &mut self.caches {
            cache.resize(needed, 0);
        }
    }

    /// Flatten the layer stack into the final composite, starting from the lowest dirty layer.
    /// Returns a reference to the final composite buffer (`caches[last]`).
    ///
    /// After flatten, all dirty flags are cleared.
    pub fn flatten(&mut self) -> &[u32] {
        if self.layers.is_empty() {
            return &[];
        }

        // Find the lowest dirty layer index.
        let lowest_dirty = self.layers.iter()
            .position(|l| l.dirty)
            .unwrap_or(self.layers.len());

        // If nothing is dirty, return the existing final composite.
        if lowest_dirty >= self.layers.len() {
            return &self.caches[self.layers.len() - 1];
        }

        // Start from the cache below the lowest dirty layer, or from the base layer.
        for i in lowest_dirty..self.layers.len() {
            if i == 0 {
                // Base layer: copy directly into cache[0].
                self.caches[0].copy_from_slice(&self.layers[0].pixels);
            } else {
                // Copy the previous cache as the starting point.
                let (left, right) = self.caches.split_at_mut(i);
                right[0].copy_from_slice(&left[i - 1]);
                // Apply this layer's blend mode.
                self.layers[i].blend.flatten(&mut right[0], &self.layers[i].pixels);
            }
            self.layers[i].dirty = false;
        }

        &self.caches[self.layers.len() - 1]
    }

    /// Dimensions.
    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_stack_flattens_to_empty() {
        let mut stack = LayerStack::new(4, 4);
        assert!(stack.flatten().is_empty());
    }

    #[test]
    fn single_replace_layer_copies_through() {
        let mut stack = LayerStack::new(4, 1);
        stack.push(BlendMode::Replace);
        stack.layer_mut(0).pixels = alloc::vec![0xFF112233; 4];
        stack.layer_mut(0).dirty = true;
        let result = stack.flatten();
        assert_eq!(result, &[0xFF112233; 4]);
    }

    #[test]
    fn additive_layer_adds() {
        let mut stack = LayerStack::new(2, 1);
        stack.push(BlendMode::Replace);
        stack.push(BlendMode::Add);
        stack.layer_mut(0).pixels = alloc::vec![0xFF100000; 2];
        stack.layer_mut(1).pixels = alloc::vec![0x00000010; 2];
        let result = stack.flatten();
        assert_eq!(result, &[0xFF100010; 2]);
    }

    #[test]
    fn xor_layer_flips_rgb() {
        let mut stack = LayerStack::new(1, 1);
        stack.push(BlendMode::Replace);
        stack.push(BlendMode::Xor);
        stack.layer_mut(0).pixels = alloc::vec![0xFF_FF_FF_FF];
        stack.layer_mut(1).pixels = alloc::vec![0x00_FF_00_FF]; // XOR R and B
        let result = stack.flatten();
        // 0xFF_FF_FF_FF ^ 0x00_FF_00_FF = 0xFF_00_FF_00 (alpha preserved from dst)
        assert_eq!(result, &[0xFF_00_FF_00]);
    }

    #[test]
    fn dirty_tracking_skips_clean_layers() {
        let mut stack = LayerStack::new(2, 1);
        stack.push(BlendMode::Replace);
        stack.push(BlendMode::Add);
        stack.layer_mut(0).pixels = alloc::vec![0xFF100000; 2];
        stack.layer_mut(1).pixels = alloc::vec![0x00000010; 2];
        // First flatten: both dirty.
        let _ = stack.flatten();

        // Modify only layer 1.
        stack.layer_mut(1).pixels = alloc::vec![0x00000020; 2];
        stack.layer_mut(1).dirty = true;
        // Layer 0 is clean — should use cached composite.
        let result = stack.flatten();
        assert_eq!(result, &[0xFF100020; 2]);
    }

    #[test]
    fn resize_marks_all_dirty() {
        let mut stack = LayerStack::new(2, 2);
        stack.push(BlendMode::Replace);
        let _ = stack.flatten();
        assert!(!stack.layer(0).dirty);
        stack.resize(4, 4);
        assert!(stack.layer(0).dirty);
        assert_eq!(stack.layer(0).pixels.len(), 16);
    }
}
