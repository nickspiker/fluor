//! The 8-bit-only text blits: the legacy byte-buffer draws (photon's logo and render pass write visible RGB bytes) and the additive / subtractive wrapping blits (reversible tints on a u32 buffer).
//! Both speak 8-bit visible bytes by design, so neither is generic over [`crate::pixel::Pixel`]; the α + darkness path every depth shares is `render_buffer_u32` in the parent module.
//! They lived in text.rs until the rendering core went generic; they are still methods of [`TextRenderer`], so callers are unchanged.

use super::*;

impl TextRenderer {
    pub fn draw_text_center_legacy(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: Vec<u8>,
        rotation: u16,
        font: &str, // "Josefin Slab" for UI, "Open Sans" for user content
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            // Calculate text width
            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;

            for glyph in run.glyphs {
                min_x = min_x.min(glyph.x);
                max_x = max_x.max(glyph.x + glyph.w);
            }

            let text_width = max_x - min_x;
            let text_height = run.line_height;

            self.render_buffer(
                &mut buffer,
                pixels,
                width,
                height,
                x,
                y,
                text_width,
                text_height,
                colour,
                rotation,
                0, // center alignment
            );

            text_width
        } else {
            0.0
        }
    }

    pub fn draw_text_left_legacy(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: Vec<u8>,
        rotation: u16,
        font: &str, // "Josefin Slab" for UI, "Open Sans" for user content
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;

            self.render_buffer(
                &mut buffer,
                pixels,
                width,
                height,
                x,
                y,
                text_width,
                text_height,
                colour,
                rotation,
                1, // left alignment
            );

            text_width
        } else {
            0.0
        }
    }

    pub fn draw_text_right_legacy(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: Vec<u8>,
        rotation: u16,
        font: &str, // "Josefin Slab" for UI, "Open Sans" for user content
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;

            self.render_buffer(
                &mut buffer,
                pixels,
                width,
                height,
                x,
                y,
                text_width,
                text_height,
                colour,
                rotation,
                2, // right alignment
            );

            text_width
        } else {
            0.0
        }
    }


    fn render_buffer(
        &mut self,
        buffer: &mut Buffer,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        anchor_x: f32,
        anchor_y: f32,
        text_width: f32,
        text_height: f32,
        colour: Vec<u8>,
        rotation: u16,
        alignment: u8, // 0=center, 1=left, 2=right
    ) {
        let channels = colour.len();

        // Calculate the offset based on alignment
        let (offset_x, offset_y) = match alignment {
            0 => (anchor_x - text_width / 2.0, anchor_y - text_height / 2.0), // center
            1 => (anchor_x, anchor_y - text_height / 2.0),                    // left
            2 => (anchor_x - text_width, anchor_y - text_height / 2.0),       // right
            _ => (anchor_x, anchor_y),
        };

        for run in buffer.layout_runs() {
            let baseline_offset = run.line_y;

            for glyph in run.glyphs {
                let physical_glyph = glyph.physical((offset_x, offset_y), 1.);

                if let Some(image) = self
                    .swash_cache
                    .get_image(&mut self.font_system, physical_glyph.cache_key)
                {
                    let glyph_x = physical_glyph.x + image.placement.left;
                    let glyph_y = physical_glyph.y + baseline_offset as i32 - image.placement.top;

                    // Draw the glyph bitmap
                    let glyph_width = image.placement.width as usize;
                    let glyph_height = image.placement.height as usize;
                    // This LEGACY path composites in VISIBLE-RGB byte space (raw channel bytes over-blended with the visible tint `colour[c]`), NOT the u32 α+darkness domain of `render_buffer_u32`. So the Color branch here writes the source's visible sRGB channels DIRECTLY (no darkness complement) — the buffer isn't XOR-flipped at finalize. Colour glyph = `Content::Color` RGBA (4 bytes/pixel); Mask glyph = 1-byte coverage.
                    let is_colour = image.content == swash::scale::image::Content::Color;

                    for cy in 0..glyph_height {
                        for cx in 0..glyph_width {
                            // Source α: coverage byte for a Mask, the RGBA α for a colour glyph.
                            let alpha = if is_colour {
                                image.data[(cy * glyph_width + cx) * 4 + 3]
                            } else {
                                image.data[cy * glyph_width + cx]
                            };
                            if alpha > 0 {
                                let py_base = glyph_y + cy as i32;
                                let px_base = glyph_x + cx as i32;

                                // Calculate position relative to anchor point
                                let rel_x = px_base as f32 - anchor_x;
                                let rel_y = py_base as f32 - anchor_y;

                                // Rotate around the anchor point
                                let (rot_x, rot_y) = match rotation {
                                    90 => (rel_y, -rel_x),
                                    180 => (-rel_x, -rel_y),
                                    270 => (-rel_y, rel_x),
                                    _ => (rel_x, rel_y),
                                };

                                // Convert back to absolute coordinates
                                let final_x = (anchor_x + rot_x) as i32;
                                let final_y = (anchor_y + rot_y) as i32;
                                // Clip to the buffer: a glyph can land partly (or wholly) off-canvas — e.g. a mid-rotation sliver resize (observed 1440×202 on Android) put final_y negative, and the `as usize` cast wrapped it to ~2^64, indexing OOB (samsung panic, 2026-07-15). Skip the off-canvas pixels instead.
                                // WHY/PROOF: final_x and final_y come from glyph positioning, negative or past the buffer whenever text runs off an edge — the cast to usize would wrap and the write would land outside the pixel buffer.
                                if final_x < 0
                                    || final_y < 0
                                    || final_x >= width as i32
                                    || final_y >= height as i32
                                {
                                    continue;
                                }
                                let offset = (final_y as usize * width as usize + final_x as usize)
                                    * channels;

                                let alpha_u16 = alpha as u16;
                                let inv_alpha = 256 - alpha_u16;

                                if is_colour {
                                    // Composite the glyph's OWN visible sRGB, ignoring the style `colour`. Map source r,g,b onto the first three channels; any 4th (alpha) channel keeps the mask-path behaviour of tinting toward `colour[3]`.
                                    let base = (cy * glyph_width + cx) * 4;
                                    for c in 0..channels {
                                        let src = if c < 3 {
                                            image.data[base + c] as u16
                                        } else {
                                            colour[c] as u16
                                        };
                                        pixels[offset + c] = ((pixels[offset + c] as u16 * inv_alpha
                                            + src * alpha_u16)
                                            >> 8)
                                            as u8;
                                    }
                                } else {
                                    for c in 0..channels {
                                        pixels[offset + c] = ((pixels[offset + c] as u16 * inv_alpha
                                            + colour[c] as u16 * alpha_u16)
                                            >> 8)
                                            as u8;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Draw left-aligned text with additive/subtractive compositing (u32 ARGB version) Uses wrapping add/sub so it's reversible - subtract same colour to remove text add_mode: true = add colour, false = subtract colour
    pub fn draw_text_left_additive_u32(
        &mut self,
        pixels: &mut [u32],
        width: usize,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: u32,
        font: &str,
        add_mode: bool,
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;

            self.render_buffer_left_additive_u32(
                &mut buffer,
                pixels,
                width,
                x,
                y,
                text_height,
                add_mode,
                colour,
            );

            text_width
        } else {
            0.
        }
    }

    /// Draw center-aligned text with additive/subtractive compositing (u32 ARGB version) Uses wrapping add/sub so it's reversible - subtract same colour to remove text add_mode: true = add colour, false = subtract colour
    pub fn draw_text_center_additive_u32(
        &mut self,
        pixels: &mut [u32],
        width: usize,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: u32,
        font: &str,
        add_mode: bool,
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;

            self.render_buffer_center_additive_u32(
                &mut buffer,
                pixels,
                width,
                x,
                y,
                text_height,
                add_mode,
                colour,
            );

            text_width
        } else {
            0.
        }
    }

    /// Draw right-aligned text with additive/subtractive compositing (u32 ARGB version) Uses wrapping add/sub so it's reversible - subtract same colour to remove text add_mode: true = add colour, false = subtract colour
    pub fn draw_text_right_additive_u32(
        &mut self,
        pixels: &mut [u32],
        width: usize,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: u32,
        font: &str,
        add_mode: bool,
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(None, None);
        let cjk = self.note_packs(text);
        set_text_vs(&mut buffer, text, &attrs, cjk);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;

            self.render_buffer_right_additive_u32(
                &mut buffer,
                pixels,
                width,
                x,
                y,
                text_height,
                add_mode,
                colour,
            );

            text_width
        } else {
            0.
        }
    }

    /// Render buffer with additive/subtractive compositing (u32 version)
    fn render_buffer_left_additive_u32(
        &mut self,
        buffer: &mut Buffer,
        pixels: &mut [u32],
        width: usize,
        anchor_x: f32,
        anchor_y: f32,
        text_height: f32,
        add_mode: bool,
        colour: u32,
    ) {
        let offset_x = anchor_x;
        let offset_y = anchor_y - text_height / 2.;

        for run in buffer.layout_runs() {
            let baseline_offset = run.line_y;

            for glyph in run.glyphs {
                let physical_glyph = glyph.physical((offset_x, offset_y), 1.);

                if let Some(image) = self
                    .swash_cache
                    .get_image(&mut self.font_system, physical_glyph.cache_key)
                {
                    let glyph_x = physical_glyph.x + image.placement.left;
                    let glyph_y = physical_glyph.y + baseline_offset as i32 - image.placement.top;

                    let glyph_width = image.placement.width as usize;
                    let glyph_height = image.placement.height as usize;

                    for cy in 0..glyph_height {
                        for cx in 0..glyph_width {
                            let alpha = image.data[cy * glyph_width + cx];
                            if alpha > 0 {
                                let final_x = glyph_x as isize + cx as isize;
                                let final_y = glyph_y as isize + cy as isize;

                                // WHY/PROOF: final_x and final_y come from glyph positioning, negative or past the buffer whenever text runs off an edge — the cast to usize would wrap and the write would land outside the pixel buffer.
                                if final_x < 0 || final_y < 0 || final_x >= width as isize {
                                    continue;
                                }
                                let idx = final_y as usize * width + final_x as usize;
                                if idx >= pixels.len() {
                                    continue;
                                }

                                // Scale colour by alpha, per-channel
                                let a = alpha as u32;
                                let r = ((colour >> 16) & 0xFF) * a >> 8;
                                let g = ((colour >> 8) & 0xFF) * a >> 8;
                                let b = (colour & 0xFF) * a >> 8;
                                let contribution = (r << 16) | (g << 8) | b;

                                if add_mode {
                                    pixels[idx] = pixels[idx].wrapping_add(contribution);
                                } else {
                                    pixels[idx] = pixels[idx].wrapping_sub(contribution);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Render buffer with additive/subtractive compositing, center-aligned (u32 version)
    fn render_buffer_center_additive_u32(
        &mut self,
        buffer: &mut Buffer,
        pixels: &mut [u32],
        width: usize,
        anchor_x: f32,
        anchor_y: f32,
        text_height: f32,
        add_mode: bool,
        colour: u32,
    ) {
        // Calculate text width for centering
        let text_width: f32 = buffer.layout_runs().fold(0.0, |max_width, run| {
            let run_width = run.glyphs.iter().map(|g| g.w).sum::<f32>();
            max_width.max(run_width)
        });

        let offset_x = anchor_x - text_width / 2.0;
        let offset_y = anchor_y - text_height / 2.;

        for run in buffer.layout_runs() {
            let baseline_offset = run.line_y;

            for glyph in run.glyphs {
                let physical_glyph = glyph.physical((offset_x, offset_y), 1.);

                if let Some(image) = self
                    .swash_cache
                    .get_image(&mut self.font_system, physical_glyph.cache_key)
                {
                    let glyph_x = physical_glyph.x + image.placement.left;
                    let glyph_y = physical_glyph.y + baseline_offset as i32 - image.placement.top;

                    let glyph_width = image.placement.width as usize;
                    let glyph_height = image.placement.height as usize;

                    for cy in 0..glyph_height {
                        for cx in 0..glyph_width {
                            let alpha = image.data[cy * glyph_width + cx];
                            if alpha > 0 {
                                let final_x = glyph_x as isize + cx as isize;
                                let final_y = glyph_y as isize + cy as isize;

                                // WHY/PROOF: final_x and final_y come from glyph positioning, negative or past the buffer whenever text runs off an edge — the cast to usize would wrap and the write would land outside the pixel buffer.
                                if final_x < 0 || final_y < 0 || final_x >= width as isize {
                                    continue;
                                }
                                let idx = final_y as usize * width + final_x as usize;
                                if idx >= pixels.len() {
                                    continue;
                                }

                                let a = alpha as u32;
                                let r = ((colour >> 16) & 0xFF) * a >> 8;
                                let g = ((colour >> 8) & 0xFF) * a >> 8;
                                let b = (colour & 0xFF) * a >> 8;
                                let contribution = (r << 16) | (g << 8) | b;

                                if add_mode {
                                    pixels[idx] = pixels[idx].wrapping_add(contribution);
                                } else {
                                    pixels[idx] = pixels[idx].wrapping_sub(contribution);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Render buffer with additive/subtractive compositing, right-aligned (u32 version)
    fn render_buffer_right_additive_u32(
        &mut self,
        buffer: &mut Buffer,
        pixels: &mut [u32],
        width: usize,
        anchor_x: f32,
        anchor_y: f32,
        text_height: f32,
        add_mode: bool,
        colour: u32,
    ) {
        // Calculate text width for right-alignment
        let text_width: f32 = buffer.layout_runs().fold(0.0, |max_width, run| {
            let run_width = run.glyphs.iter().map(|g| g.w).sum::<f32>();
            max_width.max(run_width)
        });

        let offset_x = anchor_x - text_width;
        let offset_y = anchor_y - text_height / 2.;

        for run in buffer.layout_runs() {
            let baseline_offset = run.line_y;

            for glyph in run.glyphs {
                let physical_glyph = glyph.physical((offset_x, offset_y), 1.);

                if let Some(image) = self
                    .swash_cache
                    .get_image(&mut self.font_system, physical_glyph.cache_key)
                {
                    let glyph_x = physical_glyph.x + image.placement.left;
                    let glyph_y = physical_glyph.y + baseline_offset as i32 - image.placement.top;

                    let glyph_width = image.placement.width as usize;
                    let glyph_height = image.placement.height as usize;

                    for cy in 0..glyph_height {
                        for cx in 0..glyph_width {
                            let alpha = image.data[cy * glyph_width + cx];
                            if alpha > 0 {
                                let final_x = glyph_x as isize + cx as isize;
                                let final_y = glyph_y as isize + cy as isize;

                                // WHY/PROOF: final_x and final_y come from glyph positioning, negative or past the buffer whenever text runs off an edge — the cast to usize would wrap and the write would land outside the pixel buffer.
                                if final_x < 0 || final_y < 0 || final_x >= width as isize {
                                    continue;
                                }
                                let idx = final_y as usize * width + final_x as usize;
                                if idx >= pixels.len() {
                                    continue;
                                }

                                let a = alpha as u32;
                                let r = ((colour >> 16) & 0xFF) * a >> 8;
                                let g = ((colour >> 8) & 0xFF) * a >> 8;
                                let b = (colour & 0xFF) * a >> 8;
                                let contribution = (r << 16) | (g << 8) | b;

                                if add_mode {
                                    pixels[idx] = pixels[idx].wrapping_add(contribution);
                                } else {
                                    pixels[idx] = pixels[idx].wrapping_sub(contribution);
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    /// Render a single character with additive blending (reversible with wrapping_add/sub) Returns the width of the rendered character in pixels
    pub fn render_char_additive(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        ch: char,
        x_offset: f32, // Absolute x position
        y_center: f32, // Vertical center position
        size: f32,
        weight: u16,
        font: &str,
        colour: [u8; 4],
        textbox_mask: &[u8], // Single-channel mask for textbox boundaries
        textbox_x: usize,    // Textbox top-left x
        textbox_y: usize,    // Textbox top-left y
        textbox_width: usize,
        add_mode: bool, // true = add, false = subtract
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size));
        buffer.set_size(Some(size * 2.0), Some(size * 2.0));

        let text = ch.to_string();
        buffer.set_text(&text, &attrs, Shaping::Advanced, None);
        // cosmic-text 0.19 shapes on demand: lay the text out before reading its runs.
        buffer.shape_until_scroll(&mut self.font_system, false);

        let mut char_width: f32 = 0.0;

        // Render the character glyph
        for run in buffer.layout_runs() {
            char_width = char_width.max(run.line_w);
            let baseline_offset = run.line_y;

            for glyph in run.glyphs {
                let offset_y = y_center - size / 2.0;
                let physical_glyph = glyph.physical((x_offset, offset_y), 1.);

                if let Some(image) = self
                    .swash_cache
                    .get_image(&mut self.font_system, physical_glyph.cache_key)
                {
                    let glyph_x = physical_glyph.x + image.placement.left;
                    let glyph_y = physical_glyph.y + baseline_offset as i32 - image.placement.top;

                    let glyph_width = image.placement.width as usize;
                    let glyph_height = image.placement.height as usize;

                    for cy in 0..glyph_height {
                        for cx in 0..glyph_width {
                            let glyph_alpha = image.data[cy * glyph_width + cx];
                            if glyph_alpha > 0 {
                                let final_x = glyph_x + cx as i32;
                                let final_y = glyph_y + cy as i32;

                                // Check bounds
                                if final_x >= 0
                                    && (final_x as u32) < width
                                    && final_y >= 0
                                    && (final_y as u32) < height
                                {
                                    // Get textbox mask value for this pixel
                                    let rel_x = final_x as isize - textbox_x as isize;
                                    let rel_y = final_y as isize - textbox_y as isize;

                                    let mask_alpha = if rel_x >= 0
                                        && rel_x < textbox_width as isize
                                        && rel_y >= 0
                                        && rel_y
                                            < textbox_mask.len() as isize / textbox_width as isize
                                    {
                                        textbox_mask
                                            [rel_y as usize * textbox_width + rel_x as usize]
                                    } else {
                                        0 // Outside textbox
                                    };

                                    if mask_alpha > 0 {
                                        // Apply both glyph alpha and textbox mask
                                        let combined_alpha =
                                            (glyph_alpha as u16 * mask_alpha as u16) >> 8;

                                        let pixel_idx = (final_y as usize * width as usize
                                            + final_x as usize)
                                            * 4;

                                        // Apply colour with combined alpha to RGB channels (skip alpha channel)
                                        for c in 0..3 {
                                            let value =
                                                ((colour[c] as u16 * combined_alpha) >> 8) as u8;
                                            if add_mode {
                                                pixels[pixel_idx + c] =
                                                    pixels[pixel_idx + c].wrapping_add(value);
                                            } else {
                                                pixels[pixel_idx + c] =
                                                    pixels[pixel_idx + c].wrapping_sub(value);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        char_width
    }

    /// Draw text range with horizontal scrolling, using additive/subtractive compositing Automatically clips using textbox_mask (same as blinkey blinking)
    pub fn draw_text_scrollable_additive(
        &mut self,
        pixels: &mut [u8],
        width: u32,
        height: u32,
        text: &str,
        char_widths: &[usize],
        start_char_index: usize,
        end_char_index: usize,
        x_start: f32,
        y_center: f32,
        size: f32,
        weight: u16,
        font: &str,
        colour: [u8; 4],
        textbox_mask: &[u8],
        textbox_x: usize,
        textbox_y: usize,
        textbox_width: usize,
        add_mode: bool,
    ) {
        let mut x_offset = x_start;
        let chars: Vec<char> = text.chars().collect();

        // WHY/PROOF: the char range is the caller's, which can run past the string — chars[i] would panic.
        for i in start_char_index..end_char_index.min(chars.len()) {
            let ch = chars[i];
            self.render_char_additive(
                pixels,
                width,
                height,
                ch,
                x_offset,
                y_center,
                size,
                weight,
                font,
                colour,
                textbox_mask,
                textbox_x,
                textbox_y,
                textbox_width,
                add_mode,
            );
            x_offset += char_widths[i] as f32;
        }
    }

    /// Render single character with additive/subtractive compositing (u32 ARGB version) pixel += char_alpha * mask_alpha * brightness (or -= for subtract)
    pub fn render_char_additive_u32(
        &mut self,
        pixels: &mut [u32],
        width: usize,
        ch: char,
        x_offset: f32,
        y_center: f32,
        size: f32,
        weight: u16,
        font: &str,
        colour: u32,
        textbox_mask: &[u8],
        add_mode: bool,
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(self.optical_weight(weight, size)));

        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size));
        buffer.set_size(Some(size * 2.0), Some(size * 2.0));

        let text = ch.to_string();
        buffer.set_text(&text, &attrs, Shaping::Advanced, None);
        // cosmic-text 0.19 shapes on demand: lay the text out before reading its runs.
        buffer.shape_until_scroll(&mut self.font_system, false);

        let mut char_width: f32 = 0.0;

        for run in buffer.layout_runs() {
            let baseline_offset = run.line_y;

            if run.glyphs.is_empty() {
                char_width = run.line_w;
            }

            for glyph in run.glyphs {
                char_width = char_width.max(glyph.x + glyph.w);

                let offset_y = y_center - size / 2.0;
                let physical_glyph = glyph.physical((x_offset, offset_y), 1.);

                if let Some(image) = self
                    .swash_cache
                    .get_image(&mut self.font_system, physical_glyph.cache_key)
                {
                    let glyph_x = physical_glyph.x + image.placement.left;
                    let glyph_y = physical_glyph.y + baseline_offset as i32 - image.placement.top;

                    let glyph_width = image.placement.width as usize;
                    let glyph_height = image.placement.height as usize;

                    let height = pixels.len() / width;
                    for cy in 0..glyph_height {
                        for cx in 0..glyph_width {
                            let final_x = glyph_x + cx as i32;
                            let final_y = glyph_y + cy as i32;
                            // WHY: Glyph can be partially off-screen when textbox is scrolled PROOF: final_x/final_y are i32, can be negative or exceed bounds PREVENTS: Index out of bounds panic on wrapped negative values
                            // WHY/PROOF: final_x and final_y come from glyph positioning, negative or past the buffer whenever text runs off an edge — the cast to usize would wrap and the write would land outside the pixel buffer.
                            if final_x < 0
                                || final_y < 0
                                || final_x as usize >= width
                                || final_y as usize >= height
                            {
                                continue;
                            }
                            let idx = final_y as usize * width + final_x as usize;
                            let char_alpha = image.data[cy * glyph_width + cx];
                            let mask_alpha = textbox_mask[idx];

                            let combined_alpha = (char_alpha as u32 * mask_alpha as u32) >> 8;

                            // Mask out alpha, multiply each channel by combined_alpha in one SIMD-style op
                            let rb = (colour & 0x00_FF_00_FF) * combined_alpha;
                            let g = (colour & 0x00_00_FF_00) * combined_alpha;

                            let contribution =
                                ((rb >> 8) & 0x00_FF_00_FF) | ((g >> 8) & 0x00_00_FF_00);

                            if add_mode {
                                pixels[idx] += contribution;
                            } else {
                                pixels[idx] -= contribution;
                            }
                        }
                    }
                }
            }
        }

        char_width
    }
}
