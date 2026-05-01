//! Text rendering — verbatim port of photon's [text_rasterizing.rs](/mnt/Octopus/Code/photon/src/ui/text_rasterizing.rs). Built on cosmic-text + swash. Bundles Open Sans Regular + Bold (OFL-licensed) so the default fluor build renders text out of the box; consumers can load additional fonts via the `font_system_mut()` accessor.
//!
//! Photon's full version bundles Oxanium and Josefin Slab too; those are dropped here to keep the v0 crate small (~260 KB instead of ~2 MB). They'll be added back when a consumer needs them.

use cosmic_text::{Attrs, Buffer, Family, FontSystem, Metrics, Shaping, SwashCache, Weight};

pub struct TextRenderer {
    font_system: FontSystem,
    swash_cache: SwashCache,
}

impl TextRenderer {
    pub fn new() -> Self {
        let mut font_system = FontSystem::new();

        let db = font_system.db_mut();

        // Open Sans Regular + Bold. Photon bundles a much larger family set; v0 fluor keeps the crate tiny.
        db.load_font_data(
            include_bytes!("../assets/Open_Sans/static/OpenSans-Regular.ttf").to_vec(),
        );
        db.load_font_data(
            include_bytes!("../assets/Open_Sans/static/OpenSans-Bold.ttf").to_vec(),
        );

        Self {
            font_system,
            swash_cache: SwashCache::new(),
        }
    }

    /// Mutable access to the underlying `FontSystem` so consumers can load additional fonts via `text.font_system_mut().db_mut().load_font_data(...)`.
    pub fn font_system_mut(&mut self) -> &mut FontSystem {
        &mut self.font_system
    }

    pub fn draw_text_center_u32(
        &mut self,
        pixels: &mut [u32],
        width: usize,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        weight: u16,
        colour: u32, // [RGBA]
        font: &str,
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;

            for glyph in run.glyphs {
                min_x = min_x.min(glyph.x);
                max_x = max_x.max(glyph.x + glyph.w);
            }

            let text_width = max_x - min_x;
            let text_height = run.line_height;

            self.render_buffer_u32(
                &mut buffer,
                pixels,
                width,
                x,
                y,
                text_width,
                text_height,
                colour,
                0, // center alignment
            );

            text_width
        } else {
            0.
        }
    }

    pub fn draw_text_left_u32(
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
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;

            self.render_buffer_u32(
                &mut buffer,
                pixels,
                width,
                x,
                y,
                text_width,
                text_height,
                colour,
                1, // left alignment
            );

            text_width
        } else {
            0.
        }
    }

    pub fn draw_text_right_u32(
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
    ) -> f32 {
        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
        buffer.shape_until_scroll(&mut self.font_system, false);

        if let Some(run) = buffer.layout_runs().next() {
            let mut text_width = 0.0f32;
            for glyph in run.glyphs {
                text_width = text_width.max(glyph.x + glyph.w);
            }
            let text_height = run.line_height;

            self.render_buffer_u32(
                &mut buffer,
                pixels,
                width,
                x,
                y,
                text_width,
                text_height,
                colour,
                2, // right alignment
            );

            text_width
        } else {
            0.
        }
    }

    pub fn draw_text_center(
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
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
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

    pub fn draw_text_left(
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
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
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

    pub fn draw_text_right(
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
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
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

    /// Render buffer to u32 packed pixel array
    fn render_buffer_u32(
        &mut self,
        buffer: &mut Buffer,
        pixels: &mut [u32], // [ARGB]
        width: usize,
        anchor_x: f32,
        anchor_y: f32,
        text_width: f32,
        text_height: f32,
        colour: u32,   // [ARGB]
        alignment: u8, // 0=center, 1=left, 2=right
    ) {
        // Calculate offset based on alignment
        let (offset_x, offset_y) = match alignment {
            0 => (anchor_x - text_width / 2., anchor_y - text_height / 2.), // center
            1 => (anchor_x, anchor_y - text_height / 2.),                   // left
            2 => (anchor_x - text_width, anchor_y - text_height / 2.),      // right
            _ => (anchor_x, anchor_y),
        };

        let mut colour = colour as u64;
        colour = (colour | (colour << 16)) & 0x0000FFFF0000FFFF;
        colour = (colour | (colour << 8)) & 0x00FF00FF00FF00FF;

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

                                // Bounds check to prevent underflow/overflow
                                if final_x < 0 || final_y < 0 || final_x >= width as isize {
                                    continue;
                                }
                                let idx = final_y as usize * width + final_x as usize;
                                if idx >= pixels.len() {
                                    continue;
                                }

                                let mut bg = pixels[idx] as u64;
                                let alpha = alpha as u64;
                                let inv_alpha = (255 - alpha) as u64;

                                bg = (bg | (bg << 16)) & 0x0000FFFF0000FFFF;
                                bg = (bg | (bg << 8)) & 0x00FF00FF00FF00FF;

                                let mut blended = bg * inv_alpha + colour * alpha;

                                blended = (blended >> 8) & 0x00FF00FF00FF00FF;
                                blended = (blended | (blended >> 8)) & 0x0000FFFF0000FFFF;
                                blended = blended | (blended >> 16);
                                pixels[idx] = blended as u32;
                            }
                        }
                    }
                }
            }
        }
    }

    fn render_buffer(
        &mut self,
        buffer: &mut Buffer,
        pixels: &mut [u8],
        width: u32,
        _height: u32,
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

                    for cy in 0..glyph_height {
                        for cx in 0..glyph_width {
                            let alpha = image.data[cy * glyph_width + cx];
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
                                let offset = (final_y as usize * width as usize + final_x as usize)
                                    * channels;

                                let alpha_u16 = alpha as u16;
                                let inv_alpha = 256 - alpha_u16;

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

    /// Draw left-aligned text with additive/subtractive compositing (u32 ARGB version)
    /// Uses wrapping add/sub so it's reversible - subtract same colour to remove text add_mode: true = add colour, false = subtract colour
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
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
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

    /// Draw center-aligned text with additive/subtractive compositing (u32 ARGB version)
    /// Uses wrapping add/sub so it's reversible - subtract same colour to remove text add_mode: true = add colour, false = subtract colour
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
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
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

    /// Draw right-aligned text with additive/subtractive compositing (u32 ARGB version)
    /// Uses wrapping add/sub so it's reversible - subtract same colour to remove text add_mode: true = add colour, false = subtract colour
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
            .weight(Weight(weight));

        let metrics = Metrics::relative(size, 1.2);
        let mut buffer = Buffer::new(&mut self.font_system, metrics);

        buffer.set_size(&mut self.font_system, None, None);
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);
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
            let run_width = run
                .glyphs
                .iter()
                .map(|g| g.w)
                .sum::<f32>();
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
            let run_width = run
                .glyphs
                .iter()
                .map(|g| g.w)
                .sum::<f32>();
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

    /// Measure the width of text without rendering it
    pub fn measure_text_width(&mut self, text: &str, size: f32, weight: u16, font: &str) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        let attrs = Attrs::new()
            .family(Family::Name(font))
            .weight(Weight(weight));

        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size));
        buffer.set_size(&mut self.font_system, Some(10000.0), Some(size * 2.0));
        buffer.set_text(&mut self.font_system, text, &attrs, Shaping::Advanced);

        // Calculate width using glyph advances (includes spacing for spaces!)
        buffer.layout_runs().fold(0.0, |max_width, run| {
            let run_width = run
                .glyphs
                .iter()
                .fold(0.0, |w, glyph| (glyph.x + glyph.w).max(w));
            max_width.max(run_width)
        })
    }

    /// Render a single character with additive blending (reversible with wrapping_add/sub)
    /// Returns the width of the rendered character in pixels
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
            .weight(Weight(weight));

        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size));
        buffer.set_size(&mut self.font_system, Some(size * 2.0), Some(size * 2.0));

        let text = ch.to_string();
        buffer.set_text(&mut self.font_system, &text, &attrs, Shaping::Advanced);

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

    /// Draw text range with horizontal scrolling, using additive/subtractive compositing
    /// Automatically clips using textbox_mask (same as blinkey blinking)
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
            .weight(Weight(weight));

        let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size));
        buffer.set_size(&mut self.font_system, Some(size * 2.0), Some(size * 2.0));

        let text = ch.to_string();
        buffer.set_text(&mut self.font_system, &text, &attrs, Shaping::Advanced);

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
                            // WHY: Glyph can be partially off-screen when textbox is scrolled
                            // PROOF: final_x/final_y are i32, can be negative or exceed bounds
                            // PREVENTS: Index out of bounds panic on wrapped negative values
                            if final_x < 0 || final_y < 0
                                || final_x as usize >= width
                                || final_y as usize >= height {
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
