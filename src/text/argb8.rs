//! The 8-bit-only text blits: the legacy byte-buffer draws (photon's logo and render pass write visible RGB bytes).
//! They speak 8-bit visible bytes by design, so they are not generic over [`crate::pixel::Pixel`]; the α + darkness path every depth shares is `render_buffer_u32` in the parent module.
//! They lived in text.rs until the rendering core went generic; they are still methods of [`TextRenderer`], so callers are unchanged.
//! The additive / subtractive wrapping blits (text added over a finished u32 buffer and subtracted again to remove it, drawn last) went on 2026-11-17: nothing in fluor, Photon, lumis or mahere called them.

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
                    // This LEGACY path composites in VISIBLE-RGB byte space (raw channel bytes, each glyph composed under what is there with the visible tint `colour[c]`), NOT the u32 α+darkness domain of `render_buffer_u32`. So the Color branch here writes the source's visible sRGB channels DIRECTLY (no darkness complement) — the buffer isn't XOR-flipped at finalize. Colour glyph = `Content::Color` RGBA (4 bytes/pixel); Mask glyph = 1-byte coverage.
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
                                // Front to back: the glyph goes UNDER what the buffer already holds, each byte read as the coverage laid so far, so the glyph fills only what is still open, `v + (256 − v) × glyph >> 8` with the glyph its colour × coverage (`under`'s α accumulation on one byte). Into an empty buffer, which is how every caller starts one, that is the glyph as it always was; only a pixel two glyphs share differs from the over-blend this was until 2026-11-07, and a later draw no longer covers an earlier one (it lands behind it).
                                let under = |v: u8, src: u16| -> u8 {
                                    let v = v as u16;
                                    let g = (src * alpha_u16) >> 8;
                                    // Each term stays in a byte: g ≤ 254 (255 × 255 >> 8), so v + (256 − v) × g >> 8 < 256, and (256 − v) × g < 2¹⁶.
                                    (v + (((256 - v) * g) >> 8)) as u8
                                };

                                if is_colour {
                                    // Composite the glyph's OWN visible sRGB, ignoring the style `colour`. Map source r,g,b onto the first three channels; any 4th (alpha) channel keeps the mask-path behaviour of tinting toward `colour[3]`.
                                    let base = (cy * glyph_width + cx) * 4;
                                    for c in 0..channels {
                                        let src = if c < 3 {
                                            image.data[base + c] as u16
                                        } else {
                                            colour[c] as u16
                                        };
                                        pixels[offset + c] = under(pixels[offset + c], src);
                                    }
                                } else {
                                    for c in 0..channels {
                                        pixels[offset + c] = under(pixels[offset + c], colour[c] as u16);
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
