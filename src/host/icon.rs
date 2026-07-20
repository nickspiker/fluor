//! VSF image → α + darkness pixel buffer for app icons (orb in the top-left chrome slot, future window thumbnails, drag-drop previews).
//!
//! Loads the canonical VSF image format produced by `vsfimg`: a section labeled `image` whose `data` field is either an uncompressed `Tensor<u8>` (shape `[h, w, 3]`, VSF RGB gamma2) for sources ≤ 256×256, or a `v(b'a', av1_bytes)` AV1 wrapper for larger sources. Both paths land in the same [`Icon`] struct so the rasterizer doesn't care which path the file took.
//!
//! Output is packed α + darkness `u32` per fluor's pixel convention: α = `0xFF` (image pixels are opaque — alpha shaping lives in the rasterizer's mask, not the icon), darkness = `255 − visible_RGB` so the buffer drops straight into `Blend::under` without per-pixel inversion at composite time.
//!
//! VSF YCbCr inverse (matches `vsfimg`'s encoder math): `R = Y + 2(Cr − 0.5)`, `B = Y + 2(Cb − 0.5)`, `G = (4Y − R − B) / 2`. No legacy colour tagging — files are VSF RGB by definition.

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dSettings};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_default_settings, dav1d_get_picture, dav1d_open,
    dav1d_picture_unref, dav1d_send_data,
};
use std::ptr::NonNull;
use vsf::decoding::parse::parse;
use vsf::file_format::{VsfField, VsfHeader};
use vsf::types::VsfType;

/// Decoded image ready to composite. Square or rectangular; the rasterizer crops/masks at draw time.
pub struct Icon {
    pub width: u32,
    pub height: u32,
    /// Packed α + darkness pixels, row-major, `width * height` long. α byte = `0xFF` (opaque); RGB bytes = `255 − visible_RGB` (VSF RGB gamma2 darkness). Ready to feed straight into `Blend::under`.
    pub pixels: Vec<u32>,
}

#[cfg(feature = "host-winit")]
impl Icon {
    /// Convert to a [`winit::window::Icon`] for the OS taskbar / window-list / alt-tab. Flips fluor's darkness-packed RGB back to visible RGB and reshapes into the row-major `[R, G, B, A, ...]` byte layout winit expects. Returns `None` if winit rejects the byte buffer (size mismatch — shouldn't happen with our own decoder but the conversion API is fallible).
    ///
    /// **Why this lives on Icon**: the orb file IS the canonical app-identity asset; baking the OS-icon path into the same struct keeps the chrome's `app_icon` and the OS taskbar icon perfectly synced — set one, set the other.
    pub fn to_winit_icon(&self) -> Option<winit::window::Icon> {
        let (rgba, w, h) = self.to_rgba_circular();
        winit::window::Icon::from_rgba(rgba, w, h).ok()
    }

    /// Row-major visible-RGBA bytes with a CIRCULAR alpha mask — the shape every OS icon surface (taskbar, alt-tab, tray) wants: the orb disk, transparent corners, ~1.5px anti-aliased rim. The stored pixels are square and opaque (the in-app chrome shapes them with the rasterizer's mask instead); this is the one place squareness is clipped for the OS, so window icon and tray can't drift apart.
    pub fn to_rgba_circular(&self) -> (Vec<u8>, u32, u32) {
        let (w, h) = (self.width, self.height);
        let cx = (w as f32 - 1.0) / 2.0;
        let cy = (h as f32 - 1.0) / 2.0;
        let radius = (w.min(h) as f32) / 2.0;
        let mut rgba: Vec<u8> = Vec::with_capacity(self.pixels.len() * 4);
        for (i, &p) in self.pixels.iter().enumerate() {
            // fluor pixel is 0xααDDDDDD where DDDDDD is darkness; flip via XOR with 0x00FF_FFFF to get visible RGB.
            let visible = p ^ 0x00FF_FFFF;
            let x = (i as u32 % w) as f32;
            let y = (i as u32 / w) as f32;
            let d = ((x - cx) * (x - cx) + (y - cy) * (y - cy)).sqrt();
            // Full inside, zero outside, linear ramp across the ~1.5px rim.
            let cover = ((radius - d) / 1.5).clamp(0.0, 1.0);
            let a = (((visible >> 24) & 0xFF) as f32 * cover) as u8;
            rgba.push(((visible >> 16) & 0xFF) as u8);
            rgba.push(((visible >> 8) & 0xFF) as u8);
            rgba.push((visible & 0xFF) as u8);
            rgba.push(a);
        }
        (rgba, w, h)
    }
}

#[derive(Debug)]
pub enum IconError {
    /// VSF parse or structural problem (header, section, field).
    Parse(String),
    /// File parsed but doesn't carry an `image` section.
    MissingImageSection,
    /// `image` section exists but no `data` field inside.
    MissingDataField,
    /// `data` field carried a type fluor doesn't decode (only `t_u3` tensor and `v(b'a', ...)` AV1 are supported).
    UnsupportedDataType,
    /// Tensor shape isn't `[h, w, 3]`.
    BadTensorShape(Vec<usize>),
    /// rav1d returned an error code or no picture for an AV1 payload.
    Av1Decode(String),
}

impl core::fmt::Display for IconError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            IconError::Parse(s) => write!(f, "VSF parse: {}", s),
            IconError::MissingImageSection => write!(f, "no 'image' section in file"),
            IconError::MissingDataField => write!(f, "'image' section has no 'data' field"),
            IconError::UnsupportedDataType => {
                write!(
                    f,
                    "'data' field has unsupported VSF type (expected t_u3 or v(b'a', ...))"
                )
            }
            IconError::BadTensorShape(s) => write!(f, "tensor shape {:?} not [h, w, 3]", s),
            IconError::Av1Decode(s) => write!(f, "AV1 decode: {}", s),
        }
    }
}

impl std::error::Error for IconError {}

impl Icon {
    /// Decode a full VSF file from its bytes — header + sections + tensor / AV1 payload — into a ready-to-composite α + darkness buffer.
    pub fn from_vsf_bytes(data: &[u8]) -> Result<Self, IconError> {
        let (header, _consumed) = VsfHeader::decode(data).map_err(IconError::Parse)?;

        let img_field = header
            .fields
            .iter()
            .find(|f| f.name == "image")
            .ok_or(IconError::MissingImageSection)?;

        let mut p = img_field.offset_bytes;
        if p >= data.len() {
            return Err(IconError::Parse(format!(
                "image section offset {} beyond file length {}",
                p,
                data.len()
            )));
        }
        if data[p] == b'>' {
            p += 1;
        }
        if p >= data.len() || data[p] != b'[' {
            return Err(IconError::Parse(format!(
                "expected '[' at section start, got byte {:02x}",
                data.get(p).copied().unwrap_or(0)
            )));
        }
        p += 1;

        if p < data.len() && data[p] != b'(' {
            let _name = parse(data, &mut p)
                .map_err(|e| IconError::Parse(format!("section name: {:?}", e)))?;
            let _count =
                parse(data, &mut p).map_err(|e| IconError::Parse(format!("section n: {:?}", e)))?;
            let _length =
                parse(data, &mut p).map_err(|e| IconError::Parse(format!("section b: {:?}", e)))?;
        }

        let mut data_value: Option<VsfType> = None;
        for _ in 0..img_field.child_count {
            let field = VsfField::parse(data, &mut p)
                .map_err(|e| IconError::Parse(format!("field parse: {}", e)))?;
            if field.name == "data" {
                data_value = field.values.into_iter().next();
                break;
            }
        }
        let data_value = data_value.ok_or(IconError::MissingDataField)?;

        match data_value {
            VsfType::t_u3(tensor) => Self::from_rgb_tensor(tensor.shape, tensor.data),
            VsfType::v(tag, av1_bytes) if tag == b'a' => Self::from_av1(&av1_bytes),
            _ => Err(IconError::UnsupportedDataType),
        }
    }

    /// Build directly from an uncompressed `[h, w, 3]` VSF RGB gamma2 buffer.
    fn from_rgb_tensor(shape: Vec<usize>, bytes: Vec<u8>) -> Result<Self, IconError> {
        if shape.len() != 3 || shape[2] != 3 {
            return Err(IconError::BadTensorShape(shape));
        }
        let h = shape[0] as u32;
        let w = shape[1] as u32;
        let pixels = pack_alpha_darkness(&bytes, w, h);
        Ok(Self {
            width: w,
            height: h,
            pixels,
        })
    }

    /// Decode an AV1 OBU bitstream → YCbCr → VSF RGB gamma2 → α + darkness. YCbCr inverse mirrors `vsfimg`'s encoder math byte-for-byte.
    fn from_av1(av1: &[u8]) -> Result<Self, IconError> {
        let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
        unsafe { dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap()) };
        let settings = unsafe { settings.assume_init() };

        let mut ctx: Option<Dav1dContext> = None;
        let open = unsafe {
            dav1d_open(
                NonNull::new(&mut ctx as *mut _),
                NonNull::new(&settings as *const _ as *mut _),
            )
        };
        if open.0 < 0 {
            return Err(IconError::Av1Decode(format!("dav1d_open: {}", open.0)));
        }
        let ctx =
            ctx.ok_or_else(|| IconError::Av1Decode("dav1d_open returned null context".into()))?;

        let mut data = Dav1dData::default();
        let data_ptr = unsafe { dav1d_data_create(NonNull::new(&mut data), av1.len()) };
        if data_ptr.is_null() {
            unsafe { dav1d_close(NonNull::new(&mut Some(ctx) as *mut _)) };
            return Err(IconError::Av1Decode(
                "dav1d_data_create returned null".into(),
            ));
        }
        unsafe { std::ptr::copy_nonoverlapping(av1.as_ptr(), data_ptr, av1.len()) };

        loop {
            let r = unsafe { dav1d_send_data(Some(ctx), NonNull::new(&mut data)) };
            if r.0 == 0 {
                break;
            } else if r.0 == -11 {
                continue;
            } else if r.0 < 0 {
                unsafe { dav1d_close(NonNull::new(&mut Some(ctx) as *mut _)) };
                return Err(IconError::Av1Decode(format!("dav1d_send_data: {}", r.0)));
            }
        }

        let mut pic = Dav1dPicture::default();
        loop {
            let r = unsafe { dav1d_get_picture(Some(ctx), NonNull::new(&mut pic)) };
            if r.0 == 0 {
                break;
            } else if r.0 == -11 {
                std::thread::yield_now();
                continue;
            } else {
                unsafe { dav1d_close(NonNull::new(&mut Some(ctx) as *mut _)) };
                return Err(IconError::Av1Decode(format!("dav1d_get_picture: {}", r.0)));
            }
        }

        let width = pic.p.w as usize;
        let height = pic.p.h as usize;
        let stride_y = pic.stride[0] as usize;
        let stride_uv = pic.stride[1] as usize;
        let y_ptr = pic.data[0]
            .ok_or_else(|| IconError::Av1Decode("missing Y plane".into()))?
            .as_ptr() as *const u8;
        let u_ptr = pic.data[1]
            .ok_or_else(|| IconError::Av1Decode("missing U plane".into()))?
            .as_ptr() as *const u8;
        let v_ptr = pic.data[2]
            .ok_or_else(|| IconError::Av1Decode("missing V plane".into()))?
            .as_ptr() as *const u8;

        let mut pixels = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let y_val = unsafe { *y_ptr.add(y * stride_y + x) } as f32 / 255.0;
                let u_val = unsafe { *u_ptr.add((y / 2) * stride_uv + (x / 2)) } as f32 / 255.0;
                let v_val = unsafe { *v_ptr.add((y / 2) * stride_uv + (x / 2)) } as f32 / 255.0;
                let cb = u_val - 0.5;
                let cr = v_val - 0.5;
                let r = y_val + 2.0 * cr;
                let b = y_val + 2.0 * cb;
                let g = (4.0 * y_val - r - b) / 2.0;
                // f32 → u8 in Rust saturates (< 0 → 0, > 255 → 255, NaN → 0) so out-of-gamut YCbCr maps cleanly without explicit clamps.
                let dr = (255.0 - r * 255.0) as u8 as u32;
                let dg = (255.0 - g * 255.0) as u8 as u32;
                let db = (255.0 - b * 255.0) as u8 as u32;
                pixels.push(0xFF000000 | (dr << 16) | (dg << 8) | db);
            }
        }

        unsafe {
            dav1d_picture_unref(NonNull::new(&mut pic));
            dav1d_close(NonNull::new(&mut Some(ctx) as *mut _));
        }

        Ok(Self {
            width: width as u32,
            height: height as u32,
            pixels,
        })
    }
}

/// Pack VSF RGB gamma2 u8 triples into α + darkness u32: α=0xFF, darkness = 255 − visible. The buffer length must be `w * h * 3` (caller guarantees from a verified `[h, w, 3]` tensor shape).
fn pack_alpha_darkness(rgb: &[u8], w: u32, h: u32) -> Vec<u32> {
    let count = (w as usize) * (h as usize);
    let mut pixels = Vec::with_capacity(count);
    for chunk in rgb.chunks_exact(3) {
        let dr = (255 - chunk[0]) as u32;
        let dg = (255 - chunk[1]) as u32;
        let db = (255 - chunk[2]) as u32;
        pixels.push(0xFF000000 | (dr << 16) | (dg << 8) | db);
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_alpha_darkness_inverts_rgb_and_sets_alpha_opaque() {
        let rgb = vec![0u8, 0, 0, 255, 255, 255, 128, 64, 32];
        let p = pack_alpha_darkness(&rgb, 3, 1);
        assert_eq!(p[0], 0xFFFFFFFF);
        assert_eq!(p[1], 0xFF000000);
        assert_eq!(p[2], 0xFF000000 | (127u32 << 16) | (191u32 << 8) | 223u32);
    }

    #[test]
    fn decode_example_orb_vsf() {
        let bytes = include_bytes!("../../examples/assets/example_orb.vsf");
        let icon = Icon::from_vsf_bytes(bytes).expect("orb decodes");
        assert_eq!(icon.width, 256);
        assert_eq!(icon.height, 256);
        assert_eq!(icon.pixels.len(), 256 * 256);
        for &p in &icon.pixels {
            assert_eq!(p & 0xFF000000, 0xFF000000, "α byte should be 0xFF (opaque)");
        }
    }
}
