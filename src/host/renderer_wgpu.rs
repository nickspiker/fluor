//! macOS renderer — wgpu/Metal backend, no shader.
//!
//! CPU buffer → write_texture (memcpy) → copy_texture_to_texture (DMA) → present. Zero float conversion. Pixels stay as u32 bytes the whole way.
//!
//! Pixel layout: `u32` stores `0xAARRGGBB`. Little-endian bytes are [B,G,R,A] = Bgra8Unorm — direct upload, zero byte-swapping.
//!
//! Uses PostMultiplied alpha mode so 0x00000000 pixels composite as fully transparent — required for squircle window corners.

use winit::window::Window;

pub struct WgpuBuffer<'a> {
    inner: &'a mut Renderer,
}

impl<'a> std::ops::Deref for WgpuBuffer<'a> {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        &self.inner.cpu_buffer
    }
}

impl<'a> std::ops::DerefMut for WgpuBuffer<'a> {
    fn deref_mut(&mut self) -> &mut [u32] {
        &mut self.inner.cpu_buffer
    }
}

impl<'a> WgpuBuffer<'a> {
    pub fn present(self) -> Result<(), ()> {
        self.inner.present_frame();
        Ok(())
    }
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    frame_texture: wgpu::Texture,
    cpu_buffer: Vec<u32>,
    width: u32,
    height: u32,
    /// Raw NSView pointer for deferred colorspace tagging (done on first render, not init, because init runs inside a nounwind C callback where panics abort).
    ns_view_ptr: *mut std::ffi::c_void,
    colorspace_tagged: bool,
}

impl Renderer {
    pub fn new(window: &Window, width: u32, height: u32) -> Self {
        let static_window: &'static Window = unsafe { std::mem::transmute(window) };
        pollster::block_on(Self::init(static_window, width, height))
    }

    async fn init(window: &'static Window, width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::METAL,
            ..Default::default()
        });

        let surface = instance
            .create_surface(window)
            .expect("wgpu: create_surface failed");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("wgpu: no Metal adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fluor"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .expect("wgpu: request_device failed");

        let caps = surface.get_capabilities(&adapter);

        // Lock the format. fluor's pixel convention is `0xAARRGGBB` u32 → LE bytes [B,G,R,A] = Bgra8Unorm. If the surface doesn't offer it, fail loud — we'd otherwise silently swap R↔B and produce wrong colours. The host upload boundary is the only legal place to convert; this surface does direct memcpy so it must match.
        let surface_format = caps.formats.iter().copied()
            .find(|f| *f == wgpu::TextureFormat::Bgra8Unorm)
            .expect("wgpu: surface does not support Bgra8Unorm — fluor's pixel convention requires it for zero-conversion upload");

        // Lock the alpha mode. PostMultiplied means the OS compositor performs the premultiply at present time, so we can hand it our straight-alpha pixels directly. Without it, we'd need to premultiply at upload — and we currently don't.
        let alpha_mode = caps.alpha_modes.iter().copied()
            .find(|m| *m == wgpu::CompositeAlphaMode::PostMultiplied)
            .expect("wgpu: surface does not support PostMultiplied alpha — fluor's convention requires it (straight-alpha internal pixels)");

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::COPY_DST,
            format: surface_format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Capture the NSView pointer for deferred colorspace tagging (first render, not here — init runs inside a nounwind C callback).
        let ns_view_ptr = {
            use winit::raw_window_handle::HasWindowHandle;
            window.window_handle().ok().and_then(|h| match h.as_raw() {
                winit::raw_window_handle::RawWindowHandle::AppKit(a) => Some(a.ns_view.as_ptr()),
                _ => None,
            }).unwrap_or(std::ptr::null_mut())
        };

        let frame_texture = Self::make_texture(&device, width, height);
        let cpu_buffer = vec![0u32; (width * height) as usize];

        Self {
            surface,
            device,
            queue,
            config,
            frame_texture,
            cpu_buffer,
            width,
            height,
            ns_view_ptr,
            colorspace_tagged: false,
        }
    }

    fn tag_vsf_colorspace(ns_view_ptr: *mut std::ffi::c_void) {
        if ns_view_ptr.is_null() { return; }
        use objc2::runtime::{AnyClass, AnyObject, Sel};
        unsafe {
            unsafe extern "C" {
                fn CFDataCreate(allocator: *const std::ffi::c_void, bytes: *const u8, length: isize) -> *const std::ffi::c_void;
                fn CGColorSpaceCreateWithICCData(data: *const std::ffi::c_void) -> *const std::ffi::c_void;
                fn CFRelease(cf: *const std::ffi::c_void);
            }
            let view = ns_view_ptr as *mut AnyObject;
            let sel_layer = Sel::register(c"layer");
            let sel_set_cs = Sel::register(c"setColorspace:");
            let sel_is_kind = Sel::register(c"isKindOfClass:");
            let ca_metal_layer_class = AnyClass::get(c"CAMetalLayer");
            // Get the view's layer
            let layer: *mut AnyObject = objc2::msg_send![&*view, layer];
            if layer.is_null() { return; }
            // Verify it's a CAMetalLayer
            if let Some(cls) = ca_metal_layer_class {
                let is_metal: bool = objc2::msg_send![&*layer, isKindOfClass: cls];
                if !is_metal { return; }
            }
            let icc = include_bytes!("vsf_rgb.icc");
            let cf_data = CFDataCreate(std::ptr::null(), icc.as_ptr(), icc.len() as isize);
            if cf_data.is_null() { return; }
            let cs = CGColorSpaceCreateWithICCData(cf_data);
            if !cs.is_null() {
                // setColorspace: takes a CGColorSpaceRef (which is a CFTypeRef / *const c_void). We pass it as a raw pointer argument via msg_send.
                let cs_obj = cs as *mut AnyObject;
                let () = objc2::msg_send![&*layer, setColorspace: &*cs_obj];
                CFRelease(cs);
            }
            CFRelease(cf_data);
        }
    }

    fn make_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame-tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.width = width;
        self.height = height;
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.cpu_buffer.resize((width * height) as usize, 0);
        self.frame_texture = Self::make_texture(&self.device, width, height);
    }

    pub fn lock_buffer(&mut self) -> WgpuBuffer<'_> {
        if !self.colorspace_tagged {
            self.colorspace_tagged = true;
            Self::tag_vsf_colorspace(self.ns_view_ptr);
        }
        WgpuBuffer { inner: self }
    }

    fn present_frame(&mut self) {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.cpu_buffer.as_ptr() as *const u8,
                self.cpu_buffer.len() * 4,
            )
        };
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.frame_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(_) => return,
                }
            }
            Err(_) => return,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy-enc"),
            });
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.frame_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &output.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
