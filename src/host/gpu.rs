//! The GPU present — macOS always (wgpu/Metal), Linux with the `gpu` feature (wgpu/Vulkan) when the app asks for it.
//!
//! The finished surface buffer (`persistent_screen`, premultiplied `0xAARRGGBB`) is mirrored in a texture of the surface's size, and only the rectangle that changed since the last present is uploaded.
//! The present pass then draws that texture onto the window surface, and where the app hands over an [`UnderLayer`] (its own picture, drawn on fluor's device) lays it under the chrome inside the window shape: `out = chrome + (1 − chrome.α) · clip · layer`, the same premultiplied under a CPU flatten does, without the frame ever leaving the GPU.
//!
//! Pixel layout: `u32` stores `0xAARRGGBB`; little-endian bytes are [B,G,R,A] = Bgra8Unorm, uploaded as they are.
//! The surface is a plain (non-sRGB) 8-bit format, so the bytes reach the screen as fluor wrote them.

use winit::window::Window;

/// fluor's GPU: one instance, adapter, device and queue for every surface, so an app's textures can go under the chrome on any monitor.
/// Every member is a cheap handle (wgpu resources are reference counted); an app keeps its own clone from [`super::app::FluorApp::gpu_ready`].
#[derive(Clone)]
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// What an app needs of fluor's device: features beyond WebGPU's (intersected with what the adapter has) and limits (the app derives them from the adapter's own, so they fit).
pub struct GpuRequest {
    pub features: wgpu::Features,
    pub limits: wgpu::Limits,
}

/// The app's picture under the chrome for this frame: a texture on fluor's device the size of the window's pass-0 viewport, any filterable 8-bit colour format, its values the display's (as fluor's own pixels are).
/// The app submits its drawing before returning it; fluor's present goes on the same queue after it.
#[derive(Clone)]
pub struct UnderLayer {
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl Gpu {
    /// The instance every surface is made from.
    pub(crate) fn instance() -> wgpu::Instance {
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: if cfg!(target_os = "macos") { wgpu::Backends::METAL } else { wgpu::Backends::VULKAN },
            ..Default::default()
        })
    }

    /// The adapter that can present to `surface` and a device on it with what `request` asks (the adapter's features only); `None` when there is no such adapter or device, or (off macOS) the app asks for none.
    pub(crate) fn new(instance: wgpu::Instance, surface: &wgpu::Surface<'static>, request: impl FnOnce(&wgpu::Adapter) -> Option<GpuRequest>) -> Option<Gpu> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        }))
        .map_err(|e| log::warn!("fluor gpu: no adapter presents to the window: {e:?}"))
        .ok()?;
        // Linux presents on the GPU only for an app that asked; macOS always does, on a plain device when the app asks nothing.
        let (features, limits) = match request(&adapter) {
            Some(r) => (r.features & adapter.features(), r.limits),
            None if cfg!(target_os = "macos") => (wgpu::Features::empty(), wgpu::Limits::default()),
            None => return None,
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("fluor"),
            required_features: features,
            required_limits: limits,
            ..Default::default()
        }))
        .map_err(|e| log::warn!("fluor gpu: no device: {e:?}"))
        .ok()?;
        let info = adapter.get_info();
        log::info!("fluor gpu: {} ({:?})", info.name, info.backend);
        Some(Gpu { instance, adapter, device, queue })
    }
}

/// The present pass: the surface buffer, and inside the window the app's layer under it, cut to the window shape.
const SHADER: &str = r#"
struct U {
    // The window on the surface: x, y, width, height in surface pixels.
    rect: vec4<i32>,
    // x: a layer is under the chrome; y: the window is drawn 1:1 (texel per pixel), so it is read, not filtered.
    on: vec4<u32>,
}
@group(0) @binding(0) var screen_t: texture_2d<f32>;
@group(0) @binding(1) var layer_t: texture_2d<f32>;
@group(0) @binding(2) var clip_t: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var<uniform> u: U;
// The display map (`theme::G2T_TO_SDR`): the layer is the app's own `g2t` codes, as fluor's buffer is, and goes to the screen thru the same table finalize uses.
@group(0) @binding(5) var sdr_t: texture_2d<f32>;

fn to_sdr(c: vec3<f32>) -> vec3<f32> {
    let i = vec3<i32>(round(c * 255.0));
    return vec3<f32>(textureLoad(sdr_t, vec2<i32>(i.r, 0), 0).r, textureLoad(sdr_t, vec2<i32>(i.g, 0), 0).r, textureLoad(sdr_t, vec2<i32>(i.b, 0), 0).r);
}

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let q = vec2<i32>(p.xy);
    let c = textureLoad(screen_t, q, 0);
    let l = q - u.rect.xy;
    if (u.on.x == 0u || any(l < vec2<i32>(0)) || any(l >= u.rect.zw)) {
        return c;
    }
    var m: f32;
    var under: vec3<f32>;
    if (u.on.y != 0u) {
        m = textureLoad(clip_t, l, 0).r;
        under = textureLoad(layer_t, l, 0).rgb;
    } else {
        let uv = (vec2<f32>(l) + 0.5) / vec2<f32>(u.rect.zw);
        m = textureSampleLevel(clip_t, samp, uv, 0.0).r;
        under = textureSampleLevel(layer_t, samp, uv, 0.0).rgb;
    }
    // The chrome is premultiplied by α × clip (finalized `over`), so what of the window it leaves is clip − its α.
    // WHY/PROOF: 1:1 the chrome's α is α × clip ≤ clip; resampled (a window straddling monitors of different density) the clip is filtered at another place than the surface pixel, and the two can disagree by a fraction below 0.
    let k = max(m - c.a, 0.0);
    return vec4<f32>(c.rgb + k * to_sdr(under), c.a + k);
}
"#;

/// One surface's present: the wgpu surface, the mirror of its buffer, the window's clip mask, and the pass.
pub(crate) struct Presenter {
    /// Held so the surface below (made from a clone of this Arc) and the raw NSView pointer stay valid for the presenter's whole life.
    _window: std::sync::Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The surface buffer's mirror (Bgra8Unorm, premultiplied).
    screen: wgpu::Texture,
    screen_view: wgpu::TextureView,
    /// The window's clip mask (R8), at pass-0 size, uploaded when it may have changed.
    clip: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    /// A 1×1 stand-in for the layer and the clip when there is no layer.
    blank: wgpu::TextureView,
    /// The display map, `theme::G2T_TO_SDR` as a 256×1 texture.
    sdr: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    /// What of the surface buffer changed since the last upload, in surface pixels `(x0, y0, x1, y1)`.
    dirty: Option<(usize, usize, usize, usize)>,
    /// macOS: the NSView, its CAMetalLayer tagged with the VSF colour space on the first present (not at creation, which runs inside a nounwind C callback where panics abort).
    #[cfg(target_os = "macos")]
    ns_view_ptr: *mut std::ffi::c_void,
    #[cfg(target_os = "macos")]
    colorspace_tagged: bool,
}

impl Presenter {
    /// A presenter on `window`, `None` when the device can't present to it with alpha (Linux then keeps softbuffer).
    pub(crate) fn new(gpu: &Gpu, window: std::sync::Arc<Window>, surface: wgpu::Surface<'static>, width: u32, height: u32) -> Option<Presenter> {
        let caps = surface.get_capabilities(&gpu.adapter);
        // fluor's pixels are Bgra8Unorm as they are; an RGBA surface takes them thru the shader's swizzle just as well. Never an sRGB format: the bytes are already the display's.
        let format = [wgpu::TextureFormat::Bgra8Unorm, wgpu::TextureFormat::Rgba8Unorm].into_iter().find(|f| caps.formats.contains(f));
        // The buffer is premultiplied. Metal has no PreMultiplied mode: its PostMultiplied only clears the layer's `opaque` flag and Core Animation composites premultiplied (wgpu-hal metal/surface.rs). On X11, Inherit is the visual's own convention, which is premultiplied.
        let alpha = [wgpu::CompositeAlphaMode::PreMultiplied, wgpu::CompositeAlphaMode::PostMultiplied, wgpu::CompositeAlphaMode::Inherit].into_iter().find(|m| caps.alpha_modes.contains(m));
        let (Some(format), Some(alpha_mode)) = (format, alpha) else {
            log::warn!("fluor gpu: the surface offers {:?} / {:?}, no 8-bit format with alpha", caps.formats, caps.alpha_modes);
            return None;
        };
        let device = &gpu.device;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            // WHY/PROOF: the size is the OS window's, 0 while minimized — wgpu panics configuring a zero-sized surface.
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(device, &config);
        log::info!("fluor gpu: surface {}x{} {:?} {:?}", config.width, config.height, format, alpha_mode);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fluor present"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let tex = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fluor present"),
            entries: &[
                tex(0),
                tex(1),
                tex(2),
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                tex(5),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("fluor present"), bind_group_layouts: &[&layout], immediate_size: 0 });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fluor present"),
            layout: Some(&pl),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })] }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("fluor present"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor { label: Some("fluor present"), size: 32, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let blank = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("fluor blank"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let (screen, screen_view) = Self::make_screen(device, config.width, config.height);
        let sdr = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fluor display map"),
            size: wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &sdr, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &crate::theme::G2T_TO_SDR,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        );
        let sdr = sdr.create_view(&Default::default());

        #[cfg(target_os = "macos")]
        let ns_view_ptr = {
            use winit::raw_window_handle::HasWindowHandle;
            window
                .window_handle()
                .ok()
                .and_then(|h| match h.as_raw() {
                    winit::raw_window_handle::RawWindowHandle::AppKit(a) => Some(a.ns_view.as_ptr()),
                    _ => None,
                })
                .unwrap_or(std::ptr::null_mut())
        };
        Some(Presenter {
            _window: window,
            surface,
            config,
            screen,
            screen_view,
            clip: None,
            blank,
            sdr,
            pipeline,
            layout,
            sampler,
            uniform,
            dirty: None,
            #[cfg(target_os = "macos")]
            ns_view_ptr,
            #[cfg(target_os = "macos")]
            colorspace_tagged: false,
        })
    }

    fn make_screen(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("fluor screen"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let v = t.create_view(&Default::default());
        (t, v)
    }

    /// The surface's backing size changed: the surface configured again and its mirror made anew (all of it uploaded at the next present).
    pub(crate) fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&gpu.device, &self.config);
        (self.screen, self.screen_view) = Self::make_screen(&gpu.device, width, height);
        self.mark_all();
    }

    /// Part of the surface buffer changed: surface pixels `x0..x1 × y0..y1`, any part off the surface ignored.
    pub(crate) fn mark(&mut self, x0: i32, y0: i32, x1: i32, y1: i32) {
        let (w, h) = (self.config.width as i32, self.config.height as i32);
        // WHY/PROOF: the rect is the window's place on the surface, off its edges while the window hangs past them — the upload must stay inside the texture.
        let (x0, y0, x1, y1) = (x0.clamp(0, w) as usize, y0.clamp(0, h) as usize, x1.clamp(0, w) as usize, y1.clamp(0, h) as usize);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        self.dirty = Some(match self.dirty {
            Some((a, b, c, d)) => (a.min(x0), b.min(y0), c.max(x1), d.max(y1)),
            None => (x0, y0, x1, y1),
        });
    }

    /// All of the surface buffer changed.
    pub(crate) fn mark_all(&mut self) {
        self.dirty = Some((0, 0, self.config.width as usize, self.config.height as usize));
    }

    /// The window's clip mask changed (or may have): `clip` is `w × h`, the pass-0 window.
    pub(crate) fn set_clip(&mut self, gpu: &Gpu, clip: &[u8], w: u32, h: u32) {
        if w == 0 || h == 0 || clip.len() != (w * h) as usize {
            return;
        }
        if !self.clip.as_ref().is_some_and(|c| (c.2, c.3) == (w, h)) {
            let t = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fluor clip"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.clip = Some((t, v, w, h));
        }
        let (t, ..) = self.clip.as_ref().unwrap();
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            clip,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
    }

    /// Whether a clip mask of `w × h` is on the GPU (a new window size needs one before the layer can go under it).
    pub(crate) fn has_clip(&self, w: u32, h: u32) -> bool {
        self.clip.as_ref().is_some_and(|c| (c.2, c.3) == (w, h))
    }

    /// Upload what changed of `screen` (the surface buffer, `width × height` of this surface), then draw it onto the surface with `layer` under the chrome inside `rect` (the window on this surface: x, y, width, height in surface pixels), and present.
    /// A buffer of the wrong size (a resize announced before its Resized lands) skips the present.
    pub(crate) fn present(&mut self, gpu: &Gpu, screen: &[u32], layer: Option<&UnderLayer>, rect: (i32, i32, i32, i32)) {
        #[cfg(target_os = "macos")]
        if !self.colorspace_tagged {
            self.colorspace_tagged = true;
            tag_vsf_colorspace(self.ns_view_ptr);
        }
        let (w, h) = (self.config.width as usize, self.config.height as usize);
        if screen.len() != w * h {
            return;
        }
        if let Some((x0, y0, x1, y1)) = self.dirty.take() {
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(screen.as_ptr() as *const u8, screen.len() * 4) };
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.screen, mip_level: 0, origin: wgpu::Origin3d { x: x0 as u32, y: y0 as u32, z: 0 }, aspect: wgpu::TextureAspect::All },
                &bytes[(y0 * w + x0) * 4..],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w as u32 * 4), rows_per_image: Some((y1 - y0) as u32) },
                wgpu::Extent3d { width: (x1 - x0) as u32, height: (y1 - y0) as u32, depth_or_array_layers: 1 },
            );
        }
        // The layer goes under only with a clip of its size: without one the window's shape is unknown.
        let layer = layer.filter(|l| self.has_clip(l.width, l.height) && rect.2 > 0 && rect.3 > 0);
        let exact = layer.is_some_and(|l| (l.width as i32, l.height as i32) == (rect.2, rect.3));
        let u: [u32; 8] = [rect.0 as u32, rect.1 as u32, rect.2 as u32, rect.3 as u32, layer.is_some() as u32, exact as u32, 0, 0];
        gpu.queue.write_buffer(&self.uniform, 0, unsafe { std::slice::from_raw_parts(u.as_ptr() as *const u8, 32) });

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&gpu.device, &self.config);
                match self.surface.get_current_texture() {
                    Ok(t) => t,
                    Err(_) => return,
                }
            }
            Err(_) => return,
        };
        let target = output.texture.create_view(&Default::default());
        let (layer_view, clip_view) = match (layer, &self.clip) {
            (Some(l), Some(c)) => (&l.view, &c.1),
            _ => (&self.blank, &self.blank),
        };
        let bind = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fluor present"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.screen_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(layer_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(clip_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: self.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.sdr) },
            ],
        });
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fluor present") });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fluor present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &target, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        gpu.queue.submit([enc.finish()]);
        output.present();
    }
}

/// Tag the view's CAMetalLayer with the VSF RGB colour space, so Core Animation maps fluor's pixels as VSF RGB on every display.
#[cfg(target_os = "macos")]
fn tag_vsf_colorspace(ns_view_ptr: *mut std::ffi::c_void) {
    if ns_view_ptr.is_null() {
        return;
    }
    use objc2::runtime::{AnyClass, AnyObject};
    unsafe {
        unsafe extern "C" {
            fn CFDataCreate(allocator: *const std::ffi::c_void, bytes: *const u8, length: isize) -> *const std::ffi::c_void;
            fn CGColorSpaceCreateWithICCData(data: *const std::ffi::c_void) -> *const std::ffi::c_void;
            fn CFRelease(cf: *const std::ffi::c_void);
        }
        let view = ns_view_ptr as *mut AnyObject;
        let ca_metal_layer_class = AnyClass::get(c"CAMetalLayer");
        let layer: *mut AnyObject = objc2::msg_send![&*view, layer];
        if layer.is_null() {
            return;
        }
        if let Some(cls) = ca_metal_layer_class {
            let is_metal: bool = objc2::msg_send![&*layer, isKindOfClass: cls];
            if !is_metal {
                return;
            }
        }
        let icc = include_bytes!("vsf_rgb.icc");
        let cf_data = CFDataCreate(std::ptr::null(), icc.as_ptr(), icc.len() as isize);
        if cf_data.is_null() {
            return;
        }
        let cs = CGColorSpaceCreateWithICCData(cf_data);
        if !cs.is_null() {
            // setColorspace: takes a CGColorSpaceRef (a CFTypeRef), passed as a raw object pointer.
            let cs_obj = cs as *mut AnyObject;
            let () = objc2::msg_send![&*layer, setColorspace: &*cs_obj];
            CFRelease(cs);
        }
        CFRelease(cf_data);
    }
}
