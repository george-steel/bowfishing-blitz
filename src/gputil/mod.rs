use std::{fs::File, mem::size_of, path::Path};

use image::{ImageDecoder, ImageError, ImageResult};
use wgpu::BindGroupLayoutEntry;
use winit::{dpi::PhysicalSize, window::Window};
use bytemuck::{Pod, Zeroable};
use glam::*;

pub mod mip;

pub struct GPUContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub output_format: wgpu::TextureFormat,
}


impl GPUContext {
    pub async fn with_default_limits(instance: wgpu::Instance, for_surface: Option<&wgpu::Surface<'_>>, features: wgpu::Features) -> Self {
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions{
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: for_surface,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Failed to create device");

        let output_format = match for_surface {
            None => wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(surface) => {
                let surface_caps = surface.get_capabilities(&adapter);
                surface_caps.formats.iter().copied()
                    .filter(|f| f.is_srgb())
                    .next().unwrap_or(surface_caps.formats[0])
            },
        };
        log::info!("Using output format {:?}", output_format);

        GPUContext {
            instance, adapter, device, queue, output_format
        }
    }

    pub fn configure_surface_target(&self, surface: &wgpu::Surface, size: UVec2) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.output_format,
            present_mode: wgpu::PresentMode::Fifo,
            ..surface.get_default_config(&self.adapter, size.x.max(1), size.y.max(1)).unwrap()
        };
        surface.configure(&self.device, &surface_config);
    }

    pub fn load_rgbe8_texture(&self, path: &str) -> ImageResult<wgpu::Texture> {
        let (width, height, data) = rgbe::load_rgbe8_png_file_as_rgb9e5(Path::new(path.into()))?;
        let img = PlanarImage {width: width as usize, height: height as usize, data};
        let tex = self.upload_2d_texture(path, wgpu::TextureFormat::Rgb9e5Ufloat, &img);
        Ok(tex)
    }

    pub fn load_r16f_texture(&self, path: &str) -> ImageResult<wgpu::Texture> {
        let img = load_png::<u16>(path)?;
        let tex = self.upload_2d_texture(path, wgpu::TextureFormat::R16Float, &img);
        Ok(tex)
    }

    pub fn load_r8_texture(&self, path: &str) -> ImageResult<wgpu::Texture> {
        let img = load_png::<u8>(path)?;
        let tex = self.upload_2d_texture(path, wgpu::TextureFormat::R8Unorm, &img);
        Ok(tex)
    }

    pub fn load_png_texture<P: Pod + Zeroable>(&self, path: &str, format: wgpu::TextureFormat) -> ImageResult<wgpu::Texture> {
        let img = load_png::<P>(path)?;
        let tex = self.upload_2d_texture(path, format, &img);
        Ok(tex)
    }

    pub fn upload_2d_texture<Texel: bytemuck::Pod>(&self, label: &str, format: wgpu::TextureFormat, img: &PlanarImage<Texel>) -> wgpu::Texture {
        let texel_size = std::mem::size_of::<Texel>();
        if texel_size != format.block_copy_size(None).unwrap() as usize {
            panic!("texture format must have the same size as the data buffer element")
        }

        let size = wgpu::Extent3d{width: img.width as u32, height: img.height as u32, depth_or_array_layers: 1};
        let tex = self.device.create_texture(&wgpu::TextureDescriptor{
            label: Some(label),
            dimension: wgpu::TextureDimension::D2,
            size, format,
            mip_level_count: 1,
            sample_count: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo{texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All},
            bytemuck::cast_slice(&img.data),
            wgpu::TexelCopyBufferLayout{offset: 0, bytes_per_row: Some((texel_size * img.width) as u32), rows_per_image: Some(img.height as u32)},
            size);
        tex
    }

    pub fn load_texture_make_mips<P: Pod + Zeroable>(&self, path: &str, format: wgpu::TextureFormat, num_mips: u32) -> ImageResult<wgpu::Texture> {
        let img = load_png::<P>(path)?;
        let texel_size = std::mem::size_of::<P>();
        if texel_size != format.block_copy_size(None).unwrap() as usize {
            panic!("texture format must have the same size as the data buffer element")
        }

        let size = wgpu::Extent3d{width: img.width as u32, height: img.height as u32, depth_or_array_layers: 1};
        let tex = self.device.create_texture(&wgpu::TextureDescriptor{
            label: Some(path),
            dimension: wgpu::TextureDimension::D2,
            size, format,
            mip_level_count: num_mips,
            sample_count: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo{texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All},
            bytemuck::cast_slice(&img.data),
            wgpu::TexelCopyBufferLayout{offset: 0, bytes_per_row: Some((texel_size * img.width) as u32), rows_per_image: Some(img.height as u32)},
            size);
        
        mip::MipMaker::get(&self).make_mips(&self, &tex);
        Ok(tex)
    }

    pub fn upload_texture_atlas<Texel: bytemuck::Pod>(&self, label: &str, format: wgpu::TextureFormat, img: &PlanarImage<Texel>, num_tiles: u32) -> wgpu::Texture {
        let texel_size = std::mem::size_of::<Texel>();
        if texel_size != format.block_copy_size(None).unwrap() as usize {
            panic!("texture format must have the same size as the data buffer element")
        }

        let height =(img.height as u32) / num_tiles;
        let size = wgpu::Extent3d{width: img.width as u32, height, depth_or_array_layers: num_tiles};
        let tex = self.device.create_texture(&wgpu::TextureDescriptor{
            label: Some(label),
            dimension: wgpu::TextureDimension::D2,
            size, format,
            mip_level_count: 1,
            sample_count: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo{texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All},
            bytemuck::cast_slice(&img.data),
            wgpu::TexelCopyBufferLayout{offset: 0, bytes_per_row: Some((texel_size * img.width) as u32), rows_per_image: Some(height)},
            size);
        tex
    }

    pub fn create_empty_texture(&self, size: wgpu::Extent3d, format: wgpu::TextureFormat, label: &'static str) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = self.device.create_texture(&wgpu::TextureDescriptor{
            label: Some(label),
            size, format,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        (tex, view)
    }
}

pub fn reverse_z() -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState{
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Greater,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

pub fn frag_tex_2d(n: u32) -> BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry{
        binding: n,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false
        },
        count: None,
    }
}

pub fn window_size(window: &Window) -> UVec2 {
    let isize = window.inner_size();
    uvec2(isize.width, isize.height)
}

pub fn extent_2d(size: UVec2) -> wgpu::Extent3d {
    wgpu::Extent3d { width: size.x, height: size.y, depth_or_array_layers: 1}
}

pub fn smoothstep(t: f32) -> f32 {
    let ct = t.clamp(0.0, 1.0);
    ct * ct * (3.0 - 2.0 * ct)
}

#[derive(Clone, Debug)]
pub struct PlanarImage<P> {
    pub width: usize,
    pub height: usize,
    pub data: Box<[P]>,
}

impl<P: Copy> PlanarImage<P> {
    pub fn pixel_at(&self, pos: UVec2) -> P {
        let x = pos.x as usize;
        let y = pos.y as usize;
        self.data[self.width * y + x]
    }

    pub fn sample_nearest(&self, uv: Vec2, wrap_x: bool, wrap_y: bool) -> P {
        let dim = vec2(self.width as f32, self.height as f32);
        let px = (uv * dim).floor().as_ivec2();
        self.pixel_at(self.wrap_or_clamp(px, wrap_x, wrap_y))
    }

    pub fn wrap_or_clamp(&self, pos: IVec2, wrap_x: bool, wrap_y: bool) -> UVec2 {
        let wrapped = pos.rem_euclid(ivec2(self.width as i32, self.height as i32));
        let clamped = pos.clamp(IVec2::ZERO, ivec2(self.width as i32 - 1, self.height as i32 - 1));
        let x = if wrap_x {wrapped.x} else {clamped.x};
        let y = if wrap_y {wrapped.y} else {clamped.y};
        uvec2(x as u32, y as u32)
    }
}

impl<P: Copy + Into<f32>> PlanarImage<P> {
    pub fn sample_bilinear_f32(&self, uv: Vec2, wrap_x: bool, wrap_y: bool) -> f32 {
        let dim = vec2(self.width as f32, self.height as f32);
        let px = (uv * dim - 0.5).floor().as_ivec2();
        let cell_uv = (uv * dim - 0.5).fract();

        let nw = self.pixel_at(self.wrap_or_clamp(px, wrap_x, wrap_y)).into();
        let sw = self.pixel_at(self.wrap_or_clamp(px + ivec2(0, 1), wrap_x, wrap_y)).into();
        let ne = self.pixel_at(self.wrap_or_clamp(px + ivec2(1, 0), wrap_x, wrap_y)).into();
        let se = self.pixel_at(self.wrap_or_clamp(px + ivec2(1, 1), wrap_x, wrap_y)).into();
        
        let n = f32::lerp(nw, ne, cell_uv.x);
        let s = f32::lerp(sw, se, cell_uv.x);
        f32::lerp(n, s, cell_uv.y)
    }

    pub fn sample_grad_f32(&self, uv: Vec2, wrap_x: bool, wrap_y: bool) -> Vec2 {
        let delta_u = vec2(1.0 / self.width as f32, 0.0);
        let delta_v = vec2(0.0, 1.0 / self.height as f32);
        let n = self.sample_bilinear_f32(uv - delta_v, wrap_x, wrap_y);
        let s = self.sample_bilinear_f32(uv + delta_v, wrap_x, wrap_y);
        let w = self.sample_bilinear_f32(uv - delta_u, wrap_x, wrap_y);
        let e = self.sample_bilinear_f32(uv + delta_u, wrap_x, wrap_y);

        vec2(e - w, s - n) / (2.0 * (delta_u + delta_v))
    }
}

pub fn load_png<P: Pod + Zeroable>(path: impl AsRef<std::path::Path>) -> ImageResult<PlanarImage<P>> {
    let file = File::open(path).map_err(ImageError::IoError)?;
    let decoder = image::codecs::png::PngDecoder::new(file)?;
    let (width, height) = decoder.dimensions();
    let size = (width * height) as usize;
    let mut out = bytemuck::allocation::zeroed_slice_box::<P>(size);
    decoder.read_image(bytemuck::cast_slice_mut(&mut out))?;
    Ok(PlanarImage{ width: width as usize, height: height as usize, data: out})
}
