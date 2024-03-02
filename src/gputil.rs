use std::{fs::File, path::Path};

use image::{ImageDecoder, ImageError, ImageResult};
use winit::{dpi::PhysicalSize, window::Window};
use glam::*;

pub struct GPUContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub output_format: wgpu::TextureFormat,
}


impl GPUContext {
    pub async fn with_default_limits(instance: wgpu::Instance, for_surface: Option<&wgpu::Surface<'_>>) -> Self {
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions{
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: for_surface,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::RG11B10UFLOAT_RENDERABLE,
            required_limits: wgpu::Limits::default(),
        }, None)
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

    pub fn load_rgbe8_texture(&self, path: &Path) -> ImageResult<wgpu::Texture> {
        let (width, height, data) = rgbe::load_rgbe8_png_file_as_rgb9e5(path)?;
        let size = wgpu::Extent3d{width, height, depth_or_array_layers: 1};
        let tex = self.device.create_texture(&wgpu::TextureDescriptor{
                label: path.to_str(),
                dimension: wgpu::TextureDimension::D2,
                size,
                mip_level_count: 1,
                sample_count: 1,
                format: wgpu::TextureFormat::Rgb9e5Ufloat,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
        });
        self.queue.write_texture(wgpu::ImageCopyTexture{
            texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All
        }, bytemuck::cast_slice(&data), wgpu::ImageDataLayout{
            offset: 0, bytes_per_row: Some(4 * width), rows_per_image: Some(height),
        }, size);
        Ok(tex)
    }

    pub fn load_r16f_texture(&self, path: &Path) -> ImageResult<wgpu::Texture> {
        let (width, height, data) = load_png16(path)?;
        let size = wgpu::Extent3d{width, height, depth_or_array_layers: 1};
        let tex = self.device.create_texture(&wgpu::TextureDescriptor{
                label: path.to_str(),
                dimension: wgpu::TextureDimension::D2,
                size,
                mip_level_count: 1,
                sample_count: 1,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
        });
        self.queue.write_texture(wgpu::ImageCopyTexture{
            texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All
        }, bytemuck::cast_slice(&data), wgpu::ImageDataLayout{
            offset: 0, bytes_per_row: Some(2 * width), rows_per_image: Some(height),
        }, size);
        Ok(tex)
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

pub fn window_size(window: &Window) -> UVec2 {
    let isize = window.inner_size();
    uvec2(isize.width, isize.height)
}

pub fn extent_2d(size: UVec2) -> wgpu::Extent3d {
    wgpu::Extent3d { width: size.x, height: size.y, depth_or_array_layers: 1}
}

pub fn load_png16(path: &Path) -> ImageResult<(u32, u32, Box<[u16]>)> {
    let file = File::open(path).map_err(ImageError::IoError)?;
    let decoder = image::codecs::png::PngDecoder::new(file)?;
    let (width, height) = decoder.dimensions();
    let size = (width * height) as usize;
    let mut out = bytemuck::allocation::zeroed_slice_box::<u16>(size);
    decoder.read_image(bytemuck::cast_slice_mut(&mut out))?;
    Ok((width, height, out))
}

pub fn load_png32(path: &Path) -> ImageResult<(u32, u32, Box<[u32]>)> {
    let file = File::open(path).map_err(ImageError::IoError)?;
    let decoder = image::codecs::png::PngDecoder::new(file)?;
    let (width, height) = decoder.dimensions();
    let size = (width * height) as usize;
    let mut out = bytemuck::allocation::zeroed_slice_box::<u32>(size);
    decoder.read_image(bytemuck::cast_slice_mut(&mut out))?;
    Ok((width, height, out))
}
