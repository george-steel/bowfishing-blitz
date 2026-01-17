use std::{borrow::Cow, fs::File, io::{BufRead, BufReader, Read}, mem::size_of, path::{Path, PathBuf}};

use image::{ImageDecoder, ImageError, ImageResult};
use rgbe::{RGB9E5, RGBE8};
use wgpu::{BindGroupLayoutEntry, ShaderModuleDescriptor, TextureFormat};
use winit::{dpi::PhysicalSize, window::Window};
use bytemuck::{Pod, Zeroable};
use glam::*;

pub mod mip;
pub mod asset;
pub use asset::AssetSource;

use crate::shaders;

pub struct GPUContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub output_format: wgpu::TextureFormat,
    pub output_raw_format: wgpu::TextureFormat,
    pub mip_maker: mip::MipMaker,
    can_clip: bool,
}


impl GPUContext {
    pub async fn with_limits(instance: wgpu::Instance, for_surface: Option<&wgpu::Surface<'_>>, mut features: wgpu::Features, limits: wgpu::Limits) -> Self {
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions{
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: for_surface,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");

        let can_clip = adapter.features().contains(wgpu::Features::CLIP_DISTANCES);
        if can_clip {
            features.insert(wgpu::Features::CLIP_DISTANCES);
        }
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: features,
            required_limits: limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Failed to create device");        

        let output_raw_format = match for_surface {
            None => wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(surface) => {
                let surface_caps = surface.get_capabilities(&adapter);
                surface_caps.formats.iter().copied()
                    .filter(|f| f.is_srgb())
                    .next().unwrap_or(surface_caps.formats[0])
            },
        };
        log::info!("Using output format {:?}", output_raw_format);
        let output_format = match output_raw_format {
            TextureFormat::Rgba8Unorm => TextureFormat::Rgba8UnormSrgb,
            TextureFormat::Bgra8Unorm => TextureFormat::Bgra8UnormSrgb,
            x => x,
        };

        let mip_maker = mip::MipMaker::new(&device);

        GPUContext {
            instance, adapter, device, queue, output_format, output_raw_format, mip_maker, can_clip,
        }
    }

    pub fn configure_surface_target(&self, surface: &wgpu::Surface, size: UVec2) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.output_raw_format,
            present_mode: wgpu::PresentMode::Fifo,
            view_formats: vec![self.output_format],
            ..surface.get_default_config(&self.adapter, size.x.max(1), size.y.max(1)).unwrap()
        };
        surface.configure(&self.device, &surface_config);
    }

    pub fn load_rgbe8_texture(&self, source: &impl AssetSource, path: &str) -> ImageResult<wgpu::Texture> {
        let img = load_rgbe8_png_as_9e5(source, path)?;
        let tex = self.upload_2d_texture(path, wgpu::TextureFormat::Rgb9e5Ufloat, &img);
        Ok(tex)
    }

    pub fn load_rgbe8_cube_texture(&self, source: &impl AssetSource, path: &str, num_levels: u32) -> ImageResult<wgpu::Texture> {
        let img = load_rgbe8_png_as_9e5(source, path)?;
        let tex = self.upload_texture_cube(path, wgpu::TextureFormat::Rgb9e5Ufloat, &img, num_levels);
        Ok(tex)
    }

    pub fn load_r16f_texture(&self, source: &impl AssetSource, path: &str) -> ImageResult<wgpu::Texture> {
        let img = load_png::<u16>(source, path)?;
        let tex = self.upload_2d_texture(path, wgpu::TextureFormat::R16Float, &img);
        Ok(tex)
    }

    pub fn load_r8_texture(&self, source: &impl AssetSource, path: &str) -> ImageResult<wgpu::Texture> {
        let img = load_png::<u8>(source, path)?;
        let tex = self.upload_2d_texture(path, wgpu::TextureFormat::R8Unorm, &img);
        Ok(tex)
    }

    pub fn load_png_texture<P: Pod + Zeroable>(&self, source: &impl AssetSource, path: &str, format: wgpu::TextureFormat) -> ImageResult<wgpu::Texture> {
        let img = load_png::<P>(source, path)?;
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

    pub fn load_texture_make_mips<P: Pod + Zeroable>(&self, source: &impl AssetSource, path: &str, format: wgpu::TextureFormat, num_mips: u32) -> ImageResult<wgpu::Texture> {
        let img = load_png::<P>(source, path)?;
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
        
        self.mip_maker.make_mips(&self, &tex);
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

    pub fn upload_texture_cube<Texel: bytemuck::Pod>(&self, label: &str, format: wgpu::TextureFormat, img: &PlanarImage<Texel>, num_levels: u32) -> wgpu::Texture {
        let texel_size = std::mem::size_of::<Texel>();
        if texel_size != format.block_copy_size(None).unwrap() as usize {
            panic!("texture format must have the same size as the data buffer element")
        }

        let img_width = (img.width as u32);
        let size = wgpu::Extent3d{width: img_width, height: img_width, depth_or_array_layers: 6};
        let tex = self.device.create_texture(&wgpu::TextureDescriptor{
            label: Some(label),
            dimension: wgpu::TextureDimension::D2,
            size, format,
            mip_level_count: num_levels,
            sample_count: 1,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut mip_size = img_width;
        let mut offset: usize = 0;
        for level in 0..num_levels {
            let size = (mip_size * mip_size * 6) as usize;
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo{texture: &tex, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All},
                bytemuck::cast_slice(&img.data[offset..(offset+size)]),
                wgpu::TexelCopyBufferLayout{offset: 0, bytes_per_row: Some((texel_size) as u32 * mip_size), rows_per_image: Some(mip_size)},
                wgpu::Extent3d{width: mip_size, height: mip_size, depth_or_array_layers: 6}
            );
            offset += size;
            mip_size /= 2;
        }
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

    pub fn process_shader_module(&self, name: &str, body: &str) -> wgpu::ShaderModule {
        let mut ctx = nanopre::Context::with_includes(|wanted: &str| -> Result<&'static [u8], ()> {
            for (path, contents) in shaders::INCLUDES {
                if wanted == *path {
                    return Ok(contents.as_bytes())
                }
            }
            Err(())
        });

        ctx.define("CAN_CLIP", if self.can_clip {"1"} else {"0"});
        let shader = nanopre::process_str(body, &mut ctx).unwrap();
        //log::info!("processed shader:\n{}", shader);
        self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some(name), source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader))
        })
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

pub fn slice_view(tex: &wgpu::Texture, slice: u32) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: slice,
        array_layer_count: Some(1),
        ..Default::default()
    })
}

pub fn cube_view(tex: &wgpu::Texture) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
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

pub fn load_png<P: Pod + Zeroable>(source: &impl AssetSource, path: impl AsRef<std::path::Path>) -> ImageResult<PlanarImage<P>> {
    let stream = source.get_reader(path.as_ref()).map_err(ImageError::IoError)?;
    let decoder = image::codecs::png::PngDecoder::new(stream)?;
    let (width, height) = decoder.dimensions();
    let size = (width * height) as usize;
    let mut out = bytemuck::allocation::zeroed_slice_box::<P>(size);
    decoder.read_image(bytemuck::cast_slice_mut(&mut out))?;
    Ok(PlanarImage{ width: width as usize, height: height as usize, data: out})
}

pub fn load_rgbe8_png_as_9e5(source: &impl AssetSource, path: impl AsRef<std::path::Path>) -> ImageResult<PlanarImage<RGB9E5>> {
    let stream = source.get_reader(path.as_ref()).map_err(ImageError::IoError)?;
    let decoder = image::codecs::png::PngDecoder::new(stream)?;
    let (width, height) = decoder.dimensions();
    let out = rgbe::decode_rgbe8_png_as_rgb9e5(decoder)?;
    Ok(PlanarImage{ width: width as usize, height: height as usize, data: out})
}
