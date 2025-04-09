use crate::gputil::*;
use crate::shaders;

use wgpu::*;
use glam::*;
use std::borrow::Cow;
use std::sync::Once;
use std::sync::OnceLock;
use half::f16;

// structure used for collision with heightmaps,
// each level stores the range of imput values within its texel
pub struct BakedRangeMips {
    pub orig_size: u32,
    pub num_mip_levels: usize,
    pub level_offsets: Box<[(usize, usize)]>,
    pub raw_data: Box<[[f16; 2]]>,
}

impl BakedRangeMips {
    pub fn sub_bins(ij: UVec2) -> [UVec2; 4] {
        [uvec2(2*ij.x,     2*ij.y), uvec2(2*ij.x,     2*ij.y + 1),
         uvec2(2*ij.x + 1, 2*ij.y), uvec2(2*ij.x + 1, 2*ij.y + 1)]
    } 

    pub fn get_range_uv(&self, uv: Vec2, level: usize) -> (f32, f32) {
        let bin = (uv * self.orig_size as f32).floor().as_uvec2() / (1u32 << level);
        self.get_bin(bin, level)
    }

    pub fn get_bin(&self, bin: UVec2, level: usize) -> (f32, f32) {
        if level == 0 || level > self.num_mip_levels {
            panic!{"mip level out of bounds"}
        }
        let (offset, stride) = self.level_offsets[level];
        let idx = offset + stride * (bin.y as usize) + (bin.x as usize);
        let range = self.raw_data[idx];
        (range[0].to_f32(), range[1].to_f32())
    }
}

pub struct MipMaker {
    shaders: ShaderModule,
}

impl MipMaker {
    fn new(gpu: &GPUContext) -> Self {
        let shaders = gpu.device.create_shader_module(ShaderModuleDescriptor{
            label: Some("mip.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(crate::shaders::MIP)),
        });

        Self {shaders}
    }

    pub fn get(gpu: &GPUContext) -> &'static Self {
        static INST: OnceLock<MipMaker> = OnceLock::new();
        INST.get_or_init(||{Self::new(gpu)})
    }

    // largely copied from wgpu documentation
    pub fn make_mips(&self, gpu: &GPUContext, tex: &wgpu::Texture) {
        let mip_count = tex.mip_level_count();
        if mip_count <= 1 { return; }

        let mip_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &self.shaders,
                entry_point: Some("mip_vert"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shaders,
                entry_point: Some("mip_frag"),
                compilation_options: Default::default(),
                targets: &[Some(tex.format().into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None
        });
        let bg_layout = mip_pipeline.get_bind_group_layout(0);

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mip"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let mut mip_size = tex.size();
        mip_size.width /= 2;
        mip_size.height /= 2;

        let temp_tex = gpu.device.create_texture(&TextureDescriptor {
            label: Some("temp_mip_tex"),
            size: mip_size,
            mip_level_count: mip_count - 1,
            dimension: TextureDimension::D2,
            sample_count: 1,
            format: tex.format(),
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let mut encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("mip_encoder") });

        for i in 0..(mip_count-1) {
            let in_view = tex.create_view(&TextureViewDescriptor {
                base_mip_level: i,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let out_view = temp_tex.create_view(&TextureViewDescriptor {
                base_mip_level: i,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let bg = gpu.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bg_layout,
                entries: &[
                    BindGroupEntry {binding: 0, resource: BindingResource::TextureView(&in_view)},
                    BindGroupEntry {binding: 1, resource: BindingResource::Sampler(&sampler)},
                ]
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&mip_pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.draw(0..3, 0..1);

            drop(rpass);
            encoder.copy_texture_to_texture(
                ImageCopyTexture {texture: &temp_tex, mip_level: i, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                ImageCopyTexture {texture: &tex, mip_level: i+1, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                mip_size,
            );

            mip_size.width /= 2;
            mip_size.height /= 2;
        }
    
        gpu.queue.submit([encoder.finish()]);
        temp_tex.destroy();
    }
}
