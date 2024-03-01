use crate::deferred_renderer::DeferredRenderer;
use crate::deferred_renderer::RenderObject;
use crate::gputil::*;
use crate::camera::*;
use glam::f32::*;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroupEntry, BlendComponent, BufferUsages, ShaderStages};
use std::borrow::Cow;
use std::{default::Default, slice, path::Path};


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainParams {
    pub radius: f32,
    pub z_scale: f32,
    pub grid_size: u32,
}

pub struct TerrainView {
    terrain_pipeline: wgpu::RenderPipeline,
    underwater_terrain_pipeline: wgpu::RenderPipeline,
    water_pipeline: wgpu::RenderPipeline,
    params: TerrainParams,
    params_buf: wgpu::Buffer,
    terrain_bind_group: wgpu::BindGroup,
}

impl TerrainView {
    pub fn new(gpu: &GPUContext, renderer: &DeferredRenderer) -> Self {
        let bg_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("Terrain Uniforms"),
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 1,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 2,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ]
        });
        let pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Terrain Pipeline Layout"),
            bind_group_layouts: &[&renderer.global_bind_layout, &bg_layout],
            push_constant_ranges: &[],
        });
        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("terrain.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(crate::shaders::terrain)),
        });
        let terrain_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "terrain_mesh",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "terrain_frag",
                targets: DeferredRenderer::GBUFFER_TARGETS,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: Some(wgpu::Face::Back),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let underwater_terrain_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "underwater_terrain_mesh",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "underwater_terrain_frag",
                targets: DeferredRenderer::UNDERWATER_GBUFFER_TARGETS,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: Some(wgpu::Face::Back),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let water_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "water_quad",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "water_frag",
                targets: DeferredRenderer::GBUFFER_TARGETS,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: Some(wgpu::Face::Back),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let height_tex = gpu.load_r16f_texture(Path::new("./assets/terrain_heightmap.png")).expect("Failed to load terrain");
        let height_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..wgpu::SamplerDescriptor::default()
        });
        let height_tex_view = height_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let params = TerrainParams{
            radius: 60.0,
            z_scale: 1.0,
            grid_size: 360,
        };

        let params_buf = gpu.device.create_buffer_init(&BufferInitDescriptor{
            label: Some("params_buf"),
            contents: bytemuck::cast_slice(slice::from_ref(&params)),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,

        });

        let terrain_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_bind_group"),
            layout: &bg_layout,
            entries: &[
                BindGroupEntry{binding: 0, resource: params_buf.as_entire_binding()},
                BindGroupEntry{binding: 1, resource: wgpu::BindingResource::TextureView(&height_tex_view)},
                BindGroupEntry{binding: 2, resource: wgpu::BindingResource::Sampler(&height_sampler)},
            ]
        });

        TerrainView {
            terrain_pipeline, underwater_terrain_pipeline, water_pipeline,
            params, params_buf, terrain_bind_group,
        }
    }
}

impl RenderObject for TerrainView {
    fn draw_underwater<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.underwater_terrain_pipeline);
        pass.set_bind_group(1, &self.terrain_bind_group, &[]);
        pass.draw(0..(2 * self.params.grid_size + 2), 0..(self.params.grid_size));
    }

    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_bind_group(1, &self.terrain_bind_group, &[]);

        pass.set_pipeline(&self.water_pipeline);
        pass.draw(0..4, 0..1);

        pass.set_pipeline(&self.terrain_pipeline);
        pass.draw(0..(2 * self.params.grid_size + 2), 0..(self.params.grid_size));
    }
}
