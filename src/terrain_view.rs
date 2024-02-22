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
    pub transform: Mat4,
    pub inv_transform: Mat4,
    pub uv_center: Vec2,
    pub uv_radius: f32,
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
    pub fn new(ctx: &GPUContext, renderer: &DeferredRenderer) -> Self {
        let bg_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("Terrain Uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry{
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }]
        });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("Terrain Pipeline Layout"),
            bind_group_layouts: &[&renderer.global_bind_layout, &bg_layout],
            push_constant_ranges: &[],
        });
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("terrain.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(crate::shaders::terrain)),
        });
        let terrain_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        let underwater_terrain_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        let water_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        let transform = Mat4::from_translation(vec3(0.0, 0.0, 0.0));
        let params = TerrainParams{
            transform,
            inv_transform: transform.inverse(),
            uv_center: Vec2::ZERO,
            uv_radius: 10.0,
            grid_size: 200,
        };

        let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor{
            label: Some("params_buf"),
            contents: bytemuck::cast_slice(slice::from_ref(&params)),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,

        });

        let terrain_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_bind_group"),
            layout: &terrain_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry{binding: 0, resource: params_buf.as_entire_binding()},
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
