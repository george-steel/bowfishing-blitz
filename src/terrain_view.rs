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
    //water_pipeline: wgpu::RenderPipeline,
    //sky_pipeline: wgpu::RenderPipeline,
    params: TerrainParams,
    params_buf: wgpu::Buffer,
    terrain_bind_group: wgpu::BindGroup,
    //sky_bind_group: wgpu::BindGroup,
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

        /*let water_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                targets: &[Some(wgpu::ColorTargetState{
                    format: ctx.output_format,
                    blend: Some(wgpu::BlendState{
                        color: BlendComponent::OVER,
                        alpha: BlendComponent::default(),
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: Some(wgpu::Face::Back),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });*/

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

        /*let sky_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "sky_vert",
                buffers: &[]
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "sky_frag",
                targets: &[Some(ctx.output_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let sky_tex = ctx.load_rgbe8_texture(Path::new("./assets/sky-equirect.rgbe8.png")).expect("Failed to load sky");
        let sky_sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..wgpu::SamplerDescriptor::default()
        });
        let sky_tex_view = sky_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sky_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("sky_bind_group"),
            layout: &sky_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry{binding: 0, resource: camera_buf.as_entire_binding()},
                BindGroupEntry{binding: 1, resource: wgpu::BindingResource::TextureView(&sky_tex_view)},
                BindGroupEntry{binding: 2, resource: wgpu::BindingResource::Sampler(&sky_sampler)},
            ]
        });*/

        TerrainView {
            terrain_pipeline,  params,
            params_buf,terrain_bind_group,
        }
    }
}

impl RenderObject for TerrainView {
    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut wgpu::RenderPass<'a>) {
        pass.set_pipeline(&self.terrain_pipeline);
        pass.set_bind_group(1, &self.terrain_bind_group, &[]);
        pass.draw(0..(2 * self.params.grid_size + 2), 0..(self.params.grid_size));
    }
}
