use crate::gputil::*;
use crate::camera::*;
use glam::f32::*;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroupEntry, BlendComponent, BufferUsages, ShaderStages};
use std::{default::Default, slice, time::Instant};


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
    water_pipeline: wgpu::RenderPipeline,
    params: TerrainParams,
    params_buf: wgpu::Buffer,
    camera_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    last_size: wgpu::Extent3d,
    depth_tex: Option<wgpu::Texture>,
}

impl TerrainView {
    pub fn new(ctx: &GPUContext, camera: &CameraController) -> Self {
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
            }, wgpu::BindGroupLayoutEntry{
                binding: 1,
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
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });
        let shader = ctx.device.create_shader_module(wgpu::include_wgsl!("noise.wgsl"));
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
                targets: &[Some(ctx.output_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                cull_mode: Some(wgpu::Face::Back),
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState{
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
            depth_stencil: Some(wgpu::DepthStencilState{
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
        let camera_buf = ctx.device.create_buffer_init(&BufferInitDescriptor{
            label: Some("camera_buf"),
            contents: bytemuck::cast_slice(slice::from_ref(&camera.camera(1.0))),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_bind_group"),
            layout: &terrain_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry{binding: 0, resource: params_buf.as_entire_binding()},
                BindGroupEntry{binding: 1, resource: camera_buf.as_entire_binding()},
            ]
        });

        TerrainView {
            terrain_pipeline, water_pipeline, params,
            params_buf, camera_buf, bind_group,
            last_size: wgpu::Extent3d{width: 0, height: 0, depth_or_array_layers: 0},
            depth_tex: None,
        }
    }

    fn resize(&mut self, ctx: &GPUContext, size: wgpu::Extent3d) {
        let depth = ctx.device.create_texture(&wgpu::TextureDescriptor{
            label: Some("depth"),
            mip_level_count: 1,
            size: size,
            dimension: wgpu::TextureDimension::D2,
            sample_count: 1,
            format: wgpu::TextureFormat::Depth32Float,
            view_formats: &[],
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        self.last_size = size;
        self.depth_tex = Some(depth);
    }

    pub fn render(&mut self, ctx: &GPUContext, encoder: &mut wgpu::CommandEncoder, out: &wgpu::Texture, camera: &CameraController) {
        let size = out.size();
        let aspect = size.width as f32 / size.height as f32;

        ctx.queue.write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(slice::from_ref(&camera.camera(aspect))));

        if size != self.last_size || self.depth_tex.is_none() {
            self.resize(ctx, size);
        }

        let col_view = out.create_view(&Default::default());
        let depth_view = self.depth_tex.as_ref().unwrap().create_view(&Default::default());
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &col_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment{
                view: &depth_view,
                depth_ops: Some(wgpu::Operations{
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        rpass.set_pipeline(&self.terrain_pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..(2 * self.params.grid_size + 2), 0..(self.params.grid_size));

        rpass.set_pipeline(&self.water_pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }
}