use crate::gputil::*;
use glam::f32::*;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{BindGroupEntry, BufferUsages};
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
    pipeline: wgpu::RenderPipeline,
    params: TerrainParams,
    params_buf: wgpu::Buffer,
    camera_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    last_size: wgpu::Extent3d,
    depth_tex: Option<wgpu::Texture>,
    init_time: Instant
}

impl TerrainView {
    pub fn new(ctx: &GPUContext, now: Instant) -> Self {
        let shader = ctx.device.create_shader_module(wgpu::include_wgsl!("noise.wgsl"));
        let pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
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

        let transform = Mat4::from_translation(vec3(0.0, 0.0, 0.0));
        let params = TerrainParams{
            transform,
            inv_transform: transform.inverse(),
            uv_center: Vec2::ZERO,
            uv_radius: 6.0,
            grid_size: 100,
        };

        let camera = Mat4::look_at_rh(vec3(0.0, -10.0, 4.0), Vec3::ZERO, Vec3::Z);

        let params_buf = ctx.device.create_buffer_init(&BufferInitDescriptor{
            label: Some("params_buf"),
            contents: bytemuck::cast_slice(slice::from_ref(&params)),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,

        });
        let camera_buf = ctx.device.create_buffer_init(&BufferInitDescriptor{
            label: Some("camera_buf"),
            contents: bytemuck::cast_slice(slice::from_ref(&camera)),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain_bind_group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry{binding: 0, resource: params_buf.as_entire_binding()},
                BindGroupEntry{binding: 1, resource: camera_buf.as_entire_binding()},
            ]
        });

        TerrainView {
            pipeline, params,
            params_buf, camera_buf, bind_group,
            last_size: wgpu::Extent3d{width: 0, height: 0, depth_or_array_layers: 0},
            depth_tex: None,
            init_time: now,
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

    pub fn render(&mut self, ctx: &GPUContext, encoder: &mut wgpu::CommandEncoder, out: &wgpu::Texture, now: Instant) {
        let size = out.size();
        let aspect = size.width as f32 / size.height as f32;

        let camera_angle = 0.5 * now.duration_since(self.init_time).as_secs_f32();
        let camera_loc = Mat4::from_rotation_z(camera_angle).transform_point3(vec3(0.0, -10.0, 4.0));
        let camera = Mat4::perspective_rh(0.5, aspect, 0.5, 100.0)
                * Mat4::look_at_rh(camera_loc, Vec3::ZERO, Vec3::Z);
        ctx.queue.write_buffer(&self.camera_buf, 0, bytemuck::cast_slice(slice::from_ref(&camera)));

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
        
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..(2 * self.params.grid_size + 2), 0..(self.params.grid_size));
    }
}