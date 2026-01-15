use bowfishing_blitz::{gputil::mip::MipMaker, *};
use gputil::*;
use std::borrow::Cow;

pub struct FragDisplay {
    shader: wgpu::ShaderModule,
    map_pipeline: wgpu::RenderPipeline,
}

impl FragDisplay {
    pub fn new(ctx: &GPUContext) -> Self {
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("terrain_map.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(crate::shaders::TERRAIN_MAP)),
        });
        let map_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("fullscreen_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("terrain_map"),
                compilation_options: Default::default(),
                targets: &[Some(ctx.output_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None
        });

        FragDisplay {
            shader, map_pipeline,
        }
    }

    pub fn render(&self, gpu: &GPUContext, out: &wgpu::Texture) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        let view = out.create_view(&Default::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..wgpu::RenderPassDescriptor::default()
            });
            
            rpass.set_pipeline(&self.map_pipeline);
            rpass.draw(0..4, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);
    }

    pub fn bake_height(&self, gpu: &GPUContext, texture_size: u32, path: &str)  {
        let height_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: Some("fullscreen_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: Some("terrain_heightmap"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::R16Float.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None
        });

        let size_3d = wgpu::Extent3d {
            width: texture_size,
            height: texture_size,
            depth_or_array_layers: 1,
        };
        let height_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            size: size_3d,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("heightmap"),
            view_formats: &[],
        });
        let height_view = height_texture.create_view(&Default::default());

        // we need to store this for later
        let u16_size = std::mem::size_of::<u16>() as u32;

        let output_buffer_size = (u16_size * texture_size * texture_size) as wgpu::BufferAddress;
        let output_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: Some("heightmap_buffer"),
            mapped_at_creation: false,
        });

        let mip_width = texture_size / 2;
        let all_mips_height = mip_width + (mip_width + 2) / 3;

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &height_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..wgpu::RenderPassDescriptor::default()
            });
            
            rpass.set_pipeline(&height_pipeline);
            rpass.draw(0..4, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                        texture: &height_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(u16_size * texture_size),
                    rows_per_image: None,
                },
            },
            size_3d
        );

        gpu.queue.submit([encoder.finish()]);

        log::info!("starting heightmap render");

        let (tx, rx) = std::sync::mpsc::channel();
        let height_slice = output_buffer.slice(..);
        height_slice.map_async(wgpu::MapMode::Read, move |result| {
            log::info!("got signal");
            tx.send(result).unwrap();
        });

        gpu.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        log::info!("mapped height buffer");
        let height_range = height_slice.get_mapped_range();
        let height_data = bytemuck::cast_slice(&height_range);
        let height_img = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(texture_size, texture_size, height_data).unwrap();
        height_img.save(format!("{}.r16f.png", path)).unwrap();
        log::info!("saved height buffer as PNG");
    }
}
