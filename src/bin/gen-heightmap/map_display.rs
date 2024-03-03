use bowfishing_blitz::*;
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
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(crate::shaders::terrain_map)),
        });
        let map_pipeline = ctx.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "fullscreen_quad",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "terrain_map",
                targets: &[Some(ctx.output_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
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
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
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
                entry_point: "fullscreen_quad",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: "terrain_heightmap",
                targets: &[Some(wgpu::TextureFormat::R16Float.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let mip_bg_layout = gpu.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("Terrain Uniforms"),
            entries: &[
                wgpu::BindGroupLayoutEntry{
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry{
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let mip_pipeline_layout = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mip_pipeline_layput"),
            bind_group_layouts: &[&mip_bg_layout],
            push_constant_ranges: &[],
        });

        let first_mip_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&mip_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: "fullscreen_tri",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: "first_range_mip",
                targets: &[Some(wgpu::TextureFormat::Rg16Float.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let next_mip_pipeline = gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&mip_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &self.shader,
                entry_point: "fullscreen_tri",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &self.shader,
                entry_point: "next_range_mip",
                targets: &[Some(wgpu::TextureFormat::Rg16Float.into())],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
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

        let num_mips = texture_size.ilog2();
        let mip_width = texture_size / 2;
        let all_mips_height = mip_width + (mip_width + 2) / 3;
        let mip_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: mip_width,
                height: mip_width,
                depth_or_array_layers: 1,
            },
            mip_level_count: num_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("heightmap"),
            view_formats: &[],
        });
        let mip_buffer_size = (4 * texture_size * all_mips_height) as wgpu::BufferAddress;
        let mip_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            size: mip_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
            label: Some("heightmap_buffer"),
            mapped_at_creation: false,
        });

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &height_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            rpass.set_pipeline(&height_pipeline);
            rpass.draw(0..4, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                        texture: &height_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(u16_size * texture_size),
                    rows_per_image: None,
                },
            },
            size_3d
        );

        let mut mip_num = 0;
        let mut mip_offset = 0;
        let mut last_mip = height_texture.create_view(&Default::default());
        let mut mip_size = mip_width;
        
        while mip_size > 0 {
            let out_view = mip_texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip_num,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &mip_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry{ binding: 0, resource: wgpu::BindingResource::TextureView(&last_mip)},
                    wgpu::BindGroupEntry{ binding: 1, resource: mip_buffer.as_entire_binding()},
                ],
            });

            {
                let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &out_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                
                if mip_num == 0 {
                    rpass.set_pipeline(&first_mip_pipeline);
                } else {
                    rpass.set_pipeline(&next_mip_pipeline);
                }
                rpass.set_bind_group(0, &bg, &[]);
                rpass.draw(0..4, mip_num..(mip_num + 1));
            }

            last_mip = out_view;
            mip_offset += mip_size * mip_size;
            mip_size /= 2;
            mip_num += 1;
        }
        gpu.queue.submit([encoder.finish()]);

        log::info!("starting heightmap render");

        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        let height_slice = output_buffer.slice(..);
        height_slice.map_async(wgpu::MapMode::Read, move |result| {
            log::info!("got signal");
            tx.send(result).unwrap();
        });
        let mip_slice = mip_buffer.slice(..);
        mip_slice.map_async(wgpu::MapMode::Read, move |result| {
            log::info!("got signal");
            tx2.send(result).unwrap();
        });
        gpu.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();

        log::info!("mapped height buffer");
        let height_range = height_slice.get_mapped_range();
        let height_data = bytemuck::cast_slice(&height_range);
        let height_img = image::ImageBuffer::<image::Luma<u16>, _>::from_raw(texture_size, texture_size, height_data).unwrap();
        height_img.save(format!("{}.r16f.png", path)).unwrap();

        let mip_range = mip_slice.get_mapped_range();
        let mip_data = bytemuck::cast_slice(&mip_range);
        let mip_img = image::ImageBuffer::<image::LumaA<u16>, _>::from_raw(mip_width, all_mips_height, mip_data).unwrap();
        mip_img.save(format!("{}.range-mip.r16f.png", path)).unwrap();
        log::info!("saved height buffer as PNG");
    }
}