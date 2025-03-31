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


    pub fn bake_range_mips(&self, gpu: &GPUContext, input_texture: &Texture) -> BakedRangeMips {
        let tex_dims = input_texture.size();
        if tex_dims.width != tex_dims.height {
            panic!("texture must be square");
        }

        let bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("range_mip_bg_layout"),
            entries: &[
                BindGroupLayoutEntry{
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false },
                    count: None,
                },
                BindGroupLayoutEntry{
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("range_mip_pipeline_layput"),
            bind_group_layouts: &[&bg_layout],
            push_constant_ranges: &[],
        });

        let first_mip_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("bake_first_range_mip"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &self.shaders,
                entry_point: Some("bake_mip_fullscreen_tri"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &self.shaders,
                entry_point: Some("bake_first_range_mip"),
                compilation_options: Default::default(),
                targets: &[Some(TextureFormat::Rg16Float.into())],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None
        });

        let next_mip_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("bake_first_range_mip"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &self.shaders,
                entry_point: Some("bake_mip_fullscreen_tri"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &self.shaders,
                entry_point: Some("bake_next_range_mip"),
                compilation_options: Default::default(),
                targets: &[Some(TextureFormat::Rg16Float.into())],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None
        });

        let texture_size = tex_dims.width;
        let num_mips = texture_size.ilog2();
        let mip_width = texture_size / 2;
        let all_mips_height = mip_width + (mip_width + 2) / 3;
        let mip_texture = gpu.device.create_texture(&TextureDescriptor {
            size: Extent3d {
                width: mip_width,
                height: mip_width,
                depth_or_array_layers: 1,
            },
            mip_level_count: num_mips,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC,
            label: Some("range_mips"),
            view_formats: &[],
        });

        // mips below 64x64 are too unaligned to copy_texture_to_buffer, so we write to a storage buffer in the shader.
        let small_mip_buffer_size = 4 * 32 * (32 + 11);
        let small_mip_buffer = gpu.device.create_buffer(&BufferDescriptor {
            size: small_mip_buffer_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            label: Some("small_mips_buffer"),
            mapped_at_creation: false,
        });
        let out_buffer_size = (4 * texture_size * all_mips_height) as BufferAddress;
        let out_buffer = gpu.device.create_buffer(&BufferDescriptor {
            size: out_buffer_size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            label: Some("mips_out_buffer"),
            mapped_at_creation: false,
        });

        let mut encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor {
            label: None,
        });

        let mut mip_num = 0;
        let mut mip_offset = 0;
        let mut small_mips_offset = 0;
        let mut last_mip = input_texture.create_view(&Default::default());
        let mut mip_size = mip_width;
        let mut offsets = Vec::new();

        while mip_size > 0 {
            let out_view = mip_texture.create_view(&TextureViewDescriptor {
                base_mip_level: mip_num,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let bg = gpu.device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &bg_layout,
                entries: &[
                    BindGroupEntry{ binding: 0, resource: BindingResource::TextureView(&last_mip)},
                    BindGroupEntry{ binding: 1, resource: small_mip_buffer.as_entire_binding()},
                ],
            });

            {
                let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &out_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(Color::BLACK),
                            store: StoreOp::Store,
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

            if mip_size > 32 {
                encoder.copy_texture_to_buffer(
                    ImageCopyTexture {
                        aspect: TextureAspect::All,
                                texture: &mip_texture,
                        mip_level: mip_num,
                        origin: Origin3d::ZERO,
                    },
                    ImageCopyBuffer {
                        buffer: &out_buffer,
                        layout: ImageDataLayout {
                            offset: 4 * mip_offset as u64,
                            bytes_per_row: Some(4 * mip_size),
                            rows_per_image: None,
                        },
                    },
                    Extent3d {
                        width: mip_size,
                        height: mip_size,
                        depth_or_array_layers: 1,
                    }
                );
            } else if mip_size == 32 {
                small_mips_offset = mip_offset;
            }

            offsets.push((mip_offset as usize, mip_size as usize));

            last_mip = out_view;
            mip_offset += mip_size * mip_size;
            mip_size /= 2;
            mip_num += 1;
        }
        encoder.copy_buffer_to_buffer(
            &small_mip_buffer, 0,
            &out_buffer, 4 * small_mips_offset as BufferAddress,
            small_mip_buffer_size
        );
        gpu.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        let mip_slice = out_buffer.slice(..);
        mip_slice.map_async(MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        gpu.device.poll(Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let mip_range = mip_slice.get_mapped_range();
        let mip_data: Box<[[f16; 2]]> = bytemuck::cast_slice(&mip_range).to_vec().into_boxed_slice();

        drop(mip_range);
        mip_texture.destroy();
        small_mip_buffer.destroy();
        out_buffer.destroy();

        BakedRangeMips {
            orig_size: texture_size,
            num_mip_levels: num_mips as usize,
            level_offsets: offsets.into_boxed_slice(),
            raw_data: mip_data,
        }
    }
}
