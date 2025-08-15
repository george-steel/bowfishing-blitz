use bowfishing_blitz::{gputil::mip::MipMaker, *};
use gputil::*;
use std::{borrow::Cow, default};
use wgpu::{wgt::TextureDescriptor, *};

pub struct DFGBaker {
    shaders: ShaderModule,

    dfg_pipeline: RenderPipeline,
    dirs_pipeline: RenderPipeline,
}

pub struct DFGTables {
    pub dfg: Texture,
    pub trans_dfg: Texture,
    pub dirs: Texture,
}

impl DFGBaker {
    const OUTPUT_LINEAR: bool = true;
    const DFG1_FORMAT: TextureFormat = if Self::OUTPUT_LINEAR {TextureFormat::Rgba16Unorm} else {TextureFormat::Rg16Float};
    const OUTPUT_FORMAT: TextureFormat = if Self::OUTPUT_LINEAR {TextureFormat::Rgba16Unorm} else {TextureFormat::Rgba16Float};
    const DFG1_EXT: &'static str = if Self::OUTPUT_LINEAR {"rgba16un"} else {"rg16f"};
    const OUTPUT_EXT: &'static str = if Self::OUTPUT_LINEAR {"rgba16un"} else {"rgba16f"};
    pub const LIN_CORRECTION: f64 = if Self::OUTPUT_LINEAR {8.0} else {1.0};

    const COMP_OPTIONS: PipelineCompilationOptions<'static> = PipelineCompilationOptions {
        constants: &[("LIN_CORRECTION", Self::LIN_CORRECTION)],
        zero_initialize_workgroup_memory: true,
    };

    const DFG1_WORD_SIZE: u32 = if Self::OUTPUT_LINEAR {8} else {4};
    const DFG_DIM_NV: u32 = 128;
    const DFG_DIM_ROUGH: u32 = 128;
    const DFG1_ROW_SIZE: u32 = Self::DFG_DIM_NV * Self::DFG1_WORD_SIZE;
    const DFG1_BUF_SIZE: u32 = Self::DFG1_ROW_SIZE * Self::DFG_DIM_ROUGH;
    const DFG2_ROW_SIZE: u32 = Self::DFG_DIM_NV * 8;
    const DFG2_BUF_SIZE: u32 = Self::DFG_DIM_NV * Self::DFG_DIM_ROUGH * 8;

    const DIRS_DIM_NV: u32 = 128;
    const DIRS_DIM_ROUGH: u32 = 128;
    const DIRS_ROW_SIZE: u32 = Self::DIRS_DIM_NV * 8;
    const DIRS_BUF_SIZE: u32 = Self::DIRS_DIM_NV * Self::DIRS_DIM_ROUGH * 8 * 4;

    pub fn new(gpu: &GPUContext) -> Self {
        let shaders = gpu.device.create_shader_module(ShaderModuleDescriptor{
            label: Some("integral_filter.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("integral_filter.wgsl"))),
        });

        let pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("integral_layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[]
        });

        let dfg_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("dfg_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("fullscreen_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: Some("integrate_DFG_LUT"),
                compilation_options: Self::COMP_OPTIONS,
                targets: &[Some(Self::DFG1_FORMAT.into()), Some(Self::OUTPUT_FORMAT.into())],
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

        let dirs_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("idirs_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("fullscreen_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: Some("integrate_Dirs_LUT"),
                compilation_options: Self::COMP_OPTIONS,
                targets: &[Some(Self::OUTPUT_FORMAT.into()), Some(Self::OUTPUT_FORMAT.into()), Some(Self::OUTPUT_FORMAT.into()), Some(Self::OUTPUT_FORMAT.into())],
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


        DFGBaker {
            shaders,
            dfg_pipeline, dirs_pipeline,
        }
    }

    pub fn integrate_dfg_lut(&self, gpu: &GPUContext) -> DFGTables {
        let dfg1_out_tex = gpu.device.create_texture(&TextureDescriptor{
            label: Some("dfg_lut_tex"),
            size: Extent3d { width: Self::DFG_DIM_NV, height: Self::DFG_DIM_ROUGH, depth_or_array_layers: 1 },
            format: Self::DFG1_FORMAT,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let dfg1_out_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("dfg_lut_out_buf"),
            size: Self::DFG1_BUF_SIZE as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false 
        });

        let dfg2_out_tex = gpu.device.create_texture(&TextureDescriptor{
            label: Some("dfg_lut_tex"),
            size: Extent3d { width: Self::DFG_DIM_NV, height: Self::DFG_DIM_ROUGH, depth_or_array_layers: 1 },
            format: Self::OUTPUT_FORMAT,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let dfg2_out_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("dfg_lut_out_buf"),
            size: Self::DFG2_BUF_SIZE as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false 
        });

        let dirs_out_tex = gpu.device.create_texture(&TextureDescriptor{
            label: Some("dfg_lut_tex"),
            size: Extent3d { width: Self::DIRS_DIM_NV, height: Self::DIRS_DIM_ROUGH, depth_or_array_layers: 4 },
            format: Self::OUTPUT_FORMAT,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let dirs_out_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("dfg_lut_out_buf"),
            size: Self::DIRS_BUF_SIZE as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false 
        });

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &slice_view(&dfg1_out_tex, 0),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &slice_view(&dfg2_out_tex, 0),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            rpass.set_pipeline(&self.dfg_pipeline);
            rpass.draw(0..4, 0..1);
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &slice_view(&dirs_out_tex, 0),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &slice_view(&dirs_out_tex, 1),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &slice_view(&dirs_out_tex, 2),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &slice_view(&dirs_out_tex, 3),
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            rpass.set_pipeline(&self.dirs_pipeline);
            rpass.draw(0..4, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                        texture: &dfg1_out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &dfg1_out_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(Self::DFG1_ROW_SIZE),
                    rows_per_image: Some(Self::DFG_DIM_ROUGH),
                },
            },
            dfg1_out_tex.size(),
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                        texture: &dfg2_out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &dfg2_out_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(Self::DFG2_ROW_SIZE),
                    rows_per_image: Some(Self::DFG_DIM_ROUGH),
                },
            },
            dfg1_out_tex.size(),
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                aspect: wgpu::TextureAspect::All,
                        texture: &dirs_out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &dirs_out_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(Self::DIRS_ROW_SIZE),
                    rows_per_image: Some(Self::DIRS_DIM_ROUGH),
                },
            },
            dirs_out_tex.size(),
        );

        gpu.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        let tx3 = tx.clone();
        let dfg1_out_slice = dfg1_out_buf.slice(..);
        dfg1_out_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let dfg2_out_slice = dfg2_out_buf.slice(..);
        dfg2_out_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });
        let dirs_out_slice = dirs_out_buf.slice(..);
        dirs_out_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx3.send(result).unwrap();
        });
        
        gpu.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();
        log::info!("integration complete");

        let dfg1_out_range = dfg1_out_slice.get_mapped_range();
        let dfg1_out_data = bytemuck::cast_slice(&dfg1_out_range);
        let dfg1_out_img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(Self::DFG_DIM_NV, Self::DFG_DIM_ROUGH, dfg1_out_data).unwrap();
        let dfg1_out_path = format!("./assets/staging/dfg_integral_lut.{}.png", Self::DFG1_EXT);
        dfg1_out_img.save(&dfg1_out_path).unwrap();

        let dfg2_out_range = dfg2_out_slice.get_mapped_range();
        let dfg2_out_data = bytemuck::cast_slice(&dfg2_out_range);
        let dfg2_out_img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(Self::DFG_DIM_NV, Self::DFG_DIM_ROUGH, dfg2_out_data).unwrap();
        let dfg2_out_path = format!("./assets/staging/dfg_trans_integral_lut.{}.png", Self::OUTPUT_EXT);
        dfg2_out_img.save(&dfg2_out_path).unwrap();

        let dirs_out_range = dirs_out_slice.get_mapped_range();
        let dirs_out_data = bytemuck::cast_slice(&dirs_out_range);
        let dirs_out_img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(Self::DIRS_DIM_NV, Self::DIRS_DIM_ROUGH * 4, dirs_out_data).unwrap();
        let dirs_out_path = format!("./assets/staging/dirs_integral_lut.{}.png", Self::OUTPUT_EXT);
        dirs_out_img.save(&dirs_out_path).unwrap();
        log::info!("save complete");

        DFGTables {
            dfg: dfg1_out_tex,
            trans_dfg: dfg2_out_tex,
            dirs: dirs_out_tex,
        }
    }
}