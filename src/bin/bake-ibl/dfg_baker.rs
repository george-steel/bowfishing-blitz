use bowfishing_blitz::{gputil::mip::MipMaker, *};
use gputil::*;
use std::{borrow::Cow, default};
use wgpu::{wgt::TextureDescriptor, *};

pub struct DFGBaker {
    shaders: ShaderModule,

    dfg_pipeline: RenderPipeline,
    dirs_pipeline: RenderPipeline,
}

impl DFGBaker {
    const COMP_OPTIONS: PipelineCompilationOptions<'static> = PipelineCompilationOptions {
        constants: &[],
        zero_initialize_workgroup_memory: true,
    };
    const DFG_DIM_NV: u32 = 128;
    const DFG_DIM_ROUGH: u32 = 128;
    const DFG_ROW_SIZE: u32 = Self::DFG_DIM_NV * 8;
    const DFG_BUF_SIZE: u32 = Self::DFG_DIM_NV * Self::DFG_DIM_ROUGH * 8;

    const DIRS_DIM_NV: u32 = 64;
    const DIRS_DIM_ROUGH: u32 = 64;
    const DIRS_ROW_SIZE: u32 = Self::DIRS_DIM_NV * 8;
    const DIRS_BUF_SIZE: u32 = Self::DIRS_DIM_NV * Self::DIRS_DIM_ROUGH * 8 * 2;

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
                targets: &[Some(TextureFormat::Rgba16Unorm.into())],
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
                targets: &[Some(TextureFormat::Rgba16Unorm.into()), Some(TextureFormat::Rgba16Unorm.into())],
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

    pub fn integrate_dfg_lut(&self, gpu: &GPUContext) -> (Texture, Texture) {
        let dfg_out_tex = gpu.device.create_texture(&TextureDescriptor{
            label: Some("dfg_lut_tex"),
            size: Extent3d { width: Self::DFG_DIM_NV, height: Self::DFG_DIM_ROUGH, depth_or_array_layers: 1 },
            format: TextureFormat::Rgba16Unorm,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let dfg_out_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("dfg_lut_out_buf"),
            size: Self::DFG_BUF_SIZE as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false 
        });

        let dirs_out_tex = gpu.device.create_texture(&TextureDescriptor{
            label: Some("dfg_lut_tex"),
            size: Extent3d { width: Self::DIRS_DIM_NV, height: Self::DIRS_DIM_ROUGH, depth_or_array_layers: 2 },
            format: TextureFormat::Rgba16Unorm,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3,
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
                        view: &dfg_out_tex.create_view(&Default::default()),
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
            let dirs_view = dirs_out_tex.create_view(&Default::default());
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &dirs_view,
                        depth_slice: Some(0),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &dirs_view,
                        depth_slice: Some(1),
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
                        texture: &dfg_out_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &dfg_out_buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(Self::DFG_ROW_SIZE),
                    rows_per_image: Some(Self::DFG_DIM_ROUGH),
                },
            },
            dfg_out_tex.size(),
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
        let dfg_out_slice = dfg_out_buf.slice(..);
        dfg_out_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let dirs_out_slice = dirs_out_buf.slice(..);
        dirs_out_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx2.send(result).unwrap();
        });
        
        gpu.device.poll(wgpu::PollType::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();
        log::info!("integration complete");

        let dfg_out_range = dfg_out_slice.get_mapped_range();
        let dfg_out_data = bytemuck::cast_slice(&dfg_out_range);
        let dfg_out_img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(Self::DFG_DIM_NV, Self::DFG_DIM_ROUGH, dfg_out_data).unwrap();
        dfg_out_img.save("./assets/staging/dfg_integral_lut.rgba16un.png").unwrap();

        let dirs_out_range = dirs_out_slice.get_mapped_range();
        let dirs_out_data = bytemuck::cast_slice(&dirs_out_range);
        let dirs_out_img = image::ImageBuffer::<image::Rgba<u16>, _>::from_raw(Self::DIRS_DIM_NV, Self::DIRS_DIM_ROUGH * 2, dirs_out_data).unwrap();
        dirs_out_img.save("./assets/staging/dirs_integral_lut.rgba16un.png").unwrap();
        log::info!("save complete");

        (dfg_out_tex, dirs_out_tex)
    }
}