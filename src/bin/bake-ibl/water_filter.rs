use bowfishing_blitz::{gputil::mip::MipMaker, *};
use gputil::*;
use std::{borrow::Cow, default};
use wgpu::{wgt::TextureDescriptor, *};

use crate::integral_filter::{DFGBaker, DFGTables};

pub struct WaterFilter {
    shaders: ShaderModule,
    sampler: Sampler,

    water_bg_layout: BindGroupLayout,
    clamp_bg_layout: BindGroupLayout,
    above_pipeline: RenderPipeline,
    below_pipeline: RenderPipeline,
    clamp_pipeline: RenderPipeline,
}

impl WaterFilter {
    const COMP_OPTIONS: PipelineCompilationOptions<'static> = PipelineCompilationOptions {
        constants: &[("LIN_CORRECTION", DFGBaker::LIN_CORRECTION)],
        zero_initialize_workgroup_memory: true,
    };

    pub fn new(gpu: &GPUContext) -> Self {
        let shaders = gpu.device.create_shader_module(ShaderModuleDescriptor{
            label: Some("water_filters.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("water_filters.wgsl"))),
        });

        let water_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::Cube,
                        multisampled: false
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let water_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("water_layout"),
            bind_group_layouts: &[&water_bg_layout],
            push_constant_ranges: &[]
        });

        let clamp_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false
                    },
                    count: None,
                },
            ],
        });
        let clamp_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("water_layout"),
            bind_group_layouts: &[&clamp_bg_layout],
            push_constant_ranges: &[]
        });

        let above_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&water_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("fullscreen_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: Some("above_water_equi"),
                compilation_options: Self::COMP_OPTIONS,
                targets: &[Some(TextureFormat::Rgba16Float.into())],
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

        let below_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&water_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("fullscreen_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: Some("below_water_equi"),
                compilation_options: Self::COMP_OPTIONS,
                targets: &[Some(TextureFormat::Rgba16Float.into())],
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

        let clamp_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&clamp_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("fullscreen_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: Some("clamp_tex"),
                compilation_options: Self::COMP_OPTIONS,
                targets: &[Some(TextureFormat::Rgba16Float.into())],
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

        let sampler = gpu.device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::MirrorRepeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        });

        WaterFilter {
            shaders, sampler,
            water_bg_layout, clamp_bg_layout,
            above_pipeline, below_pipeline, clamp_pipeline,
        }
    }

    pub fn render_above(&self, gpu: &GPUContext, dfg_tables: &DFGTables, raw_sky: &TextureView, filt_sky: &TextureView) -> Texture {
        let dims = raw_sky.texture().size();
        let out_tex = gpu.device.create_texture(&TextureDescriptor{
            label: Some("above_tex"),
            size: dims,
            format: TextureFormat::Rgba16Float,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("above_bg"),
            layout: &self.water_bg_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&raw_sky),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&filt_sky),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&dfg_tables.dfg.create_view(&Default::default())),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&dfg_tables.trans_dfg.create_view(&Default::default())),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&dfg_tables.dirs.create_view(&Default::default())),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Sampler(&dfg_tables.sampler),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        let out_view = out_tex.create_view(&Default::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
                    depth_slice: None,
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
            
            rpass.set_pipeline(&self.above_pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.draw(0..4, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);

        out_tex
    }

    pub fn render_below(&self, gpu: &GPUContext, dfg_tables: &DFGTables, raw_sky: &TextureView, filt_sky: &TextureView) -> Texture {
        let dims = raw_sky.texture().size();
        let out_tex = gpu.device.create_texture(&TextureDescriptor{
            label: Some("below_tex"),
            size: dims,
            format: TextureFormat::Rgba16Float,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("below_bg"),
            layout: &self.water_bg_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&raw_sky),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&filt_sky),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&dfg_tables.dfg.create_view(&Default::default())),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&dfg_tables.trans_dfg.create_view(&Default::default())),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(&dfg_tables.dirs.create_view(&Default::default())),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Sampler(&dfg_tables.sampler),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        let out_view = out_tex.create_view(&Default::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
                    depth_slice: None,
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
            
            rpass.set_pipeline(&self.below_pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.draw(0..4, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);

        out_tex
    }

    pub fn render_clamp(&self, gpu: &GPUContext, raw_sky: &TextureView,) -> Texture {
        let dims = raw_sky.texture().size();
        let out_tex = gpu.device.create_texture(&TextureDescriptor{
            label: Some("clamp_tex"),
            size: dims,
            format: TextureFormat::Rgba16Float,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("clamp_bg"),
            layout: &self.clamp_bg_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&raw_sky),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        let out_view = out_tex.create_view(&Default::default());
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
                    depth_slice: None,
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
            
            rpass.set_pipeline(&self.clamp_pipeline);
            rpass.set_bind_group(0, &bg, &[]);
            rpass.draw(0..4, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);

        out_tex
    }
}