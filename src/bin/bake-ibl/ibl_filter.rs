use bowfishing_blitz::{gputil::mip::MipMaker, *};
use gputil::*;
use std::{borrow::Cow, default};
use wgpu::*;

pub struct IBLFilter {
    bake_shader: ShaderModule,
    disp_pipeline: RenderPipeline,
    disp_bg: BindGroup,
    sampler: Sampler,
    filtered_tex: Texture,
    filtered_view: TextureView,
}


impl IBLFilter {
    const INPUT_HEIGHT: u32 = 512;
    const OUTPUT_LEVELS: u32 = 8;
    const FHT_IN_BUFFER_SIZE: u64 = (2*Self::INPUT_HEIGHT*Self::INPUT_HEIGHT*4*4) as u64;
    const FHT_OUT_BUFFER_SIZE: u64 = Self::FHT_IN_BUFFER_SIZE*(Self::OUTPUT_LEVELS as u64);
    const SPECTRUM_BUFFER_SIZE: u64 = (Self::INPUT_HEIGHT*Self::OUTPUT_LEVELS*4) as u64;

    pub fn new(gpu: &GPUContext) -> Self {
        let bake_shader = gpu.device.create_shader_module(ShaderModuleDescriptor{
            label: Some("spherical_filters.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("spherical_filters.wgsl"))),
        });

        let disp_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("disp_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ]
        });
        let disp_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("disp_layout"),
            bind_group_layouts: &[&disp_bg_layout],
            push_constant_ranges: &[]
        });
        let disp_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&disp_pipeline_layout),
            vertex: VertexState {
                module: &bake_shader,
                entry_point: Some("fullscreen_quad"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &bake_shader,
                entry_point: Some("display_tex"),
                compilation_options: Default::default(),
                targets: &[Some(gpu.output_format.into())],
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

        let filtered_tex = gpu.device.create_texture(&wgpu::TextureDescriptor{
            label: Some("filtered_tex"),
            size: Extent3d{width: 2*Self::INPUT_HEIGHT, height: Self::INPUT_HEIGHT , depth_or_array_layers: Self::OUTPUT_LEVELS},
            format: TextureFormat::Rgba16Float,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let filtered_view = filtered_tex.create_view(&Default::default());
        let sampler = gpu.device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::MirrorRepeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let disp_bg = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("fht_bind"),
            layout: &disp_bg_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&filtered_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&sampler),
                },
            ],
        });

        IBLFilter {
            bake_shader, 
            disp_pipeline, disp_bg,
            filtered_tex, filtered_view, sampler,
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
            
            rpass.set_pipeline(&self.disp_pipeline);
            rpass.set_bind_group(0, &self.disp_bg, &[]);
            rpass.draw(0..4, 0..1);
        }
        gpu.queue.submit([encoder.finish()]);
    }

    pub fn do_filtering(&self, gpu: &GPUContext)  {
        let in_tex = gpu.load_rgbe8_texture("./assets/staging/kloofendal_48d_partly_cloudy_1k.rgbe.png").expect("Failed to load sky");
        let in_view = in_tex.create_view(&Default::default());

        let spectrum_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("spectrum_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
            ]
        });
        let spectrum_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("spectrum_layout"),
            bind_group_layouts: &[&spectrum_bg_layout],
            push_constant_ranges: &[]
        });
        let spectrum_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor{
            label: Some("spectrum_pipeline"),
            layout: Some(&spectrum_layout),
            module: &self.bake_shader,
            entry_point: Some("get_kernel_spectra"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fht1_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("fht1_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer{
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        });
        let fht1_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("fht1_layout"),
            bind_group_layouts: &[&fht1_bg_layout],
            push_constant_ranges: &[]
        });
        let fht1_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor{
            label: Some("fht1_pipeline"),
            layout: Some(&fht1_layout),
            module: &self.bake_shader,
            entry_point: Some("horiz_fht_tex_to_buf"),
            compilation_options: Default::default(),
            cache: None,
        });

        let fht2_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("fht2_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ]
        });
        let fht2_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("fht2_layout"),
            bind_group_layouts: &[&fht2_bg_layout],
            push_constant_ranges: &[]
        });
        let fht2_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor{
            label: Some("fht2_pipeline"),
            layout: Some(&fht2_layout),
            module: &self.bake_shader,
            entry_point: Some("horiz_fht_buf_to_tex"),
            compilation_options: Default::default(),
            cache: None,
        });

        let blur_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("blur_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        });
        let blur_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("blur_layout"),
            bind_group_layouts: &[&blur_bg_layout],
            push_constant_ranges: &[]
        });
        let blur_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor{
            label: Some("blur_pipeline"),
            layout: Some(&blur_layout),
            module: &self.bake_shader,
            entry_point: Some("blur_spectra"),
            compilation_options: Default::default(),
            cache: None,
        });

        let spectrum_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("spectrum_buf"),
            size: Self::SPECTRUM_BUFFER_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false 
        });
        let hspec_in_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("hspec_in_buf"),
            size: Self::FHT_IN_BUFFER_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false 
        });
        let hspec_out_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("hspec_out_buf"),
            size: Self::FHT_OUT_BUFFER_SIZE,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false 
        });
        let spectrum_bind = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("spectrum_bind"),
            layout: &spectrum_bg_layout,
            entries: &[BindGroupEntry {
                    binding: 0,
                    resource: spectrum_buf.as_entire_binding(),
                },
            ],
        });
        let fht1_bind = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("fht1_bind"),
            layout: &fht1_bg_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&in_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: hspec_in_buf.as_entire_binding(),
                },
            ],
        });
        let fht2_bind = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("fht2_bind"),
            layout: &fht2_bg_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: hspec_out_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&self.filtered_view),
                },
            ],
        });
        let blur_bind = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("blur_bind"),
            layout: &blur_bg_layout,
            entries: &[BindGroupEntry {
                    binding: 0,
                    resource: spectrum_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: hspec_in_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: hspec_out_buf.as_entire_binding(),
                },
            ],
        });


        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("dht_pass"), timestamp_writes: None });
            pass.set_pipeline(&spectrum_pipeline);
            pass.set_bind_group(0, &spectrum_bind, &[]);
            pass.dispatch_workgroups(Self::OUTPUT_LEVELS as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("dht_pass"), timestamp_writes: None });
            pass.set_pipeline(&fht1_pipeline);
            pass.set_bind_group(0, &fht1_bind, &[]);
            pass.dispatch_workgroups(Self::INPUT_HEIGHT / 2 as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("blur_pass"), timestamp_writes: None });
            pass.set_pipeline(&blur_pipeline);
            pass.set_bind_group(0, &blur_bind, &[]);
            pass.dispatch_workgroups(Self::INPUT_HEIGHT * 2 as u32, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("dht_pass"), timestamp_writes: None });
            pass.set_pipeline(&fht2_pipeline);
            pass.set_bind_group(0, &fht2_bind, &[]);
            pass.dispatch_workgroups(Self::INPUT_HEIGHT / 2 as u32, Self::OUTPUT_LEVELS as u32, 1);
        }

        gpu.queue.submit([encoder.finish()]);
    }
}
