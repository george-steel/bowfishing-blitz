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
                        view_dimension: TextureViewDimension::D2,
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
            size: Extent3d{width: 1024, height: 512, depth_or_array_layers: 1},
            format: TextureFormat::Rgba32Float,
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
        let in_tex = gpu.load_rgbe8_texture("./assets/sky-equirect.rgbe8.png").expect("Failed to load sky");
        let in_view = in_tex.create_view(&Default::default());

        let fht_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor{
            label: Some("fht_bg_layout"),
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
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ]
        });
        let fht_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor{
            label: Some("fht_layout"),
            bind_group_layouts: &[&fht_bg_layout],
            push_constant_ranges: &[]
        });
        let dht_pipeline = gpu.device.create_compute_pipeline(&ComputePipelineDescriptor{
            label: Some("dht_pipeline"),
            layout: Some(&fht_layout),
            module: &self.bake_shader,
            entry_point: Some("horiz_fht"),
            compilation_options: Default::default(),
            cache: None,
        });
        let hspec_tex_1 = gpu.device.create_texture(&wgpu::TextureDescriptor{
            label: Some("filtered_tex"),
            size: Extent3d{width: 1024, height: 512, depth_or_array_layers: 1},
            format: TextureFormat::Rgba32Float,
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC | TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let hspec_view_1 = hspec_tex_1.create_view(&Default::default());
        let fht1_bind = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("fht_bind"),
            layout: &fht_bg_layout,
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
                    resource: BindingResource::TextureView(&hspec_view_1),
                },
            ],
        });
        let fht2_bind = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("fht_bind"),
            layout: &fht_bg_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&hspec_view_1),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&self.filtered_view),
                },
            ],
        });


        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("dht_pass"), timestamp_writes: None });
            pass.set_pipeline(&dht_pipeline);
            pass.set_bind_group(0, &fht1_bind, &[]);
            pass.dispatch_workgroups(256, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("dht_pass"), timestamp_writes: None });
            pass.set_pipeline(&dht_pipeline);
            pass.set_bind_group(0, &fht2_bind, &[]);
            pass.dispatch_workgroups(256, 1, 1);
        }

        gpu.queue.submit([encoder.finish()]);
    }
}
