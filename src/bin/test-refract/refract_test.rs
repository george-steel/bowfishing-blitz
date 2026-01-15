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
            label: Some("refract_test.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("refract_test.wgsl"))),
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
                entry_point: Some("refr_test"),
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
}
