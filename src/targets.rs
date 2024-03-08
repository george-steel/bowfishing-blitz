use glam::*;
use kira::manager::AudioManager;
use kira::sound::static_sound::StaticSoundData;
use wgpu::{util::BufferInitDescriptor, *};
use wgpu::util::DeviceExt;
use rand::random;
use std::mem::size_of;

use crate::arrows::{collide_ray_sphere, ArrowTarget};
use crate::audio_util::SoundAtlas;
use crate::{deferred_renderer::{DeferredRenderer, RenderObject}, gputil::*, terrain_view::HeightmapTerrain};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Target {
    center: Vec3,
    state: u32, // 0 is live, 1 is hit
}

const NUM_TARGETS: usize = 128;
const TARGET_RADIUS: f32 = 0.4;


#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TargetVertex {
    pos: Vec3,
    norm: Vec3,
}

const OCTAHEDHRON: &[TargetVertex] = &[
    TargetVertex {pos: vec3(0.0, 0.0, -1.0), norm: vec3(1.0, 1.0, -1.0)},
    TargetVertex {pos: vec3(0.0, 1.0, 0.0), norm: vec3(1.0, 1.0, -1.0)},
    TargetVertex {pos: vec3(1.0, 0.0, 0.0), norm: vec3(1.0, 1.0, -1.0)},

    TargetVertex {pos: vec3(0.0, 0.0, -1.0), norm: vec3(1.0, -1.0, -1.0)},
    TargetVertex {pos: vec3(1.0, 0.0, 0.0), norm: vec3(1.0, -1.0, -1.0)},
    TargetVertex {pos: vec3(0.0, -1.0, 0.0), norm: vec3(1.0, -1.0, -1.0)},

    TargetVertex {pos: vec3(0.0, 0.0, -1.0), norm: vec3(-1.0, -1.0, -1.0)},
    TargetVertex {pos: vec3(0.0, -1.0, 0.0), norm :vec3(-1.0, -1.0, -1.0)},
    TargetVertex {pos: vec3(-1.0, 0.0, 0.0), norm: vec3(-1.0, -1.0, -1.0)},

    TargetVertex {pos: vec3(0.0, 0.0, -1.0), norm: vec3(-1.0, 1.0, -1.0)},
    TargetVertex {pos: vec3(-1.0, 0.0, 0.0), norm: vec3(-1.0, 1.0, -1.0)},
    TargetVertex {pos: vec3(0.0, 1.0, 0.0), norm: vec3(-1.0, 1.0, -1.0)},

    TargetVertex {pos: vec3(0.0, 0.0, 1.0), norm: vec3(1.0, 1.0, 1.0)},
    TargetVertex {pos: vec3(1.0, 0.0, 0.0), norm: vec3(1.0, 1.0, 1.0)},
    TargetVertex {pos: vec3(0.0, 1.0, 0.0), norm: vec3(1.0, 1.0, 1.0)},

    TargetVertex {pos: vec3(0.0, 0.0, 1.0), norm: vec3(1.0, -1.0, 1.0)},
    TargetVertex {pos: vec3(0.0, -1.0, 0.0), norm: vec3(1.0, -1.0, 1.0)},
    TargetVertex {pos: vec3(1.0, 0.0, 0.0), norm: vec3(1.0, -1.0, 1.0)},

    TargetVertex {pos: vec3(0.0, 0.0, 1.0), norm: vec3(-1.0, -1.0, 1.0)},
    TargetVertex {pos: vec3(-1.0, 0.0, 0.0), norm: vec3(-1.0, -1.0, 1.0)},
    TargetVertex {pos: vec3(0.0, -1.0, 0.0), norm: vec3(-1.0, -1.0, 1.0)},

    TargetVertex {pos: vec3(0.0, 0.0, 1.0), norm: vec3(-1.0, 1.0, 1.0)},
    TargetVertex {pos: vec3(0.0, 1.0, 0.0), norm: vec3(-1.0, 1.0, 1.0)},
    TargetVertex {pos: vec3(-1.0, 0.0, 0.0), norm: vec3(-1.0, 1.0, 1.0)},
];

pub struct TargetController {
    targets_above_pipeline: RenderPipeline,
    targets_below_pipeline: RenderPipeline,
    targets_buf: Buffer,
    targets_bg: BindGroup,
    targets_vertex_buf: Buffer,
    smash_sounds: SoundAtlas,

    pub all_targets: Box<[Target]>,
    dirty: bool,
}

impl TargetController {
    fn gen_targets(num_targets: usize, terrain: &HeightmapTerrain, inner_radius: f32) -> Box<[Target]> {
        let mut targets = Vec::with_capacity(num_targets);

        while targets.len() < num_targets {
            let xy = (vec2(random(), random()) - 0.5) * 2.0 * inner_radius;
            if let Some(z) = terrain.height_at(xy) {
                if z < 0.0 {
                    targets.push(Target {
                        center: vec3(xy.x, xy.y, z + 0.3),
                        state: 0,
                    });
                }
            }
        }
        targets.into_boxed_slice()
    }

    pub fn new(gpu: &GPUContext, renderer: &DeferredRenderer, terrain: &HeightmapTerrain) -> Self {
        let all_targets = Self::gen_targets(NUM_TARGETS, terrain, 40.0);

        let shaders = gpu.device.create_shader_module(ShaderModuleDescriptor{
            label: Some("pots.wgsl"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(crate::shaders::TARGETS)),
        });

        let targets_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("pots_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }
            ]
        });

        let targets_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pots_pipeline_layout"),
            bind_group_layouts: &[
                &renderer.global_bind_layout,
                &targets_bg_layout,
            ],
            push_constant_ranges: &[],
        });

        let targets_vertex_layout = VertexBufferLayout {
            array_stride: size_of::<TargetVertex>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };

        let targets_vertex_buf = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("pots_vertex_buf"),
            contents: bytemuck::cast_slice(OCTAHEDHRON),
            usage: BufferUsages::VERTEX,
        });

        let targets_above_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("pots_above"),
            layout: Some(&targets_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: "pot_vert_above",
                buffers: &[targets_vertex_layout.clone()],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: "pot_frag_above",
                targets: DeferredRenderer::GBUFFER_TARGETS,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let targets_below_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("pots_above"),
            layout: Some(&targets_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: "pot_vert_below",
                buffers: &[targets_vertex_layout],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: "pot_frag_below",
                targets: DeferredRenderer::UNDERWATER_GBUFFER_TARGETS,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let targets_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("pots_buf"),
            size: (size_of::<Target>() * NUM_TARGETS) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        let targets_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("pots_bg"),
            layout: &targets_bg_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: targets_buf.as_entire_binding()}
            ]
        });

        let smash_sounds = SoundAtlas::load_with_starts("./assets/glass_smash.ogg", -3.0,
            &[0.0, 1.57, 2.84, 4.02, 5.43, 6.98, 8.38, 9.68, 10.97, 12.32, 13.58, 15.30, 16.73, 18.12]).unwrap();

        TargetController {
            targets_above_pipeline, targets_below_pipeline,
            targets_buf, targets_bg, targets_vertex_buf,
            smash_sounds,

            all_targets,
            dirty: true,
        }
    }
}

impl RenderObject for TargetController {
    fn prepass(&mut self, gpu: &GPUContext, renderer: &DeferredRenderer, encoder: &mut CommandEncoder) {
        if self.dirty {
            gpu.queue.write_buffer(&self.targets_buf, 0, bytemuck::cast_slice(&self.all_targets));
        }
        self.dirty = false;
    }

    fn draw_underwater<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {
        pass.set_pipeline(&self.targets_below_pipeline);
        pass.set_bind_group(1, &self.targets_bg, &[]);
        pass.set_vertex_buffer(0, self.targets_vertex_buf.slice(..));
        pass.draw(0..(OCTAHEDHRON.len() as u32), 0..(NUM_TARGETS as u32));
    }

    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {
        pass.set_pipeline(&self.targets_above_pipeline);
        pass.set_bind_group(1, &self.targets_bg, &[]);
        pass.set_vertex_buffer(0, self.targets_vertex_buf.slice(..));
        pass.draw(0..(OCTAHEDHRON.len() as u32), 0..(NUM_TARGETS as u32));
    }
}

impl ArrowTarget for TargetController {
    fn process_hits(&mut self, audio: &mut AudioManager, start: Vec3, end: Vec3) -> bool {
        let mut was_hit = false;

        for t in self.all_targets.iter_mut() {
            if t.state == 0 && collide_ray_sphere(start, end, t.center, TARGET_RADIUS) {
                t.state = 1;
                was_hit = true;

                audio.play(self.smash_sounds.random_sound()).unwrap();
            }
        }

        self.dirty = self.dirty || was_hit;
        was_hit
    }
}
