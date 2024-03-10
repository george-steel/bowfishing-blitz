use glam::*;
use kira::manager::AudioManager;
use kira::sound::static_sound::StaticSoundData;
use wgpu::{util::BufferInitDescriptor, *};
use wgpu::util::DeviceExt;
use rand::random;
use std::f32::consts::TAU;
use std::mem::size_of;
use std::time::Instant;

use crate::arrows::{collide_ray_sphere, ArrowTarget};
use crate::audio_util::SoundAtlas;
use crate::{deferred_renderer::{DeferredRenderer, RenderObject}, gputil::*, terrain_view::HeightmapTerrain};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Target {
    bottom: Vec3,
    time_hit: f32, // 0 is live, 1 is hit
    orientation: Quat,
}

const NUM_TARGETS: usize = 128;
const TARGET_RADIUS: f32 = 0.5;

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LathePoint {
    pub pos_rz: Vec2,
    pub norm1_rz: Vec2,
    pub norm2_rz: Vec2,
    pub v: f32,
    pub pad: f32,
}

impl LathePoint {
    pub fn smooth(pos: Vec2, v: f32, norm: Vec2) -> Self {
        let norm_rz = norm.normalize();
        LathePoint {
            pos_rz: pos,
            norm1_rz: norm_rz,
            norm2_rz: norm_rz,
            v,
            pad: 0.0,
        }
    }

    pub fn sharp(pos: Vec2, v: f32, norm1: Vec2, norm2: Vec2) -> Self {
        let norm1_rz = norm1.normalize();
        let norm2_rz = norm2.normalize();
        LathePoint {
            pos_rz: pos,
            norm1_rz,
            norm2_rz,
            v,
            pad: 0.0,
        }
    }
}

fn pot_model() -> Box<[LathePoint]> {
    // Modelled on graph paper
    let points = vec![
        LathePoint::smooth(vec2(4.0/16.0, 15.0/16.0), 0.0, vec2(0.0, -1.0)),
        LathePoint::smooth(vec2(3.0/16.0, 16.0/16.0), 1.0/16.0, vec2(-1.0, 0.0)),
        LathePoint::smooth(vec2(4.0/16.0, 17.0/16.0), 2.0/16.0, vec2(0.0, 1.0)),
        LathePoint::smooth(vec2(5.0/16.0, 16.0/16.0), 3.0/16.0, vec2(1.0, 0.0)),
        LathePoint::sharp(vec2(4.0/16.0, 15.0/16.0), 0.25, vec2(1.0, 2.0), vec2(0.0, -1.0)),

        LathePoint::smooth(vec2(7.0/16.0, 13.0/16.0), 1.0 - (23.5 / 28.0)*0.75, vec2(1.0, 1.0)),
        LathePoint::smooth(vec2(8.0/16.0, 11.0/16.0), 1.0 - (20.5 / 28.0)*0.75, vec2(4.0, 1.0)),
        LathePoint::smooth(vec2(8.0/16.0, 9.0/16.0), 1.0 - (18.0 / 28.0)*0.75, vec2(8.0, -1.0)),
        LathePoint::smooth(vec2(7.0/16.0, 5.0/16.0), 1.0 - (12.5 / 28.0)*0.75, vec2(3.0, -1.0)),
        LathePoint::smooth(vec2(6.0/16.0, 3.0/16.0), 1.0 - (9.5 / 28.0)*0.75, vec2(1.75, -1.0)),
        LathePoint::sharp(vec2(4.0/16.0, 0.0), 1.0 - (5.0 / 28.0)*0.75, vec2(1.0, -1.0), vec2(-1.0, -3.0), ),
        LathePoint::smooth(vec2(0.0, 1.0/16.0), 1.0, vec2(0.0, -1.0)),
    ];
    points.into_boxed_slice()
}

const POT_U_DIVS: u32 = 8;
const NUM_POT_VERTS: u32 = 11 * 6 * POT_U_DIVS;

pub struct TargetController {
    targets_above_pipeline: RenderPipeline,
    targets_below_pipeline: RenderPipeline,
    targets_buf: Buffer,
    targets_bg: BindGroup,
    smash_sounds: SoundAtlas,

    updated_at: f64,
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
                    let rot_z = TAU * random::<f32>();
                    let norm = terrain.normal_at(xy).unwrap(); //same domain as height
                    let rot = Quat::from_rotation_arc(vec3(0.0, 0.0, 1.0), norm) * Quat::from_rotation_z(rot_z);
                    targets.push(Target {
                        bottom: vec3(xy.x, xy.y, z),
                        time_hit: -1.0,
                        orientation: rot,
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
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
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

        let target_lathe_points = pot_model();
        let target_lathe_buf = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("target_lathe_buf"),
            contents: bytemuck::cast_slice(&target_lathe_points),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let targets_above_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("pots_above"),
            layout: Some(&targets_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: "pot_vert_above",
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: "pot_frag_above",
                targets: DeferredRenderer::GBUFFER_TARGETS,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
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
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: "pot_frag_below",
                targets: DeferredRenderer::UNDERWATER_GBUFFER_TARGETS,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
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
                BindGroupEntry {binding: 0, resource: targets_buf.as_entire_binding()},
                BindGroupEntry {binding: 1, resource: target_lathe_buf.as_entire_binding()},
            ]
        });

        let smash_sounds = SoundAtlas::load_with_starts("./assets/glass_smash.ogg", -3.0,
            &[0.0, 1.57, 2.84, 4.02, 5.43, 6.98, 8.38, 9.68, 10.97, 12.32, 13.58, 15.30, 16.73, 18.12]).unwrap();

        TargetController {
            targets_above_pipeline, targets_below_pipeline,
            targets_buf, targets_bg,
            smash_sounds,

            updated_at: 0.0,

            all_targets,
            dirty: true,
        }
    }

    pub fn tick(&mut self, time: f64) {
        self.updated_at = time;
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
        pass.draw(0..NUM_POT_VERTS, 0..(NUM_TARGETS as u32));
    }

    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {
        pass.set_pipeline(&self.targets_above_pipeline);
        pass.set_bind_group(1, &self.targets_bg, &[]);
        pass.draw(0..NUM_POT_VERTS, 0..(NUM_TARGETS as u32));
    }
}

impl ArrowTarget for TargetController {
    fn process_hits(&mut self, audio: &mut AudioManager, start: Vec3, end: Vec3) -> bool {
        let mut was_hit = false;

        for t in self.all_targets.iter_mut() {
            if t.time_hit < 0.0 {
                let center = t.bottom + t.orientation.mul_vec3(vec3(0.0, 0.0, TARGET_RADIUS));
                if collide_ray_sphere(start, end, center, TARGET_RADIUS) {
                    t.time_hit = self.updated_at as f32;
                    was_hit = true;

                    audio.play(self.smash_sounds.random_sound()).unwrap();
                }
            }
        }

        self.dirty = self.dirty || was_hit;
        was_hit
    }
}
