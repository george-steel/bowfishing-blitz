use glam::*;
use half::f16;
use kira::manager::AudioManager;
use wgpu::{util::BufferInitDescriptor, *};
use wgpu::util::DeviceExt;
use rand::{thread_rng, Rng};
use std::f32::consts::TAU;
use std::mem::size_of;
use std::time::Instant;

use crate::arrows::{collide_ray_sphere, ArrowTarget};
use crate::audio_util::SoundAtlas;
use crate::boat_rail::LoopedRail;
use crate::camera::sphere_visible;
use crate::{deferred_renderer::{DeferredRenderer, RenderObject}, gputil::*, terrain_view::HeightmapTerrain};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Target {
    bottom: Vec3,
    time_hit: f32, // 0 is live, 1 is hit
    orientation: Quat,
    color_a: [f16; 3],
    color_b: [f16; 3],
    seed: u32
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
        LathePoint::sharp(vec2(4.0/16.0, 15.0/16.0), 0.25, vec2(0.0, -1.0), vec2(1.0, 2.0)),

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

fn color_rail() -> LoopedRail<Vec3> {
    // rainbow palette
    // https://iamkate.com/data/12-bit-rainbow/
    LoopedRail {points: vec![
        vec3(0.24575, 0.00582, 0.18417),
        vec3(0.40228, 0.03304, 0.09082),
        vec3(0.60439, 0.13263, 0.13266),
        vec3(0.85646, 0.31808, 0.05692),
        vec3(0.85454, 0.723, 0.00119),
        vec3(0.31869, 0.72314, 0.0909),
        vec3(0.05705, 0.72332, 0.24573),
        vec3(0.05705, 0.72332, 0.24573),
        vec3(0.0, 0.49693, 0.60383),
        vec3(0.00033, 0.31845, 0.6031),
        vec3(0.03299, 0.13279, 0.49739),
        vec3(0.13277, 0.03324, 0.31803),
    ].into_boxed_slice()}
}

fn pack_h3(v: Vec3) -> [f16; 3] {
    [f16::from_f32(v.x), f16::from_f32(v.y), f16::from_f32(v.z)]
}

pub struct TargetController {
    targets_pipeline: RenderPipeline,
    targets_refr_pipeline: RenderPipeline,
    targets_refl_pipeline: RenderPipeline,
    shadow_targets_pipeline: RenderPipeline,
    targets_buf: Buffer,
    targets_bg: BindGroup,
    shadow_targets_bg: BindGroup,
    smash_sounds: SoundAtlas,
    max_target_inst: u32,

    updated_at: f64,
    pub all_targets: Box<[Target]>,
    pub targets_hit: u32,
}

impl TargetController {
    fn gen_targets(num_targets: usize, terrain: &HeightmapTerrain, inner_radius: f32) -> Box<[Target]> {
        let mut targets = Vec::with_capacity(num_targets);
        let mut rng = thread_rng();

        let colors = color_rail();

        let seed = rng.random();
        let mut i = 0;
        while targets.len() < num_targets {
            let rand = sobol_burley::sample_4d(i, 0, seed);
            let xy = (vec2(rand[0], rand[1]) - 0.5) * 2.0 * inner_radius;
            if let Some(z) = terrain.height_at(xy) {
                if z < -0.1 {
                    let rot_z: f32 = TAU * rand[2];
                    let norm = terrain.normal_at(xy).unwrap(); //same domain as height
                    let rot = Quat::from_rotation_arc(vec3(0.0, 0.0, 1.0), norm) * Quat::from_rotation_z(rot_z);
                    let col_idx: f64 = rand[3] as f64;
                    let col_step = if rng.gen_bool(0.5) {0.33} else {-0.33};
                    let col_fac = 0.4 * smoothstep((col_idx as f32 - 0.33).abs() * 5.0);
                    targets.push(Target {
                        bottom: vec3(xy.x, xy.y, z),
                        time_hit: -1.0,
                        orientation: rot,
                        color_a: pack_h3(colors.sample(col_idx)),
                        color_b: pack_h3(colors.sample(col_idx + col_step).lerp(vec3(0.7, 0.7, 0.7), col_fac)),
                        seed: rng.random(),
                    });
                }
            }
            i += 1
        }
        targets.into_boxed_slice()
    }

    pub fn new(gpu: &GPUContext, assets: &impl AssetSource, renderer: &DeferredRenderer, terrain: &HeightmapTerrain) -> Self {
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
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
            ]
        });

        let shadow_targets_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
        let shadow_targets_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pots_pipeline_layout"),
            bind_group_layouts: &[
                &renderer.global_bind_layout,
                &shadow_targets_bg_layout,
            ],
            push_constant_ranges: &[],
        });

        let target_lathe_points = pot_model();
        let target_lathe_buf = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("target_lathe_buf"),
            contents: bytemuck::cast_slice(&target_lathe_points),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let targets_pipeline_desc = RenderPipelineDescriptor {
            label: Some("pots_above"),
            layout: Some(&targets_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("pot_vert"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: Some("pot_frag"),
                compilation_options: Default::default(),
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
            cache: None,
        };
        let targets_pipeline = gpu.device.create_render_pipeline(&targets_pipeline_desc);
        let targets_refr_pipeline = DeferredRenderer::create_refracted_pipeline(&gpu.device, &targets_pipeline_desc);
        let targets_refl_pipeline = DeferredRenderer::create_reflected_pipeline(&gpu.device, &targets_pipeline_desc);

        let shadow_targets_pipeline_desc = RenderPipelineDescriptor {
            label: Some("pots_above"),
            layout: Some(&shadow_targets_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("pot_vert_shadow"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: None,
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        };
        let shadow_targets_pipeline = gpu.device.create_render_pipeline(&shadow_targets_pipeline_desc);

        let targets_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("pots_buf"),
            size: (size_of::<Target>() * NUM_TARGETS) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let tex_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 4,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..wgpu::SamplerDescriptor::default()
        });

        let pot_co_tex = gpu.load_texture_make_mips::<u32>(assets, "pot-co.png", TextureFormat::Rgba8UnormSrgb, 4).unwrap();
        let pot_nr_tex = gpu.load_texture_make_mips::<u32>(assets, "pot-nr.png", TextureFormat::Rgba8Unorm, 4).unwrap();

        let targets_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("pots_bg"),
            layout: &targets_bg_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: targets_buf.as_entire_binding()},
                BindGroupEntry {binding: 1, resource: target_lathe_buf.as_entire_binding()},
                BindGroupEntry {binding: 2, resource: BindingResource::Sampler(&tex_sampler)},
                BindGroupEntry {binding: 3, resource: BindingResource::TextureView(&pot_co_tex.create_view(&Default::default()))},
                BindGroupEntry {binding: 4, resource: BindingResource::TextureView(&pot_nr_tex.create_view(&Default::default()))},
            ]
        });
        let shadow_targets_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("pots_bg"),
            layout: &shadow_targets_bg_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: targets_buf.as_entire_binding()},
                BindGroupEntry {binding: 1, resource: target_lathe_buf.as_entire_binding()},
            ]
        });

        let smash_sounds = SoundAtlas::load_with_starts(assets, "glass_smash.ogg", -3.0,
            &[0.0, 1.57, 2.84, 4.02, 5.43, 6.98, 8.38, 9.68, 10.97, 12.32, 13.58, 15.30, 16.73, 18.12]).unwrap();

        TargetController {
            targets_pipeline, targets_refr_pipeline, targets_refl_pipeline, shadow_targets_pipeline,
            targets_buf, targets_bg, shadow_targets_bg,
            smash_sounds,
            max_target_inst: 0,

            updated_at: 0.0,
            all_targets,
            targets_hit: 0,
        }
    }

    pub fn reset(&mut self, terrain: &HeightmapTerrain) {
        self.all_targets = Self::gen_targets(NUM_TARGETS, terrain, 40.0);
        self.updated_at = 0.0;
        self.targets_hit = 0;
    }

    pub fn tick(&mut self, time: f64) {
        if time >= 0.0 {
            self.updated_at = time;
        }
    }
}

impl RenderObject for TargetController {
    fn prepass(&mut self, gpu: &GPUContext, renderer: &DeferredRenderer, encoder: &mut CommandEncoder) {
        let planes = renderer.camera.perspective_clipping_planes();

        let mut visible_targets: Vec<Target> = self.all_targets.iter().copied().filter(|t| {
            sphere_visible(planes, t.bottom, 3.0)
        }).collect();

        self.max_target_inst = visible_targets.len() as u32;
        if self.max_target_inst != 0 {
            gpu.queue.write_buffer(&self.targets_buf, 0, bytemuck::cast_slice(&visible_targets));
        }
    }

    fn draw_shadow_casters<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {
        if self.max_target_inst != 0 {
            pass.set_pipeline(&self.shadow_targets_pipeline);
            pass.set_bind_group(1, &self.shadow_targets_bg, &[]);
            pass.draw(0..NUM_POT_VERTS, 0..self.max_target_inst);
        }
    }

    fn draw_underwater<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {
        if self.max_target_inst != 0 {
            pass.set_pipeline(&self.targets_refr_pipeline);
            pass.set_bind_group(1, &self.targets_bg, &[]);
            pass.draw(0..NUM_POT_VERTS, 0..self.max_target_inst);
        }
    }
    fn draw_reflected<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {
        if self.max_target_inst != 0 {
            pass.set_pipeline(&self.targets_refl_pipeline);
            pass.set_bind_group(1, &self.targets_bg, &[]);
            pass.draw(0..NUM_POT_VERTS, 0..self.max_target_inst);
        }
    }

    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {
        if self.max_target_inst != 0 {
            pass.set_pipeline(&self.targets_pipeline);
            pass.set_bind_group(1, &self.targets_bg, &[]);
            pass.draw(0..NUM_POT_VERTS, 0..self.max_target_inst);
        }
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
                    self.targets_hit += 1;

                    audio.play(self.smash_sounds.random_sound()).unwrap();
                }
            }
        }

        was_hit
    }
}
