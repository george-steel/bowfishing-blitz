use std::cmp::max;
use std::collections::VecDeque;
use std::f32::consts::TAU;
use std::mem::size_of;
use std::num;
use std::slice::from_ref;
use std::time::Instant;

use glam::*;
use kira::manager::AudioManager;
use kira::sound::Sound;
use wgpu::util::BufferInitDescriptor;
use wgpu::util::DeviceExt;
use wgpu::*;
use crate::audio_util::SoundAtlas;
use crate::deferred_renderer::*;
use crate::gputil::*;
use crate::camera::*;
use crate::terrain_view::HeightmapTerrain;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ArrowVert {
    pos: Vec3,
    norm: Vec3,
    uv: Vec2,
}

fn arr_vert(pos: Vec3, norm: Vec3, uv: Vec2) -> ArrowVert {
    ArrowVert { pos, norm: norm.normalize(), uv}
}

fn orthogonal_verts(verts: &[ArrowVert], xform: Mat3, v_low: f32, v_high: f32) -> Vec<ArrowVert> {
    let mut out = Vec::new();
    for v in verts {
        out.push(ArrowVert {
            pos: xform * v.pos,
            norm: xform * v.norm,
            uv: vec2(v.uv.x, v_low.lerp(v_high, v.uv.y)),
        });
    }
    out
}

fn arrow_model() -> Box<[ArrowVert]> {
    // modeled on graph paper
    let mut out = Vec::new();
    let arrow_quarter = [
        // head
        arr_vert(vec3(0.0, 0.02, 0.0), vec3(-2.0, 1.0, 4.0), vec2(0.0, 0.0)),
        arr_vert(vec3(-0.04, -0.06, 0.0), vec3(-2.0, 0.0, 3.0), vec2(0.1, 1.0)),
        arr_vert(vec3(0.0, -0.06, 0.02), vec3(0.0, 0.0, 1.0), vec2(0.1, 0.0)),
        arr_vert(vec3(-0.02, -0.12, 0.0), vec3(-2.0, -1.0, -0.0), vec2(0.2, 1.0)),
        arr_vert(vec3(0.0, -0.12, 0.02), vec3(0.0, 0.0, 1.0), vec2(0.2, 0.0)),
        //shaft
        arr_vert(vec3(-0.02, -0.12, 0.0), vec3(-1.0, 0.0, 0.0), vec2(0.2, 1.0)),
        arr_vert(vec3(0.0, -1.00, 0.02), vec3(0.0, 0.0, 1.0), vec2(0.7, 0.0)),
        arr_vert(vec3(-0.02, -1.00, 0.0), vec3(-1.0, 0.0, 0.0), vec2(0.7, 1.0)),
        //nock
        arr_vert(vec3(0.0, -1.01, 0.0), vec3(0.0, -1.0, 0.0), vec2(0.71, 0.0)),
        //reset strip
        arr_vert(Vec3::NAN, Vec3::ZERO, Vec2::NAN),
    ];
    // mirror with D4 symmetry
    out.extend_from_slice(&orthogonal_verts(&arrow_quarter, Mat3::IDENTITY, 0.25, 0.5));
    out.extend_from_slice(&orthogonal_verts(&arrow_quarter, Mat3::from_diagonal(vec3(-1.0, 1.0, -1.0)), 0.75, 1.0));
    out.push(arr_vert(Vec3::NAN, Vec3::ZERO, Vec2::NAN)); // switch parity
    out.extend_from_slice(&orthogonal_verts(&arrow_quarter, Mat3::from_diagonal(vec3(1.0, 1.0, -1.0)), 0.75, 0.5));
    out.extend_from_slice(&orthogonal_verts(&arrow_quarter, Mat3::from_diagonal(vec3(-1.0, 1.0, 1.0)), 0.25, 0.0));

    let fletching = [
        arr_vert(vec3(0.0, -0.78, 0.0), vec3(2.0, 0.0, 6.0), vec2(0.8, 0.0)),
        arr_vert(vec3(0.058, -0.81, -0.015), vec3(1.5, 1.0, 6.0), vec2(1.0, 0.0)),
        arr_vert(vec3(0.0, -0.84, 0.0), vec3(1.0, 0.0, 6.0), vec2(0.8, 0.333)),
        arr_vert(vec3(0.06, -0.87, -0.005), vec3(0.5, 1.0, 6.0), vec2(1.0, 0.333)),
        arr_vert(vec3(0.0, -0.90, 0.0), vec3(0.0, 0.0, 6.0), vec2(0.8, 0.667)),
        arr_vert(vec3(0.06, -0.93, 0.005), vec3(-0.5, 1.0, 6.0), vec2(1.0, 0.667)),
        arr_vert(vec3(0.0, -0.96, 0.0), vec3(-1.0, 0.0, 6.0), vec2(0.8, 1.0)),
        arr_vert(vec3(0.06, -0.99, 0.005), vec3(-1.5, 1.0, 6.0), vec2(1.0, 1.0)),
        //needed twice for parity
        arr_vert(Vec3::NAN, Vec3::ZERO, Vec2::NAN),
        arr_vert(Vec3::NAN, Vec3::ZERO, Vec2::NAN),
    ];
    // repeat with Z3 symmetry
    out.extend_from_slice(&fletching);
    out.extend_from_slice(&orthogonal_verts(&fletching, Mat3::from_axis_angle(Vec3::Y, TAU / 3.0), 0.0, 1.0));
    out.extend_from_slice(&orthogonal_verts(&fletching, Mat3::from_axis_angle(Vec3::Y, 2.0 * TAU / 3.0), 0.0, 1.0));

    out.into_boxed_slice()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Arrow {
    end_pos: Vec3,
    state: u32, // 0 = dead, 1 = live
    dir: Vec3, // normalized
    len: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Splish {
    center: Vec2,
    start_time: f32,
}

const MAX_DEAD_ARROWS: usize = 64;
const MAX_LIVE_ARROWS: usize = 4;
const ARROW_SPEED: f32 = 50.0;
const ARROW_LEN: f32 = 1.0;
const MOVING_ARROW_LEN: f32 = 1.5;
const RAYMARCH_RES: f32 = 0.2;
const MAX_SPLISHES: usize = 16;
const SPLISH_DURATION: f32 = 2.0;

pub struct ArrowController {
    arrows_pipeline: RenderPipeline,
    arrows_refr_pipeline: RenderPipeline,
    arrows_refl_pipeline: RenderPipeline,
    shadow_arrows_pipeline: RenderPipeline,
    arrows_model: Box<[ArrowVert]>,
    arrows_vertex_buf: Buffer,
    arrows_buf: Buffer,
    arrows_bg: BindGroup,
    max_arrow_inst: u32,

    splish_pipeline: RenderPipeline,
    splish_buf: Buffer,
    max_splish_inst: u32,
    all_splishes: VecDeque<Splish>,

    release_sounds: SoundAtlas,
    splish_sounds: SoundAtlas,
    thunk_sounds: SoundAtlas,

    dead_arrows: Box<[Arrow]>, // ring buffer
    num_dead_arrows: usize,
    next_dead_arrow: usize,
    live_arrows: Vec<Arrow>,
    pub arrows_shot: u32,
    updated_at: f64,
}

impl ArrowController {
    pub fn new(gpu: &GPUContext, assets: &impl AssetSource, renderer: &DeferredRenderer) -> Self {
        let shaders = gpu.device.create_shader_module(ShaderModuleDescriptor{
            label: Some("arrows.wgsl"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(crate::shaders::ARROWS)),
        });

        let arrows_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("arrows_bg_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                }
            ]
        });

        let arrows_model = arrow_model();

        let arrows_vertex_layout = VertexBufferLayout {
            array_stride: size_of::<ArrowVert>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
        };

        let arrows_vertex_buf = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("arrows_vertex_buf"),
            contents: bytemuck::cast_slice(&arrows_model),
            usage: BufferUsages::VERTEX,
        });

        let arrows_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("arrows_pipeline_layout"),
            bind_group_layouts: &[
                &renderer.global_bind_layout,
                &arrows_bg_layout,
            ],
            push_constant_ranges: &[],
        });

        let arrows_above_pipeline_desc = RenderPipelineDescriptor {
            label: Some("arrows"),
            layout: Some(&arrows_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("arrow_vert"),
                compilation_options: Default::default(),
                buffers: &[arrows_vertex_layout.clone()],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: Some("arrow_frag"),
                compilation_options: Default::default(),
                targets: DeferredRenderer::GBUFFER_TARGETS,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                //cull_mode: Some(wgpu::Face::Back),
                ..PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        };

        let arrows_pipeline = gpu.device.create_render_pipeline(&arrows_above_pipeline_desc);
        let arrows_refr_pipeline = DeferredRenderer::create_refracted_pipeline(&gpu.device, &arrows_above_pipeline_desc);
        let arrows_refl_pipeline = DeferredRenderer::create_reflected_pipeline(&gpu.device, &arrows_above_pipeline_desc);

        let shadow_arrows_pipeline_desc = RenderPipelineDescriptor {
            label: Some("arrows"),
            layout: Some(&arrows_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("arrow_vert_shadow"),
                compilation_options: Default::default(),
                buffers: &[arrows_vertex_layout.clone()],
            },
            fragment: None,
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                //cull_mode: Some(wgpu::Face::Back),
                ..PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        };
        let shadow_arrows_pipeline = gpu.device.create_render_pipeline(&shadow_arrows_pipeline_desc);

        let arrows_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("arrows_buf"),
            size: (size_of::<Arrow>() * (MAX_DEAD_ARROWS + MAX_LIVE_ARROWS)) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false
        });
        let arrows_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("arrows_bg"),
            layout: &arrows_bg_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: arrows_buf.as_entire_binding()}
            ]
        });

        let splish_vertex_layout = VertexBufferLayout {
            array_stride: size_of::<Splish>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: &vertex_attr_array![0 => Float32x2, 1 => Float32],
        };
        let splish_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("splish_pipeline_layout"),
            bind_group_layouts: &[&renderer.global_bind_layout],
            push_constant_ranges: &[],
        });
        // decal pipeline
        let splish_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("splish_pipeline"),
            layout: Some(&splish_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: Some("splish_vert"),
                compilation_options: Default::default(),
                buffers: from_ref(&splish_vertex_layout),
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: Some("splish_frag"),
                compilation_options: Default::default(),
                targets: &[
                    Some(ColorTargetState{ format: TextureFormat::Rg11b10Ufloat, blend: None, write_mask: ColorWrites::empty() }),
                    Some(ColorTargetState{
                        format: TextureFormat::Rgb10a2Unorm,
                        blend: Some(BlendState {
                            color: BlendComponent { src_factor: BlendFactor::SrcAlpha, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add },
                            alpha: BlendComponent { src_factor: BlendFactor::SrcAlpha, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add },
                        }),
                        write_mask: ColorWrites::ALL
                    }),
                    Some(ColorTargetState{ format: TextureFormat::Rg8Unorm, blend: None, write_mask: ColorWrites::empty() }),
                    Some(ColorTargetState{ format: TextureFormat::R8Unorm, blend: None, write_mask: ColorWrites::empty() }),
                    Some(ColorTargetState{ format: TextureFormat::R8Uint, blend: None, write_mask: ColorWrites::empty() }),
                ],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::GreaterEqual,
                stencil: StencilState::default(),
                bias: DepthBiasState {
                    constant: 16, // in ULP,
                    slope_scale: 1.0,
                    clamp: 0.001,
                },
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let splish_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("splish_buf"),
            size: (size_of::<Splish>() * MAX_SPLISHES) as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });


        let release_sounds = SoundAtlas::load_with_stride(assets, "arrow_release.ogg", 5.0, 0.4).unwrap();
        let thunk_sounds = SoundAtlas::load_with_stride(assets, "arrow_thunk.ogg", -3.0, 0.5).unwrap();
        let splish_sounds = SoundAtlas::load_with_stride(assets, "water_splish.ogg", -2.0, 1.0).unwrap();

        let dead_arrows = bytemuck::zeroed_slice_box(MAX_DEAD_ARROWS);
        ArrowController {
            arrows_pipeline, arrows_refr_pipeline, arrows_refl_pipeline, shadow_arrows_pipeline,
            arrows_model, arrows_vertex_buf, arrows_buf, arrows_bg,
            release_sounds, splish_sounds, thunk_sounds,
            max_arrow_inst: 0,
            splish_pipeline, splish_buf,
            all_splishes: VecDeque::new(),
            max_splish_inst: 0,

            dead_arrows,
            num_dead_arrows: 0,
            next_dead_arrow: 0,
            live_arrows: Vec::new(),
            arrows_shot: 0,
            updated_at: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.num_dead_arrows = 0;
        self.next_dead_arrow = 0;
        self.live_arrows.clear();
        self.arrows_shot = 0;
        self.updated_at = 0.0;
    }

    pub fn shoot(&mut self, audio: Option<&mut AudioManager>, camera: &impl CameraController) {
        self.arrows_shot += 1;
        let eye = camera.eye();
        let start_pos = eye - vec3(0.0, 0.0, 0.08);
        let dir = camera.look_dir().normalize();
        let end_pos = start_pos + dir * MOVING_ARROW_LEN;
        let arrow = Arrow {
            end_pos, dir, state: 1, len: MOVING_ARROW_LEN,
        };

        if self.live_arrows.len() < MAX_LIVE_ARROWS {
            self.live_arrows.push(arrow);
        } else {
            // replace the furthest-away (oldest) live arrow if full
            let mut dmax = 0.0;
            let mut imax = 0;
            for i in 0..self.live_arrows.len() {
                let d = (self.live_arrows[i].end_pos - eye).length();
                if d > dmax {
                    dmax = d;
                    imax = i;
                }
            }
            self.live_arrows[imax] = arrow;
        }
        if let Some(audio) = audio {
            audio.play(self.release_sounds.random_sound()).unwrap();
        }
    }

    pub fn tick(&mut self, time: f64, terrain: &HeightmapTerrain, mut audio: Option<&mut AudioManager>, targets: &mut[&mut dyn ArrowTarget]) -> bool {
        if time <= 0.0 {
            return false
        }

        while let Some(s) = self.all_splishes.front() {
            if s.start_time + SPLISH_DURATION < time as f32 {
                self.all_splishes.pop_front();
            } else {
                break;
            }
        }

        let mut did_hit = false;
        let delta_t = time - self.updated_at;

        self.live_arrows.retain_mut(|live_arrow: &mut Arrow| {
            let mut stays_live = true;
            let old_pos = live_arrow.end_pos;
            let frame_dist = delta_t as f32 * ARROW_SPEED;
            let mut new_pos = old_pos + frame_dist * live_arrow.dir;

            let num_tests = (frame_dist / RAYMARCH_RES).ceil() as u32;
            let delta_p = live_arrow.dir * frame_dist / (num_tests as f32);
            let mut last_pos = old_pos;
            let mut last_height = f32::INFINITY;
            for i in 0..=num_tests {
                let test_pos = old_pos + delta_p * (i as f32) ;
                match terrain.height_at(test_pos.xy()) {
                    None => {
                        stays_live = false;
                        break;
                    }
                    Some(h) => {
                        if i > 0 && h > test_pos.z {
                            let t = (last_pos.z - last_height) / (last_pos.z - last_height + h - test_pos.z);
                            let stop_pos = last_pos + t * delta_p;
                            new_pos = stop_pos;
                            stays_live = false;

                            // put this arrow into the ring buffer
                            self.dead_arrows[self.next_dead_arrow ] = Arrow {
                                end_pos: stop_pos, state: 0, dir: live_arrow.dir, len: ARROW_LEN,
                            };
                            self.next_dead_arrow = (self.next_dead_arrow + 1) % MAX_DEAD_ARROWS;
                            self.num_dead_arrows = MAX_DEAD_ARROWS.min(self.num_dead_arrows + 1);

                            if let Some(audio) = &mut audio {
                                audio.play(self.thunk_sounds.random_sound()).unwrap();
                            }
                            break;
                        }
                        last_height = h;
                        last_pos = test_pos;
                    }
                }
            }
            live_arrow.end_pos = new_pos;

            if old_pos.z > 0.0 && new_pos.z <= 0.0 {
                if let Some(audio) = &mut audio {
                    audio.play(self.splish_sounds.random_sound()).unwrap();
                }
                if self.all_splishes.len() >= MAX_SPLISHES {
                    self.all_splishes.pop_front();
                }
                self.all_splishes.push_back(Splish {
                    center: old_pos.xy().lerp(new_pos.xy(), old_pos.z / (old_pos.z - new_pos.z)),
                    start_time: time as f32,
                });
            }

            for target in targets.iter_mut() {
                let hit_target =  target.process_hits(audio.as_deref_mut(), old_pos, new_pos);
                did_hit = did_hit || hit_target;
            }

            stays_live
        });
        self.updated_at = time;
        did_hit
    }

}

impl RenderObject for ArrowController {
    fn prepass(&mut self, gpu: &GPUContext, renderer: &DeferredRenderer, encoder: &mut CommandEncoder) {
        let planes = renderer.camera.perspective_clipping_planes();
        let mut visible_arrows: Vec<Arrow> = self.dead_arrows.iter().copied().filter(|arr| {
            sphere_visible(planes, arr.end_pos, 1.5 * arr.len)
        }).collect();
        visible_arrows.append(&mut self.live_arrows.iter().copied().filter(|arr| {
            sphere_visible(planes, arr.end_pos, 1.5 * arr.len)
        }).collect());

        self.max_arrow_inst = visible_arrows.len() as u32;
        if self.max_arrow_inst != 0 {
            gpu.queue.write_buffer(&self.arrows_buf, 0, bytemuck::cast_slice(&visible_arrows));
        }

        let visible_splishes: Vec<Splish> = self.all_splishes.iter().copied().filter(|s| {
            sphere_visible(planes, (s.center, 0.0).into(), 1.0)
        }).collect();
        self.max_splish_inst = visible_splishes.len() as u32;
        if self.max_splish_inst != 0 {
            gpu.queue.write_buffer(&self.splish_buf, 0, bytemuck::cast_slice(&visible_splishes));
        }
    }

    fn draw_shadow_casters<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {
        if self.max_arrow_inst != 0  {
            pass.set_pipeline(&self.shadow_arrows_pipeline);
            pass.set_vertex_buffer(0, self.arrows_vertex_buf.slice(..));
            pass.set_bind_group(1, &self.arrows_bg, &[]);
            pass.draw(0..(self.arrows_model.len() as u32), 0..self.max_arrow_inst);
        }
    }

    fn draw_underwater<'a>(&'a self, gpu: &GPUContext, renderer: &crate::deferred_renderer::DeferredRenderer, pass: &mut RenderPass<'a>) {
        if self.max_arrow_inst != 0 {
            pass.set_pipeline(&self.arrows_refr_pipeline);
            pass.set_vertex_buffer(0, self.arrows_vertex_buf.slice(..));
            pass.set_bind_group(1, &self.arrows_bg, &[]);
            pass.draw(0..(self.arrows_model.len() as u32), 0..self.max_arrow_inst);
        }

    }
    fn draw_reflected<'a>(&'a self, gpu: &GPUContext, renderer: &crate::deferred_renderer::DeferredRenderer, pass: &mut RenderPass<'a>) {
        if self.max_arrow_inst != 0 {
            pass.set_pipeline(&self.arrows_refl_pipeline);
            pass.set_vertex_buffer(0, self.arrows_vertex_buf.slice(..));
            pass.set_bind_group(1, &self.arrows_bg, &[]);
            pass.draw(0..(self.arrows_model.len() as u32), 0..self.max_arrow_inst);
        }

    }
    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &crate::deferred_renderer::DeferredRenderer, pass: &mut RenderPass<'a>) {
        if self.max_arrow_inst != 0  {
            pass.set_pipeline(&self.arrows_pipeline);
            pass.set_vertex_buffer(0, self.arrows_vertex_buf.slice(..));
            pass.set_bind_group(1, &self.arrows_bg, &[]);
            pass.draw(0..(self.arrows_model.len() as u32), 0..self.max_arrow_inst);
        }
        if self.max_splish_inst != 0 {
            pass.set_pipeline(&self.splish_pipeline);
            pass.set_vertex_buffer(0, self.splish_buf.slice(..));
            pass.draw(0..4, 0..self.max_splish_inst);
        }

    }
}

pub trait ArrowTarget {
    fn process_hits(&mut self, audio: Option<&mut AudioManager>, start: Vec3, end: Vec3) -> bool;
}

pub fn collide_ray_sphere(start: Vec3, end: Vec3, center: Vec3, radius: f32) -> bool {
    let delta = end - start;
    let proj = delta.dot(center - start) / delta.length_squared();
    if proj <= 0.0 {
        (center - start).length_squared() < radius * radius
    } else if proj >= 1.0 {
        (center - end).length_squared() < radius * radius
    } else {
        let perp = center - start - proj * delta;
        perp.length_squared() < radius * radius
    }
}
