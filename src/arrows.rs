use std::cmp::max;
use std::f32::consts::TAU;
use std::mem::size_of;
use std::num;
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

const MAX_DEAD_ARROWS: usize = 64;
const ARROW_SPEED: f32 = 50.0;
const ARROW_LEN: f32 = 1.0;
const MOVING_ARROW_LEN: f32 = 1.5;
const RAYMARCH_RES: f32 = 0.2;

pub struct ArrowController {
    arrows_above_pipeline: RenderPipeline,
    arrows_below_pipeline: RenderPipeline,
    arrows_model: Box<[ArrowVert]>,
    arrows_vertex_buf: Buffer,
    arrows_buf: Buffer,
    arrows_bg: BindGroup,

    release_sounds: SoundAtlas,
    splish_sounds: SoundAtlas,
    thunk_sounds: SoundAtlas,

    all_arrows: Box<[Arrow]>, // live arrow + ring buffer
    num_dead_arrows: usize,
    next_dead_arrow: usize,
    arrow_is_live: bool,
    //arrow_buf: Buffer,
    updated_at: Instant,
}

impl ArrowController {
    pub fn new(gpu: &GPUContext, renderer: &DeferredRenderer, now: Instant) -> Self {
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

        let arrows_above_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("arrows_above"),
            layout: Some(&arrows_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: "arrow_vert_above",
                buffers: &[arrows_vertex_layout.clone()],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: "arrow_frag_above",
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
        });

        let arrows_below_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("arrows_above"),
            layout: Some(&arrows_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: "arrow_vert_below",
                buffers: &[arrows_vertex_layout],
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: "arrow_frag_below",
                targets: DeferredRenderer::UNDERWATER_GBUFFER_TARGETS,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                //cull_mode: Some(wgpu::Face::Back),
                ..PrimitiveState::default()
            },
            depth_stencil: reverse_z(),
            multisample: MultisampleState::default(),
            multiview: None,
        });

        let arrows_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("arrows_buf"),
            size: (size_of::<Arrow>() * (MAX_DEAD_ARROWS + 1)) as u64,
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


        let release_sounds = SoundAtlas::load_with_stride("./assets/arrow_release.ogg", 5.0, 0.4).unwrap();
        let thunk_sounds = SoundAtlas::load_with_stride("./assets/arrow_thunk.ogg", -3.0, 0.5).unwrap();
        let splish_sounds = SoundAtlas::load_with_stride("./assets/water_splish.ogg", -2.0, 1.0).unwrap();

        let all_arrows = bytemuck::zeroed_slice_box(MAX_DEAD_ARROWS + 1);
        ArrowController {
            arrows_above_pipeline, arrows_below_pipeline,
            arrows_model, arrows_vertex_buf, arrows_buf, arrows_bg,
            release_sounds, splish_sounds, thunk_sounds,

            all_arrows,
            num_dead_arrows: 0,
            next_dead_arrow: 0,
            arrow_is_live: false,
            updated_at: now,
        }
    }

    pub fn shoot(&mut self, audio: &mut AudioManager, camera: &impl CameraController) {
        self.arrow_is_live = true;
        let start_pos = camera.eye() - vec3(0.0, 0.0, 0.08);
        let dir = camera.look_dir().normalize();
        let end_pos = start_pos + dir * MOVING_ARROW_LEN;
        self.all_arrows[0] = Arrow {
            end_pos, dir, state: 1, len: MOVING_ARROW_LEN,
        };
        audio.play(self.release_sounds.random_sound()).unwrap();
    }

    pub fn tick(&mut self, now: Instant, terrain: &HeightmapTerrain, audio: &mut AudioManager, targets: &mut[&mut dyn ArrowTarget]) -> bool {
        let mut did_hit = false;
        if self.arrow_is_live {
            let delta_t = now - self.updated_at;
            let live_arrow = self.all_arrows[0];
            let old_pos = live_arrow.end_pos;
            let frame_dist = delta_t.as_secs_f32() * ARROW_SPEED;
            let mut new_pos = old_pos + frame_dist * live_arrow.dir;

            let num_tests = (frame_dist / RAYMARCH_RES).ceil() as u32;
            let delta_p = live_arrow.dir * frame_dist / (num_tests as f32);
            let mut last_pos = old_pos;
            let mut last_height = f32::INFINITY;
            for i in 0..=num_tests {
                let test_pos = old_pos + delta_p * (i as f32) ;
                match terrain.height_at(test_pos.xy()) {
                    None => {
                        self.arrow_is_live = false;
                        break;
                    }
                    Some(h) => {
                        if i > 0 && h > test_pos.z {
                            let t = (last_pos.z - last_height) / (last_pos.z - last_height + h - test_pos.z);
                            let stop_pos = last_pos + t * delta_p;
                            new_pos = stop_pos;
                            self.arrow_is_live = false;

                            // put this arrow into the ring buffer
                            self.all_arrows[self.next_dead_arrow + 1] = Arrow {
                                end_pos: stop_pos, state: 0, dir: live_arrow.dir, len: ARROW_LEN,
                            };
                            self.next_dead_arrow = (self.next_dead_arrow + 1) % MAX_DEAD_ARROWS;
                            self.num_dead_arrows = MAX_DEAD_ARROWS.min(self.num_dead_arrows + 1);

                            audio.play(self.thunk_sounds.random_sound()).unwrap();
                            break;
                        }
                        last_height = h;
                        last_pos = test_pos;
                    }
                }
            }
            self.all_arrows[0].end_pos = new_pos;

            if self.arrow_is_live && old_pos.z > 0.0 && new_pos.z <= 0.0 {
                audio.play(self.splish_sounds.random_sound()).unwrap();
            }

            for target in targets.iter_mut() {
                let hit_target =  target.process_hits(audio, old_pos, new_pos);
                did_hit = did_hit || hit_target;
            }

            // collide with targets and water
        }
        self.updated_at = now;
        did_hit
    }

}

impl RenderObject for ArrowController {
    fn prepass(&mut self, gpu: &GPUContext, renderer: &DeferredRenderer, encoder: &mut CommandEncoder) {
        gpu.queue.write_buffer(&self.arrows_buf, 0, bytemuck::cast_slice(&self.all_arrows));
    }

    fn draw_underwater<'a>(&'a self, gpu: &GPUContext, renderer: &crate::deferred_renderer::DeferredRenderer, pass: &mut RenderPass<'a>) {
        let min_inst = if self.arrow_is_live {0} else {1};
        let max_inst = self.num_dead_arrows as u32 + 1;
        if min_inst != max_inst {
            pass.set_pipeline(&self.arrows_below_pipeline);
            pass.set_vertex_buffer(0, self.arrows_vertex_buf.slice(..));
            pass.set_bind_group(1, &self.arrows_bg, &[]);
            pass.draw(0..(self.arrows_model.len() as u32), min_inst..max_inst);
        }

    }
    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &crate::deferred_renderer::DeferredRenderer, pass: &mut RenderPass<'a>) {
        let min_inst = if self.arrow_is_live {0} else {1};
        let max_inst = self.num_dead_arrows as u32 + 1;
        if min_inst != max_inst {
            pass.set_pipeline(&self.arrows_above_pipeline);
            pass.set_vertex_buffer(0, self.arrows_vertex_buf.slice(..));
            pass.set_bind_group(1, &self.arrows_bg, &[]);
            pass.draw(0..(self.arrows_model.len() as u32), min_inst..max_inst);
        }

    }
}

pub trait ArrowTarget {
    fn process_hits(&mut self, audio: &mut AudioManager, start: Vec3, end: Vec3) -> bool;
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
