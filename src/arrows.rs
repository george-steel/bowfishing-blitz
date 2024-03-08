use std::cmp::max;
use std::mem::size_of;
use std::num;
use std::time::Instant;

use glam::*;
use kira::manager::AudioManager;
use kira::sound::Sound;
use wgpu::*;
use crate::audio_util::SoundAtlas;
use crate::deferred_renderer::*;
use crate::gputil::*;
use crate::camera::*;
use crate::terrain_view::HeightmapTerrain;

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
const ARROW_LEN: f32 = 0.7;
const MOVING_ARROW_LEN: f32 = 1.0;
const RAYMARCH_RES: f32 = 0.2;

pub struct ArrowController {
    arrows_above_pipeline: RenderPipeline,
    arrows_below_pipeline: RenderPipeline,
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
                buffers: &[],
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
                buffers: &[],
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
            arrows_buf, arrows_bg,
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
                            log::info!("Hit, t={}", t);
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
            pass.set_bind_group(1, &self.arrows_bg, &[]);
            pass.draw(0..14, min_inst..max_inst);
        }

    }
    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &crate::deferred_renderer::DeferredRenderer, pass: &mut RenderPass<'a>) {
        let min_inst = if self.arrow_is_live {0} else {1};
        let max_inst = self.num_dead_arrows as u32 + 1;
        if min_inst != max_inst {
            pass.set_pipeline(&self.arrows_above_pipeline);
            pass.set_bind_group(1, &self.arrows_bg, &[]);
            pass.draw(0..14, min_inst..max_inst);
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
