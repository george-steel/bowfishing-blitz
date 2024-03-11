use std::{default, mem::size_of, sync::Arc, time::{Duration, Instant}};
use image::imageops::FilterType::Nearest;
use wgpu::{util::{BufferInitDescriptor, DeviceExt}, *};
use glam::*;

use crate::{arrows::ArrowController, boat_rail::RailController, deferred_renderer::{DeferredRenderer, RenderObject}, gputil::{load_png, GPUContext}, targets::TargetController};

// state machine for title screen, pausing, and restart
#[derive(Clone, Copy, Debug)]
pub enum GameState {
    Title {started_at: Instant, is_restart: bool},
    Fade {done_at: Instant},
    Countdown {done_at: Instant},
    Playing,
    Paused,
    Finish {done_at: Instant},
}

impl GameState {
    pub const FADE_DURATION: Duration = Duration::from_millis(600);
    pub const COUNTDOWN_DURATION: Duration = Duration::from_millis(4000);
    pub const FINISH_DURATION: Duration = Duration::from_millis(5000);
    pub const GAME_PERIOD: f64 = 180.0;

    pub fn is_playing(&self) -> bool {
        if let GameState::Playing = self { true } else {false}
    }
    pub fn is_paused(&self) -> bool {
        if let GameState::Paused = self { true } else {false}
    }

    pub fn should_reset_world(&self, now: Instant) -> bool {
        if let GameState::Fade { done_at } = *self {
            done_at <= now
        } else {
            false
        }
    }

    pub fn do_timeout(&mut self, now: Instant) {
        match *self {
            GameState::Title {..} => {},
            GameState::Fade { done_at } => {
                if done_at <= now {
                    *self = GameState::Countdown {done_at: now + GameState::COUNTDOWN_DURATION};
                }
            },
            GameState::Countdown { done_at } => {
                if done_at <= now {
                    *self = GameState::Playing;
                }
            },
            GameState::Playing => {},
            GameState::Paused => {},
            GameState::Finish { done_at } => {
                if done_at <= now {
                    *self = GameState::Title { started_at: now, is_restart: true }
                }
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SDFTextParams {
    viewport_loc: Vec2,
    size_vh: Vec2,
    color: Vec4,
    shadow_color: Vec4,
    shadow_size: f32,
    margin_vh: f32,
    sdf_rad: f32,
    num_chars: u32,
    chars: [u32;4],
}

pub struct UIDisplay {
    text_pipeline: RenderPipeline,
    title_buf: Buffer,
    states_buf: Buffer,
    numbers_buf: Buffer,
    title_bg: BindGroup,
    states_bg: BindGroup,
    numbers_bg: BindGroup,


    enc_music: Arc<[u8]>,

    // for display
    old_state: GameState,
    cycle_time: f64,
    arrows_shot: u32,
    targets_hit: u32,
    secs_left: u32,
}

impl UIDisplay {
    pub fn new(gpu: &GPUContext, renderer: &DeferredRenderer) -> Self {
        let title_img = load_png::<u8>("./assets/title.sdf.png").unwrap();
        let title_atlas = gpu.upload_texture_atlas("title_atlas", TextureFormat::R8Unorm, &title_img, 1);
        let states_img = load_png::<u8>("./assets/states.sdf.png").unwrap();
        let states_atlas = gpu.upload_texture_atlas("states_atlas", TextureFormat::R8Unorm, &states_img, 4);
        let numbers_img = load_png::<u8>("./assets/numbers.sdf.png").unwrap();
        let numbers_atlas = gpu.upload_texture_atlas("numbers_atlas", TextureFormat::R8Unorm, &numbers_img, 14);

        let bilinear_sampler = gpu.device.create_sampler(&SamplerDescriptor {
            label: Some("bilinear_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..Default::default()
        });

        let shaders = gpu.device.create_shader_module(ShaderModuleDescriptor{
            label: Some("ui.wgsl"),
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(crate::shaders::UI)),
        });

        let text_bg_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("text_bg_layout"),
            entries: &[
                BindGroupLayoutEntry{
                    binding: 0, visibility: ShaderStages::VERTEX_FRAGMENT, count: None,
                    ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }
                },
                BindGroupLayoutEntry{
                    binding: 1, visibility: ShaderStages::VERTEX_FRAGMENT, count: None,
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2Array, multisampled: false }
                },
                BindGroupLayoutEntry{
                    binding: 2, visibility: ShaderStages::VERTEX_FRAGMENT, count: None,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering)
                },
            ]
        });

        let text_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("text_pipeline_layout"),
            bind_group_layouts: &[&renderer.global_bind_layout, &text_bg_layout],
            push_constant_ranges: &[],
        });

        let text_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("text_pipeline"),
            layout: Some(&text_pipeline_layout),
            vertex: VertexState {
                module: &shaders,
                entry_point: "sdf_text_vert",
                buffers: &[],
            },
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            fragment: Some(FragmentState {
                module: &shaders,
                entry_point: "sdf_text_frag",
                targets: &[Some(ColorTargetState{
                    format: gpu.output_format,
                    blend: Some(BlendState {
                        color: BlendComponent::OVER,
                        alpha: BlendComponent::OVER,
                    }),
                    write_mask: ColorWrites::ALL })],
            }),
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
        });

        let title_params = SDFTextParams {
            viewport_loc: vec2(0.5, 0.4),
            size_vh: vec2(1.0, 0.25),
            color: vec4(1.0, 1.0, 1.0, 1.0),
            shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
            shadow_size: 1.0,
            margin_vh: 0.05,
            sdf_rad: 20.0,
            num_chars: 1,
            chars: [0, 0, 0, 0],
        };

        let title_buf = gpu.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("title_buf"),
            contents: bytemuck::bytes_of(&title_params),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let title_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("title_bg"),
            layout: &text_bg_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: title_buf.as_entire_binding()},
                BindGroupEntry {binding: 1, resource: BindingResource::TextureView(&title_atlas.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    format: Some(TextureFormat::R8Unorm),
                    ..Default::default()
                }))},
                BindGroupEntry {binding: 2, resource: BindingResource::Sampler(&bilinear_sampler)},
            ]
        });

        let states_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("states_buf"),
            size: size_of::<SDFTextParams>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let states_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("states_bg"),
            layout: &text_bg_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: states_buf.as_entire_binding()},
                BindGroupEntry {binding: 1, resource: BindingResource::TextureView(&states_atlas.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    format: Some(TextureFormat::R8Unorm),
                    ..Default::default()
                }))},
                BindGroupEntry {binding: 2, resource: BindingResource::Sampler(&bilinear_sampler)},
            ]
        });

        let numbers_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("numbers_buf"),
            size: 5 * size_of::<SDFTextParams>() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false
        });

        let numbers_bg = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("numbers_bg"),
            layout: &text_bg_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: numbers_buf.as_entire_binding()},
                BindGroupEntry {binding: 1, resource: BindingResource::TextureView(&numbers_atlas.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    format: Some(TextureFormat::R8Unorm),
                    ..Default::default()
                }))},
                BindGroupEntry {binding: 2, resource: BindingResource::Sampler(&bilinear_sampler)},
            ]
        });


        let enc_music = std::fs::read("./assets/river_valley_breakdown.ogg").unwrap().into();

        UIDisplay {
            text_pipeline,
            title_buf, states_buf, numbers_buf,
            title_bg, states_bg, numbers_bg,
            
            enc_music,

            old_state: GameState::Title { started_at: Instant::now(), is_restart: false },
            cycle_time: 0.0,
            arrows_shot: 0,
            targets_hit: 0,
            secs_left: 0,
        }
    }

    pub fn tick(&mut self, new_state: GameState, now: Instant, camera: &RailController, arrows: &ArrowController, targets: &TargetController) {
        self.cycle_time = camera.current_time;
        self.arrows_shot = arrows.arrows_shot;
        self.targets_hit = targets.targets_hit;
        self.secs_left = (GameState::GAME_PERIOD - camera.current_time).ceil() as u32;
        self.old_state = new_state;
    }
}

impl RenderObject for UIDisplay {
    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut wgpu::RenderPass<'a>) { }

    fn draw_transparent<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut wgpu::RenderPass<'a>) {
        match self.old_state {
            GameState::Title {..} | GameState::Fade {..} => {
                pass.set_pipeline(&self.text_pipeline);
                pass.set_bind_group(1, &self.title_bg, &[]);
                pass.draw(0..4, 0..1);
            }
            _ => {}
        }

        let maybe_state_data = match self.old_state {
            GameState::Title {..} => Some(SDFTextParams {
                viewport_loc: vec2(0.5, 0.8),
                size_vh: vec2(0.6, 0.15),
                shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
                color: vec4(1.0, 1.0, 1.0, 1.0),
                shadow_size: 0.6,
                margin_vh: 0.05,
                sdf_rad: 20.0,
                num_chars: 1,
                chars: [0, 0, 0, 0],
            }),
            GameState::Playing => {
                if self.cycle_time < 0.8 {
                    Some(SDFTextParams {
                        viewport_loc: vec2(0.5, 0.2),
                        size_vh: vec2(0.8, 0.2),
                        shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
                        color: vec4(1.0, 1.0, 1.0, 1.0),
                        shadow_size: 0.6,
                        margin_vh: 0.05,
                        sdf_rad: 20.0,
                        num_chars: 1,
                        chars: [1, 0, 0, 0],
                    })
                } else {None}
            }
            GameState::Paused => Some(SDFTextParams {
                viewport_loc: vec2(0.5, 0.5),
                size_vh: vec2(0.8, 0.2),
                shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
                color: vec4(1.0, 1.0, 1.0, 1.0),
                shadow_size: 0.6,
                margin_vh: 0.05,
                sdf_rad: 20.0,
                num_chars: 1,
                chars: [2, 0, 0, 0],
            }),
            GameState::Finish {..} => Some(SDFTextParams {
                viewport_loc: vec2(0.5, 0.5),
                size_vh: vec2(0.8, 0.2),
                shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
                color: vec4(1.0, 1.0, 1.0, 1.0),
                shadow_size: 0.6,
                margin_vh: 0.05,
                sdf_rad: 20.0,
                num_chars: 1,
                chars: [3, 0, 0, 0],
            }),
            _ => None
        };
        if let Some(state_data) = maybe_state_data {
            gpu.queue.write_buffer(&self.states_buf, 0, bytemuck::bytes_of(&state_data));
            pass.set_pipeline(&self.text_pipeline);
            pass.set_bind_group(1, &self.states_bg, &[]);
            pass.draw(0..4, 0..1);
        }

        let mut numbers_data = Vec::new();
        let num_arrows = 999.min(self.arrows_shot);
        numbers_data.push(SDFTextParams {
            viewport_loc: vec2(0.0, 0.0),
            size_vh: vec2(0.15, 0.05),
            shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
            color: vec4(1.0, 1.0, 1.0, 1.0),
            shadow_size: 1.0,
            margin_vh: 0.05,
            sdf_rad: 12.0,
            num_chars: 4,
            chars: [12, num_arrows / 100, (num_arrows / 10) % 10, num_arrows % 10],
        });
        let num_targets = 999.min(self.targets_hit);
        numbers_data.push(SDFTextParams {
            viewport_loc: vec2(1.0, 0.0),
            size_vh: vec2(0.15, 0.05),
            shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
            color: vec4(1.0, 1.0, 1.0, 1.0),
            shadow_size: 1.0,
            margin_vh: 0.05,
            sdf_rad: 12.0,
            num_chars: 4,
            chars: [10, num_targets / 100, (num_targets / 10) % 10, num_targets % 10],
        });

        match self.old_state {
            GameState::Playing | GameState::Paused => {
                numbers_data.push(SDFTextParams {
                    viewport_loc: vec2(0.5, 0.0),
                    size_vh: vec2(0.15, 0.05),
                    shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
                    color: vec4(1.0, 1.0, 1.0, 1.0),
                    shadow_size: 1.0,
                    margin_vh: 0.05,
                    sdf_rad: 12.0,
                    num_chars: 4,
                    chars: [11, self.secs_left / 100, (self.secs_left / 10) % 10, self.secs_left % 10],
                });
            }
            _ => {}
        }

        if let GameState::Countdown {..} = self.old_state {
            let countdown_num = (-self.cycle_time).ceil() as u32;
            if countdown_num > 0 && countdown_num <= 3 {
                numbers_data.push(SDFTextParams {
                    viewport_loc: vec2(0.5, 0.2),
                    size_vh: vec2(0.15, 0.2),
                    shadow_color: vec4(0.0, 0.0, 0.0, 0.7),
                    color: vec4(1.0, 1.0, 1.0, 1.0),
                    shadow_size: 1.0,
                    margin_vh: 0.05,
                    sdf_rad: 12.0,
                    num_chars: 1,
                    chars: [countdown_num, 0, 0, 0],
                });
            }
        }

        match self.old_state {
            GameState::Playing | GameState::Countdown {..} => {
                numbers_data.push(SDFTextParams {
                    viewport_loc: vec2(0.5, 0.5),
                    size_vh: vec2(0.06, 0.08),
                    shadow_color: vec4(0.3, 0.3, 0.3, 0.0),
                    color: vec4(0.3196, 0.00723, 0.00535, 1.0),
                    shadow_size: 0.3,
                    margin_vh: 0.05,
                    sdf_rad: 12.0,
                    num_chars: 1,
                    chars: [13, 0, 0, 0],
                });
            }
            _ => {}
        }

        if numbers_data.len() > 0 {
            gpu.queue.write_buffer(&self.numbers_buf, 0, bytemuck::cast_slice(&numbers_data));
            pass.set_pipeline(&self.text_pipeline);
            pass.set_bind_group(1, &self.numbers_bg, &[]);
            pass.draw(0..4, 0..(numbers_data.len() as u32));
        }

    }
}
