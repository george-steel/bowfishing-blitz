use std::borrow::Borrow;
use web_time::Instant;
use kira::{manager::{backend::DefaultBackend, AudioManager, AudioManagerSettings}};
use wgpu::{Surface, Texture, wgt::TextureViewDescriptor};
use glam::{UVec2, vec3};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::{arrows::ArrowController, boat_rail::RailController, camera::ShadowSettings, deferred_renderer::DeferredRenderer, gputil::AssetSource, targets::TargetController, terrain_view::{HeightmapTerrain, TerrainView}, ui::{GameState, UIDisplay}};

pub mod gputil;
pub mod terrain_view;
pub mod camera;
pub mod deferred_renderer;
pub mod shaders;
pub mod arrows;
pub mod targets;
pub mod boat_rail;
pub mod audio_util;
pub mod ui;

pub use gputil::GPUContext;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct GameSystem {
    gpu: GPUContext,
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(skip))]
    pub surface: wgpu::Surface<'static>,
    audio: Option<kira::manager::AudioManager>,
    game_state: GameState,
    camera: RailController,
    renderer: Box<DeferredRenderer>,
    terrain: HeightmapTerrain,
    terrain_view: TerrainView,
    arrows: ArrowController,
    targets: TargetController,
    ui_disp: UIDisplay,
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct FrameResult {
    pub need_resize: bool,
    pub should_release_cursor: bool,
}

impl GameSystem {
    pub fn new(gpu: GPUContext, surface: wgpu::Surface<'static>, size: UVec2, assets: &impl AssetSource) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let audio = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).ok();
        #[cfg(target_arch = "wasm32")]
        let audio = None;

        let init_time = Instant::now();
        let game_state = GameState::Title {started_at: init_time, is_restart: false };

        let shadow_settings = ShadowSettings {
            sun_dir: vec3(0.548, -0.380, 0.745),
            range_xy: 60.0,
            range_z: 10.0,
        };

        let camera = RailController::new(shadow_settings, init_time);
        let renderer = DeferredRenderer::new(&gpu, assets, &camera, size);

        let terrain = terrain_view::HeightmapTerrain::load(assets);
        let terrain_view = crate::terrain_view::TerrainView::new(&gpu, assets, &renderer, &terrain);

        let arrows = ArrowController::new(&gpu, assets, &renderer);
        let targets = TargetController::new(&gpu, assets, &renderer, &terrain);

        let ui_disp = UIDisplay::new(&gpu, assets, &renderer);

        GameSystem {
            gpu, surface, audio,
            game_state, camera, renderer,
            terrain, terrain_view, arrows, targets, ui_disp
        }
    }

    pub fn tick_and_render(&mut self, output: &wgpu::Texture) -> bool {
        let mut should_release_cursor = false;

        let now = Instant::now();
        if self.game_state.should_reset_world(now) {
            self.targets.reset(&self.terrain);
            self.arrows.reset();
            self.camera.reset(now, -GameState::COUNTDOWN_DURATION.as_secs_f64());
        }
        self.game_state.do_timeout(now);

        // movement and hits continue after the finish to allow for buzzer beater shots
        if !self.game_state.is_paused() {
            let time = self.camera.tick(now);
            if self.game_state.is_playing() && time >= GameState::GAME_PERIOD {
                should_release_cursor = true;
                self.game_state = GameState::Finish { done_at: now + GameState::FINISH_DURATION };
            }

            self.arrows.tick(time, &self.terrain, self.audio.as_mut(), &mut [
                &mut self.targets,
            ]);
            self.targets.tick(time);
        }
        self.ui_disp.tick(self.audio.as_mut(), self.game_state, now, &self.camera, &self.arrows, &self.targets);

        let out_view = output.create_view(&TextureViewDescriptor{
            format: Some(self.gpu.output_format),
            ..Default::default()
        });
        self.renderer.render(&self.gpu, &out_view, &self.camera, &mut [
            &mut self.terrain_view,
            &mut self.arrows,
            &mut self.targets,
            &mut self.ui_disp,
        ]);

        should_release_cursor
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl GameSystem {
    pub async fn init_from_canvas(canvas: web_sys::HtmlCanvasElement, raw_asset_bundle: Box<[u8]>) -> Self {
        use web_sys::console::{log_1, log_2};
        use crate::gputil::asset::LoadedZipBundle;

        let init_size = UVec2::new(canvas.width(), canvas.height());
        log_1(&"got canvas size".into());

        let wgpu_inst = wgpu::Instance::default();
        let surface = wgpu_inst.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).unwrap();
        let gpu = GPUContext::with_limits(
            wgpu_inst,
            Some(&surface),
            wgpu::Features::RG11B10UFLOAT_RENDERABLE | wgpu::Features::CLIP_DISTANCES,
            Default::default(),
        ).await;
        log_1(&"initialized gpu".into());

        gpu.configure_surface_target(&surface, init_size);
        log_1(&"configured canvas".into());

        let raw_asset_ref: &[u8] = raw_asset_bundle.borrow();
        log_2(&"asset bundle size (wasm)".into(), &raw_asset_ref.len().into());
        let assets = LoadedZipBundle::new(&raw_asset_ref).unwrap();
        log_1(&"parsed asset bundle".into());

        Self::new(gpu, surface, init_size, &assets)
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl GameSystem {
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn resize(&mut self, width: u32, height: u32) {
        let new_size = UVec2::new(width, height);
        self.renderer.resize(&self.gpu, new_size);
        self.gpu.configure_surface_target(&self.surface, new_size);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    // returns if the cursor should grab
    pub fn on_click(&mut self) -> bool {
        #[cfg(target_arch = "wasm32")]
        if let None = self.audio {
            self.audio = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).ok();
            web_sys::console::log_1(&"initialized audio".into());
        }
        match self.game_state {
            GameState::Playing => {
                self.arrows.shoot(self.audio.as_mut(), &self.camera);
                false
            },
            GameState::Title {..} => {
                self.game_state = GameState::Fade { done_at: Instant::now() + GameState::FADE_DURATION };
                true
            }
            GameState::Paused => {
                self.camera.unpause(Instant::now());
                self.game_state = GameState::Playing;
                true
            }
            _ => {false}
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn on_cursor_ungrab(&mut self) {
        match self.game_state {
            GameState::Playing => {
                self.game_state = GameState::Paused;
            }
            GameState::Countdown {..} | GameState::Fade {..} => {
                self.game_state = GameState::Title {started_at: Instant::now(), is_restart: false};
            }
            _ => {}
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn on_mouse_move(&mut self, dx: f64, dy: f64) {
        match self.game_state {
            GameState::Playing | GameState::Countdown {..} => {
                self.camera.mouse(dx, dy);
            }
            _ => {}
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn stop_music(&mut self) {
        self.ui_disp.stop_music();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn on_frame(&mut self,) -> FrameResult {
        let surface_result = self.surface.get_current_texture();
        let surface_tex = match surface_result {
            Ok(t) => t,
            Err(e) => {
                log::error!("get_current_texture: {}", e);
                return FrameResult {need_resize: true, should_release_cursor: false}
            }
        };

        let should_release_cursor = self.tick_and_render(&surface_tex.texture);
        surface_tex.present();

        FrameResult {need_resize: false, should_release_cursor}
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn init_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"Hello from wasm".into());
    Ok(())
}
