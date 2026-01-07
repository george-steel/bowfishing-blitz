use std::time::Instant;
use kira::{manager::{backend::DefaultBackend, AudioManager, AudioManagerSettings}};
use wgpu::{Surface, Texture};
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

pub struct GameSystem {
    audio: kira::manager::AudioManager,
    game_state: GameState,
    camera: RailController,
    renderer: Box<DeferredRenderer>,
    terrain: HeightmapTerrain,
    terrain_view: TerrainView,
    arrows: ArrowController,
    targets: TargetController,
    ui_disp: UIDisplay,
}

pub struct FrameResult {
    pub should_release_cursor: bool,
}

impl GameSystem {
    pub fn new(gpu: &GPUContext, size: UVec2, assets: &impl AssetSource) -> Self {
        let audio = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).unwrap();

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
            audio,
            game_state, camera, renderer,
            terrain, terrain_view, arrows, targets, ui_disp
        }
    }

    pub fn resize(&mut self, gpu: &GPUContext, surface: &Surface, new_size: UVec2) {
        self.renderer.resize(&gpu, new_size);
        gpu.configure_surface_target(&surface, new_size);
    }

    // returns if the cursor should grab
    pub fn on_click(&mut self) -> bool {
        match self.game_state {
            GameState::Playing => {
                self.arrows.shoot(&mut self.audio, &self.camera);
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

    pub fn on_mouse_move(&mut self, dx: f64, dy: f64) {
        match self.game_state {
            GameState::Playing | GameState::Countdown {..} => {
                self.camera.mouse(dx, dy);
            }
            _ => {}
        }
    }

    pub fn stop_music(&mut self) {
        self.ui_disp.stop_music(&mut self.audio);
    }

    pub fn on_frame(&mut self, gpu: &GPUContext, output: &Texture) -> FrameResult {
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

            self.arrows.tick(time, &self.terrain, &mut self.audio, &mut [
                &mut self.targets,
            ]);
            self.targets.tick(time);
        }
        self.ui_disp.tick(&mut self.audio, self.game_state, now, &self.camera, &self.arrows, &self.targets);

        let out_view = output.create_view(&Default::default());
        self.renderer.render(&gpu, &out_view, &self.camera, &mut [
            &mut self.terrain_view,
            &mut self.arrows,
            &mut self.targets,
            &mut self.ui_disp,
        ]);

        FrameResult {should_release_cursor}
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    Ok(())
}
