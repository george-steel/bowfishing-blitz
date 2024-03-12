// disable windows console on release build
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use bowfishing_blitz::{arrows::ArrowController, boat_rail::RailController, camera::*, deferred_renderer::*, gputil::*, targets::TargetController, ui::{GameState, UIDisplay}, *};
use env_logger::init;
use kira::{manager::{backend::DefaultBackend, AudioManager, AudioManagerSettings}, sound::streaming::{StreamingSoundData, StreamingSoundSettings}, Volume};

use std::time::{Duration, Instant};

use glam::*;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    platform::pump_events::{EventLoopExtPumpEvents, PumpStatus},
    window::{CursorGrabMode, Window}
};

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Info).init();
    log::info!("starting up");

    let wgpu_inst = wgpu::Instance::default();
    let mut event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new()
            .with_title("Bowfishing Blitz")
            .with_maximized(true)
            .build(&event_loop).unwrap();
    let surface = wgpu_inst.create_surface(&window).unwrap();
    
    let gpu = pollster::block_on(GPUContext::with_default_limits(
        wgpu_inst,
        Some(&surface),
        wgpu::Features::RG11B10UFLOAT_RENDERABLE,
    ));
    
    let mut size = window_size(&window);
    gpu.configure_surface_target(&surface, size);

    let mut audio = AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).unwrap();

    let init_time = Instant::now();
    let mut game_state = GameState::Title {started_at: init_time, is_restart: false };


    let mut camera = RailController::new(init_time);
    let mut renderer = DeferredRenderer::new(&gpu, &camera, size);

    let terrain = terrain_view::HeightmapTerrain::load();
    let mut terrain_view = crate::terrain_view::TerrainView::new(&gpu, &renderer, &terrain);

    let mut arrows = ArrowController::new(&gpu, &renderer);
    let mut targets = TargetController::new(&gpu, &renderer, &terrain);

    let mut ui_disp = UIDisplay::new(&gpu, &renderer);

    let window = &window;
    'mainloop: loop{
        let surface_result = surface.get_current_texture();
        //log::info!("FRAME ------------------------------------------------------");

        let mut must_resize: Option<UVec2> = None;
        let loop_status = event_loop.pump_events(Some(Duration::ZERO),  |event, target| {
            match event {
                Event::DeviceEvent {device_id: _, event: dev_event} => match dev_event {
                    DeviceEvent::Key(key_event) => {
                        if window.has_focus() {
                            //camera.key(key_event.physical_key, key_event.state);
                        }
                    }
                    DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                        if window.has_focus() {
                            match game_state {
                                GameState::Playing | GameState::Countdown {..} => {
                                    camera.mouse(dx, dy);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }
                Event::WindowEvent {window_id: _, event,} => match event {
                    WindowEvent::Resized(new_size) => {
                        must_resize = Some(uvec2(new_size.width, new_size.height));
                        //renderer.resize(&gpu, size);
                        //gpu.configure_surface_target(&surface, size);
                        window.request_redraw();
                    }
                    WindowEvent::Focused(false) |
                    WindowEvent::KeyboardInput { device_id: _, event: KeyEvent {
                        physical_key: _, logical_key: Key::Named(NamedKey::Escape), text: _, location: _, state: _, repeat: _, ..
                    }, is_synthetic: _ }=> {
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                        match game_state {
                            GameState::Playing => {
                                game_state = GameState::Paused;
                            }
                            GameState::Countdown {..} | GameState::Fade {..} => {
                                game_state = GameState::Title {started_at: Instant::now(), is_restart: false};
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::MouseInput {device_id: _, state: ElementState::Pressed, button: MouseButton::Left} => {
                        if window.has_focus() {
                            match game_state {
                                GameState::Playing => {
                                    arrows.shoot(&mut audio, &camera);
                                },
                                GameState::Title {..} => {
                                    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                    window.set_cursor_visible(false);
                                    game_state = GameState::Fade { done_at: Instant::now() + GameState::FADE_DURATION }
                                }
                                GameState::Paused => {
                                    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                    window.set_cursor_visible(false);
                                    camera.unpause(Instant::now());
                                    game_state = GameState::Playing
                                }
                                _ => {}
                            }
                        }
                    }
                    WindowEvent::CloseRequested => {
                        log::info!("CLOSE REQUESTED");
                        target.exit();
                    }
                    _ => {}
                }
                _ => {}
            }
        });
        match loop_status {
            PumpStatus::Continue => {}
            PumpStatus::Exit(_) => {
                log::info!("EXITING");
                break 'mainloop
            }
        }
        
        if let Some(new_size) = must_resize {
            drop(surface_result);
            renderer.resize(&gpu, new_size);
            gpu.configure_surface_target(&surface, new_size);
            window.request_redraw();
            continue
        }
        
        let surface_tex = match surface_result {
            Ok(t) => t,
            Err(e) => {
                log::error!("get_current_texture: {}", e);
                size = window_size(&window);
                renderer.resize(&gpu, size);
                gpu.configure_surface_target(&surface, size);
                window.request_redraw();
                continue
            }
        };
        let now = Instant::now();
        if game_state.should_reset_world(now) {
            targets.reset(&terrain);
            arrows.reset();
            camera.reset(now, -GameState::COUNTDOWN_DURATION.as_secs_f64());
        }
        game_state.do_timeout(now);

        // movement and hits continue after the finish to allow for buzzer beater shots
        if !game_state.is_paused() {
            let time = camera.tick(now);
            if game_state.is_playing() && time >= GameState::GAME_PERIOD {
                let _ = window.set_cursor_grab(CursorGrabMode::None);
                window.set_cursor_visible(true);
                game_state = GameState::Finish { done_at: now + GameState::FINISH_DURATION };
            }

            arrows.tick(time, &terrain, &mut audio, &mut [
                &mut targets,
            ]);
            targets.tick(time);
        }
        ui_disp.tick(&mut audio, game_state, now, &camera, &arrows, &targets);

        let out_view = surface_tex.texture.create_view(&Default::default());
        renderer.render(&gpu, &out_view, &camera, &mut [
            &mut terrain_view,
            &mut arrows,
            &mut targets,
            &mut ui_disp,
        ]);
        surface_tex.present();
        //window.request_redraw();
    }
}