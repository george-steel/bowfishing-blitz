// disable windows console on release build
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::time::{Duration};

use bowfishing_blitz::{GameSystem, gputil::{asset::LocalAssetFolder, *}};

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
    let window = event_loop.create_window(
            Window::default_attributes().with_title("Bowfishing Blitz").with_maximized(true)
        ).unwrap();
    let surface = wgpu_inst.create_surface(&window).unwrap();
    
    let gpu = pollster::block_on(GPUContext::with_limits(
        wgpu_inst,
        Some(&surface),
        wgpu::Features::RG11B10UFLOAT_RENDERABLE | wgpu::Features::CLIP_DISTANCES,
        Default::default(),
    ));
    
    let mut size = window_size(&window);
    gpu.configure_surface_target(&surface, size);

    let assets = LocalAssetFolder::new("./assets");

    let mut game = GameSystem::new(&gpu, size, &assets);

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
                            game.on_mouse_move(dx, dy);
                        }
                    }
                    _ => {}
                }
                Event::WindowEvent {window_id: _, event,} => match event {
                    WindowEvent::Resized(new_size) => {
                        must_resize = Some(uvec2(new_size.width, new_size.height));
                        window.request_redraw();
                    }
                    WindowEvent::Focused(false) |
                    WindowEvent::KeyboardInput { device_id: _, event: KeyEvent {
                        physical_key: _, logical_key: Key::Named(NamedKey::Escape), text: _, location: _, state: _, repeat: _, ..
                    }, is_synthetic: _ }=> {
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                        game.on_cursor_ungrab();
                    }
                    WindowEvent::KeyboardInput { device_id: _, event: KeyEvent {
                        physical_key: _, logical_key: Key::Character(c), text: _, location: _, state: _, repeat: _, ..
                    }, is_synthetic: _ }=> {
                        if c == "m" {
                            game.stop_music();
                        }
                    }
                    WindowEvent::MouseInput {device_id: _, state: ElementState::Pressed, button: MouseButton::Left} => {
                        if window.has_focus() {
                            let should_grab = game.on_click();
                            if should_grab {
                                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                window.set_cursor_visible(false);
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
            game.resize(&gpu, &surface, new_size);
            window.request_redraw();
            continue
        }
        
        let surface_tex = match surface_result {
            Ok(t) => t,
            Err(e) => {
                log::error!("get_current_texture: {}", e);
                size = window_size(&window);
                game.resize(&gpu, &surface, size);
                window.request_redraw();
                continue
            }
        };

        let frame_result = game.on_frame(&gpu, &surface_tex.texture);
        if frame_result.should_release_cursor {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
        surface_tex.present();
    }
}