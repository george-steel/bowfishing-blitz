use bowfishing_blitz::{*, arrows::ArrowController, gputil::*, camera::*, deferred_renderer::*};

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
            .build(&event_loop).unwrap();
    let surface = wgpu_inst.create_surface(&window).unwrap();
    
    let gpu = pollster::block_on(GPUContext::with_default_limits(
        wgpu_inst,
        Some(&surface),
        wgpu::Features::RG11B10UFLOAT_RENDERABLE,
    ));
    
    let mut size = window_size(&window);
    gpu.configure_surface_target(&surface, size);

    let init_time = Instant::now();
    //let ft = fragtex::FragDisplay::new(&gctx);
    let mut camera = FreeCam::new(CameraSettings::default(), vec3(0.0, -5.0, 3.0), 90.0, init_time);
    let mut renderer = DeferredRenderer::new(&gpu, &camera, size);

    let terrain = terrain_view::HeightmapTerrain::load();
    let mut terrain_view = crate::terrain_view::TerrainView::new(&gpu, &renderer, &terrain);

    let mut arrows = ArrowController::new(&gpu, &renderer, init_time);

    let mut grabbed = false;
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
                            camera.key(key_event.physical_key, key_event.state);
                        }
                    }
                    DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                        if window.has_focus() && grabbed {
                            //log::info!("mouse {} {}", dx, dy);
                            camera.mouse(dx, dy);
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
                        grabbed = false;
                    }
                    WindowEvent::MouseInput {device_id: _, state: ElementState::Pressed, button: MouseButton::Left } => {
                        if window.has_focus() {
                            if grabbed {
                                log::info!("SHOOT");
                                arrows.shoot(&camera);
                            } else {
                                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                                window.set_cursor_visible(false);
                                grabbed = true;
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
        camera.tick(now);
        arrows.tick(now, &terrain, &mut []);

        let out_view = surface_tex.texture.create_view(&Default::default());
        renderer.render(&gpu, &out_view, &camera, &mut [
            &mut terrain_view,
            &mut arrows,
        ]);
        surface_tex.present();
        //window.request_redraw();
    }
}