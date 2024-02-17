mod gputil;
mod terrain_view;
mod camera;
use gputil::*;
use camera::*;

use std::time::Instant;

use glam::f32::*;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent, MouseButton}, event_loop::EventLoop, keyboard::{Key, NamedKey}, window::CursorGrabMode
};


fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Info).init();
    log::info!("starting up");

    let gpu = wgpu::Instance::default();
    let event_loop = EventLoop::new().unwrap();
    let window = winit::window::WindowBuilder::new().build(&event_loop).unwrap();
    let surface = gpu.create_surface(&window).unwrap();

    let gctx = pollster::block_on(GPUContext::with_default_limits(gpu, Some(&surface)));

    gctx.configure_surface_target(&surface, window.inner_size());

    let init_time = Instant::now();
    //let ft = fragtex::FragDisplay::new(&gctx);
    let mut camera = CameraController::new(CameraSettings::default(), vec3(0.0, -5.0, 3.0), 90.0, init_time);
    let mut terrain = terrain_view::TerrainView::new(&gctx, &camera);

    let mut grabbed = false;
    let window = &window;
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = &gctx;

            if let Event::DeviceEvent {device_id: _, event} = event.clone() {
                match event {
                    DeviceEvent::Key(key_event) => {
                        if window.has_focus() {
                            camera.key(key_event.physical_key, key_event.state);
                        }
                    }
                    DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                        if window.has_focus() && grabbed {
                            camera.mouse(dx, dy);
                        }
                    }
                    _ => {}
                }
            }
            if let Event::WindowEvent {window_id: _, event,} = event {
                match event {
                    WindowEvent::Resized(new_size) => {
                        gctx.configure_surface_target(&surface, new_size);
                        window.request_redraw();
                    }
                    WindowEvent::Focused(false) |
                    WindowEvent::KeyboardInput { device_id: _, event: KeyEvent {
                        physical_key: _, logical_key: Key::Named(NamedKey::Escape), text: _, location: _, state: _, repeat: _, ..
                    }, is_synthetic: _ }=> {
                        window.set_cursor_grab(CursorGrabMode::None);
                        window.set_cursor_visible(true);
                        grabbed = false;
                    }
                    WindowEvent::MouseInput {device_id: _, state: ElementState::Pressed, button: MouseButton::Left } => {
                        if window.has_focus() {
                            window.set_cursor_grab(CursorGrabMode::Confined);
                            window.set_cursor_visible(false);
                            grabbed = true;
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        camera.tick(now);
                        match surface.get_current_texture() {
                            Err(e) => {
                                log::error!("get_current_tecture: {}", e);
                                gctx.configure_surface_target(&surface, window.inner_size());
                                window.request_redraw();
                            }
                            Ok(frame) => {
                                let mut encoder = gctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                    label: None,
                                });
                            
                                //ft.render(&gctx, &mut encoder, &frame.texture);
                                terrain.render(&gctx, &mut encoder, &frame.texture, &camera);

                                gctx.queue.submit(Some(encoder.finish()));
                                frame.present();
                                window.request_redraw();
                            }
                        }
                    }
                    WindowEvent::CloseRequested => target.exit(),
                    _ => {}
                };
            }
        })
        .unwrap();
}