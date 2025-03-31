use bowfishing_blitz::*;
use gputil::*;
use camera::*;

mod refract_test;

use std::time::Instant;

use glam::*;
use winit::{
    event::{DeviceEvent, ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::{Key, NamedKey},
    window::{CursorGrabMode, Window}
};

fn main() {
    env_logger::builder().filter_level(log::LevelFilter::Info).init();
    log::info!("starting up");

    let wgpu_inst = wgpu::Instance::default();
    let event_loop = EventLoop::new().unwrap();
    let window = event_loop.create_window(Window::default_attributes()).unwrap();
    let surface = wgpu_inst.create_surface(&window).unwrap();
    
    let gpu = pollster::block_on(GPUContext::with_default_limits(
        wgpu_inst,
        Some(&surface),
        wgpu::Features::MAPPABLE_PRIMARY_BUFFERS
    ));
    let limits = gpu.adapter.limits();
    //log::info!("Limits: {:#?}", &limits);
    
    let mut size = window_size(&window);
    gpu.configure_surface_target(&surface, size);

    let init_time = Instant::now();
    let window = &window;
    let map_disp = refract_test::FragDisplay::new(&gpu);
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = &gpu;

            if let Event::WindowEvent {window_id: _, event,} = event {
                match event {
                    WindowEvent::Resized(new_size) => {
                        size = uvec2(new_size.width, new_size.height);
                        gpu.configure_surface_target(&surface, size);
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        match surface.get_current_texture() {
                            Err(e) => {
                                log::error!("get_current_texture: {}", e);
                                size = window_size(&window);
                                gpu.configure_surface_target(&surface, size);
                                window.request_redraw();
                            }
                            Ok(frame) => {
                                map_disp.render(&gpu, &frame.texture);
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