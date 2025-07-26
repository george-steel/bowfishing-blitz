use bowfishing_blitz::*;
use gputil::*;
use camera::*;
use wgpu::Features;

mod ibl_filter;

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
    let window = event_loop.create_window(
        Window::default_attributes()
        .with_inner_size(winit::dpi::LogicalSize::new(1024.0, 512.0)))
        .unwrap();
    let surface = wgpu_inst.create_surface(&window).unwrap();
    
    let gpu = pollster::block_on(GPUContext::with_limits(
        wgpu_inst,
        Some(&surface),
        Features::SUBGROUP | Features::MAPPABLE_PRIMARY_BUFFERS,
        wgpu::Limits {
            max_compute_workgroup_size_x: 512,
            max_compute_invocations_per_workgroup: 512,
            max_compute_workgroup_storage_size: 65536,
            max_buffer_size: 512 * 1024 * 1024,
            max_storage_buffer_binding_size: 512 * 1024 * 1024,
            ..Default::default()
        },
    ));
    let limits = gpu.adapter.limits();
    //log::info!("Limits: {:#?}", &limits);
    
    let mut size = window_size(&window);
    gpu.configure_surface_target(&surface, size);

    let init_time = Instant::now();
    let window = &window;
    let baker = ibl_filter::IBLFilter::new(&gpu);

    let in_tex = gpu.load_rgbe8_texture("./assets/staging/kloofendal_48d_partly_cloudy_2k.rgbe.png").expect("Failed to load sky");
    let in_view = in_tex.create_view(&Default::default());
    baker.bake_maps(&gpu, &in_view, "test");

    let sky_tex = gpu.load_rgbe8_texture("./assets/staging/kloofendal_48d_partly_cloudy_puresky_2k.rgbe.png").expect("Failed to load sky");
    let sky_view = sky_tex.create_view(&Default::default());
    baker.bake_cube(&gpu, &sky_view, "skybox", 512);

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
                                baker.render(&gpu, &frame.texture);
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