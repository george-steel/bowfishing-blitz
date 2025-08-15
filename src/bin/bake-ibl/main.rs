use bowfishing_blitz::*;
use gputil::*;
use camera::*;
use wgpu::{wgt::TextureViewDescriptor, Features, TextureViewDimension};

mod spherical_filter;
mod water_filter;
mod integral_filter;

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
        Features::SUBGROUP | Features::MAPPABLE_PRIMARY_BUFFERS | Features::TEXTURE_FORMAT_16BIT_NORM | Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
        wgpu::Limits {
            max_compute_workgroup_size_x: 512,
            max_compute_invocations_per_workgroup: 512,
            max_compute_workgroup_storage_size: 65536,
            max_buffer_size: 512 * 1024 * 1024,
            max_storage_buffer_binding_size: 512 * 1024 * 1024,
            max_color_attachment_bytes_per_sample: 40,
            ..Default::default()
        },
    ));
    let limits = gpu.adapter.limits();
    //log::info!("Limits: {:#?}", &limits);
    
    let mut size = window_size(&window);
    gpu.configure_surface_target(&surface, size);

    let window = &window;
    let baker = spherical_filter::IBLFilter::new(&gpu);
    let water_baker = water_filter::WaterFilter::new(&gpu);
    let dfg_baker = integral_filter::DFGBaker::new(&gpu);

    //let test_cube = baker.make_test_cube(&gpu, 32);

    let dfg_tables = dfg_baker.integrate_dfg_lut(&gpu);

    let raw_sky_tex = gpu.load_rgbe8_texture("./assets/staging/kloofendal_48d_partly_cloudy_2k.rgbe.png").expect("Failed to load sky");
    let raw_sky_view = raw_sky_tex.create_view(&Default::default());

    let clamped_tex = water_baker.render_clamp(&gpu, &raw_sky_view);
    let clamped_view = clamped_tex.create_view(&Default::default());
    let (clamped_cube, _) = baker.bake_maps(&gpu, &clamped_view, None);
    let clamped_cube_view = clamped_cube.create_view(&TextureViewDescriptor{dimension: Some(TextureViewDimension::Cube), ..Default::default()});

    let above_tex = water_baker.render_above(&gpu, &dfg_tables, &raw_sky_view, &clamped_cube_view);
    let above_view = above_tex.create_view(&Default::default());
    let (above_cube, _) = baker.bake_maps(&gpu, &above_view, Some("above"));

    let below_tex = water_baker.render_below(&gpu, &dfg_tables, &raw_sky_view, &clamped_cube_view);
    let below_view = below_tex.create_view(&Default::default());
    let (below_cube, _) = baker.bake_maps(&gpu, &below_view, Some("below"));

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
                                baker.render(&gpu, &above_cube, &frame.texture);
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