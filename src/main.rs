mod gputil;
mod fragtex;
mod terrain_view;

use std::time::Instant;

use gputil::*;

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
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
    let mut terrain = terrain_view::TerrainView::new(&gctx, init_time);

    let window = &window;
    event_loop
        .run(move |event, target| {
            // Have the closure take ownership of the resources.
            // `event_loop.run` never returns, therefore we must do this to ensure
            // the resources are properly cleaned up.
            let _ = &gctx;

            if let Event::WindowEvent {
                window_id: _,
                event,
            } = event
            {
                match event {
                    WindowEvent::Resized(new_size) => {
                        gctx.configure_surface_target(&surface, new_size);
                        window.request_redraw();
                    }
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
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
                                terrain.render(&gctx, &mut encoder, &frame.texture, now);

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