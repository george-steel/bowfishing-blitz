use winit::dpi::PhysicalSize;

pub struct GPUContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub output_format: wgpu::TextureFormat,
}


impl GPUContext {
    pub async fn with_default_limits(instance: wgpu::Instance, for_surface: Option<&wgpu::Surface<'_>>) -> Self {
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions{
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: for_surface,
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }, None)
        .await
        .expect("Failed to create device");

        let output_format = match for_surface {
            None => wgpu::TextureFormat::Rgba8UnormSrgb,
            Some(surface) => {
                let surface_caps = surface.get_capabilities(&adapter);
                surface_caps.formats.iter().copied()
                    .filter(|f| f.is_srgb())
                    .next().unwrap_or(surface_caps.formats[0])
            },
        };
        log::info!("Using output format {:?}", output_format);

        GPUContext {
            instance, adapter, device, queue, output_format
        }
    }

    pub fn configure_surface_target(&self, surface: &wgpu::Surface, size: PhysicalSize<u32>) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.output_format,
            present_mode: wgpu::PresentMode::Fifo,
            ..surface.get_default_config(&self.adapter, size.width.max(1), size.height.max(1)).unwrap()
        };
        surface.configure(&self.device, &surface_config);
    }
}
