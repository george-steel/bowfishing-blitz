use crate::gputil::*;
use crate::camera::*;
use glam::*;
use wgpu::*;
use wgpu::util::*;
use std::collections::HashMap;
use std::f32;
use std::{borrow::Cow, path::Path};

pub trait RenderObject {
    // To update biffers or run compute shaders
    fn prepass(&mut self, gpu: &GPUContext, renderer: &DeferredRenderer, encoder: &mut CommandEncoder) {}

    fn draw_shadow_casters<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {}

    // Draw underwater geometry to its refracted positions.
    fn draw_underwater<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {}

    // Draw geometry to its reflected positions.
    fn draw_reflected<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {}

    // draw opaque geometry
    fn draw_opaque<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>);

    // draw transparant geometry after lighting calculations
    fn draw_transparent<'a>(&'a self, gpu: &GPUContext, renderer: &DeferredRenderer, pass: &mut RenderPass<'a>) {}
}

#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlobalLighting {
    pub sun_color: Vec3,
    pub sky_fac: f32,
    pub sun_dir: Vec3, // towards sun
    pub pad0: f32,
    pub refr_sun_dir: Vec3,
    pub refr_sun_trans: f32,
    pub water_lim_color: Vec3,
    pub half_secci: f32,
}

impl GlobalLighting {
    pub fn new(sun_color: Vec3, sky_fac: f32, sun_dir: Vec3) -> Self {
        let norm_sun = sun_dir.normalize();
        let sin_above = norm_sun.xy().length();
        let cos_below = (1.0 - sin_above * sin_above / 1.7689).sqrt();
        let refr_sun_dir = vec3(norm_sun.x/1.33, norm_sun.y/1.33, cos_below);
        let refr_sun_trans = (norm_sun.z / cos_below) * 0.98 * (1.0 - (1.0 - norm_sun.z).powi(5));
        let water_lim_color = vec3(0.03, 0.05, 0.1);
        let half_secci = 15.0;
        GlobalLighting {
            sun_color, sky_fac,
            sun_dir: norm_sun,
            refr_sun_dir, refr_sun_trans,
            water_lim_color, half_secci,
            pad0: 0.0,
        }
    }
}

struct DeferredRendererTextures {
    size: UVec2,
    water_size: UVec2,
    // main gbuffers
    dist: Texture,
    dist_view: TextureView,
    albedo: Texture,
    albedo_view: TextureView,
    normal: Texture,
    normal_view: TextureView,
    rough_metal: Texture,
    rm_view: TextureView,
    ao: Texture,
    ao_view: TextureView,
    material: Texture,
    material_view: TextureView,

    // reflected
    water_refl: Texture,
    water_refl_view: TextureView,
    water_refl_dist: Texture,
    water_refl_dist_view: TextureView,
    water_refl_albedo: Texture,
    water_refl_albedo_view: TextureView,
    water_refl_normal: Texture,
    water_refl_normal_view: TextureView,
    water_refl_rough_metal: Texture,
    water_refl_rm_view: TextureView,
    water_refl_ao: Texture,
    water_refl_ao_view: TextureView,
    water_refl_material: Texture,
    water_refl_material_view: TextureView,

    // refracted
    water_trans: Texture,
    water_trans_view: TextureView,
    water_trans_dist: Texture,
    water_trans_dist_view: TextureView,
    water_trans_albedo: Texture,
    water_trans_albedo_view: TextureView,
    water_trans_normal: Texture,
    water_trans_normal_view: TextureView,
    water_trans_rough_metal: Texture,
    water_trans_rm_view: TextureView,
    water_trans_ao: Texture,
    water_trans_ao_view: TextureView,
    water_trans_material: Texture,
    water_trans_material_view: TextureView,

    // shadow
    shadow_dist: Texture,
    shadow_dist_view: TextureView,
}

impl DeferredRendererTextures {
    fn destroy(&self) {
        self.dist.destroy();
        self.albedo.destroy();
        self.normal.destroy();
        self.rough_metal.destroy();
        self.ao.destroy();
        self.material.destroy();
        self.water_refl.destroy();
        self.water_refl_dist.destroy();
        self.water_refl_albedo.destroy();
        self.water_refl_normal.destroy();
        self.water_refl_rough_metal.destroy();
        self.water_refl_ao.destroy();
        self.water_refl_material.destroy();
        self.water_trans.destroy();
        self.water_trans_dist.destroy();
        self.water_trans_albedo.destroy();
        self.water_trans_normal.destroy();
        self.water_trans_rough_metal.destroy();
        self.water_trans_ao.destroy();
        self.water_trans_material.destroy();
        self.shadow_dist.destroy();
    }

    fn create(gpu: &GPUContext, output_size: UVec2) -> Self {
        let size = output_size.max(uvec2(1,1));
        let water_y = 720.clamp(output_size.y / 2, output_size.y);
        let water_x = (size.x * water_y / size.y).max(1);
        let water_size = uvec2(water_x, water_y);
        let size_3d = extent_2d(size);
        let water_size_3d = extent_2d(water_size);

        let (dist, dist_view) = gpu.create_empty_texture(size_3d, TextureFormat::Depth32Float, "dist");
        let (albedo, albedo_view) = gpu.create_empty_texture(size_3d, TextureFormat::Rg11b10Ufloat, "albedo");
        let (normal, normal_view) = gpu.create_empty_texture(size_3d, TextureFormat::Rgb10a2Unorm, "normal");
        let (rough_metal, rm_view) = gpu.create_empty_texture(size_3d, TextureFormat::Rg8Unorm, "rough-metal");
        let (ao, ao_view) = gpu.create_empty_texture(size_3d, TextureFormat::R8Unorm, "ao");
        let (material, material_view) = gpu.create_empty_texture(size_3d, TextureFormat::R8Uint, "material");
        let (water_refl, water_refl_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Rg11b10Ufloat, "water-refl-out");
        let (water_refl_dist, water_refl_dist_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Depth32Float, "water-refl-dist");
        let (water_refl_albedo, water_refl_albedo_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Rg11b10Ufloat, "water-refl-albedo");
        let (water_refl_normal, water_refl_normal_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Rgb10a2Unorm, "water-refl-normal");
        let (water_refl_rough_metal, water_refl_rm_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Rg8Unorm, "water-refl-rough-metal");
        let (water_refl_ao, water_refl_ao_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::R8Unorm, "water-refl-ao");
        let (water_refl_material, water_refl_material_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::R8Uint, "water-refl-material");
        let (water_trans, water_trans_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Rg11b10Ufloat, "water-trans-out");
        let (water_trans_dist, water_trans_dist_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Depth32Float, "water-trans-dist");
        let (water_trans_albedo, water_trans_albedo_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Rg11b10Ufloat, "water-trans-albedo");
        let (water_trans_normal, water_trans_normal_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Rgb10a2Unorm, "water-trans-normal");
        let (water_trans_rough_metal, water_trans_rm_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::Rg8Unorm, "water-trans-rough-metal");
        let (water_trans_ao, water_trans_ao_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::R8Unorm, "water-trans-ao");
        let (water_trans_material, water_trans_material_view) = gpu.create_empty_texture(water_size_3d, TextureFormat::R8Uint, "water-trans-material");
        let (shadow_dist, shadow_dist_view) = gpu.create_empty_texture(extent_2d(uvec2(2000, 2000)), TextureFormat::Depth32Float, "shadow_dist");

        Self {
            size, water_size,
            dist, dist_view,
            albedo, albedo_view,
            normal, normal_view,
            rough_metal, rm_view,
            ao, ao_view,
            material, material_view,
            water_refl, water_refl_view,
            water_refl_dist, water_refl_dist_view,
            water_refl_albedo, water_refl_albedo_view,
            water_refl_normal, water_refl_normal_view,
            water_refl_rough_metal, water_refl_rm_view,
            water_refl_ao, water_refl_ao_view,
            water_refl_material, water_refl_material_view,
            water_trans, water_trans_view,
            water_trans_dist, water_trans_dist_view,
            water_trans_albedo, water_trans_albedo_view,
            water_trans_normal, water_trans_normal_view,
            water_trans_rough_metal, water_trans_rm_view,
            water_trans_ao, water_trans_ao_view,
            water_trans_material, water_trans_material_view,
            shadow_dist, shadow_dist_view,
        }
    }
}

pub struct DeferredRenderer {
    lighting_shaders: ShaderModule,
    gbuffers: DeferredRendererTextures,
    pub global_bind_layout: BindGroupLayout,
    pub global_bind_group: BindGroup,
    pub main_camera_buf: Buffer,
    pub camera: Camera,
    pub global_lighting: GlobalLighting,

    gbuffer_bind_layout: BindGroupLayout,
    gbuffer_bind_group: BindGroup,
    water_sampler: Sampler,
    shadow_sampler: Sampler,
    water_gbuffer_bind_layout: BindGroupLayout,
    water_trans_gbuffer_bind_group: BindGroup,
    water_refl_gbuffer_bind_group: BindGroup,
    above_lighting_bind_group: BindGroup,
    below_lighting_bind_group: BindGroup,
    lighting_pipeline: RenderPipeline,
    underwater_lighting_pipeline: RenderPipeline,
    reflected_lighting_pipeline: RenderPipeline,
}

impl DeferredRenderer {
    pub fn new(gpu: &GPUContext, camera_ctrl: &impl CameraController, output_size: UVec2) -> Box<Self> {
        let lighting_shaders = gpu.device.create_shader_module(ShaderModuleDescriptor{
            label: Some("lighting.wgsl"),
            source: ShaderSource::Wgsl(Cow::Borrowed(crate::shaders::LIGHTING)),
        });

        let gbuffers = DeferredRendererTextures::create(gpu, output_size);

        let global_bind_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("global_bind_layout"),
            entries: &[
                BindGroupLayoutEntry{
                    binding: 0,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    count: None,
                }
            ]
        });

        let camera = camera_ctrl.camera(gbuffers.size.as_vec2(), gbuffers.water_size.as_vec2());
        let main_camera_buf = gpu.device.create_buffer_init(&BufferInitDescriptor{
            label: Some("camera_buf"),
            contents: bytemuck::bytes_of(&camera),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let global_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("global_bind_group"),
            layout: &global_bind_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: main_camera_buf.as_entire_binding()},
            ],
        });

        let gbuffer_bind_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("gbuffer_bind_layout"),
            entries: &[
                BindGroupLayoutEntry{
                    binding: 0, // dist
                    ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 1, // albedo
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 2, // normal
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 3, // rough_metal
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 4, // ao
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 5, // material
                    ty: BindingType::Texture { sample_type: TextureSampleType::Uint, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 6, // shadow map
                    ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 7, // bilinear for water transmission
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 8, // underwater color
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 9, // underwater distance
                    ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 10, // reflected color
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 11, // reflected distance
                    ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 12, // shadow map
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
            ]
        });
        let shadow_sampler = gpu.device.create_sampler(&SamplerDescriptor {
            label: Some("shadow_sampler"),
            compare: Some(CompareFunction::Greater),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let water_sampler = gpu.device.create_sampler(&SamplerDescriptor {
            label: Some("water_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let gbuffer_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("gbuffer_bind_group"),
            layout: &gbuffer_bind_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: wgpu::BindingResource::TextureView(&gbuffers.dist_view)},
                BindGroupEntry {binding: 1, resource: wgpu::BindingResource::TextureView(&gbuffers.albedo_view)},
                BindGroupEntry {binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffers.normal_view)},
                BindGroupEntry {binding: 3, resource: wgpu::BindingResource::TextureView(&gbuffers.rm_view)},
                BindGroupEntry {binding: 4, resource: wgpu::BindingResource::TextureView(&gbuffers.ao_view)},
                BindGroupEntry {binding: 5, resource: wgpu::BindingResource::TextureView(&gbuffers.material_view)},
                BindGroupEntry {binding: 6, resource: wgpu::BindingResource::TextureView(&gbuffers.shadow_dist_view)},
                BindGroupEntry {binding: 7, resource: wgpu::BindingResource::Sampler(&shadow_sampler)},
                BindGroupEntry {binding: 8, resource: wgpu::BindingResource::TextureView(&gbuffers.water_trans_view)},
                BindGroupEntry {binding: 9, resource: wgpu::BindingResource::TextureView(&gbuffers.water_trans_dist_view)},
                BindGroupEntry {binding: 10, resource: wgpu::BindingResource::TextureView(&gbuffers.water_refl_view)},
                BindGroupEntry {binding: 11, resource: wgpu::BindingResource::TextureView(&gbuffers.water_refl_dist_view)},
                BindGroupEntry {binding: 12, resource: wgpu::BindingResource::Sampler(&water_sampler)},
            ]
        });

        let water_gbuffer_bind_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_gbuffer_bind_layout"),
            entries: &[
                BindGroupLayoutEntry{
                    binding: 0, // dist
                    ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 1, // albedo
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 2, // normal
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 3, // rough_metal
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 4, // ao
                    ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 5, // material
                    ty: BindingType::Texture { sample_type: TextureSampleType::Uint, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 6, // shadow map
                    ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 7, // shadow map sampler
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
            ]
        });

        let water_trans_gbuffer_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("water_gbuffer_bind_group"),
            layout: &water_gbuffer_bind_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: wgpu::BindingResource::TextureView(&gbuffers.water_trans_dist_view)},
                BindGroupEntry {binding: 1, resource: wgpu::BindingResource::TextureView(&gbuffers.water_trans_albedo_view)},
                BindGroupEntry {binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffers.water_trans_normal_view)},
                BindGroupEntry {binding: 3, resource: wgpu::BindingResource::TextureView(&gbuffers.water_trans_rm_view)},
                BindGroupEntry {binding: 4, resource: wgpu::BindingResource::TextureView(&gbuffers.water_trans_ao_view)},
                BindGroupEntry {binding: 5, resource: wgpu::BindingResource::TextureView(&gbuffers.water_trans_material_view)},
                BindGroupEntry {binding: 6, resource: wgpu::BindingResource::TextureView(&gbuffers.shadow_dist_view)},
                BindGroupEntry {binding: 7, resource: wgpu::BindingResource::Sampler(&shadow_sampler)},
            ]
        });

        let water_refl_gbuffer_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("water_gbuffer_bind_group"),
            layout: &water_gbuffer_bind_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: wgpu::BindingResource::TextureView(&gbuffers.water_refl_dist_view)},
                BindGroupEntry {binding: 1, resource: wgpu::BindingResource::TextureView(&gbuffers.water_refl_albedo_view)},
                BindGroupEntry {binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffers.water_refl_normal_view)},
                BindGroupEntry {binding: 3, resource: wgpu::BindingResource::TextureView(&gbuffers.water_refl_rm_view)},
                BindGroupEntry {binding: 4, resource: wgpu::BindingResource::TextureView(&gbuffers.water_refl_ao_view)},
                BindGroupEntry {binding: 5, resource: wgpu::BindingResource::TextureView(&gbuffers.water_refl_material_view)},
                BindGroupEntry {binding: 6, resource: wgpu::BindingResource::TextureView(&gbuffers.shadow_dist_view)},
                BindGroupEntry {binding: 7, resource: wgpu::BindingResource::Sampler(&shadow_sampler)},
            ]
        });

        let lighting_bind_group_layout = gpu.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("lighting_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry{
                    binding: 0,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    visibility: ShaderStages::FRAGMENT,
                    count: None,
                },
                BindGroupLayoutEntry{
                    binding: 1, // lut_sampler
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 2, // dfg_lut
                    ty: BindingType::Texture { sample_type:TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 3, // cube_trilinear
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 4, // cube_bilinear
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 5, // radiance_map
                    ty: BindingType::Texture { sample_type:TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::Cube, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 6, // irradiance_map
                    ty: BindingType::Texture { sample_type:TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::Cube, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
                BindGroupLayoutEntry{
                    binding: 7, // sky
                    ty: BindingType::Texture { sample_type:TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::Cube, multisampled: false },
                    visibility: ShaderStages::FRAGMENT, count: None,
                },
            ],
        });

        let global_lighting = GlobalLighting::new(
            0.8 * vec3(1.0, 1.0, 0.8),
            0.6,
            vec3(0.548, -0.380, 0.745)
        );

        let global_lighting_buf = gpu.device.create_buffer_init(&BufferInitDescriptor{
            label: Some("global_ligting_buf"),
            contents: bytemuck::bytes_of(&global_lighting),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let sky_tex = gpu.load_rgbe8_cube_texture("./assets/staging/skybox.cube.rgbe8.png", 1).expect("Failed to load sky");
        let above_radiance_tex = gpu.load_rgbe8_cube_texture("./assets/staging/above-radiance.mipcube.rgbe8.png", 8).expect("Failed to load sky");
        let above_irradiance_tex = gpu.load_rgbe8_cube_texture("./assets/staging/above-irradiance.mipcube.rgbe8.png", 1).expect("Failed to load sky");
        let below_radiance_tex = gpu.load_rgbe8_cube_texture("./assets/staging/below-radiance.mipcube.rgbe8.png", 8).expect("Failed to load sky");
        let below_irradiance_tex = gpu.load_rgbe8_cube_texture("./assets/staging/below-irradiance.mipcube.rgbe8.png", 1).expect("Failed to load sky");
        let dfg_lut_tex = gpu.load_png_texture::<u32>("./assets/staging/dfg_integral_lut.rg16f.png", TextureFormat::Rg16Float).expect("Failed to load DFG");

        let lut_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..wgpu::SamplerDescriptor::default()
        });
        let cube_bilinear_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            ..wgpu::SamplerDescriptor::default()
        });
        let cube_trilinear_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::MirrorRepeat,
            address_mode_v: wgpu::AddressMode::MirrorRepeat,
            ..wgpu::SamplerDescriptor::default()
        });
        let sky_tex_view = cube_view(&sky_tex);
        let dfg_tex_view = dfg_lut_tex.create_view(&Default::default());

        let above_lighting_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("lighting_bind_group"),
            layout: &lighting_bind_group_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: global_lighting_buf.as_entire_binding()},
                BindGroupEntry {binding: 1, resource: wgpu::BindingResource::Sampler(&lut_sampler)},
                BindGroupEntry {binding: 2, resource: wgpu::BindingResource::TextureView(&dfg_tex_view)},
                BindGroupEntry {binding: 3, resource: wgpu::BindingResource::Sampler(&cube_trilinear_sampler)},
                BindGroupEntry {binding: 4, resource: wgpu::BindingResource::Sampler(&cube_bilinear_sampler)},
                BindGroupEntry {binding: 5, resource: wgpu::BindingResource::TextureView(&cube_view(&above_radiance_tex))},
                BindGroupEntry {binding: 6, resource: wgpu::BindingResource::TextureView(&cube_view(&above_irradiance_tex))},
                BindGroupEntry {binding: 7, resource: wgpu::BindingResource::TextureView(&sky_tex_view)},
            ]
        });

        let below_lighting_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: Some("lighting_bind_group"),
            layout: &lighting_bind_group_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: global_lighting_buf.as_entire_binding()},
                BindGroupEntry {binding: 1, resource: wgpu::BindingResource::Sampler(&lut_sampler)},
                BindGroupEntry {binding: 2, resource: wgpu::BindingResource::TextureView(&dfg_tex_view)},
                BindGroupEntry {binding: 3, resource: wgpu::BindingResource::Sampler(&cube_trilinear_sampler)},
                BindGroupEntry {binding: 4, resource: wgpu::BindingResource::Sampler(&cube_bilinear_sampler)},
                BindGroupEntry {binding: 6, resource: wgpu::BindingResource::TextureView(&cube_view(&below_irradiance_tex))},
                BindGroupEntry {binding: 5, resource: wgpu::BindingResource::TextureView(&cube_view(&below_radiance_tex))},
                BindGroupEntry {binding: 7, resource: wgpu::BindingResource::TextureView(&sky_tex_view)},
            ]
        });


        let lighting_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("lighting_pipeline_layout"),
            bind_group_layouts: &[
                &global_bind_layout,
                &gbuffer_bind_layout,
                &lighting_bind_group_layout,
            ],
            push_constant_ranges: &[]
        });

        let lighting_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("lighting_pipeline"),
            layout: Some(&lighting_pipeline_layout),
            vertex: VertexState {
                module: &lighting_shaders,
                entry_point: Some("fullscreen_tri"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &lighting_shaders,
                entry_point: Some("do_global_lighting"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState{ format: gpu.output_format, blend: None, write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None
        });

        let indirect_lighting_pipeline_layout = gpu.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("lighting_pipeline_layout"),
            bind_group_layouts: &[
                &global_bind_layout,
                &water_gbuffer_bind_layout,
                &lighting_bind_group_layout,
            ],
            push_constant_ranges: &[]
        });

        let underwater_lighting_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("underwater_lighting_pipeline"),
            layout: Some(&indirect_lighting_pipeline_layout),
            vertex: VertexState {
                module: &lighting_shaders,
                entry_point: Some("fullscreen_tri"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &lighting_shaders,
                entry_point: Some("do_underwater_lighting"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState{ format: TextureFormat::Rg11b10Ufloat, blend: None, write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let reflected_lighting_pipeline = gpu.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("reflected_lighting_pipeline"),
            layout: Some(&indirect_lighting_pipeline_layout),
            vertex: VertexState {
                module: &lighting_shaders,
                entry_point: Some("fullscreen_tri"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &lighting_shaders,
                entry_point: Some("do_reflected_lighting"),
                compilation_options: Default::default(),
                targets: &[Some(ColorTargetState{ format: TextureFormat::Rg11b10Ufloat, blend: None, write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });


        Box::new(DeferredRenderer {
            lighting_shaders,
            gbuffers,
            global_bind_layout, global_bind_group,
            main_camera_buf, camera,
            global_lighting,

            gbuffer_bind_layout, gbuffer_bind_group, water_sampler, shadow_sampler,
            water_gbuffer_bind_layout, water_trans_gbuffer_bind_group, water_refl_gbuffer_bind_group,
            above_lighting_bind_group, below_lighting_bind_group,
            lighting_pipeline, underwater_lighting_pipeline, reflected_lighting_pipeline,
        })
    }

    pub fn resize(&mut self, gpu: &GPUContext, size: UVec2) {
        if size == self.gbuffers.size {return}
        self.gbuffers.destroy();
        self.gbuffers = DeferredRendererTextures::create(gpu, size);

        self.gbuffer_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("gbuffer_bind_group"),
            layout: &self.gbuffer_bind_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: wgpu::BindingResource::TextureView(&self.gbuffers.dist_view)},
                BindGroupEntry {binding: 1, resource: wgpu::BindingResource::TextureView(&self.gbuffers.albedo_view)},
                BindGroupEntry {binding: 2, resource: wgpu::BindingResource::TextureView(&self.gbuffers.normal_view)},
                BindGroupEntry {binding: 3, resource: wgpu::BindingResource::TextureView(&self.gbuffers.rm_view)},
                BindGroupEntry {binding: 4, resource: wgpu::BindingResource::TextureView(&self.gbuffers.ao_view)},
                BindGroupEntry {binding: 5, resource: wgpu::BindingResource::TextureView(&self.gbuffers.material_view)},
                BindGroupEntry {binding: 6, resource: wgpu::BindingResource::TextureView(&self.gbuffers.shadow_dist_view)},
                BindGroupEntry {binding: 7, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler)},
                BindGroupEntry {binding: 8, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_trans_view)},
                BindGroupEntry {binding: 9, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_trans_dist_view)},
                BindGroupEntry {binding: 10, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_refl_view)},
                BindGroupEntry {binding: 11, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_refl_dist_view)},
                BindGroupEntry {binding: 12, resource: wgpu::BindingResource::Sampler(&self.water_sampler)},
            ]
        });

        self.water_trans_gbuffer_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("water_gbuffer_bind_group"),
            layout: &self.water_gbuffer_bind_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_trans_dist_view)},
                BindGroupEntry {binding: 1, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_trans_albedo_view)},
                BindGroupEntry {binding: 2, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_trans_normal_view)},
                BindGroupEntry {binding: 3, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_trans_rm_view)},
                BindGroupEntry {binding: 4, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_trans_ao_view)},
                BindGroupEntry {binding: 5, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_trans_material_view)},
                BindGroupEntry {binding: 6, resource: wgpu::BindingResource::TextureView(&self.gbuffers.shadow_dist_view)},
                BindGroupEntry {binding: 7, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler)},
            ]
        });

        self.water_refl_gbuffer_bind_group = gpu.device.create_bind_group(&BindGroupDescriptor{
            label: Some("water_gbuffer_bind_group"),
            layout: &self.water_gbuffer_bind_layout,
            entries: &[
                BindGroupEntry {binding: 0, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_refl_dist_view)},
                BindGroupEntry {binding: 1, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_refl_albedo_view)},
                BindGroupEntry {binding: 2, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_refl_normal_view)},
                BindGroupEntry {binding: 3, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_refl_rm_view)},
                BindGroupEntry {binding: 4, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_refl_ao_view)},
                BindGroupEntry {binding: 5, resource: wgpu::BindingResource::TextureView(&self.gbuffers.water_refl_material_view)},
                BindGroupEntry {binding: 6, resource: wgpu::BindingResource::TextureView(&self.gbuffers.shadow_dist_view)},
                BindGroupEntry {binding: 7, resource: wgpu::BindingResource::Sampler(&self.shadow_sampler)},
            ]
        });
    }

    pub fn render(&mut self, gpu: &GPUContext, out: &wgpu::TextureView, camera_ctrl: &impl CameraController, scene: &mut[&mut dyn RenderObject]) {
        let mut command_encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("deferred_renderer") });

        self.camera = camera_ctrl.camera(self.gbuffers.size.as_vec2(), self.gbuffers.water_size.as_vec2());
        gpu.queue.write_buffer(&self.main_camera_buf, 0, bytemuck::bytes_of(&self.camera));

        for obj in scene.iter_mut() {
            obj.prepass(gpu, &self, &mut command_encoder);
        }
        {
            let mut shadow_pass = command_encoder.begin_render_pass(&RenderPassDescriptor{
                label: Some("shadow-pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.gbuffers.shadow_dist_view,
                    depth_ops: Some(Operations {load: LoadOp::Clear(0.0), store: StoreOp::Store}),
                    stencil_ops: None
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            shadow_pass.set_bind_group(0, &self.global_bind_group, &[]);

            for obj in scene.iter() {
                obj.draw_shadow_casters(gpu, &self, &mut shadow_pass);
            }
        }
        {
            let mut underwater_pass = command_encoder.begin_render_pass(&RenderPassDescriptor{
                label: Some("refracted-opaque-pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_trans_albedo_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_trans_normal_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_trans_rm_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_trans_ao_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_trans_material_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                ],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment{
                    view: &self.gbuffers.water_trans_dist_view,
                    depth_ops: Some(Operations{load: LoadOp::Clear(0.0), store: StoreOp::Store}),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            underwater_pass.set_bind_group(0, &self.global_bind_group, &[]);

            for obj in scene.iter() {
                obj.draw_underwater(gpu, &self, &mut underwater_pass);
            }
        }
        {
            let mut underwater_lighting_pass = command_encoder.begin_render_pass(&RenderPassDescriptor{
                label: Some("refracted-lighting_pass"),
                color_attachments: &[Some(RenderPassColorAttachment{
                    view: &self.gbuffers.water_trans_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: ZERO_COLOR,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            underwater_lighting_pass.set_pipeline(&self.underwater_lighting_pipeline);
            underwater_lighting_pass.set_bind_group(0, &self.global_bind_group, &[]);
            underwater_lighting_pass.set_bind_group(1, &self.water_trans_gbuffer_bind_group, &[]);
            underwater_lighting_pass.set_bind_group(2, &self.below_lighting_bind_group, &[]);
            underwater_lighting_pass.draw(0..3, 0..1);
        }
        {
            let mut reflected_pass = command_encoder.begin_render_pass(&RenderPassDescriptor{
                label: Some("reflected-opaque-pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_refl_albedo_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_refl_normal_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_refl_rm_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_refl_ao_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.water_refl_material_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                ],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment{
                    view: &self.gbuffers.water_refl_dist_view,
                    depth_ops: Some(Operations{load: LoadOp::Clear(0.0), store: StoreOp::Store}),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            reflected_pass.set_bind_group(0, &self.global_bind_group, &[]);

            for obj in scene.iter() {
                obj.draw_reflected(gpu, &self, &mut reflected_pass);
            }
        }
        {
            let mut reflected_lighting_pass = command_encoder.begin_render_pass(&RenderPassDescriptor{
                label: Some("reflected-lighting_pass"),
                color_attachments: &[Some(RenderPassColorAttachment{
                    view: &self.gbuffers.water_refl_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: ZERO_COLOR,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            reflected_lighting_pass.set_pipeline(&self.reflected_lighting_pipeline);
            reflected_lighting_pass.set_bind_group(0, &self.global_bind_group, &[]);
            reflected_lighting_pass.set_bind_group(1, &self.water_refl_gbuffer_bind_group, &[]);
            reflected_lighting_pass.set_bind_group(2, &self.above_lighting_bind_group, &[]);
            reflected_lighting_pass.draw(0..3, 0..1);
        }

        // direct path
        {
            let mut opaque_pass = command_encoder.begin_render_pass(&RenderPassDescriptor{
                label: Some("opaque-pass"),
                color_attachments: &[
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.albedo_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.normal_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.rm_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.ao_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                    Some(RenderPassColorAttachment{
                        view: &self.gbuffers.material_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: ZERO_COLOR,
                    }),
                ],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment{
                    view: &self.gbuffers.dist_view,
                    depth_ops: Some(Operations{load: LoadOp::Clear(0.0), store: StoreOp::Store}),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            opaque_pass.set_bind_group(0, &self.global_bind_group, &[]);

            for obj in scene.iter() {
                obj.draw_opaque(gpu, &self, &mut opaque_pass);
            }
        }
        {
            let mut lighting_pass = command_encoder.begin_render_pass(&RenderPassDescriptor{
                label: Some("lighting_pass"),
                color_attachments: &[Some(RenderPassColorAttachment{
                    view: &out,
                    depth_slice: None,
                    resolve_target: None,
                    ops: ZERO_COLOR,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            lighting_pass.set_pipeline(&self.lighting_pipeline);
            lighting_pass.set_bind_group(0, &self.global_bind_group, &[]);
            lighting_pass.set_bind_group(1, &self.gbuffer_bind_group, &[]);
            lighting_pass.set_bind_group(2, &self.above_lighting_bind_group, &[]);
            lighting_pass.draw(0..3, 0..1);

            for obj in scene.iter() {
                obj.draw_transparent(gpu, &self, &mut lighting_pass);
            }
        }
        gpu.queue.submit(Some(command_encoder.finish()));
    }

    pub const GBUFFER_TARGETS: &'static [Option<ColorTargetState>] = &[
        Some(ColorTargetState{ format: TextureFormat::Rg11b10Ufloat, blend: None, write_mask: ColorWrites::ALL }),
        Some(ColorTargetState{ format: TextureFormat::Rgb10a2Unorm, blend: None, write_mask: ColorWrites::ALL }),
        Some(ColorTargetState{ format: TextureFormat::Rg8Unorm, blend: None, write_mask: ColorWrites::ALL }),
        Some(ColorTargetState{ format: TextureFormat::R8Unorm, blend: None, write_mask: ColorWrites::ALL }),
        Some(ColorTargetState{ format: TextureFormat::R8Uint, blend: None, write_mask: ColorWrites::ALL }),
    ];

    const PATH_REFRACT: u32 = 1;
    const PATH_REFLECT: u32 = 2;
    const OVERRIDES_REFRACT: &[(&str, f64)] = &[("PATH_ID", Self::PATH_REFRACT as f64)];
    const OVERRIDES_REFLECT: &[(&str, f64)] = &[("PATH_ID", Self::PATH_REFLECT as f64)];

    pub fn create_refracted_pipeline(device: &wgpu::Device, desc: &wgpu::RenderPipelineDescriptor) -> wgpu::RenderPipeline {
        let mut desc2 = desc.clone();

        desc2.vertex.compilation_options.constants = Self::OVERRIDES_REFRACT;
        if let Some(frag) = desc2.fragment.as_mut() {
           frag.compilation_options.constants = Self::OVERRIDES_REFRACT;
        };
        device.create_render_pipeline(&desc2)
    }

    pub fn create_reflected_pipeline(device: &wgpu::Device, desc: &wgpu::RenderPipelineDescriptor) -> wgpu::RenderPipeline {
        let mut desc2 = desc.clone();

        desc2.vertex.compilation_options.constants = Self::OVERRIDES_REFLECT;
        if let Some(frag) = desc2.fragment.as_mut() {
           frag.compilation_options.constants = Self::OVERRIDES_REFLECT;
        };
        desc2.primitive.front_face = match desc2.primitive.front_face {
            FrontFace::Cw => FrontFace::Ccw,
            FrontFace::Ccw => FrontFace::Cw,
        };
        device.create_render_pipeline(&desc2)
    }
}

const ZERO_COLOR: Operations<Color> = Operations {load: LoadOp::Clear(Color {r: 0.0, g: 0.0, b: 0.0, a: 0.0 }), store: StoreOp::Store};
