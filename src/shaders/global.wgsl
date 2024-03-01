const PI  = 3.1415926535;
const TAU = 6.2821853072;

struct Camera {
    matrix: mat4x4f,
    inv_matrix: mat4x4f,
    eye: vec3f,
    clip_near: f32,
    fb_size: vec2f,
    time: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
//@group(0) @binding(0) var<uniform> sun: DirectionLight;

const MAT_SKY: u32 = 0; // cleared value;
const MAT_SOLID: u32 = 1; // general Cook-Torrance
const MAT_WATER: u32 = 2; // albedo is transmitted value
const MAT_LEAF: u32 = 3; // semi-translucent


struct GBufferPoint {
    @location(0) albedo: vec4f, // rgba8-srgb
    @location(1) normal: vec4f, // world-space normal ramapped from [-1,1] to [0,1] and stored in rgb10a2-unorm
    @location(2) rough_metal: vec2f, //rg8-unorm
    @location(3) occlusion: f32, //r8-unorm, collects micro-occlusion from textures and then multiplied with SSAO.
    @location(4) mat_type: u32, //r8-uint
}

struct UnderwaterPoint {
    @location(0) albedo: vec4f, // rgba8-srgb
    @location(1) normal: vec4f, // world-space normal ramapped from [-1,1] to [0,1] and stored in rgb10a2-unorm
    @location(2) rough_metal: vec2f, //rg8-unorm
    @location(3) occlusion: f32, //r8-unorm, collects micro-occlusion from textures and then multiplied with SSAO.
    @location(4) mat_type: u32, //r8-uint
    @location(5) depth_adj: f32, //r8-unorm, refracted depth/worldspace depth
}

// planar refraction of underwater geometry.
// alters the depth of vertices to their virtual images.
fn apparent_depth(dist: f32, eye_height: f32, depth: f32) -> f32 {
    var ratio: f32 = 1.33; // on-axis value
    var oblique = dist / (abs(depth) / ratio  + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
    oblique = dist / (abs(depth) / ratio + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
    oblique = dist / (abs(depth) / ratio + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
    oblique = dist / (abs(depth) / ratio + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
    return depth / ratio;
}

