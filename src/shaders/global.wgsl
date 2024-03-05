const PI  = 3.1415926535;
const TAU = 6.2821853072;

struct Camera {
    matrix: mat4x4f,
    inv_matrix: mat4x4f,
    eye: vec3f,
    clip_near: f32,
    fb_size: vec2f,
    water_fb_size: vec2f,
    time: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
//@group(0) @binding(0) var<uniform> sun: DirectionLight;

const MAT_SKY: u32 = 0; // cleared value;
const MAT_SOLID: u32 = 1; // general Cook-Torrance
const MAT_WATER: u32 = 2; // albedo is transmitted value
const MAT_LEAF: u32 = 3; // semi-translucent
const MAT_EMIT: u32 = 4; // emissive material. Albedo = final color


struct GBufferPoint {
    @location(0) albedo: vec4f, // rg11b10-ufloat
    @location(1) normal: vec4f, // world-space normal ramapped from [-1,1] to [0,1] and stored in rgb10a2-unorm
    @location(2) rough_metal: vec2f, //rg8-unorm, metal channel is mode-dependant
    @location(3) occlusion: f32, //r8-unorm, collects micro-occlusion from textures and then multiplied with SSAO.
    @location(4) mat_type: u32, //r8-uint
}

struct UnderwaterPoint {
    @location(0) albedo: vec4f, // rg11b10-ufloat
    @location(1) normal: vec4f, // world-space normal ramapped from [-1,1] to [0,1] and stored in rgb10a2-unorm
    @location(2) rough_metal: vec2f, //rg8-unorm metal channel is mode-dependant
    @location(3) occlusion: f32, //r8-unorm, collects micro-occlusion from textures and then multiplied with SSAO.
    @location(4) mat_type: u32, //r8-uint
    @location(5) depth_adj: f32, //r16-float, refracted vertical depth/worldspace depth
    // depth buffer uses reverse z based on apparant distance according to horizontal parallax.
}

// planar refraction of underwater geometry.
// alters the depth of vertices to their virtual images.
fn apparent_depth(dist: f32, eye_height: f32, depth: f32) -> f32 {
    // Finding apparent depth off-axis requires finding the fixed point of a nasty quartic.
    // To approximate start with a crude estimate (a fitted manually in Desmos)
    // then refine by ineration of the actual equation 
    let x = dist / (depth + 3 * eye_height);
    let init_ratio = 1.33 * (x * x + 1);

    var ratio: f32 = init_ratio; 
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

fn refracted_z(world_pos: vec3f) -> f32 {
    let cam_dist = length(world_pos.xy - camera.eye.xy);
    return apparent_depth(cam_dist, camera.eye.z, world_pos.z);
}
