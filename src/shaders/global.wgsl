const PI  = 3.1415926535;
const TAU = 6.2821853072;

const PATH_DIRECT: u32 = 0;
const PATH_REFRACT: u32 = 1;
const PATH_REFLECT: u32 = 2;

override PATH_ID: u32 = PATH_DIRECT;

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
    // depth buffer uses reverse z based on apparant distance according to horizontal parallax.
}

// planar refraction of underwater geometry.
// alters the depth of vertices to their virtual images.
fn apparent_depth(dist: f32, eye_height: f32, depth: f32) -> f32 {
    let d = abs(depth);
    let h = eye_height;
    let x = dist;

    // starting point in correct bucket
    let oi = dist / (d * 0.01 + eye_height);
    var ratio: f32 = sqrt(0.777 * oi * oi + 1.777);
    // use newton's method to find apparant depth ratio
    for (var i = 0; i < 4; i++) {
        let q = ratio;
        let od = d + q * h;
        let o = q * x / od;
        let Do = x * d / od / od;
        let r = sqrt(0.777 * o * o + 1.777);
        let Dr = 0.777 * o * Do / r;
        ratio = q - (r - q) / (Dr - 1);
    }
    return depth / ratio;
}

fn refracted_z(world_pos: vec3f) -> f32 {
    let cam_dist = length(world_pos.xy - camera.eye.xy);
    return apparent_depth(cam_dist, camera.eye.z, world_pos.z);
}

fn clip_point(world_pos: vec3f) -> vec4f {
    var z = world_pos.z;
    if PATH_ID == PATH_REFLECT {
        z = -z;
    } else if PATH_ID == PATH_REFRACT {
        let cam_dist = length(world_pos.xy - camera.eye.xy);
        z = apparent_depth(cam_dist, camera.eye.z, world_pos.z);
    }

    let virt_pos = vec4f(world_pos.xy, z, 1.0);
    return camera.matrix * virt_pos;
}

fn guard_frag(world_pos: vec3f) {
    switch PATH_ID {
        case PATH_DIRECT{
            if world_pos.z < -0.05 {
                discard;
            }
        }
        case PATH_REFLECT {
            if world_pos.z < -0.1 {
                discard;
            }
        }
        case PATH_REFRACT {
            if world_pos.z > 0.05 {
                discard;
            }
        }
        default {}
    }
}

// ported from glam no-sse fallback
fn quat_rotate(q: vec4f, v: vec3f) -> vec3f { 
	    let w = q.w;
        let b = q.xyz;
        let b2 = dot(b, b);
        return v * (w * w - b2) + (b * dot(v, b) * 2.0) + (cross(b, v) * 2.0 * w);
} 
