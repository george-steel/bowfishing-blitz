#if CAN_CLIP
enable clip_distances;
#endif

#include global.wgsl

struct Arrow {
    end_pos: vec3f,
    state: u32,
    dir: vec3f,
    len: f32,
}

@group(1) @binding(0) var<storage, read> arrows: array<Arrow>;

struct ArrowVSIn {
    @location(0) pos: vec3f,
    @location(1) norm: vec3f,
    @location(2) uv: vec2f,
}

struct ArrowVSOut {
    #if CAN_CLIP
        @builtin(clip_distances) clip: array<f32, 1>,
    #endif
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) world_norm: vec3f,
    @location(3) uv: vec2f,
}

struct ArrowFragIn {
    @location(0) world_pos: vec3f,
    @location(1) world_norm: vec3f,
    @location(3) uv: vec2f,
}

@vertex fn arrow_vert(vert: ArrowVSIn, @builtin(instance_index) inst: u32) -> ArrowVSOut {
    let arr = arrows[inst];
    let right = -normalize(cross(vec3f(0.0, 0.0, 1.0), arr.dir));
    let up = normalize(cross(right, arr.dir));

    // includes stretch and squash for visibility
    let width = clamp(sqrt(length(arr.end_pos - camera.eye) / 8), 1.0, 3.0);
    let pos_mat = mat3x3f(width * right, arr.len * normalize(arr.dir), width * up);
    let norm_mat = mat3x3f(right, normalize(arr.dir), up);

    let world_pos = arr.end_pos + pos_mat * vert.pos;
    let world_norm = norm_mat * vert.norm;

    var out: ArrowVSOut;
    #if CAN_CLIP
        out.clip[0] = clip_dist(world_pos);
    #endif
    out.clip_pos = clip_point(world_pos);
    out.world_pos = world_pos;
    out.world_norm = world_norm;
    out.uv = vert.uv;
    return out;
}

@vertex fn arrow_vert_shadow(vert: ArrowVSIn, @builtin(instance_index) inst: u32) -> @builtin(position) vec4f {
    let arr = arrows[inst];
    let right = -normalize(cross(vec3f(0.0, 0.0, 1.0), arr.dir));
    let up = normalize(cross(right, arr.dir));

    // includes stretch and squash for visibility
    let width = clamp(sqrt(length(arr.end_pos - camera.eye) / 8), 1.0, 3.0);
    let pos_mat = mat3x3f(width * right, arr.len * normalize(arr.dir), width * up);

    let world_pos = arr.end_pos + pos_mat * vert.pos;

    return shadow_clip_point(world_pos);
}

@fragment fn arrow_frag(v: ArrowFragIn, @builtin(front_facing) is_forward: bool) -> GBufferPoint {
    #if !CAN_CLIP
        guard_frag(v.world_pos.z);
    #endif

    let norm = normalize(v.world_norm) * select(-1.0, 1.0, is_forward);

    var albedo: vec4f;
    var rough: f32;
    var metal: f32;
    var mat_type = MAT_SOLID;
    if v.uv.x < 0.25 {
        albedo = vec4f(0.7, 0.7, 0.7, 1.0);
        rough = 0.15;
        metal = 1.0;
    } else if v.uv.x > 0.75 {
        albedo = vec4f(1.0, 0.1, 0.0, 1.0);
        rough = 0.25;
        metal = 0.4;
        mat_type = MAT_EMIT;
    } else {
        albedo = vec4f(0.57012, 0.13881, 0.05111, 1);
        rough = 0.5;
        metal = 0.0;
    }

    var out: GBufferPoint;
    out.albedo = albedo;
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.rough_metal = vec2f(rough, metal);
    out.occlusion = 1.0;
    out.mat_type = mat_type;
    return out;
}

struct Splish {
    @location(0) center: vec2f,
    @location(1) start_time: f32,
}

@group(1) @binding(0) var<storage, read> splish_buf: array<Splish>;

struct SplishVSOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) local_xy: vec2f,
    @location(1) time: f32,
}

@vertex fn splish_vert(@builtin(vertex_index) vert: u32, inst: Splish) -> SplishVSOut {
    let local_xy = vec2f(2.0 * f32(vert / 2) - 1.0, 1.0 - 2.0 * f32(vert % 2));
    let world_pos = vec3f(inst.center + local_xy, 0.0);
    
    var out: SplishVSOut;
    out.clip_pos = camera.matrix * vec4f(world_pos, 1.0);
    out.local_xy = local_xy;
    out.time = 0.5 * (camera.time - inst.start_time);
    return out;
}

@fragment fn splish_frag(v: SplishVSOut) -> GBufferPoint {
    let t_fac: f32 = (1 - v.time) * (1 - v.time);
    let r_center = 0.8 * v.time;
    let r = length(v.local_xy);
    let r_delta = abs(r - r_center) * 10;
    if r_delta > 3 {
        discard;
    }
    let r_fac = exp(- r_delta * r_delta);
    let alpha = 0.5 * t_fac * r_fac;

    let dzdr = sin(TAU * 20 * (r - 1.2 * v.time));
    let r_dir = normalize(v.local_xy);

    var out: GBufferPoint;
    out.normal = vec4f(0.5 - 0.4 * dzdr * r_dir, 0.0, alpha);
    return out;
}
