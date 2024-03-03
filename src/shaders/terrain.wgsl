const ls1mat = mat2x2f(0.2, 0.1, -0.1, 0.2);
const ls2mat = mat2x2f(0.5, -0.5, 0.5, 0.5);
const oct = mat2x2f(2.159, 0.978, -0.978, 2.159);
fn terrain(uv: vec2f) -> vec3f {
    let ls1 = 1.5 * perlin_noise_deriv(uv, ls1mat, 0) + const_gradval(0.2);
    let ls2 = perlin_noise_deriv(uv, ls2mat, 1);
    let ls = ls1 + 0.3 * mult_gradval(ls1 + const_gradval(1.0), ls2);
    let fs = fbm_deriv(uv, id2, 6u, oct, 0.35, 5);
    return ls + mult_gradval(fs, const_gradval(0.2) + 0.5 * abs_gradval(ls));// - const_gradval(1.0);
}

struct TerrainParams {
    radius: f32,
    z_scale: f32,
    grid_size: u32,
}


struct TerrainVertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) refract_pos: vec3f,
    @location(2) uv: vec2f,
}

@group(1) @binding(0) var<uniform> tparams: TerrainParams;
@group(1) @binding(1) var terrain_height: texture_2d<f32>;
@group(1) @binding(2) var terrain_sampler: sampler;

const DELTA_U = vec2f(1.0 / 2048.0, 0.0);
const DELTA_V = vec2f(0.0, 1.0 / 2048.0);
// sample finite differences of the heightmap
fn terrain_grad(uv: vec2f) -> vec2f {
    let n = textureSampleLevel(terrain_height, terrain_sampler, uv - DELTA_V, 0.0).x;
    let s = textureSampleLevel(terrain_height, terrain_sampler, uv + DELTA_V, 0.0).x;
    let w = textureSampleLevel(terrain_height, terrain_sampler, uv - DELTA_U, 0.0).x;
    let e = textureSampleLevel(terrain_height, terrain_sampler, uv + DELTA_U, 0.0).x;
    return tparams.z_scale * 512.0 * vec2f(e - w, n - s) / tparams.radius;
}

@vertex fn terrain_mesh(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> TerrainVertexOut {
    let ij = vec2i(vec2u(inst_idx + (vert_idx % 2), vert_idx / 2));
    let uv = vec2f(0.0, 1.0) + vec2f(1.0, -1.0) * vec2f(ij) / f32(tparams.grid_size);
    let xy = tparams.radius * (2 * vec2f(ij) / f32(tparams.grid_size) - 1);
    let z = tparams.z_scale * textureSampleLevel(terrain_height, terrain_sampler, uv, 0.0).x;
    let world_pos = vec4f(xy, z, 1);

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * world_pos;
    out.refract_pos = world_pos.xyz;
    out.world_pos = world_pos.xyz;
    out.uv = uv;
    return out;
}

@vertex fn underwater_terrain_mesh(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> TerrainVertexOut {
    let ij = vec2i(vec2u(inst_idx + (vert_idx % 2), vert_idx / 2));
    let uv = vec2f(0.0, 1.0) + vec2f(1.0, -1.0) * vec2f(ij) / f32(tparams.grid_size);
    let xy = tparams.radius * (2 * vec2f(ij) / f32(tparams.grid_size) - 1);
    let z = tparams.z_scale * textureSampleLevel(terrain_height, terrain_sampler, uv, 0.0).x;
    let world_pos = vec4f(xy, z, 1);


    // Clip-space planar refraction: move vertices to their virtual images when converting to clip space.
    var refract_pos = world_pos;
    let cam_dist = length(world_pos.xy - camera.eye.xy);
    refract_pos.z = apparent_depth(cam_dist, camera.eye.z, world_pos.z);

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * refract_pos;
    out.refract_pos = refract_pos.xyz;
    out.world_pos = world_pos.xyz;
    out.uv = uv;
    return out;
}

const grass_col = vec3f(0.26406, 0.46721, 0.12113);
const beach_col = vec3f(0.45, 0.37, 0.25);
const rock_col = vec3f(0.168, 0.171, 0.216);

fn terrain_albedo(uv: vec2f, z: f32, norm: vec3f, shore: f32) -> vec3f {
    let bias: f32 = 0.0; //fbm_deriv(uv, mat2x2f(6.0, 0.0, 0.0, 6.0), 4u, oct, 0.6, 0).z;
    let flat_col = mix(beach_col - 0.1 * bias, grass_col - 0.05 * bias, smoothstep(0.0, shore, z - 0.1 * bias));
    return mix(rock_col + 0.05 * bias, flat_col, smoothstep(0.5, 0.9, norm.z + 0.1 * bias));
}

@fragment fn terrain_frag(v: TerrainVertexOut) -> GBufferPoint {
    if v.world_pos.z <  -0.1 {
        discard; // skip texturing above water
    }
    let grad = terrain_grad(v.uv);
    let norm = normalize(vec3f(-grad, 1));

    let z = tparams.z_scale * textureSampleLevel(terrain_height, terrain_sampler, v.uv, 0.0).x;
    let albedo = terrain_albedo(v.uv, z, norm, 0.2);

    var out: GBufferPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_SOLID;
    return out;
}

@fragment fn underwater_terrain_frag(v: TerrainVertexOut) -> UnderwaterPoint {
    if v.world_pos.z > 0.1 {
        discard; // skip texturing above water
    }

    let grad = terrain_grad(v.uv);
    let norm = normalize(vec3f(-grad, 1));

    let z = tparams.z_scale * textureSampleLevel(terrain_height, terrain_sampler, v.uv, 0.0).x;
    let albedo = terrain_albedo(v.uv, z, norm, 0.2);

    var out: UnderwaterPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_SOLID;
    out.depth_adj = v.world_pos.z / v.refract_pos.z;
    return out;
}

fn water_ripples(xy: vec2f) -> gradval {
    return 0.015 * perlin_noise_deriv(xy + vec2f(0.1, -0.55) * camera.time, mat2x2f(0.8, -1.9, 3.8, 0.4), 1)
        + 0.012 * perlin_noise_deriv(xy + vec2f(-0.05, 0.4) * camera.time, mat2x2f(3.5, 0.0, 0.0, 6.7), 0);
}

@vertex fn water_quad(@builtin(vertex_index) vert_idx: u32) -> TerrainVertexOut {
    let ij = vec2u(vert_idx % 2, vert_idx / 2);
    let uv = tparams.radius * (2.0 * vec2f(ij) - 1.0);
    let world_pos = vec4f(uv, 0, 1);

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * world_pos;
    out.world_pos = world_pos.xyz;
    out.uv = uv;
    return out;
}

@fragment fn water_frag(v: TerrainVertexOut) -> GBufferPoint {
    let ripple = water_ripples(v.world_pos.xy);
    let norm = normalize(vec3f(-ripple.xy, 1.0));

    var out: GBufferPoint;
    out.albedo = vec4f(0.5, 0.5, 0.5, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_WATER;
    return out;
}


