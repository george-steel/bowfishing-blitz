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
    transform: mat4x4f, // must be affine with the linear portion having have z as an eigenvector
    inv_transform: mat4x4f,
    uv_center: vec2f,
    uv_radius: f32,
    grid_size: u32,
}


struct TerrainVertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) refract_pos: vec3f,
    @location(2) terrain_coord: vec2f,
}

@group(1) @binding(0) var<uniform> tparams: TerrainParams;

@vertex fn terrain_mesh(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> TerrainVertexOut {
    let ij = vec2i(vec2u(inst_idx + (vert_idx % 2), vert_idx / 2));
    let uv = (2.0 * vec2f(ij) / f32(tparams.grid_size) - 1.0) * tparams.uv_radius + tparams.uv_center;
    let zd = terrain(uv);
    let local_pos = vec4f(uv, zd.z, 1);
    let world_pos = tparams.transform * local_pos;

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * world_pos;
    out.refract_pos = world_pos.xyz;
    out.world_pos = world_pos.xyz;
    out.terrain_coord = uv;
    return out;
}

@vertex fn underwater_terrain_mesh(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> TerrainVertexOut {
    let ij = vec2i(vec2u(inst_idx + (vert_idx % 2), vert_idx / 2));
    let uv = (2.0 * vec2f(ij) / f32(tparams.grid_size) - 1.0) * tparams.uv_radius + tparams.uv_center;
    let zd = terrain(uv);
    let local_pos = vec4f(uv, zd.z, 1);
    let world_pos = tparams.transform * local_pos;

    // Clip-space planar refraction: move vertices to their virtual images when converting to clip space.
    var refract_pos = world_pos;
    let sealevel = tparams.transform[3].z;
    let cam_height = camera.eye.z - sealevel;
    let cam_dist = length(world_pos.xy - camera.eye.xy);
    let depth = world_pos.z - sealevel;
    refract_pos.z = sealevel + apparent_depth(cam_dist, cam_height, depth);

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * refract_pos;
    out.refract_pos = refract_pos.xyz;
    out.world_pos = world_pos.xyz;
    out.terrain_coord = uv;
    return out;
}

const grass_col = vec3f(0.26406, 0.46721, 0.12113);
const beach_col = vec3f(0.45, 0.37, 0.25);
const rock_col = vec3f(0.168, 0.171, 0.216);

fn terrain_albedo(uv: vec2f, z: f32, norm: vec3f, shore: f32) -> vec3f {
    let bias = fbm_deriv(uv, mat2x2f(6.0, 0.0, 0.0, 6.0), 4u, oct, 0.6, 0).z;
    let flat_col = mix(beach_col - 0.1 * bias, grass_col - 0.05 * bias, smoothstep(0.0, shore, z - 0.1 * bias));
    return mix(rock_col + 0.05 * bias, flat_col, smoothstep(0.5, 0.9, norm.z + 0.1 * bias));
}

@fragment fn terrain_frag(v: TerrainVertexOut) -> GBufferPoint {
    let uv = v.terrain_coord;
    let zd = terrain(uv);
    let local_norm = normalize(vec3f(-zd.xy, 1));
    let norm = normalize((vec4f(local_norm, 0) * tparams.inv_transform).xyz);

    let z = zd.z;
    let col = vec3f(saturate(0.5 + 0.5 * z), 0.9, saturate(0.3 + 0.3 * z));

    // contour lines
    /*let hspan = length(vec2f(dpdx(z), dpdy(z)));
    let cdist = 0.1 * abs(10 * z - round(10 * z)) / hspan; // in pixels
    let line_fac = smoothstep(1.0, 1.5, cdist);
    let lcol = vec3f(0.1, 0.5, 0.5);
    let albedo = mix(lcol, col, line_fac);*/
    let albedo = terrain_albedo(uv, z, norm, 0.2);

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

    let uv = v.terrain_coord;
    let zd = terrain(uv);
    let local_norm = normalize(vec3f(-zd.xy, 1));
    let norm = normalize((vec4f(local_norm, 0) * tparams.inv_transform).xyz);

    let z = zd.z;
    let col = vec3f(0.7, 0.7, 0.8);

    let albedo = terrain_albedo(uv, z, norm, 0.2);

    var out: UnderwaterPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_SOLID;
    out.depth_adj = v.refract_pos.z / v.world_pos.z;
    return out;
}

fn water_ripples(xy: vec2f) -> gradval {
    return 0.005 * sin_wave_deriv(xy, vec2f(0.5, 1.8), (camera.time / 3.0))
        + 0.015 * perlin_noise_deriv(xy + vec2f(0.1, -0.55) * camera.time, mat2x2f(0.8, -1.9, 3.8, 0.4), 1)
        + 0.012 * perlin_noise_deriv(xy + vec2f(-0.05, -0.4) * camera.time, mat2x2f(3.5, 0.0, 0.0, 6.7), 0);
}

@vertex fn water_quad(@builtin(vertex_index) vert_idx: u32) -> TerrainVertexOut {
    let ij = vec2u(vert_idx % 2, vert_idx / 2);
    let uv = tparams.uv_center + tparams.uv_radius * (2.0 * vec2f(ij) - 1.0);
    let local_pos = vec4f(uv, 0, 1);
    let world_pos = tparams.transform * local_pos;
    let world_norm = (vec4f(0, 0, 1, 0) * tparams.inv_transform).xyz;

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * world_pos;
    out.world_pos = world_pos.xyz;
    out.terrain_coord = uv;
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


