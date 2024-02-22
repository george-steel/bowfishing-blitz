const ls1mat = mat2x2f(0.2, 0.1, -0.1, 0.2);
const ls2mat = mat2x2f(0.5, -0.5, 0.5, 0.5);
const oct = mat2x2f(2.159, 0.978, -0.978, 2.159);
fn terrain(uv: vec2f) -> vec3f {
    let ls1 = 1.5 * perlin_noise_deriv(uv, ls1mat, 0) + const_gradval(0.2);
    let ls2 = perlin_noise_deriv(uv, ls2mat, 1);
    let ls = ls1 + 0.3 * mult_gradval(ls1 + const_gradval(1.0), ls2);
    let fs = fbm_deriv(uv, id2, 6u, oct, 0.35, 4);
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
    @location(1) terrain_coord: vec2f,
    @location(2) world_normal: vec3f,
}

@group(1) @binding(0) var<uniform> tparams: TerrainParams;

@vertex fn terrain_mesh(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> TerrainVertexOut {
    let ij = vec2i(vec2u(inst_idx + (vert_idx % 2), vert_idx / 2));
    let uv = (2.0 * vec2f(ij) / f32(tparams.grid_size) - 1.0) * tparams.uv_radius + tparams.uv_center;
    let zd = terrain(uv);
    let local_pos = vec4f(uv, zd.z, 1);
    let local_norm = normalize(vec3f(-zd.xy, 1));
    let world_pos = tparams.transform * local_pos;
    let world_norm = (vec4f(local_norm, 0) * tparams.inv_transform).xyz;

    var refract_pos = world_pos;
    /*if zd.z < 0 {
        let sealevel = tparams.transform[3].z;
        let cam_height = camera.eye.z - sealevel;
        let cam_dist = length(world_pos.xy - camera.eye.xy);
        let depth = world_pos.z - sealevel;
        refract_pos.z = sealevel + apparent_depth(cam_dist, cam_height, depth);
    }*/

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * refract_pos;
    out.world_pos = world_pos.xyz;
    out.terrain_coord = uv;
    out.world_normal = world_norm;
    return out;
}

const sun = vec3(0.548, -0.380, 0.745);
const uw_sun = vec3(0.412, -0.285, 0.865);

@fragment fn terrain_frag(v: TerrainVertexOut) -> GBufferPoint {
    let uv = v.terrain_coord;
    let zd = terrain(uv);
    let local_norm = normalize(vec3f(-zd.xy, 1));
    let norm = normalize((vec4f(local_norm, 0) * tparams.inv_transform).xyz);

    let z = zd.z;
    var col: vec3f;
    if z >= 0 {
        col = vec3f(saturate(0.5 + 0.5 * z), 0.9, saturate(0.3 + 0.3 * z));
    } else {
        col = vec3f(0.7, 0.7, 0.8);
    }
    // contour lines
    let hspan = length(vec2f(dpdx(z), dpdy(z)));
    let cdist = 0.1 * abs(10 * z - round(10 * z)) / hspan; // in pixels
    let line_fac = smoothstep(0.5, 1.0, cdist);
    let lcol = vec3f(0.1, 0.5, 0.5);

    let albedo = mix(lcol, col, line_fac);
    /*
    var spec: f32 = 0.0;
    if zd.z > 0 {
        let view = normalize(camera.eye.xyz - v.world_pos);
        let h = normalize(view + sun);
        let crd = length(norm - h);
        spec = 1.0 - smoothstep(0.02, 0.05, crd);
    }*/

    var out: GBufferPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.mat_type = MAT_SOLID;
    return out;
    
    //let norm = normalize(v.world_normal);
    //let light = 0.02 + 0.9 * max(0.0, dot(select(sun, uw_sun, z < 0), norm));
    //return vec4f(vec3f(spec) + albedo * light, 1);
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
    out.world_normal = world_norm;
    return out;
}

@fragment fn water_frag(v: TerrainVertexOut) -> @location(0) vec4f {
    let norm = normalize((vec4f(0,0,1,0) * tparams.inv_transform).xyz);

    let to_eye = normalize(camera.eye.xyz - v.world_pos);
    let refl = 0.02 + 0.98 * pow(1.0 - dot(norm, to_eye), 5.0);
    let h = normalize(to_eye + sun);
    let crd = length(norm - h);
    let spec = 1.0-smoothstep(0.05, 0.15, crd);
    return vec4f(spec, spec, spec, refl);
}

struct SkyQuadOut {
    @builtin(position) pos: vec4f,
    @location(0) clip_xy: vec2f,
}

@group(0) @binding(1) var sky_tex: texture_2d<f32>;
@group(0) @binding(2) var sky_sampler: sampler;
const pi = 3.1415926535;

@vertex fn sky_vert(@builtin(vertex_index) idx: u32) -> SkyQuadOut {
    let u = f32(idx % 2);
    let v = f32((idx / 2) % 2);
    let xy = vec2f(2*u -1, 2*v-1);
    var out: SkyQuadOut;
    out.pos = vec4f(xy, 0.0001, 1);
    out.clip_xy = xy;
    return out;
}

@fragment fn sky_frag(vsout: SkyQuadOut) -> @location(0) vec4f {
    let look = normalize((camera.inv_matrix * vec4f(vsout.clip_xy * camera.clip_near, camera.clip_near, camera.clip_near)).xyz - camera.eye);
    let v = 0.5 - atan2(look.z, length(look.xy)) / pi;
    let u = 0.5 - atan2(look.y, look.x) / (2*pi);
    let c = textureSample(sky_tex, sky_sampler, vec2f(u, v));
    return c;
}

