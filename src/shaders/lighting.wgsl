
@vertex fn fullscreen_tri(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
    let u = f32(idx % 2);
    let v = f32(idx / 2);
    let xy = vec2f(4*u -1, 4*v-1);
    return vec4f(xy, 1, 1);
}

@group(1) @binding(0) var dist_buf: texture_depth_2d;
@group(1) @binding(1) var albedo_buf: texture_2d<f32>;
@group(1) @binding(2) var normal_buf: texture_2d<f32>;
@group(1) @binding(3) var rm_buf: texture_2d<f32>;
@group(1) @binding(4) var ao_buf: texture_2d<f32>;
@group(1) @binding(5) var material_buf: texture_2d<u32>;

@group(1) @binding(6) var water_buf: texture_2d<f32>;
@group(1) @binding(7) var water_dist_buf: texture_depth_2d;
@group(1) @binding(8) var water_sampler: sampler;

const sun = vec3(0.548, -0.380, 0.745);

@fragment fn do_global_lighting(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let px = vec2i(floor(pos.xy));
    let albedo = textureLoad(albedo_buf, px, 0).xyz;
    let normal = 2 * textureLoad(normal_buf, px, 0).xyz - 1;
    let light = 0.1 + 0.9 * max(0.0, dot(normal, sun));
    return vec4f(albedo * light, 1);
}

@group(1) @binding(6) var depth_adj_buf: texture_2d<f32>;

@fragment fn do_underwater_lighting(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let px = vec2i(floor(pos.xy));
    let albedo = textureLoad(albedo_buf, px, 0).xyz;
    let normal = 2 * textureLoad(normal_buf, px, 0).xyz - 1;
    let light = 0.1 + 0.9 * max(0.0, dot(normal, sun));
    return vec4f(albedo * light, 1);
}

