struct SDFTextParams {
    viewport_loc: vec2f,
    size_vh: vec2f,
    color: vec4f,
    shadow_color: vec4f,
    shadow_size: f32,
    margin_vh: f32,
    sdf_rad: f32,
    num_chars: u32,
    chars: array<u32,4>,
}

@group(1) @binding(0) var<storage, read> params_buf: array<SDFTextParams>;
@group(1) @binding(1) var sdf_atlas: texture_2d_array<f32>;
@group(1) @binding(2) var bilinear_sampler: sampler;

struct TextVSOut {
    @builtin(position) clip_pos: vec4f,
    @interpolate(flat) @location(0) inst_idx: u32,
    @location(1) uv: vec2f,
}

@vertex fn sdf_text_vert(@builtin(vertex_index) vert: u32, @builtin(instance_index) inst_idx: u32) -> TextVSOut {
    let inst = params_buf[inst_idx];
    let ij = vec2f(vec2u(vert / 2, vert % 2));
    let uv = vec2f(ij.x * f32(inst.num_chars), ij.y);

    let aspect = camera.fb_size.x / camera.fb_size.y;
    let n = (aspect - 2 * inst.margin_vh - inst.size_vh.x) * inst.viewport_loc.x + inst.margin_vh;
    let w = (1.0 - 2 * inst.margin_vh - inst.size_vh.y) * inst.viewport_loc.y + inst.margin_vh;
    let xy = vec2f(n, w) + ij * inst.size_vh;

    var out: TextVSOut;
    out.clip_pos = vec4f(2 * xy.x / aspect - 1, 1 - 2 * xy.y, 0, 1);
    //out.clip_pos = vec4f(ij, 1, 1);
    out.inst_idx = inst_idx;
    out.uv = uv;
    return out;
}

@fragment fn sdf_text_frag(v: TextVSOut) -> @location(0) vec4f {
    var inst = params_buf[v.inst_idx];
    let char = inst.chars[u32(floor(v.uv.x))];
    let uv = vec2f(fract(v.uv.x), v.uv.y);

    let px_scale = dpdy(v.uv.y) * f32(textureDimensions(sdf_atlas).y);
    let px_dist = 0.5 * px_scale / inst.sdf_rad;
    let sdf_val = textureSampleLevel(sdf_atlas, bilinear_sampler, uv, char, 0.0).x;

    let shadow_fac = smoothstep(0.5 - 0.5 * inst.shadow_size, 0.5 - 0.15 * inst.shadow_size, sdf_val);
    let text_fac = smoothstep(0.5 - px_dist, 0.5 + px_dist, sdf_val);

    let shadow = mix(vec4f(0), inst.shadow_color, shadow_fac);
    return mix(shadow, inst.color, text_fac);
}

struct BlackoutVSOut {
    @builtin(position) pos: vec4f,
    @location(0) alpha: f32,
}

@vertex fn blackout_vert(@builtin(vertex_index) idx: u32, @builtin(instance_index) inst: u32) -> BlackoutVSOut {
    let u = f32(idx % 2);
    let v = f32(idx / 2);
    let xy = vec2f(4*u -1, 4*v-1);
    var out: BlackoutVSOut;
    out.pos = vec4f(xy, 1, 1);
    out.alpha = smoothstep(0.0, 900.0, f32(inst));
    return out;
}

@fragment fn blackout_frag(v: BlackoutVSOut) -> @location(0) vec4f {
    return vec4f(0, 0, 0, v.alpha);
}
