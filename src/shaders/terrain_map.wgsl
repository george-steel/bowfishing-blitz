const PI  = 3.1415926535;
const TAU = 6.2821853072;

struct FQOut {
    @builtin(position) pos: vec4f,
    @location(0) xy: vec2f,
}

@vertex fn fullscreen_quad(@builtin(vertex_index) vert_idx: u32) -> FQOut {
    let ij = vec2u(vert_idx % 2, vert_idx / 2);
    let xy = 2.0 * vec2f(ij) - 1.0;
    var out: FQOut;
    out.pos = vec4f(xy, 0.0, 1.0);
    out.xy = 60 * xy;
    return out;
}

const sun = vec3f(-3.0, 4.0, 8.0) / 13.0;

@fragment fn terrain_map(vert: FQOut) -> @location(0) vec4f {
    let xy = vert.xy;
    let zd = terrain(xy);
    let norm = normalize(vec3f(-zd.xy, 1));
    let z = zd.z;

    let hspan = length(vec2f(dpdx(z), dpdy(z)));
    let cdist = abs(z - round(z)) / hspan; // in pixels
    let line_fac = smoothstep(1.0, 1.5, cdist);

    let xdist = 10 * abs(0.1 * xy.x - round(0.1 * xy.x)) / abs(dpdx(xy.x));
    let ydist = 10 * abs(0.1 * xy.y - round(0.1 * xy.y)) / abs(dpdy(xy.y));
    let grid_fac = 0.5 - 0.5 * smoothstep(1.0, 1.5, min(xdist, ydist));

    var col: vec3f;
    if z > 0 {
        let rb = 0.1 * z;
        col = vec3f(rb, 0.9, rb);
    } else {
        let b = 1.0 + 0.1 * z;
        col = vec3f(0.3, 0.3, b);
    }
    let lcol = vec3f(0.1, 0.2, 0.2);
    let albedo = mix(mix(lcol, col, line_fac), vec3f(0), grid_fac);
    let light = 0.8 * dot(sun, norm) + 0.2;
    return vec4f(albedo * light, 1.0);
}

@fragment fn terrain_heightmap(vert: FQOut) -> @location(0) vec4f {
    let zd = terrain(vert.xy);
    return vec4f(zd.z, 0, 0, 1);
}

const ls1mat = mat2x2f(0.1, 0., -0.1, 0.2);
const ls2mat = mat2x2f(0.5, -0.5, 0.5, 0.5);
const oct = mat2x2f(1.572, 0.602, -0.602, 1.572);

fn terrain(uv: vec2f) -> gradval {
    let gain =  0.9 / sqrt(determinant(oct));
    let r = length(uv);
    let base_z = -2.5 * cos(r * PI / 80) + 4.0 * cos(r * TAU / 45) * select(1.0, 0.5, r > 45.0) + select(0.0, 2.0, r > 45.0) ;
    let base_grad = normalize(uv) * ((2.7 * PI / 80) * sin(r * PI / 80) - (4.0 * TAU / 45) * sin(r * TAU / 45) * select(1.0, 0.5, r > 45.0));
    let base_shape = vec3f(base_grad, base_z);
    let ls = 3 * perlin_noise_deriv(uv, mat2x2f(0.02, -0.01, 0.01, 0.03), -3) - 4 * perlin_noise_deriv(uv, mat2x2f(0.02,-0.03, 0.03, 0.02), -1);
    let fs = 8 * fbm_deriv(uv, mat2x2f(0.05, 0, 0, 0.05), 8u, oct, gain, 0);
    return base_shape + ls + fs ;
}

@group(0) @binding(0) var prev_mip: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> mip_buffer: array<u32>;

struct MipTriOut {
    @builtin(position) pos: vec4f,
    @location(0) mip_num: u32,
    @location(1) mip_size: u32,
    @location(2) buf_offset: u32,
}

@vertex fn fullscreen_tri(@builtin(vertex_index) idx: u32, @builtin(instance_index) inst: u32) -> MipTriOut {
    let u = f32(idx % 2);
    let v = f32(idx / 2);
    let xy = vec2f(4*u -1, 4*v-1);

    var out: MipTriOut;
    out.pos = vec4f(xy, 1, 1);
    out.mip_num = inst;
    out.mip_size = textureDimensions(prev_mip).x / 2;
    out.buf_offset = 0u;

    var sz: u32 = out.mip_size;
    for (var i = inst; i > 0; i = i - 1) {
        sz = sz * 2;
        out.buf_offset += sz * sz;
    }
    return out;
}


@fragment fn first_range_mip(v: MipTriOut) -> @location(0) vec4f {
    let px = vec2u(floor(v.pos.xy));

    let nw = textureLoad(prev_mip, vec2u(2*px.x, 2*px.y), 0);
    let sw = textureLoad(prev_mip, vec2u(2*px.x, 2*px.y + 1), 0);
    let ne = textureLoad(prev_mip, vec2u(2*px.x + 1, 2*px.y), 0);
    let se = textureLoad(prev_mip, vec2u(2*px.x + 1, 2*px.y + 1), 0);

    let bottom = min(min(nw.x, sw.x), min(ne.x, se.x));
    let top = max(max(nw.x, sw.x), max(ne.x, se.x));

    let idx = v.buf_offset + v.mip_size * px.y + px.x;
    mip_buffer[idx] = pack2x16float(vec2f(bottom, top));

    return vec4f(bottom, top, 0.0, 1.0);
}

@fragment fn next_range_mip(v: MipTriOut) -> @location(0) vec4f {
    let px = vec2u(floor(v.pos.xy));

    let nw = textureLoad(prev_mip, vec2u(2*px.x, 2*px.y), 0);
    let sw = textureLoad(prev_mip, vec2u(2*px.x, 2*px.y + 1), 0);
    let ne = textureLoad(prev_mip, vec2u(2*px.x + 1, 2*px.y), 0);
    let se = textureLoad(prev_mip, vec2u(2*px.x + 1, 2*px.y + 1), 0);

    let bottom = min(min(nw.x, sw.x), min(ne.x, se.x));
    let top = max(max(nw.y, sw.y), max(ne.y, se.y));

    let idx = v.buf_offset + v.mip_size * px.y + px.x;
    mip_buffer[idx] = pack2x16float(vec2f(bottom, top));

    return vec4f(bottom, top, 0.0, 1.0);
}
