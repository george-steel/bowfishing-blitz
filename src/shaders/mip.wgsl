@group(0) @binding(0) var prev_mip: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> small_mip_buffer: array<u32>;

struct BakeMipTriOut {
    @builtin(position) pos: vec4f,
    @location(0) mip_num: u32,
    @location(1) mip_size: u32,
    @location(2) buf_offset: u32,
}

@vertex fn bake_mip_fullscreen_tri(@builtin(vertex_index) idx: u32, @builtin(instance_index) inst: u32) -> BakeMipTriOut {
    let u = f32(idx % 2);
    let v = f32(idx / 2);
    let xy = vec2f(4*u -1, 4*v-1);

    var out: BakeMipTriOut;
    out.pos = vec4f(xy, 1, 1);
    out.mip_num = inst;
    out.mip_size = textureDimensions(prev_mip).x / 2;
    out.buf_offset = 0u;

    for (var sz: u32 = out.mip_size; sz < 32; sz = sz * 2) {
        out.buf_offset += sz * sz;
    }
    return out;
}


@fragment fn bake_first_range_mip(v: BakeMipTriOut) -> @location(0) vec4f {
    let px = vec2u(floor(v.pos.xy));

    let nw = textureLoad(prev_mip, vec2u(2*px.x, 2*px.y), 0);
    let sw = textureLoad(prev_mip, vec2u(2*px.x, 2*px.y + 1), 0);
    let ne = textureLoad(prev_mip, vec2u(2*px.x + 1, 2*px.y), 0);
    let se = textureLoad(prev_mip, vec2u(2*px.x + 1, 2*px.y + 1), 0);

    let bottom = min(min(nw.x, sw.x), min(ne.x, se.x));
    let top = max(max(nw.x, sw.x), max(ne.x, se.x));

    if v.mip_size <= 32 {
        let idx = v.buf_offset + v.mip_size * px.y + px.x;
        small_mip_buffer[idx] = pack2x16float(vec2f(bottom, top));
    }

    return vec4f(bottom, top, 0.0, 1.0);
}

@fragment fn bake_next_range_mip(v: BakeMipTriOut) -> @location(0) vec4f {
    let px = vec2u(floor(v.pos.xy));

    let nw = textureLoad(prev_mip, vec2u(2*px.x, 2*px.y), 0);
    let sw = textureLoad(prev_mip, vec2u(2*px.x, 2*px.y + 1), 0);
    let ne = textureLoad(prev_mip, vec2u(2*px.x + 1, 2*px.y), 0);
    let se = textureLoad(prev_mip, vec2u(2*px.x + 1, 2*px.y + 1), 0);

    let bottom = min(min(nw.x, sw.x), min(ne.x, se.x));
    let top = max(max(nw.y, sw.y), max(ne.y, se.y));

    if v.mip_size <= 32 {
        let idx = v.buf_offset + v.mip_size * px.y + px.x;
        small_mip_buffer[idx] = pack2x16float(vec2f(bottom, top));
    }

    return vec4f(bottom, top, 0.0, 1.0);
}
