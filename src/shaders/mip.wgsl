@group(0) @binding(0) var prev_mip: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> small_mip_buffer: array<u32>;

// From WGPU documentation
struct MipVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
};

@vertex fn mip_vert(@builtin(vertex_index) vertex_index: u32) -> MipVertexOutput {
    var result: MipVertexOutput;
    let x = i32(vertex_index) / 2;
    let y = i32(vertex_index) & 1;
    let tc = vec2<f32>(
        f32(x) * 2.0,
        f32(y) * 2.0
    );
    result.position = vec4<f32>(
        tc.x * 2.0 - 1.0,
        1.0 - tc.y * 2.0,
        0.0, 1.0
    );
    result.tex_coords = tc;
    return result;
}

@group(0) @binding(0) var r_color: texture_2d<f32>;
@group(0) @binding(1) var r_sampler: sampler;

@fragment fn mip_frag(vertex: MipVertexOutput) -> @location(0) vec4<f32> {
    return textureSample(r_color, r_sampler, vertex.tex_coords);
}

struct BakeMipTriOut {
    @builtin(position) pos: vec4f,
    @location(0) mip_num: u32,
    @location(1) mip_size: u32,
    @location(2) buf_offset: u32,
}
