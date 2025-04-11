const TAU = 6.2831853072;

override FFTSIZE: u32 = 1024;

alias comp_colors = mat2x3f;

fn comp_smul(s: vec2f, v: comp_colors) -> comp_colors {
    return s.x * v + s.y * mat2x3f(-v[1], v[0]);
}

fn root_unity(num: i32, denom: u32) -> vec2f {
    var revs = fract(f32(num) / f32(denom));
    let theta = TAU * revs;
    return vec2f(cos(theta), sin(theta));
}

// complex color fft. Can process two lines of real data at once with post-transform
var<workgroup> fft_buf: array<comp_colors, FFTSIZE>;
fn do_fft(local_x: u32, inverse: bool) {
    {
        // bit-reverse shuffle step
        let i = 2 * local_x;
        let j = i + 1;
        let x = fft_buf[i];
        let y = fft_buf[j];
        workgroupBarrier();
        let bits = firstLeadingBit(FFTSIZE);
        let shift = 32 - bits;
        let i2 = extractBits(reverseBits(i), shift, bits);
        let j2 = extractBits(reverseBits(j), shift, bits);
        fft_buf[i2] = x;
        fft_buf[j2] = y;
        workgroupBarrier();
    }

    let inverse_fac: i32 = select(-1, 1, inverse);
    for (var group_size: u32 = 2; group_size <= FFTSIZE; group_size *= 2) {
        let stride = group_size / 2;
        let t = local_x % stride;
        let i = 2 * local_x - t;
        let j = i + stride;

        let p = fft_buf[i];
        let qq = fft_buf[j];
        let twiddle = root_unity(i32(t) * inverse_fac, group_size);
        let q = comp_smul(twiddle, qq);
        fft_buf[i] = p + q;
        fft_buf[j] = p - q;

        workgroupBarrier();
    }
}

override IMG_HEIGHT = 256;

@group(0) @binding(0) var tex_in: texture_2d<f32>;
@group(0) @binding(1) var tex_in_samp: sampler;
@group(0) @binding(2) var tex_out: texture_storage_2d<rgba32float, write>;

fn get_tex_in(col: u32, row: u32) -> vec3f {
    let u = (f32(col) + 0.5) / f32(FFTSIZE);
    let v = f32(2 * row + 1) / f32(FFTSIZE);
    return textureSampleLevel(tex_in, tex_in_samp, vec2f(u, v), 0.0).xyz;
}


@compute @workgroup_size(FFTSIZE / 2, 1, 1) fn horiz_fht(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let row = 2 * wg_id.x;
    let col = 2 * local_id.x;

    fft_buf[col] = mat2x3f(get_tex_in(col, row), get_tex_in(col, row+1));
    fft_buf[col+1] = mat2x3f(get_tex_in(col+1, row), get_tex_in(col+1, row+1));
    
    workgroupBarrier();
    do_fft(local_id.x, false);
    
    // extract two Hartley transforms from FFT
    let i = local_id.x;
    let j = select(FFTSIZE - i, FFTSIZE / 2, i == 0);
    let fi = fft_buf[i];
    let fj = fft_buf[j];
    var hi = fi;
    var hj = fj;
    if i != 0 {
        hi = 0.5 * (comp_smul(vec2f(1, 1), fi) + comp_smul(vec2f(1, -1), fj));
        hj = 0.5 * (comp_smul(vec2f(1, 1), fj) + comp_smul(vec2f(1, -1), fi));
    }
    


    textureStore(tex_out, vec2u(i, row), vec4f(hi[0], 1));
    textureStore(tex_out, vec2u(i, row+1), vec4f(hi[1], 1));
    textureStore(tex_out, vec2u(j, row), vec4f(hj[0], 1));
    textureStore(tex_out, vec2u(j, row+1), vec4f(hj[1], 1));
}

struct FQOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex fn fullscreen_quad(@builtin(vertex_index) vert_idx: u32) -> FQOut {
    let ij = vec2u(vert_idx % 2, vert_idx / 2);
    let xy = 2.0 * vec2f(ij) - 1.0;
    var out: FQOut;
    out.pos = vec4f(xy, 0.0, 1.0);
    out.uv = vec2f(vec2u(ij.x, 1-ij.y));
    return out;
}

@fragment fn display_tex(v: FQOut) -> @location(0) vec4f {
    let raw = textureSampleLevel(tex_in, tex_in_samp, v.uv, 0.0).xyz;
    let col = abs(raw) / 512;
    return vec4f(col, 1);
}
