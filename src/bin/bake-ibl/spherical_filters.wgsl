const TAU = 6.2831853072;
const PI  = 3.1415926535;

const HEIGHT: u32 = 512;
const WIDTH: u32 = HEIGHT * 2;
const LEVELS: u32 = 8;

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
var<workgroup> fft_buf: array<comp_colors, WIDTH>;
fn do_fft(local_x: u32, inverse: bool) {
    {
        // bit-reverse shuffle step
        let i = 2 * local_x;
        let j = i + 1;
        let x = fft_buf[i];
        let y = fft_buf[j];
        workgroupBarrier();
        let bits = firstLeadingBit(WIDTH);
        let shift = 32 - bits;
        let i2 = extractBits(reverseBits(i), shift, bits);
        let j2 = extractBits(reverseBits(j), shift, bits);
        fft_buf[i2] = x;
        fft_buf[j2] = y;
        workgroupBarrier();
    }

    let inverse_fac: i32 = select(-1, 1, inverse);
    for (var group_size: u32 = 2; group_size <= WIDTH; group_size *= 2) {
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

@group(0) @binding(0) var raw_tex_in: texture_2d<f32>;
@group(0) @binding(1) var tex_in_samp: sampler;
@group(0) @binding(2) var<storage, read_write> raw_spectra_out: array<vec3f>;

fn get_tex_in(col: u32, row: u32) -> vec3f {
    let u = (f32(col) + 0.5) / f32(WIDTH);
    let v = (f32(row) + 0.5) / f32(HEIGHT);
    let tex = textureSampleLevel(raw_tex_in, tex_in_samp, vec2f(u, v), 0.0).xyz;
    return min(tex, vec3f(4.0));
    //return tex;
}

@compute @workgroup_size(WIDTH / 2, 1, 1) fn horiz_fht_tex_to_buf(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let row = 2 * wg_id.x;
    let col = 2 * local_id.x;

    fft_buf[col] = mat2x3f(get_tex_in(col, row), get_tex_in(col, row+1));
    fft_buf[col+1] = mat2x3f(get_tex_in(col+1, row), get_tex_in(col+1, row+1));
    
    workgroupBarrier();
    do_fft(local_id.x, false);
    
    // extract two Hartley transforms from FFT
    let i = local_id.x;
    let j = select(WIDTH - i, WIDTH / 2, i == 0);
    let fi = fft_buf[i];
    let fj = fft_buf[j];
    var hi = fi;
    var hj = fj;
    if i != 0 {
        hi = 0.5 * (comp_smul(vec2f(1, 1), fi) + comp_smul(vec2f(1, -1), fj));
        hj = 0.5 * (comp_smul(vec2f(1, 1), fj) + comp_smul(vec2f(1, -1), fi));
    }
    
    let norm: f32 = 1.0 / f32(WIDTH);
    raw_spectra_out[i * HEIGHT + row] = hi[0] * norm;
    raw_spectra_out[i * HEIGHT + row + 1] = hi[1] * norm;
    raw_spectra_out[j * HEIGHT + row] = hj[0] * norm;
    raw_spectra_out[j * HEIGHT + row + 1] = hj[1] * norm;
}

@group(0) @binding(0) var<storage, read> blurred_spectra_in: array<vec3f>;
@group(0) @binding(1) var tex_out: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(WIDTH / 2, 1, 1) fn horiz_fht_buf_to_tex(@builtin(workgroup_id) wg_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let row = 2 * wg_id.x;
    let col = 2 * local_id.x;
    let level = wg_id.y;
    let colptr = level * 2 * HEIGHT * HEIGHT + col * HEIGHT;

    fft_buf[col] = mat2x3f(blurred_spectra_in[colptr + row], blurred_spectra_in[colptr + row + 1]);
    fft_buf[col+1] = mat2x3f(blurred_spectra_in[colptr + HEIGHT + row], blurred_spectra_in[colptr + HEIGHT + row + 1]);
    
    workgroupBarrier();
    do_fft(local_id.x, false);
    
    // extract two Hartley transforms from FFT
    let i = local_id.x;
    let j = select(WIDTH - i, WIDTH / 2, i == 0);
    let fi = fft_buf[i];
    let fj = fft_buf[j];
    var hi = fi;
    var hj = fj;
    if i != 0 {
        hi = 0.5 * (comp_smul(vec2f(1, 1), fi) + comp_smul(vec2f(1, -1), fj));
        hj = 0.5 * (comp_smul(vec2f(1, 1), fj) + comp_smul(vec2f(1, -1), fi));
    }
    
    textureStore(tex_out, vec2u(i, row), level, vec4f(hi[0], 1));
    textureStore(tex_out, vec2u(i, row+1), level, vec4f(hi[1], 1));
    textureStore(tex_out, vec2u(j, row), level, vec4f(hj[0] , 1));
    textureStore(tex_out, vec2u(j, row+1), level, vec4f(hj[1], 1));
}

////////////////////////////////////////////////////////////////////////////////

var<workgroup> sg_counter: atomic<u32>;
var<workgroup> sg_counter_result: u32;

fn enumerate_subgroups(sg_inv: u32) -> u32 {
    var sgid = 0u;
    if sg_inv == 0 {
        sgid = atomicAdd(&sg_counter, 1u);
    }
    return subgroupBroadcastFirst(sgid);
}

const MAX_SG = HEIGHT / 32;
var<workgroup> sum_buffer: array<vec3f, MAX_SG>;
var<workgroup> sum_result: vec3f;

fn workgroupAddVec3f(sg_id: u32, sg_inv: u32, num_subgroups: u32, val: vec3f) -> vec3f {
    let warp_sum = subgroupAdd(val);
    if sg_inv == 0 {
        sum_buffer[sg_id] = warp_sum;
    }
    workgroupBarrier();
    var subtotal = vec3f(0);
    if sg_id == 0 {
        if sg_inv < num_subgroups {
            subtotal = sum_buffer[sg_inv];
        }
        let total = subgroupAdd(subtotal);
        if sg_inv == 0 {
            sum_result = total;
        }
    }
    return workgroupUniformLoad(&sum_result);
}
fn workgroupAddF32(sg_id: u32, sg_inv: u32, num_subgroups: u32, val: f32) -> f32 {
    let warp_sum = subgroupAdd(val);
    if sg_inv == 0 {
        sum_buffer[sg_id] = vec3f(warp_sum, 0, 0);
    }
    workgroupBarrier();
    var subtotal = 0.0;
    if sg_id == 0 {
        if sg_inv < num_subgroups {
            subtotal = sum_buffer[sg_inv].x;
        }
        let total = subgroupAdd(subtotal);
        if sg_inv == 0 {
            sum_result = vec3f(total, 0, 0);
        }
    }
    return workgroupUniformLoad(&sum_result).x;
}


fn workgroupMulF32(sg_id: u32, sg_inv: u32, num_subgroups: u32, val: f32) -> f32 {
    let warp_sum = subgroupMul(val);
    if sg_inv == 0 {
        sum_buffer[sg_id] = vec3f(warp_sum, 0, 0);
    }
    workgroupBarrier();
    var subtotal = 1.0;
    if sg_id == 0 {
        if sg_inv < num_subgroups {
            subtotal = sum_buffer[sg_inv].x;
        }
        let total = subgroupMul(subtotal);
        if sg_inv == 0 {
            sum_result = vec3f(total, 0, 0);
        }
    }
    return workgroupUniformLoad(&sum_result).x;
}

@group(0) @binding(0) var<storage, read> spectrum_buf_in: array<f32>;
@group(0) @binding(1) var<storage, read> raw_spectra_in: array<vec3f>;
@group(0) @binding(2) var<storage, read_write> blurred_spectra_out: array<vec3f>;
const BANDWIDTH: u32 = HEIGHT;

var<workgroup> twiddles: array<vec2f, BANDWIDTH>;
var<private> px_out: array<vec3f, 16>;

// Blur th DHT slices using a spherical harmonic decomposition and the convolution theorem
// Adapted from https://people.csail.mit.edu/ythomas/unpublished/6869.pdf
@compute @workgroup_size(HEIGHT, 1, 1) fn blur_spectra(
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_inv: u32
) {
    let sg_id = enumerate_subgroups(sg_inv);
    workgroupBarrier();
    if sg_id == 0 && sg_inv == 0 {
        sg_counter_result = atomicLoad(&sg_counter);
    }
    let num_subgroups: u32 = workgroupUniformLoad(&sg_counter_result);

    let n: u32 = (wg_id.x + 1) / 2;
    let col = select(n, WIDTH - n, (wg_id.x & 1u) != 0);
    let row = local_id.x;
    let buf_idx = col * HEIGHT + row;
    let px_in = raw_spectra_in[buf_idx];

    if n < BANDWIDTH {
        let theta = PI * (f32(row) + 0.5) / f32(HEIGHT);
        let z = cos(theta);
        let r = sin(theta);
        let dw = PI * r / f32(HEIGHT) / 2;

        let p_init_fac = select(1.0, sqrt(f32(n+row) / f32(row) / 4), row > 0 && row <= n);
        let p_init = workgroupMulF32(sg_id, sg_inv, num_subgroups, p_init_fac);

        // calculate coefficients for Legendre polynomial recurrence
        // https://doi.org/10.1016/S0377-0427(03)00546-6
        if row < BANDWIDTH {
            let nf = f32(n);
            let kf = f32(row);
            let denom = (kf - nf + 1) * (kf + nf + 1);
            let twv = (2 * kf + 1) / sqrt(denom);
            let tww = sqrt((kf + nf) * (kf - nf) / denom);
            twiddles[row] = vec2f(twv, tww);
        }
        workgroupBarrier();

        var p = p_init * pow(r, f32(n));
        var p1 = 0.0;
        var p2 = 0.0;

        for (var k = n; k < BANDWIDTH; k += 1) {
            let cap = (p * dw * (2 * f32(k) + 1)) * px_in;
            let cleancap = max(cap, vec3f(0)) + min(cap, vec3f(0));
            let coeff = workgroupAddVec3f(sg_id, sg_inv, num_subgroups, cleancap);

            for (var lv = 0u; lv < LEVELS; lv += 1) {
                px_out[lv] += p * coeff * spectrum_buf_in[k * LEVELS + lv];
            }

            let vw = twiddles[k];
            p2 = p1;
            p1 = p;
            p = vw.x * z * p1 - vw.y * p2; 
        }
    }

    workgroupBarrier();
    for (var lv = 0u; lv < LEVELS; lv += 1) {
        let colptr = lv * 2 * HEIGHT * HEIGHT + col * HEIGHT;
        blurred_spectra_out[colptr + row] = px_out[lv];
    }

}

@group(0) @binding(0) var<storage, read_write> spectrum_buf_out: array<f32>;
var<workgroup> spectrum: array<f32, BANDWIDTH>;
const CONV_FAC = 4.0 * pow(PI, 1.5);

@compute @workgroup_size(BANDWIDTH, 1, 1) fn get_kernel_spectra(
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_inv: u32
) {
    let sg_id = enumerate_subgroups(sg_inv);
    workgroupBarrier();
    if sg_id == 0 && sg_inv == 0 {
        sg_counter_result = atomicLoad(&sg_counter);
    }
    let num_subgroups: u32 = workgroupUniformLoad(&sg_counter_result);

    let row = local_id.x;


    let theta = PI * (f32(row) + 0.5) / f32(HEIGHT);
    let z = cos(theta);
    let r = sin(theta);
    let dw = PI * r / f32(HEIGHT) / 2;

    let level = wg_id.x;
    let alpha = ldexp(1.0, -i32(level)); // roughness^2, approx lobe radius in rad at small angles
    let z_clip = max(z, 0.0) / PI;
    let a2 = alpha * alpha;
    let nh = cos(theta / 2);
    let nh2 = nh * nh;
    let denom = nh2 * (a2 - 1.0) + 1.0;
    let kernel_px = select(0.0, z_clip * a2 / (denom * denom), z > 0);

    // calculate coefficients for Legendre polynomial recurrence
    // https://doi.org/10.1016/S0377-0427(03)00546-6
    if row < BANDWIDTH {
        let kf = f32(row);
        let twv = (2 * kf + 1) / (kf + 1);
        let tww = kf / (kf + 1);
        twiddles[row] = vec2f(twv, tww);
    }
    workgroupBarrier();

    var p = 1.0;
    var p1 = 0.0;
    var p2 = 0.0;

    for (var k: u32 = 0; k < BANDWIDTH; k += 1) {
        let cap = (p * dw) * kernel_px;
        let cleancap = max(cap, 0.0) + min(cap, 0.0); // remove NaNs
        let coeff = workgroupAddF32(sg_id, sg_inv, num_subgroups, cleancap);

        if row == 0 {
            spectrum[k] = coeff;
        }

        let vw = twiddles[k];
        p2 = p1;
        p1 = p;
        p = vw.x * z * p1 - vw.y * p2; 
    }

    workgroupBarrier();
    if row < BANDWIDTH {
        spectrum_buf_out[row * LEVELS + level] = spectrum[row] / spectrum[0];
    }

}

////////////////////////////////////////////////////////////////////////////////

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

@group(0) @binding(0) var blurred_tex_in: texture_2d_array<f32>;

@fragment fn display_tex(v: FQOut) -> @location(0) vec4f {
    let raw = textureSampleLevel(blurred_tex_in, tex_in_samp, v.uv, 4, 0.0).xyz;
    let col = raw / 2;
    return vec4f(col, 1);
}
