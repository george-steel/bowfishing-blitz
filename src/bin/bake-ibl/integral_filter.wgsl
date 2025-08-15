const TAU = 6.2831853072;
const PI  = 3.1415926535;

struct FQOut {
    @builtin(position) pos: vec4f,
    @location(0) xy: vec2f,
    @location(1) uv: vec2f,
}

@vertex fn fullscreen_quad(@builtin(vertex_index) vert_idx: u32) -> FQOut {
    let ij = vec2u(vert_idx % 2, vert_idx / 2);
    let xy = 2.0 * vec2f(ij) - 1.0;
    var out: FQOut;
    out.pos = vec4f(xy, 0.0, 1.0);
    out.xy = xy;
    out.uv = vec2f(0.5 + 0.5 * xy.x, 0.5 - 0.5 * xy.y);
    return out;
}

const SAMPLES: u32 = 8192;
const ACCUM_SIZE: u32 = 64;

struct DFGOut {
    @location(0) spec: vec4f,
    @location(1) trans: vec4f,
}

@fragment fn integrate_DFG_LUT(vert_in: FQOut) -> DFGOut {
    let rough = vert_in.uv.y;
    let alpha = rough * rough;
    let alpha2 = alpha * alpha;

    let nv = vert_in.uv.x*vert_in.uv.x;
    let nv2 = nv * nv;
    let v = vec3f(sqrt(1 - nv2), 0, nv);
    let shad_v = (-1 + sqrt(alpha2 * (1 - nv2) / nv2 + 1)) * 0.5;

    var total_spec = vec3f(0.0);
    var slice_spec = vec3f(0.0);
    var total_trans = vec2f(0.0);
    var slice_trans = vec2f(0.0);

    for (var i = 0u; i < SAMPLES; i += 1) {
        if i % ACCUM_SIZE == 0 {
            total_spec += slice_spec;
            slice_spec = vec3f(0.0);
            total_trans += slice_trans;
            slice_trans = vec2f(0.0);
        }

        let xi2 = 1 - (0.5 + f32(i)) / f32(SAMPLES);
        let xi1 = ldexp(f32(reverseBits(i)), -32);
        let phi = TAU * xi1;
        let rho = vec2f(cos(phi), sin(phi));

        // GGX ICDF
        let rz = normalize(vec2f(alpha * sqrt(xi2), sqrt(1 - xi2)));
        let h = vec3f(rz.x * rho, rz.y);
        let vh = dot(v, h);
        if vh <= 0 { continue; }

        let l = reflect(-v, h);
        let l_in = refract(-v, h, 0.75);
        let l_out = refract(-v, h, 1.33);

        let out_h = saturate(dot(normalize(l_out), -h));
        let in_h = saturate(dot(normalize(l_in), -h));
        let shlick = pow(saturate(1.0 - vh), 5.0);
        var shlick_out = 1.0;
        if l_out.z < 0.0 {
            let oh = dot(normalize(l_out), -h);
            shlick_out = pow(1.0 - out_h, 5.0);
        }

        if l.z > 0 {
            let nl2 = l.z * l.z;
            let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
            let shad = 1 / (1 + shad_l + shad_v);
            let gvis = shad * vh / (h.z * nv);
            let contrib = vec3f(1-shlick, shlick, mix(0.02, 1.0, shlick_out)) * gvis;
            slice_spec += contrib;
        }

        if l_in.z < 0 {
            let nl2 = l_in.z * l_in.z;
            let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
            let shad = 1 / (1 + shad_l + shad_v);
            let gvis = shad * vh / (h.z * nv);
            let contrib = mix(0.98, 0.0, shlick) * gvis;
            slice_trans.x += contrib;
        }

        if l_in.z < 0 {
            let nl2 = l_out.z * l_out.z;
            let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
            let shad = 1 / (1 + shad_l + shad_v);
            let gvis = shad * vh / (h.z * nv);
            let contrib = mix(0.98, 0.0, shlick_out) * gvis;
            slice_trans.y += contrib;
        }
    }

    total_spec += slice_spec;
    total_trans += slice_trans;

    var out: DFGOut;
    out.spec = vec4f(total_spec / f32(SAMPLES), 1.0);
    out.trans = vec4f(total_trans / f32(SAMPLES), 0.0, 1.0);

    return out;
}

override LIN_CORRECTION: f32 = 1.0;

struct DirsOut {
    @location(0) dir_ins: vec4f,
    @location(1) dir_metal: vec4f,
    @location(2) dir_in: vec4f,
    @location(3) dir_out: vec4f,
}

@fragment fn integrate_Dirs_LUT(vert_in: FQOut) -> DirsOut {
    let rough = vert_in.uv.y;
    let alpha = rough * rough;
    let alpha2 = alpha * alpha;

    let nv = vert_in.uv.x*vert_in.uv.x;
    let nv2 = nv * nv;
    let v = vec3f(sqrt(1 - nv2), 0, nv);
    let shad_v = (-1 + sqrt(alpha2 * (1 - nv2) / nv2 + 1)) * 0.5;
    let mx = acos(nv);

    var total_nvar = vec2f(0.0);
    var total_dist_ins = vec4f(0.0);
    var total_dist_metal = vec4f(0.0);
    var total_dist_in = vec4f(0.0);
    var total_dist_out = vec4f(0.0);

    var slice_nvar = vec2f(0.0);
    var slice_dist_ins = vec4f(0.0);
    var slice_dist_metal = vec4f(0.0);
    var slice_dist_in = vec4f(0.0);
    var slice_dist_out = vec4f(0.0);

    for (var i = 0u; i < SAMPLES; i += 1) {
        if i % ACCUM_SIZE == 0 {
            total_nvar += slice_nvar;
            total_dist_ins += slice_dist_ins;
            total_dist_metal += slice_dist_metal;
            total_dist_in += slice_dist_in;
            total_dist_out += slice_dist_out;
            
            slice_nvar = vec2f(0.0);
            slice_dist_ins = vec4f(0.0);
            slice_dist_metal = vec4f(0.0);
            slice_dist_in = vec4f(0.0);
            slice_dist_out = vec4f(0.0);
        }

        let xi2 = 1 - (0.5 + f32(i)) / f32(SAMPLES);
        let xi1 = ldexp(f32(reverseBits(i)), -32);
        let phi = TAU * xi1;
        let rho = vec2f(cos(phi), sin(phi));

        // GGX ICDF
        let rz = normalize(vec2f(alpha * sqrt(xi2), sqrt(1 - xi2)));
        let h = vec3f(rz.x * rho, rz.y);
        let vh = dot(v, h);
        if vh <= 0 { continue; }

        let l = reflect(-v, h);
        let l_in = refract(-v, h, 0.75);
        let l_out = refract(-v, h, 1.33);

        let out_h = saturate(dot(normalize(l_out), -h));
        let in_h = saturate(dot(normalize(l_in), -h));
        let shlick = pow(saturate(1.0 - vh), 5.0);
        var shlick_out = 1.0;
        if l_out.z < 0.0 {
            let oh = dot(normalize(l_out), -h);
            shlick_out = pow(1.0 - out_h, 5.0);
        }

        if l.z > 0 && vh > 0 {
            let nl2 = l.z * l.z;
            let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
            let shad = 1 / (1 + shad_l + shad_v);
            let gvis = shad * vh / (h.z * nv);
            let contrib = vec2f(1-shlick, shlick) * gvis;
            
            let tx = -atan2(l.x, l.z);
            let dtx = mx - tx;
            let ty = asin(l.y);
            let moments = vec3f(tx, dtx * dtx, ty * ty);
            let mcontrib = vec2f(0.04, 0.8) * contrib.x + contrib.y;
            slice_dist_ins += vec4f(moments, 1.0) * mcontrib.x;
            slice_dist_metal += vec4f(moments, 1.0) * mcontrib.y;
        }
        if l_in.z < 0 {
            let nl2 = l_in.z * l_in.z;
            let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
            let shad = 1 / (1 + shad_l + shad_v);
            let gvis = shad * vh / (h.z * nv);
            let contrib = mix(0.98, 0.0, shlick) * gvis;
            
            let tx = -atan2(l_in.x, -l_in.z);
            let dtx = mx - tx;
            let ty = asin(l_in.y);
            let moments = vec3f(tx, dtx * dtx, ty * ty);
            slice_dist_in += vec4f(moments, 1.0) * contrib;
        }
        if l_out.z < 0 {
            let nl2 = l_out.z * l_out.z;
            let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
            let shad = 1 / (1 + shad_l + shad_v);
            let gvis = shad * vh / (h.z * nv);
            let contrib = mix(0.98, 0.0, shlick_out) * gvis;
            
            let tx = -atan2(l_out.x, -l_out.z);
            let dtx = mx - tx;
            let ty = asin(l_out.y);
            let moments = vec3f(tx, dtx * dtx, ty * ty);
            slice_dist_out += vec4f(moments, 1.0) * contrib;
        }

        // get variance of roughness at V=N 
        let vnl = reflect(vec3f(0.0, 0.0, -1.0), h);
        if vnl.z > 0 {
            let tx = asin(vnl.x);
            slice_nvar += vec2f(tx * tx, 1.0) * vnl.z; 
        }
    }

    total_nvar += slice_nvar;
    total_dist_ins += slice_dist_ins;
    total_dist_metal += slice_dist_metal;
    total_dist_in += slice_dist_in;
    total_dist_out += slice_dist_out;

    let exp_dl = sqrt(total_nvar.x / total_nvar.y);
    let exc = exp_dl * LIN_CORRECTION;
    let inv_corr = 1.0 / LIN_CORRECTION;
    var out: DirsOut;
    let ins_moments = total_dist_ins.xyz / total_dist_ins.w;
    let ins_dx = mx - ins_moments.x;
    out.dir_ins = vec4f(inv_corr * ins_moments.x / mx, sqrt(ins_moments.y - ins_dx * ins_dx) / exc, sqrt(ins_moments.z) / exc, 1.0);
    let metal_moments = total_dist_metal.xyz / total_dist_metal.w;
    let metal_dx = mx - metal_moments.x;
    out.dir_metal = vec4f(inv_corr * metal_moments.x / mx, sqrt(metal_moments.y - metal_dx * metal_dx) / exc, sqrt(metal_moments.z) / exc, 1.0);

    let in_moments = total_dist_in.xyz / total_dist_in.w;
    let in_dx = mx - in_moments.x;
    out.dir_in = vec4f(inv_corr * in_moments.x / mx, sqrt(in_moments.y - in_dx * in_dx) / exc, sqrt(in_moments.z) / exc, 1.0);
    let out_moments = total_dist_out.xyz / total_dist_out.w;
    let out_dx = mx - out_moments.x;
    out.dir_out = vec4f(inv_corr * out_moments.x / mx, sqrt(out_moments.y - out_dx * out_dx) / exc, sqrt(out_moments.z) / exc, 1.0);

    return out;
}

