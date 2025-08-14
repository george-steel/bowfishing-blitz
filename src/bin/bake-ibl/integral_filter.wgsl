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

const SAMPLES: u32 = 4096;
const ACCUM_SIZE: u32 = 64;



@fragment fn integrate_DFG_LUT(vert_in: FQOut) -> @location(0) vec4f{
    let rough = vert_in.uv.y;
    let alpha = rough * rough;

    let nv = vert_in.uv.x*vert_in.uv.x;
    let nv2 = nv * nv;
    let v = vec3f(sqrt(1 - nv2), 0, nv);

    var total_spec = vec2f(0.0);
    var slice_spec = vec2f(0.0);

    for (var i = 0u; i < SAMPLES; i += 1) {
        if i % ACCUM_SIZE == 0 {
            total_spec += slice_spec;
            slice_spec = vec2f(0.0);
        }

        let xi2 = 1 - (0.5 + f32(i)) / f32(SAMPLES);
        let xi1 = ldexp(f32(reverseBits(i)), -32);
        let phi = TAU * xi1;
        let rho = vec2f(cos(phi), sin(phi));

        // GGX ICDF
        let rz = normalize(vec2f(alpha * sqrt(xi2), sqrt(1 - xi2)));
        let h = vec3f(rz.x * rho, rz.y);

        let l = reflect(-v, h);
        let vh = dot(v, h);
        if l.z > 0 && vh > 0 {
            let shlick = pow(saturate(1.0 - vh), 5.0);
            let alpha2 = alpha * alpha;
            let nl2 = l.z * l.z;
            let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
            let shad_v = (-1 + sqrt(alpha2 * (1 - nv2) / nv2 + 1)) * 0.5;
            let shad = 1 / (1 + shad_l + shad_v);
            let gvis = shad * vh / (h.z * nv);
            let contrib = vec2f(1-shlick, shlick) * gvis;
            slice_spec += contrib;
        }
    }

    total_spec += slice_spec;

    return vec4f(total_spec / f32(SAMPLES), 0.0, 1.0);
}

struct DirsOut {
    @location(0) dir_ins: vec4f,
    @location(1) dir_metal: vec4f,
}

@fragment fn integrate_Dirs_LUT(vert_in: FQOut) -> DirsOut {
    let rough = vert_in.uv.y;
    let alpha = rough * rough;

    let nv = vert_in.uv.x*vert_in.uv.x;
    let nv2 = nv * nv;
    let v = vec3f(sqrt(1 - nv2), 0, nv);
    let mx = acos(nv);

    var total_exp = vec2f(0.0);
    var total_m_weight = vec2f(0.0);
    var total_dist_ins = vec3f(0.0);
    var total_dist_metal = vec3f(0.0);

    var slice_exp = vec2f(0.0);
    var slice_m_weight = vec2f(0.0);
    var slice_dist_ins = vec3f(0.0);
    var slice_dist_metal = vec3f(0.0);

    for (var i = 0u; i < SAMPLES; i += 1) {
        if i % ACCUM_SIZE == 0 {
            total_exp += slice_exp;
            total_m_weight += slice_m_weight;
            total_dist_ins += slice_dist_ins;
            total_dist_metal += slice_dist_metal;
            
            slice_exp = vec2f(0.0);
            slice_m_weight = vec2f(0.0);
            slice_dist_ins = vec3f(0.0);
            slice_dist_metal = vec3f(0.0);
        }

        let xi2 = 1 - (0.5 + f32(i)) / f32(SAMPLES);
        let xi1 = ldexp(f32(reverseBits(i)), -32);
        let phi = TAU * xi1;
        let rho = vec2f(cos(phi), sin(phi));

        // GGX ICDF
        let rz = normalize(vec2f(alpha * sqrt(xi2), sqrt(1 - xi2)));
        let h = vec3f(rz.x * rho, rz.y);

        let l = reflect(-v, h);
        let vh = dot(v, h);
        if l.z > 0 && vh > 0 {
            let shlick = pow(saturate(1.0 - vh), 5.0);
            let alpha2 = alpha * alpha;
            let nl2 = l.z * l.z;
            let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
            let shad_v = (-1 + sqrt(alpha2 * (1 - nv2) / nv2 + 1)) * 0.5;
            let shad = 1 / (1 + shad_l + shad_v);
            let gvis = shad * vh / (h.z * nv);
            let contrib = vec2f(1-shlick, shlick) * gvis;
            
            let tx = -atan2(l.x, l.z);
            let dtx = mx - tx;
            let ty = asin(l.y);
            let moments = vec3f(tx, dtx * dtx, ty * ty);
            let mcontrib = vec2f(0.04, 0.8) * contrib.x + contrib.y;
            slice_m_weight += mcontrib;
            slice_dist_ins += moments * mcontrib.x;
            slice_dist_metal += moments * mcontrib.y;
        }

        let vnl = reflect(vec3f(0.0, 0.0, -1.0), h);
        if vnl.z > 0 {
            let tx = asin(vnl.x);
            slice_exp += vec2f(tx * tx, 1.0) * vnl.z; 
        }
    }

    total_exp += slice_exp;
    total_m_weight += slice_m_weight;
    total_dist_ins += slice_dist_ins;
    total_dist_metal += slice_dist_metal;

    let exp_dl = sqrt(total_exp.x / total_exp.y);
    let ex2 = exp_dl * 8;
    var out: DirsOut;
    let ins_moments = total_dist_ins / total_m_weight.x;
    let ins_dx = mx - ins_moments.x;
    out.dir_ins = vec4f(0.5 * ins_moments.x / mx, sqrt(ins_moments.y - ins_dx * ins_dx) / ex2, sqrt(ins_moments.z) / ex2, 1.0);
    let metal_moments = total_dist_metal / total_m_weight.y;
    let metal_dx = mx - metal_moments.x;
    out.dir_metal = vec4f(0.5 * metal_moments.x / mx, sqrt(metal_moments.y - metal_dx * metal_dx) / ex2, sqrt(metal_moments.z) / ex2, 1.0);

    return out;
}

