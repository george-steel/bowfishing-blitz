const TAU = 6.2831853072;
const PI  = 3.1415926535;
const X = vec3f(1.0, 0.0, 0.0);
const Y = vec3f(0.0, 1.0, 0.0);
const Z = vec3f(0.0, 0.0, 1.0);

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

const MAX_CUBE_ANISO: u32 = 12;
fn sampleCubeAniso(radiance: texture_cube<f32>, samp: sampler, dir: vec3f, major_axis: vec3f, alphas: vec2f) -> vec4f {
    let total_mips = textureNumLevels(radiance);

    var hi_axis = major_axis;
    var alpha_hi = alphas.x;
    var alpha_lo = alphas.y;
    if alphas.y > alphas.x {
        alpha_hi = alphas.y;
        alpha_lo = alphas.x;
        hi_axis = cross(major_axis, dir);
    }
    alpha_hi = max(alpha_hi, 0.002);
    alpha_lo = max(max(alpha_lo, 0.002), alpha_hi / f32(MAX_CUBE_ANISO));

    let eccentricity = alpha_hi / alpha_lo;
    let n_samples = 2 * u32(round(eccentricity)) + 1;

    let mip = max(0.0, f32(total_mips) - 1 + log2(alpha_lo));

    let ndir = normalize(dir);
    let adir = normalize(hi_axis - dot(hi_axis, ndir) * ndir);

    let samp_rad = 4 * alpha_hi * (eccentricity - 1);

    var total_tex = vec4f(0);
    var total_weight = 0u;

    for (var i = 0u; i < n_samples; i += 1) {
        let t = 2 * f32(i) / f32(n_samples - 1) - 1;
        let sdir = ndir + samp_rad * t * adir;

        let weight = min(i + 1, n_samples - i);
        total_weight += weight;

        let stex = textureSampleLevel(radiance, samp, sdir, mip);
        total_tex += f32(weight) * stex;
    }

    return total_tex / f32(total_weight);
}

// adapted from Learn OpenGL
fn dist_GGX(nh: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let nhc = max(0.0, nh);
    let nh2 = nhc * nhc;
    let denom = nh2 * (a2 - 1.0) + 1.0;
    return a2 / (denom * denom);
}

override U_AT_PLUSX: f32 = 0.5;
override WATER_STRENGTH: f32 = 0.6;
override WATER_ROUGH: f32 = 0.2;
override MAX_BRIGHT_PX: f32 = 3.0;

const SUN_COLOR = 2 * vec3f(0.98, 0.99, 0.90);
const SUN_DIR = vec3f(0.548, -0.380, 0.745);

@group(0) @binding(0) var trilinear: sampler;
@group(0) @binding(1) var sky_in: texture_2d<f32>;
@group(0) @binding(2) var sky_filt: texture_cube<f32>;
@group(0) @binding(3) var dfg_refl: texture_2d<f32>;
@group(0) @binding(4) var dfg_trans: texture_2d<f32>;
@group(0) @binding(5) var dfg_dirs: texture_2d_array<f32>;
@group(0) @binding(6) var lut_sampler: sampler;

override LIN_CORRECTION: f32 = 1.0;

@fragment fn above_water_equi(v: FQOut) -> @location(0) vec4f {
    let max_cube_mip = f32(textureNumLevels(sky_filt) - 1);

    var px_in = textureSampleLevel(sky_in, trilinear, v.uv, 0.0);

    let bright = dot(px_in, vec4f(1, 1, 1, 0));
    if bright > MAX_BRIGHT_PX {
        px_in *= vec4f(vec3f(MAX_BRIGHT_PX / bright), 1);
    }

    if v.xy.y >= 0 {
        return px_in;
    }

    let theta = v.xy.y * PI / 2;
    let phi = (- U_AT_PLUSX - v.uv.x ) * TAU;
    let z = sin(theta);
    let rho = cos(theta);
    let dir = vec3f(rho * cos(phi), rho * sin(phi), z);
    let mirror_dir = vec3f(dir.xy, -dir.z);
    let alpha = WATER_ROUGH * WATER_ROUGH;

    let dfg_uv = vec2f(sqrt(abs(dir.z)), WATER_ROUGH);
    let dfg_vals = textureSampleLevel(dfg_refl, lut_sampler, dfg_uv, 0.0);
    let trans_vals = textureSampleLevel(dfg_trans, lut_sampler, dfg_uv, 0.0);
    let dirs_refl = textureSampleLevel(dfg_dirs, lut_sampler, dfg_uv, 0, 0.0);
    let dirs_trans = textureSampleLevel(dfg_dirs, lut_sampler, dfg_uv, 2, 0.0);

    let refl_dir = mix(Z, mirror_dir, dirs_refl.x * LIN_CORRECTION);
    let refl_alphas = alpha * dirs_refl.yz * LIN_CORRECTION;
    let refl_fac = 0.02 * dfg_vals.x + dfg_vals.y;
    let refl_col = sampleCubeAniso(sky_filt, trilinear, refl_dir, Z, refl_alphas).xyz;

    let trans_dir = mix(-Z, dir, dirs_trans.x * LIN_CORRECTION);
    let trans_alphas = alpha * dirs_trans.yz * LIN_CORRECTION;
    let trans_fac = trans_vals.x;
    let trans_col = sampleCubeAniso(sky_filt, trilinear, trans_dir, Z, trans_alphas).xyz;

    let l = normalize(SUN_DIR);
    let h = normalize(l - dir);
    let vh = saturate(dot(-dir, h));
    let ks = mix(0.02, 1.0, pow(1 - vh, 5.0));
    let ndf = dist_GGX(h.z, alpha);
    let nv2 = z * z;
    let nl2 = l.z * l.z;
    let alpha2 = alpha * alpha;
    let shad_v = (-1 + sqrt(alpha2 * (1 - nv2) / nv2 + 1)) * 0.5;
    let shad_l = (-1 + sqrt(alpha2 * (1 - nl2) / nl2 + 1)) * 0.5;
    let shad = 1 / (1 + shad_l + shad_v);
    let spec_fac = ks * ndf * shad / ((4.0 * (-dir.z)) + 0.0001);

    //return vec4f(spec_fac);
    //return refl_fac * refl_col;
    //return trans_fac * trans_col;
    let water_col = refl_fac * refl_col + trans_fac * trans_col + spec_fac * SUN_COLOR;
    return vec4f(water_col, 1.0);
}

const LIMIT_COLOR = vec4f(0.03, 0.05, 0.1, 1.0);

@fragment fn below_water_equi(v: FQOut) -> @location(0) vec4f {
    let max_cube_mip = f32(textureNumLevels(sky_filt) - 1);

    var px_in = textureSampleLevel(sky_in, trilinear, v.uv, 0.0);
    let theta = v.xy.y * PI / 2;
    let falloff = 0.9 * exp(0.3 * (-1 / abs(sin(theta)) + 1)) + 0.1;
    //let falloff = abs(sin(theta));

    let bright = dot(px_in, vec4f(1, 1, 1, 0));
    if bright > MAX_BRIGHT_PX {
        px_in *= vec4f(vec3f(MAX_BRIGHT_PX / bright), 1);
    }

    if v.xy.y <= 0 {
        return mix(LIMIT_COLOR, px_in, falloff);
    }

    let phi = (- U_AT_PLUSX - v.uv.x ) * TAU;
    let z = sin(theta);
    let rho = cos(theta);
    let dir = vec3f(rho * cos(phi), rho * sin(phi), z);
    let mirror_dir = vec3f(dir.xy, -dir.z);
    let alpha = WATER_ROUGH * WATER_ROUGH;

    let dfg_uv = vec2f(sqrt(abs(dir.z)), WATER_ROUGH);
    let trans_vals = textureSampleLevel(dfg_trans, lut_sampler, dfg_uv, 0.0);
    let dirs_refl = textureSampleLevel(dfg_dirs, lut_sampler, dfg_uv, 4, 0.0);
    let dirs_trans = textureSampleLevel(dfg_dirs, lut_sampler, dfg_uv, 3, 0.0);

    let refl_dir = mix(-Z, mirror_dir, dirs_refl.x * LIN_CORRECTION);
    let refl_alphas = alpha * min(vec2f(1.0), dirs_refl.yz * LIN_CORRECTION);
    let refl_fac = trans_vals.z;
    let refl_col = sampleCubeAniso(sky_filt, trilinear, refl_dir, Z, refl_alphas).xyz;

    let trans_dir = mix(Z, dir, dirs_trans.x * LIN_CORRECTION);
    let trans_alphas = alpha * dirs_trans.yz * LIN_CORRECTION;
    let trans_fac = trans_vals.y;
    let trans_col = sampleCubeAniso(sky_filt, trilinear, trans_dir, Z, trans_alphas).xyz;

    //return vec4f(refl_col, 1);
    let water_col = refl_fac * refl_col + trans_fac * trans_col;
    return mix(LIMIT_COLOR, vec4f(water_col, 1.0), falloff);
}

@fragment fn clamp_tex(v: FQOut) -> @location(0) vec4f {
    var px_in = textureSampleLevel(sky_in, trilinear, v.uv, 0.0);

    let bright = dot(px_in, vec4f(1, 1, 1, 0));
    if bright > MAX_BRIGHT_PX {
        px_in *= vec4f(vec3f(MAX_BRIGHT_PX / bright), 1);
    }
    return px_in;
}
