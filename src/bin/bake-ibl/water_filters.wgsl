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

override U_AT_PLUSX: f32 = 0.5;
override WATER_STRENGTH: f32 = 0.6;
override WATER_ROUGH: f32 = 0.25;
override MAX_BRIGHT_PX: f32 = 3.0;

@group(0) @binding(0) var trilinear: sampler;
@group(0) @binding(1) var sky_in: texture_2d<f32>;
@group(0) @binding(2) var sky_filt: texture_cube<f32>;

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

    let refr_dir = normalize(refract(dir, vec3f(0.0, 0.0, 1.0), 0.75));
    let refl_dir = normalize(vec3f(dir.xy, -dir.z));

    let rough = WATER_ROUGH * abs(dir.z);

    let shlick = pow(saturate(1.0 - abs(dir.z)), 5.0);
    let ks = mix(0.02, 1.0, shlick);

    let refr_col = textureSampleLevel(sky_filt, trilinear, refr_dir, max(0.0, max_cube_mip + 2 * log2(rough)));
    let refl_col = textureSampleLevel(sky_filt, trilinear, refl_dir, max(0.0, max_cube_mip + 2 * log2(rough)));

    let water_col = mix(refr_col, refl_col, ks);
    return mix(px_in, water_col, WATER_STRENGTH);
}

const LIMIT_COLOR = vec4f(0.03, 0.05, 0.1, 1.0);

@fragment fn below_water_equi(v: FQOut) -> @location(0) vec4f {
    let max_cube_mip = f32(textureNumLevels(sky_filt) - 1);
    let theta = v.xy.y * PI / 2;

    var px_in = textureSampleLevel(sky_in, trilinear, v.uv, 0.0);
    let falloff = 0.85 * exp(0.3 * (-1 / abs(sin(theta)) + 1)) + 0.15;
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

    let refr_dir = normalize(refract(dir, vec3f(0.0, 0.0, -1.0), 1.33));
    let refl_dir = normalize(vec3f(dir.xy, -dir.z));

    let refr_rough = WATER_ROUGH * abs(refr_dir.z);
    let refl_rough = WATER_ROUGH * abs(refl_dir.z);

    let shlick = pow(saturate(1.0 - abs(refr_dir.z)), 5.0);
    //let shlick = saturate(1.0 - abs(refr_dir.z));
    let ks = mix(mix(0.02, 1.0, shlick), 1.0, smoothstep(0.2, 0.0, refr_dir.z));

    let refl_col = textureSampleLevel(sky_filt, trilinear, refl_dir, max(0.0, max_cube_mip + 2 * log2(refl_rough)));
    var surface_col = refl_col;

    if refr_dir.z != 0 {
        let refr_col = textureSampleLevel(sky_filt, trilinear, refr_dir, max(0.0, max_cube_mip -1 + 2 * log2(refr_rough)));
        //let refr_adj = vec4f(vec3f(refr_dir.z / dir.z), 1);
        surface_col = mix(refr_col, refl_col, ks);
    }

    return mix(LIMIT_COLOR, surface_col, falloff);
}

@fragment fn clamp_tex(v: FQOut) -> @location(0) vec4f {
    var px_in = textureSampleLevel(sky_in, trilinear, v.uv, 0.0);

    let bright = dot(px_in, vec4f(1, 1, 1, 0));
    if bright > MAX_BRIGHT_PX {
        px_in *= vec4f(vec3f(MAX_BRIGHT_PX / bright), 1);
    }
    return px_in;
}
