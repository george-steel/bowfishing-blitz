
@vertex fn fullscreen_tri(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
    let u = f32(idx % 2);
    let v = f32(idx / 2);
    let xy = vec2f(4*u -1, 4*v-1);
    return vec4f(xy, 1, 1);
}

// adapted from Learn OpenGL
fn dist_GGX(normal: vec3f, half: vec3f, rough: f32) -> f32 {
    let alpha = rough * rough;
    let a2 = alpha * alpha;
    let nh = max(0.0, dot(normal, half));
    let nh2 = nh * nh;
    let denom = nh2 * (a2 - 1.0) + 1.0;
    return a2 / (denom * denom);
}

fn shad_GGX(nv: f32, rough: f32) -> f32{
    let r = rough + 1.0;
    let k = r * r / 8.0;
    return nv / mix(nv, 1.0, k);
}

fn direct_illumination(to_eye: vec3f, to_light: vec3f, normal: vec3f, rough: f32, metal: f32, albedo: vec3f, ao: f32, F0: f32) -> vec3f {
    let h = normalize(to_eye + to_light);
    let vh = dot(to_eye, h);
    let nv = max(0.0, dot(to_eye, normal));
    let nl = max(0.0, dot(to_light, normal));

    let shlick_vh = pow(clamp(1.0 - vh, 0.0, 1.0), 5.0);
    let ks = mix(mix(vec3f(F0), albedo, metal), vec3f(1.0), shlick_vh);
    let ndf = dist_GGX(normal, h, rough);
    let spec_ao = saturate(pow(nv + ao, exp2(-16.0 * rough - 1.0)) -1.0 + ao);
    let shad = shad_GGX(nv, rough) * shad_GGX(nl, rough) * spec_ao;
    let spec_fac = ndf * shad / ((4.0 * nv) + 0.0001);
    let spec = spec_fac * ks;

    let d0 = sqrt(mix(1-F0, 0, metal));
    let d90 = 2 * rough * vh * vh;
    let shlick_nl = pow(clamp(1.0 - nl, 0.0, 1.0), 5.0);
    let shlick_nv = pow(clamp(1.0 - nv, 0.0, 1.0), 5.0);
    let kd = mix(d0, d90, shlick_nl) * mix(d0, d90, shlick_nv);
    let diff = albedo * nl * kd * mix(ao, 1.0, nl);
    return spec + diff;
}

fn direct_diffuse_illumination(to_eye: vec3f, to_light: vec3f, normal: vec3f, rough: f32, metal: f32, albedo: vec3f, ao: f32, F0: f32) -> vec3f {
    let nl = max(0.0, dot(to_light, normal));
    return albedo * nl * mix(ao, 1.0, nl);
}

struct GlobalLighting {
    sun_color: vec3f,
    sky_fac: f32,
    sun_dir: vec3f, // towards sun
    refr_sun_dir: vec3f,
    refr_sun_trans: f32,
    water_lim_color: vec3f,
    half_secci: f32,
}

@group(2) @binding(0) var<uniform> sun: GlobalLighting;
@group(2) @binding(1) var lut_sampler: sampler;
@group(2) @binding(2) var dfg_lut: texture_2d<f32>;
@group(2) @binding(3) var cube_trilinear: sampler;
@group(2) @binding(4) var cube_bilinear: sampler;
@group(2) @binding(5) var radiance_map: texture_cube<f32>;
@group(2) @binding(6) var irradiance_map: texture_cube<f32>;
@group(2) @binding(7) var sky_tex: texture_cube<f32>;

fn get_sky(look_dir: vec3f) -> vec3f {
    return sun.sky_fac * textureSampleLevel(sky_tex, cube_bilinear, look_dir, 0.0).xyz;
}

fn ibl_illumination(to_eye: vec3f, normal: vec3f, rough: f32, metal: f32, albedo: vec3f, ao: f32, F0: f32) -> vec3f {
    let nv = max(0.0, dot(to_eye, normal));
    let dfg_uv = vec2f(sqrt(nv), rough);
    let dfg_vals = textureSampleLevel(dfg_lut, lut_sampler, dfg_uv, 0.0);
    let ks = mix(vec3f(F0), albedo, metal) * dfg_vals.x + dfg_vals.y;
    let spec_ao = saturate(pow(nv + ao, exp2(-16.0 * rough - 1.0)) -1.0 + ao);
    let shlick_nl = pow(clamp(1.0 - nv, 0.0, 1.0), 5.0);
    let kd = (1-F0) * (1-metal) * mix(1, rough, shlick_nl);

    let alpha = rough * rough;
    // from "moving frostbite to PBR"
    let dir_fac = (1-alpha) * (sqrt(1-alpha) + alpha);
    let spec_dir = mix(normal, reflect(-to_eye, normal), dir_fac);
    let max_radiance_mip = f32(textureNumLevels(radiance_map) - 1);

    let spec_col = textureSampleLevel(radiance_map, cube_trilinear, spec_dir, max_radiance_mip + log2(alpha)).xyz;
    let diff_col = textureSampleLevel(irradiance_map, cube_bilinear, spec_dir, 0.0).xyz;

    return sun.sky_fac * (ks * spec_ao * spec_col + kd * ao * albedo * diff_col);
}

@group(1) @binding(0) var dist_buf: texture_depth_2d;
@group(1) @binding(1) var albedo_buf: texture_2d<f32>;
@group(1) @binding(2) var normal_buf: texture_2d<f32>;
@group(1) @binding(3) var rm_buf: texture_2d<f32>;
@group(1) @binding(4) var ao_buf: texture_2d<f32>;
@group(1) @binding(5) var material_buf: texture_2d<u32>;
@group(1) @binding(6) var shadow_buf: texture_depth_2d;
@group(1) @binding(7) var shadow_sampler: sampler_comparison;

@group(1) @binding(8) var trans_buf: texture_2d<f32>;
@group(1) @binding(9) var trans_dist_buf: texture_depth_2d;
@group(1) @binding(10) var refl_buf: texture_2d<f32>;
@group(1) @binding(11) var refl_dist_buf: texture_depth_2d;
@group(1) @binding(12) var water_sampler: sampler;

@fragment fn do_global_lighting(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let px = vec2i(floor(pos.xy));
    let uv = pos.xy / camera.fb_size;
    let clip_xy = (uv - 0.5)  * vec2f(2, -2);
    let material = textureLoad(material_buf, px, 0).x;
    if material == MAT_SKY {
        let clip_pos = vec4f(clip_xy * camera.clip_near, camera.clip_near, camera.clip_near);
        let look_dir = normalize((camera.inv_matrix * clip_pos).xyz - camera.eye);
        return vec4f(get_sky(look_dir), 1.0);
    }

    let dist_val = textureLoad(dist_buf, px, 0);
    let clip_w = camera.clip_near / dist_val;
    let clip_pos = vec4f(clip_xy * clip_w, camera.clip_near, clip_w);
    let world_pos = (camera.inv_matrix * clip_pos).xyz;

    let normal = normalize(2 * textureLoad(normal_buf, px, 0).xyz - 1);
    var color: vec3f;
    if material == MAT_WATER {
        let look_dir = normalize(world_pos - camera.eye);

        // transmitted color
        let refr_up = refract(look_dir, vec3f(0.0, 0.0, 1.0), 0.75);
        let refr_norm = refract(look_dir, normal, 0.75);
        let corr_xy = length(look_dir.xy) / length(refr_up.xy);
        let corr_z = look_dir.z / refr_up.z;
        let virt_trans_dir = normalize(vec3f(refr_norm.xy * corr_xy, refr_norm.z * corr_z)); // in planar-refracted space
        let uw_dist_val = textureSampleLevel(trans_dist_buf, water_sampler, uv, 0);
        let uw_dist = length(world_pos - camera.eye) * (dist_val / uw_dist_val - 1.0);
        let refr_point = world_pos + uw_dist * virt_trans_dir; // CSPR on
        //let refr_point = world_pos + uw_dist * normalize(look_dir + 0.75 * (refr_norm - refr_up)); // CSPR off
        let refr_clip = camera.matrix * vec4f(refr_point, 1.0);
        let refr_uv1 = vec2f(0.5, -0.5) * refr_clip.xy / refr_clip.w + 0.5;
        //let uw_dist_val_2 = textureSampleLevel(trans_dist_buf, water_sampler, refr_uv1, 0.0);
        //let corr_uv = min(2.0, (1 / uw_dist_val_2 - 1 / dist_val) / (refr_clip.w / refr_clip.z - 1 / dist_val));
        //let refr_uv = uv + (refr_uv1 - uv) * corr_uv;
        let trans = textureSampleLevel(trans_buf, water_sampler, refr_uv1, 0.0).xyz;

        // reflected color
        let refl_dist_val = max(textureSampleLevel(refl_dist_buf, water_sampler, uv, 0), 1e-4);
        let refl_dist = length(world_pos - camera.eye) * (dist_val / refl_dist_val - 1.0);
        let refl_norm = reflect(look_dir, normal);
        let virt_refl_dir = vec3f(refl_norm.xy, -refl_norm.z);
        let refl_point = world_pos + refl_dist * virt_refl_dir;
        let refl_clip = camera.matrix * vec4f(refl_point, 1.0);
        let refl_uv1 = vec2f(0.5, -0.5) * refl_clip.xy / refl_clip.w + 0.5;
        let refl_raw = textureSampleLevel(refl_buf, water_sampler, refl_uv1, 0.0).xyz;
        let refl_px = vec2i(floor(clamp(refl_uv1, vec2f(0), vec2f(1)) * camera.fb_size));
        let refl_mask = textureLoad(material_buf, refl_px, 0).x;
        let refl = refl_raw * select(0.1, 1.0, refl_mask == MAT_WATER);

        let fresnel = 0.02 + 0.98 * pow(1.0 - max(0.0, dot(normal, -look_dir)), 5.0);
        color = mix(trans, refl, fresnel);
    } else {
        let albedo = textureLoad(albedo_buf, px, 0).xyz;
        let rm_val = textureLoad(rm_buf, px, 0).xy;
        let rough = rm_val.x;
        let metal = select(0.0, rm_val.y, material == MAT_SOLID);

        var emit = vec3f(0);
        if material == MAT_EMIT {
            let emit_fac = rm_val.y / (1 - rm_val.y);
            emit = emit_fac * albedo;
        }
        
        let to_eye = normalize(camera.eye - world_pos);
        let ao = textureLoad(ao_buf, px, 0).x;
        let ambient = ibl_illumination(to_eye, normal, rough, metal, albedo, ao, 0.04);

        let to_light = sun.sun_dir;
        let shadow_point = shadow_map_point(world_pos);
        let shadow_fac = textureSampleCompareLevel(shadow_buf, shadow_sampler, shadow_point.xy, shadow_point.z);
        let direct_refl = direct_illumination(to_eye, to_light, normal, rough, metal, albedo, ao, 0.04);
        let direct_radiance = shadow_fac * sun.sun_color;
        let direct = direct_radiance * direct_refl;

        color = ambient + direct + emit;
    }
    return vec4f(color, 1.0);
}

@fragment fn do_reflected_lighting(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let px = vec2i(floor(pos.xy));
    let uv = pos.xy / camera.water_fb_size;
    let clip_xy = (uv - 0.5)  * vec2f(2, -2);
    let material = textureLoad(material_buf, px, 0).x;
    if material == MAT_SKY {
        let clip_pos = vec4f(clip_xy * camera.clip_near, camera.clip_near, camera.clip_near);
        let look_dir = normalize((camera.inv_matrix * clip_pos).xyz - camera.eye);
        let sky = get_sky(vec3f(look_dir.xy, -look_dir.z));
        return vec4f(sky, 1.0);
    }

    let dist_val = textureLoad(dist_buf, px, 0);
    let clip_w = camera.clip_near / dist_val;
    let clip_pos = vec4f(clip_xy * clip_w, camera.clip_near, clip_w);
    let virt_pos = (camera.inv_matrix * clip_pos).xyz;
    let world_pos = vec3f(virt_pos.xy, -virt_pos.z);

    let normal = normalize(2 * textureLoad(normal_buf, px, 0).xyz - 1);
    let albedo = textureLoad(albedo_buf, px, 0).xyz;
    let rm_val = textureLoad(rm_buf, px, 0).xy;
    let rough = rm_val.x;
    let metal = select(0.0, rm_val.y, material == MAT_SOLID); 

    var emit = vec3f(0);
    if material == MAT_EMIT {
        let emit_fac = rm_val.y / (1 - rm_val.y);
        emit = emit_fac * albedo;
    }

    let virt_to_eye = normalize(camera.eye - virt_pos);
    let to_eye = vec3f(virt_to_eye.xy, -virt_to_eye.z);

    let ao = textureLoad(ao_buf, px, 0).x;
    let ambient = ibl_illumination(to_eye, normal, rough, metal, albedo, ao, 0.04);

    let to_light = sun.sun_dir;
    let shadow_point = shadow_map_point(world_pos);
    let shadow_fac = textureSampleCompareLevel(shadow_buf, shadow_sampler, shadow_point.xy, shadow_point.z);
    let direct_refl = direct_illumination(to_eye, to_light, normal, rough, metal, albedo, ao, 0.04);
    let direct_radiance = shadow_fac * sun.sun_color;
    let direct = direct_radiance * direct_refl;

    let color = ambient + direct + emit;
    return vec4f(color, 1.0);
}

@fragment fn do_underwater_lighting(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let px = vec2i(floor(pos.xy));
    let material = textureLoad(material_buf, px, 0).x;
    if material == MAT_SKY {
        return vec4f(sun.water_lim_color, 1);
    }

    let clip_xy = ((pos.xy / camera.water_fb_size) - 0.5)  * vec2f(2, -2);
    let dist_val = textureLoad(dist_buf, px, 0);
    let clip_w = camera.clip_near / dist_val;
    let clip_pos = vec4f(clip_xy * clip_w, camera.clip_near, clip_w);
    let virt_pos = (camera.inv_matrix * clip_pos).xyz;

    // recover world position using snell's law
    let look_dir_above = normalize(virt_pos - camera.eye);
    let look_dir_below = normalize(refract(look_dir_above, vec3f(0.0, 0.0, 1.0), 0.75));
    let depth_adj = (look_dir_below.z / look_dir_above.z) * (length(look_dir_above.xy) / length(look_dir_below.xy));
    let world_pos = vec3f(virt_pos.xy, depth_adj * virt_pos.z);

    let amb_falloff = exp(world_pos.z / sun.half_secci);
    let sun_falloff = exp(world_pos.z / (sun.refr_sun_dir.z * sun.half_secci));
    let look_falloff = exp(world_pos.z / (-look_dir_below.z * sun.half_secci));

    let normal = normalize(2 * textureLoad(normal_buf, px, 0).xyz - 1);
    let albedo = textureLoad(albedo_buf, px, 0).xyz;
    let rm_val = textureLoad(rm_buf, px, 0).xy;
    let rough = rm_val.x;
    let metal = select(0.0, rm_val.y, material == MAT_SOLID); 

    var emit = vec3f(0);
    if material == MAT_EMIT {
        let emit_fac = rm_val.y / (1 - rm_val.y);
        emit = emit_fac * albedo;
    }

    let to_eye = -look_dir_below;
    let ao = textureLoad(ao_buf, px, 0).x;
    let ambient = amb_falloff * ibl_illumination(to_eye, normal, rough, metal, albedo, ao, 0.01);

    let to_light = sun.refr_sun_dir;
    let shadow_point = shadow_map_point(world_pos);
    let shadow_fac = textureSampleCompareLevel(shadow_buf, shadow_sampler, shadow_point.xy, shadow_point.z);
    let direct_refl = direct_illumination(to_eye, to_light, normal, rough, metal, albedo, ao, 0.01);
    let direct_radiance = shadow_fac * sun_falloff * sun.refr_sun_trans * sun.sun_color;
    let direct = direct_radiance * direct_refl;

    let color = ambient + direct + emit;
    return vec4f(mix(sun.water_lim_color, color, look_falloff), 1);
}

