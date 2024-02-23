
@vertex fn fullscreen_tri(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4f {
    let u = f32(idx % 2);
    let v = f32(idx / 2);
    let xy = vec2f(4*u -1, 4*v-1);
    return vec4f(xy, 1, 1);
}

struct GlobalLighting {
    upper_ambient_color: vec3f,
    lower_ambient_color: vec3f,
    sun_color: vec3f,
    sun_dir: vec3f, // towards sun
    refr_sun_dir: vec3f,
    refr_sun_trans: f32,
}

@group(2) @binding(0) var<uniform> sun : GlobalLighting;
@group(2) @binding(1) var sky_tex: texture_2d<f32>;
@group(2) @binding(2) var sky_sampler: sampler;

fn get_sky(look_dir: vec3f) -> vec3f {
    let look = normalize(look_dir);
    let v = 0.5 - atan2(look.z, length(look.xy)) / PI;
    let u = 0.5 - atan2(look.y, look.x) / (2*PI);
    return textureSampleLevel(sky_tex, sky_sampler, vec2f(u, v), 0.0).xyz;
}

@group(1) @binding(0) var dist_buf: texture_depth_2d;
@group(1) @binding(1) var albedo_buf: texture_2d<f32>;
@group(1) @binding(2) var normal_buf: texture_2d<f32>;
@group(1) @binding(3) var rm_buf: texture_2d<f32>;
@group(1) @binding(4) var ao_buf: texture_2d<f32>;
@group(1) @binding(5) var material_buf: texture_2d<u32>;

@group(1) @binding(6) var water_buf: texture_2d<f32>;
@group(1) @binding(7) var water_dist_buf: texture_depth_2d;
@group(1) @binding(8) var water_sampler: sampler;

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

    let normal = 2 * textureLoad(normal_buf, px, 0).xyz - 1;
    var color: vec3f;
    if material == MAT_WATER {
        let look_dir = normalize(world_pos - camera.eye);

        let refr_up = refract(look_dir, vec3f(0.0, 0.0, 1.0), 0.75);
        let refr_norm = refract(look_dir, normal, 0.75);
        let corr_xy = length(look_dir.xy) / length(refr_up.xy);
        let corr_z = look_dir.z / refr_up.z;
        let trans_dir = normalize(vec3f(refr_norm.xy * corr_xy, refr_norm.z * corr_z)); // in planar-refracted space
        let uw_dist_val = textureSampleLevel(water_dist_buf, water_sampler, uv, 0.0);
        let uw_dist = length(world_pos - camera.eye) * (dist_val / uw_dist_val - 1.0);
        let refr_point = world_pos + uw_dist * trans_dir; // CSPR on
        //let refr_point = world_pos + uw_dist * normalize(look_dir + 0.75 * (refr_norm - refr_up)); // CSPR off
        let refr_clip = camera.matrix * vec4f(refr_point, 1.0);
        let refr_uv1 = vec2f(0.5, -0.5) * refr_clip.xy / refr_clip.w + 0.5;
        //let uw_dist_val_2 = textureSampleLevel(water_dist_buf, water_sampler, refr_uv1, 0.0);
        //let corr_uv = min(2.0, (1 / uw_dist_val_2 - 1 / dist_val) / (refr_clip.w / refr_clip.z - 1 / dist_val));
        //let refr_uv = uv + (refr_uv1 - uv) * corr_uv;

        let fresnel = 0.02 + 0.98 * pow(1.0 - max(0.0, dot(normal, -look_dir)), 5.0);
        let trans = textureSampleLevel(water_buf, water_sampler, refr_uv1, 0.0).xyz;
        let refl = get_sky(reflect(look_dir, normal));
        color = mix(trans, refl, fresnel);
    } else {
        let albedo = textureLoad(albedo_buf, px, 0).xyz;
        let ambient = mix(sun.lower_ambient_color, sun.upper_ambient_color, 0.5 * (1+normal.z));
        let direct = sun.sun_color * max(0.0, dot(normal, sun.sun_dir));
        color = albedo * (ambient + direct);
    }
    return vec4f(color, 1.0);
}

@group(1) @binding(6) var depth_adj_buf: texture_2d<f32>;

@fragment fn do_underwater_lighting(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let px = vec2i(floor(pos.xy));
    let material = textureLoad(material_buf, px, 0).x;
    if material == MAT_SKY {
        return vec4f(0.5, 0.5, 0.75, 1.0);
    }
    if material == MAT_SKY {
        return vec4f(0.5, 0.5, 0.5, 1.0);
    }
    let albedo = textureLoad(albedo_buf, px, 0).xyz;
    let normal = 2 * textureLoad(normal_buf, px, 0).xyz - 1;
    let ambient = mix(sun.lower_ambient_color, sun.upper_ambient_color, 0.5 * (1+normal.z));
    let direct = sun.sun_color * sun.refr_sun_trans * max(0.0, dot(normal, sun.refr_sun_dir));
    return vec4f(albedo * (ambient + direct), 1);
}

