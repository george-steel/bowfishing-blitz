struct Arrow {
    end_pos: vec3f,
    state: u32,
    dir: vec3f,
    len: f32,
}

@group(1) @binding(0) var<storage, read> arrows: array<Arrow>;

struct ArrowVSOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) world_norm: vec3f,
    @location(2) refr_z: f32,
}

fn arrow_model(vert: u32, inst: u32, underwater: bool) -> ArrowVSOut {
    let arr = arrows[inst];
    let right = normalize(cross(vec3f(0.0, 0.0, 1.0), arr.dir));
    let up = normalize(cross(right, arr.dir));

    let t = f32(vert % 2) - 1.0;
    let theta = f32(vert / 2) / 6.0;
    let from_center = cos(theta * TAU) * right + sin(theta * TAU) * up;

    let world_pos = arr.end_pos + t * arr.len * arr.dir + 0.04 * from_center;

    var refr_z = world_pos.z;
    if underwater {
        refr_z = refracted_z(world_pos);
    }

    var out: ArrowVSOut;
    out.clip_pos = camera.matrix * vec4f(world_pos.xy, refr_z, 1.0);
    out.world_pos = world_pos;
    out.world_norm = from_center;
    out.refr_z = refr_z;
    return out;
}

@vertex fn arrow_vert_above(@builtin(vertex_index) vert: u32, @builtin(instance_index) inst: u32) -> ArrowVSOut {
    return arrow_model(vert, inst, false);
}

@vertex fn arrow_vert_below(@builtin(vertex_index) vert: u32, @builtin(instance_index) inst: u32) -> ArrowVSOut {
    return arrow_model(vert, inst, true);
}

@fragment fn arrow_frag_above(v: ArrowVSOut) -> GBufferPoint {
    let norm = normalize(v.world_norm);

    var out: GBufferPoint;
    out.albedo = vec4f(0.0, 0.0, 1.0, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.rough_metal = vec2f(0.0, 0.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_EMIT;
    return out;
}

@fragment fn arrow_frag_below(v: ArrowVSOut) -> UnderwaterPoint {
    if v.world_pos.z > 0 {
        discard;
    }
    let norm = normalize(v.world_norm);

    var out: UnderwaterPoint;
    out.albedo = vec4f(0.0, 0.0, 1.0, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.rough_metal = vec2f(0.0, 0.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_EMIT;
    out.depth_adj = v.world_pos.z / v.refr_z;
    return out;
}
