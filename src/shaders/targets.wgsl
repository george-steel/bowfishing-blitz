struct Pot {
    center: vec3f,
    state: u32,
}

@group(1) @binding(0) var<storage, read> pots: array<Pot>;

struct PotVSIn {
    @location(0) pos: vec3f,
    @location(1) norm: vec3f,
}

struct PotVSOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) world_norm: vec3f,
    @location(2) refr_z: f32,
    @location(3) state: u32,
}

fn pot_vert(v: PotVSIn, inst: u32, underwater: bool) -> PotVSOut {
    let pot = pots[inst];

    let world_pos = pot.center + 0.4 * v.pos;
    var refr_z = world_pos.z;
    if underwater {
        refr_z = refracted_z(world_pos);
    }
    var out: PotVSOut;
    out.clip_pos = camera.matrix * vec4f(world_pos.xy, refr_z, 1.0);
    out.world_pos = world_pos;
    out.world_norm = v.norm;
    out.refr_z = refr_z;
    out.state = pot.state;
    return out;
}

@vertex fn pot_vert_above(vert: PotVSIn, @builtin(instance_index) inst: u32) -> PotVSOut {
    return pot_vert(vert, inst, false);
}

@vertex fn pot_vert_below(vert: PotVSIn, @builtin(instance_index) inst: u32) -> PotVSOut {
    return pot_vert(vert, inst, true);
}

@fragment fn pot_frag_above(v: PotVSOut) -> GBufferPoint {
    if camera.eye.z > 0 && v.world_pos.z < 0 {
        discard;
    }
    let norm = normalize(v.world_norm);
    let red = vec3f(1.0, 0.0, 0.0);
    let green = vec3f(0.0, 1.0, 0.0);
    let albedo = select(red, green, v.state != 0);

    var out: GBufferPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.rough_metal = vec2f(0.0, 0.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_SOLID;
    return out;
}

@fragment fn pot_frag_below(v: PotVSOut) -> UnderwaterPoint {
    if v.world_pos.z > 0 {
        discard;
    }
    let norm = normalize(v.world_norm);

    let red = vec3f(1.0, 0.0, 0.0);
    let green = vec3f(0.0, 1.0, 0.0);
    let albedo = select(red, green, v.state != 0);

    var out: UnderwaterPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.rough_metal = vec2f(0.0, 0.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_SOLID;
    out.depth_adj = v.world_pos.z / v.refr_z;
    return out;
}