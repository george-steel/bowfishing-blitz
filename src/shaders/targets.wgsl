struct PotInst {
    // must preserve angles
    base_point: vec3f,
    time_hit: f32,
    rotate: vec4f,
}

@group(1) @binding(0) var<storage, read> pots: array<PotInst>;

struct LathePoint {
    pos_rz: vec2f,
    norm1_rz: vec2f,
    norm2_rz: vec2f,
    v: f32,
}

@group(1) @binding(1) var<uniform> pot_model: array<LathePoint, 12>;

struct PotVSOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) refr_z: f32,
    @location(2) uv: vec2f,
    @location(3) world_norm: vec3f,
    @location(4) world_tan: vec3f,
    @location(5) state: u32,
}

const POT_U_DIVS: u32 = 8;
const SMASH_TIME: f32 = 0.8;

var<private> QUAD_U: array<f32, 6> = array(0, 0.5, 1, 1, 0.5, 1.5);
var<private> QUAD_V: array<u32, 6> = array(0, 1, 0, 0, 1, 1);

fn pot_vert(vert_idx: u32, inst_idx: u32, underwater: bool) -> PotVSOut {
    let ring_idx = vert_idx / (6 * POT_U_DIVS);
    let quad_idx = (vert_idx / 6) % (POT_U_DIVS);
    let quad_corner = vert_idx % 6;

    let phi: f32 = f32(quad_idx) + QUAD_U[quad_corner] - 0.5 * f32(ring_idx % 2);
    let u = phi / f32(POT_U_DIVS);
    let rho = vec2f(cos(TAU * u), sin(TAU * u));
    let ring_offset = QUAD_V[quad_corner];
    let point = pot_model[ring_idx + ring_offset];
    
    let local_pos = vec3f(rho * point.pos_rz.x, point.pos_rz.y);
    let norm_rz = select(point.norm2_rz, point.norm1_rz, ring_offset == 1);
    let local_norm = vec3f(rho * norm_rz.x, norm_rz.y);
    let local_tan = vec3f(rho.y, -rho.x, 0.0);

    let pot = pots[inst_idx];
    let world_pos = quat_rotate(pot.rotate, local_pos) + pot.base_point;
    let world_norm = quat_rotate(pot.rotate, local_norm);
    let world_tan = quat_rotate(pot.rotate, local_tan);

    var refr_z = world_pos.z;
    if underwater {
        refr_z = refracted_z(world_pos);
    }
    var out: PotVSOut;
    out.clip_pos = camera.matrix * vec4f(world_pos.xy, refr_z, 1.0);
    out.world_pos = world_pos;
    out.refr_z = refr_z;
    out.uv = vec2f(u, point.v);
    out.world_norm = world_norm;
    out.world_tan = world_tan;
    out.state = select(0u, 1u, pot.time_hit >= 0);
    return out;
}

@vertex fn pot_vert_above(@builtin(vertex_index) vert: u32, @builtin(instance_index) inst: u32) -> PotVSOut {
    return pot_vert(vert, inst, false);
}

@vertex fn pot_vert_below(@builtin(vertex_index) vert: u32, @builtin(instance_index) inst: u32) -> PotVSOut {
    return pot_vert(vert, inst, true);
}

@fragment fn pot_frag_above(v: PotVSOut, @builtin(front_facing) is_forward: bool) -> GBufferPoint {
    if camera.eye.z > 0 && v.world_pos.z < 0 {
        discard;
    }
    let norm = normalize(v.world_norm) * select(-1.0, 1.0, is_forward);
    let checker = (floor(v.uv.x * 6) + floor(v.uv.y * 10)) % 2;
    let red = vec3f(1.0, 0.0, 0.0);
    let green = vec3f(0.0, 1.0, 0.0);
    let albedo = select(select(red, green, v.state != 0), vec3f(0.5), checker == 0.0);

    var out: GBufferPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.rough_metal = vec2f(0.0, 0.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_SOLID;
    return out;
}

@fragment fn pot_frag_below(v: PotVSOut, @builtin(front_facing) is_forward: bool) -> UnderwaterPoint {
    if v.world_pos.z > 0 {
        discard;
    }
    let norm = normalize(v.world_norm) * select(-1.0, 1.0, is_forward);
    let checker = (floor(v.uv.x * 6) + floor(v.uv.y * 10)) % 2;

    let red = vec3f(1.0, 0.0, 0.0);
    let green = vec3f(0.0, 1.0, 0.0);
    let albedo = select(select(red, green, v.state != 0), vec3f(0.5), checker == 0.0);

    var out: UnderwaterPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (norm + 1), 1.0);
    out.rough_metal = vec2f(0.0, 0.0);
    out.occlusion = 1.0;
    out.mat_type = MAT_SOLID;
    out.depth_adj = v.world_pos.z / v.refr_z;
    return out;
}