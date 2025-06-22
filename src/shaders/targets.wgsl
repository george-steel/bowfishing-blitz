struct PotInst {
    // must preserve angles
    base_point: vec3f,
    time_hit: f32,
    rotate: vec4f,
    colors_packed: vec3u,
    seed: u32,
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
    @location(1) uv: vec2f,
    @location(2) world_norm: vec3f,
    @location(3) world_tan: vec3f,
    @location(4) world_bitan: vec3f,
    @location(5) color_a: vec3f,
    @location(6) color_b: vec3f,
    @location(7) explode_progress: f32,
}

const POT_U_DIVS: u32 = 8;
const EXPLODE_TIME: f32 = 0.5;
const SINK_TIME: f32 = 1.0;

var<private> QUAD_U: array<f32, 6> = array(0, 0.5, 1, 1, 0.5, 1.5);
var<private> QUAD_V: array<u32, 6> = array(0, 1, 0, 0, 1, 1);

@vertex fn pot_vert(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> PotVSOut {
    let ring_idx = vert_idx / (6 * POT_U_DIVS);
    let quad_idx = (vert_idx / 6) % (POT_U_DIVS);
    let quad_corner = vert_idx % 6;

    // solid of rotation
    let phi: f32 = f32(quad_idx) + QUAD_U[quad_corner] + 0.5 * f32(ring_idx % 2);
    let u = phi / f32(POT_U_DIVS);
    let rho = vec2f(cos(TAU * u), sin(TAU * u));
    let ring_offset = QUAD_V[quad_corner];
    let point = pot_model[ring_idx + ring_offset];
    
    var local_pos = vec3f(rho * point.pos_rz.x, point.pos_rz.y);
    let norm_rz = select(point.norm2_rz, point.norm1_rz, ring_offset == 1);
    var local_norm = vec3f(rho * norm_rz.x, norm_rz.y);
    var local_tan = vec3f(-rho.y, rho.x, 0.0);

    let pot = pots[inst_idx];

    var explode_progress:f32 = 0;
    if pot.time_hit > 0 {
        let explode_time = saturate((camera.time - pot.time_hit) / EXPLODE_TIME);
        explode_progress = (2.0 - explode_time) * explode_time;
        let sink_progress = smoothstep(0.0, 1.0, (camera.time - pot.time_hit) / SINK_TIME);

        let tri_idx = (vert_idx / 3) % (2 * POT_U_DIVS);
        let tri_u = f32(tri_idx + (ring_idx % 2)) / f32(2 * POT_U_DIVS);
        let tri_rho = vec2f(cos(TAU * tri_u), sin(TAU * tri_u));
        let tri_point = pot_model[ring_idx + (quad_corner / 3)];
        let anchor_pos = vec3f(tri_rho * tri_point.pos_rz.x, tri_point.pos_rz.y);
        let corner_delta = local_pos - anchor_pos;

        let noise = pcg3d_snorm(vec3i(vec3u(ring_idx, tri_idx, pot.seed)));
        let explode_delta = explode_progress * (vec3f(tri_rho * tri_point.pos_rz.x, 1.0) + 0.5 * noise);
        let exploded_pos = anchor_pos + explode_delta;

        let shard_rot = normalize(vec4f(0.3 * explode_progress * noise.z * local_tan, 1.0));

        let world_down = quat_rotate(pot.rotate * vec4f(1, 1, 1, -1), vec3f(0, 0, -1));
        let world_down_adj = mix(1.0, -1/world_down.z, 0.7);
        let down = mix(vec3f(0, 0, -1), world_down * world_down_adj, 0.6);
        let sink_delta = 0.9 * sink_progress * exploded_pos.z * down;
        let sunk_pos = exploded_pos + sink_delta;
        local_pos = sunk_pos + quat_rotate(shard_rot, corner_delta);
        local_norm = quat_rotate(shard_rot, local_norm);
    }

    let world_pos = quat_rotate(pot.rotate, local_pos) + pot.base_point;
    let world_norm = quat_rotate(pot.rotate, local_norm);
    let world_tan = quat_rotate(pot.rotate, local_tan);

    let color_a = unpack2x16float(pot.colors_packed.x);
    let color_ab = unpack2x16float(pot.colors_packed.y);
    let color_b = unpack2x16float(pot.colors_packed.z);
    
    var out: PotVSOut;
    out.clip_pos = clip_point(world_pos);
    out.world_pos = world_pos;
    out.uv = vec2f(u, point.v);
    out.world_norm = world_norm;
    out.world_tan = world_tan;
    out.world_bitan = cross(world_norm, world_tan);
    out.color_a = vec3f(color_a, color_ab.x);
    out.color_b = vec3f(color_ab.y, color_b);
    out.explode_progress = explode_progress;
    return out;
}

@vertex fn pot_vert_shadow(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> @builtin(position) vec4f {
    let ring_idx = vert_idx / (6 * POT_U_DIVS);
    let quad_idx = (vert_idx / 6) % (POT_U_DIVS);
    let quad_corner = vert_idx % 6;

    // solid of rotation
    let phi: f32 = f32(quad_idx) + QUAD_U[quad_corner] + 0.5 * f32(ring_idx % 2);
    let u = phi / f32(POT_U_DIVS);
    let rho = vec2f(cos(TAU * u), sin(TAU * u));
    let ring_offset = QUAD_V[quad_corner];
    let point = pot_model[ring_idx + ring_offset];
    
    var local_pos = vec3f(rho * point.pos_rz.x, point.pos_rz.y);
    var local_tan = vec3f(-rho.y, rho.x, 0.0);

    let pot = pots[inst_idx];

    var explode_progress:f32 = 0;
    if pot.time_hit > 0 {
        let explode_time = saturate((camera.time - pot.time_hit) / EXPLODE_TIME);
        explode_progress = (2.0 - explode_time) * explode_time;
        let sink_progress = smoothstep(0.0, 1.0, (camera.time - pot.time_hit) / SINK_TIME);

        let tri_idx = (vert_idx / 3) % (2 * POT_U_DIVS);
        let tri_u = f32(tri_idx + (ring_idx % 2)) / f32(2 * POT_U_DIVS);
        let tri_rho = vec2f(cos(TAU * tri_u), sin(TAU * tri_u));
        let tri_point = pot_model[ring_idx + (quad_corner / 3)];
        let anchor_pos = vec3f(tri_rho * tri_point.pos_rz.x, tri_point.pos_rz.y);
        let corner_delta = local_pos - anchor_pos;

        let noise = pcg3d_snorm(vec3i(vec3u(ring_idx, tri_idx, pot.seed)));
        let explode_delta = explode_progress * (vec3f(tri_rho * tri_point.pos_rz.x, 1.0) + 0.5 * noise);
        let exploded_pos = anchor_pos + explode_delta;

        let shard_rot = normalize(vec4f(0.3 * explode_progress * noise.z * local_tan, 1.0));

        let world_down = quat_rotate(pot.rotate * vec4f(1, 1, 1, -1), vec3f(0, 0, -1));
        let world_down_adj = mix(1.0, -1/world_down.z, 0.7);
        let down = mix(vec3f(0, 0, -1), world_down * world_down_adj, 0.6);
        let sink_delta = 0.9 * sink_progress * exploded_pos.z * down;
        let sunk_pos = exploded_pos + sink_delta;
        local_pos = sunk_pos + quat_rotate(shard_rot, corner_delta);
    }

    let world_pos = quat_rotate(pot.rotate, local_pos) + pot.base_point;
    
    return shadow_clip_point(world_pos);
}

@group(1) @binding(2) var tex_sampler: sampler;
@group(1) @binding(3) var pot_co_tex: texture_2d<f32>;
@group(1) @binding(4) var pot_nr_tex: texture_2d<f32>;

@fragment fn pot_frag(v: PotVSOut, @builtin(front_facing) is_forward: bool) -> GBufferPoint {
    guard_frag(v.world_pos);

    let back_corr = select(-1.0, 1.0, is_forward);
    let norm = normalize(v.world_norm) * back_corr;
    let tan = normalize(v.world_tan);
    let bitan = normalize(v.world_bitan) * back_corr;
    let norm_mat = mat3x3f(tan, bitan, norm);

    let col_mat = mat3x3f(v.color_a, v.color_b, vec3f(1.0));

    let uv = vec2f(3 * v.uv.x, 0.5 * select(2 - v.uv.y, v.uv.y, is_forward));

    let co = textureSample(pot_co_tex, tex_sampler, uv);
    let nr = textureSample(pot_nr_tex, tex_sampler, uv);
    var albedo = col_mat * co.xyz;
    let frag_norm = norm_mat * normalize(2 * nr.xyz - 1);

    if !is_forward {
        albedo *= mix(1.0, mix(0.2, 1.0, v.explode_progress), smoothstep(0.25, 0.35, v.uv.x));
    }

    var out: GBufferPoint;
    out.albedo = vec4f(albedo, 1.0);
    out.normal = vec4f(0.5 * (frag_norm + 1), 1.0);
    out.rough_metal = vec2f(nr.w, 0.0);
    out.occlusion = co.w;
    out.mat_type = MAT_SOLID;
    return out;
}
