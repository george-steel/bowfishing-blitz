// LGC/Feistel-based 3d noise function
// https://www.jcgt.org/published/0009/03/02/paper.pdf
fn pcg3d_snorm(p: vec3i) -> vec3f {
	var v = vec3u(p);
	v = v * 1664525u + 1013904223u;

	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;
    v = v ^ (v >> vec3u(16));
	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;
	return ldexp(vec3f(v), vec3i(-31)) - 1.0;
}

fn smoother_step(x: f32) -> f32 {
    let t = saturate(x);
    return t*t*t*(t*(t*6.0 - 15.0) + 10.0);
}

fn perlin_noise_deriv(xy: vec2f, xform: mat2x2f, seed: i32) -> vec3f {
    let uv = xform * xy;
    let cell = vec2i(floor(uv));
    let f = fract(uv);

    let u = f*f*f*(f*(f*6 - 15) + 10);
    let du = 30.0*f*f*(f*(f-2)+1);

    let gsw = pcg3d_snorm(vec3i(cell.x, cell.y, seed)).xy;
    let gnw = pcg3d_snorm(vec3i(cell.x, cell.y+1, seed)).xy;
    let gse = pcg3d_snorm(vec3i(cell.x+1, cell.y, seed)).xy;
    let gne = pcg3d_snorm(vec3i(cell.x+1, cell.y+1, seed)).xy;
    let sw = vec3f(gsw, dot(gsw, vec2f(f.x, f.y)));
    let nw = vec3f(gnw, dot(gnw, vec2f(f.x, f.y - 1)));
    let se = vec3f(gse, dot(gse, vec2f(f.x - 1, f.y)));
    let ne = vec3f(gne, dot(gne, vec2f(f.x - 1, f.y - 1)));

    let w = mix(sw, nw, u.y) + vec3f(0, (nw.z - sw.z) * du.y, 0);
    let e = mix(se, ne, u.y) + vec3f(0, (ne.z - se.z) * du.y, 0);
    let n = mix(w, e, u.x) + vec3f((e.z - w.z) * du.x, 0, 0);
    return vec3f(n.xy * xform, n.z);
}

const id2 = mat2x2f(vec2f(1,0), vec2f(0,1));

const oct = mat2x2f(
    vec2f(2.159, 0.978),
    vec2f(-0.978, 2.159),
);

const large = mat2x2f(
    vec2f(0.3, 0.1),
    vec2f(-0.1, 0.3),
);

fn fbm_deriv(uv: vec2f, init_transform: mat2x2f, octaves: u32, lancuarity: mat2x2f, gain: f32, init_seed: i32) -> vec3f {
    var out = vec3f(0);
    var transform = init_transform;
    var amp = 1.0;
    var seed = init_seed;

    for (var i:u32 = 0; i < octaves; i++) {
        out += amp * perlin_noise_deriv(uv, transform, seed);
        seed += 1;
        amp *= gain;
        transform = lancuarity * transform;
    }
    return out;
}

fn terrain(uv: vec2f) -> vec3f {
    return 2 * perlin_noise_deriv(uv, large, 0) + 0.6 * fbm_deriv(uv, id2, 6u, oct, 0.3, 1);
}

struct ClipQuadOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}


@vertex fn fullscreen_quad(@builtin(vertex_index) idx: u32) -> ClipQuadOut {
    let u = f32(idx % 2);
    let v = f32((idx / 2) % 2);
    let xy = vec2f(2*u -1, 2*v-1);
    var out: ClipQuadOut;
    out.pos = vec4f(xy, 0, 1);
    out.uv = vec2f(u, v);
    return out;
}

@group(0) @binding(0) var ramp: texture_1d<f32>;
@group(0) @binding(1) var ramp_sampler: sampler;

@fragment fn my_image(vs: ClipQuadOut) -> @location(0) vec4f {
    let p = vs.uv * 10;
    let n = terrain(p);
    let ramped = textureSampleLevel(ramp, ramp_sampler, 0.5 * (n.z + 1), 0.0);
    //let locald = ((vs.pos.xy % 20) - 10) * vec2f(1, -1);
    //if dot(locald, n.xy) > 0 {
    //    return vec4f(n.zz + 0.5, 0, ramped.w);
    //} else {
    //    return vec4f(0, n.zz + 0.5, ramped.w);
    //}
    return ramped;
}

struct TerrainParams {
    transform: mat4x4f,
    inv_transform: mat4x4f,
    grid_size: vec2u,
    terrain_size: vec2f,
    xy_scale: f32,
    z_scale: f32,
}

struct TerrainVertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) terrain_coord: vec2f,
    @location(2) world_normal: vec3f,
}

@group(0) @binding(0) var<uniform> tparams: TerrainParams;
@group(0) @binding(1) var<uniform> camera: mat4x4f;

@vertex fn terrain_mesh(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> TerrainVertexOut {
    let ij = vec2i(vec2u(vert_idx / 2, inst_idx + (vert_idx % 2)));
    let uv = vec2f(ij) / vec2f(tparams.grid_size) * vec2f(tparams.terrain_size);
    let zd = terrain(uv);
    let local_pos = vec4f(uv * tparams.xy_scale, zd.z * tparams.z_scale, 1);
    let local_norm = normalize(vec3f(zd.xy * tparams.z_scale, tparams.xy_scale));
    let world_pos = tparams.transform * local_pos;
    let world_norm = (vec4f(local_norm, 0) * tparams.inv_transform).xyz;
    var out: TerrainVertexOut;
    out.clip_pos = camera * world_pos;
    out.world_pos = world_pos.xyz;
    out.terrain_coord = uv;
    out.world_normal = world_norm;
    return out;
}

const light = vec3(5.0/13.0, 0.0, 12.0/13.0);

@fragment fn terrain_frag(v: TerrainVertexOut) -> @location(0) vec4f {
    let uv = v.terrain_coord;
    let zd = terrain(uv);
    let local_norm = normalize(vec3f(zd.xy * tparams.z_scale, tparams.xy_scale));
    let norm = normalize((vec4f(local_norm, 0) * tparams.inv_transform).xyz);

    //let norm = normalize(v.world_normal);
    let light = 0.02 + 0.9 * max(0.0, dot(light, norm));
    return vec4f(light, light, light, 1);
}

