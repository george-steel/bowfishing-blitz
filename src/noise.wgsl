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

alias gradval = vec3f; // gradient is x and y, value is z

fn perlin_noise_deriv(xy: vec2f, xform: mat2x2f, seed: i32) -> gradval {
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

fn fbm_deriv(uv: vec2f, init_transform: mat2x2f, octaves: u32, lancuarity: mat2x2f, gain: f32, init_seed: i32) -> gradval {
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

fn mult_gradval(a: gradval, b: gradval) -> gradval {
    return vec3f(a.xy * b.z + b.xy * a.z, a.z * b.z);
}

fn abs_gradval(a: gradval) -> gradval {
    return a * sign(a.z);
}

fn const_gradval(a: f32) -> gradval {
    return vec3f(0.0, 0.0, a);
}


const ls1mat = mat2x2f(0.2, 0.1, -0.1, 0.2);
const ls2mat = mat2x2f(0.5, -0.5, 0.5, 0.5);
const oct = mat2x2f(2.159, 0.978, -0.978, 2.159);
fn terrain(uv: vec2f) -> vec3f {
    let ls1 = 1.5 * perlin_noise_deriv(uv, ls1mat, 0) + const_gradval(0.2);
    let ls2 = perlin_noise_deriv(uv, ls2mat, 1);
    let ls = ls1 + 0.3 * mult_gradval(ls1 + const_gradval(1.0), ls2);
    let fs = fbm_deriv(uv, id2, 6u, oct, 0.35, 4);
    return ls + mult_gradval(fs, const_gradval(0.2) + 0.5 * abs_gradval(ls));// - const_gradval(1.0);
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
    return ramped;
}

// perform refraction
fn apparent_depth(dist: f32, eye_height: f32, depth: f32) -> f32 {
    var oblique = dist / (0.75 * abs(depth) + eye_height);
    var ratio = sqrt(0.77 * oblique * oblique + 1.77);
    oblique = dist / (abs(depth) / ratio + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
    return depth / ratio;
}

struct TerrainParams {
    transform: mat4x4f, // must be affine with the linear portion having have z as an eigenvector
    inv_transform: mat4x4f,
    uv_center: vec2f,
    uv_radius: f32,
    grid_size: u32,
}

struct Camera {
    matrix: mat4x4f,
    eye: vec3f,
    time: f32,
}

struct TerrainVertexOut {
    @builtin(position) clip_pos: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) terrain_coord: vec2f,
    @location(2) world_normal: vec3f,
}

@group(0) @binding(0) var<uniform> tparams: TerrainParams;
@group(0) @binding(1) var<uniform> camera: Camera;

@vertex fn terrain_mesh(@builtin(vertex_index) vert_idx: u32, @builtin(instance_index) inst_idx: u32) -> TerrainVertexOut {
    let ij = vec2i(vec2u(inst_idx + (vert_idx % 2), vert_idx / 2));
    let uv = (2.0 * vec2f(ij) / f32(tparams.grid_size) - 1.0) * tparams.uv_radius + tparams.uv_center;
    let zd = terrain(uv);
    let local_pos = vec4f(uv, zd.z, 1);
    let local_norm = normalize(vec3f(-zd.xy, 1));
    let world_pos = tparams.transform * local_pos;
    let world_norm = (vec4f(local_norm, 0) * tparams.inv_transform).xyz;

    var refract_pos = world_pos;
    if zd.z < 0 {
        let sealevel = tparams.transform[3].z;
        let cam_height = camera.eye.z - sealevel;
        let cam_dist = length(world_pos.xy - camera.eye.xy);
        let depth = world_pos.z - sealevel;
        refract_pos.z = sealevel + apparent_depth(cam_dist, cam_height, depth);
    }

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * refract_pos;
    out.world_pos = world_pos.xyz;
    out.terrain_coord = uv;
    out.world_normal = world_norm;
    return out;
}

const sun = vec3(5.0/13.0, 0.0, 12.0/13.0);
const uw_sun = vec3(0.29, 0.0, 0.96);

@fragment fn terrain_frag(v: TerrainVertexOut) -> @location(0) vec4f {
    let uv = v.terrain_coord;
    let zd = terrain(uv);
    let local_norm = normalize(vec3f(-zd.xy, 1));
    let norm = normalize((vec4f(local_norm, 0) * tparams.inv_transform).xyz);

    let z = zd.z;
    var col: vec3f;
    if z >= 0 {
        col = vec3f(saturate(0.5 + 0.5 * z), 0.9, saturate(0.3 + 0.3 * z));
    } else {
        col = vec3f(0.7, 0.7, 0.8);
    }
    // contour lines
    let hspan = length(vec2f(dpdx(z), dpdy(z)));
    let cdist = 0.1 * abs(10 * z - round(10 * z)) / hspan; // in pixels
    let line_fac = smoothstep(0.5, 1.0, cdist);
    let lcol = vec3f(0.1, 0.5, 0.5);

    let albedo = mix(lcol, col, line_fac);

    var spec: f32 = 0.0;
    if zd.z > 0 {
        let view = normalize(camera.eye.xyz - v.world_pos);
        let h = normalize(view + sun);
        let crd = length(norm - h);
        spec = 1.0 - smoothstep(0.02, 0.05, crd);
    }
    
    //let norm = normalize(v.world_normal);
    let light = 0.02 + 0.9 * max(0.0, dot(select(sun, uw_sun, z < 0), norm));
    return vec4f(vec3f(spec) + albedo * light, 1);
}

@vertex fn water_quad(@builtin(vertex_index) vert_idx: u32) -> TerrainVertexOut {
    let ij = vec2u(vert_idx % 2, vert_idx / 2);
    let uv = tparams.uv_center + tparams.uv_radius * (2.0 * vec2f(ij) - 1.0);
    let local_pos = vec4f(uv, 0, 1);
    let world_pos = tparams.transform * local_pos;
    let world_norm = (vec4f(0, 0, 1, 0) * tparams.inv_transform).xyz;

    var out: TerrainVertexOut;
    out.clip_pos = camera.matrix * world_pos;
    out.world_pos = world_pos.xyz;
    out.terrain_coord = uv;
    out.world_normal = world_norm;
    return out;
}

@fragment fn water_frag(v: TerrainVertexOut) -> @location(0) vec4f {
    let norm = normalize((vec4f(0,0,1,0) * tparams.inv_transform).xyz);

    let to_eye = normalize(camera.eye.xyz - v.world_pos);
    let refl = 0.02 + 0.98 * pow(1.0 - dot(norm, to_eye), 5.0);
    let h = normalize(to_eye + sun);
    let crd = length(norm - h);
    let spec = 1.0-smoothstep(0.05, 0.15, crd);
    return vec4f(spec, spec, spec, refl);
}