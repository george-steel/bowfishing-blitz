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

// 4 RNGs at once at adjacent points using SIMD
fn pcg3d_snorm_perlin_quad(p:vec3i) -> mat4x2f {
    let a = vec3u(p) * 1664525u + 1013904223u;
    let b = vec3u(p + 1) * 1664525u + 1013904223u;

    var x = vec4u(a.x, a.x, b.x, b.x);
    var y = vec4u(a.y, b.y, a.y, b.y);
    var z = vec4u(a.z);

    x += y * z;
    y += z * x;
    z += x * y;
    x = x ^ (x >> vec4u(16));
    y = y ^ (y >> vec4u(16));
    z = z ^ (z >> vec4u(16));
    x += y * z;
    y += z * x;

    return transpose(mat2x4f(ldexp(vec4f(x), vec4i(-31)) - 1.0, ldexp(vec4f(y), vec4i(-31)) - 1.0));
}

alias gradval = vec3f; // gradient is x and y, value is z

fn perlin_noise_deriv(xy: vec2f, freq: mat2x2f, seed: i32) -> gradval {
    let uv = freq * xy;
    let cell = vec2i(floor(uv));
    let f = fract(uv);

    let u = f*f*f*(f*(f*6 - 15) + 10);
    let du = 30.0*f*f*(f*(f-2)+1);

    let grads = pcg3d_snorm_perlin_quad(vec3i(cell, seed));
    let gsw = grads[0]; //pcg3d_snorm(vec3i(cell.x, cell.y, seed)).xy;
    let gnw = grads[1]; //pcg3d_snorm(vec3i(cell.x, cell.y+1, seed)).xy;
    let gse = grads[2]; //pcg3d_snorm(vec3i(cell.x+1, cell.y, seed)).xy;
    let gne = grads[3]; //pcg3d_snorm(vec3i(cell.x+1, cell.y+1, seed)).xy;
    let sw = vec3f(gsw, dot(gsw, f));
    let nw = vec3f(gnw, dot(gnw, f - vec2f(0.0, 1.0)));
    let se = vec3f(gse, dot(gse, f - vec2f(1.0, 0.0)));
    let ne = vec3f(gne, dot(gne, f - vec2f(1.0, 1.0)));

    let w = mix(sw, nw, u.y) + vec3f(0, (nw.z - sw.z) * du.y, 0);
    let e = mix(se, ne, u.y) + vec3f(0, (ne.z - se.z) * du.y, 0);
    let n = mix(w, e, u.x) + vec3f((e.z - w.z) * du.x, 0, 0);
    return vec3f(n.xy * freq, n.z);
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

fn sin_wave_deriv(xy: vec2f, freq_dir: vec2f, phase: f32) -> gradval {
    let u = (dot(freq_dir, xy) + phase) * TAU;
    return vec3f(TAU * cos(u) * freq_dir, sin(u));
}
