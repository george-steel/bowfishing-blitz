// planar refraction of underwater geometry.
// alters the depth of vertices to their virtual images.
fn apparent_depth(dist: f32, eye_height: f32, depth: f32) -> f32 {
    let d = abs(depth);
    let h = eye_height;
    let x = dist;

    // starting point in correct bucket
    let oi = dist / (depth * 0.01 + eye_height);
    var ratio: f32 = sqrt(0.777 * oi * oi + 1.777);
    // use newton's method to find apparant depth ratio
    for (var i = 0; i < 4; i++) {
        let q = ratio;
        let od = d + q * h;
        let o = q * x / od;
        let Do = x * d / od / od;
        let r = sqrt(0.777 * o * o + 1.777);
        let Dr = 0.777 * o * Do / r;
        ratio = q - (r - q) / (Dr - 1);
    }
    return depth / ratio;
}

fn refracted_error(dist: f32, depth: f32) -> f32 {
    let virt_depth = apparent_depth(dist, 1.0, depth);
    let a_dist = dist / (virt_depth + 1.0);
    let b_dist = dist - a_dist;
    let a_dir = vec2f(dist, virt_depth + 1.0);
    let b_dir = refract(normalize(a_dir), vec2f(0.0, -1.0), 0.75);
    let recon_depth = virt_depth * (b_dir.y / a_dir.y) / (b_dir.x / a_dir.x);
    return recon_depth / depth;
}

struct FQOut {
    @builtin(position) pos: vec4f,
    @location(0) xy: vec2f,
}

@vertex fn fullscreen_quad(@builtin(vertex_index) vert_idx: u32) -> FQOut {
    let ij = vec2u(vert_idx % 2, vert_idx / 2);
    let xy = 2.0 * vec2f(ij) - 1.0;
    var out: FQOut;
    out.pos = vec4f(xy, 0.0, 1.0);
    out.xy = vec2f(ij);
    return out;
}

@fragment fn refr_test(vert: FQOut) -> @location(0) vec4f {
    let dist = vert.xy.x * 500;
    let depth = vert.xy.y * 500;
    let err = refracted_error(dist, depth);
    if err >= 1.0 {
        let e = 5.0 * (err - 1.0);
        return vec4f(0.0, e, 0.0, 1.0);
    } else {
        let e = 5.0 * (1.0 - err);
        return vec4f(e, 0.0, 0.0, 1.0);
    }
}