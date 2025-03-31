// planar refraction of underwater geometry.
// alters the depth of vertices to their virtual images.
fn apparent_depth(dist: f32, eye_height: f32, depth: f32) -> f32 {
    // Finding apparent depth off-axis requires finding the fixed point of a nasty quartic.
    // To approximate start with a crude estimate (a fitted manually in Desmos)
    // then refine by ineration of the actual equation 
    let x = dist / (depth + 3 * eye_height);
    let init_ratio = 1.33 * (x * x + 1);

    var ratio: f32 = init_ratio; 
    var oblique = dist / (abs(depth) / ratio  + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
    oblique = dist / (abs(depth) / ratio + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
    oblique = dist / (abs(depth) / ratio + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
    oblique = dist / (abs(depth) / ratio + eye_height);
    ratio = sqrt(0.77 * oblique * oblique + 1.77);
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
    let dist = vert.xy.x * 100;
    let depth = vert.xy.y * 100;
    let err = refracted_error(dist, depth);
    if err >= 1.0 {
        let e = 1.0 * (err - 1.0);
        return vec4f(0.0, e, 0.0, 1.0);
    } else {
        let e = 1.0 * (1.0 - err);
        return vec4f(e, 0.0, 0.0, 1.0);
    }
}