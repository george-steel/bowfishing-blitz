![screenshot](./doc/bb-title-screen.jpg)

Bowfishing Blitz: A Game of Refractive Aberration
=================================================

A tribute to the minigames of the Zelda series, and a demonstration of a new technique for rendering real-time water refraction.
Made for Acerola Jam 0.

**Objective:** Smash as many pots as possible in three laps of the pond before time runs out.
Be careful, *water refracts light, but it does **not** refract arrows.* 
When shooting, remember that objects in water are deeper than they appear, and adjust your aim accordingly. 

[Builds available on itch.io](https://george-steel.itch.io/bowfishing-blitz).

Credits
-------

Copyright 2024 by George Steel. Available under the [Mozilla Public License 2.0](./LICENSE.txt).

[Background music](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1300032) by Kevin MacLeod used under the [Creative Commons Attribution 3.0 license](http://creativecommons.org/licenses/by/3.0/).

Rendering pipeline setup inspired by [Jasper's excellent video](https://www.youtube.com/watch?v=By7qcgaqGI4) on deferred lighting, transmission, and decals.

Key insights into the nature of clip space and vertex shaders inspired by [Manifold Garden](https://store.steampowered.com/app/473950/Manifold_Garden/).

Additional CC0 textures and audio listed in [`assets/SOURCES.md`](./assets/SOURCES.md).

----

Introducing Clip-Space Planar Refraction
=========================================

This game was created as a demonstration of how to adapt a deferred rendering pipeline in order to achieve physically-accurate planar refraction
without the heavy computational cost of raytracing (commonly believed to be required for this effect).
By performing a form of non-perspective path tracing on each vertex of the underwater geometry to produce clip space positions,
we can use normal triangle rasterization to render an accurate image refracted by flat water, without needing to raytrace each fragment.
Unlike screen-space transmission, this is fully capable of rendering refracted geometry that would not otherwise be visible to the camera.

In order extend this technique efficiently to wavy water,
we can then use the flat-water refracted image as the basis for screen-space effect similar to parallax mapping,
projecting the true refracted ray path directions into planar-refracted space,
then combining these projected directions with the planar-refracted distance buffer,
to find a displaced point to sample the transmitted colour from on the underwater image.
As this displacement tends to be fairly small (unlike the difference between view direction and refracted direction),
this makes it viable to use the actual refracted paths for transmission sampling without a large fraction of them going offscreen
(a problem which forces most games implementing water transmission to use non-physical forms of distortion which nave no planar component).



Clip space and paths
--------------------

In order to convert a traditionally raytraced technique (refraction) to a rasterized one,
we must first understand the relationship between the two approaches.
Both rendering techniques are, at their core, ways of finding optical paths
from the the camera to scene geometry for every direction in the camera's field of view.
In raytracing, we use a pull approach, processing pixels independently and finding the intersection of a pixel's ray with the scene by
searching through an indexed representation of the scene in world-space (for which many possible data structures exist).
Although computationally intensive, this as an extremely flexible approach
and can deal with paths that bend or reflect by continuing the search with a new origin and direction.

Rasterizing, on the other hand, solves the same problem by using a push approach,
breaking a up a scene up into worldspace triangles and then calculating which pixels each triangle can reach.
By merging the results for all triangles (taking the shortest path for each pixel to account for occlusion),
we get a path for each pixel that can reach the scene.
This significantly reduces the processing requirements needed, as there is no need to search and index every pixel.
Pathfinding (which is usually quite simple) is only done for each triangle vertex by a vertex shader,
and the results are simply interpolated between when passing data to fragment shaders.

To represent optical paths between the camera and vertices, we use a coordinate system called clip space.
Each path is described by a four-vector `vec4f(x, y, z, w)`,
where `x/w`, `y/w`, and the sign of `z/w` represent the direction of the path as it leaves the camera
(where `z/w` is positive, `x/w` and `y/w` determine pixel position),
and `w/z` (possibly adjusted for depth clipping) represents the length of the path
(used to select the shortest path and accumulated in the distance buffer). 

In the case of straight lines through empty space, direct paths can be found using the perspective transformation,
with the path to each vertex determined (as a clip-space vector) by

```wgsl
clip_pos = camera.matrix * vec4f(world_pos, 1.0);
```

where `camera.matrix` maps the direction the camera is pointing to `w`, perpendicular directions to `x` and `y` (scaled for field of view) and a constant to `z`.
When applied to all triangles, this this results in each pixel being covered by the closest triangle that its ray passes through.

In order to simulate to additional, indirect optical paths, we can render the triangles of a scene additional times,
tracing a different path (with a different clip-space representation) with each repetition. 
In order to correctly merge the results of these paths with the direct paths,
we render the indirect paths in separate passes to intermediate textures,
combining the results in the fragment shader or lighting shader of the main pass in order to arrive at the pixel's final colour.

This is a very common technique used to render reflections from flat, mirror-like surfaces,
as we can find those path using the same type of perspective transformations as direct paths
(using the same shaders with different parameters), reflecting the camera across the mirror.


Virtual space and refracted paths
---------------------------------

Although the case of reflection above is simple enough to calculate by just reflecting the camera,
when dealing with more complex indirect paths, it is helpful to introduce an additional, intermediate coordinate system: virtual image space.

We define a path's representation in virtual space as the endpoint (in world space) of a straight path
with the same origin (the camera), initial direction, and length as the path in question.
This is closely related to clip space, but somewhat more generic: for all paths (both direct and indirect),

```wgsl
clip_pos = camera.matrix * vec4f(virtual_pos, 1.0);
```
In the case of reflection in a planar mirror discussed above, the virtual position of a point's reflection is
the point's world-space position inverted across the mirror plane,
which is where the reflection appears to be when looking "through" the mirror.
As this inversion operation can also be represented as a matrix,
composing it with the original camera produces the reflected camera discussed above.

Although the transformation from world space to virtual space of a planar reflection is linear and can be represented as a matrix,
nonlinear transformations such as curved reflections and planar refraction are also possible.
Although they do not preserve triangles exactly,
they will render with reasonable accuracy as long as they are smooth at the scale of the triangles being transformed,
which are often approximations of curved geometry themselves.
If excessively-large triangles exist that are meant to be truly flat, they can be broken up using tessellation. 

In the case of planar refraction into flat water, the optical path from the camera to a point
is bent downwards (towards the surface's vertical normal) as it crosses the water's surface.
As this bending is purely vertical, horizontal parallax, which we use as our path length metric to keep the math simple, is unaffected,
putting the paths's position in virtual space (for the this refracted path) directly above the point's world-space position,
somewhere between the point and the water's surface directly above it.

To find the position where the path intersects the water's surface (and thus the depth of the virtual position)
we solve for the point where the path directions match Snell's law of refraction:
```wgsl
sin(angle_from_vertical_above) == ior * sin(angle_from_vertical_below);
```
For a point `world_depth` below the water, and a camera `horiz_dist` away horizontally and `cam_height` above the water,
let `virt_depth` be the depth of the point's virtual position below the water. We then have
```wgsl
1/tan(angle_from_vertical_above) == (cam_height + virt_depth) / horiz_dist;
1/tan(angle_from_vertical_below) == (world_depth / vert_depth) * (cam_height + virt_depth) / horiz_dist;
```
Applying trigonometry, we combine these with Snell's law to get
```wgsl
pow(world_depth / virt_depth, 2) == (pow(ior, 2) - 1) * pow(horiz_dist / (cam_height + virt_depth), 2) + pow(ior, 2);
```
which we can solve to find `virt_depth`. 

Since the analytic solution to this quartic equation is extremely messy, to find `virt_depth` in shader code,
we approximate a solution numerically using Newton's method,
which converges in 4 iterations for points within 500 times the camera height (tested with `bin/test-refract`).
Letting `ior = 1.33` (water's index of refraction) gives us the following function:

```wgsl
fn apparent_depth(dist: f32, eye_height: f32, world_depth: f32) -> f32 {
    let d = abs(world_depth);
    let h = eye_height;
    let x = dist;

    // starting point in correct bucket
    let oi = dist / (d * 0.01 + eye_height);
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
    return world_depth / ratio;
}

fn clip_point(world_pos: vec3f) -> vec4f {
    var z = world_pos.z;
    if PATH_ID == PATH_REFLECT {
        z = -z;
    } else if PATH_ID == PATH_REFRACT {
        let cam_dist = length(world_pos.xy - camera.eye.xy);
        z = apparent_depth(cam_dist, camera.eye.z, world_pos.z);
    }

    let virt_pos = vec4f(world_pos.xy, z, 1.0);
    return camera.matrix * virt_pos;
}
```

By taking the absolute value of world_depth when calculating our virtual depth,
this gives a sensible (although technically non-physical) extrapolation of virtual position for points above water. 
Using this, we can render refracted triangles that cross the water's surface by using this function to find the virtual positions of the vertices, 
then discarding the above-water part of the triangle in the fragment shader.

In order to perform deferred lighting calculations for a refracted image,
we need to recover the world-space position of each fragment, along with the optical path direction where it meets the fragment.
We start with the usual technique of passing the pixel position and depth-buffer value through the inverted camera matrix,
recovering the virtual position of the point and the direction the path leaves the camera.
We can then use Snell's Law to recover the direction where the path meets the point,
then use those two directions of to find the point's worldspace depth.

```wgsl
let virt_pos = (camera.inv_matrix * clip_pos).xyz;
let look_dir_above = normalize(virt_pos - camera.eye);
let look_dir_below = normalize(refract(look_dir_above, vec3f(0.0, 0.0, 1.0), 0.75));
let depth_adj = (look_dir_below.z / look_dir_above.z) * (length(look_dir_above.xy) / length(look_dir_below.xy));
let world_pos = vec3f(virt_pos.xy, depth_adj * virt_pos.z);
```

By using the techniques above and applying some simple lighting, we end up with a refracted image of underwater geometry
(note how further-away objects appear flatter as the angle of incidence becomes shallower).

![a rendered image of underwater geometry](./doc/underwater-buf.jpg)

Ripples and screen-space distortion
-----------------------------------

In order to render refraction from a wavy surface instead of a flat one,
we start with a planar refracted image (plus its corresponding distance buffer) using the technique above
and then sample it using parallax mapping based on the true normals
in order to obtain the transmitted colour for each fragment of the water's surface.

In the lighting shader, when we processing water fragments,
we calculated the actual refracted direction of a path from the camera (using the fragment's true normal),
and then project that direction into planar-refracted virtual space by shrinking its z component by the
same amount that we shrunk the z component of the world-space position to get the virtual position.
In this case, we use Snell's law with `refract` WGSL builtin.

```wgsl
let look_dir_above = normalize(water_world_pos - camera.world_pos);

let refr_dir_planar = refract(look_dir, vec3f(0.0, 0.0, 1.0), 0.75);
let virt_z_ratio = (look_dir.z / refr_dir_planar.z) * (length(refr_dir_planar.xy) / length(look_dir.xy));

let refr_dir_world = refract(look_dir, water_normal, 0.75);
let refr_dir_virt = normalize(vec3f(refr_dir_world.xy, refr_dir_world.z * virt_z_ratio)); 
```

We then find the length of the underwater portion of the planar refracted path by
sampling using the underwater distance buffer at the pixel's location.
(Note that we use `textureSampleLevel` here instead of `textureLoad` as it allows the underwater image to be rendered at reduced resolution.)

```wgsl
// this line is part of recovering water_world_pos, it is included here for clarity
let dist_val = textureLoad(dist_buf, pixel_xy, 0); // a depth buffer stores clip_near / distance
// ... 
let uw_dist_val = textureSampleLevel(water_dist_buf, water_sampler, pixel_uv, 0.0);
let uw_dist_planar = length(water_world_pos - camera.world_pos) * (dist_val / uw_dist_val - 1.0);
```

In order to find the correct point on the underwater image to sample,
we use a form of parallax mapping where we take the planar-refracted underwater distance (`uw_dist_planar`)
and apply it to the virtual refracted direction (`refr_dir_virt`) to find our sampling point in virtual space.
We can then apply the camera matrix to this point,
taking the xy coordinates to get a point on the underwater image to sample.
For small ripples, this tends to be quite close to water's pixel,
minimizing the chance of going off-screen.

```wgsl
let refr_point_virt = water_world_pos + uw_dist_planar * refr__dir_virt; // simple parallax mapping
let refr_point_clip = camera.matrix * vec4f(refr_point_virt, 1.0);
let refr_point_uv = vec2f(0.5, -0.5) * (refr_clip.xy / refr_clip.w) + 0.5;

let trans_color = textureSampleLevel(water_image_buf, water_sampler, refr_point_uv, 0.0).xyz;
```

Combining this transmitted colour with the reflected colour
(also rendered using a virtual space method, then distorted simillarly to the refracted image)
according to a Fresnel factor gives us the fragment's final colour as seen by the camera.
Using the same capture as the underwater image above, this gives us the following final frame.

![rendered image with ripples](./doc/rippled-img.jpg)

Since we used used simple parallax mapping (only sampling the distance buffer once)
to find the correct point on the planar-refracted image,
the ripples are not quite accurate around edges of objects (such as the blue and green pot in the center)
or when showing transmitted geometry nearly parallel to paths from the camera
(in the distance, although this is obscured by reflection due to the shallow angle and high Fresnel factor),
but this effect is close enough to look convincing, especially when the waves are animated.

If additional accuracy is desired (at the cost of additional processing load),
one could instead replace the simple parallax mapping with parallax occlusion mapping,
marching through the underwater distance buffer along the refracted path
to find the true intersection of the refracted path and the underwater scene.
For the purposes of this game, this was not required.

----

By first rendering a planar-refracted underwater image,
using vertex shaders to draw triangles at their virtual positions,
then parallax mapping the resulting image to account for waves,
we have a very efficient way to render a properly refracted underwater scene
without the need for the overhead of raytracing.
When testing this game at 4k resolution on an NVidia mobile 1050Ti,
render times averaged approximately 7ms/frame,
less than half of the 16ms limit for 60fps.

Note
----

Although the game and its source code is licensed under the [Mozilla Public License 2.0](./LICENSE.txt),
the code samples in this README may additionally be used under the MIT OR APACHE-2.0 licenses.