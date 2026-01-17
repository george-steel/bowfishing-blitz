pub const INCLUDES: &[(&str, &str)] = &[
    ("global.wgsl", include_str!("global.wgsl")),
    ("noise.wgsl", include_str!("noise.wgsl")),
];

pub const LIGHTING: &str = include_str!("lighting.wgsl");

pub const TERRAIN: &str = include_str!("terrain.wgsl");

pub const TERRAIN_MAP: &str = concat!(
    include_str!("noise.wgsl"),
    include_str!("terrain_map.wgsl"),
);

pub const MIP: &str = include_str!("mip.wgsl");

pub const ARROWS: &str = include_str!("arrows.wgsl");

pub const TARGETS: &str = include_str!("targets.wgsl");

pub const UI: &str = include_str!("ui.wgsl");