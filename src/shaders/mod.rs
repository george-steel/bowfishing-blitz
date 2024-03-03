pub const LIGHTING: &str = concat!(
    include_str!("global.wgsl"),
    include_str!("lighting.wgsl"),
);

pub const TERRAIN: &str = concat!(
    include_str!("global.wgsl"),
    include_str!("noise.wgsl"),
    include_str!("terrain.wgsl"),
);

pub const TERRAIN_MAP: &str = concat!(
    include_str!("noise.wgsl"),
    include_str!("terrain_map.wgsl"),
);

pub const MIP: &str = include_str!("mip.wgsl");
