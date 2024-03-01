
pub const lighting: &str = concat!(
    include_str!("global.wgsl"),
    include_str!("lighting.wgsl"),
);

pub const terrain: &str = concat!(
    include_str!("global.wgsl"),
    include_str!("noise.wgsl"),
    include_str!("terrain.wgsl"),
);

pub const terrain_map: &str = concat!(
    include_str!("noise.wgsl"),
    include_str!("terrain_map.wgsl"),
);
