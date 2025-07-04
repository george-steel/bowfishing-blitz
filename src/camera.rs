use std::{ops::Rem, time::Instant};
use winit::{event::ElementState, keyboard::{KeyCode, PhysicalKey::{self, *}}};

use glam::*;

// Camera data for GPU use.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    pub matrix: Mat4,
    pub inv_matrix: Mat4,
    pub eye: Vec3,
    pub clip_near: f32,
    pub fb_size: Vec2,
    pub water_fb_size: Vec2,
    pub shadow_skew: Vec2,
    pub shadow_range_xy: f32,
    pub shadow_range_z: f32,
    pub shadow_depth_corr: f32,
    pub time_s: f32,
    pub pad: Vec2,
}

impl Camera {
    pub fn perspective_clipping_planes(&self) -> [Vec4; 4] {
        let zmin = self.clip_near;
        let nw = ((self.inv_matrix * vec4(-zmin, zmin, zmin, zmin)).xyz() - self.eye).normalize();
        let ne = ((self.inv_matrix * vec4(zmin, zmin, zmin, zmin)).xyz() - self.eye).normalize();
        let sw = ((self.inv_matrix * vec4(-zmin, -zmin, zmin, zmin)).xyz() - self.eye).normalize();
        let se = ((self.inv_matrix * vec4(zmin, -zmin, zmin, zmin)).xyz() - self.eye).normalize();

        let n = nw.cross(ne).normalize();
        let w = sw.cross(nw).normalize();
        let s = se.cross(sw).normalize();
        let e = ne.cross(se).normalize();

        [
            (n, -n.dot(self.eye)).into(),
            (s, -s.dot(self.eye)).into(),
            (w, -w.dot(self.eye)).into(),
            (e, -e.dot(self.eye)).into(),
        ]
    }
}

// Tests if a sphere appears in frame. Used for frustum culling
// Subject to false positives due to refraction compensation.
pub fn sphere_visible(planes: [Vec4; 4], center: Vec3, radius: f32) -> bool {
    let above = (center, 1.0).into();
    let below = vec4(center.z, center.y, center.z.max(0.0), 1.0);
    for p in planes {
        if p.dot(above) < -radius && p.dot(below) < -radius {
            return false;
        }
    }
    true
}

pub trait CameraController {
    fn camera(&self, fb_size: Vec2, water_fb_size: Vec2) -> Camera;
    fn eye(&self) -> Vec3;
    fn look_dir(&self) -> Vec3;
}


#[derive(Copy, Clone, Debug)]
pub struct FreeCamSettings {
    pub lin_speed: f32, // units/s
    pub rot_speed: f32, // deg/count
    pub fov_y: f32, // deg
    pub clip_near: f32,
}

impl Default for FreeCamSettings {
    fn default() -> Self {
        FreeCamSettings {
            lin_speed: 5.0,
            rot_speed: 0.05,
            fov_y: 45.0,
            clip_near: 0.1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ShadowSettings {
    pub sun_dir: Vec3,
    pub range_xy: f32,
    pub range_z: f32,
}

#[derive(Clone, Debug)]
struct CameraInputAccum {
    vel_fwd: f32,
    vel_rev: f32,
    vel_left: f32,
    vel_right: f32,
    vel_up: f32,
    vel_down: f32,
    rot_x: f32,
    rot_y: f32,
}

impl CameraInputAccum {
    fn new() -> Self {
        CameraInputAccum {
            vel_fwd: 0.0,
            vel_rev: 0.0,
            vel_left: 0.0,
            vel_right: 0.0,
            vel_up: 0.0,
            vel_down: 0.0,
            rot_x: 0.0,
            rot_y: 0.0,
        }
    }
    //fn clear(&mut self) {
    //    *self = Self::new();
    //}

    fn vel_long(&self) -> f32 {
        self.vel_fwd - self.vel_rev
    }
    fn vel_trans(&self) -> f32 {
        self.vel_left - self.vel_right
    }
    fn vel_vert(&self) -> f32 {
        self.vel_up - self.vel_down
    }

    fn key(&mut self, key: PhysicalKey, state: ElementState) {
        let new_vel = match state {
            ElementState::Pressed => 1.0,
            ElementState::Released => 0.0,
        };

        match key {
            Code(KeyCode::KeyW) => {self.vel_fwd = new_vel}
            Code(KeyCode::KeyS) => {self.vel_rev = new_vel}
            Code(KeyCode::KeyA) => {self.vel_left = new_vel}
            Code(KeyCode::KeyD) => {self.vel_right = new_vel}
            Code(KeyCode::Space) => {self.vel_up = new_vel}
            Code(KeyCode::ShiftLeft) => {self.vel_down = new_vel}
            Code(KeyCode::Escape) => {*self = Self::new()}
            _ => {}
        }
    }

    fn mouse(&mut self, dx: f64, dy: f64) {
        self.rot_x -= dx as f32;
        self.rot_y -= dy as f32;
    }

    fn take_rot(&mut self) -> (f32, f32) {
        let xy = (self.rot_x, self.rot_y);
        self.rot_x = 0.0;
        self.rot_y = 0.0;
        xy
    }
}

/*
#[derive(Clone, Debug)]
pub struct FreeCam {
    pub settings: FreeCamSettings,
    pub eye_pos: Vec3,
    pub yaw: f32, // radians, 0 is towards +X
    pub pitch: f32, // radians, 0 is horizontal
    pub created_at: Instant,
    pub updated_at: Instant,
    accum: CameraInputAccum,
}

impl FreeCam {
    pub const MAX_PITCH: f32 = 88.0;

    pub fn new(settings: FreeCamSettings, eye: Vec3, yaw: f32, now: Instant) -> Self {
        FreeCam {
            settings, eye_pos: eye, yaw,
            pitch: 0.0,
            created_at: now,
            updated_at: now,
            accum: CameraInputAccum::new(),
        }
    }

    pub fn tick(&mut self, now: Instant) {
        let dt = now - self.updated_at;
        self.updated_at = now;

        let yaw_rad = self.yaw.to_radians();
        let long_dir = vec3(yaw_rad.cos(), yaw_rad.sin(), 0.0);
        let trans_dir = vec3(-yaw_rad.sin(), yaw_rad.cos(), 0.0);
        let vert_dir = vec3(0.0, 0.0, 1.0);

        let vel =
            self.accum.vel_long() * long_dir +
            self.accum.vel_trans() * trans_dir +
            self.accum.vel_vert() * vert_dir;
        self.eye_pos += dt.as_secs_f32() * self.settings.lin_speed * vel;

        let (dx, dy) = self.accum.take_rot();

        self.yaw = (self.yaw + self.settings.rot_speed * dx).rem(360.0);
        self.pitch = (self.pitch + self.settings.rot_speed * dy).clamp(-Self::MAX_PITCH, Self::MAX_PITCH);
    }

    pub fn key(&mut self, key: PhysicalKey, state: ElementState) {
        self.accum.key(key, state);
    }
    pub fn mouse(&mut self, dx: f64, dy: f64) {
        self.accum.mouse(dx, dy);
    }
}

impl CameraController for FreeCam {
    fn camera(&self, fb_size: Vec2, water_fb_size: Vec2) -> Camera {
        let aspect_ratio = fb_size.x / fb_size.y;
        let mat = Mat4::perspective_infinite_reverse_rh(self.settings.fov_y.to_radians(), aspect_ratio, self.settings.clip_near,)
            * Mat4::look_to_rh(self.eye_pos, self.look_dir(), Vec3::new(0.0, 0.0, 1.0));
        if mat.determinant() == 0.0 {
            panic!("Singular camera matrix: {:?}", mat);
        }
        Camera {
            matrix: mat,
            inv_matrix: mat.inverse(),
            eye: self.eye_pos,
            clip_near: self.settings.clip_near,
            fb_size, water_fb_size,
            time_s: (self.updated_at - self.created_at).as_secs_f32(),
            pad: 0.0,
        }
    }

    fn eye(&self) -> Vec3 {
        self.eye_pos
    }

    fn look_dir(&self) -> Vec3 {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        Vec3::new(pitch_rad.cos() * yaw_rad.cos(), pitch_rad.cos() * yaw_rad.sin(), pitch_rad.sin())
    }
}
*/