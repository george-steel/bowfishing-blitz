use glam::*;
use std::{f32::consts::TAU, time::Instant};

use crate::camera::{Camera, CameraController};



pub struct RailController {
    center: Vec2,
    radius: f32,
    period: f64,
    pitch: f32,
    yaw: f32,
    created_at: Instant,
    updated_at: Instant,
    mouse_accum: DVec2,
}

const EYE_HEIGHT: f32 = 2.0;
const FOV_Y_DEG: f32 = 45.0;
const CLIP_NEAR: f32 = 0.1;
const MAX_PITCH: f32 = 88.0;
const ROT_SPEED: f32 = 0.05;

impl CameraController for RailController {
    fn camera(&self, fb_size: Vec2, water_fb_size: Vec2) -> crate::camera::Camera {
        let eye = self.eye();
        let aspect_ratio = fb_size.x / fb_size.y;
        let mat = Mat4::perspective_infinite_reverse_rh(FOV_Y_DEG.to_radians(), aspect_ratio, CLIP_NEAR)
            * Mat4::look_to_rh(eye, self.look_dir(), Vec3::new(0.0, 0.0, 1.0));
        if mat.determinant() == 0.0 {
            panic!("Singular camera matrix: {:?}", mat);
        }
        Camera {
            matrix: mat,
            inv_matrix: mat.inverse(),
            eye: eye,
            clip_near: CLIP_NEAR,
            fb_size, water_fb_size,
            time_s: (self.updated_at - self.created_at).as_secs_f32(),
            pad: Vec3::ZERO,
        }
    }

    fn eye(&self) -> Vec3 {
        let t = (self.updated_at - self.created_at).as_secs_f64();
        let theta = ((t / self.period).fract() as f32) * TAU;
        let x = self.center.x - self.radius * theta.cos();
        let y = self.center.y + self.radius * theta.sin();
        vec3(x, y, EYE_HEIGHT)
    }

    fn look_dir(&self) -> Vec3 {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        Vec3::new(pitch_rad.cos() * yaw_rad.cos(), pitch_rad.cos() * yaw_rad.sin(), pitch_rad.sin())
    }
}

impl RailController {
    pub fn new(now: Instant) -> Self {
        RailController {
            center: vec2(0.0, 0.0),
            radius: 25.0,
            period: 60.0,
            pitch: 0.0,
            yaw: 90.0,
            created_at: now,
            updated_at: now,
            mouse_accum: DVec2::ZERO,
        }
    }

    pub fn mouse(&mut self, dx: f64, dy: f64) {
        self.mouse_accum += dvec2(dx, dy);
    }

    pub fn tick(&mut self, now: Instant) {
        self.updated_at = now;
        self.yaw = (self.yaw - ROT_SPEED * self.mouse_accum.x as f32) % 360.0;
        self.pitch = (self.pitch - ROT_SPEED * self.mouse_accum.y as f32).clamp(-MAX_PITCH, MAX_PITCH);
        self.mouse_accum = DVec2::ZERO;
    }
}
