use glam::*;
use std::{f32::consts::TAU, time::Instant};

use crate::camera::{Camera, CameraController};

// use the compiler to parse static csv
const RAIL_CSV: &[f32] = &include!("rail-path.csv");

fn rail_points() -> Box<[Vec2]> {
    let raw_points: &[[f32; 2]] = bytemuck::cast_slice(RAIL_CSV);
    raw_points.iter().map(|p| {
        let x = p[0] / 10.0 - 60.0;
        let y = -(p[1] / 10.0 - 60.0);
        vec2(x, y)
    }).collect()
}

fn sample_rail(rail: &[Vec2], t: f64) -> Vec2 {
    let x = t.fract() * (rail.len() - 1) as f64;
    let i = x.floor() as usize;
    rail[i].lerp(rail[i + 1], x.fract() as f32)
}

pub struct RailController {
    rail: Box<[Vec2]>,
    period: f64,
    current_time: f64,
    pitch: f32,
    yaw: f32,
    updated_at: Instant,
    mouse_accum: DVec2,
}

const EYE_HEIGHT: f32 = 2.0;
const FOV_Y_DEG: f32 = 60.0;
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
            time_s: self.current_time as f32,
            pad: Vec3::ZERO,
        }
    }

    fn eye(&self) -> Vec3 {
        let xy = sample_rail(&self.rail, self.current_time / self.period);
        vec3(xy.x, xy.y, EYE_HEIGHT)
    }

    fn look_dir(&self) -> Vec3 {
        let yaw_rad = self.yaw.to_radians();
        let pitch_rad = self.pitch.to_radians();
        Vec3::new(pitch_rad.cos() * yaw_rad.cos(), pitch_rad.cos() * yaw_rad.sin(), pitch_rad.sin())
    }
}

impl RailController {
    pub fn new(now: Instant) -> Self {
        let rail = rail_points();
        log::info!("path has {} points", rail.len());

        RailController {
            rail,
            period: 180.0,
            pitch: 0.0,
            yaw: 90.0,
            current_time: 0.0,
            updated_at: now,
            mouse_accum: DVec2::ZERO,
        }
    }

    pub fn mouse(&mut self, dx: f64, dy: f64) {
        self.mouse_accum += dvec2(dx, dy);
    }

    pub fn tick(&mut self, now: Instant) -> f64 {
        let delta_t = (now - self.updated_at).as_secs_f64();
        self.current_time += delta_t;
        self.updated_at = now;
        self.yaw = (self.yaw - ROT_SPEED * self.mouse_accum.x as f32) % 360.0;
        self.pitch = (self.pitch - ROT_SPEED * self.mouse_accum.y as f32).clamp(-MAX_PITCH, MAX_PITCH);
        self.mouse_accum = DVec2::ZERO;
        self.current_time
    }
}
