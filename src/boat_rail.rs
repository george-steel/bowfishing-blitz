use glam::*;
use std::{f32::consts::TAU, ops::{Add, Mul, Sub}};
use web_time::Instant;

use crate::{camera::{Camera, CameraController, ShadowSettings}, ui::GameState};

// use the compiler to parse static csv
const RAIL_CSV: &[f32] = &include!("rail-path.csv");

pub struct LoopedRail<A> {
    pub points: Box<[A]>
}

impl<A> LoopedRail<A> where
    A: Copy + Add<A,Output=A> + Sub<A,Output=A> + Mul<f32, Output=A> {
    fn get_point(&self, pos: f64) -> A {
        let n = self.points.len();
        let i = pos.floor().rem_euclid(n as f64) as usize;
        let t = pos.rem_euclid(1.0) as f32;
        catmull_rom(t, self.points[(i + n - 1) % n], self.points[i], self.points[(i + 1) % n], self.points[(i + 2) % n])
    }

    pub fn sample(&self, u: f64) -> A {
        self.get_point(u * self.points.len() as f64)
    }

    pub fn sample_dir(&self, u: f64, window: f64) -> A {
        let n = self.points.len() as f64;
        (self.get_point(u * n) - self.get_point(u * n - window))
    }
}

fn catmull_rom<A>(t:f32, a: A, x:A, y:A, b: A) -> A where 
        A: Copy + Add<A,Output=A> + Sub<A,Output=A> + Mul<f32, Output=A> {
    let tx = (y - a) * 0.5;
    let ty = (b - x) * 0.5;
    let p = t * t * (3.0 - 2.0 * t);
    let my = t * t * (t - 1.0);
    let mx = t * (t - 1.0) * (t - 1.0);

    y*p + x*(1.0-p) + tx* mx + ty*my
}

fn rail_points() -> LoopedRail<Vec2> {
    let raw_points: &[[f32; 2]] = bytemuck::cast_slice(RAIL_CSV);
    let points = raw_points.iter().map(|p| {
        let x = p[0] / 10.0 - 60.0;
        let y = -(p[1] / 10.0 - 60.0);
        vec2(x, y)
    }).collect();
    LoopedRail {points}
}

pub struct RailController {
    shadow_settings: ShadowSettings,
    rail: LoopedRail<Vec2>,
    period: f64,
    pub current_time: f64,
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

        let shadow_skew = self.shadow_settings.sun_dir.xy() / self.shadow_settings.sun_dir.z;
        let norm_sun = self.shadow_settings.sun_dir.normalize();
        let sin_above = norm_sun.xy().length();
        let cos_below = (1.0 - sin_above * sin_above / 1.7689).sqrt();
        let refr_sun_dir = vec3(norm_sun.x/1.33, norm_sun.y/1.33, cos_below);
        let shadow_depth_corr = (norm_sun.z * refr_sun_dir.xy().length()) / (refr_sun_dir.z * norm_sun.xy().length());
        
        Camera {
            matrix: mat,
            inv_matrix: mat.inverse(),
            eye: eye,
            clip_near: CLIP_NEAR,
            fb_size, water_fb_size,
            shadow_skew,
            shadow_range_xy: self.shadow_settings.range_xy,
            shadow_range_z: self.shadow_settings.range_z,
            shadow_depth_corr,
            time_s: self.current_time as f32,
            pad: Vec2::ZERO,
        }
    }

    fn eye(&self) -> Vec3 {
        let xy = self.rail.sample(self.current_time / self.period);
        vec3(xy.x, xy.y, EYE_HEIGHT)
    }

    fn look_dir(&self) -> Vec3 {
        let yaw_rad = ((self.yaw + 180.0).rem_euclid(360.0) - 180.0).to_radians();
        let pitch_rad = self.pitch.to_radians();
        let rail_xy = self.rail.sample_dir(self.current_time / self.period, 2.0).normalize();

        let dir_fac = if self.current_time > self.period {
            let dt = (self.current_time - self.period).min(10.0) / 10.0;
            (1.0 - dt * dt * (3.0 - 2.0 * dt)) as f32 // smoothstep
        } else {
            1.0
        };
        let xy = Vec2::from_angle(yaw_rad * dir_fac).rotate(rail_xy);
        let rz = Vec2::from_angle(pitch_rad * dir_fac);
        Vec3::new(xy.x * rz.x, xy.y * rz.x, rz.y)
    }
}

impl RailController {
    pub fn new(shadow_settings: ShadowSettings, now: Instant) -> Self {
        let rail = rail_points();
        log::info!("path has {} points", rail.points.len());

        RailController {
            shadow_settings,
            rail,
            period: GameState::GAME_PERIOD,
            pitch: 0.0,
            yaw: 00.0,
            current_time: 0.0,
            updated_at: now,
            mouse_accum: DVec2::ZERO,
        }
    }

    pub fn reset(&mut self, now: Instant, start_time: f64) {
        self.updated_at = now;
        self.current_time = start_time;
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.mouse_accum = DVec2::ZERO;
    }

    pub fn unpause(&mut self, now: Instant) {
        self.updated_at = now
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
