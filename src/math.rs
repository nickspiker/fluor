//! Thin wrappers around `libm` free functions so `f32` math works in `no_std`.
//!
//! In `std` builds these compile down to the same instructions as the inherent methods
//! (the compiler sees through the `libm` call). In `no_std` builds they provide software
//! implementations of the standard math functions.

#[inline] pub fn ceil(x: f32) -> f32 { libm::ceilf(x) }
#[inline] pub fn floor(x: f32) -> f32 { libm::floorf(x) }
#[inline] pub fn sqrt(x: f32) -> f32 { libm::sqrtf(x) }
#[inline] pub fn sin_cos(x: f32) -> (f32, f32) { (libm::sinf(x), libm::cosf(x)) }
#[inline] pub fn atan2(y: f32, x: f32) -> f32 { libm::atan2f(y, x) }
#[inline] pub fn powf(x: f32, y: f32) -> f32 { libm::powf(x, y) }
#[inline] pub fn powi(x: f32, n: i32) -> f32 { libm::powf(x, n as f32) }
#[inline] pub fn fract(x: f32) -> f32 { x - libm::floorf(x) }
#[inline] pub fn rem_euclid(x: f32, y: f32) -> f32 {
    let r = libm::fmodf(x, y);
    if r < 0.0 { r + y } else { r }
}
