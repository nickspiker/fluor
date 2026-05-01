//! Center-origin, +y-down 2D coordinates in `f32`.
//!
//! `RuVec2` is the layout-layer point/extent type. It is dimensionless in RU space: conversion to pixel coordinates lives in [`Viewport`](crate::Viewport).
//!
//! Storage is `f32` (see `Coord` below). Hardware-native addition and multiplication on every relevant target — same speed as the bespoke compositors fluor replaces. Spirix is welcome in precision-critical app code or via a future `spirix-coord` feature flag, but it is not the default for windowing.

use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Single scalar used for all RU coordinate components. `f32` by default — hardware-native on aarch64 NEON and x86 AVX/SSE.
pub type Coord = f32;

/// 2D vector in RU (relative-unit) space. Origin is the viewport center; +x right, +y down.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct RuVec2 {
    pub x: Coord,
    pub y: Coord,
}

impl RuVec2 {
    pub const ZERO: RuVec2 = RuVec2 { x: 0.0, y: 0.0 };

    #[inline]
    pub const fn new(x: Coord, y: Coord) -> Self {
        Self { x, y }
    }

    #[inline]
    pub const fn splat(v: Coord) -> Self {
        Self { x: v, y: v }
    }
}

impl Add for RuVec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { Self { x: self.x + rhs.x, y: self.y + rhs.y } }
}
impl AddAssign for RuVec2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { self.x += rhs.x; self.y += rhs.y; }
}
impl Sub for RuVec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self { Self { x: self.x - rhs.x, y: self.y - rhs.y } }
}
impl SubAssign for RuVec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { self.x -= rhs.x; self.y -= rhs.y; }
}
impl Neg for RuVec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { Self { x: -self.x, y: -self.y } }
}
impl Mul<Coord> for RuVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, k: Coord) -> Self { Self { x: self.x * k, y: self.y * k } }
}
impl MulAssign<Coord> for RuVec2 {
    #[inline]
    fn mul_assign(&mut self, k: Coord) { self.x *= k; self.y *= k; }
}
impl Div<Coord> for RuVec2 {
    type Output = Self;
    #[inline]
    fn div(self, k: Coord) -> Self { Self { x: self.x / k, y: self.y / k } }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_round_trip() {
        let a = RuVec2::new(1.5, -2.25);
        let b = RuVec2::new(0.5, 0.25);
        let sum = a + b;
        assert_eq!(sum, RuVec2::new(2., -2.));
        assert_eq!(sum - b, a);
        assert_eq!(-a, RuVec2::new(-1.5, 2.25));
    }

    #[test]
    fn zero_is_identity() {
        let v = RuVec2::new(3., -1.);
        assert_eq!(v + RuVec2::ZERO, v);
    }

    #[test]
    fn scalar_mul_div() {
        let v = RuVec2::new(2., -4.);
        assert_eq!(v * 0.5, RuVec2::new(1., -2.));
        assert_eq!(v / 2., RuVec2::new(1., -2.));
    }
}
