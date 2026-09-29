//! Mathematical routines.

use num::complex::Complex32;
use std::{f32::consts::PI as PI_f32, f64::consts::PI};

/// Wraps a phase to `[-pi, pi)`.
pub fn phase_wrap_f64(phase: f64) -> f64 {
    // The % operator in Rust preserves the sign of the left operand, so we need
    // to use rem_euclid here.
    (phase + PI).rem_euclid(2.0 * PI) - PI
}

/// Wraps a phase to `[-pi, pi)` assuming that the phase is already in `[-3*pi, 3*pi)`.
pub fn phase_wrap_once_f64(phase: f64) -> f64 {
    if phase >= PI {
        phase - 2.0 * PI
    } else if phase < -PI {
        phase + 2.0 * PI
    } else {
        phase
    }
}

/// Wraps a phase to `[-pi, pi)` assuming that the phase is already in `[-3*pi, 3*pi)`.
pub fn phase_wrap_once_f32(phase: f32) -> f32 {
    if phase >= PI_f32 {
        phase - 2.0 * PI_f32
    } else if phase < -PI_f32 {
        phase + 2.0 * PI_f32
    } else {
        phase
    }
}

/// Algebraic operators for complex numbers.
///
/// This trait is used to implement the algebraic operators from floating point
/// types (new in Rust 1.98.0) for complex numbers.
pub trait AlgebraicComplex {
    /// Scalar floating point type corresponding to the complex type.
    type Scalar;
    /// Algebraic sum.
    fn algebraic_add(self, rhs: Self) -> Self;
    /// Algebraic subtraction.
    fn algebraic_sub(self, rhs: Self) -> Self;
    /// Algebraic product.
    fn algebraic_mul(self, rhs: Self) -> Self;
    /// Algebraic product conjugate.
    fn algebraic_mul_conj(self, rhs: Self) -> Self;
    /// Algebraic division.
    fn algebraic_div(self, rhs: Self) -> Self;
    /// Algebraic division by scalar.
    fn algebraic_div_scalar(self, rhs: Self::Scalar) -> Self;
    /// Algebraic norm squared.
    fn algebraic_norm_sqr(self) -> Self::Scalar;
}

impl AlgebraicComplex for Complex32 {
    type Scalar = f32;

    fn algebraic_add(self, rhs: Self) -> Self {
        Complex32::new(self.re.algebraic_add(rhs.re), self.im.algebraic_add(rhs.im))
    }

    fn algebraic_sub(self, rhs: Self) -> Self {
        Complex32::new(self.re.algebraic_sub(rhs.re), self.im.algebraic_sub(rhs.im))
    }

    fn algebraic_mul(self, rhs: Self) -> Self {
        // Naïve implementation with 4 products
        Complex32::new(
            self.re
                .algebraic_mul(rhs.re)
                .algebraic_sub(self.im.algebraic_mul(rhs.im)),
            self.re
                .algebraic_mul(rhs.im)
                .algebraic_add(self.im.algebraic_mul(rhs.re)),
        )
    }

    fn algebraic_mul_conj(self, rhs: Self) -> Self {
        // Naïve implementation with 4 products
        Complex32::new(
            self.re
                .algebraic_mul(rhs.re)
                .algebraic_add(self.im.algebraic_mul(rhs.im)),
            self.im
                .algebraic_mul(rhs.re)
                .algebraic_sub(self.re.algebraic_mul(rhs.im)),
        )
    }

    fn algebraic_div(self, rhs: Self) -> Self {
        // Calculate as z/w = z*conj(w) / |w|^2
        self.algebraic_mul_conj(rhs)
            .algebraic_div_scalar(rhs.algebraic_norm_sqr())
    }

    fn algebraic_div_scalar(self, rhs: f32) -> Self {
        Complex32::new(self.re.algebraic_div(rhs), self.im.algebraic_div(rhs))
    }

    fn algebraic_norm_sqr(self) -> f32 {
        self.re
            .algebraic_mul(self.re)
            .algebraic_add(self.im.algebraic_mul(self.im))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn phase_wrap_f64(x in -PI..PI, wraps in -1024i32..=1024) {
            let tolerance = 1e-10;
            let wrapped = super::phase_wrap_f64(x + wraps as f64 * 2.0 * PI);
            let diff = (wrapped - x).abs();
            assert!(diff <= tolerance, "|{wrapped} - {x}| = {diff:e} > tolerance");
        }
    }

    proptest! {
        #[test]
        fn phase_wrap_once_f64(x in -PI..PI, wraps in -1..=1) {
            let tolerance = 1e-14;
            let wrapped = super::phase_wrap_once_f64(x + wraps as f64 * 2.0 * PI);
            let diff = (wrapped - x).abs();
            assert!(diff <= tolerance, "|{wrapped} - {x}| = {diff:e} > tolerance");
        }
    }

    proptest! {
        #[test]
        fn phase_wrap_once_f32(x in -PI_f32..PI_f32, wraps in -1..=1) {
            let tolerance = 1e-6;
            let wrapped = super::phase_wrap_once_f32(x + wraps as f32 * 2.0 * PI_f32);
            let diff = (wrapped - x).abs();
            assert!(diff <= tolerance, "|{wrapped} - {x}| = {diff:e} > tolerance");
        }
    }

    prop_compose! {
        fn complex32_1x1()(a in -1.0_f32..=1.0, b in -1.0_f32..=1.0) -> Complex32 {
            Complex32::new(a, b)
        }
    }

    proptest! {
        #[test]
        fn algebraic_add(z in complex32_1x1(), w in complex32_1x1()) {
            let tolerance = 1e-6;
            let expected = z + w;
            assert!((z.algebraic_add(w) - expected).norm() <= tolerance);
        }
    }

    proptest! {
        #[test]
        fn algebraic_sub(z in complex32_1x1(), w in complex32_1x1()) {
            let tolerance = 1e-6;
            let expected = z - w;
            assert!((z.algebraic_sub(w) - expected).norm() <= tolerance);
        }
    }

    proptest! {
        #[test]
        fn algebraic_mul(z in complex32_1x1(), w in complex32_1x1()) {
            let tolerance = 1e-6;
            let expected = z * w;
            assert!((z.algebraic_mul(w) - expected).norm() <= tolerance);
        }
    }

    proptest! {
        #[test]
        fn algebraic_mul_conj(z in complex32_1x1(), w in complex32_1x1()) {
            let tolerance = 1e-6;
            let expected = z * w.conj();
            assert!((z.algebraic_mul_conj(w) - expected).norm() <= tolerance);
        }
    }

    prop_compose! {
        fn complex32_ring()(r in -0.1..=10.0, theta in -PI..PI) -> Complex32 {
            Complex32::new((r * theta.cos()) as f32, (r * theta.sin()) as f32)
        }
    }

    proptest! {
        #[test]
        fn algebraic_div(z in complex32_1x1(), w in complex32_ring()) {
            let tolerance = 1e-6;
            let expected = z / w;
            assert!((z.algebraic_div(w) - expected).norm() <= tolerance);
        }
    }

    proptest! {
        #[test]
        fn algebraic_div_scalar(z in complex32_1x1(), x in 0.1_f32..10.0) {
            let tolerance = 1e-6;
            let expected = z / x;
            assert!((z.algebraic_div_scalar(x) - expected).norm() <= tolerance);
        }
    }

    proptest! {
        #[test]
        fn algebraic_norm_sqr(z in complex32_1x1()) {
            let tolerance = 1e-6;
            let expected = z.norm_sqr();
            assert!((z.algebraic_norm_sqr() - expected).abs() <= tolerance);
        }
    }
}
