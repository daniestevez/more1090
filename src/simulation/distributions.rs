//! Distributions of simulation parameters.
//!
//! This module defines random distributions that are used to generate
//! simulation parameters.

use super::pulse_shape::MAX_CLOCK_RATE_ERROR;
use crate::{constants::dsp::SAMPLE_RATE_F64, math};
use anyhow::Result;
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};
use std::{
    f64::consts::PI,
    ops::{Bound, Range, RangeBounds},
};

/// Distributions for a message configuration.
///
/// This struct gives the distributions that are sampled to choose the
/// parameters of a message.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MessageDistributions {
    /// CN0 distribution.
    ///
    /// The CN0 is referred to the peak power of an ideal rectangular pulse. It
    /// does not take into account the 50% duty cycle of ADS-B or the pulse
    /// shaping that is used in the simulation.
    pub cn0: CN0Distribution,
    /// Carrier frequency offset distribution.
    pub carrier_frequency_offset: CarrierFrequencyOffsetDistribution,
    /// Carrier phase distribution.
    pub carrier_phase: CarrierPhaseDistribution,
    /// Symbol clock error distribution.
    pub symbol_clock_error: SymbolClockErrorDistribution,
}

/// CN0 distribution.
///
/// This enum defines the possible CN0 distributions.
///
/// The CN0 is referred to the peak power of an ideal rectangular pulse. It
/// does not take into account the 50% duty cycle of ADS-B or the pulse
/// shaping that is used in the simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CN0Distribution {
    /// Constant (deterministic) CN0.
    Constant(ConstantCN0),
    /// Distribution according to a maximum CN0 and free space path loss over a
    /// given distance range.
    DistanceRange(DistanceRangeCN0),
}

impl Default for CN0Distribution {
    fn default() -> CN0Distribution {
        // TODO: make this values match realistic ground-based recordings
        CN0Distribution::DistanceRange(DistanceRangeCN0::new(100.0, 10.0..300.0).unwrap())
    }
}

impl Distribution<f64> for CN0Distribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            CN0Distribution::Constant(d) => d.sample(rng),
            CN0Distribution::DistanceRange(d) => d.sample(rng),
        }
    }
}

/// Constant CN0 distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantCN0 {
    cn0: f64,
}

impl ConstantCN0 {
    /// Creates a constant CN0 distribution.
    ///
    /// The CN0 value is given in dB·Hz units. An error is returned if the value
    /// is invalid (for instance an `f64` which is not finite).
    pub fn new(cn0: f64) -> Result<ConstantCN0> {
        anyhow::ensure!(cn0.is_finite(), "CN0 {cn0} must be finite");
        Ok(ConstantCN0 { cn0 })
    }

    /// Returns the constant CN0 in dB·Hz set in the distribution.
    pub fn cn0(&self) -> f64 {
        self.cn0
    }
}

impl Distribution<f64> for ConstantCN0 {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
        self.cn0()
    }
}

/// Distribution according to a maximum CN0 and free space path loss over a
/// given distance range.
///
/// This distribution models CN0 as follows. Transmitters are assumed to be
/// uniformly distributed over a disk of radius `sqrt(max_distance**2 -
/// min_distance**2)` that sits in the horizontal plane `z = 0`. The
/// receiver is assumed to be located over the center of the disk, at height
/// `z = min_distance`. In this way, the range of distances between the
/// transmitters and the receiver is `[min_distance, max_distance]`. The CN0
/// for a transmitter at distance `min_distance` is assumed to be
/// `max_cn0`. The CN0 for transmitters at any other distance `d` is assumed
/// to scale proportionally according to free space path loss, giving
/// `max_cn0 + 20*log10(min_distance) - 20*log10(d)` (in dB units).
///
/// This simplified model is roughly representative of the following two
/// situations:
///
/// - A receiver located on the ground, with transmitters flying at cruise
///   altitude. In this case `min_distance` is the typical cruise altitude and
///   `max_distance` is the maximum distance at which signals can be received
///   (essentially the radio horizon).
///
/// - A receiver located on a satellite in a circular Earth orbit (specially
///   LEO). In this case `min_distance` is the altitude of the satellite minus
///   the typical cruise altitude and `max_distance` is the maximum distance at
///   which signals can be received (essentially the radius of the satellite
///   footprint).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceRangeCN0 {
    max_cn0: f64,
    min_distance: f64,
    max_distance: f64,
    // CN0 normalized to distance one. Used to save some calculations when
    // sampling.
    distance_corrected_cn0: f64,
    // min_distance**2. Used to save some calculations when sampling.
    min_distance_squared: f64,
    // max_distance**2 - min_distance**2. Used to save some calculations when sampling.
    radius_squared: f64,
}

impl DistanceRangeCN0 {
    /// Creates a distance range CN0 distribution.
    ///
    /// The parameters are the following:
    ///
    /// - `max_cn0` is the maximum CN0 in dB·Hz. It corresponds to transmitters
    ///   at distance `min_distance`.
    ///
    /// - `distance_range` is a range (typically `min_distance..max_distance`)
    ///   that specifies the distance range. The units of distance do not
    ///   matter, but they need to be the same for both `min_distance` and
    ///   `max_distance`. Ranges of the form `..max_distance`, `min_distance..`
    ///   and `..` are invalid. There is no distinction between including or
    ///   excluding `max_distance` in the range (that is, between
    ///   `min_distance..max_distance` and `min_distance..=max_distance`).
    ///
    /// This function returns an error if the parameters are invalid.
    pub fn new<B: RangeBounds<f64>>(max_cn0: f64, distance_range: B) -> Result<DistanceRangeCN0> {
        anyhow::ensure!(max_cn0.is_finite(), "Maximum CN0 {max_cn0} must be finite");
        let min_distance = match distance_range.start_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => {
                anyhow::bail!("The start bound of the distance range must be bounded")
            }
        };
        let max_distance = match distance_range.end_bound() {
            Bound::Included(x) => *x,
            Bound::Excluded(x) => *x,
            Bound::Unbounded => {
                anyhow::bail!("The end bound of the distance range must be bounded")
            }
        };
        anyhow::ensure!(
            min_distance.is_finite(),
            "The minimum distance {min_distance} must be finite"
        );
        anyhow::ensure!(
            max_distance.is_finite(),
            "The maximum distance {max_distance} must be finite"
        );
        anyhow::ensure!(
            min_distance > 0.0,
            "The minimum distance {min_distance} must be positive"
        );
        anyhow::ensure!(
            max_distance >= min_distance,
            "The maximum distance {max_distance} must be greater than or equal to \
             the minimum distance {min_distance}"
        );

        let distance_corrected_cn0 = max_cn0 + 20.0 * min_distance.log10();
        let min_distance_squared = min_distance * min_distance;
        anyhow::ensure!(
            min_distance_squared > 0.0,
            "The minimum distance squared must be positive (min_distance was too small)"
        );
        let radius_squared = max_distance * max_distance - min_distance_squared;
        Ok(DistanceRangeCN0 {
            max_cn0,
            min_distance,
            max_distance,
            distance_corrected_cn0,
            min_distance_squared,
            radius_squared,
        })
    }

    /// Returns the maximum CN0 set in the distribution.
    pub fn max_cn0(&self) -> f64 {
        self.max_cn0
    }

    /// Returns the minimum distance set in the distribution.
    pub fn min_distance(&self) -> f64 {
        self.min_distance
    }

    /// Returns the maximum distance set in the distribution.
    pub fn max_distance(&self) -> f64 {
        self.max_distance
    }

    /// Returns the distance range set in the distribution.
    ///
    /// This function always returns an exclusive range
    /// `min_distance..max_distance`, regardless of how the distribution was
    /// constructed.
    pub fn distance_range(&self) -> Range<f64> {
        self.min_distance()..self.max_distance()
    }
}

impl Distribution<f64> for DistanceRangeCN0 {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        // The CDF for the distance between the satellite and transmitter is
        //
        // F_d(t) = P[d <= t] = (t**2 - min_distance**2) / (max_distance**2 - min_distance**2).
        //
        // A random value following this CDF can be obtained by drawing
        // u from a Uniform([0, 1]) distribution and computing d = F_d^{-1}(u). This gives
        //
        // d = sqrt(min_distance**2 + (max_distance**2 - min_distance**2) * u).

        let u: f64 = rng.sample(rand::distr::StandardUniform);
        // Computing d**2 allows us to avoid computing a sqrt(), because we only
        // need log10(d).
        let d_squared = self.min_distance_squared + self.radius_squared * u;
        self.distance_corrected_cn0 - 10.0 * d_squared.log10()
    }
}

/// Carrier frequency offset distribution.
///
/// This enum defines the possible distributions for carrier frequency offset.
///
/// The [`Default`] implementation gives a clamped normal with a standard
/// deviation of 100 kHz and clamping to +/-1 MHz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CarrierFrequencyOffsetDistribution {
    /// Constant (deterministic) carrier frequency offset.
    Constant(ConstantCarrierFrequencyOffset),
    /// Normal with standard deviation `sigma` clamped to `[-max, max]`.
    ClampedNormal(ClampedNormalCarrierFrequencyOffset),
}

impl Distribution<f64> for CarrierFrequencyOffsetDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            CarrierFrequencyOffsetDistribution::Constant(d) => d.sample(rng),
            CarrierFrequencyOffsetDistribution::ClampedNormal(d) => d.sample(rng),
        }
    }
}

impl Default for CarrierFrequencyOffsetDistribution {
    fn default() -> CarrierFrequencyOffsetDistribution {
        CarrierFrequencyOffsetDistribution::ClampedNormal(
            ClampedNormalCarrierFrequencyOffset::new(100e3, 1e6).unwrap(),
        )
    }
}

/// Constant carrier frequency offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantCarrierFrequencyOffset {
    carrier_frequency_offset: f64,
}

impl ConstantCarrierFrequencyOffset {
    /// Creates a constant carrier frequency offset distribution.
    ///
    /// The carrier frequency offset is given in Hz. An error is returned if the
    /// value is invalid (for instance an `f64` which is not finite or exceeds
    /// Nyquist).
    pub fn new(carrier_frequency_offset: f64) -> Result<ConstantCarrierFrequencyOffset> {
        anyhow::ensure!(
            carrier_frequency_offset.is_finite(),
            "Carrier frequency offset {carrier_frequency_offset} must be finite"
        );
        let nyquist = 0.5 * SAMPLE_RATE_F64;
        anyhow::ensure!(
            carrier_frequency_offset.abs() <= 0.5 * SAMPLE_RATE_F64,
            "Carrier frequency offset {carrier_frequency_offset} must not be larger \
             than Nyquist frequency {nyquist}"
        );
        Ok(ConstantCarrierFrequencyOffset {
            carrier_frequency_offset,
        })
    }

    /// Returns the constant carrier frequency offset in Hz set in the
    /// distribution.
    pub fn carrier_frequency_offset(&self) -> f64 {
        self.carrier_frequency_offset
    }
}

impl Distribution<f64> for ConstantCarrierFrequencyOffset {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
        self.carrier_frequency_offset()
    }
}

/// Clamped normal carrier frequency offset.
///
/// A carrier frequency offset that is a normal distribution with standard
/// deviation `sigma` clamped to `[-max, max]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClampedNormalCarrierFrequencyOffset {
    sigma: f64,
    max: f64,
}

impl ClampedNormalCarrierFrequencyOffset {
    /// Creates a clamped normal carrier frequency offset distribution.
    ///
    /// The parameter `sigma` is the standard deviation of the normal
    /// distribution in Hz. The parameter `max` clamps the normal to the
    /// interval `[-max, max]` (it is also given in Hz). An error is returned if
    /// the parameters are invalid (for instance an `f64` which is not finite, a
    /// negative `sigma` or `max`, or a `max` that exceeds Nyquist).
    pub fn new(sigma: f64, max: f64) -> Result<ClampedNormalCarrierFrequencyOffset> {
        anyhow::ensure!(
            sigma.is_finite(),
            "Clamped normal sigma {sigma} must be finite"
        );
        anyhow::ensure!(max.is_finite(), "Clamped normal max {max} must be finite");
        anyhow::ensure!(
            sigma >= 0.0,
            "Clamped normal sigma {sigma} must be non-negative"
        );
        anyhow::ensure!(max >= 0.0, "Clamped normal max {max} must be non-negative");
        let nyquist = 0.5 * SAMPLE_RATE_F64;
        anyhow::ensure!(
            max <= 0.5 * SAMPLE_RATE_F64,
            "Clamped normal max {max} must not be larger than Nyquist frequency {nyquist}"
        );
        Ok(ClampedNormalCarrierFrequencyOffset { sigma, max })
    }

    /// Returns the standard deviation in Hz set in the normal distribution.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Returns the clamping value `max` in Hz set in the distribution.
    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Distribution<f64> for ClampedNormalCarrierFrequencyOffset {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        (rng.sample::<f64, _>(StandardNormal) * self.sigma()).clamp(-self.max(), self.max())
    }
}

/// Carrier phase distribution.
///
/// This enum defines the possible distributions for carrier phase.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CarrierPhaseDistribution {
    /// Constant (deterministic) carrier phase.
    Constant(ConstantCarrierPhase),
    /// Uniform carrier phase in [-pi, pi) radians.
    #[default]
    Uniform,
}

impl Distribution<f64> for CarrierPhaseDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            CarrierPhaseDistribution::Constant(d) => d.sample(rng),
            CarrierPhaseDistribution::Uniform => rng.random_range(-PI..PI),
        }
    }
}

/// Constant carrier phase distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantCarrierPhase {
    carrier_phase: f64,
}

impl ConstantCarrierPhase {
    /// Creates a constant carrier phase distribution.
    ///
    /// The carrier phase is given in radians and it is reduced internally to
    /// [-pi, pi). An error is returned if the value is invalid (for instance an
    /// `f64` which is not finite).
    pub fn new(carrier_phase: f64) -> Result<ConstantCarrierPhase> {
        anyhow::ensure!(
            carrier_phase.is_finite(),
            "Carrier phase {carrier_phase} must be finite"
        );
        let carrier_phase = math::phase_wrap_f64(carrier_phase);
        Ok(ConstantCarrierPhase { carrier_phase })
    }

    /// Returns the constant carrier phase in radians set in the distribution.
    pub fn carrier_phase(&self) -> f64 {
        self.carrier_phase
    }
}

impl Distribution<f64> for ConstantCarrierPhase {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
        self.carrier_phase()
    }
}

/// Symbol clock error distribution.
///
/// This enum defines the possible distributions for symbol clock frequency
/// error.
///
/// The [`Default`] implementation gives a uniform over [-50e-6, 50e-6] parts
/// per one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SymbolClockErrorDistribution {
    /// Constant (deterministic) symbol clock error.
    Constant(ConstantSymbolClockError),
    /// Uniform symbol clock error in [-max, max].
    Uniform(UniformSymbolClockError),
}

impl Distribution<f64> for SymbolClockErrorDistribution {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        match self {
            SymbolClockErrorDistribution::Constant(d) => d.sample(rng),
            SymbolClockErrorDistribution::Uniform(d) => d.sample(rng),
        }
    }
}

impl Default for SymbolClockErrorDistribution {
    fn default() -> SymbolClockErrorDistribution {
        SymbolClockErrorDistribution::Uniform(UniformSymbolClockError::new(50e-6).unwrap())
    }
}

/// Constant symbol clock error distribution
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantSymbolClockError {
    symbol_clock_error: f64,
}

impl ConstantSymbolClockError {
    /// Creates a constant symbol clock error distribution.
    ///
    /// The symbol clock error is given in parts per one. An error is returned
    /// if the value is invalid (for instance an `f64` which is not finite or
    /// too large).
    pub fn new(symbol_clock_error: f64) -> Result<ConstantSymbolClockError> {
        anyhow::ensure!(
            symbol_clock_error.is_finite(),
            "Symbol clock error {symbol_clock_error} must be finite"
        );
        anyhow::ensure!(
            symbol_clock_error.abs() <= MAX_CLOCK_RATE_ERROR,
            "Symbol clock error {symbol_clock_error} must not be larger (in absolute value) \
             than {MAX_CLOCK_RATE_ERROR}"
        );
        Ok(ConstantSymbolClockError { symbol_clock_error })
    }

    /// Returns the symbol clock error in parts per one set in the distribution.
    pub fn symbol_clock_error(&self) -> f64 {
        self.symbol_clock_error
    }
}

impl Distribution<f64> for ConstantSymbolClockError {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
        self.symbol_clock_error()
    }
}

/// Uniform symbol clock error distribution.
///
/// This is a symbol clock error that is uniformly distributed over the interval
/// `[-max, max]`, where `max` is given in parts per one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformSymbolClockError {
    max: f64,
}

impl UniformSymbolClockError {
    /// Creates a uniform symbol clock error distribution.
    ///
    /// The maximum symbol clock error is given in parts per one. An error is returned
    /// if the value is invalid (for instance an `f64` which is not finite or
    /// too large).
    pub fn new(max: f64) -> Result<UniformSymbolClockError> {
        anyhow::ensure!(
            max.is_finite(),
            "Maximum symbol clock error {max} must be finite"
        );
        anyhow::ensure!(
            max >= 0.0,
            "Maximum symbol clock error {max} must be non-negative"
        );
        anyhow::ensure!(
            max <= MAX_CLOCK_RATE_ERROR,
            "Maximum symbol clock error {max} must not be larger than {MAX_CLOCK_RATE_ERROR}"
        );
        Ok(UniformSymbolClockError { max })
    }

    /// Returns the maximum symbol clock error in parts per one set in the
    /// distribution.
    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Distribution<f64> for UniformSymbolClockError {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.random_range(-self.max()..=self.max())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::num::f64::{NEGATIVE, POSITIVE, SUBNORMAL, ZERO};
    use proptest::prelude::*;

    fn any_cn0() -> proptest::num::f64::Any {
        POSITIVE | NEGATIVE | ZERO | SUBNORMAL
    }

    macro_rules! assert_samples_constant {
        ($d:expr, $constant:expr) => {{
            let mut rng = rand::rng();
            for _ in 0..1024 {
                let sample = $d.sample(&mut rng);
                assert_eq!(
                    sample, $constant,
                    "distribution does not sample expected constant {} (sampled {})",
                    $constant, sample,
                );
            }
        }};
    }

    #[test]
    fn constant_cn0_rejects_invalid() {
        assert!(ConstantCN0::new(f64::NAN).is_err());
        assert!(ConstantCN0::new(f64::INFINITY).is_err());
        assert!(ConstantCN0::new(f64::NEG_INFINITY).is_err());
    }

    proptest! {
        #[test]
        fn constant_cn0_accepts_valid_and_samples_constant(cn0 in any_cn0()) {
            let d = ConstantCN0::new(cn0).unwrap();
            assert_eq!(d.cn0(), cn0);
            assert_samples_constant!(d, cn0);
        }
    }

    #[test]
    fn distance_range_rejects_invalid() {
        macro_rules! test_case {
            ($case_name:expr, $max_cn0:expr, $range:expr) => {
                assert!(
                    DistanceRangeCN0::new($max_cn0, $range).is_err(),
                    "{} should be rejected",
                    $case_name
                );
            };
        }
        test_case!("NaN max_cn0", f64::NAN, 10.0..300.0);
        test_case!("infinite max_cn0", f64::INFINITY, 10.0..300.0);
        test_case!("unbounded start", 100.0, ..300.0);
        test_case!("unbounded end", 100.0, 10.0..);
        test_case!("zero min_distance", 100.0, 0.0..300.0);
        test_case!("negative min_distance", 100.0, -10.0..300.0);
        test_case!("NaN min_distance", 100.0, f64::NAN..300.0);
        test_case!("NaN max_distance", 100.0, 10.0..f64::NAN);
        test_case!("max_distance < min_distance", 100.0, 300.0..10.0);
        test_case!("infinite max_distance", 100.0, 10.0..f64::INFINITY);
        test_case!("min_distance squared underflows", 100.0, 1e-200..1.0);
    }

    const ANY_DISTANCE: std::ops::RangeInclusive<f64> = 1e-12..=1e12;

    fn min_max_distance() -> impl Strategy<Value = (f64, f64)> {
        (ANY_DISTANCE, ANY_DISTANCE).prop_map(|(a, b)| if a <= b { (a, b) } else { (b, a) })
    }

    proptest! {
        #[test]
        fn distance_range_accepts_valid(
            max_cn0 in any_cn0(),
            (min_distance, max_distance) in min_max_distance(),
        ) {
            let d_exclusive = DistanceRangeCN0::new(max_cn0, min_distance..max_distance).unwrap();
            let d_inclusive = DistanceRangeCN0::new(max_cn0, min_distance..=max_distance).unwrap();
            assert_eq!(d_exclusive.max_cn0(), max_cn0);
            assert_eq!(d_exclusive.min_distance(), min_distance);
            assert_eq!(d_exclusive.max_distance(), max_distance);
            assert_eq!(d_inclusive.max_cn0(), max_cn0);
            assert_eq!(d_inclusive.min_distance(), min_distance);
            assert_eq!(d_inclusive.max_distance(), max_distance);
            // check that these two ranges give the same object
            assert_eq!(d_exclusive, d_inclusive);
        }
    }

    proptest! {
        /// Test that min_distance == max_distance is allowed and it gives max_cn0
        /// as output always.
        #[test]
        fn distance_range_constant_range_is_constant(
            max_cn0 in any_cn0(),
            distance in ANY_DISTANCE,
        ) {
            let mut rng = rand::rng();
            // exclusive range
            let d_exclusive = DistanceRangeCN0::new(max_cn0, distance..distance).unwrap();
            let d_inclusive = DistanceRangeCN0::new(max_cn0, distance..=distance).unwrap();
            for _ in 0..1024 {
                let tolerance = 1e-12;
                assert!((d_exclusive.sample(&mut rng) - max_cn0).abs() <= tolerance);
                assert!((d_inclusive.sample(&mut rng) - max_cn0).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn constant_cfo_rejects_invalid() {
        assert!(ConstantCarrierFrequencyOffset::new(f64::NAN).is_err());
        assert!(ConstantCarrierFrequencyOffset::new(f64::INFINITY).is_err());
        assert!(ConstantCarrierFrequencyOffset::new(1.1 * 0.5 * SAMPLE_RATE_F64).is_err());
        assert!(ConstantCarrierFrequencyOffset::new(-1.1 * 0.5 * SAMPLE_RATE_F64).is_err());
    }

    proptest! {
        #[test]
        fn constant_cfo_accepts_valid_and_samples_constant(
            freq in -0.5 * SAMPLE_RATE_F64..=0.5 * SAMPLE_RATE_F64
        ) {
            let d = ConstantCarrierFrequencyOffset::new(freq).unwrap();
            assert_eq!(d.carrier_frequency_offset(), freq);
            assert_samples_constant!(d, freq);
        }
    }

    #[test]
    fn clamped_normal_rejects_invalid() {
        assert!(ClampedNormalCarrierFrequencyOffset::new(f64::NAN, 1e6).is_err());
        assert!(ClampedNormalCarrierFrequencyOffset::new(100e3, f64::NAN).is_err());
        assert!(ClampedNormalCarrierFrequencyOffset::new(-1.0, 1e6).is_err());
        assert!(ClampedNormalCarrierFrequencyOffset::new(100e3, -1.0).is_err());
        assert!(
            ClampedNormalCarrierFrequencyOffset::new(100e3, 1.1 * 0.5 * SAMPLE_RATE_F64).is_err()
        );
    }

    proptest! {
        #[test]
        fn clamped_normal_accepts_valid(
            sigma in POSITIVE | ZERO,
            max in 0.0..=0.5 * SAMPLE_RATE_F64,
        ) {
            let d = ClampedNormalCarrierFrequencyOffset::new(sigma, max).unwrap();
            assert_eq!(d.sigma(), sigma);
            assert_eq!(d.max(), max);
        }
    }

    /// Tests that sigma = 0 generates a valid (constant) distribution.
    #[test]
    fn clamped_normal_sigma_zero() {
        let d = ClampedNormalCarrierFrequencyOffset::new(0.0, 1e6).unwrap();
        assert_eq!(d.sigma(), 0.0);
        assert_eq!(d.max(), 1e6);
        assert_samples_constant!(d, 0.0);
    }

    #[test]
    fn constant_phase_rejects_invalid() {
        assert!(ConstantCarrierPhase::new(f64::NAN).is_err());
        assert!(ConstantCarrierPhase::new(f64::INFINITY).is_err());
        assert!(ConstantCarrierPhase::new(f64::NEG_INFINITY).is_err());
    }

    proptest! {
        #[test]
        fn constant_phase_accepts_valid_and_samples_constat(
            phase in POSITIVE | NEGATIVE | ZERO | SUBNORMAL
        ) {
            let d = ConstantCarrierPhase::new(phase).unwrap();
            // phase0 is phase wrapped to [-pi, pi)
            // There is another test below that checks this.
            let phase0 = d.carrier_phase();
            assert_samples_constant!(d, phase0);
        }
    }

    proptest! {
        #[test]
        fn constant_phase_normalizes(
            phase in -PI..PI,
            wraps in -1024_i32..=1024,
        ) {
            let d = ConstantCarrierPhase::new(phase + wraps as f64 * 2.0 * PI).unwrap();
            let phase0 = d.carrier_phase();
            let tolerance = 1e-10;
            assert!((phase0 - phase).abs() <= tolerance);
            assert_samples_constant!(d, phase0);
        }
    }

    #[test]
    fn constant_clock_error_rejects_invalid() {
        assert!(ConstantSymbolClockError::new(f64::NAN).is_err());
        assert!(ConstantSymbolClockError::new(f64::INFINITY).is_err());
        assert!(ConstantSymbolClockError::new(-f64::INFINITY).is_err());
        assert!(ConstantSymbolClockError::new(1.1 * MAX_CLOCK_RATE_ERROR).is_err());
        assert!(ConstantSymbolClockError::new(-1.1 * MAX_CLOCK_RATE_ERROR).is_err());
    }

    proptest! {
        #[test]
        fn constant_clock_error_accepts_valid_and_samples_constant(
            clock_error in -MAX_CLOCK_RATE_ERROR..=MAX_CLOCK_RATE_ERROR
        ) {
            let d = ConstantSymbolClockError::new(clock_error).unwrap();
            assert_eq!(d.symbol_clock_error(), clock_error);
            assert_samples_constant!(d, clock_error);
        }
    }

    #[test]
    fn uniform_clock_error_rejects_invalid() {
        assert!(UniformSymbolClockError::new(f64::NAN).is_err());
        assert!(UniformSymbolClockError::new(f64::INFINITY).is_err());
        assert!(UniformSymbolClockError::new(-f64::INFINITY).is_err());
        assert!(UniformSymbolClockError::new(1.1 * MAX_CLOCK_RATE_ERROR).is_err());
        assert!(UniformSymbolClockError::new(-1.1 * MAX_CLOCK_RATE_ERROR).is_err());
        assert!(UniformSymbolClockError::new(-1e-9).is_err());
    }

    proptest! {
        #[test]
        fn uniform_clock_error_accepts_valid(clock_error in 0.0..=MAX_CLOCK_RATE_ERROR) {
            assert!(UniformSymbolClockError::new(clock_error).is_ok());
        }
    }

    /// Tests that max = 0 produces a constant distribution.
    #[test]
    fn uniform_clock_error_zero() {
        let d = UniformSymbolClockError::new(0.0).unwrap();
        assert_eq!(d.max(), 0.0);
        assert_samples_constant!(d, 0.0);
    }

    /// Tests that the default constructors do not panic (they use unwrap
    /// internally to check parameters).
    #[test]
    fn defaults_do_not_panic() {
        CN0Distribution::default();
        CarrierFrequencyOffsetDistribution::default();
        CarrierPhaseDistribution::default();
        SymbolClockErrorDistribution::default();
    }
}
