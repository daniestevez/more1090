//! ADS-B pulse shape.
//!
//! This module implements a polyphase filter used to simulate a band-limited ADS-B pulse shape.

use super::Pulses;
use crate::constants::dsp::SPS;
use pm_remez::{BandSetting, constant, pm_parameters, pm_remez};
use std::sync::OnceLock;

/// Number of branches in the polyphase filter.
pub const NUM_BRANCHES: usize = 64 * SPS; // this needs to be a multiple of SPS

/// Number of taps per branch in the polyphase filter.
pub const TAPS_PER_BRANCH: usize = 7;

const PROTOTYPE_LEN: usize = NUM_BRANCHES * TAPS_PER_BRANCH;

// times in samples
const RISE_TIME: usize = 38;
const FALL_TIME: usize = 64;
const PULSE_DURATION: usize = 256;
const PULSE_TOTAL_LEN: usize = PULSE_DURATION + (RISE_TIME + FALL_TIME) / 2;

/// Pulse shape filterbank.
///
/// The filterbank is organized as a 2D array of dimensions
/// [`NUM_BRANCHES`]x[`TAPS_PER_BRANCH`]. Each row of the filterbank contains
/// the corresponding [`TAPS_PER_BRANCH`] taps of the filter corresponding to
/// that branch.
pub type Filterbank = [[f32; TAPS_PER_BRANCH]; NUM_BRANCHES];

#[derive(Debug, Clone, PartialEq)]
struct FilterbankData {
    filterbank: Filterbank,
    // The group delay is measured in output samples (at 8 Msps)
    group_delay: f64,
}

impl FilterbankData {
    fn get() -> &'static FilterbankData {
        static FILTERBANK_DATA: OnceLock<FilterbankData> = OnceLock::new();
        FILTERBANK_DATA.get_or_init(|| {
            let prototype = build_prototype_filter();
            let group_delay = compute_group_delay(&prototype);
            let filterbank = (0..NUM_BRANCHES)
                .map(|j| {
                    prototype
                        .iter()
                        .copied()
                        .skip(j)
                        .step_by(NUM_BRANCHES)
                        .collect::<Vec<f32>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            FilterbankData {
                filterbank,
                group_delay,
            }
        })
    }
}

fn compute_group_delay(prototype: &[f32]) -> f64 {
    // Find where the rising edge of the prototype crosses the trigger level.
    let trigger_level = 0.5_f32;
    let (j, &[a0, a1]) = prototype
        .array_windows()
        .enumerate()
        .find(|&(_, &[a0, a1])| (a0..a1).contains(&trigger_level))
        .unwrap();
    // Find where the trigger_level crossing would be exactly.
    let tau = (trigger_level as f64 - a0 as f64) / (a1 as f64 - a0 as f64);
    // Convert to output samples.
    (j as f64 + tau) / (NUM_BRANCHES / SPS) as f64
}

/// Returns the pulse shape filterbank.
///
/// Since building the filterbank is expensive, a [`OnceLock`] is used to build
/// the filterbank on the first call only. The following calls re-use the same
/// filterbank.
pub fn filterbank() -> &'static Filterbank {
    &FilterbankData::get().filterbank
}

/// Returns the group delay of the filterbank.
///
/// The group delay is given in output samples refered to the interpolator
/// output, that is, in samples at 8 Msps. Computing the group delay of the
/// filterbank requires building the filterbank. Since that is an expensive
/// operation, the same [`OnceLock`] that is used in [`filterbank`] is used
/// here, so the filterbank is only built once and its groupd delay is
/// calculated when it is built.  The following calls use the previously
/// computed group delay value.
pub fn group_delay() -> f64 {
    FilterbankData::get().group_delay
}

/// Applies the filterbank to a sequence of pulses.
///
/// The pulses are marked as `true` elements in a boolean slice of symbols (each
/// symbol is interpolated to [`SPS`] samples). The initial clock phase must be
/// in the interval `[0, 1)`. Its units are output samples. A larger initial
/// clock phase means that the sequence of pulses arrives earlier in the
/// output. The clock rate error is given in parts per one.
///
/// # Panics
///
/// This function panics if the initial clock phase is not in `[0, 1)`, or if
/// `clock_rate_error.abs() > MAX_CLOCK_RATE_ERROR` (this latter constrain is in
/// place to avoid weird behaviour such as skipping or repeating samples coming
/// from an atypically large clock rate error).
pub fn apply_filterbank<'a>(
    filterbank: &'a Filterbank,
    pulses: Pulses,
    initial_clock_phase: f64,
    clock_rate_error: f64,
) -> ApplyFilterbank<'a> {
    assert!((0.0..1.0).contains(&initial_clock_phase));
    assert!(clock_rate_error.abs() <= MAX_CLOCK_RATE_ERROR);
    ApplyFilterbank {
        filterbank,
        pulses,
        clock_phase: initial_clock_phase / SPS as f64,
        clock_rate_error,
        current_index: 0,
    }
}

/// Maximum clock rate error allowed by [`apply_filterbank`].
pub const MAX_CLOCK_RATE_ERROR: f64 = 0.1;

/// Apply filterbank iterator.
///
/// This struct is produced by the [`apply_filterbank`] function.
#[derive(Debug)]
pub struct ApplyFilterbank<'a> {
    filterbank: &'a Filterbank,
    pulses: Pulses,
    clock_phase: f64,
    clock_rate_error: f64,
    current_index: usize,
}

impl<'a> Iterator for ApplyFilterbank<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        if self.pulses.is_empty() {
            // no reason to do anything on empty pulses; this avoids outputting zeros
            return None;
        }
        if self.current_index >= self.pulses.len() + TAPS_PER_BRANCH - 1 {
            // already past the end of self.pulses
            return None;
        }
        let out = convolve_taps_pulses(
            &self.filter_taps(),
            self.pulses.as_slice(),
            self.current_index,
        );
        self.advance();
        Some(out)
    }
}

fn convolve_taps_pulses(taps: &[f32; TAPS_PER_BRANCH], pulses: &[bool], index: usize) -> f32 {
    let mut out = 0.0;
    let first_tap = if index >= pulses.len() {
        index - pulses.len() + 1
    } else {
        0
    };
    for (tap, &pulse) in taps[first_tap..]
        .iter()
        .zip(pulses[..(index + 1).min(pulses.len())].iter().rev())
    {
        if pulse {
            out += tap;
        }
    }
    out
}

impl<'a> ApplyFilterbank<'a> {
    fn filter_taps(&self) -> [f32; TAPS_PER_BRANCH] {
        let mut taps = [0.0; TAPS_PER_BRANCH];
        let clock_phase_scaled = self.clock_phase * NUM_BRANCHES as f64;
        let a = clock_phase_scaled as usize;
        let tau = (clock_phase_scaled - a as f64) as f32;
        for (x, &y) in taps.iter_mut().zip(self.filterbank[a].iter()) {
            *x = (1.0 - tau) * y;
        }
        if a == NUM_BRANCHES - 1 {
            // Special case in which we are at the end of the bank and need to
            // roll over for the interpolation. We use the fact that
            // self.filterbank[NUM_BRANCHES - 1][:-1] is very close to
            // self.filterbank[0][1:]. For the last tap we do nothing, since it
            // is going to be quite a small contribution in any case.
            for (x, &y) in taps.iter_mut().zip(self.filterbank[0].iter().skip(1)) {
                *x += tau * y;
            }
        } else {
            for (x, &y) in taps.iter_mut().zip(self.filterbank[a + 1].iter()) {
                *x += tau * y;
            }
        }
        taps
    }

    fn advance(&mut self) {
        self.clock_phase += 1.0 / SPS as f64 * (1.0 + self.clock_rate_error);
        if self.clock_phase >= 1.0 {
            self.clock_phase -= 1.0;
            assert!(self.clock_phase < 1.0);
            self.current_index += 1;
        }
    }
}

/// Returns a tight upper bound for the number of samples generated by
/// [`apply_filterbank`].
///
/// Given a number of pulses, this function returns an upper bound for the
/// number of samples that a call to [`apply_filterbank`] using the filterbank
/// returned by [`filterbank`], a `pulses` array of such number of pulses, and
/// any initial clock phase and clock rate error will generate. The bound is
/// tight, but only for clock rate errors equal to `-MAX_CLOCK_RATE_ERROR`.
pub const fn required_samples_for_apply(num_pulses: usize) -> usize {
    if num_pulses == 0 {
        0
    } else {
        (((num_pulses + TAPS_PER_BRANCH - 1) * SPS) as f64 / (1.0 - MAX_CLOCK_RATE_ERROR)).ceil()
            as usize
    }
}

fn build_prototype_filter() -> Vec<f32> {
    let filter = convolve_full(&unfiltered_pulse(), &lowpass_filter());
    assert_eq!(filter.len(), PROTOTYPE_LEN);
    filter
}

fn convolve_full(a: &[f32], b: &[f32]) -> Vec<f32> {
    let n = a.len() + b.len() - 1;
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        // TODO: write this using iterators
        let mut conv = 0.0;
        let start = if j >= b.len() { j - b.len() + 1 } else { 0 };
        for k in start..=j.min(a.len() - 1) {
            conv += a[k] * b[j - k];
        }
        out.push(conv);
    }
    out
}

/// Builds an unfiltered pulse.
///
/// The pulse is based on the ADS-B specifications, and it has the following
/// characteristics.
///
/// - 74.2 ns rise time (spec requires 50-100 ns)
/// - 125 ns fall time (spec requires 50-200 ns)
/// - 500 ns pulse width measured at 50% level (spec requires 500 +/- 50 ns)
fn unfiltered_pulse() -> Vec<f32> {
    let mut pulse = Vec::with_capacity(PULSE_TOTAL_LEN);

    // rising edge
    for j in 0..RISE_TIME {
        pulse.push(j as f32 / RISE_TIME as f32);
    }

    // pulse top
    pulse.resize(PULSE_TOTAL_LEN - FALL_TIME, 1.0);

    // falling edge
    for j in 0..FALL_TIME {
        pulse.push(1.0 - (j + 1) as f32 / FALL_TIME as f32);
    }

    assert_eq!(pulse.len(), PULSE_TOTAL_LEN);
    pulse
}

/// Builds the low-pass filter.
///
/// This low-pass filter has a cutoff frequency around 4 MHz, which is the
/// Nyquist frequency for 8 Msps IQ.
fn lowpass_filter() -> Vec<f32> {
    let num_taps = PROTOTYPE_LEN - PULSE_TOTAL_LEN + 1;
    let transition_bw = 0.4;
    let bands = [
        BandSetting::with_weight(
            0.0,
            (2.0 - 0.5 * transition_bw) / NUM_BRANCHES as f64,
            constant(1.0),
            constant(1.0),
        )
        .unwrap(),
        BandSetting::with_weight(
            (2.0 + 0.5 * transition_bw) / NUM_BRANCHES as f64,
            0.5,
            constant(0.0),
            constant(10.0),
        )
        .unwrap(),
    ];
    let parameters = pm_parameters(num_taps, &bands).unwrap();
    let design = pm_remez(&parameters).unwrap();
    assert!(design.flatness < 1e-6);
    design.impulse_response.iter().map(|&x| x as f32).collect()
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn convolve_pulses() {
        let taps = [0.1, 0.3, 0.5, 0.42, 0.33, 0.2, 0.05];
        let pulses = [true, false, false, true, true, false, true, false];
        let expected = [
            0.1, 0.3, 0.5, 0.52, 0.73, 1.0, 1.07, 1.05, 1.03, 0.67, 0.38, 0.2, 0.05, 0.0,
        ];
        for (j, &expected_value) in expected.iter().enumerate() {
            assert!((convolve_taps_pulses(&taps, &pulses, j) - expected_value).abs() < 1e-6);
        }
    }

    #[test]
    fn convolve_numbers() {
        let a = [1.7, -3.5, 4.32, 2.13];
        let b = [7.25, 4.53, -5.4];
        let conv = convolve_full(&a, &b);
        let expected = [12.325, -17.674, 6.285, 53.9121, -13.6791, -11.502];
        assert_all_close!(&conv, &expected, 1e-5, 0.0);
    }

    #[test]
    fn convolve_ones() {
        let a = [1.0; 5];
        let b = [1.0; 7];
        let conv = convolve_full(&a, &b);
        let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(&conv, &expected);
    }

    /// Tests that the filterbank is built without panicking.
    #[test]
    fn filterbank_does_not_panic() {
        let f = filterbank();
        assert_eq!(f.len(), NUM_BRANCHES);
    }

    fn single_pulse() -> Pulses {
        Pulses {
            data: [true; _],
            len: 1,
        }
    }

    /// Tests the group delay calculation by checking that the rising edge is
    /// where it should be.
    #[test]
    fn group_delay_vs_rising_edge() {
        let delay = group_delay();
        let delay_int = delay as usize;
        let delay_frac = delay - delay_int as f64;
        // Set the clock phase equal to the fractional delay so that an output
        // sample is exactly aligned with the trigger level crossing.
        let clock_phase = delay_frac;
        let clock_rate_error = 0.0;
        let x = apply_filterbank(filterbank(), single_pulse(), clock_phase, clock_rate_error)
            .collect::<Vec<_>>();
        let trigger_level = 0.5;
        assert!((x[delay_int] - trigger_level).abs() < 1e-16);
    }

    proptest! {
        #[test]
        fn apply_filterbank_empty_pulses(
            clock_phase in 0.0..1.0,
            clock_rate_error in -100e-6..=100e-6)
        {
            let empty_pulses = Pulses {
                data: [false; _],
                len: 0,
            };
            assert_eq!(
                apply_filterbank(filterbank(), empty_pulses, clock_phase, clock_rate_error).next(),
                None
            );
        }
    }

    proptest! {
        #[test]
        fn apply_filterbank_one_pulse_without_clock_error(clock_phase in 0.0..1.0) {
            let n = apply_filterbank(filterbank(), single_pulse(), clock_phase, 0.0)
                .count();
            assert_eq!(n, SPS * TAPS_PER_BRANCH);
        }
    }

    proptest! {
        #[test]
        fn apply_filterbank_one_pulse_with_small_clock_error(
            clock_phase in 0.0..1.0,
            clock_rate_error in -100e-6..=100e-6
        ) {
            let n = apply_filterbank(filterbank(), single_pulse(), clock_phase, clock_rate_error)
                .count();
            // In some extreme cases depending on clock error we might get one
            // more or one fewer samples than nominal.
            assert!((n as isize - (SPS * TAPS_PER_BRANCH) as isize).abs() <= 1);
        }
    }

    fn max_pulses() -> Pulses {
        use super::super::MAX_MESSAGE_PULSES;
        Pulses {
            data: [false; MAX_MESSAGE_PULSES],
            len: MAX_MESSAGE_PULSES,
        }
    }

    proptest! {
        #[test]
        fn required_samples_for_apply_is_upper_bound(
            clock_phase in 0.0..1.0,
            clock_rate_error in -MAX_CLOCK_RATE_ERROR..=MAX_CLOCK_RATE_ERROR,
        ) {
            use super::super::MAX_MESSAGE_PULSES;
            let n = apply_filterbank(filterbank(), max_pulses(), clock_phase, clock_rate_error)
                .count();
            assert!(n <= required_samples_for_apply(MAX_MESSAGE_PULSES));
        }
    }

    proptest! {
        #[test]
        fn required_samples_for_apply_is_upper_bound_worst_clock_error(
            clock_phase in 0.0..1.0,
        ) {
            use super::super::MAX_MESSAGE_PULSES;
            let clock_rate_error = -MAX_CLOCK_RATE_ERROR;
            let n = apply_filterbank(filterbank(), max_pulses(), clock_phase, clock_rate_error)
                .count();
            assert!(n <= required_samples_for_apply(MAX_MESSAGE_PULSES));
        }
    }
}
