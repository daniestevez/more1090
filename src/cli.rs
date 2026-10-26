//! CLI utilities.
//!
//! This module contains miscellaneous functions that are used in the CLI.

use crate::receiver::{ADSBReceiver, ErrorCorrectionLevel};
use anyhow::Result;
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Tries to convert a string representing a duration in seconds into a [`Duration`].
pub fn try_secs_into_duration(s: &str) -> Result<Duration> {
    let secs: f64 = s
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid number of seconds: {e}"))?;
    Ok(Duration::try_from_secs_f64(secs)?)
}

/// Newtype wrapper for [`Duration`] that implements
/// [`FromStr`](std::str::FromStr) and [`Display`](std::fmt::Display) as a
/// number of seconds.
///
/// This newtype wrapper implements `FromStr` and `Display` so that clap can
/// parse user-provided durations in seconds and render `default_value_t`
/// defaults in the help output.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Seconds(pub Duration);

impl std::str::FromStr for Seconds {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        try_secs_into_duration(s).map(Seconds)
    }
}

impl std::fmt::Display for Seconds {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0.as_secs_f64())
    }
}

impl From<Seconds> for Duration {
    fn from(seconds: Seconds) -> Duration {
        seconds.0
    }
}

/// Tries to convert a string representing a number of bit errors into an [`ErrorCorrectionLevel`].
pub fn try_bit_errors_into_error_correction_level(s: &str) -> Result<ErrorCorrectionLevel> {
    let bit_errors: usize = s
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid number of bit errors: {e}"))?;
    Ok(match bit_errors {
        0 => ErrorCorrectionLevel::NoErrors,
        1 => ErrorCorrectionLevel::SingleError,
        2 => ErrorCorrectionLevel::DoubleError,
        _ => anyhow::bail!("cannot correct {bit_errors} bit errors; the maximum supported is 2"),
    })
}

/// ADS-B receiver configuration CLI arguments.
///
/// This struct is flattened into CLI arguments of applications that need to
/// allow the user to configure all the receiver parameters.
#[derive(Debug, Clone, PartialEq, Parser, Serialize, Deserialize)]
pub struct ADSBReceiverArgs {
    /// Error correction level.
    ///
    /// This sets the maximum number of bit errors that the ADS-B decoder is
    /// allowed to correct. The maximum correction level allowed is 2 bits.
    #[arg(long, default_value = "2", value_parser = try_bit_errors_into_error_correction_level)]
    error_correction_level: ErrorCorrectionLevel,
    /// Carrier metric threshold.
    ///
    /// This sets the threshold used for carrier detection. The provided default
    /// value is generally a good choice.
    #[arg(long, default_value_t = crate::constants::dsp::CARRIER_METRIC_THRESHOLD)]
    carrier_metric_threshold: f32,
    /// CN0 threshold in dB·Hz.
    ///
    /// This sets the threshold used for CN0 validation in messages with 2 bit
    /// errors. The provided default is generally a good choice.
    #[arg(long, default_value_t = crate::constants::dsp::CN0_THRESHOLD)]
    cn0_threshold: f32,
    /// Hamming distance threshold for rejection of 2-bit error noncoherent
    /// decodes.
    ///
    /// In order to reduce the number of false decodes, for a noncoherent decode
    /// candidate with 2 bit errors, the receiver computes the Hamming distance
    /// between the coherent and noncoherent demodulations corresponding to that
    /// candidate. If the distance is greater or equal than this threshold, then
    /// the CN0 validation is done against a higher threshold.
    #[arg(long, default_value_t = crate::constants::dsp::NONCOHERENT_HIGH_HAMMING_DISTANCE)]
    noncoherent_high_hamming_distance: u32,
    /// Increased CN0 threshold in dB·Hz for rejection of 2-bit error
    /// noncoherent decodes.
    #[arg(long, default_value_t = crate::constants::dsp::NONCOHERENT_HIGH_HAMMING_CN0_THRESHOLD)]
    noncoherent_high_hamming_cn0_threshold: f32,
}

impl Default for ADSBReceiverArgs {
    fn default() -> ADSBReceiverArgs {
        ADSBReceiverArgs::try_parse_from(std::iter::empty::<std::ffi::OsString>()).unwrap()
    }
}

impl ADSBReceiverArgs {
    /// Creates a new [`ADSBReceiver`] and applies the configuration described
    /// by `self`.
    pub fn new_receiver(&self) -> Result<ADSBReceiver> {
        let mut receiver = ADSBReceiver::new();
        receiver.set_error_correction_level(self.error_correction_level);
        receiver.set_carrier_metric_threshold(self.carrier_metric_threshold)?;
        receiver.set_cn0_threshold(self.cn0_threshold)?;
        receiver.set_noncoherent_high_hamming_distance(self.noncoherent_high_hamming_distance)?;
        receiver.set_noncoherent_high_hamming_cn0_threshold(
            self.noncoherent_high_hamming_cn0_threshold,
        )?;
        Ok(receiver)
    }

    /// Validates the arguments.
    ///
    /// This validates the arguments by constructing a receiver with these
    /// arguments and dropping it.
    pub fn validate(&self) -> Result<()> {
        self.new_receiver()?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    /// Tests that the default implementation of ADSBReceiverArgs does not
    /// panic.
    #[test]
    fn adsb_receiver_args_default() {
        ADSBReceiverArgs::default();
    }

    #[test]
    fn secs_into_duration() {
        assert_eq!(try_secs_into_duration("1").unwrap(), Duration::from_secs(1));
        assert_eq!(
            try_secs_into_duration("0.1").unwrap(),
            Duration::from_millis(100)
        );
        assert_eq!(
            try_secs_into_duration("0.001").unwrap(),
            Duration::from_millis(1)
        );
        assert_eq!(
            try_secs_into_duration("0.000001").unwrap(),
            Duration::from_nanos(1000)
        );
        assert!(try_secs_into_duration("-0.1").is_err());
        assert!(try_secs_into_duration("-2").is_err());
        assert!(try_secs_into_duration("").is_err());
        assert!(try_secs_into_duration("abc").is_err());
    }

    #[test]
    fn bit_errrors_into_error_correction_level() {
        assert_eq!(
            try_bit_errors_into_error_correction_level("0").unwrap(),
            ErrorCorrectionLevel::NoErrors
        );
        assert_eq!(
            try_bit_errors_into_error_correction_level("1").unwrap(),
            ErrorCorrectionLevel::SingleError
        );
        assert_eq!(
            try_bit_errors_into_error_correction_level("2").unwrap(),
            ErrorCorrectionLevel::DoubleError
        );
        assert!(try_bit_errors_into_error_correction_level("3").is_err());
        assert!(try_bit_errors_into_error_correction_level("").is_err());
        assert!(try_bit_errors_into_error_correction_level("abc").is_err());
    }

    #[test]
    fn new_receiver() {
        let args = ADSBReceiverArgs {
            error_correction_level: ErrorCorrectionLevel::SingleError,
            carrier_metric_threshold: 1.7,
            cn0_threshold: 70.0,
            noncoherent_high_hamming_distance: 2,
            noncoherent_high_hamming_cn0_threshold: 75.0,
        };
        let receiver = args.new_receiver().unwrap();
        assert_eq!(
            receiver.error_correction_level(),
            args.error_correction_level
        );
        assert_eq!(
            receiver.carrier_metric_threshold(),
            args.carrier_metric_threshold
        );
        assert_eq!(receiver.cn0_threshold(), args.cn0_threshold);
        assert_eq!(
            receiver.noncoherent_high_hamming_distance(),
            args.noncoherent_high_hamming_distance
        );
        assert_eq!(
            receiver.noncoherent_high_hamming_cn0_threshold(),
            args.noncoherent_high_hamming_cn0_threshold
        );
    }
}
