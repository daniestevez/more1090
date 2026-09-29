//! CLI utilities.
//!
//! This module contains miscellaneous functions that are used in the CLI.

use crate::receiver::ErrorCorrectionLevel;
use anyhow::Result;
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
