//! # more1090
//!
//! more1090 is a high-sensitivity ADS-B receiver that performs better in
//! scenarios with low SNR and packet collisions than other open source
//! receivers such as [readsb](https://github.com/wiedehopf/readsb). more1090
//! requires an input sample rate of 8 Msps IQ. It decodes extended squitter
//! (DF17 and DF18) Mode-S reply messages. In addition to the decoded message,
//! the receiver provides estimates for the time of arrival, the carrier
//! frequency and the CN0 of the ADS-B message.
//!
//! more1090 can be used as a CLI application called `more1090` that reads a
//! SigMF file and produces an annotation for each decoded message, or reads a
//! real time stream from stdin and outputs each decoded message in AVR format
//! (allowing the output to be piped into other applications such as
//! [tar1090](https://github.com/wiedehopf/tar1090)). more1090 also contains a
//! Mode-S signal simulator called `more1090-simulator` and a benchmarking tool
//! called `more1090-benchmark` that is used to benchmark the sensitivity and
//! throughput of the more1090 ADS-B receiver. Additionally, all of this can be
//! used as a Rust library.
//!
//! ## Example
//!
//! The receiver is instantiated with some fixed buffer sizes at compile time,
//! which can be optimized to trade off lower overhead or better cache
//! locality. Based on these buffer sizes, the receiver's
//! [`process_chunk`](ADSBReceiver::process_chunk) method supports a maximum
//! chunk size. IQ data is fed into the receiver in chunks of samples whose
//! length must be a multiple of
//! [`NFFT_SHORT`](crate::constants::dsp::NFFT_SHORT) and lower than or equal to
//! the maximum chunk size. Each `process_chunk` call returns a slice containing
//! the detected and decoded packets.
//!
//! ```
//! # fn main() {
//! use more1090::{ADSBReceiver, CHUNK_LEN};
//! use num::complex::Complex32;
//!
//! let mut receiver = ADSBReceiver::new();
//! // Usually the IQ input will come from an external source or file
//! let iq = vec![Complex32::new(0.0, 0.0); 16_000]; // 2 ms of IQ samples
//! for chunk in iq.as_chunks::<CHUNK_LEN>().0.iter() {
//!     if let Some(packet) = receiver.process_chunk(chunk) {
//!         dbg!(packet);
//!     }
//! }
//! # }
//! ```

#![warn(missing_docs)]

#[cfg(test)]
/// Checks that two slices have the same length and their elements are closer
/// than a limit given by absolute and relative tolerances.
macro_rules! assert_all_close {
    ($x:expr, $y:expr, $absolute_tolerance:expr, $relative_tolerance:expr$(,)?) => {{
        let abs_tol = $absolute_tolerance;
        let rel_tol = $relative_tolerance;
        let x = $x;
        let y = $y;
        #[allow(unused_imports)]
        use num::complex::ComplexFloat;
        assert_eq!(x.len(), y.len());
        for (j, (a, b)) in x.iter().zip(y.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                (diff <= abs_tol) || (diff <= rel_tol * a.abs().min(b.abs())),
                "elements x[{j}] = {a} and y[{j}] = {b} \
                 are further away than absolute tolerance {abs_tol:e} \
                 and relative tolerance {rel_tol:e}"
            );
        }
    }};
}

pub mod benchmark;
pub mod circbuf;
pub mod cli;
pub mod constants;
pub mod io;
pub mod math;
pub mod mode_s;
pub mod receiver;
pub mod sigmf;
pub mod simulation;

pub use receiver::{A, ADSBReceiver, Align};

/// Chunk length used in the input of the [`ADSBReceiver`].
///
/// This is equal to [`NFFT_SHORT`](constants::dsp::NFFT_SHORT).
pub const CHUNK_LEN: usize = constants::dsp::NFFT_SHORT;
