//! ADS-B receiver.
//!
//! This module implements an [`ADSBReceiver`] struct that performs ADS-B
//! message detection and decoding, as well as auxiliary data types.

use crate::{
    circbuf::CircBuf,
    constants::{
        dsp::*,
        mode_s::{MODE_S_LONG_BITS, MODE_S_LONG_BYTES, REPLY_PREAMBLE_BITS},
    },
    math::AlgebraicComplex,
    mode_s::{double_bit_error_from_crc24, single_bit_error_from_crc24},
};
use anyhow::Result;
use generic_array::typenum::{self, marker_traits::Unsigned};
use num::complex::Complex32;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

mod kernels;
use kernels::*;

/// Preferred alignment for SIMD data.
///
/// This type indicates the preferred alignment for SIMD data using the
/// [`aligned`] crate.
pub type Align = aligned::A64;

/// Data aligned to [`Align`] alignment.
///
/// This type aligns data to the preferred alignment for SIMD data using the
/// [`aligned`] crate.
pub type A<T> = aligned::Aligned<Align, T>;

/// ADS-B receiver.
///
/// This struct implements a full ADS-B receiver. The receiver detects Mode-S
/// long replies using an FFT-based approach that detects the residual carrier
/// (DC bias of PPM modulation) of the reply payload, and decodes extended
/// squitter messages.
#[derive(Clone)]
pub struct ADSBReceiver {
    error_correction_level: ErrorCorrectionLevel,
    carrier_metric_threshold: f32,
    preamble_metric_threshold: f32,
    fft: Arc<dyn Fft<f32>>,
    // The real parts and imaginary parts are separated into two different
    // sub-arrays, as this is advantageous for vectorization of IQ x LO.
    freq_bin_lo: Box<[[A<[f32; FREQ_BIN_LO_LEN]>; 2]; FREQ_BIN_FACTOR_SHORT]>,
    fft_buf: Box<A<[Complex32; NUM_FREQ_BINS]>>,
    fft_scratch: Box<[Complex32]>,
    short_ffts: CircBuf<A<[Complex32; NUM_FREQ_BINS_KEEP]>, ShortFFTEntries>,
    long_ffts: CircBuf<LongFFT, LongFFTEntries>,
    iq: CircBuf<Complex32, IqSamplesEntries>,
    packet_samples: Box<[Complex32; PACKET_WINDOW_SAMPLES]>,
    symbols: Box<[[Complex32; PACKET_WINDOW_SYMBOLS]; SPS]>,
    coherent_demod: [[[u8; PACKET_WINDOW_BYTES]; 2]; SPS],
    noncoherent_demod: [[[u8; PACKET_WINDOW_BYTES]; 2]; SPS],
    packet: ADSBPacket,
}

// Circular buffer sizes
type ShortFFTEntries = typenum::consts::U8;
type LongFFTEntries = typenum::consts::U8;
type IqSamplesEntries = typenum::consts::U2048;

// compile-time check that buffer sizes satisfy constraints
const _: () = {
    assert!(
        <ShortFFTEntries as Unsigned>::USIZE >= FFT_OVERLAP_FACTOR,
        "short FFT circular buffer is too small"
    );
    assert!(
        <LongFFTEntries as Unsigned>::USIZE > 2 * LOCAL_MAX_WINDOW as usize,
        "long FFT circular buffer is too small"
    );
    assert!(
        <IqSamplesEntries as Unsigned>::USIZE
            >= (LOCAL_MAX_WINDOW as usize + FFT_OVERLAP_FACTOR) * NFFT_SHORT + LEFT_MARGIN,
        "IQ circular buffer is too small"
    );
};

#[derive(Debug, Clone, PartialEq)]
struct LongFFT {
    fft_power: A<[f32; NUM_FREQ_BINS_KEEP]>,
    max_bin: u32,
    max_value: f32,
}

impl Default for LongFFT {
    fn default() -> LongFFT {
        LongFFT {
            fft_power: aligned::Aligned([0.0; NUM_FREQ_BINS_KEEP]),
            max_bin: 0,
            max_value: 0.0,
        }
    }
}

/// ADS-B packet.
///
/// This struct is an ADS-B packet output by the [`ADSBReceiver`]. The packet
/// can be just a detection, for which decoding failed, in which case `decode`
/// is set to `None`, or it can be a detection for which decoding succeeded, in
/// which case `decode` is set to a `Some` value that contains the decode
/// results. In both cases, `detection` is set to the detection results.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ADSBPacket {
    /// Detection results for the packet.
    pub detection: ADSBDetection,
    /// Decode results for the packet.
    ///
    /// This is only present when decoding was successful.
    pub decode: Option<ADSBDecode>,
}

/// ADS-B detection.
///
/// This struct contains the parameters produced by [`ADSBReceiver`] for each
/// detected packet.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ADSBDetection {
    /// Sample index in which the Mode-S reply payload begins.
    ///
    /// This is a sample index that corresponds to the count of samples fed to
    /// the ADSB-Receiver using [`ADSBReceiver::process_chunk`]. Detection
    /// performs only a coarse time estimate for the packet. In particular, the
    /// index given here is always a multiple of [`NFFT_SHORT`].
    pub sample_index: u64,
    /// Carrier frequency offset estimate in Hz.
    pub carrier_frequency_offset: f64,
    /// Carrier power estimate.
    pub carrier_power: f32,
    /// Floor power estimate.
    ///
    /// The floor is defined as the integrated power in the central 4 MHz of
    /// spectrum, except for a +/- 200 kHz notch about the residual carrier.
    pub floor_power: f32,
}

/// ADS-B decode.
///
/// This struct contains the message and other parameters produced by
/// [`ADSBReceiver`] for each successfully decoded message.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ADSBDecode {
    /// Sample index in which the Mode-S reply payload begins.
    ///
    /// This is a sample index that corresponds to the count of samples fed to
    /// the ADSB-Receiver using [`ADSBReceiver::process_chunk`]. The sample
    /// index corresponds to the sample in which the receiver estimates that the
    /// 50% level of the rising edge of the first Mode-S reply payload bit
    /// (assuming that this bit is a one, since if it is a zero the pulse would
    /// be placed 500 ns later), rounded to the nearest integer sample.
    pub sample_index: u64,
    /// Sample index fractional part.
    ///
    /// This is the fractional part of `sample_index`. It is a float in the
    /// range `[-0.5, 0.5)` such that when added to `sample_index`, it gives the
    /// receiver estimate of the 50% level of the rising edge with subsample
    /// precision.
    pub sample_index_frac: f32,
    /// Mode-S long reply message payload.
    ///
    /// This array contains the 112-bit message, including the CRC-24.
    pub message: [u8; MODE_S_LONG_BYTES],
    /// Number and location of corrected bit errors.
    pub corrected_errors: BitErrors,
    /// Demodulation type used for the decode.
    pub demodulation: DemodulationType,
    /// CN0 estimation in dB·Hz.
    pub cn0_db: f32,
}

/// Error correction level.
///
/// This enum is used to indicate the maximum number of bit errors that the
/// ADS-B decoder tries to correct in a message.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ErrorCorrectionLevel {
    /// No bit errors.
    NoErrors,
    /// A single bit error.
    SingleError,
    /// Two bit errors.
    DoubleError,
}

/// Bit error locations.
///
/// This enum is used to indicate the number and location of the bit errors that
/// the ADS-B decoder corrected in a decoded message.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
pub enum BitErrors {
    /// No bit errors.
    #[default]
    NoErrors,
    /// A single bit error, whose position is indicated by the argument.
    SingleError(u8),
    /// Two bit errors, whose positions are indicated by the arguments.
    DoubleError(u8, u8),
}

impl BitErrors {
    /// Returns the corresponding number of bit errors.
    pub fn num_errors(&self) -> usize {
        match self {
            BitErrors::NoErrors => 0,
            BitErrors::SingleError(_) => 1,
            BitErrors::DoubleError(_, _) => 2,
        }
    }
}

/// Demodulation type.
///
/// This enum is used to indicate the demodulation type that was successfully
/// used for an ADS-B packet decode.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Default)]
pub enum DemodulationType {
    /// Coherent demodulation.
    #[default]
    Coherent,
    /// Non-coherent demodulation.
    NonCoherent,
}

impl std::fmt::Display for DemodulationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(
            f,
            "{}",
            match self {
                DemodulationType::Coherent => "coherent",
                DemodulationType::NonCoherent => "noncoherent",
            }
        )
    }
}

impl std::fmt::Debug for ADSBReceiver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("ADSBReceiver")
            .field("error_correction_level", &self.error_correction_level)
            .field("freq_bin_lo", &self.freq_bin_lo)
            .field("fft_buf", &self.fft_buf)
            .field("fft_scratch", &self.fft_scratch)
            .field("short_ffts", &self.short_ffts)
            .field("long_ffts", &self.long_ffts)
            .field("iq", &self.iq)
            .field("packet_samples", &self.packet_samples)
            .field("symbols", &self.symbols)
            .field("coherent_demod", &self.coherent_demod)
            .field("noncoherent_demod", &self.noncoherent_demod)
            .field("packet", &self.packet)
            .finish()
    }
}

impl Default for ADSBReceiver {
    fn default() -> ADSBReceiver {
        ADSBReceiver::new()
    }
}

impl ADSBReceiver {
    /// Creates a new ADS-B receiver.
    ///
    /// This creates an ADS-B receiver with default parameters. The parameters
    /// can be modified by calling the setter methods defined below.
    pub fn new() -> ADSBReceiver {
        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(NFFT_SHORT);
        let scratch_len = fft.get_inplace_scratch_len();
        ADSBReceiver {
            error_correction_level: ErrorCorrectionLevel::DoubleError,
            carrier_metric_threshold: CARRIER_METRIC_THRESHOLD,
            preamble_metric_threshold: PREAMBLE_METRIC_THRESHOLD,
            fft,
            freq_bin_lo: make_freq_bin_lo(),
            fft_buf: Box::new(aligned::Aligned([Complex32::default(); NUM_FREQ_BINS])),
            fft_scratch: vec![Complex32::default(); scratch_len].into_boxed_slice(),
            short_ffts: CircBuf::new_filled_with(&aligned::Aligned(
                [Complex32::default(); NUM_FREQ_BINS_KEEP],
            )),
            long_ffts: CircBuf::new_filled_with(&LongFFT::default()),
            iq: CircBuf::new(),
            packet_samples: Box::new([Complex32::default(); _]),
            symbols: Box::new([[Complex32::default(); PACKET_WINDOW_SYMBOLS]; SPS]),
            coherent_demod: Default::default(),
            noncoherent_demod: Default::default(),
            packet: ADSBPacket::default(),
        }
    }

    /// Returns the current error correction level.
    ///
    /// This is the maximum number of bit errors that the decoder attempts to
    /// correct. The default is two bit errors.
    pub fn error_correction_level(&self) -> ErrorCorrectionLevel {
        self.error_correction_level
    }

    /// Sets the current error correction level.
    ///
    /// See [`error_correction_level`](Self::error_correction_level).
    pub fn set_error_correction_level(&mut self, error_correction_level: ErrorCorrectionLevel) {
        self.error_correction_level = error_correction_level;
    }

    /// Returns the current carrier metric threshold for packet detection.
    ///
    /// The default is [`CARRIER_METRIC_THRESHOLD`].
    pub fn carrier_metric_threshold(&self) -> f32 {
        self.carrier_metric_threshold
    }

    /// Sets the current carrier metric threshold.
    ///
    /// See [`carrier_metric_threshold`](Self::carrier_metric_threshold). An
    /// error is returned if the `carrier_metric_threshold` is negative or not finite.
    pub fn set_carrier_metric_threshold(&mut self, carrier_metric_threshold: f32) -> Result<()> {
        anyhow::ensure!(
            carrier_metric_threshold.is_finite(),
            "carrier metric threshold {carrier_metric_threshold} must be finite"
        );
        anyhow::ensure!(
            carrier_metric_threshold >= 0.0,
            "carrier metric threshold {carrier_metric_threshold} must be non-negative"
        );
        self.carrier_metric_threshold = carrier_metric_threshold;
        Ok(())
    }

    /// Returns the current preamble metric threshold for preamble validation.
    ///
    /// The default is [`PREAMBLE_METRIC_THRESHOLD`]. This metric is used to
    /// validate the preamble for messages with 2 or more corrected bit
    /// errors. It is a mechanism to reduce the number of false decodes caused
    /// by message misalignment.
    pub fn preamble_metric_threshold(&self) -> f32 {
        self.preamble_metric_threshold
    }

    /// Sets the current preamble metric threshold.
    ///
    /// See [`preamble_metric_threshold`](Self::preamble_metric_threshold). An
    /// error is returned if the `preamble_metric_threshold` is negative or not
    /// finite.
    pub fn set_preamble_metric_threshold(&mut self, preamble_metric_threshold: f32) -> Result<()> {
        anyhow::ensure!(
            preamble_metric_threshold.is_finite(),
            "preamble metric threshold {preamble_metric_threshold} must be finite"
        );
        anyhow::ensure!(
            preamble_metric_threshold >= 0.0,
            "preamble metric threshold {preamble_metric_threshold} must be non-negative"
        );
        self.preamble_metric_threshold = preamble_metric_threshold;
        Ok(())
    }

    /// Process a chunk of IQ samples.
    ///
    /// This feeds a chunk of [`CHUNK_LEN`](crate::CHUNK_LEN) IQ samples into
    /// the receiver. The function returns an `Option` that may contain a
    /// reference to a new packet that was detected or decoded as a consequence
    /// of these samples being processed by the receiver. There is some internal
    /// processing delay in the receiver (the receiver needs to look some
    /// samples beyond the end of a packet before attempting to detect it), so
    /// the packet returned by this call generally corresponds to IQ samples fed
    /// in previous calls.
    ///
    /// The reference returned by this function will be overwritten on the next
    /// [`process_chunk`](Self::process_chunk) call, so the caller should do
    /// something with this packet before calling the function again (the Rust
    /// ownership system guarantees that `process_chunk` cannot be called while
    /// the returned reference is still kept alive by the caller).
    pub fn process_chunk<'a>(&'a mut self, iq: &[Complex32; NFFT_SHORT]) -> Option<&'a ADSBPacket> {
        let lo_offset =
            usize::try_from(self.iq.samples_written() % u64::try_from(FREQ_BIN_LO_LEN).unwrap())
                .unwrap();
        self.prepare_fft_input(iq, lo_offset);
        self.fft
            .process_with_scratch(&mut self.fft_buf[..], &mut self.fft_scratch);
        self.copy_short_ffts_to_circbuf();
        self.update_long_ffts();
        self.iq.extend_from_slice(iq);
        self.perform_detection()
    }

    fn prepare_fft_input(&mut self, iq: &[Complex32; NFFT_SHORT], lo_offset: usize) {
        for freq_bin in 0..FREQ_BIN_FACTOR_SHORT {
            let lo_re: &[f32; NFFT_SHORT] = self.freq_bin_lo[freq_bin][0]
                [lo_offset..lo_offset + NFFT_SHORT]
                .try_into()
                .unwrap();
            let lo_im: &[f32; NFFT_SHORT] = self.freq_bin_lo[freq_bin][1]
                [lo_offset..lo_offset + NFFT_SHORT]
                .try_into()
                .unwrap();
            let out_offset = freq_bin * NFFT_SHORT;
            let out_chunk: &mut [Complex32; NFFT_SHORT] = (&mut self.fft_buf
                [out_offset..out_offset + NFFT_SHORT])
                .try_into()
                .unwrap();
            for (out, (z, (w_re, w_im))) in out_chunk
                .iter_mut()
                .zip(iq.iter().zip(lo_re.iter().zip(lo_im.iter())))
            {
                *out = z.algebraic_mul(Complex32::new(*w_re, *w_im));
            }
        }
    }

    fn copy_short_ffts_to_circbuf(&mut self) {
        let output_fft = self.short_ffts.writer_ref();
        // Interleave the FREQ_BIN_FACTOR_SHORT FFTs, perform fftshift on each
        // of them while copying, and drop outermost 1/4's of spectrum. Linear
        // access in input. Strided access in output.
        for (lo_bin, input_fft) in self.fft_buf.as_chunks::<NFFT_SHORT>().0.iter().enumerate() {
            // We want to drop the outermost 1/4's of the spectrum, to keep
            // only the central 1/2. Before FFT-shifting, these outermost
            // 1/4's are the central half. So we operate on the first and
            // last 1/4's of the input.
            for (k, z) in input_fft[..NFFT_SHORT / 4].iter().enumerate() {
                // The DC bin is k = 0, and it would go to position
                // NFFT_SHORT / 4, since we are reducing the FFT size by a
                // factor of 2 by keeping only the central 1/2.
                let k_fftshift = k + NFFT_SHORT / 4;
                let dest_bin = k_fftshift * FREQ_BIN_FACTOR_SHORT + lo_bin;
                output_fft[dest_bin] = *z;
            }
            for (k, z) in input_fft[3 * NFFT_SHORT / 4..].iter().enumerate() {
                // The first bin that we get in the last 1/4 is the first
                // bin of the FFT-shifted output, so there is no indexing change.
                let k_fftshift = k;
                let dest_bin = k_fftshift * FREQ_BIN_FACTOR_SHORT + lo_bin;
                output_fft[dest_bin] = *z;
            }
        }
    }

    fn update_long_ffts(&mut self) {
        const OVERLAP: u64 = FFT_OVERLAP_FACTOR as u64;
        if self.long_ffts.samples_written() + OVERLAP - 1 < self.short_ffts.samples_written() {
            let start = self.long_ffts.samples_written();
            let end = start + OVERLAP;
            let (a, b) = self.short_ffts.get(start..end).unwrap();
            // Convert short FFTs references into an array of references to
            // allow the compiler to do a better job at vectorizing the
            // accumulation.
            let mut short_ffts = a.iter().chain(b.iter());
            let short_ffts = std::array::from_fn(|_| short_ffts.next().unwrap());
            let long_fft = self.long_ffts.writer_ref();
            accumulate_short_ffts(&mut long_fft.fft_power, &short_ffts);
            long_fft_max_search(long_fft);
        }
    }

    fn perform_detection(&mut self) -> Option<&ADSBPacket> {
        if self.long_ffts.samples_written() < LOCAL_MAX_WINDOW_SIZE {
            // Not enough long FFTs to do anything yet
            return None;
        }
        let k = self.long_ffts.samples_written() - LOCAL_MAX_WINDOW - 1;
        if self.check_local_maximum(k)
            && let Some(floor_power) = self.check_carrier_metric(k)
        {
            self.packet.detection.sample_index = k * u64::try_from(NFFT_SHORT).unwrap();
            let frequency_estimate = self.estimate_carrier_frequency_offset(k);
            self.packet.detection.carrier_frequency_offset = frequency_estimate;
            self.packet.detection.carrier_power = self.long_ffts[k].max_value;
            self.packet.detection.floor_power = floor_power;
            self.extract_packet_samples(k);
            self.decode_packet();
            Some(&self.packet)
        } else {
            None
        }
    }

    fn check_local_maximum(&self, long_fft_idx: u64) -> bool {
        let (w0, w1) = self
            .long_ffts
            .get(long_fft_idx - LOCAL_MAX_WINDOW..=long_fft_idx + LOCAL_MAX_WINDOW)
            .unwrap();
        let mut window_max = 0.0;
        for fft in w0.iter().chain(w1.iter()) {
            if fft.max_value > window_max {
                window_max = fft.max_value;
            }
        }
        self.long_ffts[long_fft_idx].max_value == window_max
    }

    fn check_carrier_metric(&self, long_fft_idx: u64) -> Option<f32> {
        let fft = &self.long_ffts[long_fft_idx];
        let floor_power = carrier_notched_power(fft);
        if fft.max_value >= self.carrier_metric_threshold * floor_power {
            Some(floor_power)
        } else {
            None
        }
    }

    fn estimate_carrier_frequency_offset(&self, long_fft_idx: u64) -> f64 {
        let fft = &self.long_ffts[long_fft_idx];
        let bin_int = usize::try_from(fft.max_bin).unwrap();
        let max_window = fft.fft_power[bin_int - 1..=bin_int + 1].try_into().unwrap();
        let bin_frac = f64::from(parabolic_interpolation(max_window));
        let bin = bin_int as f64 + bin_frac;
        (bin - (NUM_FREQ_BINS_KEEP / 2) as f64) * LONG_FFT_BINWIDTH
    }

    fn extract_packet_samples(&mut self, long_fft_index: u64) {
        let packet_start = long_fft_index * u64::try_from(NFFT_SHORT).unwrap();
        let (a, b) = self
            .iq
            .get(
                packet_start - u64::try_from(LEFT_MARGIN).unwrap()
                    ..packet_start + u64::try_from(MODE_S_LONG_SAMPLES + RIGHT_MARGIN).unwrap(),
            )
            .unwrap();
        self.packet_samples[..a.len()].copy_from_slice(a);
        if !b.is_empty() {
            self.packet_samples[a.len()..].copy_from_slice(b);
        }
    }

    fn decode_packet(&mut self) {
        let frequency_estimate = self.packet.detection.carrier_frequency_offset;
        frequency_shift(&mut self.packet_samples, -frequency_estimate);
        let carrier_phasor = estimate_carrier_phasor(
            self.packet_samples[LEFT_MARGIN..LEFT_MARGIN + MODE_S_LONG_SAMPLES]
                .try_into()
                .unwrap(),
        );
        phasor_shift(&mut self.packet_samples, carrier_phasor.conj());
        accumulate_symbols(&self.packet_samples, &mut self.symbols);
        for (symbols, (coherent, noncoherent)) in self.symbols.iter().zip(
            self.coherent_demod
                .iter_mut()
                .zip(self.noncoherent_demod.iter_mut()),
        ) {
            demodulate(symbols, coherent, noncoherent);
        }

        let carrier_power = self.packet.detection.carrier_power;
        let Some(location) = self.find_best_decode_candidate(carrier_power) else {
            self.packet.decode = None;
            return;
        };
        let demod = match location.demodulation {
            DemodulationType::Coherent => &self.coherent_demod,
            DemodulationType::NonCoherent => &self.noncoherent_demod,
        };
        let mut sample_index = self.packet.detection.sample_index
            - u64::try_from(LEFT_MARGIN).unwrap()
            + u64::try_from(
                SPS * (2 * location.bit_offset + location.sym_offset) + location.clock_phase,
            )
            .unwrap();
        let mut message = extract_message(
            &demod[location.clock_phase][location.sym_offset],
            location.bit_offset,
        );
        correct_message_bit_errors(&mut message, location.corrected_errors);
        let estimates = self.postdecode_estimates(&message, &location);
        // The reference point for the sample index is the 50% of the rising
        // edge of the first bit, but the receiver subsample timing gives zero
        // (equal early and late powers) when the 50% level is exactly halfway
        // between sample_index - 1 and sample_index. This means that we need to
        // subtract 0.5 samples to the subsample timing.
        let mut sample_index_frac = estimates.subsample_timing - 0.5;
        if sample_index_frac < -0.5 {
            sample_index -= 1;
            sample_index_frac += 1.0;
        } else if sample_index_frac >= 0.5 {
            sample_index += 1;
            sample_index_frac -= 1.0;
        }
        debug_assert!((-0.5..0.5).contains(&sample_index_frac));
        self.packet.decode = Some(ADSBDecode {
            sample_index,
            sample_index_frac,
            message,
            corrected_errors: location.corrected_errors,
            demodulation: location.demodulation,
            cn0_db: estimates.cn0_db,
        });
    }

    fn find_best_decode_candidate(&self, carrier_power: f32) -> Option<DecodeLocation> {
        let preamble_corr_threshold = self.preamble_metric_threshold * carrier_power;
        const PACKET_WINDOW_BITS: usize = 8 * PACKET_WINDOW_BYTES;
        let mut best_preamble_corr = 0.0;
        let mut best_location: Option<DecodeLocation> = None;
        // Technically the maximum offset that we could use in these loops to
        // avoid going out of bounds (including in the late correlator
        // calculation), is
        //
        // (bit_offset, clock_phase, sym_offset) =
        //    (PACKET_WINDOW_BITS - MODE_S_LONG_BITS, SPS - 2, 0).
        //
        // However, for simplicity we keep uniform limits (that is, we search on
        // a 3D box of indices rather than on some "jagged" volume in 3D index
        // space), and stop at
        // (PACKET_WINDOW_BITS - MODE_S_LONG_BITS - 1, SPS - 1, 1).
        //
        // The start of this range, bit_offset = REPLY_PREAMBLE_BITS, comes from
        // the requirement to avoid out-of-bounds when computing preamble_loc
        // below.
        for bit_offset in REPLY_PREAMBLE_BITS..PACKET_WINDOW_BITS - MODE_S_LONG_BITS {
            for clock_phase in 0..SPS {
                for sym_offset in 0..2 {
                    if let Some(location) = self.try_decode(bit_offset, clock_phase, sym_offset) {
                        const PREAMBLE_SYMBOLS: usize = 2 * REPLY_PREAMBLE_BITS;
                        let preamble_loc = 2 * bit_offset + sym_offset - PREAMBLE_SYMBOLS;
                        let corr = preamble_corr(
                            self.symbols[clock_phase]
                                [preamble_loc..preamble_loc + PREAMBLE_MASK_LEN]
                                .try_into()
                                .unwrap(),
                        );
                        // To cut down false decodes, if there are 2 or more bit
                        // errors we require the preamble correlation to be
                        // above a quality metric threshold.
                        if location.corrected_errors.num_errors() >= 2
                            && corr < preamble_corr_threshold
                        {
                            continue;
                        }
                        // The best location is chosen according to what gives
                        // the largest value for lexicographic order in
                        // (-num_errors, preamble_corr).
                        if best_location.is_none()
                            || (
                                std::cmp::Reverse(location.corrected_errors.num_errors()),
                                corr,
                            ) > (
                                std::cmp::Reverse(
                                    best_location
                                        .as_ref()
                                        .unwrap()
                                        .corrected_errors
                                        .num_errors(),
                                ),
                                best_preamble_corr,
                            )
                        {
                            best_preamble_corr = corr;
                            best_location = Some(location)
                        }
                    }
                }
            }
        }
        best_location
    }

    fn try_decode(
        &self,
        bit_offset: usize,
        clock_phase: usize,
        sym_offset: usize,
    ) -> Option<DecodeLocation> {
        let mut location = DecodeLocation {
            bit_offset,
            clock_phase,
            sym_offset,
            // the following fields are overwritten below if needed
            corrected_errors: BitErrors::NoErrors,
            demodulation: DemodulationType::Coherent,
        };
        // first try coherent decoding
        let coherent_crc = compute_crc(&self.coherent_demod[clock_phase][sym_offset], bit_offset);
        if coherent_crc == 0
            && df_is_valid_extended_squitter(
                &self.coherent_demod[clock_phase][sym_offset],
                bit_offset,
                location.corrected_errors,
            )
        {
            return Some(location);
        }
        // if that fails, try noncoherent decoding
        let noncoherent_crc =
            compute_crc(&self.noncoherent_demod[clock_phase][sym_offset], bit_offset);
        if noncoherent_crc == 0
            && df_is_valid_extended_squitter(
                &self.noncoherent_demod[clock_phase][sym_offset],
                bit_offset,
                location.corrected_errors,
            )
        {
            location.demodulation = DemodulationType::NonCoherent;
            return Some(location);
        }
        // if we cannot correct bit errors, fail decoding
        if self.error_correction_level == ErrorCorrectionLevel::NoErrors {
            return None;
        }
        // try decoding with single-bit error patterns on the coherent decode
        if let Some(j) = single_bit_error_from_crc24(coherent_crc) {
            location.corrected_errors = BitErrors::SingleError(j);
            if df_is_valid_extended_squitter(
                &self.coherent_demod[clock_phase][sym_offset],
                bit_offset,
                location.corrected_errors,
            ) {
                return Some(location);
            }
        }
        // if that fails, and the noncoherent decode is different, try single-bit error patterns there
        if noncoherent_crc != coherent_crc
            && let Some(j) = single_bit_error_from_crc24(noncoherent_crc)
        {
            location.corrected_errors = BitErrors::SingleError(j);
            location.demodulation = DemodulationType::NonCoherent;
            if df_is_valid_extended_squitter(
                &self.noncoherent_demod[clock_phase][sym_offset],
                bit_offset,
                location.corrected_errors,
            ) {
                return Some(location);
            }
        }
        // if we can only correct single-bit errors, fail decoding
        if self.error_correction_level == ErrorCorrectionLevel::SingleError {
            return None;
        }
        assert_eq!(
            self.error_correction_level,
            ErrorCorrectionLevel::DoubleError
        );
        // try decoding with double-bit error patterns on the coherent decode
        if let Some((j, k)) = double_bit_error_from_crc24(coherent_crc) {
            location.corrected_errors = BitErrors::DoubleError(j, k);
            if df_is_valid_extended_squitter(
                &self.coherent_demod[clock_phase][sym_offset],
                bit_offset,
                location.corrected_errors,
            ) {
                return Some(location);
            }
        }
        // if that fails, and the noncoherent decode is different, try double-bit error patterns there
        if noncoherent_crc != coherent_crc
            && let Some((j, k)) = double_bit_error_from_crc24(noncoherent_crc)
        {
            location.corrected_errors = BitErrors::DoubleError(j, k);
            location.demodulation = DemodulationType::NonCoherent;
            if df_is_valid_extended_squitter(
                &self.noncoherent_demod[clock_phase][sym_offset],
                bit_offset,
                location.corrected_errors,
            ) {
                return Some(location);
            }
        }
        // at this point there is nothing we can do to decode
        None
    }

    fn postdecode_estimates(
        &self,
        message: &[u8; MODE_S_LONG_BYTES],
        location: &DecodeLocation,
    ) -> PostdecodeEstimates {
        let prompt_power = postdecode_power(self.symbols_for_location(location), message);
        let early_power = postdecode_power(
            self.symbols_for_location(&location.previous_clock_phase()),
            message,
        );
        let late_power = postdecode_power(
            self.symbols_for_location(&location.next_clock_phase()),
            message,
        );
        let subsample_timing = parabolic_interpolation([early_power, prompt_power, late_power]);
        // We don't have a guarantee that prompt_power is larger than
        // early_power and late_power, so parabolic interpolation could give a
        // result that is very large. We clamp the result to [-1.0, 1.0],
        // because if decoding chose `location` as the best candidate, the true
        // timing cannot be much further off than one sample.
        let subsample_timing = subsample_timing.clamp(-1.0, 1.0);

        let noise_power = postdecode_noise_power(self.samples_for_location(location), message);
        // The noise power measurement has the same coherent integration length
        // as the prompt_power measurement, but the noncoherent integration
        // length is a factor of two shorter. Therefore, prompt_power / (2 *
        // noise_power) is an estimate for (S+N)/N, and from there we can estimate the CN0.
        let noise = 2.0 * noise_power;
        let signal = (prompt_power - noise).max(0.0);
        let sn = signal / noise;
        // Due to the coherent integration of length SPS done in the
        // noise_power, the noise bandwidth here is SAMPLE_RATE / SPS.
        const SCALE_FACTOR: f32 = SAMPLE_RATE_F32 / SPS as f32;
        let cn0_db = 10.0 * (sn * SCALE_FACTOR).log10();

        PostdecodeEstimates {
            subsample_timing,
            cn0_db,
        }
    }

    fn samples_for_location(&self, location: &DecodeLocation) -> &[Complex32; MODE_S_LONG_SAMPLES] {
        let start = SPS * (2 * location.bit_offset + location.sym_offset) + location.clock_phase;
        self.packet_samples[start..start + MODE_S_LONG_SAMPLES]
            .try_into()
            .unwrap()
    }

    fn symbols_for_location(&self, location: &DecodeLocation) -> &[Complex32; MODE_S_LONG_SYMBOLS] {
        let start = 2 * location.bit_offset + location.sym_offset;
        self.symbols[location.clock_phase][start..start + MODE_S_LONG_SYMBOLS]
            .try_into()
            .unwrap()
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
struct DecodeLocation {
    bit_offset: usize,
    clock_phase: usize,
    sym_offset: usize,
    corrected_errors: BitErrors,
    demodulation: DemodulationType,
}

impl DecodeLocation {
    fn previous_clock_phase(&self) -> DecodeLocation {
        let mut bit_offset = self.bit_offset;
        let mut sym_offset = self.sym_offset;
        let clock_phase = if self.clock_phase == 0 {
            if sym_offset == 0 {
                debug_assert_ne!(bit_offset, 0);
                bit_offset -= 1;
                sym_offset = 1;
            } else {
                sym_offset -= 1;
            }
            SPS - 1
        } else {
            self.clock_phase - 1
        };
        DecodeLocation {
            bit_offset,
            clock_phase,
            sym_offset,
            corrected_errors: self.corrected_errors,
            demodulation: self.demodulation,
        }
    }

    fn next_clock_phase(&self) -> DecodeLocation {
        let mut bit_offset = self.bit_offset;
        let mut sym_offset = self.sym_offset;
        let clock_phase = if self.clock_phase == SPS - 1 {
            if sym_offset == 1 {
                bit_offset += 1;
                sym_offset = 0;
            } else {
                sym_offset += 1;
            }
            0
        } else {
            self.clock_phase + 1
        };
        DecodeLocation {
            bit_offset,
            clock_phase,
            sym_offset,
            corrected_errors: self.corrected_errors,
            demodulation: self.demodulation,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct PostdecodeEstimates {
    subsample_timing: f32,
    cn0_db: f32,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::constants::dsp::NFFT_LONG;
    use proptest::prelude::*;
    use std::f64::consts::PI;

    prop_compose! {
        fn any_decode_location()(
            bit_offset in REPLY_PREAMBLE_BITS..8 * PACKET_WINDOW_BYTES - MODE_S_LONG_BITS,
            clock_phase in 0..SPS,
            sym_offset in 0_usize..2,
            corrected_errors: BitErrors,
            noncoherent: bool,
        ) -> DecodeLocation {
            DecodeLocation {
                bit_offset,
                clock_phase,
                sym_offset,
                corrected_errors,
                demodulation: if noncoherent {
                    DemodulationType::NonCoherent
                } else {
                    DemodulationType::Coherent
                },
            }
        }
    }

    impl Arbitrary for DecodeLocation {
        type Parameters = ();
        type Strategy = BoxedStrategy<DecodeLocation>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            any_decode_location().boxed()
        }
    }

    impl Arbitrary for BitErrors {
        type Parameters = ();
        type Strategy = BoxedStrategy<BitErrors>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            let num_bits = u8::try_from(MODE_S_LONG_BITS).unwrap();
            prop_oneof![
                Just(BitErrors::NoErrors),
                (0..num_bits).prop_map(BitErrors::SingleError),
                (0..num_bits - 1).prop_flat_map(move |j| {
                    (j + 1..num_bits).prop_map(move |k| BitErrors::DoubleError(j, k))
                }),
            ]
            .boxed()
        }
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    enum SignalType {
        Dc,
        Awgn,
    }

    fn dc_iq(num_samples: usize) -> Vec<Complex32> {
        vec![Complex32::new(1.0, 0.0); num_samples]
    }

    fn any_awgn() -> impl Strategy<Value = Complex32> {
        any::<u64>().prop_map(|seed| {
            let mut rng = crate::simulation::rng::rng_from_seed(seed);
            crate::simulation::generate_awgn(&mut rng, 1.0)
        })
    }

    #[test]
    fn set_error_correction_level() {
        let mut receiver = ADSBReceiver::new();
        assert_eq!(
            receiver.error_correction_level(),
            ErrorCorrectionLevel::DoubleError
        );
        for level in [
            ErrorCorrectionLevel::NoErrors,
            ErrorCorrectionLevel::SingleError,
            ErrorCorrectionLevel::DoubleError,
        ] {
            receiver.set_error_correction_level(level);
            assert_eq!(receiver.error_correction_level(), level);
        }
    }

    fn any_threshold() -> proptest::num::f32::Any {
        use proptest::num::f32::{POSITIVE, SUBNORMAL, ZERO};
        POSITIVE | ZERO | SUBNORMAL
    }

    #[test]
    fn set_carrier_metric_threshold_rejects_invalid() {
        let mut receiver = ADSBReceiver::new();
        assert!(receiver.set_carrier_metric_threshold(f32::NAN).is_err());
        assert!(
            receiver
                .set_carrier_metric_threshold(f32::INFINITY)
                .is_err()
        );
        assert!(
            receiver
                .set_carrier_metric_threshold(f32::NEG_INFINITY)
                .is_err()
        );
        assert!(receiver.set_carrier_metric_threshold(-1.0).is_err());
    }

    proptest! {
        #[test]
        fn set_carrier_metric_threshold_accepts_valid(threshold in any_threshold()) {
            let mut receiver = ADSBReceiver::new();
            receiver.set_carrier_metric_threshold(threshold).unwrap();
            assert_eq!(receiver.carrier_metric_threshold(), threshold);
        }
    }

    #[test]
    fn set_preamble_metric_threshold_rejects_invalid() {
        let mut receiver = ADSBReceiver::new();
        assert!(receiver.set_preamble_metric_threshold(f32::NAN).is_err());
        assert!(
            receiver
                .set_preamble_metric_threshold(f32::INFINITY)
                .is_err()
        );
        assert!(
            receiver
                .set_preamble_metric_threshold(f32::NEG_INFINITY)
                .is_err()
        );
        assert!(receiver.set_preamble_metric_threshold(-1.0).is_err());
    }

    proptest! {
        #[test]
        fn set_preamble_metric_threshold_accepts_valid(threshold in any_threshold()) {
            let mut receiver = ADSBReceiver::new();
            receiver.set_preamble_metric_threshold(threshold).unwrap();
            assert_eq!(receiver.preamble_metric_threshold(), threshold);
        }
    }

    fn short_fft(signal: SignalType, iq: Vec<Complex32>) {
        let mut receiver = ADSBReceiver::new();
        receiver.process_chunk(iq[..].try_into().unwrap());
        assert_eq!(receiver.short_ffts.samples_written(), 1);
        let short_fft = receiver.short_ffts[0];
        let mut iq_zero_pad = iq.clone();
        iq_zero_pad.resize(NUM_FREQ_BINS, Complex32::new(0.0, 0.0));
        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(NUM_FREQ_BINS);
        fft.process(&mut iq_zero_pad);
        let fft_zero_pad = iq_zero_pad;
        // fft_zero_pad is non-FFT-shifted, while short_fft is FFT-shifted
        // and is only the central 1/2
        let rel_tolerance = 5e-4;
        // AWGN needs an absolute tolerance because otherwise it's possible to
        // get small values in some bins in some cases that do not satisfy the relative tolerance.
        let abs_tolerance = match signal {
            SignalType::Dc => 0.0,
            SignalType::Awgn => 1e-5,
        };
        assert_all_close!(
            &short_fft[..NUM_FREQ_BINS_KEEP / 2],
            &fft_zero_pad[3 * NUM_FREQ_BINS / 4..],
            abs_tolerance,
            rel_tolerance,
        );
        assert_all_close!(
            &short_fft[NUM_FREQ_BINS_KEEP / 2..],
            &fft_zero_pad[..NUM_FREQ_BINS / 4],
            abs_tolerance,
            rel_tolerance,
        );
        if matches!(signal, SignalType::Dc) {
            // The central element is the DC term complex amplitude. This checks the
            // FFT scaling.
            let expected_dc = Complex32::new(NFFT_SHORT as f32, 0.0);
            let dc = short_fft[NUM_FREQ_BINS_KEEP / 2];
            assert!((dc - expected_dc).norm() <= rel_tolerance * expected_dc.re);
        }
    }

    /// Tests that, with a DC input, the short FFTs that are computed match an
    /// equivalent zero-padded FFT.
    #[test]
    fn short_fft_dc_input() {
        short_fft(SignalType::Dc, dc_iq(NFFT_SHORT));
    }

    proptest! {
        #[test]
        fn short_fft_awgn(
            iq in proptest::collection::vec(any_awgn(), NFFT_SHORT)
        ) {
            short_fft(SignalType::Awgn, iq);
        }
    }

    fn long_fft(signal: SignalType, iq: Vec<Complex32>) {
        let mut receiver = ADSBReceiver::new();
        assert!(iq.len().is_multiple_of(NFFT_SHORT));
        for chunk in iq.as_chunks::<NFFT_SHORT>().0 {
            receiver.process_chunk(chunk);
        }
        assert_eq!(
            receiver.short_ffts.samples_written(),
            u64::try_from(FFT_OVERLAP_FACTOR).unwrap()
        );
        assert_eq!(receiver.long_ffts.samples_written(), 1);
        let results = &receiver.long_ffts[0];
        let long_fft = &results.fft_power;
        let mut iq_zero_pad = iq.clone();
        iq_zero_pad.resize(NUM_FREQ_BINS, Complex32::new(0.0, 0.0));
        let mut fft_planner = FftPlanner::new();
        let fft = fft_planner.plan_fft_forward(NUM_FREQ_BINS);
        fft.process(&mut iq_zero_pad);
        let fft_zero_pad = iq_zero_pad
            .into_iter()
            .map(|z| z.norm_sqr())
            .collect::<Vec<f32>>();
        // fft_zero_pad is non-FFT-shifted, while long_fft is FFT-shifted
        // and is only the central 1/2
        let rel_tolerance = 1e-3;
        let abs_tolerance = 1e-6;
        assert_all_close!(
            &long_fft[..NUM_FREQ_BINS_KEEP / 2],
            &fft_zero_pad[3 * NUM_FREQ_BINS / 4..],
            abs_tolerance,
            rel_tolerance,
        );
        assert_all_close!(
            &long_fft[NUM_FREQ_BINS_KEEP / 2..],
            &fft_zero_pad[..NUM_FREQ_BINS / 4],
            abs_tolerance,
            rel_tolerance,
        );
        assert_eq!(
            long_fft[usize::try_from(results.max_bin).unwrap()],
            results.max_value
        );
        const START: usize = NUM_FREQ_BINS_KEEP / 2 - MAX_CARRIER_FREQ_OFFSET_BINS;
        const END: usize = NUM_FREQ_BINS_KEEP / 2 + MAX_CARRIER_FREQ_OFFSET_BINS + 1;
        let mut max = 0.0;
        for &x in long_fft[START..END].iter() {
            if x > max {
                max = x;
            }
        }
        assert_eq!(max, results.max_value);
        if matches!(signal, SignalType::Dc) {
            // The central element is the DC term power. This checks the FFT
            // scaling.
            let expected_dc = Complex32::new((NFFT_LONG * NFFT_LONG) as f32, 0.0);
            let dc = long_fft[NUM_FREQ_BINS_KEEP / 2];
            assert!((dc - expected_dc).norm() <= rel_tolerance * expected_dc.re);
            assert_eq!(
                usize::try_from(results.max_bin).unwrap(),
                NUM_FREQ_BINS_KEEP / 2
            );
        }
    }

    /// Tests that, with a DC input, the long FFTs that are computed match an
    /// equivalent zero-padded FFT.
    #[test]
    fn long_fft_dc_input() {
        long_fft(SignalType::Dc, dc_iq(NFFT_LONG));
    }

    proptest! {
        /// Tests that, with AWGN input, the long FFTs that are computed match an
        /// equivalent zero-padded FFT.
        #[test]
        fn long_fft_awgn_input(
            iq in proptest::collection::vec(any_awgn(), NFFT_LONG),
        ) {
            long_fft(SignalType::Awgn, iq);
        }
    }

    proptest! {
        #[test]
        fn check_local_maximum(
            fft_max_values in proptest::collection::vec(
                0.0_f32..1000.0,
                LOCAL_MAX_WINDOW_SIZE as usize..LOCAL_MAX_WINDOW_SIZE as usize + 128)
        ) {
            let mut receiver = ADSBReceiver::new();
            for &max_value in fft_max_values.iter() {
                let long_fft = LongFFT {
                    // these two fields are unused by check_local_maximum
                    fft_power: aligned::Aligned([0.0; NUM_FREQ_BINS_KEEP]),
                    max_bin: 0,
                    max_value,
                };
                receiver.long_ffts.push(long_fft);
            }
            const WINDOW: usize = LOCAL_MAX_WINDOW as usize;
            let k = fft_max_values.len() - WINDOW - 1;
            let window = &fft_max_values[k - WINDOW..=k + WINDOW];
            let is_local_max = fft_max_values[k] == *window
                .iter()
                .max_by(|x, y| x.partial_cmp(y).unwrap())
                .unwrap();
            assert_eq!(receiver.check_local_maximum(u64::try_from(k).unwrap()),
                       is_local_max);
        }
    }

    fn check_carrier_metric(signal: SignalType, iq: Vec<Complex32>) {
        let mut receiver = ADSBReceiver::new();
        assert!(iq.len().is_multiple_of(NFFT_SHORT));
        for chunk in iq.as_chunks::<NFFT_SHORT>().0 {
            receiver.process_chunk(chunk);
        }
        for k in 0..receiver.long_ffts.samples_written() {
            let expected = match signal {
                SignalType::Dc => true,
                SignalType::Awgn => false,
            };
            assert_eq!(receiver.check_carrier_metric(k).is_some(), expected);
        }
    }

    #[test]
    fn check_carrier_metric_dc_input() {
        check_carrier_metric(SignalType::Dc, dc_iq(NFFT_LONG));
    }

    proptest! {
        #[test]
        fn check_carrier_metric_awgn_input(
            iq in proptest::collection::vec(any_awgn(), NFFT_LONG),
        ) {
            check_carrier_metric(SignalType::Awgn, iq);
        }
    }

    proptest! {
        #[test]
        fn estimate_carrier_frequency_offset_cw(
            phase in -PI..PI,
            freq in -1e6_f64..=1e6,
        ) {
            let cw = (0..NFFT_LONG)
                .map(|j| {
                    let phi = phase + 2.0 * PI * freq * j as f64 / SAMPLE_RATE_F64;
                    let (im, re) = phi.sin_cos();
                    Complex32::new(re as f32, im as f32)
                })
                .collect::<Vec<_>>();
            let mut receiver = ADSBReceiver::new();
            for chunk in cw.as_chunks::<NFFT_SHORT>().0 {
                receiver.process_chunk(chunk);
            }
            let fft_idx = 0;
            let estimate = receiver.estimate_carrier_frequency_offset(fft_idx);
            let error_bound = 40.0;
            assert!((estimate - freq).abs() <= error_bound,
                    "estimate = {estimate}, freq = {freq}");
        }
    }

    prop_compose! {
        fn awgn_chunks()(num_chunks in (LOCAL_MAX_WINDOW_SIZE as usize + FFT_OVERLAP_FACTOR - 1)..=128)
            (iq in proptest::collection::vec(any_awgn(), num_chunks * NFFT_SHORT))
             -> Vec<Complex32> {
                iq
            }
    }

    proptest! {
        #[test]
        fn extract_packet_samples(iq in awgn_chunks()) {
            let mut receiver = ADSBReceiver::new();
            for chunk in iq.as_chunks::<NFFT_SHORT>().0 {
                receiver.process_chunk(chunk);
            }
            let k = receiver.long_ffts.samples_written() - LOCAL_MAX_WINDOW - 1;
            receiver.extract_packet_samples(k);
            let offset = usize::try_from(k).unwrap() * NFFT_SHORT;
            let window = &iq[offset - LEFT_MARGIN..offset + MODE_S_LONG_SAMPLES + RIGHT_MARGIN];
            assert_eq!(&receiver.packet_samples[..], window);
        }
    }

    prop_compose! {
        pub fn any_message()(
            message in proptest::collection::vec(any::<u8>(), MODE_S_LONG_BYTES)
        ) ->[u8; MODE_S_LONG_BYTES] {
            message.try_into().unwrap()
        }
    }

    pub fn copy_message_to_bytes(message: &[u8], bytes: &mut [u8], bit_offset: usize) {
        let byte_offset = bit_offset / 8;
        let bit_offset = bit_offset - 8 * byte_offset;
        if bit_offset == 0 {
            bytes[byte_offset..byte_offset + MODE_S_LONG_BYTES].copy_from_slice(message);
        } else {
            for (k, b) in message.iter().enumerate() {
                let n = byte_offset + k;
                bytes[n] &= 0xff << (8 - bit_offset);
                bytes[n] |= b >> bit_offset;
                bytes[n + 1] &= 0xff >> bit_offset;
                bytes[n + 1] |= b << (8 - bit_offset);
            }
        }
    }

    proptest! {
        #[test]
        fn try_decode(
            location: DecodeLocation,
            mut message in any_message(),
        ) {
            let mut receiver = ADSBReceiver::new();
            // fill all the buffers with a pattern that doesn't produce decodes
            // (the all-zeros pattern produces valid decodes)
            for a in receiver.coherent_demod.iter_mut() {
                for b in a.iter_mut() {
                    b.fill(0x55);
                }
            }
            for a in receiver.noncoherent_demod.iter_mut() {
                for b in a.iter_mut() {
                    b.fill(0x55);
                }
            }
            assert!(receiver.try_decode(
                location.bit_offset,
                location.clock_phase,
                location.sym_offset).is_none());

            // put a valid DF17 header to ensure that the message passes the
            // df_is_valid_extended_squitter check
            message[0] = 17 << 3;
            // ovewrite the message CRC-24 with the correct one
            let crc = crate::mode_s::CRC24.checksum(&message[..MODE_S_LONG_BYTES - 3]);
            message[MODE_S_LONG_BYTES - 3] = u8::try_from((crc >> 16) & 0xff).unwrap();
            message[MODE_S_LONG_BYTES - 2] = u8::try_from((crc >> 8) & 0xff).unwrap();
            message[MODE_S_LONG_BYTES - 1] = u8::try_from(crc & 0xff).unwrap();
            assert_eq!(crate::mode_s::CRC24.checksum(&message), 0);

            // apply bit errors to message
            correct_message_bit_errors(&mut message, location.corrected_errors);

            // copy message to buffer
            let buffer = match location.demodulation {
                DemodulationType::Coherent =>
                    &mut receiver.coherent_demod[location.clock_phase][location.sym_offset],
                DemodulationType::NonCoherent =>
                    &mut receiver.noncoherent_demod[location.clock_phase][location.sym_offset],
            };
            copy_message_to_bytes(&message, buffer, location.bit_offset);

            let decoded_location = receiver.try_decode(
                location.bit_offset,
                location.clock_phase,
                location.sym_offset).unwrap();
            assert_eq!(decoded_location, location);
        }
    }

    proptest! {
        #[test]
        fn samples_for_location(
            location: DecodeLocation,
            packet_samples in proptest::collection::vec(any_awgn(), PACKET_WINDOW_SAMPLES),
            expected_samples in proptest::collection::vec(any_awgn(), MODE_S_LONG_SAMPLES),
        ) {
            let mut receiver = ADSBReceiver::new();
            receiver.packet_samples.copy_from_slice(&packet_samples);
            let a = SPS * (2 * location.bit_offset + location.sym_offset) + location.clock_phase;
            receiver.packet_samples[a..a + MODE_S_LONG_SAMPLES].copy_from_slice(&expected_samples);
            let samples = receiver.samples_for_location(&location);
            assert_eq!(samples, &expected_samples[..]);
        }
    }

    proptest! {
        fn symbols_for_location(
            location: DecodeLocation,
            symbols in proptest::collection::vec(any_awgn(), PACKET_WINDOW_SYMBOLS),
            expected_symbols in proptest::collection::vec(any_awgn(), MODE_S_LONG_SYMBOLS),
        ) {
            let mut receiver = ADSBReceiver::new();
            for clock_phase in 0..SPS {
                receiver.symbols[clock_phase].copy_from_slice(&symbols);
            }
            let a = 2 * location.bit_offset + location.sym_offset;
            receiver.symbols[location.clock_phase][a..a + MODE_S_LONG_SYMBOLS]
                .copy_from_slice(&expected_symbols);
            let symbols = receiver.symbols_for_location(&location);
            assert_eq!(symbols, &expected_symbols[..]);
        }
    }

    #[test]
    fn previous_clock_phase() {
        macro_rules! check {
            (($b0:expr, $c0:expr, $s0:expr), ($b1:expr, $c1:expr, $s1:expr)) => {
                assert_eq!(
                    DecodeLocation {
                        bit_offset: $b0,
                        clock_phase: $c0,
                        sym_offset: $s0,
                        corrected_errors: BitErrors::NoErrors,
                        demodulation: DemodulationType::Coherent,
                    }
                    .previous_clock_phase(),
                    DecodeLocation {
                        bit_offset: $b1,
                        clock_phase: $c1,
                        sym_offset: $s1,
                        corrected_errors: BitErrors::NoErrors,
                        demodulation: DemodulationType::Coherent,
                    }
                );
            };
        }

        check!((8, 0, 0), (7, 3, 1));
        check!((8, 0, 1), (8, 3, 0));
        check!((8, 1, 0), (8, 0, 0));
    }

    #[test]
    fn next_clock_phase() {
        macro_rules! check {
            (($b0:expr, $c0:expr, $s0:expr), ($b1:expr, $c1:expr, $s1:expr)) => {
                assert_eq!(
                    DecodeLocation {
                        bit_offset: $b0,
                        clock_phase: $c0,
                        sym_offset: $s0,
                        corrected_errors: BitErrors::NoErrors,
                        demodulation: DemodulationType::Coherent,
                    }
                    .next_clock_phase(),
                    DecodeLocation {
                        bit_offset: $b1,
                        clock_phase: $c1,
                        sym_offset: $s1,
                        corrected_errors: BitErrors::NoErrors,
                        demodulation: DemodulationType::Coherent,
                    }
                );
            };
        }

        check!((8, 3, 1), (9, 0, 0));
        check!((8, 3, 0), (8, 0, 1));
        check!((8, 2, 1), (8, 3, 1));
    }

    #[test]
    fn high_cn0_decode() {
        use crate::simulation::{
            MessageConfig, MessageType, SingleMessageSimulation,
            distributions::{CN0Distribution, ConstantCN0, MessageDistributions},
            rng::rng_from_seed,
        };
        use std::time::Duration;

        let mut rng = rng_from_seed(0);
        let cn0_db = 75.0;
        let start = 1600.0; // 200 usec
        let simulation = SingleMessageSimulation {
            duration: Duration::from_millis(2),
            message_config: MessageConfig::from_distributions(
                start,
                MessageDistributions {
                    cn0: CN0Distribution::Constant(ConstantCN0::new(cn0_db).unwrap()),
                    ..Default::default()
                },
                MessageType::ExtendedSquitter,
                &mut rng,
            ),
        };
        let iq = simulation.run(&mut rng);
        assert!(iq.len().is_multiple_of(NFFT_SHORT));
        let mut packet = None;
        let mut receiver = ADSBReceiver::new();
        for chunk in iq.as_chunks::<NFFT_SHORT>().0 {
            if let Some(p) = receiver.process_chunk(chunk) {
                assert!(packet.is_none());
                packet = Some(p.clone());
            }
        }
        let packet = packet.unwrap();
        let reply_preamble_duration =
            REPLY_PREAMBLE_SAMPLES as f64 * (1.0 - simulation.message_config.symbol_clock_error);
        let packet_start = simulation.message_config.timestamp_samples + reply_preamble_duration;
        // In infinite SNR, the detection timing error should be bounded by
        // NFFT_SHORT / 2, which is half of the time step, but in finite SNR
        // there can be a slightly larger error because of noise.
        assert!(
            (packet.detection.sample_index as f64 - packet_start).abs()
                <= (5 * NFFT_SHORT / 8) as f64
        );
        assert!(
            (packet.detection.carrier_frequency_offset
                - simulation.message_config.carrier_frequency_offset)
                .abs()
                <= 200.0
        );
        let decode = packet.decode.as_ref().unwrap();
        assert!(
            (decode.sample_index as f64 + f64::from(decode.sample_index_frac) - packet_start).abs()
                <= 0.2
        );
        let crate::mode_s::ModeSMessage::Long(expected_message) =
            &simulation.message_config.message
        else {
            panic!();
        };
        assert_eq!(&decode.message, expected_message);
        assert_eq!(decode.corrected_errors, BitErrors::NoErrors);
        assert_eq!(decode.demodulation, DemodulationType::Coherent);
        assert!((f64::from(decode.cn0_db) - cn0_db).abs() <= 0.5);
    }
}
