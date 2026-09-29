//! Constant definitions.
//!
//! This module contains constant definitions used by [`more1090`](crate).

/// Digital signal processing constants.
pub mod dsp {
    /// The sample rate used by the receiver (8 Msps) as a `usize`.
    pub const SAMPLE_RATE_USIZE: usize = 8_000_000;

    /// The sample rate used by the receiver (8 Msps) as an `f32`.
    pub const SAMPLE_RATE_F32: f32 = SAMPLE_RATE_USIZE as f32;

    /// The sample rate used by the receiver (8 Msps) as an `f64`.
    pub const SAMPLE_RATE_F64: f64 = SAMPLE_RATE_USIZE as f64;

    /// The ADS-B carrier frequency (1090 MHz) in Hz.
    pub const CARRIER_FREQUENCY: f64 = 1_090_000_000.0;

    /// Samples per symbol.
    ///
    /// A symbol here is a 500 ns pulse.
    pub const SPS: usize = 4;

    /// Short FFT size.
    ///
    /// This is the short FFT size used for carrier detection.
    pub const NFFT_SHORT: usize = 128;

    /// FFT overlap factor.
    ///
    /// Long FFTs are advanced by [`NFFT_SHORT`], and the long FFT size
    /// ([`NFFT_LONG`]) is defined as `NFFT_SHORT` times this factor.
    pub const FFT_OVERLAP_FACTOR: usize = 7;

    /// Long FFT size.
    ///
    /// This is the long FFT size used for carrier detection. Long FFTs are not
    /// calculated directly. They are calculated by stacking
    /// [`FFT_OVERLAP_FACTOR`] short FFTs.
    pub const NFFT_LONG: usize = NFFT_SHORT * FFT_OVERLAP_FACTOR;

    /// Frequency bin factor.
    ///
    /// FFTs are spaced in the frequency domain by the inverse of the long FFT
    /// bin size times this factor.
    pub const FREQ_BIN_FACTOR: usize = 4;

    /// Frequency bin factor referenced to the short FFT.
    pub const FREQ_BIN_FACTOR_SHORT: usize = FREQ_BIN_FACTOR * FFT_OVERLAP_FACTOR;

    /// Total number of frequency bins.
    ///
    /// This is equal to the long FFT size times the frequency bin factor.
    pub const NUM_FREQ_BINS: usize = NFFT_LONG * FREQ_BIN_FACTOR;

    /// Total number of frequency bins that are kept.
    ///
    /// This is equal to [`NUM_FREQ_BINS`] divided by 2, since only the central
    /// half of the spectrum is kept for carrier detection.
    pub const NUM_FREQ_BINS_KEEP: usize = NUM_FREQ_BINS / 2;

    /// Frequency bin periodicity
    ///
    /// This is `lcm(FFT_OVERLAP_FACTOR, FREQ_BIN_FACTOR)`, but Rust doesn't
    /// support calculating this as a constant, so the scalar is hardcoded and
    /// unit tested.
    pub const FREQ_BIN_PERIODICITY: usize = 28;

    /// Length of frequency bin local oscillator.
    pub const FREQ_BIN_LO_LEN: usize = FREQ_BIN_PERIODICITY * NFFT_SHORT;

    /// Window used for local maximum detection criteria.
    ///
    /// The local maximum detection criterion requires time bin `k` to be
    /// largest than or equal to all time bins between `k - LOCAL_MAX_WINDOW`
    /// and `k + LOCAL_MAX_WINDOW` (both included.
    pub const LOCAL_MAX_WINDOW: u64 = 2;

    /// Size of the local maximum window.
    ///
    /// This is the total size of the local maximum window.
    pub const LOCAL_MAX_WINDOW_SIZE: u64 = 2 * LOCAL_MAX_WINDOW + 1;

    /// Long FFT binwidth in Hz.
    ///
    /// This takes into account the FFT bin factor, so the number of bins that
    /// divide the spectrum is [`NUM_FREQ_BINS`].
    pub const LONG_FFT_BINWIDTH: f64 = SAMPLE_RATE_F64 / NUM_FREQ_BINS as f64;

    /// Maximum carrier frequency offset to detect in Hz.
    pub const MAX_CARRIER_FREQ_OFFSET: f64 = 1.5e6;

    /// Maximum carrier frequency offset to detect in long FFT bins.
    pub const MAX_CARRIER_FREQ_OFFSET_BINS: usize =
        (MAX_CARRIER_FREQ_OFFSET / LONG_FFT_BINWIDTH).round() as usize;

    /// Carrier notching window in Hz for carrier detection.
    ///
    /// The carrier notching window is +/- this value around the carrier.
    pub const CARRIER_NOTCHING_WINDOW: f64 = 200e3;

    /// Carrier notching window in long FFT bins for carrier detection.
    ///
    /// The carrier notching window is +/- this value around the carrier.
    pub const CARRIER_NOTCHING_WINDOW_BINS: usize =
        (CARRIER_NOTCHING_WINDOW / LONG_FFT_BINWIDTH).round() as usize;

    /// Default carrier metric threshold used for carrier detection.
    pub const CARRIER_METRIC_THRESHOLD: f32 = 0.0148;

    /// Default preamble metric threshold used for preamble validation.
    pub const PREAMBLE_METRIC_THRESHOLD: f32 = 2.5e-4;

    /// Number of IQ samples in a Mode-S long message payload.
    pub const MODE_S_LONG_SAMPLES: usize =
        super::mode_s::MODE_S_LONG_BITS * SAMPLE_RATE_USIZE / 1_000_000;

    /// Left margin in samples used for packet cropping.
    pub const LEFT_MARGIN: usize = NFFT_SHORT + SPS * 2 * super::mode_s::REPLY_PREAMBLE_BITS;

    /// Right margin in samples used for packet cropping.
    pub const RIGHT_MARGIN: usize = NFFT_SHORT + SPS - 1;

    /// Number of samples in packet decoding window.
    pub const PACKET_WINDOW_SAMPLES: usize = LEFT_MARGIN + MODE_S_LONG_SAMPLES + RIGHT_MARGIN;

    /// Number of symbols in packet decoding window.
    pub const PACKET_WINDOW_SYMBOLS: usize = PACKET_WINDOW_SAMPLES / SPS;

    /// Number of decoded bytes in packet decoding window.
    ///
    /// Each pair of symbols contributes one bit, so the number of bytes is
    /// equal to the number of symbols divided by 16.
    pub const PACKET_WINDOW_BYTES: usize = PACKET_WINDOW_SYMBOLS / 16;

    /// Mask for preamble correlation.
    pub const PREAMBLE_MASK: [i8; 10] = [1, -1, 1, -1, 0, 0, -1, 1, -1, 1];

    /// Length of preamble correlation window.
    pub const PREAMBLE_MASK_LEN: usize = PREAMBLE_MASK.len();

    /// Length of Mode-S reply preamble in samples.
    pub const REPLY_PREAMBLE_SAMPLES: usize = 2 * super::mode_s::REPLY_PREAMBLE_BITS * SPS;

    /// Number of symbols in Mode-S long message payload.
    ///
    /// This is just twice the number of bits in the payload.
    pub const MODE_S_LONG_SYMBOLS: usize = 2 * super::mode_s::MODE_S_LONG_BITS;

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn freq_bin_periodicity() {
            assert_eq!(
                FREQ_BIN_PERIODICITY,
                num::integer::lcm(FFT_OVERLAP_FACTOR, FREQ_BIN_FACTOR)
            );
        }

        #[test]
        fn packet_window_samples_multiple() {
            assert!((PACKET_WINDOW_SAMPLES - (SPS - 1)).is_multiple_of(SPS));
        }

        #[test]
        fn packet_window_symbols_multiple() {
            assert!(PACKET_WINDOW_SYMBOLS.is_multiple_of(16));
        }
    }
}

/// Mode-S constants.
pub mod mode_s {
    /// Number of bits in a Mode-S long message payload.
    pub const MODE_S_LONG_BITS: usize = 112;

    /// Number of bytes in Mode-S long message payload.
    pub const MODE_S_LONG_BYTES: usize = MODE_S_LONG_BITS / 8;

    /// Number of bits in a Mode-S short message payload.
    pub const MODE_S_SHORT_BITS: usize = 56;

    /// Number of bytes in Mode-S short message payload.
    pub const MODE_S_SHORT_BYTES: usize = MODE_S_SHORT_BITS / 8;

    /// Number of bits in a CRC-24.
    pub const CRC24_BITS: usize = 24;

    /// Number of bytes in a CRC-24.
    pub const CRC24_BYTES: usize = CRC24_BITS / 8;

    /// Number of "bits" in a Mode-S reply preamble.
    ///
    /// The mode-S reply preamble lasts 8 usec. These can be thought of as 8 bits
    /// modulated in PPM following the sequence `11_00___`, where `_` denotes
    /// that no pulse is transmitted.
    pub const REPLY_PREAMBLE_BITS: usize = 8;

    /// Total number of "bits" in an Mode-S long reply.
    ///
    /// This includes both the message and the preamble.
    pub const TOTAL_MODE_S_LONG_REPLY_BITS: usize = MODE_S_LONG_BITS + REPLY_PREAMBLE_BITS;

    /// DF values that correspond to a Mode-S long reply.
    ///
    /// This array lists the DF fields that are currently assigned to Mode-S
    /// long replies. These are 16-21 and 24-31. DF field values 0, 4, 5, 11 are
    /// defined for short (56-bit) replies, so they are excluded from this
    /// list. The remaining DF values are reserved currently, so they are not
    /// present in the list either. If any of these reserved values becomes
    /// used, it will be assigned to either a short reply or a long reply. If it
    /// is assigned to a long reply, this list will be updated to add that DF
    /// value.
    pub const DF_LONG_REPLY: [u8; 14] = [16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31];

    /// DF value that corresponds to an acquisition squitter reply or all-call
    /// reply.
    pub const DF_ACQUISITION_SQUITTER: u8 = 11;

    /// DF values that correspond to a Mode-S extended squitter reply.
    ///
    /// Extended squitter replies are the only long replies that are guaranteed
    /// to use a CRC-24 that is not XORed with other additional information.
    ///
    /// In particular, DFs 0, 4, 5, 16 and 24-31 use address parity AP, so the
    /// CRC-24 is XORed with the 24-bit ICAO address. DFs 20, 21 use AP or data
    /// parity, which XORs with the ICAO address and potentially a field
    /// containing BDS data.
    ///
    /// DFs 11, 17, 18 use a parity/interrogator PI field, which is the XOR of
    /// the CRC-24 with a field the concatenation of 17 zero bits, the code
    /// label CL (3 bits), and the interrogator code IC (4 bits). Section
    /// 3.1.2.3.2.1.4 in the ICAO Annex 10 Vol IV specifies that "If the reply
    /// is made in response to a Mode A/C/S all-call, a Mode S-only all-call
    /// with CL field (3.1.2.5.2.1.3) and IC field (3.1.2.5.2.1.2) equal to 0,
    /// or is an acquisition or an extended squitter (3.1.2.8.5, 3.1.2.8.6 or
    /// 3.1.2.8.7), the II and the SI codes shall be 0."
    ///
    /// The interrogator identifier II (4 bits) and surveillance identifier SI
    /// (6 bit, value zero forbidden) identify Mode-S interrogators. The CL + IC
    /// field carries an II code in IC when CL is zero, or an SI code by putting
    /// the 4 LSBs in IC and specifying the 2 MSBs by the value of CL (1, 2, 3,
    /// 4). Therefore, this quote from the spec means that the CL + IC field is
    /// zero if the request was Mode A/C/S all-call (in which case there is no
    /// interrogator identifier), a Mode-S all-call with a zero interrogator
    /// identifier, an acqusition (DF 11 sent periodically and unsolicted with
    /// the purpose of announcing the ICAO address to sensors that could track
    /// the sender selectively) or extended squitter (DF 17 and 18).
    pub const DF_EXTENDED_SQUITTER: [u8; 2] = [17, 18];

    /// DF values that correspond to Mode-S short replies that use address
    /// parity.
    ///
    /// These message types are DF0 (short air-air surveillance ACAS), DF4
    /// (surveillance altitude reply) and DF5 (surveillance identity reply).
    pub const DF_SHORT_AP: [u8; 3] = [0, 4, 5];

    /// DF values that correspond to Mode-S long replies that use address
    /// parity.
    ///
    /// These message types are DF16 (long air-air surveillance ACAS), DF20
    /// (Comm-B altitude reply), DF21 (Comm-B identity reply), and DF24-31
    /// (Comm-D ELM).
    pub const DF_LONG_AP: [u8; 11] = [16, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31];

    /// DF value that corresponds to military extended squitter messages (DF19).
    pub const DF_MILITARY_EXTENDED_SQUITTER: u8 = 19;

    /// CA values that are valid.
    ///
    /// This lists the values of the CA (capability) field in DF11 and DF17
    /// messages that are not reserved. These are the only values that can occur
    /// in good messages.
    pub const CA_VALID: [u8; 5] = [0, 4, 5, 6, 7];

    /// DF18 CF values that are valid.
    ///
    /// This lists the values of the CF (control field) field in DF18 messages
    /// that are not reserved. These are the only values that can occur in good
    /// messages.
    pub const DF18_CF_VALID: [u8; 6] = [0, 2, 3, 4, 5, 6];
}

/// Constants used for simulation.
pub mod simulation {
    /// Standard deviation of real/imaginary parts of AWGN.
    ///
    /// This is set to achieve a standard deviation of 32 when the data is
    /// converted to complex int16. This gives minimal quantization losses, and
    /// allows an SNR of up to 60 dB.
    pub const AWGN_RE_SIGMA: f32 = 1.0 / (1 << 10) as f32;
}
