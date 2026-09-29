use super::{A, BitErrors, LongFFT};
use crate::{
    constants::{
        dsp::*,
        mode_s::{CA_VALID, DF_EXTENDED_SQUITTER, DF18_CF_VALID, MODE_S_LONG_BYTES},
    },
    math::{AlgebraicComplex, phase_wrap_once_f32},
    mode_s::CRC24,
};
use num::complex::Complex32;
use std::f64::consts::PI;

pub fn make_freq_bin_lo() -> Box<[[A<[f32; FREQ_BIN_LO_LEN]>; 2]; FREQ_BIN_FACTOR_SHORT]> {
    let mut los = Box::new([[aligned::Aligned([0.0; _]); _]; _]);
    for (n, [lo_re, lo_im]) in los.iter_mut().enumerate() {
        let freq_bin = n as f64 / FREQ_BIN_FACTOR_SHORT as f64;
        for (k, (lo_re, lo_im)) in lo_re.iter_mut().zip(lo_im.iter_mut()).enumerate() {
            let phi = -2.0 * PI * freq_bin * k as f64 / NFFT_SHORT as f64;
            let (im, re) = phi.sin_cos();
            *lo_re = re as f32;
            *lo_im = im as f32;
        }
    }
    los
}

pub fn accumulate_short_ffts(
    long_fft: &mut A<[f32; NUM_FREQ_BINS_KEEP]>,
    short_ffts: &[&A<[Complex32; NUM_FREQ_BINS_KEEP]>; FFT_OVERLAP_FACTOR],
) {
    for (j, x) in long_fft.iter_mut().enumerate() {
        let mut z = Complex32::new(0.0, 0.0);
        for short_fft in short_ffts {
            z = z.algebraic_add(short_fft[j]);
        }
        *x = z.algebraic_norm_sqr();
    }
}

pub fn long_fft_max_search(long_fft: &mut LongFFT) {
    let mut max_bin = 0;
    let mut max_value = 0.0;
    // Do max search only over center +/- MAX_CARRIER_FREQ_OFFSET
    const START: usize = NUM_FREQ_BINS_KEEP / 2 - MAX_CARRIER_FREQ_OFFSET_BINS;
    const END: usize = NUM_FREQ_BINS_KEEP / 2 + MAX_CARRIER_FREQ_OFFSET_BINS + 1;
    for (j, &x) in long_fft.fft_power[START..END].iter().enumerate() {
        if x > max_value {
            max_bin = j;
            max_value = x;
        }
    }
    long_fft.max_bin = u32::try_from(max_bin + START).unwrap();
    long_fft.max_value = max_value;
}

pub fn carrier_notched_power(long_fft: &LongFFT) -> f32 {
    let k = usize::try_from(long_fft.max_bin).unwrap();
    let notch_start = k.saturating_sub(CARRIER_NOTCHING_WINDOW_BINS);
    let notch_end = (k + CARRIER_NOTCHING_WINDOW_BINS + 1).min(NUM_FREQ_BINS_KEEP);
    let mut power: f32 = 0.0;
    // Use algebraic_add to let the compiler reorder the sums and vectorize
    // this, since otherwise the sums need to be performed in order because IEEE
    // sum does not commute exactly, and the compiler can only do scalar sum in
    // order.
    for &x in long_fft.fft_power[..notch_start].iter() {
        power = power.algebraic_add(x);
    }
    for &x in long_fft.fft_power[notch_end..].iter() {
        power = power.algebraic_add(x);
    }
    power
}

pub fn parabolic_interpolation(x: [f32; 3]) -> f32 {
    let d = x[2] + x[0] - 2.0 * x[1];
    // under the assumption that x[1] >= x[0].max(x[2]), d == 0 means that the 3
    // numbers are equal, so we should return zero
    if d == 0.0 {
        return 0.0;
    }
    -0.5 * (x[2] - x[0]) / d
}

pub fn frequency_shift(x: &mut [Complex32; PACKET_WINDOW_SAMPLES], freq: f64) {
    let phi_incr = (2.0 * PI * freq / SAMPLE_RATE_F64) as f32;
    let mut phi: f32 = 0.0;
    for z in x.iter_mut() {
        let (sin, cos) = phi.sin_cos();
        let lo = Complex32::new(cos, sin);
        *z = z.algebraic_mul(lo);
        phi += phi_incr;
        phi = phase_wrap_once_f32(phi);
    }
}

pub fn estimate_carrier_phasor(x: &[Complex32; MODE_S_LONG_SAMPLES]) -> Complex32 {
    let mut z = Complex32::new(0.0, 0.0);
    for &w in x.iter() {
        z = z.algebraic_add(w);
    }
    // Division by |z| causes the return value to be NaN if z == 0, which can
    // happen for instance if the samples in x are all zeros. This is not
    // concerning. As the samples get rotated in phasor_shift(), they become all
    // NaNs. These NaNs are decoded as an all-zeros message, because NaN > NaN
    // is false. The all-zeros message has correct CRC-24 but it is not a valid
    // extended squitter message, so the decode is discarded. Note that an
    // all-zeros input to the ADSBDecoder results in many detections, because
    // the detection threshold check has a >= that is trivially satisfied as 0
    // >= threshold * 0 when the input is all zeros.
    z.algebraic_div_scalar(z.algebraic_norm_sqr().sqrt())
}

pub fn phasor_shift(x: &mut [Complex32; PACKET_WINDOW_SAMPLES], phasor: Complex32) {
    for z in x.iter_mut() {
        *z = z.algebraic_mul(phasor);
    }
}

pub fn accumulate_symbols(
    samples: &[Complex32; PACKET_WINDOW_SAMPLES],
    symbols: &mut [[Complex32; PACKET_WINDOW_SYMBOLS]; SPS],
) {
    for n in 0..PACKET_WINDOW_SYMBOLS {
        for k in 0..SPS {
            let w: &[Complex32; SPS] = samples[n * SPS + k..(n + 1) * SPS + k].try_into().unwrap();
            let mut symbol = Complex32::new(0.0, 0.0);
            for &z in w.iter() {
                symbol = symbol.algebraic_add(z);
            }
            symbols[k][n] = symbol;
        }
    }
}

pub fn demodulate(
    symbols: &[Complex32; PACKET_WINDOW_SYMBOLS],
    coherent: &mut [[u8; PACKET_WINDOW_BYTES]; 2],
    noncoherent: &mut [[u8; PACKET_WINDOW_BYTES]; 2],
) {
    let coherent_slicer = |z0: Complex32, z1: Complex32| -> u8 { (z0.re > z1.re).into() };
    let noncoherent_slicer = |z0: Complex32, z1: Complex32| -> u8 {
        (z0.algebraic_norm_sqr() > z1.algebraic_norm_sqr()).into()
    };
    for n in 0..PACKET_WINDOW_BYTES {
        let mut coh0 = 0;
        let mut coh1 = 0;
        let mut noncoh0 = 0;
        let mut noncoh1 = 0;
        for k in 0..8 {
            let z0 = symbols[16 * n + 2 * k];
            let z1 = symbols[16 * n + 2 * k + 1];
            coh0 = (coh0 << 1) | coherent_slicer(z0, z1);
            noncoh0 = (noncoh0 << 1) | noncoherent_slicer(z0, z1);
            if n == PACKET_WINDOW_BYTES - 1 && k == 7 {
                // the last bit for the second pairing goes out of bounds, so we
                // hardcode it to zero
                coh1 <<= 1;
                noncoh1 <<= 1;
            } else {
                let z2 = symbols[16 * n + 2 * k + 2];
                coh1 = (coh1 << 1) | coherent_slicer(z1, z2);
                noncoh1 = (noncoh1 << 1) | noncoherent_slicer(z1, z2);
            };
        }
        coherent[0][n] = coh0;
        coherent[1][n] = coh1;
        noncoherent[0][n] = noncoh0;
        noncoherent[1][n] = noncoh1;
    }
}

pub fn compute_crc(bytes: &[u8; PACKET_WINDOW_BYTES], bit_offset: usize) -> u32 {
    let mut digest = CRC24.digest();
    let byte_offset = bit_offset / 8;
    let bit_offset = bit_offset - 8 * byte_offset;
    if bit_offset == 0 {
        digest.update(&bytes[byte_offset..byte_offset + MODE_S_LONG_BYTES]);
    } else {
        for k in byte_offset..byte_offset + MODE_S_LONG_BYTES {
            let b = (bytes[k] << bit_offset) | (bytes[k + 1] >> (8 - bit_offset));
            digest.update(&[b]);
        }
    }
    digest.finalize()
}

// Checks that the DF, taking the bit errors into account, matches a value
// assigned to extended squitter replies. Additionally, it also checks that for
// DF17, the value of the CA field is in CA_VALID and for DF18 the
// value of the CF field is in DF18_CF_VALID.
pub fn df_is_valid_extended_squitter(
    bytes: &[u8; PACKET_WINDOW_BYTES],
    bit_offset: usize,
    bit_errors: BitErrors,
) -> bool {
    // extract DF from bytes
    let byte_offset = bit_offset / 8;
    let bit_offset = bit_offset - 8 * byte_offset;
    let mut first_byte = (bytes[byte_offset] << bit_offset)
        | bytes[byte_offset + 1].unbounded_shr(u32::try_from(8 - bit_offset).unwrap());

    // correct bit errors that land on the first byte
    macro_rules! correct_bit {
        ($j:expr) => {{
            let j = $j;
            if j < 8 {
                first_byte ^= 1 << (7 - j);
            }
        }};
    }

    match bit_errors {
        BitErrors::NoErrors => {}
        BitErrors::SingleError(j) => correct_bit!(j),
        BitErrors::DoubleError(j, k) => {
            correct_bit!(j);
            correct_bit!(k);
        }
    }
    let df = first_byte >> 3;
    if !DF_EXTENDED_SQUITTER.contains(&df) {
        return false;
    }
    // bits 5-7 are the CA field for DF17 and the CF field for DF18.
    let ca_cf = first_byte & 0x7;
    match df {
        17 => CA_VALID.contains(&ca_cf),
        18 => DF18_CF_VALID.contains(&ca_cf),
        _ => unreachable!(),
    }
}

pub fn extract_message(
    bytes: &[u8; PACKET_WINDOW_BYTES],
    bit_offset: usize,
) -> [u8; MODE_S_LONG_BYTES] {
    let mut message = [0; MODE_S_LONG_BYTES];
    let byte_offset = bit_offset / 8;
    let bit_offset = bit_offset - 8 * byte_offset;
    if bit_offset == 0 {
        message.copy_from_slice(&bytes[byte_offset..byte_offset + MODE_S_LONG_BYTES])
    } else {
        for (k, message_byte) in message.iter_mut().enumerate() {
            let n = k + byte_offset;
            *message_byte = (bytes[n] << bit_offset) | (bytes[n + 1] >> (8 - bit_offset));
        }
    }
    message
}

pub fn correct_message_bit_errors(
    message: &mut [u8; MODE_S_LONG_BYTES],
    corrected_errors: BitErrors,
) {
    match corrected_errors {
        BitErrors::NoErrors => {}
        BitErrors::SingleError(j) => {
            let j = usize::from(j);
            message[j / 8] ^= 1 << (7 - (j % 8));
        }
        BitErrors::DoubleError(j, k) => {
            let j = usize::from(j);
            let k = usize::from(k);
            message[j / 8] ^= 1 << (7 - (j % 8));
            message[k / 8] ^= 1 << (7 - (k % 8));
        }
    }
}

pub fn preamble_corr(symbols: &[Complex32; PREAMBLE_MASK_LEN]) -> f32 {
    let mut z = Complex32::new(0.0, 0.0);
    for (&w, s) in symbols.iter().zip(PREAMBLE_MASK) {
        match s {
            1 => z = z.algebraic_add(w),
            -1 => z = z.algebraic_sub(w),
            0 => {}
            _ => unreachable!(),
        }
    }
    z.algebraic_norm_sqr()
}

fn extract_message_bit(message: &[u8; MODE_S_LONG_BYTES], bit_index: usize) -> u8 {
    (message[bit_index / 8] >> (7 - (bit_index % 8))) & 1
}

pub fn postdecode_power(
    symbols: &[Complex32; MODE_S_LONG_SYMBOLS],
    message: &[u8; MODE_S_LONG_BYTES],
) -> f32 {
    let mut power: f32 = 0.0;
    for (j, pair) in symbols.as_chunks::<2>().0.iter().enumerate() {
        let bit = extract_message_bit(message, j);
        let pulse = pair[usize::from(1 - bit)];
        power = power.algebraic_add(pulse.algebraic_norm_sqr());
    }
    power
}

// This kernel computes noise power in the following way. For each message bit,
// the 2 central samples in the pulse position where the message does not
// transmit a pulse are taken. The other 2 samples in this pulse position are
// ignored because they can be contaminated with the power from adjacent
// pulses. The 4 taken samples from each pair of bits are integrated coherently
// as a crude lowpass filter that is similar to the Mode-S PSD. The main reason
// to do this filtering is to have some rejection to DME/TACAN stations in
// adjacent channels, which are in-band at 8 Msps but have reduced impact on
// Mode-S decoding. All these 4-sample coherent integrations are integrated
// non-coherently.
pub fn postdecode_noise_power(
    samples: &[Complex32; MODE_S_LONG_SAMPLES],
    message: &[u8; MODE_S_LONG_BYTES],
) -> f32 {
    let mut power: f32 = 0.0;
    for (j, z) in samples.as_chunks::<{ 4 * SPS }>().0.iter().enumerate() {
        let b0 = usize::from(extract_message_bit(message, 2 * j));
        let b1 = usize::from(extract_message_bit(message, 2 * j + 1));
        let start0 = SPS * b0 + 1;
        let start1 = 2 * SPS + SPS * b1 + 1;
        let gap0 = &z[start0..start0 + 2];
        let gap1 = &z[start1..start1 + 2];
        let w = gap0[0]
            .algebraic_add(gap0[1])
            .algebraic_add(gap1[0])
            .algebraic_add(gap1[1]);
        power = power.algebraic_add(w.algebraic_norm_sqr());
    }
    power
}

#[cfg(test)]
mod test {
    use super::{
        super::test::{any_message, copy_message_to_bytes},
        *,
    };
    use proptest::num::f32::{NEGATIVE, POSITIVE, SUBNORMAL, ZERO};
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn carrier_notched_power(
            fft_power in proptest::collection::vec(0.0_f32..1.0, NUM_FREQ_BINS_KEEP),
            max_bin in NUM_FREQ_BINS_KEEP / 2 - MAX_CARRIER_FREQ_OFFSET_BINS..=
                NUM_FREQ_BINS_KEEP / 2 + MAX_CARRIER_FREQ_OFFSET_BINS,
        ) {
            let long_fft = LongFFT {
                fft_power: aligned::Aligned(fft_power.try_into().unwrap()),
                // the kernel does not need max_bin to be consistent with the values of fft_power
                max_bin: u32::try_from(max_bin).unwrap(),
                // this is not used by the kernel
                max_value: 0.0,
            };
            let mut expected = 0.0;
            for (k, f) in long_fft.fft_power.iter().enumerate() {
                if (isize::try_from(k).unwrap() - isize::try_from(max_bin).unwrap()).abs()
                    > isize::try_from(CARRIER_NOTCHING_WINDOW_BINS).unwrap() {
                    expected += f;
                }
            }
            let tolerance = 3e-3;
            let result = super::carrier_notched_power(&long_fft);
            assert!((result - expected).abs() <= tolerance,
                    "result = {result}, expected = {expected}");
        }
    }

    prop_compose! {
        fn peak_in_middle()(mut x in [POSITIVE | NEGATIVE | ZERO | SUBNORMAL; 3]) -> [f32; 3] {
            // reorder elements of x if needed to make x[1] be the largest
            let mut j = 0;
            if x[1] > x[0] {
                j = 1;
            }
            if x[2] > x[j] {
                j = 2;
            }
            x.swap(1, j);
            assert!((x[1] >= x[0]) && (x[1] >= x[2]));
            x
        }
    }

    proptest! {
        /// Checks that parabolic interpolation always gives a value in (-0.501, 0.501)
        /// when x[1] is a local maximum.
        ///
        /// Note: the interval cannot be tight ([-0.5, 0.5]) due to numerical
        /// precision errors.
        #[test]
        fn parabolic_interpolation_range(x in peak_in_middle()) {
            let y = parabolic_interpolation(x);
            assert!((0.0..0.501).contains(&y.abs()),
                    "parabolic interpolation {y} not contained in open interval (-0.501, 0.501)");
        }
    }

    proptest! {
        fn frequency_shift_cw(freq in -1e6_f64..=1e6) {
            let mut samples = (0..PACKET_WINDOW_SAMPLES).map(|n| {
                let phase: f64 = 2.0 * PI * n as f64 * freq / SAMPLE_RATE_F64;
                Complex32::new(phase.cos() as f32, phase.sin() as f32)
            }).collect::<Vec<_>>();
            frequency_shift((&mut samples[..]).try_into().unwrap(), -freq);
            let expected = [Complex32::new(1.0, 0.0); PACKET_WINDOW_SAMPLES];
            assert_all_close!(samples, expected, 0.0, 1e-5);
        }
    }

    proptest! {
        #[test]
        fn estimate_carrier_phasor_cw(phase in -PI..PI) {
            let z = Complex32::new(phase.cos() as f32, phase.sin() as f32);
            let samples = [z; MODE_S_LONG_SAMPLES];
            let estimate = estimate_carrier_phasor(&samples);
            let tolerance = 1e-4;
            assert!((estimate - z).norm() <= tolerance,
                    "estimate = {estimate}, z = {z}");
        }
    }

    proptest! {
        #[test]
        fn phasor_shift_cw(phase in -PI..PI) {
            let phasor = Complex32::new(phase.cos() as f32, phase.sin() as f32);
            let mut samples = [phasor; PACKET_WINDOW_SAMPLES];
            phasor_shift(&mut samples, phasor.conj());
            let expected = [Complex32::new(1.0, 0.0); PACKET_WINDOW_SAMPLES];
            assert_all_close!(samples, expected, 0.0, 1e-5);
        }
    }

    prop_compose! {
        fn complex_2x2()(re in -2.0_f32..2.0, im in -2.0_f32..2.0) -> Complex32 {
            Complex32::new(re, im)
        }
    }

    proptest! {
        #[test]
        fn accumulate_symbols(
            samples in proptest::collection::vec(complex_2x2(), PACKET_WINDOW_SAMPLES)
        ) {
            let mut symbols = [[Complex32::new(0.0, 0.0); PACKET_WINDOW_SYMBOLS]; SPS];
            super::accumulate_symbols(samples[..].try_into().unwrap(), &mut symbols);
            let mut expected = [[Complex32::new(0.0, 0.0); PACKET_WINDOW_SYMBOLS]; SPS];
            for clock_phase in 0..SPS {
                for (n, w) in samples[clock_phase..]
                    .as_chunks::<SPS>()
                    .0
                    .iter()
                    .enumerate()
                {
                    expected[clock_phase][n] = w.iter().sum();
                }
            }
            for (syms, expect) in symbols.iter().zip(expected.iter()) {
                assert_all_close!(syms, expect, 1e-6, 1e-4);
            }
        }
    }

    proptest! {
        #[test]
        fn demodulate(
            symbols in proptest::collection::vec(complex_2x2(), PACKET_WINDOW_SYMBOLS)
        ) {
            let mut coherent = [[0; PACKET_WINDOW_BYTES]; 2];
            let mut noncoherent = [[0; PACKET_WINDOW_BYTES]; 2];
            super::demodulate(
                symbols[..].try_into().unwrap(),
                &mut coherent,
                &mut noncoherent,
            );
            let mut expected_coherent = [[0; PACKET_WINDOW_BYTES]; 2];
            let mut expected_noncoherent = [[0; PACKET_WINDOW_BYTES]; 2];
            for skip in 0..2 {
                for n in 0..PACKET_WINDOW_BYTES {
                    for k in 0..8 {
                        let (coh, ncoh): (u8, u8) =
                            if skip == 1 && n == PACKET_WINDOW_BYTES - 1 && k == 7 {
                                // avoid out-of-bound access; assign 0, 0 to the demodulation
                                (0, 0)
                            } else {
                                let z0 = symbols[2 * (8 * n + k) + skip];
                                let z1 = symbols[2 * (8 * n + k) + skip + 1];
                                let coh = (z0.re > z1.re).into();
                                let ncoh = (z0.norm_sqr() > z1.norm_sqr()).into();
                                (coh, ncoh)
                            };
                        expected_coherent[skip][n] = (expected_coherent[skip][n] << 1) | coh;
                        expected_noncoherent[skip][n] = (expected_noncoherent[skip][n] << 1) | ncoh;
                    }
                }
            }
            assert_eq!(coherent, expected_coherent);
            assert_eq!(noncoherent, expected_noncoherent);
        }
    }

    prop_compose! {
        fn any_bytes()(
            bytes in proptest::collection::vec(any::<u8>(), PACKET_WINDOW_BYTES)
        ) -> [u8; PACKET_WINDOW_BYTES] {
            bytes.try_into().unwrap()
        }
    }

    proptest! {
        #[test]
        fn compute_crc(
            bit_offset in 0..8*(PACKET_WINDOW_BYTES-MODE_S_LONG_BYTES),
            mut bytes in any_bytes(),
            message in any_message(),
        ) {
            copy_message_to_bytes(&message, &mut bytes, bit_offset);
            let crc = super::compute_crc(&bytes, bit_offset);
            assert_eq!(crc, CRC24.checksum(&message));
        }
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    enum MessageHeader {
        DF17(CA),
        DF18(CF),
        OtherDF(u8),
    }

    impl MessageHeader {
        fn header_byte(&self) -> u8 {
            match self {
                MessageHeader::DF17(ca) => (17 << 3) | ca.ca(),
                MessageHeader::DF18(cf) => (18 << 3) | cf.cf(),
                MessageHeader::OtherDF(byte) => *byte,
            }
        }

        fn is_df_ca_cf_valid(&self) -> bool {
            match self {
                MessageHeader::DF17(CA::Valid(_)) => true,
                MessageHeader::DF17(CA::Invalid(_)) => false,
                MessageHeader::DF18(CF::Valid(_)) => true,
                MessageHeader::DF18(CF::Invalid(_)) => false,
                MessageHeader::OtherDF(_) => false,
            }
        }
    }

    impl Arbitrary for MessageHeader {
        type Parameters = ();
        type Strategy = BoxedStrategy<MessageHeader>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            prop_oneof![
                any::<CA>().prop_map(MessageHeader::DF17),
                any::<CF>().prop_map(MessageHeader::DF18),
                any::<u8>()
                    .prop_filter("other DF must not be DF17 or DF18", |byte| ![17, 18]
                        .contains(&(byte >> 3)))
                    .prop_map(MessageHeader::OtherDF)
            ]
            .boxed()
        }
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    enum CA {
        Valid(u8),
        Invalid(u8),
    }

    impl CA {
        fn ca(&self) -> u8 {
            match self {
                CA::Valid(ca) => *ca,
                CA::Invalid(ca) => *ca,
            }
        }
    }

    impl Arbitrary for CA {
        type Parameters = ();
        type Strategy = BoxedStrategy<CA>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (0u8..8)
                .prop_map(|ca| {
                    if CA_VALID.contains(&ca) {
                        CA::Valid(ca)
                    } else {
                        CA::Invalid(ca)
                    }
                })
                .boxed()
        }
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
    enum CF {
        Valid(u8),
        Invalid(u8),
    }

    impl CF {
        fn cf(&self) -> u8 {
            match self {
                CF::Valid(cf) => *cf,
                CF::Invalid(cf) => *cf,
            }
        }
    }

    impl Arbitrary for CF {
        type Parameters = ();
        type Strategy = BoxedStrategy<CF>;

        fn arbitrary_with(_: ()) -> Self::Strategy {
            (0u8..8)
                .prop_map(|ca| {
                    if DF18_CF_VALID.contains(&ca) {
                        CF::Valid(ca)
                    } else {
                        CF::Invalid(ca)
                    }
                })
                .boxed()
        }
    }

    proptest! {
        #[test]
        fn df_is_valid_extended_squitter(
            header: MessageHeader,
            bit_offset in 0..8*(PACKET_WINDOW_BYTES-MODE_S_LONG_BYTES),
            bit_errors: BitErrors,
            mut bytes in any_bytes(),
            mut message in any_message(),
        ) {
            message[0] = header.header_byte();
            super::correct_message_bit_errors(&mut message, bit_errors);
            copy_message_to_bytes(&message, &mut bytes, bit_offset);

            assert_eq!(
                super::df_is_valid_extended_squitter(&bytes, bit_offset, bit_errors),
                header.is_df_ca_cf_valid());
        }
    }

    proptest! {
        #[test]
        fn extract_message(
            bit_offset in 0..8*(PACKET_WINDOW_BYTES-MODE_S_LONG_BYTES),
            mut bytes in any_bytes(),
            message in any_message(),
        ) {
            copy_message_to_bytes(&message, &mut bytes, bit_offset);
            assert_eq!(
                super::extract_message(&bytes, bit_offset),
                message,
            );
        }
    }

    fn flip_bit(message: &mut [u8; MODE_S_LONG_BYTES], bit_pos: usize) {
        message[bit_pos / 8] ^= 1 << (7 - (bit_pos % 8));
    }

    proptest! {
        #[test]
        fn correct_message_bit_errors(
            bit_errors: BitErrors,
            message in any_message(),
        ) {
            let mut scratch = message;
            match bit_errors {
                BitErrors::NoErrors => {},
                BitErrors::SingleError(j) => flip_bit(&mut scratch, usize::from(j)),
                BitErrors::DoubleError(j, k) => {
                    assert_ne!(j , k);
                    flip_bit(&mut scratch, usize::from(j));
                    flip_bit(&mut scratch, usize::from(k));
                }
            }
            super::correct_message_bit_errors(&mut scratch, bit_errors);
            assert_eq!(scratch, message);
        }
    }

    #[test]
    fn preamble_corr_of_preamble() {
        let one = Complex32::new(1.0, 0.0);
        let zero = Complex32::new(0.0, 0.0);
        let preamble = [one, zero, one, zero, zero, zero, zero, one, zero, one];
        let corr = preamble_corr(&preamble);
        let expected = 16.0;
        assert_eq!(corr, expected);
    }

    proptest! {
        #[test]
        fn postdecode_power(
            symbols in proptest::collection::vec(complex_2x2(), MODE_S_LONG_SYMBOLS),
            message in any_message(),
        ) {
            let expected = symbols.as_chunks::<2>()
                .0
                .iter()
                .enumerate()
                .map(|(n, [z0, z1])| {
                    let bit = (message[n / 8] >> (7 - (n % 8))) & 1;
                    let z = if bit == 1 {
                        z0
                    } else {
                        z1
                    };
                    z.norm_sqr()
                })
                .sum::<f32>();
            let power = super::postdecode_power(&symbols[..].try_into().unwrap(), &message);
            assert!((power - expected).abs() <= 1e-5 * expected);
        }
    }

    proptest! {
        #[test]
        fn postdecode_power_clean(
            message in any_message(),
        ) {
            let mut symbols = [Complex32::new(0.0, 0.0); MODE_S_LONG_SYMBOLS];
            for (n, [z0, z1]) in symbols.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                let bit = (message[n / 8] >> (7 - (n % 8))) & 1;
                let one = Complex32::new(1.0, 0.0);
                if bit == 1 {
                    *z0 = one;
                } else {
                    *z1 = one;
                }
            }
            let power = super::postdecode_power(&symbols, &message);
            assert_eq!(power, crate::constants::mode_s::MODE_S_LONG_BITS as f32);
        }
    }

    proptest! {
        #[test]
        fn postdecode_noise_power(
            samples in proptest::collection::vec(complex_2x2(), MODE_S_LONG_SAMPLES),
            message in any_message(),
        ) {
            let mut noise_samples = samples.as_chunks::<{ 2 * SPS }>()
                .0
                .iter()
                .enumerate()
                .map(|(n, zs)| {
                    let [z0, z1] = zs.as_chunks::<SPS>().0 else {
                        panic!()
                    };
                    let bit = (message[n / 8] >> (7 - (n % 8))) & 1;
                    let z = if bit == 1 {
                        z1
                    } else {
                        z0
                    };
                    z[1] + z[2]
                });
            let mut expected = 0.0;
            while let Some(z) = noise_samples.next() {
                expected += (z + noise_samples.next().unwrap()).norm_sqr()
            }
            let power = super::postdecode_noise_power(&samples[..].try_into().unwrap(), &message);
            assert!((power - expected).abs() <= 1e-5 * expected);
        }
    }

    proptest! {
        #[test]
        fn postdecode_noise_power_clean(
            message in any_message(),
        ) {
            let mut symbols = [Complex32::new(0.0, 0.0); MODE_S_LONG_SAMPLES];
            for (n, zs) in symbols.as_chunks_mut::<{ 2 * SPS }>().0.iter_mut().enumerate() {
                let [z0, z1] = zs.as_chunks_mut::<SPS>().0 else {
                    panic!()
                };
                let bit = (message[n / 8] >> (7 - (n % 8))) & 1;
                let z = if bit == 1 {
                    z0
                } else {
                    z1
                };
                z.fill(Complex32::new(1.0, 0.0));
            }
            let power = super::postdecode_noise_power(&symbols, &message);
            assert_eq!(power, 0.0);
        }
    }
}
