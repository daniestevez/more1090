//! Simulation of ADS-B signals.
use crate::{
    constants::{
        dsp::{SAMPLE_RATE_F64, SAMPLE_RATE_USIZE},
        mode_s::{
            CA_VALID, CRC24_BYTES, DF_ACQUISITION_SQUITTER, DF_EXTENDED_SQUITTER, DF_LONG_AP,
            DF_MILITARY_EXTENDED_SQUITTER, DF_SHORT_AP, DF18_CF_VALID, MODE_S_LONG_BITS,
            MODE_S_LONG_BYTES, MODE_S_SHORT_BYTES, REPLY_PREAMBLE_BITS,
        },
        simulation::AWGN_RE_SIGMA,
    },
    io::WriteComplex32,
    math,
    mode_s::{CRC24, ModeSMessage},
};
use anyhow::Result;
use num::complex::Complex32;
use rand::{Rng, RngExt, distr::weighted::WeightedIndex, seq::IndexedRandom};
use rand_distr::{Distribution, StandardNormal};
use std::{
    f64::consts::{PI, SQRT_2},
    time::Duration,
};

pub mod distributions;
pub mod pulse_shape;
pub mod rng;

use distributions::MessageDistributions;

/// Configuration for a single-message simulation.
///
/// This struct defines the configuration parameters for a single-message
/// simulation that generates an IQ snippet containing a single ADS-B message.
#[derive(Debug, Clone, PartialEq)]
pub struct SingleMessageSimulation {
    /// Duration of the simulation.
    ///
    /// This defines the length of the IQ data.
    pub duration: Duration,
    /// Configuration for the message in the simulation.
    pub message_config: MessageConfig,
}

/// Configuration for a full simulation.
///
/// This struct defines the configuration parameters for a full simulation that
/// generates IQ data containing multiple Mode-S messages of the types listed by
/// the [`MessageType`] enum. Each of these message types appears following a
/// Poisson process with a given number of messages per second. These Poisson
/// processes are independent.
#[derive(Debug, Clone, PartialEq)]
pub struct Simulation {
    /// Duration of the simulation.
    ///
    /// This defines the length of the IQ data.
    pub duration: Duration,
    /// Messages per second for each message type.
    pub messages_per_second: MessagesPerSecond,
    /// Message parameters distributions.
    ///
    /// This defines the random distributions used to generate the message
    /// parameters.
    pub message_distributions: MessageDistributions,
}

/// Number of messages per second for each message type.
///
/// This defines the average number of messages per second for each message type
/// according to its Poisson process.
#[derive(Debug, Clone, PartialEq)]
pub struct MessagesPerSecond {
    /// Average number of extended squitter (DF17/18) messages per second.
    pub extended_squitter: f64,
    /// Average number of acquisition squitter (DF11) messages per second.
    pub acquisition_squitter: f64,
    /// Average number of short messages using address parity (DF0/4/5) per
    /// second.
    pub short_ap: f64,
    /// Average number of all-call reply messages per second.
    pub all_call_reply: f64,
    /// Average number of long messages using address parity (DF16/20/21/24-31)
    /// per second.
    pub long_ap: f64,
    /// Average number of military extended squitter (DF19) messages per second.
    pub military_extended_squitter: f64,
}

impl MessagesPerSecond {
    /// Creates a messages per second instance according to a total number of
    /// messages per second and the typical probabilities of each message type.
    pub fn from_total_messages_per_second(messages_per_second: f64) -> Result<MessagesPerSecond> {
        anyhow::ensure!(
            messages_per_second > 0.0 && messages_per_second.is_finite(),
            "messages per second must be positive and finite"
        );
        Ok(MessagesPerSecond {
            extended_squitter: messages_per_second
                * MessageType::ExtendedSquitter.typical_probability(),
            acquisition_squitter: messages_per_second
                * MessageType::AcquisitionSquitter.typical_probability(),
            short_ap: messages_per_second * MessageType::ShortAP.typical_probability(),
            all_call_reply: messages_per_second * MessageType::AllCallReply.typical_probability(),
            long_ap: messages_per_second * MessageType::LongAP.typical_probability(),
            military_extended_squitter: messages_per_second
                * MessageType::MilitaryExtendedSquitter.typical_probability(),
        })
    }

    /// Returns the total number of messages per second.
    pub fn total_messages_per_second(&self) -> f64 {
        self.extended_squitter
            + self.acquisition_squitter
            + self.short_ap
            + self.all_call_reply
            + self.long_ap
            + self.military_extended_squitter
    }
}

/// Mode-S message type.
///
/// This enum lists the different message types that the simulator can generate.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum MessageType {
    /// Extended squitter (DF17/18) messages, generated with
    /// [`random_extended_squitter`].
    ExtendedSquitter,
    /// Acquisition squitter (DF11) messages, generated with
    /// [`random_acquisition_squitter`].
    AcquisitionSquitter,
    /// Short messages using address parity (DF0/4/5), generated with
    /// [`random_short_ap`].
    ShortAP,
    /// All-call reply (DF11) messages using parity/interrogator identifier,
    /// generated with [`random_all_call_reply`].
    AllCallReply,
    /// Long messages using address parity (DF16/20/21/24-31), generated with
    /// [`random_long_ap`].
    LongAP,
    /// Military extended squitter (DF19), generated with
    /// [`random_military_extended_squitter`].
    MilitaryExtendedSquitter,
}

impl MessageType {
    /// Generates a random message of the given type.
    pub fn random<R: Rng + ?Sized>(&self, rng: &mut R) -> ModeSMessage {
        match self {
            MessageType::ExtendedSquitter => random_extended_squitter(rng),
            MessageType::AcquisitionSquitter => random_acquisition_squitter(rng),
            MessageType::ShortAP => random_short_ap(rng),
            MessageType::AllCallReply => random_all_call_reply(rng),
            MessageType::LongAP => random_long_ap(rng),
            MessageType::MilitaryExtendedSquitter => random_military_extended_squitter(rng),
        }
    }

    /// Array listing all the message types.
    pub const ALL: [MessageType; 6] = [
        MessageType::ExtendedSquitter,
        MessageType::AcquisitionSquitter,
        MessageType::ShortAP,
        MessageType::AllCallReply,
        MessageType::LongAP,
        MessageType::MilitaryExtendedSquitter,
    ];

    // Typical relative weights of these message types.
    //
    // This is inspired by the OpenSky report 2016
    // https://s3.opensky-network.org/website-public-files/publications/dasc2016.pdf
    // Table II, which contains data mostly for Europe covering 2013-2016.
    //
    // Deviations from this report include the fact that ADS-B usage has grown
    // since and the fact that DF20/21 are more common in Europe, while DF18 is
    // more common in the US
    const TYPICAL_WEIGHTS: [f64; 6] = [
        0.50,  // extended squitter
        0.19,  // acquisition squitter
        0.12,  // short AP
        0.065, // all-call reply
        0.12,  // long AP
        0.005, // military extended squitter
    ];

    /// Returns the typical probability tha a message is of this type.
    pub fn typical_probability(&self) -> f64 {
        match self {
            MessageType::ExtendedSquitter => Self::TYPICAL_WEIGHTS[0],
            MessageType::AcquisitionSquitter => Self::TYPICAL_WEIGHTS[1],
            MessageType::ShortAP => Self::TYPICAL_WEIGHTS[2],
            MessageType::AllCallReply => Self::TYPICAL_WEIGHTS[3],
            MessageType::LongAP => Self::TYPICAL_WEIGHTS[4],
            MessageType::MilitaryExtendedSquitter => Self::TYPICAL_WEIGHTS[5],
        }
    }
}

/// Configuration for a single ADS-B message.
///
/// This struct defines the configuration of a single message to be inserted
/// into the simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageConfig {
    /// Timestamp of the message.
    ///
    /// This is indicated in samples since the beginning of the simulation. The
    /// reference for the timestamp is the 50% crossing of the rising edge of
    /// the first pulse.
    pub timestamp_samples: f64,
    /// CN0 of the message in dB·Hz.
    ///
    /// The CN0 is referred to the peak power of an ideal rectangular pulse. It
    /// does not take into account the 50% duty cycle of ADS-B or the pulse
    /// shaping that is used in the simulation.
    pub cn0_db: f64,
    /// Carrier frequency offset of the message in Hz.
    pub carrier_frequency_offset: f64,
    /// Initial carrier phase of the message in radians.
    pub carrier_phase: f64,
    /// Symbol clock error of the message in parts per one.
    pub symbol_clock_error: f64,
    /// Mode-S message.
    ///
    /// This contains the message that is modulated in the Mode-S reply payload.
    pub message: ModeSMessage,
}

impl MessageConfig {
    /// Generate a message configuration from distributions for its parameters.
    ///
    /// The Mode-S message is generated randomly with [`MessageType::random`].
    pub fn from_distributions<R: Rng + ?Sized>(
        timestamp_samples: f64,
        distributions: MessageDistributions,
        message_type: MessageType,
        rng: &mut R,
    ) -> MessageConfig {
        MessageConfig {
            timestamp_samples,
            cn0_db: distributions.cn0.sample(rng),
            carrier_frequency_offset: distributions.carrier_frequency_offset.sample(rng),
            carrier_phase: distributions.carrier_phase.sample(rng),
            symbol_clock_error: distributions.symbol_clock_error.sample(rng),
            message: message_type.random(rng),
        }
    }

    /// First IQ sample index occupied by the message as a floating point number.
    fn first_sample_index_f64(&self) -> f64 {
        self.timestamp_samples - pulse_shape::group_delay() * (1.0 - self.symbol_clock_error)
    }

    /// First IQ sample index occupied by the message.
    fn first_sample_index(&self) -> i64 {
        self.first_sample_index_f64().ceil() as i64
    }

    /// Clock phase to be supplied to the filterbank.
    fn clock_phase(&self) -> f64 {
        let phase = self.first_sample_index() as f64 - self.first_sample_index_f64();
        debug_assert!((0.0..1.0).contains(&phase));
        phase
    }

    fn amplitude(&self) -> f32 {
        let a = 10.0_f64.powf(self.cn0_db / 20.0) * SQRT_2 * AWGN_RE_SIGMA as f64
            / SAMPLE_RATE_F64.sqrt();
        a as f32
    }

    fn samples(&self) -> impl Iterator<Item = Complex32> {
        let pulses = message_to_pulses(&self.message);
        let a = self.amplitude();
        let message_samples = pulse_shape::apply_filterbank(
            pulse_shape::filterbank(),
            pulses,
            self.clock_phase(),
            self.symbol_clock_error,
        );

        let carrier_rate = 2.0 * PI * self.carrier_frequency_offset / SAMPLE_RATE_F64;
        // Compute phase at first sample so that phase at the timestamp
        // reference (which is the 50% level of the rising edge of the first
        // pulse) is equal to self.carrier_phase. Note that the first sample has
        // associated time self.first_sample_index() (which is always an
        // integer), while the timestamp reference is, by definition, at
        // self.timestamp_samples. The timestamp reference is not an integer
        // sample in general. This means that the 50% level of the rising edge
        // will not be attained in general in an integer sample present in
        // message_samples. It will be attained in between two integer samples
        // (and its exact location can be pinned down by interpolation). In the
        // same way, the desired self.phase will in general be attained (at the
        // same point) in between two integer samples, and can also be pinned
        // down by interpolation of the phases of the samples that straddle this point.
        //
        // The amount of samples that the phase advances between these two
        // instants is their difference, which is represented by the variable
        // time_delta > 0 below. We need to convert that time difference to
        // radians at the given carrier rate, and subtract it from the desired
        // carrier phase at the timestamp reference to get the carrier phase at
        // the first sample.
        let time_delta = self.timestamp_samples - self.first_sample_index() as f64;
        debug_assert!(time_delta > 0.0);
        let mut carrier_phase: f64 =
            math::phase_wrap_f64(self.carrier_phase - time_delta * carrier_rate);

        message_samples.map(move |x| {
            let x = a * x;
            let (sin, cos) = carrier_phase.sin_cos();
            let z = Complex32::new(x * cos as f32, x * sin as f32);

            // increment carrier phase and wrap
            carrier_phase = math::phase_wrap_once_f64(carrier_phase + carrier_rate);

            z
        })
    }
}

/// Converts a [`Duration`] to a number of IQ samples.
pub fn duration_as_samples(duration: Duration) -> u64 {
    const NANOS_IN_SECOND: u128 = 1_000_000_000;
    u64::try_from(
        duration
            .as_nanos()
            .checked_mul(u128::try_from(SAMPLE_RATE_USIZE).unwrap())
            .unwrap()
            / NANOS_IN_SECOND,
    )
    .unwrap()
}

/// Converts a number of IQ samples to a [`Duration`].
pub fn samples_as_duration(samples: u64) -> Duration {
    let nanos_in_sample = 1_000_000_000 / u64::try_from(SAMPLE_RATE_USIZE).unwrap();
    Duration::from_nanos(samples.checked_mul(nanos_in_sample).unwrap())
}

impl SingleMessageSimulation {
    /// Runs the simulation, generating an IQ snippet containing a single ADS-B message.
    pub fn run<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<Complex32> {
        let num_samples = duration_as_samples(self.duration)
            .try_into()
            .expect("number of IQ samples too large to fit into usize");
        let mut iq = Vec::with_capacity(num_samples);
        iq.resize_with(num_samples, || generate_awgn(rng, AWGN_RE_SIGMA));
        let message_samples = self.message_config.samples();
        let j0 = self.message_config.first_sample_index();
        let s = usize::try_from(-j0.min(0)).unwrap();
        let j0 = usize::try_from(j0.max(0)).unwrap();
        for (z, w) in iq.iter_mut().skip(j0).zip(message_samples.skip(s)) {
            *z += w;
        }
        iq
    }
}

impl Simulation {
    /// Runs the simulation, writing IQ data to the writer.
    pub fn run<R: Rng + ?Sized, W: WriteComplex32>(
        &self,
        rng: &mut R,
        writer: &mut W,
    ) -> Result<()> {
        let num_samples = duration_as_samples(self.duration);
        // Number of samples generated at a time: 1 second of IQ data
        const BATCH_SIZE: usize = SAMPLE_RATE_USIZE;
        // Right-hand margin for samples carried out to the next batch.
        const MARGIN_SIZE: usize = pulse_shape::required_samples_for_apply(MAX_MESSAGE_PULSES);
        // Total buffer size. We have one extra sample as a "canary".
        const BUF_SIZE: usize = BATCH_SIZE + MARGIN_SIZE + 1;

        let mut buffer = vec![Complex32::new(0.0, 0.0); BUF_SIZE];
        // This indicates the number of samples that have already been written,
        // which is also the timestamp (in simulation samples) corresponding to
        // the first sample in `buffer`.
        let mut samples_written: u64 = 0;

        // This macro does the following:
        //
        // 1. Determine how many samples in the non-margin part of the current
        // buffer are to be written out.
        //
        // 2. Add AWGN to those samples.
        //
        // 3. Write them out.
        //
        // 4. If we need to write out more samples in future calls, "recycle"
        // the buffer by copying the margin samples to the front and zeroing out
        // the rest of the buffer.
        //
        // It is a macro because the same snippet of code needs to be run in two
        // places below.
        macro_rules! write_out_buffer {
            () => {
                let to_write = usize::try_from(
                    (num_samples - samples_written).min(u64::try_from(BATCH_SIZE).unwrap()),
                )
                .unwrap();
                for z in buffer[..to_write].iter_mut() {
                    *z += generate_awgn(rng, AWGN_RE_SIGMA);
                }
                writer.write_complex32(&buffer[..to_write])?;
                samples_written += u64::try_from(to_write).unwrap();

                if samples_written < num_samples {
                    // Recycle the buffer
                    //
                    // Copy margin samples to the front. They are the first
                    // samples of the next buffer.
                    buffer.copy_within(BATCH_SIZE..BATCH_SIZE + MARGIN_SIZE, 0);
                    // Zero out the rest of the buffer (except for the canary
                    // sample). These are fresh samples.
                    buffer[MARGIN_SIZE..BATCH_SIZE + MARGIN_SIZE].fill(Complex32::new(0.0, 0.0));
                }
            };
        }

        macro_rules! check_messages_per_second {
            ($field:ident, $name:expr) => {
                anyhow::ensure!(
                    self.messages_per_second.$field >= 0.0
                        && self.messages_per_second.$field.is_finite(),
                    "the number of {} messages per second must be non-negative and finite",
                    $name
                );
            };
        }

        check_messages_per_second!(extended_squitter, "extended squitter");
        check_messages_per_second!(acquisition_squitter, "acquisition squitter");
        check_messages_per_second!(short_ap, "short AP");
        check_messages_per_second!(all_call_reply, "all-call reply");
        check_messages_per_second!(long_ap, "long AP");
        check_messages_per_second!(military_extended_squitter, "military extended squitter");

        let messages_per_second = self.messages_per_second.total_messages_per_second();
        anyhow::ensure!(
            messages_per_second > 0.0,
            "the number of messages per second for at least one message type must be positive"
        );
        anyhow::ensure!(
            messages_per_second.is_finite(),
            "the total number of messages per second must be finite"
        );

        // We can simulate K independent Poisson processes as a single Poisson
        // process whose parameter is the sum of the parameters of the K Possion
        // processes, and each event is tagged with an event type (that
        // indicates to which Poisson process it belongs) that is chosen from a
        // discrete distribution on K elements that is weighted according to the
        // parameters of each Poission process.
        //
        // Since the message arrivals are a Poisson process, the inter-arrival
        // times are distributed as an exponential whose parameter lambda is
        // average number of messages per unit. We use units of samples here. So
        // lambda is equal to the average number of messages per sample.
        let messages_per_sample = messages_per_second / SAMPLE_RATE_F64;
        let message_inter_arrival_distr = rand_distr::Exp::new(messages_per_sample).unwrap();
        // WeightedIndex does not require normalized weights. The order of these
        // weights must match the order in MessageType::ALL.
        let message_type_weights = [
            self.messages_per_second.extended_squitter,
            self.messages_per_second.acquisition_squitter,
            self.messages_per_second.short_ap,
            self.messages_per_second.all_call_reply,
            self.messages_per_second.long_ap,
            self.messages_per_second.military_extended_squitter,
        ];
        let message_type_distr = WeightedIndex::new(message_type_weights).unwrap();

        let mut last_message_time = 0.0;
        loop {
            let message_time = last_message_time + message_inter_arrival_distr.sample(rng);
            let message_type = MessageType::ALL[message_type_distr.sample(rng)];
            let message_config = MessageConfig::from_distributions(
                message_time,
                self.message_distributions,
                message_type,
                rng,
            );

            let first_sample = message_config.first_sample_index();
            if first_sample < 0 {
                // A negative sample index can happen at the beginning of
                // the simulation. In this case, we ignore this message. Do not
                // put it in the output to avoid truncating the beginning of
                // the message in the output.
                last_message_time = message_time;
                continue;
            }
            // At this point we know first_sample >= 0.
            let first_sample = u64::try_from(first_sample).unwrap();
            assert!(
                first_sample >= samples_written,
                "Message starts before the beginning of the buffer. \
                 Buffer management bug. (first_sample = {first_sample}, samples_written = {samples_written}"
            );

            if first_sample + u64::try_from(MARGIN_SIZE).unwrap() >= num_samples {
                // The message is not guaranteed to fit fully into the
                // simulation. We are done generating messages. Finalize writing
                // what remains. Typically this loop only iterates once (except
                // when the average number of messages per second is small).
                while samples_written < num_samples {
                    write_out_buffer!();
                }
                // We are done here.
                break;
            }

            while first_sample >= samples_written + u64::try_from(BATCH_SIZE).unwrap() {
                // The message starts beyond the non-margin part of the
                // buffer. Write out the buffer and "recycle" it. Typically this
                // loop only iterates once (except when the average number of
                // messages per second is small).
                write_out_buffer!();
            }

            let message_samples = message_config.samples();
            // We have checked that first_sample >= samples_written.
            let j0 = usize::try_from(first_sample - samples_written).unwrap();
            assert!(j0 < BATCH_SIZE);
            let mut message_samples_written = 0;
            for (z, w) in buffer[j0..].iter_mut().zip(message_samples) {
                *z += w;
                message_samples_written += 1;
            }
            // Check that the canary sample was not written, which means that
            // our margin was large enough to fit the message.
            assert!(j0 + message_samples_written < BUF_SIZE);

            last_message_time = message_time;
        }

        Ok(())
    }
}

/// Generates an AWGN IQ sample.
///
/// The `real_sigma` argument indicates the standard deviation of the real and
/// imaginary parts of the AWGN. The standard deviation of the complex data is
/// then `sqrt(2) * real_sigma`.
pub fn generate_awgn<R: Rng + ?Sized>(rng: &mut R, real_sigma: f32) -> Complex32 {
    Complex32::new(
        real_sigma * rng.sample::<f32, _>(StandardNormal),
        real_sigma * rng.sample::<f32, _>(StandardNormal),
    )
}

/// Generates a random extended squitter reply message.
///
/// The DF field is chosen randomly among the values listed in
/// [`DF_EXTENDED_SQUITTER`]. For DF17, the CA field is chosen randomly among
/// the values listed in [`CA_VALID`]. For DF18, the CF is chosen randomly
/// among the values listed in [`DF18_CF_VALID`]. The rest of the contents of
/// the message are generated randomly, and the CRC-24 is computed based on the
/// message.
pub fn random_extended_squitter<R: Rng + ?Sized>(rng: &mut R) -> ModeSMessage {
    let mut message = [0; MODE_S_LONG_BYTES];
    // overwrite first 5 bits with DF, chosen randomly
    let df = *DF_EXTENDED_SQUITTER.choose(rng).unwrap();
    let ca_cf = match df {
        17 => *CA_VALID.choose(rng).unwrap(),
        18 => *DF18_CF_VALID.choose(rng).unwrap(),
        _ => unreachable!(),
    };
    message[0] = (df << 3) | ca_cf;
    debug_assert_eq!(message[0] >> 3, df);
    debug_assert_eq!(message[0] & 0x7, ca_cf);
    let data_bytes = MODE_S_LONG_BYTES - CRC24_BYTES;
    rng.fill(&mut message[1..data_bytes]);
    let crc = CRC24.checksum(&message[..data_bytes]);
    message[data_bytes] = (crc >> 16) as u8;
    message[data_bytes + 1] = (crc >> 8) as u8;
    message[data_bytes + 2] = crc as u8;
    ModeSMessage::Long(message)
}

/// Generates a random acquisition squitter message.
///
/// The DF field is fixed to [`DF_ACQUISITION_SQUITTER`]. The CA field is chosen
/// randomly among the values listed in [`CA_VALID`]. The aircraft address is
/// chosen randomly. The CRC-24 is computed based on the message.
pub fn random_acquisition_squitter<R: Rng + ?Sized>(rng: &mut R) -> ModeSMessage {
    let mut message = [0; MODE_S_SHORT_BYTES];
    let df = DF_ACQUISITION_SQUITTER;
    let ca = *CA_VALID.choose(rng).unwrap();
    message[0] = (df << 3) | ca;
    debug_assert_eq!(message[0] >> 3, df);
    debug_assert_eq!(message[0] & 0x7, ca);
    let data_bytes = MODE_S_SHORT_BYTES - CRC24_BYTES;
    rng.fill(&mut message[1..data_bytes]);
    let crc = CRC24.checksum(&message[..data_bytes]);
    message[data_bytes] = (crc >> 16) as u8;
    message[data_bytes + 1] = (crc >> 8) as u8;
    message[data_bytes + 2] = crc as u8;
    ModeSMessage::Short(message)
}

/// Generates a random short Mode-S reply with address parity.
///
/// The DF field is chosen randomly among the values listed in [`DF_SHORT_AP`].
/// The remaining contents of the message are chosen randomly, including the
/// CRC-24 field (which simulates choosing a random aircraft address).
pub fn random_short_ap<R: Rng + ?Sized>(rng: &mut R) -> ModeSMessage {
    let mut message = [0; MODE_S_SHORT_BYTES];
    rng.fill(&mut message);
    let df = *DF_SHORT_AP.choose(rng).unwrap();
    message[0] = (df << 3) | (message[0] & 0x7);
    debug_assert_eq!(message[0] >> 3, df);
    ModeSMessage::Short(message)
}

/// Generates a random all-call reply message.
///
/// The DF field is fixed to [`DF_ACQUISITION_SQUITTER`]. The CA field is chosen
/// randomly among the values listed in [`CA_VALID`]. The aircraft address is
/// chosen randomly. The CRC-24 is computed based on the message. The last 7
/// bits of the message are XOR-ed with a non-zero 7-bit value to emulate the CL
/// and IC fields of the interrogator identifier.
pub fn random_all_call_reply<R: Rng + ?Sized>(rng: &mut R) -> ModeSMessage {
    let mut message = [0; MODE_S_SHORT_BYTES];
    let df = DF_ACQUISITION_SQUITTER;
    let ca = *CA_VALID.choose(rng).unwrap();
    message[0] = (df << 3) | ca;
    debug_assert_eq!(message[0] >> 3, df);
    debug_assert_eq!(message[0] & 0x7, ca);
    let data_bytes = MODE_S_SHORT_BYTES - CRC24_BYTES;
    rng.fill(&mut message[1..data_bytes]);
    let crc = CRC24.checksum(&message[..data_bytes]);
    message[data_bytes] = (crc >> 16) as u8;
    message[data_bytes + 1] = (crc >> 8) as u8;
    message[data_bytes + 2] = crc as u8;
    let interrogator_identifier: u8 = rng.random_range(1..128);
    message[data_bytes + 2] ^= interrogator_identifier;
    ModeSMessage::Short(message)
}

/// Generates a random long Mode-S reply with address parity.
///
/// The DF field is chosen randomly among the values listed in [`DF_LONG_AP`].
/// The remaining contents of the message are chosen randomly, including the
/// CRC-24 field (which simulates choosing a random aircraft address).
pub fn random_long_ap<R: Rng + ?Sized>(rng: &mut R) -> ModeSMessage {
    let mut message = [0; MODE_S_LONG_BYTES];
    rng.fill(&mut message);
    let df = *DF_LONG_AP.choose(rng).unwrap();
    message[0] = (df << 3) | (message[0] & 0x7);
    debug_assert_eq!(message[0] >> 3, df);
    ModeSMessage::Long(message)
}

/// Generates a random military extended squitter (DF19) message.
///
/// The DF field is fixed to [`DF_MILITARY_EXTENDED_SQUITTER`]. The remaining
/// contents of the message are chosen randomly, including the CRC-24 field.
pub fn random_military_extended_squitter<R: Rng + ?Sized>(rng: &mut R) -> ModeSMessage {
    let mut message = [0; MODE_S_LONG_BYTES];
    rng.fill(&mut message);
    let df = DF_MILITARY_EXTENDED_SQUITTER;
    message[0] = (df << 3) | (message[0] & 0x7);
    debug_assert_eq!(message[0] >> 3, df);
    ModeSMessage::Long(message)
}

const MAX_MESSAGE_PULSES: usize = 2 * (REPLY_PREAMBLE_BITS + MODE_S_LONG_BITS);

/// Mode-S reply pulses.
///
/// This structure stores an array of pulses that is sized for a Mode-S long
/// reply, together with a length field that indicates how many pulses are
/// valid, so that the same structure can also be used to represent Mode-S short
/// replies.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Pulses {
    data: [bool; MAX_MESSAGE_PULSES],
    len: usize,
}

impl Pulses {
    /// Returns the pulses as a slice.
    fn as_slice(&self) -> &[bool] {
        &self.data[..self.len]
    }

    /// Returns the number of pulses.
    fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the pulses slice is empty.
    fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Converts a mode-S long reply message to a sequence of PPM pulses.
///
/// The output includes the reply preamble.
pub fn message_to_pulses(message: &ModeSMessage) -> Pulses {
    let mut pulses = [false; MAX_MESSAGE_PULSES];

    // preamble pulses
    pulses[0] = true;
    pulses[2] = true;
    pulses[7] = true;
    pulses[9] = true;

    for (j, byte) in message.as_slice().iter().enumerate() {
        for k in 0..8 {
            let bit = (byte >> (7 - k)) & 1 != 0;
            let base = 2 * (REPLY_PREAMBLE_BITS + 8 * j + k);
            if bit {
                pulses[base] = true;
            } else {
                pulses[base + 1] = true;
            }
        }
    }

    Pulses {
        data: pulses,
        len: 2 * (REPLY_PREAMBLE_BITS + message.num_bits()),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn message_type_weights() {
        let sum = MessageType::TYPICAL_WEIGHTS.iter().sum::<f64>();
        assert_eq!(sum, 1.0);
    }

    #[test]
    fn duration_as_samples_one_second() {
        assert_eq!(
            duration_as_samples(Duration::from_secs(1)),
            u64::try_from(SAMPLE_RATE_USIZE).unwrap()
        );
    }

    #[test]
    fn samples_as_duration_one_second() {
        assert_eq!(
            samples_as_duration(u64::try_from(SAMPLE_RATE_USIZE).unwrap()),
            Duration::from_secs(1)
        );
    }

    #[test]
    fn generate_awgn_mean_and_sigma() {
        use num::complex::Complex64;
        let real_sigma = 1.0;
        let num_samples = 1_000_000;
        let mut sum = Complex64::new(0.0, 0.0);
        let mut sum_norm_sqr: f64 = 0.0;
        let mut rng = rand::rng();
        for _ in 0..num_samples {
            let z = generate_awgn(&mut rng, real_sigma as f32);
            let z = Complex64::new(f64::from(z.re), f64::from(z.im));
            sum += z;
            sum_norm_sqr += z.norm_sqr();
        }
        let mean = sum / num_samples as f64;
        assert!(mean.norm() <= 5e-3, "mean = {mean}");
        let variance = sum_norm_sqr / num_samples as f64 - mean.norm_sqr();
        assert!(
            (variance - 2.0 * real_sigma * real_sigma).abs() < 5e-3,
            "variance = {variance}"
        );
    }

    proptest! {
        /// Generates a single-message simulation and checks that the first rising
        /// edge is at the right sample and has the required phase.
        #[test]
        fn rising_edge_time_and_phase(
            timestamp_samples in 100.0_f64..=2400.0, // in samples
            // Frequency error (generally) large enough to have significant rotation on each
            // microsecond interval
            carrier_frequency_offset in -1.5e6..=1.5e6,
            carrier_phase in -PI..PI,
            symbol_clock_error in -100e-6..=100e-6,
            message: [u8; MODE_S_LONG_BYTES],
        )
        {
            let simulation = SingleMessageSimulation {
                duration: Duration::from_nanos(400_000),
                message_config: MessageConfig {
                    timestamp_samples,
                    // CN0 high enough so that noise can be ignored
                    cn0_db: 120.0,
                    carrier_frequency_offset,
                    carrier_phase,
                    symbol_clock_error,
                    message: ModeSMessage::Long(message),
                },
            };
            let mut rng = rand::rng();
            let iq = simulation.run(&mut rng);
            // In general the first rising edge occurs in between two
            // samples. We recover it by linear interpolation, which is crude,
            // but good enough for this test. Because the phase can be rotating
            // rather quickly between z0 and z1, and linear interpolation has a
            // large error in that situation, we first counter-rotate z0 and z1
            // by their expected phases before interpolating linearly. In this
            // way we obtain a value that is already counter-rotated to be real
            // and positive, which makes checking easier.
            let z0 = iq[timestamp_samples as usize];
            let z1 = iq[timestamp_samples as usize + 1];
            let t0 = (timestamp_samples as usize) as f64;
            let phase_rate = 2.0 * PI * carrier_frequency_offset / SAMPLE_RATE_F64;
            // Expected phases for z0 and z1.
            let phi0 = carrier_phase + (t0 - timestamp_samples) * phase_rate;
            let phi1 = carrier_phase + (t0 + 1.0 - timestamp_samples) * phase_rate;
            let z0 = z0 * Complex32::new(phi0.cos() as f32, -phi0.sin() as f32);
            let z1 = z1 * Complex32::new(phi1.cos() as f32, -phi1.sin() as f32);
            let tau = (timestamp_samples - t0) as f32;
            let z = (1.0 - tau) * z0 + tau * z1;
            let trigger_level = 0.5 * simulation.message_config.amplitude();
            // Tolerance is relatively high for this test because the sample we
            // check sees inter-symbol interference from the adjacent pulse.
            assert!(
                (z.re - trigger_level).abs() < 3e-2,
                "z.re = {}, trigger_level = {}", z.re, trigger_level,
            );
            assert!(z.im.abs() < 5.0 * AWGN_RE_SIGMA, "z.im = {}", z.im);
        }
    }

    prop_compose! {
        fn random_extended_squitter_any()(seed: u64) -> ModeSMessage {
            let mut rng = crate::simulation::rng::rng_from_seed(seed);
            random_extended_squitter(&mut rng)
        }
    }

    proptest! {
        #[test]
        fn random_extended_squitter_crc(message in random_extended_squitter_any()) {
            assert_eq!(CRC24.checksum(message.as_slice()), 0);
        }
    }

    proptest! {
        #[test]
        fn random_extended_squitter_is_valid(message in random_extended_squitter_any()) {
            let message = message.as_slice();
            let df = message[0] >> 3;
            assert!(DF_EXTENDED_SQUITTER.contains(&df));
            let ca_cf = message[0] & 0x7;
            match df {
                17 => assert!(CA_VALID.contains(&ca_cf)),
                18 => assert!(DF18_CF_VALID.contains(&ca_cf)),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn long_message_to_pulses_all_ones() {
        let mut pulses = Pulses {
            data: [false; MAX_MESSAGE_PULSES],
            len: MAX_MESSAGE_PULSES,
        };
        pulses.data[0] = true;
        pulses.data[2] = true;
        pulses.data[7] = true;
        pulses.data[9] = true;
        for n in 0..8 * MODE_S_LONG_BYTES {
            pulses.data[16 + 2 * n] = true;
        }
        let message = ModeSMessage::Long([0xff; MODE_S_LONG_BYTES]);
        assert_eq!(message_to_pulses(&message), pulses);
    }

    proptest! {
        fn long_message_to_pulses_num_pulses(message: [u8; MODE_S_LONG_BYTES]) {
            let pulses = message_to_pulses(&ModeSMessage::Long(message));
            let num_pulses = pulses.as_slice().iter().filter(|pulse| **pulse).count();
            let num_preamble_pulses = 4;
            assert_eq!(num_pulses, num_preamble_pulses + 8 * MODE_S_LONG_BYTES);
        }
    }
}
