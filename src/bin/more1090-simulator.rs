//! [`more1090`] ADS-B signal simulator.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use more1090::{
    cli::Seconds,
    constants::{
        dsp::{SAMPLE_RATE_F64, SAMPLE_RATE_USIZE, SPS},
        mode_s::TOTAL_MODE_S_LONG_REPLY_BITS,
    },
    io::{CF32LEWriter, WriteComplex32},
    sigmf,
    simulation::{
        MessageConfig, MessageType, MessagesPerSecond, Simulation, SingleMessageSimulation,
        distributions::{
            CN0Distribution, CarrierFrequencyOffsetDistribution, CarrierPhaseDistribution,
            ConstantCN0, ConstantCarrierFrequencyOffset, ConstantCarrierPhase,
            ConstantSymbolClockError, DistanceRangeCN0, MessageDistributions,
            SymbolClockErrorDistribution,
        },
        duration_as_samples,
        rng::rng_from_seed,
    },
};
use rand::prelude::*;
use std::{fs::File, path::PathBuf, time::Duration};

/// Simulation CLI arguments.
#[derive(Debug, Clone, PartialEq, Parser)]
#[command(
    version,
    about = "Secondary surveillance radar reply signal simulator from the more1090 project"
)]
pub struct Args {
    /// Subcommand.
    #[command(subcommand)]
    pub command: Command,
}

/// Global arguments that apply to all the commands.
#[derive(Debug, Clone, PartialEq, clap::Args)]
pub struct GlobalArgs {
    /// Output SigMF file.
    pub sigmf_file: PathBuf,
    /// Random seed.
    ///
    /// Random seed used to generate simulation. If a seed is not specified,
    /// then a random seed will be generated and printed out.
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Commands supported by the simulator.
#[derive(Debug, Clone, PartialEq, Subcommand)]
pub enum Command {
    /// IQ snippet with a single message.
    SingleMessage {
        /// Global arguments.
        #[clap(flatten)]
        global: GlobalArgs,
        /// Duration in seconds.
        #[arg(long, default_value_t = Seconds(Duration::from_millis(1)))]
        duration: Seconds,
        /// Message start in seconds.
        ///
        /// The default is to center the message in the simulation time span, with a
        /// random offset of +/- 120 usec). The time is referenced to the first
        /// rising edge of the message.
        #[arg(long)]
        message_start: Option<f64>,
        /// CN0 in dB·Hz.
        ///
        /// The CN0 is referred to the peak power of an ideal rectangular
        /// pulse. It does not take into account the 50% duty cycle of ADS-B or
        /// the pulse shaping that is used in the simulation.
        #[arg(long, default_value_t = 90.0)]
        cn0: f64,
        /// Carrier frequency offset in Hz.
        ///
        /// The default is to choose randomly from a normal distribution with a
        /// standard deviation of 100 kHz, clamped to 1 MHz.
        // This documentation needs to match CarrierFrequencyOffsetDistribution::default
        #[arg(long)]
        carrier_frequency_offset: Option<f64>,
        /// Initial carrier phase in radians.
        ///
        /// The default is to choose randomly uniformly in [-pi, pi). This value
        /// gives the carrier phase at the first rising edge of the message.
        #[arg(long)]
        carrier_phase: Option<f64>,
        /// Symbol clock error in parts per one.
        ///
        /// The default is to choose randomly uniformly in [-50e-6, 50e-6].
        // This documentation needs to match SymbolClockErrorDistribution::default
        #[arg(long)]
        symbol_clock_error: Option<f64>,
        /// Message type.
        #[arg(long, default_value = "extended-squitter")]
        message_type: MessageType,
    },
    /// Full scenario containing messages with parameters drawn from random
    /// distributions.
    Scenario {
        /// Global arguments.
        #[clap(flatten)]
        global: GlobalArgs,
        /// Duration in seconds.
        #[arg(long, default_value_t = Seconds(Duration::from_secs(10)))]
        duration: Seconds,
        /// Average total number of messages per second.
        #[arg(long, default_value_t = 2000.0)]
        messages_per_second: f64,
        /// Maximum CN0 in dB·Hz.
        #[arg(long, default_value_t = 105.0)]
        max_cn0: f64,
        /// Minimum distance to transmitters in km.
        #[arg(long, default_value_t = 10.0)]
        min_distance: f64,
        /// Maximum distance to transmitters in km.
        #[arg(long, default_value_t = 500.0)]
        max_distance: f64,
    },
}

impl Args {
    fn duration(&self) -> Duration {
        match &self.command {
            Command::SingleMessage { duration, .. } => duration.0,
            Command::Scenario { duration, .. } => duration.0,
        }
    }

    fn global(&self) -> &GlobalArgs {
        match &self.command {
            Command::SingleMessage { global, .. } => global,
            Command::Scenario { global, .. } => global,
        }
    }

    fn rng(&self) -> impl Rng + SeedableRng + use<> {
        let seed = if let Some(seed) = self.global().seed {
            seed
        } else {
            let seed = rand::random();
            eprintln!("random seed = {seed}");
            seed
        };
        rng_from_seed(seed)
    }
}

fn map_or_ok_default<T, U, E, F>(option: Option<T>, f: F) -> Result<U, E>
where
    U: Default,
    F: FnOnce(T) -> Result<U, E>,
{
    if let Some(x) = option {
        f(x)
    } else {
        Ok(U::default())
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct MessageCounter {
    extended_squitter: u64,
    acquisition_squitter: u64,
    short_ap: u64,
    all_call_reply: u64,
    long_ap: u64,
    military_extended_squitter: u64,
    mode_ac: u64,
}

impl Default for MessageCounter {
    fn default() -> MessageCounter {
        MessageCounter::new()
    }
}

impl MessageCounter {
    fn new() -> MessageCounter {
        MessageCounter {
            extended_squitter: 0,
            acquisition_squitter: 0,
            short_ap: 0,
            all_call_reply: 0,
            long_ap: 0,
            military_extended_squitter: 0,
            mode_ac: 0,
        }
    }

    fn get_and_increment(&mut self, message_type: MessageType) -> u64 {
        macro_rules! post_incr {
            ($field:ident) => {{
                let ret = self.$field;
                self.$field += 1;
                ret
            }};
        }

        match message_type {
            MessageType::ExtendedSquitter => post_incr!(extended_squitter),
            MessageType::AcquisitionSquitter => post_incr!(acquisition_squitter),
            MessageType::ShortAP => post_incr!(short_ap),
            MessageType::AllCallReply => post_incr!(all_call_reply),
            MessageType::LongAP => post_incr!(long_ap),
            MessageType::MilitaryExtendedSquitter => post_incr!(military_extended_squitter),
            MessageType::ModeAC => post_incr!(mode_ac),
        }
    }
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    let duration = args.duration();

    macro_rules! create_writer {
        () => {
            CF32LEWriter::new(
                File::create(sigmf::data_path(&args.global().sigmf_file)?)
                    .context("failed to create output .sigmf-data file")?,
            )
        };
    }

    macro_rules! flush_writer {
        ($writer:expr) => {
            $writer.flush().context("failed to flush output file")?
        };
    }

    let mut meta = sigmf::simulation_meta();

    match args.command {
        Command::SingleMessage {
            message_start,
            cn0,
            carrier_frequency_offset,
            carrier_phase,
            symbol_clock_error,
            message_type,
            ..
        } => {
            let num_samples = duration_as_samples(duration);
            let mut rng = args.rng();
            let timestamp_samples = if let Some(message_start) = message_start {
                message_start * SAMPLE_RATE_F64
            } else {
                // This assumes that the symbol clock error is zero.  If the
                // symbol clock error is non-zero, the message will not be centered
                // exactly in the middle of the simulation time span.
                let message_samples = 2 * TOTAL_MODE_S_LONG_REPLY_BITS * SPS;
                let nominal = (num_samples as f64 - message_samples as f64) * 0.5;
                let max_error = message_samples as f64;
                let error = rng.random_range(-max_error..=max_error);
                nominal + error
            };
            let distributions = MessageDistributions {
                cn0: CN0Distribution::Constant(ConstantCN0::new(cn0)?),
                carrier_frequency_offset: map_or_ok_default(carrier_frequency_offset, |x| {
                    ConstantCarrierFrequencyOffset::new(x)
                        .map(CarrierFrequencyOffsetDistribution::Constant)
                })?,
                carrier_phase: map_or_ok_default(carrier_phase, |x| {
                    ConstantCarrierPhase::new(x).map(CarrierPhaseDistribution::Constant)
                })?,
                symbol_clock_error: map_or_ok_default(symbol_clock_error, |x| {
                    ConstantSymbolClockError::new(x).map(SymbolClockErrorDistribution::Constant)
                })?,
            };
            let simulation = SingleMessageSimulation {
                duration,
                message_config: MessageConfig::from_distributions(
                    timestamp_samples,
                    distributions,
                    message_type,
                    &mut rng,
                ),
            };
            let label = "";
            if let Err(err) =
                meta.add_simulation_message_annotation(&simulation.message_config, label)
            {
                eprintln!("failed to add SigMF annotation: {err}");
            }
            let iq = simulation.run(&mut rng);
            let mut writer = create_writer!();
            writer
                .write_complex32(&iq)
                .context("failed to write to output file")?;
            flush_writer!(writer);
        }
        Command::Scenario {
            messages_per_second,
            max_cn0,
            min_distance,
            max_distance,
            ..
        } => {
            let simulation = Simulation {
                duration,
                messages_per_second: MessagesPerSecond::from_total_messages_per_second(
                    messages_per_second,
                )?,
                message_distributions: MessageDistributions {
                    cn0: CN0Distribution::DistanceRange(DistanceRangeCN0::new(
                        max_cn0,
                        min_distance..max_distance,
                    )?),
                    carrier_frequency_offset: Default::default(),
                    carrier_phase: Default::default(),
                    symbol_clock_error: Default::default(),
                },
                // A batch size of 1 second is good to reduce overhead.
                batch_size: SAMPLE_RATE_USIZE,
            };
            let mut rng = args.rng();
            let mut writer = create_writer!();
            let mut num_messages = MessageCounter::new();
            simulation.run(&mut rng, &mut writer, |message_config| {
                let message_type = MessageType::from_message(&message_config.message).unwrap();
                let num_message = num_messages.get_and_increment(message_type);
                let label = format!("{message_type} {num_message}");
                meta.add_simulation_message_annotation(message_config, &label)?;
                Ok(())
            })?;
            flush_writer!(writer);
        }
    }

    meta.to_file(sigmf::meta_path(&args.global().sigmf_file)?)
        .context("failed to write .sigmf-meta file")?;

    Ok(())
}
