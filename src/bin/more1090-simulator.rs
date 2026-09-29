//! [`more1090`] ADS-B signal simulator.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use more1090::{
    cli::Seconds,
    constants::{
        dsp::{SAMPLE_RATE_F64, SPS},
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
    about,
    long_about = "ADS-B signal simulator from the more1090 project"
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
        #[arg(long, default_value_t = 100.0)]
        max_cn0: f64,
        /// Minimum distance to transmitters in km.
        #[arg(long, default_value_t = 10.0)]
        min_distance: f64,
        /// Maximum distance to transmitters in km.
        #[arg(long, default_value_t = 300.0)]
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

    fn rng(&self) -> impl Rng + use<> {
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

    match args.command {
        Command::SingleMessage {
            message_start,
            cn0,
            carrier_frequency_offset,
            carrier_phase,
            symbol_clock_error,
            ..
        } => {
            let num_samples = duration_as_samples(duration);
            let mut rng = args.rng();
            let timestamp_samples = if let Some(message_start) = message_start {
                message_start * SAMPLE_RATE_F64
            } else {
                // This assumes that the symbol clock error is zero.  If the
                // symbol clock error is non-zero, the message will not be centered
                // exactly in the middle of the simualation time span.
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
                    MessageType::ExtendedSquitter,
                    &mut rng,
                ),
            };
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
            };
            let mut rng = args.rng();
            let mut writer = create_writer!();
            simulation.run(&mut rng, &mut writer)?;
            flush_writer!(writer);
        }
    }

    let meta = sigmf::simulation_meta();
    meta.to_file(sigmf::meta_path(&args.global().sigmf_file)?)
        .context("failed to write .sigmf-meta file")?;

    Ok(())
}
