//! [`more1090`] ADS-B receiver benchmark.

use anyhow::Result;
use clap::{Parser, Subcommand};
use more1090::{
    benchmark::{
        sensitivity::{DecodingBenchmark, FalseAcquisitionBenchmark, ScenarioBenchmark},
        throughput::ThroughputBenchmark,
    },
    cli::{ADSBReceiverArgs, Seconds},
    constants::dsp::SAMPLE_RATE_USIZE,
    simulation::{MessagesPerSecond, Simulation, distributions::MessageDistributions},
};

/// Benchmark CLI arguments.
#[derive(Debug, Clone, PartialEq, Parser)]
#[command(version, about = "more1090 ADS-B receiver benchmark")]
pub struct Args {
    #[command(subcommand)]
    command: Command,
}

/// Benchmark commands.
#[derive(Debug, Clone, PartialEq, Subcommand)]
pub enum Command {
    /// Throughput benchmark.
    Throughput {
        /// Run benchmark for a single iteration only.
        #[arg(long)]
        single_iteration: bool,
        /// Simulated scenario duration.
        #[arg(long,
              default_value_t = Seconds(ThroughputBenchmark::default().scenario.duration))]
        scenario_duration: Seconds,
        /// Extended squitter (DF17/18) messages per second in simulated scenario.
        #[arg(long,
              default_value_t = ThroughputBenchmark::default().scenario.messages_per_second.extended_squitter)]
        messages_per_second_extended_squitter: f64,
        /// Acquisition squitter (DF11) messages per second in simulated scenario.
        #[arg(long,
              default_value_t = ThroughputBenchmark::default().scenario.messages_per_second.acquisition_squitter)]
        messages_per_second_acquisition_squitter: f64,
        /// Short address parity (DF0/4/5) messages per second in simulated scenario.
        #[arg(long,
              default_value_t = ThroughputBenchmark::default().scenario.messages_per_second.short_ap)]
        messages_per_second_short_ap: f64,
        /// All-call reply (DF11) messages per second in simulated scenario.
        #[arg(long,
              default_value_t = ThroughputBenchmark::default().scenario.messages_per_second.all_call_reply)]
        messages_per_second_all_call_reply: f64,
        /// Long address parity (DF16/20/21/24-31) messages per second in simulated scenario.
        #[arg(long,
              default_value_t = ThroughputBenchmark::default().scenario.messages_per_second.long_ap)]
        messages_per_second_long_ap: f64,
        /// Military extended squitter (DF19) messages per second in simulated scenario.
        #[arg(long,
              default_value_t = ThroughputBenchmark::default().scenario.messages_per_second.military_extended_squitter)]
        messages_per_second_military_extended_squitter: f64,
        #[arg(long,
              default_value_t = ThroughputBenchmark::default().scenario.messages_per_second.mode_ac)]
        messages_per_second_mode_ac: f64,
        /// Report period.
        #[arg(long,
              default_value_t = Seconds(ThroughputBenchmark::default().report_period))]
        report_period: Seconds,
        #[command(flatten)]
        receiver_args: ADSBReceiverArgs,
    },
    /// False acquisition benchmark.
    FalseAcquisition {
        /// Benchmark duration.
        ///
        /// This defines the benchmark duration in terms of IQ input data
        /// processed.
        #[arg(long,
              default_value_t = Seconds(FalseAcquisitionBenchmark::default().duration))]
        duration: Seconds,
        /// Random seed.
        #[arg(long, default_value_t = 0)]
        random_seed: u64,
        /// Number of threads to use for running the benchmark.
        #[arg(long,
              default_value_t = FalseAcquisitionBenchmark::default().num_threads)]
        num_threads: std::num::NonZero<usize>,
        /// Output results in JSON format.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        receiver_args: ADSBReceiverArgs,
    },
    /// Decoding benchmark.
    Decoding {
        /// CN0 in dB·Hz.
        #[arg(long)]
        cn0: f64,
        /// Number of packets to simulate.
        #[arg(long,
              default_value_t = DecodingBenchmark::default().num_packets)]
        num_packets: u64,
        /// Random seed.
        #[arg(long, default_value_t = 0)]
        random_seed: u64,
        /// Number of threads to use for running the benchmark.
        #[arg(long,
              default_value_t = DecodingBenchmark::default().num_threads)]
        num_threads: std::num::NonZero<usize>,
        /// Output results in JSON format.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        receiver_args: ADSBReceiverArgs,
    },
    /// Scenario benchmark.
    Scenario {
        /// Total scenario duration to simulate.
        #[arg(long,
              default_value_t = Seconds(ScenarioBenchmark::default().duration))]
        duration: Seconds,
        /// Total number of Mode-S messages per second.
        #[arg(long,
              default_value_t = ScenarioBenchmark::default().messages_per_second)]
        messages_per_second: f64,
        /// Maximum CN0 in dB·Hz.
        #[arg(long,
              default_value_t = ScenarioBenchmark::default().max_cn0)]
        max_cn0: f64,
        /// Minimum transmitter distance in km.
        #[arg(long,
              default_value_t = ScenarioBenchmark::default().min_distance)]
        min_distance: f64,
        /// Maximum transmitter distance in km.
        #[arg(long,
              default_value_t = ScenarioBenchmark::default().max_distance)]
        max_distance: f64,
        /// Random seed.
        #[arg(long, default_value_t = 0)]
        random_seed: u64,
        /// Number of threads to use for running the benchmark.
        #[arg(long,
              default_value_t = ScenarioBenchmark::default().num_threads)]
        num_threads: std::num::NonZero<usize>,
        /// Output results in JSON format.
        #[arg(long)]
        json: bool,
        #[command(flatten)]
        receiver_args: ADSBReceiverArgs,
    },
}

fn output_results<B, R>(benchmark: &B, results: &R, json: bool) -> Result<()>
where
    B: serde::Serialize,
    R: serde::Serialize + std::fmt::Display,
{
    if json {
        let out = serde_json::json!({
            "benchmark": benchmark,
            "results": results,
        });
        print!("{}", serde_json::to_string_pretty(&out)?);
    } else {
        print!("{}", results);
    }
    Ok(())
}

pub fn main() -> Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Throughput {
            single_iteration,
            scenario_duration,
            messages_per_second_extended_squitter,
            messages_per_second_acquisition_squitter,
            messages_per_second_short_ap,
            messages_per_second_all_call_reply,
            messages_per_second_long_ap,
            messages_per_second_military_extended_squitter,
            messages_per_second_mode_ac,
            report_period,
            receiver_args,
        } => ThroughputBenchmark {
            scenario: Simulation {
                duration: scenario_duration.into(),
                messages_per_second: MessagesPerSecond {
                    extended_squitter: messages_per_second_extended_squitter,
                    acquisition_squitter: messages_per_second_acquisition_squitter,
                    short_ap: messages_per_second_short_ap,
                    all_call_reply: messages_per_second_all_call_reply,
                    long_ap: messages_per_second_long_ap,
                    military_extended_squitter: messages_per_second_military_extended_squitter,
                    mode_ac: messages_per_second_mode_ac,
                },
                message_distributions: MessageDistributions::default(),
                // A batch size of 1 second is good to reduce overhead.
                batch_size: SAMPLE_RATE_USIZE,
            },
            loop_scenario: !single_iteration,
            report_period: report_period.into(),
            receiver_args,
        }
        .run(),
        Command::FalseAcquisition {
            duration,
            random_seed,
            num_threads,
            json,
            receiver_args,
        } => {
            let benchmark = FalseAcquisitionBenchmark {
                duration: duration.into(),
                receiver_args,
                random_seed,
                num_threads,
            };
            let results = benchmark.run()?;
            output_results(&benchmark, &results, json)?;
            Ok(())
        }
        Command::Decoding {
            cn0,
            num_packets,
            random_seed,
            num_threads,
            json,
            receiver_args,
        } => {
            let benchmark = DecodingBenchmark {
                num_packets,
                cn0,
                receiver_args,
                random_seed,
                num_threads,
            };
            let results = benchmark.run()?;
            output_results(&benchmark, &results, json)?;
            Ok(())
        }
        Command::Scenario {
            duration,
            messages_per_second,
            max_cn0,
            min_distance,
            max_distance,
            random_seed,
            num_threads,
            json,
            receiver_args,
        } => {
            let benchmark = ScenarioBenchmark {
                duration: duration.into(),
                messages_per_second,
                max_cn0,
                min_distance,
                max_distance,
                receiver_args,
                random_seed,
                num_threads,
            };
            let results = benchmark.run()?;
            output_results(&benchmark, &results, json)?;
            Ok(())
        }
    }
}
