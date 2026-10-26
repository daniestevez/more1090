//! Sensitivity benchmarking.
//!
//! This module contains routines to benchmark the sensitivity of the more1090
//! ADS-B receiver.

use crate::{
    A, ADSBReceiver, AVec, CHUNK_LEN,
    cli::ADSBReceiverArgs,
    constants::{
        dsp::{REPLY_PREAMBLE_SAMPLES, SPS},
        mode_s::TOTAL_MODE_S_LONG_REPLY_BITS,
        simulation::AWGN_RE_SIGMA,
    },
    io::WriteComplex32,
    mode_s::ModeSMessage,
    receiver::{ADSBDecode, ADSBPacket, BitErrors, DemodulationType},
    simulation::{
        MessageConfig, MessageType, MessagesPerSecond, SSRReply, Simulation,
        SingleMessageSimulation,
        distributions::{CN0Distribution, ConstantCN0, DistanceRangeCN0, MessageDistributions},
        duration_as_samples, generate_awgn, message_num_samples,
        rng::rng_from_seed,
        samples_as_duration,
    },
};
use anyhow::Result;
use num::complex::Complex32;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    cell::RefCell,
    num::NonZero,
    rc::Rc,
    thread,
    time::{Duration, Instant},
};

/// False acquisition benchmark.
///
/// This structure defines the configuration of a false acquisition benchmark,
/// which runs the receiver on AWGN input and measures the number of detections
/// per second. The benchmark results, except for the througput, depend
/// deterministically on all these parameters, including the number of threads
/// (if the benchmark is run with the same parameters except for the number of
/// threads, the results will be different but close).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FalseAcquisitionBenchmark {
    /// Benchmark duration.
    ///
    /// This defines the benchmark duration in terms of IQ input data processed.
    pub duration: Duration,
    /// ADS-B receiver arguments.
    pub receiver_args: ADSBReceiverArgs,
    /// Random seed.
    pub random_seed: u64,
    /// Number of threads to use for running the benchmark.
    pub num_threads: NonZero<usize>,
}

/// False acquisition benchmark results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FalseAcquisitionBenchmarkResults {
    /// Average number of false detections per second.
    pub false_detections_per_second: f64,
    /// Average time in seconds between false detections.
    pub seconds_per_false_detection: f64,
    /// Total number of false detections.
    pub false_detections: u64,
    /// Benchmark duration (IQ input time).
    pub benchmark_duration: Duration,
    /// Real time benchmark execution duration.
    pub real_time_duration: Duration,
    /// Total throughput in samples/s.
    pub throughput_sps: f64,
    /// Per thread throughput in samples/s.
    pub thread_throughput_sps: f64,
}

impl std::fmt::Display for FalseAcquisitionBenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        writeln!(
            f,
            "Detections per second: {:.e}",
            self.false_detections_per_second
        )?;
        writeln!(
            f,
            "Seconds per detection: {:.3}",
            self.seconds_per_false_detection
        )?;
        writeln!(f, "Total detections: {}", self.false_detections)?;
        writeln!(f, "Benchmark duration: {:?}", self.benchmark_duration)?;
        writeln!(f, "Real-time duration: {:?} s", self.real_time_duration)?;
        writeln!(
            f,
            "Aggregate throughput: {:.3} Msps",
            1e-6 * self.throughput_sps
        )?;
        writeln!(
            f,
            "Per-thread throughput: {:.3} Msps",
            1e-6 * self.thread_throughput_sps
        )?;
        Ok(())
    }
}

fn default_num_threads() -> NonZero<usize> {
    std::thread::available_parallelism().unwrap_or(NonZero::new(1).unwrap())
}

impl Default for FalseAcquisitionBenchmark {
    fn default() -> FalseAcquisitionBenchmark {
        FalseAcquisitionBenchmark {
            duration: Duration::from_secs(10000),
            receiver_args: ADSBReceiverArgs::default(),
            random_seed: 0,
            num_threads: default_num_threads(),
        }
    }
}

#[derive(Debug)]
struct FalseAcquisitionRunner {
    chunks_per_thread: u64,
    receiver_args: ADSBReceiverArgs,
    num_threads: usize,
}

impl FalseAcquisitionRunner {
    fn new(benchmark: &FalseAcquisitionBenchmark) -> Result<FalseAcquisitionRunner> {
        benchmark.receiver_args.validate()?;
        let chunks =
            duration_as_samples(benchmark.duration).div_ceil(u64::try_from(CHUNK_LEN).unwrap());
        let num_threads = benchmark.num_threads.get();
        Ok(FalseAcquisitionRunner {
            chunks_per_thread: chunks.div_ceil(u64::try_from(num_threads).unwrap()),
            receiver_args: benchmark.receiver_args.clone(),
            num_threads,
        })
    }

    fn run(&self, random_seed: u64) -> u64 {
        let mut rng = rng_from_seed(random_seed);
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(self.num_threads);
            for _ in 0..self.num_threads {
                let thread_seed = rng.next_u64();
                handles.push(s.spawn(move || self.run_thread(thread_seed)));
            }
            let mut detections = 0;
            for handle in handles {
                detections += handle.join().expect("runner thread panicked");
            }
            detections
        })
    }

    fn run_thread(&self, thread_seed: u64) -> u64 {
        let mut rng = rng_from_seed(thread_seed);
        let mut receiver = self.receiver_args.new_receiver().unwrap();
        let mut buf: A<_> = aligned::Aligned([Complex32::default(); CHUNK_LEN]);
        let mut detections = 0;
        for _ in 0..self.chunks_per_thread {
            buf.fill_with(|| generate_awgn(&mut rng, AWGN_RE_SIGMA));
            let detected = receiver.process_chunk(&buf).is_some();
            if detected {
                detections += 1;
            }
        }
        detections
    }

    fn results(
        &self,
        false_detections: u64,
        elapsed: Duration,
    ) -> FalseAcquisitionBenchmarkResults {
        let samples_done = self.chunks_per_thread
            * u64::try_from(self.num_threads.checked_mul(CHUNK_LEN).unwrap()).unwrap();
        let benchmark_duration = samples_as_duration(samples_done);
        let throughput_sps = samples_done as f64 / elapsed.as_secs_f64();
        let false_detections_per_second =
            false_detections as f64 / benchmark_duration.as_secs_f64();
        FalseAcquisitionBenchmarkResults {
            false_detections_per_second,
            seconds_per_false_detection: 1.0 / false_detections_per_second,
            false_detections,
            benchmark_duration,
            real_time_duration: elapsed,
            throughput_sps,
            thread_throughput_sps: throughput_sps / self.num_threads as f64,
        }
    }
}

impl FalseAcquisitionBenchmark {
    /// Runs the false acquisition benchmark.
    ///
    /// This function runs until the maximum benchmark duration is reached, and
    /// returns the results.
    pub fn run(&self) -> Result<FalseAcquisitionBenchmarkResults> {
        let runner = FalseAcquisitionRunner::new(self)?;
        let start = Instant::now();
        let false_detections = runner.run(self.random_seed);
        let elapsed = start.elapsed();
        Ok(runner.results(false_detections, elapsed))
    }
}

/// Decoding benchmark.
///
/// This structure defines the configuration of a decoding benchmark, which runs
/// the receiver on signal snippets containing a single ADS-B packet in AWGN at
/// a given CN0 and counts the number of successful detections and decodes. The
/// benchmark results, except for the throughput, depend deterministically on
/// all these parameters, including the number of threads (if the benchmark is
/// run with the same parameters except for the number of threads, the results
/// will be different but close).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodingBenchmark {
    /// Number of packets (signal snippets) to test in the benchmark.
    pub num_packets: u64,
    /// CN0 (in dB·Hz) of the packets.
    pub cn0: f64,
    /// ADS-B receiver arguments.
    pub receiver_args: ADSBReceiverArgs,
    /// Random seed.
    pub random_seed: u64,
    /// Number of threads to use for running the benchmark.
    pub num_threads: NonZero<usize>,
}

/// Decoding benchmark results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodingBenchmarkResults {
    /// Average number detected packets.
    ///
    /// This is defined as the number of detections divided by the number of
    /// simulated packets. Because of false detections, this number can exceed
    /// one in some cases.
    pub detection_probability: f64,
    /// Total number of detected packets.
    pub num_detections: u64,
    /// Probability of successfully decoding a packet.
    pub decode_probability: f64,
    /// Probability of successfully decoding a packet with zero bit errors.
    pub decode_probability_zero_bit_errors: f64,
    /// Probability of successfully decoding a packet with exactly one bit
    /// error.
    pub decode_probability_one_bit_error: f64,
    /// Probability of successfully decoding a packet with exactly two bit
    /// errors.
    pub decode_probability_two_bit_errors: f64,
    /// Probability that a successful decoded used noncoherent demodulation.
    pub decode_probability_noncoherent: f64,
    /// Total number of successfully decoded packets.
    pub num_decoded: u64,
    /// Total number of successfully decoded packets with zero bit errors.
    pub num_decoded_zero_bit_errors: u64,
    /// Total number of successfully decoded packets with one bit error.
    pub num_decoded_one_bit_error: u64,
    /// Total number of successfully decoded packets with two bit errors.
    pub num_decoded_two_bit_errors: u64,
    /// Total number of successfully decoded packets that used noncoherent
    /// demodulation.
    pub num_decoded_noncoherent: u64,
    /// Probability of incorrectly decoding a packet.
    ///
    /// This counts the probability that a packet is claimed to be successfully
    /// decoded but the decoded message does not match the simulated message.
    pub decode_wrong_probability: f64,
    /// Probability of incorrectly decoding a packet with zero bit errors.
    pub decode_wrong_probability_zero_bit_errors: f64,
    /// Probability of incorrectly decoding a packet with exactly one bit error.
    pub decode_wrong_probability_one_bit_error: f64,
    /// Probability of incorrectly decoding a packet with exactly two bit
    /// errors.
    pub decode_wrong_probability_two_bit_errors: f64,
    /// Probability that an incorrect decode used noncoherent demodulation.
    pub decode_wrong_probability_noncoherent: f64,
    /// Total number of incorrectly decoded packets.
    pub num_decoded_wrong: u64,
    /// Total number of incorrectly decoded packets with zero bit errors.
    pub num_decoded_wrong_zero_bit_errors: u64,
    /// Total number of incorrectly decoded packets with exactly one bit error.
    pub num_decoded_wrong_one_bit_error: u64,
    /// Total number of incorrectly decoded packets with exactly two bit errors.
    pub num_decoded_wrong_two_bit_errors: u64,
    /// Total number of incorrectly decoded packets that used noncoherent
    /// demodulation.
    pub num_decoded_wrong_noncoherent: u64,
    /// Maximum error of carrier frequency offset estimate in Hz.
    pub max_frequency_error: f64,
    /// RMS error of carrier frequency offset estimate in Hz.
    pub rms_frequency_error: f64,
    /// Maximum detection time estimate error in samples.
    pub max_detection_time_error: f64,
    /// Mean of detection time estimate error in samples.
    pub avg_detection_time_error: f64,
    /// Standard deviation of detection time estimate error in samples.
    pub std_detection_time_error: f64,
    /// Maximum decode time estimate error in samples.
    ///
    /// This only considers correctly decoded packets. Since incorrectly decoded
    /// packets often have very large time errors because the incorrect decode
    /// corresponds to a wrong message start determination.
    pub max_decode_time_error: f64,
    /// Mean of decode time estimate error in samples.
    ///
    /// This only considers correctly decoded packets. Since incorrectly decoded
    /// packets often have very large time errors because the incorrect decode
    /// corresponds to a wrong message start determination.
    pub avg_decode_time_error: f64,
    /// Standard deviation of decode time estimate error in samples.
    ///
    /// This only considers correctly decoded packets. Since incorrectly decoded
    /// packets often have very large time errors because the incorrect decode
    /// corresponds to a wrong message start determination.
    pub std_decode_time_error: f64,
    /// Maximum CN0 estimate error in dB.
    ///
    /// This only considers correctly decoded packets.
    pub max_cn0_error: f64,
    /// Mean of CN0 estimate error in dB.
    ///
    /// The mean is computed for the estimate in dB·Hz units, not in linear
    /// units. This only consideres correctly decoded packets.
    pub avg_cn0_error: f64,
    /// Standard deviation of CN0 estimate error in dB.
    ///
    /// The standard deviation is computed for the estimate in dB·Hz units, not
    /// in linear units. This only consideres correctly decoded packets.
    pub std_cn0_error: f64,
    /// Total number of packets tested.
    ///
    /// This can differ from the `num_packets` in the configuration due to
    /// worker threads stopping slightly after the end of the test is reached.
    pub num_packets: u64,
    /// Real time benchmark execution duration.
    pub real_time_duration: Duration,
    /// Total throughput in samples/s.
    pub throughput_sps: f64,
    /// Per thread throughput in samples/s.
    pub thread_throughput_sps: f64,
}

macro_rules! fmt_results_common {
    ($self:expr, $f:expr) => {
        writeln!($f, "Total detections: {}", $self.num_detections)?;
        writeln!($f, "Decode probability: {:.3}", $self.decode_probability)?;
        writeln!(
            $f,
            "Decode probability (0 bit errors): {:.3}",
            $self.decode_probability_zero_bit_errors
        )?;
        writeln!(
            $f,
            "Decode probability (1 bit error): {:.3}",
            $self.decode_probability_one_bit_error
        )?;
        writeln!(
            $f,
            "Decode probability (2 bit errors): {:.3}",
            $self.decode_probability_two_bit_errors
        )?;
        writeln!(
            $f,
            "Noncoherent demodulation probability: {:.3}",
            $self.decode_probability_noncoherent
        )?;
        writeln!($f, "Total decoded: {}", $self.num_decoded)?;
        writeln!(
            $f,
            "Total decoded (0 bit errors): {}",
            $self.num_decoded_zero_bit_errors
        )?;
        writeln!(
            $f,
            "Total decoded (1 bit error): {}",
            $self.num_decoded_one_bit_error
        )?;
        writeln!(
            $f,
            "Total decoded (2 bit errors): {}",
            $self.num_decoded_two_bit_errors
        )?;
        writeln!(
            $f,
            "Total decoded using noncoherent demodulation: {}",
            $self.num_decoded_noncoherent
        )?;
        writeln!(
            $f,
            "Wrong decode probability: {:.3}",
            $self.decode_wrong_probability
        )?;
        writeln!(
            $f,
            "Wrong decode probability (0 bit errors): {:.3}",
            $self.decode_wrong_probability_zero_bit_errors
        )?;
        writeln!(
            $f,
            "Wrong decode probability (1 bit error): {:.3}",
            $self.decode_wrong_probability_one_bit_error
        )?;
        writeln!(
            $f,
            "Wrong decode probability (2 bit errors): {:.3}",
            $self.decode_wrong_probability_two_bit_errors
        )?;
        writeln!(
            $f,
            "Wrong decode probability (noncoherent demodulation): {:.3}",
            $self.decode_wrong_probability_noncoherent
        )?;
        writeln!($f, "Total wrong decoded: {}", $self.num_decoded_wrong)?;
        writeln!(
            $f,
            "Total wrong decoded (0 bit errors): {}",
            $self.num_decoded_wrong_zero_bit_errors
        )?;
        writeln!(
            $f,
            "Total wrong decoded (1 bit error): {}",
            $self.num_decoded_wrong_one_bit_error
        )?;
        writeln!(
            $f,
            "Total wrong decoded (2 bit errors): {}",
            $self.num_decoded_wrong_two_bit_errors
        )?;
        writeln!(
            $f,
            "Total wrong decoded (noncoherent demodulation): {}",
            $self.num_decoded_wrong_noncoherent
        )?;
        writeln!($f, "Number of packets: {}", $self.num_packets)?;
        writeln!(
            $f,
            "Maximum frequency error: {:.3} Hz",
            $self.max_frequency_error
        )?;
        writeln!(
            $f,
            "RMS frequency error: {:.3} Hz",
            $self.rms_frequency_error
        )?;
        writeln!(
            $f,
            "Maximum detection time: {:.3} samples",
            $self.max_detection_time_error
        )?;
        writeln!(
            $f,
            "Mean of detection time error: {:.3} samples",
            $self.avg_detection_time_error
        )?;
        writeln!(
            $f,
            "Standard deviation of detection time error: {:.3} samples",
            $self.std_detection_time_error
        )?;
        writeln!(
            $f,
            "Maximum decode time: {:.3} samples",
            $self.max_decode_time_error
        )?;
        writeln!(
            $f,
            "Mean of decode time error: {:.3} samples",
            $self.avg_decode_time_error
        )?;
        writeln!(
            $f,
            "Standard deviation of decode time error: {:.3} samples",
            $self.std_decode_time_error
        )?;
        writeln!(
            $f,
            "Maximum CN0 estimate error: {:.3} dB",
            $self.max_cn0_error
        )?;
        writeln!($f, "Mean CN0 estimate error: {:.3} dB", $self.avg_cn0_error)?;
        writeln!(
            $f,
            "Standard deviation of CN0 estimate error: {:.3} dB",
            $self.std_cn0_error
        )?;
        writeln!(
            $f,
            "Real-time duration: {:.3} s",
            $self.real_time_duration.as_secs_f64()
        )?;
        writeln!(
            $f,
            "Aggregate throughput: {:.3} Msps",
            1e-6 * $self.throughput_sps
        )?;
        writeln!(
            $f,
            "Per-thread throughput: {:.3} Msps",
            1e-6 * $self.thread_throughput_sps
        )?;
    };
}

impl std::fmt::Display for DecodingBenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        writeln!(
            f,
            "Detection probability: {:.3}",
            self.detection_probability
        )?;
        fmt_results_common!(self, f);
        Ok(())
    }
}

impl Default for DecodingBenchmark {
    fn default() -> DecodingBenchmark {
        DecodingBenchmark {
            num_packets: 100_000,
            cn0: 75.0,
            receiver_args: ADSBReceiverArgs::default(),
            random_seed: 0,
            num_threads: default_num_threads(),
        }
    }
}

#[derive(Debug)]
struct DecodingRunner {
    packets_per_thread: u64,
    cn0: CN0Distribution,
    receiver_args: ADSBReceiverArgs,
    num_threads: usize,
}

macro_rules! calculate_results {
    ($worker_results:expr, $packets:expr, $detected:expr, $struct:ident { $($field:tt)* } ) => {
        $struct {
            num_detections: $worker_results.num_detections,
            decode_probability: $worker_results.num_ok_decodes() as f64 / $packets as f64,
            decode_probability_zero_bit_errors: $worker_results.num_ok_decodes_zero_errors as f64
                / $packets as f64,
            decode_probability_one_bit_error: $worker_results.num_ok_decodes_one_error as f64
                / $packets as f64,
            decode_probability_two_bit_errors: $worker_results.num_ok_decodes_two_errors as f64
                / $packets as f64,
            decode_probability_noncoherent: $worker_results.num_ok_decodes_noncoherent as f64
                / $worker_results.num_ok_decodes() as f64,
            num_decoded: $worker_results.num_ok_decodes(),
            num_decoded_zero_bit_errors: $worker_results.num_ok_decodes_zero_errors,
            num_decoded_one_bit_error: $worker_results.num_ok_decodes_one_error,
            num_decoded_two_bit_errors: $worker_results.num_ok_decodes_two_errors,
            num_decoded_noncoherent: $worker_results.num_ok_decodes_noncoherent,
            decode_wrong_probability: $worker_results.num_wrong_decodes() as f64
                / $packets as f64,
            decode_wrong_probability_zero_bit_errors: $worker_results.num_wrong_decodes_zero_errors
                as f64
                / $packets as f64,
            decode_wrong_probability_one_bit_error: $worker_results.num_wrong_decodes_one_error
                as f64
                / $packets as f64,
            decode_wrong_probability_two_bit_errors: $worker_results.num_wrong_decodes_two_errors
                as f64
                / $packets as f64,
            decode_wrong_probability_noncoherent: $worker_results.num_wrong_decodes_noncoherent
                as f64
                / $worker_results.num_wrong_decodes() as f64,
            num_decoded_wrong: $worker_results.num_wrong_decodes(),
            num_decoded_wrong_zero_bit_errors: $worker_results.num_wrong_decodes_zero_errors,
            num_decoded_wrong_one_bit_error: $worker_results.num_wrong_decodes_one_error,
            num_decoded_wrong_two_bit_errors: $worker_results.num_wrong_decodes_two_errors,
            num_decoded_wrong_noncoherent: $worker_results.num_wrong_decodes_noncoherent,
            max_frequency_error: $worker_results.max_freq_error,
            rms_frequency_error: ($worker_results.freq_error_sqr_sum
                                  / $detected as f64)
                .sqrt(),
            max_detection_time_error: $worker_results.max_detection_time_error,
            avg_detection_time_error: $worker_results.detection_time_error_sum
                / $detected as f64,
            std_detection_time_error: standard_deviation(
                $worker_results.detection_time_error_sqr_sum,
                $worker_results.detection_time_error_sum,
                $detected,
            ),
            max_decode_time_error: $worker_results.max_decode_time_error,
            avg_decode_time_error: $worker_results.decode_time_error_sum
                / $worker_results.num_ok_decodes() as f64,
            std_decode_time_error: standard_deviation(
                $worker_results.decode_time_error_sqr_sum,
                $worker_results.decode_time_error_sum,
                $worker_results.num_ok_decodes(),
            ),
            max_cn0_error: $worker_results.max_cn0_error,
            avg_cn0_error: $worker_results.cn0_error_sum
                / $worker_results.num_ok_decodes() as f64,
            std_cn0_error: standard_deviation(
                $worker_results.cn0_error_sqr_sum,
                $worker_results.cn0_error_sum,
                $worker_results.num_ok_decodes(),
            ),
            $($field)*
        }
    }
}

impl DecodingRunner {
    fn new(benchmark: &DecodingBenchmark) -> Result<DecodingRunner> {
        anyhow::ensure!(
            benchmark.num_packets > 0,
            "the number of packets cannot be zero"
        );
        benchmark.receiver_args.validate()?;
        let cn0 = CN0Distribution::Constant(ConstantCN0::new(benchmark.cn0)?);
        Ok(DecodingRunner {
            packets_per_thread: benchmark
                .num_packets
                .div_ceil(u64::try_from(benchmark.num_threads.get()).unwrap()),
            cn0,
            receiver_args: benchmark.receiver_args.clone(),
            num_threads: benchmark.num_threads.get(),
        })
    }

    const NUM_SAMPLES: u64 = 8320; // 1.04 ms

    fn packet_start<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let message_samples = 2 * TOTAL_MODE_S_LONG_REPLY_BITS * SPS;
        let nominal_start = (Self::NUM_SAMPLES as f64 - message_samples as f64) * 0.5;
        let max_start_error = message_samples as f64;
        let start_error = rng.random_range(-max_start_error..max_start_error);
        nominal_start + start_error
    }

    fn run(&self, random_seed: u64) -> DecodingWorkerResults {
        let mut rng = rng_from_seed(random_seed);
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(self.num_threads);
            for _ in 0..self.num_threads {
                let thread_seed = rng.next_u64();
                handles.push(s.spawn(move || self.run_thread(thread_seed)));
            }
            let mut results = DecodingWorkerResults::new();
            for handle in handles {
                let worker_results = handle.join().expect("runner thread panicked");
                results.combine(&worker_results)
            }
            results
        })
    }

    fn run_thread(&self, thread_seed: u64) -> DecodingWorkerResults {
        let mut results = DecodingWorkerResults::new();
        let mut rng = rng_from_seed(thread_seed);
        let mut receiver = self.receiver_args.new_receiver().unwrap();
        let mut receiver_time_offset: u64 = 0;
        for _ in 0..self.packets_per_thread {
            let simulation = self.create_simulation(&mut rng);
            let snippet = simulation.run(&mut rng);
            assert!(snippet.len().is_multiple_of(CHUNK_LEN));
            for chunk in snippet.as_chunks::<CHUNK_LEN>().0 {
                let result = receiver.process_chunk(chunk);
                results.update(result, &simulation.message_config, receiver_time_offset);
            }
            receiver_time_offset += u64::try_from(snippet.len()).unwrap();
        }
        results
    }

    fn create_simulation<R: Rng + ?Sized>(&self, rng: &mut R) -> SingleMessageSimulation {
        SingleMessageSimulation {
            duration: samples_as_duration(Self::NUM_SAMPLES),
            message_config: MessageConfig::from_distributions(
                self.packet_start(rng),
                MessageDistributions {
                    cn0: self.cn0,
                    ..Default::default()
                },
                MessageType::ExtendedSquitter,
                rng,
            ),
        }
    }

    fn results(
        &self,
        worker_results: &DecodingWorkerResults,
        elapsed: Duration,
    ) -> DecodingBenchmarkResults {
        let packets_done = self
            .packets_per_thread
            .checked_mul(u64::try_from(self.num_threads).unwrap())
            .unwrap();
        let samples_done = packets_done * Self::NUM_SAMPLES;
        let throughput_sps = samples_done as f64 / elapsed.as_secs_f64();
        calculate_results!(
            worker_results,
            packets_done,
            worker_results.num_detections,
            DecodingBenchmarkResults {
                detection_probability: worker_results.num_detections as f64 / packets_done as f64,
                num_packets: packets_done,
                real_time_duration: elapsed,
                throughput_sps,
                thread_throughput_sps: throughput_sps / self.num_threads as f64,
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DecodingWorkerResults {
    num_detections: u64,
    num_ok_decodes_zero_errors: u64,
    num_ok_decodes_one_error: u64,
    num_ok_decodes_two_errors: u64,
    num_ok_decodes_noncoherent: u64,
    num_wrong_decodes_zero_errors: u64,
    num_wrong_decodes_one_error: u64,
    num_wrong_decodes_two_errors: u64,
    num_wrong_decodes_noncoherent: u64,
    max_freq_error: f64,
    freq_error_sqr_sum: f64,
    max_detection_time_error: f64,
    detection_time_error_sum: f64,
    detection_time_error_sqr_sum: f64,
    max_decode_time_error: f64,
    decode_time_error_sum: f64,
    decode_time_error_sqr_sum: f64,
    max_cn0_error: f64,
    cn0_error_sum: f64,
    cn0_error_sqr_sum: f64,
}

impl DecodingWorkerResults {
    fn new() -> DecodingWorkerResults {
        DecodingWorkerResults {
            num_detections: 0,
            num_ok_decodes_zero_errors: 0,
            num_ok_decodes_one_error: 0,
            num_ok_decodes_two_errors: 0,
            num_ok_decodes_noncoherent: 0,
            num_wrong_decodes_zero_errors: 0,
            num_wrong_decodes_one_error: 0,
            num_wrong_decodes_two_errors: 0,
            num_wrong_decodes_noncoherent: 0,
            max_freq_error: 0.0,
            freq_error_sqr_sum: 0.0,
            max_detection_time_error: 0.0,
            detection_time_error_sum: 0.0,
            detection_time_error_sqr_sum: 0.0,
            max_decode_time_error: 0.0,
            decode_time_error_sum: 0.0,
            decode_time_error_sqr_sum: 0.0,
            max_cn0_error: 0.0,
            cn0_error_sum: 0.0,
            cn0_error_sqr_sum: 0.0,
        }
    }

    fn num_ok_decodes(&self) -> u64 {
        self.num_ok_decodes_zero_errors
            + self.num_ok_decodes_one_error
            + self.num_ok_decodes_two_errors
    }

    fn num_wrong_decodes(&self) -> u64 {
        self.num_wrong_decodes_zero_errors
            + self.num_wrong_decodes_one_error
            + self.num_wrong_decodes_two_errors
    }

    fn combine(&mut self, other: &DecodingWorkerResults) {
        self.num_detections += other.num_detections;
        self.num_ok_decodes_zero_errors += other.num_ok_decodes_zero_errors;
        self.num_ok_decodes_one_error += other.num_ok_decodes_one_error;
        self.num_ok_decodes_two_errors += other.num_ok_decodes_two_errors;
        self.num_ok_decodes_noncoherent += other.num_ok_decodes_noncoherent;
        self.num_wrong_decodes_zero_errors += other.num_wrong_decodes_zero_errors;
        self.num_wrong_decodes_one_error += other.num_wrong_decodes_one_error;
        self.num_wrong_decodes_two_errors += other.num_wrong_decodes_two_errors;
        self.num_wrong_decodes_noncoherent += other.num_wrong_decodes_noncoherent;
        self.max_freq_error = self.max_freq_error.max(other.max_freq_error);
        self.freq_error_sqr_sum += other.freq_error_sqr_sum;
        self.max_detection_time_error = self
            .max_detection_time_error
            .max(other.max_detection_time_error);
        self.detection_time_error_sum += other.detection_time_error_sum;
        self.detection_time_error_sqr_sum += other.detection_time_error_sqr_sum;
        self.max_decode_time_error = self.max_decode_time_error.max(other.max_decode_time_error);
        self.decode_time_error_sum += other.decode_time_error_sum;
        self.decode_time_error_sqr_sum += other.decode_time_error_sqr_sum;
        self.max_cn0_error = self.max_cn0_error.max(other.max_cn0_error);
        self.cn0_error_sum += other.cn0_error_sum;
        self.cn0_error_sqr_sum += other.cn0_error_sqr_sum;
    }

    fn update(
        &mut self,
        receiver_result: Option<&ADSBPacket>,
        message_config: &MessageConfig,
        receiver_time_offset: u64,
    ) {
        let Some(packet) = receiver_result else {
            return;
        };
        self.num_detections += 1;
        let freq_error =
            packet.detection.carrier_frequency_offset - message_config.carrier_frequency_offset;
        self.max_freq_error = self.max_freq_error.max(freq_error.abs());
        self.freq_error_sqr_sum += freq_error * freq_error;

        let detection_time_error = (i64::try_from(packet.detection.sample_index).unwrap()
            - i64::try_from(receiver_time_offset).unwrap())
            as f64
            - simulation_message_start(message_config);
        self.max_detection_time_error = self
            .max_detection_time_error
            .max(detection_time_error.abs());
        self.detection_time_error_sum += detection_time_error;
        self.detection_time_error_sqr_sum += detection_time_error * detection_time_error;

        if let Some(decode) = packet.decode.as_ref() {
            self.update_decode(decode, message_config, receiver_time_offset);
        }
    }

    fn update_decode(
        &mut self,
        decode: &ADSBDecode,
        message_config: &MessageConfig,
        receiver_time_offset: u64,
    ) {
        let SSRReply::ModeS(ModeSMessage::Long(expected_message)) = &message_config.message else {
            panic!("unexpected short mode-s message");
        };
        let decode_correct = &decode.message == expected_message;

        if decode_correct {
            self.update_correct_decode(decode, message_config, receiver_time_offset);
        } else {
            self.update_false_decode(decode);
        }
    }

    fn update_correct_decode(
        &mut self,
        decode: &ADSBDecode,
        message_config: &MessageConfig,
        receiver_time_offset: u64,
    ) {
        let decode_time_error = (i64::try_from(decode.sample_index).unwrap()
            - i64::try_from(receiver_time_offset).unwrap()) as f64
            + f64::from(decode.sample_index_frac)
            - simulation_message_start(message_config);
        self.max_decode_time_error = self.max_decode_time_error.max(decode_time_error.abs());
        self.decode_time_error_sum += decode_time_error;
        self.decode_time_error_sqr_sum += decode_time_error * decode_time_error;

        let cn0_error = f64::from(decode.cn0_db) - message_config.cn0_db;
        self.max_cn0_error = self.max_cn0_error.max(cn0_error.abs());
        self.cn0_error_sum += cn0_error;
        self.cn0_error_sqr_sum += cn0_error * cn0_error;

        match decode.corrected_errors {
            BitErrors::NoErrors => self.num_ok_decodes_zero_errors += 1,
            BitErrors::SingleError(_) => self.num_ok_decodes_one_error += 1,
            BitErrors::DoubleError(_, _) => self.num_ok_decodes_two_errors += 1,
        }
        if decode.demodulation == DemodulationType::NonCoherent {
            self.num_ok_decodes_noncoherent += 1;
        }
    }

    fn update_false_decode(&mut self, decode: &ADSBDecode) {
        match decode.corrected_errors {
            BitErrors::NoErrors => self.num_wrong_decodes_zero_errors += 1,
            BitErrors::SingleError(_) => self.num_wrong_decodes_one_error += 1,
            BitErrors::DoubleError(_, _) => self.num_wrong_decodes_two_errors += 1,
        }
        if decode.demodulation == DemodulationType::NonCoherent {
            self.num_wrong_decodes_noncoherent += 1;
        }
    }
}

impl Default for DecodingWorkerResults {
    fn default() -> DecodingWorkerResults {
        DecodingWorkerResults::new()
    }
}

fn simulation_message_start(message_config: &MessageConfig) -> f64 {
    // Correct preamble duration according to the symbol clock frequency error
    let reply_preamble_duration =
        REPLY_PREAMBLE_SAMPLES as f64 / (1.0 + message_config.symbol_clock_error);
    message_config.timestamp_samples + reply_preamble_duration
}

fn standard_deviation(sum_squares: f64, sum: f64, num_samples: u64) -> f64 {
    if num_samples <= 1 {
        return f64::NAN;
    }
    ((sum_squares - sum * sum / num_samples as f64) / (num_samples - 1) as f64).sqrt()
}

impl DecodingBenchmark {
    /// Runs the decoding benchmark.
    ///
    /// This function runs until the number of packets defined in the
    /// configuration is reached, and returns the results.
    pub fn run(&self) -> Result<DecodingBenchmarkResults> {
        let runner = DecodingRunner::new(self)?;
        let start = Instant::now();
        let results = runner.run(self.random_seed);
        let elapsed = start.elapsed();
        Ok(runner.results(&results, elapsed))
    }
}

/// Scenario benchmark.
///
/// This structure defines the configuration of a scenario benchmark, which runs
/// the receiver on a simulation scenario containing many Mode-S messages in
/// AWGN. The benchmark counts the number of successful decodes. The benchmark
/// results, except for the througput, depend deterministically on all these
/// parameters, including the number of threads (if the benchmark is run with
/// the same parameters except for the number of threads, the results will be
/// different but close).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioBenchmark {
    /// Total benchmark scenario duration.
    ///
    /// This duration is divided evenly into the threads that run the
    /// benchmark. Each thread runs an independent scenario.
    pub duration: Duration,
    /// Average total number of messages per second.
    pub messages_per_second: f64,
    /// Maximum CN0 in dB·Hz.
    pub max_cn0: f64,
    /// Minimum distance to transmitters in km.
    pub min_distance: f64,
    /// Maximum distance to transmitters in km.
    pub max_distance: f64,
    /// ADS-B receiver arguments.
    pub receiver_args: ADSBReceiverArgs,
    /// Random seed.
    pub random_seed: u64,
    /// Number of threads to use for running the benchmark.
    pub num_threads: NonZero<usize>,
}

/// Scenario benchmark results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScenarioBenchmarkResults {
    /// Total number of detections.
    ///
    /// This is generally larger than the total number of simulated ADS-B
    /// packets due to other Mode-S packets also triggering detections.
    pub num_detections: u64,
    /// Probability of successfully decoding an ADS-B packet.
    pub decode_probability: f64,
    /// Probability of successfully decoding an ADS-B packet with zero bit errors.
    pub decode_probability_zero_bit_errors: f64,
    /// Probability of successfully decoding an ADS-B packet with exactly one bit
    /// error.
    pub decode_probability_one_bit_error: f64,
    /// Probability of successfully decoding an ADS-B packet with exactly two bit
    /// errors.
    pub decode_probability_two_bit_errors: f64,
    /// Probability that a successful decoded used noncoherent demodulation.
    pub decode_probability_noncoherent: f64,
    /// Total number of successfully decoded ADS-B packets.
    pub num_decoded: u64,
    /// Total number of successfully decoded ADS-B packets with zero bit errors.
    pub num_decoded_zero_bit_errors: u64,
    /// Total number of successfully decoded ADS-B packets with one bit error.
    pub num_decoded_one_bit_error: u64,
    /// Total number of successfully decoded ADS-B packets with two bit errors.
    pub num_decoded_two_bit_errors: u64,
    /// Total number of successfully decoded ADS-B packets that used noncoherent
    /// demodulation.
    pub num_decoded_noncoherent: u64,
    /// Probability of incorrectly decoding an ADS-B packet.
    ///
    /// This counts the probability that a packet is claimed to be successfully
    /// decoded but the decoded message does not match a simulated message.
    pub decode_wrong_probability: f64,
    /// Probability of incorrectly decoding an ADS-B packet with zero bit errors.
    pub decode_wrong_probability_zero_bit_errors: f64,
    /// Probability of incorrectly decoding an ADS-B packet with exactly one bit error.
    pub decode_wrong_probability_one_bit_error: f64,
    /// Probability of incorrectly decoding an ADS-B packet with exactly two bit
    /// errors.
    pub decode_wrong_probability_two_bit_errors: f64,
    /// Probability that an incorrect decode used noncoherent demodulation.
    pub decode_wrong_probability_noncoherent: f64,
    /// Total number of incorrectly decoded ADS-B packets.
    pub num_decoded_wrong: u64,
    /// Total number of incorrectly decoded ADS-B packets with zero bit errors.
    pub num_decoded_wrong_zero_bit_errors: u64,
    /// Total number of incorrectly decoded ADS-B packets with exactly one bit error.
    pub num_decoded_wrong_one_bit_error: u64,
    /// Total number of incorrectly decoded ADS-B packets with exactly two bit errors.
    pub num_decoded_wrong_two_bit_errors: u64,
    /// Total number of incorrectly decoded ADS-B packets that used noncoherent
    /// demodulation.
    pub num_decoded_wrong_noncoherent: u64,
    /// Maximum error of carrier frequency offset estimate in Hz.
    ///
    /// This only considers correctly decoded packets.
    pub max_frequency_error: f64,
    /// RMS error of carrier frequency offset estimate in Hz.
    ///
    /// This only considers correctly decoded packets.
    pub rms_frequency_error: f64,
    /// Maximum detection time estimate error in samples.
    ///
    /// This only considers correctly decoded packets.
    pub max_detection_time_error: f64,
    /// Mean of detection time estimate error in samples.
    ///
    /// This only considers correctly decoded packets.
    pub avg_detection_time_error: f64,
    /// Standard deviation of detection time estimate error in samples.
    ///
    /// This only considers correctly decoded packets.    
    pub std_detection_time_error: f64,
    /// Maximum decode time estimate error in samples.
    ///
    /// This only considers correctly decoded packets.
    pub max_decode_time_error: f64,
    /// Mean of decode time estimate error in samples.
    ///
    /// This only considers correctly decoded packets.
    pub avg_decode_time_error: f64,
    /// Standard deviation of decode time estimate error in samples.
    ///
    /// This only considers correctly decoded packets.
    pub std_decode_time_error: f64,
    /// Maximum CN0 estimate error in dB.
    ///
    /// This only considers correctly decoded packets.
    pub max_cn0_error: f64,
    /// Mean of CN0 estimate error in dB.
    ///
    /// The mean is computed for the estimate in dB·Hz units, not in linear
    /// units. This only considers correctly decoded packets.
    pub avg_cn0_error: f64,
    /// Standard deviation of CN0 estimate error in dB.
    ///
    /// The standard deviation is computed for the estimate in dB·Hz units, not
    /// in linear units. This only considers correctly decoded packets.
    pub std_cn0_error: f64,
    /// Total number of ADS-B packets in the simulation.
    pub num_packets: u64,
    /// Total number of ADS-B packets in the simulation that have collisions
    /// with other messages in the simulation.
    pub num_packets_collision: u64,
    /// Probability that an ADS-B packet in the simulation has a collision.
    pub collision_probability: f64,
    /// Real time benchmark execution duration.
    pub real_time_duration: Duration,
    /// Total throughput in samples/s.
    pub throughput_sps: f64,
    /// Per thread throughput in samples/s.
    pub thread_throughput_sps: f64,
}

impl std::fmt::Display for ScenarioBenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        writeln!(f, "ADS-B packets: {}", self.num_packets)?;
        writeln!(
            f,
            "ADS-B packets with collision: {}",
            self.num_packets_collision,
        )?;
        writeln!(
            f,
            "ADS-B packet collision probability: {:.3}",
            self.collision_probability,
        )?;
        fmt_results_common!(self, f);
        Ok(())
    }
}

impl Default for ScenarioBenchmark {
    fn default() -> ScenarioBenchmark {
        ScenarioBenchmark {
            duration: Duration::from_secs(1000),
            messages_per_second: 2000.0,
            max_cn0: 105.0,
            min_distance: 10.0,
            max_distance: 500.0,
            receiver_args: ADSBReceiverArgs::default(),
            random_seed: 0,
            num_threads: default_num_threads(),
        }
    }
}

#[derive(Debug)]
struct ScenarioRunner {
    simulation: Simulation,
    receiver_args: ADSBReceiverArgs,
    num_threads: usize,
}

#[derive(Debug)]
struct ScenarioWorker {
    receiver: ADSBReceiver,
    results: ScenarioWorkerResults,
    previous_messages: Vec<MessageConfig>,
    current_messages: Vec<MessageConfig>,
    previous_messages_max_end_time: f64,
    adsb_noncolliding_candidate_end_time: Option<f64>,
}

#[derive(Debug)]
struct ScenarioWorkerWriter {
    worker: Rc<RefCell<ScenarioWorker>>,
}

impl ScenarioRunner {
    fn new(benchmark: &ScenarioBenchmark) -> Result<ScenarioRunner> {
        anyhow::ensure!(
            benchmark.duration > Duration::ZERO,
            "scenario duration cannot be zero"
        );
        benchmark.receiver_args.validate()?;
        let num_threads = benchmark.num_threads.get();
        // we need at least 1 sample to prevent samples_per_thread from being
        // zero
        let total_samples = duration_as_samples(benchmark.duration).max(1);
        // ensure that each thread has an integer number of CHUNK_LEN blocks
        let total_samples = total_samples
            .next_multiple_of(u64::try_from(num_threads.checked_mul(CHUNK_LEN).unwrap()).unwrap());
        let samples_per_thread = total_samples / u64::try_from(num_threads).unwrap();
        let duration_per_thread = samples_as_duration(samples_per_thread);
        let simulation = Simulation {
            duration: duration_per_thread,
            messages_per_second: MessagesPerSecond::from_total_messages_per_second(
                benchmark.messages_per_second,
            )?,
            message_distributions: MessageDistributions {
                cn0: CN0Distribution::DistanceRange(DistanceRangeCN0::new(
                    benchmark.max_cn0,
                    benchmark.min_distance..benchmark.max_distance,
                )?),
                carrier_frequency_offset: Default::default(),
                carrier_phase: Default::default(),
                symbol_clock_error: Default::default(),
            },
            // The batch size must be a multiple of CHUNK_LEN, and it must be
            // large enough so that a message decoded from IQ data of the
            // current batch has been output via the message_config callback
            // after the previous IQ batch was output. Within these constraints,
            // we prefer to keep the batch size small to reduce the number of
            // messages that we need to store and search over.
            batch_size: 80_000, // 10 ms
        };
        Ok(ScenarioRunner {
            simulation,
            receiver_args: benchmark.receiver_args.clone(),
            num_threads,
        })
    }

    fn run(&self, random_seed: u64) -> ScenarioWorkerResults {
        let mut rng = rng_from_seed(random_seed);
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(self.num_threads);
            for _ in 0..self.num_threads {
                let thread_seed = rng.next_u64();
                handles.push(s.spawn(move || self.run_thread(thread_seed)));
            }
            let mut results = ScenarioWorkerResults::new();
            for handle in handles {
                let worker_results = handle.join().expect("runner thread panicked");
                results.combine(&worker_results)
            }
            results
        })
    }

    fn run_thread(&self, thread_seed: u64) -> ScenarioWorkerResults {
        let mut rng = rng_from_seed(thread_seed);
        let worker = Rc::new(RefCell::new(
            ScenarioWorker::from_receiver_args(&self.receiver_args).unwrap(),
        ));
        let mut writer = ScenarioWorkerWriter {
            worker: Rc::clone(&worker),
        };
        self.simulation
            .run(&mut rng, &mut writer, |message_config| {
                worker.borrow_mut().message_callback(message_config)
            })
            .unwrap();
        // Write a batch of AWGN to flush the receiver
        let mut awgn = AVec::with_capacity(0, self.simulation.batch_size);
        for _ in 0..self.simulation.batch_size {
            awgn.push(generate_awgn(&mut rng, AWGN_RE_SIGMA));
        }
        writer.write_complex32(&awgn).unwrap();
        // Drop the writer so that there is a single reference to worker and we
        // can extract its inner value.
        drop(writer);
        Rc::into_inner(worker).unwrap().into_inner().results
    }

    fn results(
        &self,
        worker_results: &ScenarioWorkerResults,
        elapsed: Duration,
    ) -> ScenarioBenchmarkResults {
        // The batch_size is added to account for the final AWGN flush of the
        // receiver.
        let samples_per_thread = duration_as_samples(self.simulation.duration)
            + u64::try_from(self.simulation.batch_size).unwrap();
        let total_samples = samples_per_thread
            .checked_mul(u64::try_from(self.num_threads).unwrap())
            .unwrap();
        let throughput_sps = total_samples as f64 / elapsed.as_secs_f64();
        calculate_results!(
            worker_results.results,
            worker_results.num_packets,
            worker_results.results.num_ok_decodes(),
            ScenarioBenchmarkResults {
                num_packets: worker_results.num_packets,
                num_packets_collision: worker_results.num_packets_with_collision,
                collision_probability: worker_results.num_packets_with_collision as f64
                    / worker_results.num_packets as f64,
                real_time_duration: elapsed,
                throughput_sps,
                thread_throughput_sps: throughput_sps / self.num_threads as f64,
            }
        )
    }
}

impl ScenarioWorker {
    fn from_receiver_args(receiver_args: &ADSBReceiverArgs) -> Result<ScenarioWorker> {
        Ok(ScenarioWorker {
            receiver: receiver_args.new_receiver()?,
            results: ScenarioWorkerResults::new(),
            previous_messages: Vec::new(),
            current_messages: Vec::new(),
            previous_messages_max_end_time: 0.0,
            adsb_noncolliding_candidate_end_time: None,
        })
    }

    fn message_callback(&mut self, message_config: &MessageConfig) -> Result<()> {
        if let Some(t) = self.adsb_noncolliding_candidate_end_time {
            // The previous message was an ADS-B message that did not collide
            // with its previous message. Check if it collides with this message.
            if t > message_config.timestamp_samples {
                self.results.num_packets_with_collision += 1;
            }
            self.adsb_noncolliding_candidate_end_time = None;
        }

        let message_type = MessageType::from_message(&message_config.message).unwrap();
        let message_end_time =
            message_config.timestamp_samples + message_num_samples(&message_config.message) as f64;

        if message_type == MessageType::ExtendedSquitter {
            self.results.num_packets += 1;

            if self.previous_messages_max_end_time > message_config.timestamp_samples {
                // This ADS-B message collides with the previous message (the
                // previous message can be ADS-B or a different type)
                self.results.num_packets_with_collision += 1;
            } else {
                // This ADS-B message does not collide with the previous
                // message, but it might collide with the next message (which is
                // checked in the next call to this callback).
                self.adsb_noncolliding_candidate_end_time = Some(message_end_time);
            }

            self.current_messages.push(message_config.clone());
        }

        self.previous_messages_max_end_time =
            self.previous_messages_max_end_time.max(message_end_time);
        Ok(())
    }

    fn process_chunk(&mut self, chunk: &[Complex32; CHUNK_LEN]) {
        let receiver_result = self.receiver.process_chunk(chunk);
        let Some(packet) = receiver_result else {
            return;
        };
        let Some(decode) = &packet.decode else {
            self.results.results.num_detections += 1;
            return;
        };
        let message = &decode.message;
        // Search for a simulated message within the messages of the previous
        // and current batches that matches the decoded message.
        let Some(message_config) = self
            .current_messages
            .iter()
            .chain(self.previous_messages.iter())
            .find(|message_config| {
                let SSRReply::ModeS(this_message) = &message_config.message else {
                    panic!("stored non-Mode-S message in ground truth list");
                };
                this_message.as_slice() == message
            })
        else {
            self.results.results.num_detections += 1;
            self.results.results.update_false_decode(decode);
            return;
        };
        let receiver_time_offset = 0;
        self.results
            .results
            .update(receiver_result, message_config, receiver_time_offset);
    }

    fn advance_messages(&mut self) {
        self.previous_messages.clear();
        std::mem::swap(&mut self.previous_messages, &mut self.current_messages);
    }
}

impl WriteComplex32 for ScenarioWorker {
    fn write_complex32(&mut self, samples: &[Complex32]) -> std::io::Result<()> {
        assert!(samples.len().is_multiple_of(CHUNK_LEN));
        for chunk in samples.as_chunks::<CHUNK_LEN>().0 {
            self.process_chunk(chunk);
        }
        self.advance_messages();
        Ok(())
    }
}

impl WriteComplex32 for ScenarioWorkerWriter {
    fn write_complex32(&mut self, samples: &[Complex32]) -> std::io::Result<()> {
        self.worker.borrow_mut().write_complex32(samples)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ScenarioWorkerResults {
    num_packets: u64,
    num_packets_with_collision: u64,
    results: DecodingWorkerResults,
}

impl ScenarioWorkerResults {
    fn new() -> ScenarioWorkerResults {
        ScenarioWorkerResults {
            num_packets: 0,
            num_packets_with_collision: 0,
            results: DecodingWorkerResults::new(),
        }
    }

    fn combine(&mut self, other: &ScenarioWorkerResults) {
        self.num_packets += other.num_packets;
        self.num_packets_with_collision += other.num_packets_with_collision;
        self.results.combine(&other.results);
    }
}

impl Default for ScenarioWorkerResults {
    fn default() -> ScenarioWorkerResults {
        ScenarioWorkerResults::new()
    }
}

impl ScenarioBenchmark {
    /// Runs the scenario benchmark.
    ///
    /// This function runs until the simulated data for the duration configured
    /// in `self` has been generated and processed, and returns the results.
    pub fn run(&self) -> Result<ScenarioBenchmarkResults> {
        let runner = ScenarioRunner::new(self)?;
        let start = Instant::now();
        let results = runner.run(self.random_seed);
        let elapsed = start.elapsed();
        Ok(runner.results(&results, elapsed))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::constants::dsp::SAMPLE_RATE_USIZE;

    #[test]
    fn false_acquisition_new() {
        let benchmark = FalseAcquisitionBenchmark {
            num_threads: 3.try_into().unwrap(),
            ..Default::default()
        };
        let runner = FalseAcquisitionRunner::new(&benchmark).unwrap();
        assert_eq!(runner.chunks_per_thread, 208_333_334);
        assert_eq!(runner.num_threads, 3);
    }

    #[test]
    fn false_acquisition_results() {
        let benchmark = FalseAcquisitionBenchmark {
            duration: Duration::from_secs(1000),
            num_threads: 2.try_into().unwrap(),
            ..Default::default()
        };
        let runner = FalseAcquisitionRunner::new(&benchmark).unwrap();
        let false_detections = 10;
        let runtime = Duration::from_secs(10);
        let throughput_sps = (1000 / 10 * SAMPLE_RATE_USIZE) as f64;
        let expected = FalseAcquisitionBenchmarkResults {
            false_detections_per_second: 10.0 / 1000.0,
            seconds_per_false_detection: 1000.0 / 10.0,
            false_detections,
            benchmark_duration: Duration::from_secs(1000),
            real_time_duration: runtime,
            throughput_sps,
            thread_throughput_sps: throughput_sps * 0.5,
        };
        assert_eq!(runner.results(false_detections, runtime), expected);
    }

    #[test]
    fn false_acquisition_default_short() {
        let benchmark = FalseAcquisitionBenchmark {
            duration: Duration::from_millis(10),
            num_threads: 3.try_into().unwrap(),
            ..Default::default()
        };
        let results0 = benchmark.run().unwrap();
        let mut results1 = benchmark.run().unwrap();
        // throughput is different, so we copy it before testing for equality of
        // results
        results1.real_time_duration = results0.real_time_duration;
        results1.throughput_sps = results0.throughput_sps;
        results1.thread_throughput_sps = results0.thread_throughput_sps;
        assert_eq!(results0, results1);
    }

    #[test]
    fn decoding_new() {
        let benchmark = DecodingBenchmark {
            num_threads: 3.try_into().unwrap(),
            ..Default::default()
        };
        let runner = DecodingRunner::new(&benchmark).unwrap();
        assert_eq!(runner.packets_per_thread, 33_334);
        let CN0Distribution::Constant(cn0) = runner.cn0 else {
            panic!();
        };
        assert_eq!(cn0.cn0(), 75.0);
        assert_eq!(runner.num_threads, 3);
    }

    #[test]
    fn decoding_zero_packets() {
        let benchmark = DecodingBenchmark {
            num_packets: 0,
            ..Default::default()
        };
        assert!(DecodingRunner::new(&benchmark).is_err());
    }

    #[test]
    fn decoding_simulation_cn0() {
        let benchmark = DecodingBenchmark {
            cn0: 63.5,
            ..Default::default()
        };
        let runner = DecodingRunner::new(&benchmark).unwrap();
        let simulation = runner.create_simulation(&mut rand::rng());
        assert_eq!(simulation.message_config.cn0_db, benchmark.cn0);
    }

    #[test]
    fn decoding_default_short() {
        let benchmark = DecodingBenchmark {
            num_packets: 100,
            num_threads: 3.try_into().unwrap(),
            ..Default::default()
        };
        let mut results0 = benchmark.run().unwrap();
        let mut results1 = benchmark.run().unwrap();
        // throughput is different, so we copy it before testing for equality of
        // results
        results1.real_time_duration = results0.real_time_duration;
        results1.throughput_sps = results0.throughput_sps;
        results1.thread_throughput_sps = results0.thread_throughput_sps;
        // fields that are NaN need to be overwritten with a finite value, or
        // otherwise the comparison check fails
        assert!(results0.decode_wrong_probability_noncoherent.is_nan());
        assert!(results1.decode_wrong_probability_noncoherent.is_nan());
        results0.decode_wrong_probability_noncoherent = 0.0;
        results1.decode_wrong_probability_noncoherent = 0.0;
        assert_eq!(results0, results1);
    }

    #[test]
    fn standard_deviation_invalid() {
        assert!(standard_deviation(123.4, 56.7, 0).is_nan());
        assert!(standard_deviation(123.4, 56.7, 1).is_nan());
    }

    #[test]
    fn standard_deviation_uniform() {
        let mut rng = rand::rng();
        let num_samples = 100_000;
        let mut sum_squares = 0.0;
        let mut sum = 0.0;
        for _ in 0..num_samples {
            let x = rng.random_range(0.0_f64..1.0);
            sum_squares += x * x;
            sum += x;
        }
        let std = standard_deviation(sum_squares, sum, num_samples);
        // The variance of a uniform distribution on [a, b] is (b-a)^2/12.
        let expected = 1.0 / 12.0_f64.sqrt();
        assert!(
            (std - expected) <= 2e-3,
            "std = {std}, expected = {expected}"
        );
    }

    #[test]
    fn scenario_new() {
        let benchmark = ScenarioBenchmark {
            num_threads: 3.try_into().unwrap(),
            ..Default::default()
        };
        let runner = ScenarioRunner::new(&benchmark).unwrap();
        assert!(runner.simulation.duration * 3 >= benchmark.duration);
        assert!(matches!(
            runner.simulation.message_distributions.cn0,
            CN0Distribution::DistanceRange(_)
        ));
        assert_eq!(runner.simulation.batch_size, 80000);
        assert_eq!(runner.num_threads, 3);
    }

    #[test]
    fn scenario_zero_duration() {
        let benchmark = ScenarioBenchmark {
            duration: Duration::ZERO,
            ..Default::default()
        };
        assert!(ScenarioRunner::new(&benchmark).is_err());
    }

    /// Tests that a scenario with a tiny duration still runs successfully.
    ///
    /// This is because the duration is rounded up to an integer number of
    /// blocks per thread.
    #[test]
    fn scenario_tiny_duration() {
        let benchmark = ScenarioBenchmark {
            duration: Duration::from_nanos(1),
            num_threads: 3.try_into().unwrap(),
            ..Default::default()
        };
        let runner = ScenarioRunner::new(&benchmark).unwrap();
        let min_duration = samples_as_duration(u64::try_from(CHUNK_LEN).unwrap());
        assert_eq!(runner.simulation.duration, min_duration);
        let random_seed = 0;
        runner.run(random_seed);
    }

    #[test]
    fn scenario_cn0_range() {
        let benchmark = ScenarioBenchmark {
            max_cn0: 100.23,
            min_distance: 13.5,
            max_distance: 102.7,
            ..Default::default()
        };
        let runner = ScenarioRunner::new(&benchmark).unwrap();
        let CN0Distribution::DistanceRange(cn0) = &runner.simulation.message_distributions.cn0
        else {
            panic!();
        };
        assert_eq!(cn0.max_cn0(), benchmark.max_cn0);
        assert_eq!(cn0.min_distance(), benchmark.min_distance);
        assert_eq!(cn0.max_distance(), benchmark.max_distance);
    }

    fn sample_adsb_message() -> MessageConfig {
        MessageConfig {
            timestamp_samples: 123.4,
            cn0_db: 71.5,
            carrier_frequency_offset: -1023.4,
            carrier_phase: -0.72,
            symbol_clock_error: 500e-9,
            message: SSRReply::ModeS(ModeSMessage::Long(
                const_hex::decode_to_array("8da4c34cea3ca864013c081c17f0").unwrap(),
            )),
        }
    }

    fn sample_non_adsb_message() -> MessageConfig {
        MessageConfig {
            timestamp_samples: 123.4,
            cn0_db: 71.5,
            carrier_frequency_offset: -1023.4,
            carrier_phase: -0.72,
            symbol_clock_error: 500e-9,
            message: SSRReply::ModeS(ModeSMessage::Long(
                const_hex::decode_to_array("a02803352004e179c565a04d5e18").unwrap(),
            )),
        }
    }

    fn new_scenario_worker() -> ScenarioWorker {
        ScenarioWorker::from_receiver_args(&ADSBReceiverArgs::default()).unwrap()
    }

    #[test]
    fn scenario_worker_callback_adsb() {
        let mut worker = new_scenario_worker();
        worker.message_callback(&sample_adsb_message()).unwrap();
        assert_eq!(worker.current_messages.len(), 1);
        assert_eq!(worker.current_messages[0], sample_adsb_message());
        assert!(worker.previous_messages.is_empty());
        let end_time = 123.4 + 960.0;
        assert_eq!(worker.previous_messages_max_end_time, end_time);
        assert_eq!(worker.adsb_noncolliding_candidate_end_time, Some(end_time));
        assert_eq!(worker.results.num_packets, 1);
        assert_eq!(worker.results.num_packets_with_collision, 0);
    }

    #[test]
    fn scenario_worker_callback_non_adsb() {
        let mut worker = new_scenario_worker();
        worker.message_callback(&sample_non_adsb_message()).unwrap();
        assert!(worker.current_messages.is_empty());
        assert!(worker.previous_messages.is_empty());
        let end_time = 123.4 + 960.0;
        assert_eq!(worker.previous_messages_max_end_time, end_time);
        assert_eq!(worker.adsb_noncolliding_candidate_end_time, None);
        assert_eq!(worker.results.num_packets, 0);
        assert_eq!(worker.results.num_packets_with_collision, 0);
    }

    #[test]
    fn scenario_worker_callback_collision_front() {
        let mut worker = new_scenario_worker();
        let mut message0 = sample_non_adsb_message();
        message0.timestamp_samples = 87.23;
        worker.message_callback(&message0).unwrap();
        assert!(worker.current_messages.is_empty());
        assert!(worker.previous_messages.is_empty());
        let end_time = 87.23 + 960.0;
        assert_eq!(worker.previous_messages_max_end_time, end_time);
        assert_eq!(worker.adsb_noncolliding_candidate_end_time, None);
        assert_eq!(worker.results.num_packets, 0);
        assert_eq!(worker.results.num_packets_with_collision, 0);
        worker.message_callback(&sample_adsb_message()).unwrap();
        assert_eq!(worker.current_messages.len(), 1);
        assert_eq!(worker.current_messages[0], sample_adsb_message());
        assert!(worker.previous_messages.is_empty());
        let end_time = 123.4 + 960.0;
        assert_eq!(worker.previous_messages_max_end_time, end_time);
        assert_eq!(worker.adsb_noncolliding_candidate_end_time, None);
        assert_eq!(worker.results.num_packets, 1);
        assert_eq!(worker.results.num_packets_with_collision, 1);
    }

    #[test]
    fn scenario_worker_callback_collision_back() {
        let mut worker = new_scenario_worker();
        worker.message_callback(&sample_adsb_message()).unwrap();
        assert_eq!(worker.current_messages.len(), 1);
        assert_eq!(worker.current_messages[0], sample_adsb_message());
        assert!(worker.previous_messages.is_empty());
        let end_time = 123.4 + 960.0;
        assert_eq!(worker.previous_messages_max_end_time, end_time);
        assert_eq!(worker.adsb_noncolliding_candidate_end_time, Some(end_time));
        assert_eq!(worker.results.num_packets, 1);
        assert_eq!(worker.results.num_packets_with_collision, 0);
        let mut message1 = sample_non_adsb_message();
        message1.timestamp_samples = 857.9;
        worker.message_callback(&message1).unwrap();
        assert_eq!(worker.current_messages.len(), 1);
        assert_eq!(worker.current_messages[0], sample_adsb_message());
        assert!(worker.previous_messages.is_empty());
        let end_time = 857.9 + 960.0;
        assert_eq!(worker.previous_messages_max_end_time, end_time);
        assert_eq!(worker.adsb_noncolliding_candidate_end_time, None);
        assert_eq!(worker.results.num_packets, 1);
        assert_eq!(worker.results.num_packets_with_collision, 1);
    }

    #[test]
    fn scenario_worker_callback_collision_two_adsb() {
        let mut worker = new_scenario_worker();
        worker.message_callback(&sample_adsb_message()).unwrap();
        assert_eq!(worker.current_messages.len(), 1);
        assert_eq!(worker.current_messages[0], sample_adsb_message());
        assert!(worker.previous_messages.is_empty());
        let end_time = 123.4 + 960.0;
        assert_eq!(worker.previous_messages_max_end_time, end_time);
        assert_eq!(worker.adsb_noncolliding_candidate_end_time, Some(end_time));
        assert_eq!(worker.results.num_packets, 1);
        assert_eq!(worker.results.num_packets_with_collision, 0);
        let mut message1 = sample_adsb_message();
        message1.timestamp_samples = 857.9;
        worker.message_callback(&message1).unwrap();
        assert_eq!(worker.current_messages.len(), 2);
        assert_eq!(worker.current_messages[0], sample_adsb_message());
        assert_eq!(worker.current_messages[1], message1);
        assert!(worker.previous_messages.is_empty());
        let end_time = 857.9 + 960.0;
        assert_eq!(worker.previous_messages_max_end_time, end_time);
        assert_eq!(worker.adsb_noncolliding_candidate_end_time, None);
        assert_eq!(worker.results.num_packets, 2);
        assert_eq!(worker.results.num_packets_with_collision, 2);
    }

    /// Tests that a long message followed by a short message contained fully in
    /// the long message followed by an ADS-B message that overlaps only with
    /// the long message (and not with the short message, which is the previous
    /// message) is counted as a collision.
    #[test]
    fn scenario_worker_callback_collision_nested() {
        let mut worker = new_scenario_worker();
        let mut message0 = sample_non_adsb_message();
        message0.timestamp_samples = 100.0;
        worker.message_callback(&message0).unwrap();
        let mut message1 = sample_non_adsb_message();
        message1.message = SSRReply::ModeS(ModeSMessage::Short(
            const_hex::decode_to_array("02e197b01a4c46").unwrap(),
        ));
        message1.timestamp_samples = 400.0;
        worker.message_callback(&message1).unwrap();
        let mut message2 = sample_adsb_message();
        message2.timestamp_samples = 950.0;
        worker.message_callback(&message2).unwrap();
        assert_eq!(worker.results.num_packets_with_collision, 1);
    }

    #[test]
    fn scenario_worker_callback_collision_train() {
        let train_length = 20;
        // this is smaller than the message duration
        let train_spacing = 400.273;
        let mut worker = new_scenario_worker();
        for j in 0..train_length {
            let mut message = sample_adsb_message();
            message.timestamp_samples += j as f64 * train_spacing;
            worker.message_callback(&message).unwrap();
        }
        assert_eq!(worker.results.num_packets, train_length);
        assert_eq!(worker.results.num_packets_with_collision, train_length);
    }

    #[test]
    fn scenario_worker_callback_noncollision_train() {
        let train_length = 20;
        // this is larger than the message duration
        let train_spacing = 981.21;
        let mut worker = new_scenario_worker();
        for j in 0..train_length {
            let mut message = sample_adsb_message();
            message.timestamp_samples += j as f64 * train_spacing;
            worker.message_callback(&message).unwrap();
        }
        assert_eq!(worker.results.num_packets, train_length);
        assert_eq!(worker.results.num_packets_with_collision, 0);
    }

    #[test]
    fn scenario_worker_advance_messages() {
        let mut worker = new_scenario_worker();
        for _ in 0..37 {
            worker.message_callback(&sample_adsb_message()).unwrap();
        }
        assert_eq!(worker.previous_messages.len(), 0);
        assert_eq!(worker.current_messages.len(), 37);
        worker.advance_messages();
        assert_eq!(worker.previous_messages.len(), 37);
        assert_eq!(worker.current_messages.len(), 0);
        for _ in 0..51 {
            worker.message_callback(&sample_adsb_message()).unwrap();
        }
        assert_eq!(worker.previous_messages.len(), 37);
        assert_eq!(worker.current_messages.len(), 51);
        worker.advance_messages();
        assert_eq!(worker.previous_messages.len(), 51);
        assert_eq!(worker.current_messages.len(), 0);
    }

    #[test]
    fn scenario_short_high_cn0() {
        let benchmark = ScenarioBenchmark {
            duration: Duration::from_millis(250),
            messages_per_second: 150.0,
            max_cn0: 120.0,
            min_distance: 10.0,
            max_distance: 100.0,
            // The random seed and number of threads fix the scenario
            // deterministically, so we can ensure a choice that guarantees no
            // packet collisions that would cause missed packets.
            random_seed: 12,
            num_threads: 10.try_into().unwrap(),
            ..Default::default()
        };
        let mut results = benchmark.run().unwrap();
        assert_eq!(results.decode_probability, 1.0);
        assert_eq!(results.decode_probability_zero_bit_errors, 1.0);
        assert_eq!(results.decode_probability_one_bit_error, 0.0);
        assert_eq!(results.decode_probability_two_bit_errors, 0.0);
        assert_eq!(results.decode_probability_noncoherent, 0.0);
        assert!(results.num_packets > 10);
        assert_eq!(results.num_packets, results.num_decoded);
        assert_eq!(results.num_decoded, results.num_decoded_zero_bit_errors);
        assert_eq!(results.num_decoded_one_bit_error, 0);
        assert_eq!(results.num_decoded_two_bit_errors, 0);
        assert_eq!(results.num_decoded_noncoherent, 0);
        assert_eq!(results.decode_wrong_probability, 0.0);
        assert_eq!(results.decode_wrong_probability_zero_bit_errors, 0.0);
        assert_eq!(results.decode_wrong_probability_one_bit_error, 0.0);
        assert_eq!(results.decode_wrong_probability_two_bit_errors, 0.0);
        assert!(results.decode_wrong_probability_noncoherent.is_nan());
        assert_eq!(results.num_decoded_wrong, 0);
        assert_eq!(results.num_decoded_wrong_zero_bit_errors, 0);
        assert_eq!(results.num_decoded_wrong_one_bit_error, 0);
        assert_eq!(results.num_decoded_wrong_two_bit_errors, 0);
        assert_eq!(results.num_decoded_wrong_noncoherent, 0);
        assert_eq!(results.num_packets_collision, 0);
        assert_eq!(results.collision_probability, 0.0);
        // These thresholds are calibrated to the random_seed used
        // above. Another seed might exceed them.
        assert!(results.max_frequency_error < 50.0);
        assert!(results.max_decode_time_error < 0.15);
        assert!(results.max_cn0_error < 4.0);
        let mut results2 = benchmark.run().unwrap();
        // throughput is different, so we copy it before testing for equality of
        // results
        results2.real_time_duration = results.real_time_duration;
        results2.throughput_sps = results.throughput_sps;
        results2.thread_throughput_sps = results.thread_throughput_sps;
        // fields that are NaN need to be overwritten with a finite value, or
        // otherwise the comparison check fails
        assert!(results.decode_wrong_probability_noncoherent.is_nan());
        assert!(results2.decode_wrong_probability_noncoherent.is_nan());
        results.decode_wrong_probability_noncoherent = 0.0;
        results2.decode_wrong_probability_noncoherent = 0.0;
        assert_eq!(results, results2);
    }
}
