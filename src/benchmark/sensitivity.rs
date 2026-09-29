//! Sensitivity benchmarking.
//!
//! This module contains routines to benchmark the sensitivity of the more1090
//! ADS-B receiver.

use crate::{
    ADSBReceiver, CHUNK_LEN,
    constants::{
        dsp::{REPLY_PREAMBLE_SAMPLES, SPS},
        mode_s::TOTAL_MODE_S_LONG_REPLY_BITS,
        simulation::AWGN_RE_SIGMA,
    },
    mode_s::ModeSMessage,
    receiver::{ADSBDecode, ADSBPacket, BitErrors, DemodulationType},
    simulation::{
        MessageConfig, MessageType, SingleMessageSimulation,
        distributions::{CN0Distribution, ConstantCN0, MessageDistributions},
        duration_as_samples, generate_awgn,
        rng::rng_from_seed,
        samples_as_duration,
    },
};
use anyhow::Result;
use num::complex::Complex32;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    num::NonZero,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    thread,
    time::{Duration, Instant},
};

/// False acquisition benchmark.
///
/// This structure defines the configuration of a false acquisition benchmark,
/// which runs the receiver on AWGN input and measures the number of detections
/// per second.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FalseAcquisitionBenchmark {
    /// Maximum benchmark duration.
    ///
    /// This defines the maximum benchmark duration in terms of IQ input data
    /// processed.
    pub max_duration: Duration,
    /// Maximum number of false detections.
    pub max_false_detections: u64,
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
        write!(
            f,
            "Detections per second: {:.e}",
            self.false_detections_per_second
        )?;
        write!(
            f,
            "Seconds per detection: {:.3}",
            1.0 / self.false_detections_per_second
        )?;
        write!(f, "Total detections: {}", self.false_detections)?;
        write!(f, "Benchmark duration: {:?}", self.benchmark_duration)?;
        write!(f, "Real-time duration: {:?}", self.real_time_duration)?;
        write!(
            f,
            "Aggregate throughput: {:.3} Msps",
            1e-6 * self.throughput_sps
        )?;
        write!(
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
            max_duration: Duration::from_secs(10000),
            max_false_detections: 100,
            random_seed: 0,
            num_threads: default_num_threads(),
        }
    }
}

#[derive(Debug)]
struct FalseAcquisitionRunner {
    chunks_done: AtomicU64,
    false_detections: AtomicU64,
    max_false_detections: u64,
    max_chunks: u64,
    num_threads: usize,
}

impl FalseAcquisitionRunner {
    fn new(benchmark: &FalseAcquisitionBenchmark) -> FalseAcquisitionRunner {
        let max_chunks =
            duration_as_samples(benchmark.max_duration) / u64::try_from(CHUNK_LEN).unwrap();
        FalseAcquisitionRunner {
            chunks_done: AtomicU64::new(0),
            false_detections: AtomicU64::new(0),
            max_false_detections: benchmark.max_false_detections,
            max_chunks,
            num_threads: benchmark.num_threads.get(),
        }
    }

    fn run(&self, random_seed: u64) {
        let mut rng = rng_from_seed(random_seed);
        thread::scope(|s| {
            let mut handles = Vec::with_capacity(self.num_threads);
            for _ in 0..self.num_threads {
                let thread_seed = rng.next_u64();
                handles.push(s.spawn(move || self.run_thread(thread_seed)));
            }
            for handle in handles {
                handle.join().expect("runner thread panicked");
            }
        });
    }

    fn run_thread(&self, thread_seed: u64) {
        let mut rng = rng_from_seed(thread_seed);
        let mut receiver = ADSBReceiver::new();
        let mut buf = [Complex32::default(); CHUNK_LEN];
        loop {
            buf.fill_with(|| generate_awgn(&mut rng, AWGN_RE_SIGMA));
            let detected = receiver.process_chunk(&buf).is_some();
            if self.update_counters(detected) {
                break;
            }
        }
    }

    // Returns true if the run is done
    fn update_counters(&self, is_detection: bool) -> bool {
        let mut done = false;
        if is_detection
            && self.false_detections.fetch_add(1, Relaxed) + 1 >= self.max_false_detections
        {
            done = true;
        }
        if self.chunks_done.fetch_add(1, Relaxed) + 1 >= self.max_chunks {
            done = true;
        }
        done
    }

    fn results(&self, elapsed: Duration) -> FalseAcquisitionBenchmarkResults {
        let samples_done = self.chunks_done.load(Relaxed) * u64::try_from(CHUNK_LEN).unwrap();
        let benchmark_duration = samples_as_duration(samples_done);
        let false_detections = self.false_detections.load(Relaxed);
        let throughput_sps = samples_done as f64 / elapsed.as_secs_f64();
        FalseAcquisitionBenchmarkResults {
            false_detections_per_second: false_detections as f64 / benchmark_duration.as_secs_f64(),
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
    /// This function runs until the maximum benchmark duration or maximum
    /// number of false detections is reached, and returns the results.
    pub fn run(&self) -> FalseAcquisitionBenchmarkResults {
        let runner = FalseAcquisitionRunner::new(self);
        let start = Instant::now();
        runner.run(self.random_seed);
        let elapsed = start.elapsed();
        runner.results(elapsed)
    }
}

/// Decoding benchmark.
///
/// This structure defines the configuration of a decoding benchmark, which runs
/// the receiver on signal snippets containing a single ADS-B packet in AWGN at
/// a given CN0 and counts the number of successful detections and decodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodingBenchmark {
    /// Number of packets (signal snippets) to test in the benchmark.
    pub num_packets: u64,
    /// CN0 (in dB·Hz) of the packets.
    pub cn0: f64,
    /// Random seed.
    pub random_seed: u64,
    /// Number of threads to use for running the benchmark.
    pub num_threads: NonZero<usize>,
}

/// Decoding benchmark results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecodingBenchmarkResults {
    /// Average number detected packets.
    pub detection_probability: f64,
    /// Total number of detected packets.
    pub num_detections: u64,
    /// Probability of successfully decoding a packet.
    pub decode_probability: f64,
    /// Probability of successfully decoding a packets with zero bit errors.
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
    /// Total number of successfully decoded with zero bit errors.
    pub num_decoded_zero_bit_errors: u64,
    /// Total number of successfully decoded with one bit error.
    pub num_decoded_one_bit_error: u64,
    /// Total number of successfully decoded with two bit errors.
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

impl std::fmt::Display for DecodingBenchmarkResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        writeln!(
            f,
            "Detection probability: {:.3}",
            self.detection_probability
        )?;
        writeln!(f, "Total detections: {}", self.num_detections)?;
        writeln!(f, "Decode probability: {:.3}", self.decode_probability)?;
        writeln!(
            f,
            "Decode probability (0 bit errors): {:.3}",
            self.decode_probability_zero_bit_errors
        )?;
        writeln!(
            f,
            "Decode probability (1 bit error): {:.3}",
            self.decode_probability_one_bit_error
        )?;
        writeln!(
            f,
            "Decode probability (2 bit errors): {:.3}",
            self.decode_probability_two_bit_errors
        )?;
        writeln!(
            f,
            "Noncoherent demodulation probability: {:.3}",
            self.decode_probability_noncoherent
        )?;
        writeln!(f, "Total decoded: {}", self.num_decoded)?;
        writeln!(
            f,
            "Total decoded (0 bit errors): {}",
            self.num_decoded_zero_bit_errors
        )?;
        writeln!(
            f,
            "Total decoded (1 bit error): {}",
            self.num_decoded_one_bit_error
        )?;
        writeln!(
            f,
            "Total decoded (2 bit errors): {}",
            self.num_decoded_two_bit_errors
        )?;
        writeln!(
            f,
            "Total decoded using noncoherent demodulation: {}",
            self.num_decoded_noncoherent
        )?;
        writeln!(
            f,
            "Wrong decode probability: {:.3}",
            self.decode_wrong_probability
        )?;
        writeln!(
            f,
            "Wrong decode probability (0 bit errors): {:.3}",
            self.decode_wrong_probability_zero_bit_errors
        )?;
        writeln!(
            f,
            "Wrong decode probability (1 bit error): {:.3}",
            self.decode_wrong_probability_one_bit_error
        )?;
        writeln!(
            f,
            "Wrong decode probability (2 bit errors): {:.3}",
            self.decode_wrong_probability_two_bit_errors
        )?;
        writeln!(
            f,
            "Wrong decode probability (noncoherent demodulation): {:.3}",
            self.decode_wrong_probability_noncoherent
        )?;
        writeln!(f, "Total wrong decoded: {}", self.num_decoded_wrong)?;
        writeln!(
            f,
            "Total wrong decoded (0 bit errors): {}",
            self.num_decoded_wrong_zero_bit_errors
        )?;
        writeln!(
            f,
            "Total wrong decoded (1 bit error): {}",
            self.num_decoded_wrong_one_bit_error
        )?;
        writeln!(
            f,
            "Total wrong decoded (2 bit errors): {}",
            self.num_decoded_wrong_two_bit_errors
        )?;
        writeln!(
            f,
            "Total wrong decoded (noncoherent demodulation): {}",
            self.num_decoded_wrong_noncoherent
        )?;
        writeln!(f, "Number of packets: {}", self.num_packets)?;
        writeln!(
            f,
            "Maximum frequency error: {:.3} Hz",
            self.max_frequency_error
        )?;
        writeln!(f, "RMS frequency error: {:.3} Hz", self.rms_frequency_error)?;
        writeln!(
            f,
            "Maximum detection time: {:.3} samples",
            self.max_detection_time_error
        )?;
        writeln!(
            f,
            "Mean of detection time error: {:.3} samples",
            self.avg_detection_time_error
        )?;
        writeln!(
            f,
            "Standard deviation of detection time error: {:.3} samples",
            self.std_detection_time_error
        )?;
        writeln!(
            f,
            "Maximum decode time: {:.3} samples",
            self.max_decode_time_error
        )?;
        writeln!(
            f,
            "Mean of decode time error: {:.3} samples",
            self.avg_decode_time_error
        )?;
        writeln!(
            f,
            "Standard deviation of decode time error: {:.3} samples",
            self.std_decode_time_error
        )?;
        writeln!(
            f,
            "Maximum CN0 estimate error: {:.3} dB",
            self.max_cn0_error
        )?;
        writeln!(f, "Mean CN0 estimate error: {:.3} dB", self.avg_cn0_error)?;
        writeln!(
            f,
            "Standard deviation of CN0 estimate error: {:.3} dB",
            self.std_cn0_error
        )?;
        writeln!(
            f,
            "Real-time duration: {:.3}",
            self.real_time_duration.as_secs_f64()
        )?;
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

impl Default for DecodingBenchmark {
    fn default() -> DecodingBenchmark {
        DecodingBenchmark {
            num_packets: 100_000,
            cn0: 75.0,
            random_seed: 0,
            num_threads: default_num_threads(),
        }
    }
}

#[derive(Debug)]
struct DecodingRunner {
    packets_done: AtomicU64,
    max_packets: u64,
    cn0: CN0Distribution,
    num_threads: usize,
}

impl DecodingRunner {
    fn new(benchmark: &DecodingBenchmark) -> Result<DecodingRunner> {
        anyhow::ensure!(
            benchmark.num_packets > 0,
            "the number of packets cannot be zero"
        );
        let cn0 = CN0Distribution::Constant(ConstantCN0::new(benchmark.cn0)?);
        Ok(DecodingRunner {
            packets_done: AtomicU64::new(0),
            max_packets: benchmark.num_packets,
            cn0,
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
        let mut receiver = ADSBReceiver::new();
        let mut receiver_time_offset: u64 = 0;
        loop {
            let simulation = self.create_simulation(&mut rng);
            let snippet = simulation.run(&mut rng);
            assert!(snippet.len().is_multiple_of(CHUNK_LEN));
            for chunk in snippet.as_chunks::<CHUNK_LEN>().0 {
                let result = receiver.process_chunk(chunk);
                results.update(result, &simulation.message_config, receiver_time_offset);
            }
            receiver_time_offset += u64::try_from(snippet.len()).unwrap();
            if self.update_counter() {
                return results;
            }
        }
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

    // Returns true if the run is done
    fn update_counter(&self) -> bool {
        self.packets_done.fetch_add(1, Relaxed) + 1 >= self.max_packets
    }

    fn results(
        &self,
        worker_results: &DecodingWorkerResults,
        elapsed: Duration,
    ) -> DecodingBenchmarkResults {
        let packets_done = self.packets_done.load(Relaxed);
        let samples_done = packets_done * Self::NUM_SAMPLES;
        let throughput_sps = samples_done as f64 / elapsed.as_secs_f64();
        DecodingBenchmarkResults {
            detection_probability: worker_results.num_detections as f64 / packets_done as f64,
            num_detections: worker_results.num_detections,
            decode_probability: worker_results.num_ok_decodes() as f64 / packets_done as f64,
            decode_probability_zero_bit_errors: worker_results.num_ok_decodes_zero_errors as f64
                / packets_done as f64,
            decode_probability_one_bit_error: worker_results.num_ok_decodes_one_error as f64
                / packets_done as f64,
            decode_probability_two_bit_errors: worker_results.num_ok_decodes_two_errors as f64
                / packets_done as f64,
            decode_probability_noncoherent: worker_results.num_ok_decodes_noncoherent as f64
                / worker_results.num_ok_decodes() as f64,
            num_decoded: worker_results.num_ok_decodes(),
            num_decoded_zero_bit_errors: worker_results.num_ok_decodes_zero_errors,
            num_decoded_one_bit_error: worker_results.num_ok_decodes_one_error,
            num_decoded_two_bit_errors: worker_results.num_ok_decodes_two_errors,
            num_decoded_noncoherent: worker_results.num_ok_decodes_noncoherent,
            decode_wrong_probability: worker_results.num_wrong_decodes() as f64
                / packets_done as f64,
            decode_wrong_probability_zero_bit_errors: worker_results.num_wrong_decodes_zero_errors
                as f64
                / packets_done as f64,
            decode_wrong_probability_one_bit_error: worker_results.num_wrong_decodes_one_error
                as f64
                / packets_done as f64,
            decode_wrong_probability_two_bit_errors: worker_results.num_wrong_decodes_two_errors
                as f64
                / packets_done as f64,
            decode_wrong_probability_noncoherent: worker_results.num_wrong_decodes_noncoherent
                as f64
                / worker_results.num_wrong_decodes() as f64,
            num_decoded_wrong: worker_results.num_wrong_decodes(),
            num_decoded_wrong_zero_bit_errors: worker_results.num_wrong_decodes_zero_errors,
            num_decoded_wrong_one_bit_error: worker_results.num_wrong_decodes_one_error,
            num_decoded_wrong_two_bit_errors: worker_results.num_wrong_decodes_two_errors,
            num_decoded_wrong_noncoherent: worker_results.num_wrong_decodes_noncoherent,
            max_frequency_error: worker_results.max_freq_error,
            rms_frequency_error: (worker_results.freq_error_sqr_sum
                / worker_results.num_detections as f64)
                .sqrt(),
            max_detection_time_error: worker_results.max_detection_time_error,
            avg_detection_time_error: worker_results.detection_time_error_sum
                / worker_results.num_detections as f64,
            std_detection_time_error: standard_deviation(
                worker_results.detection_time_error_sqr_sum,
                worker_results.detection_time_error_sum,
                worker_results.num_detections,
            ),
            max_decode_time_error: worker_results.max_decode_time_error,
            avg_decode_time_error: worker_results.decode_time_error_sum
                / worker_results.num_decodes() as f64,
            std_decode_time_error: standard_deviation(
                worker_results.decode_time_error_sqr_sum,
                worker_results.decode_time_error_sum,
                worker_results.num_decodes(),
            ),
            max_cn0_error: worker_results.max_cn0_error,
            avg_cn0_error: worker_results.cn0_error_sum / worker_results.num_decodes() as f64,
            std_cn0_error: standard_deviation(
                worker_results.cn0_error_sqr_sum,
                worker_results.cn0_error_sum,
                worker_results.num_decodes(),
            ),
            num_packets: packets_done,
            real_time_duration: elapsed,
            throughput_sps,
            thread_throughput_sps: throughput_sps / self.num_threads as f64,
        }
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

    fn num_decodes(&self) -> u64 {
        self.num_ok_decodes() + self.num_wrong_decodes()
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
        let ModeSMessage::Long(expected_message) = &message_config.message else {
            panic!("unexpected short mode-s message");
        };
        let decode_correct = &decode.message == expected_message;

        if decode_correct {
            self.update_correct_decode(decode, message_config, receiver_time_offset);
        }

        match (decode_correct, decode.corrected_errors) {
            (true, BitErrors::NoErrors) => self.num_ok_decodes_zero_errors += 1,
            (true, BitErrors::SingleError(_)) => self.num_ok_decodes_one_error += 1,
            (true, BitErrors::DoubleError(_, _)) => self.num_ok_decodes_two_errors += 1,
            (false, BitErrors::NoErrors) => self.num_wrong_decodes_zero_errors += 1,
            (false, BitErrors::SingleError(_)) => self.num_wrong_decodes_one_error += 1,
            (false, BitErrors::DoubleError(_, _)) => self.num_wrong_decodes_two_errors += 1,
        }
        if decode.demodulation == DemodulationType::NonCoherent {
            if decode_correct {
                self.num_ok_decodes_noncoherent += 1;
            } else {
                self.num_wrong_decodes_noncoherent += 1;
            }
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
        REPLY_PREAMBLE_SAMPLES as f64 * (1.0 - message_config.symbol_clock_error);
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
        let runner = FalseAcquisitionRunner::new(&benchmark);
        assert_eq!(runner.chunks_done.load(Relaxed), 0);
        assert_eq!(runner.false_detections.load(Relaxed), 0);
        assert_eq!(runner.max_false_detections, 100);
        assert_eq!(
            runner.max_chunks,
            10000 * u64::try_from(SAMPLE_RATE_USIZE / CHUNK_LEN).unwrap()
        );
        assert_eq!(runner.num_threads, 3);
    }

    #[test]
    fn false_acuisition_update_counters_no_detection() {
        let runner = FalseAcquisitionRunner::new(&FalseAcquisitionBenchmark::default());
        assert!(!runner.update_counters(false));
        assert_eq!(runner.chunks_done.load(Relaxed), 1);
        assert_eq!(runner.false_detections.load(Relaxed), 0);
    }

    #[test]
    fn false_acuisition_update_counters_detection() {
        let runner = FalseAcquisitionRunner::new(&FalseAcquisitionBenchmark::default());
        assert!(!runner.update_counters(true));
        assert_eq!(runner.chunks_done.load(Relaxed), 1);
        assert_eq!(runner.false_detections.load(Relaxed), 1);
    }

    #[test]
    fn false_acqusition_stop_by_detections() {
        let runner = FalseAcquisitionRunner::new(&FalseAcquisitionBenchmark::default());
        runner
            .false_detections
            .store(runner.max_false_detections - 1, Relaxed);
        assert!(runner.update_counters(true));
        assert_eq!(runner.chunks_done.load(Relaxed), 1);
        assert_eq!(
            runner.false_detections.load(Relaxed),
            runner.max_false_detections
        );
    }

    #[test]
    fn false_acqusition_stop_by_chunks() {
        let runner = FalseAcquisitionRunner::new(&FalseAcquisitionBenchmark::default());
        runner.chunks_done.store(runner.max_chunks - 1, Relaxed);
        assert!(runner.update_counters(true));
        assert_eq!(runner.chunks_done.load(Relaxed), runner.max_chunks);
        assert_eq!(runner.false_detections.load(Relaxed), 1);
    }

    #[test]
    fn false_acquisition_stop_by_detections_and_chunks() {
        let runner = FalseAcquisitionRunner::new(&FalseAcquisitionBenchmark::default());
        runner.chunks_done.store(runner.max_chunks - 1, Relaxed);
        runner
            .false_detections
            .store(runner.max_false_detections - 1, Relaxed);
        assert!(runner.update_counters(true));
        assert_eq!(runner.chunks_done.load(Relaxed), runner.max_chunks);
        assert_eq!(
            runner.false_detections.load(Relaxed),
            runner.max_false_detections
        );
    }

    #[test]
    fn false_acquisition_results() {
        let benchmark = FalseAcquisitionBenchmark {
            num_threads: 2.try_into().unwrap(),
            ..Default::default()
        };
        let runner = FalseAcquisitionRunner::new(&benchmark);
        // 1000 seconds worth of chunks
        let chunks = 1000 * u64::try_from(SAMPLE_RATE_USIZE / CHUNK_LEN).unwrap();
        runner.chunks_done.store(chunks, Relaxed);
        runner.false_detections.store(10, Relaxed);
        let runtime = Duration::from_secs(10);
        let throughput_sps = (1000 / 10 * SAMPLE_RATE_USIZE) as f64;
        let expected = FalseAcquisitionBenchmarkResults {
            false_detections_per_second: 10.0 / 1000.0,
            false_detections: 10,
            benchmark_duration: Duration::from_secs(1000),
            real_time_duration: runtime,
            throughput_sps,
            thread_throughput_sps: throughput_sps * 0.5,
        };
        assert_eq!(runner.results(runtime), expected);
    }

    #[test]
    fn decoding_new() {
        let benchmark = DecodingBenchmark {
            num_threads: 3.try_into().unwrap(),
            ..Default::default()
        };
        let runner = DecodingRunner::new(&benchmark).unwrap();
        assert_eq!(runner.packets_done.load(Relaxed), 0);
        assert_eq!(runner.max_packets, 100_000);
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
    fn decoding_update_counter() {
        let runner = DecodingRunner::new(&DecodingBenchmark::default()).unwrap();
        assert!(!runner.update_counter());
        assert_eq!(runner.packets_done.load(Relaxed), 1);
    }

    #[test]
    fn decoding_update_counters_stop() {
        let runner = DecodingRunner::new(&DecodingBenchmark::default()).unwrap();
        runner.packets_done.store(runner.max_packets - 1, Relaxed);
        assert!(runner.update_counter());
        assert_eq!(runner.packets_done.load(Relaxed), runner.max_packets);
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
}
