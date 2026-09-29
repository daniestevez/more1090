//! Throughput benchmarking.
//!
//! This module contains routines to benchmark the throughput (CPU performance)
//! of the more1090 ADS-B receiver.

use crate::{
    ADSBReceiver, CHUNK_LEN,
    constants::dsp::SAMPLE_RATE_F64,
    receiver::ADSBPacket,
    simulation::{MessagesPerSecond, Simulation, distributions::MessageDistributions},
};
use anyhow::Result;
use num::complex::Complex32;
use std::{
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    thread,
    time::{Duration, Instant},
};

/// Throughput benchmark.
///
/// This structure defines the configuration of a throughput benchmark, which
/// runs the receiver on a pre-simulated scenario with multiple ADS-B packets.
#[derive(Debug, Clone, PartialEq)]
pub struct ThroughputBenchmark {
    /// Simulated scenario configuration.
    pub scenario: Simulation,
    /// Loop the scenario forever.
    pub loop_scenario: bool,
    /// Statistics report period.
    pub report_period: Duration,
}

impl Default for ThroughputBenchmark {
    fn default() -> ThroughputBenchmark {
        ThroughputBenchmark {
            scenario: Simulation {
                duration: Duration::from_secs(10),
                messages_per_second: MessagesPerSecond::from_total_messages_per_second(2000.0)
                    .unwrap(),
                message_distributions: MessageDistributions::default(),
            },
            loop_scenario: true,
            report_period: Duration::from_secs(1),
        }
    }
}

impl ThroughputBenchmark {
    /// Runs the throughput benchmark.
    ///
    /// This function pre-generates the simulated scenario and runs the receiver
    /// on it. It only returns if the scenario is not configured to loop
    /// forever, or if there is an error. The function prints the throughput
    /// statistics periodically.
    pub fn run(&self) -> Result<()> {
        anyhow::ensure!(
            self.report_period > Duration::ZERO,
            "report period cannot be zero"
        );
        let mut rng = rand::rng();
        let mut iq = Vec::new();
        self.scenario.run(&mut rng, &mut iq)?;
        anyhow::ensure!(
            iq.len().is_multiple_of(CHUNK_LEN),
            "scenario duration must have a length that is a multiple of {CHUNK_LEN} IQ samples"
        );

        let runner = Runner::new();
        let mut receiver = ADSBReceiver::new();
        let mut reporter = Reporter::new(Instant::now());
        if !self.loop_scenario {
            runner.run_receiver(&mut receiver, &iq, self.loop_scenario);
            eprintln!("{}", reporter.report(&runner, Instant::now()));
        } else {
            // use a thread to report periodically
            thread::scope(|s| {
                s.spawn(|| {
                    loop {
                        std::thread::sleep(self.report_period);
                        eprintln!("{}", reporter.report(&runner, Instant::now()));
                    }
                });
                runner.run_receiver(&mut receiver, &iq, self.loop_scenario);
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Runner {
    samples_processed: AtomicU64,
    detections: AtomicU64,
    decodes: AtomicU64,
}

impl Default for Runner {
    fn default() -> Runner {
        Runner::new()
    }
}

impl Runner {
    fn new() -> Runner {
        Runner {
            samples_processed: AtomicU64::new(0),
            detections: AtomicU64::new(0),
            decodes: AtomicU64::new(0),
        }
    }

    fn run_receiver(&self, receiver: &mut ADSBReceiver, iq: &[Complex32], loop_scenario: bool) {
        assert!(iq.len().is_multiple_of(CHUNK_LEN));
        loop {
            for chunk in iq.as_chunks::<CHUNK_LEN>().0 {
                let result = receiver.process_chunk(chunk);
                self.update_counters(result);
            }
            if !loop_scenario {
                break;
            }
        }
    }

    fn update_counters(&self, receiver_result: Option<&ADSBPacket>) {
        if let Some(packet) = receiver_result {
            self.detections.fetch_add(1, Relaxed);
            if packet.decode.is_some() {
                self.decodes.fetch_add(1, Relaxed);
            }
        }
        self.samples_processed
            .fetch_add(u64::try_from(CHUNK_LEN).unwrap(), Relaxed);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Reporter {
    t: Instant,
    samples_processed: u64,
    detections: u64,
    decodes: u64,
}

impl Reporter {
    fn new(now: Instant) -> Reporter {
        Reporter {
            t: now,
            samples_processed: 0,
            detections: 0,
            decodes: 0,
        }
    }

    fn report(&mut self, runner: &Runner, now: Instant) -> Report {
        let samples_processed = runner.samples_processed.load(Relaxed);
        let detections = runner.detections.load(Relaxed);
        let decodes = runner.decodes.load(Relaxed);
        let t = now;
        let samples_per_second =
            (samples_processed - self.samples_processed) as f64 / (t - self.t).as_secs_f64();
        let detections_per_second = (detections - self.detections) as f64
            / (samples_processed - self.samples_processed) as f64
            * SAMPLE_RATE_F64;
        let decodes_per_second = (decodes - self.decodes) as f64
            / (samples_processed - self.samples_processed) as f64
            * SAMPLE_RATE_F64;
        self.samples_processed = samples_processed;
        self.detections = detections;
        self.decodes = decodes;
        self.t = t;
        Report {
            samples_per_second,
            detections_per_second,
            decodes_per_second,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Report {
    samples_per_second: f64,
    detections_per_second: f64,
    decodes_per_second: f64,
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.3} Msps, {:.3} detections/s, {:.3} decodes/s",
            self.samples_per_second * 1e-6,
            self.detections_per_second,
            self.decodes_per_second
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::receiver::{ADSBDecode, ADSBDetection, BitErrors, DemodulationType};

    #[test]
    fn runner_init_counters_zero() {
        let runner = Runner::new();
        assert_eq!(runner.samples_processed.load(Relaxed), 0);
        assert_eq!(runner.detections.load(Relaxed), 0);
        assert_eq!(runner.decodes.load(Relaxed), 0);
    }

    #[test]
    fn update_counters_no_detection() {
        let runner = Runner::new();
        runner.update_counters(None);
        assert_eq!(
            runner.samples_processed.load(Relaxed),
            u64::try_from(CHUNK_LEN).unwrap()
        );
        assert_eq!(runner.detections.load(Relaxed), 0);
        assert_eq!(runner.decodes.load(Relaxed), 0);
    }

    #[test]
    fn update_counters_detection() {
        let runner = Runner::new();
        let packet = ADSBPacket {
            detection: ADSBDetection {
                sample_index: 0,
                carrier_frequency_offset: 0.0,
                carrier_power: 0.0,
                floor_power: 0.0,
            },
            decode: None,
        };
        runner.update_counters(Some(&packet));
        assert_eq!(
            runner.samples_processed.load(Relaxed),
            u64::try_from(CHUNK_LEN).unwrap()
        );
        assert_eq!(runner.detections.load(Relaxed), 1);
        assert_eq!(runner.decodes.load(Relaxed), 0);
    }

    #[test]
    fn update_counters_decode() {
        let runner = Runner::new();
        let packet = ADSBPacket {
            detection: ADSBDetection {
                sample_index: 0,
                carrier_frequency_offset: 0.0,
                carrier_power: 0.0,
                floor_power: 0.0,
            },
            decode: Some(ADSBDecode {
                sample_index: 0,
                sample_index_frac: 0.0,
                message: [0; _],
                corrected_errors: BitErrors::NoErrors,
                demodulation: DemodulationType::Coherent,
                cn0_db: 0.0,
            }),
        };
        runner.update_counters(Some(&packet));
        assert_eq!(
            runner.samples_processed.load(Relaxed),
            u64::try_from(CHUNK_LEN).unwrap()
        );
        assert_eq!(runner.detections.load(Relaxed), 1);
        assert_eq!(runner.decodes.load(Relaxed), 1);
    }

    #[test]
    fn report() {
        let t0 = Instant::now();
        let t1 = t0 + Duration::from_secs(2);
        let mut reporter = Reporter::new(t0);
        let runner = Runner {
            samples_processed: AtomicU64::new(32_000_000),
            detections: AtomicU64::new(8000),
            decodes: AtomicU64::new(6000),
        };
        let report = reporter.report(&runner, t1);
        let expected = Report {
            samples_per_second: 16_000_000.0,
            detections_per_second: 2000.0,
            decodes_per_second: 1500.0,
        };
        assert_eq!(report, expected);
    }
}
