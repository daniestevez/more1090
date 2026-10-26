# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Configuration of `more1090` receiver parameters through CLI arguments.

- Increased threshold to filter 2-bit error false decodes with noncoherent
  demodulation.

- Simulation of Mode-A/C replies.

- Ground-truth message configuration in simulator.

- SigMF annotations in `more1090-simulator` output.

- Scenario benchmark. Benchmarking for terrestrial and LEO scenarios.

### Changed

- Replaced preamble metric used to filter 2-bit error false decodes by CN0
  threshold.

- Timing estimation error reduced, specially in cases giving an error larger
  than 0.5 samples.

- Changed noise estimator filter shape. Before it integrated coherently 2 blocks
  of 2 samples spaced by either 6 or 10 samples. Now it integrates coherently
  just 2 samples.

- Simulations are now fully reproducible.

- Simulation output batch size is now configurable.

## [0.1.0] - 2026-09-29

### Added

- Initial more1090 release.

[unreleased]: https://github.com/daniestevez/more1090/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/daniestevez/more1090/releases/tag/v0.1.0
