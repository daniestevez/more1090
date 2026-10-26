# more1090

more1090 is a high-sensitivity ADS-B receiver that performs better in scenarios
with low SNR and packet collisions than other open source receivers such as
[readsb](https://github.com/wiedehopf/readsb). more1090 requires an input sample
rate of 8 Msps IQ. It decodes extended squitter (DF17 and DF18) Mode-S reply
messages. In addition to the decoded message, the receiver provides estimates
for the time of arrival, the carrier frequency and the CN0 of the decoded ADS-B
messages.

more1090 can be used as a CLI application called `more1090` that reads a SigMF
file and produces an annotation for each decoded message, or reads a real-time
stream from stdin and outputs each decoded message in AVR format (allowing the
output to be piped into other applications such as
[tar1090](https://github.com/wiedehopf/tar1090)). more1090 also contains a
secondary surveillance radar reply signal simulator called `more1090-simulator`
that is able to generate all types of Mode-A/C and Mode-S replies, and a
benchmarking tool called `more1090-benchmark` that is used to benchmark the
sensitivity and throughput of the more1090 ADS-B receiver. Additionally, all of
this can be used as a Rust library.

## Installation

more1090 uses SIMD heavily. It benefits from being built with the appropriate
flags for the CPU where it is run (so you can get AVX2 or AVX-512 if
available). For instance, to install it for running on the current CPU, build it
with `target-cpu=native` in the following way.

```
RUSTFLAGS="-C target-cpu=native" cargo install --locked more1090
```

## Quick start with a simulated dataset

Create a SigMF dataset that contains a simulation of secondary surveillance
radar reply messages at 1090 MHz.

```
more1090-simulator scenario ssr-scenario.sigmf-meta
```

Run `more1090` on this SigMF file to decode ADS-B messages.

```
more1090 ssr-scenario.sigmf-meta
```

`more1090-simulator` adds a SigMF annotation for each SSR message in the
simulated scenario. `more1090` adds a SigMF annotation for each decoded
message. If you want to delete the simulator annotations when running
`more1090`, use the `--clear-annotations` flag.

## Running in real-time with an SDR

`more1090` can be run by feeding it 8 Msps IQ data in (native-endian) complex
float32 format through its standard input. When run in this way, `more1090`
outputs decoded messages in AVR format.

With a USRP, the following can be used.

```
uhd_rx_cfile -r 8e6 -f 1090e6 --lo-offset 4e6 -g 60 -A RX2 /dev/stdout | more1090 --stdin
```

Other SDR hardware may have similar CLI tools that can pipe complex float IQ
data into `more1090`.

## CPU benchmark

Run the following to test how fast `more1090` can run in your CPU.

```
more1090-benchmark throughput
```

This generates a 10 second simulated scenario and loops the more1090 over it as
fast as the CPU allows. 

## Profiling

First build using the `profiling` profile to get optimizations and debug
information.

```
RUSTFLAGS="-C target-cpu=native" cargo build --profile profiling
```

Run the throughput benchmark under `perf`. The following runs for 30 seconds
with an additional warmp-up time of 10 seconds. It also uses `taskset` to pin
the benchmark to CPU 0.

```
timeout 40 perf record -F 299 -D 10000 --call-graph dwarf taskset -c 0 \
    ./target/profiling/more1090-benchmark throughput
```

Use the following to open the profile trace in [Firefox
Profiler](https://profiler.firefox.com/).

```
perf script report gecko
```

The results can also be viewed in `perf annotate` or `perf report`.

## Sending decoded data to tar1090

[tar1090](https://github.com/wiedehopf/tar1090) can be used to display a map of
the decoded aircraft. A simple way of running it is to use the Docker image from
[sdr-enthusiasts/docker-tar1090](https://github.com/sdr-enthusiasts/docker-tar1090). It
can be launched as follows.

```
docker run --rm --name=tar1090 -p 8078:80 -p 30001:30001 ghcr.io/sdr-enthusiasts/docker-tar1090:latest
```

Then the output of `more1090` can be piped into TCP port 30001. For instance,
this uses `uhd_rx_cfile` to get IQ data from a USRP and `socat` to send
`more1090`'s output by TCP.

```
uhd_rx_cfile -r 8e6 -f 1090e6 --lo-offset 4e6 -g 60 -A RX2 /dev/stdout | \
    more1090 --stdin | socat -u - TCP:127.0.0.1:30001
```

The web interface of `tar1090` can is in [http://localhost:8078/](http://localhost:8078/).

## API documentation

The documentation for the more1090 Rust crate is hosted in [docs.rs](https://docs.rs/more1090/).

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
