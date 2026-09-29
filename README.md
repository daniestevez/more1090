# more1090

more1090 is a high-sensitivity ADS-B receiver that performs better in scenarios
with low SNR and packet collisions than other open source receivers such as
[readsb](https://github.com/wiedehopf/readsb). more1090 requires an input sample
rate of 8 Msps IQ. It decodes extended squitter (DF17 and DF18) Mode-S reply
messages. In addition to the decoded message, the receiver provides estimates
for the time of arrival, the carrier frequency and the CN0 of the ADS-B message.

more1090 can be used as a CLI application called `more1090` that reads a SigMF
file and produces an annotation for each decoded message, or reads a real time
stream from stdin and outputs each decoded message in AVR format (allowing the
output to be piped into other applications such as
[tar1090](https://github.com/wiedehopf/tar1090)). more1090 also contains a
Mode-S signal simulator called `more1090-simulator` and a benchmarking tool
called `more1090-benchmark` that is used to benchmark the sensitivity and
throughput of the more1090 ADS-B receiver. Additionally, all of this can be used
as a Rust library.

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
