use more1090::sigmf::{self, Annotation, SigMFMeta};
use std::{collections::HashSet, process::Command};

fn run_command(command: &mut Command) {
    let out = command
        .output()
        .unwrap_or_else(|err| panic!("{:?} failed to execute: {err}", command.get_program()));
    if !out.status.success() {
        panic!(
            "{:?} failed: {}",
            command.get_program(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

fn simulator_extended_squitter_hex(annotation: &Annotation) -> Option<String> {
    let message_type = annotation
        .get_key(sigmf::keys::MESSAGE_TYPE_KEY)
        .unwrap()
        .as_str()
        .unwrap();
    if message_type != "extended squitter" {
        return None;
    }
    let hex = annotation
        .get_key(sigmf::keys::MODE_S_HEX_KEY)
        .unwrap()
        .as_str()
        .unwrap();
    Some(hex.to_string())
}

fn receiver_hex(annotation: &Annotation) -> String {
    let hex = annotation
        .get_key(sigmf::keys::MODE_S_HEX_KEY)
        .unwrap()
        .as_str()
        .unwrap();
    hex.to_string()
}

/// Tests generating a SigMF scenario with more1090-simulator and decoding it
/// with more1090.
#[test]
fn simulator_scenario_and_receiver() {
    let tmpdir = tempfile::TempDir::new().expect("tempdir creation failed");
    let scenario = tmpdir.path().join("scenario.sigmf-meta");
    run_command(
        Command::new(env!("CARGO_BIN_EXE_more1090-simulator")).args([
            "scenario",
            scenario.to_str().unwrap(),
            "--duration",
            "1",
            "--messages-per-second",
            "2000",
            "--seed",
            "0",
        ]),
    );
    let simulator = SigMFMeta::from_file(&scenario).unwrap();
    let simulator_hexes = simulator
        .annotations()
        .into_iter()
        .filter_map(simulator_extended_squitter_hex)
        .collect::<HashSet<_>>();
    let num_simulator_es_messages = simulator_hexes.len();
    assert!(num_simulator_es_messages >= 750);

    run_command(
        Command::new(env!("CARGO_BIN_EXE_more1090"))
            .args(["--clear-annotations", scenario.to_str().unwrap()]),
    );
    let receiver = SigMFMeta::from_file(&scenario).unwrap();
    let receiver_hexes = receiver
        .annotations()
        .into_iter()
        .map(receiver_hex)
        .collect::<HashSet<_>>();
    let num_receiver_messages = receiver_hexes.len();
    let minimum_decode_probability = 0.8;
    assert!(
        num_receiver_messages
            >= (minimum_decode_probability * num_simulator_es_messages as f64).round() as usize
    );
    let num_false_decodes = receiver_hexes
        .iter()
        .filter(|&hex| !simulator_hexes.contains(hex))
        .count();
    let maximum_false_decode_probability = 0.005;
    assert!(
        num_false_decodes
            <= (maximum_false_decode_probability * num_simulator_es_messages as f64).round()
                as usize
    );
}

/// Tests generating a SigMF message with more1090-simulator and decoding it
/// with more1090.
#[test]
fn simulator_single_message_and_receiver() {
    let tmpdir = tempfile::TempDir::new().expect("tempdir creation failed");
    let single_message = tmpdir.path().join("single-message.sigmf-meta");
    run_command(
        Command::new(env!("CARGO_BIN_EXE_more1090-simulator")).args([
            "single-message",
            single_message.to_str().unwrap(),
            "--seed",
            "0",
        ]),
    );
    let simulator = SigMFMeta::from_file(&single_message).unwrap();
    let simulator_hexes = simulator
        .annotations()
        .into_iter()
        .filter_map(simulator_extended_squitter_hex)
        .collect::<HashSet<_>>();
    let num_simulator_es_messages = simulator_hexes.len();
    assert_eq!(num_simulator_es_messages, 1);

    run_command(
        Command::new(env!("CARGO_BIN_EXE_more1090"))
            .args(["--clear-annotations", single_message.to_str().unwrap()]),
    );
    let receiver = SigMFMeta::from_file(&single_message).unwrap();
    let receiver_hexes = receiver
        .annotations()
        .into_iter()
        .map(receiver_hex)
        .collect::<HashSet<_>>();
    let num_receiver_messages = receiver_hexes.len();
    assert_eq!(num_receiver_messages, 1);
    let num_false_decodes = receiver_hexes
        .iter()
        .filter(|&hex| !simulator_hexes.contains(hex))
        .count();
    assert_eq!(num_false_decodes, 0);
}
