#!/usr/bin/env python3

import logging
import pathlib
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


def false_acquisition_benchmark(output_dir):
    threshold_min = 0.0138
    threshold_max = 0.0158
    threshold_step = 0.0001
    thresholds = np.arange(
        threshold_min, threshold_max + threshold_step, threshold_step
    )
    for threshold in thresholds:
        logger.info(f"false acquisition benchmark for threshold {threshold:.4f}")
        res = subprocess.run(
            [
                "more1090-benchmark",
                "false-acquisition",
                "--json",
                "--carrier-metric-threshold",
                str(threshold),
            ],
            capture_output=True,
            check=True,
        )
        with open(
            output_dir / f"false_acquisition_threshold_{threshold:.4f}.json", "wb"
        ) as f:
            f.write(res.stdout)


def decode_benchmark(output_dir):
    cn0_min = 55
    cn0_max = 115
    cn0_step = 0.25
    cn0s = np.arange(cn0_min, cn0_max + cn0_step, cn0_step)
    for cn0 in cn0s:
        logger.info(f"decode benchmark for CN0 {cn0:.2f} dB·Hz")
        res = subprocess.run(
            ["more1090-benchmark", "decoding", "--json", "--cn0", str(cn0)],
            capture_output=True,
            check=True,
        )
        with open(output_dir / f"decoding_cn0_{cn0:.2f}.json", "wb") as f:
            f.write(res.stdout)


def scenario_benchmark(output_dir):
    scenarios = {
        "terrestrial": {
            "max-cn0": 105.0,
            "min-distance": 10.0,
            "max-distance": 500.0,
        },
        "LEO": {
            "max-cn0": 75.0,
            "min-distance": 500.0,
            "max-distance": 2600.0,
        },
    }
    messages_per_second = [45, 100, 200, 450, 1000, 2000, 4500, 10000, 20000, 45000]
    for scenario, settings in scenarios.items():
        scenario_args = [[f"--{arg}", str(value)] for arg, value in settings.items()]
        scenario_args = [a for b in scenario_args for a in b]
        for mps in messages_per_second:
            logger.info(f"scenario benchmark for {scenario} and {mps} messages/second")
            res = subprocess.run(
                [
                    "more1090-benchmark",
                    "scenario",
                    "--json",
                    "--messages-per-second",
                    str(mps),
                ]
                + scenario_args,
                capture_output=True,
                check=True,
            )
            with open(output_dir / f"scenario_{scenario}_{mps}.json", "wb") as f:
                f.write(res.stdout)


def main():
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    output_dir = pathlib.Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    false_acquisition_benchmark(output_dir)
    decode_benchmark(output_dir)
    scenario_benchmark(output_dir)


if __name__ == "__main__":
    main()
