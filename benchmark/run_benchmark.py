#!/usr/bin/env python3

import logging
import pathlib
import subprocess

import numpy as np

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    cn0_min = 55
    cn0_max = 95
    cn0_step = 0.25
    cn0s = np.arange(cn0_min, cn0_max + cn0_step, cn0_step)
    output_dir = pathlib.Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    for cn0 in cn0s:
        logger.info(f"decode benchmark for CN0 {cn0:.2f} dB·Hz")
        res = subprocess.run(
            ["more1090-benchmark", "decoding", "--json", "--cn0", str(cn0)],
            capture_output=True,
            check=True,
        )
        with open(output_dir / f"decoding_cn0_{cn0:.2f}.json", "wb") as f:
            f.write(res.stdout)


if __name__ == "__main__":
    main()
