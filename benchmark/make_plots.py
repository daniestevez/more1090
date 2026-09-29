#!/usr/bin/env python3

import json
import math
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.special import erfc

PACKET_NUM_BITS = 112  # number of bits in Mode-S reply
BITS_PER_SECOND = 1_000_000  # bits/second in Mode-S reply
NFFT_SHORT = 128  # short FFT size in samples (equal to FFT overlap step)


class Plotter:
    def __init__(self):
        self.output_dir = pathlib.Path(__file__).parent / "plots"
        self.output_dir.mkdir(exist_ok=True)
        data = []
        input_dir = pathlib.Path(__file__).parent / "results"
        for p in input_dir.glob("decoding_*.json"):
            with open(p) as f:
                data.append(json.load(f))
        data.sort(key=lambda entry: entry["benchmark"]["cn0"])
        self.cn0s = np.array([d["benchmark"]["cn0"] for d in data])
        self.results = {
            k: np.array([d["results"][k] for d in data]) for k in data[0]["results"]
        }

    def make_plots(self):
        self.plot_decode()
        self.plot_decode_wrong()
        self.plot_noncoherent()
        self.plot_detect()
        self.plot_freq_error()
        self.plot_time_error()
        self.plot_cn0_error()

    def new_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))

    def finish_plot(self, title, yaxis, figpath, *, legend_loc=None):
        if legend_loc is None:
            legend_kwargs = {}
        elif legend_loc == "right":
            legend_kwargs = {
                "bbox_to_anchor": (1.01, 1),
                "loc": "upper left",
                "borderaxespad": 0,
            }
        else:
            raise ValueError(f"invalid legend_loc '{legend_loc}'")
        self.ax.legend(**legend_kwargs)
        self.ax.xaxis.set_minor_locator(MultipleLocator(1))
        self.ax.grid(which="major")
        self.ax.grid(which="minor", linewidth=0.2)
        self.ax.set_xlabel("CN0 (dB·Hz)")
        self.ax.set_ylabel(yaxis)
        self.ax.set_title(title)
        self.fig.savefig(self.output_dir / figpath, bbox_inches="tight", dpi=300)
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def plot_key(self, key, label):
        self.plot_line(self.results[key], label)

    def plot_line(self, data, label):
        self.ax.plot(self.cn0s, data, ".-", markersize=4, label=label)

    def set_probability_yaxis(self):
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    def plot_decode(self):
        self.new_figure()
        self.plot_key("decode_probability", "more1090 $\\leq 2$ bit errors")
        self.plot_line(
            self.results["decode_probability_zero_bit_errors"]
            + self.results["decode_probability_one_bit_error"],
            "more1090 $\\leq 1$ bit errors",
        )
        self.plot_key("decode_probability_zero_bit_errors", "more1090 $= 0$ bit errors")
        self.plot_key("decode_probability_one_bit_error", "more1090 $= 1$ bit error")
        self.plot_key("decode_probability_two_bit_errors", "more1090 $= 2$ bit errors")
        self.plot_decode_theory()
        self.ax.set_xlim(left=65, right=76)
        self.set_probability_yaxis()
        self.finish_plot(
            "more1090 decode probability compared to theory",
            "Probability",
            "decode.png",
            legend_loc="right",
        )

    def plot_decode_wrong(self):
        self.new_figure()
        self.plot_key("decode_wrong_probability", "$\\leq 2$ bit errors")
        self.plot_line(
            self.results["decode_wrong_probability_zero_bit_errors"]
            + self.results["decode_wrong_probability_one_bit_error"],
            "$\\leq 1$ bit errors",
        )
        self.plot_key("decode_wrong_probability_zero_bit_errors", "$= 0$ bit errors")
        self.plot_key("decode_wrong_probability_one_bit_error", "$= 1$ bit error")
        self.plot_key("decode_wrong_probability_two_bit_errors", "$= 2$ bit errors")
        self.set_probability_yaxis()
        self.finish_plot(
            "more1090 wrong decode probability",
            "Probability",
            "decode_wrong.png",
            legend_loc="right",
        )

    def plot_noncoherent(self):
        self.new_figure()
        self.plot_key(
            "decode_probability_noncoherent",
            "Probability that a successful decode used noncoherent demodulation",
        )
        self.plot_key(
            "decode_wrong_probability_noncoherent",
            "Probability that an incorrect decode used noncoherent demodulation",
        )
        self.ax.set_xlim(left=65)
        self.set_probability_yaxis()
        self.finish_plot(
            "more1090 noncoherent demodulation usage", "Probability", "noncoherent.png"
        )

    def plot_detect(self):
        self.new_figure()
        self.plot_key("detection_probability", "Detection probability")
        self.ax.set_xlim(right=64)
        self.set_probability_yaxis()
        self.finish_plot("more1090 detection probability", "Probability", "detect.png")

    def plot_freq_error(self):
        self.new_figure()
        self.plot_key("rms_frequency_error", "RMS frequency error")
        self.plot_key("max_frequency_error", "Max frequency error")
        packet_duration = PACKET_NUM_BITS / BITS_PER_SECOND
        self.ax.axhline(
            y=0.1 / packet_duration,
            linestyle=":",
            color="gray",
            label="0.1 / Mode-S reply duration",
        )
        self.finish_plot(
            "more1090 carrier frequency estimation error",
            "Frequency error (Hz)",
            "freq_error.png",
        )

    def plot_time_error(self):
        self.new_figure()
        self.plot_key("max_decode_time_error", "Max time error")
        self.plot_key("std_decode_time_error", "Standard deviation of time error")
        self.plot_key("avg_decode_time_error", "Average time error")
        self.ax.set_xlim(left=65)
        self.finish_plot(
            "more1090 post-decode time estimation error",
            "Time error (samples)",
            "time_error.png",
        )

        self.new_figure()
        self.plot_key("max_detection_time_error", "Max time error")
        self.plot_key("std_detection_time_error", "Standard deviation of time error")
        self.plot_key("avg_detection_time_error", "Average time error")
        self.ax.axhline(
            y=NFFT_SHORT, linestyle=":", color="gray", label="Detection time step"
        )
        self.finish_plot(
            "more1090 detection time estimation error",
            "Time error (samples)",
            "time_error_detection.png",
        )

    def plot_cn0_error(self):
        self.new_figure()
        self.plot_key("max_cn0_error", "Max CN0 error")
        self.plot_key("std_cn0_error", "Standard deviation of CN0 error")
        self.plot_key("avg_cn0_error", "Average CN0 error")
        self.ax.set_xlim(left=65)
        self.finish_plot(
            "more1090 CN0 estimate error",
            "CN0 error (dB)",
            "cn0_error.png",
        )

    def plot_decode_theory(self):
        # The CN0 is measured as the CN0 of each pulse, without taking into
        # account the 50% duty cycle of the signal. Therefore, the CN0 taking
        # into account average power is 3 dB less. The Mode-S PPM modulation can
        # be understood as residual carrier Manchester coding modulation at 1
        # Mbaud, where 50% of the power goes to the residual carrier and 50% of
        # the power goes to the data. Therefore the data modulation has another
        # 3 dB less. As the number of bits per second is 1e6 (60 dB), we get
        # that the Eb/N0 is 66 dB less than the CN0. (This ignores the fact that
        # CRC-24 bits do not really transmit information, but they are not a
        # true error correction code either).
        ebn0s = self.cn0s - 66
        # Coherent BPSK demodulation BER formula
        bpsk_ber = 0.5 * erfc(10 ** (ebn0s / 20))

        # For non-coherent demodulation of PPM, the theory is equivalent to
        # non-coherent 2-FSK, since the two possible pulse positions are
        # orthogonal symbols. The CN0 that matters is the CN0 of each pulse, but
        # we should consider the modulation as 2 Mbaud rather than 1 Mbaud,
        # because the symbol duration is only 0.5 usec. Therefore, the Eb/N0
        # that must be used for the non-coherent 2-FSK formula is the CN0 minus
        # 63 dB.
        ebn0s_fsk = self.cn0s - 63
        # Non-coherent FSK demodulation formula
        fsk_ber = 0.5 * np.exp(-0.5 * 10 ** (ebn0s_fsk / 10))

        def k_bit_errors_prob(k, coherent=True):
            ber = bpsk_ber if coherent else fsk_ber
            return (
                math.comb(PACKET_NUM_BITS, k)
                * ber**k
                * (1 - ber) ** (PACKET_NUM_BITS - k)
            )

        def k_or_less_bit_errors_prob(k, coherent=True):
            return sum(k_bit_errors_prob(j, coherent) for j in range(k + 1))

        for coherent, linestyle, label in zip(
            [True, False], ["--", ":"], ["Coh.", "Noncoh."]
        ):
            self.ax.set_prop_cycle(None)
            for k in range(2, 0, -1):
                self.ax.plot(
                    self.cn0s,
                    k_or_less_bit_errors_prob(k, coherent),
                    linestyle,
                    label=f"{label} theory $\\leq {k}$ bit errors",
                )
            for k in range(3):
                self.ax.plot(
                    self.cn0s,
                    k_bit_errors_prob(k, coherent),
                    linestyle,
                    label=f"{label} theory $= {k}$ bit errors",
                )


def main():
    plotter = Plotter()
    plotter.make_plots()


if __name__ == "__main__":
    main()
