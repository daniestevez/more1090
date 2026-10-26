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


def load_jsons(input_dir, glob_expr):
    data = []
    for p in input_dir.glob(glob_expr):
        with open(p) as f:
            data.append(json.load(f))
    return data


def extract_results(data):
    return {k: np.array([d["results"][k] for d in data]) for k in data[0]["results"]}


def load_data(input_dir, glob_expr, get_xaxis):
    data = load_jsons(input_dir, glob_expr)
    data.sort(key=lambda entry: get_xaxis(entry))
    x_axis = np.array([get_xaxis(d) for d in data])
    results = extract_results(data)
    return x_axis, results


class Plotter:
    def __init__(self):
        self.output_dir = pathlib.Path(__file__).parent / "plots"
        self.output_dir.mkdir(exist_ok=True)
        input_dir = pathlib.Path(__file__).parent / "results"

        # load false acquisition results
        self.false_acquisition_thresholds, self.false_acquisition_results = load_data(
            input_dir,
            "false_acquisition_*.json",
            lambda entry: entry["benchmark"]["receiver_args"][
                "carrier_metric_threshold"
            ],
        )

        # load decoding benchmark results
        self.decoding_cn0s, self.decoding_results = load_data(
            input_dir, "decoding_*.json", lambda entry: entry["benchmark"]["cn0"]
        )

        # load scenario benchmark results
        self.scenarios = list(
            {p.name.split("_")[1] for p in input_dir.glob("scenario_*.json")}
        )
        self.scenario_mps = {}
        self.scenario_results = {}
        for scenario in self.scenarios:
            self.scenario_mps[scenario], self.scenario_results[scenario] = load_data(
                input_dir,
                f"scenario_{scenario}_*.json",
                lambda entry: entry["benchmark"]["messages_per_second"],
            )

    def make_plots(self):
        self.plot_false_acquisition()
        self.plot_decode()
        self.plot_decode_wrong()
        self.plot_noncoherent()
        self.plot_detect()
        self.plot_freq_error()
        self.plot_time_error()
        self.plot_cn0_error()
        for scenario in self.scenarios:
            self.plot_scenario_decode(scenario)
            self.plot_scenario_decode_wrong(scenario)
            self.plot_scenario_noncoherent(scenario)
            self.plot_scenario_freq_error(scenario)
            self.plot_scenario_time_error(scenario)
            self.plot_scenario_cn0_error(scenario)
            self.plot_scenario_collision(scenario)

    def new_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 4))

    def finish_decode_plot(self, title, yaxis, figpath, *, legend_loc=None):
        self.ax.xaxis.set_minor_locator(MultipleLocator(1))
        self.ax.set_xlabel("CN0 (dB·Hz)")
        self.finish_plot(title, yaxis, figpath, legend_loc=legend_loc)

    def finish_false_acquisition_plot(self, title, yaxis, figpath, *, legend_loc=None):
        self.ax.set_xlabel("Carrier metric threshold")
        self.finish_plot(title, yaxis, figpath, legend_loc=legend_loc)

    def finish_scenario_plot(self, title, yaxis, figpath, *, legend_loc=None):
        self.ax.set_xlabel("Messages/second")
        self.ax.set_xscale("log")
        self.finish_plot(title, yaxis, figpath, legend_loc=legend_loc)

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
        self.ax.grid(which="major")
        self.ax.grid(which="minor", linewidth=0.2)
        self.ax.set_ylabel(yaxis)
        self.ax.set_title(title)
        self.fig.savefig(self.output_dir / figpath, bbox_inches="tight", dpi=300)
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def plot_key(self, key, label):
        self.plot_line(self.decoding_results[key], label)

    def plot_line(self, data, label):
        self.ax.plot(self.decoding_cn0s, data, ".-", markersize=4, label=label)

    def plot_false_acquisition_key(self, key, label):
        self.plot_false_acquisition_line(self.false_acquisition_results[key], label)

    def plot_false_acquisition_line(self, data, label):
        self.ax.plot(
            self.false_acquisition_thresholds, data, ".-", markersize=4, label=label
        )

    def plot_scenario_key(self, scenario, key, label):
        self.plot_scenario_line(scenario, self.scenario_results[scenario][key], label)

    def plot_scenario_line(self, scenario, data, label):
        self.ax.plot(self.scenario_mps[scenario], data, ".-", markersize=4, label=label)

    def set_probability_yaxis(self):
        self.ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    def plot_false_acquisition(self):
        self.new_figure()
        self.plot_false_acquisition_line(
            1 / self.false_acquisition_results["false_detections_per_second"],
            "seconds per false detection",
        )
        self.finish_false_acquisition_plot(
            "more1090 false detection in AWGN",
            "Time between false detections (s)",
            "false_acquisition.png",
        )

    def plot_decode(self):
        self.new_figure()
        self.plot_key("decode_probability", "more1090 $\\leq 2$ bit errors")
        self.plot_line(
            self.decoding_results["decode_probability_zero_bit_errors"]
            + self.decoding_results["decode_probability_one_bit_error"],
            "more1090 $\\leq 1$ bit errors",
        )
        self.plot_key("decode_probability_zero_bit_errors", "more1090 $= 0$ bit errors")
        self.plot_key("decode_probability_one_bit_error", "more1090 $= 1$ bit error")
        self.plot_key("decode_probability_two_bit_errors", "more1090 $= 2$ bit errors")
        self.plot_decode_theory()
        self.ax.set_xlim(left=65, right=76)
        self.set_probability_yaxis()
        self.finish_decode_plot(
            "more1090 decode probability compared to theory",
            "Probability",
            "decode.png",
            legend_loc="right",
        )

    def plot_decode_wrong(self):
        self.new_figure()
        self.plot_key("decode_wrong_probability", "$\\leq 2$ bit errors")
        self.plot_line(
            self.decoding_results["decode_wrong_probability_zero_bit_errors"]
            + self.decoding_results["decode_wrong_probability_one_bit_error"],
            "$\\leq 1$ bit errors",
        )
        self.plot_key("decode_wrong_probability_zero_bit_errors", "$= 0$ bit errors")
        self.plot_key("decode_wrong_probability_one_bit_error", "$= 1$ bit error")
        self.plot_key("decode_wrong_probability_two_bit_errors", "$= 2$ bit errors")
        self.set_probability_yaxis()
        self.finish_decode_plot(
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
        self.finish_decode_plot(
            "more1090 noncoherent demodulation usage", "Probability", "noncoherent.png"
        )

    def plot_detect(self):
        self.new_figure()
        self.plot_key("detection_probability", "Detection probability")
        self.ax.set_xlim(left=55, right=64)
        self.set_probability_yaxis()
        self.finish_decode_plot(
            "more1090 detection probability", "Probability", "detect.png"
        )

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
        self.finish_decode_plot(
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
        self.finish_decode_plot(
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
        self.finish_decode_plot(
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
        self.finish_decode_plot(
            "more1090 CN0 estimate error",
            "CN0 error (dB)",
            "cn0_error.png",
        )

    def plot_scenario_decode(self, scenario):
        self.new_figure()
        self.plot_scenario_key(
            scenario, "decode_probability", "good $\\leq 2$ bit errors"
        )
        self.plot_scenario_line(
            scenario,
            self.scenario_results[scenario]["decode_probability_zero_bit_errors"]
            + self.scenario_results[scenario]["decode_probability_one_bit_error"],
            "good $\\leq 1$ bit errors",
        )
        self.plot_scenario_key(
            scenario, "decode_probability_zero_bit_errors", "good $= 0$ bit errors"
        )
        self.plot_scenario_key(
            scenario, "decode_probability_one_bit_error", "good $= 1$ bit error"
        )
        self.plot_scenario_key(
            scenario, "decode_probability_two_bit_errors", "good $= 2$ bit errors"
        )
        self.plot_scenario_key(
            scenario, "decode_wrong_probability", "false $\\leq 2$ bit errors"
        )
        self.plot_scenario_line(
            scenario,
            self.scenario_results[scenario]["decode_wrong_probability_zero_bit_errors"]
            + self.scenario_results[scenario]["decode_wrong_probability_one_bit_error"],
            "false $\\leq 1$ bit errors",
        )
        self.set_probability_yaxis()
        self.finish_scenario_plot(
            f"more1090 decode probability in {scenario} scenario",
            "Probability",
            f"scenario_{scenario}_decode.png",
            legend_loc="right",
        )

    def plot_scenario_decode_wrong(self, scenario):
        self.new_figure()
        self.plot_scenario_key(
            scenario, "decode_wrong_probability", "$\\leq 2$ bit errors"
        )
        self.plot_scenario_line(
            scenario,
            self.scenario_results[scenario]["decode_wrong_probability_zero_bit_errors"]
            + self.scenario_results[scenario]["decode_wrong_probability_one_bit_error"],
            "$\\leq 1$ bit errors",
        )
        self.plot_scenario_key(
            scenario, "decode_wrong_probability_zero_bit_errors", "$= 0$ bit errors"
        )
        self.plot_scenario_key(
            scenario, "decode_wrong_probability_one_bit_error", "$= 1$ bit error"
        )
        self.plot_scenario_key(
            scenario, "decode_wrong_probability_two_bit_errors", "$= 2$ bit errors"
        )
        self.set_probability_yaxis()
        self.finish_scenario_plot(
            f"more1090 false decode probability in {scenario} scenario",
            "Probability",
            f"scenario_{scenario}_decode_wrong.png",
            legend_loc="right",
        )

    def plot_scenario_noncoherent(self, scenario):
        self.new_figure()
        self.plot_scenario_key(
            scenario,
            "decode_probability_noncoherent",
            "Probability that a successful decode used noncoherent demodulation",
        )
        self.plot_scenario_key(
            scenario,
            "decode_wrong_probability_noncoherent",
            "Probability that an incorrect decode used noncoherent demodulation",
        )
        self.set_probability_yaxis()
        self.finish_scenario_plot(
            f"more1090 noncoherent demodulation usage in {scenario} scenario",
            "Probability",
            f"scenario_{scenario}_noncoherent.png",
        )

    def plot_scenario_freq_error(self, scenario):
        self.new_figure()
        self.plot_scenario_key(scenario, "rms_frequency_error", "RMS frequency error")
        self.plot_scenario_key(scenario, "max_frequency_error", "Max frequency error")
        packet_duration = PACKET_NUM_BITS / BITS_PER_SECOND
        self.ax.axhline(
            y=0.1 / packet_duration,
            linestyle=":",
            color="gray",
            label="0.1 / Mode-S reply duration",
        )
        self.finish_scenario_plot(
            f"more1090 carrier frequency estimation error in {scenario} scenario",
            "Frequency error (Hz)",
            f"scenario_{scenario}_freq_error.png",
        )

    def plot_scenario_time_error(self, scenario):
        self.new_figure()
        self.plot_scenario_key(scenario, "max_decode_time_error", "Max time error")
        self.plot_scenario_key(
            scenario, "std_decode_time_error", "Standard deviation of time error"
        )
        self.plot_scenario_key(scenario, "avg_decode_time_error", "Average time error")
        self.finish_scenario_plot(
            f"more1090 post-decode time estimation error in {scenario} scenario",
            "Time error (samples)",
            f"scenario_{scenario}_time_error.png",
        )

        self.new_figure()
        self.plot_scenario_key(scenario, "max_detection_time_error", "Max time error")
        self.plot_scenario_key(
            scenario, "std_detection_time_error", "Standard deviation of time error"
        )
        self.plot_scenario_key(
            scenario, "avg_detection_time_error", "Average time error"
        )
        self.ax.axhline(
            y=NFFT_SHORT, linestyle=":", color="gray", label="Detection time step"
        )
        self.finish_scenario_plot(
            f"more1090 detection time estimation error in {scenario} scenario",
            "Time error (samples)",
            f"scenario_{scenario}_time_error_detection.png",
        )

    def plot_scenario_cn0_error(self, scenario):
        self.new_figure()
        self.plot_scenario_key(scenario, "max_cn0_error", "Max CN0 error")
        self.plot_scenario_key(
            scenario, "std_cn0_error", "Standard deviation of CN0 error"
        )
        self.plot_scenario_key(scenario, "avg_cn0_error", "Average CN0 error")
        self.finish_scenario_plot(
            f"more1090 CN0 estimate error in {scenario} scenario",
            "CN0 error (dB)",
            f"scenario_{scenario}_cn0_error.png",
        )

    def plot_scenario_collision(self, scenario):
        self.new_figure()
        self.plot_scenario_key(
            scenario, "collision_probability", "Collision probability"
        )
        self.set_probability_yaxis()
        self.finish_scenario_plot(
            f"ADS-B message collision probability in {scenario} scenario",
            "Probability",
            f"scenario_{scenario}_collision.png",
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
        ebn0s = self.decoding_cn0s - 66
        # Coherent BPSK demodulation BER formula
        bpsk_ber = 0.5 * erfc(10 ** (ebn0s / 20))

        # For non-coherent demodulation of PPM, the theory is equivalent to
        # non-coherent 2-FSK, since the two possible pulse positions are
        # orthogonal symbols. The CN0 that matters is the CN0 of each pulse, but
        # we should consider the modulation as 2 Mbaud rather than 1 Mbaud,
        # because the symbol duration is only 0.5 usec. Therefore, the Eb/N0
        # that must be used for the non-coherent 2-FSK formula is the CN0 minus
        # 63 dB.
        ebn0s_fsk = self.decoding_cn0s - 63
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
                    self.decoding_cn0s,
                    k_or_less_bit_errors_prob(k, coherent),
                    linestyle,
                    label=f"{label} theory $\\leq {k}$ bit errors",
                )
            for k in range(3):
                self.ax.plot(
                    self.decoding_cn0s,
                    k_bit_errors_prob(k, coherent),
                    linestyle,
                    label=f"{label} theory $= {k}$ bit errors",
                )


def main():
    plotter = Plotter()
    plotter.make_plots()


if __name__ == "__main__":
    main()
