"""A loader for data from the PLE module of the qudi_at_iqp package (unreleased).

Data is stored in three files per measurement:
 - trace: 2D matrix of count rates for the forward scan direction
 - retrace: ... for the backward scan direction
 - feedback: timestamps and frequencies from wavemeter readout during the measurement
"""
import glob
from configparser import ConfigParser
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import lmfit

from pleasant.measurement import Measurement


__all__ = ["load_folder"]


def load_folder(folder, plot=True, interp_eval_scan_index=0):
    data_files = glob.glob(f"{folder}*.dat")

    suffix = "_trace.dat"
    stubs = [i.removesuffix(suffix) for i in data_files if i.endswith(suffix)]

    measurements = []
    for stub in stubs:
        trace, retrace = read_data_files(stub, plot=plot,
                                         interp_eval_scan_index=interp_eval_scan_index)
        measurements.append(trace)
        measurements.append(retrace)

    return measurements


def read_data_files(stub, plot=True, interp_eval_scan_index=0):
    filename = stub.split("/")[-1]
    timestamp, description = filename.split("_", 1)

    trace_file = f"{stub}_trace.dat"
    retrace_file = f"{stub}_retrace.dat"
    feedback_file = f"{stub}_feedback.dat"
    count_rate_trace = np.loadtxt(trace_file)
    count_rate_retrace = np.fliplr(np.loadtxt(retrace_file))
    t_wavemeter, f_wavemeter = np.loadtxt(feedback_file, unpack=True)

    scan_repetitions, bin_count = count_rate_trace.shape

    _, metadata = read_header(trace_file)
    scan_range_start = float(metadata["scan range start"])
    scan_range_stop = float(metadata["scan range stop"])
    scan_speed = float(metadata["scan speed"])
    scan_resolution = float(metadata["scan resolution"])
    scan_duration = abs(scan_range_start - scan_range_stop) / scan_speed
    break_duration = float(metadata["scan pause"])

    # the actual break duration depends on the scan rate!
    rate = scan_speed * scan_resolution / abs(scan_range_stop - scan_range_start)
    break_duration = int(round(rate * break_duration)) / rate

    # fit start of first scan in an iterative way
    # start should be within approximately one trace and retrace
    t_sel = t_wavemeter[:t_wavemeter.size // scan_repetitions]
    f_sel = f_wavemeter[:t_wavemeter.size // scan_repetitions]
    t_measurement_start = find_measurement_start(
        t_sel, f_sel,
    )
    t_measurement_start_retrace = t_measurement_start + scan_duration + break_duration

    t_rates_trace, f_rates_trace = interp_wavemeter_readings(
        t_wavemeter,
        f_wavemeter,
        t_measurement_start,
        scan_duration,
        bin_count,
        scan_repetitions,
        break_duration,
    )
    t_rates_retrace, f_rates_retrace = interp_wavemeter_readings(
        t_wavemeter,
        f_wavemeter,
        t_measurement_start_retrace,
        scan_duration,
        bin_count,
        scan_repetitions,
        break_duration,
    )

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 2), sharey=True)
        for ax in axs:
            ax.plot(t_wavemeter, 1e-12 * f_wavemeter, ".", label="data")

        axs[0].plot(t_rates_trace[0], 1e-12 * f_rates_trace[0], label="trace")
        axs[0].plot(t_rates_retrace[0], 1e-12 * f_rates_retrace[0], label="retrace")
        axs[1].plot(t_rates_trace[-1], 1e-12 * f_rates_trace[-1], label="trace")
        axs[1].plot(t_rates_retrace[-1], 1e-12 * f_rates_retrace[-1], label="retrace")

        axs[0].set_xlim(0, t_measurement_start + 2 * (scan_duration + break_duration))
        last_stop = t_measurement_start + 2 * scan_repetitions * (
                    scan_duration + break_duration)
        last_start = last_stop - break_duration - 2 * (scan_duration + break_duration)
        axs[1].set_xlim(last_start, last_stop)

        for ax in axs:
            ax.set_xlabel("Time (s)")
        axs[0].legend(title="start")
        axs[1].legend(title="end")
        axs[0].yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        axs[0].set_ylabel("Wavemeter (THz)")
        fig.suptitle(f"{timestamp} | {description}")
        plt.subplots_adjust(wspace=0.05)
        plt.show()

    f_unified_trace = f_rates_trace[interp_eval_scan_index]
    count_rate_trace = interp_count_rate(
        f_unified_trace, f_rates_trace, count_rate_trace
    )

    f_unified_retrace = f_rates_retrace[interp_eval_scan_index]
    count_rate_retrace = interp_count_rate(
        f_unified_retrace, f_rates_retrace, count_rate_retrace
    )

    m_trace = Measurement(
        count_rate_trace,
        f_unified_trace,
        timestamp=timestamp,
        description=description,
        scan_duration=scan_duration,
        break_duration=break_duration,
    )
    m_retrace = Measurement(
        count_rate_retrace,
        f_unified_retrace,
        timestamp=timestamp,
        description=description,
        scan_duration=scan_duration,
        break_duration=break_duration,
    )

    return m_trace, m_retrace


def get_header_from_file(file_path):
    offset = 0
    line_start = 2
    with open(file_path, "r") as file:
        for line in file:
            if line.endswith("---- END HEADER ----\n"):
                break
            offset += len(line)
        file.seek(0)
        header_lines = file.read(offset).splitlines()
    return "\n".join(line[line_start:] for line in header_lines)


def read_header(file_path):
    header = get_header_from_file(file_path)

    config = ConfigParser(comment_prefixes=None, delimiters=("=",))
    config.read_string(header)

    timestamp = datetime.fromisoformat(
        config.get("General", "timestamp", raw=True, fallback=None)
    )

    metadata = dict(config.items("Metadata", raw=True))
    return timestamp, metadata


def find_measurement_start(t: np.ndarray, f: np.ndarray, window: int = 5, threshold: float = 0.2) -> float:
    """Find the change point from constant to linear in the wavemeter data.

    :param t: array containing the timestamps
    :param f: array containing the frequencies
    :param window: number of samples where iterative fits should agree
    :param threshold: how well fits have to agree within in sampling intervals
    :raises ValueError: if change point could not be found
    :return: fitted change time
    """
    t_all = t.copy()
    # normalize frequencies
    f_all = f - f[0]
    f_all = np.abs(f_all)
    f_all /= f_all.max()

    # broken stick model
    def constant_to_linear_change(x, c, x_change, slope):
        return np.where(x <= x_change, c, c + slope * (x - x_change))
    model = lmfit.Model(constant_to_linear_change)

    # perform iterative fits: include one more sample each iteration
    recent_fit_results = np.full(window, np.nan)
    i = window
    while i < t_all.size - 1:
        t = t_all[:i]
        f = f_all[:i]

        x_change_guess = t.mean()
        slope_guess = (f[-1] - f[0]) / (t[-1] - x_change_guess)
        c_guess = f[0]

        res = model.fit(f, x=t, c=c_guess, x_change=x_change_guess,
                        slope=slope_guess)
        recent_fit_results[0] = res.params["x_change"].value
        recent_fit_results = np.roll(recent_fit_results, 1)

        # check if fit results agree to less than one sample precision
        spacing = np.diff(t).mean()
        if recent_fit_results.std() / spacing < threshold:
            return recent_fit_results.mean()

        i += 1
    raise ValueError("Could not find a change point.")


def interp_wavemeter_readings(
    t_wavemeter,
    f_wavemeter,
    t_measurement_start,
    scan_duration,
    bin_count,
    scan_repetitions,
    scan_pause,
):
    # timings of bins during first trace
    t_rates_first_scan = t_measurement_start + np.linspace(0, scan_duration, bin_count)

    # offset by scan period and tile to complete measurement
    scan_period = 2 * (scan_duration + scan_pause)
    offset = np.tile(scan_period * np.arange(scan_repetitions), (bin_count, 1))
    t_rates = np.tile(t_rates_first_scan, (scan_repetitions, 1)) + offset.T

    # if times are beyond the wavemeter data: nan
    # can occur e.g. if wavemeter stream stopped too soon
    f_rates = np.interp(t_rates, t_wavemeter, f_wavemeter, left=np.nan, right=np.nan)
    return t_rates, f_rates


def interp_count_rate(f_unified, f_rates, count_rate):
    # xp values must be ascending for the interpolation
    ascending = f_rates[0, -1] > f_rates[0, 0]
    if ascending:
        f_rates_ascending = f_rates
        count_rate_ascending = count_rate
    else:
        f_rates_ascending = np.fliplr(f_rates)
        count_rate_ascending = np.fliplr(count_rate)

    count_rate_interpolated = np.empty_like(count_rate)
    for i in range(count_rate.shape[0]):
        count_rate_interpolated[i] = np.interp(
            f_unified,
            f_rates_ascending[i],
            count_rate_ascending[i],
            left=0.0,
            right=0.0,
        )

    nans = np.isnan(count_rate_interpolated)
    if nans.sum() > 0:
        print("Replacing NaNs in the count rate matrix with zeros. "
              "Did the wavemeter stream stop too early?")
        count_rate_interpolated[nans] = 0.0

    return count_rate_interpolated
