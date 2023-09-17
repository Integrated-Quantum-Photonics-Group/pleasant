"""A loader for data from qudi (https://github.com/Ulm-IQO/qudi).

Specifically, data from the laser scanner module can be loaded.
"""

from os import listdir

import numpy as np
from scipy.constants import speed_of_light

from pleasant.measurement import Measurement


__all__ = ["get_descriptions_in_folder", "load_qudi_folder"]


def get_descriptions_in_folder(folder_name: str) -> list[str]:
    """Get all unique measurements descriptions of the qudi data files in a folder.

    :param folder_name: directory where the data files are located
    :return: list of unique descriptions
    """
    filename_stubs = _get_filename_stubs(folder_name)
    descriptions = {i.split("_", 1)[1] for i in filename_stubs}
    return list(descriptions)


def load_qudi_folder(
    folder_name: str,
    description_contains: str = "",
    break_duration: float = np.nan,
) -> list[Measurement]:
    """Load all qudi data files in folder into Measurement objects.

    There are four files per measurement pair:
        - a _volt_data_raw_trace.dat file containing a 2D matrix of the count rates
        registered during all scans
        - a _wl_data_trace.dat file containing the wavemeter readings before
        and after each scan
        - retrace versions of the above containing the same data but for scans
        in the opposite spectral direction
    :param folder_name: directory where the data files are located
    :param description_contains: filenames that do not contain this string
    in the description part will be ignored
    :param break_duration: argument to be passed on to the Measurement object creator
    :return: list of Measurement objects
    """
    filename_stubs = [
        i for i in _get_filename_stubs(folder_name) if description_contains in i
    ]
    if not filename_stubs:
        print(
            "No file containing the description {} found.".format(description_contains)
        )
        return []

    _measurements = []
    for filename_stub in filename_stubs:
        # the timestamp is at the beginning of a filename and followed by an underscore
        # split at first underscore
        timestamp, description = filename_stub.split("_", 1)
        filename_stub_with_folder = folder_name + "/" + filename_stub
        for which in ["trace", "retrace"]:
            rate, exc_freq, scan_duration = _read_data_files(
                filename_stub_with_folder, which=which
            )
            _m = Measurement(
                rate,
                exc_freq,
                timestamp=timestamp,
                description=description,
                scan_duration=scan_duration,
                break_duration=break_duration,
            )
            _measurements.append(_m)

    return _measurements


def _get_filename_stubs(folder_name: str) -> list[str]:
    """Generate a set of filename stubs that uniquely identify measurement pairs.

    Pairs consist of a trace and retrace measurement.
    :param folder_name: directory where the data files are located
    :return: list of filename stubs
    """
    suffix = "_volt_data_raw_trace.dat"
    filename_stubs = [
        i.removesuffix(suffix) for i in listdir(folder_name) if i.endswith(suffix)
    ]
    return filename_stubs


def _get_scan_params(filename: str) -> dict[str, float]:
    """Read header of a qudi data file and extract voltage scan parameters.

    :param filename: path to the qudi data file
    :return: dictionary containing voltage scan parameters
    """
    with open(filename) as f:
        header = [next(f) for _ in range(11)]
    p_dict = {
        "v_start": float(header[6].split(":")[1]),
        "v_stop": float(header[7].split(":")[1]),
        "v_speed": float(header[8].split(":")[1]),
    }
    return p_dict


def _read_wavemeter_file(filename: str) -> tuple[float, float]:
    """Read wavemeter file and compute average values for start and stop frequencies.

    The excitation frequency during all scans of the measurement is then assumed as
    linear and equal for all scans.
    This approximation is only appropriate if during the course of a measurement:
        - the drift of the laser from scan to scan is negligible
        - any hysteresis or other deviation from a linear behavior is negligible
        - the wavelength/frequency of the laser is read our precisely before and
        after each scan, not during a scan.
    :param filename: path to the qudi data file
    :return: average frequency before the scan, average frequency after the scan
    """
    wavelengths = np.genfromtxt(filename)

    # replace under/overexposed values with nan
    # replace zeros with nan
    wavelengths[(wavelengths <= 0).any(axis=1)] = np.nan
    unsuccessful = np.count_nonzero(np.isnan(wavelengths).any(axis=1))
    ratio = (len(wavelengths) - unsuccessful) / len(wavelengths)
    if ratio == 0:
        raise ValueError("No successful wavemeter readings.")
    elif ratio < 1:
        print(f"Not all wavemeter readings successful ({ratio:.2%}).")

    freq = speed_of_light / (1e-9 * wavelengths)
    avg_start_freq, avg_stop_freq = np.nanmean(freq, axis=0)

    return avg_start_freq, avg_stop_freq


def _read_data_files(
    filename_stub: str, which: str = "trace"
) -> tuple[np.ndarray, np.ndarray, float]:
    """Read main qudi data files containing the registered count rates during the scans.

    :param filename_stub: stub from which all qudi data files can be derived
    :param which: whether to read trace or retrace files (scan direction)
    :return: 1D or 2D count rates, 1D excitation frequencies, scan duration (in s)
    """
    # determine correct filenames depending on which argument
    if which == "trace":
        rate_file = filename_stub + "_volt_data_raw_trace.dat"
        wm_file = filename_stub + "_wl_data_trace.dat"
    elif which == "retrace":
        rate_file = filename_stub + "_volt_data_raw_retrace.dat"
        wm_file = filename_stub + "_wl_data_retrace.dat"
    else:
        raise AssertionError("which kwarg must be trace or retrace")

    count_rate = np.loadtxt(rate_file)
    if count_rate.ndim == 1:
        step_count = count_rate.size
    else:
        step_count = count_rate.shape[1]

    # determine frequency axis
    avg_start_freq, avg_stop_freq = _read_wavemeter_file(wm_file)

    # create frequency array
    freq = np.linspace(avg_start_freq, avg_stop_freq, step_count)

    # determine scan duration
    scan_params = _get_scan_params(rate_file)
    scan_duration = (
        abs(scan_params["v_start"] - scan_params["v_stop"]) / scan_params["v_speed"]
    )

    return count_rate, freq, scan_duration
