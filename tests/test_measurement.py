import numpy as np
import pandas as pd
import pytest

from pleasant.measurement import Measurement


@pytest.fixture
def zero_measurement() -> Measurement:
    count_rate = np.zeros((10, 100))
    exc_freq = np.arange(100)
    return Measurement(count_rate, exc_freq)


@pytest.fixture
def measurement_with_data(datadir) -> Measurement:
    count_rate = np.load(str(datadir) + "/count_rate.npy")
    exc_freq = np.load(str(datadir) + "/exc_freq.npy")
    scan_duration = 1.5
    return Measurement(count_rate, exc_freq, scan_duration=scan_duration)


@pytest.fixture
def measurement_with_fits(measurement_with_data) -> Measurement:
    measurement_with_data.peak_window_filter()
    # fit only 10 lines
    measurement_with_data.photon_count_mask[10:] = False
    measurement_with_data.fit_scans()
    return measurement_with_data


# Measurement instantiation


def test_count_rate_1d():
    measurement = Measurement(np.zeros(10), np.zeros(10))
    assert measurement.count_rate.ndim == 2


def test_count_rate_not_2d():
    with pytest.raises(AssertionError) as exc:
        _ = Measurement(np.zeros((3, 3, 10)), np.zeros(10))
    assert "should be a 1D or 2D numpy array" in str(exc.value)


def test_count_rate_not_array():
    with pytest.raises(AssertionError) as exc:
        _ = Measurement(None, np.zeros(10))
    assert "should be a numpy array" in str(exc.value)


def test_exc_freq_not_array():
    with pytest.raises(AssertionError) as exc:
        _ = Measurement(np.zeros((2, 10)), None)
    assert "should be a numpy array" in str(exc.value)


def test_exc_freq_not_1d():
    with pytest.raises(AssertionError) as exc:
        _ = Measurement(np.zeros((2, 10)), np.zeros((2, 10)))
    assert "should be a 1D numpy array" in str(exc.value)


def test_count_rate_exc_freq_match():
    with pytest.raises(AssertionError) as exc:
        _ = Measurement(np.zeros((2, 10)), np.zeros(9))
    assert "should have the same bin count" in str(exc.value)


# Measurement methods and attributes


def test_scan_count(zero_measurement: Measurement):
    assert zero_measurement.scan_count == 10


def test_bin_count(zero_measurement: Measurement):
    assert zero_measurement.bin_count == 100


def test_freq_range(zero_measurement: Measurement):
    assert zero_measurement.freq_range == 99


def test_scan_direction(zero_measurement: Measurement):
    assert zero_measurement.scan_direction == 1


def test_scan_direction_monotonically():
    measurement = Measurement(np.zeros((2, 10)), np.zeros(10))
    with pytest.raises(AssertionError):
        _ = measurement.scan_direction


def test_scan_speed(zero_measurement: Measurement):
    zero_measurement.scan_duration = 1
    assert zero_measurement.scan_speed == 99.0


def test_scan_speed_nan(zero_measurement: Measurement):
    assert np.isnan(zero_measurement.scan_speed)


def test_bin_width(zero_measurement: Measurement):
    assert zero_measurement.bin_width == 0.99


def test_print_info(zero_measurement: Measurement, capsys):
    zero_measurement.print_info()
    captured = capsys.readouterr()
    assert "Measurement" in captured.out


def test_rebin(zero_measurement: Measurement):
    zero_measurement.rebin(bins_to_merge=2)
    assert zero_measurement.bin_count == 50


def test_rebin_remainder(zero_measurement: Measurement, capsys):
    zero_measurement.rebin(bins_to_merge=3, verbose=True)
    captured = capsys.readouterr()
    assert "trimming 1 bin(s)" in captured.out


def test_rebin_to_width(zero_measurement: Measurement):
    zero_measurement.rebin_to_width(target_bin_width=2)
    assert zero_measurement.bin_count == 50

def test_rebin_data_deprecated(zero_measurement: Measurement):
    with pytest.deprecated_call():
        zero_measurement.rebin_data(bins_to_merge=2)


# Tests with real data


def test_fit_sum_of_scans(measurement_with_data: Measurement):
    res = measurement_with_data.fit_sum_of_scans()
    measurement_with_data.plot_sum_of_scans()
    fwhm = res.params["fwhm"].value
    assert round(1e-9 * fwhm, 2) == 4.35


def test_fit_sum_of_scans_range(measurement_with_data: Measurement):
    res = measurement_with_data.fit_sum_of_scans(scan_index_range=(0, 100))
    fwhm = res.params["fwhm"].value
    assert round(1e-9 * fwhm, 2) == 5.05


def test_plot_sum_of_scans(measurement_with_data: Measurement):
    fig = measurement_with_data.plot_sum_of_scans()
    # TODO this can be more precise
    assert len(fig.axes) == 4


def test_photon_count_filter(measurement_with_data: Measurement):
    mask = measurement_with_data.photon_count_filter(10)
    assert mask.sum() == 237


def test_peak_window_filter(measurement_with_data: Measurement):
    mask = measurement_with_data.peak_window_filter(min_snr=3, window=10)
    assert mask.sum() == 336


def test_fit_scans(measurement_with_fits: Measurement):
    n_mask = measurement_with_fits.photon_count_mask.sum()
    n_fit_result = 0
    for i in measurement_with_fits.scan_fit_results:
        if i is not None:
            n_fit_result += 1
    assert n_fit_result == n_mask

    fit_result = measurement_with_fits.scan_fit_results[0]
    assert fit_result is not None

    fwhm = fit_result.params["fwhm"].value
    assert round(fwhm, 3) == 0.043


def test_fit_scans_gaussian(measurement_with_fits: Measurement):
    measurement_with_fits.fit_scans(model_name="Gaussian")
    fit_result = measurement_with_fits.scan_fit_results[0]
    assert fit_result is not None
    fwhm = fit_result.params["fwhm"].value
    assert round(fwhm, 3) == 0.071


def test_fit_scans_voigt(measurement_with_fits: Measurement):
    measurement_with_fits.fit_scans(model_name="Voigt")
    fit_result = measurement_with_fits.scan_fit_results[0]
    assert fit_result is not None
    fwhm = fit_result.params["fwhm"].value
    assert round(fwhm, 3) == 0.071


def test_fit_scans_pseudo_voigt(measurement_with_fits: Measurement):
    measurement_with_fits.fit_scans(model_name="Pseudo Voigt")
    fit_result = measurement_with_fits.scan_fit_results[0]
    assert fit_result is not None
    fwhm = fit_result.params["fwhm"].value
    assert round(fwhm, 3) == 0.071


def test_plot_individual_scan(measurement_with_fits: Measurement):
    print(measurement_with_fits.scan_fit_results)
    fig = measurement_with_fits.plot_individual_scan(0)
    assert len(fig.axes[0].lines) == 4


def test_plot_individual_scan_no_fit(measurement_with_fits: Measurement):
    fig = measurement_with_fits.plot_individual_scan(15)
    assert len(fig.axes[0].lines) == 1


def test_scan_fit_data(measurement_with_fits: Measurement, shared_datadir):
    df = measurement_with_fits.scan_fit_data
    df = df[:10]
    df_ref = pd.read_pickle(str(shared_datadir) + "/scan_fit_data.pkl")
    pd.testing.assert_frame_equal(df, df_ref)
