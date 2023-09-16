import pytest
import numpy as np
from pleasant.measurement import Measurement

@pytest.fixture
def zero_measurement():
    count_rate = np.zeros((10, 100))
    exc_freq = np.arange(100)
    return Measurement(count_rate, exc_freq)

@pytest.fixture
def zero_measurement_kwargs():
    count_rate = np.zeros((10, 100))
    exc_freq = np.arange(100)
    return Measurement(count_rate, exc_freq, scan_duration=1.0)

### Measurement instantiation

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

### Measurement methods and attributes

def test_scan_count(zero_measurement):
    assert zero_measurement.scan_count == 10

def test_bin_count(zero_measurement):
    assert zero_measurement.bin_count == 100

def test_freq_range(zero_measurement):
    assert zero_measurement.freq_range == 99

def test_scan_direction(zero_measurement):
    assert zero_measurement.scan_direction == 1

def test_scan_direction_monotonically():
    measurement = Measurement(np.zeros((2, 10)), np.zeros(10))
    with pytest.raises(AssertionError):
        _ = measurement.scan_direction

def test_scan_speed(zero_measurement_kwargs):
    assert zero_measurement_kwargs.scan_speed == 99.0

def test_scan_speed_nan(zero_measurement):
    assert np.isnan(zero_measurement.scan_speed)

def test_bin_width(zero_measurement):
    assert zero_measurement.bin_width == 0.99

def test_print_info(zero_measurement, capsys):
    zero_measurement.print_info()
    captured = capsys.readouterr()
    assert 'Measurement' in captured.out

def test_rebin_data_bins_to_merge(zero_measurement):
    zero_measurement.rebin_data(bins_to_merge=2)
    assert zero_measurement.bin_count == 50

def test_rebin_data_bins_to_merge_remainder(zero_measurement, capsys):
    zero_measurement.rebin_data(bins_to_merge=3, verbose=True)
    captured = capsys.readouterr()
    assert 'trimming 1 bin(s)' in captured.out

def test_rebin_data_target_bin_width(zero_measurement):
    zero_measurement.rebin_data(target_bin_width=2)
    assert zero_measurement.bin_count == 50
