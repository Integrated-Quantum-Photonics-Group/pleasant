import pytest
import numpy as np
from pleasant.measurement import Measurement

@pytest.fixture
def zero_measurement():
    count_rate = np.zeros((10, 100))
    exc_freq = np.arange(100)
    return Measurement(count_rate, exc_freq)

def test_scan_count(zero_measurement):
    assert zero_measurement.scan_count == 10

def test_bin_count(zero_measurement):
    assert zero_measurement.bin_count == 100

def test_freq_range(zero_measurement):
    assert zero_measurement.freq_range == 99

def test_scan_direction(zero_measurement):
    assert zero_measurement.scan_direction == 1

def test_bin_width(zero_measurement):
    assert zero_measurement.bin_width == 0.99

def test_rebin_data_bins_to_merge(zero_measurement):
    zero_measurement.rebin_data(bins_to_merge=2)
    assert zero_measurement.bin_count == 50

def test_rebin_data_target_bin_width(zero_measurement):
    zero_measurement.rebin_data(target_bin_width=2)
    assert zero_measurement.bin_count == 50
