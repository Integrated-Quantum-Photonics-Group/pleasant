import pytest
import numpy as np
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
from pleasant import fitting

@pytest.fixture
def gauss_params():
    return GaussianModel().make_params()

@pytest.fixture
def lorentz_params():
    return LorentzianModel().make_params()

@pytest.fixture
def voigt_params():
    # TODO: tests of Voigt functions not working so far
    return VoigtModel().make_params()

def _round(x):
    return round(x, 7)

def test_gauss_sigma(gauss_params):
    fwhm = gauss_params['fwhm'].value
    sigma = _round(fitting.gauss_sigma(fwhm))
    # rounding is necessary
    assert sigma == gauss_params['sigma'].value

def test_gauss_fwhm(gauss_params):
    sigma = gauss_params['sigma'].value
    fwhm = _round(fitting.gauss_fwhm(sigma))
    # rounding is necessary
    assert fwhm == gauss_params['fwhm'].value

def test_lorentz_sigma(lorentz_params):
    fwhm = lorentz_params['fwhm'].value
    sigma = _round(fitting.lorentz_sigma(fwhm))
    # rounding is necessary
    assert sigma == lorentz_params['sigma'].value

def test_lorentz_fwhm(lorentz_params):
    sigma = lorentz_params['sigma'].value
    fwhm = _round(fitting.lorentz_fwhm(sigma))
    # rounding is necessary
    assert fwhm == lorentz_params['fwhm'].value

def test_gauss_amplitude(gauss_params):
    height = gauss_params['height'].value
    sigma = gauss_params['sigma'].value
    amplitude = _round(fitting.gauss_amplitude(height, sigma))
    assert amplitude == gauss_params['amplitude'].value

def test_gauss_height(gauss_params):
    amplitude = gauss_params['amplitude'].value
    sigma = gauss_params['sigma'].value
    height = _round(fitting.gauss_height(amplitude, sigma))
    assert height == gauss_params['height'].value

def test_lorentz_amplitude(lorentz_params):
    height = lorentz_params['height'].value
    sigma = lorentz_params['sigma'].value
    amplitude = _round(fitting.lorentz_amplitude(height, sigma))
    assert amplitude == lorentz_params['amplitude'].value

def test_lorentz_height(lorentz_params):
    amplitude = lorentz_params['amplitude'].value
    sigma = lorentz_params['sigma'].value
    height = _round(fitting.lorentz_height(amplitude, sigma))
    assert height == lorentz_params['height'].value

def test_max_within_fwhm():
    x = 0.5 * np.arange(10)
    y = np.zeros(10)
    y_max = 1
    y[2] = y_max
    assert fitting.max_within_fwhm(x, y, 2, 3) == y_max

def test_max_within_fwhm_value_error():
    x = np.arange(10)
    assert np.isnan(fitting.max_within_fwhm(x, x, 12, 1))
