"""Custom fitting models and related functions."""

import lmfit
import numpy as np


# define models with constant background
gaussian = lmfit.models.GaussianModel() + lmfit.models.ConstantModel()
lorentzian = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()
pseudo_voigt = lmfit.models.PseudoVoigtModel() + lmfit.models.ConstantModel()
voigt = lmfit.models.VoigtModel() + lmfit.models.ConstantModel()


def gauss_sigma(fwhm: float) -> float:
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def lorentz_sigma(fwhm: float) -> float:
    return fwhm / 2


def voigt_sigma(fwhm: float) -> float:
    return fwhm / 3.6013  # approximate


def gauss_fwhm(sigma: float) -> float:
    return sigma * (2 * np.sqrt(2 * np.log(2)))


def lorentz_fwhm(sigma: float) -> float:
    return sigma * 2


def gauss_amplitude(height: float, sigma: float) -> float:
    return height * sigma * np.sqrt(2 * np.pi)


def gauss_height(amplitude: float, sigma: float) -> float:
    return amplitude / (sigma * np.sqrt(2 * np.pi))


def lorentz_amplitude(height: float, sigma: float) -> float:
    return height * np.pi * sigma


def lorentz_height(amplitude: float, sigma: float) -> float:
    return amplitude / (np.pi * sigma)


def voigt_amplitude(height: float, sigma: float) -> float:
    return gauss_amplitude(height, sigma)


def voigt_height(amplitude: float, sigma: float) -> float:
    return gauss_height(amplitude, sigma)


def max_within_fwhm(x: np.ndarray, y: np.ndarray, x_c: float, fwhm: float) -> float:
    """
    Determine maximum of y within an area of width fwhm around x_c
    :param x: x values
    :param y: y values
    :param x_c: center around which to find the maximum
    :param fwhm: width in which to find the maximum
    :return: maximum within fwhm or nan if empty
    """
    x_min = x_c - (fwhm / 2)
    x_max = x_c + (fwhm / 2)
    x_within_peak = np.logical_and(x > x_min, x < x_max)
    try:
        return y[x_within_peak].max()
    except ValueError:
        # if data[x_within_peak] is empty
        return np.nan
