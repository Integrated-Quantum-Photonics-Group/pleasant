import lmfit
import numpy as np


# define models with constant background
gaussian = lmfit.models.GaussianModel() + lmfit.models.ConstantModel()
lorentzian = lmfit.models.LorentzianModel() + lmfit.models.ConstantModel()
pseudo_voigt = lmfit.models.PseudoVoigtModel() + lmfit.models.ConstantModel()
voigt = lmfit.models.VoigtModel() + lmfit.models.ConstantModel()


def gauss_sigma(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def lorentz_sigma(fwhm):
    return fwhm / 2


def voigt_sigma(fwhm):
    return fwhm / 3.6013  # approximate


def gauss_fwhm(sigma):
    return sigma * (2 * np.sqrt(2 * np.log(2)))


def lorentz_fwhm(sigma):
    return sigma * 2


def gauss_amplitude(height, sigma):
    return height * sigma * np.sqrt(2 * np.pi)


def gauss_height(amplitude, sigma):
    return amplitude / (sigma * np.sqrt(2 * np.pi))


def lorentz_amplitude(height, sigma):
    return height * np.pi * sigma


def lorentz_height(amplitude, sigma):
    return amplitude / (np.pi * sigma)


def voigt_amplitude(height, sigma):
    return gauss_amplitude(height, sigma)


def voigt_height(amplitude, sigma):
    return gauss_height(amplitude, sigma)


def max_within_fwhm(x, y, x_c, fwhm):
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
