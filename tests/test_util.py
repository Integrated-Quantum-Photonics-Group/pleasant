import numpy as np

from pleasant.util import inv_variance_weighting, get_spectral_diffusion_rates

# TODO: try out a test fixture - maybe a measurement object with the example data?

def test_inv_variance_weighting():
    """ Compute the inverse variance weighted average in a simple case. """
    n = 10
    y = 1.0
    y_std = 0.5
    data = np.zeros((n, 2))
    data[:, 0] = y
    data[:, 1] = y_std
    weighted_avg = y
    weighted_std = y_std * np.sqrt(1 / n)
    assert inv_variance_weighting(data) == (weighted_avg, weighted_std)
