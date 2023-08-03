import numpy as np

from pleasant.util import inv_variance_weighting, get_spectral_diffusion_rates

# TODO: try out a test fixture - maybe a measurement object with the example data?

def test_inv_variance_weighting():
    data = np.array([
        [3.1, np.nan, 1.0, 2.2, 65.1, 7.4],
        [0.1, 1, 0.2, 0.5, np.nan, 1.1]
    ]).T
    assert inv_variance_weighting(data) == (2.6952574957030997, 0.08776440549263095)


