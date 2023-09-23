"""Test utility functions."""

import numpy as np
import pandas as pd

from pleasant.util import get_spectral_diffusion_rates, inv_variance_weighting


def test_inv_variance_weighting():
    """Inverse variance weighting works in a simple case."""
    n = 10
    y = 1.0
    y_std = 0.5
    data = np.zeros((n, 2))
    data[:, 0] = y
    data[:, 1] = y_std
    weighted_avg = y
    weighted_std = y_std * np.sqrt(1 / n)
    assert inv_variance_weighting(data) == (weighted_avg, weighted_std)


def test_spectral_diffusion_rates(shared_datadir):
    """Spectral diffusion rate computation works for example data."""
    df = pd.read_pickle(str(shared_datadir) + "/scan_fit_data.pkl")
    # does not work without timestamps and break duration
    df["timestamp"] = 0
    df["break_duration"] = 0.15
    mean_rate = get_spectral_diffusion_rates(df)[:, 0].mean()
    assert round(1e-6 * mean_rate, 2) == 319.04
