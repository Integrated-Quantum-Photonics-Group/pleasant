"""Routines for post-processing and analyzing PLE experiment data."""

from importlib.metadata import version

from .util import get_spectral_diffusion_rates, inv_variance_weighting

__all__ = ["inv_variance_weighting", "get_spectral_diffusion_rates"]

__version__ = version(__name__)
