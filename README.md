# PLEasant

A package aimed at making the analysis of [photoluminescence excitation](https://en.wikipedia.org/wiki/Photoluminescence_excitation) (PLE)
measurement data a pleasant experience.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/Integrated-Quantum-Photonics-Group/pleasant/workflows/Tests/badge.svg)](https://github.com/Integrated-Quantum-Photonics-Group/pleasant/actions?workflow=Tests)
[![Docs](https://github.com/Integrated-Quantum-Photonics-Group/pleasant/workflows/Docs/badge.svg)](https://github.com/Integrated-Quantum-Photonics-Group/pleasant/actions?workflow=Docs)

## Features

- independent and extendable data loading functions
- convenient handling of measurement metadata with the dedicated [`Measurement`](src/pleasant/measurement.py) class
  - access to measurement metadata (scan speed, scan range, user-defined description, ...)
  - rebinning of data
  - simple photon-count threshold filtering
- typical PLE analysis routines, e.g. for the extraction of homogeneous and inhomogeneous linewidths
  - fitting and plotting a sum of all PLE scans with [lmfit](https://lmfit.github.io/lmfit-py/) and [matplotlib](https://matplotlib.org)
  - fitting of individual PLE scans with a peak-like model (Gaussian, Lorentzian, Voigt, Pseudo-Voigt)
  - plotting of scans with or without fit
  - straight-forward export of scan fit results as a [pandas](https://pandas.pydata.org) dataframe retaining all measurement metadata
- computation of time-normalized spectral jumps for extraction of the spectral diffusion rate

## Installation and Documentation

PLEasant is provided as a pip-installable Python package.
Take a look at the [documentation](https://integrated-quantum-photonics-group.github.io/pleasant)
for details. There is also a [demo notebook](examples/demo.ipynb)
to get you started quickly.

## Attribution

If you are publishing any work based on using PLEasant as an analysis tool,
please mention it e.g. in the methods section
and consider citing the original scientific work that this package was written for:

[Optically coherent nitrogen-vacancy defect centers in diamond nanostructures](https://doi.org/10.1103/PhysRevX.13.011042)
