# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- More specific `rebin` and `rebin_to_width` methods to `Measurement`, replacing `rebin_data`.

### Deprecated

- Method `rebin_data` of `Measurement` class

### Fixed

- Default value of `break_duration` in `qudi` loader is now `NaN`.

## [1.1.0] - 2023-09-16

### Added

- Extensive unit tests.
- A `nox` file for test automation.
- Improved documentation in README and example notebook.
- More advanced filter `peak_window_filter`.
- Argument `scan_index_range` to `fit_sum_of_scans` and `plot_sum_of_scans`.

### Changed

- Both `Measurement.rebin_data` and `util.get_spectral_diffusion_rates` are non-verbose by default, added keyword argument to make verbose.
- Upgrade dependencies: `pandas`, `scipy`
- Use `NaN` as default for `scan_duration` and `break_duration`.
- Assert that `scan_duration` is not `NaN` where required.
- Improve type checking in `Measurement.__init__`.
- `fit_sum_of_scans` now returns the fit result instead of saving it in an attribute.

### Fixed

- Add missing brackets in `gauss_height`.
- Minor bug in `scan_direction` property.

### Removed

- Attribute `sum_fit_result` of `Measurement`.

## [1.0.0] - 2023-01-24

### Added

- Upload code to GitHub.

[unreleased]: https://github.com/Integrated-Quantum-Photonics-Group/pleasant/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Integrated-Quantum-Photonics-Group/pleasant/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Integrated-Quantum-Photonics-Group/pleasant/releases/tag/v1.0.0
