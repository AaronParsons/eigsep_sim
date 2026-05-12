# Repository Guidelines

## Project Structure & Module Organization

This package builds a differentiable simulator for global 21cm reionization
experiments. Source modules live in `src/` and are exposed as `eigsep_sim`.
Core code includes `simulate.py`, `sky.py`, `beam.py`, `observer.py`,
`terrain.py`, `calibrator.py`, and helpers such as `coord.py`, `healpix.py`,
and `basis.py`. Tests live in `tests/`, packaged `.npz` assets in `src/data/`,
notebooks in `notebooks/`, and performance scripts in `benchmarks/`.

## Architecture Priorities

Prioritize a correct, efficient JAX forward model with a clean API for Marjum
Canyon and lunar orbiter simulations. Preserve the split between immutable
NumPy descriptors and JAX kernels. Support HEALPix radio skies, compact
bright-source and planetary components, terrain shielding from Earth DEMs or
lunar horizons, and antenna beams rotating relative to topocentric or inertial
lunar frames.

## Build, Test, and Development Commands

- `pip install -e .` installs the package in editable mode.
- `pip install -e ".[dev]"` installs `pytest`, `pytest-cov`, `black`, and
  `flake8`.
- `pip install -e ".[all]"` installs dev and notebook dependencies.
- `pytest` runs the full test suite.
- `pytest tests/test_simulate.py` runs one focused test module.
- `pytest --cov=eigsep_sim` runs tests with package coverage.
- `black src tests` formats source and tests using the configured style.
- `flake8 src tests` checks style issues not handled by formatting.

## Coding Style & Naming Conventions

Use Python 3.8+ syntax and explicit imports. Black is configured with a
79-character line length; format touched Python files before submitting.
Use snake_case for functions, variables, and modules, and CapWords for classes
such as `ForwardModel`. Include units and frames in array names where useful,
for example `freqs_hz`, `crds_gal`, `direction_topo`, or `rot_gal2top`.

## Testing Guidelines

Tests use `pytest` and are named `test_*.py` with functions named `test_*`.
Place new tests beside related coverage in `tests/`. Use small HEALPix `nside`
values and compact arrays to keep tests fast. For numerical changes, assert
shapes, dtypes, finite values, frame conventions, and tolerances.

## Commit & Pull Request Guidelines

Recent commits use short summaries such as `Fixed...`, `Added...`, or
`Updated...`, with longer bodies for substantial behavior changes. Keep commits
focused. Pull requests should describe affected simulation, calibration, or
data behavior; list tests run; mention notebook or benchmark updates; and call
out changes to packaged data in `src/data/`.

## Agent-Specific Instructions

Avoid unrelated refactors and do not regenerate notebooks unless requested.
Treat API clarity and JAX performance as first-class review criteria. Do not
modify packaged `.npz` data without explaining provenance and expected
downstream effects.
