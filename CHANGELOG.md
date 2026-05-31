# Changelog

All notable changes to this project are documented in this file.

## [1.2.2] - 2026-05-31

### Fixed
- Bumped the release version so PyPI publishes a fresh build instead of reusing the already uploaded 1.2.1 files.

### Internal
- Prepared the next tagged release after the 1.2.1 publish attempt hit PyPI file reuse protection.

## [1.2.1] - 2026-05-31

### Added
- GitHub Actions workflow for trusted PyPI publishing on version tags.
- Explicit virtual environment setup instructions using `.venv` across the README, package docs, notebooks, and app home page.

### Changed
- Reorganized the Streamlit application into `optimeo/app/` so packaged installs include the app entrypoint, pages, and bundled resources.
- Updated the command-line launcher to start the packaged Streamlit app from the installed distribution.
- Refreshed installation guidance for PyPI, GitHub installs, and local clone workflows.
- Updated documentation build workflow paths to match the packaged app/resource layout.
- Test workflow now installs the package before running pytest.

### Fixed
- Corrected the package initializer so it exposes package metadata instead of duplicated README content.
- Ensured wheel and sdist artifacts include the Streamlit app assets required to run `optimeo` after installation.

### Internal
- Built and validated `optimeo-1.2.1` distributions with `python -m build` and `twine check`.

## [1.2] - 2026-05-30

### Added
- Sensitivity analysis based on Ax analysis cards, aligned with the Ax materials-science tutorial workflow.
- Visual styling updates for sensitivity plots (impact color direction, cleaner tooltips, clearer axis labels).
- Exploration vs exploitation presets for Bayesian optimization tuning (Exploit, Balanced, Explore, Custom).
- Explicit guidance in notebooks and docs for local `uv` workflow and Colab installation.

### Changed
- Bayesian optimization tuning now supports both single and batch UCB:
  - `UpperConfidenceBound` when generating one candidate.
  - `qUpperConfidenceBound` when generating batches.
- `beta` tuning now uses direct values for more intuitive behavior.
- Multi-objective optimization now supports 3 or more objectives in the same run.
- Installation and usage docs refreshed across app pages and package docs.
- Notebook examples updated to remove wildcard imports in favor of explicit imports.

### Fixed
- Replaced broad exception handling with narrower, typed exceptions in critical paths.
- Replaced wildcard imports in app pages with explicit imports.
- Replaced unsafe `eval`-based parameter parsing with `ast.literal_eval` in analysis workflow.
- Cleared stale notebook outputs to avoid confusion with outdated logs.

### Internal
- Test suite remains passing after updates.
