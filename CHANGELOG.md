# Changelog

All notable changes to this project are documented in this file.

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
