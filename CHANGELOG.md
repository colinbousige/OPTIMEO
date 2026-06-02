# Changelog

All notable changes to this project are documented in this file.

## [1.3.2] - 2026-06-02

### Added
- Added editable categorical allow-lists in the Bayesian Optimization data-loading UI so accepted categories can be widened or narrowed intentionally.

### Changed
- Candidate tables now preserve categorical values as text instead of showing Ax one-hot internal names or encoded numbers.
- Sensitivity analysis labels now map one-hot categorical features back to human-readable category values.
- The BO page now skips prediction rendering when no fitted model is available and shows an explicit bootstrap message instead.

### Fixed
- Prevented `DataRequiredError` and `No feasible points are in the search space` failures from crashing the BO workflow when the current session cannot fit a model.
- Added no-model fallbacks for candidate suggestions, best-parameter display, and model plots.

### Internal
- Bumped package/version metadata and user-facing release references to `1.3.2`.

## [1.3.1] - 2026-06-01

### Added
- Added `objective_thresholds` support to `BOExperiment`, passed through to Ax `ObjectiveProperties` for multi-objective reference-point configuration.

### Changed
- Updated notebook preprocessing in docs build to enforce responsive Plotly rendering in published example pages.
- Updated app/resource path handling to use package-relative resolution across Streamlit pages for cloud-safe file loading.

### Fixed
- Fixed Streamlit Cloud `FileNotFoundError` issues caused by cwd-relative resource paths (`style.css`, icons, logo, figures).
- Suppressed recurring matplotlib/pyparsing deprecation warnings in docs notebook execution.
- Added backward-compatible docs publishing behavior and notebook rendering improvements in the docs pipeline.

### Internal
- Bumped package/version metadata and user-facing release references to `1.3.1`.

## [1.3.0] - 2026-06-01

### Added
- Editable Bayesian Optimization data table in the Data Loading tab, with inline cell editing and dynamic row management before modeling.

### Changed
- Harmonized installation guidance to make `uv` the recommended workflow consistently across README and generated docs.
- Updated docs build workflow to include the root `optimeo` module page so published API docs stay aligned with package-level usage guidance.

### Fixed
- Default CLI launch now disables Streamlit file watching unless explicitly overridden, avoiding `torch.classes` watcher crashes in some local environments.

### Internal
- Bumped package/version metadata and user-facing release references to `1.3.0`.

## [1.2.2b0] - 2026-05-31

### Fixed
- Updated the launcher so `optimeo` starts the packaged Streamlit app from `optimeo/app/Home.py`.

### Internal
- Prepared a pre-release after the `1.2.2` PyPI publish to validate the launcher fix without reusing the existing release files.

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
