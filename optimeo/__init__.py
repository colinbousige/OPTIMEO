"""OPTIMEO: Bayesian optimization, design of experiments, and data analysis tools.

[![PyPI version](https://img.shields.io/pypi/v/optimeo.svg)](https://pypi.org/project/optimeo/)

OPTIMEO helps scientists and engineers optimize experimental processes through:

- Design of Experiments (DoE)
- Bayesian Optimization (BO)
- Data analysis and regression/model interpretation

## Installation

Recommended install with ``uv``:

    uv venv .venv --python 3.10
    source .venv/bin/activate
    uv pip install optimeo

Alternative install with ``pip``:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install optimeo

Install from GitHub:

    python -m pip install "git+https://github.com/colinbousige/OPTIMEO.git"

## Usage

Launch the Streamlit app:

    optimeo

If needed, disable Streamlit file watcher issues explicitly:

    optimeo --server.fileWatcherType=none

You can also use the hosted app at https://optimeo.streamlit.app/.

## API Overview

- ``optimeo.doe``: design of experiments utilities and classes
- ``optimeo.bo``: Bayesian optimization workflow and helpers
- ``optimeo.analysis``: data analysis and machine learning modeling

For full API pages, use the docs sidebar or direct module pages:

- ``optimeo/analysis.html``
- ``optimeo/bo.html``
- ``optimeo/doe.html``
"""

__version__ = "1.3.1"

__all__ = ["__version__"]