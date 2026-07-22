"""OPTIMEO: Bayesian optimization, design of experiments, and data analysis tools.
================================================================================

.. image:: https://img.shields.io/pypi/v/optimeo
   :alt: PyPI - Version
   :target: https://pypi.org/project/optimeo/

OPTIMEO helps scientists and engineers optimize experimental processes through:

* Design of Experiments (DoE)
* Bayesian Optimization (BO)
* Data analysis and regression/model interpretation

Installation
------------

Recommended install with ``uv``:

.. code-block:: bash

    uv venv .venv --python 3.11
    source .venv/bin/activate
    uv pip install optimeo

Alternative install with ``pip``:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install optimeo

Install from GitHub:

.. code-block:: bash

    python -m pip install "git+https://github.com/colinbousige/OPTIMEO.git"

Usage
-----

Launch the Streamlit app:

.. code-block:: bash

    optimeo

If needed, disable Streamlit file watcher issues explicitly:

.. code-block:: bash

    optimeo --server.fileWatcherType=none

You can also use the hosted app at `https://optimeo.streamlit.app/ <https://optimeo.streamlit.app/>`_.

API Overview
------------

* ``optimeo.doe``: design of experiments utilities and classes
* ``optimeo.bo``: Bayesian optimization workflow and helpers
* ``optimeo.analysis``: data analysis and machine learning modeling

For full API pages, use the docs sidebar or direct module pages:

* `optimeo/analysis.html <optimeo/analysis.html>`_
* `optimeo/bo.html <optimeo/bo.html>`_
* `optimeo/doe.html <optimeo/doe.html>`_
"""

__version__ = "1.4.0"

__all__ = ["__version__"]
