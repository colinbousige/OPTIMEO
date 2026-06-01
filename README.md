[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15308437.svg)](https://doi.org/10.5281/zenodo.15308437)
[![paper](https://joss.theoj.org/papers/5df5fbe4e131d230b13fb3c98db545d8/status.svg)](https://joss.theoj.org/papers/5df5fbe4e131d230b13fb3c98db545d8)
[![PyPI version](https://img.shields.io/pypi/v/optimeo.svg)](https://pypi.org/project/optimeo/)
[![Tests](https://github.com/colinbousige/OPTIMEO/actions/workflows/python-app.yml/badge.svg)](https://github.com/colinbousige/OPTIMEO/actions/workflows/python-app.yml)
[![Documentation](https://github.com/colinbousige/OPTIMEO/actions/workflows/build-docs.yml/badge.svg)](https://colinbousige.github.io/OPTIMEO/)


# OPTIMEO – Bayesian Optimization Web App for Process Tuning, Modeling, and Orchestration

<p align="center">
    <img src="https://raw.githubusercontent.com/colinbousige/OPTIMEO/main/optimeo/app/resources/logo.png" alt="OPTIMEO logo" width="320" />
</p>

## About this package

[OPTIMEO](https://optimeo.streamlit.app/) is a package doubled by a web application that helps you optimize your experimental process by generating a Design of Experiment, generating new experiments using Bayesian Optimization (BO), and analyzing the results of your experiments using Machine Learning models.
The OPTIMEO package is aimed at helping scientists of any field to reach the optimum parameters of their process using the minimum amount of resources and effort.
Therefore, it is based on BO for its data efficiency: when each experiment might take one or more day to run and characterize, it is much preferable to use BO to determine which parameters to use to minimize the number of experiments to run.

This package was developed within the frame of an academic research project, MOFSONG, funded by the French National Research Agency (N° ANR-24-CE08-7639). See the related paper reference in [How to cite](#how-to-cite).

## Documentation

The package documentation is available [here](https://colinbousige.github.io/OPTIMEO/).
It is generated automatically by the GitHub Actions workflow on pushes to `main` and published to GitHub Pages.

## Changelog

Release notes are available in [CHANGELOG.md](CHANGELOG.md).  
Current release: **v1.3.1**.

## Installation

### Installing the package

Installing the package and its dependencies should take up about 1.3 GB on your hard disk, the main "heavy" dependencies being `botorch`, `scikit_learn`, `plotly`, `scipy`, `pandas` and `streamlit`.

#### Recommended: install from PyPI with uv

```bash
uv venv .venv --python 3.10
source .venv/bin/activate # Linux / macOS
uv pip install optimeo
```

This installs OPTIMEO from PyPI and registers the `optimeo` command in your environment.

#### Alternative: install from PyPI with pip

```bash
python -m venv .venv
source .venv/bin/activate # Linux / macOS
python -m pip install --upgrade pip
python -m pip install optimeo
```

#### Install from GitHub

To install the latest code directly from the repository, use:

```bash
python -m venv .venv
source .venv/bin/activate # Linux / macOS
python -m pip install "git+https://github.com/colinbousige/OPTIMEO.git"
```

You can upgrade or uninstall when using pip:

```bash
python -m pip install --upgrade optimeo
pip uninstall optimeo
```

### Launching the web app

After installation, run:

```bash
optimeo
```

If the command is not found, activate the environment first:

```bash
source .venv/bin/activate # Linux / macOS
optimeo
```

If you encounter Streamlit watcher errors related to `torch.classes` (for example `Examining the path of torch.classes raised`), OPTIMEO now starts Streamlit with file watching disabled by default. To override this behavior manually, run:

```bash
optimeo --server.fileWatcherType=auto
```

You can also use the hosted app directly: [https://optimeo.streamlit.app/](https://optimeo.streamlit.app/). Local execution is recommended for larger datasets.

## Usage

### With the web app

You can use the app on [Streamlit.io](https://optimeo.streamlit.app/) or run it locally (see [Installation](#installation)). Local execution is recommended if you process many rows or use heavier BO/modeling tasks.

Choose the page you want to use in the sidebar, and follow the instructions. Hover the mouse on the question marks to get more information about the parameters.

**1. Design of Experiment:**  
Generate a Design of Experiment (DoE) for the optimization of your process. Depending on the number of factors and levels, you can choose between different types of DoE, such as Sobol sequence, Full Factorial, Fractional Factorial, or Definitive Screening Design.

**2. New experiments using Bayesian Optimization:**  
From a previous set of experiments and their results, generate a new set of experiments to optimize your process. You can define up to 10 outcomes. Any subset of outcomes can be marked as optimization objectives (maximize/minimize), and the others can be used as constraints.

The BO page also provides model interpretation plots, including Ax Sensitivity Analysis.

**3. Data analysis and modeling:**  
Analyze the results of your experiments and model the response of your process.

### Quick local workflow from a clone

Create the `.venv` environment, activate it, install the project, then launch the app:

```bash
git clone https://github.com/colinbousige/OPTIMEO.git
cd OPTIMEO
python -m venv .venv
source .venv/bin/activate # Linux / macOS
python -m pip install --upgrade pip
python -m pip install -e .
optimeo
```

If you prefer uv:

```bash
git clone https://github.com/colinbousige/OPTIMEO.git
cd OPTIMEO
uv venv .venv --python 3.10
source .venv/bin/activate # Linux / macOS
uv sync
optimeo
```

### With the Python package

You might want to use the app as a Python package in order to integrate it in your own code, or to automate some tasks. For example:

- you are maybe using a robotic platform to run your experiments and characterize your results, and you want to use Bayesian Optimization to suggest new experiments to run automatically
- you are running a simulation and you want to optimize its parameters using Design of Experiment and Bayesian Optimization.

You can also use the app as a Python package (see [Installation](#installation)). You can import the different modules of the app and use them in your own code. Here is an example of how to use the app as a package:

#### For Design of Experiment

A more detailed example is given in [the notebook](https://colab.research.google.com/github/colinbousige/OPTIMEO/blob/main/notebooks/doe.ipynb).

```python
from optimeo.doe import * 
parameters = [
    {'name': 'Temperature', 'type': 'integer', 'values': [20, 40]},
    {'name': 'Pressure', 'type': 'float', 'values': [1, 2, 3]},
    {'name': 'Catalyst', 'type': 'categorical', 'values': ['A', 'B', 'C']}
]
doe = DesignOfExperiments(
    type='Sobol sequence',
    parameters=parameters,
    Nexp=8
)
doe
```

#### For Bayesian Optimization

A more detailed example is given in [the notebook](https://colab.research.google.com/github/colinbousige/OPTIMEO/blob/main/notebooks/bo.ipynb).

```python
from optimeo.bo import * 

features, outcomes = read_experimental_data('experimental_data.csv', out_pos=[-1])
bo = BOExperiment(
    features=features, 
    outcomes=outcomes,
    N = 2, # number of new points to generate
    maximize=True, # we want to maximize the response
    fixed_features=None, 
    feature_constraints=None, 
    optim = 'bo'
)
bo.suggest_next_trials()
```

#### For Data Analysis

A more detailed example is given in [the notebook](https://colab.research.google.com/github/colinbousige/OPTIMEO/blob/main/notebooks/MLanalysis.ipynb).

```python
from optimeo.analysis import * 

data = pd.read_csv('dataML.csv')
factors = data.columns[:-1].tolist()
response = data.columns[-1]
analysis = DataAnalysis(data, factors, response)
analysis.model_type = "ElasticNetCV"
MLmodel = analysis.compute_ML_model()
figs = analysis.plot_ML_model()
for fig in figs:
    fig.show()
```

## Support

This app was made by [Colin Bousige](mailto:colin.bousige@cnrs.fr). Contact me for support or to signal a bug, or leave a message on the [GitHub page of the app](https://github.com/colinbousige/OPTIMEO).

## How to cite

This work has been published in the article "OPTIMEO: Bayesian Optimization Web App for Process Tuning, Modeling, and Orchestration", C. Bousige, [J. Open Source Softw. **10**, 115 (2025), 8510](https://joss.theoj.org/papers/10.21105/joss.08510). Please cite this paper if you publish using this code:

```bibtex
@article{bousige_optimeo_2025,
  title = {{{OPTIMEO}}: {{Bayesian Optimization Web App}} for {{Process Tuning}}, {{Modeling}}, and {{Orchestration}}},
  author = {Bousige, Colin},
  year = 2025,
  journal = {J. Open Source Softw.},
  volume = {10},
  number = {115},
  pages = {8510},
  doi = {10.21105/joss.08510}
}
```

## Acknowledgements

This work was supported by the French National Research Agency (N° ANR-24-CE08-7639).  
Also, this work was made possible thanks to the following open-source projects:

- [ax](https://ax.dev/)
- [BoTorch](https://botorch.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [pyDOE3](https://github.com/relf/pyDOE3)
- [dexpy](https://statease.github.io/dexpy/)
- [doepy](https://doepy.readthedocs.io/en/latest/)
- [definitive-screening-design](https://pypi.org/project/definitive-screening-design/)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
