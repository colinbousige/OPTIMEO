# test_bo.py
import sys
import os

# Add the parent directory of optimeo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
import pandas as pd
from optimeo.bo import BOExperiment, read_experimental_data

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="botorch_model_class is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

features, outcomes = read_experimental_data("tests/data/experimental_data2.csv", out_pos=[-2, -1])

def test_initialization():
    """Test if the class initializes correctly with valid inputs."""
    experiment = BOExperiment(features, 
                              outcomes, 
                              N=2, 
                              maximize={'Yield':True, 'Price':False}
                              )
    next_trial = experiment.suggest_next_trials()
    assert isinstance(next_trial, pd.DataFrame)
    fig = experiment.plot_model(metricname='Yield')
    assert fig is not None
    fig = experiment.plot_model(metricname='Price', linear=True)
    assert fig is not None
    fig = experiment.plot_model(metricname='Yield', slice_values={'Temperature': 50})
    assert fig is not None
    fig = experiment.plot_feature_importances()
    assert fig is not None
    experiment.compute_pareto_frontier()
    fig = experiment.plot_pareto_frontier()
    assert fig is not None
    best_params = experiment.get_best_parameters()
    assert isinstance(best_params, pd.DataFrame)
    with pytest.raises(ValueError):
        experiment.update_experiment({'bad_column': [1, 2]}, {'Yield': [0.5, 0.6], 'Price': [100, 200]})


def test_three_objectives():
    """Test that 3-objective optimization initializes and Pareto plotting works."""
    outcomes3 = {
        "Yield": outcomes["Yield"],
        "Price": outcomes["Price"],
        "Purity": {"type": "float", "data": [0.1 * y + 0.9 * p for y, p in zip(outcomes["Yield"]["data"], outcomes["Price"]["data"]) ]},
    }
    experiment = BOExperiment(
        features,
        outcomes3,
        N=2,
        maximize={"Yield": True, "Price": False, "Purity": True},
    )
    next_trial = experiment.suggest_next_trials()
    assert isinstance(next_trial, pd.DataFrame)
    pareto = experiment.compute_pareto_frontier()
    assert pareto is not None
    fig = experiment.plot_pareto_frontier(show_error_bars=False)
    assert fig is not None


