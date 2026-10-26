# test_analysis.py
import sys
import os

# Add the parent directory of optimeo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
import numpy as np
import pandas as pd
from optimeo.analysis import DataAnalysis

data = pd.read_csv("tests/data/dataML.csv")

models_list = ["ElasticNetCV", "RidgeCV", "LinearRegression", 
               "RandomForest", "GaussianProcess", "GradientBoosting"]

def test_initialization():
    """Test if the class initializes correctly with valid inputs."""
    factors = data.columns[:-1].tolist()
    response = data.columns[-1]
    for mod in models_list:
        analysis = DataAnalysis(data, factors, response)
        analysis.model_type = mod
        MLmodel = analysis.compute_ML_model()
        analysis.encode_data()
        eq = analysis.write_equation(order=1, quadratic=[])
        assert len(eq) > 0
        linmod = analysis.compute_linear_model(order=1, quadratic=[])
        assert linmod is not None
        MLmodel = analysis.compute_ML_model()
        assert MLmodel is not None
        fig = analysis.plot_qq()
        assert fig is not None
        fig = analysis.plot_boxplot()
        assert fig is not None
        fig = analysis.plot_histogram()
        assert fig is not None
        fig = analysis.plot_scatter_response()
        assert fig is not None
        fig = analysis.plot_corr()
        assert fig is not None
        fig = analysis.plot_pareto()
        assert fig is not None
        fig = analysis.plot_pairplot_seaborn()
        assert fig is not None
        fig = analysis.plot_pairplot_plotly()
        assert fig is not None
        fig = analysis.plot_ML_model(features_in_log=False)
        assert fig is not None
        fig = analysis.plot_linear_model()
        assert fig is not None

def test_pareto_plot():
    """Test the Pareto chart of standardized effects."""
    factors = data.columns[:-1].tolist()
    response = data.columns[-1]
    analysis = DataAnalysis(data, factors, response)
    fig = analysis.plot_pareto()
    bars = fig.data[0]
    assert len(bars.y) == len(factors)
    # sorted ascending so that the largest effect is displayed on top
    assert list(bars.x) == sorted(bars.x)
    # interactions add bars
    fig2 = analysis.plot_pareto(order=2)
    assert len(fig2.data[0].y) > len(factors)
    # the other models are left untouched
    assert analysis.equation == ''
    assert analysis.linear_model is None


def test_pareto_sign_colors():
    """Positive effects are green, negative effects are red."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({'up': rng.uniform(0, 1, 30), 'down': rng.uniform(0, 1, 30)})
    df['y'] = 5 * df['up'] - 5 * df['down'] + rng.normal(0, 0.1, 30)
    fig = DataAnalysis(df, ['up', 'down'], 'y').plot_pareto()
    colors = dict(zip(fig.data[0].y, fig.data[0].marker.color))
    assert colors['up'].startswith('rgba(46,125,80')
    assert colors['down'].startswith('rgba(214,39,40')


def test_pareto_saturated():
    """Test that a model with no residual degrees of freedom raises an error."""
    factors = data.columns[:-1].tolist()
    response = data.columns[-1]
    small = data.iloc[:len(factors) + 1].copy()
    analysis = DataAnalysis(small, factors, response)
    with pytest.raises(ValueError):
        analysis.plot_pareto()


def test_invalid_model_type():
    """Test if an invalid model type raises an error."""
    factors = data.columns[:-1].tolist()
    response = data.columns[-1]
    analysis = DataAnalysis(data, factors, response)
    with pytest.raises(ValueError):
        analysis.model_type = "InvalidModel"

