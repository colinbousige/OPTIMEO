# test_doe.py
import sys
import os

# Add the parent directory of optimeo to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pytest
import pandas as pd
from optimeo.doe import DesignOfExperiments

# Sample parameters for testing
sample_parameters = [
        {'name': 'Temperature', 'type': 'integer', 'values': [20, 30, 40]},
        {'name': 'Pressure', 'type': 'float', 'values': [1, 2, 3]},
        {'name': 'Catalyst', 'type': 'categorical', 'values': ['A', 'B', 'C']},
        {'name': 'Speed', 'type': 'categorical', 'values': [100, 200, 300]}
    ]

# Sample design types for testing
sample_design_types = [
    "Full Factorial",
    # "Sobol sequence",
    "Fractional Factorial",
    "Definitive Screening",
    "Space Filling Latin Hypercube",
    "Randomized Latin Hypercube",
    "Optimal",
    "Plackett-Burman",
    "Box-Behnken",
]

def test_initialization():
    """Test if the class initializes correctly with valid inputs."""
    for design in sample_design_types:
        doe = DesignOfExperiments(
            type=design,
            parameters=sample_parameters,
            Nexp=10
        )
        print(design)
        fig = doe.plot()
        assert doe.type == design
        assert doe.parameters == sample_parameters
        assert isinstance(doe.design, pd.DataFrame)
        assert not doe.design.empty
        assert len(fig) > 0

def test_central_composite_numeric():
    """Test Central Composite with numeric parameters only."""
    doe = DesignOfExperiments(type="Central Composite", parameters=sample_parameters)
    assert isinstance(doe.design, pd.DataFrame)
    assert not doe.design.empty

def test_central_composite_with_categorical():
    """Test Central Composite with a categorical parameter — axial star points must be clipped to valid labels."""
    params = [
        {'name': 'Temperature', 'type': 'float', 'values': [20, 40]},
        {'name': 'Catalyst', 'type': 'categorical', 'values': ['A', 'B', 'C']},
    ]
    doe = DesignOfExperiments(type="Central Composite", parameters=params)
    assert isinstance(doe.design, pd.DataFrame)
    assert not doe.design.empty
    assert set(doe.design['Catalyst']).issubset({'A', 'B', 'C'})

def test_invalid_design_type():
    """Test if an invalid design type raises an error."""
    with pytest.raises(ValueError):
        DesignOfExperiments(type="Invalid Type", parameters=sample_parameters)

