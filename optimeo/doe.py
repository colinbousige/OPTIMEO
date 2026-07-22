# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

"""
This module provides a class for creating and visualizing a design of experiments (DoE).
It supports various types of designs including Full Factorial, Sobol sequence, Fractional Factorial, Definitive Screening Design, Space Filling Latin Hypercube, Randomized Latin Hypercube, Optimal, Plackett-Burman, and Box-Behnken.
The class allows the user to specify the parameters, their types, and values, and generates the design accordingly.
It also provides a method to plot the design using scatter plots.

You can see an example notebook `here <../examples/doe.ipynb>`_.
"""


import random
import plotly.graph_objects as go
from itertools import combinations
import plotly.express as px
import definitive_screening_design as dsd
from pyDOE3 import *
from sklearn.preprocessing import LabelEncoder
from doepy import build
from dexpy.design import coded_to_actual
from dexpy.model import ModelOrder
from dexpy.optimal import build_optimal
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeError)


class DesignOfExperiments:
    """
    Class to create a design of experiments (DoE) for a given model.
    This class allows the user to specify the type of design, the parameters,
    and various options for the design generation.
    The design can be visualized using scatter plots.

    Example
    -------

    .. code-block:: python

        from optimeo.doe import DesignOfExperiments

        parameters = [
            {'name': 'Temperature', 'type': 'integer', 'values': [20, 30, 40]},
            {'name': 'Pressure', 'type': 'float', 'values': [1, 2, 3]},
            {'name': 'Catalyst', 'type': 'categorical', 'values': ['A', 'B', 'C']}
        ]

        doe = DesignOfExperiments(
            type='Full Factorial',
            parameters=parameters,
        )
        design = doe.design
        print(design)

    Parameters
    ----------

    type : str
        The type of design to create. Must be one of:
        ``'Full Factorial'``, ``'Sobol sequence'``, ``'Fractional Factorial'``,
        ``'Definitive Screening'``, ``'Space Filling Latin Hypercube'``,
        ``'Randomized Latin Hypercube'``, ``'Optimal'``, ``'Plackett-Burman'``,
        ``'Box-Behnken'`` or ``'Central Composite'``.
    parameters : List[Dict[str, Dict[str, Any]]]
        List of parameters for the design, each with a dictionary of properties.
        Each dictionary should contain ``'name'``, ``'type'``, and ``'values'``.
        ``'values'`` should be a list of possible values for the parameter.
        ``'type'`` should be either "int", "integer", "float", "<other>". 
        Any <other> will be considered as "categorical".
        ``'values'`` should be a list of possible values for the parameter.
    Nexp : int, optional
        Number of experiments in the design, when applicable. Default is 4.
    order : int, optional
        Order of the model (for ``'Optimal'`` design). Default is 2.
    randomize : bool, optional
        Whether to randomize the run order. Default is True.
    reduction : int, optional
        Reduction factor for ``'Fractional Factorial'`` designs. Default is 2.
    feature_constraints : Optional[List[Dict[str, Any]]], optional
        Feature constraints of the experiment for Sobol sequence. Default is None.
        If a single dictionary is provided, it will be converted to a list.
        If a string is provided, it will be converted to a list with one element.
        If a list is provided, it will be used as is.
        If None, no constraints will be applied.

    Attributes
    ----------

    type : str
        The type of design.
    parameters : List[Dict[str, Dict[str, Any]]]
        The parameters for the design.
    Nexp : int
        Number of experiments in the design.
    order : int
        Order of the model.
    randomize : bool
        Whether to randomize the run order.
    reduction : int
        Reduction factor for ``'Fractional Factorial'`` designs.
    design : pd.DataFrame
        The design DataFrame.
    lows : Dict[str, float]
        Lower bounds for the parameters.
    highs : Dict[str, float]
        Upper bounds for the parameters.

    Methods
    -------
    create_design()
        Create the design of experiments based on the specified type and parameters.
    plot()
        Plot the design of experiments using plotly.

    """

    def __init__(self,
                 type: str,
                 parameters: List[Dict[str, Dict[str, Any]]],
                 Nexp: int = 4,
                 order: int = 2,
                 randomize: bool = True,
                 reduction: int = 2,
                 feature_constraints: Optional[List[Dict[str, Any]]] = None,
                 center=(2, 2),
                 alpha='o',
                 face='ccc',
                 seed: int = 42):
        self.type = type
        self.parameters = parameters
        self.Nexp = Nexp
        self.order = order
        self.randomize = randomize
        self.reduction = reduction
        self.center = center
        self.alpha = alpha
        self.face = face
        self.design = None
        self.lows = {}
        self.feature_constraints = feature_constraints
        self.highs = {}
        self.seed = seed
        self.create_design()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Return a string representation of the DesignOfExperiments instance.
        """
        printpar = "\n".join([str(par) for par in self.parameters])
        return f"""
- Design of Experiments type: {self.type}
- Parameters:
{printpar}
- Lows: {self._lows}
- Highs: {self._highs}
- If applicable:
    - Randomize: {self.randomize}
    - Number of Experiments: {self.Nexp}
    - Order: {self.order}
    - Reduction: {self.reduction}
- Design:
{self.design}
"""

    @property
    def type(self) -> str:
        """The type of design to create. Must be one of: `'Full Factorial'`, `'Sobol sequence'`, `'Fractional Factorial'`, `'Definitive Screening'`, `'Space Filling Latin Hypercube'`, `'Randomized Latin Hypercube'`, `'Optimal'`, `'Plackett-Burman'`, `'Box-Behnken'` or `'Central Composite'`."""
        return self._type

    @type.setter
    def type(self, value: str):
        """Set the type of design."""
        self._type = value

    @property
    def seed(self) -> int:
        """Random seed for reproducibility. Default is 42."""
        return self._seed

    @seed.setter
    def seed(self, value: int):
        """Set the random seed."""
        if isinstance(value, int):
            self._seed = value
        else:
            raise Warning("Seed must be an integer. Using default seed 42.")
            self._seed = 42
        random.seed(self.seed)
        np.random.seed(self.seed)

    @property
    def parameters(self) -> List[Dict[str, Dict[str, Any]]]:
        """List of parameters for the design, each with a dictionary of properties.
            Each dictionary should contain the keys ``"name"``, ``"type"``, and ``"values"``.
            ``"values"`` should be a list of possible values for the parameter.
            ``"type"`` should be either ``"int"``, ``"integer"``, ``"float"``, ``"<other>"``. 
            Any ``"<other>"`` will be considered as ``"categorical"``.
            ``"values"`` should be a list of possible values for the parameter."""
        return self._parameters

    @parameters.setter
    def parameters(self, value: List[Dict[str, Dict[str, Any]]]):
        """Set the parameters for the design."""
        self._parameters = value

    @property
    def Nexp(self) -> int:
        """Number of experiments in the design, when applicable. Default is ``4``."""
        return self._Nexp

    @Nexp.setter
    def Nexp(self, value: int):
        """Set the number of experiments."""
        self._Nexp = value

    @property
    def order(self) -> int:
        """Order of the model (for ``'Optimal'`` design). Default is ``2``."""
        return self._order

    @property
    def center(self) -> tuple:
        """Center for the Central Composite Design. Must be a tuple of two values."""
        return self._center

    @center.setter
    def center(self, value: tuple):
        """Set the center of the design."""
        if not isinstance(value, tuple):
            raise ValueError("Center must be a tuple of two values.")
        if len(value) != 2:
            raise ValueError("Center must be a tuple of two values.")
        if not all(isinstance(i, (int, float)) for i in value):
            raise ValueError("Center must be a tuple of two numeric values.")
        self._center = value

    @property
    def alpha(self) -> str:
        """Alpha for the Central Composite Design. Default is ``'o'`` (orthogonal).
        Can be either ``'o'`` or ``'r'`` (rotatable)."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: str):
        """Set the alpha of the design."""
        if value not in ['o', 'r']:
            raise ValueError(
                "Alpha must be either ``'o'`` (orthogonal) or ``'r'`` (rotatable).")
        self._alpha = value

    @property
    def face(self) -> str:
        """The relation between the start points and the corner (factorial) points for the Central Composite Design.

        There are three possible options for this input:

        1. ``'circumscribed'`` or ``'ccc'`` (Default)
        2. ``'inscribed'`` or ``'cci'``
        3. ``'faced'`` or ``'ccf'``"""
        return self._face

    @face.setter
    def face(self, value: str):
        """Set the face of the design."""
        if value not in ['ccc', 'cci', 'ccf']:
            raise ValueError(
                "Face must be either ``'ccc'`` (circumscribed), ``'cci'`` (inscribed), or ``'ccf'`` (faced).")
        self._face = value

    @property
    def lows(self) -> Dict[str, float]:
        """Get the lower bounds for the parameters."""
        return self._lows

    @lows.setter
    def lows(self, value: Dict[str, float]):
        """Set the lower bounds for the parameters."""
        self._lows = value

    @property
    def highs(self) -> Dict[str, float]:
        """Get the upper bounds for the parameters."""
        return self._highs

    @highs.setter
    def highs(self, value: Dict[str, float]):
        """Set the upper bounds for the parameters."""
        self._highs = value

    @order.setter
    def order(self, value: int):
        """Set the order of the model."""
        self._order = value

    @property
    def randomize(self) -> bool:
        """Whether to randomize the run order. Default is ``True``."""
        return self._randomize

    @randomize.setter
    def randomize(self, value: bool):
        """Set the randomize flag."""
        self._randomize = value

    @property
    def reduction(self) -> int:
        """Reduction factor for ``'Fractional Factorial'`` designs. Default is ``2``."""
        return self._reduction

    @reduction.setter
    def reduction(self, value: int):
        """Set the reduction factor."""
        self._reduction = value

    @property
    def design(self) -> pd.DataFrame:
        """Get the design DataFrame."""
        return self._design

    @design.setter
    def design(self, value: pd.DataFrame):
        """Set the design DataFrame."""
        self._design = value

    @property
    def feature_constraints(self):
        """
        Get the feature constraints of the experiment for Sobol sequence.
        """
        return self._feature_constraints

    @feature_constraints.setter
    def feature_constraints(self, value):
        """
        Set the feature constraints of the experiment with validation for Sobol sequence.
        """
        if isinstance(value, dict):
            self._feature_constraints = [value]
        elif isinstance(value, list):
            self._feature_constraints = value if len(value) > 0 else None
        elif isinstance(value, str):
            self._feature_constraints = [value]
        else:
            self._feature_constraints = None

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def create_design(self):
        """
        Create the design of experiments based on the specified type and parameters.
        """
        for par in self.parameters:
            if par['type'].lower() == "categorical":
                if self.type != 'Sobol sequence':
                    le = LabelEncoder()
                    label = le.fit_transform(par['values'])
                    par['values'] = label
                    par['encoder'] = le
                    self.lows[par['name']] = np.min(par['values'])
                    self.highs[par['name']] = np.max(par['values'])
                else:
                    par['encoder'] = None
            else:
                self.lows[par['name']] = np.min(par['values'])
                self.highs[par['name']] = np.max(par['values'])

        pars = {par['name']: par['values'] for par in self.parameters}

        if self.type == 'Full Factorial':
            self.design = build.full_fact(pars)
        elif self.type == 'Sobol sequence':
            from ax.adapter.registry import Generators
            from ax.generation_strategy.generation_node import GenerationStep
            from ax.generation_strategy.generation_strategy import GenerationStrategy
            from ax.service.ax_client import AxClient, ObjectiveProperties

            ax_client = AxClient()
            params = []
            for par in self.parameters:
                if par['type'].lower() == "float":
                    params.append({'name': par['name'],
                                   'type': 'range',
                                   'value_type': 'float',
                                   'bounds': [float(np.min(par['values'])), float(np.max(par['values']))]})
                elif par['type'].lower() in ["integer", 'int']:
                    params.append({'name': par['name'],
                                   'type': 'range',
                                   'value_type': 'int',
                                   'bounds': [int(np.min(par['values'])), int(np.max(par['values']))]})
                else:
                    params.append({'name': par['name'],
                                   'type': 'choice',
                                   'values': par['values']})
            ax_client.create_experiment(
                name="DOE",
                parameters=params,
                objectives={"response": ObjectiveProperties(minimize=False)},
                parameter_constraints=self._feature_constraints
            )
            gs = GenerationStrategy(
                nodes=[GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=-1,
                    should_deduplicate=True,
                    model_kwargs={"seed": self.seed},
                    model_gen_kwargs={},
                )]
            )
            generated_runs = gs.gen(
                experiment=ax_client.experiment,
                data=None,
                n=self.Nexp
            )
            generated_run = generated_runs[0]
            if len(generated_run) == 1:
                generated_run = generated_run[0]
            if self.Nexp == 1:
                ax_client.experiment.new_trial(generated_run)
            else:
                ax_client.experiment.new_batch_trial(generated_run)
            trials = ax_client.get_trials_data_frame()
            self.design = trials[trials['trial_status'] == 'CANDIDATE']
            self.design = self._design.drop(columns=['trial_index',
                                                     'trial_status',
                                                     'arm_name',
                                                     'generation_method',
                                                     'generation_node'],
                                            errors='ignore')
        elif self.type == 'Fractional Factorial':
            for par in range(len(self.parameters)):
                if self.parameters[par]['type'] == "Numerical":
                    self.parameters[par]['type'] = "Categorical"
                    le = LabelEncoder()
                    label = le.fit_transform(self.parameters[par]['values'])
                    self.parameters[par]['values'] = label
                    self.parameters[par]['encoder'] = le
            design = gsd([len(par['values'])
                         for par in self.parameters], self.reduction)
            self.design = pd.DataFrame(
                design, columns=[par['name'] for par in self.parameters])
        elif self.type == 'Definitive Screening':
            params = {par['name']: [np.min(par['values']), np.max(
                par['values'])] for par in self.parameters}
            self.design = dsd.generate(factors_dict=params)
        elif self.type == 'Space Filling Latin Hypercube':
            self.design = build.space_filling_lhs(pars, num_samples=self.Nexp)
        elif self.type == 'Randomized Latin Hypercube':
            self.design = build.lhs(pars, num_samples=self.Nexp)
        elif self.type == 'Optimal':
            reaction_design = build_optimal(
                len(self.parameters),
                order=ModelOrder(self.order),
                run_count=self.Nexp)
            reaction_design.columns = [par['name'] for par in self.parameters]
            self.design = coded_to_actual(
                reaction_design, self._lows, self._highs)
        elif self.type == 'Plackett-Burman':
            self.design = build.plackett_burman(pars)
        elif self.type == 'Box-Behnken':
            if len(self.parameters) < 3 or any([len(par['values']) < 3 for par in self.parameters]):
                self.design = pd.DataFrame({})
                raise Warning(
                    "Box-Behnken design is not possible with less than 3 parameters and with less than 3 levels for any parameter.")
            else:
                self.design = build.box_behnken(d=pars)
        elif self.type == 'Central Composite':
            pars_list = {k: v.tolist() if hasattr(v, 'tolist') else list(v)
                         for k, v in pars.items()}
            self.design = build.central_composite(pars_list,
                                                  center=self.center,
                                                  alpha=self.alpha,
                                                  face=self.face)
        else:
            raise ValueError("Unknown design type. Must be one of: 'Full Factorial', 'Sobol sequence', 'Fractional Factorial', 'Definitive Screening', 'Space Filling Latin Hypercube', 'Randomized Latin Hypercube', 'Optimal', 'Plackett-Burman', 'Box-Behnken' or 'Central Composite'.")

        for par in self.parameters:
            if par['type'].lower() == "categorical" and self.type != 'Sobol sequence':
                vals = self._design[par['name']].to_numpy()
                n_classes = len(par['encoder'].classes_)
                clipped = [int(np.clip(np.round(v), 0, n_classes - 1))
                           for v in vals]
                self.design[par['name']
                            ] = par['encoder'].inverse_transform(clipped)

        # randomize the run order
        self.design['run_order'] = np.arange(len(self._design)) + 1
        if self.randomize:
            ord = self._design['run_order'].to_numpy()
            self.design['run_order'] = np.random.permutation(ord)
        cols = self._design.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        self.design = self._design[cols]
        # apply the column types
        for col in self._design.columns:
            for par in self.parameters:
                if col == par['name']:
                    if par['type'].lower() == "float":
                        self.design[col] = self._design[col].astype(float)
                    elif par['type'].lower() in ["int", "integer"]:
                        self.design[col] = self._design[col].astype(int)
                    else:
                        self.design[col] = self._design[col].astype(str)
        return self._design

    def plot(self):
        """
        Plot the design of experiments.

        Returns
        -------
        List of plotly.graph_objs._figure.Figure
            A list of Plotly figures representing the design of experiments.
        """
        fig = []
        count = 0
        if len(self.design) > 0:
            if len(self.parameters) <= 2:
                # Create 2D scatter plots
                for i, faci in enumerate(self.parameters):
                    for j, facj in enumerate(self.parameters):
                        if j > i:
                            fig.append(px.scatter(
                                self.design,
                                x=facj['name'],
                                y=faci['name'],
                                title=f"""{faci['name']} vs {facj['name']}""",
                                labels={facj['name']: facj['name'],
                                        faci['name']: faci['name']}
                            ))
                            fig[count].update_traces(marker=dict(size=10))
                            fig[count].update_layout(
                                margin=dict(l=10, r=10, t=50, b=50),
                                xaxis=dict(
                                    showgrid=True,
                                    gridcolor="lightgray",
                                    zeroline=False,
                                    showline=True,
                                    linewidth=1,
                                    linecolor="black",
                                    mirror=True
                                ),
                                yaxis=dict(
                                    showgrid=True,
                                    gridcolor="lightgray",
                                    zeroline=False,
                                    showline=True,
                                    linewidth=1,
                                    linecolor="black",
                                    mirror=True
                                ),
                            )
                            count += 1
            else:
                # Create 3D scatter plots
                for k, (faci, facj, fack) in enumerate(combinations(self.parameters, 3)):
                    fig.append(go.Figure(data=[go.Scatter3d(
                        x=self.design[facj['name']],
                        y=self.design[faci['name']],
                        z=self.design[fack['name']],
                        mode='markers',
                        marker=dict(size=10, color='royalblue', opacity=0.7),
                    )]))
                    fig[count].update_layout(
                        template='ggplot2',
                        height=500,
                        width=500,
                        scene=dict(
                            xaxis_title=facj['name'],
                            yaxis_title=faci['name'],
                            zaxis_title=fack['name'],
                        ),
                        title=f"{faci['name']} vs {facj['name']}<br>vs {fack['name']}",
                        margin=dict(l=10, r=10, t=50, b=50)
                    )
                    count += 1
        return fig
