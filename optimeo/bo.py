# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

"""
This module provides a class for optimizing experiments using Bayesian Optimization (BO) with the [Ax platform](https://ax.dev/).
It includes methods for initializing the experiment, suggesting trials, predicting outcomes, and plotting results.

You can see an example notebook [here](../examples/bo.ipynb).

"""

import matplotlib as mpl
import re
import plotly.express as px
import plotly.graph_objects as go
from botorch.acquisition.analytic import *
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.plot.slice import plot_slice
from ax.plot.trace import optimization_trace_single_method_plotly
from ax.plot.pareto_utils import compute_posterior_pareto_frontier
from ax.plot.feature_importances import plot_feature_importance_by_feature_plotly
try:
    from ax.plot.contour import interact_contour, plot_contour
except ImportError:
    from ax.plot.contour import plot_contour
    interact_contour = None
try:
    from ax.analysis.plotly.sensitivity import SensitivityAnalysisPlot
except ImportError:
    SensitivityAnalysisPlot = None
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generation_node import GenerationStep
from ax.exceptions.core import DataRequiredError
from ax.core.base_trial import TrialStatus
from ax.core.observation import ObservationFeatures
from ax.adapter.registry import Generators
from typing import Any, Dict, List, Optional, Union, Tuple
from janitor import clean_names
import random
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeError)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class BOExperiment:
    """
    BOExperiment is a class designed to facilitate Bayesian Optimization experiments using the [Ax platform](https://ax.dev/).
    It encapsulates the experiment setup, including features, outcomes, constraints, and optimization methods.

    Example
    -------

    .. code-block:: python

        from optimeo.bo import BOExperiment, read_experimental_data

        features, outcomes = read_experimental_data('data.csv', out_pos=[-2, -1])
        experiment = BOExperiment(
            features,
            outcomes,
            N=5,
            maximize={'out1': True, 'out2': False},
        )
        experiment.suggest_next_trials()
        experiment.plot_model(metricname='outcome1')
        experiment.plot_optimization_trace()

    Parameters
    ----------
    features: Dict[str, Dict[str, Any]]
        A dictionary defining the features of the experiment, including their types and ranges.
        Each feature is represented as a dictionary with keys 'type', 'data', and 'range'.
        - 'type': The type of the feature (e.g., 'int', 'float', 'text').
        - 'data': The observed data for the feature.
        - 'range': The range of values for the feature.
    outcomes: Dict[str, Dict[str, Any]]
        A dictionary defining the outcomes of the experiment, including their types and observed data.
        Each outcome is represented as a dictionary with keys 'type' and 'data'.
        - 'type': The type of the outcome (e.g., 'int', 'float').
        - 'data': The observed data for the outcome.
    ranges: Optional[Dict[str, Dict[str, Any]]]
        A dictionary defining the ranges of the features. Default is `None`.
        If not provided, the ranges will be inferred from the features data.
        The ranges should be in the format `{'feature_name': [minvalue,maxvalue]}`.
    N: int
        The number of trials to suggest in each optimization step. Must be a positive integer.
    maximize: Union[bool, Dict[str, bool]]
        A boolean or dict indicating whether to maximize the outcomes in the form `{'outcome1':True, 'outcome2':False}`.
        If a single boolean is provided, it is applied to all outcomes. Default is `True`.
    fixed_features: Optional[Dict[str, Any]]
        A dictionary defining fixed features with their values. Default is `None`.
        If provided, the fixed features will be treated as fixed parameters in the generation process.
        The fixed features should be in the format `{'feature_name': value}`.
        The values should be the fixed values for the respective features.
    outcome_constraints: Optional[List[str]]
        Constraints on the outcomes, specified as a list of strings. Default is `None`.
        The constraints should be in the format `{'outcome_name': [minvalue,maxvalue]}`.
    objective_thresholds: Optional[Dict[str, float]]
        Reference-point thresholds for multi-objective optimization, specified per outcome
        in the format ``{'outcome_name': threshold}``. Default is `None`.
        Setting these avoids Ax's default winsorization warning for multi-objective runs.
    feature_constraints: Optional[List[str]]
        Constraints on the features, specified as a list of strings. Default is `None`.
        The constraints should be in the format `{'feature_name': [minvalue,maxvalue]}`.
    optim: str
        The optimization method to use, either 'bo' for Bayesian Optimization or 'sobol' for Sobol sequence. Default is 'bo'.
    acq_func: Optional[Dict[str, Any]]
        The acquisition function to use for the optimization process. It must be a dict with 2 keys:
        - `acqf`: the acquisition function class to use (e.g., `UpperConfidenceBound`),
        - `acqf_kwargs`: a dict of the kwargs to pass to the acquisition function class. (e.g. `{'beta': 0.1}`).

        If not provided, the default acquisition function is used (`LogExpectedImprovement` or `qLogExpectedImprovement` if N>1).

    Attributes
    ----------

    features: Dict[str, Dict[str, Any]]
        A dictionary defining the features of the experiment, including their types and ranges.
    outcomes: Dict[str, Dict[str, Any]]
        A dictionary defining the outcomes of the experiment, including their types and observed data.
    N: int
        The number of trials to suggest in each optimization step. Must be a positive integer.
    maximize: Union[bool, List[bool]]
        A boolean or list of booleans indicating whether to maximize the outcomes.
        If a single boolean is provided, it is applied to all outcomes.
    outcome_constraints: Optional[Dict[str, Dict[str, float]]]
        Constraints on the outcomes, specified as a dictionary or list of dictionaries.
    feature_constraints: Optional[List[Dict[str, Any]]]
        Constraints on the features, specified as a list of dictionaries.
    optim: str
        The optimization method to use, either 'bo' for Bayesian Optimization or 'sobol' for Sobol sequence.
    data: pd.DataFrame
        A DataFrame representing the current data in the experiment, including features and outcomes.
    acq_func: dict
        The acquisition function to use for the optimization process. 
    generator_run:
        The generator run for the experiment, used to generate new candidates.
    model:
        The model used for predictions in the experiment.
    ax_client:
        The AxClient for the experiment, used to manage trials and data.
    gs:
        The generation strategy for the experiment, used to generate new candidates.
    parameters:
        The parameters for the experiment, including their types and ranges.
    names:
        The names of the features in the experiment.
    fixed_features:
        The fixed features for the experiment, used to generate new candidates.
    candidate:
        The candidate(s) suggested by the optimization process.


    Methods
    -------
    initialize_ax_client()
        Initialize AxClient with experiment parameters, objectives, and constraints.
    suggest_next_trials()
        Suggest next trial(s) from the current model and generation strategy.
    predict(params)
        Predict outcomes for a list of parameter dictionaries.
    update_experiment(params, outcomes)
        Update the experiment with new observations and refresh internal state.
    plot_model(metricname=None, slice_values=None, linear=False)
        Plot model predictions as slices or contours.
    plot_optimization_trace(optimum=None)
        Plot optimization progress over trials.
    plot_pareto_frontier()
        Plot Pareto frontier for multi-objective problems.
    get_best_parameters()
        Return best parameter set(s) and associated outcomes.
    clear_trials()
        Remove all current trials.

    """

    def __init__(self,
                 features: Dict[str, Dict[str, Any]],
                 outcomes: Dict[str, Dict[str, Any]],
                 ranges: Optional[Dict[str, Dict[str, Any]]] = None,
                 N=1,
                 maximize: Union[bool, Dict[str, bool]] = True,
                 fixed_features: Optional[Dict[str, Any]] = None,
                 outcome_constraints: Optional[List[str]] = None,
                 objective_thresholds: Optional[Dict[str, float]] = None,
                 feature_constraints: Optional[List[str]] = None,
                 optim='bo',
                 acq_func=None,
                 seed=42) -> None:
        self._first_initialization_done = False
        self.ranges = ranges
        self.features = features
        self.names = list(self._features.keys())
        self.fixed_features = fixed_features
        self.outcomes = outcomes
        self.N = N
        self.maximize = maximize
        self.outcome_constraints = outcome_constraints
        self.objective_thresholds = objective_thresholds
        self.feature_constraints = feature_constraints
        self.optim = optim
        self.acq_func = acq_func
        self.seed = seed
        self.candidate = None
        """The candidate(s) suggested by the optimization process."""
        self.ax_client = None
        """Ax's client for the experiment."""
        self.model = None
        """Ax's Gaussian Process model."""
        self.parameters = None
        """Ax's parameters for the experiment."""
        self.generator_run = None
        """Ax's generator run for the experiment."""
        self.gs = None
        """Ax's generation strategy for the experiment."""
        self.initialize_ax_client()
        self.Nmetrics = len(self.ax_client.objective_names)
        """The number of metrics in the experiment."""
        self._first_initialization_done = True
        """To indicate that the first initialization is done so that we don't call `initialize_ax_client()` again."""
        self.pareto_frontier = None
        """The Pareto frontier for multi-objective optimization experiments."""

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
    def features(self):
        """
        A dictionary defining the features of the experiment, including their types and ranges.

        Example
        -------
        .. code-block:: python

            features = {
                'feature1': {'type': 'int', 'data': [1, 2, 3], 'range': [1, 3]},
                'feature2': {'type': 'float', 'data': [0.1, 0.2, 0.3], 'range': [0.1, 0.3]},
                'feature3': {'type': 'text', 'data': ['A', 'B', 'C'], 'range': ['A', 'B', 'C']},
            }
        """
        return self._features

    @features.setter
    def features(self, value):
        """
        Set the features of the experiment with validation.
        """
        if not isinstance(value, dict):
            raise ValueError("features must be a dictionary")
        self._features = value
        for name in self._features.keys():
            if self.ranges and name in self.ranges.keys():
                self._features[name]['range'] = self.ranges[name]
            else:
                feature_data = self._features[name].get('data', [])
                if len(feature_data) == 0:
                    self._features[name]['range'] = self._features[name].get(
                        'range', [])
                elif self._features[name]['type'] == 'text':
                    self._features[name]['range'] = list(set(feature_data))
                elif self._features[name]['type'] == 'int':
                    self._features[name]['range'] = [int(np.min(self._features[name]['data'])),
                                                     int(np.max(self._features[name]['data']))]
                elif self._features[name]['type'] == 'float':
                    self._features[name]['range'] = [float(np.min(self._features[name]['data'])),
                                                     float(np.max(self._features[name]['data']))]
        if self._first_initialization_done:
            self.initialize_ax_client()

    @property
    def ranges(self):
        """
        A dictionary defining the ranges of the features. Default is `None`.

        If not provided, the ranges will be inferred from the features data.
        The ranges should be in the format `{'feature_name': [minvalue,maxvalue]}`.
        """
        return self._ranges

    @ranges.setter
    def ranges(self, value):
        """
        Set the ranges of the features with validation.
        """
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("ranges must be a dictionary")
        self._ranges = value

    @property
    def names(self):
        """
        The names of the features.
        """
        return self._names

    @names.setter
    def names(self, value):
        """
        Set the names of the features.
        """
        if not isinstance(value, list):
            raise ValueError("names must be a list")
        self._names = value

    @property
    def outcomes(self):
        """
        A dictionary defining the outcomes of the experiment, including their types and observed data.

        Example
        -------
        .. code-block:: python

            outcomes = {
                'outcome1': {'type': 'float', 'data': [0.1, 0.2, 0.3]},
                'outcome2': {'type': 'float', 'data': [1.0, 2.0, 3.0]},
            }
        """
        return self._outcomes

    @outcomes.setter
    def outcomes(self, value):
        """
        Set the outcomes of the experiment with validation.
        """
        if not isinstance(value, dict):
            raise ValueError("outcomes must be a dictionary")
        self._outcomes = value
        self.out_names = list(value.keys())
        if self._first_initialization_done:
            self.initialize_ax_client()

    @property
    def fixed_features(self):
        """
        A dictionary defining fixed features with their values. Default is `None`.
        If provided, the fixed features will be treated as fixed parameters in the generation process.
        The fixed features should be in the format `{'feature_name': value}`.
        The values should be the fixed values for the respective features.
        """
        return self._fixed_features

    @fixed_features.setter
    def fixed_features(self, value):
        """
        Set the fixed features of the experiment.
        """
        self._fixed_features = None
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError("fixed_features must be a dictionary")
            for name in value.keys():
                if name not in self.names:
                    raise ValueError(
                        f"Fixed feature '{name}' not found in features")
            # fixed_features should be an ObservationFeatures object
            self._fixed_features = ObservationFeatures(parameters=value)
        if self._first_initialization_done:
            self.set_gs()

    @property
    def N(self):
        """
        The number of trials to suggest in each optimization step. Must be a positive integer. Default is `1`.
        """
        return self._N

    @N.setter
    def N(self, value):
        """
        Set the number of trials to suggest in each optimization step with validation.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("N must be a positive integer")
        self._N = value
        if self._first_initialization_done:
            self.set_gs()

    @property
    def maximize(self):
        """
        A boolean or dict indicating whether to maximize outcomes in the form ``{'outcome1': True, 'outcome2': False}``.
        If a single boolean is provided, it is applied to all outcomes. Default is ``True``.
        """
        return self._maximize

    @maximize.setter
    def maximize(self, value):
        """
        Set the maximization setting for the outcomes with validation.
        """
        if isinstance(value, bool):
            self._maximize = {out: value for out in self.out_names}
        elif isinstance(value, dict) and len(value) == len(self._outcomes):
            self._maximize = {k: v for k, v in value.items() if
                              (k in self.out_names and isinstance(v, bool))}
        else:
            raise ValueError(
                "maximize must be a boolean or a list of booleans with the same length as outcomes")
        if self._first_initialization_done:
            self.initialize_ax_client()

    @property
    def outcome_constraints(self):
        """
        Constraints on the outcomes, specified as a list of strings. Default is `None`.
        """
        return self._outcome_constraints

    @outcome_constraints.setter
    def outcome_constraints(self, value):
        """
        Set the outcome constraints of the experiment with validation.
        """
        if isinstance(value, str):
            self._outcome_constraints = [value]
        elif isinstance(value, list):
            self._outcome_constraints = value
        else:
            self._outcome_constraints = None
        if self._first_initialization_done:
            self.initialize_ax_client()

    @property
    def objective_thresholds(self):
        """
        Reference-point thresholds for multi-objective optimization.

        Format: ``{'outcome_name': threshold}``.
        """
        return self._objective_thresholds

    @objective_thresholds.setter
    def objective_thresholds(self, value):
        """
        Set objective thresholds with validation.
        """
        if value is None:
            self._objective_thresholds = None
        elif isinstance(value, dict):
            validated = {}
            for name, threshold in value.items():
                if name not in self.out_names:
                    raise ValueError(
                        f"Objective threshold provided for unknown outcome '{name}'")
                if not isinstance(threshold, (int, float)):
                    raise ValueError(
                        f"Objective threshold for '{name}' must be a number")
                validated[name] = float(threshold)
            self._objective_thresholds = validated
        else:
            raise ValueError(
                "objective_thresholds must be a dictionary or None")
        if self._first_initialization_done:
            self.initialize_ax_client()

    @property
    def feature_constraints(self):
        """
        Constraints on the features, specified as a list of strings. Default is `None`.

        Example
        -------
        .. code-block:: python

            feature_constraints = [
                'feature1 <= 10.0',
                'feature1 + 2*feature2 >= 3.0',
            ]
        """
        return self._feature_constraints

    @feature_constraints.setter
    def feature_constraints(self, value):
        """
        Set the feature constraints of the experiment with validation.
        """
        if isinstance(value, dict):
            self._feature_constraints = [value]
        elif isinstance(value, list):
            self._feature_constraints = value
        elif isinstance(value, str):
            self._feature_constraints = [value]
        else:
            self._feature_constraints = None
        if self._first_initialization_done:
            self.initialize_ax_client()

    @property
    def optim(self):
        """
        The optimization method to use, either `'bo'` for Bayesian Optimization or `'sobol'` for Sobol sequence. Default is `'bo'`.
        """
        return self._optim

    @optim.setter
    def optim(self, value):
        """
        Set the optimization method with validation.
        """
        value = value.lower()
        if value not in ['bo', 'sobol']:
            raise ValueError(
                "Optimization method must be either 'bo' or 'sobol'")
        self._optim = value
        if self._first_initialization_done:
            self.set_gs()

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns a DataFrame of the current data in the experiment, including features and outcomes.
        """
        feature_data = {name: info['data']
                        for name, info in self._features.items()}
        outcome_data = {name: info['data']
                        for name, info in self._outcomes.items()}
        data_dict = {**feature_data, **outcome_data}
        return pd.DataFrame(data_dict)

    @data.setter
    def data(self, value: pd.DataFrame):
        """
        Sets the features and outcomes data from a given DataFrame.
        """
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")

        feature_columns = [
            col for col in value.columns if col in self._features]
        outcome_columns = [
            col for col in value.columns if col in self._outcomes]

        for col in feature_columns:
            self._features[col]['data'] = value[col].tolist()

        for col in outcome_columns:
            self._outcomes[col]['data'] = value[col].tolist()

        if self._first_initialization_done:
            self.initialize_ax_client()

    @property
    def pareto_frontier(self):
        """
        The Pareto frontier for multi-objective optimization experiments.
        """
        return self._pareto_frontier

    @pareto_frontier.setter
    def pareto_frontier(self, value):
        """
        Set the Pareto frontier of the experiment.
        """
        self._pareto_frontier = value

    @property
    def acq_func(self):
        """
        The acquisition function to use for the optimization process. It must be a dict with 2 keys:
        - `acqf`: the acquisition function class to use (e.g., `UpperConfidenceBound`),
        - `acqf_kwargs`: a dict of the kwargs to pass to the acquisition function class. (e.g. `{'beta': 0.1}`).

        If not provided, the default acquisition function is used (`LogExpectedImprovement` or `qLogExpectedImprovement` if N>1).

        Example
        -------
        .. code-block:: python

            acq_func = {
                'acqf': UpperConfidenceBound,
                'acqf_kwargs': {'beta': 0.1},  # lower = exploitation, higher = exploration
            }
        """
        return self._acq_func

    @acq_func.setter
    def acq_func(self, value):
        """
        Set the acquisition function with validation.
        """
        self._acq_func = value
        if self._first_initialization_done:
            self.set_gs()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        """
        Return a string representation of the BOExperiment instance.
        """
        return f"""
BOExperiment(
    N={self.N},
    maximize={self.maximize},
    outcome_constraints={self.outcome_constraints},
    feature_constraints={self.feature_constraints},
    optim={self.optim}
)

Input data:

{self.data}
        """

    def initialize_ax_client(self):
        """
        Initialize the AxClient with the experiment's parameters, objectives, and constraints.
        """
        print('\n========   INITIALIZING MODEL   ========\n')
        self.ax_client = AxClient(verbose_logging=False,
                                  suppress_storage_errors=True)
        self.parameters = []
        for name, info in self._features.items():
            if info['type'] == 'text':
                values = [str(val) for val in info['range']]
                self.parameters.append({
                    "name": name,
                    "type": "choice",
                    "values": values,
                    "value_type": "str",
                    "is_ordered": len(values) == 2,
                    "sort_values": False})
            elif info['type'] == 'int':
                self.parameters.append({
                    "name": name,
                    "type": "range",
                    "bounds": [int(np.min(info['range'])),
                               int(np.max(info['range']))],
                    "value_type": "int"})
            elif info['type'] == 'float':
                self.parameters.append({
                    "name": name,
                    "type": "range",
                    "bounds": [float(np.min(info['range'])),
                               float(np.max(info['range']))],
                    "value_type": "float"})

        objectives = {}
        for k, v in self._maximize.items():
            if isinstance(v, bool) and k in self._outcomes.keys():
                threshold = None
                if self._objective_thresholds is not None:
                    threshold = self._objective_thresholds.get(k)
                objectives[k] = ObjectiveProperties(
                    minimize=not v, threshold=threshold)

        self.ax_client.create_experiment(
            name="bayesian_optimization",
            parameters=self.parameters,
            objectives=objectives,
            parameter_constraints=self._feature_constraints,
            outcome_constraints=self._outcome_constraints,
            overwrite_existing_experiment=True
        )

        if len(next(iter(self._outcomes.values()))['data']) > 0:
            for i in range(len(next(iter(self._outcomes.values()))['data'])):
                params = {name: info['data'][i]
                          for name, info in self._features.items()}
                outcomes = {name: info['data'][i]
                            for name, info in self._outcomes.items()}
                self.ax_client.attach_trial(params)
                self.ax_client.complete_trial(trial_index=i, raw_data=outcomes)

        self.set_model()
        self.set_gs()

    def _has_completed_data(self) -> bool:
        """Return whether the current Ax experiment has usable completed data."""
        if self.ax_client is None or self.ax_client.experiment is None:
            return False
        data = self.ax_client.experiment.fetch_data()
        if data is None:
            return False
        df = getattr(data, "df", None)
        return df is not None and not df.empty

    def set_model(self):
        """
        Set the model to be used for predictions.
        This method is called after initializing the AxClient.
        """
        if not self._has_completed_data():
            self.model = None
            return

        try:
            self.model = Generators.BOTORCH_MODULAR(
                experiment=self.ax_client.experiment,
                data=self.ax_client.experiment.fetch_data()
            )
        except DataRequiredError:
            self.model = None

    def set_gs(self):
        """
        Set the generation strategy for the experiment.
        This method is called after initializing the AxClient.
        """
        self.clear_trials()
        if self._optim == 'bo':
            if not self.model:
                self.set_model()
            if self.model is None:
                self.gs = GenerationStrategy(
                    nodes=[GenerationStep(
                        generator=Generators.SOBOL,
                        num_trials=-1,
                        should_deduplicate=True,
                        model_kwargs={"seed": self.seed},
                        model_gen_kwargs={},
                    )
                    ]
                )
            elif self.acq_func is None:
                self.gs = GenerationStrategy(
                    nodes=[GenerationStep(
                        generator=Generators.BOTORCH_MODULAR,
                        num_trials=-1,  # No limitation on how many trials should be produced from this step
                        max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
                    )
                    ]
                )
            else:
                self.gs = GenerationStrategy(
                    nodes=[GenerationStep(
                        generator=Generators.BOTORCH_MODULAR,
                        num_trials=-1,  # No limitation on how many trials should be produced from this step
                        max_parallelism=3,  # Parallelism limit for this step, often lower than for Sobol
                        model_kwargs={
                            "seed": self.seed,
                            "botorch_acqf_class": self.acq_func['acqf'],
                            "botorch_acqf_options": self.acq_func['acqf_kwargs'],
                        },
                    )
                    ]
                )
        elif self._optim == 'sobol':
            self.gs = GenerationStrategy(
                nodes=[GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=-1,  # How many trials should be produced from this generation step
                    should_deduplicate=True,  # Deduplicate the trials
                    # Any kwargs you want passed into the model
                    model_kwargs={"seed": self.seed},
                    model_gen_kwargs={},  # Any kwargs you want passed to `modelbridge.gen`
                )
                ]
            )
        generated_run = self.gs.gen(
            # Ax `Experiment`, for which to generate new candidates
            experiment=self.ax_client.experiment,
            data=None,  # Ax `Data` to use for model training, optional.
            n=self._N,  # Number of candidate arms to produce
            fixed_features=self._fixed_features,
        )
        self.generator_run = generated_run[0][0]

    def clear_trials(self):
        """
        Clear all trials in the experiment.
        """
        # Get all pending trial indices
        pending_trials = [k for k, i in self.ax_client.experiment.trials.items()
                          if i.status == TrialStatus.CANDIDATE]
        for i in pending_trials:
            self.ax_client.experiment.trials[i].mark_abandoned()

    def suggest_next_trials(self, with_predicted=True):
        """
        Suggest the next set of trials based on the current model and optimization strategy.

        Returns
        -------

        pd.DataFrame: 
            DataFrame containing the suggested trials and their predicted outcomes.
        """
        self.clear_trials()
        if self.ax_client is None:
            self.initialize_ax_client()
        if self._N == 1:
            self.candidate = self.ax_client.experiment.new_trial(
                self.generator_run)
        else:
            self.candidate = self.ax_client.experiment.new_batch_trial(
                self.generator_run)
        if hasattr(self.candidate, "arms"):
            arm_parameters = [arm.parameters for arm in self.candidate.arms]
        else:
            arm_parameters = [self.candidate.arm.parameters]
        trials = pd.DataFrame(arm_parameters)
        trials = trials[[name for name in self.names]]
        if with_predicted and self.model is None:
            return trials.reset_index(drop=True)
        if with_predicted:
            topred = [trials.iloc[i].to_dict() for i in range(len(trials))]
            preds = self.predict(topred)[0]
            preds = pd.DataFrame(preds)
            # add 'predicted_' to the names of the pred dataframe
            preds.columns = [f'Predicted_{col}' for col in preds.columns]
            preds = preds.reset_index(drop=True)
            trials = trials.reset_index(drop=True)
            return pd.concat([trials, preds], axis=1)
        else:
            return trials

    def _get_observed_best_parameters(self):
        """Return best observed rows when no fitted Ax model is available yet."""
        data = self.data.copy()
        objective_names = [
            name for name, maximize in self._maximize.items() if isinstance(maximize, bool)]
        if len(objective_names) == 0:
            return pd.DataFrame()

        data = data.dropna(subset=objective_names)
        if data.empty:
            return pd.DataFrame(columns=self.data.columns)

        if len(objective_names) == 1:
            objective_name = objective_names[0]
            if self._maximize[objective_name]:
                best_index = data[objective_name].idxmax()
            else:
                best_index = data[objective_name].idxmin()
            return data.loc[[best_index]].reset_index(drop=True)

        return data.reset_index(drop=True)

    def predict(self, params):
        """
        Predict the outcomes for a given set of parameters using the current model.

        Parameters
        ----------

        params : List[Dict[str, Any]]
            List of parameter dictionaries for which to predict outcomes.

        Returns
        -------

        List[Dict[str, float]]: 
            List of predicted outcomes for the given parameters.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        if self.model is None:
            raise ValueError(
                "Predictions require at least one completed experiment with numeric outcome data.")
        obs_feats = [ObservationFeatures(parameters=p) for p in params]
        f, cm = self.model.predict(obs_feats)
        # return prediction and std errors as a list of dictionaries
        # Convert to list of dictionaries
        predictions = []
        for i in range(len(obs_feats)):
            pred_dict = {}
            for metric_name in f.keys():
                pred_dict[metric_name] = {
                    'mean': f[metric_name][i],
                    'std': np.sqrt(cm[metric_name][metric_name][i])
                }
            predictions.append(pred_dict)
        preds = [{k: v['mean'] for k, v in pred.items()}
                 for pred in predictions]
        stderrs = [{k: v['std'] for k, v in pred.items()}
                   for pred in predictions]
        return preds, stderrs

    def update_experiment(self, params, outcomes):
        """
        Update the experiment with new parameters and outcomes, and reinitialize the AxClient.

        Parameters
        ----------

        params : Dict[str, Any]
            Dictionary of new parameters to update the experiment with.

        outcomes : Dict[str, Any]
            Dictionary of new outcomes to update the experiment with.
        """
        # append new data to the features and outcomes dictionaries
        for k, v in zip(params.keys(), params.values()):
            if k not in self._features:
                raise ValueError(f"Parameter '{k}' not found in features")
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if not isinstance(v, list):
                v = [v]
            self._features[k]['data'] += v
        for k, v in zip(outcomes.keys(), outcomes.values()):
            if k not in self._outcomes:
                raise ValueError(f"Outcome '{k}' not found in outcomes")
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if not isinstance(v, list):
                v = [v]
            self._outcomes[k]['data'] += v
        self.initialize_ax_client()

    def plot_model(self, metricname=None, slice_values={}, linear=False):
        """
        Plot the model's predictions for the experiment's parameters and outcomes.
        Parameters
        ----------
        metricname : Optional[str]
            The name of the metric to plot. If None, the first outcome metric is used.
        slice_values : Optional[Dict[str, Any]]
            Dictionary of slice values for plotting.
        linear : bool
            Whether to plot a linear slice plot. Default is False.
        Returns
        -------
        plotly.graph_objects.Figure: 
            Plotly figure of the model's predictions.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
            self.suggest_next_trials()
        cand_name = 'Candidate' if self._N == 1 else 'Candidates'
        mname = self.ax_client.objective_names[0] if metricname is None else metricname
        param_name = [
            name for name in self.names if name not in slice_values.keys()]
        par_numeric = [
            name for name in param_name if self._features[name]['type'] in ['int', 'float']]

        if self.model is None:
            completed_trials = self.ax_client.get_trials_data_frame()
            completed_trials = completed_trials[completed_trials['trial_status'] != 'CANDIDATE'].copy(
            )
            if mname not in completed_trials.columns:
                return go.Figure()

            if len(par_numeric) == 0:
                return go.Figure()

            if len(par_numeric) == 1:
                fig = px.scatter(
                    completed_trials,
                    x=par_numeric[0],
                    y=mname,
                    title=f"Observed {mname} vs {par_numeric[0]}",
                )
            elif len(par_numeric) == 2:
                fig = px.scatter(
                    completed_trials,
                    x=par_numeric[0],
                    y=par_numeric[1],
                    color=mname,
                    color_continuous_scale="Viridis",
                    title=f"Observed {par_numeric[1]} vs {par_numeric[0]}",
                )
            else:
                fig = px.scatter_matrix(
                    completed_trials,
                    dimensions=par_numeric + [mname],
                    title=f"Observed relationships for {mname}",
                )

            return fig

        if len(par_numeric) == 1:
            fig = plot_slice(
                model=self.model,
                metric_name=mname,
                density=100,
                param_name=par_numeric[0],
                generator_runs_dict={cand_name: self.generator_run},
                slice_values=slice_values
            )
        elif len(par_numeric) == 2:
            fig = plot_contour(
                model=self.model,
                metric_name=mname,
                param_x=par_numeric[0],
                param_y=par_numeric[1],
                generator_runs_dict={cand_name: self.generator_run},
                slice_values=slice_values
            )
        else:
            # remove sliced parameters from par_numeric
            pars = [p for p in par_numeric if p not in slice_values.keys()]
            if interact_contour is not None:
                fig = interact_contour(
                    model=self.model,
                    generator_runs_dict={cand_name: self.generator_run},
                    metric_name=mname,
                    slice_values=slice_values,
                    parameters_to_use=pars
                )
            elif len(pars) >= 2:
                fig = plot_contour(
                    model=self.model,
                    metric_name=mname,
                    param_x=pars[0],
                    param_y=pars[1],
                    generator_runs_dict={cand_name: self.generator_run},
                    slice_values=slice_values
                )
            elif len(pars) == 1:
                fig = plot_slice(
                    model=self.model,
                    metric_name=mname,
                    density=100,
                    param_name=pars[0],
                    generator_runs_dict={cand_name: self.generator_run},
                    slice_values=slice_values
                )
            else:
                return go.Figure()

        plotly_fig = go.Figure(fig.data)
        all_trials = self.ax_client.get_trials_data_frame()
        completed_trials = all_trials[all_trials['trial_status'] != 'CANDIDATE'].copy(
        )
        # compute distance to slice
        col_to_consider = completed_trials[[k for k in slice_values.keys()]]
        completed_trials.loc[:, 'signed_dist_to_slice'] = (
            # Sum of signed differences
            (col_to_consider - slice_values).sum(axis=1)
        )
        signed_dists = completed_trials['signed_dist_to_slice'].values
        positive_dists = signed_dists[signed_dists >= 0]
        negative_dists = signed_dists[signed_dists < 0]

        # Normalize positive distances to [0, 1]
        if len(positive_dists) > 0 and np.max(positive_dists) > 0:
            normalized_positive = positive_dists / np.max(positive_dists)
        else:
            normalized_positive = np.zeros_like(positive_dists)

        # Normalize negative distances to [-1, 0]
        if len(negative_dists) > 0 and np.min(negative_dists) < 0:
            normalized_negative = negative_dists / \
                np.abs(np.min(negative_dists))
        else:
            normalized_negative = np.zeros_like(negative_dists)

        # Combine the normalized distances
        normalized_signed_dists = np.zeros_like(signed_dists)
        normalized_signed_dists[signed_dists >= 0] = normalized_positive
        normalized_signed_dists[signed_dists < 0] = normalized_negative

        completed_trials.loc[:,
                             'normalized_signed_dist'] = normalized_signed_dists
        coolwarm = mpl.colormaps['bwr']
        # Map from [-1,1] to [0,1]
        normalized_values = (
            completed_trials['normalized_signed_dist'] + 1) / 2
        colors = [
            f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
            for r, g, b, _ in coolwarm(normalized_values)
        ]
        completed_trials.loc[:, 'colors'] = colors
        trials = self.ax_client.get_trials_data_frame()
        trials = trials[trials['trial_status'] == 'CANDIDATE']
        trials = trials[[name for name in self.names]]

        in_sample_trace_idx = 0
        for trace in plotly_fig.data:
            if trace.type == "contour":
                trace.colorscale = "viridis"
            if 'marker' in trace and trace.legendgroup != cand_name:
                arm_names = []
                if trace['text']:
                    for text in trace['text']:
                        print(text)
                        match = re.search(r'Arm (\d+_\d+)', text)
                        if match:
                            arm_names.append(match.group(1))
                    arm_to_color = dict(
                        zip(completed_trials['arm_name'], completed_trials['colors']))
                    trace.marker.color = [arm_to_color[arm]
                                          for arm in arm_names]
                trace.marker.symbol = "circle"
                trace.marker.size = 10
                trace.marker.line.width = 2
                trace.marker.line.color = 'black'
                # if len(opacities) > 0:
                # trace.marker.opacity = opacities
                if trace.text is not None:
                    trace.text = [t.replace('Arm', '<b>Sample').replace(
                        "_0", "</b>") for t in trace.text]
            if trace.legendgroup == cand_name:
                trace.marker.line.color = 'red'
                trace.marker.color = "orange"
                trace.name = cand_name
                trace.marker.symbol = "x"
                trace.marker.size = 12
                trace.marker.opacity = 1
                trace.hoverinfo = "text"
                trace.hoverlabel = dict(bgcolor="#f8e3cd", font_color='black')
                if trace.text is not None:
                    trace.text = [t.replace("<i>", "").replace(
                        "</i>", "") for t in trace.text]
                trace.text = [
                    "<b>Candidate {}</b><br>{}".format(
                        i + 1,
                        "<br>".join(
                            ["{}: {}".format(col, val)
                             for col, val in trials.iloc[i].items()]
                        ),
                    )
                    for t in trace.text
                    for i in range(len(trials))
                ]

        plotly_fig.update_layout(
            plot_bgcolor="white",
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
            xaxis=dict(
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                zerolinecolor="black",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                zerolinecolor="black",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True
            ),
            xaxis2=dict(
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                zerolinecolor="black",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True
            ),
            yaxis2=dict(
                showgrid=True,
                gridcolor="lightgray",
                zeroline=False,
                zerolinecolor="black",
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True
            ),
        )
        return plotly_fig

    def plot_optimization_trace(self, optimum=None):
        """
        Plot the optimization trace, showing the progress of the optimization over trials.

        Parameters
        ----------

        optimum : Optional[float]
            The optimal value to plot on the optimization trace.

        Returns
        -------

        plotly.graph_objects.Figure: 
            Plotly figure of the optimization trace.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        if len(self._outcomes) > 1:
            print("Optimization trace is not available for multi-objective optimization.")
            return None
        optimization_config = self.ax_client.experiment.optimization_config
        objective = optimization_config.objective
        metric_name = objective.metric_names[0]
        data = self.ax_client.experiment.fetch_data()
        if data is None or getattr(data, "df", None) is None:
            return None
        df = data.df
        df = df[df["metric_name"] == metric_name].sort_values("trial_index")
        if df.empty:
            return None

        y = df["mean"].to_numpy(dtype=float)[None, :]
        fig = optimization_trace_single_method_plotly(
            y=y,
            optimum=optimum,
            ylabel=metric_name,
            optimization_direction="minimize" if objective.minimize else "maximize",
        )
        for trace in fig.data:
            # add hover info
            trace.hoverinfo = "x+y"
        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=50, r=10, t=50, b=50),
            xaxis=dict(
                showgrid=True,  # Enable grid
                gridcolor="lightgray",  # Light gray grid lines
                zeroline=False,
                zerolinecolor="black",  # Black zero line
                showline=True,
                linewidth=1,
                linecolor="black",  # Black border
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,  # Enable grid
                gridcolor="lightgray",  # Light gray grid lines
                zeroline=False,
                zerolinecolor="black",  # Black zero line
                showline=True,
                linewidth=1,
                linecolor="black",  # Black border
                mirror=True
            ),
        )
        return fig

    def plot_feature_importances(self, relative=False, per_objective=False):
        """
        Plot feature importances using Ax default Sensitivity Analysis cards
        (same analysis family as in Ax tutorials).

        Parameters
        ----------
        relative : bool, optional
            Used only by the fallback Ax helper plot if analysis cards are
            unavailable. Default is False.
        per_objective : bool, optional
            If True and the experiment is multiobjective, return one figure per
            objective instead of only the first objective's figure.

        Returns
        -------
        plotly.graph_objects.Figure | list[tuple[str, plotly.graph_objects.Figure]]:
            Plotly figure of feature importances, or a list of named figures
            when per_objective=True in a multiobjective experiment.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        if self.model is None:
            self.set_model()

        def _style_sensitivity_figure(fig):
            import re as _re

            # Build lookup: full _OH_PARAM_N key → human-readable label
            oh_label_map = {}
            for name, info in self._features.items():
                if info['type'] == 'text':
                    for i, val in enumerate(info.get('range', [])):
                        oh_label_map[f"{name}_OH_PARAM_{i}"] = f"{name}: {val}"

            def resolve_label(raw):
                if not isinstance(raw, str):
                    return raw
                return oh_label_map.get(raw, raw)

            for trace in fig.data:
                trace_name = str(getattr(trace, "name", "") or "")
                if "Increases" in trace_name:
                    trace.marker.color = "#2ca02c"
                elif "Decreases" in trace_name:
                    trace.marker.color = "#d62728"

                if getattr(trace, "type", None) == "bar":
                    trace.width = 0.85
                    trace.offsetgroup = None
                    trace.alignmentgroup = None

                    # customdata[0] holds the FULL _OH_PARAM_N key (first pass)
                    # or already-resolved label (second pass) — resolve either way
                    cd = getattr(trace, "customdata", None)
                    if cd is not None:
                        cd_arr = np.array(cd, dtype=object)
                        resolved_labels = []
                        for ri in range(cd_arr.shape[0]):
                            resolved = resolve_label(str(cd_arr[ri, 0]))
                            cd_arr[ri, 0] = resolved
                            resolved_labels.append(resolved)
                        trace.customdata = cd_arr
                        # Override trace.y with the resolved labels from customdata
                        # (trace.y contains Ax's pre-truncated display strings — ignore them)
                        trace.y = tuple(resolved_labels)

                    # Fix hovertemplate:
                    # - remove the truncated_parameter_name=%{y} line (y is now resolved anyway)
                    # - keep parameter_name pointing at customdata[0] which is now resolved
                    ht = getattr(trace, "hovertemplate", None)
                    if ht is not None:
                        ht = ht.replace(
                            "truncated_parameter_name=%{y}<br>", "")
                        ht = ht.replace(
                            "truncated_parameter_name=%{y}<br />", "")
                        # if hovertemplate still uses %{y} for parameter_name, redirect to customdata
                        ht = ht.replace(
                            "parameter_name=%{y}", "parameter_name=%{customdata[0]}")
                        trace.hovertemplate = ht

            fig.update_layout(barmode="overlay")
            fig.update_yaxes(title_text="", automargin=True)
            fig.update_xaxes(title_text="Importance")
            return fig

        metric_names = list(self.ax_client.objective_names)
        if len(metric_names) == 0:
            return None

        figures = []
        labels = []
        for metric_name in metric_names:
            try:
                card = SensitivityAnalysisPlot(metric_name=metric_name).compute(
                    experiment=self.ax_client.experiment,
                    generation_strategy=self.gs,
                    adapter=self.model,
                )
                figures.append(_style_sensitivity_figure(card.get_figure()))
                labels.append(metric_name)
            except (AttributeError, RuntimeError, ValueError, TypeError, KeyError):
                continue

        if len(figures) == 0:
            # Fallback to legacy Ax plot helper if analysis cards are unavailable.
            try:
                fig = plot_feature_importance_by_feature_plotly(
                    model=self.model,
                    relative=relative,
                )
                return _style_sensitivity_figure(fig)
            except (AttributeError, RuntimeError, ValueError, TypeError, KeyError):
                return None

        if len(figures) == 1:
            return _style_sensitivity_figure(figures[0])

        if per_objective:
            return [
                (labels[i], _style_sensitivity_figure(fig))
                for i, fig in enumerate(figures)
            ]

        # Keep backward-compatible behavior for callers that expect a single
        # figure: default to the first objective.
        return _style_sensitivity_figure(figures[0])

    def compute_pareto_frontier(self):
        """
        Compute the Pareto frontier for multi-objective optimization experiments.

        Returns
        -------
        The Pareto frontier.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        if len(self._outcomes) < 2:
            print("Pareto frontier is not available for single-objective optimization.")
            return None

        if self.Nmetrics == 2:
            objective = self.ax_client.experiment.optimization_config.objective
            metric_names = list(objective.metric_names)
            metrics = self.ax_client.experiment.metrics
            self.pareto_frontier = compute_posterior_pareto_frontier(
                experiment=self.ax_client.experiment,
                data=self.ax_client.experiment.fetch_data(),
                primary_objective=metrics[metric_names[1]],
                secondary_objective=metrics[metric_names[0]],
                absolute_metrics=metric_names,
                num_points=20,
            )
        else:
            # For 3+ objectives, keep Pareto-optimal points and visualize in plot_pareto_frontier.
            self.pareto_frontier = self.ax_client.get_pareto_optimal_parameters()
        return self.pareto_frontier

    def plot_pareto_frontier(self, show_error_bars=True):
        """
        Plot the Pareto frontier for multi-objective optimization experiments.

        Parameters
        ----------
        show_error_bars : bool, optional
            Whether to show error bars on the plot. Default is True.

        Returns
        -------
        plotly.graph_objects.Figure: 
            Plotly figure of the Pareto frontier.
        """
        if self.pareto_frontier is None:
            return None

        if self.Nmetrics > 2:
            df = ordered_dict_to_dataframe(self.pareto_frontier)
            objective_names = [
                name for name in self.ax_client.objective_names if name in df.columns]
            if len(objective_names) < 2:
                return None
            fig = px.scatter_matrix(
                df,
                dimensions=objective_names,
                hover_data=[name for name in self.names if name in df.columns],
                title="Pareto-optimal objective trade-offs",
            )
            fig.update_traces(diagonal_visible=False)
        else:
            objective = self.ax_client.experiment.optimization_config.objective
            metric_names = list(objective.metric_names)
            if len(metric_names) < 2:
                return None
            x_metric, y_metric = metric_names[0], metric_names[1]
            x_maximize = bool(self._maximize.get(x_metric, True))
            y_maximize = bool(self._maximize.get(y_metric, True))

            data = self.ax_client.experiment.fetch_data()
            if data is None or getattr(data, "df", None) is None:
                return None
            df_raw = data.df
            if df_raw.empty:
                return None

            means = df_raw.pivot_table(
                index="trial_index", columns="metric_name", values="mean", aggfunc="first"
            )
            sems = df_raw.pivot_table(
                index="trial_index", columns="metric_name", values="sem", aggfunc="first"
            )

            if x_metric not in means.columns or y_metric not in means.columns:
                return None

            means = means[[x_metric, y_metric]].dropna()
            if means.empty:
                return None

            points = means.to_numpy(dtype=float)

            def _dominates(a, b):
                x_be = a[0] >= b[0] if x_maximize else a[0] <= b[0]
                y_be = a[1] >= b[1] if y_maximize else a[1] <= b[1]
                x_strict = a[0] > b[0] if x_maximize else a[0] < b[0]
                y_strict = a[1] > b[1] if y_maximize else a[1] < b[1]
                return (x_be and y_be) and (x_strict or y_strict)

            is_pareto = np.ones(len(points), dtype=bool)
            for i in range(len(points)):
                if not is_pareto[i]:
                    continue
                for j in range(len(points)):
                    if i == j:
                        continue
                    if _dominates(points[j], points[i]):
                        is_pareto[i] = False
                        break

            frontier = means.iloc[is_pareto].copy()
            frontier = frontier.sort_values(x_metric, ascending=True)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=means[x_metric],
                    y=means[y_metric],
                    mode="markers",
                    name="Observed Trials",
                    marker=dict(color="#4c78a8", size=9, opacity=0.75),
                    text=[f"Trial {idx}" for idx in means.index],
                    hovertemplate="%{text}<br>" +
                    f"{x_metric}: %{{x}}<br>{y_metric}: %{{y}}<extra></extra>",
                )
            )

            frontier_error_x = None
            frontier_error_y = None
            if show_error_bars and not sems.empty:
                if x_metric in sems.columns and y_metric in sems.columns:
                    sem_front = sems.loc[frontier.index]
                    frontier_error_x = sem_front[x_metric].to_numpy(
                        dtype=float)
                    frontier_error_y = sem_front[y_metric].to_numpy(
                        dtype=float)

            fig.add_trace(
                go.Scatter(
                    x=frontier[x_metric],
                    y=frontier[y_metric],
                    mode="lines+markers",
                    name="Pareto Frontier",
                    line=dict(color="#f2c84b", width=3, dash="dash"),
                    marker=dict(
                        color="#f2c84b",
                        size=14,
                        symbol="diamond",
                        line=dict(color="black", width=1.5),
                    ),
                    error_x=(dict(type="data", array=frontier_error_x, visible=True)
                             if frontier_error_x is not None else None),
                    error_y=(dict(type="data", array=frontier_error_y, visible=True)
                             if frontier_error_y is not None else None),
                    hovertemplate=f"{x_metric}: %{{x}}<br>{y_metric}: %{{y}}<extra></extra>",
                )
            )

            fig.update_layout(
                title=f"Pareto Frontier ({len(frontier)} point{'s' if len(frontier) != 1 else ''})",
                xaxis_title=x_metric,
                yaxis_title=y_metric,
            )

        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=50, r=10, t=50, b=50),
            xaxis=dict(
                showgrid=True,  # Enable grid
                gridcolor="lightgray",  # Light gray grid lines
                zeroline=False,
                zerolinecolor="black",  # Black zero line
                showline=True,
                linewidth=1,
                linecolor="black",  # Black border
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,  # Enable grid
                gridcolor="lightgray",  # Light gray grid lines
                zeroline=False,
                zerolinecolor="black",  # Black zero line
                showline=True,
                linewidth=1,
                linecolor="black",  # Black border
                mirror=True
            ),
        )
        return fig

    def get_best_parameters(self):
        """
        Return the best parameters found by the optimization process.

        Returns
        -------

        pd.DataFrame: 
            DataFrame containing the best parameters and their outcomes.
        """
        if self.ax_client is None:
            self.initialize_ax_client()
        if self.model is None:
            return self._get_observed_best_parameters()
        if self.Nmetrics == 1:
            best_result = self.ax_client.get_best_parameters()
            if best_result is None or best_result[0] is None or best_result[1] is None:
                return self._get_observed_best_parameters()

            best_parameters = best_result[0]
            best_outcomes = best_result[1]
            best_parameters.update(best_outcomes[0])
            best = pd.DataFrame(best_parameters, index=[0])
        else:
            best_parameters = self.ax_client.get_pareto_optimal_parameters()
            if best_parameters is None:
                return self._get_observed_best_parameters()
            best = ordered_dict_to_dataframe(best_parameters)
        return best

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def flatten_dict(d, parent_key="", sep="_"):
    """
    Flatten a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def ordered_dict_to_dataframe(data):
    """
    Convert an OrderedDict with arbitrary nesting to a DataFrame.
    """
    dflat = flatten_dict(data)
    out = []

    for key, value in dflat.items():
        main_dict = value[0]
        sub_dict = value[1][0]
        out.append([value for value in main_dict.values()] +
                   [value for value in sub_dict.values()])

    df = pd.DataFrame(out, columns=[key for key in main_dict.keys()] +
                                   [key for key in sub_dict.keys()])
    return df

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def read_experimental_data(file_path: str, out_pos=[-1]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Read experimental data from a CSV file and format it into features and outcomes dictionaries.

    Parameters
    ----------
    file_path (str) 
        Path to the CSV file containing experimental data.
    out_pos (list of int)
        Column indices of the outcome variables. Default is the last column.

    Returns
    -------
    Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]
        Formatted features and outcomes dictionaries.
    """
    data = pd.read_csv(file_path)
    data = clean_names(data, remove_special=True, case_type='preserve')
    outcome_column_name = data.columns[out_pos]
    features = data.loc[:, ~data.columns.isin(outcome_column_name)].copy()
    outcomes = data[outcome_column_name].copy()

    feature_definitions = {}
    for column in features.columns:
        if features[column].dtype == 'object':
            unique_values = features[column].unique()
            feature_definitions[column] = {'type': 'text',
                                           'range': unique_values.tolist()}
        elif features[column].dtype in ['int64', 'float64']:
            min_val = features[column].min()
            max_val = features[column].max()
            feature_type = 'int' if features[column].dtype == 'int64' else 'float'
            feature_definitions[column] = {'type': feature_type,
                                           'range': [min_val, max_val]}

    formatted_features = {name: {'type': info['type'],
                                 'data': features[name].tolist(),
                                 'range': info['range']}
                          for name, info in feature_definitions.items()}
    # same for outcomes with just type and data
    outcome_definitions = {}
    for column in outcomes.columns:
        if outcomes[column].dtype == 'object':
            unique_values = outcomes[column].unique()
            outcome_definitions[column] = {'type': 'text',
                                           'data': unique_values.tolist()}
        elif outcomes[column].dtype in ['int64', 'float64']:
            min_val = outcomes[column].min()
            max_val = outcomes[column].max()
            outcome_type = 'int' if outcomes[column].dtype == 'int64' else 'float'
            outcome_definitions[column] = {'type': outcome_type,
                                           'data': outcomes[column].tolist()}
    formatted_outcomes = {name: {'type': info['type'],
                                 'data': outcomes[name].tolist()}
                          for name, info in outcome_definitions.items()}
    return formatted_features, formatted_outcomes
