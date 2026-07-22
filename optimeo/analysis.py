# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

"""
The analysis module provides tools for data analysis and regression modeling.
The main workhorse is the ``DataAnalysis`` class, which allows for encoding categorical variables, performing regression analysis, and visualizing results.

It supports both linear regression using the ``statsmodels`` package and machine learning models from ``sklearn``.
The class also provides methods for plotting Q-Q plots, box plots, histograms, and scatter plots.
It includes functionality for bootstrap resampling to estimate the variability of model coefficients.
The ``DataAnalysis`` class is designed to be flexible and extensible, allowing users to customize the regression analysis process.

You can see an example notebook `here <../examples/MLanalysis.ipynb>`_.

"""


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import t
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.graphics.gofplots import qqplot
# import ML models and scaling functions
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import seaborn as sns


class DataAnalysis:
    """
    This class is used to analyze the data and perform regression analysis.

    Example
    -------

    .. code-block:: python

        import pandas as pd
        from optimeo.analysis import DataAnalysis

        data = pd.read_csv('dataML.csv')
        factors = data.columns[:-1].tolist()
        response = data.columns[-1]
        analysis = DataAnalysis(data, factors, response)
        analysis.model_type = "ElasticNetCV"
        analysis.compute_ML_model()

    Parameters
    ----------

    data : pd.DataFrame
        The input data.
    factors : list
        The list of factor variables.
    response : str
        The response variable.
    split_size : float, optional
        The proportion of the dataset to include in the test split. Default is ``0.2``.
    model_type : str, optional
        The type of machine learning model to use. Default is None. 
        Must be one of the following: ``"ElasticNetCV"``, ``"RidgeCV"``,
        ``"LinearRegression"``, ``"RandomForest"``, ``"GaussianProcess"``, ``"GradientBoosting"``.

    Attributes
    ----------

    data : pd.DataFrame
        The input data.
    factors : list
        The list of factor variables.
    response : str
        The response variable.
    encoders : dict
        The encoders for categorical variables.
    dtypes : pd.Series  
        The data types of the columns.
    linear_model : object
        The linear model object.
    equation : str
        The equation for the linear model, in the form ``"response ~ var1 + var2 + var1:var2"``.
    model : object
        The machine learning model object.
    model_type : str
        The type of machine learning model to use.
    split_size : float
        The proportion of the dataset to include in the test split.

    Methods
    -------
    encode_data()
        Encode categorical variables in input data.
    plot_qq()
        Plot a Q-Q plot of residuals.
    plot_boxplot()
        Plot response distribution as boxplot.
    plot_histogram()
        Plot response distribution as histogram.
    plot_scatter_response()
        Plot scatter response against factors.
    plot_corr()
        Plot factor correlation matrix.
    plot_pairplot_seaborn()
        Generate pairplot with seaborn.
    plot_pairplot_plotly()
        Generate pairplot with plotly.
    write_equation(order=1, quadratic=[])
        Build formula string for statsmodels.
    compute_linear_model(order=1, quadratic=[])
        Fit linear model with statsmodels.
    plot_linear_model()
        Visualize linear model effects.
    compute_ML_model(**kwargs)
        Fit machine-learning model.
    plot_ML_model(features_in_log=False)
        Plot ML model response surfaces.

    """

    def __init__(self,
                 data: pd.DataFrame,
                 factors: list,
                 response: str,
                 split_size=.2,
                 model_type=None):
        self._dtypes = None
        self._encoders = {}
        self._linear_model = None
        self._model = None
        self._equation = ''
        self._data = data
        self._factors = factors
        self._response = response
        self._model_type = model_type
        self._split_size = split_size
        self.encode_data()

    @property
    def data(self):
        """The input `pandas.DataFrame`."""
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")
        self._data = value

    def encode_data(self):
        """
        Called during initialization: encodes categorical variables in the data if there are any. 
        Uses ``LabelEncoder()`` from ``sklearn`` to convert categorical variables to numerical values.
        Also drops rows with NaN values.
        """
        self._dtypes = self._data.dtypes
        for factor in self._factors:
            if self._dtypes[factor] == 'object':
                le = LabelEncoder()
                self._encoders[factor] = le
                self._data[factor] = le.fit_transform(
                    [str(d) for d in self._data[factor]])
        self.data = self.data[self._factors + [self._response]]
        self.data = self.data.dropna(axis=0, how='any')

    @property
    def factors(self):
        """The list of names of the columns of the ``data`` DataFrame that contain factor variables."""
        return self._factors

    @factors.setter
    def factors(self, value):
        if not isinstance(value, list):
            raise ValueError("Factors must be a list.")
        self._factors = value

    @property
    def encoders(self):
        """The list of encoders for categorical variables."""
        return self._encoders

    @encoders.setter
    def encoders(self, value):
        if not isinstance(value, dict):
            raise ValueError("Encoders must be a dictionary.")
        self._encoders = value

    @property
    def dtypes(self):
        """Get the data types of the columns."""
        return self._dtypes

    @dtypes.setter
    def dtypes(self, value):
        self._dtypes = value

    @property
    def response(self):
        """The name of the column of the `data` DataFrame that contain the response variable."""
        return self._response

    @response.setter
    def response(self, value):
        if not isinstance(value, str):
            raise ValueError("Response must be a string.")
        self._response = value

    @property
    def linear_model(self):
        """Get the linear model."""
        return self._linear_model

    @linear_model.setter
    def linear_model(self, value):
        self._linear_model = value

    @property
    def equation(self):
        """The equation for the linear model, in the form ``"response ~ var1 + var2 + var1:var2"``.

        See statsmodels formula examples:
        https://www.statsmodels.org/dev/examples/notebooks/generated/formulas.html
        """
        return self._equation

    @equation.setter
    def equation(self, value):
        if not isinstance(value, str):
            raise ValueError("Equation must be a string.")
        self._equation = value

    @property
    def model_type(self):
        """The type of machine learning model to use. Default is None. 
            Must be one of the following: ``"ElasticNetCV"``, ``"RidgeCV"``,
            ``"LinearRegression"``, ``"RandomForest"``, ``"GaussianProcess"``, ``"GradientBoosting"``."""
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        if value is not None and value not in ["ElasticNetCV", "RidgeCV",
                                               "LinearRegression", "RandomForest",
                                               "GaussianProcess", "GradientBoosting"]:
            raise ValueError("Model must be one of the following: "
                             "ElasticNetCV, RidgeCV, LinearRegression, "
                             "RandomForest, GaussianProcess, GradientBoosting.")
        self._model_type = value

    @property
    def model(self):
        """The machine learning model object."""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def split_size(self):
        """The proportion of the dataset to include in the test split. Default is ``0.2``."""
        return self._split_size

    @split_size.setter
    def split_size(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("Split size must be a number.")
        self._split_size = value

    def __str__(self):
        """Return a string representation of the DataAnalysis object."""
        return (f"DataAnalysis(data={self._data.shape}, "
                f"factors={self._factors}, "
                f"response={self._response}, "
                f"model_type={self._model_type}, "
                f"split_size={self._split_size}, "
                f"encoders={self._encoders})"
                )

    def __repr__(self):
        """Return a string representation of the DataAnalysis object."""
        return self.__str__()

    def plot_qq(self):
        """
        Plot a Q-Q plot for the response variable.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The Q-Q plot figure.
        """
        qqplot_data = pd.Series(self._data[self._response], copy=True)
        qqplot_data = qqplot(qqplot_data, line='s').gca().lines
        fig = go.Figure()

        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[0].get_xdata(),
            'y': qqplot_data[0].get_ydata(),
            'mode': 'markers',
        })

        fig.add_trace({
            'type': 'scatter',
            'x': qqplot_data[1].get_xdata(),
            'y': qqplot_data[1].get_ydata(),
            'mode': 'lines',
            'line': {
                'color': 'red'
            }
        })

        fig.update_layout(
            title='Q-Q Plot',
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            plot_bgcolor="white",  # White background
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=50),
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
            height=300,
        )
        return fig

    def plot_boxplot(self):
        """
        Plot a boxplot for the response variable.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The boxplot figure.
        """
        fig = px.box(self._data, y=self._response,
                     points="all", title='Box Plot')
        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
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
            ), height=300,
        )
        return fig

    def plot_histogram(self):
        """
        Plot a histogram for the response variable.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The histogram figure.
        """
        fig = px.histogram(self._data, x=self._response, title='Histogram')
        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
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
            ), height=300,
        )
        return fig

    def plot_scatter_response(self):
        """
        Plot a scatter plot for the response variable.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The scatter plot figure.
        """
        fig = px.scatter(x=np.arange(1, len(self._data[self._response]) + 1),
                         y=self._data[self._response],
                         labels={'x': 'Measurement number',
                                 'y': self._response},
                         title='Scatter Plot')
        fig.update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
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
            ), height=300,
        )
        return fig

    def plot_pairplot_seaborn(self):
        """
        Plot a pairplot for the data using seaborn.

        Returns
        -------
        fig : seaborn.axisgrid.PairGrid
            The pairplot figure.
        """
        fig = sns.pairplot(
            self._data,
            kind="reg",
            diag_kind="kde",
            plot_kws={"scatter_kws": {"alpha": 0.1}},
        )
        return fig

    def plot_pairplot_plotly(self):
        """
        Plot a pairplot for the data using plotly.

        Returns
        -------
        fig : plotly.graph_objs.Figure
            The plotly figure.
        """

        # Get the column names
        columns = self._data.columns
        n_cols = len(columns)

        # Create subplot grid with partially shared axes
        fig = make_subplots(
            rows=n_cols,
            cols=n_cols,
            shared_xaxes=True,
            shared_yaxes=True,
            vertical_spacing=0.05,
            horizontal_spacing=0.05
        )

        # Add scatter plots and regression lines to each subplot
        for i in range(n_cols):
            for j in range(n_cols):
                x_data = self._data[columns[j]]
                y_data = self._data[columns[i]]

                if i == j:  # Diagonal: Add KDE plot
                    # Calculate KDE
                    data = self._data[columns[i]].dropna()
                    if len(data) > 1:  # Ensure enough data points
                        kde_x = np.linspace(data.min(), data.max(), 100)
                        kde = stats.gaussian_kde(data)
                        kde_y = kde(kde_x)
                        # Scale to match y-axis (needed for top left corner)
                        kde_y = kde_y / kde_y.max() * np.max(y_data.dropna())

                        # Add KDE plot to diagonal
                        fig.add_trace(
                            go.Scatter(
                                x=kde_x,
                                y=kde_y,
                                mode='lines',
                                fill='tozeroy',
                                line=dict(color='rgba(29, 81, 189, 1)'),
                                showlegend=False
                            ),
                            row=i+1, col=j+1
                        )
                        # Ensure the y-axis is independent for diagonal plots
                        # don't know why, doesn't work with top left corner
                        fig.update_yaxes(
                            matches=None,  # Ensure independent y-axis
                            showticklabels=True,  # Show y-axis values
                            row=i+1,
                            col=j+1
                        )
                else:  # Off-diagonal: Add scatter plot with regression line
                    # Add scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='markers',
                            marker=dict(
                                color='rgba(29, 81, 189, 0.5)',
                                size=5
                            ),
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )

                    # Calculate and add regression line
                    if len(x_data.dropna()) > 1 and len(y_data.dropna()) > 1:
                        # Drop NaN values for regression calculation
                        valid_indices = x_data.notna() & y_data.notna()
                        x_clean = x_data[valid_indices]
                        y_clean = y_data[valid_indices]
                        ytype = y_clean.dtype
                        xtype = x_clean.dtype
                        unique_x = x_clean.unique()
                        if len(x_clean) > 1 and len(unique_x) > 1 and ytype != 'object' and xtype != 'object':
                            # Calculate regression parameters
                            slope, intercept, r_value, p_value, std_err = stats.linregress(
                                x_clean, y_clean)
                            x_range = np.linspace(
                                x_clean.min(), x_clean.max(), 100)
                            y_pred = slope * x_range + intercept

                            # Standard error of estimate
                            y_fit = slope * x_clean + intercept
                            residuals = y_clean - y_fit
                            dof = len(x_clean) - 2
                            residual_std_error = np.sqrt(
                                np.sum(residuals**2) / dof)

                            mean_x = np.mean(x_clean)
                            t_val = t.ppf(0.975, dof)  # 95% confidence

                            se_line = residual_std_error * \
                                np.sqrt(1/len(x_clean) + (x_range - mean_x)
                                        ** 2 / np.sum((x_clean - mean_x)**2))
                            y_upper = y_pred + t_val * se_line
                            y_lower = y_pred - t_val * se_line

                            # Add regression line
                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=y_pred,
                                    mode='lines',
                                    line=dict(color='red', width=2),
                                    showlegend=False
                                ),
                                row=i+1, col=j+1
                            )

                            # Add confidence interval area
                            fig.add_trace(
                                go.Scatter(
                                    x=np.concatenate([x_range, x_range[::-1]]),
                                    y=np.concatenate([y_upper, y_lower[::-1]]),
                                    fill='toself',
                                    fillcolor='rgba(255, 0, 0, 0.2)',
                                    line=dict(color='rgba(255, 0, 0, 0)'),
                                    showlegend=False
                                ),
                                row=i+1, col=j+1
                            )

        # Update layout and axis properties
        fig.update_layout(
            margin=dict(l=10, r=10, t=50, b=50),
            title="Pair Plot",
            height=600,
            width=600,
            showlegend=False,
            plot_bgcolor="white"
        )

        # Update all axes properties first (hiding tick labels by default)
        fig.update_xaxes(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showticklabels=False
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showticklabels=False
        )

        # Show tick labels and titles only for bottom row and leftmost column
        for i, col_name in enumerate(columns):
            # Bottom row: show x-axis titles and tick labels
            fig.update_xaxes(
                title_text=col_name,
                showticklabels=True,
                row=n_cols,
                col=i+1
            )

            # Leftmost column: show y-axis titles and tick labels
            fig.update_yaxes(
                title_text=col_name,
                showticklabels=True,
                row=i+1,
                col=1
            )

        return fig

    def plot_corr(self):
        """
        Plot a correlation matrix for the data.

        Returns
        -------
        fig : seaborn.axisgrid.PairGrid
            The pairplot figure.
        """
        corr_matrix = self._data.corr(method='pearson')
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        # Apply mask - replace upper triangle with NaN values
        corr_matrix_lower = corr_matrix.copy()
        corr_matrix_lower.values[mask] = np.nan

        fig = px.imshow(
            corr_matrix_lower,
            text_auto='.2f',  # Display correlation values
            # Red to Blue color scale (reversed)
            color_continuous_scale='RdBu_r',
            zmin=-1,  # Minimum correlation value
            zmax=1,   # Maximum correlation value
            aspect="auto",  # Keep aspect ratio adaptive
            title="Pearson Correlation Heatmap"
        )
        # Customize hover template
        fig.update_traces(
            hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z}<extra></extra>'
        )
        # Improve layout
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "0.5", "1"],
            ),
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=50, b=50),
            xaxis=dict(
                showgrid=False,  # Enable grid
                gridcolor="lightgray",  # Light gray grid lines
                zeroline=False,
                zerolinecolor="black",  # Black zero line
                showline=False,
                linewidth=1,
                tickangle=-45,
                linecolor="black",  # Black border
                mirror=True
            ),
            yaxis=dict(
                showgrid=False,  # Enable grid
                gridcolor="lightgray",  # Light gray grid lines
                zeroline=False,
                zerolinecolor="black",  # Black zero line
                showline=False,
                linewidth=1,
                linecolor="black",  # Black border
                mirror=True
            ),
            height=600,
            width=600
        )

        return fig

    def write_equation(self, order=1, quadratic=[]):
        """
        Write R-style equation for multivariate fitting procedure using the statsmodels package.

        Parameters
        ----------
        order : int, optional
            The order of the polynomial. Default is ``1``.
        quadratic : list, optional
            The list of quadratic factors. Default is an empty list.

        Returns
        -------
        str
            The R-style equation.
        """
        myfactors = self._factors.copy()
        if self._dtypes is not None:
            if not isinstance(myfactors, list):
                myfactors = myfactors.tolist()  # Convert to list to allow mutable operations
            for i in range(len(self._factors)):
                if self._dtypes[self._factors[i]] == 'object':
                    myfactors[i] = f'C({myfactors[i]})'
        eqn = f'{self._response} ~ {myfactors[0]} '
        for factor in myfactors[1:]:
            eqn += f'+ {factor} '
        if order > 1:
            for i in range(len(myfactors)):
                for j in range(i + 1, len(myfactors)):
                    eqn += f'+ {myfactors[i]}:{myfactors[j]} '
        if order > 2:
            for i in range(len(myfactors)):
                for j in range(i + 1, len(myfactors)):
                    for k in range(j + 1, len(myfactors)):
                        eqn += f'+ {myfactors[i]}:{myfactors[j]}:{myfactors[k]} '
        if order > 3:
            for i in range(len(myfactors)):
                for j in range(i + 1, len(myfactors)):
                    for k in range(j + 1, len(myfactors)):
                        for l in range(k + 1, len(myfactors)):
                            eqn += f'+ {myfactors[i]}:{myfactors[j]}:{myfactors[k]}:{myfactors[l]} '
        if len(quadratic) > 0:
            for factor in quadratic:
                eqn += f'+ I({factor}**2) '
        self._equation = eqn
        return self._equation

    def compute_linear_model(self, order=1, quadratic=[]):
        """
        Compute the linear model using the statsmodels package.

        Parameters
        ----------
        order : int, optional
            The order of the polynomial. Default is ``1``. The parameter 
            is not used if the equation is already set.
        quadratic : list, optional
            The list of quadratic factors. Default is an empty list.
            The parameter is not used if the equation is already set.

        Returns
        -------
        statsmodels.regression.linear_model.RegressionResultsWrapper
            The fitted linear model.
        """
        if self._equation == '':
            eqn = self.write_equation(order=order, quadratic=quadratic)
        else:
            eqn = self._equation
        model = smf.ols(formula=eqn, data=self._data)
        self._linear_model = model.fit()
        return self._linear_model

    def plot_linear_model(self):
        """
        Plot the linear model using plotly.

        Returns
        -------
        fig : list
            The list of plotly figures.
        """
        if self._linear_model is None:
            raise Warning("Linear model has not been computed yet.")
        fig = [None] * 2
        fig[0] = go.Figure()
        # Fig[0]: actual vs predicted line
        fig[0].add_trace(go.Scatter(x=self._data[self._response],
                                    y=self._linear_model.predict(),
                                    mode='markers',
                                    marker=dict(size=12),
                                    name='Actual vs Predicted'))
        # Add 1:1 line
        fig[0].add_shape(type="line",
                         x0=min(self._data[self._response]),
                         y0=min(self._data[self._response]),
                         x1=max(self._data[self._response]),
                         y1=max(self._data[self._response]),
                         line=dict(color="Gray", width=1, dash="dash"))
        fig[0].update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis_title=f'Actual {self._response}',
            yaxis_title=f'Predicted {self._response}',
            height=500,  # Adjust height as needed
            margin=dict(l=10, r=10, t=50, b=50),
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
            )
        )
        # # # # # # # # # # # # # # # # # # # # #
        # Fig[1]: slope values for each factor
        # # # # # # # # # # # # # # # # # # # # #
        # Plot: Slope values for each factor
        res = self._linear_model.params.rename_axis('terms').reset_index()[1:]
        res.columns = ['terms', 'slope']
        error = self._linear_model.bse.rename_axis('terms').reset_index()[1:]
        error.columns = ['terms', 'error']
        res['error'] = error['error']
        res['pvalue'] = [self._linear_model.pvalues[res['terms']].iloc[i]
                         for i in range(len(res))]

        # Sort by p-values
        res = res.sort_values(by='pvalue', ascending=False)
        # Prepare colors and labels
        colors = ['red' if x < 0 else 'green' for x in res['slope']]
        res['slope'] = res['slope'].abs()

        fig[1] = go.Figure()

        # Add bar plot
        fig[1].add_trace(go.Bar(
            y=[term.replace('I(', '').replace('C(', '').replace(')', '').replace(
                ' ** ', '^') for term in res['terms']],
            x=res['slope'],
            error_x=dict(type='data', array=res['error']),
            marker_color=colors,
            orientation='h',
            name='Slope',
            showlegend=False  # Hide the legend entry for the bar trace
        ))

        # Update layout for log scale and labels
        fig[1].update_layout(
            xaxis_title='Magnitude of effect',
            xaxis_type="log",
            height=500,  # Adjust height as needed
            plot_bgcolor="white",  # White background
            legend=dict(
                orientation="h",  # Horizontal orientation
                y=1.1  # Place legend on top
            ),
            # Increase right margin for annotations
            margin=dict(l=10, r=150, t=50, b=50),
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
            )
        )

        # Add legend for Negative and Positive
        fig[1].add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='red'),
            name='Negative'
        ))
        fig[1].add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color='green'),
            name='Positive'
        ))

        # Add p-values as annotations outside the plot
        for i, p in enumerate(res['slope']):
            fig[1].add_annotation(
                x=1.025,  # Place annotations outside the plot
                y=i,
                text=f"p = {self._linear_model.pvalues[res['terms'].iloc[i]]:.2g}",
                showarrow=False,
                xref="paper",  # Use paper coordinates for x
                xanchor='left'
            )

        return fig

    def compute_ML_model(self, **kwargs):
        """
        Compute the machine learning model.

        Parameters
        ----------
        kwargs : dict
            Additional keyword arguments for the model. Overrides default
            parameters. See the Notes section for the defaults used by
            each model type.

        Returns
        -------
        object
            The fitted machine learning model.

        Notes
        -----
        Default parameters by model:

        ElasticNetCV
            ``l1_ratio`` : list, default=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
                List of L1 ratios to try.
            ``cv`` : int, default=5
                Cross-validation folds.
            ``max_iter`` : int, default=1000
                Maximum iterations.

        RidgeCV
            ``alphas`` : list, default=[0.1, 1.0, 10.0]
                List of alpha values to try.
            ``cv`` : int, default=5
                Cross-validation folds.

        LinearRegression
            ``fit_intercept`` : bool, default=True
                Whether to calculate the intercept.

        RandomForest
            ``n_estimators`` : int, default=100
                Number of trees in the forest.
            ``max_depth`` : int or None, default=None
                Maximum depth of trees.
            ``min_samples_split`` : int, default=2
                Minimum samples required to split a node.
            ``random_state`` : int, default=42
                Random seed for reproducibility.

        GaussianProcess
            ``kernel`` : kernel object, default=None
                Kernel for the Gaussian Process.
            ``alpha`` : float, default=1e-10
                Value added to diagonal of kernel matrix.
            ``normalize_y`` : bool, default=True
                Normalize target values.
            ``random_state`` : int, default=42
                Random seed for reproducibility.

        GradientBoosting
            ``n_estimators`` : int, default=100
                Number of boosting stages.
            ``learning_rate`` : float, default=0.1
                Learning rate.
            ``max_depth`` : int, default=3
                Maximum depth of trees.
            ``random_state`` : int, default=42
                Random seed for reproducibility.
        """
        if self._model_type is None:
            raise ValueError("Model must be provided.")

        X = self._data[self._factors]
        y = self._data[self._response]
        if self._split_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._split_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Default parameters for each model type
        default_params = {
            "ElasticNetCV": {"l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                             "cv": 5,
                             "max_iter": 1000},
            "RidgeCV": {"alphas": [0.1, 1.0, 10.0],
                        "cv": 5},
            "LinearRegression": {"fit_intercept": True},
            "RandomForest": {"n_estimators": 100,
                             "max_depth": None,
                             "min_samples_split": 2,
                             "random_state": 42},
            "GaussianProcess": {"kernel": None,
                                "alpha": 1e-10,
                                "normalize_y": True,
                                "random_state": 42},
            "GradientBoosting": {"n_estimators": 100,
                                 "learning_rate": 0.1,
                                 "max_depth": 3,
                                 "random_state": 42}
        }

        # Get default parameters for the selected model
        model_defaults = default_params.get(self._model_type, {})

        # Override defaults with any provided kwargs
        model_params = {**model_defaults, **kwargs}

        if self._model_type == "ElasticNetCV":
            self._model = make_pipeline(StandardScaler(),
                                        ElasticNetCV(**model_params))
        elif self._model_type == "RidgeCV":
            self._model = make_pipeline(StandardScaler(),
                                        RidgeCV(**model_params))
        elif self._model_type == "LinearRegression":
            self._model = make_pipeline(StandardScaler(),
                                        LinearRegression(**model_params))
        elif self._model_type == "RandomForest":
            self._model = make_pipeline(StandardScaler(),
                                        RandomForestRegressor(**model_params))
        elif self._model_type == "GaussianProcess":
            self._model = make_pipeline(StandardScaler(),
                                        GaussianProcessRegressor(**model_params))
        elif self._model_type == "GradientBoosting":
            self._model = make_pipeline(StandardScaler(),
                                        GradientBoostingRegressor(**model_params))

        # Fit the model
        self._model.fit(X_train, y_train)
        return self._model

    def plot_ML_model(self, features_in_log=False):
        """
        Plot the machine learning model using plotly.

        Parameters
        ----------
        features_in_log : bool, optional
            Whether to plot the feature importances in log scale. Default is False.

        Returns
        -------
        fig : list
            The list of plotly figures.
        """
        coef_names = self._data[self._factors].columns
        X = self._data[self._factors]
        y = self._data[self._response]
        if self._split_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self._split_size)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y
        bootstrap_coefs = bootstrap_coefficients(
            self._model[1], X, y, n_bootstrap=100, random_state=42)
        # Calculate mean and standard deviation of coefficients
        mean_coefs = np.mean(bootstrap_coefs, axis=0)
        std_coefs = np.std(bootstrap_coefs, axis=0)
        # make the fig
        fig = [None] * 2
        fig[0] = go.Figure()
        # Add actual vs predicted line
        fig[0].add_trace(go.Scatter(x=y_train,
                                    y=self._model.predict(X_train),
                                    mode='markers',
                                    marker=dict(size=12, color='royalblue'),
                                    name='Training'))
        if self._split_size > 0:
            fig[0].add_trace(go.Scatter(x=y_test,
                                        y=self._model.predict(X_test),
                                        mode='markers',
                                        marker=dict(size=12, color='orange'),
                                        name='Validation'))
        # Add 1:1 line
        fig[0].add_shape(type="line",
                         x0=min(y_train), y0=min(y_train),
                         x1=max(y_train), y1=max(y_train),
                         line=dict(color="Gray", width=1, dash="dash"))
        fig[0].update_layout(
            plot_bgcolor="white",  # White background
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis_title=f'Actual {self._response}',
            yaxis_title=f'Predicted {self._response}',
            height=500,  # Adjust height as needed
            margin=dict(l=10, r=10, t=120, b=50),
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
            )
        )
        # add the score in title
        fig[0].update_layout(
            title={
                'text': f"Predicted vs Actual for {str(self._model[1])}<br>R²_test = {r2_score(y_test, self._model.predict(X_test)):.4f}  -  R²_train = {r2_score(y_train, self._model.predict(X_train)):.4f}<br>RMSE_test = {root_mean_squared_error(y_test, self._model.predict(X_test)):.4f}  -  RMSE_train = {root_mean_squared_error(y_train, self._model.predict(X_train)):.4f}",
                'x': 0.45,
                'xanchor': 'center'
            }
        )
        # make feature importance plot
        if self._model_type != "GaussianProcess":
            fig[1] = go.Figure()
            pos_coefs = mean_coefs[mean_coefs > 0]
            pos_coefs_names = coef_names[mean_coefs > 0]
            neg_coefs = mean_coefs[mean_coefs < 0]
            neg_coefs_names = coef_names[mean_coefs < 0]
            # Add bars for positive mean coefficients
            fig[1].add_trace(go.Bar(
                y=[f"{pos_coefs_names[i]}" for i in range(len(pos_coefs))],
                x=mean_coefs[mean_coefs > 0],
                error_x=dict(
                    type='data', array=std_coefs[mean_coefs > 0], visible=True),
                orientation='h',
                marker_color='royalblue',
                name='Positive'
            ))
            fig[1].add_trace(go.Bar(
                y=[f"{neg_coefs_names[i]}" for i in range(len(neg_coefs))],
                x=-mean_coefs[mean_coefs < 0],
                error_x=dict(
                    type='data', array=std_coefs[mean_coefs < 0], visible=True),
                orientation='h',
                marker_color='orange',
                name='Negative'
            ))
            # Update layout
            fig[1].update_layout(
                title={
                    'text': f"Features importance for {str(self._model[1])}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Coefficient Value",
                yaxis_title="Features",
                barmode='relative',
                margin=dict(l=150)
            )
            if features_in_log:
                fig[1].update_xaxes(type="log")
            else:
                fig[1].update_xaxes(type="linear")
            fig[1].update_layout(
                plot_bgcolor="white",  # White background
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                height=500,  # Adjust height as needed
                margin=dict(l=10, r=10, t=120, b=50),
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
                )
            )
        else:
            print('GaussianProcess does not have coefficients or feature importances')
        return fig

    def predict(self, X=None, model='all'):
        """
        Predict using the machine learning model and the linear model,
        if they are trained. Use the encoders to encode the data.
        If X is not provided, use the original data.
        If the model has not been trained, raise a warning.

        Parameters
        ----------
        X : pd.DataFrame, optional
            The input features. Default is None, which uses the original data.

        Returns
        -------
        pd.DataFrame
            The predicted values with a column indicating the model used.
        """
        if X is None:
            X = self._data[self._factors].copy()
        else:
            X = X.copy()

        # Encode the input data using the same encoders
        for factor in self._factors:
            if factor in self._encoders:
                X[factor] = self._encoders[factor].transform(
                    X[factor].astype(str))

        predML = None
        predlin = None

        if self._model is not None:
            # Check if the model is fitted
            if not hasattr(self._model, 'predict'):
                raise Warning("Model has not been trained yet.")
            predML = self._model.predict(X)
            predML = pd.DataFrame(predML, columns=['prediction'])
            predML['model'] = self._model_type

        if self._linear_model is not None:
            # Check if the linear model is fitted
            if not hasattr(self._linear_model, 'predict'):
                raise Warning("Linear model has not been trained yet.")
            predlin = self._linear_model.predict(X)
            predlin = pd.DataFrame(predlin, columns=['prediction'])
            predlin['model'] = 'Linear Model'

        if predML is not None and predlin is not None and model == 'all':
            # Combine the predictions on top of each other
            pred = pd.concat([predML, predlin], axis=0)
            pred = pred.reset_index(drop=True)
            return pred
        elif predML is not None and model == 'ML':
            return predML
        elif predlin is not None and model == 'linear':
            return predlin
        else:
            raise Warning("No model has been trained yet.")


def bootstrap_coefficients(mod, X, y, n_bootstrap=100, random_state=None):
    """
    Perform bootstrap resampling to estimate the variability of model coefficients.

    Parameters
    ----------
    mod : object
        The machine learning model.
    X : pd.DataFrame
        The input features.
    y : pd.Series
        The target variable.
    n_bootstrap : int, optional
        The number of bootstrap samples. Default is ``100``.
    random_state : int, optional
        The seed for the random number generator.

    Returns
    -------
    results : np.ndarray
        The bootstrapped coefficients.
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    results = []

    for _ in range(n_bootstrap):
        # Resample the data
        indices = np.random.choice(
            np.arange(n_samples), size=n_samples, replace=True)
        X_resample, y_resample = X.values[indices], y.values[indices]

        # Fit the model
        if isinstance(mod, RidgeCV):
            model = RidgeCV(alphas=np.logspace(-3, 3, 10)
                            ).fit(X_resample, y_resample)
            results.append(model.coef_)
        elif isinstance(mod, ElasticNetCV):
            model = ElasticNetCV(alphas=np.logspace(-3, 3, 10),
                                 l1_ratio=0.5).fit(X_resample, y_resample)
            results.append(model.coef_)
        elif isinstance(mod, LinearRegression):
            model = LinearRegression().fit(X_resample, y_resample)
            results.append(model.coef_)
        elif isinstance(mod, RandomForestRegressor):
            model = RandomForestRegressor().fit(X_resample, y_resample)
            results.append(model.feature_importances_)
        elif isinstance(mod, GradientBoostingRegressor):
            model = GradientBoostingRegressor().fit(X_resample, y_resample)
            results.append(model.feature_importances_)
        elif isinstance(mod, GaussianProcessRegressor):
            # Gaussian Process does not have coefficients or feature importances
            model = GaussianProcessRegressor().fit(X_resample, y_resample)
            # Placeholder for Gaussian Process
            results.append(np.zeros(X.shape[1]))
        else:
            raise ValueError(f"Unsupported model type: {mod}")

    return np.array(results)
