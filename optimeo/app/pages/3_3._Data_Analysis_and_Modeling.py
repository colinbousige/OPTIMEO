# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version. 

import streamlit as st
import numpy as np
from ast import literal_eval
from resources.functions import about_items, check_data, load_data_widget, read_markdown_file, write_poly
from janitor import clean_names
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from optimeo.analysis import DataAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import t

st.set_page_config(page_title="Data Analysis and Modeling", 
                   page_icon="resources/icon.png", 
                   layout="wide", menu_items=about_items)

style = read_markdown_file("resources/style.css")
st.markdown(style, unsafe_allow_html=True)

if not 'analysis' in st.session_state:
    st.session_state.analysis = None
if not 'figml' in st.session_state:
    st.session_state.figml = None
if not 'modml' in st.session_state:
    st.session_state.modml = None
if not 'modlin' in st.session_state:
    st.session_state.modlin = None
if not 'figlin' in st.session_state:
    st.session_state.figlin = None
if "model_up_to_date" not in st.session_state:
    st.session_state['model_up_to_date'] = False
    
def data_changed():
    st.session_state.analysis = None
    st.session_state.figml = None
    st.session_state.modml = None
    st.session_state.modlin = None
    st.session_state.figlin = None

def recompute_ML_fig():
    st.session_state.figml = None

def model_updated():
    st.session_state.model_up_to_date = True

def model_changed():
    st.session_state.model_up_to_date = False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# Data Analysis and Modeling
""")

tabs = st.tabs(["Data Loading", "Visual Assessment", "Linear Regression Model", 'Machine Learning Model'])

with tabs[0]: # data loading
    data = load_data_widget()
    if data is None:
        cont = st.container(border=True)
        cont.markdown(
        """##### How to format your data?
        
The data must be in tidy format, meaning that each column is a variable and each row is an observation. We usually place the factors in the first columns and the response(s) in the last column(s). Data type can be float, integer, or text, and you can only specify one response. Spaces and special characters in the column names will be automatically removed. The first row of the file will be used as the header.

For Excel-like files, the first sheet will be used, and data should start in the A1 cell, and no unnecessary rows or columns should be present. 

"""
        )
        conti = cont.columns([1,2,1])
        conti[1].image("resources/tidy_data.jpg", caption="Example of tidy data format.")
    if data is not None:
        left,right=st.columns([1,1])
        data = clean_names(data, remove_special=True, case_type='preserve')
        cols = data.columns.to_numpy()
        st.dataframe(data, hide_index=False)
        mincol = 1 if 'run_order' in cols else 0
        factors = left.multiselect("Select the **parameter(s)** columns:", 
                data.columns, default=cols[mincol:-1], on_change=data_changed)
        # response cannot be a factor, so default are all unselected columns in factor
        available = [col for col in cols if col not in factors]
        response = [right.selectbox("Select the **outcome** column:", 
                available, index=len(available)-1, 
                on_change=data_changed,
                help="""The outcome is the response variable that you want to model. 

**Only one outcome is supported for now**"""),
                    ]
        if len(response) > 0:
            response = response[0]
        placeholder = st.empty()
        dtypes = data.dtypes
        # check if factors are valid
        messages = check_data(data, factors)
        if len(messages) > 0:
            key, value = list(messages.items())[0]
            messages[key] = '⚠️   '+messages[key]
            message = '''

⚠️   '''.join(messages.values())
            placeholder.error(message)
            for name,messsage in messages.items():
                # drop factors[name]
                factors.remove(name)
        dataclean = data[factors+[response]].copy()
        dataclean = dataclean.dropna(axis=0, how='any')
        st.session_state.analysis = DataAnalysis(dataclean, factors, response)
        encoders = st.session_state.analysis.encoders


with tabs[1]: # visual assessment
    if data is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the **Sidebar** and select the features and response in the **Data Loading** tab.""")
    if data is not None and len(factors) > 0 and len(response) > 0:

        # Set up the layout for Streamlit
        ncols = min(len(factors), 4)
        cols = st.columns(4)

        # Q-Q Plot
        fig = st.session_state.analysis.plot_qq()
        cols[0].plotly_chart(fig, key="qq_plot")

        # Box Plot
        fig = st.session_state.analysis.plot_boxplot()
        cols[1].plotly_chart(fig, key="box_plot")

        # Histogram
        fig = st.session_state.analysis.plot_histogram()
        cols[2].plotly_chart(fig, key="histogram")

        # Scatter Plot of Response Values
        fig = st.session_state.analysis.plot_scatter_response()
        cols[3].plotly_chart(fig, key="scatter_response")
        
        st.write("---")
        cols = st.columns(2)
        fig = st.session_state.analysis.plot_corr()
        cols[0].plotly_chart(fig, key="correlation")
        
        fig = st.session_state.analysis.plot_pairplot_plotly()
        cols[1].plotly_chart(fig, key="pairplot")

        # Scatter Plots for Each Factor
        st.write("---\n##### Polynomial fits")        
        cols = st.columns(ncols)
        for i, factor in enumerate(factors):
            fig = px.scatter(x=dataclean[factor], 
                             y=dataclean[response], 
                             labels={'x': factor, 'y': response}, 
                             title=f'{response} vs {factor}')
            fig.update_layout(
                plot_bgcolor="white",  # White background
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=10, r=10, t=100, b=10),
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
            fitorder = cols[i % ncols].number_input(f"Order for {response} vs {factor}:",
                                        min_value=0, value=2, max_value=10, step=1)
            # Add linear regression with red line and equation
            if dtypes[factor] != 'object':
                x_data = dataclean[factor]
                y_data = dataclean[response]
                valid_indices = x_data.notna() & y_data.notna()
                x_clean = x_data[valid_indices]
                y_clean = y_data[valid_indices]
                p = np.polyfit(dataclean[factor], dataclean[response], fitorder)
                x_range = np.linspace(np.min(dataclean[factor]), np.max(dataclean[factor]), 100)
                y_pred = np.polyval(p, x_range)
                # Standard error of estimate
                y_fit = np.polyval(p, x_clean)
                residuals = y_clean - y_fit
                dof = len(x_clean) - 2
                residual_std_error = np.sqrt(np.sum(residuals**2) / dof)

                mean_x = np.mean(x_clean)
                t_val = t.ppf(0.975, dof)  # 95% confidence

                se_line = residual_std_error * np.sqrt(1/len(x_clean) + (x_range - mean_x)**2 / np.sum((x_clean - mean_x)**2))
                y_upper = y_pred + t_val * se_line
                y_lower = y_pred - t_val * se_line
                
                fig.add_trace(go.Scatter(x=x_range, y=y_pred, 
                                         mode='lines', 
                                         name='Polynomial Fit', 
                                         line=dict(color='red')))
                fig.add_trace(go.Scatter(
                                    x=np.concatenate([x_range, x_range[::-1]]),
                                    y=np.concatenate([y_upper, y_lower[::-1]]),
                                    fill='toself',
                                    fillcolor='rgba(255, 0, 0, 0.2)',
                                    line=dict(color='rgba(255, 0, 0, 0)'),
                                    showlegend=False
                                    )
                              )
                fig.update_layout(title_subtitle_text=rf'{write_poly(p)}',
                                  title_subtitle_font_size=16,
                                  showlegend=False, height=400)

            cols[i % ncols].plotly_chart(fig)


with tabs[2]: # simple model
    if data is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the **Sidebar** and select the features and response in the **Data Loading** tab.""")
    if data is not None and len(factors) > 0 and len(response) > 0:
        cols = st.columns([1,1,4])
        order = cols[0].number_input("Interactions order:", 
                                        min_value=1, value=1, 
                                        max_value=min([4,len(factors)]))
        quadratic = cols[1].multiselect("Quadratic terms?", factors)
        def_eqn = st.session_state.analysis.write_equation(order, quadratic)
        st.session_state.analysis.equation = cols[2].text_input("Model equation:", key="eqn", value=def_eqn, help="""Interactions are written as `factor1:factor2`.  
Powered terms are written as `I(factor**power)`.  
Categorical variables are written as `C(factor)`.  
You can also use `np.log(factor)` or `np.exp(factor)` for transformations.  
To remove the intercept, add `-1` at the end of the equation.""")
        if st.button("Compute the linear model"):
            st.session_state.analysis.compute_linear_model()
            st.session_state.modlin = st.session_state.analysis.linear_model
            st.session_state.figlin = st.session_state.analysis.plot_linear_model()
            # Plot: Actual vs Predicted
        if st.session_state.modlin is not None:
            st.write(st.session_state.modlin.summary())
            cols = st.columns([1, 1])
            cols[0].plotly_chart(st.session_state.figlin[0])
            cols[1].plotly_chart(st.session_state.figlin[1])
            # # # # # # # # # # # # # # # 
            st.write("##### Predict the response for a set of factors with this linear model:")
            Xnew = {factor: 0 for factor in factors}
            left, right = st.columns(2)
            for i,factor in enumerate(factors):
                colsinput = left.columns(2)
                colsinput[0].write(f"<p style='text-align:right;font-size:1.1em'><b>{factor}</b></p>", unsafe_allow_html=True)
                if dtypes[factor] == 'object':
                    # non encoded factor
                    possible = np.unique(encoders[factor].inverse_transform(dataclean[factor].values))
                    Xnew[factor] = str(colsinput[1].selectbox(f"{factor}", possible, key=f"{factor}lm", label_visibility='collapsed'))
                else:
                    Xnew[factor] = colsinput[1].number_input(f"{factor}", 
                                                value=np.mean(dataclean[factor]), key=f"{factor}lm", label_visibility='collapsed')
            # encode the factors if they are categorical
            for i,factor in enumerate(factors):
                if dtypes[factor] == 'object':
                    toencode = Xnew[factor]
                    Xnew[factor] = encoders[factor].transform([toencode])[0]
            # Get prediction and standard error
            # Convert Xnew to a DataFrame for prediction
            Xnew_df = pd.DataFrame([Xnew])
            prediction = st.session_state.modlin.get_prediction(Xnew_df)
            predicted_value = prediction.predicted_mean[0]
            stderr = prediction.se_mean[0]  # Standard error of the prediction

            # Display the predicted value and standard error
            right.container(border=True).write(
                f"<p style='text-align:center;font-size:1.5em'>"
                f"Predicted {response}:<br><br>"
                f"{predicted_value:.4g} ± {stderr:.4g}"
                "</p>",
                unsafe_allow_html=True
            )
        st.write("")
        st.write("")
        st.write("")
        st.write("")

with tabs[3]: # machine learning model
    if data is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the **Sidebar** and select the features and response in the **Data Loading** tab.""")
    if data is not None and len(factors) > 0 and len(response) > 0:
        # Choose machine learning model
        cols = st.columns([2,3])
        with cols[0].expander("**How to choose the ML model?**"):
            st.write("""In the **Visual Assessment** tab, you can take a look at your data and make a pairplot out of it. This can help you in model selection:

- **Linear Relationships:** If the pairplot shows linear relationships between features and the target variable, linear models like **LinearRegression**, **RidgeCV**, or **ElasticNetCV** might be appropriate.

- **Non-linear Relationships:** If you observe non-linear patterns, models like **RandomForest**, **GradientBoosting**, or **GaussianProcess** could be more suitable, as they can capture complex relationships.

- **Feature Correlations:** A pairplot can also reveal correlations between features. High correlations might suggest the need for regularization techniques (e.g., **RidgeCV** or **ElasticNetCV**) to handle multicollinearity.

- **Outliers and Distribution:** Visualizing the data can help identify outliers or unusual distributions that might affect model performance. This insight can guide preprocessing steps or model choice.
""")
        recols = cols[0].columns(2)
        with recols[0].expander("**ElasticNetCV**"):
            st.write('''This model combines the properties of both Lasso and Ridge regression. It is useful when you have many correlated features and want to perform feature selection. It automatically tunes the regularization parameters using cross-validation.
            ''')
        with recols[0].expander("**RidgeCV**"):
            st.write('''This is a linear regression model with L2 regularization. It is effective when dealing with multicollinearity (highly correlated features). RidgeCV automatically tunes the regularization strength using cross-validation, making it a good choice for improving model generalization.
            ''')
        with recols[0].expander("**LinearRegression**"):
            st.write('''This is a simple and interpretable model that fits a linear relationship between the input features and the target variable. It is suitable for datasets where the relationship between features and the target is approximately linear.
            ''')
        with recols[1].expander("**RandomForest**"):
            st.write('''This is an ensemble learning method that combines multiple decision trees to improve predictive performance. It is robust to overfitting and can handle both classification and regression tasks. RandomForest is a good choice when you have a large dataset with complex interactions.
            ''')
        with recols[1].expander("**GaussianProcess**"):
            st.write('''This model is useful for small to medium-sized datasets and can capture complex relationships. It provides uncertainty estimates along with predictions, making it suitable for tasks where understanding prediction confidence is important.
            ''')
        with recols[1].expander("**GradientBoosting**"):
            st.write('''This is an ensemble technique that builds models sequentially, each correcting the errors of the previous one. It is powerful for both regression and classification tasks and often achieves high performance, but it can be computationally intensive.
            ''')
        cols[1].write("###### Machine Learning Model")
        colss = cols[1].columns(2)
        model_sel = colss[0].selectbox("Select the machine learning model:", 
                ["ElasticNetCV", "RidgeCV", "LinearRegression", 
                 "RandomForest", "GaussianProcess", "GradientBoosting"],
                on_change=model_changed)
        split_size = colss[0].number_input("Validation set size:", 
                                             min_value=0.0, value=0.2,
                                             max_value=1., step=.1, 
                                             on_change=model_changed)
        features_in_log = colss[0].toggle("Log scale for features importance", 
                                          value=True, 
                                          on_change=model_changed)
        kwargs = colss[1].text_area('Additional parameters for the model (optional):', 
            value='{}', height=68,
            on_change=model_changed, 
            help="""You can add additional parameters for the model in the form of a dictionary. 
Default parameters will be used if you do not specify them, they are:
- **[ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)**: 
    - `{"l1_ratio": [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0], "cv": 5, "max_iter": 1000}`
- **[RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html)**: 
    - `{"alphas": [0.1, 1.0, 10.0], "cv": 5}`
- **[LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)**: 
    - `{"fit_intercept": True}`
- **[RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)**: 
    - `{"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "random_state": 42}`
- **[GaussianProcess](https://scikit-learn.org/stable/modules/gaussian_process.html)**: 
    - `{"kernel": None, "alpha": 1e-10, "normalize_y": True, "random_state": 42}`
- **[GradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)**: 
    - `{"n_estimators": 100,"learning_rate": 0.1,"max_depth": 3, "random_state": 42}`""")
        if kwargs != "":
            try:
                kwargs = literal_eval(kwargs)
            except Exception as e:
                st.error(f"Error in evaluating the additional parameters: {e}")
                kwargs = {}
        if kwargs != {} and not isinstance(kwargs, dict):
            st.error("The additional parameters must be in the form of a dictionary.")
            kwargs = {}
        # colss[1].write("")
        if colss[1].button("Compute the machine learning model and plot the results",
                          disabled=st.session_state['model_up_to_date'],
                          on_click=model_updated,
                          type="primary"):
            st.session_state.analysis.model_type = model_sel
            st.session_state.analysis.split_size = split_size
            st.session_state.analysis.compute_ML_model(**kwargs)
            st.session_state.modml = st.session_state.analysis.model
            st.session_state.figml = st.session_state.analysis.plot_ML_model(features_in_log)
        
        if st.session_state.figml is not None:
            # Make plot of predicted vs actual
            cols = st.columns([1, 1])
            cols[0].plotly_chart(st.session_state.figml[0])
            if st.session_state.figml[1] is not None:
                cols[1].plotly_chart(st.session_state.figml[1])
            else:
                cols[1].warning(f"The {model_sel} model does not support feature importance.")

            st.write(f"##### Predict the response for a set of factors with this {model_sel} model:")

            Xnew = []
            left, right = st.columns(2)
            for i, factor in enumerate(factors):
                colsinput = left.columns(2)
                colsinput[0].write(f"<p style='text-align:right;font-size:1.1em'><b>{factor}</b></p>", unsafe_allow_html=True)
                if dtypes[factor] == 'object':
                    # Non-encoded factor
                    possible = np.unique(encoders[factor].inverse_transform(dataclean[factor].values))
                    Xnew.append(str(colsinput[1].selectbox(f"{factor}", possible, key=f"{factor}ml", label_visibility='collapsed')))
                else:
                    Xnew.append(colsinput[1].number_input(f"{factor}",
                                                        value=np.mean(dataclean[factor]), key=f"{factor}ml", label_visibility='collapsed'))

            # Encode the factors if they are categorical
            for i, factor in enumerate(factors):
                if dtypes[factor] == 'object':
                    toencode = Xnew[i]
                    Xnew[i] = encoders[factor].transform([toencode])[0]

            # Convert Xnew to a numpy array and reshape
            Xnew = np.array(Xnew).reshape(1, -1)

            # Make prediction
            prediction = st.session_state.modml.predict(Xnew)[0]

            # Calculate standard error
            stderr = None
            if model_sel in ["ElasticNetCV", "RidgeCV", "LinearRegression"]:
                # For linear models, use the covariance matrix
                X_train = st.session_state.analysis.data[st.session_state.analysis.factors]
                y_train = st.session_state.analysis.data[st.session_state.analysis.response]

                # Add intercept if needed
                if hasattr(st.session_state.modml, 'fit_intercept') and st.session_state.modml.fit_intercept:
                    X_train_with_intercept = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
                else:
                    X_train_with_intercept = X_train

                # Get the covariance matrix of the coefficients
                if hasattr(st.session_state.modml, 'cov_params'):
                    cov_matrix = st.session_state.modml.cov_params()
                else:
                    # For RidgeCV and ElasticNetCV, we need to estimate the covariance matrix
                    # This is a simplified approach; for a more accurate estimate, consider using the MSE and the regularization
                    residuals = y_train - st.session_state.modml.predict(X_train)
                    mse = np.mean(residuals**2)
                    cov_matrix = mse * np.linalg.inv(X_train_with_intercept.T @ X_train_with_intercept)

                # Add intercept to Xnew if needed
                if hasattr(st.session_state.modml, 'fit_intercept') and st.session_state.modml.fit_intercept:
                    Xnew_with_intercept = np.hstack([np.ones((Xnew.shape[0], 1)), Xnew])
                else:
                    Xnew_with_intercept = Xnew

                # Calculate the variance of the prediction
                var_pred = Xnew_with_intercept @ cov_matrix @ Xnew_with_intercept.T
                stderr = np.sqrt(var_pred)[0, 0]

            elif model_sel == "GaussianProcess":
                _, stderr = st.session_state.modml.predict(Xnew, return_std=True)

            else:
                # For non-linear models, use bootstrapping
                n_bootstraps = 100
                bootstrap_predictions = []
                X = st.session_state.analysis.data[st.session_state.analysis.factors]
                y = st.session_state.analysis.data[st.session_state.analysis.response]
                if split_size > 0:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=split_size, random_state=42)
                else:
                    X_train, X_test, y_train, y_test = X, X, y, y

                for _ in range(n_bootstraps):
                    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
                    # Use .iloc for integer-location based indexing
                    X_resampled = X_train.iloc[indices]
                    y_resampled = y_train.iloc[indices]
                    
                    if model_sel == "RandomForest":
                        model = RandomForestRegressor(**kwargs)
                    elif model_sel == "GradientBoosting":
                        model = GradientBoostingRegressor(**kwargs)
                    
                    model.fit(X_resampled, y_resampled)
                    bootstrap_predictions.append(model.predict(Xnew)[0])

                stderr = np.std(bootstrap_predictions, ddof=1)

            # Display prediction and standard error
            if stderr is not None:
                right.container(border=True).write(
                    f"<p style='text-align:center;font-size:1.5em'>"
                    f"Predicted {response}:<br><br>"
                    f"<b>{prediction:.4g} ± {stderr:.4g}</b>"
                    "</p>",
                    unsafe_allow_html=True
                )
            else:
                right.container(border=True).write(
                    f"<p style='text-align:center;font-size:1.5em'>"
                    f"Predicted {response}:<br><br>"
                    f"<b>{prediction:.4g}</b>"
                    "</p>",
                    unsafe_allow_html=True
                )
        st.write("")
        st.write("")
        st.write("")
        st.write("")