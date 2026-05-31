# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the Creative Commons Attribution-NonCommercial 
# 4.0 International License. 


import streamlit as st
from pathlib import Path
import sys
from io import StringIO 
import pandas as pd
import numpy as np
from janitor import clean_names
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from optimeo.bo import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import streamlit.components.v1 as components

about_items={
        'Get Help': 'mailto:colin.bousige@cnrs.fr',
        'Report a bug': "mailto:colin.bousige@cnrs.fr",
        'About': """
        ## OPTIMA
        Version date 2025-04-14.

        This app was made by [Colin Bousige](https://lmi.cnrs.fr/author/colin-bousige/). Contact me for support, requests, or to signal a bug.
        """
    }

def read_markdown_file(markdown_file):
    """Read a md file and return the content"""
    return Path(markdown_file).read_text()

def writeout(df: pd.DataFrame, format='csv'):
    """Write a pd.DataFrame to csv. To use with st.download_button()"""
    df = clean_names(df)
    return df.to_csv(index=False).encode('utf-8')

def write_poly(pp):
    signs = [np.sign(x) for x in pp]
    p = np.array([f"{x:.2e}" for x in pp])
    p = [x.split("e") for x in p]
    p = [f"{x[0]}•10<sup>{int(x[1])}</sup>" for x in p]
    p = [x.replace(f'•10<sup>{0}</sup>', '') for x in p]
    out = "y = "
    order = len(p) - 1
    for i,n in zip(range(len(p)), range(order, -1, -1)):
        if i==0 and n>1:
            out += f"{p[0]}x<sup>{n}</sup>"
        elif i==0 and n==1:
            out += f"{p[0]}x"
        elif i==0 and n==0:
            out += f"{p[0]}"
        elif n == 1:
            out += f"{'' if signs[i]<0 else ' +'} {p[i]}x"
        elif n == 0:
            out += f"{'' if signs[i]<0 else ' +'} {p[i]}"
        else:
            out += f"{'' if signs[i]<0 else ' +'} {p[i]}x<sup>{n}</sup>"
    out = out.replace('-', '–')
    return(out)

def display_figure(file_path):
    with open(file_path, 'r') as file:
        html_content = file.read()
    components.html(html_content, height=800)

# @st.cache_data
def encode_data(data, factors, response, factor_ranges):
    """
    Read experimental data from a CSV file and format it into features and outcomes dictionaries.

    Parameters:
    - file_path (str): Path to the CSV file containing experimental data.
    - out_pos (list of int): Column indices of the outcome variables. Default is the last column.

    Returns:
    - Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]: Formatted features and outcomes dictionaries.
    """
    features = data[factors].copy()
    outcomes = data[response].copy()
    message = {}
    formatted_features = {}
    for column in features.columns:
        if features[column].dtype == 'object':
            unique_values = features[column].unique()
            if len(unique_values) == 1:
                message[column] = f"Only one unique value found in **{column}**. This column will be removed from the features."
            else:
                formatted_features[column] = {
                    'type': 'text',
                    'data': [str(val) for val in features[column].tolist()],
                    'range': [str(val) for val in unique_values.tolist()]
                    }
        elif 'int' in str(features[column].dtype):
            unique_values = features[column].unique()
            min_val = int(np.min(factor_ranges[column]))
            max_val = int(np.max(factor_ranges[column]))
            if len(unique_values) == 1:
                message[column] = f"Only one unique value found in **{column}**. This column will be removed from the features."
            elif min_val >= max_val:
                message[column] = f"Invalid range for **{column}**: {min_val} >= {max_val}. This column will be removed from the features."
            # if data points are not in the range of the unique values, remove them
            elif (any(val < min_val for val in unique_values) or 
                  any(val > max_val for val in unique_values)):
                message[column] = f"Invalid range for **{column}**: some data points do not belong to the range [{min_val},{max_val}]. This column will be removed from the features."
            else:
                formatted_features[column] = {
                    'type': 'int',
                    'data': [int(val) for val in features[column].tolist()],
                    'range': [min_val, max_val]
                    }
        elif 'float' in str(features[column].dtype):
            unique_values = features[column].unique()
            min_val = float(np.min(factor_ranges[column]))
            max_val = float(np.max(factor_ranges[column]))
            if len(unique_values) == 1:
                message[column] = f"Only one unique value found in **{column}**. This column will be removed from the features."
            elif min_val >= max_val:
                message[column] = f"Invalid range for **{column}**: {min_val} >= {max_val}. This column will be removed from the features."
            elif (any(val < min_val for val in features[column]) or 
                  any(val > max_val for val in features[column])):
                message[column] = f"Invalid range for **{column}**: some data points do not belong to the range [{min_val},{max_val}]. This column will be removed from the features."
            else:
                formatted_features[column] = {
                    'type': 'float',
                    'data': [float(val) for val in features[column].tolist()],
                    'range': [min_val, max_val]
                    } 

    # same for outcomes with just type and data
    formatted_outcomes = {}
    for column in outcomes.columns:
        if outcomes[column].dtype == 'object':
            unique_values = outcomes[column].unique()
            formatted_outcomes[column] = {'type': 'text',
                                          'data': unique_values.tolist()}
        elif outcomes[column].dtype in ['int64', 'float64']:
            min_val = outcomes[column].min()
            max_val = outcomes[column].max()
            outcome_type = 'int' if outcomes[column].dtype == 'int64' else 'float'
            formatted_outcomes[column] = {'type': outcome_type,
                                          'data': outcomes[column].tolist()}

    return formatted_features, formatted_outcomes, message

def encode_data2(data, factors):
    dtypes = data.dtypes
    encoders = {}
    for factor in factors:
        if dtypes[factor] == 'object':
            le = LabelEncoder()
            encoders[factor] = le
            data[factor] = le.fit_transform([str(d) for d in data[factor]])
    return data, encoders, dtypes

# Function to check constraints
def check_constraints(df, constraints):
    results = {}
    for constraint in constraints:
        # Evaluate the constraint as a boolean expression
        results[constraint] = df.eval(constraint)
    return results

def check_ranges(ranges):
    """
    Check if the ranges are valid.
    """
    message = ""
    invalid = []
    for factor, range_ in ranges.items():
        if len(range_) != 2:
            message = f"Range for {factor} should be a tuple of length 2. **{factor} was removed from the features.**"
            invalid.append(factor)
        if range_[0] > range_[1]:
            message = f"Invalid range for {factor}: {range_}. The first value should be less than the second. **{factor} was removed from the features.**"
            invalid.append(factor)
        if len(np.unique(range_)) == 1:
            message = f"Invalid range for {factor}: {range_}. The values should be different. **{factor} was removed from the features.**"
            invalid.append(factor)
    return message, invalid

# @st.cache_data
def update_model(features, outcomes, 
                 factor_ranges, Nexp, maximize, 
                 fixed_features, feature_constraints,outcome_constraints,
                 sampler, acq_function, rseed):
    """
    Update the model if the parameters have changed.
    """
    # Check if the model is already in the session state
    # or if the parameters necessitating reinitializing the model have changed
    if (st.session_state['bo'] is None or 
            # if these changed, a new model needs to be created
            features != st.session_state['bo'].features or 
            outcomes != st.session_state['bo'].outcomes or
            factor_ranges != st.session_state['bo'].ranges or 
            maximize != st.session_state['bo'].maximize):
        st.session_state['bo'] = BOExperiment(
            features            = features, 
            outcomes            = outcomes,
            ranges              = factor_ranges,
            N                   = Nexp,
            maximize            = maximize,
            outcome_constraints = outcome_constraints,
            fixed_features      = fixed_features,
            feature_constraints = feature_constraints,
            optim               = sampler,
            acq_func            = acq_function,
            seed                = rseed
            )
    else:
        # see if parameters that just play on the generation have changed
        # N
        if Nexp != st.session_state['bo'].N:
            st.session_state['bo'].N = Nexp
        # fixed_features
        if fixed_features != st.session_state['bo'].fixed_features:
            st.session_state['bo'].fixed_features = fixed_features
        # feature_constraints
        if feature_constraints != st.session_state['bo'].feature_constraints:
            st.session_state['bo'].feature_constraints = feature_constraints
        # outcome_constraints
        if outcome_constraints != st.session_state['bo'].outcome_constraints:
            st.session_state['bo'].outcome_constraints = outcome_constraints
        # sampler
        if sampler != st.session_state['bo'].optim:
            st.session_state['bo'].optim = sampler



def check_data(data, factors):
    """
    Check if the data is valid.
    """
    features = data[factors].copy()
    message = {}
    for column in features.columns:
        unique_values = features[column].unique()
        if len(unique_values) == 1:
            message[column] = f"Only one unique value found in **{column}**. This column will be removed from the features."
    return message


def load_data_widget():
    """Reusable data loading widget for sidebar"""
    st.sidebar.header("🗂️ Data Management")
    
    # Initialize session state
    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = None
    if 'data_filename' not in st.session_state:
        st.session_state.data_filename = None
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file", 
        type=["csv",'xlsx','xls', 'xlsm', 'xlsb', 'odf', 'ods', 'odt'],
        key="persistent_uploader"
    )
    
    # Load data button
    if uploaded_file is not None:
        try:
            # Determine file type and load accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.loaded_data = data
            st.session_state.data_filename = uploaded_file.name
            st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Display current data info
    if st.session_state.loaded_data is not None:
        data = st.session_state.loaded_data
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Current Dataset:**")
        st.sidebar.markdown(f"📄 **File:** {st.session_state.data_filename}")
        st.sidebar.markdown(f"📊 **Shape:** {data.shape[0]:,} rows × {data.shape[1]} columns")
        st.sidebar.markdown(f"💾 **Memory:** {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Data management buttons
        
        if st.sidebar.button("🗑️ Clear"):
            st.session_state.loaded_data = None
            st.session_state.data_filename = None
            st.rerun()
    
    return st.session_state.loaded_data

def get_loaded_data():
    """Simple function to get loaded data from session state"""
    return st.session_state.get('loaded_data', None)