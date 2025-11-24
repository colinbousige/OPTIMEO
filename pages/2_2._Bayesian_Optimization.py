# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version. 


import streamlit as st
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeError)
from resources.functions import *
from optimeo.bo import *
import pandas as pd
import numpy as np
from resources.functions import about_items

st.set_page_config(page_title="Bayesian Optimization",
                   page_icon="resources/icon.png",
                   layout="wide", menu_items=about_items)

style = read_markdown_file("resources/style.css")
st.markdown(style, unsafe_allow_html=True)

if "bo" not in st.session_state:
    st.session_state['bo'] = None
if "next" not in st.session_state:
    st.session_state['next'] = None
if "best" not in st.session_state:
    st.session_state['best'] = None
if "model_up_to_date" not in st.session_state:
    st.session_state['model_up_to_date'] = False
if "plot_up_to_date" not in st.session_state:
    st.session_state['plot_up_to_date'] = False
if "pareto_front_up_to_date" not in st.session_state:
    st.session_state['pareto_front_up_to_date'] = False
if "showerror" not in st.session_state:
    st.session_state['showerror'] = True

def model_changed():
    st.session_state.model_up_to_date = False
    st.session_state.plot_up_to_date = False
    st.session_state.pareto_front_up_to_date = False
    st.session_state.bo = None
def model_updated():
    st.session_state.model_up_to_date = True
def plot_changed():
    st.session_state.plot_up_to_date = False
def plot_updated():
    st.session_state.plot_up_to_date = True
def pareto_front_updated():
    st.session_state.pareto_front_up_to_date = True


# if "data" not in st.session_state:
#     st.session_state['data'] = None
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# Bayesian Optimization
""")


tabs = st.tabs(["Data Loading", "Bayesian Optimization", 'Predictions'])

with tabs[0]:# Data Loading
    rseed = st.sidebar.number_input("Random seed (for reproducibility):", 
                                  min_value=0, value=42)
    colos = st.columns([2,3])
    data = load_data_widget()
    if data is None:
        with st.expander("**How to format your data?**"):
            st.markdown(
                """The data must be in tidy format, meaning that each column is a variable and each row is an observation. We usually place the factors in the first columns and the response(s) in the last column(s). Data type can be float, integer, or text, and you can only specify one response. Spaces and special characters in the column names will be automatically removed. The first row of the file will be used as the header.

For Excel-like files, the first sheet will be used, and data should start in the A1 cell, and no unnecessary rows or columns should be present. 
"""
        )
            cols = st.columns([1,2,1])
            cols[1].image("resources/tidy_data.jpg", caption="Example of tidy data format")
        with st.expander("**Bayesian Optimization in simple terms**"):
            st.markdown("""**Bayesian optimization** is a strategy used to find the best settings or parameters for a system or model, especially when evaluating each setting is expensive or time-consuming. Here's a simple explanation:

- **Imagine a Landscape:** Think of the problem as a hilly landscape where the height of the hills represents how well the system performs with different settings. Your goal is to find the highest peak (the best performance).

- **Initial Guesses:** You start by making a few initial guesses about where the highest peak might be. These guesses are based on some prior knowledge or random sampling.

- **Build a Model:** Based on these guesses, you build a simple model (often called a surrogate model) that approximates the landscape. This model helps predict what the landscape looks like, even in areas you haven't explored yet.

- **Update Beliefs:** As you evaluate more settings, you update your model. This updating process is where the "Bayesian" part comes in—you're continually refining your beliefs about the landscape based on new information. Bayes' theorem is a way to update the probability of a hypothesis as more evidence or information becomes available. It's based on the idea that the likelihood of an event can change when you consider new data, combining your initial belief (prior probability) with new evidence (likelihood) to give you a revised belief (posterior probability).

- **Choose the Next Point:** The key idea is to choose the next setting to evaluate by balancing two goals:

  - **Exploitation:** Choosing settings where your model predicts the performance will be high.
  - **Exploration:** Choosing settings where the model is uncertain, to gather more information and improve the model.

- **Iterate:** You repeat the process of updating the model and choosing new settings until you find the highest peak or run out of resources (like time or computational power).

In simple terms, Bayesian optimization is like a smart search strategy that helps you efficiently find the best settings for a complex system by learning from each attempt and making educated guesses about where to look next.
""")
        with st.expander("""**But the app tells me to go measure new points with a predicted outcome that is worse than what I already have in my database!**

**This does not work as I thought...**"""):
                st.markdown("""Let's see how it works in practice with a simple example with only one continuous feature – it is the same procedure with more features, it's just harder to visualize in more than 2 or 3 dimensions.

At the core of the Bayesian Optimization is the Gaussian Process (GP) regression and the acquisition function. Here, we use an acquisition function called Expected Improvement (EI) (in fact, its log value).

The GP regression is a non-parametric regression method that uses the data to build a probabilistic model of the response. The GP regression is used to predict the response at new points, and it also provides an uncertainty estimate (the standard deviation) for each prediction. The EI acquisition function is a function computing the expected improvement of the response at a new point compared to the best response observed so far. It is used to decide where to sample next by maximizing it.
<details><summary><b>More about the EI function</b></summary>

Let's denote:
- $f(x)$ as the objective function we want to maximize.
- $x$ as a point in the search space.
- $f^*$ as the current best observed value of the objective function.
- $\\mu(x)$ and $\\sigma(x)$ as the predicted mean and standard deviation of the objective function at point $x$, respectively, based on a Gaussian process model.
- $\\Phi$ as the cumulative distribution function (CDF) of the standard normal distribution.
- $\\phi$ as the probability density function (PDF) of the standard normal distribution.

The expected improvement (EI) at a point $x$ is defined as:

$$ \\text{EI}(x) = \\mathbb{E}[\\max(f(x) - f^*, 0)] $$

This can be expressed in terms of the Gaussian process model as:

$$ \\text{EI}(x) = (\\mu(x) - f^*) \\Phi(Z) + \\sigma(x) \\phi(Z) $$

where

$$ Z = \\frac{\\mu(x) - f^*}{\\sigma(x)} $$

##### Interpretation

- **$\\mu(x) - f^*$**: This term represents the expected improvement in the mean prediction over the current best observed value.
- **$\\Phi(Z)$**: This term represents the probability that the predicted value at $x$ is greater than the current best observed value.
- **$\\sigma(x) \\phi(Z)$**: This term accounts for the uncertainty in the prediction, encouraging exploration in regions where the model is uncertain.

The expected improvement balances exploration (trying new points with high uncertainty) and exploitation (focusing on points with high predicted mean values). It is widely used as an acquisition function in Bayesian optimization to decide where to sample next.

</details>
<br>
""", unsafe_allow_html=True)
                figi = st.slider('Bayesian Optimization step', 0, 15, 0,1)
                display_figure(f'resources/figure_{figi}.html')
    if data is not None:
        data = clean_names(data, remove_special=True, case_type='preserve')
        left, right = st.columns([3,2])
        resp = right.empty()
        fac = left.empty()
        cols = data.columns.to_numpy()
        st.dataframe(data, hide_index=False)
        mincol = 1 if 'run_order' in cols else 0
        factors = fac.multiselect("Select the **parameter(s)** column(s):", 
                data.columns, default=cols[mincol:-1],
                on_change=model_changed)
        # response cannot be a factor, so default are all unselected columns in factor
        available = [col for col in cols if col not in factors]
        responses = resp.multiselect("Select the **outcome(s)** column(s):", 
                available, max_selections=10, default=available[-1],
                on_change=model_changed)
        # add option to change type of columns
        dtypesF = data[factors].dtypes
        placeholder = st.empty()
        st.write("""##### Select the type and range of each parameter
Except for categorical parameters, you can increase the ranges to allow the optimization algorithm to explore values outside the current range of measures.""")
        factor_types = {factor: dtypesF[factor] for factor in factors}
        factor_ranges = {factor: [np.min(data[factor]), np.max(data[factor])] for factor in factors}
        type_choice = {'object':0, 'int64':1, 'float64':2}
        colos = st.columns(5)
        colos[1].write("<p style='text-align:center;'><b>Type</b></p>", unsafe_allow_html=True)
        colos[2].write("<p style='text-align:center;'><b>Min</b></p>", unsafe_allow_html=True)
        colos[3].write("<p style='text-align:center;'><b>Max</b></p>", unsafe_allow_html=True)
        for factor in factors:
            colos = st.columns(5)
            colos[0].write(f"<p style='text-align:right;'><b>{factor}</b></p>", unsafe_allow_html=True)
            factype = type_choice[f"{factor_types[factor]}"]
            factor_types[factor] = colos[1].selectbox(f"Type of **{factor}**", 
                ['Categorical', 'Integer', 'Float'], key=f"type_{factor}", 
                index = factype, label_visibility='collapsed', on_change=model_changed)
            if factor_types[factor] == 'Categorical':
                factor_types[factor] = 'object'
            elif factor_types[factor] == 'Integer':
                factor_types[factor] = 'int64'
            else:
                factor_types[factor] = 'float64'
            data[factor] = data[factor].astype(factor_types[factor])
            if factor_types[factor] != 'object':
                factor_ranges[factor][0] = colos[2].number_input(f"Min value of **{factor}**",
                    value=factor_ranges[factor][0], key=f"min_{factor}", label_visibility='collapsed',
                    on_change=model_changed)
                factor_ranges[factor][1] = colos[3].number_input(f"Max value of **{factor}**",
                    value=factor_ranges[factor][1], key=f"max_{factor}", label_visibility='collapsed',
                    on_change=model_changed)
        messages = []
        if data is not None and len(factors) > 0 and len(responses) > 0:
            dataclean = data[factors+responses].copy()
            dataclean = dataclean.dropna(axis=0, how='any')
            features, outcomes, messages = encode_data(
                dataclean, factors, responses, factor_ranges)
            if len(messages) > 0:
                key, value = list(messages.items())[0]
                messages[key] = '⚠️   '+messages[key]
                message = '''

⚠️   '''.join(messages.values())
                placeholder.error(message)
                for name,messsage in messages.items():
                    # drop factors[name]
                    factors.remove(name)
        st.write("")
        st.write("")
        st.write("")
        st.write("")


with tabs[1]:# Bayesian Optimization
    if data is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the **Sidebar** and select the parameter(s) and outcome(s) in the **Data Loading** tab.""")
    if data is not None and len(factors) > 0 and len(responses) > 0:
        left,right = st.columns([3,1])
        container = st.container(border=True)
        container.write("#### Model options")
        if len(responses) > 2:
            container.info("You have defined more than two outcomes. Know that you can only optimize up to **two** ojectives using this web app. The other outcomes need to be set to **Not an objective**, and you can use them to define outcomes constraints. If you really want to optimize more than two ojectives at once: first, ask yourself if this is really needed for your process. If it is, then either write your own code using the package version of OPTIMEO (as it handles more than two objectives), or contact the author to make an update to the web app.", icon="ℹ️")
        containerplot = st.container(border=True)
        cols = container.columns(4)
        maximize = {}
        non_metric_outcomes = []
        for i in range(len(responses)):
            temp = cols[i%2].radio(f"Direction for **{responses[i]}**:", 
                                   horizontal=False, 
                                   options=["Maximize", "Minimize", "Not an objective"],
                                   on_change=model_changed)
            if temp == "Maximize":
                maximize[responses[i]] = True
            elif temp == "Minimize":
                maximize[responses[i]] = False
            else:
                maximize[responses[i]] = None
                non_metric_outcomes.append(responses[i])
        nmetrics = len([v for v in maximize.values() if v is not None])
        if nmetrics == 0:
            cols[0].warning("You need to select at least one objective to optimize.", icon="⚠️")
            st.session_state['model_up_to_date'] = True
        if nmetrics > 2:
            cols[0].warning("You can only optimize up to two objectives at once.", icon="⚠️")
            st.session_state['model_up_to_date'] = True
        Nexp = cols[2].number_input("Number of new experiments", 
                min_value=1, value=1, max_value=100, 
                help="Number of proposed new experiments to run in parallel to look for the optimum response.", 
                on_change=model_changed)
        samplerchoice = "Bayesian Optimization"
        sampler_list = {"Sobol pseudo-random": 'sobol',
                        "Bayesian Optimization": 'bo'}
        # fix a parameter value
        fixed_features_names = cols[3].multiselect("""Select the fixed parameters (if any)""", 
                factors, help="""Select one or more parameters to fix during generation. You may want to do that if you can perform several experiments at the same time with fixed parameters. 

For example, this can happen if you are using a robot to make experiments with varying concentrations but fixed temperature.""", on_change=model_changed)
        cols = container.columns(4)
        fixed_features_values = [None]*len(fixed_features_names)
        if len(fixed_features_names) > 0:
            for i,feature in enumerate(fixed_features_names):
                if factor_types[feature] == 'object':
                    cases = dataclean[feature].unique()
                    fixed_features_values[i] = cols[(i%3)+1].selectbox(f"Value of **{feature}**:", 
                                                                    cases, 
                                                                    key=f"fixpar{i}", 
                                                                    on_change=model_changed)
                else:
                    fixed_features_values[i] = cols[(i%3)+1].number_input(f"Value of **{feature}**:", 
                                                                        value=np.mean(dataclean[feature]), key=f"fixpar{i}", 
                                                                        on_change=model_changed)
        # regroup the fixed features in a dict
        fixed_features = {}
        for i,feature in enumerate(fixed_features_names):
            fixed_features[feature] = fixed_features_values[i]
        
        # add a text input to add constraints
        cols = container.columns([2,1])
        feature_constraints = cols[0].text_input("""Add **linear** constraints to the **parameters**""",
                help="""Add **linear** constraints to the parameters. Leave blank if no constraints, and use a comma to separate multiple constraints.

The constraints should be in the form of inequalities such as:

- `x1 >= 0`
- `x2 <= 10, x4 >= -0.5`
- `x1 + 3*x2 <= 5`

If you want to add non-linear constraint like `x1^2 + x2^2 <= 5`, you should first transform your columns before loading the data file.""", on_change=model_changed)
        if len(feature_constraints)>0:
            feature_constraints = feature_constraints.replace("+", " + ")
            feature_constraints = feature_constraints.replace("<", "<=")
            feature_constraints = feature_constraints.replace(">", ">=")
            feature_constraints = feature_constraints.replace("<==", "<=")
            feature_constraints = feature_constraints.replace("<=", " <= ")
            feature_constraints = feature_constraints.replace(">==", ">=")
            feature_constraints = feature_constraints.replace(">=", " >= ")
            feature_constraints = feature_constraints.split(",")
        else:
            feature_constraints = []
        
        outcome_constraints = cols[0].text_input(f"""Add **linear** constraints to the **outcomes that are not objectives**: {', '.join(non_metric_outcomes)}""",
                disabled = False if nmetrics > 0 and len(non_metric_outcomes) > 0 else True,
                help="""You can add constraints to the outcomes **that are not objectives**. Leave blank if no constraints, and use a comma to separate multiple constraints.

The constraints should be in the form of inequalities such as:
`constrained_outcome <= some_bound`

**Note:** This corresponds to setting an **objective thresholds** to incorporate domain knowledge. Thresholds define minimum acceptable values for objectives: "If objective_1 is less than X, it doesn't matter how good objective_2 is - that solution is unacceptable." This helps focus the search on practically feasible solutions. For example, if maximizing yield, you might set a threshold of 80% to exclude any solutions below that value regardless of other objectives.""", on_change=model_changed)
        if len(outcome_constraints)>0:
            outcome_constraints = outcome_constraints.replace("+", " + ")
            outcome_constraints = outcome_constraints.replace("<", "<=")
            outcome_constraints = outcome_constraints.replace(">", ">=")
            outcome_constraints = outcome_constraints.replace("<==", "<=")
            outcome_constraints = outcome_constraints.replace("<=", " <= ")
            outcome_constraints = outcome_constraints.replace(">==", ">=")
            outcome_constraints = outcome_constraints.replace(">=", " >= ")
            outcome_constraints = outcome_constraints.split(",")
        else:
            outcome_constraints = []

        acq_function = None
        tuning = cols[1].toggle("Allow tuning Optimization vs Exploitation?",
                                disabled=False if Nexp==1 else True,
                                value=False,
                                help="""⚠️ **This will only work for a single number of experiment**.
                                
⚠️ **If you don't really know what you are doing, just stick with the default acquisition function and switch this off.**

By default, the acquisition function that is used is the logarithm of the Expected Improvement (EI), providing a good balance between exploration and exploitation. If you check this box, the acquisition function will be the Upper Confidence Bound (UCB), which allows you to tune the balance between exploration and exploitation.

The UCB is defined as:

$$ UCB(x) = \\mu(x) + \\sqrt{\\beta} \\sigma(x) $$

where $\\mu(x)$ is the predicted mean at point $x$, $\\sigma(x)$ is the predicted standard deviation at point $x$, and $\\beta$ is a tuning parameter that controls the balance between exploration and exploitation. 

A higher value of $\\beta$ will lead to more exploration, while a lower value will lead to more exploitation. The default value of $\\beta$ is 1, which simplifies to $\\mu(x) + \\sigma(x)$, such that the predictions and the model's uncertainty estimates (standard deviations) are balanced equally.

""")
        if tuning:
            beta = cols[1].number_input("Tuning parameter $\\beta$", 
                                min_value=0., 
                                value = 1.0,
                                step = 1.000,
                                format="%0.8f",
                                on_change = model_changed,
                                help="""Tuning parameter for the UCB acquisition function.

- A **higher** value will lead to more **exploration**,
- A **lower** value will lead to more **exploitation**.""")
            if Nexp==1:
                acq_function = {'acqf': UpperConfidenceBound, 
                                'acqf_kwargs': {'beta': 10**beta}}
            else:
                st.warning("""The UCB acquisition function is only available for a single number of experiment. 
                           
**-> Switching back to the default acquisition function, logEI.**""", icon="⚠️")
                acq_function = None
        
        # Perform Bayesian optimization
        colos = container.columns([6,1])
#         if samplerchoice == "Bayesian Optimization":
#             colos[0].success("**Bayesian optimization** is a probabilistic model. The results may vary slightly each time you run it.", icon=":material/info:")
#         else:
#             colos[0].warning("""You are using the **Sobol pseudo-random generator**. The results will vary each time you run it.
                            
# **You are _not_ performing an optimization**, but an uniform sampling of the parameter space. This is suitable for the first few iterations of the optimization (exploration), then switch to Bayesian optimization.""", icon="⚠️")
        modelbutton = colos[1].empty()
        plotbutton = containerplot.empty()
        plotparetobutton = containerplot.empty()
        # Check constraints
        if len(feature_constraints) > 0:
            constraint_results = check_constraints(data, feature_constraints)
            all_valid = all(result.all() for result in constraint_results.values())
            if not all_valid:
                # print which constraints are not valid
                for feature, result in constraint_results.items():
                    if not result.all():
                        whichfails = result[result == False].index.tolist()
                        colos[0].error(f"Constraint **{feature}** is invalid for the given data (see lines: {whichfails}). It was discarded.")
                        # drop feature_constraints[i]
                        feature_constraints = [f for f in feature_constraints if f != feature]
        if modelbutton.button("Compute / Update model", type="primary", 
                              disabled=st.session_state['model_up_to_date'], 
                              on_click=model_updated, 
                              help="""⚠️ **Bayesian optimization** is a probabilistic model. 

The results may vary slightly each time you run it."""):
            update_model(
                    features, outcomes,
                    factor_ranges, Nexp, maximize, 
                    fixed_features, feature_constraints, outcome_constraints,
                    sampler_list[samplerchoice],acq_function, rseed
                    )
            st.session_state.plot_up_to_date = False
            st.session_state['next'] = st.session_state['bo'].suggest_next_trials()
            st.session_state['best'] = st.session_state['bo'].get_best_parameters()
        if (st.session_state['bo'] is not None and 
            st.session_state['next'] is not None and 
            st.session_state['best'] is not None):
            cols= container.columns(2)
            cols[0].write("**Next experiments to perform:**")
            cols[0].dataframe(st.session_state['next'], hide_index=True)
            cols[1].write("**Best parameters found:**")
            cols[1].dataframe(st.session_state['best'], hide_index=True)
        figmod = []
        figopt = None
        # add a button to launch pareto frontiers plotting
        containerplot.write("#### Plot options")
        # add help about slicing parameters in a tooltip
        with containerplot.expander("**Information about slicing parameters**", icon=":material/info:"):
            st.write("""When your search space has more than two parameters, you can still visualize a 2D contour plot by fixing the extra parameters to constant values. This creates a 2D "slice" through your high-dimensional space, allowing you to focus on just two parameters at a time.

**How to Fix Parameters:**

- Enter a value in the boxes below to fix a parameter to that value for the plot.
- Leave a box empty if you want the parameter to remain free (not fixed).

**What Happens If You Leave Parameters Free?**

- If more than two parameters are free, the plot will let you choose which two parameters to visualize.
- The other free parameters will be marginalized (averaged over) in the plot.

**Understanding the Plot Colors:**

The color of each point in the plot represents its distance from the slice values (the fixed parameters).
The colors transition smoothly:

- :blue[**Blue**]: Points are below the slice value.
- **White**: Points are close to the slice value.
- :red[**Red**]: Points are above the slice value.

""")
        cols= containerplot.columns([3,3,1])
        parslice = {}
        for i,f in enumerate(factors):
            if features[f]['type'] == 'float':
                temp = cols[i%2].number_input(f"Slice for **{f}**", key=f"parslice{f}", 
                                               on_change=plot_changed,
                                               value=None, min_value=features[f]['range'][0],
                                               max_value=features[f]['range'][1])
                if temp is not None:
                    parslice[f] = temp
            if features[f]['type'] == 'text':
                temp = cols[i%2].multiselect(f"Slice for **{f}**", max_selections=1,
                                              on_change=plot_changed,
                                              options=features[f]['range'], key=f"parslice{f}")
                if len(temp) > 0:
                    parslice[f] = temp[0]
            if features[f]['type'] == 'int':
                temp = cols[i%2].number_input(f"Slice for **{f}**", key=f"parslice{f}", 
                                               on_change=plot_changed,
                                               value=None, 
                                               min_value=int(features[f]['range'][0]),
                                               max_value=int(features[f]['range'][1]))
                if temp is not None:
                    parslice[f] = temp

        # find which parameters are not in parslice and are not in fixed_features
        not_fixed = [f for f in factors if f not in parslice and f not in fixed_features_names]
        # count how many parameters in not_fixed are float or int
        count = len([name for name in not_fixed if features[name]['type']=='float' or features[name]['type']=='int'])
        if st.session_state['bo'] is not None and (cols[2].button("Plot model / Update plots", type="primary",
                             on_click=plot_updated,
                             disabled=st.session_state['plot_up_to_date']) or
            st.session_state['plot_up_to_date'] == True):
            toplot = [r for r in responses if r not in non_metric_outcomes]
            if count>0:
                for i in range(len(toplot)):
                    figmod.append(st.session_state['bo'].plot_model(metricname=toplot[i], 
                                                slice_values=parslice,
                                                linear=False if count > 1 else True,
                                                ))
            if len(responses) == 1:
                figopt = st.session_state['bo'].plot_optimization_trace()
            if figmod is not None and count>0:
                for i in range(len(toplot)):
                    st.plotly_chart(figmod[i], key=f"figmod{i}")
            elif figmod is not None and count==0:
                st.warning("Can't plot a model with no free features or with no numerical features.", 
                           icon="⚠️")
            if figopt is not None:
                st.plotly_chart(figopt, key=f"figopt")
        colos = containerplot.columns([1,1,1,2])
        if (st.session_state['bo'] is not None and 
            len(responses) >1 and
            # st.session_state['plot_up_to_date'] == True and
            st.session_state['bo'].model is not None and
            colos[0].button("Compute Pareto frontiers", 
                           type="primary",
                           disabled=st.session_state['pareto_front_up_to_date'],
                           on_click=pareto_front_updated)):
                paretofront = st.session_state['bo'].compute_pareto_frontier()        
        if (st.session_state['bo'] is not None and 
            len(responses) >1 and
            st.session_state['bo'].model is not None and
            st.session_state['pareto_front_up_to_date'] == True):
                figpareto = None
                if colos[1].button("Plot Pareto frontiers **with** error bars", type="primary"):
                    figpareto = st.session_state['bo'].plot_pareto_frontier(
                        show_error_bars=True)
                if colos[2].button("Plot Pareto frontiers **without** error bars", type="primary"):
                    figpareto = st.session_state['bo'].plot_pareto_frontier(
                        show_error_bars=False)
                if figpareto is not None:
                    st.plotly_chart(figpareto, key="figparetoplot")


with tabs[2]:# Predictions
    if data is None:
        st.warning("""The data is not yet loaded. Please upload a data file in the **Sidebar** and select the parameter(s) and outcome(s) in the **Data Loading** tab.""")
    if data is not None and len(factors) > 0 and len(responses) > 0:
        st.write(f"#### Select the parameters for prediction of {', '.join(responses)}")
        cols = st.columns(4)
        # add a button to launch predictions
        parslice = {}
        for i,f in enumerate(factors):
            if features[f]['type'] == 'float':
                parslice[f] = cols[i%4].number_input(f"**{f}**", 
                                                    value=float(np.mean(features[f]['range'])), 
                                                    min_value=float(features[f]['range'][0]),
                                                    max_value=float(features[f]['range'][1]))
            elif features[f]['type'] == 'text':
                parslice[f] = str(cols[i%4].selectbox(f"**{f}**",
                                        options=features[f]['range']))
            elif features[f]['type'] == 'int':
                parslice[f] = cols[i%4].number_input(f"**{f}**", 
                                                      value=int(np.mean(features[f]['range'])), 
                                                      min_value=int(features[f]['range'][0]),
                                                      max_value=int(features[f]['range'][1]))
        if st.session_state['bo'] is None:
            st.warning("""The model is not yet computed. Please compute the model in the **Bayesian Optimization** tab.""")
        if len(parslice) > 0 and st.session_state['bo'] is not None:
            pred, stderrs = st.session_state['bo'].predict([parslice])
            pred = pd.DataFrame(pred)
            stderrs = pd.DataFrame(stderrs)
            # Concatenate side by side
            result = pd.concat([pred, stderrs.add_suffix(' standard error')], axis=1)

            # If you want to pivot to long format with columns: response, prediction, standard error
            result_long = pd.DataFrame([
                {
                    'Response': col,
                    'Prediction': pred[col].iloc[0],
                    'Standard error': stderrs[col].iloc[0]
                }
                for col in pred.columns
            ])

            cols = st.columns([1,2,1])
            cols[1].dataframe(result_long, hide_index=True)
            # actual = {}
            # cols[1].write("Update model with actual value of the response for these parameters")
            # for i in range(len(responses)):
            #     temp = cols[1].number_input(f"Actual value of **{responses[i]}**", 
            #                                                 value=None, step=0.1)
            #     if temp is not None:
            #         actual[responses[i]] = temp
            # if cols[1].button("Update model"):
            #     if len(actual) == len(responses):
            #         # update the model with the new data
            #         newdata = pd.DataFrame({**parslice, **actual}, index=[0])
            #         st.session_state['data'] = pd.concat([st.session_state['data'], newdata], ignore_index=True)
            #         st.success("Model updated with actual value of the response.")
            #     else:
            #         st.error("Please provide actual values for all responses.")