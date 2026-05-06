# Copyright (c) 2025 Colin BOUSIGE
# Contact: colin.bousige@cnrs.fr
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version. 

import streamlit as st
import numpy as np
import pandas as pd
from resources.functions import about_items
from io import BytesIO
import xlsxwriter
from resources.functions import *
from optimeo.doe import *
from datetime import datetime
import re

st.set_page_config(page_title="Design Of Experiment", 
                   page_icon="resources/icon.png",
                   layout="wide", menu_items=about_items)
style = read_markdown_file("resources/style.css")
st.markdown(style, unsafe_allow_html=True)

defaultParNames = ["Temperature", "ConcentrationA", "ConcentrationB", "Reaction_time",
                   "Flux", "Parameter_6", "Parameter_7", "Parameter_8", "Parameter_9", 
                   "Parameter_10", "Parameter_11", "Parameter_12", "Parameter_13", 
                   "Parameter_14", "Parameter_15", "Parameter_16", "Parameter_17", 
                   "Parameter_18", "Parameter_19", "Parameter_20"]

rseed = 42
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# Definition of User Interface
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
st.write("""
# Design Of Experiment
""")

design_type = st.sidebar.selectbox("Design type",
        ['Sobol sequence','Full Factorial', 'Fractional Factorial', 
         'Definitive Screening', 'Space Filling Latin Hypercube', 
         'Randomized Latin Hypercube', 'Optimal', 'Plackett-Burman', 'Box-Behnken', 'Central Composite'])

cols = st.sidebar.columns(2)
cols[0].write("N parameters")
Npars = cols[0].number_input("N parameters", min_value=1, max_value=20, value=2, label_visibility="collapsed")
parameters = np.array([{'name': '', 'actual': 0., 'coded': 0.} for i in range(Npars)])
cols[1].write("Shuffle experiments order")
randomize = cols[1].checkbox("Randomize", value=True, label_visibility="collapsed")
reduction = 2
order = 2
Nexp = 4
center = (1,1)
alpha = 'o'
face = 'ccc'
feature_constraints = []

if design_type=='Optimal':
    model_order = st.sidebar.selectbox("Model order:", 
            ['linear','quadratic','cubic'], index=1)
    if model_order == 'linear':
        order = 1
        minexp = Npars + 1
    elif model_order == 'quadratic':
        order = 2
        minexp = int((Npars+1)*(Npars+2)/2)
    elif model_order == 'cubic':
        order = 3
        minexp = int((Npars+1)*(Npars+2)*(Npars+3)/6)
# # # # # # # # # # # # # # # # 
tab1, tab2, tab3 = st.tabs(["How to choose the proper design?", "Parameters Ranges", "Experimental Design"])

with tab1:
    # Add an explanation section at the bottom
    with st.expander("About Design of Experiments"):
        st.markdown("""
        ##### What is Design of Experiments (DoE)?
        
        Design of Experiments is a systematic method to determine the relationship between factors affecting a process and the output of that process. It helps in identifying the optimal conditions for a process, understanding which variables are most influential, and how they interact.
        
        ##### How to Choose the Right Design:
        
        1. **Consider your goals**: Are you screening for important factors, optimizing a process, or exploring relationships?
        2. **Factor count matters**: More factors often require more efficient designs
        3. **Factor types**: Some designs work only with continuous factors
        4. **Expected relationships**: If you anticipate non-linear effects or interactions, certain designs are better suited
        5. **Resource constraints**: Consider how many experimental runs you can afford
        
        Use the filters above to find the most appropriate design for your specific needs.
        """)
    # Create filter section
    st.write(f"#### Filter designs based on your requirements")

    col1, col2, col3, col4 = st.columns([2,3,3,3])

    with col1:
        # Filter options for the left column
        num_factors = st.radio(
            "Number of factors:",
            ["Any", "Few factors (2-5)", "Many factors (6+)"]
        )
    with col2:
        factor_types = st.pills(
            "Factor types:",
            ["Continuous", "Categorical (multi-level)"],
            default=["Continuous", "Categorical (multi-level)"],
            selection_mode="multi"
        )
        
    with col3:
        # Filter options for the right column
        relationships = st.pills(
            "Expected relationships:",
            ["Linear", "Non-linear/Quadratic", "Interaction effects"],
            default=["Linear", "Non-linear/Quadratic", "Interaction effects"],
            selection_mode="multi"
        )
    with col4:
        purpose = st.radio(
            "Primary purpose:",
            ["Any", "Screening (identify important factors)", "Optimization", "Comprehensive exploration"]
        )

    # Dictionary of DoE methods and their characteristics
    doe_methods = {
        "Sobol Sequence": {
            "description": "Pseudo-random uniform paving of the parameter space with a fixed number of experiments. OK with multi-level categorical factors.",
            "detailed_info": "A Sobol sequence is a type of low-discrepancy sequence used to generate quasi-random numbers. It is particularly effective for uniformly sampling high-dimensional spaces, making it ideal for applications in numerical integration and optimization. The sequence is designed to fill the parameter space more evenly than pseudo-random sequences, reducing the risk of clustering and ensuring better coverage. This is especially useful when the number of experiments is fixed, and you need a representative sampling of the parameter space. Sobol sequences are deterministic, meaning the same sequence can be reproduced exactly, which is beneficial for reproducibility in experiments.",
            "num_factors": ["Few factors (2-5)", "Many factors (6+)"],
            "factor_types": ["Continuous", "Categorical (multi-level)"],
            "relationships": ["Linear", "Non-linear/Quadratic", "Interaction effects"],
            "purpose": ["Comprehensive exploration", "Screening (identify important factors)"]
        },
        "Full Factorial Design": {
            "description": "Small number of factors and levels. OK with multi-level categorical factors.",
            "detailed_info": "A full factorial experimental design is ideal for systematically exploring all possible combinations of factors at different levels. This means that for $k$ factors, each having $n$ levels, one needs to perform $n^k$ experimental runs. This design is useful when the number of factors and levels is small, and when interactions between factors are expected to be important. It is especially useful when the goal is to understand how multiple factors interact with each other and influence the outcome of an experiment. This design provides comprehensive data, allowing for the analysis of both individual effects and interactions between factors. However, it can become resource-intensive as the number of factors increases due to the exponential growth in the number of experiments needed.",
            "num_factors": ["Few factors (2-5)"],
            "factor_types": ["Continuous", "Categorical (multi-level)"],
            "relationships": ["Linear", "Non-linear/Quadratic", "Interaction effects"],
            "purpose": ["Comprehensive exploration"]
        },
        "Fractional Factorial Design": {
            "description": "Large number of factors, interactions not expected to be important. OK with multi-level categorical factors.",
            "detailed_info": "A fractional factorial design is a reduced version of a full factorial design. The design is orthogonal, meaning that the main effects are uncorrelated with each other. The design is also confounded, meaning that some main effects are aliased with interactions. This design is particularly useful when the number of factors is large, and when interactions between factors are not expected to be important. By using fewer runs, fractional designs sacrifice some detailed information, particularly interactions between factors, but are effective in capturing the main effects. This trade-off is acceptable when interactions between factors are assumed to be less critical. The resolution $r$ of a fractional factorial design refers to the ability to distinguish between main effects and interaction effects. Typically, the number of experimental runs becomes $2^{k-r}$.",
            "num_factors": ["Many factors (6+)"],
            "factor_types": ["Continuous", "Categorical (multi-level)"],
            "relationships": ["Linear"],
            "purpose": ["Screening (identify important factors)"]
        },
        "Definitive Screening Design": {
            "description": "Many factors, but few are expected to have a large effect. Non-linear relationships or factor interactions might be present.",
            "detailed_info": "A Definitive Screening Design (DSD) is a type of experimental design used in statistics to efficiently identify important factors and their effects in a process or system. It was introduced by Bradley Jones and Christopher Nachtsheim in 2011 as a more efficient and versatile screening tool compared to traditional screening designs, such as fractional factorial or Plackett-Burman designs. The primary purpose of a DSD is to screen factors (independent variables) and identify the most significant ones that influence the response (dependent variable) in an experiment. DSDs are designed to handle experiments where there may be a large number of factors, but only a few are likely to be influential. DSDs are structured to avoid the confounding of main effects with two-factor interactions. This means that DSDs allow for the estimation of main effects independently, and the effects of interactions between factors are not confounded with them. Each factor in a DSD is typically tested at three levels: low (-1), middle (0), and high (+1). This allows for the detection of non-linear effects, which is a key advantage over traditional two-level screening designs. DSDs are highly efficient, requiring fewer experimental runs than traditional designs while still providing valuable information about the main effects and interactions. The number of experimental runs is often equal to $2k + 1$, where $k$ is the number of factors, making them much smaller in size for initial screening experiments. DSDs can detect both main effects and interactions, as well as quadratic effects (non-linear relationships between factors and responses). This is a significant improvement over many other screening designs, which focus only on main effects.",
            "num_factors": ["Many factors (6+)"],
            "factor_types": ["Continuous", "Categorical (multi-level)"],
            "relationships": ["Linear", "Non-linear/Quadratic", "Interaction effects"],
            "purpose": ["Screening (identify important factors)"]
        },
        "Space Filling Latin Hypercube": {
            "description": "Large number of factors, interactions not expected to be important. Continuous factors only.",
            "detailed_info": "A Latin hypercube design (LHD) is a statistical method used for sampling across multiple dimensions in an efficient and balanced way. It divides the range of each variable into equal intervals and ensures that each interval is sampled once, creating a well-distributed set of experimental points. This design is particularly useful in experiments with many factors and when interactions between factors are not expected to be important, as it requires fewer runs than a full factorial design while still providing good coverage of the variable space, making it valuable for complex simulations or optimization studies. It is only available for continuous factors.",
            "num_factors": ["Many factors (6+)"],
            "factor_types": ["Continuous"],
            "relationships": ["Linear"],
            "purpose": ["Comprehensive exploration"]
        },
        "Randomized Latin Hypercube": {
            "description": "Large number of factors, interactions not expected to be important. Continuous factors only. Space filling.",
            "detailed_info": "A Randomized Latin hypercube design is a statistical method used for efficiently sampling large, multidimensional spaces in experiments. It ensures that the sampling is spread evenly across all variables, reducing the chance of clustering. The design randomizes the selection of points while maintaining balanced coverage of each factor's range. It's particularly useful in situations where you have many variables, and you want to optimize sampling efficiency with fewer experimental runs compared to full factorial designs. This is commonly used in simulations or modeling complex systems. It is only available for continuous factors.",
            "num_factors": ["Many factors (6+)"],
            "factor_types": ["Continuous"],
            "relationships": ["Linear"],
            "purpose": ["Comprehensive exploration"]
        },
        "Optimal Design": {
            "description": "Large number of factors.",
            "detailed_info": "An optimal design is an experimental design that maximizes the amount of information obtained from the experiment while minimizing resources like time or costs. It is tailored to specific goals, such as estimating model parameters with high precision or detecting significant effects efficiently. Unlike standard designs, optimal designs are generated algorithmically based on the specific model and constraints of the experiment, making them flexible and adaptable to different experimental conditions. Common types include D-optimal, A-optimal, and G-optimal designs. These designs are particularly useful when the number of factors is large, and when interactions between factors are not expected to be important.",
            "num_factors": ["Many factors (6+)"],
            "factor_types": ["Continuous", "Categorical (multi-level)"],
            "relationships": ["Linear", "Non-linear/Quadratic"],
            "purpose": ["Optimization"]
        },
        "Plackett-Burman Design": {
            "description": "Screening a large number of 2-level factors.",
            "detailed_info": "A Plackett-Burman design is a type of experimental design used for screening a large number of factors to identify the most influential ones with a minimal number of experiments. It is an efficient two-level design (with high and low settings for each factor) that focuses on main effects, ignoring interactions between factors. This makes it particularly useful in the early stages of experimentation when the goal is to quickly determine which factors significantly affect the outcome.",
            "num_factors": ["Many factors (6+)"],
            "factor_types": ["Continuous", "Categorical (multi-level)"],
            "relationships": ["Linear"],
            "purpose": ["Screening (identify important factors)"]
        },
        "Box-Behnken Design": {
            "description": "Three or more factors, non-linear dependency.",
            "detailed_info": "A Box-Behnken design is a response surface methodology used for optimization. It's designed to explore the relationships between multiple factors and their interactions. This design only includes combinations where all factors are at their midpoints, highs, or lows, but it avoids extreme combinations, making it more efficient and reducing the risk of extreme conditions that could yield invalid results. It's commonly used when experimenting with three or more factors and is effective for developing quadratic models without requiring a full factorial design.",
            "num_factors": ["Few factors (2-5)"],
            "factor_types": ["Continuous"],
            "relationships": ["Non-linear/Quadratic", "Interaction effects"],
            "purpose": ["Optimization"]
        },
        "Central Composite Design": {
            "description": "Quadratic models.",
            "detailed_info": "A Central Composite Design (CCD) is an advanced design of experiments method used to build quadratic models for response surface methodology. It's useful for optimizing processes by exploring the relationships between factors and their responses. CCD combines a factorial design with center points and additional 'star' points to allow for better estimation of curvature. This design enables efficient estimation of linear, interaction, and quadratic effects, making it ideal for fine-tuning processes or conditions in experiments.",
            "num_factors": ["Few factors (2-5)"],
            "factor_types": ["Continuous"],
            "relationships": ["Non-linear/Quadratic", "Interaction effects"],
            "purpose": ["Optimization"]
        }
    }

    # Apply filters
    filtered_methods = {}

    for method, properties in doe_methods.items():
        # Check if the method passes all filters
        passes_filters = True
        
        # Filter by number of factors
        if num_factors != "Any" and num_factors not in properties["num_factors"]:
            passes_filters = False
        
        # Filter by factor types
        if not any(ft in properties["factor_types"] for ft in factor_types):
            passes_filters = False
            
        # Filter by relationships
        if not any(rel in properties["relationships"] for rel in relationships):
            passes_filters = False
            
        # Filter by purpose
        if purpose != "Any" and purpose not in properties["purpose"]:
            passes_filters = False
        
        # If the method passes all filters, add it to the filtered methods
        if passes_filters:
            filtered_methods[method] = properties

    # Display number of matching designs
    st.write(f"#### Matching designs: :blue[{len(filtered_methods)}]")

    # Display filtered methods in expanders
    if filtered_methods:
        cols = st.columns(2)
        
        for i, (method, properties) in enumerate(filtered_methods.items()):
            col_idx = i % 2
            
            with cols[col_idx].expander(f":blue[**{method}**]\n\n{properties['description']}"):
                st.write(properties["detailed_info"])
                
                # Display properties as badges
                st.write("---")
                st.write("**Characteristics:**")
                
                badge_cols = st.columns(2)
                
                with badge_cols[0]:
                    st.info(f"**Factor count:** {', '.join(properties['num_factors'])}")
                    st.info(f"**Factor types:** {', '.join(properties['factor_types'])}")
                
                with badge_cols[1]:
                    st.info(f"**Relationships:** {', '.join(properties['relationships'])}")
                    st.info(f"**Purpose:** {', '.join(properties['purpose'])}")
    else:
        st.warning("No design methods match your selected filters. Try broadening your criteria.")

    



with tab2:
    for par in range(Npars):
        cols = st.columns(5)
        parameters[par]['name'] = cols[0].text_input(f"**Parameter {par+1}:**", 
                                            key=f"par{1+par}", 
                                            value=defaultParNames[par]).replace(" ","_")
        parameters[par]['type'] = cols[1].selectbox("Parameter Type", 
                                ("Float", "Integer", "Categorical"), key=f"cat{1+par}")
        if parameters[par]['type'] != "Categorical":
            low = cols[2].number_input("Low", value=-1., key=f"low{1+par}")
            high = cols[3].number_input("High", value=1., key=f"high{1+par}")
            other = cols[4].text_input("""Intermediate values""", value="", key=f"other{1+par}", help="**Optional:** specify intermediate values (comma separated) between `Low` and `High` (only useful for Factorial designs) or Box-Behnken design (for which at least one intermediate value is required).")
            other = [float(item) for item in other.split(",") if item]
            parameters[par]['values'] = np.array([low] + other + [high])
        else:
            other = cols[2].text_input("""Values (comma separated)""", value="A,B", key=f"other{1+par}")
            parameters[par]['values'] = other.split(",")


if design_type in ['Sobol sequence', 'Space Filling Latin Hypercube', 
                   'Randomized Latin Hypercube', 'Optimal']:
    mymin = 1 if design_type!='Optimal' else minexp
    Nexp = st.sidebar.number_input("Number of experiments:",
                                    min_value=mymin, max_value=1000, value = 10)
    rseed = st.sidebar.number_input("Random seed (for reproducibility):", 
                                    min_value=0, value=42)
    feature_constraints = st.sidebar.text_input("""Add **linear** constraints on the features (if any). Use a comma to separate multiple constraints.""",
                help="""The constraints should be in the form of inequalities such as:

- `x1 >= 0`
- `x2 <= 10, x4 >= -0.5`
- `x1 + 3*x2 <= 5`""")
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
    # check that the constraints are valid
    for i,constraint in enumerate(feature_constraints):
        # simple check: each constraint should contain at 
        # least one parameter name that is among the defined parameters
        # and not contain invalid parameter names
        constraint = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', constraint)
        invalid_consts = []
        for const in constraint:
            if const not in [parameters[par]['name'] for par in range(Npars)]:
                st.error(f"Constraint '{const}' is invalid: no matching parameter name found. The constraint will be ignored.")
                invalid_consts.append(i)
        if len(invalid_consts)>0:
            feature_constraints = [c for j,c in enumerate(feature_constraints) if j not in invalid_consts]
elif design_type=='Fractional Factorial':
    reduction = st.sidebar.number_input("Reduction:", min_value=2, max_value=Npars+1, value=2)
elif design_type=='Central Composite':
    center = st.sidebar.slider("Number of replications of the center point:",
            value=1, min_value=0, max_value=10, step=1)
    center = (0, center)
    face = st.sidebar.selectbox("Type of design:", ['Circumscribed', 'Inscribed', 'Faced'])
    if face == 'Circumscribed':
        face = 'ccc'
    elif face == 'Inscribed':
        face = 'cci'
    elif face == 'Faced':
        face = 'ccf'
    # alpha = st.sidebar.selectbox("Alpha", ['orthogonal', 'rotatable'])
    # if alpha == 'orthogonal':
    #     alpha = 'o'
    # elif alpha == 'rotatable':
    #     alpha = 'r'
    # change parameter values to list
    for par in range(Npars):
        if not isinstance(parameters[par]['values'], list):
            parameters[par]['values'] = parameters[par]['values'].tolist()

doe = DesignOfExperiments(
    type                = design_type,
    parameters          = parameters,
    Nexp                = Nexp,
    reduction           = reduction,
    order               = order,
    randomize           = randomize,
    feature_constraints = feature_constraints,
    center              = center,
    alpha               = alpha,
    face                = face,
    seed                = rseed
)
design = doe.design

# # # # # # # # # # # # # # # # 

with tab3:
    # add an empty "response" column to the design
    design['response'] = ''
    timestamp = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    outfile = writeout(design)
    cols= st.columns([4,1,1,7])
    cols[0].write(f"Download Experimental Design with {len(design)} runs:")
    cols[1].download_button(
        label     = f"CSV",
        data      = outfile,
        file_name = f'DOE_{timestamp}.csv',
        mime      = 'text/csv',
        key       = 'download-csv'
    )
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df = clean_names(design)
        df.to_excel(writer, sheet_name="Sheet1", index=False)
        writer.close()
        cols[2].download_button(
            label=f"XLSX",
            data=buffer,
            file_name=f'DOE_{timestamp}.xlsx',
            mime="application/vnd.ms-excel",
        )
    st.dataframe(design, hide_index=True)
    # plot the design
    Npars = len(parameters)
    figs = doe.plot()
    cols = st.columns(1 if Npars <= 3 else 2)
    count = 0
    if len(design.values) > 0:
        for i in range(len(figs)):
            cols[i%2].plotly_chart(figs[i], use_container_width=True)