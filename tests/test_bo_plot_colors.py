from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from optimeo.bo import BOExperiment, read_experimental_data
from test_bo_plot_axes import PAGE, ROOT, PlotStub, make_experiment


def color_experiment(model):
    experiment = make_experiment(model=model)
    experiment.names = ["x", "y", "small", "large"]
    experiment._features = {
        "x": {"type": "float", "range": [0, 1]},
        "y": {"type": "float", "range": [0, 1]},
        "small": {"type": "float", "range": [0, 1]},
        "large": {"type": "float", "range": [0, 1000]},
    }
    data = pd.DataFrame({
        "x": [0.1, 0.2, 0.3, 0.4],
        "y": [0.2, 0.3, 0.4, 0.5],
        "small": [0.6, 0.4, 0.7, 0.5],
        "large": [400, 600, 700, 500],
        "result": [1, 2, 3, 4],
        "trial_status": ["COMPLETED"] * 4,
        "arm_name": ["0_0", "1_0", "2_0", "3_0"],
    })
    experiment.ax_client = SimpleNamespace(
        objective_names=["result"], get_trials_data_frame=lambda: data.copy()
    )
    return experiment


@pytest.mark.parametrize("model", [False, True])
def test_color_modes_use_normalized_slices(monkeypatch, model):
    def contour(**kwargs):
        fig = go.Figure(go.Scatter(
            x=[0.1, 0.2, 0.3, 0.4], y=[0.2, 0.3, 0.4, 0.5],
            mode="markers", text=[f"Arm {i}_0" for i in range(4)],
        ))
        return SimpleNamespace(data=fig.to_dict())

    monkeypatch.setattr("optimeo.bo.plot_contour", contour)
    experiment = color_experiment(object() if model else None)
    args = {"param_x": "x", "param_y": "y", "slice_values": {"small": 0.5, "large": 500}}
    overall = experiment.plot_model(**args, color_mode="overall")
    single = experiment.plot_model(**args, color_mode="parameter", color_parameter="small")
    none = experiment.plot_model(**args, color_mode="none")
    overall_colors = list(overall.data[0].marker.color)
    single_colors = list(single.data[0].marker.color)
    none_colors = none.data[0].marker.color
    assert overall_colors[0] == overall_colors[1] == overall_colors[3]
    assert overall_colors[2] != overall_colors[3]
    assert single_colors[0] != single_colors[1]
    assert single_colors[3] == overall_colors[3]
    assert list(none_colors) == ["white"] * 4
    if not model:
        assert none.data[0].marker.line.color == "black"
        assert none.data[0].marker.line.width > 0


@pytest.mark.parametrize("mode,parameter", [
    ("invalid", None), ("parameter", None), ("parameter", "x"),
    ("overall", "small"),
])
def test_invalid_color_options(mode, parameter):
    with pytest.raises(ValueError):
        color_experiment(None).plot_model(
            slice_values={"small": 0.5}, color_mode=mode, color_parameter=parameter
        )


def test_overall_color_rejects_zero_range():
    experiment = color_experiment(None)
    experiment._features["small"]["range"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="small.*positive finite range"):
        experiment.plot_model(slice_values={"small": 0.5}, color_mode="overall")
    experiment.plot_model(slice_values={"small": 0.5}, color_mode="none")
    experiment.plot_model(
        slice_values={"small": 0.5}, color_mode="parameter", color_parameter="small"
    )


def test_real_contour_observations_and_candidates_are_colored():
    features, outcomes = read_experimental_data(
        str(ROOT / "tests/data/experimental_data2.csv"), out_pos=[-2, -1]
    )
    features["Third"] = {"type": "float", "data": [0.0, 0.3, 0.7, 1.0]}
    experiment = BOExperiment(features, {"Yield": outcomes["Yield"]})
    experiment.suggest_next_trials()
    fig = experiment.plot_model(
        slice_values={"Third": 0.5}, color_mode="parameter", color_parameter="Third"
    )
    observations = [trace for trace in fig.data if trace.type == "scatter"
                    and trace.legendgroup != "Candidate" and trace.text]
    assert observations
    colors = [color for trace in observations for color in trace.marker.color]
    assert any(color.startswith("rgb(0, 0, 255)") for color in colors)
    assert any(color.startswith("rgb(255, 0, 0)") for color in colors)
    candidates = [trace for trace in fig.data if trace.legendgroup == "Candidate"]
    assert candidates and all(
        trace.marker.color == "orange"
        and trace.marker.symbol == "x"
        and trace.marker.line.color == "red"
        and trace.marker.line.width >= 1
        for trace in candidates
    )


def test_color_options_in_page(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "optimeo/app"))
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state.loaded_data = pd.DataFrame({
        "a": [1.0, 2.0], "b": [2.0, 3.0], "c": [3.0, 4.0],
        "d": [4.0, 5.0], "result": [4.0, 5.0],
    })
    app.session_state.data_filename = "colors.csv"
    plotter = PlotStub()
    app.session_state.bo = plotter
    app.session_state.model_up_to_date = True
    app.run()
    assert not plotter.calls
    assert any(r"D = \sum_{i \in S} \frac{x_i - s_i}{max_i - min_i}" in w.value for w in app.markdown)
    next(w for w in app.number_input if w.label == "Slice for **c**").set_value(3.0).run()
    next(w for w in app.number_input if w.label == "Slice for **d**").set_value(4.0).run()
    assert not plotter.calls
    next(b for b in app.button if b.label == "Plot model / Update plots").click().run()
    assert not app.exception
    mode = next(w for w in app.selectbox if w.label == "Color observed points")
    mode.select("Single parameter").run()
    parameter = next(w for w in app.selectbox if w.label == "Coloring parameter")
    parameter.select("d").run()
    assert not app.exception
    assert plotter.calls[-1]["color_mode"] == "parameter"
    assert plotter.calls[-1]["color_parameter"] == "d"
    next(w for w in app.selectbox if w.label == "Color observed points").select("No coloring").run()
    assert not app.exception
    assert plotter.calls[-1]["color_mode"] == "none"
    next(w for w in app.selectbox if w.label == "Color observed points").select("Overall slice").run()
    assert not app.exception
    assert plotter.calls[-1]["color_mode"] == "overall"
    num_calls = len(plotter.calls)
    next(w for w in app.number_input if w.label == "Slice for **c**").set_value(3.5).run()
    assert not app.exception
    assert len(plotter.calls) > num_calls
    assert plotter.calls[-1]["slice_values"] == {"c": 3.5, "d": 4.0}
    next(w for w in app.number_input if w.label == "Slice for **c**").set_value(None).run()
    next(w for w in app.number_input if w.label == "Slice for **d**").set_value(None).run()
    assert not app.exception
    assert not any(w.label == "Color observed points" for w in app.selectbox)
    assert plotter.calls[-1]["color_mode"] == "none"
    assert plotter.calls[-1]["slice_values"] == {}


def test_categorical_slice_updates_automatically(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "optimeo/app"))
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state.loaded_data = pd.DataFrame({
        "a": [1, 2], "b": [2, 3], "category": ["one", "two"], "result": [4.0, 5.0],
    })
    app.session_state.data_filename = "categories.csv"
    plotter = PlotStub()
    app.session_state.bo = plotter
    app.session_state.model_up_to_date = True
    app.run()
    next(b for b in app.button if b.label == "Plot model / Update plots").click().run()
    calls = len(plotter.calls)
    next(w for w in app.multiselect if w.label == "Slice for **category**").select("one").run()
    assert not app.exception
    assert len(plotter.calls) > calls
    assert plotter.calls[-1]["slice_values"] == {"category": "one"}
    assert not any(w.label == "Color observed points" for w in app.selectbox)
    assert plotter.calls[-1]["color_mode"] == "none"
    next(w for w in app.number_input if w.label == "Slice for **a**").set_value(1).run()
    next(w for w in app.number_input if w.label == "Slice for **b**").set_value(2).run()
    assert not app.exception
    assert any("no free features" in w.value for w in app.warning)
    calls = len(plotter.calls)
    next(w for w in app.number_input if w.label == "Slice for **a**").set_value(None).run()
    assert not app.exception
    assert len(plotter.calls) > calls
    assert plotter.calls[-1]["slice_values"] == {"b": 2, "category": "one"}
