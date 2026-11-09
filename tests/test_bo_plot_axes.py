from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.graph_objects as go
import pytest
from streamlit.testing.v1 import AppTest

from optimeo.bo import BOExperiment, read_experimental_data


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "optimeo/app/pages/2_2._Bayesian_Optimization.py"


def make_experiment(model=None):
    experiment = BOExperiment.__new__(BOExperiment)
    experiment.names = ["a", "b", "c"]
    experiment._features = {name: {"type": "float", "range": [0.0, 10.0]} for name in experiment.names}
    experiment._N = 1
    experiment.model = model
    experiment.generator_run = None
    data = pd.DataFrame({
        "a": [1.0], "b": [2.0], "c": [3.0], "result": [4.0],
        "trial_status": ["COMPLETED"], "arm_name": ["0_0"],
    })
    experiment.ax_client = SimpleNamespace(
        objective_names=["result"], get_trials_data_frame=lambda: data.copy()
    )
    return experiment


def test_selected_axes_used_for_observations():
    fig = make_experiment().plot_model(param_x="c", param_y="a")
    assert fig.layout.xaxis.title.text == "c"
    assert fig.layout.yaxis.title.text == "a"
    assert list(fig.data[0].x) == [3.0]
    assert list(fig.data[0].y) == [1.0]


def test_selected_axes_used_for_model(monkeypatch):
    calls = []

    def contour(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=go.Figure().to_dict())

    monkeypatch.setattr("optimeo.bo.plot_contour", contour)
    experiment = make_experiment(model=object())
    fig = experiment.plot_model(param_x="c", param_y="a", slice_values={"b": 2.0})
    assert calls[0]["param_x"] == "c"
    assert calls[0]["param_y"] == "a"
    assert calls[0]["slice_values"] == {"b": 2.0}
    assert fig.layout.xaxis.title.text == "c"
    assert fig.layout.yaxis.title.text == "a"


def test_categorical_slice_does_not_break_model_plot(monkeypatch):
    monkeypatch.setattr(
        "optimeo.bo.plot_contour",
        lambda **kwargs: SimpleNamespace(data=go.Figure().to_dict()),
    )
    experiment = make_experiment(model=object())
    experiment.names.append("category")
    experiment._features["category"] = {"type": "text"}
    data = experiment.ax_client.get_trials_data_frame()
    data["category"] = "first"
    experiment.ax_client.get_trials_data_frame = lambda: data.copy()
    fig = experiment.plot_model(
        param_x="c", param_y="a", slice_values={"category": "first"}
    )
    assert fig.layout.xaxis.title.text == "c"


def test_real_model_contour_uses_selected_axes():
    features, outcomes = read_experimental_data(
        str(ROOT / "tests/data/experimental_data2.csv"), out_pos=[-2, -1]
    )
    features["Third"] = {"type": "float", "data": [0.0, 0.3, 0.7, 1.0]}
    experiment = BOExperiment(features, {"Yield": outcomes["Yield"]})
    experiment.suggest_next_trials()
    assert experiment.model is not None
    fig = experiment.plot_model(param_x="Third", param_y="Temperature")
    contour = next(trace for trace in fig.data if trace.type == "contour")
    assert min(contour.x) == pytest.approx(0.0)
    assert max(contour.x) == pytest.approx(1.0)
    assert min(contour.y) == pytest.approx(0.0)
    assert max(contour.y) == pytest.approx(100.0)
    assert fig.layout.xaxis.title.text == "Third"
    assert fig.layout.yaxis.title.text == "Temperature"


@pytest.mark.parametrize("x,y,slices", [
    ("a", "a", {}),
    ("a", "missing", {}),
    ("a", None, {}),
    ("a", "b", {"b": 2.0}),
])
def test_invalid_axes_rejected(x, y, slices):
    with pytest.raises(ValueError):
        make_experiment().plot_model(param_x=x, param_y=y, slice_values=slices)


class PlotStub:
    model = None

    def __init__(self):
        self.calls = []

    def plot_model(self, **kwargs):
        self.calls.append(kwargs)
        return go.Figure()

    def plot_optimization_trace(self):
        return go.Figure()

    def plot_feature_importances(self, **kwargs):
        return None


def test_page_axis_selectors_update_plots(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "optimeo/app"))
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state.loaded_data = pd.DataFrame({
        "a": [1.0, 2.0], "b": [2.0, 3.0], "c": [3.0, 4.0],
        "d": [4.0, 5.0], "result": [4.0, 5.0]
    })
    app.session_state.data_filename = "axes.csv"
    plotter = PlotStub()
    app.session_state.bo = plotter
    app.session_state.model_up_to_date = True
    app.run()
    assert not app.exception
    next(button for button in app.button if button.label == "Plot model / Update plots").click().run()
    assert not app.exception
    x = next(widget for widget in app.selectbox if widget.label == "X-axis parameter")
    y = next(widget for widget in app.selectbox if widget.label == "Y-axis parameter")
    assert (x.value, y.value) == ("a", "b")
    x.select("c").run()
    assert not app.exception
    y = next(widget for widget in app.selectbox if widget.label == "Y-axis parameter")
    y.select("a").run()
    assert not app.exception
    assert plotter.calls[-1]["param_x"] == "c"
    assert plotter.calls[-1]["param_y"] == "a"
    x = next(widget for widget in app.selectbox if widget.label == "X-axis parameter")
    x.select("a").run()
    assert not app.exception
    y = next(widget for widget in app.selectbox if widget.label == "Y-axis parameter")
    assert y.value != "a"
    next(widget for widget in app.number_input if widget.label == "Slice for **d**").set_value(4.0).run()
    assert not app.exception
    x = next(widget for widget in app.selectbox if widget.label == "X-axis parameter")
    y = next(widget for widget in app.selectbox if widget.label == "Y-axis parameter")
    assert "d" not in x.options
    assert "d" not in y.options
    next(widget for widget in app.number_input if widget.label == "Slice for **b**").set_value(2.0).run()
    assert not app.exception
    assert not any(widget.label == "X-axis parameter" for widget in app.selectbox)
    assert plotter.calls[-1]["slice_values"] == {"b": 2.0, "d": 4.0}
    assert "param_x" not in plotter.calls[-1]


def test_generation_fixed_parameter_remains_plot_option(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "optimeo/app"))
    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state.loaded_data = pd.DataFrame({
        "a": [1.0, 2.0], "b": [2.0, 3.0], "c": [3.0, 4.0], "result": [4.0, 5.0]
    })
    app.session_state.data_filename = "fixed.csv"
    app.run()
    next(widget for widget in app.multiselect if widget.label.startswith("Select the fixed parameters")).select("a").run()
    assert not app.exception
    app.session_state.bo = PlotStub()
    app.session_state.model_up_to_date = True
    app.run()
    next(button for button in app.button if button.label == "Plot model / Update plots").click().run()
    assert not app.exception
    x = next(widget for widget in app.selectbox if widget.label == "X-axis parameter")
    assert "a" in x.options
