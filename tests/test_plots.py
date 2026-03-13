
import glob
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from optuna.visualization._terminator_improvement import _ImprovementInfo
from shap.plots._force import AdditiveForceVisualizer
from sklearn.metrics import f1_score, get_scorer, mean_squared_error

from experionml import ExperionMLClassifier, ExperionMLForecaster, ExperionMLRegressor
from experionml.plots.baseplot import Aesthetics, BaseFigure
from experionml.utils.types import Legend
from experionml.utils.utils import NotFittedError

from .conftest import (
    X10, X10_str, X_bin, X_class, X_ex, X_label, X_reg, X_sparse, X_text,
    bin_groups, y10, y_bin, y_class, y_ex, y_fc, y_label, y_multiclass, y_reg,
)


# Test BaseFigure ================================================== >>

def test_get_elem():
    """Assert that elements are assigned correctly."""
    base = BaseFigure()
    assert base.get_elem() == "rgb(95, 70, 144)"
    assert base.get_elem("x") == "rgb(95, 70, 144)"
    assert base.get_elem("x") == "rgb(95, 70, 144)"


# Test BasePlot ==================================================== >>

def test_aesthetics():
    """Assert that the aesthetics getter works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(experionml.aesthetics, Aesthetics)
    assert isinstance(experionml.palette, list)
    assert isinstance(experionml.title_fontsize, int)
    assert isinstance(experionml.label_fontsize, int)
    assert isinstance(experionml.tick_fontsize, int)
    assert isinstance(experionml.line_width, int)
    assert isinstance(experionml.marker_size, int)


def test_aesthetics_setter():
    """Assert that the aesthetics setter works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.aesthetics = {"line_width": 3}
    assert experionml.line_width == 3


def test_palette_setter():
    """Assert that the palette setter works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.palette = ["red", "rgb(255, 34, 20)", "#0044ff"]
    fig = experionml.plot_distribution(columns=[0, 1], display=None)
    assert "rgb(255, 34, 20)" in str(fig._data_objs[2])


def test_palette_setter_invalid_name():
    """Assert that an error is raised when an invalid palette is used."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*the palette parameter.*"):
        experionml.palette = "unknown"


def test_get_plot_index():
    """Assert that indices can be converted to timestamps."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    assert isinstance(experionml._get_plot_index(experionml.dataset), pd.DatetimeIndex)

    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(experionml._get_plot_index(experionml.dataset), pd.RangeIndex)


def test_get_show():
    """Assert that the show returns max the number of features."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml._get_show(show=80, maximum=X_bin.shape[1]) == X_bin.shape[1]
    assert experionml._get_show(show=230) == 200
    assert experionml._get_show(show=5) == 5


def test_get_set():
    """Assert that the row selection is based on an iterator."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert next(experionml._get_set(rows="test")) == ("test", "test")
    assert next(experionml._get_set(rows=["train", "test"])) == ("train", "train")
    assert next(experionml._get_set(rows={"train": "test"})) == ("train", "test")


def test_get_metric():
    """Assert that metrics can be selected."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", metric=["f1", "recall"])
    assert experionml._get_metric(metric=None) == ["f1", "recall"]
    assert experionml._get_metric(metric=1) == ["recall"]
    assert experionml._get_metric(metric=[0, 1]) == ["f1", "recall"]
    assert experionml._get_metric(metric=["f1", "recall"]) == ["f1", "recall"]
    assert experionml._get_metric(metric="f1+recall") == ["f1", "recall"]
    assert experionml._get_metric(metric="time", max_one=True) == ["time"]


def test_get_metric_invalid_int():
    """Assert that an error is raised when the value is out of range."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", metric=["f1", "recall"])
    with pytest.raises(ValueError, match=".*out of range.*"):
        experionml._get_metric(metric=3, max_one=True)


def test_get_metric_invalid_name():
    """Assert that an error is raised for an invalid metric name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", metric=["f1", "recall"])
    with pytest.raises(ValueError, match=".*wasn't used to fit the models.*"):
        experionml._get_metric(metric="precision", max_one=True)


def test_get_metric_max_one():
    """Assert that an error is raised when multiple metrics are selected."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", metric=["f1", "recall"])
    with pytest.raises(ValueError, match=".*only accepts one metric.*"):
        experionml._get_metric(metric="f1+recall", max_one=True)


def test_get_plot_models_check_fitted():
    """Assert that an error is raised when the runner is not fitted."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(NotFittedError, match=".*not yet fitted.*"):
        experionml._get_plot_models(models=0)


def test_get_plot_models_max_one():
    """Assert that an error is raised when more than one model is selected."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "Tree"])
    with pytest.raises(ValueError, match=".*only accepts one model.*"):
        experionml._get_plot_models(models=None, max_one=True)


@patch("experionml.plots.baseplot.go.Figure.show")
def test_custom_title_and_legend(func):
    """Assert that title and legend can be customized."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    experionml.plot_roc(title={"text": "test", "x": 0}, legend={"font_color": "red"})
    func.assert_called_once()


@pytest.mark.parametrize("legend", Legend.__args__)
def test_custom_legend_position(legend):
    """Assert that the legend position can be specified."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    experionml.plot_roc(legend=legend, display=False)


@patch("mlflow.tracking.MlflowClient.log_figure")
def test_figure_to_mlflow(mlflow):
    """Assert that the figure is logged to mlflow."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run(["Tree", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    experionml.log_plots = True
    experionml.plot_results(display=False)
    experionml.lgb.plot_shap_scatter(display=False)
    assert mlflow.call_count == 3


@patch("experionml.plots.baseplot.go.Figure.write_html")
def test_figure_is_saved_html(func):
    """Assert that the figure is saved as .html by default."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.plot_correlation(filename="auto", display=False)
    func.assert_called_with(Path("plot_correlation.html"))


@patch("experionml.plots.baseplot.go.Figure.write_image")
def test_figure_is_saved_png(func):
    """Assert that the figure is saved as .png if specified."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.plot_correlation(filename="corr.png", display=False)
    func.assert_called_with(Path("corr.png"))


@patch("experionml.plots.baseplot.plt.Figure.savefig")
def test_figure_is_saved_png_plt(func):
    """Assert that the figure is saved as .png if specified."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.scale()
    experionml.plot_pipeline(filename="pipeline", display=False)
    func.assert_called_with(Path("pipeline.png"))


def test_figure_is_returned():
    """Assert that the method returns the figure for display=None."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    fig = experionml.plot_correlation(display=None)
    assert fig.__class__.__name__ == "Figure"
    assert fig.__class__.__module__.startswith("plotly")

    fig = experionml.plot_shap_bar(display=None)
    assert fig.__class__.__name__ == "Figure"
    assert fig.__class__.__module__.startswith("matplotlib")


def test_canvas():
    """Assert that the canvas works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("Tree")
    with experionml.canvas(2, 2, sharex=True, sharey=True, title="Title", display=False) as fig:
        experionml.plot_residuals(title={"text": "Residuals plot", "x": 0})
        experionml.plot_feature_importance(title="Feature importance plot")
        experionml.plot_residuals()
        experionml.plot_residuals()
    assert fig.__class__.__name__ == "Figure"


def test_canvas_too_many_plots():
    """Assert that the canvas works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    with experionml.canvas(1, 2, display=False):
        experionml.plot_prc()
        experionml.plot_roc()
        with pytest.raises(ValueError, match=".*number of plots.*"):
            experionml.plot_prc()


@patch("experionml.plots.baseplot.go.Figure.write_html")
def test_figure_is_saved_canvas(func):
    """Assert that the figure is only saved after finishing the canvas."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    with experionml.canvas(1, 2, filename="canvas", display=False):
        experionml.plot_prc()
        func.assert_not_called()
        experionml.plot_roc()
        func.assert_not_called()
    func.assert_called_with(Path("canvas.html"))  # Only at the end it is saved


def test_reset_aesthetics():
    """Assert that the reset_aesthetics method set values to default."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.tick_fontsize = 30
    assert experionml.tick_fontsize == 30
    experionml.reset_aesthetics()
    assert experionml.tick_fontsize == 12


def test_update_layout():
    """Assert that the update_layout method set default layout values."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.update_layout(template="plotly-dark")
    experionml._custom_layout["template"] = "plotly-dark"


def test_update_traces():
    """Assert that the update_traces method set default trace values."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.update_traces(mode="lines+markers")
    experionml._custom_traces["mode"] = "lines+markers"
    experionml.reset_aesthetics()


# Test DataPlot ==================================================== >>

@pytest.mark.parametrize("columns", [None, -1])
def test_plot_acf(columns):
    """Assert that the plot_acf method works."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.plot_acf(columns=columns, display=False)


def test_plot_ccf():
    """Assert that the plot_ccf method works."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    with pytest.raises(ValueError, match=".*requires at least two columns.*"):
        experionml.plot_ccf(display=False)

    experionml = ExperionMLForecaster(X_ex, y=y_ex, random_state=1)
    experionml.plot_ccf(plot_interval=True, display=False)


@pytest.mark.parametrize("show", [10, None])
def test_plot_components(show):
    """Assert that the plot_components method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)

    # Didn't run PCA
    with pytest.raises(PermissionError, match=".*using the 'pca' strategy.*"):
        experionml.plot_components(display=False)

    experionml.feature_selection(strategy="pca", n_features=10)
    experionml.plot_components(show=show, display=False)


def test_plot_correlation():
    """Assert that the plot_correlation method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.plot_correlation(display=False)


def test_plot_data_splits():
    """Assert that the plot_data_splits method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, metadata=bin_groups, random_state=1)
    experionml.plot_data_splits(display=False)

    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.plot_data_splits(display=False)


@pytest.mark.parametrize("columns", [None, -1])
def test_plot_decomposition(columns):
    """Assert that the plot_decomposition method works."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.plot_decomposition(columns=columns, display=False)


def test_plot_distribution():
    """Assert that the plot_distribution method works."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    experionml.plot_distribution(columns=2, distributions=None, display=False)
    experionml.plot_distribution(columns="x0", distributions="kde", display=False)
    experionml.plot_distribution(columns=[0, 1], distributions="pearson3", display=False)


@pytest.mark.parametrize("columns", [None, -1])
def test_plot_fft(columns):
    """Assert that the plot_fft method works."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.plot_fft(columns=columns, display=False)


@pytest.mark.parametrize("ngram", [1, 2, 3, 4])
def test_plot_ngrams(ngram):
    """Assert that the plot_ngrams method works."""
    experionml = ExperionMLClassifier(X_text, y10, random_state=1)
    experionml.plot_ngrams(ngram=ngram, display=False)  # When the corpus is a str
    experionml.tokenize()
    experionml.plot_ngrams(ngram=ngram, display=False)  # When the corpus consists of tokens


@pytest.mark.parametrize("columns", [None, -1])
def test_plot_pacf(columns):
    """Assert that the plot_pacf method works."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.plot_pacf(columns=columns, display=False)


@pytest.mark.parametrize("X", [X10, X_sparse])
def test_plot_pca(X):
    """Assert that the plot_pca method works."""
    experionml = ExperionMLClassifier(X, y10, random_state=1)

    # Didn't run PCA
    with pytest.raises(PermissionError, match=".*using the 'pca' strategy.*"):
        experionml.plot_pca(display=False)

    experionml.feature_selection(strategy="pca", n_features=2)
    experionml.plot_pca(display=False)


@pytest.mark.parametrize("columns", [None, -1])
def test_plot_periodogram(columns):
    """Assert that the plot_periodogram method works."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.plot_periodogram(columns=columns, display=False)


def test_plot_qq():
    """Assert that the plot_qq method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.plot_qq(columns=[0, 1], distributions="pearson3", display=False)


def test_plot_relationships():
    """Assert that the plot_relationships method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.plot_relationships(display=False)


@pytest.mark.parametrize("scoring", [None, "auc"])
def test_plot_rfecv(scoring):
    """Assert that the plot_rfecv method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, n_rows=0.1, random_state=1)

    # Didn't run RFECV
    with pytest.raises(PermissionError, match=".*using the 'rfecv' strategy.*"):
        experionml.plot_rfecv(display=False)

    experionml.feature_selection("rfecv", solver="tree", n_features=20, scoring=scoring)
    experionml.plot_rfecv(display=False)


def test_plot_series():
    """Assert that the plot_series method works."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.plot_series(columns=None, display=False)
    experionml.plot_series(columns=-1, display=False)


def test_plot_wordcloud():
    """Assert that the plot_wordcloud method works."""
    experionml = ExperionMLClassifier(X_text, y10, random_state=1)
    experionml.plot_wordcloud(display=False)  # When the corpus is a str
    experionml.tokenize()
    experionml.plot_wordcloud(display=False)  # When the corpus consists of tokens


# Test HyperparameterTuningPlot ==================================== >>

def test_check_hyperparams():
    """Assert that an error is raised when models didn't run HT."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    with pytest.raises(PermissionError, match=".*models that ran hyperparameter.*"):
        experionml._check_hyperparams([experionml.tree])


def test_get_hyperparams():
    """Assert that hyperparameters can be retrieved."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", n_trials=3)
    assert len(experionml._get_hyperparams(params=None, model=experionml.tree)) == 7
    assert len(experionml._get_hyperparams(params=range(1, 4), model=experionml.tree)) == 3
    assert len(experionml._get_hyperparams(params=slice(1, 4), model=experionml.tree)) == 3
    assert len(experionml._get_hyperparams(params=[0, 1], model=experionml.tree)) == 2
    assert len(experionml._get_hyperparams(params=["criterion"], model=experionml.tree)) == 1
    assert len(experionml._get_hyperparams(params="criterion+splitter", model=experionml.tree)) == 2


def test_get_hyperparams_invalid_name():
    """Assert that an error is raised when a hyperparameter is invalid."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", n_trials=3)
    with pytest.raises(ValueError, match=".*value for the params parameter.*"):
        experionml._get_hyperparams(params="invalid", model=experionml.tree)


def test_get_hyperparams_empty():
    """Assert that an error is raised when no hyperparameters are selected."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", n_trials=3)
    with pytest.raises(ValueError, match=".*Didn't find any hyperparameters.*"):
        experionml._get_hyperparams(params=[], model=experionml.tree)


def test_plot_edf():
    """Assert that the plot_edf method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run(["lasso", "ridge"], n_trials=(3, 0))

    experionml.lasso.plot_edf(display=False)


def test_plot_hyperparameter_importance():
    """Assert that the plot_hyperparameter_importance method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("lasso", metric=["mse", "r2"], n_trials=3)
    experionml.plot_hyperparameter_importance(metric=1, display=False)


def test_plot_hyperparameters():
    """Assert that the plot_hyperparameters method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("lr", n_trials=3)

    # Only one hyperparameter
    with pytest.raises(ValueError, match=".*minimum of two parameters.*"):
        experionml.plot_hyperparameters(params=[0], display=False)

    experionml.plot_hyperparameters(params=(0, 1, 2), display=False)


def test_plot_parallel_coordinate():
    """Assert that the plot_parallel_coordinate method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("tree", n_trials=3)
    experionml.plot_parallel_coordinate(display=False)


def test_plot_pareto_front():
    """Assert that the plot_pareto_front method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("tree")

    # Not multi-metric
    with pytest.raises(PermissionError, match=".*models with multi-metric runs.*"):
        experionml.plot_pareto_front(display=False)

    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("tree", metric=["mae", "mse", "rmse"], n_trials=3)

    # Only one metric
    with pytest.raises(ValueError, match=".*minimum of two metrics.*"):
        experionml.plot_pareto_front(metric=[0], display=False)

    experionml.plot_pareto_front(display=False)


def test_plot_slice():
    """Assert that the plot_slice method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("lr", metric=["f1", "recall"], n_trials=3)
    experionml.plot_slice(display=False)


@patch("experionml.plots.hyperparametertuningplot._get_improvement_info")
def test_plot_terminator_improvements(improvement):
    """Assert that the plot_terminator_improvement method works."""
    improvement.return_value = _ImprovementInfo([], [], [])

    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run("tree", n_trials=1)

    # No cross-validation
    with pytest.raises(PermissionError, match=".*using cross-validation.*"):
        experionml.plot_terminator_improvement()

    experionml.run("tree", n_trials=1, ht_params={"cv": 2})
    experionml.plot_terminator_improvement(display=False)


def test_plot_timeline():
    """Assert that the plot_timeline method works."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run("tree", n_trials=1)
    experionml.plot_timeline(display=False)


def test_plot_trials():
    """Assert that the plot_bo method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("lasso", n_trials=3)
    experionml.plot_trials(display=False)


# Test PredictionPlot =================================================== >>

def test_plot_bootstrap():
    """Assert that the plot_bootstrap method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run(["OLS", "Tree", "Tree_2"], n_bootstrap=(3, 3, 0))

    with pytest.raises(ValueError, match=".*have bootstrap scores.*"):
        experionml.plot_bootstrap(models="Tree_2", display=False)

    experionml.plot_bootstrap(display=False)  # Mixed bootstrap
    experionml.plot_bootstrap(models=["OLS", "Tree"], display=False)  # All bootstrap


def test_plot_calibration():
    """Assert that the plot_calibration method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["Dummy", "Tree"], metric="f1")
    experionml.plot_calibration(display=False)


def test_plot_confusion_matrix():
    """Assert that the plot_confusion_matrix method works."""
    # For binary classification tasks
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["RF", "LGB"], est_params={"n_estimators": 5})
    experionml.plot_confusion_matrix(rows=[0, 1, 2], threshold=0.2, display=False)

    # For multiclass classification tasks
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run(["RF", "LGB"], est_params={"n_estimators": 5})

    # Not available for multiclass
    with pytest.raises(NotImplementedError, match=".*not support the comparison.*"):
        experionml.plot_confusion_matrix(display=False)

    experionml.lgb.plot_confusion_matrix(display=False)


def test_plot_cv_splits():
    """Assert that the plot_cv_splits method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, metadata=bin_groups, random_state=1)
    experionml.run("Dummy")
    with pytest.raises(ValueError, match=".*ran cross-validation.*"):
        experionml.plot_cv_splits(display=False)
    experionml.dummy.cross_validate(cv=2)
    experionml.plot_cv_splits(display=False)

    experionml = ExperionMLForecaster(y_fc, holdout_size=0.1, random_state=1)
    experionml.run("NF")
    experionml.nf.cross_validate(cv=2, include_holdout=True)
    experionml.plot_cv_splits(display=False)


def test_plot_det():
    """Assert that the plot_det method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LGB", "SVM"], est_params={"LGB": {"n_estimators": 5}})
    experionml.plot_det(display=False)


def test_plot_errors():
    """Assert that the plot_errors method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("Tree")
    experionml.plot_errors(display=False)


def test_plot_evals():
    """Assert that the plot_evals method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(
        models=["LR", "LGB", "MLP"],
        metric="f1",
        est_params={
            "LGB": {"n_estimators": 5},
            "MLP": {"hidden_layer_sizes": (5,), "max_iter": 5},
        },
    )

    # No in-training validation
    with pytest.raises(ValueError, match=".*no in-training validation.*"):
        experionml.lr.plot_evals(display=False)

    experionml.plot_evals(models=["LGB", "MLP"], display=False)


def test_plot_feature_importance():
    """Assert that the plot_feature_importance method works."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run(["KNN", "Tree", "Bag"])

    # Model has no feature importance values
    with pytest.raises(ValueError, match=".*has no scores_.*"):
        experionml.knn.plot_feature_importance(display=False)

    experionml.plot_feature_importance(models=["Tree", "Bag"], display=False)
    experionml.tree.plot_feature_importance(show=5, display=False)


def test_plot_forecast():
    """Assert that the plot_forecast method works."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.run(models="NF")

    # All values are in train set when in_sample=False
    with pytest.raises(ValueError, match=".*plot_insample parameter.*"):
        experionml.plot_forecast(fh=range(3), plot_insample=False, display=False)

    experionml.plot_forecast(inverse=False, display=False)

    experionml = ExperionMLForecaster(X_ex, y=(-2, -1), random_state=1)
    experionml.run(models=["NF", "ES"])
    experionml.plot_forecast(display=False)


def test_plot_gains():
    """Assert that the plot_gains method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    experionml.plot_gains(display=False)


def test_plot_learning_curve():
    """Assert that the plot_learning_curve method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.train_sizing(["Dummy", "Tree"], n_bootstrap=4)
    experionml.plot_learning_curve(display=False)


def test_plot_lift():
    """Assert that the plot_lift method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", metric="f1")
    experionml.plot_lift(display=False)


def test_plot_parshap():
    """Assert that the plot_parshap method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.balance("smote")  # To get samples over 500
    experionml.run(["Dummy", "Tree"])
    experionml.plot_parshap(display=False)  # With colorbar
    experionml.dummy.plot_parshap(display=False)  # Without colorbar


def test_plot_partial_dependence():
    """Assert that the plot_partial_dependence method works."""
    experionml = ExperionMLClassifier(X_label, y=y_label, random_state=1)
    experionml.run("Tree")
    with pytest.raises(PermissionError, match=".*not available for multilabel.*"):
        experionml.plot_partial_dependence(display=False)

    experionml = ExperionMLClassifier(X_bin, y_bin, n_jobs=-1, random_state=1)
    experionml.run(["KNN", "LGB"], est_params={"LGB": {"n_estimators": 5}})

    # Pair for multimodel
    with pytest.raises(ValueError, match=".*when plotting multiple models.*"):
        experionml.plot_partial_dependence(columns=2, pair=3, display=False)

    # Different features for multiple models
    experionml.branch = "b2"
    experionml.feature_selection(strategy="pca", n_features=5)
    experionml.run(["Tree"])
    with pytest.raises(ValueError, match=".*models use the same features.*"):
        experionml.plot_partial_dependence(columns=(0, 1), display=False)

    experionml = ExperionMLClassifier(X_class, y_class, n_jobs=-1, random_state=1)
    experionml.run(["Tree", "LDA"])
    experionml.plot_partial_dependence(columns=[0, 1], kind="average+individual", display=False)
    experionml.tree.plot_partial_dependence(columns=[0, 1], pair=2, display=False)


@patch("experionml.plots.predictionplot.permutation_importance")
def test_plot_permutation_importance(importances):
    """Assert that the plot_permutation_importance method works."""
    importances.return_value = {"importances": np.array(range(len(X_bin)))}

    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", metric="f1")
    experionml.plot_permutation_importance(display=False)


def test_plot_pipeline():
    """Assert that the plot_pipeline method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("KNN")
    experionml.plot_pipeline(display=False)  # No transformers

    # Invalid models
    with pytest.raises(ValueError, match=".*any model that matches.*"):
        experionml.plot_pipeline(models="invalid", display=False)

    # Called from a canvas
    with pytest.raises(PermissionError, match=".*a canvas.*"):  # noqa: PT012
        with experionml.canvas(2, 1, display=False):
            experionml.plot_results(display=False)
            experionml.plot_pipeline(display=False)

    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.scale()
    experionml.plot_pipeline(display=False)  # No model

    experionml.run("Tree", n_trials=2)
    experionml.tree.plot_pipeline(display=False)  # Only one branch

    experionml.branch = "b2"
    experionml.prune()
    experionml.run(["OLS", "EN"])
    experionml.voting(["OLS", "EN"])
    experionml.plot_pipeline(title="Pipeline plot", display=False)  # Multiple branches


def test_plot_prc():
    """Assert that the plot_prc method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    experionml.plot_prc(display=False)


def test_plot_probabilities():
    """Assert that the plot_probabilities method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["Tree", "SVM"])

    # Model has no predict_proba attribute
    with pytest.raises(PermissionError, match=".*with a predict_proba method.*"):
        experionml.svm.plot_probabilities(display=False)

    experionml.plot_probabilities("Tree", display=False)


def test_plot_probabilities_multioutput():
    """Assert that the plot_probabilities method works for multioutput tasks."""
    experionml = ExperionMLClassifier(X_label, y=y_label, random_state=1)
    experionml.run("LR")
    experionml.plot_probabilities(display=False)

    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    experionml.run("LR")
    experionml.plot_probabilities(display=False)


def test_plot_residuals():
    """Assert that the plot_residuals method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("Tree")
    experionml.plot_residuals(display=False)


def test_plot_results():
    """Assert that the plot_results method works."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("Tree")

    with pytest.raises(ValueError, match=".*can't be mixed with non-time metrics.*"):
        experionml.plot_results(metric="time+mae", display=False)

    experionml.plot_results(metric=None, display=False)
    experionml.plot_results(metric=0, display=False)
    experionml.plot_results(metric=mean_squared_error, display=False)
    experionml.plot_results(metric=["time_fit+time"], display=False)
    experionml.plot_results(metric=["mae", "mse"], display=False)


def test_plot_roc():
    """Assert that the plot_roc method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    experionml.run("Tree")
    experionml.plot_roc(display=False)


def test_plot_successive_halving():
    """Assert that the plot_successive_halving method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.successive_halving(
        models=["Bag", "RF", "LGB"],
        est_params={"n_estimators": 5},
        n_bootstrap=3,
    )
    experionml.plot_successive_halving(display=False)


@pytest.mark.parametrize("metric", [f1_score, get_scorer("f1"), "precision", "auc"])
def test_plot_threshold(metric):
    """Assert that the plot_threshold method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    experionml.plot_threshold(metric=metric, display=False)


def test_plot_threshold_multilabel():
    """Assert that the plot_threshold method works for multilabel tasks."""
    experionml = ExperionMLClassifier(X_label, y=y_label, random_state=1)
    experionml.run("Tree")
    experionml.plot_threshold(display=False)


# Test ShapPlot ==================================================== >>

def test_plot_shap_fail():
    """Assert that an error is raised when the explainer can't be created."""
    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    experionml.run("LDA")
    with pytest.raises(ValueError, match=".*Failed to get shap's explainer.*"):
        experionml.plot_shap_beeswarm(display=False)


def test_plot_shap_multioutput():
    """Assert that the shap plots work with multioutput tasks."""
    experionml = ExperionMLClassifier(X_label, y=y_label, random_state=1)
    experionml.run(["LR", "Tree"])
    experionml.lr.plot_shap_bar(display=False)  # Non-native multioutput
    experionml.tree.plot_shap_bar(display=False)  # Native multioutput


def test_plot_shap_bar():
    """Assert that the plot_shap_bar method works."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run(["LR", "Tree"], metric="f1_macro")
    experionml.lr.plot_shap_bar(display=False)


def test_plot_shap_beeswarm():
    """Assert that the plot_shap_beeswarm method works."""
    experionml = ExperionMLClassifier(X_class, y_class, n_rows=0.1, random_state=1)
    experionml.run("GNB", metric="f1_macro")
    experionml.plot_shap_beeswarm(display=False)


def test_plot_shap_decision():
    """Assert that the plot_shap_decision method works."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run("LR", metric="f1_macro")
    experionml.lr.plot_shap_decision(display=False)


@patch("shap.force_plot")
def test_plot_shap_force(plot):
    """Assert that the plot_shap_force method works."""
    plot.return_value = Mock(spec=AdditiveForceVisualizer)
    plot.return_value.html.return_value = ""

    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run(
        models=["LR", "MLP"],
        metric="MSE",
        est_params={"MLP": {"hidden_layer_sizes": (5,), "max_iter": 5}},
    )

    # Expected value from Explainer
    experionml.lr.plot_shap_force(rows=100, matplotlib=True, display=False)

    # Own calculation of expected value
    experionml.mlp.plot_shap_force(rows=100, matplotlib=True, display=False)

    experionml.lr.plot_shap_force(matplotlib=False, filename="force", display=True)
    assert glob.glob("force.html")


def test_plot_shap_heatmap():
    """Assert that the plot_shap_heatmap method works."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run("LR", metric="f1_macro")
    experionml.plot_shap_heatmap(display=False)


def test_plot_shap_scatter():
    """Assert that the plot_shap_scatter method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")

    with pytest.raises(ValueError, match=".*at most one feature.*"):
        experionml.plot_shap_scatter(columns=(0, 1), display=False)

    experionml.plot_shap_scatter(display=False)


def test_plot_shap_waterfall():
    """Assert that the plot_shap_waterfall method works."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run("Tree")

    with pytest.raises(ValueError, match=".*plotting multiple samples.*"):
        experionml.plot_shap_waterfall(rows=(0, 1), display=False)

    experionml.plot_shap_waterfall(display=False)
