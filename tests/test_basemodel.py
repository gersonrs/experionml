
import glob
from importlib.util import find_spec
from unittest.mock import patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
import requests
from optuna.distributions import CategoricalDistribution, IntDistribution
from optuna.pruners import PatientPruner
from optuna.samplers import NSGAIISampler
from optuna.study import Study
from pandas.io.formats.style import Styler
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, mean_absolute_error, mean_absolute_percentage_error, r2_score,
    recall_score,
)
from sklearn.model_selection import FixedThresholdClassifier, KFold
from sklearn.multioutput import ClassifierChain
from sklearn.tree import DecisionTreeClassifier
from sktime.forecasting.base import ForecastingHorizon
from sktime.proba.normal import Normal

from experionml import ExperionMLClassifier, ExperionMLForecaster, ExperionMLModel, ExperionMLRegressor
from experionml.utils.utils import check_is_fitted, check_scaling

from .conftest import (
    X10_str, X_bin, X_class, X_ex, X_idx, X_label, X_reg, bin_groups,
    bin_sample_weight, y10, y10_str, y_bin, y_class, y_ex, y_fc, y_idx,
    y_label, y_multiclass, y_multireg, y_reg,
)


# Test magic methods ================================== >>

def test_scaler():
    """Assert that a scaler is made for models that need scaling."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LGB", "LDA"], est_params={"LGB": {"n_estimators": 5}})
    assert experionml.lgb.scaler
    assert not experionml.lda.scaler


def test_repr():
    """Assert that the __repr__ method works as intended."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LDA")
    assert str(experionml.lda).startswith("LinearDiscriminantAnalysis")


def test_dir():
    """Assert that __dir__ contains all the extra attributes."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("dummy")
    assert all(attr in dir(experionml.dummy) for attr in ("y", "age", "head"))


def test_getattr():
    """Assert that branch attributes can be called from a model."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.balance(strategy="smote")
    experionml.run("Tree")
    assert isinstance(experionml.tree.shape, tuple)
    assert isinstance(experionml.tree.alcohol, pd.Series)
    assert isinstance(experionml.tree.head(), pd.DataFrame)
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        print(experionml.tree.data)


def test_contains():
    """Assert that we can test if model contains a column."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run("Tree")
    assert "alcohol" in experionml.tree


def test_getitem():
    """Assert that the models are subscriptable."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run("Tree")
    assert_series_equal(experionml.tree["alcohol"], experionml["alcohol"])
    assert_series_equal(experionml.tree[0], experionml[0])
    assert isinstance(experionml.tree[["alcohol", "ash"]], pd.DataFrame)


# Test training ==================================================== >>

def test_est_params_invalid_param():
    """Assert that invalid parameters in est_params are caught."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(
        models=["LR", "LGB"],
        metric="AP",
        n_trials=1,
        est_params={"test": 220, "LGB": {"n_estimators": 5}},
    )
    assert experionml.models == "LGB"  # LGB passes since it accepts kwargs


def test_est_params_unknown_param_fit():
    """Assert that unknown parameters in est_params_fit are caught."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(RuntimeError, match=".*All models failed.*"):
        experionml.run(["LR", "LGB"], n_trials=1, est_params={"test_fit": 220})


def test_custom_distributions_by_name():
    """Assert that the parameters to tune can be set by name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR", n_trials=1, ht_params={"distributions": "max_iter"})
    assert list(experionml.lr.best_params) == ["max_iter"]


def test_custom_distributions_by_name_excluded():
    """Assert that the parameters to tune can be excluded by name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("CNB", n_trials=1, ht_params={"distributions": "!fit_prior"})
    assert list(experionml.cnb.best_params) == ["alpha", "norm"]


def test_custom_distributions_name_is_invalid():
    """Assert that an error is raised when an invalid parameter is provided."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*is not a predefined hyperparameter.*"):
        experionml.run(
            models="LR",
            n_trials=1,
            ht_params={"distributions": "invalid"},
            errors="raise",
        )


def test_custom_distributions_is_dist():
    """Assert that the custom distributions are for all models."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(
        models="LR",
        n_trials=1,
        ht_params={"distributions": {"max_iter": IntDistribution(10, 20)}},
    )
    assert list(experionml.lr.best_params) == ["max_iter"]


def test_custom_distributions_include_and_excluded():
    """Assert that an error is raised when parameters are included and excluded."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*either include or exclude.*"):
        experionml.run(
            models="LR",
            n_trials=1,
            ht_params={"distributions": ["!max_iter", "penalty"]},
            errors="raise",
        )


def test_custom_distributions_meta_estimators():
    """Assert that meta-estimators can be tuned normally."""
    experionml = ExperionMLClassifier(X_label, y=y_label, random_state=1)
    experionml.run(
        models=ExperionMLModel(
            estimator=ClassifierChain(LogisticRegression(), cv=2),
            native_multilabel=True,
        ),
        n_trials=1,
        ht_params={
            "distributions": {
                "order": CategoricalDistribution([(0, 1, 2, 3), (1, 0, 3, 2)]),
                "base_estimator__solver": CategoricalDistribution(["lbfgs", "newton-cg"]),
            },
        },
    )


def test_est_params_removed_from_ht():
    """Assert that params in est_params are dropped from the optimization."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LGB", n_trials=1, est_params={"n_estimators": 5})
    assert "n_estimators" not in experionml.lgb.trials


def test_hyperparameter_tuning_with_no_hyperparameters():
    """Assert that the optimization is skipped when there are no hyperparameters."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(models="BNB", n_trials=10, est_params={"alpha": 1.0, "fit_prior": True})
    assert not hasattr(experionml.bnb, "trials")


def test_multi_objective_optimization():
    """Assert that hyperparameter tuning works for multi-metric runs."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR", metric=["f1", "auc"], n_trials=1)
    assert experionml.lr.study.sampler.__class__ == NSGAIISampler


def test_hyperparameter_tuning_with_plot():
    """Assert that you can plot the hyperparameter tuning as it runs."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(models=["LDA", "lSVM", "SVM"], n_trials=10, ht_params={"plot": True})


def test_xgb_optimizes_score():
    """Assert that the XGB model optimizes the score."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(
        models="XGB",
        n_trials=10,
        est_params={"n_estimators": 10},
        ht_params={"pruner": PatientPruner(None, patience=1)},
    )
    assert experionml.xgb.trials["f1"].sum() > 0  # All scores are positive


@patch("optuna.study.Study.get_trials")
def test_empty_study(func):
    """Assert that the optimization is skipped when there are no completed trials."""
    func.return_value = []  # No successful trials

    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(models="tree", n_trials=1)
    assert not hasattr(experionml.tree, "study")


def test_ht_with_groups():
    """Assert that the hyperparameter tuning works with groups."""
    experionml = ExperionMLClassifier(X_bin, y_bin, metadata=bin_groups, random_state=1)
    experionml.run("lr", n_trials=1, ht_params={"cv": 1})
    assert hasattr(experionml.lr, "trials")

    experionml.run("lr_2", n_trials=1, ht_params={"cv": 2})
    assert hasattr(experionml.lr_2, "trials")

    experionml = ExperionMLClassifier(X_bin, y_bin, stratify=None, metadata=bin_groups, random_state=1)
    experionml.run("lr", n_trials=1, ht_params={"cv": 2}, errors="raise")
    assert hasattr(experionml.lr, "trials")


def test_ht_with_pipeline():
    """Assert that the hyperparameter tuning works with a transformer pipeline."""
    experionml = ExperionMLClassifier(X10_str, y10, stratify=None, random_state=1)
    experionml.encode()
    experionml.run("lr", n_trials=1, ht_params={"cv": 2})
    assert hasattr(experionml.lr, "trials")


def test_ht_with_multilabel():
    """Assert that the hyperparameter tuning works with multilabel tasks."""
    experionml = ExperionMLClassifier(X_label, y=y_label, random_state=1)
    experionml.run("SGD", n_trials=1, est_params={"max_iter": 5})
    experionml.multioutput = None
    experionml.run("MLP", n_trials=1, est_params={"max_iter": 5})


def test_ht_with_multioutput():
    """Assert that the hyperparameter tuning works with multioutput tasks."""
    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    experionml.run("SGD", n_trials=1, est_params={"max_iter": 5})

    experionml = ExperionMLForecaster(y=X_ex, random_state=1)
    experionml.run("OLS", n_trials=1)

    experionml = ExperionMLRegressor(X_reg, y=y_multireg, random_state=1)
    experionml.run("Tree", n_trials=1)


def test_ht_with_pruning():
    """Assert that trials can be pruned."""
    experionml = ExperionMLClassifier(X_bin, y=y_bin, random_state=1)
    experionml.run(
        models="SGD",
        n_trials=7,
        ht_params={
            "distributions": {"max_iter": IntDistribution(5, 15)},
            "pruner": PatientPruner(None, patience=3),
        },
    )
    assert "PRUNED" in experionml.sgd.trials["state"].unique()


def test_custom_cv():
    """Assert that trials with a custom cv work for both tasks."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("dummy", n_trials=1, ht_params={"cv": KFold(n_splits=3)})


def test_cv_larger_one_forecast():
    """Assert that cv can be set to larger than one for forecast tasks."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.run("NF", n_trials=1, ht_params={"cv": 2})


def test_skip_duplicate_calls():
    """Assert that trials with the same parameters skip the calculation."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("dummy", n_trials=5)
    assert experionml.dummy.trials["f1"].nunique() < len(experionml.dummy.trials["f1"])


def test_trials_stored_correctly():
    """Assert that the `trials` attribute has the same params as the trial object."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("lr", n_trials=3, ht_params={"distributions": ["penalty", "C"]})
    assert experionml.lr.trials.loc[2, "penalty"] == experionml.lr.study.trials[2].params["penalty"]
    assert experionml.lr.trials.loc[2, "C"] == experionml.lr.study.trials[2].params["C"]


@patch("mlflow.log_params")
def test_nested_runs_to_mlflow(mlflow):
    """Assert that the trials are logged to mlflow as nested runs."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.log_ht = True
    experionml.run("Tree", n_trials=1)
    assert mlflow.call_count == 2  # n_trials + fit


@patch("mlflow.log_params")
def test_run_log_params_to_mlflow(mlflow):
    """Assert that model parameters are logged to mlflow."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("GNB")
    assert mlflow.call_count == 1


@patch("mlflow.log_metric")
def test_run_log_evals_to_mlflow(mlflow):
    """Assert that eval metrics are logged to mlflow."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("LGB")
    assert mlflow.call_count > 10


@patch("mlflow.sklearn.log_model")
def test_run_log_models_to_mlflow(mlflow):
    """Assert that models are logged to mlflow."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("LGB")
    assert mlflow.call_count == 1


@patch("mlflow.log_input")
def test_run_log_data_to_mlflow(mlflow):
    """Assert that train and test sets are logged to mlflow."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.log_data = True
    experionml.run("GNB")
    assert mlflow.call_count == 2  # Train and test set


@patch("mlflow.sklearn.log_model")
def test_run_log_pipeline_to_mlflow(mlflow):
    """Assert that renaming also changes the mlflow run."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.log_pipeline = True
    experionml.run("GNB")
    assert mlflow.call_count == 2  # Model + Pipeline


def test_fit_with_sample_weight():
    """Assert that sample weights are passed to the estimator and scorer."""
    experionml = ExperionMLClassifier(X_bin, y=y_bin, metadata=bin_sample_weight, random_state=1)
    experionml.run("SGD", est_params={"max_iter": 5})

    routing = experionml.sgd.estimator.get_metadata_routing()._serialize()
    assert routing["fit"]["sample_weight"]
    assert routing["partial_fit"]["sample_weight"]

    routing = experionml._metric[0].get_metadata_routing()._serialize()
    assert routing["score"]["sample_weight"]
    assert routing["score"]["sample_weight"]


def test_continued_hyperparameter_tuning():
    """Assert that the hyperparameter_tuning method can be recalled."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    assert not hasattr(experionml.tree, "trials")
    experionml.tree.hyperparameter_tuning(3)
    assert len(experionml.tree.trials) == 3
    experionml.tree.hyperparameter_tuning(3)
    assert len(experionml.tree.trials) == 6
    experionml.tree.hyperparameter_tuning(2, reset=True)
    assert len(experionml.tree.trials) == 2


def test_continued_bootstrapping():
    """Assert that the bootstrapping method can be recalled."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LGB", est_params={"n_estimators": 5})
    assert not hasattr(experionml.lgb, "bootstrap")
    experionml.lgb.bootstrapping(3)
    assert len(experionml.lgb.bootstrap) == 3
    experionml.lgb.bootstrapping(3)
    assert len(experionml.lgb.bootstrap) == 6
    experionml.lgb.bootstrapping(3, reset=True)
    assert len(experionml.lgb.bootstrap) == 3


# Test utility properties ========================================== >>

def test_name_property():
    """Assert that the name property can be set."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree_2")
    assert experionml.tree_2.name == "Tree_2"
    experionml.tree_2.name = ""
    assert experionml.tree.name == "Tree"
    experionml.tree.name = "Tree_3"
    assert experionml.tree_3.name == "Tree_3"
    experionml.tree_3.name = "4"
    assert experionml.tree_4.name == "Tree_4"


@patch("mlflow.MlflowClient.set_tag")
def test_name_property_to_mlflow(mlflow):
    """Assert that the new name is stored in mlflow."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("Tree")
    experionml.tree.name = "2"
    mlflow.assert_called_with(experionml.tree_2._run.info.run_id, "mlflow.runName", "Tree_2")


def test_og_property():
    """Assert that the og property returns the original Branch."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    assert experionml.tree.og is experionml.og


def test_branch_property():
    """Assert that the branch property returns the Branch."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    assert experionml.tree.branch is experionml.branch


def test_run_property():
    """Assert that the run property returns the mlflow run."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("Tree")
    assert hasattr(experionml.tree, "run")


def test_study_property():
    """Assert that the study property returns optuna's study."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", n_trials=0)
    assert not hasattr(experionml.tree, "study")
    experionml.run("Tree", n_trials=1)
    assert isinstance(experionml.tree.study, Study)


def test_trials_property():
    """Assert that the trials property returns an overview of the trials."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", n_trials=0)
    assert not hasattr(experionml.tree, "trials")
    experionml.run("Tree", n_trials=1)
    assert isinstance(experionml.tree.trials, pd.DataFrame)


def test_best_trial_property():
    """Assert that the best_trial property can be set."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", n_trials=5)
    assert experionml.tree.best_trial.number != 4
    experionml.tree.best_trial = 4
    assert experionml.tree.best_trial.number == 4
    experionml.tree.best_trial = None
    assert experionml.tree.best_trial.number == 2


def test_best_trial_property_invalid():
    """Assert that an error is raised when best_trial is invalid."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", n_trials=5)
    with pytest.raises(ValueError, match=".*should be a trial number.*"):
        experionml.tree.best_trial = 22


def test_best_params_property():
    """Assert that the best_params property returns the hyperparameters."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree", n_trials=0)
    assert experionml.tree.best_params == {}
    experionml.run("Tree", n_trials=5)
    assert isinstance(experionml.tree.best_params, dict)


def test_estimator_property():
    """Assert that the estimator property returns the estimator."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    assert isinstance(experionml.tree.estimator, DecisionTreeClassifier)
    assert check_is_fitted(experionml.tree.estimator)


def test_evals_property():
    """Assert that the estimator property returns the estimator."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LGB", est_params={"n_estimators": 5})
    assert len(experionml.lgb.evals) == 2


def test_bootstrap_property():
    """Assert that the bootstrap property returns the bootstrap results."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    assert not hasattr(experionml.tree, "bootstrap")
    experionml.run("Tree", n_bootstrap=3)
    assert len(experionml.tree.bootstrap) == 3


def test_feature_importance_property():
    """Assert that the feature_importance property works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    assert len(experionml.tree.feature_importance) == X_bin.shape[1]

    experionml = ExperionMLClassifier(X_label, y=y_label, random_state=1)
    experionml.run("LDA")
    assert len(experionml.lda.feature_importance) == X_label.shape[1]

    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    experionml.run("LDA")
    assert len(experionml.lda.feature_importance) == X_class.shape[1]


def test_results_property():
    """Assert that the property returns an overview of the model's results."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    assert isinstance(experionml.tree.results, pd.Series)


# Test data properties ============================================= >>

def test_pipeline_property():
    """Assert that the pipeline property returns the scaler as well."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.clean()
    experionml.run(["LR", "Tree"])
    assert len(experionml.lr.pipeline) == 2
    assert len(experionml.tree.pipeline) == 1


def test_dataset_property():
    """Assert that the dataset property returns scaled data if needed."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    assert_frame_equal(experionml.dataset, experionml.mnb.dataset)
    assert check_scaling(experionml.lr.dataset)


def test_train_property():
    """Assert that the train property returns scaled data if needed."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    assert_frame_equal(experionml.train, experionml.mnb.train)
    assert check_scaling(experionml.lr.train)


def test_test_property():
    """Assert that the test property returns scaled data if needed."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    assert_frame_equal(experionml.test, experionml.mnb.test)
    assert check_scaling(experionml.lr.test)


def test_holdout_property():
    """Assert that the holdout property is calculated."""
    experionml = ExperionMLClassifier(X10_str, y10, holdout_size=0.3, random_state=1)
    experionml.encode()
    experionml.run(["LR", "Tree"])
    assert not experionml.lr.holdout.equals(experionml.tree.holdout)  # Scaler vs no scaler
    assert len(experionml.lr.holdout.columns) > 3  # Holdout is transformed


def test_X_property():
    """Assert that the X property returns scaled data if needed."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    assert_frame_equal(experionml.X, experionml.mnb.X)
    assert check_scaling(experionml.lr.X)


def test_y_property():
    """Assert that the y property is returned unchanged."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    assert_series_equal(experionml.y, experionml.mnb.y)
    assert_series_equal(experionml.y, experionml.lr.y)


def test_X_train_property():
    """Assert that the X_train property returns scaled data if needed."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    assert_frame_equal(experionml.X_train, experionml.mnb.X_train)
    assert check_scaling(experionml.lr.X_train)


def test_X_test_property():
    """Assert that the X_test property returns scaled data if needed."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    assert_frame_equal(experionml.X_test, experionml.mnb.X_test)
    assert check_scaling(experionml.lr.X_test)


def test_X_holdout_property():
    """Assert that the X_holdout property is calculated."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    experionml.run("MNB")
    assert_frame_equal(experionml.mnb.X_holdout, experionml.mnb.holdout.iloc[:, :-1])


def test_y_train_property():
    """Assert that the y_train property is returned unchanged."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    assert_series_equal(experionml.y_train, experionml.mnb.y_train)
    assert_series_equal(experionml.y_train, experionml.lr.y_train)


def test_y_holdout_property():
    """Assert that the y_holdout property is calculated."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    experionml.run("MNB")
    assert_series_equal(experionml.mnb.y_holdout, experionml.mnb.holdout.iloc[:, -1])


def test_shape_property():
    """Assert that the shape property returns the shape of the dataset."""
    experionml = ExperionMLClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    experionml.run("MNB")
    assert experionml.mnb.shape == (len(X_bin), X_bin.shape[1] - 1)


def test_columns_property():
    """Assert that the columns property returns the columns of the dataset."""
    experionml = ExperionMLClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    experionml.run("MNB")
    assert len(experionml.mnb.columns) == len(experionml.columns) - 2


def test_n_columns_property():
    """Assert that the n_columns property returns the number of columns."""
    experionml = ExperionMLClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    experionml.run("MNB")
    assert experionml.mnb.n_columns == experionml.n_columns - 2


def test_features_property():
    """Assert that the features property returns the features of the dataset."""
    experionml = ExperionMLClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    experionml.run("MNB")
    assert len(experionml.mnb.features) == len(experionml.features) - 2


def test_n_features_property():
    """Assert that the n_features property returns the number of features."""
    experionml = ExperionMLClassifier(X_bin, y_bin, ignore=(0, 1), random_state=1)
    experionml.run("MNB")
    assert experionml.mnb.n_features == experionml.n_features - 2


def test_all_property():
    """Assert that the _all property returns the dataset + holdout."""
    experionml = ExperionMLRegressor(X_bin, y_bin, holdout_size=0.1, random_state=1)
    experionml.run("OLS")
    assert len(experionml.ols.dataset) != len(X_bin)
    assert len(experionml.ols._all) == len(X_bin)


# Test utility methods ============================================= >>

def test_calibrate_invalid_task():
    """Assert than an error is raised when task="regression"."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("OLS")
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        experionml.ols.calibrate()


def test_calibrate():
    """Assert that calibrate method works as intended."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("MNB")
    experionml.mnb.calibrate()
    assert isinstance(experionml.mnb.estimator, CalibratedClassifierCV)


def test_calibrate_train_on_test():
    """Assert that the calibrate method works when train_on_test=True."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("MNB")
    experionml.mnb.calibrate(train_on_test=True)
    assert isinstance(experionml.mnb.estimator, CalibratedClassifierCV)


def test_calibrate_new_mlflow_run():
    """Assert that calibrate creates a new mlflow run is created."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("GNB")
    run = experionml.gnb._run
    experionml.gnb.calibrate()
    assert experionml.gnb._run is not run


def test_set_threshold():
    """Assert that the set_threshold method works as intended."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("MNB")
    run = experionml.mnb._run

    with pytest.raises(ValueError, match=".*should lie between 0 and 1.*"):
        experionml.mnb.set_threshold(threshold=1.5)

    experionml.mnb.set_threshold(threshold=0.2)
    assert isinstance(experionml.mnb.estimator, FixedThresholdClassifier)
    assert experionml.mnb._run is not run


def test_clear():
    """Assert that the clear method resets the model's attributes."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    experionml.run("SGD", est_params={"max_iter": 5})
    experionml.plot_shap_beeswarm(display=False)
    experionml.evaluate(rows="holdout")
    assert experionml.sgd._evals
    assert not experionml.sgd._shap._shap_values.empty
    assert "holdout" in experionml.sgd.branch.__dict__
    experionml.clear()
    assert not experionml.sgd._evals
    assert experionml.sgd._shap._shap_values.empty
    assert "holdout" not in experionml.sgd.branch.__dict__


@patch("gradio.Interface")
def test_create_app(interface):
    """Assert that the create_app method calls the underlying package."""
    experionml = ExperionMLClassifier(X10_str, y10_str, random_state=1)
    experionml.clean()
    experionml.encode()
    experionml.run("Tree")
    experionml.tree.create_app()
    interface.assert_called_once()


def test_create_dashboard_multioutput():
    """Assert that the method is unavailable for multioutput tasks."""
    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    experionml.run("LR")
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        experionml.tree.create_dashboard()


@patch("explainerdashboard.ExplainerDashboard")
def test_create_dashboard_binary(func):
    """Assert that the create_dashboard method calls the underlying package."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    experionml.run("LR")
    experionml.lr.create_dashboard(dataset="holdout", filename="dashboard")
    func.assert_called_once()


@patch("explainerdashboard.ExplainerDashboard")
def test_create_dashboard_multiclass(func):
    """Assert that the create_dashboard method calls the underlying package."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    experionml.run("Tree")
    experionml.tree.create_dashboard()
    func.assert_called_once()


@patch("explainerdashboard.ExplainerDashboard")
def test_create_dashboard_regression(func):
    """Assert that the create_dashboard method calls the underlying package."""
    experionml = ExperionMLRegressor(X_reg, y_reg, holdout_size=0.1, random_state=1)
    experionml.run("Tree")
    experionml.tree.create_dashboard(dataset="both")
    func.assert_called_once()


def test_cross_validate_groups():
    """Assert that an error is raised when groups are passed directly."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    with pytest.raises(ValueError, match=".*groups can not be passed directly.*"):
        experionml.lr.cross_validate(groups=bin_groups)


def test_cross_validate():
    """Assert that the cross_validate method works as intended."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    assert isinstance(experionml.lr.cross_validate(), Styler)
    assert isinstance(experionml.lr.cross_validate(scoring="AP"), Styler)


def test_cross_validate_ts():
    """Assert that the cross_validate method works for forecast tasks."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.run("NF")
    assert isinstance(experionml.nf.cross_validate(), Styler)
    assert isinstance(experionml.nf.cross_validate(scoring="mae"), Styler)


def test_evaluate_metric_None():
    """Assert that the evaluate method works when metric is empty."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    experionml.run("MNB")
    scores = experionml.mnb.evaluate()
    assert len(scores) == 9

    experionml = ExperionMLClassifier(X_class, y_class, holdout_size=0.1, random_state=1)
    experionml.run("MNB")
    scores = experionml.mnb.evaluate()
    assert len(scores) == 6

    experionml = ExperionMLClassifier(X_label, y=y_label, holdout_size=0.1, random_state=1)
    experionml.run("MNB")
    scores = experionml.mnb.evaluate()
    assert len(scores) == 7

    experionml = ExperionMLRegressor(X_reg, y_reg, holdout_size=0.1, random_state=1)
    experionml.run("OLS")
    scores = experionml.ols.evaluate()
    assert len(scores) == 5


def test_evaluate_custom_metric():
    """Assert that custom metrics are accepted."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("MNB")
    assert isinstance(experionml.mnb.evaluate("roc_auc_ovo"), pd.Series)


def test_export_pipeline():
    """Assert that the pipeline can be retrieved from the model."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.clean()
    experionml.run("LR")
    assert len(experionml.lr.export_pipeline()) == 3


def test_full_train_no_holdout():
    """Assert that an error is raised when include_holdout=True with no set."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LGB")
    with pytest.raises(ValueError, match=".*holdout data set.*"):
        experionml.lgb.full_train(include_holdout=True)


def test_full_train():
    """Assert that the full_train method trains on the test set."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LGB")
    experionml.lgb.full_train()
    assert experionml.lgb.score("test") == 1.0  # Perfect score on test


def test_full_train_holdout():
    """Assert that the full_train method trains on the holdout set."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.2, random_state=1)
    experionml.run("Tree")
    experionml.tree.full_train(include_holdout=True)
    assert experionml.tree.score("holdout") == 1.0  # Perfect score on holdout


def test_full_train_new_mlflow_run():
    """Assert that a new mlflow run is created."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("GNB")
    run = experionml.gnb._run
    experionml.gnb.full_train()
    assert experionml.gnb._run is not run


def test_get_best_threshold():
    """Assert that the get_best_threshold method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR", metric=["f1", "auc"])
    assert 0 < experionml.lr.get_best_threshold(metric=None, train_on_test=True) < 1
    assert 0 < experionml.lr.get_best_threshold(metric=1) < 1
    assert hasattr(experionml.lr, "tuned_threshold")


def test_inverse_transform():
    """Assert that the inverse_transform method works as intended."""
    experionml = ExperionMLClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    experionml.run("LR")
    assert_frame_equal(experionml.lr.inverse_transform(experionml.lr.X), X_bin)


def test_inverse_transform_output():
    """Assert that the output type is determined by the data engine."""
    experionml = ExperionMLClassifier(X_bin, y_bin, engine="polars", random_state=1)
    experionml.run("Tree")
    assert isinstance(experionml.tree.inverse_transform(X_bin), pl.DataFrame)


def test_save_estimator():
    """Assert that the save_estimator saves a pickle file."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("MNB")
    experionml.mnb.save_estimator("auto")
    assert glob.glob("MultinomialNB.pkl")


@pytest.mark.skipif(not find_spec("ray"), reason="Ray is not installed.")
def test_serve():
    """Assert that the serve method deploys a reachable endpoint."""
    from ray import serve

    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("MNB")
    experionml.mnb.serve()
    response = requests.get("http://127.0.0.1:8000/", json=X_bin.to_json(), timeout=5)
    assert response.status_code == 200
    serve.shutdown()


def test_register_no_experiment():
    """Assert that an error is raised when there is no experiment."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("MNB")
    with pytest.raises(PermissionError, match=".*mlflow experiment.*"):
        experionml.mnb.register()


@patch("mlflow.register_model")
@patch("mlflow.MlflowClient.transition_model_version_stage")
def test_register(mlflow, client):
    """Assert that the register saves the model to a stage."""
    experionml = ExperionMLClassifier(X_bin, y_bin, experiment="test", random_state=1)
    experionml.run("MNB")
    experionml.mnb.register()
    mlflow.assert_called_once()
    client.assert_called_once()


def test_transform():
    """Assert that new data can be transformed by the model's pipeline."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    experionml.encode()
    experionml.run("LR")
    X = experionml.lr.transform(X10_str)
    assert len(X.columns) > 3  # Data is one-hot encoded
    assert all(-3 <= v <= 3 for v in X.to_numpy().ravel())  # Data is scaled


def test_transform_output():
    """Assert that the output type is determined by the data engine."""
    experionml = ExperionMLClassifier(X_bin, y_bin, engine="polars", random_state=1)
    experionml.run("Tree")
    assert isinstance(experionml.tree.transform(X_bin), pl.DataFrame)


# Test ClassRegModel ================================================== >>

def test_classreg_get_tags():
    """Assert that the get_tags method returns the tags."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    assert isinstance(experionml.lr.get_tags(), dict)


def test_predictions_from_index():
    """Assert that predictions can be made from data indices."""
    experionml = ExperionMLClassifier(X_idx, y_idx, index=True, holdout_size=0.1, random_state=1)
    experionml.run("LR")
    assert isinstance(experionml.lr.decision_function(("index_4", "index_5")), pd.Series)
    assert isinstance(experionml.lr.predict(["index_4", "index_8"]), pd.Series)
    assert isinstance(experionml.lr.predict_log_proba(-100), pd.DataFrame)
    assert isinstance(experionml.lr.predict_proba("index_4"), pd.DataFrame)
    assert isinstance(experionml.lr.score(slice(10, 20)), float)


def test_transformations_first():
    """Assert that the transformations are applied before predicting."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    experionml.encode(max_onehot=None)
    experionml.run("Tree")
    assert isinstance(experionml.tree.predict(X10_str), pd.Series)


def test_data_is_scaled():
    """Assert that the data is scaled for models that need it."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    assert sum(experionml.lr.predict(X_bin)) > 0  # Always 0 if not scaled


def test_predictions_from_new_data():
    """Assert that predictions can be made from new data."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    assert isinstance(experionml.lr.predict(X_bin), pd.Series)
    assert isinstance(experionml.lr.predict_proba(X_bin), pd.DataFrame)


def test_prediction_from_multioutput():
    """Assert that predictions can be made for multioutput datasets."""
    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    experionml.run("LR")
    assert isinstance(experionml.lr.predict_proba(X_class).index, pd.MultiIndex)


def test_prediction_inverse_transform():
    """Assert that the predict method can return the inversely transformed data."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.scale(columns=-1)
    experionml.run("Tree")
    assert check_scaling(experionml.tree.predict(X_reg, inverse=False))
    assert not check_scaling(experionml.tree.predict(X_reg, inverse=True))


def test_score_regression():
    """Assert that the score returns r2 for regression tasks."""
    experionml = ExperionMLRegressor(X_reg, y_reg, shuffle=False, random_state=1)
    experionml.run("Tree")
    r2 = r2_score(y_reg, experionml.tree.predict(X_reg))
    assert experionml.tree.score(X_reg, y_reg) == r2


def test_score_metric_is_None():
    """Assert that the score method returns the default metric."""
    experionml = ExperionMLClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    experionml.run("Tree")
    f1 = f1_score(y_bin, experionml.tree.predict(X_bin))
    assert experionml.tree.score(X_bin, y_bin) == f1


def test_score_custom_metric():
    """Assert that the score method works for a custom scorer."""
    experionml = ExperionMLClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    experionml.run("Tree")
    recall = recall_score(y_bin, experionml.tree.predict(X_bin))
    assert experionml.tree.score(X_bin, y_bin, metric="recall") == recall


def test_score_with_sample_weight():
    """Assert that the score method works when sample weights are provided."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    score = experionml.tree.score(X_bin, y_bin, sample_weight=list(range(len(y_bin))))
    assert isinstance(score, float)


# Test ForecastModel =============================================== >>

def test_forecast_get_tags():
    """Assert that the get_tags method returns the tags."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.run("NF")
    assert isinstance(experionml.nf.get_tags(), dict)


def test_predictions_invalid_fh():
    """Assert that predictions can be made using only the fh."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.run("NF")
    with pytest.raises(ValueError, match=".*Use a ForecastingHorizon.*"):
        experionml.nf.predict(fh=range(200))


def test_predictions_only_fh():
    """Assert that predictions can be made using only the fh with exogenous variables."""
    experionml = ExperionMLForecaster(X_ex, y=y_ex, random_state=1)
    experionml.run("OLS")
    assert isinstance(experionml.ols.predict(fh=experionml.test), pd.Series)
    assert isinstance(experionml.ols.predict(fh=ForecastingHorizon([1, 2]), X=X_ex.iloc[:2]), pd.Series)


def test_predictions_with_exogenous():
    """Assert that predictions can be made with exogenous variables."""
    experionml = ExperionMLForecaster(X_ex, y=y_ex, random_state=1)
    experionml.run("NF")
    assert isinstance(experionml.nf.predict(ForecastingHorizon(range(10)), X=X_ex.iloc[:10]), pd.Series)
    assert isinstance(experionml.nf.predict_proba(range(10), X=X_ex.iloc[:10]), Normal)
    assert isinstance(experionml.nf.predict_quantiles(range(10), X=X_ex.iloc[:10]), pd.DataFrame)
    assert isinstance(experionml.nf.predict_var(range(10), X=X_ex.iloc[:10]), pd.DataFrame)


def test_ts_prediction_inverse_transform():
    """Assert that the predict method can return the inversely transformed data."""
    experionml = ExperionMLForecaster(X_ex, y=y_ex, random_state=1)
    experionml.scale(columns=-1)
    experionml.run("NF")
    assert check_scaling(experionml.nf.predict(y_ex, inverse=False))
    assert not check_scaling(experionml.nf.predict(y_ex, inverse=True))
    assert check_scaling(experionml.nf.predict_interval(y_ex, inverse=False))
    assert not check_scaling(experionml.nf.predict_interval(y_ex, inverse=True))


def test_predictions_with_y():
    """Assert that predictions can be made with y."""
    experionml = ExperionMLForecaster(X_ex.iloc[:-2], y=y_ex[:-2], random_state=1)
    experionml.run("OLS")
    assert isinstance(experionml.ols.predict_residuals(y=experionml.test), pd.Series)
    assert isinstance(experionml.ols.predict_residuals(y=y_ex[-2:], X=X_ex.iloc[-2:]), pd.Series)


def test_score_ts_metric_is_None():
    """Assert that the score method returns the default metric."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.run("NF")
    mape = mean_absolute_percentage_error(y_fc[:10], experionml.nf.predict(range(10)))
    assert experionml.nf.score(y_fc[:10]) == np.negative(mape)


def test_score_ts_custom_metric():
    """Assert that the score method works for a custom scorer."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.run("NF")
    mae = mean_absolute_error(y_fc[:10], experionml.nf.predict(range(10)))
    assert experionml.nf.score(y_fc[:10], metric="mae") == np.negative(mae)
