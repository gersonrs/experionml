
from sklearn.linear_model import HuberRegressor

from experionml import ExperionMLClassifier, ExperionMLForecaster, ExperionMLModel, ExperionMLRegressor

from .conftest import X_bin, X_reg, y_bin, y_fc, y_reg


def test_experionmlmodel():
    """Assert that it returns an estimator that works with experionml."""
    model = ExperionMLModel(
        estimator=(huber := HuberRegressor()),
        name="huber1",
        acronym="huber",
        needs_scaling=True,
    )

    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run(model)
    assert model is not huber  # Is cloned
    assert model.name == "huber1"
    assert model.acronym == "huber"
    assert model.needs_scaling is True
    assert model.native_multioutput is False
    assert model.validation is None


def test_experionmlclassifier():
    """Assert that the goal is set correctly for ExperionMLClassifier."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml._goal.name == "classification"


def test_experionmlforecaster():
    """Assert that the goal is set correctly for ExperionMLForecaster."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    assert experionml._goal.name == "forecast"


def test_experionmlregressor():
    """Assert that the goal is set correctly for ExperionMLRegressor."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    assert experionml._goal.name == "regression"
