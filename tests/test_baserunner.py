
import glob
import sys
from random import choices
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.io.formats.style import Styler
from pandas.testing import (
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import Bunch

from experionml import (
    ExperionMLClassifier,
    ExperionMLForecaster,
    ExperionMLRegressor,
)
from experionml.data import Branch
from experionml.training import DirectClassifier, DirectForecaster
from experionml.utils.utils import NotFittedError, merge

from .conftest import (
    X10,
    X_bin,
    X_class,
    X_idx,
    X_reg,
    bin_groups,
    bin_test,
    bin_train,
    fc_test,
    fc_train,
    y10,
    y_bin,
    y_class,
    y_fc,
    y_idx,
    y_multiclass,
    y_reg,
)


# Test magic methods =============================================== >>


def test_getstate_and_setstate():
    """Assert that versions are checked and a warning raised."""
    experionml = ExperionMLClassifier(X_bin, y_bin, warnings=True)
    experionml.run("LR")
    experionml.save("experionml")

    sys.modules["sklearn"].__version__ = "1.2.7"  # Fake version
    with pytest.warns(Warning, match=".*while the version in this environment.*"):
        ExperionMLClassifier.load("experionml")


def test_dir():
    """Assert that __dir__ contains all the extra attributes."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run("dummy")
    assert all(attr in dir(experionml) for attr in ("X", "main", "age", "dummy"))


def test_getattr_branch():
    """Assert that branches can be called."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.branch = "b2"
    assert experionml.b2 is experionml._branches["b2"]


def test_getattr_attr_from_branch():
    """Assert that branch attributes can be called."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml.pipeline is experionml.branch.pipeline


def test_getattr_model():
    """Assert that the models can be called as attributes."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("Tree")
    assert experionml.tree is experionml._models[0]


def test_getattr_column():
    """Assert that the columns can be accessed as attributes."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    assert isinstance(experionml.alcohol, pd.Series)


def test_getattr_dataframe():
    """Assert that the dataset attributes can be called."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(experionml.head(), pd.DataFrame)


def test_getattr_invalid():
    """Assert that an error is raised when there is no such attribute."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(AttributeError, match=".*object has no attribute.*"):
        _ = experionml.invalid


def test_setattr_to_branch():
    """Assert that branch properties can be set."""
    new_dataset = merge(X_bin, y_bin)
    new_dataset.iloc[0, 3] = 4  # Change one value

    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.dataset = new_dataset
    assert experionml.dataset.iloc[0, 3] == 4  # Check the value is changed


def test_setattr_normal():
    """Assert that attributes can be set normally."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.attr = "test"
    assert experionml.attr == "test"


def test_delattr_models():
    """Assert that models can be deleted through del."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["MNB", "LR"])
    del experionml.lr
    assert experionml.models == "MNB"


def test_delattr_normal():
    """Assert that attributes can be deleted normally."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    del experionml._config
    assert not hasattr(experionml, "_config")


def test_contains():
    """Assert that we can test if a trainer contains a column."""
    trainer = DirectClassifier(models="LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert "mean radius" in trainer


def test_len():
    """Assert that the length of a trainer is the length of the dataset."""
    trainer = DirectClassifier(models="LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert len(trainer) == len(X_bin)


def test_getitem_no_dataset():
    """Assert that an error is raised when getitem is used before run."""
    trainer = DirectClassifier(models="LR", random_state=1)
    with pytest.raises(RuntimeError, match=".*has no dataset.*"):
        print(trainer[4])


def test_getitem_int():
    """Assert that getitem works for a column index."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert_series_equal(experionml[0], experionml["mean radius"])


def test_getitem_str_from_branch():
    """Assert that getitem works for a branch name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml["main"] is experionml._branches["main"]


def test_getitem_str_from_model():
    """Assert that getitem works for a model name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LDA")
    assert experionml["lda"] is experionml.lda


def test_getitem_str_from_column():
    """Assert that getitem works for a column name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert_series_equal(experionml["mean radius"], experionml.dataset["mean radius"])


def test_getitem_invalid_str():
    """Assert that an error is raised when getitem is invalid."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*has no branch, model or column.*"):
        print(experionml["invalid"])


def test_getitem_list():
    """Assert that getitem works for a list of column names."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(experionml[["mean radius", "mean texture"]], pd.DataFrame)


# Test utility properties ========================================== >>


def test_sp_property_none():
    """Assert that the sp property can be set up correctly."""
    experionml = ExperionMLForecaster(y_fc, sp=None, random_state=1)
    assert experionml.sp == experionml._config.sp == Bunch()


def test_sp_property_invalid_index():
    """Assert that an error is raised when index has no freqstr."""
    with pytest.raises(ValueError, match=".*has no attribute freqstr.*"):
        ExperionMLForecaster(y_bin, sp="index", random_state=1)


def test_sp_property_index():
    """Assert that the sp property can be set up correctly."""
    experionml = ExperionMLForecaster(y_fc, sp="index", random_state=1)
    assert experionml.sp.sp == 12


def test_sp_property_infer():
    """Assert that the sp property can be set up correctly."""
    experionml = ExperionMLForecaster(y_fc, sp="infer", random_state=1)
    assert experionml.sp.sp == [12, 24, 36, 11, 48]


def test_sp_property_invalid_str():
    """Assert that an error is raised when sp in an unknown string."""
    with pytest.raises(ValueError, match=".*a list of allowed values.*"):
        ExperionMLForecaster(y_fc, sp="T", random_state=1)


def test_sp_property_str():
    """Assert that the sp property can be set up correctly."""
    experionml = ExperionMLForecaster(y_fc, sp="W", random_state=1)
    assert experionml.sp.sp == 52


def test_sp_property_int():
    """Assert that the sp property can be set up correctly."""
    experionml = ExperionMLForecaster(y_fc, sp=3, random_state=1)
    assert experionml.sp.sp == 3


def test_sp_property_sequence():
    """Assert that the sp property can be set up correctly."""
    experionml = ExperionMLForecaster(y_fc, sp=(12, 24), random_state=1)
    assert experionml.sp == Bunch(sp=[12, 24])


def test_sp_property_dict():
    """Assert that the sp property can be set up correctly."""
    experionml = ExperionMLForecaster(y_fc, sp={"seasonal_model": "multiplicative"})
    assert experionml.sp == Bunch(seasonal_model="multiplicative")


def test_branch_property():
    """Assert that the branch property returns the current branch."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert isinstance(experionml.branch, Branch)


def test_delete_last_branch():
    """Assert that an error is raised when the last branch is deleted."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(PermissionError, match=".*last branch.*"):
        del experionml.branch


def test_delete_depending_models():
    """Assert that dependent models are deleted with the branch."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.branch = "b2"
    experionml.run("LR")
    del experionml.branch
    assert not experionml.models


def test_delete_current():
    """Assert that we can delete the current branch."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.branch = "b2"
    del experionml.branch
    assert "b2" not in experionml._branches


def test_models_property():
    """Assert that the models property returns the model names."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "Tree"])
    assert experionml.models == ["LR", "Tree"]


def test_metric_property():
    """Assert that the metric property returns the metric names."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("lr", metric="f1")
    assert experionml.metric == "f1"


def test_winners_property():
    """Assert that the winners property returns the best models."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml.winners is None
    experionml.run(["LR", "Tree", "LDA"])
    assert experionml.winners == [experionml.lr, experionml.lda, experionml.tree]


def test_winner_property():
    """Assert that the winner property returns the best model."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml.winner is None
    experionml.run(["LR", "Tree", "LDA"])
    assert experionml.winner is experionml.lr


def test_winner_deleter():
    """Assert that the winning model can be deleted through del."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "Tree", "LDA"])
    del experionml.winner
    assert experionml.models == ["Tree", "LDA"]


def test_results_property():
    """Assert that the results property returns an overview of the results."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    assert experionml.results.data.shape == (1, 4)


def test_results_property_dropna():
    """Assert that the results property doesn't return columns with NaNs."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    assert "mean_bootstrap" not in experionml.results.data


def test_results_property_successive_halving():
    """Assert that the results property works for successive halving runs."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.successive_halving(["OLS", "Tree"])
    assert experionml.results.data.shape == (3, 4)
    assert list(experionml.results.data.index.get_level_values(0)) == [0.5, 0.5, 1.0]


def test_results_property_train_sizing():
    """Assert that the results property works for train sizing runs."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.train_sizing("LR")
    assert experionml.results.data.shape == (5, 4)
    assert list(experionml.results.data.index.get_level_values(0)) == [0.2, 0.4, 0.6, 0.8, 1.0]


# Test _get_data =================================================== >>


def test_groups_with_forecast():
    """Assert that an error is raised when groups are provided in a forecast task."""
    with pytest.raises(ValueError, match=".*'groups' is unavailable for forecast.*"):
        ExperionMLForecaster(
            y_fc, metadata={"groups": choices(["A", "B"], k=len(y_fc))}, random_state=1
        )


def test_index_is_true():
    """Assert that the indices are left as is when index=True."""
    experionml = ExperionMLClassifier(X_idx, y_idx, index=True, shuffle=False, random_state=1)
    assert experionml.dataset.index[0] == "index_0"


def test_index_is_False():
    """Assert that the indices are reset when index=False."""
    experionml = ExperionMLClassifier(X_idx, y_idx, index=False, shuffle=False, random_state=1)
    assert experionml.dataset.index[0] == 0


def test_index_is_int_invalid():
    """Assert that an error is raised when the index is an invalid int."""
    with pytest.raises(IndexError, match=".*is out of range.*"):
        ExperionMLClassifier(X_bin, y_bin, index=1000, random_state=1)


def test_index_is_int():
    """Assert that a column can be selected from a position."""
    X = X_bin.copy()
    X["mean radius"] = range(len(X))
    experionml = ExperionMLClassifier(X, y_bin, index=0, random_state=1)
    assert experionml.dataset.index.name == "mean radius"


def test_index_is_str_invalid():
    """Assert that an error is raised when the index is an invalid str."""
    with pytest.raises(ValueError, match=".*not found in the dataset.*"):
        ExperionMLClassifier(X_bin, y_bin, index="invalid", random_state=1)


def test_index_is_str():
    """Assert that a column can be selected from a name."""
    X = X_bin.copy()
    X["mean texture"] = range(len(X))
    experionml = ExperionMLClassifier(X, y_bin, index="mean texture", random_state=1)
    assert experionml.dataset.index.name == "mean texture"


def test_index_is_range():
    """Assert that a column can be selected from a name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, index=range(len(X_bin)), shuffle=False)
    assert list(experionml.dataset.index) == list(range(len(X_bin)))


def test_index_is_target():
    """Assert that an error is raised when the index is the target column."""
    with pytest.raises(ValueError, match=".*same as the target column.*"):
        ExperionMLRegressor(X_bin, index="worst fractal dimension", random_state=1)


def test_index_is_sequence_no_data_sets_invalid_length():
    """Assert that an error is raised when len(index) != len(data)."""
    with pytest.raises(IndexError, match=".*Length of index.*"):
        ExperionMLClassifier(X_bin, y_bin, index=[1, 2, 3], random_state=1)


def test_index_is_sequence_no_data_sets():
    """Assert that a sequence is set as index when provided."""
    index = [f"index_{i}" for i in range(len(X_bin))]
    experionml = ExperionMLClassifier(X_bin, y_bin, index=index, random_state=1)
    assert experionml.dataset.index[0] == "index_242"


def test_index_is_sequence_has_data_sets_invalid_length():
    """Assert that an error is raised when len(index) != len(data)."""
    with pytest.raises(IndexError, match=".*Length of index.*"):
        ExperionMLClassifier(bin_train, bin_test, index=[1, 2, 3], random_state=1)


def test_index_is_sequence_has_data_sets():
    """Assert that a sequence is set as index when provided."""
    index = [f"index_{i}" for i in range(len(bin_train) + 2 * len(bin_test))]
    experionml = ExperionMLClassifier(bin_train, bin_test, bin_test, index=index, random_state=1)
    assert experionml.dataset.index[0] == "index_0"
    assert experionml.holdout.index[0] == "index_569"


@pytest.mark.parametrize("stratify", [-1, "target"])
def test_stratify_options(stratify):
    """Assert that the data can be stratified among data sets."""
    experionml = ExperionMLClassifier(X_bin, y_bin, stratify=stratify, random_state=1)
    train_balance = experionml.classes["train"][0] / experionml.classes["train"][1]
    test_balance = experionml.classes["test"][0] / experionml.classes["test"][1]
    np.testing.assert_almost_equal(train_balance, test_balance, decimal=2)


def test_stratify_is_None():
    """Assert that the data is not stratified when stratify=None."""
    experionml = ExperionMLClassifier(X_bin, y_bin, stratify=None, random_state=1)
    train_balance = experionml.classes["train"][0] / experionml.classes["train"][1]
    test_balance = experionml.classes["test"][0] / experionml.classes["test"][1]
    assert abs(train_balance - test_balance) > 0.05


def test_stratify_invalid_column_int():
    """Assert that an error is raised when the value is invalid."""
    with pytest.raises(ValueError, match=".*out of range for a dataset.*"):
        ExperionMLClassifier(X_bin, y_bin, stratify=100, random_state=1)


def test_stratify_invalid_column_str():
    """Assert that an error is raised when the value is invalid."""
    with pytest.raises(ValueError, match=".*not found in the dataset.*"):
        ExperionMLClassifier(X_bin, y_bin, stratify="invalid", random_state=1)


def test_input_is_y_without_arrays():
    """Assert that input y through parameter works."""
    experionml = ExperionMLForecaster(y=y_fc, random_state=1)
    assert experionml.dataset.shape == (len(y_fc), 1)


def test_empty_data_arrays():
    """Assert that an error is raised when no data is provided."""
    with pytest.raises(ValueError, match=".*data arrays are empty.*"):
        ExperionMLClassifier(n_rows=100, random_state=1)


def test_data_already_set():
    """Assert that if there already is data, the call to run can be empty."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    trainer.run()
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))
    assert_index_equal(trainer.branch._data.train_idx, bin_train.index)
    assert_index_equal(trainer.branch._data.test_idx, bin_test.index)


def test_input_is_X():
    """Assert that input X works."""
    experionml = ExperionMLRegressor(X_bin, random_state=1)
    assert experionml.dataset.shape == X_bin.shape


def test_input_is_y():
    """Assert that input y works for forecasting tasks."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    assert experionml.dataset.shape == (len(y_fc), 1)


def test_input_is_X_with_parameter_y():
    """Assert that input X can be combined with parameter y."""
    experionml = ExperionMLRegressor(X_bin, y="mean texture", random_state=1)
    assert experionml.target == "mean texture"


def test_input_invalid_holdout():
    """Assert that an error is raised when holdout is invalid."""
    with pytest.raises(ValueError, match=".*holdout_size parameter.*"):
        ExperionMLClassifier(X_bin, test_size=0.3, holdout_size=0.8)


@pytest.mark.parametrize("holdout_size", [0.1, 40])
def test_input_is_X_with_holdout(holdout_size):
    """Assert that input X can be combined with a holdout set."""
    experionml = ExperionMLRegressor(X_bin, holdout_size=holdout_size, random_state=1)
    assert isinstance(experionml.holdout, pd.DataFrame)


def test_input_holdout_with_groups():
    """Assert that the holdout size is determined based on groups."""
    experionml = ExperionMLRegressor(X_bin, holdout_size=0.2, metadata=bin_groups, random_state=1)
    assert len(experionml.holdout) in experionml.metadata["groups"].value_counts().to_numpy()


@pytest.mark.parametrize("shuffle", [True, False])
def test_input_is_train_test_with_holdout(shuffle):
    """Assert that input train and test can be combined with a holdout set."""
    experionml = ExperionMLClassifier(bin_train, bin_test, bin_test, shuffle=shuffle)
    assert isinstance(experionml.holdout, pd.DataFrame)


@pytest.mark.parametrize("n_rows", [0.7, 1])
def test_n_rows_X_y_frac(n_rows):
    """Assert that n_rows<=1 work for input X and X, y."""
    experionml = ExperionMLClassifier(X_bin, y_bin, n_rows=n_rows, random_state=1)
    assert len(experionml.dataset) == int(len(X_bin) * n_rows)


def test_n_rows_X_y_int():
    """Assert that n_rows>1 work for input X and X, y."""
    experionml = ExperionMLClassifier(X_bin, y_bin, n_rows=200, random_state=1)
    assert len(experionml.dataset) == 200


def test_n_rows_forecasting():
    """Assert that the rows are cut from the dataset's head when forecasting."""
    experionml = ExperionMLForecaster(y_fc, n_rows=142, random_state=1)
    assert len(experionml.dataset) == 142
    assert experionml.dataset.index[0] == y_fc.index[len(y_fc) - 142]


def test_n_rows_too_large():
    """Assert that an error is raised when n_rows>len(data)."""
    with pytest.raises(ValueError, match=".*n_rows parameter.*"):
        ExperionMLClassifier(X_bin, y_bin, n_rows=1e6, random_state=1)


def test_no_shuffle_X_y():
    """Assert that the order is kept when shuffle=False."""
    experionml = ExperionMLClassifier(X_bin, y_bin, shuffle=False)
    assert_frame_equal(experionml.X, X_bin)


def test_length_dataset():
    """Assert that the dataset is always len>=5."""
    with pytest.raises(ValueError, match=".*n_rows=1 for small.*"):
        ExperionMLClassifier(X10, y10, n_rows=0.01, random_state=1)


@pytest.mark.parametrize("test_size", [-2, 0, 1000])
def test_test_size_parameter(test_size):
    """Assert that the test_size parameter is in correct range."""
    with pytest.raises(ValueError, match=".*test_size parameter.*"):
        ExperionMLClassifier(X_bin, test_size=test_size, random_state=1)


def test_test_size_fraction():
    """Assert that the test_size parameters splits the sets correctly when <1."""
    experionml = ExperionMLClassifier(X_bin, y_bin, test_size=0.2, random_state=1)
    assert len(experionml.test) == int(0.2 * len(X_bin))
    assert len(experionml.train) == len(X_bin) - int(0.2 * len(X_bin))


def test_test_size_int():
    """Assert that the test_size parameters splits the sets correctly when >=1."""
    experionml = ExperionMLClassifier(X_bin, y_bin, test_size=100, random_state=1)
    assert len(experionml.test) == 100
    assert len(experionml.train) == len(X_bin) - 100


def test_input_is_X_y():
    """Assert that input X, y works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml.dataset.shape == merge(X_bin, y_bin).shape


def test_input_is_2_tuples():
    """Assert that the 2 tuples input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run((X_train, y_train), (X_test, y_test))
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_input_is_train_test():
    """Assert that input train, test works."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test)
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_input_is_train_test_with_parameter_y():
    """Assert that input X works can be combined with y."""
    experionml = ExperionMLClassifier(bin_train, bin_test, y="mean texture", random_state=1)
    assert experionml.target == "mean texture"


def test_input_is_train_test_for_forecast():
    """Assert that input train, test works for forecast tasks."""
    trainer = DirectForecaster("Croston", random_state=1)
    trainer.run(fc_train, fc_test)
    assert_series_equal(trainer.y, pd.concat([fc_train, fc_test]))


def test_input_is_3_tuples():
    """Assert that the 3 tuples input works."""
    X_train = bin_train.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    X_test = bin_test.iloc[:-20, :-1]
    y_test = bin_test.iloc[:-20, -1]
    X_holdout = bin_test.iloc[-20:, :-1]
    y_holdout = bin_test.iloc[-20:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run((X_train, y_train), (X_test, y_test), (X_holdout, y_holdout))
    assert_frame_equal(trainer.X, pd.concat([X_train, X_test]))
    assert_frame_equal(trainer.holdout, pd.concat([X_holdout, y_holdout], axis=1))


def test_input_is_train_test_holdout():
    """Assert that input train, test, holdout works."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(bin_train, bin_test.iloc[:100], bin_test.iloc[100:])
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test.iloc[:100]]))
    assert_frame_equal(trainer.holdout, bin_test.iloc[100:])


def test_4_data_provided():
    """Assert that the four-element input works."""
    X_train = bin_train.iloc[:, :-1]
    X_test = bin_test.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    y_test = bin_test.iloc[:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(X_train, X_test, y_train, y_test)
    assert_frame_equal(trainer.dataset, pd.concat([bin_train, bin_test]))


def test_6_data_provided():
    """Assert that the six-element input works."""
    X_train = bin_train.iloc[:, :-1]
    y_train = bin_train.iloc[:, -1]
    X_test = bin_test.iloc[:-20, :-1]
    y_test = bin_test.iloc[:-20, -1]
    X_holdout = bin_test.iloc[-20:, :-1]
    y_holdout = bin_test.iloc[-20:, -1]

    trainer = DirectClassifier("LR", random_state=1)
    trainer.run(X_train, X_test, X_holdout, y_train, y_test, y_holdout)
    assert_frame_equal(trainer.X, pd.concat([X_train, X_test]))
    assert_frame_equal(trainer.holdout, pd.concat([X_holdout, y_holdout], axis=1))


def test_invalid_input():
    """Assert that an error is raised when input arrays are invalid."""
    trainer = DirectClassifier("LR", random_state=1)
    with pytest.raises(ValueError, match=".*Invalid data arrays.*"):
        trainer.run(X_bin, y_bin, X_bin, y_bin, y_bin, X_bin, X_bin)


def test_n_rows_train_test_frac():
    """Assert that n_rows<=1 work for input with train and test."""
    experionml = ExperionMLClassifier(bin_train, bin_test, n_rows=0.8, random_state=1)
    assert len(experionml.train) == int(len(bin_train) * 0.8)
    assert len(experionml.test) == int(len(bin_test) * 0.8)


def test_no_shuffle_train_test():
    """Assert that the order is kept when shuffle=False."""
    experionml = ExperionMLClassifier(bin_train, bin_test, shuffle=False)
    assert_frame_equal(
        left=experionml.train,
        right=bin_train.reset_index(drop=True),
        check_dtype=False,
    )


def test_n_rows_train_test_int():
    """Assert that an error is raised when n_rows>1 for input with train and test."""
    with pytest.raises(ValueError, match=".*must be <1 when the train and test.*"):
        ExperionMLClassifier(bin_train, bin_test, n_rows=100, random_state=1)


def test_dataset_is_shuffled():
    """Assert that the dataset is shuffled before splitting."""
    experionml = ExperionMLClassifier(X_bin, y_bin, shuffle=True, random_state=1)
    assert not X_bin.equals(experionml.X)


def test_holdout_is_shuffled():
    """Assert that the holdout set is shuffled."""
    experionml = ExperionMLClassifier(bin_train, bin_test, bin_test, shuffle=True, random_state=1)
    assert not bin_test.equals(experionml.holdout)


def test_reset_index():
    """Assert that the indices are reset for the all data sets."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    assert list(experionml.dataset.index) == list(range(len(experionml.dataset)))


def test_unequal_columns_train_test():
    """Assert that an error is raised when train and test have different columns."""
    with pytest.raises(ValueError, match=".*train and test set do not have.*"):
        ExperionMLClassifier(X10, bin_test, random_state=1)


def test_unequal_columns_holdout():
    """Assert that an error is raised when holdout has different columns."""
    with pytest.raises(ValueError, match=".*holdout set does not have.*"):
        ExperionMLClassifier(bin_train, bin_test, X10, random_state=1)


def test_merger_to_dataset():
    """Assert that the merger between X and y was successful."""
    # Reset index since the order of rows is different after shuffling
    merger = X_bin.merge(y_bin.to_frame(), left_index=True, right_index=True)
    df1 = merger.sort_values(by=merger.columns.tolist())

    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    df2 = experionml.dataset.sort_values(by=experionml.dataset.columns.tolist())
    assert_frame_equal(
        left=df1.reset_index(drop=True),
        right=df2.reset_index(drop=True),
        check_dtype=False,
    )


def test_invalid_index_forecast():
    """Assert that an error is raised when the index is invalid."""
    with pytest.raises(ValueError, match=".*index of the dataset must.*"):
        ExperionMLForecaster(pd.Series([1, 2, 3, 4, 5], index=[1, 4, 2, 3, 5]), random_state=1)


def test_duplicate_indices():
    """Assert that an error is raised when there are duplicate indices."""
    with pytest.raises(ValueError, match=".*duplicate indices.*"):
        ExperionMLClassifier(X_bin, X_bin, index=True, random_state=1)


# Test utility methods ============================================= >>


def test_get_models_is_None():
    """Assert that all models are returned by default."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR_1", "LR_2"])
    assert experionml._get_models(models=None) == [experionml.lr_1, experionml.lr_2]


def test_get_models_by_int():
    """Assert that models can be selected by index."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(IndexError, match=".*out of range.*"):
        experionml._get_models(models=0)
    experionml.run(["LR_1", "LR_2"])
    assert experionml._get_models(models=1) == [experionml.lr_2]


def test_get_models_by_slice():
    """Assert that a slice of models is returned."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.train_sizing(["LR_1", "LR_2"])
    assert len(experionml._get_models(models=slice(1, 4))) == 3


def test_get_models_winner():
    """Assert that the winner is returned when used as name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LDA"])
    assert experionml._get_models(models="winner") == [experionml.lr]


def test_get_models_by_str():
    """Assert that models can be retrieved by name or regex."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["GNB", "LR_1", "LR_2"])
    assert experionml._get_models("gnb+lr_1") == [experionml.gnb, experionml.lr_1]
    assert experionml._get_models(["gnb+lr_1", "lr_2"]) == [
        experionml.gnb,
        experionml.lr_1,
        experionml.lr_2,
    ]
    assert experionml._get_models("lr.*") == [experionml.lr_1, experionml.lr_2]
    assert experionml._get_models("!lr_1") == [experionml.gnb, experionml.lr_2]
    assert experionml._get_models("!lr.*") == [experionml.gnb]
    with pytest.raises(ValueError, match=".*any model that matches.*"):
        experionml._get_models(models="invalid")


def test_get_models_exclude():
    """Assert that models can be excluded using `!`."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*not find any model.*"):
        experionml._get_models(models="!invalid")
    experionml.run(["LR_1", "LR_2"])
    assert experionml._get_models(models="!lr_1") == [experionml.lr_2]
    assert experionml._get_models(models="!.*_2$") == [experionml.lr_1]


def test_get_models_by_model():
    """Assert that a model can be called using a Model instance."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    assert experionml._get_models(models=experionml.lr) == [experionml.lr]


def test_get_models_include_or_exclude():
    """Assert that an error is raised when models are included and excluded."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR_1", "LR_2"])
    with pytest.raises(ValueError, match=".*either include or exclude models.*"):
        experionml._get_models(models=["LR_1", "!LR_2"])


def test_get_models_remove_ensembles():
    """Assert that ensembles can be excluded."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR_1", "LR_2"])
    experionml.voting()
    assert "Vote" not in experionml._get_models(models=None, ensembles=False)


def test_get_models_invalid_branch():
    """Assert that an error is raised when the branch is invalid."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    experionml.branch = "2"
    experionml.run("LDA")
    with pytest.raises(ValueError, match=".*have been fitted.*"):
        experionml._get_models(models=None, branch=experionml.branch)


def test_get_models_remove_duplicates():
    """Assert that duplicate models are returned."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR_1", "LR_2"])
    assert experionml._get_models(["LR_1", "LR_1"]) == [experionml.lr_1]


def test_available_models():
    """Assert that the available_models method shows the models per task."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    models = experionml.available_models(native_multioutput=True, supports_engines="cuml")
    assert isinstance(models, pd.DataFrame)
    assert "RF" in models["acronym"].unique()
    assert "BR" not in models["acronym"].unique()  # Is not a classifier
    assert "MLP" not in models["acronym"].unique()  # Is not native multioutput
    assert models["supports_engines"].str.contains("cuml").all()


def test_clear():
    """Assert that the clear method resets all model's attributes."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LDA"])
    experionml.lda.plot_shap_beeswarm(display=False)
    assert not experionml.lda._shap._shap_values.empty
    experionml.clear()
    assert experionml.lda._shap._shap_values.empty


def test_delete_default():
    """Assert that all models in branch are deleted as default."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LDA"])
    experionml.delete()  # All models
    assert not experionml.models
    assert not experionml.metric
    assert experionml.results.data.empty


@pytest.mark.parametrize("metric", ["ap", "f1"])
def test_evaluate(metric):
    """Assert that the evaluate method works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, experionml.evaluate)
    experionml.run("Tree")
    assert isinstance(experionml.evaluate(metric), Styler)


def test_export_pipeline_same_transformer():
    """Assert that two same transformers get different names."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.clean()
    experionml.clean()
    experionml.clean()
    pl = experionml.export_pipeline()
    assert list(pl.named_steps) == ["cleaner", "cleaner-2", "cleaner-3"]


def test_export_pipeline_with_model():
    """Assert that the model's branch is used."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.scale()
    experionml.run("GNB")
    experionml.branch = "b2"
    experionml.normalize()
    assert len(experionml.export_pipeline(model="GNB")) == 2


def test_get_class_weight_regression():
    """Assert that an error is raised when called from regression tasks."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError):
        experionml.get_class_weight()


def test_get_class_weight():
    """Assert that the get_class_weight method returns a dict of the classes."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    assert list(experionml.get_class_weight()) == [0, 1, 2]


def test_get_class_weight_multioutput():
    """Assert that the get_class_weight method works for multioutput."""
    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    assert list(experionml.get_class_weight()) == ["a", "b", "c"]


def test_get_sample_weights_regression():
    """Assert that an error is raised when called from regression tasks."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError):
        experionml.get_sample_weight()


def test_get_sample_weight():
    """Assert that the get_sample_weight method returns a series."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    assert len(experionml.get_sample_weight()) == len(experionml.train)


def test_get_sample_weight_multioutput():
    """Assert that the get_sample_weight method works for multioutput."""
    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    assert len(experionml.get_sample_weight()) == len(experionml.train)


def test_get_seasonal_period_no_harmonics():
    """Assert that the get_seasonal_period returns a list of periods."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    assert experionml.get_seasonal_period(harmonics=None) == [12, 24, 36, 11, 48]


def test_get_seasonal_period_drop_harmonics():
    """Assert that the harmonics are dropped from the seasonal periods."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    assert experionml.get_seasonal_period(harmonics="drop") == [12, 11]


def test_get_seasonal_period_raw_strength_harmonics():
    """Assert that the strongest harmonics are kept in the seasonal periods."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    assert experionml.get_seasonal_period(harmonics="raw_strength") == [11, 48]


def test_get_seasonal_period_harmonic_strength_harmonics():
    """Assert that the strongest harmonics are pushed forward."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    assert experionml.get_seasonal_period(harmonics="harmonic_strength") == [48, 11]


def test_get_seasonal_period_no_periods():
    """Assert that an error is raised when no periods are detected."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    with pytest.raises(ValueError, match=".*No seasonal periods.*"):
        experionml.get_seasonal_period(max_sp=2)


def test_merge_invalid_class():
    """Assert that an error is raised when the class is not a trainer."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(TypeError, match=".*Expecting a.*"):
        experionml.merge(ExperionMLRegressor(X_reg, y_reg, random_state=1))


def test_merge_different_dataset():
    """Assert that an error is raised when the og dataset is different."""
    experionml_1 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_2 = ExperionMLClassifier(X10, y10, random_state=1)
    with pytest.raises(ValueError, match=".*different dataset.*"):
        experionml_1.merge(experionml_2)


def test_merge_adopts_metrics():
    """Assert that the metric of the merged instance is adopted."""
    experionml_1 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_2 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_2.run("Tree", metric="f1")
    experionml_1.merge(experionml_2)
    assert experionml_1.metric == "f1"


def test_merge_different_metrics():
    """Assert that an error is raised when the metrics are different."""
    experionml_1 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_1.run("Tree", metric="f1")
    experionml_2 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_2.run("Tree", metric="auc")
    with pytest.raises(ValueError, match=".*different metric.*"):
        experionml_1.merge(experionml_2)


def test_merge():
    """Assert that the merger handles branches, models and attributes."""
    experionml_1 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_1.run("Tree")
    experionml_2 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_2.branch.name = "b2"
    experionml_2.missing = ["missing"]
    experionml_2.run("LR")
    experionml_1.merge(experionml_2)
    assert list(experionml_1._branches) == [experionml_1.main, experionml_1.b2]
    assert experionml_1.models == ["Tree", "LR"]
    assert experionml_1.missing[-1] == "missing"


def test_merge_with_suffix():
    """Assert that the merger handles branches, models and attributes."""
    experionml_1 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_1.run(["Tree", "LDA"], n_trials=1, ht_params={"distributions": {"LDA": "test"}})
    experionml_2 = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml_2.run(["Tree", "LDA"], n_trials=1, ht_params={"distributions": {"LDA": "test"}})
    experionml_1.merge(experionml_2)
    assert list(experionml_1._branches) == [experionml_1.main, experionml_1.main2]
    assert experionml_1.models == ["Tree", "Tree2"]


def test_file_is_saved():
    """Assert that the pickle file is saved."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.save("auto")
    assert glob.glob("ExperionMLClassifier.pkl")


@patch("experionml.baserunner.pickle", MagicMock())
def test_save_data_false():
    """Assert that the dataset is restored after saving with save_data=False."""
    experionml = ExperionMLClassifier(X_bin, y_bin, holdout_size=0.1, random_state=1)
    experionml.save(filename="experionml", save_data=False)
    assert experionml.dataset is not None  # Dataset is restored after saving
    assert experionml.holdout is not None  # Holdout is restored after saving


def test_stacking():
    """Assert that the stacking method creates a Stack model."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, experionml.stacking)
    experionml.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    experionml.stacking()
    assert hasattr(experionml, "stack")
    assert "Stack" in experionml.models


def test_stacking_non_ensembles():
    """Assert that stacking ignores other ensembles."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    experionml.voting()
    experionml.stacking()
    assert len(experionml.stack.estimator.estimators) == 2  # No voting


def test_stacking_invalid_models():
    """Assert that an error is raised when <2 models."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    with pytest.raises(ValueError, match=".*contain at least two.*"):
        experionml.stacking()


def test_stacking_invalid_name():
    """Assert that an error is raised when the model already exists."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "Tree"])
    experionml.stacking()
    with pytest.raises(ValueError, match=".*multiple Stacking.*"):
        experionml.stacking()


def test_stacking_custom_models():
    """Assert that stacking can be created selecting the models."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, experionml.stacking)
    experionml.run(["LR", "LDA", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    experionml.stacking(models=["LDA", "LGB"])
    assert list(experionml.stack._models) == [experionml.lda, experionml.lgb]


def test_stacking_different_name():
    """Assert that the acronym is added in front of the new name."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    experionml.stacking(name="stack_1")
    experionml.stacking(name="_2")
    assert hasattr(experionml, "Stack_1")
    assert hasattr(experionml, "Stack_2")


def test_stacking_unknown_predefined_final_estimator():
    """Assert that an error is raised when the final estimator is unknown."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    with pytest.raises(ValueError, match=".*Unknown model.*"):
        experionml.stacking(final_estimator="invalid")


def test_stacking_invalid_predefined_final_estimator():
    """Assert that an error is raised when the final estimator is invalid."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    with pytest.raises(ValueError, match=".*can not perform.*"):
        experionml.stacking(final_estimator="OLS")


def test_stacking_predefined_final_estimator():
    """Assert that the final estimator accepts predefined models."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    experionml.stacking(final_estimator="LDA")
    assert isinstance(experionml.stack.estimator.final_estimator_, LDA)


def test_stacking_train_on_test():
    """Assert that the stacking model can be trained on the test set."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    experionml.stacking(train_on_test=True)
    assert experionml.stack.score("test") > 0.95


def test_voting():
    """Assert that the voting method creates a Vote model."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    pytest.raises(NotFittedError, experionml.voting)
    experionml.run(["LR", "LGB"], est_params={"LGB": {"n_estimators": 5}})
    experionml.voting(name="2")
    assert hasattr(experionml, "Vote2")
    assert "Vote2" in experionml.models


def test_voting_invalid_name():
    """Assert that an error is raised when the model already exists."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run(["LR", "Tree"])
    experionml.voting()
    with pytest.raises(ValueError, match=".*multiple Voting.*"):
        experionml.voting()


def test_voting_invalid_models():
    """Assert that an error is raised when <2 models."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR")
    with pytest.raises(ValueError, match=".*contain at least two.*"):
        experionml.voting()
    with pytest.raises(ValueError, match=".*contain at least two.*"):
        experionml.voting()
