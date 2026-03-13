
import glob
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from category_encoders.target_encoder import TargetEncoder
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MultiLabelBinarizer,
    OneHotEncoder,
    StandardScaler,
)
from sktime.transformations.series.impute import Imputer
from sktime.transformations.series.summarize import WindowSummarizer

from experionml import (
    ExperionMLClassifier,
    ExperionMLForecaster,
    ExperionMLRegressor,
)
from experionml.data_cleaning import Cleaner, Pruner
from experionml.training import DirectClassifier
from experionml.utils.utils import check_scaling

from .conftest import (
    X10,
    DummyTransformer,
    X10_dt,
    X10_nan,
    X10_str,
    X10_str2,
    X20_out,
    X_bin,
    X_class,
    X_ex,
    X_label,
    X_pa,
    X_reg,
    X_sparse,
    X_text,
    bin_sample_weight,
    y10,
    y10_label,
    y10_label2,
    y10_sn,
    y10_str,
    y_bin,
    y_class,
    y_ex,
    y_fc,
    y_label,
    y_multiclass,
    y_multireg,
    y_reg,
)


# Test __init__ ==================================================== >>


def test_task_assignment():
    """Assert that the correct task is assigned."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml.task.name == "binary_classification"

    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    assert experionml.task.name == "multiclass_classification"

    experionml = ExperionMLClassifier(X_label, y=y_label, random_state=1)
    assert experionml.task.name == "multilabel_classification"

    experionml = ExperionMLClassifier(X10, y=y10_label, stratify=None, random_state=1)
    assert experionml.task.name == "multilabel_classification"

    experionml = ExperionMLClassifier(X10, y=y10_label2, random_state=1)
    assert experionml.task.name == "multilabel_classification"

    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    assert experionml.task.name == "multiclass_multioutput_classification"

    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    assert experionml.task.name == "regression"

    experionml = ExperionMLRegressor(X_class, y=y_multiclass, random_state=1)
    assert experionml.task.name == "multioutput_regression"


def test_raise_one_target_value():
    """Assert that error raises when there is only one target value."""
    with pytest.raises(ValueError, match=".*1 target value.*"):
        ExperionMLClassifier(X_bin, [1] * len(X_bin), random_state=1)


def test_backend_with_n_jobs_1():
    """Assert that a warning is raised."""
    with pytest.warns(UserWarning, match=".*Leaving n_jobs=1 ignores.*"):
        ExperionMLClassifier(X_bin, y_bin, warnings=True, backend="threading", random_state=1)


# Test magic methods =============================================== >>


def test_init():
    """Assert that the __init__ method works for non-standard parameters."""
    experionml = ExperionMLClassifier(
        X_bin, y_bin, n_jobs=2, device="gpu", backend="multiprocessing"
    )
    assert experionml.device == "gpu"
    assert experionml.backend == "multiprocessing"


def test_repr():
    """Assert that the __repr__ method visualizes the pipeline(s)."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.scale()
    assert "Branches: main" in str(experionml)
    experionml.branch = "b2"
    assert "Branches:\n   --> main\n   --> b2 !" in str(experionml)


def test_iter():
    """Assert that we can iterate over experionml's pipeline."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.clean()
    experionml.impute()
    assert list(experionml) == list(experionml.pipeline.named_steps.values())


# Test utility properties =========================================== >>


def test_branch():
    """Assert that we can get the current branch."""
    experionml = ExperionMLClassifier(X10, y10, random_state=1)
    assert experionml.branch.name == "main"


def test_branch_same():
    """Assert that we can stay on the same branch."""
    experionml = ExperionMLClassifier(X10, y10, random_state=1)
    experionml.branch = "main"
    assert experionml.branch.name == "main"


def test_branch_change():
    """Assert that we can change to another branch."""
    experionml = ExperionMLClassifier(X10, y10, random_state=1)
    experionml.branch = "b2"
    experionml.clean()
    experionml.branch = "main"
    assert experionml.pipeline.steps == []  # Has no Cleaner


def test_branch_existing_name():
    """Assert that an error is raised when the name already exists."""
    experionml = ExperionMLClassifier(X10, y10, random_state=1)
    experionml.branch = "b2"
    with pytest.raises(ValueError, match=".*already exists.*"):
        experionml.branch = "b2_from_main"


def test_branch_unknown_parent():
    """Assert that an error is raised when the parent doesn't exist."""
    experionml = ExperionMLClassifier(X10, y10, random_state=1)
    with pytest.raises(ValueError, match=".*does not exist.*"):
        experionml.branch = "b2_from_invalid"


def test_branch_new():
    """Assert that we can create a new branch."""
    experionml = ExperionMLClassifier(X10, y10, random_state=1)
    experionml.clean()
    experionml.branch = "b2"
    assert len(experionml._branches) == 2


def test_branch_from_valid():
    """Assert that we can create a new branch, not from the current one."""
    experionml = ExperionMLClassifier(X10_nan, y10, random_state=1)
    experionml.branch = "b2"
    experionml.impute()
    experionml.branch = "b3_from_main"
    assert experionml.branch.name == "b3"
    assert experionml.n_nans > 0


def test_pos_label():
    """Assert that the pos_label property is set for all metrics."""
    experionml = ExperionMLClassifier(X_bin, y=[2 if i else 3 for i in y_bin], random_state=1)
    assert experionml.pos_label == 3

    experionml.pos_label = 2
    experionml.run("LR")
    assert experionml._metric[0]._kwargs["pos_label"] == 2


def test_pos_label_invalid_task():
    """Assert that the pos_label property is set for all metrics."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(ValueError, match=".*pos_label property can only be set.*"):
        experionml.pos_label = 0


def test_metadata():
    """Assert that the metadata property works."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert not experionml.metadata
    experionml.metadata = {"sample_weights": range(len(X_bin))}
    assert "sample_weights" in experionml.metadata


def test_missing():
    """Assert that the missing property returns the values considered 'missing'."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert "NA" in experionml.missing


def test_missing_setter():
    """Assert that we can change the missing property."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.missing = (1, 2)
    assert isinstance(experionml.missing, list)
    assert "NA" not in experionml.missing


def test_scaled():
    """Assert that scaled returns if the dataset is scaled."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert not experionml.scaled
    experionml.scale()
    assert experionml.scaled


def test_duplicates():
    """Assert that duplicates returns the number of duplicated samples."""
    experionml = ExperionMLClassifier(X10, y10, random_state=1)
    assert experionml.duplicates == 2


def test_nans():
    """Assert that the nans property returns a series of missing values."""
    experionml = ExperionMLClassifier(X10_nan, y10, random_state=1)
    assert experionml.nans.sum() == 2


def test_n_nans():
    """Assert that n_nans returns the number of rows with missing values."""
    experionml = ExperionMLClassifier(X10_nan, y10, random_state=1)
    assert experionml.n_nans == 2


def test_numerical():
    """Assert that numerical returns the names of the numerical columns."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    assert len(experionml.numerical) == 3


def test_n_numerical():
    """Assert that n_categorical returns the number of numerical columns."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    assert experionml.n_numerical == 3


def test_categorical():
    """Assert that categorical returns the names of categorical columns."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    assert len(experionml.categorical) == 1


def test_n_categorical():
    """Assert that n_categorical returns the number of categorical columns."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    assert experionml.n_categorical == 1


def test_outliers():
    """Assert that nans returns a series of outlier values."""
    experionml = ExperionMLClassifier(X20_out, y10 * 2, random_state=1)
    assert experionml.outliers.sum() == 2


def test_n_outliers():
    """Assert that n_outliers returns the number of rows with outliers."""
    experionml = ExperionMLClassifier(X20_out, y10 * 2, random_state=1)
    assert experionml.n_outliers == 2


def test_classes():
    """Assert that the classes property returns a df of the classes in y."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    assert list(experionml.classes.index) == [0, 1, 2]


def test_n_classes():
    """Assert that the n_classes property returns the number of classes."""
    experionml = ExperionMLClassifier(X_class, y_class, random_state=1)
    assert experionml.n_classes == 3


def test_unavailable_sparse_properties():
    """Assert that certain properties are unavailable for sparse datasets."""
    experionml = ExperionMLClassifier(X_sparse, y10, random_state=1)
    with pytest.raises(AttributeError):
        print(experionml.nans)
    with pytest.raises(AttributeError):
        print(experionml.n_nans)
    with pytest.raises(AttributeError):
        print(experionml.outliers)
    with pytest.raises(AttributeError):
        print(experionml.n_outliers)


def test_unavailable_regression_properties():
    """Assert that certain properties are unavailable for regression tasks."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError):
        print(experionml.classes)
    with pytest.raises(AttributeError):
        print(experionml.n_classes)


# Test utility methods ============================================= >>


def test_checks():
    """Assert that the checks method works as expected."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    checks = experionml.checks()
    assert isinstance(checks, pd.DataFrame)


@pytest.mark.parametrize("distributions", [None, "norm", ["norm", "pearson3"]])
def test_distribution(distributions):
    """Assert that the distribution method works as expected."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    dist = experionml.distributions(distributions=distributions, columns=(0, 1))
    assert isinstance(dist, pd.DataFrame)


@patch("sweetviz.analyze")
def test_eda_analyze(cls):
    """Assert that the eda method creates a report for one dataset."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.eda(rows="test", filename="report")
    cls.assert_called_once()


@patch("sweetviz.compare")
def test_eda_compare(cls):
    """Assert that the eda method creates a report for two datasets."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.eda(rows={"train": "train", "test": "test"})
    cls.assert_called_once()


def test_eda_invalid_rows():
    """Assert that an error is raised with more than two datasets."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*maximum number of.*"):
        experionml.eda(rows=("train", "test", "train"))


def test_inverse_transform():
    """Assert that the inverse_transform method works as intended."""
    experionml = ExperionMLClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    experionml.scale()
    experionml.impute()  # Does nothing, but doesn't crash either
    assert_frame_equal(experionml.inverse_transform(experionml.X), X_bin)


def test_inverse_transform_output():
    """Assert that the output type is determined by the data engine."""
    experionml = ExperionMLClassifier(X_bin, y_bin, engine="pyarrow", random_state=1)
    experionml.scale()
    assert isinstance(experionml.inverse_transform(X_bin), pa.Table)


def test_load_no_experionml():
    """Assert that an error is raised when the instance is not experionml."""
    trainer = DirectClassifier("LR", random_state=1)
    trainer.save("trainer")
    with pytest.raises(ValueError, match=".*ExperionMLClassifier, ExperionMLRegressor nor.*"):
        ExperionMLClassifier.load("trainer")


def test_load_already_contains_data():
    """Assert that an error is raised when data is provided without needed."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.save("experionml", save_data=True)
    with pytest.raises(ValueError, match=".*already contains data.*"):
        ExperionMLClassifier.load("experionml", data=(X_bin, y_bin))


def test_load_transform_data():
    """Assert that the data is transformed correctly."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.scale(columns=slice(3, 10))
    experionml.apply(np.exp, columns=2)
    experionml.feature_generation(strategy="dfs", n_features=5)
    experionml.feature_selection(strategy="sfm", solver="lgb", n_features=10)
    experionml.save("experionml", save_data=False)

    experionml_2 = ExperionMLClassifier.load("experionml", data=(X_bin, y_bin))
    assert experionml_2.dataset.shape == experionml.dataset.shape


def test_load_transform_data_multiple_branches():
    """Assert that the data is transformed with multiple branches."""
    experionml = ExperionMLClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    experionml.prune()
    experionml.branch = "b2"
    experionml.balance()
    experionml.feature_generation(strategy="dfs", n_features=5)
    experionml.branch = "b3"
    experionml.feature_selection(strategy="sfm", solver="lgb", n_features=20)
    experionml.save("experionml_2", save_data=False)

    experionml_2 = ExperionMLClassifier.load("experionml_2", data=(X_bin, y_bin))

    assert_frame_equal(experionml_2.og.X, X_bin)
    for branch in experionml._branches:
        assert_frame_equal(
            left=experionml_2._branches[branch.name]._data.data,
            right=experionml._branches[branch.name]._data.data,
            check_dtype=False,
        )


def test_reset():
    """Assert that the reset method deletes models and branches."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    experionml.scale()
    experionml.branch = "2"
    experionml.encode()
    experionml.run("LR")
    experionml.reset(hard=True)
    assert not experionml.models
    assert len(experionml._branches) == 1
    assert experionml["x2"].dtype.name == "object"  # Is reset back to str


def test_save_data():
    """Assert that the dataset is saved to a csv file."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.save_data("auto", rows="test")
    experionml.save_data("auto", rows=range(100))
    assert glob.glob("ExperionMLClassifier_test.csv")
    assert glob.glob("ExperionMLClassifier.csv")


def test_shrink_dtypes_excluded():
    """Assert that some dtypes are excluded from changing."""
    X = X_bin.copy()
    X["date"] = pd.date_range(start="1/1/2018", periods=len(X))

    experionml = ExperionMLClassifier(X, y_bin, random_state=1)
    assert experionml.dtypes[-2].name == "datetime64[ns]"
    experionml.shrink()
    assert experionml.dtypes[-2].name == "datetime64[ns]"  # Unchanged


def test_shrink_str2cat():
    """Assert that the str2cat parameter works as intended."""
    experionml = ExperionMLClassifier(X10_str2, y10, random_state=1)
    experionml.shrink(str2cat=False)
    assert experionml.dtypes[2].name == "string"

    experionml.shrink(str2cat=True)
    assert experionml.dtypes[2].name == "category"


def test_shrink_int2bool():
    """Assert that the int2bool parameter works as intended."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    assert experionml.dtypes[0].name == "int64"

    experionml.shrink(int2bool=True)
    assert experionml.dtypes[0].name == "boolean"


def test_shrink_int2uint():
    """Assert that the int2uint parameter works as intended."""
    experionml = ExperionMLClassifier(X10_str2, y10, random_state=1)
    assert experionml.dtypes[0].name == "int64"

    experionml.shrink(int2uint=False)
    assert experionml.dtypes[0].name == "Int8"

    experionml.shrink(int2uint=True)
    assert experionml.dtypes[0].name == "UInt8"


def test_shrink_sparse_arrays():
    """Assert that sparse arrays are also transformed."""
    experionml = ExperionMLClassifier(X_sparse, y10, random_state=1)
    assert experionml.dtypes[0].name == "Sparse[int64, 0]"
    experionml.shrink()
    assert experionml.dtypes[0].name == "Sparse[int8, 0]"


def test_shrink_dtypes_unchanged():
    """Assert that optimal dtypes are left unchanged."""
    experionml = ExperionMLClassifier(X_bin.astype("Float32"), y_bin, random_state=1)
    assert experionml.dtypes[3].name == "Float32"
    experionml.shrink()
    assert experionml.dtypes[3].name == "Float32"


def test_shrink_dense2sparse():
    """Assert that the dataset can be converted to sparse."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml.dtypes[0].name == "float64"
    experionml.shrink(dense2sparse=True)
    assert experionml.dtypes[0].name.startswith("Sparse[float32")


def test_shrink_pyarrow():
    """Assert that it works with pyarrow dtypes."""
    experionml = ExperionMLClassifier(X_pa, y_bin, engine="pandas-pyarrow", random_state=1)
    assert experionml.dtypes[0].name == "double[pyarrow]"
    experionml.shrink()
    assert experionml.dtypes[0].name == "float[pyarrow]"


def test_shrink_exclude_columns():
    """Assert that columns can be excluded."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    assert experionml.dtypes[0].name == "float64"
    assert experionml.dtypes[-1].name != "Int8"
    experionml.shrink(columns=-1)
    assert experionml.dtypes[0].name == "float64"
    assert experionml.dtypes[-1].name == "Int8"


def test_stats_mixed_sparse_dense():
    """Assert that stats show new information for mixed datasets."""
    X = X_sparse.copy()
    X["dense column"] = 2

    experionml = ExperionMLClassifier(X, y10, random_state=1)
    experionml.stats()


def test_status():
    """Assert that the status method prints an overview of the instance."""
    experionml = ExperionMLClassifier(*make_classification(100000), random_state=1)
    experionml.status()


def test_transform():
    """Assert that the transform method works as intended."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    experionml.encode(max_onehot=None)
    assert experionml.transform(X10_str)["x2"].dtype.kind in "ifu"


def test_transform_not_train_only():
    """Assert that train_only transformers are not used."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.prune(max_sigma=2)
    assert len(experionml.transform(X_bin)) == len(X_bin)


def test_transform_output():
    """Assert that the output type is determined by the data engine."""
    experionml = ExperionMLClassifier(X_bin, y_bin, engine="pyarrow", random_state=1)
    experionml.scale()
    assert isinstance(experionml.transform(X_bin), pa.Table)


# Test base transformers =========================================== >>


def test_add_after_model():
    """Assert that an error is raised when adding after training a model."""
    experionml = ExperionMLClassifier(X_bin, y_bin, verbose=1, random_state=1)
    experionml.run("Dummy")
    with pytest.raises(PermissionError, match=".*not allowed to add transformers.*"):
        experionml.scale()


def test_custom_params_to_method():
    """Assert that a custom parameter is passed to the method."""
    experionml = ExperionMLClassifier(X_bin, y_bin, verbose=1, random_state=1)
    experionml.scale(verbose=2)
    assert experionml.pipeline[0].verbose == 2


def test_add_basetransformer_params_are_attached():
    """Assert that the n_jobs and random_state params from experionml are used."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.add(PCA)  # When class
    experionml.add(PCA(random_state=2))  # When instance
    assert experionml.pipeline[0].get_params()["random_state"] == 1
    assert experionml.pipeline[1].get_params()["random_state"] == 2


def test_add_results_from_cache():
    """Assert that cached transformers are retrieved."""
    experionml = ExperionMLClassifier(X_bin, y_bin, memory=True, random_state=1)
    experionml.scale()

    experionml = ExperionMLClassifier(X_bin, y_bin, memory=True, random_state=1)
    experionml.scale()


def test_add_train_only():
    """Assert that experionml accepts transformers for the train set only."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.add(StandardScaler(), train_only=True)
    assert check_scaling(experionml.X_train)
    assert not check_scaling(experionml.X_test)

    len_train, len_test = len(experionml.train), len(experionml.test)
    experionml.add(Pruner(), train_only=True)
    assert len(experionml.train) != len_train
    assert len(experionml.test) == len_test


def test_add_complete_dataset():
    """Assert that experionml accepts transformers for the complete dataset."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.add(StandardScaler())
    assert check_scaling(experionml.dataset)

    len_dataset = len(experionml.dataset)
    experionml.add(Pruner())
    assert len(experionml.dataset) != len_dataset


def test_add_transformer_only_y():
    """Assert that experionml accepts transformers with only y."""
    experionml = ExperionMLClassifier(X10, y10_str, random_state=1)
    experionml.add(LabelEncoder())
    assert np.all((experionml["target"] == 0) | (experionml["target"] == 1))


def test_add_transformer_y_ignore_X():
    """Assert that experionml accepts transformers with y and default X."""
    experionml = ExperionMLClassifier(X10, y10_str, random_state=1)
    experionml.clean()  # Cleaner has X=None and y=None
    y = experionml.transform(y=y10_str)
    assert np.all((y == 0) | (y == 1))


def test_add_default_X_is_used():
    """Assert that X is autofilled when required but not provided."""
    experionml = ExperionMLClassifier(X10, y10_str, random_state=1)
    experionml.clean(columns=-1)
    assert experionml.mapping


def test_only_y_transformation():
    """Assert that only the target column can be transformed."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.scale(columns=-1)
    assert check_scaling(experionml.y)


def test_only_y_transformation_return_series():
    """Assert that the output is correctly converted to a dataframe."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.add(WindowSummarizer(), columns=-1)
    assert isinstance(experionml.y, pd.Series)
    assert isinstance(experionml.dataset, pd.DataFrame)


def test_only_y_transformation_multioutput():
    """Assert that only the target columns can be transformed for multioutput."""
    experionml = ExperionMLRegressor(X_reg, y=y_multireg, random_state=1)
    experionml.scale(columns=[-3, -1])
    assert check_scaling(experionml.y.iloc[:, [0, 2]])
    assert list(experionml.y.columns) == ["a", "b", "c"]


def test_X_and_y_transformation():
    """Assert that only the features are transformed when y is also provided."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.scale(columns=[-2, -1])
    assert check_scaling(experionml.X.iloc[:, -1])
    assert not check_scaling(experionml.y)


def test_returned_column_already_exists():
    """Assert that an error is raised if an existing column is returned."""

    def func_test(df):
        df["mean texture"] = 1
        return df

    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(ValueError, match=".*already exists in the original.*"):
        experionml.apply(func_test, columns="!mean texture")


def test_ignore_columns():
    """Assert that columns can be ignored from transformations."""
    experionml = ExperionMLRegressor(X_reg, y_reg, ignore="age", random_state=1)
    experionml.scale()
    experionml.run("OLS")
    assert "age" in experionml
    assert "age" not in experionml.pipeline.named_steps["scaler"].feature_names_in_
    assert "age" not in experionml.ols.estimator.feature_names_in_


def test_add_sparse_matrices():
    """Assert that transformers that return sp.matrix are accepted."""
    ohe = OneHotEncoder(handle_unknown="ignore").set_output(transform="default")
    experionml = ExperionMLClassifier(X10_str, y10, shuffle=False, random_state=1)
    experionml.add(ohe, columns=2)
    assert experionml.shape == (10, 8)  # Creates 4 extra columns


def test_add_keep_column_names():
    """Assert that the column names are kept after transforming."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)

    # Transformer has method get_feature_names_out
    experionml.add(TargetEncoder(return_df=False))
    assert experionml.features.tolist() == ["x0", "x1", "x2", "x3"]

    # Transformer keeps rows equal
    experionml.add(DummyTransformer(strategy="equal"), feature_names_out=None)
    assert experionml.features.tolist() == ["x0", "x1", "x2", "x3"]

    # Transformer drops rows
    experionml.add(DummyTransformer(strategy="drop"), feature_names_out=None)
    assert experionml.features.tolist() == ["x0", "x2", "x3"]

    # Transformer adds a new column
    experionml.add(DummyTransformer(strategy="add"), columns="!x2", feature_names_out=None)
    assert experionml.features.tolist() == ["x0", "x2", "x3", "x4"]


def test_raise_length_mismatch():
    """Assert that an error is raised when there's a mismatch in row length."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    with pytest.raises(IndexError, match=".*does not match length.*"):
        experionml.prune(columns=[2, 4])


def test_keep_pyarrow_dtypes():
    """Assert that columns keep the pyarrow dtype."""
    experionml = ExperionMLClassifier(X_pa, y_bin, random_state=1)
    assert isinstance(experionml.dtypes[0], pd.ArrowDtype)
    experionml.scale()
    assert isinstance(experionml.dtypes[0], pd.ArrowDtype)


def test_add_derivative_columns_keep_position():
    """Assert that derivative columns go after the original."""
    experionml = ExperionMLClassifier(X10_str, y10, shuffle=False, random_state=1)
    experionml.encode(columns="x2")
    assert list(experionml.columns[2:5]) == ["x2_b", "x2_a", "x2_c"]


def test_multioutput_y_return():
    """Assert that y returns a dataframe when multioutput."""
    experionml = ExperionMLClassifier(X10, y10_label, stratify=None, random_state=1)
    experionml.add(Cleaner())
    assert isinstance(experionml.y, pd.DataFrame)

    experionml = ExperionMLClassifier(X10, y10_label, stratify=None, random_state=1)
    experionml.add(MultiLabelBinarizer())
    assert isinstance(experionml.y, pd.DataFrame)


def test_add_sets_are_kept_equal():
    """Assert that the train and test sets always keep the same rows."""
    experionml = ExperionMLClassifier(X_bin, y_bin, index=True, random_state=1)
    train_idx, test_idx = experionml.train.index, experionml.test.index
    experionml.add(Pruner())
    assert all(idx in train_idx for idx in experionml.train.index)
    assert_index_equal(test_idx, experionml.test.index)


def test_add_reset_index():
    """Assert that the indices are reset when index=False."""
    experionml = ExperionMLClassifier(X_bin, y_bin, index=False, random_state=1)
    experionml.prune()
    assert list(experionml.dataset.index) == list(range(len(experionml.dataset)))


def test_add_raise_duplicate_indices():
    """Assert that an error is raised when indices are duplicated."""

    class AddRowsTransformer(BaseEstimator):
        def transform(self, X, y):
            return pd.concat([X, X.iloc[:5]]), pd.concat([y, y.iloc[:5]])

    experionml = ExperionMLClassifier(X_bin, y_bin, index=True, random_state=1)
    with pytest.raises(ValueError, match=".*Duplicate indices.*"):
        experionml.add(AddRowsTransformer)


def test_add_params_to_method():
    """Assert that experionml's parameters are passed to the method."""
    experionml = ExperionMLClassifier(X_bin, y_bin, verbose=1, random_state=1)
    experionml.scale()
    assert experionml.pipeline[0].verbose == 1


def test_add_wrap_fit():
    """Assert that sklearn attributes are added to the estimator."""
    experionml = ExperionMLForecaster(X_ex, y=y_ex, random_state=1)
    experionml.add(Imputer())
    assert hasattr(experionml.pipeline[0], "feature_names_in_")
    assert hasattr(experionml.pipeline[0], "n_features_in_")

    # Check there's no double wrapping
    experionml.add(Imputer())
    assert hasattr(experionml.pipeline[1].fit, "__wrapped__")
    assert not hasattr(experionml.pipeline[1].fit.__wrapped__, "__wrapped__")


def test_add_wrap_get_feature_names_out_one_to_one():
    """Assert that get_feature_names_out is added to the estimator."""
    experionml = ExperionMLForecaster(X_ex, y=y_ex, random_state=1)
    experionml.add(Imputer(), feature_names_out="one-to-one")
    assert hasattr(experionml.pipeline[0], "get_feature_names_out")
    assert list(experionml.pipeline[0].get_feature_names_out()) == list(X_ex.columns)


def test_add_wrap_get_feature_names_out_callable():
    """Assert that get_feature_names_out is added to the estimator."""
    experionml = ExperionMLForecaster(y_fc, random_state=1)
    experionml.add(Imputer(), columns=-1, feature_names_out=lambda _: ["test"])
    assert hasattr(experionml.pipeline[0], "get_feature_names_out")
    assert list(experionml.pipeline[0].get_feature_names_out()) == ["test"]


def test_add_with_sample_weights():
    """Assert that sample weights are passed to the method."""
    experionml = ExperionMLClassifier(X_bin, y=y_bin, metadata=bin_sample_weight, random_state=1)
    experionml.scale()
    assert experionml.pipeline[0].get_metadata_routing()._serialize()["fit"]["sample_weight"]


def test_add_pipeline():
    """Assert that adding a pipeline adds every step."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("sfm", SelectFromModel(RandomForestClassifier())),
        ],
    )
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.add(pipeline)
    assert isinstance(experionml.pipeline[0], StandardScaler)
    assert isinstance(experionml.pipeline[1], SelectFromModel)


def test_attributes_are_attached():
    """Assert that the transformer's attributes are attached to the branch."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.branch = "b2"
    experionml.scale()
    assert hasattr(experionml, "standard_")
    experionml.branch = "main"
    assert not hasattr(experionml, "standard_")


def test_apply():
    """Assert that a function can be applied to the dataset."""
    experionml = ExperionMLClassifier(X_bin, y_bin, shuffle=False, random_state=1)
    experionml.apply(np.exp, columns=0)
    assert experionml.iloc[0, 0] == np.exp(X_bin.iloc[0, 0])


# Test data cleaning transformers =================================== >>


def test_balance_wrong_task():
    """Assert that an error is raised for regression and multioutput tasks."""
    # For regression tasks
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        experionml.balance()

    # For multioutput tasks
    experionml = ExperionMLClassifier(X_class, y=y_multiclass, random_state=1)
    with pytest.raises(AttributeError, match=".*has no attribute.*"):
        experionml.balance()


def test_balance_with_sample_weight():
    """Assert that an error is raised when sample weights are provided."""
    experionml = ExperionMLClassifier(X_bin, y_bin, metadata=bin_sample_weight, random_state=1)
    with pytest.raises(PermissionError, match=".*not support sample weights.*"):
        experionml.balance()


def test_balance():
    """Assert that the balance method balances the training set."""
    experionml = ExperionMLClassifier(X10, y10_str, random_state=1)
    experionml.clean()  # To have column mapping
    experionml.balance(strategy="NearMiss")
    assert (experionml.y_train == 0).sum() == (experionml.y_train == 1).sum()


def test_clean():
    """Assert that the clean method cleans the dataset."""
    experionml = ExperionMLClassifier(X10, y10_sn, stratify=None, random_state=1)
    experionml.clean()
    assert len(experionml.dataset) == 9
    assert experionml.mapping == {"target": {"n": 0, "y": 1}}


def test_decompose():
    """Assert that the decompose method works."""
    experionml = ExperionMLForecaster(y_fc, sp=12, random_state=1)
    experionml.decompose(columns=-1)
    assert experionml.dataset.iloc[0, 0] != experionml.og.dataset.iloc[0, 0]


def test_discretize():
    """Assert that the discretize method bins the numerical columns."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.discretize()
    assert all(dtype.name == "object" for dtype in experionml.X.dtypes)


def test_encode():
    """Assert that the encode method encodes all categorical columns."""
    experionml = ExperionMLClassifier(X10_str, y10, random_state=1)
    experionml.encode()
    assert all(experionml.X[col].dtype.kind in "ifu" for col in experionml.X.columns)


def test_impute():
    """Assert that the impute method imputes all missing values."""
    experionml = ExperionMLClassifier(X10_nan, y10, random_state=1)
    experionml.impute()
    assert experionml.dataset.isna().sum().sum() == 0


def test_normalize():
    """Assert that the normalize method transforms the features."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    X = experionml.X
    experionml.normalize()
    assert not experionml.X.equals(X)


def test_prune():
    """Assert that the prune method handles outliers in the training set."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    len_train, len_test = len(experionml.train), len(experionml.test)
    experionml.prune(strategy="lof")
    assert len(experionml.train) != len_train
    assert len(experionml.test) == len_test


def test_scale():
    """Assert that the scale method normalizes the features."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.scale()
    assert check_scaling(experionml.dataset)


# Test nlp transformers ============================================ >>


def test_textclean():
    """Assert that the textclean method cleans the corpus."""
    experionml = ExperionMLClassifier(X_text, y10, shuffle=False, random_state=1)
    experionml.textclean()
    assert experionml["corpus"][0] == "i am in new york"


def test_textnormalize():
    """Assert that the textnormalize method normalizes the corpus."""
    experionml = ExperionMLClassifier(X_text, y10, shuffle=False, random_state=1)
    experionml.textnormalize(stopwords=False, custom_stopwords=["yes"], lemmatize=False)
    assert experionml["corpus"][0] == ["I", "àm", "in", "ne'w", "york"]


def test_tokenize():
    """Assert that the tokenize method tokenizes the corpus."""
    experionml = ExperionMLClassifier(X_text, y10, shuffle=False, random_state=1)
    experionml.tokenize()
    assert experionml["corpus"][0] == ["I", "àm", "in", "ne", "'", "w", "york"]


def test_vectorize():
    """Assert that the vectorize method converts the corpus to numerical."""
    experionml = ExperionMLClassifier(X_text, y10, test_size=0.25, random_state=1)
    experionml.vectorize(strategy="hashing", n_features=5)
    assert "corpus" not in experionml
    assert experionml.shape == (10, 6)


# Test feature engineering transformers ============================ >>


def test_feature_extraction():
    """Assert that the feature_extraction method creates datetime features."""
    experionml = ExperionMLClassifier(X10_dt, y10, random_state=1)
    experionml.feature_extraction(fmt="%d/%m/%Y")
    assert experionml.X.shape[1] == 6


def test_feature_generation():
    """Assert that the feature_generation method creates extra features."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.feature_generation(n_features=2)
    assert experionml.X.shape[1] == X_bin.shape[1] + 2


def test_feature_grouping():
    """Assert that the feature_grouping method group features."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.feature_grouping({"g1": [0, 1], "g2": "mean.*"})
    assert experionml.n_features == 32


def test_default_solver_univariate():
    """Assert that the default solver is selected for strategy="univariate"."""
    # For classification tasks
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert experionml.pipeline[0].univariate_.score_func.__name__ == "f_classif"

    # For regression tasks
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.feature_selection(strategy="univariate", solver=None, n_features=8)
    assert experionml.pipeline[0].univariate_.score_func.__name__ == "f_regression"


def test_default_solver_from_task():
    """Assert that the solver is inferred from the task when a model is selected."""
    # For classification tasks
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.feature_selection(strategy="rfe", solver="tree", n_features=8)
    assert experionml.pipeline[0].rfe_.estimator_.__class__.__name__ == "DecisionTreeClassifier"

    # For regression tasks
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.feature_selection(strategy="rfe", solver="tree", n_features=25)
    assert experionml.pipeline[0].rfe_.estimator_.__class__.__name__ == "DecisionTreeRegressor"


@patch("experionml.feature_engineering.SequentialFeatureSelector", MagicMock())
def test_default_scoring():
    """Assert that the scoring is experionml's metric when exists."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("lr", metric="recall")
    experionml.branch = "fs_branch"
    experionml.feature_selection(strategy="sfs", solver="lgb", n_features=25)
    assert experionml.pipeline[0].kwargs["scoring"].name == "recall"


# Test training methods ============================================ >>


def test_non_numerical_target_column():
    """Assert that an error is raised when the target column is categorical."""
    experionml = ExperionMLClassifier(X10, y10_str, random_state=1)
    with pytest.raises(ValueError, match=".*target column is not numerical.*"):
        experionml.run("Tree")


def test_assign_existing_metric():
    """Assert that the existing metric_ is assigned if rerun."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR", metric="recall")
    experionml.run("Tree")
    assert experionml.metric == "recall"


def test_raises_invalid_metric_consecutive_runs():
    """Assert that an error is raised for a different metric."""
    experionml = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    experionml.run("LR", metric="recall")
    pytest.raises(ValueError, experionml.run, "Tree", metric="f1")


def test_scaling_is_passed():
    """Assert that the scaling is passed to experionml."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.scale("minmax")
    experionml.run("LGB")
    assert_frame_equal(experionml.dataset, experionml.lgb.dataset)


def test_models_are_replaced():
    """Assert that models with the same name are replaced."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run(["OLS", "Tree"])
    experionml.run("OLS")
    assert experionml.models == ["Tree", "OLS"]


def test_models_and_metric_are_updated():
    """Assert that the models and metric attributes are updated correctly."""
    experionml = ExperionMLRegressor(X_reg, y_reg, random_state=1)
    experionml.run(["OLS", "Tree"], metric=get_scorer("max_error"))
    assert experionml.models == ["OLS", "Tree"]
    assert experionml.metric == "max_error"
    assert experionml.models == ["OLS", "Tree"]
    assert experionml.metric == "max_error"
