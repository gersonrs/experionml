from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import TypeVar

from beartype import beartype
from joblib.memory import Memory
from sklearn.base import clone

from experionml.experionml import ExperionML
from experionml.utils.types import (
    Backend,
    Bool,
    ColumnSelector,
    Engine,
    IndexSelector,
    Int,
    IntLargerEqualZero,
    MetadataDict,
    NJobs,
    Predictor,
    Scalar,
    Seasonality,
    SPDict,
    Verbose,
    Warnings,
    YSelector,
)
from experionml.utils.utils import Goal


T_Predictor = TypeVar("T_Predictor", bound=Predictor)


@beartype
def ExperionMLModel(
    estimator: T_Predictor,
    name: str | None = None,
    *,
    acronym: str | None = None,
    needs_scaling: Bool = False,
    native_multilabel: Bool = False,
    native_multioutput: Bool = False,
    validation: str | None = None,
) -> T_Predictor:
    """Converte um estimador em um modelo que pode ser consumido pelo experionml.

    Esta função adiciona ao estimador as tags relevantes para que ele
    possa ser usado pelo `experionml`. Note que apenas estimadores que
    seguem a [API do sklearn][api] são compatíveis.

    Leia mais sobre modelos personalizados no [guia do usuário][custom-models].

    Parameters
    ----------
    estimator: Predictor
        Estimador personalizado. Deve implementar os métodos `fit` e `predict`.

    name: str or None, default=None
        Nome do modelo. Este é o valor usado para chamar o modelo no
        experionml. O valor deve começar com o `acronym` do modelo,
        quando especificado. Se for None, são usadas as letras maiúsculas
        do nome do estimador, desde que haja duas ou mais; caso contrário,
        usa-se o nome inteiro.

    acronym: str or None, default=None
        Sigla do modelo. Se for None, usa o `name` do modelo. Especifique
        este parâmetro quando quiser treinar múltiplos modelos personalizados
        que compartilham o mesmo estimador.

    needs_scaling: bool, default=False
        Indica se o modelo deve usar [escalonamento automático de atributos][].

    native_multilabel: bool, default=False
        Indica se o modelo possui suporte nativo a tarefas [multilabel][].
        Se for False e a tarefa for multilabel, um metaestimador multilabel
        será encapsulado ao redor do estimador.

    native_multioutput: bool, default=False
        Indica se o modelo possui suporte nativo a tarefas [multioutput tasks][].
        Se for False e a tarefa for multioutput, um metaestimador
        multioutput será encapsulado ao redor do estimador.

    validation: str or None, default=None
                Indica se o modelo permite [validação durante o treinamento][].

                - Se for None: Não há suporte para validação durante o treinamento.
                - Se for str: Nome do parâmetro do estimador que define o número
                    de iterações, por exemplo, `n_estimators` para
                    [RandomForestClassifier][].

    Returns
    -------
    Predictor
        Estimador com as informações fornecidas. Passe esta instância
        para o parâmetro `models` do método [run][experionmlclassifier-run].

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor, ExperionMLModel
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import RANSACRegressor

    ransac = ExperionMLModel(
        estimator=RANSACRegressor(),
        name="RANSAC",
        needs_scaling=False,
    )

    X, y = load_diabetes(return_X_y=True, as_frame=True)

    experionml = ExperionMLRegressor(X, y, verbose=2)
    experionml.run(ransac)
    ```

    """
    if callable(estimator):
        estimator_c = estimator()
    else:
        estimator_c = clone(estimator)

    if name:
        estimator_c.name = name
    if acronym:
        estimator_c.acronym = acronym
    estimator_c.needs_scaling = needs_scaling
    estimator_c.native_multilabel = native_multilabel
    estimator_c.native_multioutput = native_multioutput
    estimator_c.validation = validation

    return estimator_c


@beartype
class ExperionMLClassifier(ExperionML):
    """Classe principal para tarefas de classificação.

    Aplica todas as transformações de dados e o gerenciamento de modelos
    fornecidos pelo pacote sobre um conjunto de dados. Note que, ao
    contrário da API do sklearn, a instância contém o conjunto de dados
    sobre o qual a análise será executada. Ao chamar um método, ele será
    aplicado automaticamente ao conjunto de dados contido na instância.

    Todas as funcionalidades de [limpeza de dados][], [engenharia de
    atributos][], [treinamento de modelos][training] e [visualização][plots]
    podem ser acessadas a partir de uma instância desta classe.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing features and target. Allowed formats are:

        - X
        - X, y
        - train, test
        - train, test, holdout
        - X_train, X_test, y_train, y_test
        - X_train, X_test, X_holdout, y_train, y_test, y_holdout
        - (X_train, y_train), (X_test, y_test)
        - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

        **X, train, test: dataframe-like**<br>
        Feature set with shape=(n_samples, n_features).

        **y: int, str, sequence or dataframe-like**<br>
        Target column(s) corresponding to `X`.

        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe-like: Target columns for multioutput tasks.

    y: int, str, sequence or dataframe-like, default=-1
        Target column(s) corresponding to `X`.

        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

        This parameter is ignored if the target column is provided
        through `arrays`.

    index: bool, int, str or sequence, default=False
        Handle the index in the resulting dataframe.

        - If False: Reset to [RangeIndex][].
        - If True: Use the provided index.
        - If int: Position of the column to use as index.
        - If str: Name of the column to use as index.
        - If sequence: Array with shape=(n_samples,) to use as index.

    metadata: dict or None, default=None
        [Metadata][] to route to estimators, scorers, and CV splitters.
        If None, no metadata is used. If dict, the available keys are:

        - groups: sequence of shape=(n_samples,)
            Group labels for the samples used while splitting the
            dataset into train and test sets.
        - sample_weight: sequence of shape=(n_samples,)
            Individual weights for each sample.

    ignore: int, str, sequence or None, default=None
        Features in X to ignore during data transformations and model
        training. The features are still used in the remaining methods.

    test_size: int or float, default=0.2
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        This parameter is ignored if the test set is provided
        through `arrays`.

        If 'groups' is provided in the `metadata` parameter, `test_size`
        represents the proportion of groups to include in the test split
        or the absolute number of test groups.

    holdout_size: int, float or None, default=None
        - If None: No holdout data set is kept apart.
        - If <=1: Fraction of the dataset to include in the holdout set.
        - If >1: Number of rows to include in the holdout set.

        This parameter is ignored if the holdout set is provided
        through `arrays`.

    shuffle: bool, default=True
        Whether to shuffle the dataset before splitting the data sets.

    stratify: int, str or None, default=-1
        Handle stratification of the target classes over the data sets.

        - If None: No stratification is applied.
        - If int: Position of the column to use for stratification.
        - If str: Name of the column to use for stratification.

        The stratification column can't contain `NaN` values.

        This parameter is ignored if `shuffle=False` or if the test
        set is provided through `arrays`.

    n_rows: int or float, default=1
        Random subsample of the dataset to use. The default value selects
        all rows.

        - If <=1: Fraction of the dataset to select.
        - If >1: Exact number of rows to select. Only if `arrays` is X
                 or X, y.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str, dict or None, default=None
        Execution engine to use for [data][data-engines] and
        [estimators][estimator-acceleration]. The value should be
        one of the possible values to change one of the two engines,
        or a dictionary with keys `data` and `estimator`, with their
        corresponding choice as values to change both engines. If
        None, the default values are used. Choose from:

        - "data":

            - "numpy"
            - "pandas" (default)
            - "pandas-pyarrow"
            - "polars"
            - "polars-lazy"
            - "pyarrow"
            - "modin"
            - "dask"
            - "pyspark"
            - "pyspark-pandas"

        - "estimator":

            - "sklearn" (default)
            - "sklearnex"
            - "cuml"

    backend: str, default="loky"
        Parallelization backend. Read more in the
        [user guide][parallel-execution]. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than `loky`.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.
        - "dask": Multi-node, process-based parallelism.

    memory: bool, str, Path or Memory, default=False
        Enables caching for memory optimization. Read more in the
        [user guide][memory-considerations].

        - If False: No caching is performed.
        - If True: A default temp directory is used.
        - If str: Path to the caching directory.
        - If Path: A [pathlib.Path][] to the caching directory.
        - If Memory: Object with the [joblib.Memory][] interface.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "once").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ExperionML can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic name.
        - If Path: A [pathlib.Path][] to the log file.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the [mlflow experiment][experiment] to use for tracking.
        If None, no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    See Also
    --------
    experionml.api:ExperionMLForecaster
    experionml.api:ExperionMLRegressor

    Examples
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Inicializa o experionml
    experionml = ExperionMLClassifier(X, y, verbose=2)

    # Aplica métodos de limpeza de dados e engenharia de atributos
    experionml.balance(strategy="smote")
    experionml.feature_selection(strategy="rfe", solver="lr", n_features=22)

    # Treina os modelos
    experionml.run(models=["LR", "RF", "XGB"])

    # Analisa os resultados
    experionml.results
    ```

    """

    _goal = Goal.classification

    def __init__(
        self,
        *arrays,
        y: YSelector = -1,
        index: IndexSelector = False,
        metadata: MetadataDict | None = None,
        ignore: ColumnSelector | None = None,
        shuffle: Bool = True,
        stratify: Int | str | None = -1,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            arrays=arrays,
            y=y,
            index=index,
            metadata=metadata,
            ignore=ignore,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=shuffle,
            stratify=stratify,
            n_rows=n_rows,
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            memory=memory,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )


@beartype
class ExperionMLForecaster(ExperionML):
    """Classe principal para tarefas de previsão.

    Aplica todas as transformações de dados e o gerenciamento de modelos
    fornecidos pelo pacote sobre um conjunto de dados. Note que, ao
    contrário da API do sklearn, a instância contém o conjunto de dados
    sobre o qual a análise será executada. Ao chamar um método, ele será
    aplicado automaticamente ao conjunto de dados contido na instância.

    Todas as funcionalidades de [limpeza de dados][], [engenharia de
    atributos][], [treinamento de modelos][training] e [visualização][plots]
    podem ser acessadas a partir de uma instância desta classe.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing exogenous features and time series. Allowed
        formats are:

        - X
        - y
        - X, y
        - train, test
        - train, test, holdout
        - X_train, X_test, y_train, y_test
        - X_train, X_test, X_holdout, y_train, y_test, y_holdout
        - (X_train, y_train), (X_test, y_test)
        - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

        **X, train, test: dataframe-like**<br>
        Exogenous feature set corresponding to y, with shape=(n_samples,
        n_features).

        **y: int, str, sequence or dataframe-like**<br>
        Time series.

        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe-like: Target columns for multioutput tasks.

    y: int, str, sequence or dataframe-like, default=-1
        Time series.

        - If None: `y` is ignored.
        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe-like: Target columns for multioutput tasks.

        This parameter is ignored if the time series is provided
        through `arrays`.

    metadata: dict or None, default=None
        [Metadata][] to route to estimators, scorers, and CV splitters.
        If None, no metadata is used. If dict, the available keys are:

        - sample_weight: sequence of shape=(n_samples,)
            Individual weights for each sample.

    ignore: int, str, sequence or None, default=None
        Exogenous features in X to ignore during data transformations
        and model training. The features are still used in the remaining
        methods.

    sp: int, str, sequence, dict or None, default=None
        [Seasonality][] of the time series.

        - If None: No seasonality.
        - If int: Seasonal period, e.g., 7 for weekly data, and 12 for
          monthly data. The value must be >=2.
        - If str:

            - Seasonal period provided as [PeriodAlias][], e.g., "M" for
              12 or "H" for 24.
            - "index": The frequency of the data index is mapped to a
              seasonal period.
            - "infer": Automatically infer the seasonal period from the
              data (calls [get_seasonal_period][self-get_seasonal_period]
              under the hood, using default parameters).

        - If sequence: Multiple seasonal periods provided as int or str.
        - If dict: Dictionary with keys:

            - "sp": Seasonal period provided as one of the aforementioned
              options.
            - "seasonal_model" (optional): "additive" or "multiplicative".
            - "trend_model" (optional): "additive" or "multiplicative".

    test_size: int or float, default=0.2
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        This parameter is ignored if the test set is provided
        through `arrays`.

    holdout_size: int, float or None, default=None
        - If None: No holdout data set is kept apart.
        - If <=1: Fraction of the dataset to include in the holdout set.
        - If >1: Number of rows to include in the holdout set.

        This parameter is ignored if the holdout set is provided
        through `arrays`.

    n_rows: int or float, default=1
        Subsample of the dataset to use. The cut is made from the head
        of the dataset (older entries are dropped when sorted by date
        ascending). The default value selects all rows.

        - If <=1: Fraction of the dataset to select.
        - If >1: Exact number of rows to select. Only if `arrays` is X
                 or X, y.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str, dict or None, default=None
        Execution engine to use for [data][data-engines] and
        [estimators][estimator-acceleration]. The value should be
        one of the possible values to change one of the two engines,
        or a dictionary with keys `data` and `estimator`, with their
        corresponding choice as values to change both engines. If
        None, the default values are used. Choose from:

        - "data":

            - "numpy"
            - "pandas" (default)
            - "pandas-pyarrow"
            - "polars"
            - "polars-lazy"
            - "pyarrow"
            - "modin"
            - "dask"
            - "pyspark"
            - "pyspark-pandas"

        - "estimator":

            - "sklearn" (default)
            - "sklearnex"
            - "cuml"

    backend: str, default="loky"
        Parallelization backend. Read more in the
        [user guide][parallel-execution]. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than `loky`.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.
        - "dask": Multi-node, process-based parallelism.

    memory: bool, str, Path or Memory, default=False
        Enables caching for memory optimization. Read more in the
        [user guide][memory-considerations].

        - If False: No caching is performed.
        - If True: A default temp directory is used.
        - If str: Path to the caching directory.
        - If Path: A [pathlib.Path][] to the caching directory.
        - If Memory: Object with the [joblib.Memory][] interface.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "once").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ExperionML can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic name.
        - If Path: A [pathlib.Path][] to the log file.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the [mlflow experiment][experiment] to use for tracking.
        If None, no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    See Also
    --------
    experionml.api:ExperionMLClassifier
    experionml.api:ExperionMLRegressor

    Examples
    --------
    ```pycon
    from experionml import ExperionMLForecaster
    from sktime.datasets import load_airline

    y = load_airline()

    # Inicializa o experionml
    experionml = ExperionMLForecaster(y, verbose=2)

    # Treina os modelos
    experionml.run(models=["NF", "ES", "ETS"])

    # Analisa os resultados
    experionml.results
    ```

    """

    _goal = Goal.forecast

    def __init__(
        self,
        *arrays,
        y: YSelector = -1,
        metadata: MetadataDict | None = None,
        ignore: ColumnSelector | None = None,
        sp: Seasonality | SPDict = None,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            arrays=arrays,
            y=y,
            index=True,
            metadata=metadata,
            ignore=ignore,
            sp=sp,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=False,
            stratify=None,
            n_rows=n_rows,
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            memory=memory,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )


@beartype
class ExperionMLRegressor(ExperionML):
    """Classe principal para tarefas de regressão.

    Aplica todas as transformações de dados e o gerenciamento de modelos
    fornecidos pelo pacote sobre um conjunto de dados. Note que, ao
    contrário da API do sklearn, a instância contém o conjunto de dados
    sobre o qual a análise será executada. Ao chamar um método, ele será
    aplicado automaticamente ao conjunto de dados contido na instância.

    Todas as funcionalidades de [limpeza de dados][], [engenharia de
    atributos][], [treinamento de modelos][training] e [visualização][plots]
    podem ser acessadas a partir de uma instância desta classe.

    Parameters
    ----------
    *arrays: sequence of indexables
        Dataset containing features and target. Allowed formats are:

        - X
        - X, y
        - train, test
        - train, test, holdout
        - X_train, X_test, y_train, y_test
        - X_train, X_test, X_holdout, y_train, y_test, y_holdout
        - (X_train, y_train), (X_test, y_test)
        - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

        **X, train, test: dataframe-like**<br>
        Feature set with shape=(n_samples, n_features).

        **y: int, str, sequence or dataframe-like**<br>
        Target column(s) corresponding to `X`.

        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe: Target columns for multioutput tasks.

    y: int, str, sequence or dataframe-like, default=-1
        Target column(s) corresponding to `X`.

        - If None: `y` is ignored.
        - If int: Position of the target column in `X`.
        - If str: Name of the target column in `X`.
        - If sequence: Target column with shape=(n_samples,) or
          sequence of column names or positions for multioutput tasks.
        - If dataframe-like: Target columns for multioutput tasks.

        This parameter is ignored if the target column is provided
        through `arrays`.

    index: bool, int, str or sequence, default=False
        Handle the index in the resulting dataframe.

        - If False: Reset to [RangeIndex][].
        - If True: Use the provided index.
        - If int: Position of the column to use as index.
        - If str: Name of the column to use as index.
        - If sequence: Array with shape=(n_samples,) to use as index.

    metadata: dict or None, default=None
        [Metadata][] to route to estimators, scorers, and CV splitters.
        If None, no metadata is used. If dict, the available keys are:

        - groups: sequence of shape=(n_samples,)
            Group labels for the samples used while splitting the
            dataset into train and test sets.
        - sample_weight: sequence of shape=(n_samples,)
            Individual weights for each sample.

    ignore: int, str, sequence or None, default=None
        Features in X to ignore during data transformations and model
        training. The features are still used in the remaining methods.

    test_size: int or float, default=0.2
        - If <=1: Fraction of the dataset to include in the test set.
        - If >1: Number of rows to include in the test set.

        This parameter is ignored if the test set is provided
        through `arrays`.

        If 'groups' is provided in the `metadata` parameter, `test_size`
        represents the proportion of groups to include in the test split
        or the absolute number of test groups.

    holdout_size: int, float or None, default=None
        - If None: No holdout data set is kept apart.
        - If <=1: Fraction of the dataset to include in the holdout set.
        - If >1: Number of rows to include in the holdout set.

        This parameter is ignored if the holdout set is provided
        through `arrays`.

    shuffle: bool, default=True
        Whether to shuffle the dataset before splitting the data sets.

    n_rows: int or float, default=1
        Random subsample of the dataset to use. The default value selects
        all rows.

        - If <=1: Fraction of the dataset to select.
        - If >1: Exact number of rows to select. Only if `arrays` is X
                 or X, y.

    n_jobs: int, default=1
        Number of cores to use for parallel processing.

        - If >0: Number of cores to use.
        - If -1: Use all available cores.
        - If <-1: Use number of cores - 1 + `n_jobs`.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str, dict or None, default=None
        Execution engine to use for [data][data-engines] and
        [estimators][estimator-acceleration]. The value should be
        one of the possible values to change one of the two engines,
        or a dictionary with keys `data` and `estimator`, with their
        corresponding choice as values to change both engines. If
        None, the default values are used. Choose from:

        - "data":

            - "numpy"
            - "pandas" (default)
            - "pandas-pyarrow"
            - "polars"
            - "polars-lazy"
            - "pyarrow"
            - "modin"
            - "dask"
            - "pyspark"
            - "pyspark-pandas"

        - "estimator":

            - "sklearn" (default)
            - "sklearnex"
            - "cuml"

    backend: str, default="loky"
        Parallelization backend. Read more in the
        [user guide][parallel-execution]. Choose from:

        - "loky": Single-node, process-based parallelism.
        - "multiprocessing": Legacy single-node, process-based
          parallelism. Less robust than `loky`.
        - "threading": Single-node, thread-based parallelism.
        - "ray": Multi-node, process-based parallelism.
        - "dask": Multi-node, process-based parallelism.

    memory: bool, str, Path or Memory, default=False
        Enables caching for memory optimization. Read more in the
        [user guide][memory-considerations].

        - If False: No caching is performed.
        - If True: A default temp directory is used.
        - If str: Path to the caching directory.
        - If Path: A [pathlib.Path][] to the caching directory.
        - If Memory: Object with the [joblib.Memory][] interface.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    warnings: bool or str, default=False
        - If True: Default warning action (equal to "once").
        - If False: Suppress all warnings (equal to "ignore").
        - If str: One of python's [warnings filters][warnings].

        Changing this parameter affects the `PYTHONWarnings` environment.
        ExperionML can't manage warnings that go from C/C++ code to stdout.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic name.
        - If Path: A [pathlib.Path][] to the log file.
        - Else: Python `logging.Logger` instance.

    experiment: str or None, default=None
        Name of the [mlflow experiment][experiment] to use for tracking.
        If None, no mlflow tracking is performed.

    random_state: int or None, default=None
        Seed used by the random number generator. If None, the random
        number generator is the `RandomState` used by `np.random`.

    See Also
    --------
    experionml.api:ExperionMLClassifier
    experionml.api:ExperionMLForecaster

    Examples
    --------
    ```pycon
    from experionml import ExperionMLRegressor
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)

    # Initialize experionml
    experionml = ExperionMLRegressor(X, y, verbose=2)

    # Apply data cleaning and feature engineering methods
    experionml.scale()
    experionml.feature_selection(strategy="rfecv", solver="xgb", n_features=12)

    # Train models
    experionml.run(models=["OLS", "RF", "XGB"])

    # Analyze the results
    experionml.results
    ```

    """

    _goal = Goal.regression

    def __init__(
        self,
        *arrays,
        y: YSelector = -1,
        index: IndexSelector = False,
        metadata: MetadataDict | None = None,
        ignore: ColumnSelector | None = None,
        shuffle: Bool = True,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            arrays=arrays,
            y=y,
            index=index,
            ignore=ignore,
            metadata=metadata,
            test_size=test_size,
            holdout_size=holdout_size,
            shuffle=shuffle,
            stratify=None,
            n_rows=n_rows,
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            memory=memory,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )
