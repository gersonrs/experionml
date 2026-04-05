from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Hashable
from typing import Any, Literal, TypeVar, overload

import numpy as np
import pandas as pd
import sklearn
from beartype import beartype
from category_encoders import (
    BackwardDifferenceEncoder,
    BaseNEncoder,
    BinaryEncoder,
    CatBoostEncoder,
    HelmertEncoder,
    JamesSteinEncoder,
    MEstimateEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialEncoder,
    SumEncoder,
    TargetEncoder,
    WOEEncoder,
)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import (
    ADASYN,
    SMOTE,
    SMOTEN,
    SMOTENC,
    SVMSMOTE,
    BorderlineSMOTE,
    KMeansSMOTE,
    RandomOverSampler,
)
from imblearn.under_sampling import (
    AllKNN,
    CondensedNearestNeighbour,
    EditedNearestNeighbours,
    InstanceHardnessThreshold,
    NearMiss,
    NeighbourhoodCleaningRule,
    OneSidedSelection,
    RandomUnderSampler,
    RepeatedEditedNearestNeighbours,
    TomekLinks,
)
from scipy.stats import zscore
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, _clone_parametrized
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.utils.validation import _check_feature_names_in
from sktime.transformations.series.detrend import (
    ConditionalDeseasonalizer,
    Deseasonalizer,
    Detrender,
)
from sktime.transformations.series.impute import Imputer as SktimeImputer
from typing_extensions import Self

from experionml.basetransformer import BaseTransformer
from experionml.utils.constants import CAT_TYPES, DEFAULT_MISSING
from experionml.utils.types import (
    Bins,
    Bool,
    CategoricalStrats,
    DiscretizerStrats,
    Engine,
    EngineDataOptions,
    EngineTuple,
    Estimator,
    FloatLargerZero,
    Int,
    IntLargerEqualZero,
    IntLargerTwo,
    IntLargerZero,
    NJobs,
    NormalizerStrats,
    NumericalStrats,
    Predictor,
    PrunerStrats,
    Scalar,
    ScalerStrats,
    SeasonalityModels,
    Sequence,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
    YReturn,
    sequence_t,
)
from experionml.utils.utils import (
    Goal,
    check_is_fitted,
    get_col_names,
    get_col_order,
    get_cols,
    it,
    lst,
    make_sklearn,
    merge,
    n_cols,
    replace_missing,
    sign,
    to_df,
    to_series,
    to_tabular,
    variable_return,
)


T_Transformer = TypeVar("T_Transformer", bound=Transformer)


@beartype
class TransformerMixin(BaseEstimator, BaseTransformer):
    """Classe mixin para todos os transformadores do ExperionML.

    Diferente do sklearn nas seguintes formas:

    - Considera a transformação de y.
    - Sempre adiciona um método fit.
    - Encapsula o método fit com atributos e verificação dos dados.
    - Encapsula os métodos de transformação com verificação dos dados.
    - Mantém os atributos internos ao ser clonado.

    """

    def __repr__(self, N_CHAR_MAX: Int = 700) -> str:
        """Remove named tuples com valores padrão da representação em string."""
        out = super().__repr__(N_CHAR_MAX)

        # Remove o engine padrão para uma representação mais limpa
        if hasattr(self, "engine") and sklearn.get_config()["print_changed_only"]:
            if self.engine.data == EngineTuple().data:
                out = re.sub(f"'data': '{self.engine.data}'", "", out)
            if self.engine.estimator == EngineTuple().estimator:
                out = re.sub(f", 'estimator': '{self.engine.estimator}'", "", out)
            out = re.sub("engine={}", "", out)
            out = re.sub(
                r"((?<=[{(]),\s|,\s(?=[})])|,\s(?=,\s))", "", out
            )  # Remove vírgulas e espaços

        return out

    def __sklearn_clone__(self: T_Transformer) -> T_Transformer:
        """Encapsula o método de clonagem para anexar atributos internos."""
        cloned = _clone_parametrized(self)

        for attr in ("_cols", "_train_only"):
            if hasattr(self, attr):
                setattr(cloned, attr, getattr(self, attr))

        return cloned

    def fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> Self:
        """Não faz nada.

        Implementado para continuidade da API.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`. Se None, `y` é
            ignorado.

        **fit_params
            Argumentos de palavra-chave adicionais para o método fit.

        Retorna
        -------
        self
            Instância do estimador.

        """
        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self._log(f"Fitting {self.__class__.__name__}...", 1)

        return self

    @overload
    def fit_transform(
        self,
        X: Literal[None],
        y: YConstructor,
        **fit_params,
    ) -> YReturn: ...

    @overload
    def fit_transform(
        self,
        X: XConstructor,
        y: Literal[None] = ...,
        **fit_params,
    ) -> XReturn: ...

    @overload
    def fit_transform(
        self,
        X: XConstructor,
        y: YConstructor,
        **fit_params,
    ) -> tuple[XReturn, YReturn]: ...

    def fit_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Ajusta aos dados e depois os transforma.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`. Se None, `y` é
            ignorado.

        **fit_params
            Argumentos de palavra-chave adicionais para o método fit.

        Retorna
        -------
        dataframe
            Conjunto de variáveis transformado. Retornado somente se fornecido.

        series or dataframe
            Coluna alvo transformada. Retornada somente se fornecida.

        """
        return self.fit(X, y, **fit_params).transform(X, y)

    @overload
    def inverse_transform(
        self,
        X: Literal[None],
        y: YConstructor,
        **fit_params,
    ) -> YReturn: ...

    @overload
    def inverse_transform(
        self,
        X: XConstructor,
        y: Literal[None] = ...,
        **fit_params,
    ) -> XReturn: ...

    @overload
    def inverse_transform(
        self,
        X: XConstructor,
        y: YConstructor,
        **fit_params,
    ) -> tuple[XReturn, YReturn]: ...

    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Não faz nada.

        Retorna a entrada sem alterações. Implementado para continuidade da API.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`. Se None, `y` é
            ignorado.

        Retorna
        -------
        dataframe
            Conjunto de variáveis. Retornado somente se fornecido.

        series or dataframe
            Coluna(s) alvo. Retornada(s) somente se fornecida(s).

        """
        check_is_fitted(self)

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        return variable_return(self._convert(Xt), self._convert(yt))

    def set_output(self, *, transform: EngineDataOptions | None = None) -> Self:
        """Define o contêiner de saída.

        Consulte o [guia do usuário][set_output] do sklearn sobre como usar a
        API `set_output`. Veja [aqui][data-engines] uma descrição
        das opções disponíveis.

        Parâmetros
        ----------
        transform: str or None, default=None
            Configura a saída dos métodos `transform`, `fit_transform`
            e `inverse_transform`. Se None, a configuração não é alterada.
            Escolha entre:

            - "numpy"
            - "pandas" (padrão)
            - "pandas-pyarrow"
            - "polars"
            - "polars-lazy"
            - "pyarrow"
            - "modin"
            - "dask"
            - "pyspark"
            - "pyspark-pandas"

        Retorna
        -------
        Self
            Instância do estimador.

        """
        if not hasattr(self, "_engine"):
            self.engine = EngineTuple()

        if transform is not None:
            self.engine = EngineTuple(estimator=self.engine.estimator, data=transform)

        return self


@beartype
class Balancer(TransformerMixin, OneToOneFeatureMixin):
    """Balanceia o número de amostras por classe na coluna alvo.

    Ao fazer oversampling, as novas amostras criadas têm um índice inteiro
    crescente para índices numéricos, e um índice da forma
    [estimator]_N para índices não numéricos, onde N representa a
    N-ésima amostra no conjunto de dados. Use apenas para tarefas de classificação.

    Esta classe pode ser acessada pelo experionml através do método [balance]
    [experionmlclassifier-balance]. Leia mais no [guia do usuário]
    [balancing-the-data].

    !!! warning
         * O estimador [clustercentroids][] não está disponível devido a
           incompatibilidades entre as APIs.
         * A classe Balancer não suporta [tarefas multioutput][].

    Parâmetros
    ----------
    strategy: str or transformer, default="ADASYN"
        Tipo de algoritmo com o qual balancear o conjunto de dados. Escolha
        pelo nome de qualquer estimador do pacote imbalanced-learn
        ou forneça uma instância personalizada.

    n_jobs: int, default=1
        Número de núcleos a usar para processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usar todos os núcleos disponíveis.
        - Se <-1: Usar número de núcleos - 1 - valor.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.
        - 2 para exibir informações detalhadas.

    random_state: int or None, default=None
        Semente usada pelo gerador de números aleatórios. Se None, o gerador
        é o `RandomState` utilizado pelo `np.random`.

    **kwargs
        Argumentos de palavra-chave adicionais para o estimador `strategy`.

    Atributos
    ----------
    [strategy]_: imblearn estimator
        Objeto (strategy em minúsculas) usado para balancear os dados,
        ex.: `balancer.adasyn_` para a estratégia padrão.

    mapping_: dict
        Valores alvo mapeados para seus respectivos inteiros codificados.

    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    target_names_in_: np.ndarray
        Nomes da coluna alvo observados durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Encoder
    experionml.data_cleaning:Imputer
    experionml.data_cleaning:Pruner

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.train)

        experionml.balance(strategy="smote", verbose=2)

        # Observe que o número de linhas aumentou
        print(experionml.train)
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Balancer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        print(X)

        balancer = Balancer(strategy="smote", verbose=2)
        X, y = balancer.fit_transform(X, y)

        # Observe que o número de linhas aumentou
        print(X)
        ```

    """

    _train_only = True

    def __init__(
        self,
        strategy: str | Estimator = "ADASYN",
        *,
        n_jobs: NJobs = 1,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
        **kwargs,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        self.strategy = strategy
        self.kwargs = kwargs

    def _log_changes(self, y: pd.Series):
        """Exibe as mudanças por classe da variável alvo.

        Parâmetros
        ----------
        y: pd.Series
            Coluna alvo.

        """
        for key, value in self.mapping_.items():
            diff = self._counts[key] - np.sum(y == value)
            if diff > 0:
                self._log(f" --> Removing {diff} samples from class {key}.", 2)
            elif diff < 0:
                self._log(f" --> Adding {-diff} samples to class {key}.", 2)

    def fit(self, X: XConstructor, y: YConstructor) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence
            Coluna alvo correspondente a `X`.

        Retorna
        -------
        Self
            Instância do estimador.

        """
        Xt = to_df(X)
        yt = to_tabular(y, index=Xt.index)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if isinstance(yt, pd.Series):
            self.target_names_in_ = np.array([yt.name])
        else:
            raise ValueError("A classe Balancer não suporta tarefas multioutput.")

        # ClusterCentroids não está disponível pois não possui sample_indices_
        strategies = {
            "condensednearestneighbour": CondensedNearestNeighbour,
            "editednearestneighborus": EditedNearestNeighbours,
            "repeatededitednearestneighbours": RepeatedEditedNearestNeighbours,
            "allknn": AllKNN,
            "instancehardnessthreshold": InstanceHardnessThreshold,
            "nearmiss": NearMiss,
            "neighbourhoodcleaningrule": NeighbourhoodCleaningRule,
            "onesidedselection": OneSidedSelection,
            "randomundersampler": RandomUnderSampler,
            "tomeklinks": TomekLinks,
            "randomoversampler": RandomOverSampler,
            "smote": SMOTE,
            "smotenc": SMOTENC,
            "smoten": SMOTEN,
            "adasyn": ADASYN,
            "borderlinesmote": BorderlineSMOTE,
            "kmeanssmote": KMeansSMOTE,
            "svmsmote": SVMSMOTE,
            "smoteenn": SMOTEENN,
            "smotetomek": SMOTETomek,
        }

        if isinstance(self.strategy, str):
            if self.strategy.lower() not in strategies:
                raise ValueError(
                    f"Valor inválido para o parâmetro strategy, valor recebido: {self.strategy}. "
                    f"Escolha entre: {', '.join(strategies)}."
                )
            est_class = strategies[self.strategy.lower()]
            estimator = self._inherit(est_class(**self.kwargs), fixed=tuple(self.kwargs))
        elif not hasattr(self.strategy, "fit_resample"):
            raise TypeError(
                "Tipo inválido para o parâmetro strategy. Um "
                "estimador personalizado deve ter o método fit_resample."
            )
        elif callable(self.strategy):
            estimator = self._inherit(self.strategy(**self.kwargs), fixed=tuple(self.kwargs))
        else:
            estimator = self.strategy

        # Cria dicionário de contagem de classes em y
        if not hasattr(self, "mapping_"):
            self.mapping_ = {str(v): v for v in yt.sort_values().unique()}

        self._counts = {}
        for key, value in self.mapping_.items():
            self._counts[key] = np.sum(yt == value)

        # Fit apenas verifica a entrada e a estratégia de amostragem
        self._estimator = estimator.fit(Xt, yt)

        # Adiciona o estimador como atributo à instância
        setattr(self, f"{estimator.__class__.__name__.lower()}_", self._estimator)

        return self

    def transform(self, X: XConstructor, y: YConstructor) -> tuple[XReturn, YReturn]:
        """Balanceia os dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence
            Coluna alvo correspondente a `X`.

        Retorna
        -------
        dataframe
            DataFrame balanceado.

        series
            Coluna alvo transformada.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)
        yt = to_series(y, index=Xt.index, name=self.target_names_in_[0])  # type: ignore[arg-type]

        if "over_sampling" in self._estimator.__module__:
            self._log(f"Oversampling with {self._estimator.__class__.__name__}...", 1)

            index = Xt.index  # Salva os índices para reatribuição posterior
            Xt, yt = self._estimator.fit_resample(Xt, yt)

            # Cria índices para as novas amostras
            n_idx: list[int | str]
            if index.dtype.kind in "ifu":
                n_idx = list(range(max(index) + 1, max(index) + len(Xt) - len(index) + 1))
            else:
                n_idx = [
                    f"{self._estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(Xt) - len(index) + 1)
                ]

            # Atribui os índices antigos + novos
            Xt.index = pd.Index(list(index) + n_idx)
            yt.index = pd.Index(list(index) + n_idx)

            self._log_changes(yt)

        elif "under_sampling" in self._estimator.__module__:
            self._log(f"Undersampling with {self._estimator.__class__.__name__}...", 1)

            self._estimator.fit_resample(Xt, yt)

            # Seleciona as linhas escolhidas (imblearn não as retorna em ordem)
            samples = np.asarray(sorted(self._estimator.sample_indices_))
            Xt, yt = Xt.iloc[samples], yt.iloc[samples]

            self._log_changes(yt)

        elif "combine" in self._estimator.__module__:
            self._log(f"Balancing with {self._estimator.__class__.__name__}...", 1)

            index = Xt.index
            X_new, y_new = self._estimator.fit_resample(Xt, yt)

            # Seleciona as linhas mantidas pelo undersampler
            if self._estimator.__class__.__name__ == "SMOTEENN":
                samples = np.asarray(sorted(self._estimator.enn_.sample_indices_))
            elif self._estimator.__class__.__name__ == "SMOTETomek":
                samples = np.asarray(sorted(self._estimator.tomek_.sample_indices_))

            # Seleciona as amostras restantes do dataframe original
            o_samples = [s for s in samples if s < len(Xt)]
            Xt, yt = Xt.iloc[o_samples], yt.iloc[o_samples]  # type: ignore[call-overload]

            # Cria índices para as novas amostras
            if index.dtype.kind in "ifu":
                n_idx = list(range(max(index) + 1, max(index) + len(X_new) - len(Xt) + 1))
            else:
                n_idx = [
                    f"{self._estimator.__class__.__name__.lower()}_{i}"
                    for i in range(1, len(X_new) - len(Xt) + 1)
                ]

            # Seleciona as novas amostras e atribui os novos índices
            X_new = X_new.iloc[-len(X_new) + len(o_samples) :]
            X_new.index = pd.Index(n_idx)
            y_new = y_new.iloc[-len(y_new) + len(o_samples) :]
            y_new.index = pd.Index(n_idx)

            # Primeiro, exibe as amostras criadas
            for key, value in self.mapping_.items():
                if (diff := np.sum(y_new == value)) > 0:
                    self._log(f" --> Adding {diff} samples to class: {key}.", 2)

            # Depois, exibe as amostras removidas
            for key, value in self.mapping_.items():
                if (diff := self._counts[key] - np.sum(yt == value)) > 0:
                    self._log(f" --> Removing {diff} samples from class: {key}.", 2)

            # Adiciona as novas amostras ao dataframe original
            Xt, yt = pd.concat([Xt, X_new]), pd.concat([yt, y_new])

        return self._convert(Xt), self._convert(yt)


@beartype
class Cleaner(TransformerMixin):
    """Aplica etapas padrão de limpeza de dados a um conjunto de dados.

    Use os parâmetros para escolher quais transformações realizar.
    As etapas disponíveis são:

    - Converter dtypes para os melhores tipos possíveis.
    - Remover colunas com tipos de dados específicos.
    - Remover caracteres dos nomes das colunas.
    - Remover espaços de colunas categóricas.
    - Remover linhas duplicadas.
    - Remover linhas com valores ausentes na coluna alvo.
    - Codificar a coluna alvo.

    Esta classe pode ser acessada pelo experionml através do método [clean]
    [experionmlclassifier-clean]. Leia mais no [guia do usuário]
    [standard-data-cleaning].

    Parâmetros
    ----------
    convert_dtypes: bool, default=True
        Converte os tipos de dados das colunas para os melhores tipos possíveis
        que suportam `pd.NA`.

    drop_dtypes: str, sequence or None, default=None
        Colunas com esses tipos de dados são removidas do conjunto de dados.

    drop_chars: str or None, default=None
        Remove o padrão regex especificado dos nomes das colunas, ex.:
        `[^A-Za-z0-9]+` para remover todos os caracteres não alfanuméricos.

    strip_categorical: bool, default=True
        Se deve remover espaços das colunas categóricas.

    drop_duplicates: bool, default=False
        Se deve remover linhas duplicadas. Apenas a primeira ocorrência de
        cada linha duplicada é mantida.

    drop_missing_target: bool, default=True
        Se deve remover linhas com valores ausentes na coluna alvo.
        Esta transformação é ignorada se `y` não for fornecido.

    encode_target: bool, default=True
        Se deve codificar a(s) coluna(s) alvo. Isso inclui
        converter colunas categóricas para numérico e binarizar
        colunas [multilabel][]. Esta transformação é ignorada se `y`
        não for fornecido.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], ex.:
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Engine de execução para [estimadores][estimator-acceleration].
        Se None, o valor padrão é usado. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.
        - 2 para exibir informações detalhadas.

    Atributos
    ----------
    missing_: list
        Valores considerados "ausentes". Os valores padrão são: None,
        NaN, NA, NaT, +inf, -inf, "", "?", "NA", "nan", "NaN", "NaT",
        "none", "None", "inf", "-inf". Note que None, NaN, NA, +inf e
        -inf são sempre considerados ausentes pois são incompatíveis
        com estimadores sklearn.

    mapping_: dict
        Valores alvo mapeados para seus respectivos inteiros codificados. Apenas
        disponível se encode_target=True.

    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    target_names_in_: np.ndarray
        Nomes da(s) coluna(s) alvo observados durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Encoder
    experionml.data_cleaning:Discretizer
    experionml.data_cleaning:Scaler

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        y = ["a" if i else "b" for i in y]

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.y)

        experionml.clean(verbose=2)

        print(experionml.y)
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Cleaner
        from numpy.random import randint

        y = ["a" if i else "b" for i in range(randint(100))]

        cleaner = Cleaner(verbose=2)
        y = cleaner.fit_transform(y=y)

        print(y)
        ```

    """

    def __init__(
        self,
        *,
        convert_dtypes: Bool = True,
        drop_dtypes: str | Sequence[str] | None = None,
        drop_chars: str | None = None,
        strip_categorical: Bool = True,
        drop_duplicates: Bool = False,
        drop_missing_target: Bool = True,
        encode_target: Bool = True,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose)
        self.convert_dtypes = convert_dtypes
        self.drop_dtypes = drop_dtypes
        self.drop_chars = drop_chars
        self.strip_categorical = strip_categorical
        self.drop_duplicates = drop_duplicates
        self.drop_missing_target = drop_missing_target
        self.encode_target = encode_target

    def fit(self, X: XConstructor | None = None, y: YConstructor | None = None) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        Self
            Instância do estimador.

        """
        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self.mapping_: dict[str, Any] = {}
        self.target_names_in_ = np.array([])
        self._drop_cols = []
        self._estimators = {}

        if not hasattr(self, "missing_"):
            self.missing_ = DEFAULT_MISSING

        self._log("Ajustando Cleaner...", 1)

        if Xt is not None and self.drop_dtypes is not None:
            self._drop_cols = list(Xt.select_dtypes(include=lst(self.drop_dtypes)).columns)

        if yt is not None:
            self.target_names_in_ = np.array(get_col_names(yt))

            if self.drop_chars:
                if isinstance(yt, pd.DataFrame):
                    yt = yt.rename(lambda x: re.sub(self.drop_chars, "", str(x)), axis=1)
                else:
                    yt.name = re.sub(self.drop_chars, "", str(yt.name))

            if self.drop_missing_target:
                yt = replace_missing(yt, self.missing_).dropna(axis=0)

            if self.encode_target:
                for col in get_cols(yt):
                    if isinstance(col.iloc[0], sequence_t):  # Multilabel (múltiplos rótulos)
                        MultiLabelBinarizer = self._get_est_class(
                            name="MultiLabelBinarizer",
                            module="preprocessing",
                        )
                        self._estimators[col.name] = MultiLabelBinarizer().fit(col)
                    elif list(uq := np.unique(col)) != list(range(col.nunique())):
                        LabelEncoder = self._get_est_class("LabelEncoder", "preprocessing")
                        self._estimators[col.name] = LabelEncoder().fit(col)
                        self.mapping_.update(
                            {str(col.name): {str(it(v)): i for i, v in enumerate(uq)}}
                        )

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Retorna os nomes das variáveis após a transformação.

        Parâmetros
        ----------
        input_features: sequence or None, default=None
            Usado apenas para validar os nomes das variáveis com os nomes
            observados durante o `fit`.

        Retorna
        -------
        np.ndarray
            Nomes das variáveis transformadas.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        columns = [col for col in self.feature_names_in_ if col not in self._drop_cols]

        if self.drop_chars:
            # Remove caracteres proibidos dos nomes das colunas
            columns = [re.sub(self.drop_chars, "", str(c)) for c in columns]

        return np.array(columns)

    def transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Aplica as etapas de limpeza de dados.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        dataframe
            Conjunto de variáveis transformado. Retornado somente se fornecido.

        series or dataframe
            Coluna alvo transformada. Retornada somente se fornecida.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        yt = to_tabular(y, index=getattr(Xt, "index", None), columns=self.target_names_in_)

        self._log("Limpando os dados...", 1)

        if Xt is not None:
            # Unifica todos os valores ausentes
            Xt = replace_missing(Xt, self.missing_)

            for name, column in Xt.items():
                # Remove variáveis com tipo de dado inválido
                if name in self._drop_cols:
                    self._log(
                        f" --> Removendo a variável {name} por "
                        f"ter o tipo: {column.dtype.name}.",
                        2,
                    )
                    Xt = Xt.drop(columns=name)

                elif column.dtype.name in CAT_TYPES:
                    if self.strip_categorical:
                        # Remove espaços em branco de strings
                        Xt[name] = column.apply(
                            lambda val: val.strip() if isinstance(val, str) else val
                        )

            # Remove caracteres proibidos dos nomes das colunas
            if self.drop_chars:
                Xt = Xt.rename(columns=lambda x: re.sub(self.drop_chars, "", str(x)))

            # Remove amostras duplicadas
            if self.drop_duplicates:
                Xt = Xt.drop_duplicates(ignore_index=True)

            if self.convert_dtypes:
                Xt = Xt.convert_dtypes()

        if yt is not None:
            if self.drop_chars:
                if isinstance(y, pd.Series):
                    yt.name = re.sub(self.drop_chars, "", str(yt.name))
                else:
                    yt = yt.rename(lambda x: re.sub(self.drop_chars, "", str(x)), axis=1)

            # Remove amostras com valores ausentes no alvo
            if self.drop_missing_target:
                length = len(yt)  # Salva o comprimento original para contar as linhas removidas
                yt = replace_missing(yt, self.missing_).dropna()

                if Xt is not None:
                    Xt = Xt[Xt.index.isin(yt.index)]  # Seleciona apenas os índices restantes

                if (d := length - len(yt)) > 0:
                    self._log(f" --> Removendo {d} linhas com valores ausentes no alvo.", 2)

            if self.encode_target and self._estimators:
                y_new = yt.__class__(dtype="object")
                for col in get_cols(yt):
                    if est := self._estimators.get(col.name):
                        if n_cols(out := est.transform(col)) == 1:
                            self._log(f" --> Aplicando label encoding na coluna {col.name}.", 2)
                            out = to_series(out, yt.index, str(col.name))
                        else:
                            self._log(f" --> Aplicando label binarization na coluna {col.name}.", 2)
                            out = to_df(
                                data=out,
                                index=yt.index,
                                columns=[f"{col.name}_{c}" for c in est.classes_],
                            )

                        # Substitui o alvo pela(s) coluna(s) codificada(s)
                        if isinstance(yt, pd.Series):
                            y_new = out
                        else:
                            y_new = merge(y_new, out)

                    else:  # Adiciona coluna inalterada
                        y_new = merge(y_new, col)

                yt = y_new

            if self.convert_dtypes:
                yt = yt.convert_dtypes()

        return variable_return(self._convert(Xt), self._convert(yt))

    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Reverte a codificação de rótulos.

        Este método reverte apenas a codificação do alvo.
        As demais transformações não podem ser revertidas. Se
        `encode_target=False`, os dados são retornados como estão.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        dataframe
            Conjunto de variáveis inalterado. Retornado somente se fornecido.

        series or dataframe
            Coluna alvo original. Retornada somente se fornecida.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        self._log("Revertendo a limpeza dos dados...", 1)

        if yt is not None and self._estimators:
            y_new = yt.__class__(dtype="object")
            for col in self.target_names_in_:
                if est := self._estimators.get(col):
                    if est.__class__.__name__ == "LabelEncoder":
                        self._log(f" --> Revertendo label encoding da coluna {col}.", 2)
                        out = est.inverse_transform(pd.DataFrame(yt)[col])

                    elif isinstance(yt, pd.DataFrame):
                        self._log(f" --> Revertendo label binarization da coluna {col}.", 2)
                        out = est.inverse_transform(
                            yt.loc[:, yt.columns.str.startswith(f"{col}_")].to_numpy()
                        )

                    # Substitui as colunas codificadas pela coluna alvo
                    if isinstance(yt, pd.Series):
                        y_new = to_series(out, yt.index, col)
                    else:
                        y_new = merge(y_new, to_series(out, yt.index, col))

                else:  # Adiciona coluna inalterada
                    y_new = merge(y_new, pd.DataFrame(yt)[col])

            yt = y_new

        return variable_return(self._convert(Xt), self._convert(yt))


@beartype
class Decomposer(TransformerMixin, OneToOneFeatureMixin):
    """Remove tendência e sazonalidade da série temporal.

    Esta classe realiza duas operações:

    - Remove a tendência de cada coluna, retornando os resíduos in-sample
      dos valores previstos pelo modelo.
    - Remove o componente sazonal de cada coluna, condicionado
      a um teste de sazonalidade.

    Colunas categóricas são ignoradas.

    Esta classe pode ser acessada pelo experionml através do método [decompose]
    [experionmlforecaster-decompose]. Leia mais no [guia do usuário]
    [time-series-decomposition].

    !!! note
        Ao usar esta classe pelo experionml, os parâmetros `trend_model`, `sp` e
        `seasonal_model` são definidos automaticamente com base no
        atributo `experionml.sp`.

    Parâmetros
    ----------
    model: str, predictor or None, default=None
        O modelo de previsão para remover a tendência. Deve ser
        um modelo que suporte a tarefa de previsão. Se None,
        [PolynomialTrend][](degree=1) é usado.

    trend_model: str, default="additive"
        Modo de decomposição da tendência. Escolha entre:

        - "additive": O `model.transform` subtrai a tendência, ou seja,
          `transform(X)` retorna `X - model.predict(fh=X.index)`.
        - "multiplicative": O `model.transform` divide pela tendência,
          ou seja, `transform(X)` retorna `X / model.predict(fh=X.index)`.

    test_seasonality: bool, default=True

        - Se True, ajusta um teste de sazonalidade de autocorrelação de 90% e,
          se a série temporal tiver componente sazonal,
          aplica a decomposição sazonal. Se o teste for negativo,
          a dessazonalização é ignorada.
        - Se False, sempre realiza a dessazonalização.

    sp: int or None, default=None
        Período de sazonalidade da série temporal. Se None, não há
        sazonalidade.

    seasonal_model: str, default="additive"
        Modo de decomposição sazonal. Escolha entre:

        - "additive": Assume que os componentes têm relação linear,
          ou seja, y(t) = nível + tendência + sazonalidade + ruído.
        - "multiplicative": Assume que os componentes têm relação não linear,
          ou seja, y(t) = nível * tendência * sazonalidade * ruído.

    n_jobs: int, default=1
        Número de núcleos a usar para processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usar todos os núcleos disponíveis.
        - Se <-1: Usar número de núcleos - 1 + `n_jobs`.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.
        - 2 para exibir informações detalhadas.

    random_state: int or None, default=None
        Semente usada pelo gerador de números aleatórios. Se None, o gerador
        é o `RandomState` utilizado pelo `np.random`.

    Atributos
    ----------
    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Encoder
    experionml.data_cleaning:Discretizer
    experionml.data_cleaning:Scaler

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        experionml = ExperionMLForecaster(y, random_state=1)
        print(experionml.y)

        experionml.decompose(columns=-1, verbose=2)

        print(experionml.y)
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Decomposer
        from sktime.datasets import load_longley

        X, _ = load_longley()

        decomposer = Decomposer(verbose=2)
        X = decomposer.fit_transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        *,
        model: str | Predictor | None = None,
        trend_model: SeasonalityModels = "additive",
        test_seasonality: Bool = True,
        sp: IntLargerZero | None = None,
        seasonal_model: SeasonalityModels = "additive",
        n_jobs: NJobs = 1,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        self.model = model
        self.trend_model = trend_model
        self.test_seasonality = test_seasonality
        self.sp = sp
        self.seasonal_model = seasonal_model

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        Self
            Instância do estimador.

        """
        from experionml.models import MODELS

        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if isinstance(self.model, str):
            if self.model in MODELS:
                model = MODELS[self.model](
                    goal=Goal.forecast,
                    **{x: getattr(self, x) for x in BaseTransformer.attrs if hasattr(self, x)},
                )
                forecaster = model._get_est({})
            else:
                raise ValueError(
                    "Valor inválido para o parâmetro model. Modelo desconhecido: "
                    f"{self.model}. Os modelos disponíveis são:\n"
                    + "\n".join(
                        [
                            f" --> {m.__name__} ({m.acronym})"
                            for m in MODELS
                            if "forecast" in m._estimators
                        ]
                    )
                )
        elif callable(self.model):
            forecaster = self._inherit(self.model())
        else:
            forecaster = self.model

        self._log("Ajustando Decomposer...", 1)

        self._estimators: dict[Hashable, tuple[Transformer, Transformer]] = {}
        for name, column in Xt.select_dtypes(include="number").items():
            trend = Detrender(
                forecaster=forecaster,
                model=self.trend_model,
            ).fit(column)

            if self.test_seasonality:
                season = ConditionalDeseasonalizer(
                    sp=self.sp or 1,
                    model=self.seasonal_model,
                ).fit(trend.transform(column))
            else:
                season = Deseasonalizer(
                    sp=self.sp or 1,
                    model=self.seasonal_model,
                ).fit(trend.transform(column))

            self._estimators[name] = (trend, season)

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Decompõe os dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            Conjunto de variáveis transformado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Decompondo os dados...", 1)

        for col, (trend, season) in self._estimators.items():
            Xt[col] = season.transform(trend.transform(Xt[col]))

        return self._convert(Xt)

    def inverse_transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Aplica a transformação inversa nos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            Conjunto de variáveis original.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Revertendo a decomposição dos dados...", 1)

        for col, (trend, season) in self._estimators.items():
            Xt[col] = trend.inverse_transform(season.inverse_transform(Xt[col]))

        return self._convert(Xt)


@beartype
class Discretizer(TransformerMixin, OneToOneFeatureMixin):
    """Agrupa dados contínuos em intervalos.

    Para cada variável, as bordas dos intervalos são calculadas durante o fit e,
    juntamente com o número de intervalos, definem os bins.
    Colunas categóricas são ignoradas.

    Esta classe pode ser acessada pelo experionml através do método [discretize]
    [experionmlclassifier-discretize]. Leia mais no [guia do usuário]
    [binning-numerical-features].

    !!! tip
        A transformação retorna colunas categóricas. Use a
        classe [Encoder][] para convertê-las de volta a tipos numéricos.

    Parâmetros
    ----------
    strategy: str, default="quantile"
        Estratégia usada para definir a largura dos bins. Escolha entre:

        - "uniform": Todos os bins têm larguras idênticas.
        - "quantile": Todos os bins têm o mesmo número de pontos.
        - "kmeans": Valores em cada bin têm o mesmo centro mais próximo de
          um cluster k-means 1D.
        - "custom": Use bordas de bin personalizadas fornecidas pelo `bins`.

    bins: int, sequence or dict, default=5
        Número ou bordas de bins em que dividir cada coluna.

        - Se int: Número de bins para todas as colunas. Apenas para
          strategy!="custom".
        - Se sequence:

            - Para strategy!="custom": Número de bins por coluna. O
              n-ésimo valor corresponde à n-ésima coluna transformada.
              Colunas categóricas são ignoradas.
            - Para strategy="custom": Bordas de bin com comprimento=n_bins - 1.
              As bordas externas são sempre `-inf` e `+inf`, ex.,
              bins `[1, 2]` indicam `(-inf, 1], (1, 2], (2, inf]`.

        - Se dict: Uma das opções acima por coluna, onde
          a chave é o nome da coluna. Colunas não presentes no
          dicionário não são transformadas.

    labels: sequence, dict or None, default=None
        Nomes dos rótulos para substituir os intervalos.

        - Se None: Usa rótulos padrão no formato `(borda_min, borda_max]`.
        - Se sequence: Rótulos a usar para todas as colunas.
        - Se dict: Rótulos por coluna, onde a chave é o nome da coluna.
          Colunas não presentes no dicionário usam os rótulos padrão.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], ex.:
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Engine de execução para [estimadores][estimator-acceleration].
        Se None, o valor padrão é usado. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.
        - 2 para exibir informações detalhadas.

    random_state: int or None, default=None
        Semente usada pelo gerador de números aleatórios. Se None, o gerador
        é o `RandomState` utilizado pelo `np.random`. Apenas
        para strategy="quantile".

    Atributos
    ----------
    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Encoder
    experionml.data_cleaning:Imputer
    experionml.data_cleaning:Normalizer

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml["mean radius"])

        experionml.discretize(
            strategy="custom",
            bins=[13, 18],
            labels=["small", "medium", "large"],
            verbose=2,
            columns="mean radius",
        )

        print(experionml["mean radius"])
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Discretizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        print(X["mean radius"])

        discretizer = Discretizer(
            strategy="custom",
            bins={"mean radius": [13, 18]},
            labels=["small", "medium", "large"],
            verbose=2,
        )
        X = discretizer.fit_transform(X)

        print(X["mean radius"])
        ```

    """

    def __init__(
        self,
        strategy: DiscretizerStrats = "quantile",
        *,
        bins: Bins = 5,
        labels: Sequence[str] | dict[str, Sequence[str]] | None = None,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            device=device,
            engine=engine,
            verbose=verbose,
            random_state=random_state,
        )
        self.strategy = strategy
        self.bins = bins
        self.labels = labels

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        Self
            Instância do estimador.

        """

        def get_labels(col: str, bins: Sequence[Scalar]) -> tuple[str, ...]:
            """Retorna os rótulos para os bins especificados.

            Parâmetros
            ----------
            col: str
                Nome da coluna.

            bins: sequence
                Bordas dos bins.

            Retorna
            -------
            tuple
                Rótulos para a coluna.

            """
            default = [
                f"({np.round(bins[i], 2)}, {np.round(bins[i + 1], 1)}]"
                for i in range(len(bins[:-1]))
            ]

            if self.labels is None:
                labels = tuple(default)
            elif isinstance(self.labels, dict):
                labels = tuple(self.labels.get(col, default))
            else:
                labels = tuple(self.labels)

            if len(bins) - 1 != len(labels):
                raise ValueError(
                    "Valor inválido para o parâmetro labels. O número de "
                    "bins não corresponde ao número de labels, obtido "
                    f"len(bins)={len(bins) - 1} e len(labels)={len(labels)}."
                )

            return labels

        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self._estimators: dict[Hashable, Estimator] = {}
        self._labels: dict[Hashable, Sequence[str]] = {}

        self._log("Ajustando Discretizer...", 1)

        for i, col in enumerate(Xt.select_dtypes(include="number")):
            # Assign bins per column
            if isinstance(self.bins, dict):
                if col in self.bins:
                    bins_c = self.bins[str(col)]
                else:
                    continue  # Ignora coluna existente não especificada no dict
            else:
                bins_c = self.bins

            if self.strategy != "custom":
                if isinstance(bins_c, sequence_t):
                    try:
                        bins_x = bins_c[i]  # Obtém o i-ésimo bin para a i-ésima coluna
                    except IndexError:
                        raise ValueError(
                            "Valor inválido para o parâmetro bins. O número de "
                            "bins não corresponde ao número de colunas, obtido len"
                            f"(bins)={len(bins_c)} e len(columns)={Xt.shape[1]}."
                        ) from None
                else:
                    bins_x = bins_c

                KBinsDiscretizer = self._get_est_class("KBinsDiscretizer", "preprocessing")

                # Implementação cuML não possui subsample e random_state
                kwargs: dict[str, Any] = {}
                if "subsample" in sign(KBinsDiscretizer):
                    kwargs["subsample"] = 200000
                    kwargs["random_state"] = self.random_state

                self._estimators[col] = KBinsDiscretizer(
                    n_bins=bins_x,
                    encode="ordinal",
                    strategy=self.strategy,
                    **kwargs,
                ).fit(Xt[[col]])

                # Salva os rótulos para o método transform
                self._labels[col] = get_labels(
                    col=str(col),
                    bins=self._estimators[col].bin_edges_[0],
                )

            else:
                if not isinstance(bins_c, sequence_t):
                    raise TypeError(
                        f"Tipo inválido para o parâmetro bins, valor recebido: {bins_c}. Apenas "
                        "uma sequência de bordas de bin é aceita quando strategy='custom'."
                    )
                else:
                    bins_c = [-np.inf, *bins_c, np.inf]

                FunctionTransformer = self._get_est_class(
                    name="FunctionTransformer",
                    module="preprocessing",
                )

                # Transforma pd.cut em um transformer
                self._estimators[col] = FunctionTransformer(
                    func=pd.cut,
                    kw_args={"bins": bins_c, "labels": get_labels(str(col), bins_c)},
                ).fit(Xt[[col]])

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Agrupa os dados em intervalos.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            Conjunto de variáveis transformado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Agrupando as variáveis em intervalos...", 1)

        for col in self._estimators:
            if self.strategy == "custom":
                Xt[col] = self._estimators[col].transform(Xt[col])
            else:
                Xt[col] = self._estimators[col].transform(Xt[[col]]).iloc[:, 0]

                # Substitui os valores de cluster pelos rótulos
                for i, label in enumerate(self._labels[col]):
                    Xt[col] = Xt[col].replace(i, label)

            self._log(f" --> Discretizando a variável {col} em {Xt[col].nunique()} bins.", 2)

        return self._convert(Xt)


@beartype
class Encoder(TransformerMixin):
    """Executa a codificação de variáveis categóricas.

    O tipo de codificação depende do número de classes na coluna:

    - Se n_classes=2 ou variável ordinal, usa codificação Ordinal.
    - Se 2 < n_classes <= `max_onehot`, usa codificação OneHot.
    - Se n_classes > `max_onehot`, usa codificação `strategy`.

    Valores ausentes são propagados para a coluna de saída. Classes desconhecidas
    encontradas durante a transformação são imputadas de acordo
    com a estratégia selecionada. Classes infrequentes podem ser substituídas por
    um valor para evitar cardinalidade muito alta.

    Esta classe pode ser acessada pelo experionml através do método [encode]
    [experionmlclassifier-encode]. Leia mais no [guia do usuário]
    [encoding-categorical-features].

    !!! warning
        Três estimadores do category-encoders não estão disponíveis:

        * [OneHotEncoder][]: Use o parâmetro max_onehot.
        * [HashingEncoder][]: Incompatibilidade de APIs.
        * [LeaveOneOutEncoder][]: Incompatibilidade de APIs.

    Parâmetros
    ----------
    strategy: str or transformer, default="Target"
        Tipo de codificação para variáveis de alta cardinalidade. Escolha
        qualquer estimador do pacote category-encoders
        ou forneça um personalizado.

    max_onehot: int or None, default=10
        Número máximo de valores únicos em uma variável para realizar
        codificação one-hot. Se None, codificação `strategy` é sempre
        usada para colunas com mais de duas classes.

    ordinal: dict or None, default=None
        Ordem das variáveis ordinais, onde a chave é o nome da variável
        e o valor é a ordem das classes, ex.: `{"salary": ["low",
        "medium", "high"]}`.

    infrequent_to_value: int, float or None, default=None
        Substitui ocorrências de classes infrequentes nas colunas categóricas
        pela string no parâmetro `value`. Esta transformação é
        feita antes da codificação da coluna.

        - Se None: Ignora esta etapa.
        - Se int: Número mínimo de ocorrências em uma classe.
        - Se float: Fração mínima de ocorrências em uma classe.

    value: str, default="infrequent"
        Valor com o qual substituir classes raras. Este parâmetro é
        ignorado se `infrequent_to_value=None`.

    n_jobs: int, default=1
        Número de núcleos a usar para processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usar todos os núcleos disponíveis.
        - Se <-1: Usar número de núcleos - 1 - valor.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.
        - 2 para exibir informações detalhadas.

    **kwargs
        Argumentos de palavra-chave adicionais para o estimador `strategy`.

    Atributos
    ----------
    mapping_: dict of dicts
        Valores codificados e seus respectivos mapeamentos. O nome da coluna é
        a chave para seu dicionário de mapeamento. Apenas para codificação ordinal.

    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Cleaner
    experionml.data_cleaning:Imputer
    experionml.data_cleaning:Pruner

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer
        from numpy.random import randint

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X["cat_feature_1"] = [f"x{i}" for i in randint(0, 2, len(X))]
        X["cat_feature_2"] = [f"x{i}" for i in randint(0, 3, len(X))]
        X["cat_feature_3"] = [f"x{i}" for i in randint(0, 20, len(X))]

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.X)

        experionml.encode(strategy="target", max_onehot=10, verbose=2)

        # Observe a coluna codificada one-hot com nome [feature]_[classe]
        print(experionml.X)
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Encoder
        from sklearn.datasets import load_breast_cancer
        from numpy.random import randint

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X["cat_feature_1"] = [f"x{i}" for i in randint(0, 2, len(X))]
        X["cat_feature_2"] = [f"x{i}" for i in randint(0, 3, len(X))]
        X["cat_feature_3"] = [f"x{i}" for i in randint(0, 20, len(X))]
        print(X)

        encoder = Encoder(strategy="target", max_onehot=10, verbose=2)
        X = encoder.fit_transform(X, y)

        # Observe a coluna codificada one-hot com nome [feature]_[classe]
        print(X)
        ```

    """

    def __init__(
        self,
        strategy: str | Transformer = "Target",
        *,
        max_onehot: IntLargerTwo | None = 10,
        ordinal: dict[str, Sequence[Any]] | None = None,
        infrequent_to_value: FloatLargerZero | None = None,
        value: str = "infrequent",
        n_jobs: NJobs = 1,
        verbose: Verbose = 0,
        **kwargs,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose)
        self.strategy = strategy
        self.max_onehot = max_onehot
        self.ordinal = ordinal
        self.infrequent_to_value = infrequent_to_value
        self.value = value
        self.kwargs = kwargs

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Ajusta aos dados.

        Deixar y=None pode levar a erros se o codificador `strategy`
        requer valores alvo. Para tarefas multioutput, apenas
        a primeira coluna alvo é usada para ajustar o codificador.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence or dataframe-like
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        Self
            Instância do estimador.

        """
        self.mapping_ = {}
        self._to_value = {}
        self._categories = {}

        strategies = {
            "backwarddifference": BackwardDifferenceEncoder,
            "basen": BaseNEncoder,
            "binary": BinaryEncoder,
            "catboost": CatBoostEncoder,
            "helmert": HelmertEncoder,
            "jamesstein": JamesSteinEncoder,
            "mestimate": MEstimateEncoder,
            "ordinal": OrdinalEncoder,
            "polynomial": PolynomialEncoder,
            "sum": SumEncoder,
            "target": TargetEncoder,
            "woe": WOEEncoder,
        }

        Xt = to_df(X)
        yt = to_tabular(y, index=Xt.index)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if isinstance(self.strategy, str):
            if self.strategy.lower().endswith("encoder"):
                self.strategy = self.strategy[:-7]  # Remove 'Encoder' do final
            if self.strategy.lower() not in strategies:
                raise ValueError(
                    f"Valor inválido para o parâmetro strategy, valor recebido: {self.strategy}. "
                    f"Escolha entre: {', '.join(strategies)}."
                )
            estimator = strategies[self.strategy.lower()]
        elif callable(self.strategy):
            estimator = self.strategy
        else:
            raise TypeError(
                f"Tipo inválido para o parâmetro strategy, valor recebido: {self.strategy}. "
                "Para estimadores personalizados, esperava-se uma classe, mas foi recebida uma instância."
            )

        if self.max_onehot is None:
            max_onehot = 0
        else:
            max_onehot = int(self.max_onehot)

        if self.infrequent_to_value:
            if self.infrequent_to_value < 1:
                infrequent_to_value = int(self.infrequent_to_value * len(Xt))
            else:
                infrequent_to_value = int(self.infrequent_to_value)

        self._log("Ajustando Encoder...", 1)

        encoders: dict[str, list[str]] = defaultdict(list)

        for name, column in Xt.select_dtypes(include=CAT_TYPES).items():  # type: ignore[arg-type]
            # Substitui classes infrequentes pela string em `value`
            if self.infrequent_to_value:
                values = column.value_counts()
                self._to_value[name] = values[values <= infrequent_to_value].index.tolist()
                Xt[name] = column.replace(self._to_value[name], self.value)

            # Obtém as categorias únicas antes do ajuste
            self._categories[name] = column.dropna().sort_values().unique().tolist()

            # Realiza o tipo de codificação de acordo com o número de valores únicos
            ordinal = self.ordinal or {}
            if name in ordinal or len(self._categories[name]) == 2:
                # Verifica se as classes fornecidas correspondem às da coluna
                ordinal_c = ordinal.get(str(name), self._categories[name])
                if column.nunique(dropna=True) != len(ordinal_c):
                    self._log(
                        f" --> O número de classes passado para a variável {name} no "
                        f"parâmetro ordinal ({len(ordinal_c)}) não corresponde ao número "
                        f"de classes nos dados ({column.nunique(dropna=True)}).",
                        1,
                        severity="warning",
                    )

                # Cria mapeamento personalizado de 0 a N - 1
                mapping: dict[Hashable, Scalar] = {v: i for i, v in enumerate(ordinal_c)}
                mapping.setdefault(np.nan, -1)  # Encoder sempre precisa do mapeamento de NaN
                self.mapping_[str(name)] = mapping

                encoders["ordinal"].append(str(name))
            elif 2 < len(self._categories[name]) <= max_onehot:
                encoders["onehot"].append(str(name))
            else:
                encoders["rest"].append(str(name))

        ordinal_enc = OrdinalEncoder(
            mapping=[{"col": c, "mapping": self.mapping_[c]} for c in encoders["ordinal"]],
            cols=encoders["ordinal"],
            handle_missing="return_nan",
            handle_unknown="value",
        )

        onehot_enc = OneHotEncoder(
            cols=encoders["onehot"],
            use_cat_names=True,
            handle_missing="return_nan",
            handle_unknown="value",
        )

        rest_enc = estimator(
            cols=encoders["rest"],
            handle_missing="return_nan",
            handle_unknown="value",
            **self.kwargs,
        )

        self._estimator = ColumnTransformer(
            transformers=[
                ("ordinal", ordinal_enc, encoders["ordinal"]),
                ("onehot", onehot_enc, encoders["onehot"]),
                ("rest", rest_enc, encoders["rest"]),
            ],
            remainder="passthrough",
            n_jobs=self.n_jobs,
            verbose_feature_names_out=False,
        ).fit(Xt, yt)

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Retorna os nomes das variáveis após a transformação.

        Parâmetros
        ----------
        input_features: sequence or None, default=None
            Usado apenas para validar os nomes das variáveis com os nomes
            observados durante o `fit`.

        Retorna
        -------
        np.ndarray
            Nomes das variáveis transformadas.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        # Remove colunas _nan (pois valores ausentes são propagados)
        cols = [c for c in self._estimator.get_feature_names_out() if not c.endswith("_nan")]

        return get_col_order(cols, self.feature_names_in_, self._estimator.feature_names_in_)

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Codifica os dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            DataFrame codificado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Codificando colunas categóricas...", 1)

        # Converte classes infrequentes para o valor especificado
        Xt = Xt.replace(self._to_value, self.value)

        for name, categories in self._categories.items():
            if name in self._estimator.transformers_[0][2]:
                estimator = self._estimator.transformers_[0][1]
            elif name in self._estimator.transformers_[1][2]:
                estimator = self._estimator.transformers_[1][1]
            else:
                estimator = self._estimator.transformers_[2][1]

            self._log(
                f" --> Aplicando {estimator.__class__.__name__[:-7]}-encoding à variável "
                f"{name}. Ela contém {Xt[name].nunique()} classes.",
                2,
            )

            # Conta os valores ausentes propagados
            if n_nans := Xt[name].isna().sum():
                self._log(f"   --> Propagando {n_nans} valores ausentes.", 2)

            # Verifica classes desconhecidas
            if uc := len(Xt[name].dropna()[~Xt[name].isin(categories)]):
                self._log(f"   --> Tratando {uc} classes desconhecidas.", 2)

        Xt = self._estimator.transform(Xt)

        return self._convert(Xt[self.get_feature_names_out()])


@beartype
class Imputer(TransformerMixin):
    """Trata valores ausentes nos dados.

    Imputa ou remove valores ausentes de acordo com a estratégia selecionada.
    Também remove linhas e colunas com muitos valores ausentes. Use
    o atributo `missing_` para personalizar o que é considerado como
    "valores ausentes".

    Esta classe pode ser acessada pelo experionml através do método [impute]
    [experionmlclassifier-impute]. Leia mais no [guia do usuário]
    [imputing-missing-values].

    Parâmetros
    ----------
    strat_num: int, float, str or callable, default="mean"
        Estratégia de imputação para colunas numéricas. Escolha entre:

        - "drop": Remove linhas contendo valores ausentes.
        - "mean": Imputa com a média da coluna.
        - "median": Imputa com a mediana da coluna.
        - "most_frequent": Imputa com o valor mais frequente.
        - "knn": Imputa usando uma abordagem K-Nearest Neighbors.
        - "iterative": Imputa usando um imputador multivariado.
        - "drift": Imputa valores usando um modelo [PolynomialTrend][].
        - "linear": Imputa usando interpolação linear.
        - "nearest": Imputa com o valor mais próximo.
        - "bfill": Imputa usando a próxima observação válida para preencher
           a lacuna.
        - "ffill": Imputa propagando a última observação válida
          para a próxima válida.
        - "random": Imputa com valores aleatórios entre o mínimo e o máximo
           da coluna.
        - int ou float: Imputa com o valor numérico fornecido.
        - callable: Substitui valores ausentes usando a estatística escalar
          retornada pelo callable aplicado a um array 1d denso
          contendo valores não ausentes de cada coluna.

    strat_cat: str, default="most_frequent"
        Estratégia de imputação para colunas categóricas. Escolha entre:

        - "drop": Remove linhas contendo valores ausentes.
        - "most_frequent": Imputa com o valor mais frequente.
        - str: Imputa com a string fornecida.

    max_nan_rows: int, float or None, default=None
        Número ou fração máxima de valores ausentes em uma linha
        (se maior, a linha é removida). Se None, ignora esta etapa.

    max_nan_cols: int, float or None, default=None
        Número ou fração máxima de valores ausentes em uma coluna
        (se maior, a coluna é removida). Se None, ignora esta etapa.

    n_jobs: int, default=1
        Número de núcleos a usar para processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usar todos os núcleos disponíveis.
        - Se <-1: Usar número de núcleos - 1 - valor.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], ex.:
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Engine de execução para [estimadores][estimator-acceleration].
        Se None, o valor padrão é usado. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.
        - 2 para exibir informações detalhadas.

    random_state: int or None, default=None
        Semente usada pelo gerador de números aleatórios. Se None, o gerador
        é o `RandomState` utilizado pelo `np.random`. Apenas
        usado quando strat_num="iterative".

    Atributos
    ----------
    missing_: list
        Valores considerados "ausentes". Os valores padrão são: None,
        NaN, NA, NaT, +inf, -inf, "", "?", "NA", "nan", "NaN", "NaT",
        "none", "None", "inf", "-inf". Note que None, NaN, NA, +inf e
        -inf são sempre considerados ausentes pois são incompatíveis
        com estimadores sklearn.

    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Balancer
    experionml.data_cleaning:Discretizer
    experionml.data_cleaning:Encoder

    Exemplos
    --------
    === "experionml"
        ```pycon
        import numpy as np
        from experionml import ExperionMLClassifier
        from numpy.random import randint
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Adiciona alguns valores ausentes aleatórios aos dados
        for i, j in zip(randint(0, X.shape[0], 600), randint(0, 4, 600)):
            X.iloc[i, j] = np.NaN

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.nans)

        experionml.impute(strat_num="median", max_nan_rows=0.1, verbose=2)

        print(experionml.n_nans)
        ```

    === "stand-alone"
        ```pycon
        import numpy as np
        from experionml.data_cleaning import Imputer
        from numpy.random import randint
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Adiciona alguns valores ausentes aleatórios aos dados
        for i, j in zip(randint(0, X.shape[0], 600), randint(0, 4, 600)):
            X.iloc[i, j] = np.nan

        imputer = Imputer(strat_num="median", max_nan_rows=0.1, verbose=2)
        X, y = imputer.fit_transform(X, y)

        print(X)
        ```

    """

    def __init__(
        self,
        strat_num: Scalar | NumericalStrats | Callable[[Sequence[Scalar]], Scalar] = "mean",
        strat_cat: str | CategoricalStrats = "most_frequent",
        *,
        max_nan_rows: FloatLargerZero | None = None,
        max_nan_cols: FloatLargerZero | None = None,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            verbose=verbose,
            random_state=random_state,
        )
        self.strat_num = strat_num
        self.strat_cat = strat_cat
        self.max_nan_rows = max_nan_rows
        self.max_nan_cols = max_nan_cols

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        Self
            Instância do estimador.

        """
        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if not hasattr(self, "missing_"):
            self.missing_ = DEFAULT_MISSING

        self._log("Ajustando Imputer...", 1)

        # Unifica todos os valores a imputar
        Xt = replace_missing(Xt, self.missing_)

        if self.max_nan_rows is not None:
            if self.max_nan_rows <= 1:
                self._max_nan_rows = int(Xt.shape[1] * self.max_nan_rows)
            else:
                self._max_nan_rows = int(self.max_nan_rows)

            Xt = Xt.dropna(axis=0, thresh=Xt.shape[1] - self._max_nan_rows)
            if Xt.empty:
                raise ValueError(
                    "Valor inválido para o parâmetro max_nan_rows, valor recebido: "
                    f"{self.max_nan_rows}. Todas as linhas contêm mais de "
                    f"{self._max_nan_rows} valores ausentes. Escolha um "
                    f"valor maior ou defina o parâmetro como None."
                )

        if self.max_nan_cols is not None:
            if self.max_nan_cols <= 1:
                max_nan_cols = int(Xt.shape[0] * self.max_nan_cols)
            else:
                max_nan_cols = int(self.max_nan_cols)

            Xt = Xt.drop(columns=Xt.columns[Xt.isna().sum() > max_nan_cols])

        # Carrega a classe imputadora do sklearn ou cuml (observe os módulos diferentes)
        SimpleImputer = self._get_est_class(
            name="SimpleImputer",
            module="preprocessing" if self.engine.estimator == "cuml" else "impute",
        )

        # Nota: missing_values=pd.NA também imputa np.NaN
        num_imputer: Estimator | Literal["passthrough"]
        if isinstance(self.strat_num, str):
            if self.strat_num in ("mean", "median", "most_frequent"):
                num_imputer = SimpleImputer(missing_values=pd.NA, strategy=self.strat_num)
            elif self.strat_num == "knn":
                num_imputer = KNNImputer()
            elif self.strat_num == "iterative":
                num_imputer = IterativeImputer(random_state=self.random_state)
            elif self.strat_num == "drop":
                num_imputer = "passthrough"
            else:
                num_imputer = make_sklearn(SktimeImputer)(
                    method=self.strat_num,
                    missing_values=[pd.NA],
                    random_state=self.random_state,
                )
        elif callable(self.strat_num):
            num_imputer = SimpleImputer(missing_values=pd.NA, strategy=self.strat_num)
        else:
            num_imputer = SimpleImputer(
                missing_values=pd.NA,
                strategy="constant",
                fill_value=self.strat_num,
            )

        cat_imputer: Estimator | Literal["passthrough"]
        if self.strat_cat == "most_frequent":
            cat_imputer = SimpleImputer(missing_values=pd.NA, strategy=self.strat_cat)
        elif self.strat_cat == "drop":
            cat_imputer = "passthrough"
        else:
            cat_imputer = SimpleImputer(
                missing_values=pd.NA,
                strategy="constant",
                fill_value=self.strat_cat,
            )

        ColumnTransformer = self._get_est_class("ColumnTransformer", "compose")

        self._estimator = ColumnTransformer(
            transformers=[
                ("num_imputer", num_imputer, list(Xt.select_dtypes(include="number"))),
                (
                    "cat_imputer",
                    cat_imputer,
                    list(Xt.select_dtypes(include=CAT_TYPES)),
                ),  # type:ignore[arg-type]
            ],
            remainder="passthrough",
            n_jobs=self.n_jobs,
            verbose_feature_names_out=False,
        ).fit(Xt)

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Retorna os nomes das variáveis após a transformação.

        Parâmetros
        ----------
        input_features: sequence or None, default=None
            Usado apenas para validar os nomes das variáveis com os nomes
            observados durante o `fit`.

        Retorna
        -------
        np.ndarray
            Nomes das variáveis transformadas.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        return np.array(
            [c for c in self.feature_names_in_ if c in self._estimator.get_feature_names_out()]
        )

    def transform(
        self,
        X: XConstructor,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Imputa os valores ausentes.

        Deixar y=None pode levar a inconsistências no
        comprimento dos dados entre X e y se linhas forem removidas durante
        a transformação.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        dataframe
            DataFrame imputado.

        series or dataframe
            Coluna alvo transformada. Retornada somente se fornecida.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)
        yt = to_tabular(y, index=Xt.index)

        num_imputer = self._estimator.named_transformers_["num_imputer"]
        cat_imputer = self._estimator.named_transformers_["cat_imputer"]

        get_stat = lambda est, n: est.statistics_[est.feature_names_in_.tolist().index(n)]

        self._log("Imputando valores ausentes...", 1)

        # Unifica todos os valores a imputar
        Xt = replace_missing(Xt, self.missing_)

        # Remove linhas com muitos valores ausentes
        if self.max_nan_rows is not None:
            length = len(Xt)
            Xt = Xt.dropna(axis=0, thresh=Xt.shape[1] - self._max_nan_rows)
            if diff := length - len(Xt):
                self._log(
                    f" --> Removendo {diff} amostras por conterem mais "
                    f"de {self._max_nan_rows} valores ausentes.",
                    2,
                )

        if self.strat_num == "drop":
            length = len(Xt)
            Xt = Xt.dropna(subset=self._estimator.transformers_[0][2])
            if diff := length - len(Xt):
                self._log(
                    f" --> Removendo {diff} amostras por conterem "
                    f"valores ausentes em colunas numéricas.",
                    2,
                )

        if self.strat_cat == "drop":
            length = len(Xt)
            Xt = Xt.dropna(subset=self._estimator.transformers_[1][2])
            if diff := length - len(Xt):
                self._log(
                    f" --> Removendo {diff} amostras por conterem "
                    f"valores ausentes em colunas categóricas.",
                    2,
                )

        # Exibe informações de imputação por variável
        for name, column in Xt.items():
            if nans := column.isna().sum():
                # Remove colunas com muitos valores ausentes
                if name not in self._estimator.feature_names_in_:
                    self._log(
                        f" --> Removendo a variável {name}. Ela contém {nans} "
                        f"({nans * 100 // len(Xt)}%) valores ausentes.",
                        2,
                    )
                    Xt = Xt.drop(columns=name)
                    continue

                if name in getattr(num_imputer, "feature_names_in_", []):
                    if not isinstance(self.strat_num, str):
                        self._log(
                            f" --> Imputando {nans} valores ausentes com o "
                            f"número '{self.strat_num}' na coluna {name}.",
                            2,
                        )
                    elif self.strat_num in ("knn", "iterative"):
                        self._log(
                            f" --> Imputando {nans} valores ausentes usando "
                            f"o imputador {self.strat_num} na coluna {name}.",
                            2,
                        )
                    elif self.strat_num in ("mean", "median", "most_frequent"):
                        self._log(
                            f" --> Imputando {nans} valores ausentes com {self.strat_num} "
                            f"({np.round(get_stat(num_imputer, name), 2)}) na coluna {name}.",
                            2,
                        )
                    else:
                        self._log(
                            f" --> Imputando {nans} valores ausentes com {self.strat_num} "
                            f"na coluna {name}.",
                            2,
                        )
                elif name in getattr(cat_imputer, "feature_names_in_", []):
                    if self.strat_cat == "most_frequent":
                        self._log(
                            f" --> Imputando {nans} valores ausentes com most_frequent "
                            f"({get_stat(cat_imputer, name)}) na coluna {name}.",
                            2,
                        )
                    elif self.strat_cat != "drop":
                        self._log(
                            f" --> Imputando {nans} valores ausentes com o valor "
                            f"'{self.strat_cat}' na coluna {name}.",
                            2,
                        )

        Xt = self._estimator.transform(Xt)

        # Torna y consistente com X
        if yt is not None:
            yt = yt.loc[yt.index.isin(Xt.index)]

        # Reordena as colunas na ordem original
        Xt = Xt[self.get_feature_names_out()]

        return variable_return(self._convert(Xt), self._convert(yt))


@beartype
class Normalizer(TransformerMixin, OneToOneFeatureMixin):
    """Transforma os dados para seguirem uma distribuição Normal/Gaussiana.

    Esta transformação é útil para problemas de modelagem relacionados à
    heterocedasticidade (variância não constante), ou outras situações
    em que a normalidade é desejada. Valores ausentes são ignorados no
    ajuste e mantidos na transformação. Colunas categóricas são ignoradas.

    Esta classe pode ser acessada pelo experionml através do método [normalize]
    [experionmlclassifier-normalize]. Leia mais no [guia do usuário]
    [normalizing-the-feature-set].

    !!! warning
        A estratégia quantile realiza uma transformação não linear.
        Isso pode distorcer correlações lineares entre variáveis medidas
        na mesma escala, mas torna variáveis medidas em escalas diferentes
        mais diretamente comparáveis.

    !!! note
        As estratégias yeojohnson e boxcox escalonam os dados após
        a transformação. Use `kwargs` para alterar esse comportamento.

    Parâmetros
    ----------
    strategy: str, default="yeojohnson"
        A estratégia de transformação. Escolha entre:

        - "[yeojohnson][]"
        - "[boxcox][]" (funciona apenas com valores estritamente positivos)
        - "[quantile][]": Transforma variáveis usando informações de quantis.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], ex.:
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Engine de execução para [estimadores][estimator-acceleration].
        Se None, o valor padrão é usado. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.

    random_state: int or None, default=None
        Semente para a estratégia quantile. Se None, o gerador
        é o `RandomState` utilizado pelo `np.random`.

    **kwargs
        Argumentos de palavra-chave adicionais para o estimador `strategy`.

    Atributos
    ----------
    [strategy]_: sklearn transformer
        Objeto com o qual os dados são transformados, ex.:
        `normalizer.yeojohnson` para a estratégia padrão.

    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Cleaner
    experionml.data_cleaning:Pruner
    experionml.data_cleaning:Scaler

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.dataset)

        experionml.plot_distribution(columns=0)

        experionml.normalize(verbose=2)

        print(experionml.dataset)

        experionml.plot_distribution(columns=0)
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Normalizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        normalizer = Normalizer(verbose=2)
        X = normalizer.fit_transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        strategy: NormalizerStrats = "yeojohnson",
        *,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
        **kwargs,
    ):
        super().__init__(
            device=device,
            engine=engine,
            verbose=verbose,
            random_state=random_state,
        )
        self.strategy = strategy
        self.kwargs = kwargs

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        Self
            Instância do estimador.

        """
        strategies = {
            "yeojohnson": "PowerTransformer",
            "boxcox": "PowerTransformer",
            "quantile": "QuantileTransformer",
        }

        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        if self.strategy in ("yeojohnson", "boxcox"):
            estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
            self._estimator = estimator(
                method=self.strategy[:3] + "-" + self.strategy[3:],
                **self.kwargs,
            )
        elif self.strategy == "quantile":
            kwargs = self.kwargs.copy()
            estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
            self._estimator = estimator(
                output_distribution=kwargs.pop("output_distribution", "normal"),
                random_state=kwargs.pop("random_state", self.random_state),
                **kwargs,
            )

        num_cols = Xt.select_dtypes(include="number")

        if num_cols.empty:
            raise ValueError(
                "A classe Normalizer não encontrou colunas durante o ajuste. "
                "Verifique se X contém colunas numéricas."
            )

        self._log("Ajustando Normalizer...", 1)
        self._estimator.fit(num_cols)

        # Adiciona o estimador como atributo à instância
        setattr(self, f"{self.strategy}_", self._estimator)

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Aplica as transformações aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            DataFrame normalizado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Normalizando as variáveis...", 1)

        Xt.update(self._estimator.transform(Xt[self._estimator.feature_names_in_]))

        return self._convert(Xt)

    def inverse_transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Aplica a transformação inversa aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            DataFrame original.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Revertendo a normalização das variáveis...", 1)

        out: np.ndarray = self._estimator.inverse_transform(Xt[self._estimator.feature_names_in_])

        Xt.update(to_df(out, index=Xt.index, columns=self._estimator.feature_names_in_))

        return self._convert(Xt)


@beartype
class Pruner(TransformerMixin, OneToOneFeatureMixin):
    """Remove valores atípicos dos dados.

    Substitui ou remove valores atípicos. A definição de valor atípico depende
    da estratégia selecionada e pode variar bastante entre elas.
    Colunas categóricas são ignoradas.

    Esta classe pode ser acessada pelo experionml através do método [prune]
    [experionmlclassifier-prune]. Leia mais no [guia do usuário]
    [handling-outliers].

    !!! info
        Os engines "sklearnex" e "cuml" são suportados apenas para
        strategy="dbscan".

    Parâmetros
    ----------
    strategy: str or sequence, default="zscore"
        Estratégia com a qual selecionar os valores atípicos. Se uma sequência de
        estratégias, apenas amostras marcadas como atípicas por todas as estratégias
        escolhidas são removidas. Escolha entre:

        - "zscore": Z-score de cada valor de dado.
        - "[iforest][]": Isolation Forest.
        - "[ee][]": Elliptic Envelope.
        - "[lof][]": Local Outlier Factor.
        - "[svm][]": One-class SVM.
        - "[dbscan][]": Density-Based Spatial Clustering.
        - "[hdbscan][]": Hierarchical Density-Based Spatial Clustering.
        - "[optics][]": Abordagem de clustering similar ao DBSCAN.

    method: int, float or str, default="drop"
        Método a aplicar nos valores atípicos. Apenas a estratégia zscore
        aceita outro método além de "drop". Escolha entre:

        - "drop": Remove qualquer amostra com valores atípicos.
        - "minmax": Substitui o valor atípico pelo mínimo/máximo da coluna.
        - Qualquer valor numérico com o qual substituir os valores atípicos.

    max_sigma: int or float, default=3
        Máximo de desvios padrão permitidos em relação à média da coluna.
        Se maior, é considerado um valor atípico. Apenas para strategy="zscore".

    include_target: bool, default=False
        Se deve incluir a coluna alvo na busca por valores atípicos.
        Pode ser útil para tarefas de regressão. Apenas
        para strategy="zscore".

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], ex.:
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Engine de execução para [estimadores][estimator-acceleration].
        Se None, o valor padrão é usado. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.
        - 2 para exibir informações detalhadas.

    **kwargs
        Argumentos de palavra-chave adicionais para o estimador `strategy`. Se
        sequência de estratégias, os parâmetros devem ser fornecidos em um
        dicionário com o nome da estratégia como chave.

    Atributos
    ----------
    [strategy]_: sklearn estimator
        Objeto usado para remover os valores atípicos, ex.: `pruner.iforest` para
        a estratégia isolation forest. Não disponível para strategy="zscore".

    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Balancer
    experionml.data_cleaning:Normalizer
    experionml.data_cleaning:Scaler

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.dataset)

        experionml.prune(stratgey="iforest", verbose=2)

        # Observe o número reduzido de linhas
        print(experionml.dataset)

        experionml.plot_distribution(columns=0)
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Normalizer
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        normalizer = Normalizer(verbose=2)
        X = normalizer.fit_transform(X)

        # Observe o número reduzido de linhas
        print(X)
        ```

    """

    _train_only = True

    def __init__(
        self,
        strategy: PrunerStrats | Sequence[PrunerStrats] = "zscore",
        *,
        method: Scalar | Literal["drop", "minmax"] = "drop",
        max_sigma: FloatLargerZero = 3,
        include_target: Bool = False,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose)
        self.strategy = strategy
        self.method = method
        self.max_sigma = max_sigma
        self.include_target = include_target
        self.kwargs = kwargs

    def transform(
        self,
        X: XConstructor,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Aplica a estratégia de detecção de valores atípicos.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        dataframe
            Conjunto de variáveis transformado.

        series or dataframe
            Coluna alvo transformada. Retornada somente se fornecida.

        """
        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        yt = to_tabular(y, index=Xt.index)

        # Estimadores com seus módulos
        strategies = {
            "iforest": ["IsolationForest", "ensemble"],
            "ee": ["EllipticEnvelope", "covariance"],
            "lof": ["LocalOutlierFactor", "neighbors"],
            "svm": ["OneClassSVM", "svm"],
            "dbscan": ["DBSCAN", "cluster"],
            "hdbscan": ["HDBSCAN", "cluster"],
            "optics": ["OPTICS", "cluster"],
        }

        for strat in lst(self.strategy):
            if strat != "zscore" and str(self.method) != "drop":
                raise ValueError(
                    "Valor inválido para o parâmetro method. Apenas a estratégia zscore "
                    f"aceita outro método além de 'drop', valor recebido: {self.method}."
                )

        # Aloca kwargs para cada estimador
        kwargs: dict[PrunerStrats, dict[str, Any]] = {}
        for strat in lst(self.strategy):
            kwargs[strat] = {}
            for key, value in self.kwargs.items():
                # Parâmetros apenas para este estimador
                if key == strat:
                    kwargs[strat].update(value)
                # Parâmetros para todos os estimadores
                elif key not in lst(self.strategy):
                    kwargs[strat].update({key: value})

        self._log("Removendo valores atípicos...", 1)

        # Prepara o dataset (une com y e exclui colunas categóricas)
        objective = merge(Xt, yt) if self.include_target and yt is not None else Xt
        objective = objective.select_dtypes(include=["number"])

        outliers = []
        for strat in lst(self.strategy):
            if strat == "zscore":
                # stats.zscore funciona apenas com tipos numpy, portanto, converte
                z_scores = zscore(objective.values.astype(float), nan_policy="propagate")

                if not isinstance(self.method, str):
                    cond = np.abs(z_scores) > self.max_sigma
                    objective = objective.mask(cond, self.method)
                    self._log(
                        f" --> Substituindo {cond.sum()} valores atípicos por {self.method}.",
                        2,
                    )

                elif self.method.lower() == "minmax":
                    counts = 0
                    for i, col in enumerate(objective):
                        # Substitui outliers por NaN e depois pelo máximo,
                        # para que o máximo não seja calculado com os outliers
                        cond1 = z_scores[:, i] > self.max_sigma
                        mask = objective[col].mask(cond1, np.nan)
                        objective[col] = mask.replace(np.nan, mask.max(skipna=True))

                        # Substitui outliers pelo mínimo
                        cond2 = z_scores[:, i] < -self.max_sigma
                        mask = objective[col].mask(cond2, np.nan)
                        objective[col] = mask.replace(np.nan, mask.min(skipna=True))

                        # Soma o número de substituições
                        counts += cond1.sum() + cond2.sum()

                    self._log(
                        f" --> Substituindo {counts} valores atípicos "
                        "pelo mínimo ou máximo da coluna.",
                        2,
                    )

                elif self.method.lower() == "drop":
                    mask = (np.abs(zscore(z_scores)) <= self.max_sigma).all(axis=1)
                    outliers.append(mask)
                    if len(lst(self.strategy)) > 1:
                        self._log(
                            f" --> A estratégia zscore detectou "
                            f"{len(mask) - sum(mask)} valores atípicos.",
                            2,
                        )

            else:
                estimator = self._get_est_class(*strategies[strat])(**kwargs[strat])
                mask = estimator.fit_predict(objective) >= 0
                outliers.append(mask)
                if len(lst(self.strategy)) > 1:
                    self._log(
                        f" --> {estimator.__class__.__name__} "
                        f"detectou {len(mask) - sum(mask)} valores atípicos.",
                        2,
                    )

                # Add the estimator as attribute to the instance
                setattr(self, f"{strat}_", estimator)

        if outliers:
            # Seleciona outliers pela intersecção das estratégias
            outlier_rows = [any(strats) for strats in zip(*outliers, strict=True)]
            self._log(
                f" --> Removendo {len(outlier_rows) - sum(outlier_rows)} valores atípicos.",
                2,
            )

            # Mantém apenas as amostras não atípicas nos dados
            Xt = Xt[outlier_rows]
            if yt is not None:
                yt = yt[outlier_rows]

        else:
            # Substitui as colunas em X com os novos valores de objective
            Xt.update(objective)

        return variable_return(self._convert(Xt), self._convert(yt))


@beartype
class Scaler(TransformerMixin, OneToOneFeatureMixin):
    """Escalona os dados.

    Aplica uma das estratégias de escalonamento do sklearn. Colunas categóricas
    são ignoradas.

    Esta classe pode ser acessada pelo experionml através do método [scale]
    [experionmlclassifier-scale]. Leia mais no [guia do usuário]
    [scaling-the-feature-set].

    Parâmetros
    ----------
    strategy: str, default="standard"
        Estratégia com a qual escalonar os dados. Escolha entre:

        - "[standard][]": Remove a média e escala para variância unitária.
        - "[minmax][]": Escala as variáveis para um intervalo dado.
        - "[maxabs][]": Escala as variáveis pelo seu valor absoluto máximo.
        - "[robust][]": Escala usando estatísticas robustas a valores atípicos.

    include_binary: bool, default=False
        Se deve escalonar colunas binárias (apenas 0s e 1s).

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], ex.:
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Engine de execução para [estimadores][estimator-acceleration].
        Se None, o valor padrão é usado. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.

    **kwargs
        Argumentos de palavra-chave adicionais para o estimador `strategy`.

    Atributos
    ----------
    [strategy]_: sklearn transformer
        Objeto com o qual os dados são escalonados, ex.:
        `scaler.standard` para a estratégia padrão.

    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Balancer
    experionml.data_cleaning:Normalizer
    experionml.data_cleaning:Scaler

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.dataset)

        experionml.scale(verbose=2)

        # Observe o número reduzido de linhas
        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Scaler
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        scaler = Scaler(verbose=2)
        X = scaler.fit_transform(X)

        # Observe o número reduzido de linhas
        print(X)
        ```

    """

    def __init__(
        self,
        strategy: ScalerStrats = "standard",
        *,
        include_binary: Bool = False,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose)
        self.strategy = strategy
        self.include_binary = include_binary
        self.kwargs = kwargs

    def fit(
        self,
        X: XConstructor,
        y: YConstructor | None = None,
        sample_weight: Sequence[Scalar] | None = None,
    ) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        sample_weight: sequence or None, default=None
            Pesos das amostras com shape=(n_amostras,).

        Retorna
        -------
        Self
            Instância do estimador.

        """
        strategies = {
            "standard": "StandardScaler",
            "minmax": "MinMaxScaler",
            "maxabs": "MaxAbsScaler",
            "robust": "RobustScaler",
        }

        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        num_cols = Xt.select_dtypes(include="number")

        if not self.include_binary:
            num_cols = Xt[
                [n for n, c in num_cols.items() if ~np.isin(c.dropna().unique(), [0, 1]).all()]
            ]

        if num_cols.empty:
            raise ValueError(
                "A classe Scaler não encontrou colunas durante o ajuste. Verifique "
                "se X contém colunas numéricas ou se existem colunas não binárias "
                "quando include_binary=False."
            )

        self._log("Ajustando Scaler...", 1)

        estimator = self._get_est_class(strategies[self.strategy], "preprocessing")
        self._estimator = estimator(**self.kwargs)

        if "sample_weight" in sign(estimator.fit):
            self._estimator.fit(num_cols, sample_weight=sample_weight)
        else:
            self._estimator.fit(num_cols)

        # Adiciona o estimador como atributo à instância
        setattr(self, f"{self.strategy}_", self._estimator)

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Padroniza os dados centralizando e escalonando.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            DataFrame escalonado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Escalonando as variáveis...", 1)

        Xt.update(self._estimator.transform(Xt[self._estimator.feature_names_in_]))

        return self._convert(Xt)

    def inverse_transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Aplica a transformação inversa aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de variáveis com shape=(n_amostras, n_variáveis).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            DataFrame escalonado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Revertendo o escalonamento das variáveis...", 1)

        out: np.ndarray = self._estimator.inverse_transform(Xt[self._estimator.feature_names_in_])

        Xt.update(to_df(out, index=Xt.index, columns=self._estimator.feature_names_in_))

        return self._convert(Xt)
