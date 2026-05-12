from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeVar

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.utils.validation import _check_feature_names_in
from sktime.transformations.series.impute import Imputer as SktimeImputer
from typing_extensions import Self

from experionml.cleaning.base import TransformerMixin
from experionml.utils.constants import CAT_TYPES, DEFAULT_MISSING
from experionml.utils.types import (
    CategoricalStrats,
    Engine,
    Estimator,
    FloatLargerZero,
    IntLargerEqualZero,
    NJobs,
    NumericalStrats,
    Scalar,
    Sequence,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
    YReturn,
)
from experionml.utils.utils import (
    check_is_fitted,
    make_sklearn,
    replace_missing,
    to_df,
    to_tabular,
    variable_return,
)

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


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
            self.missing_ = list(DEFAULT_MISSING)

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
                ),  # type: ignore[arg-type]
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
