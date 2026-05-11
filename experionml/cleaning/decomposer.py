from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

from beartype import beartype
from sklearn.base import OneToOneFeatureMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sktime.transformations.series.detrend import (
    ConditionalDeseasonalizer,
    Deseasonalizer,
    Detrender,
)
from typing_extensions import Self

from experionml.basetransformer import BaseTransformer
from experionml.utils.types import (
    Bool,
    IntLargerEqualZero,
    IntLargerZero,
    NJobs,
    Predictor,
    SeasonalityModels,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
)
from experionml.utils.utils import (
    Goal,
    check_is_fitted,
    to_df,
)


from experionml.cleaning.base import TransformerMixin

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


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
