from __future__ import annotations

from typing import TypeVar

import numpy as np
from beartype import beartype
from sklearn.base import OneToOneFeatureMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from typing_extensions import Self

from experionml.utils.types import (
    Bool,
    Engine,
    Scalar,
    ScalerStrats,
    Sequence,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
)
from experionml.utils.utils import (
    check_is_fitted,
    sign,
    to_df,
)


from experionml.cleaning.base import TransformerMixin

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


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
