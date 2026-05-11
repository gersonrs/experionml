from __future__ import annotations

from typing import TypeVar

import numpy as np
from beartype import beartype
from sklearn.base import OneToOneFeatureMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from typing_extensions import Self

from experionml.utils.types import (
    Engine,
    IntLargerEqualZero,
    NormalizerStrats,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
)
from experionml.utils.utils import (
    check_is_fitted,
    to_df,
)


from experionml.cleaning.base import TransformerMixin

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


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
