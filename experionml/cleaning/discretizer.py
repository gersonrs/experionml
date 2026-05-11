from __future__ import annotations

from collections.abc import Hashable
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.base import OneToOneFeatureMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from typing_extensions import Self

from experionml.utils.types import (
    Bins,
    DiscretizerStrats,
    Engine,
    Estimator,
    IntLargerEqualZero,
    Scalar,
    Sequence,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
    sequence_t,
)
from experionml.utils.utils import (
    check_is_fitted,
    sign,
    to_df,
)


from experionml.cleaning.base import TransformerMixin

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


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
