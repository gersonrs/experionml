from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np
from beartype import beartype
from scipy.stats import zscore
from sklearn.base import OneToOneFeatureMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

from experionml.utils.types import (
    Bool,
    Engine,
    FloatLargerZero,
    PrunerStrats,
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
    lst,
    merge,
    to_df,
    to_tabular,
    variable_return,
)


from experionml.cleaning.base import TransformerMixin

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


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
