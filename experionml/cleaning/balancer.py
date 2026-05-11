from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd
from beartype import beartype
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
from sklearn.base import OneToOneFeatureMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from typing_extensions import Self

from experionml.utils.types import (
    Estimator,
    IntLargerEqualZero,
    NJobs,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
    YReturn,
)
from experionml.utils.utils import (
    check_is_fitted,
    to_df,
    to_series,
    to_tabular,
)


from experionml.cleaning.base import TransformerMixin

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


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
