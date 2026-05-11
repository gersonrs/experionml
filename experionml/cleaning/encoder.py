from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from typing import Any, TypeVar

import numpy as np
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
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.utils.validation import _check_feature_names_in
from typing_extensions import Self

from experionml.utils.constants import CAT_TYPES
from experionml.utils.types import (
    FloatLargerZero,
    IntLargerTwo,
    NJobs,
    Scalar,
    Sequence,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
)
from experionml.utils.utils import (
    check_is_fitted,
    get_col_order,
    to_df,
    to_tabular,
)


from experionml.cleaning.base import TransformerMixin

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


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
