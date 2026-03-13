from __future__ import annotations

from collections.abc import Hashable
from random import sample
from typing import Any, Literal, cast

import featuretools as ft
import numpy as np
import pandas as pd
from beartype import beartype
from gplearn.genetic import SymbolicTransformer
from scipy import stats
from sklearn.base import is_classifier
from sklearn.feature_selection import (
    RFE,
    RFECV,
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import _check_feature_names_in
from typing_extensions import Self
from zoofs import (
    DragonFlyOptimization,
    GeneticOptimization,
    GreyWolfOptimization,
    HarrisHawkOptimization,
    ParticleSwarmOptimization,
)

from experionml.basetransformer import BaseTransformer
from experionml.data_cleaning import Scaler, TransformerMixin
from experionml.utils.types import (
    Bool,
    Engine,
    FeatureSelectionSolvers,
    FeatureSelectionStrats,
    FloatLargerEqualZero,
    FloatLargerZero,
    FloatZeroToOneInc,
    IntLargerEqualZero,
    IntLargerZero,
    NJobs,
    Operators,
    Scalar,
    Sequence,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
)
from experionml.utils.utils import (
    Goal,
    Task,
    check_is_fitted,
    check_scaling,
    get_custom_scorer,
    is_sparse,
    lst,
    merge,
    sign,
    to_df,
    to_tabular,
)


@beartype
class FeatureExtractor(TransformerMixin):
    """Extrai atributos de colunas de data e hora.

    Cria novos atributos extraindo elementos de data e hora, como dia,
    mês, ano etc., das colunas fornecidas. Colunas com dtype
    `datetime64` são usadas diretamente. Colunas categóricas que podem
    ser convertidas com sucesso para um formato datetime, isto é, com
    menos de 30% de valores NaT após a conversão, também são usadas.

    Esta classe pode ser acessada a partir do experionml por meio do
    método [feature_extraction][experionmlclassifier-feature_extraction].
    Leia mais no [guia do usuário][extracting-datetime-features].

    !!! warning
        Algoritmos baseados em árvores de decisão constroem suas regras
        de divisão considerando um atributo por vez. Isso significa que
        eles não processam corretamente atributos cíclicos, já que os
        atributos sin/cos deveriam ser considerados como um único sistema
        de coordenadas.

    Parâmetros
    ----------
    features: str or sequence, default=("year", "month", "day")
        Atributos a criar a partir das colunas datetime. Observe que
        atributos criados com variância zero, por exemplo, o atributo
        hour em uma coluna que contém apenas datas, são ignorados. Os
        valores permitidos são atributos de `pandas.Series.dt`.

    fmt: str, dict or None, default=None
        Formato (`strptime`) das colunas categóricas que precisam ser
        convertidas para datetime. Se for um dict, usa o nome da coluna
        como chave e o formato como valor. Se for None, o formato é
        inferido automaticamente a partir do primeiro valor não NaN.
        Valores que não puderem ser convertidos são retornados como `NaT`.

    encoding_type: str, default="ordinal"
        Tipo de codificação a ser usado. Escolha entre:

        - "ordinal": Codifica os atributos em ordem crescente.
        - "cyclic": Codifica os atributos usando seno e cosseno para
          capturar sua natureza cíclica. Essa abordagem cria duas colunas
          para cada atributo. Atributos não cíclicos continuam usando
          codificação ordinal.

    from_index: bool, default=False
        Indica se o índice deve ser usado como coluna datetime a ser convertida.

    drop_columns: bool, default=True
        Indica se as colunas originais devem ser removidas após a
        transformação. Esse parâmetro é ignorado se `from_index=True`.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    Atributos
    ---------
    feature_names_in_: np.ndarray
        Nomes dos atributos vistos durante o `fit`.

    n_features_in_: int
        Número de atributos vistos durante o `fit`.

    See Also
    --------
    experionml.feature_engineering:FeatureGenerator
    experionml.feature_engineering:FeatureGrouper
    experionml.feature_engineering:FeatureSelector

    Examples
    --------
    === "experionml"
        ```pycon
        import pandas as pd
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Adiciona uma coluna datetime
        X["date"] = pd.date_range(start="1/1/2018", periods=len(X))

        experionml = ExperionMLClassifier(X, y)
        experionml.feature_extraction(features=["day"], fmt="%d/%m/%Y", verbose=2)

        # Observe a coluna date_day
        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        import pandas as pd
        from experionml.feature_engineering import FeatureExtractor
        from sklearn.datasets import load_breast_cancer

        X, _ = load_breast_cancer(return_X_y=True, as_frame=True)

        # Adiciona uma coluna datetime
        X["date"] = pd.date_range(start="1/1/2018", periods=len(X))

        fe = FeatureExtractor(features=["day"], fmt="%Y-%m-%d", verbose=2)
        X = fe.transform(X)

        # Observe a coluna date_day
        print(X)
        ```

    """

    def __init__(
        self,
        features: str | Sequence[str] = ("year", "month", "day"),
        fmt: str | dict[str, str] | None = None,
        *,
        encoding_type: Literal["ordinal", "cyclic"] = "ordinal",
        drop_columns: Bool = True,
        from_index: Bool = False,
        verbose: Verbose = 0,
    ):
        super().__init__(verbose=verbose)
        self.fmt = fmt
        self.features = features
        self.encoding_type = encoding_type
        self.drop_columns = drop_columns
        self.from_index = from_index

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Extrai os novos atributos.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de atributos com shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para manter a continuidade da API.

        Retorna
        -------
        dataframe
            Conjunto de atributos transformado.

        """
        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))

        self._log("Extraindo atributos datetime...", 1)

        if self.from_index:
            if hasattr(Xt.index, "to_timestamp"):
                Xc = pd.DataFrame(Xt.index.to_timestamp())
                order = Xc.columns.tolist() + Xt.columns.tolist()
            else:
                raise ValueError("Não foi possível converter o índice para um formato timestamp.")
        else:
            Xc = Xt.select_dtypes(exclude="number")
            order = Xt.columns.tolist()

        X_new = pd.DataFrame(index=Xt.index)
        for name, column in Xc.items():
            col_dt = pd.to_datetime(
                arg=column,
                errors="coerce",  # Converte para NaT se não conseguir formatar
                format=self.fmt.get(str(name)) if isinstance(self.fmt, dict) else self.fmt,
            )

            # Se mais de 30% dos valores forem NaT, a conversão falhou
            if col_dt.isna().sum() / len(Xc) >= 0.3:
                continue  # Ignora esta coluna

            self._log(f" --> Extraindo atributos da coluna {name}.", 1)

            # Extrai atributos da coluna datetime
            # Inverte para manter a ordem de atributos fornecida
            for fx in map(str.lower, reversed(lst(self.features))):
                if hasattr(col_dt.dt, fx.lower()):
                    series = getattr(col_dt.dt, fx)
                else:
                    raise ValueError(
                        "Valor inválido para o parâmetro feature. O valor "
                        f"{fx.lower()} não é um atributo de pd.Series.dt."
                    )

                if not isinstance(series, pd.Series):
                    self._log(
                        f"   --> A extração do atributo {fx} "
                        "falhou. O resultado não é um Series.dt.",
                        2,
                    )
                    continue  # Ignora se a informação não estiver presente no formato
                elif (series == series[0]).all():
                    continue  # Ignora se o atributo resultante tiver variância zero

                min_val: int = 0
                max_val: Scalar | pd.Series | None = None  # None se não for cíclico
                if self.encoding_type == "cyclic":
                    if fx == "microsecond":
                        min_val, max_val = 0, 1e6 - 1
                    elif fx in ("second", "minute"):
                        min_val, max_val = 0, 59
                    elif fx == "hour":
                        min_val, max_val = 0, 23
                    elif fx in ("weekday", "dayofweek", "day_of_week"):
                        min_val, max_val = 0, 6
                    elif fx in ("day", "dayofmonth", "day_of_month"):
                        min_val, max_val = 1, col_dt.dt.daysinmonth
                    elif fx in ("dayofyear", "day_of_year"):
                        min_val = 1
                        max_val = pd.Series([365 if i else 366 for i in col_dt.dt.is_leap_year])
                    elif fx == "month":
                        min_val, max_val = 1, 12
                    elif fx == "quarter":
                        min_val, max_val = 1, 4

                new_name = f"{name}_{fx}"
                if self.encoding_type == "ordinal" or max_val is None:
                    self._log(f"   --> Criando atributo {new_name}.", 2)
                    X_new[new_name] = series.to_numpy()
                    order.insert(order.index(str(name)) + 1, new_name)
                elif self.encoding_type == "cyclic":
                    self._log(f"   --> Criando atributo cíclico {new_name}.", 2)
                    pos = 2 * np.pi * (series.to_numpy() - min_val) / np.array(max_val)
                    X_new[f"{new_name}_sin"] = np.sin(pos)
                    X_new[f"{new_name}_cos"] = np.cos(pos)
                    order.insert(order.index(str(name)) + 1, f"{new_name}_sin")
                    order.insert(order.index(str(name)) + 2, f"{new_name}_cos")

            # Remove a coluna original
            if self.drop_columns or self.from_index:
                order.remove(str(name))

        return self._convert(merge(X_new, Xt)[order])


@beartype
class FeatureGenerator(TransformerMixin):
    r"""Gera novos atributos.

    Cria novas combinações de atributos existentes para capturar as
    relações não lineares entre os atributos originais.

    Esta classe pode ser acessada a partir do experionml por meio do
    método [feature_generation][experionmlclassifier-feature_generation].
    Leia mais no [guia do usuário][generating-new-features].

    !!! warning
        * O uso dos operadores `div`, `log` ou `sqrt` pode retornar novos
          atributos com valores `inf` ou `NaN`. Verifique os warnings que
          possam aparecer ou use o atributo [nans][experionmlclassifier-nans]
          do experionml.
        * Ao usar dfs com `n_jobs>1`, certifique-se de proteger o código
          com `if __name__ == "__main__"`. O Featuretools usa
          [dask](https://dask.org/), que utiliza multiprocessing do Python
          para paralelização. O método spawn do multiprocessing inicia um
          novo processo Python, exigindo a importação do módulo `__main__`
          antes de executar sua tarefa.
        * gfg pode ser lento para populações muito grandes.

    !!! tip
        O dfs pode criar muitos atributos novos e nem todos serão úteis.
        Use a classe [FeatureSelector][] para reduzir a quantidade de
        atributos.

    Parâmetros
    ----------
    strategy: str, default="dfs"
        Estratégia para criar novos atributos. Escolha entre:

        - "[dfs][]": Deep Feature Synthesis.
        - "[gfg][]": Genetic Feature Generation.

    n_features: int or None, default=None
        Número máximo de atributos recém-gerados a adicionar ao dataset.
        Se for None, seleciona todos os atributos criados.

    operators: str, sequence or None, default=None
        Operadores matemáticos a aplicar aos atributos. Use None para
        usar todos. Escolha entre: `add`, `sub`, `mul`, `div`, `abs`,
        `sqrt`, `log`, `inv`, `sin`, `cos`, `tan`.

    n_jobs: int, default=1
        Número de núcleos a usar no processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usa todos os núcleos disponíveis.
        - Se <-1: Usa número de núcleos - 1 + `n_jobs`.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    random_state: int or None, default=None
        Semente usada pelo gerador de números aleatórios. Se for None,
        o gerador usado é o `RandomState` de `np.random`.

    **kwargs
        Argumentos nomeados adicionais para a instância de
        SymbolicTransformer. Apenas para a estratégia gfg.

    Atributos
    ---------
    gfg_: [SymbolicTransformer][]
        Objeto usado para calcular os atributos genéticos. Disponível
        apenas quando strategy="gfg".

    genetic_features_: pd.DataFrame
        Informações sobre os novos atributos não lineares criados.
        Disponível apenas quando strategy="gfg". As colunas incluem:

        - **name:** Nome do atributo, gerado automaticamente.
        - **description:** Operadores usados para criar esse atributo.
        - **fitness:** Pontuação de fitness.

    feature_names_in_: np.ndarray
        Nomes dos atributos vistos durante o `fit`.

    n_features_in_: int
        Número de atributos vistos durante o `fit`.

    See Also
    --------
    experionml.feature_engineering:FeatureExtractor
    experionml.feature_engineering:FeatureGrouper
    experionml.feature_engineering:FeatureSelector

    Examples
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y)
        experionml.feature_generation(strategy="dfs", n_features=5, verbose=2)

        # Observe a coluna texture error / worst symmetry
        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        from experionml.feature_engineering import FeatureGenerator
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        fg = FeatureGenerator(strategy="dfs", n_features=5, verbose=2)
        X = fg.fit_transform(X, y)

        # Observe a coluna radius error * worst smoothness
        print(X)
        ```

    """

    def __init__(
        self,
        strategy: Literal["dfs", "gfg"] = "dfs",
        *,
        n_features: IntLargerZero | None = None,
        operators: Operators | Sequence[Operators] | None = None,
        n_jobs: NJobs = 1,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
        **kwargs,
    ):
        super().__init__(n_jobs=n_jobs, verbose=verbose, random_state=random_state)
        self.strategy = strategy
        self.n_features = n_features
        self.operators = operators
        self.kwargs = kwargs

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Ajusta o estimador aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de atributos com shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondentes a `X`.

        Retorna
        -------
        self
            Instância do estimador.

        """
        Xt = to_df(X)
        yt = to_tabular(y, index=Xt.index)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        all_operators = {
            "add": "add_numeric",
            "sub": "subtract_numeric",
            "mul": "multiply_numeric",
            "div": "divide_numeric",
            "abs": "absolute",
            "sqrt": "square_root",
            "log": "natural_logarithm",
            "sin": "sine",
            "cos": "cosine",
            "tan": "tangent",
        }

        if not self.operators:  # None ou lista vazia
            operators = list(all_operators)
        else:
            operators = lst(self.operators)

        self._log("Ajustando FeatureGenerator...", 1)

        if self.strategy == "dfs":
            # Executa deep feature synthesis com primitivas de transformação
            es = ft.EntitySet(dataframes={"X": (Xt, "_index", None, None, None, True)})
            self._dfs = ft.dfs(
                target_dataframe_name="X",
                entityset=es,
                trans_primitives=[all_operators[x] for x in operators],
                max_depth=1,
                features_only=True,
                ignore_columns={"X": ["_index"]},
            )

            # Seleciona os novos atributos, pois dfs também retorna os originais
            self._dfs = self._dfs[Xt.shape[1] - 1 :]

            # Obtém uma seleção aleatória de atributos
            if self.n_features and self.n_features < len(self._dfs):
                self._dfs = sample(self._dfs, int(self.n_features))

            # Ordena os atributos alfabeticamente
            self._dfs = sorted(self._dfs, key=lambda x: x._name)

        else:
            kwargs = self.kwargs.copy()  # Copia em caso de fit repetido
            hall_of_fame = kwargs.pop("hall_of_fame", max(400, self.n_features or 400))
            self.gfg_ = SymbolicTransformer(
                population_size=kwargs.pop("population_size", 2000),
                hall_of_fame=hall_of_fame,
                n_components=hall_of_fame,
                init_depth=kwargs.pop("init_depth", (1, 2)),
                const_range=kwargs.pop("const_range", None),
                function_set=operators,
                feature_names=Xt.columns,
                verbose=kwargs.pop("verbose", 0 if self.verbose < 2 else 1),
                n_jobs=kwargs.pop("n_jobs", self.n_jobs),
                random_state=kwargs.pop("random_state", self.random_state),
                **kwargs,
            ).fit(Xt, yt)

        return self

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Gera novos atributos.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de atributos com shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para manter a continuidade da API.

        Retorna
        -------
        dataframe
            Conjunto de atributos transformado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Gerando novos atributos...", 1)

        if self.strategy == "dfs":
            es = ft.EntitySet(dataframes={"X": (Xt, "index", None, None, None, True)})
            dfs = ft.calculate_feature_matrix(self._dfs, entityset=es, n_jobs=self.n_jobs)

            # Adiciona os novos atributos ao conjunto de atributos
            Xt = pd.concat([Xt, dfs], axis=1).set_index("index")

            self._log(f" --> {len(self._dfs)} novos atributos foram adicionados.", 2)

        else:
            # Obtém os nomes e o fitness dos novos atributos
            df = pd.DataFrame(
                data=[
                    ["", str(fx), fx.fitness_]
                    for i, fx in enumerate(self.gfg_)
                    if str(fx) not in Xt.columns
                ],
                columns=["name", "description", "fitness"],
            )

            # Verifica se restou algum novo atributo
            if len(df) == 0:
                self._log(
                    " --> O algoritmo genético não encontrou atributos que melhorassem o resultado.",
                    2,
                )
                return Xt

            # Seleciona os n_features com maior fitness
            df = df.drop_duplicates()
            df = df.nlargest(int(self.n_features or len(df)), columns="fitness")

            # Se não houver atributos suficientes restantes, avisa o usuário
            if len(df) != self.n_features:
                self._log(
                    f" --> Removendo {(self.n_features or len(self.gfg_)) - len(df)} "
                    "atributos devido à repetição.",
                    2,
                )

            for i, array in enumerate(self.gfg_.transform(Xt)[:, df.index].T):
                # Se a coluna for nova, usa um nome padrão
                counter = 0
                while True:
                    name = f"x{Xt.shape[1] + counter}"
                    if name not in Xt:
                        Xt[name] = array  # Adiciona novo atributo a X
                        df.iloc[i, 0] = name
                        break
                    else:
                        counter += 1

            self._log(f" --> {len(df)} novos atributos foram adicionados.", 2)
            self.genetic_features_ = df.reset_index(drop=True)

        return self._convert(Xt)


@beartype
class FeatureGrouper(TransformerMixin):
    """Extrai estatísticas de atributos semelhantes.

    Substitui grupos de atributos com características relacionadas por
    novos atributos que resumem propriedades estatísticas do grupo. Os
    operadores estatísticos são calculados em cada linha do grupo. Os
    nomes dos grupos e seus atributos podem ser acessados pelo método
    `groups`.

    Esta classe pode ser acessada a partir do experionml por meio do
    método [feature_grouping][experionmlclassifier-feature_grouping].
    Leia mais no [guia do usuário][grouping-similar-features].

    Parâmetros
    ----------
    groups: dict
        Nomes dos grupos e [atributos][row-and-column-selection]. Um
        atributo pode pertencer a múltiplos grupos.

    operators: str, sequence or None, default=None
        Operadores estatísticos a aplicar aos grupos. Pode ser qualquer
        operador de `numpy` ou `scipy.stats`, verificados nessa ordem,
        que seja aplicado a um array. Se for None, usa: `min`, `max`,
        `mean`, `median`, `mode` e `std`.

    drop_columns: bool, default=True
        Indica se as colunas em `groups` devem ser removidas após a
        transformação.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    Atributos
    ---------
    feature_names_in_: np.ndarray
        Nomes dos atributos vistos durante o `fit`.

    n_features_in_: int
        Número de atributos vistos durante o `fit`.

    See Also
    --------
    experionml.feature_engineering:FeatureExtractor
    experionml.feature_engineering:FeatureGenerator
    experionml.feature_engineering:FeatureSelector

    Examples
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y)
        experionml.feature_grouping({"group1": "mean.*"}, verbose=2)

        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        from experionml.feature_engineering import FeatureGrouper
        from sklearn.datasets import load_breast_cancer

        X, _ = load_breast_cancer(return_X_y=True, as_frame=True)

        fg = FeatureGrouper({"group1": ["mean texture", "mean radius"]}, verbose=2)
        X = fg.transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        groups: dict[str, list[str]],
        *,
        operators: str | Sequence[str] | None = None,
        drop_columns: Bool = True,
        verbose: Verbose = 0,
    ):
        super().__init__(verbose=verbose)
        self.groups = groups
        self.operators = operators
        self.drop_columns = drop_columns

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Agrupa atributos.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de atributos com shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para manter a continuidade da API.

        Retorna
        -------
        dataframe
            Conjunto de atributos transformado.

        """
        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))

        self._log("Agrupando atributos...", 1)

        if self.operators is None:
            operators = ["min", "max", "mean", "median", "mode", "std"]
        else:
            operators = lst(self.operators)

        to_drop = []
        for name, group in self.groups.items():
            for operator in operators:
                try:
                    result = Xt[group].apply(getattr(np, operator), axis=1)
                except AttributeError:
                    try:
                        result = getattr(stats, operator)(Xt[group], axis=1)[0]
                    except AttributeError:
                        raise ValueError(
                            "Valor inválido para o parâmetro operators. O valor "
                            f"{operator} não é um atributo do numpy nem do scipy.stats."
                        ) from None

                try:
                    Xt[f"{operator}({name})"] = result
                except ValueError:
                    raise ValueError(
                        "Valor inválido para o parâmetro operators. O valor "
                        f"{operator} não retorna um array unidimensional."
                    ) from None

            to_drop.extend(group)
            self._log(f" --> Grupo {name} criado com sucesso.", 2)

        if self.drop_columns:
            Xt = Xt.drop(columns=to_drop)

        return self._convert(Xt)


@beartype
class FeatureSelector(TransformerMixin):
    """Reduz o número de atributos nos dados.

    Aplica seleção de atributos ou redução de dimensionalidade, seja
    para melhorar a acurácia dos estimadores ou aumentar seu desempenho
    em datasets com dimensionalidade muito alta. Além disso, remove
    atributos multicolineares e de baixa variância.

    Esta classe pode ser acessada a partir do experionml por meio do
    método [feature_selection][experionmlclassifier-feature_selection].
    Leia mais no [guia do usuário][selecting-useful-features].

    !!! warning
        - Empates entre atributos com pontuações iguais são desfeitos
          de forma não especificada.
        - Para strategy="rfecv", o parâmetro `n_features` representa o
          número **mínimo** de atributos a selecionar, e não o número
          real de atributos retornado pelo transformador. É possível que
          ele retorne mais.

    !!! info
        - Os engines "sklearnex" e "cuml" são suportados apenas para
          strategy="pca" com datasets densos.
        - Se strategy="pca" e os dados forem densos e não escalonados,
          eles são escalonados para média=0 e desvio padrão=1 antes do
          ajuste do transformador PCA.
        - Se strategy="pca" e os dados fornecidos forem esparsos, o
          estimador usado é [TruncatedSVD][], que funciona de forma mais
          eficiente com matrizes esparsas.

    !!! tip
        * Use os métodos [plot_pca][] e [plot_components][] para
          examinar os resultados após usar strategy="pca".
        * Use o método [plot_rfecv][] para examinar os resultados após
          usar strategy="rfecv".
        * Use o método [plot_feature_importance][] para examinar quanto
          um atributo específico contribui para as predições finais. Se
          o modelo não tiver um atributo `feature_importances_`, use
          [plot_permutation_importance][].

    Parâmetros
    ----------
    strategy: str or None, default=None
        Estratégia de seleção de atributos a usar. Escolha entre:

        - None: Não executa nenhuma estratégia de seleção de atributos.
        - "[univariate][selectkbest]": Teste F estatístico univariado.
        - "[pca][]": Principal Component Analysis.
        - "[sfm][]": Seleciona os melhores atributos de acordo com um modelo.
        - "[sfs][]": Sequential Feature Selection.
        - "[rfe][]": Recursive Feature Elimination.
        - "[rfecv][]": RFE com seleção validada por cross-validation.
        - "[pso][]": Particle Swarm Optimization.
        - "[hho][]": Harris Hawks Optimization.
        - "[gwo][]": Grey Wolf Optimization.
        - "[dfo][]": Dragonfly Optimization.
        - "[go][]": Genetic Optimization.

    solver: str, func, predictor or None, default=None
        Solver ou estimador a usar para a estratégia de seleção. Consulte
        a documentação correspondente para uma descrição mais detalhada
        das opções. Se for None, usa o valor padrão, apenas se
        strategy="pca". Escolha entre:

        - Se strategy="univariate":

            - "[f_classif][]"
            - "[f_regression][]"
            - "[mutual_info_classif][]"
            - "[mutual_info_regression][]"
            - "[chi2][]"
            - Qualquer função com assinatura `func(X, y) -> tuple[scores, p-values]`.

        - Se strategy="pca":

            - Se os dados forem densos:

                - Se engine="sklearn":

                    - "auto" (padrão)
                    - "full"
                    - "covariance_eigh"
                    - "arpack"
                    - "randomized"

                - Se engine="sklearnex":

                    - "full" (padrão)

                - Se engine="cuml":

                    - "full" (padrão)
                    - "jacobi"

            - Se os dados forem esparsos:

                - "randomized" (padrão)
                - "covariance_eigh"
                - "arpack"

        - Para as demais estratégias:<br>
          O estimador base. Para sfm, rfe e rfecv, ele deve ter um
          atributo `feature_importances_` ou `coef_` após o ajuste.
          Você pode usar um dos [modelos predefinidos][]. Adicione
          `_class` ou `_reg` ao final do nome do modelo para especificar
          uma tarefa de classificação ou regressão, por exemplo,
          `solver="LGB_reg"`. Isso não é necessário quando chamado a
          partir do experionml. Não há opção padrão.

    n_features: int, float or None, default=None
        Número de atributos a selecionar.

        - Se None: Seleciona todos os atributos.
        - Se <1: Fração do total de atributos a selecionar.
        - Se >=1: Número de atributos a selecionar.

        Se strategy="sfm" e o parâmetro threshold não for especificado,
        o threshold é automaticamente definido como `-inf` para selecionar
        `n_features` atributos.

        Se strategy="rfecv", `n_features` é o número mínimo de atributos
        a selecionar.

        Esse parâmetro é ignorado se qualquer uma destas estratégias for
        selecionada: pso, hho, gwo, dfo, go.

    min_repeated: int, float or None, default=2
        Remove atributos categóricos se não houver um valor repetido em
        pelo menos `min_repeated` linhas. O padrão é manter todos os
        atributos com variância não máxima, isto é, remover atributos cujo
        número de valores únicos seja igual ao número de linhas, como em
        nomes, IDs etc.

        - Se None: Não faz verificação de repetição mínima.
        - Se >1: Número mínimo de repetições.
        - Se <=1: Fração mínima de repetição.

    max_repeated: int, float or None, default=1.0
        Remove atributos categóricos com o mesmo valor em pelo menos
        `max_repeated` linhas. O padrão é manter todos os atributos com
        variância não nula, isto é, remover atributos com o mesmo valor
        em todas as amostras.

        - Se None: Não faz verificação de repetição máxima.
        - Se >1: Número máximo de ocorrências repetidas.
        - Se <=1: Fração máxima de ocorrências repetidas.

    max_correlation: float or None, default=1.0
        Correlação absoluta mínima de [Pearson][pearson] para identificar
        atributos correlacionados. Para cada grupo, remove todos exceto
        o atributo com maior correlação com `y`, quando fornecido, ou,
        caso contrário, remove todos exceto o primeiro. O valor padrão
        remove colunas iguais. Se for None, ignora esta etapa.

    n_jobs: int, default=1
        Número de núcleos a usar no processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usa todos os núcleos disponíveis.
        - Se <-1: Usa número de núcleos - 1 + `n_jobs`.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string
        que siga o seletor [SYCL_DEVICE_FILTER][], por exemplo,
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Engine de execução a usar para os [estimadores][estimator-acceleration].
        Se for None, usa o valor padrão. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    random_state: int or None, default=None
        Semente usada pelo gerador de números aleatórios. Se for None,
        o gerador usado é o `RandomState` de `np.random`.

    **kwargs
        Qualquer argumento nomeado extra para o estimador de `strategy`.
        Consulte a documentação correspondente para ver as opções disponíveis.

    Atributos
    ---------
    collinear_: pd.DataFrame
        Informações sobre os atributos colineares removidos. As colunas incluem:

        - **drop:** Nome do atributo removido.
        - **corr_feature:** Nomes dos atributos correlacionados.
        - **corr_value:** Coeficientes de correlação correspondentes.

    [strategy]_: sklearn transformer
        Objeto usado para transformar os dados, por exemplo, `fs.pca`
        para a estratégia pca.

    feature_names_in_: np.ndarray
        Nomes dos atributos vistos durante o `fit`.

    n_features_in_: int
        Número de atributos vistos durante o `fit`.

    See Also
    --------
    experionml.feature_engineering:FeatureExtractor
    experionml.feature_engineering:FeatureGenerator
    experionml.feature_engineering:FeatureGrouper

    Examples
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y)
        experionml.feature_selection(strategy="pca", n_features=12, verbose=2)

        # Observe que os nomes das colunas mudaram
        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        from experionml.feature_engineering import FeatureSelector
        from sklearn.datasets import load_breast_cancer

        X, _ = load_breast_cancer(return_X_y=True, as_frame=True)

        fs = FeatureSelector(strategy="pca", n_features=12, verbose=2)
        X = fs.fit_transform(X)

        # Observe que os nomes das colunas mudaram
        print(X)
        ```

    """

    def __init__(
        self,
        strategy: FeatureSelectionStrats | None = None,
        *,
        solver: FeatureSelectionSolvers = None,
        n_features: FloatLargerZero | None = None,
        min_repeated: FloatLargerEqualZero | None = 2,
        max_repeated: FloatLargerEqualZero | None = 1.0,
        max_correlation: FloatZeroToOneInc | None = 1.0,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        random_state: IntLargerEqualZero | None = None,
        **kwargs,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            verbose=verbose,
            random_state=random_state,
        )
        self.strategy = strategy
        self.solver = solver
        self.n_features = n_features
        self.min_repeated = min_repeated
        self.max_repeated = max_repeated
        self.max_correlation = max_correlation
        self.kwargs = kwargs

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Ajusta o seletor de atributos aos dados.

        As estratégias univariate, sfm, quando o modelo não está ajustado,
        sfs, rfe e rfecv precisam de uma coluna alvo. Deixar esse valor
        como None levanta uma exceção.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de atributos com shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondentes a `X`.

        Retorna
        -------
        self
            Instância do estimador.

        """
        from experionml.models import MODELS

        def objective_function(model, X_train, y_train, X_valid, y_valid, scoring):
            """Função objetivo para estratégias avançadas de otimização."""
            if X_train.equals(X_valid):
                cv = cross_val_score(model, X_train, y_train, cv=3, scoring=scoring)
                return np.mean(cv, axis=0)
            else:
                model.fit(X_train, y_train)
                return scoring(model, X_valid, y_valid)

        Xt = to_df(X)
        yt = to_tabular(y, index=Xt.index)

        if yt is None and self.strategy not in ("pca", "sfm", None):
            raise ValueError(
                "Valor inválido para o parâmetro y. O valor não pode "
                f"ser None para strategy='{self.strategy}'."
            )

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self.collinear_ = pd.DataFrame(columns=["drop", "corr_feature", "corr_value"])
        self.scaler_ = None

        kwargs = self.kwargs.copy()
        self._high_variance: dict[Hashable, tuple[Hashable, int]] = {}
        self._low_variance: dict[Hashable, tuple[Hashable, float]] = {}
        self._estimator: Any = None
        self._n_features: int | None = None

        if isinstance(self.strategy, str):
            if self.strategy not in ("univariate", "pca"):
                solver: FeatureSelectionSolvers

                if self.solver is None:
                    raise ValueError(
                        "Invalid value for the solver parameter. The "
                        f"value can't be None for strategy={self.strategy}"
                    )
                elif isinstance(self.solver, str):
                    # Define o goal para inicializar o modelo predefinido
                    if self.solver[-6:] == "_class":
                        goal = Goal.classification
                        solver = self.solver[:-6]
                    elif self.solver[-4:] == "_reg":
                        goal = Goal.regression
                        solver = self.solver[:-4]
                    else:
                        raise ValueError(
                            "Valor inválido para o parâmetro solver. O nome do modelo "
                            "deve ser seguido por '_class' ou '_reg' para especificar a "
                            "tarefa, por exemplo, solver='RF_class'."
                        )

                    # Obtém o estimador a partir dos modelos predefinidos
                    if solver in MODELS:
                        model = MODELS[solver](
                            goal=goal,
                            **{
                                x: getattr(self, x)
                                for x in BaseTransformer.attrs
                                if hasattr(self, x)
                            },
                        )
                        if yt is not None:
                            model.task = goal.infer_task(yt)
                        solver = model._get_est({})
                    else:
                        raise ValueError(
                            "Valor inválido para o parâmetro solver. Modelo desconhecido: "
                            f"{solver}. Os modelos disponíveis são:\n"
                            + "\n".join(
                                [
                                    f" --> {m.__name__} ({m.acronym})"
                                    for m in MODELS
                                    if goal.name in m._estimators
                                ]
                            )
                        )
                elif callable(self.solver):
                    solver = self._inherit(self.solver())  # type: ignore[type-var, assignment]
                else:
                    solver = self.solver

        elif self.kwargs:
            kw = ", ".join([f"{k}={v}" for k, v in self.kwargs.items()])
            raise ValueError(
                f"Os argumentos nomeados ({kw}) foram especificados para "
                "o estimador da estratégia, mas nenhuma estratégia foi selecionada."
            )

        if self.n_features is None:
            self._n_features = Xt.shape[1]
        elif self.n_features < 1:
            self._n_features = int(self.n_features * Xt.shape[1])
        else:
            self._n_features = int(self.n_features)

        min_repeated: Scalar
        if self.min_repeated is None:
            min_repeated = 1
        elif self.min_repeated <= 1:
            min_repeated = self.min_repeated * len(Xt)
        else:
            min_repeated = int(self.min_repeated)

        max_repeated: Scalar
        if self.max_repeated is None:
            max_repeated = len(Xt)
        elif self.max_repeated <= 1:
            max_repeated = self.max_repeated * len(Xt)
        else:
            max_repeated = int(self.max_repeated)

        if min_repeated > max_repeated:
            raise ValueError(
                "O parâmetro min_repeated não pode ser maior que "
                f"max_repeated, mas recebeu {min_repeated} > {max_repeated}. "
            )

        self._log("Ajustando FeatureSelector...", 1)

        # Remove atributos com variância excessivamente alta
        if self.min_repeated is not None:
            for name, column in Xt.select_dtypes(exclude="number").items():
                max_counts = column.value_counts()
                if min_repeated > max_counts.max():
                    self._high_variance[name] = (max_counts.idxmax(), max_counts.max())
                    Xt = Xt.drop(columns=name)
                    break

        # Remove atributos com variância excessivamente baixa
        if self.max_repeated is not None:
            for name, column in Xt.select_dtypes(exclude="number").items():
                for category, count in column.value_counts().items():
                    if count >= max_repeated:
                        self._low_variance[name] = (category, 100.0 * count / len(Xt))
                        Xt = Xt.drop(columns=name)
                        break

        # Remove atributos com correlação excessivamente alta
        self.collinear = pd.DataFrame(columns=["drop", "corr_feature", "corr_value"])
        if self.max_correlation:
            # Obtém a matriz de coeficiente de correlação de Pearson
            if yt is None:
                corr_X = Xt.corr()
            else:
                corr_matrix = merge(Xt, yt).corr()
                corr_X, corr_y = corr_matrix.iloc[:-1, :-1], corr_matrix.iloc[:-1, -1]

            corr = {}
            to_drop = []
            for col in corr_X:
                # Seleciona colunas que estão correlacionadas
                corr[col] = corr_X[col][corr_X[col] >= self.max_correlation]

                # Sempre encontra a si mesmo com correlação 1
                if len(corr[col]) > 1:
                    if yt is None:
                        # Remove todas exceto a primeira
                        to_drop.extend(list(corr[col][1:].index))
                    else:
                        # Mantém o atributo com maior correlação com y
                        keep = corr_y[corr[col].index].idxmax()
                        to_drop.extend(list(corr[col].index.drop(keep)))

            for col in list(dict.fromkeys(to_drop)):
                corr_feature = corr[col].drop(col).index
                corr_value = corr[col].drop(col).round(4).astype(str)
                self.collinear_ = pd.concat(
                    [
                        self.collinear_,
                        pd.DataFrame(
                            {
                                "drop": [col],
                                "corr_feature": [", ".join(corr_feature)],
                                "corr_value": [", ".join(corr_value)],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

            Xt = Xt.drop(columns=self.collinear_["drop"].tolist())

        if self.strategy is None:
            return self  # Encerra a etapa de engenharia de atributos

        elif self.strategy == "univariate":
            solvers_dct = {
                "f_classif": f_classif,
                "f_regression": f_regression,
                "mutual_info_classif": mutual_info_classif,
                "mutual_info_regression": mutual_info_regression,
                "chi2": chi2,
            }

            if not self.solver:
                raise ValueError(
                    "Valor inválido para o parâmetro solver. O "
                    f"valor não pode ser None para strategy={self.strategy}"
                )
            elif isinstance(self.solver, str):
                if self.solver in solvers_dct:
                    solver = solvers_dct[self.solver]
                else:
                    raise ValueError(
                        "Valor inválido para o parâmetro solver: "
                        f"{self.solver}. Escolha entre: {', '.join(solvers_dct)}."
                    )
            else:
                solver = self.solver

            self._estimator = SelectKBest(solver, k=self._n_features).fit(Xt, yt)

        elif self.strategy == "pca":
            if not is_sparse(Xt):
                # PCA exige que os atributos estejam escalonados
                if not check_scaling(Xt):
                    self.scaler_ = Scaler(device=self.device, engine=self.engine)
                    Xt = cast(pd.DataFrame, self.scaler_.fit_transform(Xt))

                estimator = self._get_est_class("PCA", "decomposition")
                solver_param = "svd_solver"
            else:
                estimator = self._get_est_class("TruncatedSVD", "decomposition")
                solver_param = "algorithm"

            if self.solver is None:
                solver = sign(estimator)[solver_param].default
            else:
                solver = self.solver

            # PCA e TruncatedSVD obtêm todos os componentes possíveis para uso
            # nos gráficos, n_components deve ser < n_features e <= n_rows
            self._estimator = estimator(
                n_components=min(len(Xt), Xt.shape[1] - 1),
                **{solver_param: solver},
                random_state=self.random_state,
                **self.kwargs,
            ).fit(Xt)

            self._estimator._comps = min(self._estimator.components_.shape[0], self._n_features)

        elif self.strategy == "sfm":
            # Se qualquer um desses atributos existir, o modelo já foi ajustado
            if any(hasattr(solver, a) for a in ("coef_", "feature_importances_")):
                prefit = kwargs.pop("prefit", True)
            else:
                prefit = False

            # Se não houver threshold especificado, seleciona só com base em _n_features
            if not self.kwargs.get("threshold"):
                kwargs["threshold"] = -np.inf

            self._estimator = SelectFromModel(
                estimator=solver,
                max_features=self._n_features,
                prefit=prefit,
                **kwargs,
            )
            if prefit:
                if list(getattr(solver, "feature_names_in_", [])) != list(Xt.columns):
                    raise ValueError(
                        "Valor inválido para o parâmetro solver. O estimador "
                        f"{solver.__class__.__name__} estimator "
                        "foi ajustado usando colunas diferentes de X!"
                    )
                self._estimator.estimator_ = solver
            else:
                self._estimator.fit(Xt, yt)

        elif self.strategy in ("sfs", "rfe", "rfecv"):
            if self.strategy == "sfs":
                if self.kwargs.get("scoring"):
                    kwargs["scoring"] = get_custom_scorer(self.kwargs["scoring"])

                self._estimator = SequentialFeatureSelector(
                    estimator=solver,
                    n_features_to_select=self._n_features,
                    n_jobs=self.n_jobs,
                    **kwargs,
                )

            elif self.strategy == "rfe":
                self._estimator = RFE(
                    estimator=solver,
                    n_features_to_select=self._n_features,
                    **kwargs,
                )

            elif self.strategy == "rfecv":
                if self.kwargs.get("scoring"):
                    kwargs["scoring"] = get_custom_scorer(self.kwargs["scoring"])

                # Inverte n_features para selecionar todos, opção padrão
                if self._n_features == Xt.shape[1]:
                    self._n_features = 1

                self._estimator = RFECV(
                    estimator=solver,
                    min_features_to_select=self._n_features,
                    n_jobs=self.n_jobs,
                    **kwargs,
                )

            self._estimator.fit(Xt, yt)

        else:
            strategies = {
                "pso": ParticleSwarmOptimization,
                "hho": HarrisHawkOptimization,
                "gwo": GreyWolfOptimization,
                "dfo": DragonFlyOptimization,
                "go": GeneticOptimization,
            }

            # Usa um conjunto de validação fornecido ou cross-validation sobre X
            if "X_valid" in kwargs:
                if "y_valid" in kwargs:
                    X_valid, y_valid = self._check_input(
                        kwargs.pop("X_valid"), kwargs.pop("y_valid")
                    )
                else:
                    raise ValueError(
                        "Valor inválido para o parâmetro y_valid. O valor "
                        "não pode estar ausente quando X_valid é fornecido."
                    )
            else:
                X_valid, y_valid = Xt, yt

            # Obtém o scoring para a objective_function padrão
            if "objective_function" not in kwargs:
                if kwargs.get("scoring"):
                    kwargs["scoring"] = get_custom_scorer(kwargs["scoring"])
                else:
                    goal = Goal(0) if is_classifier(solver) else Goal(1)
                    if yt is not None:
                        task = goal.infer_task(yt)
                    if task is Task.binary_classification:
                        kwargs["scoring"] = get_custom_scorer("f1")
                    elif task.is_multiclass:
                        kwargs["scoring"] = get_custom_scorer("f1_weighted")
                    else:
                        kwargs["scoring"] = get_custom_scorer("r2")

            self._estimator = strategies[self.strategy](
                objective_function=kwargs.pop("objective_function", objective_function),
                minimize=kwargs.pop("minimize", False),
                **kwargs,
            )

            self._estimator.fit(
                model=solver,
                X_train=Xt,
                y_train=yt,
                X_valid=X_valid,
                y_valid=y_valid,
                verbose=self.verbose >= 2,
            )

        # Adiciona o estimador da estratégia como atributo da classe
        setattr(self, f"{self.strategy}_", self._estimator)

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Obtém os nomes dos atributos de saída da transformação.

        Parâmetros
        ----------
        input_features: sequence or None, default=None
            Usado apenas para validar os nomes dos atributos com os
            nomes vistos no `fit`.

        Retorna
        -------
        np.ndarray
            Nomes dos atributos transformados.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        if self._estimator:
            if hasattr(self._estimator, "get_feature_names_out"):
                if self.strategy == "rfecv":
                    return self._estimator.get_feature_names_out()
                else:
                    # _n_features é o número mínimo de atributos com rfecv
                    return self._estimator.get_feature_names_out()[: self._n_features]
            else:
                raise NotImplementedError(
                    "O método get_feature_names_out não está implementado "
                    f"para as estratégias avançadas, mas recebeu {self.strategy}. "
                    "Use uma estratégia do sklearn, por exemplo, SFS, SFM ou RFE."
                )
        else:
            return np.array(
                [
                    c
                    for c in self.feature_names_in_
                    if (
                        c not in self._high_variance
                        and c not in self._low_variance
                        and c not in self.collinear_["drop"].to_numpy()
                    )
                ]
            )

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Transforma os dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de atributos com shape=(n_samples, n_features).

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para manter a continuidade da API.

        Retorna
        -------
        dataframe
            Conjunto de atributos transformado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Executando seleção de atributos...", 1)

        # Remove atributos com variância excessivamente alta
        for fx, h_variance in self._high_variance.items():
            self._log(
                f" --> O atributo {fx} foi removido por alta variância. "
                f"O valor {h_variance[0]} foi o mais repetido, com "
                f"{h_variance[1]} ({h_variance[1] / len(Xt):.1f}%) ocorrências.",
                2,
            )
            Xt = Xt.drop(columns=fx)

        # Remove atributos com variância excessivamente baixa
        for fx, l_variance in self._low_variance.items():
            self._log(
                f" --> O atributo {fx} foi removido por baixa variância. O valor "
                f"{l_variance[0]} se repetiu em {l_variance[1]:.1f}% das linhas.",
                2,
            )
            Xt = Xt.drop(columns=fx)

        # Remove atributos com correlação excessivamente alta
        for col in self.collinear_["drop"]:
            self._log(
                f" --> O atributo {col} foi removido devido à " "colinearidade com outro atributo.",
                2,
            )
            Xt = Xt.drop(columns=col)

        # Executa a seleção com base na estratégia
        if self.strategy is None:
            return self._convert(Xt)

        elif self.strategy == "univariate":
            self._log(
                f" --> O teste univariado selecionou " f"{self._n_features} atributos do dataset.",
                2,
            )
            for n, column in enumerate(Xt):
                if not self.univariate_.get_support()[n]:
                    self._log(
                        f"   --> Removendo o atributo {column} "
                        f"(score: {self.univariate_.scores_[n]:.2f}  "
                        f"p-value: {self.univariate_.pvalues_[n]:.2f}).",
                        2,
                    )
                    Xt = Xt.drop(columns=column)

        elif self.strategy == "pca":
            self._log(" --> Aplicando Principal Component Analysis...", 2)

            if self.scaler_:
                self._log("   --> Escalonando atributos...", 2)
                Xt = cast(pd.DataFrame, self.scaler_.transform(Xt))

            Xt = self._estimator.transform(Xt).iloc[:, : self._estimator._comps]

            var = np.array(self._estimator.explained_variance_ratio_[: self._n_features])
            self._log(f"   --> Mantendo {self._estimator._comps} componentes.", 2)
            self._log(f"   --> Razão de variância explicada: {round(var.sum(), 3)}", 2)

        elif self.strategy in ("sfm", "sfs", "rfe", "rfecv"):
            mask = self._estimator.get_support()
            self._log(
                f" --> {self.strategy} selecionou {sum(mask)} atributos do dataset.",
                2,
            )

            for n, column in enumerate(Xt):
                if not mask[n]:
                    if hasattr(self._estimator, "ranking_"):
                        self._log(
                            f"   --> Removendo o atributo {column} "
                            f"(rank {self._estimator.ranking_[n]}).",
                            2,
                        )
                    else:
                        self._log(f"   --> Removendo o atributo {column}.", 2)
                    Xt = Xt.drop(columns=column)

        else:  # Estratégias avançadas
            self._log(
                f" --> {self.strategy} selecionou {len(self._estimator.best_feature_list)} "
                "atributos do dataset.",
                2,
            )

            for column in Xt:
                if column not in self._estimator.best_feature_list:
                    self._log(f"   --> Removendo o atributo {column}.", 2)
                    Xt = Xt.drop(columns=column)

        return self._convert(Xt)
