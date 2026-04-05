from __future__ import annotations

from collections.abc import Hashable
from functools import cached_property
from pathlib import Path
from typing import Literal, overload
from warnings import filterwarnings

import dill as pickle
import pandas as pd
from beartype import beartype
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning
from joblib.memory import Memory
from sklearn.utils.validation import check_memory

from experionml.pipeline import Pipeline
from experionml.utils.types import (
    Bool,
    ColumnSelector,
    Int,
    IntLargerEqualZero,
    Pandas,
    RowSelector,
    Scalar,
    TargetSelector,
    TargetsSelector,
    XConstructor,
    XDatasets,
    YConstructor,
    YDatasets,
    int_t,
    pandas_t,
    segment_t,
)
from experionml.utils.utils import (
    DataContainer,
    check_scaling,
    flt,
    get_col_names,
    get_cols,
    is_sparse,
    lst,
    merge,
    to_tabular,
)


filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)


@beartype
class Branch:
    """Objeto que contém os dados.

    Um branch contém um pipeline específico, o conjunto de dados transformado
    por esse pipeline, os modelos ajustados nesse conjunto de dados, e todos
    os atributos de dados e utilitários que se referem a esse conjunto. Branches
    podem ser criados e acessados pelo atributo `branch` do experionml.

    Todas as propriedades e atributos públicos do branch podem ser acessados
    a partir do pai.

    Leia mais no [guia do usuário][branches].

    !!! warning
        Esta classe não deve ser chamada diretamente. Branches são criados
        internamente pelas classes [ExperionMLClassifier][], [ExperionMLForecaster][] e
        [ExperionMLRegressor][].

    Parâmetros
    ----------
    name: str
        Nome do branch.

    data: DataContainer or None, default=None
        Dados para o branch.

    holdout: pd.DataFrame or None, default=None
        Conjunto de dados holdout.

    memory: str, [Memory][joblibmemory] or None, default=None
        Objeto de memória para cache do pipeline e para armazenar os dados quando
        o branch está inativo.

    Veja também
    --------
    experionml.data:BranchManager

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Inicializa o experionml
    experionml = ExperionMLClassifier(X, y, verbose=2)

    # Treina um modelo
    experionml.run("RF")

    # Muda o branch e aplica escalonamento de features
    experionml.branch = "scaled"

    experionml.scale()
    experionml.run("RF_scaled")

    # Compara os modelos
    experionml.plot_roc()
    ```

    """

    _shared_attrs = (
        "pipeline",
        "mapping",
        "dataset",
        "train",
        "test",
        "X",
        "y",
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "shape",
        "columns",
        "n_columns",
        "features",
        "n_features",
        "target",
    )

    def __init__(
        self,
        name: str,
        data: DataContainer | None = None,
        holdout: pd.DataFrame | None = None,
        *,
        memory: str | Memory | None = None,
    ):
        self.name = name
        self.memory = check_memory(memory)

        self._container = data
        self._holdout = holdout
        self._pipeline = Pipeline([], memory=memory)
        self._mapping: dict[str, dict[Hashable, Scalar]] = {}

        # Caminho para armazenar os dados
        if self.memory.location is None:
            self._location = None
        else:
            self._location = Path(self.memory.location).joinpath("joblib", "experionml")

    def __repr__(self) -> str:
        """Exibe o nome do branch."""
        return f"Branch({self.name})"

    @property
    def _data(self) -> DataContainer:
        """Obtém os dados do branch.

        Carrega da memória se o contêiner de dados estiver vazio. Esta propriedade
        é necessária para acessar os dados de branches inativos.

        """
        if data := self.load(assign=False):
            return data

        # AttributeError é usado para falhar __getattr__ de BaseRunner ao acessar branch vazio
        raise AttributeError(f"O branch {self.name} não possui um conjunto de dados atribuído.")

    @property
    def name(self) -> str:
        """Nome do branch."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Atribui um novo nome ao branch."""
        if not value:
            raise ValueError("Um branch não pode ter um nome vazio.")

        self._name = value

    # Propriedades de dados ============================================== >>

    @overload
    def _check_setter(self, name: XDatasets, value: YConstructor) -> pd.DataFrame: ...

    @overload
    def _check_setter(self, name: YDatasets, value: YConstructor) -> pd.Series: ...

    def _check_setter(self, name: XDatasets | YDatasets, value: YConstructor) -> Pandas:
        """Verifica a propriedade setter do conjunto de dados.

        Converte a propriedade em um objeto 'pandas' e compara com o
        restante do conjunto de dados, para verificar se possui os índices e
        dimensões corretos.

        Parâmetros
        ----------
        name: str
            Nome do conjunto de dados a verificar.

        value: sequence or dataframe-like
            Novos valores para o conjunto de dados.

        Retorna
        -------
        pd.Series or pd.DataFrame
            Conjunto de dados.

        """

        def counter(name: str, dim: str) -> str | None:
            """Retorna a dimensão oposta do conjunto de dados fornecido.

            Parâmetros
            ----------
            name: str
                Nome do conjunto de dados.

            dim: str
                Dimensão a verificar. Pode ser side ou under.

            Retorna
            -------
            str or None
                Nome da dimensão oposta. Retorna None quando não há
                dimensão oposta, ex.: train com dim="side".

            """
            if name == "dataset":
                return name
            if dim == "side":
                if "X" in name:
                    return name.replace("X", "y")
                if "y" in name:
                    return name.replace("y", "X")
            else:
                if "train" in name:
                    return name.replace("train", "test")
                if "test" in name:
                    return name.replace("test", "train")

            return None

        # Define os atributos de dados side e under
        if side_name := counter(name, "side"):
            side = getattr(self, side_name)
        if under_name := counter(name, "under"):
            under = getattr(self, under_name)

        if (columns := get_col_names(value)) is None:
            columns = get_col_names(under) if under_name else None

        obj = to_tabular(
            data=value,
            index=side.index if side_name else None,
            columns=columns,
        )

        if side_name:  # Verifica linhas iguais
            if len(obj) != len(side):
                raise ValueError(
                    f"{name} e {side_name} devem ter o mesmo "
                    f"número de linhas, got {len(obj)} != {len(side)}."
                )
            if not obj.index.equals(side.index):
                raise ValueError(
                    f"{name} e {side_name} devem ter os mesmos "
                    f"índices, got {obj.index} != {side.index}."
                )

        if under_name:  # Verifica colunas iguais
            if isinstance(obj, pd.Series):
                if obj.name != under.name:
                    raise ValueError(
                        f"{name} e {under_name} devem ter o "
                        f"mesmo nome, got {obj.name} != {under.name}."
                    )
            else:
                if obj.shape[1] != under.shape[1]:
                    raise ValueError(
                        f"{name} e {under_name} devem ter o mesmo número "
                        f"de colunas, got {obj.shape[1]} != {under.shape[1]}."
                    )

                if not obj.columns.equals(under.columns):
                    raise ValueError(
                        f"{name} e {under_name} devem ter as mesmas "
                        f"colunas, got {obj.columns} != {under.columns}."
                    )

        # Reinicia o cálculo do holdout
        self.__dict__.pop("holdout", None)

        return obj

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline de transformadores.

        !!! tip
            Use o método [plot_pipeline][] para visualizar o pipeline.

        """
        return self._pipeline

    @property
    def mapping(self) -> dict[str, dict[Hashable, Scalar]]:
        """Valores codificados e seus respectivos valores mapeados.

        O nome da coluna é a chave para seu dicionário de mapeamento. Apenas para
        colunas mapeadas para uma única coluna (ex.: Ordinal, Leave-one-out,
        etc...).

        """
        return self._mapping

    @property
    def dataset(self) -> pd.DataFrame:
        """Conjunto de dados completo."""
        return self._data.data

    @dataset.setter
    def dataset(self, value: XConstructor):
        self._data.data = self._check_setter("dataset", value)

    @property
    def train(self) -> pd.DataFrame:
        """Conjunto de treinamento."""
        return self._data.data.loc[self._data.train_idx]

    @train.setter
    def train(self, value: XConstructor):
        df = self._check_setter("train", value)
        self._data.data = pd.concat([df, self.test])
        self._data.train_idx = df.index

    @property
    def test(self) -> pd.DataFrame:
        """Conjunto de teste."""
        return self._data.data.loc[self._data.test_idx]

    @test.setter
    def test(self, value: XConstructor):
        df = self._check_setter("test", value)
        self._data.data = pd.concat([self.train, df])
        self._data.test_idx = df.index

    @cached_property
    def holdout(self) -> pd.DataFrame | None:
        """Conjunto holdout."""
        if self._holdout is not None:
            return merge(
                *self.pipeline.transform(
                    X=self._holdout[self.features],
                    y=self._holdout[self.target],
                )
            )
        else:
            return None

    @property
    def X(self) -> pd.DataFrame:
        """Conjunto de features."""
        return self._data.data[self.features]

    @X.setter
    def X(self, value: XConstructor):
        df = self._check_setter("X", value)
        self._data.data = merge(df, self.y)

    @property
    def y(self) -> Pandas:
        """Coluna(s) alvo."""
        return self._data.data[self.target]

    @y.setter
    def y(self, value: YConstructor):
        series = self._check_setter("y", value)
        self._data.data = merge(self.X, series)

    @property
    def X_train(self) -> pd.DataFrame:
        """Features do conjunto de treinamento."""
        return self.train[self.features]

    @X_train.setter
    def X_train(self, value: XConstructor):
        df = self._check_setter("X_train", value)
        self._data.data = pd.concat([merge(df, self.y_train), self.test])

    @property
    def y_train(self) -> Pandas:
        """Coluna(s) alvo do conjunto de treinamento."""
        return self.train[self.target]

    @y_train.setter
    def y_train(self, value: YConstructor):
        series = self._check_setter("y_train", value)
        self._data.data = pd.concat([merge(self.X_train, series), self.test])

    @property
    def X_test(self) -> pd.DataFrame:
        """Features do conjunto de teste."""
        return self.test[self.features]

    @X_test.setter
    def X_test(self, value: XConstructor):
        df = self._check_setter("X_test", value)
        self._data.data = pd.concat([self.train, merge(df, self.y_test)])

    @property
    def y_test(self) -> Pandas:
        """Coluna(s) alvo do conjunto de teste."""
        return self.test[self.target]

    @y_test.setter
    def y_test(self, value: YConstructor):
        series = self._check_setter("y_test", value)
        self._data.data = pd.concat([self.train, merge(self.X_test, series)])

    @property
    def shape(self) -> tuple[Int, Int]:
        """Forma do conjunto de dados (n_linhas, n_colunas)."""
        return self.dataset.shape

    @property
    def columns(self) -> pd.Index:
        """Nome de todas as colunas."""
        return self.dataset.columns

    @property
    def n_columns(self) -> int:
        """Número de colunas."""
        return len(self.columns)

    @property
    def features(self) -> pd.Index:
        """Nome das features."""
        return self.columns[: -self._data.n_targets]

    @property
    def n_features(self) -> int:
        """Número de features."""
        return len(self.features)

    @property
    def target(self) -> str | list[str]:
        """Nome da(s) coluna(s) alvo."""
        return flt(list(self.columns[-self._data.n_targets :]))

    @property
    def _all(self) -> pd.DataFrame:
        """Dataset + holdout.

        Note que chamar esta propriedade dispara o cálculo do conjunto
        holdout.

        """
        return pd.concat([self.dataset, self.holdout])

    # Métodos utilitários ============================================== >>

    def _get_shared_attrs(self) -> list[str]:
        """Obtém os atributos que podem ser acessados a partir de um runner.

        Retorna
        -------
        list of str
            Atributos da instância.

        """
        instance_vars = [x for x in vars(self) if not x.startswith("_") and x.endswith("_")]
        return list(self._shared_attrs) + instance_vars

    @overload
    def _get_rows(
        self,
        rows: RowSelector,
        *,
        return_X_y: Literal[False] = ...,
    ) -> pd.DataFrame: ...

    @overload
    def _get_rows(
        self,
        rows: RowSelector,
        *,
        return_X_y: Literal[True],
    ) -> tuple[pd.DataFrame, Pandas]: ...

    def _get_rows(
        self,
        rows: RowSelector,
        *,
        return_X_y: Bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, Pandas]:
        """Obtém um subconjunto das linhas.

        Linhas podem ser selecionadas por nome, índice, conjunto de dados ou padrão regex.
        Se uma string for fornecida, use `+` para selecionar múltiplas linhas e `!`
        para excluí-las. Linhas não podem ser incluídas e excluídas na
        mesma chamada.

        !!! note
            Esta chamada ativa o cálculo do holdout para este branch.

        Parâmetros
        ----------
        rows: hashable, segment, sequence or dataframe
            Linhas a selecionar.

        return_X_y: bool, default=False
            Se deve retornar X e y separadamente ou concatenados.

        Retorna
        -------
        pd.DataFrame
            Subconjunto das linhas.

        pd.Series or pd.Dataframe
            Subconjunto da coluna alvo. Retornado apenas se return_X_y=True.

        """
        _all = self._all  # Evita múltiplas chamadas -> pode ser custoso

        inc: list[Hashable] = []
        exc: list[Hashable] = []
        if isinstance(rows, pandas_t):
            inc.extend(rows.index)
        elif isinstance(rows, pd.Index):
            inc.extend(rows)
        elif isinstance(rows, segment_t):
            inc.extend(_all.index[rows])
        else:
            for row in lst(rows):
                if row in _all.index:
                    inc.append(row)
                elif isinstance(row, int_t):
                    if -len(_all.index) <= row < len(_all.index):
                        inc.append(_all.index[int(row)])
                    else:
                        raise IndexError(
                            f"Valor inválido para o parâmetro rows. O valor {rows} "
                            f"está fora do intervalo para dados com {len(_all)} linhas."
                        )
                elif isinstance(row, str):
                    for r in row.split("+"):
                        array = inc
                        if r.startswith("!") and r not in _all.index:
                            array = exc
                            r = r[1:]

                        # Encontra correspondência no conjunto de dados
                        if r.lower() in ("dataset", "train", "test", "holdout"):
                            try:
                                array.extend(getattr(self, r.lower()).index)
                            except AttributeError:
                                raise ValueError(
                                    "Valor inválido para o parâmetro rows. Nenhum conjunto "
                                    "de dados holdout foi declarado ao inicializar o experionml."
                                ) from None
                        elif (matches := _all.index.str.fullmatch(r)).sum() > 0:
                            array.extend(_all.index[matches])

        if len(inc) + len(exc) == 0:
            raise ValueError(
                "Valor inválido para o parâmetro rows, got "
                f"{rows}. Nenhuma linha foi selecionada."
            )
        elif inc and exc:
            raise ValueError(
                "Valor inválido para o parâmetro rows. Você pode "
                "incluir ou excluir linhas, mas não combinações dos dois."
            )
        elif exc:
            # Se linhas foram excluídas com `!`, seleciona todas exceto essas
            inc = list(_all.index[~_all.index.isin(exc)])

        rows_c = _all.loc[inc]

        if return_X_y:
            return rows_c[self.features], rows_c[self.target]
        else:
            return rows_c

    def _get_columns(
        self,
        columns: ColumnSelector | None = None,
        *,
        include_target: Bool = True,
        only_numerical: Bool = False,
    ) -> list[str]:
        """Obtém um subconjunto das colunas.

        Colunas podem ser selecionadas por nome, índice, dtype ou padrão regex.
        Se uma string for fornecida, use `+` para selecionar múltiplas colunas e
        `!` para excluí-las. Colunas não podem ser incluídas e excluídas na
        mesma chamada.

        Parâmetros
        ----------
        columns: ColumnSelector or None, default=None
            Colunas a selecionar. Se None, retorna todas as colunas do
            conjunto de dados, respeitando os outros parâmetros.

        include_target: bool, default=True
            Se deve incluir a coluna alvo no conjunto de dados de
            seleção.

        only_numerical: bool, default=False
            Se deve selecionar apenas colunas numéricas quando
            `columns=None`.

        Retorna
        -------
        list of str
            Nomes das colunas incluídas.

        """
        # Seleciona o dataframe do qual obter as colunas
        df = self.dataset if include_target else self.X

        inc: list[str] = []
        exc: list[str] = []
        if columns is None:
            if only_numerical:
                return list(df.select_dtypes(include=["number"]).columns)
            else:
                return list(df.columns)
        elif isinstance(columns, pd.DataFrame):
            inc.extend(list(columns.columns))
        elif isinstance(columns, segment_t):
            inc.extend(list(df.columns[columns]))
        else:
            for col in lst(columns):
                if isinstance(col, int_t):
                    if -df.shape[1] <= col < df.shape[1]:
                        inc.append(df.columns[int(col)])
                    else:
                        raise IndexError(
                            f"Seleção de coluna inválida. O valor {col} está fora "
                            f"do intervalo para dados com {df.shape[1]} colunas."
                        )
                elif isinstance(col, str):
                    for c in col.split("+"):
                        array = inc
                        if c.startswith("!") and c not in df.columns:
                            array = exc
                            c = c[1:]

                        # Encontra colunas usando correspondências de regex
                        if (matches := df.columns.str.fullmatch(c)).sum() > 0:
                            array.extend(df.columns[matches])
                        else:
                            # Encontra colunas por tipo
                            try:
                                array.extend(df.select_dtypes(c).columns)  # type: ignore[call-overload]
                            except TypeError:
                                raise ValueError(
                                    "Seleção de coluna inválida. Não foi "
                                    f"possível encontrar nenhuma coluna que corresponda a {c}."
                                ) from None

        if len(inc) + len(exc) == 0:
            raise ValueError(
                f"Seleção de coluna inválida, got {columns}. "
                f"Pelo menos uma coluna deve ser selecionada."
            )
        elif inc and exc:
            raise ValueError(
                "Seleção de coluna inválida. Você pode incluir "
                "ou excluir colunas, mas não combinações dos dois."
            )
        elif exc:
            # Se colunas foram excluídas com `!`, seleciona todas exceto essas
            inc = list(df.columns[~df.columns.isin(exc)])

        return list(dict.fromkeys(inc))  # Evita duplicatas

    @overload
    def _get_target(
        self,
        target: TargetsSelector,
        *,
        only_columns: Literal[False] = ...,
    ) -> tuple[int, int]: ...

    @overload
    def _get_target(
        self,
        target: TargetsSelector,
        *,
        only_columns: Literal[True],
    ) -> str: ...

    def _get_target(
        self,
        target: TargetsSelector,
        *,
        only_columns: Bool = False,
    ) -> str | tuple[int, int]:
        """Obtém uma coluna alvo e/ou classe na coluna alvo.

        Parâmetros
        ----------
        target: int, str or tuple
            Coluna alvo ou classe a recuperar. Para tarefas multioutput,
            use uma tupla no formato (coluna, classe) para selecionar uma classe
            em uma coluna alvo específica.

        only_columns: bool, default=False
            Se deve verificar apenas colunas alvo ou também as classes alvo
            (para tarefas multilabel e multiclass-multioutput).

        Retorna
        -------
        str or tuple
            Nome da coluna alvo selecionada (se only_columns=True)
            ou tupla no formato (coluna, classe).

        """

        def get_column(target: TargetSelector) -> str:
            """Obtém a coluna alvo.

            Parâmetros
            ----------
            target: int or str
                Nome ou posição da coluna alvo.

            Retorna
            -------
            str
                Coluna alvo.

            """
            if isinstance(target, str):
                if target not in self.target:
                    raise ValueError(
                        "Valor inválido para o parâmetro target. O valor "
                        f"{target} não é uma das colunas alvo."
                    )
                else:
                    return target
            else:
                if not 0 <= target < len(self.target):
                    raise ValueError(
                        "Valor inválido para o parâmetro target. Há "
                        f"{len(self.target)} colunas alvo, got {target}."
                    )
                else:
                    return lst(self.target)[target]

        def get_class(
            target: TargetSelector,
            column: IntLargerEqualZero = 0,
        ) -> int:
            """Obtém a classe na coluna alvo.

            Parâmetros
            ----------
            target: int or str
                Nome ou posição da coluna alvo.

            column: int, default=0
                Coluna da qual obter a classe. Para tarefas multioutput.

            Retorna
            -------
            int
                Índice da classe.

            """
            if isinstance(target, str):
                try:
                    return int(self.mapping[lst(self.target)[column]][target])
                except (TypeError, KeyError):
                    raise ValueError(
                        f"Valor inválido para o parâmetro target. O valor {target} "
                        "não foi encontrado no mapeamento da coluna alvo."
                    ) from None
            else:
                n_classes = get_cols(self.y)[column].nunique(dropna=False)
                if not 0 <= target < n_classes:
                    raise ValueError(
                        "Valor inválido para o parâmetro target. "
                        f"Há {n_classes} classes, got {target}."
                    )
                else:
                    return int(target)

        if only_columns and not isinstance(target, tuple):
            return get_column(target)
        elif isinstance(target, tuple):
            if not isinstance(self.y, pd.DataFrame):
                raise ValueError(
                    f"Invalid value for the target parameter, got {target}. "
                    "A tuple is only accepted for multioutput tasks."
                )
            elif len(target) == 1:
                return self.target.index(get_column(target[0])), 0
            elif len(target) == 2:
                column = self.target.index(get_column(target[0]))
                return column, get_class(target[1], column)
            else:
                raise ValueError(
                    "Valor inválido para o parâmetro target. "
                    f"Esperada uma tupla de comprimento 2, got len={len(target)}."
                )
        else:
            return 0, get_class(target)

    def load(self, *, assign: Bool = True) -> DataContainer | None:
        """Carrega os dados do branch da memória.

        Este método é usado para restaurar os dados de branches inativos.

        Parâmetros
        ----------
        assign: bool, default=True
            Se deve atribuir os dados carregados ao `self`.

        Retorna
        -------
        DataContainer or None
            Informações dos próprios dados. Retorna None se nenhum dado estiver definido.

        """
        if self._container is None and self._location:
            try:
                with open(self._location.joinpath(f"{self}.pkl"), "rb") as file:
                    data = pickle.load(file)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"O branch {self.name} não possui dados armazenados."
                ) from None

            if assign:
                self._container = data
            else:
                return data

        return self._container

    def store(self, *, assign: Bool = True):
        """Armazena os dados do branch como um pickle na memória.

        Após o armazenamento, os dados são excluídos e o branch não é mais
        utilizável até que [load][self-load] seja chamado. Este método é usado
        para armazenar os dados de branches inativos.

        !!! note
            Este método é ignorado silenciosamente para branches sem alocação
            de memória.

        Parâmetros
        ----------
        assign: bool, default=True
            Se deve atribuir `None` aos dados em `self`.

        """
        if self._container is not None and self._location:
            try:
                with open(self._location.joinpath(f"{self}.pkl"), "wb") as file:
                    pickle.dump(self._container, file)
            except FileNotFoundError:
                raise FileNotFoundError(f"O diretório {self._location} não existe.") from None

            if assign:
                self._container = None

    def check_scaling(self) -> bool:
        """Verifica se o conjunto de features está escalonado.

        Um conjunto de dados é considerado escalonado quando possui média~0 e desvio~1,
        ou quando há um escalonador no pipeline. Colunas categóricas e
        binárias (apenas zeros e uns) são excluídas do
        cálculo. [Sparse datasets][] sempre retornam False.

        Retorna
        -------
        bool
            Se o conjunto de features está escalonado.

        """
        if any("scaler" in name.lower() for name in self.pipeline.named_steps):
            return True

        if is_sparse(self.X):
            return False

        df = self.X.loc[:, (~self.X.isin([0, 1])).any(axis=0)]  # Remove colunas binárias

        if df.empty:  # Todas as colunas são binárias -> escalonamento não necessário
            return True
        else:
            return check_scaling(df)
