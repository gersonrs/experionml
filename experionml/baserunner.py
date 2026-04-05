from __future__ import annotations

import math
import random
import re
from abc import ABCMeta
from collections.abc import Hashable
from copy import deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any

import dill as pickle
import numpy as np
import pandas as pd
from beartype import beartype
from pandas.io.formats.style import Styler
from pandas.tseries.frequencies import to_offset
from pmdarima.arima.utils import ndiffs
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils import Bunch
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.metaestimators import available_if
from sktime.datatypes import check_is_mtype
from sktime.param_est.seasonality import SeasonalityACF
from sktime.transformations.series.difference import Differencer

from experionml.basetracker import BaseTracker
from experionml.basetransformer import BaseTransformer
from experionml.data import Branch
from experionml.models import MODELS, create_stacking_model, create_voting_model
from experionml.pipeline import Pipeline
from experionml.utils.constants import COLOR_SCHEME, DF_ATTRS
from experionml.utils.types import (
    Bool,
    HarmonicsSelector,
    IndexSelector,
    Int,
    IntLargerOne,
    MetricConstructor,
    Model,
    ModelSelector,
    ModelsSelector,
    Pandas,
    RowSelector,
    Scalar,
    Seasonality,
    Segment,
    Sequence,
    SPDict,
    TargetSelector,
    YSelector,
    bool_t,
    int_t,
    pandas_t,
    segment_t,
    sequence_t,
)
from experionml.utils.utils import (
    ClassMap,
    DataContainer,
    Goal,
    SeasonalPeriod,
    Task,
    check_is_fitted,
    composed,
    crash,
    divide,
    flt,
    get_cols,
    get_segment,
    get_versions,
    has_task,
    lst,
    merge,
    method_to_log,
    n_cols,
)


class BaseRunner(BaseTracker, metaclass=ABCMeta):
    """Classe base para runners.

    Contém atributos e métodos compartilhados para o experionml e treinadores.

    """

    def __getstate__(self) -> dict[str, Any]:
        """Requer o armazenamento de um atributo extra com as versões dos pacotes."""
        return self.__dict__ | {"_versions": get_versions(self._models)}

    def __setstate__(self, state: dict[str, Any]):
        """Verifica se as versões carregadas correspondem às instaladas."""
        versions = state.pop("_versions", None)
        self.__dict__.update(state)

        # Verifica se todas as versões dos pacotes correspondem ou emite um aviso
        if versions:
            for key, value in get_versions(state["_models"]).items():
                if versions[key] != value:
                    self._log(
                        f"The loaded instance used the {key} package with version "
                        f"{versions[key]} while the version in this environment is "
                        f"{value}.",
                        1,
                        severity="warning",
                    )

    def __dir__(self) -> list[str]:
        """Adiciona atributos extras de __getattr__ ao dir."""
        # Exclui das condições _available_if
        attrs = [x for x in super().__dir__() if hasattr(self, x)]

        # Adiciona atributos extras da branch
        attrs += self.branch._get_shared_attrs()

        # Adiciona atributos extras do dataset
        attrs += [x for x in DF_ATTRS if hasattr(self.dataset, x)]

        # Adiciona nomes de branches em minúsculas
        attrs += [b.name.lower() for b in self._branches]

        # Adiciona nomes de colunas (excluindo as que têm espaços)
        attrs += [c for c in self.columns if re.fullmatch(r"\w+$", c)]

        # Adiciona nomes de modelos em minúsculas
        if isinstance(self._models, ClassMap):
            attrs += [m.name.lower() for m in self._models]

        return attrs

    def __getattr__(self, item: str) -> Any:
        """Obtém branch, atributo da branch, modelo, coluna ou atributo do dataset."""
        if item in self.__dict__["_branches"]:
            return self._branches[item]  # Obtém branch
        elif item in self.branch._get_shared_attrs():
            if isinstance(attr := getattr(self.branch, item), pandas_t):
                return self._convert(attr)  # Transforma dados pelo mecanismo de dados
            else:
                return attr
        elif item in self.__dict__["_models"]:
            return self._models[item]  # Obtém modelo
        elif item in self.branch.columns:
            return self.branch.dataset[item]  # Obtém coluna do dataset
        elif item in DF_ATTRS and hasattr(self.dataset, item):
            return getattr(self.dataset, item)  # Obtém atributo do dataset
        else:
            raise AttributeError(
                f"O objeto '{self.__class__.__name__}' não possui o atributo '{item}'."
            )

    def __setattr__(self, item: str, value: Any):
        """Define atributo na branch quando é uma propriedade de Branch."""
        if isinstance(getattr(Branch, item, None), property):
            setattr(self.branch, item, value)
        else:
            super().__setattr__(item, value)

    def __delattr__(self, item: str):
        """Exclui modelo."""
        if item in self._models:
            self.delete(item)
        else:
            super().__delattr__(item)

    def __len__(self) -> int:
        """Retorna o comprimento do dataset."""
        return len(self.branch.dataset)

    def __contains__(self, item: str) -> bool:
        """Verifica se o item é uma coluna no dataset."""
        return item in self.dataset

    def __getitem__(self, item: Int | str | list) -> Any:
        """Obtém uma branch, modelo ou coluna do dataset."""
        if self.branch._container is None:
            raise RuntimeError(
                "Esta instância não possui um dataset anexado. "
                "Use o método run antes de chamar __getitem__."
            )
        elif isinstance(item, int_t):
            return self.dataset[self.columns[item]]
        elif isinstance(item, str):
            if item in self._branches:
                return self._branches[item]  # Obtém branch
            elif item in self._models:
                return self._models[item]  # Obtém modelo
            elif item in self.dataset:
                return self.dataset[item]  # Obtém coluna do dataset
            else:
                raise ValueError(
                    f"O objeto {self.__class__.__name__} não possui "
                    f"branch, modelo ou coluna chamado {item}."
                )
        else:
            return self.dataset[item]  # Obtém subconjunto do dataset

    def __sklearn_is_fitted__(self) -> bool:
        """Retorna True quando há modelos treinados."""
        return bool(self._models)

    # Utility properties =========================================== >>

    @cached_property
    def task(self) -> Task:
        """Tipo de [tarefa][task] do dataset."""
        return self._goal.infer_task(self.branch.y)

    @property
    def sp(self) -> Bunch:
        """Sazonalidade da série temporal.

        Leia mais sobre sazonalidade no [guia do usuário][seasonality].

        """
        return self._config.sp

    @sp.setter
    @beartype
    def sp(self, sp: Seasonality | SPDict):
        """Converte sazonalidade para contêner de informações."""
        if sp is None:
            self._config.sp = Bunch()
        elif isinstance(sp, dict):
            self._config.sp = Bunch(**sp)
        else:
            self._config.sp = Bunch(sp=self._get_sp(sp))

    @property
    def og(self) -> Branch:
        """Branch contendo o dataset original.

        Esta branch contém os dados anteriores a quaisquer transformações.
        Redireciona para a branch atual se o seu pipeline estiver vazio,
        para não ter os mesmos dados na memória duas vezes.

        """
        return self._branches.og

    @property
    def branch(self) -> Branch:
        """Branch ativa atual."""
        return self._branches.current

    @property
    def holdout(self) -> pd.DataFrame | None:
        """Conjunto holdout.

        Este conjunto de dados não é transformado pelo pipeline. Leia mais em
        [guia do usuário][data-sets].

        """
        return self._convert(self.branch._holdout)

    @property
    def models(self) -> str | list[str] | None:
        """Nome do(s) modelo(s)."""
        if self._models:
            return flt(self._models.keys())
        else:
            return None

    @property
    def metric(self) -> str | list[str] | None:
        """Nome da(s) métrica(s)."""
        if self._metric:
            return flt(self._metric.keys())
        else:
            return None

    @property
    def winners(self) -> list[Model] | None:
        """Modelos ordenados por desempenho.

        O desempenho é medido pela pontuação mais alta em
        `[main_metric]_bootstrap` ou `[main_metric]_test`, verificados
        nessa ordem. Empates são resolvidos pelo menor `time_fit`.

        """
        if self._models:  # Returns None if not fitted
            return sorted(self._models, key=lambda x: (x._best_score(), x._time_fit), reverse=True)
        else:
            return None

    @property
    def winner(self) -> Model | None:
        """Modelo com melhor desempenho.

        O desempenho é medido pela pontuação mais alta em
        `[main_metric]_bootstrap` ou `[main_metric]_test`, verificados
        nessa ordem. Empates são resolvidos pelo menor `time_fit`.

        """
        if self.winners:  # Returns None if not fitted
            return self.winners[0]
        else:
            return None

    @winner.deleter
    def winner(self):
        """[Exclui][experionmlclassifier-delete] o modelo com melhor desempenho."""
        if self._models:  # Do nothing if not fitted
            self.delete(self.winner.name)

    @property
    def results(self) -> Styler:
        """Visão geral dos resultados de treinamento.

        Todas as durações estão em segundos. Os valores possíveis incluem:

        - **[metric]_ht:** Pontuação obtida pelo ajuste de hiperparâmetros.
        - **time_ht:** Duração do ajuste de hiperparâmetros.
        - **[metric]_train:** Pontuação da métrica no conjunto de treino.
        - **[metric]_test:** Pontuação da métrica no conjunto de teste.
        - **time_fit:** Duração do ajuste do modelo no conjunto de treino.
        - **[metric]_bootstrap:** Pontuação média nas amostras bootstrap.
        - **time_bootstrap:** Duração do bootstrapping.
        - **time:** Duração total da execução.

        !!! tip
            Este atributo retorna um objeto [Styler][] do pandas. Converta
            o resultado de volta para um dataframe regular usando o atributo
            `data`.

        """

        def frac(m: Model) -> float:
            """Retorna a fração do conjunto de treino utilizada.

            Parâmetros
            ----------
            m: Model
                Modelo utilizado.

            Retorna
            -------
            float
                Fração calculada.

            """
            if (n_models := len(m.branch.train) / m._train_idx) == int(n_models):
                return round(1.0 / n_models, 2)
            else:
                return round(m._train_idx / len(m.branch.train), 2)

        df = pd.DataFrame(
            data=[m.results for m in self._models],
            columns=self._models[0].results.index if self._models else [],
            index=lst(self.models),
        ).dropna(axis=1, how="all")

        # Para execuções sh e ts, inclui a fração do conjunto de treino
        if any(m._train_idx != len(m.branch.train) for m in self._models):
            df = df.set_index(
                pd.MultiIndex.from_arrays(
                    arrays=[[frac(m) for m in self._models], self.models],
                    names=["frac", "model"],
                )
            ).sort_index(level=0, ascending=True)

        return df.style.highlight_max(
            props=COLOR_SCHEME, subset=[c for c in df if not c.startswith("time")]
        ).highlight_min(props=COLOR_SCHEME, subset=[c for c in df if c.startswith("time")])

    # Utility methods ============================================== >>

    def _get_sp(self, sp: Seasonality) -> int | list[int] | None:
        """Obtém o período sazonal.

        Parâmetros
        ----------
        sp: int, str, sequence or None
            Período sazonal fornecido. Se None, significa que não há
            sazonalidade.

        Retorna
        -------
        int, list or None
            Período sazonal.

        """

        def get_single_sp(sp: Int | str) -> int:
            """Obtém um período sazonal a partir de um único valor.

            Parâmetros
            ----------
            sp: int, str or None
                Período sazonal como int ou [DateOffset][].

            Retorna
            -------
            int
                Período sazonal.

            """
            if isinstance(sp, str):
                if offset := to_offset(sp):  # Converte para DateOffset do pandas
                    name, period = offset.name.split("-")[0][0], offset.n

                if name not in SeasonalPeriod.__members__:
                    raise ValueError(
                        f"Valor inválido para o período sazonal, recebido {name}. Veja "
                        "https://pandas.pydata.org/pandas-docs/stable/user_guide/"
                        "timeseries.html#period-aliases para uma lista de valores permitidos."
                    )

                # A fórmula é a mesma que SeasonalPeriod[name] para period=1
                return math.lcm(SeasonalPeriod[name].value, period) // period
            else:
                return int(sp)

        if sp == "infer":
            return self.get_seasonal_period()
        elif sp == "index":
            if not hasattr(self.dataset.index, "freqstr"):
                raise ValueError(
                    f"Valor inválido para o período sazonal, recebido {sp}. "
                    f"O índice do dataset não possui o atributo freqstr."
                )
            else:
                return get_single_sp(self.dataset.index.freqstr)
        else:
            return flt([get_single_sp(x) for x in lst(sp)])

    def _get_data(
        self,
        arrays: tuple[Any, ...],
        y: YSelector = -1,
        *,
        index: IndexSelector | None = None,
    ) -> tuple[DataContainer, pd.DataFrame | None]:
        """Obtém conjuntos de dados a partir de uma sequência de indexáveis.

        Também atribui um índice, embaralha (de forma estratificada) e seleciona
        uma subamostra de linhas dependendo dos atributos.

        Parâmetros
        ----------
        arrays: tuple of indexables
            Conjunto(s) de dados fornecidos. Deve seguir o formato de entrada da API.

        y: int, str or sequence, default=-1
            Coluna alvo transformada.

        index: bool, int, str, sequence or None, default=None
            Parâmetro de índice conforme fornecido no construtor. Se None, o
            índice é obtido de `self._config`.

        Retorna
        -------
        DataContainer
            Conjuntos de treino e teste.

        pd.DataFrame or None
            Conjunto holdout. Retorna None se não especificado.

        """

        def _subsample(df: pd.DataFrame) -> pd.DataFrame:
            """Seleciona uma subamostra aleatória de um dataframe.

            Se shuffle=True, a subamostra é embaralhada, caso contrário a ordem
            das linhas é mantida. Para tarefas de previsão, as linhas são removidas
            da cauda do conjunto de dados.

            Parâmetros
            ----------
            df: pd.DataFrame
                Dataset.

            Retorna
            -------
            pd.DataFrame
                Subconjunto de df.

            """
            if self._config.n_rows <= 1:
                n_rows = int(len(df) * self._config.n_rows)
            else:
                n_rows = int(self._config.n_rows)

            if self._goal is Goal.forecast:
                return df.iloc[-n_rows:]  # Para previsão, seleciona da cauda
            else:
                k = random.sample(range(len(df)), k=n_rows)

                if self._config.shuffle:
                    self._config.reindex_metadata(iloc=pd.Index(k))
                    return df.iloc[k]
                else:
                    self._config.reindex_metadata(iloc=pd.Index(sorted(k)))
                    return df.iloc[sorted(k)]

        def _split_sets(data: pd.DataFrame, size: Scalar) -> tuple[pd.DataFrame, pd.DataFrame]:
            """Divide o conjunto de dados em dois conjuntos.

            Parâmetros
            ----------
            data: pd.DataFrame
                Dataset.

            size: int or float
                Tamanho do segundo conjunto.

            Retorna
            -------
            pd.DataFrame
                Primeiro conjunto.

            pd.DataFrame
                Segundo conjunto.

            """
            if (groups := self._config.get_groups(data)) is None:
                return train_test_split(
                    data,
                    test_size=size,
                    random_state=self.random_state,
                    shuffle=self._config.shuffle,
                    stratify=self._config.get_stratify_column(data),
                )
            else:
                if self._goal is Goal.forecast:
                    raise ValueError(
                        "Valor inválido para o parâmetro metadata. A chave "
                        "'groups' não está disponível para tarefas de previsão."
                    )

                # Não levamos a estratificação em consideração pois o comportamento
                # seria indefinido (portanto não implementado no sklearn)
                gss = GroupShuffleSplit(n_splits=1, test_size=size, random_state=self.random_state)
                idx1, idx2 = next(gss.split(data, groups=groups))

                return data.iloc[idx1], data.iloc[idx2]

        def _set_index(
            df: pd.DataFrame,
            y: Pandas | None,
            index: IndexSelector | None = None,
        ) -> pd.DataFrame:
            """Atribui um índice ao dataframe.

            Parâmetros
            ----------
            df: pd.DataFrame
                Dataset.

            y: pd.Series, pd.DataFrame or None
                Coluna(s) alvo. Usada para verificar que o índice fornecido
                não é uma das colunas alvo. Se None, a verificação é ignorada.

            index: bool, int, str or sequence or None, default=None
                Parâmetro de índice conforme fornecido no construtor. Se None, o
                índice é obtido de `self._config`.

            Retorna
            -------
            pd.DataFrame
                Dataset com índices atualizados.

            """
            if index is None:
                index = self._config.index

            # Reorganiza os metadados para o dataset atual
            self._config.reindex_metadata(loc=df.index)

            if index is True:  # True gets caught by isinstance(int)
                pass
            elif index is False:
                df = df.reset_index(drop=True)
            elif isinstance(index, int_t):
                if -df.shape[1] <= index <= df.shape[1]:
                    df = df.set_index(df.columns[int(index)], drop=True)
                else:
                    raise IndexError(
                        f"Valor inválido para o parâmetro index. O valor {index} "
                        f"está fora do intervalo para um dataset com {df.shape[1]} colunas."
                    )
            elif isinstance(index, str):
                if index in df:
                    df = df.set_index(index, drop=True)
                else:
                    raise ValueError(
                        "Valor inválido para o parâmetro index. "
                        f"Coluna {index} não encontrada no dataset."
                    )

            if y is not None and df.index.name in (c.name for c in get_cols(y)):
                raise ValueError(
                    "Valor inválido para o parâmetro index. A coluna de índice "
                    f"não pode ser a mesma que a coluna alvo, recebido {df.index.name}."
                )

            # Atribui os mesmos índices que o dataset atual
            self._config.reindex_metadata(df.index)

            return df

        def _no_data_sets(
            X: pd.DataFrame,
            y: Pandas,
        ) -> tuple[DataContainer, pd.DataFrame | None]:
            """Gera conjuntos de dados a partir de um único dataset.

            Adicionalmente, atribui um índice, embaralha os dados, seleciona
            uma subamostra se `n_rows` for especificado e divide em conjuntos
            de forma estratificada.

            Parâmetros
            ----------
            X: pd.DataFrame
                Conjunto de features com shape=(n_amostras, n_features).

            y: pd.Series or pd.DataFrame
                Coluna(s) alvo correspondente(s) a `X`.

            Retorna
            -------
            DataContainer
                Conjuntos de treino e teste.

            pd.DataFrame or None
                Conjunto holdout. Retorna None se não especificado.

            """
            data = merge(X, y)

            # Embaralha o dataset
            if not 0 < self._config.n_rows <= len(data):
                raise ValueError(
                    "Valor inválido para o parâmetro n_rows. O valor deve "
                    f"estar entre 0 e len(X)={len(data)}, recebido {self._config.n_rows}."
                )
            data = _subsample(data)

            if isinstance(index, sequence_t):
                if len(index) != len(data):
                    raise IndexError(
                        "Valor inválido para o parâmetro index. O comprimento do índice "
                        f"({len(index)}) não corresponde ao do dataset ({len(data)})."
                    )
                data.index = pd.Index(index)

            if len(data) < 5:
                raise ValueError(
                    f"O comprimento do dataset não pode ser <5, recebido {len(data)}. "
                    "Certifique-se de que n_rows=1 para datasets pequenos."
                )

            if not 0 < self._config.test_size < len(data):
                raise ValueError(
                    "Valor inválido para o parâmetro test_size. O valor "
                    f"deve estar entre 0 e len(X), recebido {self._config.test_size}."
                )

            # Define o tamanho do conjunto de teste
            if self._config.test_size < 1:
                if (groups := self._config.get_groups()) is not None:
                    test_size = max(1, int(self._config.test_size * groups.nunique()))
                else:
                    test_size = max(1, int(self._config.test_size * len(data)))
            else:
                test_size = self._config.test_size

            # Define o tamanho do conjunto holdout
            if self._config.holdout_size:
                if self._config.holdout_size < 1:
                    if (groups := self._config.get_groups()) is not None:
                        holdout_size = max(1, int(self._config.holdout_size * groups.nunique()))
                    else:
                        holdout_size = max(1, int(self._config.holdout_size * len(data)))
                else:
                    holdout_size = self._config.holdout_size

                if not 0 <= holdout_size <= len(data) - test_size:
                    raise ValueError(
                        "Valor inválido para o parâmetro holdout_size. "
                        "O valor deve estar entre 0 e len(X) - len(test), "
                        f"recebido {self._config.holdout_size}."
                    )

                data, holdout = _split_sets(data, size=holdout_size)
            else:
                holdout = None

            train, test = _split_sets(data, size=test_size)

            complete_set = _set_index(pd.concat([train, test, holdout]), y, index)

            container = DataContainer(
                data=(data := complete_set.iloc[: len(data)]),
                train_idx=data.index[: -len(test)],
                test_idx=data.index[-len(test) :],
                n_targets=n_cols(y),
            )

            if holdout is not None:
                holdout = complete_set.iloc[len(data) :]

            return container, holdout

        def _has_data_sets(
            X_train: pd.DataFrame,
            y_train: Pandas,
            X_test: pd.DataFrame,
            y_test: Pandas,
            X_holdout: pd.DataFrame | None = None,
            y_holdout: Pandas | None = None,
        ) -> tuple[DataContainer, pd.DataFrame | None]:
            """Gera conjuntos de dados a partir dos conjuntos fornecidos.

            Adicionalmente, atribui um índice, embaralha os dados e
            seleciona uma subamostra se `n_rows` for especificado.

            Parâmetros
            ----------
            X_train: pd.DataFrame
                Conjunto de treino.

            y_train: pd.Series or pd.DataFrame
                Coluna(s) alvo correspondente(s) a `X`_train.

            X_test: pd.DataFrame
                Conjunto de teste.

            y_test: pd.Series or pd.DataFrame
                Coluna(s) alvo correspondente(s) a `X`_test.

            X_holdout: pd.DataFrame or None, default=None
                Conjunto holdout. Pode ser None se não fornecido pelo usuário.

            y_holdout: pd.Series, pd.DataFrame or None, default=None
                Coluna(s) alvo correspondente(s) a `X`_holdout.

            Retorna
            -------
            DataContainer
                Conjuntos de treino e teste.

            pd.DataFrame or None
                Conjunto holdout. Retorna None se não especificado.

            """
            train = merge(X_train, y_train)
            test = merge(X_test, y_test)
            if X_holdout is None:
                holdout = None
            else:
                holdout = merge(X_holdout, y_holdout)

            if not train.columns.equals(test.columns):
                raise ValueError("Os conjuntos de treino e teste não possuem as mesmas colunas.")

            if holdout is not None:
                if not train.columns.equals(holdout.columns):
                    raise ValueError(
                        "O conjunto holdout não possui as "
                        "mesmas colunas que os conjuntos de treino e teste."
                    )

            if self._config.n_rows <= 1:
                train = _subsample(train)
                test = _subsample(test)
                if holdout is not None:
                    holdout = _subsample(holdout)
            else:
                raise ValueError(
                    "Valor inválido para o parâmetro n_rows. O valor deve "
                    "ser <1 quando os conjuntos de treino e teste são fornecidos."
                )

            # If the index is a sequence, assign it before shuffling
            if isinstance(index, sequence_t):
                len_data = len(train) + len(test)
                if holdout is not None:
                    len_data += len(holdout)

                if len(index) != len_data:
                    raise IndexError(
                        "Valor inválido para o parâmetro index. O comprimento do índice "
                        f"({len(index)}) não corresponde ao dos conjuntos de dados ({len_data})."
                    )
                train.index = pd.Index(index[: len(train)])
                test.index = pd.Index(index[len(train) : len(train) + len(test)])
                if holdout is not None:
                    holdout.index = pd.Index(index[-len(holdout) :])

            complete_set = _set_index(pd.concat([train, test, holdout]), y_test, index)

            container = DataContainer(
                data=(data := complete_set.iloc[: len(train) + len(test)]),
                train_idx=data.index[: len(train)],
                test_idx=data.index[-len(test) :],
                n_targets=n_cols(y_train),
            )

            if holdout is not None:
                holdout = complete_set.iloc[len(train) + len(test) :]

            return container, holdout

        # Processa os arrays de entrada ===================================== >>

        if len(arrays) == 0:
            if self.branch._container:
                return self.branch._data, self.branch._holdout
            elif self._goal is Goal.forecast and not isinstance(y, (*int_t, str)):
                # arrays=() e y=y para previsão
                sets = _no_data_sets(*self._check_input(y=y))
            else:
                raise ValueError(
                    "Os arrays de dados estão vazios! Forneça os dados para executar o pipeline "
                    "com sucesso. Consulte a documentação para os formatos permitidos."
                )

        elif len(arrays) == 1:
            # X ou y para previsão
            sets = _no_data_sets(*self._check_input(arrays[0], y=y))

        elif len(arrays) == 2:
            if isinstance(arrays[0], tuple) and len(arrays[0]) == len(arrays[1]) == 2:
                # (X_train, y_train), (X_test, y_test)
                X_train, y_train = self._check_input(arrays[0][0], arrays[0][1])
                X_test, y_test = self._check_input(arrays[1][0], arrays[1][1])
                sets = _has_data_sets(X_train, y_train, X_test, y_test)
            elif isinstance(arrays[1], (*int_t, str)) or n_cols(arrays[1]) == 1:
                if self._goal is not Goal.forecast:
                    # X, y
                    sets = _no_data_sets(*self._check_input(arrays[0], arrays[1]))
                else:
                    # treino, teste para previsão
                    X_train, y_train = self._check_input(y=arrays[0])
                    X_test, y_test = self._check_input(y=arrays[1])
                    sets = _has_data_sets(X_train, y_train, X_test, y_test)
            else:
                # treino, teste
                X_train, y_train = self._check_input(arrays[0], y=y)
                X_test, y_test = self._check_input(arrays[1], y=y)
                sets = _has_data_sets(X_train, y_train, X_test, y_test)

        elif len(arrays) == 3:
            if len(arrays[0]) == len(arrays[1]) == len(arrays[2]) == 2:
                # (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)
                X_train, y_train = self._check_input(arrays[0][0], arrays[0][1])
                X_test, y_test = self._check_input(arrays[1][0], arrays[1][1])
                X_hold, y_hold = self._check_input(arrays[2][0], arrays[2][1])
                sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)
            else:
                # treino, teste, holdout
                X_train, y_train = self._check_input(arrays[0], y=y)
                X_test, y_test = self._check_input(arrays[1], y=y)
                X_hold, y_hold = self._check_input(arrays[2], y=y)
                sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)

        elif len(arrays) == 4:
            # X_train, X_test, y_train, y_test
            X_train, y_train = self._check_input(arrays[0], arrays[2])
            X_test, y_test = self._check_input(arrays[1], arrays[3])
            sets = _has_data_sets(X_train, y_train, X_test, y_test)

        elif len(arrays) == 6:
            # X_train, X_test, X_holdout, y_train, y_test, y_holdout
            X_train, y_train = self._check_input(arrays[0], arrays[3])
            X_test, y_test = self._check_input(arrays[1], arrays[4])
            X_hold, y_hold = self._check_input(arrays[2], arrays[5])
            sets = _has_data_sets(X_train, y_train, X_test, y_test, X_hold, y_hold)

        else:
            raise ValueError(
                "Arrays de dados inválidos. Consulte a documentação para os formatos permitidos."
            )

        if self._goal is Goal.forecast:
            # Para previsão, verifica se o índice está em conformidade com o padrão do sktime
            valid, msg, _ = check_is_mtype(
                obj=pd.DataFrame(pd.concat([sets[0].data, sets[1]])),
                mtype="pd.DataFrame",
                return_metadata=True,
                var_name="the dataset",
            )

            if not valid:
                raise ValueError(msg)
        elif index is not False:
            # Caso contrário, verifica índices duplicados
            if pd.concat([sets[0].data, sets[1]]).index.duplicated().any():
                raise ValueError(
                    "Existem índices duplicados no dataset. "
                    "Use index=False para redefinir o índice para RangeIndex."
                )

        return sets

    def _get_models(
        self,
        models: ModelsSelector = None,
        *,
        ensembles: Bool = True,
        branch: Branch | None = None,
    ) -> list[Model]:
        """Obtém modelos.

        Os modelos podem ser selecionados por nome, índice ou padrão regex. Se uma
        string for fornecida, use `+` para selecionar múltiplos modelos e `!`
        para excluí-los. Modelos não podem ser incluídos e excluídos na
        mesma chamada. A entrada é insensível a maiúsculas/minúsculas.

        Parâmetros
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Modelos a selecionar. Se None, retorna todos os modelos.

        ensembles: bool, default=True
            Se deve incluir modelos ensemble na saída. Se False,
            são silenciosamente excluídos de qualquer retorno.

        branch: Branch or None, default=None
            Força os modelos retornados a terem sido ajustados nesta branch,
            caso contrário lança uma exceção. Se None, este filtro é ignorado.

        Retorna
        -------
        list
            Modelos selecionados.

        """
        inc: list[Model] = []
        exc: list[Model] = []
        if models is None:
            inc = self._models.values()
        elif isinstance(models, segment_t):
            inc = get_segment(self._models, models)
        else:
            for model in lst(models):
                if isinstance(model, int_t):
                    try:
                        inc.append(self._models[model])
                    except KeyError:
                        raise IndexError(
                            f"Valor inválido para o parâmetro models. O valor {model} está "
                            f"fora do intervalo. Existem {len(self._models)} modelos."
                        ) from None
                elif isinstance(model, str):
                    for mdl in model.split("+"):
                        array = inc
                        if mdl.startswith("!") and mdl not in self._models:
                            array = exc
                            mdl = mdl[1:]

                        if mdl.lower() == "winner" and self.winner:
                            array.append(self.winner)
                        elif matches := [
                            m for m in self._models if re.fullmatch(mdl, m.name, re.I)
                        ]:
                            array.extend(matches)
                        else:
                            raise ValueError(
                                "Valor inválido para o parâmetro models. Não foi "
                                f"possível encontrar nenhum modelo que corresponda a {mdl}. Os "
                                f"modelos disponíveis são: {', '.join(self._models.keys())}."
                            )
                elif isinstance(model, Model):
                    inc.append(model)

        if len(inc) + len(exc) == 0:
            raise ValueError(
                "Valor inválido para o parâmetro models, "
                f"recebido {models}. Nenhum modelo foi selecionado."
            )
        elif inc and exc:
            raise ValueError(
                "Valor inválido para o parâmetro models. Você pode incluir "
                "ou excluir modelos, mas não combinações de ambos."
            )
        elif exc:
            # Se modelos foram excluídos com `!`, seleciona todos exceto eles
            inc = [m for m in self._models if m not in exc]

        if not ensembles:
            inc = [m for m in inc if m.acronym not in ("Stack", "Vote")]

        if branch and not all(m.branch is branch for m in inc):
            raise ValueError(
                "Valor inválido para o parâmetro models. Nem "
                f"todos os modelos foram ajustados em {branch}."
            )

        return list(dict.fromkeys(inc))  # Evita duplicatas

    def _delete_models(self, models: str | Model | Sequence[str | Model]):
        """Exclui modelos.

        Remove os modelos da instância. Todos os atributos são excluídos,
        exceto `errors`. Se todos os modelos forem removidos, a métrica é
        redefinida.

        Parâmetros
        ----------
        models: str, Model or sequence
            Modelo(s) a excluir.

        """
        for model in lst(models):
            if model in self._models:
                self._models.remove(model)

        # Se não há modelos, redefine a métrica
        if not self._models:
            self._metric = ClassMap()

    @crash
    def available_models(self, **kwargs) -> pd.DataFrame:
        """Fornece uma visão geral dos modelos predefinidos disponíveis.

        Parâmetros
        ----------
        **kwargs
            Filtra os modelos retornados fornecendo qualquer coluna como
            argumento de palavra-chave, onde o valor é o filtro desejado,
            ex.: `accepts_sparse=True`, para obter todos os modelos que aceitam
            entrada esparsa ou `supports_engines="cuml"` para obter todos os modelos
            que suportam o mecanismo [cuML][].

        Retorna
        -------
        pd.DataFrame
            Tags dos [modelos predefinidos][] disponíveis. As colunas
            dependem da tarefa, mas podem incluir:

            - **acronym:** Acrônimo do modelo (usado para chamar o modelo).
            - **fullname:** Nome da classe do modelo.
            - **estimator:** Nome do estimador subjacente do modelo.
            - **module:** O módulo do estimador.
            - **handles_missing:** Se o modelo pode lidar com valores ausentes
              sem pré-processamento. Se False, considere usar a classe
              [Imputer][] antes de treinar os modelos.
            - **needs_scaling:** Se o modelo requer escalonamento de features.
              Se True, o [escalonamento automático de features][] é aplicado.
            - **accepts_sparse:** Se o modelo aceita [entrada esparsa][sparse-datasets].
            - **uses_exogenous:** Se o modelo usa [variáveis exógenas][].
            - **multiple_seasonality:** Se o modelo pode lidar com mais de
              um [período sazonal][seasonality].
            - **native_multilabel:** Se o modelo tem suporte nativo para
              tarefas [multilabel][].
            - **native_multioutput:** Se o modelo tem suporte nativo para
              [tarefas multioutput][].
            - **validation:** Se o modelo tem [validação durante o treinamento][].
            - **supports_engines:** Mecanismos suportados pelo modelo.

        """
        rows = []
        for model in MODELS:
            if self._goal.name in model._estimators:
                tags = model(goal=self._goal).get_tags()

                for key, value in kwargs.items():
                    k = tags.get(key)
                    if isinstance(value, bool_t) and value is not bool(k):
                        break
                    elif isinstance(value, str) and not re.search(value, k, re.I):
                        break
                else:
                    rows.append(tags)

        return pd.DataFrame(rows)

    @composed(crash, method_to_log)
    def clear(self):
        """Redefine atributos e limpa o cache de todos os modelos.

        Redefine certos atributos do modelo para o estado inicial, excluindo
        arrays de dados potencialmente grandes. Use este método para liberar
        memória antes de [salvar][self-save] a instância. Os atributos
        afetados são:

        - Pontuações de [validação durante o treinamento][]
        - [Valores Shap][shap]
        - [Instância do App][adaboost-create_app]
        - [Instância do Dashboard][adaboost-create_dashboard]
        - [Conjuntos holdout][data-sets] calculados

        """
        for model in self._models:
            model.clear()

    @composed(crash, method_to_log, beartype)
    def delete(self, models: ModelsSelector = None):
        """Exclui modelos.

        Se todos os modelos forem removidos, a métrica é redefinida. Use este
        método para descartar modelos indesejados ou liberar memória antes de
        [salvar][self-save]. Os modelos excluídos não são removidos de
        nenhum [experimento mlflow][tracking] ativo.

        Parâmetros
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Modelos a excluir. Se None, todos os modelos são excluídos.

        """
        self._log(f"Deleting {len(models := self._get_models(models))} models...", 1)
        for m in models:
            self._delete_models(m.name)
            self._log(f" --> Modelo {m.name} excluído com sucesso.", 1)

    @composed(crash, beartype)
    def evaluate(self, metric: MetricConstructor = None, rows: RowSelector = "test") -> Styler:
        """Obtém as pontuações de todos os modelos para as métricas fornecidas.

        !!! tip
            Este método retorna um objeto [Styler][] do pandas. Converta
            o resultado de volta para um dataframe regular usando o atributo
            `data`.

        Parâmetros
        ----------
        metric: str, func, scorer, sequence or None, default=None
            Métrica a calcular. Se None, retorna uma visão geral das
            métricas mais comuns por tarefa.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] para calcular
            a métrica.

        Retorna
        -------
        [Styler][]
            Pontuações dos modelos.

        """
        check_is_fitted(self)

        df = pd.DataFrame([m.evaluate(metric, rows) for m in self._models])

        return df.style.highlight_max(props=COLOR_SCHEME)

    @composed(crash, beartype)
    def export_pipeline(self, model: str | Model | None = None) -> Pipeline:
        """Exporta o pipeline interno.

        Este método retorna uma cópia profunda do pipeline da branch.
        Opcionalmente, você pode adicionar um modelo como estimador final. O
        pipeline retornado já está ajustado no conjunto de treino.

        Parâmetros
        ----------
        model: str, Model or None, default=None
            Modelo para o qual exportar o pipeline. Se o modelo usou
            [escalonamento automático de features][], o [Scaler][] é adicionado ao
            pipeline. Se None, o pipeline da branch atual é exportado
            (sem nenhum modelo).

        Retorna
        -------
        [Pipeline][]
            Branch atual como um objeto Pipeline similar ao sklearn.

        """
        if model:
            return self._get_models(model)[0].export_pipeline()
        else:
            return deepcopy(self.pipeline)

    @available_if(has_task("classification"))
    @composed(crash, beartype)
    def get_class_weight(
        self,
        rows: RowSelector = "train",
    ) -> dict[Hashable, float] | dict[Hashable, dict[Hashable, float]]:
        """Retorna pesos de classe para um conjunto de dados balanceado.

        Estatisticamente, os pesos de classe reequilibram o conjunto de dados para
        que o conjunto de dados amostrado represente a população alvo
        da forma mais próxima possível. Os pesos retornados são inversamente
        proporcionais às frequências de classe nas linhas selecionadas.

        Parâmetros
        ----------
        rows: hashable, segment, sequence or dataframe, default="train"
            [Seleção de linhas][row-and-column-selection] para as quais
            obter os pesos.

        Retorna
        -------
        dict
            Classes com os pesos correspondentes. Um dict de dicts é
            retornado para [tarefas multioutput][].

        """

        def get_weights(col: pd.Series) -> dict[Hashable, float]:
            """Obtém os pesos de classe para uma coluna.

            Parâmetros
            ----------
            col: pd.Series
                Coluna da qual obter os pesos.

            Retorna
            -------
            dict
                Pesos de classe.

            """
            counts = col.value_counts().sort_index()
            return {n: divide(counts.iloc[0], v, 3) for n, v in counts.items()}

        _, y = self.branch._get_rows(rows, return_X_y=True)

        if isinstance(y, pd.Series):
            return get_weights(y)
        else:
            return {col.name: get_weights(col) for col in get_cols(y)}

    @available_if(has_task("classification"))
    @composed(crash, beartype)
    def get_sample_weight(self, rows: RowSelector = "train") -> pd.Series:
        """Retorna pesos de amostra para um conjunto de dados balanceado.

        Os pesos retornados são inversamente proporcionais às frequências de classe
        no conjunto de dados selecionado. Para [tarefas multioutput][],
        os pesos de cada coluna de `y` serão multiplicados.

        Parâmetros
        ----------
        rows: hashable, segment, sequence or dataframe, default="train"
            [Seleção de linhas][row-and-column-selection] para as quais
            obter os pesos.

        Retorna
        -------
        pd.Series
            Sequência de pesos com shape=(n_amostras,).

        """
        _, y = self.branch._get_rows(rows, return_X_y=True)
        weights = compute_sample_weight("balanced", y=y)
        return pd.Series(weights, index=y.index, name="sample_weight").round(3)

    @available_if(has_task("forecast"))
    @composed(crash, beartype)
    def get_seasonal_period(
        self,
        max_sp: IntLargerOne | None = None,
        harmonics: HarmonicsSelector | None = None,
        target: TargetSelector = 0,
    ) -> int | list[int]:
        """Obtém os períodos sazonais da série temporal.

        Usa os dados do conjunto de treino para calcular o período sazonal.
        Os dados são diferenciados internamente antes de a sazonalidade
        ser detectada usando ACF.

        !!! tip
            Leia mais sobre sazonalidade no [guia do usuário][seasonality].

        Parâmetros
        ----------
        max_sp: int or None, default=None
            Período sazonal máximo a considerar. Se None, o período máximo
            é dado por `(len(y_train) - 1) // 2`.

        harmonics: str or None, default=None
            Define a estratégia de como lidar com harmônicos dos
            períodos sazonais detectados. Escolha entre as seguintes opções:

            - None: Os períodos sazonais detectados são mantidos inalterados
              (sem remoção de harmônicos).
            - "drop": Remove todos os harmônicos.
            - "raw_strength": Mantém os harmônicos de maior ordem, preservando
              a ordem de significância.
            - "harmonic_strength": Substitui os períodos sazonais pelo seu
              harmônico mais alto.

            Ex.: se os períodos sazonais detectados em ordem de força são
            `[2, 3, 4, 7, 8]` (note que 4 e 8 são harmônicos de 2), então:

            - Se "drop", resultado=[2, 3, 7]
            - Se "raw_strength", resultado=[3, 7, 8]
            - Se "harmonic_strength", resultado=[8, 3, 7]

        target: int or str, default=0
            Coluna alvo a analisar. Apenas para tarefas [multivariadas][].

        Retorna
        -------
        int or list of int
            Períodos sazonais, ordenados por significância.

        """
        yt = self.dataset[self.branch._get_target(target, only_columns=True)]
        max_sp = max_sp or (len(yt) - 1) // 2

        for _ in np.arange(ndiffs(yt)):
            yt = Differencer().fit_transform(yt)

        acf = SeasonalityACF(nlags=max_sp).fit(pd.DataFrame(yt))
        seasonal_periods = acf.get_fitted_params().get("sp_significant")

        if harmonics and len(seasonal_periods) > 1:
            # Create a dictionary of the seasonal periods and their harmonics
            harmonic_dict: dict[int, list[int]] = {}
            for sp in seasonal_periods:
                for k in harmonic_dict:
                    if sp % k == 0:
                        harmonic_dict[k].append(sp)
                        break
                else:
                    harmonic_dict[sp] = []

            # Para períodos sem harmônicos, simplifica as operações
            # definindo o valor da chave como ela mesma
            harmonic_dict = {k: (v or [k]) for k, v in harmonic_dict.items()}

            if harmonics == "drop":
                seasonal_periods = list(harmonic_dict.keys())
            elif harmonics == "raw_strength":
                seasonal_periods = [
                    sp
                    for sp in seasonal_periods
                    if any(max(v) == sp for v in harmonic_dict.values())
                ]
            elif harmonics == "harmonic_strength":
                seasonal_periods = [max(v) for v in harmonic_dict.values()]

        if not (seasonal_periods := [int(sp) for sp in seasonal_periods if sp <= max_sp]):
            raise ValueError(
                "Nenhum período sazonal foi detectado. Tente "
                f"aumentar o parâmetro max_sp, recebido {max_sp}."
            )

        return flt(seasonal_periods)

    @composed(crash, method_to_log, beartype)
    def merge(self, other: BaseRunner, /, suffix: str = "2"):
        """Mescla outra instância da mesma classe nesta.

        Branches, modelos, métricas e atributos da outra instância
        são mesclados nesta. Se houver branches e/ou modelos com o
        mesmo nome, eles são mesclados adicionando o parâmetro `suffix`
        ao final de seus nomes. Os erros e atributos ausentes são
        estendidos com os da outra instância. Só é possível mesclar duas
        instâncias se forem inicializadas com o mesmo dataset e treinadas
        com a mesma métrica.

        Parâmetros
        ----------
        other: Runner
            Instância com a qual mesclar. Deve ser da mesma classe que self.

        suffix: str, default="2"
            Branches e modelos com nomes conflitantes são mesclados adicionando
            `suffix` ao final de seus nomes.

        """
        if other.__class__.__name__ != self.__class__.__name__:
            raise TypeError(
                "Classe inválida para o parâmetro other. Esperando uma instância de "
                f"{self.__class__.__name__}, recebido {other.__class__.__name__}."
            )

        # Verifica se ambas as instâncias têm o mesmo dataset original
        if not self.og._data.data.equals(other.og._data.data):
            raise ValueError(
                "Valor inválido para o parâmetro other. A instância fornecida "
                "foi inicializada com um dataset diferente desta."
            )

        # Verifica se ambas as instâncias têm a mesma métrica
        if not self._metric:
            self._metric = other._metric
        elif other.metric and self.metric != other.metric:
            raise ValueError(
                "Valor inválido para o parâmetro other. A instância fornecida usa "
                f"uma métrica diferente ({other.metric}) desta ({self.metric})."
            )

        self._log("Mesclando instâncias...", 1)
        for branch in other._branches:
            self._log(f" --> Mesclando branch {branch.name}.", 1)
            if branch.name in self._branches:
                branch._name = f"{branch.name}{suffix}"
            self._branches.branches[branch.name] = branch

        for model in other._models:
            self._log(f" --> Mesclando modelo {model.name}.", 1)
            if model.name in self._models:
                model._name = f"{model.name}{suffix}"
            self._models[model.name] = model

        self._log(" --> Mesclando atributos.", 1)
        if hasattr(self, "missing"):
            self.missing.extend([x for x in other.missing if x not in self.missing])

    @composed(crash, method_to_log, beartype)
    def save(self, filename: str | Path = "auto", *, save_data: Bool = True):
        """Salva a instância em um arquivo pickle.

        Parâmetros
        ----------
        filename: str or Path, default="auto"
            Nome do arquivo ou [pathlib.Path][] do arquivo a salvar. Use
            "auto" para nomeação automática.

        save_data: bool, default=True
            Se deve salvar o dataset junto com a instância. Este
            parâmetro é ignorado se o método não for chamado pelo experionml.
            Se False, adicione os dados ao método [load][experionmlclassifier-load]
            para recarregar a instância.

        """
        if not save_data:
            data = {}
            if (og := self._branches.og).name not in self._branches:
                self._branches._og._container = None
            for branch in self._branches:
                data[branch.name] = {
                    "_data": deepcopy(branch._container),
                    "_holdout": deepcopy(branch._holdout),
                    "holdout": branch.__dict__.pop("holdout", None),  # Clear cached holdout
                }
                branch._container = None
                branch._holdout = None

        if (path := Path(filename)).suffix != ".pkl":
            path = path.with_suffix(".pkl")

        if path.name == "auto.pkl":
            path = path.with_name(f"{self.__class__.__name__}.pkl")

        with open(path, "wb") as f:
            pickle.settings["recurse"] = True
            pickle.dump(self, f)

        # Restaura os dados para os atributos
        if not save_data:
            if og.name not in self._branches:
                self._branches._og._container = og._container
            for branch in self._branches:
                branch._container = data[branch.name]["_data"]
                branch._holdout = data[branch.name]["_holdout"]
                if data[branch.name]["holdout"] is not None:
                    branch.__dict__["holdout"] = data[branch.name]["holdout"]

        self._log(f"{self.__class__.__name__} salvo com sucesso.", 1)

    @composed(crash, method_to_log, beartype)
    def stacking(
        self,
        models: Segment | Sequence[ModelSelector] | None = None,
        name: str = "Stack",
        *,
        train_on_test: Bool = False,
        **kwargs,
    ):
        """Adiciona um modelo [Stacking][] ao pipeline.

        !!! warning
            Combinar modelos treinados em branches diferentes em um
            ensemble não é permitido e lançará uma exceção.

        Parâmetros
        ----------
        models: segment, sequence or None, default=None
            Modelos que alimentam o estimador de stacking. Os modelos devem
            ter sido ajustados na branch atual.

        name: str, default="Stack"
            Nome do modelo. O nome é sempre precedido pelo acrônimo do
            modelo: `Stack`.

        train_on_test: bool, default=False
            Se deve treinar o estimador final do modelo de stacking no
            conjunto de teste em vez do conjunto de treino. Note que treiná-lo
            no conjunto de treino (opção padrão) significa que há um alto
            risco de overfitting. É recomendado usar esta opção se você tiver
            outro conjunto independente para testes ([conjunto holdout][data-sets]).

        **kwargs
            Argumentos de palavra-chave adicionais para um destes estimadores.

            - Para tarefas de classificação: [StackingClassifier][].
            - Para tarefas de regressão: [StackingRegressor][].
            - Para tarefas de previsão: [StackingForecaster][].

            !!! tip
                Os acrônimos dos modelos podem ser usados para o parâmetro
                `final_estimator`, ex.: `experionml.stacking(final_estimator="LR")`.

        """
        check_is_fitted(self)
        models_c = self._get_models(models, ensembles=False, branch=self.branch)

        if len(models_c) < 2:
            raise ValueError(
                "Valor inválido para o parâmetro models. Um modelo Stacking deve "
                f"conter pelo menos dois estimadores subjacentes, recebido apenas {models_c[0]}."
            )

        if not name.lower().startswith("stack"):
            name = f"Stack{name}"

        if name in self._models:
            raise ValueError(
                "Valor inválido para o parâmetro name. Parece que um modelo com "
                f"o nome {name} já existe. Adicione um nome diferente para "
                "treinar múltiplos modelos Stacking dentro da mesma instância."
            )

        kw_model = {
            "goal": self._goal,
            "config": self._config,
            "branches": self._branches,
            "metric": self._metric,
            **{attr: getattr(self, attr) for attr in BaseTransformer.attrs},
        }

        # O nome do parâmetro é diferente no sklearn e no sktime
        regressor = "regressor" if self.task.is_forecast else "final_estimator"
        if isinstance(kwargs.get(regressor), str):
            if kwargs["final_estimator"] not in MODELS:
                valid_models = [m.acronym for m in MODELS if self._goal.name in m._estimators]
                raise ValueError(
                    f"Valor inválido para o parâmetro {regressor}. Modelo desconhecido: "
                    f"{kwargs[regressor]}. Escolha entre: {', '.join(valid_models)}."
                )
            else:
                model = MODELS[kwargs[regressor]](**kw_model)
                if self._goal.name not in model._estimators:
                    raise ValueError(
                        f"Valor inválido para o parâmetro {regressor}. O modelo "
                        f"{model.fullname} não pode executar tarefas {self.task}."
                    )

                kwargs[regressor] = model._get_est({})

        self._models.append(create_stacking_model(models=models_c, name=name, **kw_model))
        self[name]._est_params = kwargs if self.task.is_forecast else {"cv": "prefit"} | kwargs

        if train_on_test:
            self[name].fit(self.X_test, self.y_test)
        else:
            self[name].fit()

    @composed(crash, method_to_log, beartype)
    def voting(
        self,
        models: Segment | Sequence[ModelSelector] | None = None,
        name: str = "Vote",
        **kwargs,
    ):
        """Adiciona um modelo [Voting][] ao pipeline.

        !!! warning
            Combinar modelos treinados em branches diferentes em um
            ensemble não é permitido e lançará uma exceção.

        Parâmetros
        ----------
        models: segment, sequence or None, default=None
            Modelos que alimentam o estimador de stacking. Os modelos devem
            ter sido ajustados na branch atual.

        name: str, default="Vote"
            Nome do modelo. O nome é sempre precedido pelo acrônimo do
            modelo: `Vote`.

        **kwargs
            Argumentos de palavra-chave adicionais para um destes estimadores.

            - Para tarefas de classificação: [VotingClassifier][].
            - Para tarefas de regressão: [VotingRegressor][].
            - Para tarefas de previsão: [EnsembleForecaster][].

        """
        check_is_fitted(self)
        models_c = self._get_models(models, ensembles=False, branch=self.branch)

        if len(models_c) < 2:
            raise ValueError(
                "Valor inválido para o parâmetro models. Um modelo Voting deve "
                f"conter pelo menos dois estimadores subjacentes, recebido apenas {models_c[0]}."
            )

        if not name.lower().startswith("vote"):
            name = f"Vote{name}"

        if name in self._models:
            raise ValueError(
                "Valor inválido para o parâmetro name. Parece que um modelo com "
                f"o nome {name} já existe. Adicione um nome diferente para "
                "treinar múltiplos modelos Voting dentro da mesma instância."
            )

        if kwargs.get("voting") == "soft":
            for m in models_c:
                if not hasattr(m.estimator, "predict_proba"):
                    raise ValueError(
                        "Valor inválido para o parâmetro voting. Se "
                        "'soft', todos os modelos no ensemble devem ter "
                        f"um método predict_proba, recebido {m.name}."
                    )

        self._models.append(
            create_voting_model(
                models=models_c,
                name=name,
                goal=self._goal,
                config=self._config,
                branches=self._branches,
                metric=self._metric,
                **{attr: getattr(self, attr) for attr in BaseTransformer.attrs},
            )
        )
        self[name]._est_params = kwargs
        self[name].fit()
