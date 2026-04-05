from __future__ import annotations

import functools
import sys
import warnings
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from functools import cached_property, wraps
from importlib import import_module
from inspect import Parameter, signature
from itertools import cycle
from types import GeneratorType, MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

import mlflow
import nltk
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sps
from beartype.door import is_bearable
from IPython.display import display
from matplotlib.colors import to_rgba
from mlflow.models.signature import infer_signature
from pandas._libs.missing import NAType
from pandas._typing import Axes, DtypeObj
from pandas.api.types import is_numeric_dtype
from shap import Explainer
from sklearn.base import BaseEstimator
from sklearn.base import OneToOneFeatureMixin as FMixin
from sklearn.metrics import (
    confusion_matrix,
    get_scorer,
    get_scorer_names,
    make_scorer,
    matthews_corrcoef,
)
from sklearn.utils import Bunch
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.validation import _check_response_method, _is_fitted

from experionml.utils.constants import CAT_TYPES, __version__
from experionml.utils.types import (
    Bool,
    EngineDataOptions,
    EngineTuple,
    Estimator,
    FeatureNamesOut,
    Float,
    Int,
    IntLargerEqualZero,
    MetricFunction,
    Model,
    Pandas,
    PandasConvertible,
    PosLabel,
    Predictor,
    Scalar,
    Scorer,
    Segment,
    Sequence,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
    YReturn,
    int_t,
    segment_t,
    sequence_t,
)


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial
    from shap import Explanation

    from experionml.basemodel import BaseModel
    from experionml.baserunner import BaseRunner
    from experionml.data import Branch


T = TypeVar("T")
T_Pandas = TypeVar("T_Pandas", pd.Series, pd.DataFrame, pd.Series | pd.DataFrame)
T_Transformer = TypeVar("T_Transformer", bound=Transformer)
T_Estimator = TypeVar("T_Estimator", bound=Estimator)


# Classes ========================================================== >>


class NotFittedError(ValueError, AttributeError):
    """Exceção lançada quando a instância ainda não foi ajustada.

    Esta classe herda de ValueError e AttributeError para facilitar
    o tratamento de exceções e manter compatibilidade retroativa.

    """


class Goal(Enum):
    """Objetivos suportados pelo ExperionML."""

    classification = 0
    regression = 1
    forecast = 2

    def infer_task(self, y: Pandas) -> Task:
        """Infere a tarefa correspondente a uma coluna alvo.

        Parâmetros
        ----------
        y: pd.Series or pd.DataFrame
            Coluna(s) alvo.

        Retorna
        -------
        Task
            Tarefa inferida.

        """
        if self.value == 1:
            if isinstance(y, pd.Series):
                return Task.regression
            else:
                return Task.multioutput_regression
        elif self.value == 2:
            if isinstance(y, pd.Series):
                return Task.univariate_forecast
            else:
                return Task.multivariate_forecast

        if isinstance(y, pd.DataFrame):
            if all(y[col].nunique() == 2 for col in y.columns):
                return Task.multilabel_classification
            else:
                return Task.multiclass_multioutput_classification
        elif isinstance(y.iloc[0], sequence_t):
            return Task.multilabel_classification
        elif y.nunique() == 1:  # noqa: PD101
            raise ValueError(f"Foi encontrado apenas 1 valor alvo: {y.unique()[0]}")
        elif y.nunique() == 2:
            return Task.binary_classification
        else:
            return Task.multiclass_classification


class Task(Enum):
    """Tarefas suportadas pelo ExperionML."""

    binary_classification = 0
    multiclass_classification = 1
    multilabel_classification = 2
    multiclass_multioutput_classification = 3
    regression = 4
    multioutput_regression = 5
    univariate_forecast = 6
    multivariate_forecast = 7

    def __str__(self) -> str:
        """Exibe a tarefa com inicial maiúscula."""
        return self.name.replace("_", " ").capitalize()

    @property
    def is_classification(self) -> bool:
        """Retorna se a tarefa é de classificação."""
        return self.value in (0, 1, 2, 3)

    @property
    def is_regression(self) -> bool:
        """Retorna se a tarefa é de regressão."""
        return self.value in (4, 5)

    @property
    def is_forecast(self) -> bool:
        """Retorna se a tarefa é de previsão."""
        return self.value in (6, 7)

    @property
    def is_binary(self) -> bool:
        """Retorna se a tarefa é binária ou multilabel."""
        return self.value in (0, 2)

    @property
    def is_multiclass(self) -> bool:
        """Retorna se a tarefa é multiclasse ou multiclasse multioutput."""
        return self.value in (1, 3)

    @property
    def is_multioutput(self) -> bool:
        """Retorna se a tarefa possui mais de uma coluna alvo."""
        return self.value in (2, 3, 5, 7)


class SeasonalPeriod(IntEnum):
    """Periodicidade sazonal.

    Abrange os aliases de períodos do pandas.
    Veja: https://pandas.pydminata.org/pandas-docs/stable/user_guide/timeseries.html#period-aliases

    """

    B = 5  # dia útil
    D = 7  # dia corrido
    W = 52  # semana
    M = 12  # mês
    Q = 4  # trimestre
    Y = 1  # ano
    h = 24  # horas
    min = 60  # minutos
    s = 60  # segundos
    ms = 1e3  # milissegundos
    us = 1e6  # microssegundos
    ns = 1e9  # nanossegundos


@dataclass
class DataContainer:
    """Armazena os dados de uma branch."""

    data: pd.DataFrame  # Conjunto de dados completo
    train_idx: pd.Index  # Índices do conjunto de treino
    test_idx: pd.Index  # Índices do conjunto de teste
    n_targets: int  # Número de colunas alvo


@dataclass
class TrackingParams:
    """Parâmetros de rastreamento para um experimento do mlflow."""

    log_ht: bool  # Rastreia cada trial do ajuste de hiperparâmetros
    log_plots: bool  # Salva os artefatos dos gráficos
    log_data: bool  # Salva os conjuntos de treino e teste
    log_pipeline: bool  # Salva o pipeline do modelo


@dataclass
class Aesthetics:
    """Mantém o controle da estética dos gráficos."""

    palette: str | Sequence[str]  # Sequência de cores
    title_fontsize: Scalar  # Tamanho da fonte dos títulos
    label_fontsize: Scalar  # Tamanho da fonte dos rótulos, legenda e hoverinfo
    tick_fontsize: Scalar  # Tamanho da fonte das marcações
    line_width: Scalar  # Largura das linhas
    marker_size: Scalar  # Tamanho dos marcadores


@dataclass
class DataConfig:
    """Armazena a configuração dos dados.

    Classe utilitária para armazenar a configuração dos dados em um
    único atributo e repassá-la aos modelos. Os valores padrão são os
    adotados pelos treinadores.

    """

    index: bool = False
    metadata: Bunch = field(default_factory=Bunch)
    ignore: tuple[str, ...] = ()
    sp: Bunch = field(default_factory=Bunch)
    shuffle: Bool = False
    pos_label: PosLabel = 1
    stratify: Int | str | None = None
    n_rows: Scalar = 1
    test_size: Scalar = 0.2
    holdout_size: Scalar | None = None

    def reindex_metadata(
        self,
        new_index: pd.Index | None = None,
        loc: pd.Index | None = None,
        iloc: pd.Index | None = None,
    ):
        """Reindexa as chaves 'groups' e 'sample_weight' em metadata.

        Parâmetros
        ----------
        new_index: pd.Index or None, default=None
            Novo índice para substituir o atual. As linhas permanecem
            na mesma ordem.

        loc: pd.Index or None, default=None
            Nova ordem para reindexar o metadata com base nos nomes atuais.

        iloc: pd.Index or None, default=None
            Nova ordem para reindexar o metadata com base nas posições.

        """
        for key, value in self.metadata.items():
            if new_index is not None:
                self.metadata[key].index = new_index
            elif loc is not None:
                self.metadata[key] = value.loc[loc]
            elif iloc is not None:
                self.metadata[key] = value.iloc[iloc]

    def get_groups(self, data: Pandas | None = None) -> pd.Series | None:
        """Obtém os grupos do metadata.

        Retorna apenas os índices do metadata que correspondem aos
        fornecidos.

        Parâmetros
        ----------
        data: pd.Series, pd.DataFrame or None, default=None
            Dados correspondentes aos quais o metadata se refere. Se
            None, todos os grupos são retornados.

        Retorna
        -------
        pd.Series or None
            Parâmetros do metadata.

        """
        if "groups" in self.metadata:
            if data is None:
                return self.metadata.groups
            else:
                return self.metadata.groups.loc[data.index]
        else:
            return None

    def get_sample_weight(self, data: Pandas | None = None) -> pd.Series | None:
        """Obtém os pesos das amostras no metadata.

        Retorna apenas os índices do metadata que correspondem aos
        fornecidos.

        Parâmetros
        ----------
        data: pd.Series, pd.DataFrame or None, default=None
            Dados correspondentes aos quais o metadata se refere. Se
            None, todos os pesos de amostra são retornados.

        Retorna
        -------
        pd.Series or None
            Parâmetros do metadata.

        """
        if "sample_weight" in self.metadata:
            if data is None:
                return self.metadata.sample_weight
            else:
                return self.metadata.sample_weight.loc[data.index]
        else:
            return None

    def get_metadata(self, data: Pandas | None = None) -> dict[str, Any]:
        """Obtém todo o metadata.

        Retorna apenas os índices do metadata que correspondem aos
        fornecidos.

        Parâmetros
        ----------
        data: pd.Series, pd.DataFrame or None, default=None
            Dados correspondentes aos quais o metadata se refere. Se
            None, todo o metadata é retornado.

        Retorna
        -------
        dict
            Metadata para os índices solicitados.

        """
        return {k: getattr(self, f"get_{k}")(data) for k, v in self.metadata.items()}

    def get_stratify_column(self, df: pd.DataFrame) -> pd.Series | None:
        """Obtém a coluna usada para estratificação.

        Parâmetros
        ----------
        df: pd.DataFrame
            Conjunto de dados do qual a coluna será obtida.

        Retorna
        -------
        pd.Series or None
            Coluna de estratificação. Retorna None se não houver
            estratificação.

        """
        # Não há estratificação quando os dados não podem mudar de ordem
        if self.stratify is None or self.shuffle is False:
            return None

        if isinstance(self.stratify, int_t):
            if -df.shape[1] <= self.stratify <= df.shape[1]:
                return df[df.columns[int(self.stratify)]]
            else:
                raise ValueError(
                    f"Valor inválido para o parâmetro stratify. O valor {self.stratify} "
                    f"está fora do intervalo para um conjunto de dados com {df.shape[1]} colunas."
                )
        elif isinstance(self.stratify, str):
            if self.stratify in df:
                return df[self.stratify]
            else:
                raise ValueError(
                    "Valor inválido para o parâmetro stratify. "
                    f"A coluna {self.stratify} não foi encontrada no conjunto de dados."
                )


class CatBMetric:
    """Métrica de avaliação personalizada para o modelo CatBoost.

    Parâmetros
    ----------
    scorer: Scorer
        Scorer a ser avaliado. É sempre a métrica principal do runner.

    task: Task
        Tarefa do modelo.

    """

    def __init__(self, scorer: Scorer, task: Task):
        self.scorer = scorer
        self.task = task

    def get_final_error(self, error: Float, weight: Float) -> Float:
        """Retorna o valor final da métrica com base no erro e no peso.

        Não pode ser `staticmethod` devido à implementação do CatBoost.

        Parâmetros
        ----------
        error: float
            Soma dos erros em todas as instâncias.

        weight: float
            Soma dos pesos de todas as instâncias.

        Retorna
        -------
        float
            Valor da métrica.

        """
        return error / (weight + 1e-38)

    @staticmethod
    def is_max_optimal() -> bool:
        """Retorna se valores maiores da métrica são melhores."""
        return True

    def evaluate(
        self,
        approxes: list[Float],
        targets: list[Float],
        weight: list[Float],
    ) -> tuple[Float, Float]:
        """Avalia o valor da métrica.

        Parâmetros
        ----------
        approxes: list
            Vetores com os rótulos aproximados.

        targets: list
            Vetores com os rótulos verdadeiros.

        weight: list
            Vetores de pesos.

        Retorna
        -------
        float
            Erros ponderados.

        float
            Peso total.

        """
        if self.task.is_binary:
            # Converte as predições do CatBoost em probabilidades
            e = np.exp(approxes[0])
            y_pred = e / (1 + e)
            if self.scorer._response_method == "predict":
                y_pred = (y_pred > 0.5).astype(int)

        elif self.task.is_multiclass:
            y_pred = np.array(approxes).T
            if self.scorer._response_method == "predict":
                y_pred = np.argmax(y_pred, axis=1)

        else:
            y_pred = approxes[0]

        kwargs = {}
        if "sample_weight" in sign(self.scorer._score_func):
            kwargs["sample_weight"] = weight

        score = self.scorer._score_func(targets, y_pred, **self.scorer._kwargs)

        return self.scorer._sign * score, 1.0


class LGBMetric:
    """Métrica de avaliação personalizada para o modelo LightGBM.

    Parâmetros
    ----------
    scorer: Scorer
        Scorer a ser avaliado. É sempre a métrica principal do runner.

    task: Task
        Tarefa do modelo.

    """

    def __init__(self, scorer: Scorer, task: Task):
        self.scorer = scorer
        self.task = task

    def __call__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weight: np.ndarray,
    ) -> tuple[str, Float, bool]:
        """Avalia o valor da métrica.

        Parâmetros
        ----------
        y_true: np.array
            Vetores com os rótulos aproximados.

        y_pred: np.array
            Vetores com os rótulos verdadeiros.

        weight: np.array
            Vetores de pesos.

        Retorna
        -------
        str
            Nome da métrica.

        float
            Pontuação da métrica.

        bool
            Indica se valores maiores são melhores.

        """
        if self.scorer._response_method == "predict":
            if self.task.is_binary:
                y_pred = (y_pred > 0.5).astype(int)
            elif self.task.is_multiclass:
                y_pred = y_pred.reshape(len(np.unique(y_true)), len(y_true)).T
                y_pred = np.argmax(y_pred, axis=1)

        kwargs = {}
        if "sample_weight" in sign(self.scorer._score_func):
            kwargs["sample_weight"] = weight

        score = self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs, **kwargs)

        return self.scorer.name, self.scorer._sign * score, True


class XGBMetric:
    """Métrica de avaliação personalizada para o modelo XGBoost.

    Parâmetros
    ----------
    scorer: Scorer
        Scorer a ser avaliado. É sempre a métrica principal do runner.

    task: str
        Tarefa do modelo.

    """

    def __init__(self, scorer: Scorer, task: Task):
        self.scorer = scorer
        self.task = task

    @property
    def __name__(self) -> str:
        """Retorna o nome do scorer."""
        return self.scorer.name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> Float:
        """Calcula a pontuação.

        Parâmetros
        ----------
        y_true: np.array
            Vetores com os rótulos aproximados.

        y_pred: np.array
            Vetores com os rótulos verdadeiros.

        Retorna
        -------
        float
            Pontuação da métrica.

        """
        if self.scorer._response_method == "predict":
            if self.task.is_binary:
                y_pred = (y_pred > 0.5).astype(int)
            elif self.task.is_multiclass:
                y_pred = np.argmax(y_pred, axis=1)

        score = self.scorer._score_func(y_true, y_pred, **self.scorer._kwargs)
        return -self.scorer._sign * score  # Negativo porque o XGBoost minimiza


class Table:
    """Classe para imprimir tabelas formatadas linha a linha.

    Parâmetros
    ----------
    headers: sequence
        Nome de cada coluna da tabela.

    spaces: sequence
        Largura de cada coluna. Deve ter o mesmo tamanho de `headers`.

    """

    def __init__(self, headers: Sequence[str], spaces: Sequence[Int]):
        self.headers = headers
        self.spaces = spaces
        self.positions = ["left"] + (len(headers) - 1) * ["right"]

    @staticmethod
    def to_cell(text: Scalar | str, position: str, space: Int) -> str:
        """Obtém a formatação de string para uma célula.

        Parâmetros
        ----------
        text: int, float or str
            Valor a adicionar à célula.

        position: str
            Posição do texto na célula. Escolha entre: right, left.

        space: int
            Número máximo de caracteres na célula.

        Retorna
        -------
        str
            Valor a ser exibido na célula.

        """
        text = str(text)
        if len(text) > space:
            text = text[: space - 2] + ".."

        if position == "right":
            return text.rjust(space)
        else:
            return text.ljust(space)

    def print_header(self) -> str:
        """Imprime o cabeçalho.

        Retorna
        -------
        str
            Nova linha com os nomes das colunas.

        """
        return self.pprint({k: k for k in self.headers})

    def print_line(self) -> str:
        """Imprime uma linha com traços.

        Use este método após imprimir o cabeçalho para obter uma
        estrutura de tabela mais legível.

        Retorna
        -------
        str
            Nova linha com traços.

        """
        return self.pprint({k: "-" * s for k, s in zip(self.headers, self.spaces, strict=True)})

    def pprint(self, sequence: dict[str, Any] | pd.Series) -> str:
        """Converte uma sequência em uma linha de tabela formatada.

        Parâmetros
        ----------
        sequence: dict
            Nomes das colunas e valor a adicionar à linha.

        Retorna
        -------
        str
            Nova linha com valores.

        """
        out = []
        for h, p, s in zip(self.headers, self.positions, self.spaces, strict=True):
            out.append(self.to_cell(rnd(sequence.get(h, "---")), p, s))

        return "| " + " | ".join(out) + " |"


class TrialsCallback:
    """Exibe a visão geral dos trials durante a execução do estudo.

    Callback para o estudo de ajuste de hiperparâmetros em que uma
    tabela com as informações do trial é exibida. Além disso, o
    atributo `trials` do modelo é preenchido.

    Parâmetros
    ----------
    model: Model
        Modelo a partir do qual o estudo é criado.

    n_jobs: int
        Número de jobs paralelos. Se >1, nada é exibido.

    """

    def __init__(self, model: BaseModel, n_jobs: Int):
        self.T = model
        self.n_jobs = n_jobs

        if self.n_jobs == 1:
            self._table = self.create_table()
            self.T._log(self._table.print_header(), 2)
            self.T._log(self._table.print_line(), 2)

    def __call__(self, study: Study, trial: FrozenTrial):
        """Imprime informações do trial e as armazena no experimento do mlflow."""
        try:  # Falha quando não há trials bem-sucedidos
            trials = self.T.trials.reset_index(names="trial")
            trial_info = cast(
                pd.Series, trials.loc[trial.number]
            )  # loc retorna dataframe ou series
        except KeyError:
            return

        # Salva os trials no experimento do mlflow como execuções aninhadas
        if self.T.experiment and self.T.log_ht:
            with mlflow.start_run(run_id=self.T.run.info.run_id):
                run_name = f"{self.T.name} - {trial.number}"
                with mlflow.start_run(run_name=run_name, nested=True):
                    mlflow.set_tags(
                        {
                            "name": self.T.name,
                            "model": self.T.fullname,
                            "branch": self.T.branch.name,
                            "trial_state": trial_info.state,
                            **self.T._ht["tags"],
                        }
                    )

                    mlflow.log_metric("time_trial", trial_info["time_trial"])
                    for met in self.T._metric.keys():
                        mlflow.log_metric(f"{met}_validation", trial_info[met])

                    # O mlflow só aceita parâmetros com tamanho <=250 caracteres
                    mlflow.log_params(
                        {
                            k: v
                            for k, v in trial_info["estimator"].get_params().items()
                            if len(str(v)) <= 250
                        }
                    )

                    mlflow.sklearn.log_model(
                        sk_model=(est := trial_info["estimator"]),
                        artifact_path=est.__class__.__name__,
                        signature=infer_signature(
                            model_input=self.T.branch.X,
                            model_output=est.predict(self.T.branch.X.iloc[[0]]),
                        ),
                        input_example=self.T.branch.X.iloc[[0], :],
                    )

        if self.n_jobs == 1:
            # Imprime a visão geral dos trials
            trial_info["time_trial"] = time_to_str(trial_info["time_trial"])
            trial_info["time_ht"] = time_to_str(trial_info["time_ht"])
            self.T._log(self._table.pprint(trial_info), 2)

    def create_table(self) -> Table:
        """Cria a tabela de trials.

        Retorna
        -------
        Table
            Objeto para exibir a visão geral dos trials.

        """
        headers = ["trial", *self.T._ht["distributions"]]
        for m in self.T._metric:
            headers.extend([m.name, "best_" + m.name])
        headers.extend(["time_trial", "time_ht", "state"])

        # Define a largura de cada coluna da tabela
        spaces = [len(headers[0])]
        for name, dist in self.T._ht["distributions"].items():
            # Se a distribuição for categórica, usa a média das larguras
            # Caso contrário, usa o máximo entre sete e a largura do nome
            if hasattr(dist, "choices"):
                options = np.mean([len(str(x)) for x in dist.choices], axis=0, dtype=int)
            else:
                options = 0

            spaces.append(max(7, len(name), options))

        spaces.extend(
            [max(7, len(column)) for column in headers[1 + len(self.T._ht["distributions"]) : -1]]
        )

        return Table(headers, [*spaces, 8])


class PlotCallback:
    """Plota o progresso do ajuste de hiperparâmetros durante a execução.

    Cria uma figura com dois gráficos: o primeiro mostra a pontuação de
    cada trial e o segundo mostra a distância entre as pontuações dos
    últimos passos consecutivos.

    Parâmetros
    ----------
    name: str
        Nome do modelo.

    metric: list of str
        Nome(s) das métricas a plotar.

    aesthetics: Aesthetics
        Propriedades que definem a estética do gráfico.

    """

    max_len = 15  # Número máximo de trials exibidos de uma vez no gráfico

    def __init__(self, name: str, metric: list[str], aesthetics: Aesthetics):
        self.y1: dict[int, deque] = {i: deque(maxlen=self.max_len) for i in range(len(metric))}
        self.y2: dict[int, deque] = {i: deque(maxlen=self.max_len) for i in range(len(metric))}

        traces = []
        colors = cycle(aesthetics.palette)
        for met in metric:
            color = next(colors)
            traces.extend(
                [
                    go.Scatter(
                        mode="lines+markers",
                        line={"width": aesthetics.line_width, "color": color},
                        marker={
                            "symbol": "circle",
                            "size": aesthetics.marker_size,
                            "line": {"width": 1, "color": "white"},
                            "opacity": 1,
                        },
                        name=met,
                        legendgroup=met,
                        xaxis="x2",
                        yaxis="y1",
                    ),
                    go.Scatter(
                        mode="lines+markers",
                        line={"width": aesthetics.line_width, "color": color},
                        marker={
                            "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                            "symbol": "circle",
                            "size": aesthetics.marker_size,
                            "opacity": 1,
                        },
                        name=met,
                        legendgroup=met,
                        showlegend=False,
                        xaxis="x2",
                        yaxis="y2",
                    ),
                ]
            )

        self.figure = go.FigureWidget(
            data=traces,
            layout={
                "xaxis1": {"domain": (0, 1), "anchor": "y1", "showticklabels": False},
                "yaxis1": {
                    "domain": (0.31, 1.0),
                    "title": {"text": "Score", "font_size": aesthetics.label_fontsize},
                    "anchor": "x1",
                },
                "xaxis2": {
                    "domain": (0, 1),
                    "title": {"text": "Trial", "font_size": aesthetics.label_fontsize},
                    "anchor": "y2",
                },
                "yaxis2": {
                    "domain": (0, 0.29),
                    "title": {"text": "d", "font_size": aesthetics.label_fontsize},
                    "anchor": "x2",
                },
                "title": {
                    "text": f"Hyperparameter tuning for {name}",
                    "x": 0.5,
                    "y": 1,
                    "pad": {"t": 15, "b": 15},
                    "xanchor": "center",
                    "yanchor": "top",
                    "xref": "paper",
                    "font_size": aesthetics.title_fontsize,
                },
                "legend": {
                    "x": 0.99,
                    "y": 0.99,
                    "xanchor": "right",
                    "yanchor": "top",
                    "font_size": aesthetics.label_fontsize,
                    "bgcolor": "rgba(255, 255, 255, 0.5)",
                },
                "hovermode": "x unified",
                "hoverlabel": {"font_size": aesthetics.label_fontsize},
                "font_size": aesthetics.tick_fontsize,
                "margin": {
                    "l": 0,
                    "b": 0,
                    "r": 0,
                    "t": 25 + aesthetics.title_fontsize,
                    "pad": 0,
                },
                "width": 900,
                "height": 800,
            },
        )

        display(self.figure)

    def __call__(self, study: Study, trial: FrozenTrial):
        """Calcula novos valores para as linhas e os plota.

        Parâmetros
        ----------
        study: Study
            Estudo atual.

        trial: FrozenTrial
            Trial concluído.

        """
        x = range(x_min := max(0, trial.number - self.max_len), x_min + self.max_len)

        for i, score in enumerate(lst(trial.value or trial.values)):
            self.y1[i].append(score)
            if self.y2[i]:
                self.y2[i].append(abs(self.y1[i][-1] - self.y1[i][-2]))
            else:
                self.y2[i].append(None)

            # Atualiza os dados dos traces
            self.figure.data[i * 2].x = list(x[: len(self.y1[i])])
            self.figure.data[i * 2].y = list(self.y1[i])
            self.figure.data[i * 2 + 1].x = list(x[: len(self.y1[i])])
            self.figure.data[i * 2 + 1].y = list(self.y2[i])


class ShapExplanation:
    """Wrapper de SHAP Explanation para evitar recalcular valores shap.

    Calcular valores shap ou de interação é computacionalmente caro.
    Esta classe "lembra" todos os valores calculados e os reutiliza
    quando necessário.

    Parâmetros
    ----------
    estimator: Predictor
        Estimador a partir do qual os valores shap serão obtidos.

    task: Task
        Tarefa do modelo.

    branch: Branch
        Dados a partir dos quais os valores shap serão obtidos.

    random_state: int or None, default=None
        Semente aleatória para reprodutibilidade.

    """

    def __init__(
        self,
        estimator: Predictor,
        task: Task,
        branch: Branch,
        random_state: IntLargerEqualZero | None = None,
    ):
        self.estimator = estimator
        self.task = task
        self.branch = branch
        self.random_state = random_state

        self._explanation: Explainer
        self._shap_values = pd.Series(dtype="object")
        self._interaction_values = pd.Series(dtype="object")

    @cached_property
    def explainer(self) -> Explainer:
        """Obtém o explainer do shap.

        Retorna
        -------
        shap.Explainer
            Objeto explainer inicializado.

        """
        kwargs = {
            "masker": self.branch.X_train,
            "feature_names": list(self.branch.features),
            "seed": self.random_state,
        }

        try:  # Falha quando o modelo não se encaixa nos explainers padrão
            return Explainer(self.estimator, **kwargs)
        except TypeError:
            # Se um método for fornecido como primeiro argumento, usa sempre Permutation
            responses = ("predict_proba", "decision_function", "predict")
            return Explainer(_check_response_method(self.estimator, responses), **kwargs)

    def get_explanation(
        self,
        df: pd.DataFrame,
        target: tuple[Int, ...],
    ) -> Explanation:
        """Obtém um objeto Explanation.

        Os valores shap são memoizados para não repetir cálculos nas
        mesmas linhas.

        Parâmetros
        ----------
        df: pd.DataFrame
            Conjunto de dados a analisar, subconjunto do conjunto completo.

        target: tuple
            Índices da coluna alvo e da classe na coluna alvo que serão analisadas.

        Retorna
        -------
        shap.Explanation
            Objeto contendo todas as informações: values, base_values e data.

        """
        # Obtém as linhas que ainda precisam ser calculadas
        if not (calculate := df.loc[~df.index.isin(self._shap_values.index)]).empty:
            kwargs: dict[str, Any] = {}

            # Mínimo de 2 * n_features + 1 avaliações requerido
            if "max_evals" in sign(self.explainer.__call__):
                kwargs["max_evals"] = "auto"

            # A checagem de aditividade às vezes falha sem motivo aparente
            if "check_additivity" in sign(self.explainer.__call__):
                kwargs["check_additivity"] = False

            with warnings.catch_warnings():
                # Evita warning de incompatibilidade de nomes de features no sklearn
                warnings.filterwarnings("ignore", message="X does not have valid.*")

                # Calcula os novos valores shap
                try:
                    self._explanation = self.explainer(calculate.to_numpy(), **kwargs)
                except (ValueError, AssertionError) as ex:
                    raise ValueError(
                        "Falha ao obter o explainer do shap para o estimador "
                        f"{self.estimator} com tarefa {self.task}. Exceção: {ex}"
                    ) from None

            # Armazena os valores shap no atributo _shap_values
            self._shap_values = pd.concat(
                [
                    self._shap_values,
                    pd.Series(list(self._explanation.values), index=calculate.index),
                ]
            )

        # Não usa o atributo diretamente para não salvar mudanças específicas do gráfico
        # Cópia rasa para não copiar os dados da branch
        explanation = copy(self._explanation)

        # Atualiza o objeto explanation
        explanation.values = np.stack(self._shap_values.loc[df.index].tolist())
        explanation.base_values = self._explanation.base_values[0]

        if self.task.is_multioutput:
            if explanation.values.shape[-1] == self.branch.y.shape[1]:
                # Uma explanation por coluna
                explanation.values = explanation.values[:, :, target[0]]
                explanation.base_values = explanation.base_values[target[0]]
            else:
                # Para multilabel nativo ou multiclasse multioutput, os valores
                # têm shape[-1]=n_targets x max(n_cls)
                n_classes = explanation.values.shape[-1] // self.branch.y.shape[1]

                select = target[0] * n_classes + target[1]
                explanation.values = explanation.values[:, :, select]
                explanation.base_values = explanation.base_values[select]
        elif explanation.values.ndim > 2 and len(target) == 2:
            explanation.values = explanation.values[:, :, target[1]]
            explanation.base_values = explanation.base_values[target[1]]

        return explanation


class ClassMap:
    """Lista de classes com mapeamento por atributo.

    Esta classe funciona de forma semelhante a uma lista, em que todos
    os elementos devem ter um certo atributo. É possível acessar a
    classe por índice ou por atributo. O acesso não diferencia maiúsculas
    de minúsculas.

    """

    @staticmethod
    def _conv(key: Any) -> Any:
        return key.lower() if isinstance(key, str) else key

    def _get_data(self, key: Any) -> Any:
        if isinstance(key, int_t) and key not in self.keys():
            try:
                return self.__data[key]
            except IndexError:
                raise KeyError(key) from None
        else:
            for data in self.__data:
                if self._conv(getattr(data, self.__key)) == self._conv(key):
                    return data

        raise KeyError(key)

    def _check(self, elem: T) -> T:
        if not hasattr(elem, self.__key):
            raise ValueError(f"O elemento {elem} não possui o atributo {self.__key}.")
        else:
            return elem

    def __init__(self, *args, key: str = "name"):
        """Define a chave e os dados.

        Imita a inicialização de uma lista e aceita um argumento extra
        para especificar o atributo usado como chave.

        """
        self.__key = key
        self.__data: list[Any] = []
        for elem in args:
            if isinstance(elem, GeneratorType):
                self.__data.extend(self._check(e) for e in elem)
            else:
                self.__data.append(self._check(elem))

    def __getitem__(self, key: Any) -> Any:
        """Obtém um valor ou subconjunto do mapeamento."""
        if isinstance(key, sequence_t):
            return self.__class__(*[self._get_data(k) for k in key], key=self.__key)
        elif isinstance(key, segment_t):
            return self.__class__(*get_segment(self.__data, key), key=self.__key)
        else:
            return self._get_data(key)

    def __setitem__(self, key: Any, value: Any):
        """Adiciona um novo item ao mapeamento."""
        if isinstance(key, int_t):
            self.__data[key] = self._check(value)
        else:
            try:
                self.__data = [e if self[key] == e else value for e in self.__data]
            except KeyError:
                self.append(value)

    def __delitem__(self, key: Any):
        """Exclui um item."""
        del self.__data[self.index(self._get_data(key))]

    def __iter__(self) -> Iterator[Any]:
        """Percorre os valores."""
        yield from self.__data

    def __len__(self) -> int:
        """Comprimento do mapeamento."""
        return len(self.__data)

    def __contains__(self, key: Any) -> bool:
        """Indica se a chave ou o valor existe."""
        return key in self.__data or self._conv(key) in self.keys_lower()

    def __repr__(self) -> str:
        """Exibe a representação do mapeamento."""
        return self.__data.__repr__()

    def __reversed__(self) -> Iterator[Any]:
        """Inverte a ordem do mapeamento."""
        yield from reversed(list(self.__data))

    def __eq__(self, other: object) -> bool:
        """Compara a igualdade entre instâncias."""
        return self.__data == other

    def __add__(self, other: ClassMap) -> ClassMap:
        """Mescla dois mapeamentos."""
        self.__data += other
        return self

    def __bool__(self) -> bool:
        """Indica se o mapeamento possui valores."""
        return bool(self.__data)

    def keys(self) -> list[Any]:
        """Retorna as chaves do mapeamento."""
        return [getattr(x, self.__key) for x in self.__data]

    def values(self) -> list[Any]:
        """Retorna os valores do mapeamento."""
        return self.__data

    def keys_lower(self) -> list[Any]:
        """Retorna as chaves do mapeamento em minúsculas."""
        return list(map(self._conv, self.keys()))

    def append(self, value: T) -> T:
        """Adiciona um item ao mapeamento."""
        self.__data.append(self._check(value))
        return value

    def extend(self, value: Any):
        """Estende o mapeamento com outra sequência."""
        self.__data.extend(list(map(self._check, value)))

    def remove(self, value: Any):
        """Remove um item."""
        if value in self.__data:
            self.__data.remove(value)
        else:
            self.__data.remove(self._get_data(value))

    def clear(self):
        """Limpa o conteúdo."""
        self.__data = []

    def index(self, key: Any) -> Any:
        """Retorna o índice da chave."""
        if key in self.__data:
            return self.__data.index(key)
        else:
            return self.__data.index(self._get_data(key))


# Funções ========================================================== >>


def flt(x: Any) -> Any:
    """Retorna o item de uma sequência quando ela possui apenas um elemento.

    Parâmetros
    ----------
    x: Any
        Item ou sequência.

    Retorna
    -------
    Any
        Objeto.

    """
    return x[0] if isinstance(x, sequence_t) and len(x) == 1 else x


def lst(x: Any) -> list[Any]:
    """Transforma um item em lista caso ele ainda não seja uma sequência.

    Parâmetros
    ----------
    x: Any
        Item ou sequência.

    Retorna
    -------
    list
        Item como lista de tamanho 1 ou a sequência fornecida como lista.

    """
    return list(x) if isinstance(x, (dict, *sequence_t, ClassMap)) else [x]


def it(x: Any) -> Any:
    """Converte floats arredondados em int.

    Se o item fornecido não for numérico, retorna-o sem alterações.

    Parâmetros
    ----------
    x: Any
        Item a verificar para arredondamento.

    Retorna
    -------
    Any
        Float arredondado ou item não numérico.

    """
    try:
        is_equal = int(x) == float(x)
    except ValueError:  # O item pode não ser numérico
        return x

    return int(x) if is_equal else float(x)


def rnd(x: Any, decimals: Int = 4) -> Any:
    """Arredonda um float para a casa `decimals`.

    Se o valor não for um float, retorna-o sem alterações.

    Parâmetros
    ----------
    x: Any
        Item numérico a arredondar.

    decimals: int, default=4
        Casa decimal para arredondamento.

    Retorna
    -------
    Any
        Float arredondado ou item não numérico.

    """
    return round(x, decimals) if np.issubdtype(type(x), np.floating) else x


def divide(a: Scalar, b: Scalar, decimals: Int = 4) -> int | float:
    """Divide dois números e retorna 0 em caso de divisão por zero.

    Parâmetros
    ----------
    a: int or float
        Numerador.

    b: int or float
        Denominador.

    decimals: int, default=4
        Casa decimal para arredondamento.

    Retorna
    -------
    int or float
        Resultado da divisão ou 0.

    """
    return float(np.round(np.divide(a, b), decimals)) if b != 0 else 0


def to_rgb(c: str) -> str:
    """Converte um nome de cor ou hexadecimal em rgb.

    Parâmetros
    ----------
    c: str
        Nome ou código da cor.

    Retorna
    -------
    str
        Representação RGB da cor.

    """
    if not c.startswith("rgb"):
        colors = to_rgba(c)[:3]
        return f"rgb({colors[0]}, {colors[1]}, {colors[2]})"

    return c


def sign(obj: Callable) -> MappingProxyType:
    """Obtém os parâmetros de um objeto.

    Parâmetros
    ----------
    obj: Callable
        Objeto do qual obter os parâmetros.

    Retorna
    -------
    mappingproxy
        Parâmetros do objeto.

    """
    return signature(obj).parameters


def merge(*args) -> pd.DataFrame:
    """Concatena objetos do pandas por coluna.

    Objetos None e vazios são ignorados.

    Parâmetros
    ----------
    *args
        Objetos a concatenar.

    Retorna
    -------
    pd.DataFrame
        Dataframe concatenado.

    """
    if len(args_c := [x for x in args if x is not None and not x.empty]) == 1:
        return pd.DataFrame(args_c[0])
    else:
        return pd.concat(args_c, axis=1)


def replace_missing(X: T_Pandas, missing_values: list[Any] | None = None) -> T_Pandas:
    """Substitui todos os valores considerados ausentes em um conjunto de dados.

    Este método substitui os valores ausentes em colunas com dtypes
    anuláveis do pandas por `pd.NA`; caso contrário, usa `np.NaN`.
    Conjuntos esparsos são ignorados, já que colunas esparsas não
    suportam atribuição item a item.

    Parâmetros
    ----------
    X: pd.Series or pd.DataFrame
        Conjunto de dados a ser tratado.

    missing_values: list or None, default=None
        Valores considerados ausentes. Se None, usa apenas os valores padrão.

    Retorna
    -------
    pd.Series or pd.DataFrame
        Conjunto de dados sem valores ausentes.

    """
    # Sempre converte esses valores
    default_values = [None, pd.NA, pd.NaT, np.nan, np.inf, -np.inf]

    if not is_sparse(X):
        get_nan: Callable[[DtypeObj], float | NAType] = lambda dtype: (
            np.nan if isinstance(dtype, np.dtype) else pd.NA
        )

        if isinstance(X, pd.DataFrame):
            return X.replace(
                to_replace={c: (missing_values or []) + default_values for c in X.columns},
                value={c: get_nan(d) for c, d in X.dtypes.items()},
            )
        else:
            return X.replace(
                to_replace=(missing_values or []) + default_values,
                value=get_nan(X.dtype),
            )
    else:
        return X


def n_cols(obj: YConstructor | None) -> int:
    """Obtém o número de colunas de um conjunto de dados.

    Parâmetros
    ----------
    obj: dict, sequence, dataframe-like or None
        Conjunto de dados a verificar.

    Retorna
    -------
    int
        Número de colunas.

    """
    if hasattr(obj, "shape"):
        return obj.shape[1] if len(obj.shape) > 1 else 1  # type: ignore[union-attr]
    elif isinstance(obj, dict):
        return 2  # Dict sempre vira dataframe

    try:
        if (array := np.asarray(obj)).ndim > 1:
            return array.shape[1]
        else:
            return array.ndim
    except ValueError:
        # Falha para dados heterogêneos, retorna series
        return 1


def get_cols(obj: Pandas) -> list[pd.Series]:
    """Obtém uma lista de colunas em dataframe ou series.

    Parâmetros
    ----------
    obj: pd.Series or pd.DataFrame
        Elemento do qual obter as colunas.

    Retorna
    -------
    list of pd.Series
        Colunas.

    """
    if isinstance(obj, pd.Series):
        return [obj]
    else:
        return [obj[col] for col in obj.columns]


def get_col_names(obj: Any) -> list[str] | None:
    """Obtém uma lista de nomes de colunas em objetos tabulares.

    Parâmetros
    ----------
    obj: object
        Elemento do qual obter os nomes das colunas.

    Retorna
    -------
    list of str
        Nomes das colunas. Retorna None quando o objeto fornecido não
        é um objeto do pandas.

    """
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    elif isinstance(obj, pd.Series):
        return [str(obj.name)]
    else:
        return None


def variable_return(
    X: XReturn | None,
    y: YReturn | None,
) -> XReturn | tuple[XReturn, YReturn]:
    """Retorna um ou dois argumentos dependendo de qual é None.

    Esta função utilitária é usada para fazer métodos retornarem apenas
    o conjunto de dados fornecido.

    Parâmetros
    ----------
    X: dataframe or None
        Conjunto de atributos.

    y: series, dataframe or None
        Coluna(s) alvo.

    Retorna
    -------
    series, dataframe or tuple
        Conjuntos de dados que não são None.

    """
    if y is None and X is not None:
        return X
    elif X is None and y is not None:
        return y
    elif X is not None and y is not None:
        return X, y
    else:
        raise ValueError("X e y não podem ser ambos None.")


def get_segment(obj: list[T], segment: Segment) -> list[T]:
    """Obtém um subconjunto de uma sequência por range ou slice.

    Parâmetros
    ----------
    obj: sequence
        Objeto a ser fatiado.

    segment: range or slice
        Segmento a extrair da sequência.

    Retorna
    -------
    sequence
        Subconjunto da sequência original.

    """
    if isinstance(segment, slice):
        return obj[segment]
    else:
        return obj[slice(segment.start, segment.stop, segment.step)]


def is_sparse(obj: Pandas) -> bool:
    """Verifica se o dataframe é esparso.

    Um conjunto de dados é considerado esparso se qualquer uma de suas
    colunas for esparsa.

    Parâmetros
    ----------
    obj: pd.Series or pd.DataFrame
        Conjunto de dados a verificar.

    Retorna
    -------
    bool
        Indica se o conjunto de dados é esparso.

    """
    return any(isinstance(col.dtype, pd.SparseDtype) for col in get_cols(obj))


def check_empty(obj: Pandas | None) -> Pandas | None:
    """Verifica se um objeto do pandas está vazio.

    Parâmetros
    ----------
    obj: pd.Series, pd.DataFrame or None
        Objeto do pandas a verificar.

    Retorna
    -------
    pd.Series, pd.DataFrame or None
        O mesmo objeto ou None se estiver vazio ou se obj for None.

    """
    return obj if isinstance(obj, pd.DataFrame) and not obj.empty else None


def check_dependency(name: str, pypi_name: str | None = None):
    """Verifica uma dependência opcional.

    Levanta um erro se o pacote não estiver instalado.

    Parâmetros
    ----------
    name: str
        Nome do pacote a verificar.

    pypi_name: str or None, default=None
        Nome do pacote no PyPI. Se None, é igual a `name`.

    """
    try:
        import_module(name)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            f"Não foi possível importar o pacote {name}. Instale-o com "
            f"`pip install {pypi_name or name.replace('_', '-')}` ou "
            "instale todas as dependências opcionais do experionml com "
            "`pip install experionml[full]`."
        ) from None


def check_nltk_module(module: str, *, quiet: bool):
    """Verifica se um módulo do pacote NLTK está disponível.

    Se o módulo não estiver disponível, ele é baixado.

    Parâmetros
    ----------
    module: str
        Nome do módulo a verificar.

    quiet: bool
        Indica se logs devem ser exibidos durante o download.

    """
    try:
        nltk.data.find(module)
    except LookupError:
        nltk.download(module.split("/")[-1], quiet=quiet)


def check_canvas(is_canvas: Bool, method: str):
    """Levanta um erro se um método incompatível for chamado a partir de um canvas.

    Parâmetros
    ----------
    is_canvas: bool
        Indica se o gráfico está em um canvas. Se True, um erro é levantado.

    method: str
        Nome do método a partir do qual a verificação é feita.

    """
    if is_canvas:
        raise PermissionError(
            f"O método {method} não pode ser chamado a partir de um "
            "canvas porque usa o backend do matplotlib."
        )


def check_predict_proba(models: Model | Sequence[Model], method: str):
    """Levanta um erro se um modelo não possuir método `predict_proba`.

    Parâmetros
    ----------
    models: model or sequence of models
        Modelos a verificar quanto ao atributo.

    method: str
        Nome do método a partir do qual a verificação é feita.

    """
    for m in lst(models):
        if not hasattr(m.estimator, "predict_proba"):
            raise PermissionError(
                f"O método {method} só está disponível para "
                f"modelos com método predict_proba, mas recebeu {m.name}."
            )


def check_scaling(obj: Pandas) -> bool:
    """Verifica se os dados estão escalonados.

    Um conjunto de dados é considerado escalonado quando a média das
    médias de todas as colunas está entre -0.25 e 0.15 e a média dos
    desvios padrão de todas as colunas está entre 0.85 e 1.15.

    Parâmetros
    ----------
    obj: pd.Series or pd.DataFrame
        Conjunto de dados a verificar.

    Retorna
    -------
    bool
        Indica se o conjunto de dados está escalonado.

    """
    if isinstance(obj, pd.DataFrame):
        mean = obj.mean(numeric_only=True).mean()
        std = obj.std(numeric_only=True).mean()
    else:
        mean = obj.mean()
        std = obj.std()
    return bool(-0.15 < mean < 0.15 and 0.85 < std < 1.15)


@contextmanager
def keep_attrs(estimator: Estimator):
    """Salva temporariamente os atributos personalizados de um estimador.

    O pipeline do ExperionML usa dois atributos personalizados em seus
    transformadores: _train_only e _cols. Como alguns transformadores
    redefinem seus atributos durante o fit, como os do sktime,
    encapsulamos o método fit em um contextmanager que salva e restaura
    esses atributos.

    """
    try:
        train_only = getattr(estimator, "_train_only", None)
        cols = getattr(estimator, "_cols", None)
        yield
    finally:
        if train_only is not None:
            estimator._train_only = train_only
        if cols is not None:
            estimator._cols = cols


@contextmanager
def adjust(
    estimator: Estimator | Model,
    *,
    transform: EngineDataOptions | None = None,
    verbose: Verbose | None = None,
) -> Iterator[Estimator]:
    """Ajusta temporariamente os parâmetros de saída de um estimador.

    O engine de dados e o nível de verbosidade do estimador são
    temporariamente alterados para os valores fornecidos. Para
    estimadores que não são do ExperionML, o efeito é nulo.

    Parâmetros
    ----------
    estimator: Estimator or Model
        Estimador cuja configuração será alterada temporariamente.

    transform: str or None, default=None
        Engine de dados do estimador. Se None, mantém o engine original.

    verbose: int or None, default=None
        Nível de verbosidade do estimador. Se None, mantém a verbosidade original.

    """
    if "experionml" in estimator.__module__:
        try:
            if transform is not None and hasattr(estimator, "set_output"):
                output = getattr(estimator, "_engine", EngineTuple())
                estimator.set_output(transform=transform)
            if verbose is not None and hasattr(estimator, "verbose"):
                verbosity = estimator.verbose
                estimator.verbose = verbose
            yield estimator
        finally:
            if transform is not None and hasattr(estimator, "set_output"):
                estimator._engine = output  # type: ignore[union-attr]
            if verbose is not None and hasattr(estimator, "verbose"):
                estimator.verbose = verbosity
    else:
        yield estimator


def get_versions(models: ClassMap) -> dict[str, str]:
    """Obtém as versões do ExperionML e dos pacotes dos modelos.

    Parâmetros
    ----------
    models: ClassMap
        Modelos dos quais verificar a versão.

    Retorna
    -------
    dict
        Versões atuais do ExperionML e dos modelos.

    """
    versions = {"experionml": __version__}
    for model in models:
        module = model._est_class.__module__.split(".")[0]
        versions[module] = sys.modules.get(module, import_module(module)).__version__

    return versions


def get_corpus(df: pd.DataFrame) -> str:
    """Obtém a coluna de texto de um dataframe.

    A coluna de texto deve se chamar `corpus`, sem diferenciar maiúsculas
    de minúsculas. Também verifica se a coluna consiste em uma string ou
    em uma sequência de strings.

    Parâmetros
    ----------
    df: pd.DataFrame
        Conjunto de dados do qual o corpus será obtido.

    Retorna
    -------
    str
        Nome da coluna do corpus.

    """
    try:
        corpus = next(col for col in df.columns if col.lower() == "corpus")

        if not is_bearable(df[corpus].iloc[0], (str, Sequence[str])):
            raise TypeError("O corpus deve consistir em uma string ou em uma sequência de strings.")
        else:
            return corpus
    except StopIteration as ex:
        raise ValueError(
            "O conjunto de dados fornecido não contém uma coluna chamada corpus."
        ) from ex


def time_to_str(t: Scalar) -> str:
    """Converte tempo para uma representação de string legível.

    A string resultante tem o formato 00h:00m:00s ou 1.000s se o valor
    estiver abaixo de 1 minuto.

    Parâmetros
    ----------
    t: int or float
        Tempo a converter, em segundos.

    Retorna
    -------
    str
        Representação textual do tempo.

    """
    h = int(t) // 3600
    m = int(t) % 3600 // 60
    s = t % 3600 % 60
    if not h and not m:  # Apenas segundos
        return f"{s:.3f}s"
    elif not h:  # Também minutos
        return f"{m:02.0f}m:{s:02.0f}s"
    else:  # Também horas
        return f"{h:02.0f}h:{m:02.0f}m:{s:02.0f}s"


@overload
def to_df(
    data: Literal[None],
    index: Axes | None = ...,
    columns: Axes | None = ...,
) -> None: ...


@overload
def to_df(
    data: XConstructor,
    index: Axes | None = ...,
    columns: Axes | None = ...,
) -> pd.DataFrame: ...


def to_df(
    data: XConstructor | None,
    index: Axes | None = None,
    columns: Axes | None = None,
) -> pd.DataFrame | None:
    """Converte um conjunto de dados em um dataframe do pandas.

    Parâmetros
    ----------
    data: dataframe-like or None
        Conjunto de dados a converter em dataframe. Se for None ou já
        for um dataframe do pandas, retorna sem alterações.

    index: sequence or None, default=None
        Valores do índice.

    columns: sequence or None, default=None
        Nomes das colunas. Use None para nomeação automática.

    Retorna
    -------
    pd.DataFrame or None
        Dados como dataframe. Retorna None se data for None.

    """
    if data is not None:
        if isinstance(data, pd.DataFrame):
            data_c = data.copy()
        elif hasattr(data, "to_pandas"):
            data_c = data.to_pandas()  # type: ignore[operator]
        elif hasattr(data, "__dataframe__"):
            # Converte a partir do protocolo de intercâmbio de dataframes
            data_c = pd.api.interchange.from_dataframe(data.__dataframe__())
        else:
            # Define nomes padrão de colunas, pois dict e series já possuem nomes
            if columns is None and not isinstance(data, dict | pd.Series):
                columns = [f"x{i}" for i in range(n_cols(data))]

            if sps.issparse(data):
                data_c = pd.DataFrame.sparse.from_spmatrix(data, index, columns)
            else:
                data_c = pd.DataFrame(
                    data=data,  # type: ignore[arg-type]
                    index=index,
                    columns=columns,
                    copy=True,
                )

        # Se for um conjunto textual, renomeia a coluna para corpus
        if list(data_c.columns) == ["x0"] and data_c.dtypes[0].name in CAT_TYPES:
            data_c = data_c.rename(columns={data_c.columns[0]: "corpus"})
        else:
            if isinstance(data_c.columns, pd.MultiIndex):
                raise ValueError("Colunas MultiIndex não são suportadas.")
            else:
                # Converte todos os nomes de colunas para str
                data_c.columns = data_c.columns.astype(str)

            # Não são permitidas linhas ou nomes de colunas duplicados
            if data_c.columns.duplicated().any():
                raise ValueError("Foram encontrados nomes de colunas duplicados em X.")

            if columns is not None:
                # Reordena as colunas para a ordem fornecida
                try:
                    data_c = data_c[list(columns)]  # Força a ordem determinada por columns
                except KeyError:
                    raise ValueError(
                        f"As colunas são diferentes das vistas no fit. As features "
                        f"{set(data_c.columns) - set(columns)} "  # type: ignore[arg-type]
                        "estão ausentes em X."
                    ) from None

        return data_c
    else:
        return None


@overload
def to_series(
    data: Literal[None],
    index: Axes | None = ...,
    name: str | None = ...,
) -> None: ...


@overload
def to_series(
    data: dict[str, Any] | Sequence[Any] | pd.DataFrame | PandasConvertible,
    index: Axes | None = ...,
    name: str | None = ...,
) -> pd.Series: ...


def to_series(
    data: dict[str, Any] | Sequence[Any] | pd.DataFrame | PandasConvertible | None,
    index: Axes | None = None,
    name: str | None = None,
) -> pd.Series | None:
    """Converte uma sequência em uma series do pandas.

    Parâmetros
    ----------
    data: dict, sequence, pd.DataFrame or None
        Dados a converter. Se forem None ou já forem uma series do
        pandas, retorna sem alterações.

    index: sequence, index or None, default=None
        Valores do índice.

    name: str or None, default=None
        Nome da series.

    Retorna
    -------
    pd.Series or None
        Dados como series. Retorna None se data for None.

    """
    if data is not None:
        if isinstance(data, pd.Series):
            data_c = data.copy()
        elif isinstance(data, pd.DataFrame):
            data_c = data.iloc[:, 0].copy()
        elif hasattr(data, "to_pandas"):
            data_c = data.to_pandas()
        else:
            try:
                # Achata arrays com shape=(n_samples, 1)
                array = np.asarray(data).ravel().tolist()
            except ValueError:
                # Falha para dados heterogêneos
                array = data

            data_c = pd.Series(array, index=index, name=name or "target", copy=True)

        return data_c
    else:
        return None


@overload
def to_tabular(
    data: Literal[None],
    index: Axes | None = ...,
    columns: str | Axes | None = ...,
) -> None: ...


@overload
def to_tabular(
    data: YConstructor,
    index: Axes | None = ...,
    columns: str | Axes | None = ...,
) -> Pandas: ...


def to_tabular(
    data: YConstructor | None,
    index: Axes | None = None,
    columns: str | Axes | None = None,
) -> Pandas | None:
    """Converte para um tipo tabular do pandas.

    Se os dados forem unidimensionais, converte para series; caso
    contrário, para dataframe.

    Parâmetros
    ----------
    data: dict, sequence, pd.DataFrame or None
        Dados a converter. Se None, retorna sem alterações.

    index: sequence, index or None, default=None
        Valores do índice.

    columns: str, sequence or None, default=None
        Nome das colunas. Use None para nomeação automática.

    Retorna
    -------
    pd.Series, pd.DataFrame or None
        Dados como um objeto do pandas.

    """
    if (n_targets := n_cols(data)) == 1:
        return to_series(data, index=index, name=flt(columns))  # type: ignore[arg-type]
    else:
        if columns is None and not isinstance(data, dict) and not hasattr(data, "__dataframe__"):
            columns = [f"y{i}" for i in range(n_targets)]

        return to_df(data, index=index, columns=columns)  # type: ignore[arg-type]


def check_is_fitted(
    obj: Any,
    *,
    exception: Bool = True,
    attributes: str | Sequence[str] | None = None,
) -> bool:
    """Verifica se um estimador está ajustado.

    Verifica se o estimador está ajustado conferindo a presença de
    atributos ajustados, isto é, não None nem vazios. Caso contrário,
    levanta NotFittedError. É um wrapper da função do sklearn, mas não
    checa a presença do método `fit` e pode retornar um booleano em vez
    de sempre levantar exceção.

    Parâmetros
    ----------
    obj: object
        Instância a verificar.

    exception: bool, default=True
        Indica se uma exceção deve ser levantada caso o estimador não
        esteja ajustado. Se False, retorna False.

    attributes: str, sequence or None, default=None
        Atributo(s) a verificar. Se None, o estimador é considerado
        ajustado se existir um atributo que termine com underscore e
        não comece com underscore duplo.

    Retorna
    -------
    bool
        Indica se o estimador está ajustado.

    """
    if not (is_fitted := _is_fitted(obj, attributes)) and exception:
        raise NotFittedError(
            f"A instância {type(obj).__name__} ainda não foi ajustada. "
            f"Chame {'run' if hasattr(obj, 'run') else 'fit'} com "
            "argumentos apropriados antes de usar este objeto."
        )

    return is_fitted


def get_custom_scorer(metric: str | MetricFunction | Scorer, pos_label: PosLabel = 1) -> Scorer:
    """Obtém um scorer a partir de uma string, função ou scorer.

    Os scorers usados pelo ExperionML possuem um atributo name.

    Parâmetros
    ----------
    metric: str, func or scorer
        Nome, função ou scorer a partir do qual obter o scorer. Se for
        uma função, o scorer é criado usando os parâmetros padrão de
        `make_scorer` do sklearn.

    pos_label: bool, int, float or str, default=1
        Rótulo positivo para classificação binária ou multilabel.

    Retorna
    -------
    scorer
        Scorer customizado do sklearn com atributo name.

    """
    if isinstance(metric, str):
        custom_acronyms = {
            "ap": "average_precision",
            "ba": "balanced_accuracy",
            "auc": "roc_auc",
            "logloss": "neg_log_loss",
            "ev": "explained_variance",
            "me": "max_error",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "msle": "neg_mean_squared_log_error",
            "mape": "neg_mean_absolute_percentage_error",
            "medae": "neg_median_absolute_error",
            "poisson": "neg_mean_poisson_deviance",
            "gamma": "neg_mean_gamma_deviance",
        }

        custom_scorers = {
            "tn": true_negatives,
            "fp": false_positives,
            "fn": false_negatives,
            "tp": true_positives,
            "fpr": false_positive_rate,
            "tpr": true_positive_rate,
            "tnr": true_negative_rate,
            "fnr": false_negative_rate,
            "mcc": matthews_corrcoef,
        }

        metric = metric.lower()
        if metric in get_scorer_names():
            scorer = get_scorer(metric)
        elif metric in custom_acronyms:
            scorer = get_scorer(custom_acronyms[metric])
        elif metric in custom_scorers:
            scorer = make_scorer(custom_scorers[metric])
        else:
            raise ValueError(
                f"Valor desconhecido para o parâmetro metric: {metric}. "
                f"Escolha entre: {', '.join(get_scorer_names())}."
            )

        scorer.name = metric

    elif hasattr(metric, "_score_func"):  # scoring é um scorer
        scorer = copy(metric)

        # Alguns scorers usam kwargs padrão
        if scorer._score_func.__name__.startswith(("precision", "recall", "f1", "jaccard")):
            if not scorer._kwargs:
                scorer._kwargs = {"average": "binary"}

        for key in get_scorer_names():
            if scorer.__dict__ == get_scorer(key).__dict__:
                scorer.name = key
                break

    else:  # scoring é uma função com assinatura metric(y_true, y_pred)
        scorer = make_scorer(score_func=metric)

    # Se nenhum nome foi atribuído, usa o nome da função
    if not hasattr(scorer, "name"):
        scorer.name = scorer._score_func.__name__
    if not hasattr(scorer, "fullname"):
        scorer.fullname = scorer._score_func.__name__

    if "pos_label" in sign(scorer._score_func):
        scorer._kwargs["pos_label"] = pos_label

    return scorer


# Funções de pipeline ============================================= >>


def name_cols(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    col_names: list[str],
) -> pd.Index:
    """Obtém os nomes das colunas após uma transformação.

    Se o número de colunas não mudou, os nomes originais são
    retornados. Caso contrário, a coluna recebe um nome padrão se os
    valores tiverem mudado.

    Parâmetros
    ----------
    df: pd.DataFrame
        Conjunto de dados transformado.

    original_df: pd.DataFrame
        Conjunto de dados original.

    col_names: list of str
        Colunas usadas no transformador.

    Retorna
    -------
    pd.Index
        Nomes das colunas.

    """
    # Se as colunas foram apenas transformadas, retorna os nomes originais
    if df.shape[1] == len(col_names):
        return pd.Index(col_names)

    # Se colunas foram adicionadas ou removidas
    temp_cols = []
    for i, column in enumerate(get_cols(df)):
        # equal_nan=True falha para dtypes não numéricos
        mask = original_df.apply(  # type: ignore[type-var]
            lambda c: np.array_equal(
                a1=c,
                a2=column,
                equal_nan=is_numeric_dtype(c) and np.issubdtype(column.dtype.name, np.number),
            )
        )

        if any(mask) and mask[mask].index[0] not in temp_cols:
            # Se a coluna for igual, usa o nome existente
            temp_cols.append(mask[mask].index[0])
        else:
            # Se a coluna for nova, usa um nome padrão
            counter = 0
            while True:
                n = f"x{i + counter + original_df.shape[1] - len(col_names)}"
                if (n not in original_df or n in col_names) and n not in temp_cols:
                    temp_cols.append(n)
                    break
                else:
                    counter += 1

    return pd.Index(temp_cols)


def get_col_order(
    new_columns: list[str],
    og_columns: list[str],
    col_names: list[str],
) -> np.ndarray:
    """Determina a ordem das colunas de um dataframe.

    A ordem das colunas é determinada pela ordem no conjunto de dados
    original. Colunas derivadas são colocadas na posição de sua coluna
    de origem.

    Parâmetros
    ----------
    new_columns: list of str
        Colunas do novo dataframe.

    og_columns: list of str
        Colunas do dataframe original.

    col_names: list of str
        Nomes das colunas usadas no transformador.

    Retorna
    -------
    np.ndarray
        Nova ordem das colunas.

    """
    columns: list[str] = []
    for col in og_columns:
        if col in new_columns or col not in col_names:
            columns.append(col)

        # Adiciona todas as colunas derivadas, por exemplo, colunas one-hot
        columns.extend([c for c in new_columns if c.startswith(f"{col}_") and c not in og_columns])

    # Adiciona as novas colunas restantes, que não são derivadas
    columns.extend([col for col in new_columns if col not in columns])

    return np.array(columns)


def reorder_cols(
    transformer: Transformer,
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    col_names: list[str],
) -> pd.DataFrame:
    """Reordena as colunas para sua ordem original.

    Esta função é necessária quando apenas um subconjunto das colunas do
    conjunto de dados foi usado. Nesse caso, precisamos reordená-las
    para sua ordem original.

    Parâmetros
    ----------
    transformer: Transformer
        Instância que transformou `df`.

    df: pd.DataFrame
        Conjunto de dados a reordenar.

    original_df: pd.DataFrame
        Conjunto de dados original, que define a ordem.

    col_names: list of str
        Nomes das colunas usadas no transformador.

    Retorna
    -------
    pd.DataFrame
        Conjunto de dados com colunas reordenadas.

    """
    # Verifica se as colunas retornadas pelo transformador já existem no conjunto
    for col in df:
        if col in original_df and col not in col_names:
            raise ValueError(
                f"Coluna '{col}' retornada pelo transformador {transformer} "
                "já existe no conjunto de dados original."
            )

    # Força novos índices no conjunto antigo para o merge
    try:
        original_df.index = df.index
    except ValueError as ex:  # Incompatibilidade de tamanho
        raise IndexError(
            f"Comprimento dos valores ({len(df)}) não corresponde ao comprimento "
            f"do índice ({len(original_df)}). Isso geralmente ocorre quando "
            "transformações que removem linhas não são aplicadas a "
            "todas as colunas."
        ) from ex

    columns = get_col_order(df.columns.tolist(), original_df.columns.tolist(), col_names)

    # Mescla os conjuntos novo e antigo mantendo as colunas mais recentes
    new_df = df.merge(
        right=original_df[[col for col in original_df if col in columns]],
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=("", "__drop__"),
    )
    new_df = new_df.drop(columns=new_df.filter(regex="__drop__$").columns)

    return new_df[columns]


def fit_one(
    estimator: Estimator,
    X: pd.DataFrame | None = None,
    y: Pandas | None = None,
    message: str | None = None,
    **fit_params,
) -> Estimator:
    """Ajusta os dados em um estimador.

    Parâmetros
    ----------
    estimator: Estimator
        Instância a ajustar.

    X: pd.DataFrame or None, default=None
        Conjunto de atributos com shape=(n_samples, n_features). Se
        None, `X` é ignorado.

    y: pd.Series, pd.DataFrame or None, default=None
        Coluna(s) alvo correspondentes a `X`.

    message: str or None, default=None
        Mensagem curta. Se None, nada será impresso.

    **fit_params
        Argumentos nomeados adicionais passados ao método `fit`.

    Retorna
    -------
    Estimator
        Estimador ajustado.

    """
    with _print_elapsed_time("Pipeline", message):
        if hasattr(estimator, "fit"):
            kwargs: dict[str, Pandas] = {}
            inc = getattr(estimator, "_cols", getattr(X, "columns", []))
            if "X" in (params := sign(estimator.fit)):
                if X is not None and (cols := [c for c in inc if c in X]):
                    kwargs["X"] = X[cols]

                # X é obrigatório, mas não foi fornecido
                if len(kwargs) == 0:
                    if y is not None and hasattr(estimator, "_cols"):
                        kwargs["X"] = to_df(y)[inc]
                    elif X is None:
                        raise ValueError(
                            "Exceção ao tentar ajustar o transformador "
                            f"{estimator.__class__.__name__}. O parâmetro "
                            "X é obrigatório, mas não foi fornecido."
                        )
                    elif X.empty:
                        raise ValueError(
                            "Exceção ao tentar ajustar o transformador "
                            f"{estimator.__class__.__name__}. O parâmetro X é "
                            "obrigatório, mas o conjunto de atributos fornecido está vazio. "
                            "Use o parâmetro columns para transformar apenas a "
                            "coluna alvo, por exemplo, experionml.decompose(columns=-1)."
                        )

            if y is not None:
                if "y" in params:
                    kwargs["y"] = y
                if "Y" in params:  # Alguns estimadores como ClassifierChain usam Y
                    kwargs["Y"] = y

            # Mantém atributos customizados, pois alguns transformadores os redefinem no fit
            with keep_attrs(estimator):
                estimator.fit(**kwargs, **fit_params)

        return estimator


def transform_one(
    transformer: Transformer,
    X: pd.DataFrame | None = None,
    y: Pandas | None = None,
    method: Literal["transform", "inverse_transform"] = "transform",
    **transform_params,
) -> tuple[pd.DataFrame | None, Pandas | None]:
    """Transforma os dados usando um estimador.

    Parâmetros
    ----------
    transformer: Transformer
        Instância a ajustar.

    X: pd.DataFrame or None, default=None
        Conjunto de atributos com shape=(n_samples, n_features). Se
        None, `X` é ignorado.

    y: pd.Series, pd.DataFrame or None, default=None
        Coluna(s) alvo correspondentes a `X`.

    method: str, default="transform"
        Método a aplicar: transform ou inverse_transform.

    **transform_params
        Argumentos nomeados adicionais passados ao método.

    Retorna
    -------
    pd.DataFrame or None
        Conjunto de atributos. Retorna None se não for fornecido.

    pd.Series, pd.DataFrame or None
        Coluna(s) alvo. Retorna None se não forem fornecidas.

    """

    def prepare_df(out: XConstructor, og: pd.DataFrame) -> pd.DataFrame:
        """Converte para dataframe e define os nomes corretos das colunas.

        Parâmetros
        ----------
        out: dataframe-like
            Dados retornados pela transformação.

        og: pd.DataFrame
            Dataframe original, antes das transformações.

        Retorna
        -------
        pd.DataFrame
            Conjunto de dados transformado.

        """
        out_c = to_df(out, index=og.index)

        # Atribui nomes apropriados às colunas
        use_cols = [c for c in inc if c in og.columns]
        if not isinstance(out, pd.DataFrame):
            if hasattr(transformer, "get_feature_names_out"):
                out_c.columns = transformer.get_feature_names_out()
            else:
                out_c.columns = name_cols(out_c, og, use_cols)

        # Reordena as colunas se apenas um subconjunto foi usado
        if len(use_cols) != og.shape[1]:
            return reorder_cols(transformer, out_c, og, use_cols)
        else:
            return out_c

    use_y = True

    kwargs: dict[str, Any] = {}
    inc = list(getattr(transformer, "_cols", getattr(X, "columns", [])))
    if "X" in (params := sign(getattr(transformer, method))):
        if X is not None and (cols := [c for c in inc if c in X]):
            kwargs["X"] = X[cols]

        # X é obrigatório, mas não foi fornecido
        if len(kwargs) == 0:
            if y is not None and hasattr(transformer, "_cols"):
                kwargs["X"] = to_df(y)[inc]
                use_y = False
            elif params["X"].default != Parameter.empty:
                kwargs["X"] = params["X"].default  # Preenche X com o valor padrão
            else:
                return X, y  # Se X for necessário, ignora o transformador

    if "y" in params:
        # Ignora `y` quando ele já foi adicionado a `X`
        if y is not None and use_y:
            kwargs["y"] = y
        elif "X" not in params:
            return X, y  # Se y for None e não houver X, ignora o transformador

    caller = getattr(transformer, method)
    out: YConstructor | tuple[XConstructor, YConstructor] = caller(**kwargs, **transform_params)

    # A transformação pode retornar X, y ou ambos
    X_new: pd.DataFrame | None
    y_new: Pandas | None
    if isinstance(out, tuple) and X is not None:
        X_new = prepare_df(out[0], X)
        y_new = to_tabular(out[1], index=X_new.index)
    elif "X" in params and X is not None and any(c in X for c in inc):
        # X entra -> X sai
        X_new = prepare_df(out, X)  # type: ignore[arg-type]
        y_new = y if y is None else y.set_axis(X_new.index, axis=0)
    elif y is not None:
        y_new = to_tabular(out)
        X_new = X if X is None else X.set_index(y_new.index)
        if isinstance(y, pd.DataFrame):
            y_new = prepare_df(y_new, y)

    return X_new, y_new


def fit_transform_one(
    transformer: Transformer,
    X: pd.DataFrame | None,
    y: Pandas | None,
    message: str | None = None,
    **fit_params,
) -> tuple[pd.DataFrame | None, Pandas | None, Transformer]:
    """Ajusta e transforma os dados usando um estimador.

    Estimadores sem método `transform` não são transformados.

    Parâmetros
    ----------
    transformer: Transformer
        Instância a ajustar.

    X: pd.DataFrame or None
        Conjunto de atributos com shape=(n_samples, n_features). Se
        None, `X` é ignorado.

    y: pd.Series, pd.DataFrame or None
        Coluna(s) alvo correspondentes a `X`.

    message: str or None, default=None
        Mensagem curta. Se None, nada será impresso.

    **fit_params
        Argumentos nomeados adicionais passados ao método `fit`.

    Retorna
    -------
    pd.DataFrame or None
        Conjunto de atributos. Retorna None se não for fornecido.

    pd.Series, pd.DataFrame or None
        Coluna(s) alvo. Retorna None se não forem fornecidas.

    Transformer
        Transformador ajustado.

    """
    fit_one(transformer, X, y, message, **fit_params)
    Xt, yt = transform_one(transformer, X, y)

    return Xt, yt, transformer


# Decoradores ====================================================== >>


def cache(f: Callable) -> Callable:
    """Utilitário para cache de método.

    Este decorador verifica se `functools.cache` funciona, o que falha
    quando args não são hashable, e, caso contrário, retorna o resultado
    sem usar cache.

    """

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return memoize(*args, **kwargs)
        except TypeError:
            return f(*args, **kwargs)

    memoize = functools.cache(f)

    # Adiciona ao decorador os métodos do memoizer
    wrapper.cache_info = memoize.cache_info
    wrapper.clear_cache = memoize.cache_clear

    return wrapper


def has_task(task: str | Sequence[str]) -> Callable:
    """Verifica se a instância possui uma tarefa específica.

    Se a verificação retornar False, a função decorada se torna
    indisponível para a instância.

    Parâmetros
    ----------
    task: str or sequence
        Tarefas a verificar. Escolha entre: classification, regression,
        forecast, binary e multioutput. Adicione o caractere `!` antes
        de uma tarefa para proibi-la.

    """

    def check(runner: BaseRunner) -> bool:
        checks = []
        for t in lst(task):
            if t.startswith("!"):
                checks.append(not getattr(runner.task, f"is_{t[1:]}"))
            else:
                checks.append(getattr(runner.task, f"is_{t}"))

        return all(checks)

    return check


def estimator_has_attr(attr: str) -> Callable:
    """Verifica se o estimador possui o atributo `attr`.

    Parâmetros
    ----------
    attr: str
        Nome do atributo a verificar.

    """

    def check(model: BaseModel) -> bool:
        # Levanta o `AttributeError` original se `attr` não existir
        getattr(model.estimator, attr)
        return True

    return check


def composed(*decs) -> Callable:
    """Adiciona múltiplos decoradores em uma única linha.

    Parâmetros
    ----------
    *decs
        Decoradores a executar.

    """

    def decorator(f: Callable) -> Callable:
        for dec in reversed(decs):
            f = dec(f)
        return f

    return decorator


def crash(
    f: Callable,
    cache: dict[str, Exception | None] = {"last_exception": None},  # noqa: B006
) -> Callable:
    """Salva falhas do programa no arquivo de log.

    Usamos um argumento mutável para armazenar em cache a última exceção
    levantada. Se a exceção atual for a mesma, o que pode ocorrer quando
    há captura de erro ou múltiplas chamadas de crash, ela não é escrita
    novamente no logger.

    """

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        try:  # Executa a função
            return f(*args, **kwargs)

        except Exception as ex:
            # Se a exceção não for a mesma da última vez, escreve no log
            if ex is not cache["last_exception"] and getattr(args[0], "logger", None):
                cache["last_exception"] = ex
                args[0].logger.exception("Exceção encontrada:")

            raise ex

    return wrapper


def method_to_log(f: Callable) -> Callable:
    """Salva as funções chamadas no arquivo de log."""

    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        if getattr(args[0], "logger", None):
            if f.__name__ != "__init__":
                args[0].logger.info("")
            args[0].logger.info(f"{args[0].__class__.__name__}.{f.__name__}()")

        return f(*args, **kwargs)

    return wrapper


def make_sklearn(
    obj: T_Estimator,
    feature_names_out: FeatureNamesOut = "one-to-one",
) -> T_Estimator:
    """Adiciona funcionalidades a uma classe para aderir à API do sklearn.

    O método `fit` de objetos que não são do sklearn é encapsulado para
    sempre adicionar os atributos `n_features_in_` e
    `feature_names_in_`, e o método `get-feature_names_out` é adicionado
    a transformadores que ainda não o possuem.

    Parâmetros
    ----------
    obj: Estimator
        Objeto a encapsular.

    feature_names_out: "one-to-one", callable or None, default="one-to-one"
                Determina a lista de nomes de features que será retornada pelo
                método `get_feature_names_out`.

                - Se None: o método `get_feature_names_out` não é definido.
                - Se "one-to-one": os nomes das features de saída serão iguais
                    aos nomes das features de entrada.
                - Se callable: função que recebe os argumentos posicionais self
                    e uma sequência de nomes de features de entrada. Ela deve
                    retornar uma sequência de nomes de features de saída.

        Retorna
    -------
    Estimator
        Objeto com o método fit encapsulado.

    """

    def wrap_fit(f: Callable) -> Callable:

        @wraps(f)
        def wrapper(self, *args, **kwargs):
            out = f(self, *args, **kwargs)

            # Para estimadores do sktime, o que importa é y, não X
            X = args[0] if len(args) > 0 else kwargs.get("X")

            # Adiciona os atributos e métodos após o fit para evitar
            # que sejam removidos por chamadas a .reset()
            if X is not None:
                if not hasattr(self, "feature_names_in_"):
                    BaseEstimator._check_feature_names(self, X, reset=True)
                if not hasattr(self, "n_features_in_"):
                    BaseEstimator._check_n_features(self, X, reset=True)

                if hasattr(self, "transform") and not hasattr(self, "get_feature_names_out"):
                    if feature_names_out == "one-to-one":
                        self.get_feature_names_out = FMixin.get_feature_names_out.__get__(self)
                    elif callable(feature_names_out):
                        self.get_feature_names_out = feature_names_out.__get__(self)

            return out

        # Evita encapsulamento duplo
        if getattr(f, "_fit_wrapped", False):
            return f
        else:
            wrapper._fit_wrapped = True

        return wrapper

    if not obj.__module__.startswith(("experionml.", "sklearn.", "imblearn.")):
        if isinstance(obj, type) and hasattr(obj, "fit"):
            obj.fit = wrap_fit(obj.fit)
        elif hasattr(obj.__class__, "fit"):
            obj.fit = wrap_fit(obj.__class__.fit).__get__(obj)  # type: ignore[method-assign]

    return obj


# Scorers customizados ============================================= >>


def true_negatives(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Int:
    """Resultado em que o modelo prediz corretamente a classe negativa."""
    return confusion_matrix(y_true, y_pred).ravel()[0]


def false_positives(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Int:
    """Resultado em que o modelo prediz incorretamente a classe negativa."""
    return confusion_matrix(y_true, y_pred).ravel()[1]


def false_negatives(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Int:
    """Resultado em que o modelo prediz incorretamente a classe positiva."""
    return confusion_matrix(y_true, y_pred).ravel()[2]


def true_positives(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Int:
    """Resultado em que o modelo prediz corretamente a classe positiva."""
    return confusion_matrix(y_true, y_pred).ravel()[3]


def false_positive_rate(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Float:
    """Probabilidade de um negativo real ser classificado como positivo."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def true_positive_rate(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Float:
    """Probabilidade de um positivo real ser classificado como positivo, sensibilidade."""
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


def true_negative_rate(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Float:
    """Probabilidade de um negativo real ser classificado como negativo, especificidade."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def false_negative_rate(y_true: Sequence[Int], y_pred: Sequence[Int]) -> Float:
    """Probabilidade de um positivo real ser classificado como negativo."""
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp)
