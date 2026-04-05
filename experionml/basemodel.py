from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime as dt
from functools import cached_property
from importlib import import_module
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast, overload
from unittest.mock import patch

import dill as pickle
import mlflow
import numpy as np
import optuna
import pandas as pd
from beartype import beartype
from joblib.memory import Memory
from joblib.parallel import Parallel, delayed
from mlflow.data import from_pandas
from mlflow.entities import Run
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from optuna import TrialPruned, create_study
from optuna.samplers import NSGAIISampler, TPESampler
from optuna.storages import InMemoryStorage
from optuna.study import Study
from optuna.terminator import report_cross_validation_scores
from optuna.trial import FrozenTrial, Trial, TrialState
from pandas.io.formats.style import Styler
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    BaseCrossValidator,
    FixedThresholdClassifier,
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TunedThresholdClassifierCV,
)
from sklearn.model_selection._validation import cross_validate
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier, MultiOutputRegressor
from sklearn.utils import resample
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import _check_response_method
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import make_forecasting_scorer
from sktime.split import ExpandingWindowSplitter, SingleWindowSplitter


try:
    from sktime.proba.normal import Normal
except ModuleNotFoundError:
    from sktime.base._proba._normal import Normal

from experionml.data import Branch, BranchManager
from experionml.data_cleaning import Scaler
from experionml.pipeline import Pipeline
from experionml.plots import RunnerPlot
from experionml.utils.constants import COLOR_SCHEME, DF_ATTRS
from experionml.utils.patches import fit_and_score
from experionml.utils.types import (
    HT,
    Backend,
    Bool,
    Engine,
    Float,
    Int,
    IntLargerEqualZero,
    MetricConstructor,
    MetricFunction,
    NJobs,
    Pandas,
    PredictionMethods,
    PredictionMethodsTS,
    Predictor,
    RowSelector,
    Scalar,
    Scorer,
    Sequence,
    Stages,
    TargetSelector,
    Verbose,
    Warnings,
    XReturn,
    XSelector,
    YConstructor,
    YReturn,
    YSelector,
    int_t,
)
from experionml.utils.utils import (
    ClassMap,
    DataConfig,
    Goal,
    PlotCallback,
    ShapExplanation,
    Task,
    TrialsCallback,
    adjust,
    cache,
    check_dependency,
    check_empty,
    composed,
    crash,
    estimator_has_attr,
    get_cols,
    get_custom_scorer,
    has_task,
    is_sparse,
    it,
    lst,
    merge,
    method_to_log,
    rnd,
    sign,
    time_to_str,
    to_df,
    to_series,
    to_tabular,
)


if TYPE_CHECKING:
    from starlette.requests import Request


# Desabilita logs de info do optuna (ExperionML já exibe as mesmas informações)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class BaseModel(RunnerPlot):
    """Classe base para todos os modelos.

    Parâmetros
    ----------
    goal: Goal
        Objetivo do modelo (classificação, regressão ou previsão).

    name: str or None, default=None
        Nome para o modelo. Se None, o nome é igual ao acrônimo do modelo.

    config: DataConfig or None, default=None
        Configuração dos dados. Se None, usa os valores padrão de configuração.

    branches: BranchManager or None, default=None
        BranchManager.

    metric: ClassMap or None, default=None
        Métrica na qual o modelo será ajustado.

    n_jobs: int, default=1
        Número de núcleos a usar para processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usa todos os núcleos disponíveis.
        - Se <-1: Usa número de núcleos - 1 + `n_jobs`.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], por ex.,
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str, dict or None, default=None
        Motor de execução a usar para [dados][data-engines] e
        [estimadores][estimator-acceleration]. O valor deve ser
        um dos possíveis valores para alterar um dos dois motores,
        ou um dicionário com chaves `data` e `estimator`, com suas
        escolhas correspondentes como valores para alterar ambos os motores. Se
        None, os valores padrão são usados. Escolha entre:

        - "data":

            - "numpy"
            - "pandas" (padrão)
            - "pandas-pyarrow"
            - "polars"
            - "polars-lazy"
            - "pyarrow"
            - "modin"
            - "dask"
            - "pyspark"
            - "pyspark-pandas"

        - "estimator":

            - "sklearn" (padrão)
            - "sklearnex"
            - "cuml"

    backend: str, default="loky"
        Backend de paralelização. Leia mais no
        [guia do usuário][parallel-execution]. Escolha entre:

        - "loky": Paralelismo baseado em processos, nó único.
        - "multiprocessing": Paralelismo legado baseado em processos, nó único.
          Menos robusto que `loky`.
        - "threading": Paralelismo baseado em threads, nó único.
        - "ray": Paralelismo baseado em processos, múltiplos nós.
        - "dask": Paralelismo baseado em processos, múltiplos nós.

    memory: bool, str, Path or Memory, default=False
        Habilita o cache para otimização de memória. Leia mais no
        [guia do usuário][memory-considerations].

        - Se False: Nenhum cache é realizado.
        - Se True: Um diretório temporário padrão é usado.
        - Se str: Caminho para o diretório de cache.
        - Se Path: Um [pathlib.Path][] para o diretório de cache.
        - Se Memory: Objeto com a interface [joblib.Memory][].

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    warnings: bool or str, default=False
        - Se True: Ação de aviso padrão (equivalente a "once").
        - Se False: Suprime todos os avisos (equivalente a "ignore").
        - Se str: Um dos [filtros de aviso][warnings] do Python.

        Alterar este parâmetro afeta o ambiente `PYTHONWarnings`.
        O ExperionML não consegue gerenciar avisos que passam de código C/C++ para stdout.

    logger: str, Logger or None, default=None
        - Se None: O registro em log não é usado.
        - Se str: Nome do arquivo de log. Use "auto" para nome automático.
        - Se Path: Um [pathlib.Path][] para o arquivo de log.
        - Caso contrário: Instância de `logging.Logger` do Python.

    experiment: str or None, default=None
        Nome do [experimento mlflow][experiment] a usar para rastreamento.
        Se None, nenhum rastreamento mlflow é realizado.

    random_state: int or None, default=None
        Semente usada pelo gerador de números aleatórios. Se None, o gerador
        de números aleatórios é o `RandomState` usado por `np.random`.

    """

    def __init__(
        self,
        goal: Goal,
        name: str | None = None,
        config: DataConfig | None = None,
        branches: BranchManager | None = None,
        metric: ClassMap | None = None,
        *,
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | Warnings = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            n_jobs=n_jobs,
            device=device,
            engine=engine,
            backend=backend,
            memory=memory,
            verbose=verbose,
            warnings=warnings,
            logger=logger,
            experiment=experiment,
            random_state=random_state,
        )

        self._goal = goal
        self._name = name or self.acronym

        self._config = config or DataConfig()
        self._metric = metric or ClassMap()

        self.scaler: Scaler | None = None

        if self.experiment:
            self._run = mlflow.start_run(run_name=self.name)
            mlflow.end_run()

        self._group = self._name  # sh and ts models belong to the same group
        self._evals: dict[str, list[Float]] = defaultdict(list)
        self._shap_explanation: ShapExplanation | None = None

        # Atributos de parâmetros
        self._est_params: dict[str, Any] = {}
        self._est_params_fit: dict[str, Any] = {}

        # Atributos de ajuste de hiperparâmetros
        self._ht: HT = {"distributions": {}, "cv": 1, "plot": False, "tags": {}}
        self._study: Study | None = None
        self._best_trial: FrozenTrial | None = None

        self._estimator: Predictor | None = None
        self._time_fit = 0.0

        self._bootstrap: pd.DataFrame | None = None
        self._time_bootstrap = 0.0

        # Injeta métodos específicos do objetivo do ForecastModel
        if goal is Goal.forecast and ClassRegModel in self.__class__.__bases__:
            for n, m in vars(ForecastModel).items():
                if hasattr(m, "__get__"):
                    setattr(self, n, m.__get__(self, ForecastModel))

        # Ignora esta parte se inicializado apenas para o estimador
        if branches:
            self._og = branches.og
            self._branch = branches.current
            self._train_idx = len(self.branch._data.train_idx)  # Pode mudar para sh e ts

            if getattr(self, "needs_scaling", None) and not self.branch.check_scaling():
                self.scaler = Scaler(
                    with_mean=not is_sparse(self.X_train),
                    device=self.device,
                    engine=self.engine.estimator,
                ).fit(self.X_train)

    def __repr__(self) -> str:
        """Exibe o nome da classe."""
        return f"{self.__class__.__name__}()"

    def __dir__(self) -> list[str]:
        """Adiciona atributos adicionais de __getattr__ ao dir."""
        # Exclui das condições _available_if
        attrs = [x for x in super().__dir__() if hasattr(self, x)]

        if "_branch" in self.__dict__:
            # Adiciona atributos adicionais do branch
            attrs += self.branch._get_shared_attrs()

            # Adiciona atributos adicionais do dataset
            attrs += [x for x in DF_ATTRS if hasattr(self.dataset, x)]

            # Adiciona nomes de colunas (excluindo os que contêm espaços)
            attrs += [c for c in self.columns if re.fullmatch(r"\w+$", c)]

        return attrs

    def __getattr__(self, item: str) -> Any:
        """Obtém atributos do branch ou dos dados."""
        if "_branch" in self.__dict__:
            if item in self.branch._get_shared_attrs():
                return getattr(self.branch, item)  # Obtém atributo do branch
            elif item in self.branch.columns:
                return self.branch.dataset[item]  # Obtém coluna
            elif item in DF_ATTRS:
                return getattr(self.branch.dataset, item)  # Obtém atributo do dataset

        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'.")

    def __contains__(self, item: str) -> bool:
        """Verifica se o item é uma coluna no dataset."""
        return item in self.dataset

    def __getitem__(self, item: Int | str | list) -> Pandas:
        """Obtém um subconjunto do dataset."""
        if isinstance(item, int_t):
            return self.dataset[self.columns[int(item)]]
        else:
            return self.dataset[item]  # Obtém um subconjunto do dataset

    @property
    def fullname(self) -> str:
        """Retorna o nome da classe do modelo."""
        return self.__class__.__name__

    @cached_property
    def _est_class(self) -> type[Predictor]:
        """Retorna a classe do estimador (não a instância).

        Esta propriedade verifica qual motor de estimador está habilitado e
        recupera o estimador do modelo da biblioteca correta.

        """
        locator = self._estimators.get(self._goal.name, self._estimators.get("regression"))
        module, est_name = locator.rsplit(".", 1)

        # Tenta o motor, senão importa do módulo padrão
        try:
            mod = import_module(f"{self.engine.estimator}.{module.split('.', 1)[1]}")
        except (ModuleNotFoundError, AttributeError, IndexError):
            mod = import_module(module)

        return getattr(mod, est_name)

    @property
    def _shap(self) -> ShapExplanation:
        """Retorna a instância ShapExplanation para este modelo."""
        if not self._shap_explanation:
            self._shap_explanation = ShapExplanation(
                estimator=self.estimator,
                task=self.task,
                branch=self.branch,
                random_state=self.random_state,
            )

        return self._shap_explanation

    def _check_est_params(self):
        """Verifica se os parâmetros são válidos para o estimador.

        Um parâmetro é sempre aceito se o método aceitar kwargs.

        """
        for param in self._est_params:
            if all(p not in sign(self._est_class) for p in (param, "kwargs")):
                raise ValueError(
                    "Valor inválido para o parâmetro est_params. Parâmetro desconhecido "
                    f"{param} fornecido para o estimador {self._est_class.__name__}."
                )

        for param in self._est_params_fit:
            if all(p not in sign(self._est_class.fit) for p in (param, "kwargs")):
                raise ValueError(
                    f"Valor inválido para o parâmetro est_params. "
                    f"Parâmetro desconhecido {param} fornecido para o método fit do "
                    f"estimador {self._est_class.__name__}."
                )

    def _get_param(self, name: str, params: dict[str, Any]) -> Any:
        """Obtém um parâmetro de est_params ou da função objetivo.

        Parâmetros
        ----------
        name: str
            Nome do parâmetro.

        params: dict
            Parâmetros no trial atual.

        Retorna
        -------
        Any
            Valor do parâmetro.

        """
        return self._est_params.get(name) or params.get(name)

    def _get_parameters(self, trial: Trial) -> dict[str, Any]:
        """Obtém os hiperparâmetros do trial.

        Este método busca as sugestões do trial e
        arredonda os floats para o quarto dígito.

        Parâmetros
        ----------
        trial: [Trial][]
            Trial atual.

        Retorna
        -------
        dict
            Hiperparâmetros do trial.

        """
        return {
            parameter: rnd(trial._suggest(parameter, value))
            for parameter, value in self._ht["distributions"].items()
        }

    def _trial_to_est(self, params: dict[str, Any]) -> dict[str, Any]:
        """Converte os parâmetros do trial para parâmetros do estimador.

        Alguns modelos, como MLP, usam hiperparâmetros diferentes para o
        estudo e para o estimador (isso ocorre quando o parâmetro do estimador
        não pode ser modelado segundo uma distribuição inteira, float ou
        categórica). Este método converte os parâmetros do trial para aqueles
        que podem ser ingeridos pelo estimador.
        Este método é sobrescrito nas classes filho. O método base
        simplesmente retorna os parâmetros como estão.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros do trial.

        Retorna
        -------
        dict
            Cópia dos hiperparâmetros do estimador.

        """
        return deepcopy(params)

    def _get_cv(self, cv: Int | BaseCrossValidator, max_length: Int) -> BaseCrossValidator:
        """Retorna uma classe de validador cruzado.

        O validador cruzado é selecionado com base na tarefa e na
        presença/ausência de grupos e estratificação.

        Parâmetros
        ----------
        cv: int or CrossValidator
            Número de folds ou objeto cv. Se objeto cv, é retornado como está.

        total_length: int
            Comprimento total do dataset. Usado apenas para tarefas de
            previsão quando cv > 1.

        Retorna
        -------
        CrossValidator
            Classe de validador cruzado.

        """
        if isinstance(cv, int_t):
            if self.task.is_forecast:
                if cv == 1:
                    return SingleWindowSplitter(fh=range(1, len(self.og.test)))
                else:
                    return ExpandingWindowSplitter(
                        fh=range(1, len(self.og.test) + 1),
                        initial_window=(max_length - len(self.og.test)) // cv,
                        step_length=(max_length - len(self.og.test)) // cv,
                    )
            else:
                if cv == 1:
                    if self._config.metadata.get("groups") is not None:
                        return GroupShuffleSplit(
                            n_splits=cv,
                            test_size=self._config.test_size,
                            random_state=self.random_state,
                        )
                    elif self._config.stratify is None:
                        return ShuffleSplit(
                            n_splits=cv,
                            test_size=self._config.test_size,
                            random_state=self.random_state,
                        )
                    else:
                        return StratifiedShuffleSplit(
                            n_splits=cv,
                            test_size=self._config.test_size,
                            random_state=self.random_state,
                        )
                else:
                    rs = self.random_state if self._config.shuffle else None
                    if self._config.metadata.get("groups") is None:
                        if self._config.stratify is None:
                            return KFold(n_splits=cv)
                        else:
                            return StratifiedKFold(
                                n_splits=cv,
                                shuffle=self._config.shuffle,
                                random_state=rs,
                            )
                    else:
                        if self._config.stratify is None:
                            return GroupKFold(n_splits=cv)
                        else:
                            return StratifiedGroupKFold(
                                n_splits=cv,
                                shuffle=self._config.shuffle,
                                random_state=rs,
                            )
        else:
            return cv

    def _get_est(self, params: dict[str, Any]) -> Predictor:
        """Obtém a instância do estimador.

        Usa o meta-estimador multioutput se o estimador não tiver
        suporte nativo para múltiplas saídas.

        Usa a função [make_reduction][] do sktime para regressores
        em tarefas de previsão.

        Parâmetros
        ----------
        params: dict
            Hiperparâmetros para o estimador.

        Retorna
        -------
        Predictor
            Instância do estimador.

        """
        # Separa os parâmetros do estimador daqueles nos sub-estimadores
        base_params, sub_params = {}, {}
        for name, value in params.items():
            if "__" not in name:
                base_params[name] = value
            else:
                sub_params[name] = value

        estimator = self._est_class(**base_params).set_params(**sub_params)

        fixed = tuple(params)
        if hasattr(self, "task"):
            if self.task.is_forecast and self._goal.name not in self._estimators:
                fixed = tuple(f"estimator__{f}" for f in fixed)

                # Tarefa de previsão com um regressor
                if self.task.is_multioutput:
                    estimator = make_reduction(estimator, strategy="multioutput")
                else:
                    estimator = make_reduction(estimator, strategy="recursive")

            elif self.task is Task.multilabel_classification:
                if not self.native_multilabel:
                    fixed = tuple(f"base_estimator__{f}" for f in fixed)
                    estimator = ClassifierChain(estimator)

            elif self.task.is_multioutput and not self.native_multioutput:
                fixed = tuple(f"estimator__{f}" for f in fixed)
                if self.task.is_classification:
                    estimator = MultiOutputClassifier(estimator)
                elif self.task.is_regression:
                    estimator = MultiOutputRegressor(estimator)

        return self._inherit(estimator, fixed=fixed)

    def _fit_estimator(
        self,
        estimator: Predictor,
        data: tuple[pd.DataFrame, Pandas],
        validation: tuple[pd.DataFrame, Pandas] | None = None,
        trial: Trial | None = None,
    ) -> Predictor:
        """Ajusta o estimador e realiza validação durante o treinamento.

        A avaliação durante o treinamento é realizada em modelos com o
        método `partial_fit`. Após cada ajuste parcial, o estimador
        é avaliado (usando apenas a métrica principal) nos dados de validação
        e, opcionalmente, a poda é realizada.

        Parâmetros
        ----------
        estimator: Predictor
            Instância a ser ajustada.

        data: tuple
            Dados de treinamento no formato (X, y).

        validation: tuple or None
            Dados de validação no formato (X, y). Se None, nenhuma validação
            é realizada.

        trial: [Trial][] or None
            Trial ativo (durante o ajuste de hiperparâmetros).

        Retorna
        -------
        Predictor
            Instância ajustada.

        """
        kwargs: dict[str, Any] = {}
        if (sample_weight := self._config.get_sample_weight(data[0])) is not None:
            if "sample_weight" in sign(estimator.fit):
                kwargs["sample_weight"] = sample_weight
            if hasattr(estimator, "set_fit_request"):
                estimator.set_fit_request(sample_weight=True)
            if hasattr(estimator, "set_partial_fit_request"):
                estimator.set_partial_fit_request(sample_weight=True)

        if getattr(self, "validation", False) and hasattr(estimator, "partial_fit") and validation:
            # Loop over first parameter in estimator
            try:
                steps = estimator.get_params()[self.validation]
            except KeyError:
                # For meta-estimators like multioutput
                steps = estimator.get_params()[f"estimator__{self.validation}"]

            if self.task.is_classification:
                if self.task.is_multioutput:
                    if self.native_multilabel:
                        kwargs["classes"] = list(range(self.y.shape[1]))
                    else:
                        kwargs["classes"] = [np.unique(y) for y in get_cols(self.y)]
                else:
                    kwargs["classes"] = list(np.unique(self.y))

            for step in range(steps):
                estimator.partial_fit(*data, **self._est_params_fit, **kwargs)

                if not trial:
                    # Store train and validation scores on the main metric in evals attr
                    self._evals[f"{self._metric[0].name}_train"].append(
                        self._score_from_est(self._metric[0], estimator, *data)
                    )
                    self._evals[f"{self._metric[0].name}_test"].append(
                        self._score_from_est(self._metric[0], estimator, *validation)
                    )

                # Otimização multi-objetivo não suporta poda
                if trial and len(self._metric) == 1:
                    trial.report(
                        value=float(self._score_from_est(self._metric[0], estimator, *validation)),
                        step=step,
                    )

                    if trial.should_prune():
                        # Solução alternativa para adicionar o passo podado à saída
                        if self.validation in trial.params:
                            trial.params[self.validation] = f"{step}/{steps}"

                        trial.set_user_attr("estimator", estimator)
                        raise TrialPruned

        else:
            if isinstance(estimator, BaseForecaster):
                if estimator.get_tag("requires-fh-in-fit") and "fh" not in self._est_params_fit:
                    # Adiciona o horizonte de previsão aos estimadores sktime quando necessário
                    kwargs["fh"] = self.test.index

                estimator.fit(data[1], X=check_empty(data[0]), **self._est_params_fit, **kwargs)
            else:
                estimator.fit(*data, **self._est_params_fit, **kwargs)

        return estimator

    def _best_score(self, metric: str | None = None) -> Scalar:
        """Retorna a melhor pontuação para o modelo.

        A melhor pontuação é a pontuação de bootstrap ou de teste, verificadas
        nessa ordem.

        Parâmetros
        ----------
        metric: str or None, default=None
            Nome da métrica a usar (para execuções com múltiplas métricas). Se None,
            a métrica principal é selecionada.

        Retorna
        -------
        float
            Melhor pontuação.

        """
        if self._bootstrap is None:
            return self.results[f"{metric or self._metric[0].name}_test"]
        else:
            return self.results[f"{metric or self._metric[0].name}_bootstrap"]

    def _final_output(self) -> str:
        """Retorna a saída final do modelo como string.

        Se [bootstrapping][] foi usado, usa o formato: média +- desvio.

        Retorna
        -------
        str
            Representação da pontuação final.

        """
        try:
            if self._bootstrap is None:
                out = "   ".join(
                    [f"{met}: {rnd(self._best_score(met))}" for met in self._metric.keys()]
                )
            else:
                out = "   ".join(
                    [
                        f"{met}: {rnd(self.bootstrap[met].mean())} "
                        f"\u00b1 {rnd(self.bootstrap[met].std())}"
                        for met in self._metric.keys()
                    ]
                )

            if not self.task.is_forecast:
                # Anota se o modelo superajustou quando treino 20% > teste na métrica principal
                score_train = self.results[f"{self._metric[0].name}_train"]
                score_test = self.results[f"{self._metric[0].name}_test"]
                if (1.2 if score_train < 0 else 0.8) * score_train > score_test:
                    out += " ~"

        except (TypeError, AttributeError):  # Falha quando errors="keep"
            out = "FAIL"

        return out

    @cache
    def _get_pred(
        self,
        rows: RowSelector,
        target: TargetSelector | None = None,
        method: PredictionMethods | Sequence[PredictionMethods] = "predict",
    ) -> tuple[Pandas, Pandas]:
        """Obtém os valores reais e previstos para uma coluna.

        As previsões são feitas usando os atributos `decision_function` ou
        `predict_proba` sempre que disponíveis, verificados nessa ordem.

        Parâmetros
        ----------
        rows: hashable, segment, sequence or dataframe
            [Seleção de linhas][row-and-column-selection] para as quais obter
            as previsões.

        target: str or None, default=None
            Coluna alvo a verificar. Apenas para [tarefas multioutput][].
            Se None, todas as colunas são retornadas.

        method: str or sequence, default="predict"
            Método(s) de resposta usados para obter previsões. Se sequença,
            a ordem fornecida define a ordem em que os métodos são tentados.

        Retorna
        -------
        series or dataframe
            Valores reais.

        series or dataframe
            Valores previstos.

        """
        method_caller = _check_response_method(self.estimator, method).__name__

        X, y = self.branch._get_rows(rows, return_X_y=True)

        # Filter for indices in dataset (required for sh and ts)
        X = X.loc[X.index.isin(self._all.index)]
        y_true = y.loc[y.index.isin(self._all.index)]

        if self.task.is_forecast:
            try:
                if X.empty:
                    exog = None
                else:
                    # Modelos Statsmodels como SARIMAX e DF requerem todos os
                    # dados exógenos após a última linha do conjunto de treino
                    # Outros modelos aceitam este formato
                    Xe = pd.concat([self.test, self.holdout])
                    exog = Xe.loc[Xe.index <= X.index.max(), self.features]  # type: ignore[index]

                y_pred = self._prediction(
                    fh=X.index,
                    X=exog,
                    verbose=0,
                    method=method_caller,
                )

            except (ValueError, NotImplementedError) as ex:
                # Pode falhar para modelos que não permitem previsões in-sample
                self._log(
                    f"Falha ao obter previsões para o modelo {self.name} "
                    f"nas linhas {rows}. Retornando NaN. Exceção: {ex}.",
                    3,
                )
                y_pred = pd.Series([np.nan] * len(X), index=X.index)

        else:
            y_pred = self._prediction(X.index, verbose=0, method=method_caller)

        if self.task.is_multioutput:
            if target is not None:
                target = self.branch._get_target(target, only_columns=True)
                return y_true.loc[:, target], y_pred.loc[:, target]
        elif self.task.is_binary and y_pred.ndim > 1:
            return y_true, y_pred.iloc[:, 1]

        return y_true, y_pred

    def _score_from_est(
        self,
        scorer: Scorer,
        estimator: Predictor,
        X: pd.DataFrame,
        y: Pandas,
        sample_weight: Sequence[Scalar] | None = None,
    ) -> Float:
        """Calcula a pontuação da métrica a partir de um estimador.

        Parâmetros
        ----------
        scorer: Scorer
            Métrica a calcular.

        estimator: Predictor
            Instância do estimador para obter a pontuação.

        X: pd.DataFrame
            Conjunto de features.

        y: pd.Series or pd.DataFrame
            Coluna(s) alvo correspondente(s) a `X`.

        sample_weight: sequence or None, default=None
            Pesos de amostra para o método `score`.

        Retorna
        -------
        float
            Pontuação calculada.

        """
        if self.task.is_forecast:
            y_pred = estimator.predict(fh=y.index, X=check_empty(X))
        else:
            y_pred = to_tabular(
                data=_check_response_method(estimator, scorer._response_method)(X),
                index=y.index,
            )
            if isinstance(y_pred, pd.DataFrame) and self.task is Task.binary_classification:
                y_pred = y_pred.iloc[:, 1]  # Retorna a probabilidade da classe positiva

        return self._score_from_pred(scorer, y, y_pred, sample_weight=sample_weight)

    def _score_from_pred(
        self,
        scorer: Scorer,
        y_true: Pandas,
        y_pred: Pandas,
        sample_weight: Sequence[Scalar] | None = None,
    ) -> Float:
        """Calcula a pontuação da métrica a partir de valores previstos.

        Como as métricas do sklearn não suportam tarefas multiclass-multioutput,
        calcula a média das pontuações sobre as colunas alvo para tais tarefas.

        Parâmetros
        ----------
        scorer: Scorer
            Métrica a calcular.

        y_true: pd.Series or pd.DataFrame
            Valores reais nas coluna(s) alvo.

        y_pred: pd.Series or pd.DataFrame
            Valores previstos correspondentes a y_true.

        sample_weight: sequence or None, default=None
            Pesos de amostra para o método `score`. Se não fornecido mas
            disponível nos metadados, esses são usados.

        Retorna
        -------
        float
            Pontuação calculada.

        """
        kwargs: dict[str, Any] = {}
        if "sample_weight" in sign(scorer._score_func):
            if sample_weight is not None:
                kwargs = {"sample_weight": sample_weight}
            elif (sw := self._config.get_sample_weight(y_true)) is not None:
                kwargs = {"sample_weight": sw}
                if hasattr(scorer, "set_score_request"):
                    scorer.set_score_request(sample_weight=True)

        func = lambda y1, y2: scorer._score_func(y1, y2, **scorer._kwargs, **kwargs)

        # Modelos de previsão podem ter previsões NaN, por exemplo, quando
        # usam internamente uma transformação boxcox em previsões negativas
        if self.task.is_forecast:
            y_pred = y_pred.dropna()
            y_true = y_true.loc[y_pred.index]

        if y_pred.empty:
            return np.nan

        if self.task is Task.multiclass_multioutput_classification:
            # Obtém a média das pontuações sobre as colunas alvo
            return np.mean(
                [
                    scorer._sign * func(y_true[col1], y_pred[col2])
                    for col1, col2 in zip(y_true, y_pred, strict=True)
                ],
                axis=0,
            )
        else:
            return scorer._sign * func(y_true, y_pred)

    def _get_score(self, scorer: Scorer, rows: RowSelector) -> Scalar:
        """Calcula uma pontuação de métrica.

        Parâmetros
        ----------
        scorer: Scorer
            Métricas a calcular. Se None, uma seleção das métricas mais
            comuns por tarefa é usada.

        rows: hashable, segment, sequence or dataframe
            [Seleção de linhas][row-and-column-selection] na qual
            calcular a métrica.

        Retorna
        -------
        int or float
            Pontuação da métrica no conjunto de dados selecionado.

        """
        y_true, y_pred = self._get_pred(rows, method=scorer._response_method)
        result = rnd(self._score_from_pred(scorer, y_true, y_pred))
        # Registra a métrica na execução mlflow para previsões nos conjuntos de dados
        if self.experiment and isinstance(rows, str):
            MlflowClient().log_metric(
                run_id=self.run.info.run_id,
                key=f"{scorer.name}_{rows}",
                value=it(result),
            )

        return result

    @composed(crash, method_to_log, beartype)
    def hyperparameter_tuning(self, n_trials: Int, *, reset: Bool = False):
        """Executa o algoritmo de ajuste de hiperparâmetros.

        Busca a melhor combinação de hiperparâmetros. A função
        a otimizar é avaliada com uma validação cruzada K-fold
        no conjunto de treino ou usando uma divisão aleatória de treino e
        validação a cada trial. Use este método para continuar a otimização.

        Parâmetros
        ----------
        n_trials: int
            Número de trials para o ajuste de hiperparâmetros.

        reset: bool, default=False
            Se deve iniciar um novo estudo ou continuar o existente.

        """

        def objective(trial: Trial) -> list[float]:
            """Função objetivo para ajuste de hiperparâmetros.

            Parâmetros
            ----------
            trial: optuna.trial.Trial
               Hiperparâmetros do modelo usados nesta chamada do BO.

            Retorna
            -------
            list of float
                Pontuações do estimador neste trial.

            """

            def fit_model(
                estimator: Predictor,
                train_idx: np.ndarray,
                val_idx: np.ndarray,
            ) -> tuple[Predictor, list[Float]]:
                """Ajusta o modelo. Função para paralelização.

                Divide o conjunto de treino em um (sub) treino e validação
                para este ajuste. Os conjuntos são criados a partir do dataset original
                para evitar vazamento de dados, pois o conjunto de treino é
                transformado usando o pipeline ajustado no mesmo conjunto.
                Ajusta o modelo em custom_fit se existir, senão normalmente.
                Retorna a pontuação no conjunto de validação.

                Parâmetros
                ----------
                estimator: Predictor
                    Estimador do modelo a ajustar.

                train_idx: np.array
                    Índices para o subconjunto de treino.

                val_idx: np.array
                    Índices para o conjunto de validação.

                Retorna
                -------
                Predictor
                    Estimador ajustado.

                list of float
                    Pontuações do estimador no conjunto de validação.

                """
                X_sub = self.og.X_train.iloc[train_idx]
                y_sub = self.og.y_train.iloc[train_idx]
                X_val = self.og.X_train.iloc[val_idx]
                y_val = self.og.y_train.iloc[val_idx]

                # Transforma os subconjuntos se houver um pipeline
                if len(pl := clone(self.pipeline)) > 0:
                    X_sub, y_sub = pl.fit_transform(X_sub, y_sub)
                    X_val, y_val = pl.transform(X_val, y_val)

                estimator = self._fit_estimator(
                    estimator=estimator,
                    data=(X_sub, y_sub),
                    validation=(X_val, y_val),
                    trial=trial,
                )

                scores = [
                    self._score_from_est(metric, estimator, X_val, y_val) for metric in self._metric
                ]

                return estimator, scores

            # Inicia o trial ========================================== >>

            params = self._get_parameters(trial)

            # Como os valores sugeridos não são os valores exatos usados
            # no estimador (arredondados por _get_parameters), implementamos
            # este método alternativo para sobrescrever os params no armazenamento
            if isinstance(self.study._storage, InMemoryStorage):
                trial._cached_frozen_trial.params = params
                frozen_trial = self.study._storage._get_trial(trial.number)
                frozen_trial.params = params
                self.study._storage._set_trial(trial.number, frozen_trial)

            # Armazena tags definidas pelo usuário
            for key, value in self._ht["tags"].items():
                trial.set_user_attr(key, value)

            # Cria instância do estimador com hiperparâmetros específicos do trial
            estimator = self._get_est(self._est_params | self._trial_to_est(params))

            # Verifica se os mesmos parâmetros já foram avaliados
            for t in trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))[::-1]:
                if trial.params == t.params:
                    # Obtém o mesmo estimador e pontuação da avaliação anterior
                    estimator = deepcopy(t.user_attrs["estimator"])
                    score = t.value if len(self._metric) == 1 else t.values
                    break
            else:
                splitter = self._get_cv(self._ht["cv"], max_length=len(self.og.train))

                # Segue a mesma estratégia de divisão que o experionml
                stratify = self._config.get_stratify_column(self.og.train)
                groups = self._config.get_metadata(self.og.train).get("groups")

                kwargs: dict[str, Any] = {next(iter(sign(splitter.split))): self.og.X_train}
                if stratify is not None:
                    kwargs["y"] = stratify
                if groups is not None:
                    kwargs["groups"] = groups

                # Loop paralelo sobre fit_model
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(fit_model)(estimator, i, j) for i, j in splitter.split(**kwargs)
                )

                estimator = results[0][0]
                score = np.mean(scores := [r[1] for r in results], axis=0).tolist()

                if len(results) > 1:
                    # Report cv scores for termination judgment
                    report_cross_validation_scores(trial, scores)

            trial.set_user_attr("estimator", estimator)

            return score

        # Executando ajuste de hiperparâmetros ============================ >>

        self._log(f"Executando ajuste de hiperparâmetros para {self.fullname}...", 1)

        # Verifica a validade dos parâmetros fornecidos
        self._check_est_params()

        # Atribui distribuições personalizadas ou usa as predefinidas
        dist = self._get_distributions() if hasattr(self, "_get_distributions") else {}
        if self._ht.get("distributions"):
            # Select default distributions
            inc, exc = [], []
            for name in [k for k, v in self._ht["distributions"].items() if v is None]:
                # Se for um nome, usa a dimensão predefinida
                if name.startswith("!"):
                    exc.append(n := name[1:])
                else:
                    inc.append(n := name)

                if n not in dist:
                    raise ValueError(
                        "Valor inválido para o parâmetro distributions. "
                        f"O parâmetro {n} não é um hiperparâmetro predefinido "
                        f"do modelo {self.fullname}. Consulte a documentação do modelo "
                        "para uma visão geral dos hiperparâmetros disponíveis "
                        "e suas distribuições."
                    )

            if inc and exc:
                raise ValueError(
                    "Valor inválido para o parâmetro distributions. Você pode incluir "
                    "ou excluir hiperparâmetros, mas não combinações de ambos."
                )
            elif exc:
                # Se distribuições foram excluídas com `!`, seleciona todas exceto essas
                self._ht["distributions"] = {k: v for k, v in dist.items() if k not in exc}
            elif inc:
                self._ht["distributions"] = {k: v for k, v in dist.items() if k in inc}
        else:
            self._ht["distributions"] = dist

        # Remove hiperparâmetro se já definido em est_params
        self._ht["distributions"] = {
            k: v for k, v in self._ht["distributions"].items() if k not in self._est_params
        }

        # Se não houver hiperparâmetros a otimizar, ignora ht
        if not self._ht["distributions"]:
            self._log(" --> Ignorando estudo. Nenhum hiperparâmetro a otimizar.", 2)
            return

        if not self._study or reset:
            kw: dict[str, Any] = {k: v for k, v in self._ht.items() if k in sign(create_study)}

            if len(self._metric) == 1:
                kw["direction"] = "maximize"
                kw["sampler"] = kw.pop("sampler", TPESampler(seed=self.random_state))
            else:
                kw["directions"] = ["maximize"] * len(self._metric)
                kw["sampler"] = kw.pop("sampler", NSGAIISampler(seed=self.random_state))

            self._study = create_study(study_name=self.name, **kw)

        kw = {k: v for k, v in self._ht.items() if k in sign(Study.optimize)}
        n_jobs = kw.pop("n_jobs", 1)

        # Inicializa o gráfico ao vivo do estudo
        if self._ht.get("plot", False) and n_jobs == 1:
            plot_callback = PlotCallback(
                name=self.fullname,
                metric=self._metric.keys(),
                aesthetics=self.aesthetics,
            )
        else:
            plot_callback = None

        callbacks = [*kw.pop("callbacks", []), TrialsCallback(self, n_jobs)]
        callbacks += [plot_callback] if plot_callback else []

        self._study.optimize(
            func=objective,
            n_trials=int(n_trials),
            n_jobs=n_jobs,
            callbacks=callbacks,
            show_progress_bar=kw.pop("show_progress_bar", self.verbose == 1),
            **kw,
        )

        if len(self.study.get_trials(states=[TrialState.COMPLETE])) == 0:
            self._study = None
            self._log(
                "O estudo não completou nenhum trial com sucesso. "
                "Ignorando o ajuste de hiperparâmetros.",
                1,
                severity="warning",
            )
            return

        self._log(f"Ajuste de hiperparâmetros {'-' * 27}", 1)
        self._log(f"Melhor trial --> {self.best_trial.number}", 1)
        self._log("Melhores parâmetros:", 1)
        self._log("\n".join([f" --> {k}: {v}" for k, v in self.best_params.items()]), 1)
        out = [
            f"{met}: {rnd(self.trials.loc[self.best_trial.number, met])}"
            for met in self._metric.keys()
        ]
        self._log(f"Melhor avaliação --> {'   '.join(out)}", 1)
        self._log(f"Tempo decorrido: {time_to_str(self.trials.iat[-1, -2])}", 1)

    @composed(crash, method_to_log, beartype)
    def fit(self, X: pd.DataFrame | None = None, y: Pandas | None = None, *, prefit: bool = False):
        """Ajusta e valida o modelo.

        O estimador é ajustado usando os melhores hiperparâmetros encontrados
        durante o ajuste de hiperparâmetros. Em seguida, o estimador é
        avaliado no conjunto de teste. Use este método apenas para reajustar o
        modelo após ter continuado o estudo.

        Parâmetros
        ----------
        X: pd.DataFrame or None
            Conjunto de features com shape=(n_amostras, n_features). Se None,
            `self.X_train` é usado.

        y: pd.Series, pd.DataFrame or None
            Coluna(s) alvo correspondente(s) a `X`. Se None, `self.y_train`
            é usado.

        prefit: bool, default=False
            Se o estimador já está ajustado. Se True, apenas avalia
            o modelo.

        """
        t_init = dt.now()

        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        self.clear()  # Reinicia o estado do modelo

        if self._study is None:
            self._log(f"Results for {self.fullname}:", 1)
        self._log(f"Fit {'-' * 45}", 1)

        # Assign estimator if not done already
        if self._estimator is None:
            self._check_est_params()
            self._estimator = self._get_est(self._est_params | self.best_params)

        if not prefit:
            self._estimator = self._fit_estimator(
                estimator=self.estimator,
                data=(X, y),
                validation=(self.X_test, self.y_test),
            )

        for ds in ("train", "test"):
            out = [f"{met.name}: {self._get_score(met, ds)}" for met in self._metric]
            self._log(f"T{ds[1:]} avaliação --> {'   '.join(out)}", 1)

        # Obtém a duração e imprime no log
        self._time_fit += (dt.now() - t_init).total_seconds()
        self._log(f"Tempo decorrido: {time_to_str(self._time_fit)}", 1)

        # Rastreia resultados no mlflow ================================== >>

        # Registra parâmetros, métricas, modelo e dados no mlflow
        if self.experiment:
            with mlflow.start_run(run_id=self.run.info.run_id):
                mlflow.set_tags(
                    {
                        "name": self.name,
                        "model": self.fullname,
                        "branch": self.branch.name,
                        **self._ht["tags"],
                    }
                )

                # Mlflow só aceita params com comprimento de char <250
                mlflow.log_params(
                    {k: v for k, v in self.estimator.get_params().items() if len(str(v)) <= 250}
                )

                # Salva evals para modelos com validação durante o treinamento
                if self.evals:
                    for key, value in self.evals.items():
                        for step in range(len(value)):
                            mlflow.log_metric(f"evals_{key}", value[step], step=step)

                # O restante das métricas é rastreado ao chamar _get_score
                mlflow.log_metric("time_fit", self._time_fit)

                mlflow.sklearn.log_model(
                    sk_model=self.estimator,
                    artifact_path=self._est_class.__name__,
                    signature=infer_signature(
                        model_input=self.X,
                        model_output=self.estimator.predict(self.X_test.iloc[[0]]),
                    ),
                    input_example=self.X.iloc[[0]],
                )

                if self.log_data:
                    for ds in ("train", "test"):
                        mlflow.log_input(dataset=from_pandas(getattr(self, ds)), context=ds)

                if self.log_pipeline:
                    mlflow.sklearn.log_model(
                        sk_model=self.export_pipeline(),
                        artifact_path=f"{self._est_class.__name__}_pipeline",
                        signature=infer_signature(
                            model_input=self.X,
                            model_output=self.estimator.predict(self.X_test.iloc[[0]]),
                        ),
                        input_example=self.X.iloc[[0]],
                    )

    @composed(crash, method_to_log, beartype)
    def bootstrapping(self, n_bootstrap: Int, *, reset: Bool = False):
        """Aplica um algoritmo de bootstrap.

        Extrai amostras bootstrap do conjunto de treino e as testa
        no conjunto de teste para obter uma distribuição dos resultados do modelo.

        Parâmetros
        ----------
        n_bootstrap: int
           Número de amostras bootstrap para ajustar.

        reset: bool, default=False
            Se deve iniciar uma nova execução ou continuar a existente.

        """
        t_init = dt.now()

        if self._bootstrap is None or reset:
            self._bootstrap = pd.DataFrame(columns=self._metric.keys())
            self._bootstrap.index.name = "sample"
            self._time_bootstrap = 0

        for i in range(n_bootstrap):
            # Cria amostras estratificadas com reposição
            sample_x, sample_y = resample(
                self.X_train,
                self.y_train,
                replace=True,
                random_state=i + (self.random_state or 0),
                stratify=self.y_train,
            )

            # Ajusta no conjunto bootstrap
            estimator = self._fit_estimator(self.estimator, data=(sample_x, sample_y))

            # Obtém pontuações no conjunto de teste
            scores = pd.DataFrame(
                {
                    m.name: [self._score_from_est(m, estimator, self.X_test, self.y_test)]
                    for m in self._metric
                }
            )

            self._bootstrap = pd.concat([self._bootstrap, scores], ignore_index=True)

        self._log(f"Bootstrap {'-' * 39}", 1)
        out = [
            f"{m.name}: {rnd(self.bootstrap.mean()[i])} \u00b1 {rnd(self.bootstrap.std()[i])}"
            for i, m in enumerate(self._metric)
        ]
        self._log(f"Avaliação --> {'   '.join(out)}", 1)

        self._time_bootstrap += (dt.now() - t_init).total_seconds()
        self._log(f"Tempo decorrido: {time_to_str(self._time_bootstrap)}", 1)

    # Propriedades Utilitárias =========================================== >>

    @property
    def name(self) -> str:
        """Nome do modelo.

        Use o `@setter` da propriedade para alterar o nome do modelo. O
        acrônimo sempre permanece no início do nome do modelo. Se
        o modelo estiver sendo rastreado pelo [mlflow][tracking], o nome da
        execução correspondente também muda.

        """
        return self._name

    @name.setter
    @beartype
    def name(self, value: str):
        """Altera o nome do modelo."""
        # Remove o acrônimo se fornecido pelo usuário
        if re.match(f"{self.acronym}_", value, re.I):
            value = value[len(self.acronym) + 1 :]

        # Adiciona o acrônimo na frente (com capitalização correta)
        self._name = f"{self.acronym}{f'_{value}' if value else ''}"

        if self.experiment:  # Muda o nome na execução do mlflow
            MlflowClient().set_tag(self.run.info.run_id, "mlflow.runName", self.name)

        self._log(f"Modelo {self.name} renomeado com sucesso para {self._name}.", 1)

    @cached_property
    def task(self) -> Task:
        """Tipo de [tarefa][task] do dataset."""
        return self._goal.infer_task(self.y)

    @property
    def og(self) -> Branch:
        """Branch contendo o dataset original.

        Este branch contém os dados antes de qualquer transformação.
        Redireciona para o branch atual se seu pipeline estiver vazio
        para não ter os mesmos dados na memória duas vezes.

        """
        return self._og

    @property
    def branch(self) -> Branch:
        """Branch ativo atual."""
        return self._branch

    @property
    def run(self) -> Run:
        """Execução mlflow correspondente a este modelo.

        Esta propriedade só está disponível para modelos com mlflow
        [tracking][] habilitado.

        """
        if self.experiment:
            return self._run

        raise AttributeError("Este modelo não possui experimento mlflow.")

    @property
    def study(self) -> Study:
        """Estudo Optuna usado para ajuste de hiperparâmetros.

        Esta propriedade só está disponível para modelos que executaram
        [ajuste de hiperparâmetros][hyperparameter tuning].

        """
        if self._study is not None:
            return self._study

        raise AttributeError("Este modelo não executou ajuste de hiperparâmetros.")

    @property
    def trials(self) -> pd.DataFrame:
        """Visão geral dos resultados dos trials.

        Esta propriedade só está disponível para modelos que executaram
        [ajuste de hiperparâmetros][hyperparameter tuning]. Todas as durações estão em segundos.
        As colunas incluem:

        - **[param_name]:** Valor do parâmetro usado neste trial.
        - **estimator:** Estimador usado neste trial.
        - **[metric_name]:** Pontuação da métrica do trial.
        - **[best_metric_name]:** Melhor pontuação até agora neste estudo.
        - **time_trial:** Duração do trial.
        - **time_ht:** Duração do ajuste de hiperparâmetros.
        - **state:** Estado do trial (COMPLETE, PRUNED, FAIL).

        """
        if self._study is not None:
            data: dict[str, list[Any]] = defaultdict(list)
            for t in self._study.trials:
                data["trial"].append(t.number)
                for p in t.params:
                    data[p].append(t.params[p])
                data["estimator"].append(t.user_attrs.get("estimator"))
                for i, met in enumerate(self._metric.keys()):
                    if len(self._metric) == 1:
                        data[met].append(t.value or np.nan)
                    else:
                        data[met].append(t.values[i] if t.values else np.nan)
                    data[f"best_{met}"] = np.nanmax(data[met])
                if t.datetime_complete and t.datetime_start:
                    data["time_trial"].append(
                        (t.datetime_complete - t.datetime_start).total_seconds()
                    )
                data["time_ht"].append(np.sum(data["time_trial"]))
                data["state"].append(t.state.name)

            return pd.DataFrame(data).set_index("trial")

        raise AttributeError("O modelo não executou ajuste de hiperparâmetros.")

    @property
    def best_trial(self) -> FrozenTrial:
        """Trial que retornou a maior pontuação.

        Para [execuções com múltiplas métricas][multi-metric runs], o melhor trial é aquele que
        obteve melhor desempenho na métrica principal. Use o `@setter` da propriedade
        para alterar o melhor trial. Veja [aqui][example-hyperparameter-tuning]
        um exemplo. Esta propriedade só está disponível para modelos que
        executaram [ajuste de hiperparâmetros][hyperparameter tuning].

        """
        if self._study is not None:
            if self._best_trial:
                return self._best_trial
            elif len(self._metric) == 1:
                return self.study.best_trial
            else:
                # Sort trials by the best score on the main metric
                return sorted(self.study.best_trials, key=lambda x: x.values[0])[0]

        raise AttributeError("The model didn't run hyperparameter tuning.")

    @best_trial.setter
    @beartype
    def best_trial(self, value: IntLargerEqualZero | None):
        """Atribui o melhor trial do estudo.

        Os hiperparâmetros selecionados neste trial são usados para treinar
        o estimador. Esta propriedade só está disponível para modelos que
        executaram [ajuste de hiperparâmetros][hyperparameter tuning].

        """
        if value is None:
            self._best_trial = None  # Reinicia a seleção do melhor trial
        elif value not in self.trials.index:
            raise ValueError(
                "Valor inválido para best_trial. O "
                f"valor deve ser um número de trial, recebido {value}."
            )
        else:
            self._best_trial = self.study.trials[value]

    @property
    def best_params(self) -> dict[str, Any]:
        """Parâmetros do estimador no [melhor trial][self-best_trial].

        Esta propriedade só está disponível para modelos que executaram
        [ajuste de hiperparâmetros][hyperparameter tuning].

        """
        if self._study is not None:
            return dict(self._trial_to_est(self.best_trial.params))
        else:
            return {}

    @property
    def estimator(self) -> Predictor:
        """Estimador ajustado no conjunto de treino."""
        if self._estimator is not None:
            return self._estimator

        raise AttributeError("Este modelo ainda não foi ajustado.")

    @property
    def evals(self) -> dict[str, list[Float]]:
        """Pontuações obtidas por iteração do treinamento.

        Apenas as pontuações da [métrica principal][metric] são rastreadas.
        As chaves incluídas são: train e test. Esta propriedade só
        está disponível para modelos com [validação-durante-o-treinamento][in-training-validation].

        """
        return self._evals

    @property
    def bootstrap(self) -> pd.DataFrame:
        """Visão geral das pontuações de bootstrapping.

        O dataframe tem shape=(n_bootstrap, metric) e mostra a
        pontuação obtida por cada amostra bootstrap para cada métrica.
        Usar `experionml.bootstrap.mean()` retorna os mesmos valores que
        `[metric]_bootstrap`. Esta propriedade só está disponível para
        modelos que executaram [bootstrapping][].

        """
        if self._bootstrap is not None:
            return self._bootstrap

        raise AttributeError("Este modelo não executou bootstrapping.")

    @property
    def results(self) -> pd.Series:
        """Visão geral dos resultados do modelo.

        Todas as durações estão em segundos. Os possíveis valores incluem:

        - **[metric]_ht:** Pontuação obtida pelo ajuste de hiperparâmetros.
        - **time_ht:** Duração do ajuste de hiperparâmetros.
        - **[metric]_train:** Pontuação da métrica no conjunto de treino.
        - **[metric]_test:** Pontuação da métrica no conjunto de teste.
        - **time_fit:** Duração do ajuste do modelo no conjunto de treino.
        - **[metric]_bootstrap:** Pontuação média nas amostras bootstrap.
        - **time_bootstrap:** Duração do bootstrapping.
        - **time:** Duração total da execução.

        """
        data = {}
        if self._study is not None:
            for met in self._metric.keys():
                data[f"{met}_ht"] = self.trials.loc[self.best_trial.number, met]
            data["time_ht"] = self.trials.iloc[-1, -2]
        for met in self._metric:
            for ds in ["train", "test"] + ([] if self.holdout is None else ["holdout"]):
                data[f"{met.name}_{ds}"] = self._get_score(met, ds)
        data["time_fit"] = self._time_fit
        if self._bootstrap is not None:
            for met in self._metric.keys():
                data[f"{met}_bootstrap"] = self.bootstrap[met].mean()
            data["time_bootstrap"] = self._time_bootstrap
        data["time"] = data.get("time_ht", 0) + self._time_fit + self._time_bootstrap
        return pd.Series(data, name=self.name)

    @property
    def feature_importance(self) -> pd.Series:
        """Pontuações de importância de features normalizadas.

        A soma das importâncias de todas as features é 1. As pontuações são
        extraídas dos atributos `scores_`, `coef_` ou
        `feature_importances_` do estimador, verificados nessa ordem.
        Esta propriedade só está disponível para estimadores com pelo menos
        um desses atributos.

        """
        data: np.ndarray | None = None

        for attr in ("scores_", "coef_", "feature_importances_"):
            if hasattr(self.estimator, attr):
                data = getattr(self.estimator, attr)

        # Obtém o valor médio para meta-estimadores
        if data is None and hasattr(self.estimator, "estimators_"):
            estimators = self.estimator.estimators_
            if all(hasattr(x, "feature_importances_") for x in estimators):
                data_l = [fi.feature_importances_ for fi in estimators]
            elif all(hasattr(x, "coef_") for x in estimators):
                data_l = [
                    np.linalg.norm(fi.coef_, axis=np.argmin(fi.coef_.shape), ord=1)
                    for fi in estimators
                ]
            else:
                # Para ensembles que misturam atributos
                raise ValueError(
                    "Falha ao calcular a importância de features para o meta-estimador "
                    f"{self._est_class.__name__}. Os estimadores subjacentes possuem "
                    "uma mistura de atributos feature_importances_ e coef_."
                )

            # Trunca cada coef ao número de features no 1º estimador
            # ClassifierChain adiciona features aos estimadores subsequentes
            min_length = np.min([len(c) for c in data_l])
            data = np.mean([c[:min_length] for c in data_l], axis=0)

        if data is not None:
            return pd.Series(
                data=(data_c := np.abs(data.flatten())) / data_c.sum(),
                index=self.features,
                name="feature_importance",
                dtype=float,
            ).sort_values(ascending=False)
        else:
            raise AttributeError(
                "Falha ao calcular a importância de features para o estimador "
                f"{self._est_class.__name__}. O estimador não possui o"
                f"atributo scores_, coef_ nem feature_importances_."
            )

    # Propriedades de Dados ============================================== >>

    @property
    def pipeline(self) -> Pipeline:
        """Pipeline de transformadores.

        Modelos que usaram [escalonamento automático de features][] têm o escalonador
        adicionado.

        !!! tip
            Use o método [plot_pipeline][] para visualizar o pipeline.

        """
        if self.scaler:
            return Pipeline(
                steps=[*self.branch.pipeline.steps, ("AutomatedScaler", self.scaler)],
                memory=self.memory,
            )
        else:
            return self.branch.pipeline

    @property
    def dataset(self) -> pd.DataFrame:
        """Conjunto de dados completo."""
        return merge(self.X, self.y)

    @property
    def train(self) -> pd.DataFrame:
        """Conjunto de treino."""
        return merge(self.X_train, self.y_train)

    @property
    def test(self) -> pd.DataFrame:
        """Conjunto de teste."""
        return merge(self.X_test, self.y_test)

    @property
    def holdout(self) -> pd.DataFrame | None:
        """Conjunto holdout."""
        if (holdout := self.branch.holdout) is not None:
            if self.scaler:
                return merge(self.scaler.transform(holdout[self.features]), holdout[self.target])
            else:
                return holdout
        else:
            return None

    @property
    def X(self) -> pd.DataFrame:
        """Conjunto de features."""
        return pd.concat([self.X_train, self.X_test])

    @property
    def y(self) -> Pandas:
        """Coluna(s) alvo."""
        return pd.concat([self.y_train, self.y_test])

    @property
    def X_train(self) -> pd.DataFrame:
        """Features do conjunto de treino."""
        exclude = self.branch.features.isin(self._config.ignore)
        X_train = self.branch.X_train.iloc[-self._train_idx :, ~exclude]
        if self.scaler:
            return cast(pd.DataFrame, self.scaler.transform(X_train))
        else:
            return X_train

    @property
    def y_train(self) -> Pandas:
        """Coluna alvo do conjunto de treino."""
        return self.branch.y_train[-self._train_idx :]

    @property
    def X_test(self) -> pd.DataFrame:
        """Features do conjunto de teste."""
        exclude = self.branch.features.isin(self._config.ignore)
        X_test = self.branch.X_test.iloc[:, ~exclude]
        if self.scaler:
            return cast(pd.DataFrame, self.scaler.transform(X_test))
        else:
            return X_test

    @property
    def X_holdout(self) -> pd.DataFrame | None:
        """Features do conjunto holdout."""
        if self.holdout is not None:
            return self.holdout[self.features]
        else:
            return None

    @property
    def y_holdout(self) -> Pandas | None:
        """Coluna alvo do conjunto holdout."""
        if self.holdout is not None:
            return self.holdout[self.branch.target]
        else:
            return None

    @property
    def shape(self) -> tuple[Int, Int]:
        """Shape do dataset (n_linhas, n_colunas)."""
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
        return self.columns[: -self.branch._data.n_targets]

    @property
    def n_features(self) -> int:
        """Número de features."""
        return len(self.features)

    @property
    def _all(self) -> pd.DataFrame:
        """Dataset + holdout.

        Observe que chamar esta propriedade dispara o cálculo do conjunto holdout.

        """
        return pd.concat([self.dataset, self.holdout])

    # Métodos Utilitários ============================================== >>

    @available_if(has_task("classification"))
    @composed(crash, method_to_log)
    def calibrate(
        self,
        method: Literal["sigmoid", "isotonic"] = "sigmoid",
        *,
        train_on_test: bool = False,
    ):
        """Calibrar e retreinar o modelo.

        Usa o [CalibratedClassifierCV][] do sklearn para aplicar
        calibração de probabilidade no modelo. O novo classificador substitui o
        atributo `estimator`. Se houver um experimento mlflow ativo,
        uma nova execução é iniciada com o nome `[model_name]_calibrate`.
        Como o estimador foi alterado, o modelo é [limpo][self-clear].
        Somente para classificadores.

        !!! note
            Por padrão, a calibração é otimizada usando o conjunto de
            treinamento (que já é usado para o treinamento inicial). Esta
            abordagem está sujeita a overfitting indesejado. É preferível
            usar `train_on_test=True`, que usa o conjunto de teste para
            calibração, mas somente se houver outro conjunto independente
            para teste ([conjunto holdout][data-sets]).

        Parâmetros
        ----------
        method: str, default="sigmoid"
            O método a ser usado para calibração. Escolha entre:

            - "sigmoid": Corresponde ao [método de Platt][plattsmethod]
              (i.e., um modelo de regressão logística).
            - "isotonic": Abordagem não-paramétrica. Não é recomendado
              usar este método de calibração com poucos exemplos (<1000)
              pois tende a sofrer overfitting.

        train_on_test: bool, default=False
            Se deve treinar o calibrador no conjunto de teste.

        """
        self._estimator = CalibratedClassifierCV(
            estimator=self.estimator,
            method=method,
            cv="prefit",
            n_jobs=self.n_jobs,
        )

        # Atribui uma execução mlflow ao novo estimador
        if self.experiment:
            self._run = mlflow.start_run(run_name=f"{self.name}_calibrate")
            mlflow.end_run()

        if not train_on_test:
            self.fit(self.X_train, self.y_train)
        else:
            self.fit(self.X_test, self.y_test)

    @composed(crash, method_to_log)
    def clear(self):
        """Redefinir atributos e limpar o cache do modelo.

        Redefine certos atributos do modelo para seu estado inicial, excluindo
        potencialmente grandes arrays de dados. Use este método para liberar
        memória antes de [salvar][experionmlclassifier-save] a instância. Os
        atributos afetados são:

        - Pontuações de [validação em treinamento][In-training validation]
        - [Previsões em cache][predicting].
        - [Valores Shap][shap]
        - [Instância do App][self-create_app]
        - [Instância do Dashboard][self-create_dashboard]
        - [Conjuntos de dados holdout][data-sets] calculados

        """
        self._evals = defaultdict(list)
        self._get_pred.clear_cache()
        self._shap_explanation = None
        self.__dict__.pop("app", None)
        self.__dict__.pop("dashboard", None)
        self.branch.__dict__.pop("holdout", None)

    @composed(crash, method_to_log)
    def create_app(self, **kwargs):
        """Criar um aplicativo interativo para testar previsões do modelo.

        Demonstre seu modelo de machine learning com uma interface web amigável.
        Este aplicativo é iniciado diretamente no notebook ou em uma página
        de navegador externo. A instância [Interface][] criada pode ser acessada
        através do atributo `app`.

        Parâmetros
        ----------
        **kwargs
            Argumentos de palavra-chave adicionais para a instância [Interface][]
            ou para o método [Interface.launch][launch].

        """

        def inference(*X) -> Scalar | str | list[Scalar | str]:
            """Aplicar inferência na linha fornecida pelo aplicativo.

            Parâmetros
            ----------
            *X
                Features fornecidas pelo usuário no aplicativo.

            Retorna
            -------
            int, float, str or list
                Rótulo original ou lista de rótulos para tarefas multioutput.

            """
            conv = lambda elem: elem.item() if hasattr(elem, "item") else elem

            with adjust(self, transform="pandas"):
                y_pred = self.predict([X])

            if isinstance(y_pred, pd.DataFrame):
                return [conv(elem) for elem in y_pred.iloc[0, :]]
            else:
                return conv(y_pred[0])

        check_dependency("gradio")
        from gradio import Interface
        from gradio.components import Dropdown, Textbox

        self._log("Iniciando aplicativo...", 1)

        inputs = []
        for name, column in self.og.X.items():
            if column.dtype.kind in "ifu":
                inputs.append(Textbox(label=name))
            else:
                inputs.append(Dropdown(list(column.unique()), label=name))

        self.app = Interface(
            fn=inference,
            inputs=inputs,
            outputs=["label"] * self.branch._data.n_targets,
            allow_flagging=kwargs.pop("allow_flagging", "never"),
            **{k: v for k, v in kwargs.items() if k in sign(Interface)},
        )

        self.app.launch(**{k: v for k, v in kwargs.items() if k in sign(Interface.launch)})

    @available_if(has_task("!multioutput"))
    @composed(crash, method_to_log, beartype)
    def create_dashboard(
        self,
        rows: RowSelector = "test",
        *,
        filename: str | Path | None = None,
        **kwargs,
    ):
        """Criar um dashboard interativo para analisar o modelo.

        O ExperionML usa o pacote [explainerdashboard][explainerdashboard_package]
        para fornecer uma maneira rápida e fácil de analisar e explicar
        as previsões e o funcionamento do modelo. O dashboard permite
        investigar valores SHAP, importâncias de permutação,
        efeitos de interação, gráficos de dependência parcial, todos os tipos de
        gráficos de desempenho, e até árvores de decisão individuais.

        Por padrão, o dashboard é renderizado em uma nova aba no seu navegador
        padrão, mas se preferir, você pode renderizá-lo dentro do
        notebook usando o parâmetro `mode="inline"`. A instância
        [ExplainerDashboard][] criada pode ser acessada através do
        atributo `dashboard`. Este método não está disponível para
        [tarefas multioutput][multioutput tasks].

        !!! note
            Os gráficos exibidos pelo dashboard não são criados pelo ExperionML e
            podem diferir dos obtidos através deste pacote.

        Parâmetros
        ----------
        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] para obter o
            relatório.

        filename: str, Path or None, default=None
            Nome do arquivo ou [pathlib.Path][] para salvar. None para não
            salvar nada.

        **kwargs
            Argumentos de palavra-chave adicionais para a instância
            [ExplainerDashboard][].

        """
        check_dependency("explainerdashboard")
        from explainerdashboard import ClassifierExplainer, ExplainerDashboard, RegressionExplainer

        self._log("Criando dashboard...", 1)

        Xt, yt = self.branch._get_rows(rows, return_X_y=True)

        if self.scaler:
            Xt = cast(pd.DataFrame, self.scaler.transform(Xt))

        # Obtém valores shap do objeto ShapExplanation interno
        exp = self._shap.get_explanation(Xt, target=(0,))

        # Explainerdashboard requer todas as classes alvo
        if self.task.is_classification:
            if self.task.is_binary:
                if exp.values.shape[-1] != 2:
                    exp.base_values = [np.array(1 - exp.base_values), exp.base_values]
                    exp.values = [np.array(1 - exp.values), exp.values]
            else:
                # Explainer espera uma lista de np.array com valores shap para cada classe
                exp.values = list(np.moveaxis(exp.values, -1, 0))

        params = {"permutation_metric": self._metric, "n_jobs": self.n_jobs}
        if self.task.is_classification:
            explainer = ClassifierExplainer(self.estimator, Xt, yt, **params)
        else:
            explainer = RegressionExplainer(self.estimator, Xt, yt, **params)

        explainer.set_shap_values(exp.base_values, exp.values)

        self.dashboard = ExplainerDashboard(
            explainer=explainer,
            mode=kwargs.pop("mode", "external"),
            **kwargs,
        )
        self.dashboard.run()

        if filename:
            if (path := Path(filename)).suffix != ".html":
                path = path.with_suffix(".html")

            self.dashboard.save_html(path)
            self._log("Dashboard salvo com sucesso.", 1)

    @composed(crash, method_to_log)
    def cross_validate(self, *, include_holdout: Bool = False, **kwargs) -> Styler:
        """Avaliar o modelo usando validação cruzada.

        Este método aplica validação cruzada em todo o pipeline no conjunto de
        dados completo. Use-o para avaliar a robustez do desempenho do modelo.
        Se o método de pontuação não for especificado em `kwargs`,
        usa a métrica do experionml. Os resultados da validação cruzada são
        armazenados no atributo `cv` do modelo.

        !!! tip
            Este método retorna um objeto [Styler][] do pandas. Converta
            o resultado de volta para um dataframe regular usando seu atributo `data`.

        Parâmetros
        ----------
        include_holdout: bool, default=False
            Se deve incluir o conjunto holdout (se disponível) na
            validação cruzada.

        **kwargs
            Argumentos de palavra-chave adicionais para uma destas funções.

            - Para tarefas de previsão: [evaluate][sktimeevaluate].
            - Caso contrário: [cross_validate][sklearncrossvalidate].

        Retorna
        -------
        [Styler][]
            Visão geral dos resultados.

        """
        self._log("Aplicando validação cruzada...", 1)

        if not include_holdout:
            X = self.og.X
            y = self.og.y
        else:
            X = self.og._all[self.og.features]
            y = self.og._all[self.og.target]

        # Atribui pontuação do experionml se não especificada
        if kwargs.get("scoring"):
            scorer = get_custom_scorer(kwargs.pop("scoring"), pos_label=self._config.pos_label)
            scoring = {scorer.name: scorer}
        else:
            scoring = dict(self._metric)

        if self.task.is_forecast:
            self.cv = evaluate(
                forecaster=self.export_pipeline(),
                y=y,
                X=X,
                scoring=[
                    make_forecasting_scorer(
                        func=metric._score_func,
                        name=name,
                        greater_is_better=metric._sign == 1,
                    )
                    for name, metric in scoring.items()
                ],
                cv=self._get_cv(kwargs.pop("cv", 5), max_length=len(X)),
                backend=kwargs.pop("backend", self.backend if self.backend != "ray" else None),
                backend_params=kwargs.pop("backend_params", {"n_jobs": self.n_jobs}),
                return_data=True,  # Required for plot_cv_splits
            )
        else:
            if "groups" in kwargs:
                raise ValueError(
                    "O parâmetro groups não pode ser passado diretamente para cross_validate. "
                    "O ExperionML usa roteamento de metadados para gerenciar grupos de dados. Passe os grupos para "
                    "o parâmetro 'metadata' do experionml no construtor."
                )

            # Monkey patch da função _fit_and_score do sklearn para permitir
            # pipelines que descartam amostras durante a transformação
            with patch("sklearn.model_selection._validation._fit_and_score", fit_and_score):
                self.cv = cross_validate(
                    estimator=self.export_pipeline(),
                    X=X,
                    y=y,
                    scoring=scoring,
                    cv=self._get_cv(kwargs.pop("cv", 5), max_length=len(X)),
                    params=self._config.get_metadata(X),
                    return_train_score=kwargs.pop("return_train_score", True),
                    error_score=kwargs.pop("error_score", "raise"),
                    n_jobs=kwargs.pop("n_jobs", self.n_jobs),
                    return_indices=True,  # Required for plot_cv_splits
                    **kwargs,
                )

        df = pd.DataFrame()
        for m in scoring:
            if f"train_{m}" in self.cv:
                df[f"train_{m}"] = self.cv[f"train_{m}"]
            if f"test_{m}" in self.cv:
                df[f"test_{m}"] = self.cv[f"test_{m}"]
        df["time"] = self.cv["fit_time"]
        df.loc["mean"] = df.mean()

        return df.style.highlight_max(
            props=COLOR_SCHEME, subset=[c for c in df if not c.startswith("time")]
        ).highlight_min(props=COLOR_SCHEME, subset=[c for c in df if c.startswith("time")])

    @composed(crash, beartype)
    def evaluate(self, metric: MetricConstructor = None, rows: RowSelector = "test") -> pd.Series:
        """Obter as pontuações do modelo para as métricas fornecidas.

        !!! tip
            Use o método [get_best_threshold][self-get_best_threshold] ou
            [plot_threshold][] para determinar um threshold adequado
            para um classificador binário.

        Parâmetros
        ----------
        metric: str, func, scorer, sequence or None, default=None
            Métricas a calcular. Se None, uma seleção das métricas mais
            comuns por tarefa é usada.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] para calcular
            a métrica.

        Retorna
        -------
        pd.Series
            Pontuações do modelo.

        """
        # Métricas predefinidas para exibir
        if metric is None:
            if self.task.is_classification:
                if self.task is Task.binary_classification:
                    metric = [
                        "accuracy",
                        "ap",
                        "ba",
                        "f1",
                        "jaccard",
                        "mcc",
                        "precision",
                        "recall",
                        "auc",
                    ]
                elif self.task.is_multiclass:
                    metric = [
                        "ba",
                        "f1_weighted",
                        "jaccard_weighted",
                        "mcc",
                        "precision_weighted",
                        "recall_weighted",
                    ]
                elif self.task is Task.multilabel_classification:
                    metric = [
                        "accuracy",
                        "ap",
                        "f1_weighted",
                        "jaccard_weighted",
                        "precision_weighted",
                        "recall_weighted",
                        "auc",
                    ]
            else:
                # Sem msle pois falha para valores negativos
                metric = ["mae", "mape", "mse", "r2", "rmse"]

        scores = pd.Series(name=self.name, dtype=float)
        for met in lst(metric):
            scorer = get_custom_scorer(met, pos_label=self._config.pos_label)
            scores[scorer.name] = self._get_score(scorer, rows=rows)

        return scores

    @crash
    def export_pipeline(self) -> Pipeline:
        """Exportar o pipeline de transformadores com o estimador final.

        O pipeline retornado já está ajustado no conjunto de treinamento.
        Note que se o modelo usou [escalonamento automático de features][],
        o [Scaler][] é adicionado ao pipeline.

        Retorna
        -------
        [Pipeline][]
            Branch atual como um objeto Pipeline similar ao sklearn.

        """
        pipeline = deepcopy(self.pipeline)
        pipeline.steps.append((self._est_class.__name__, deepcopy(self.estimator)))
        return pipeline

    @composed(crash, method_to_log, beartype)
    def full_train(self, *, include_holdout: Bool = False):
        """Treinar o estimador no conjunto de dados completo.

        Em alguns casos, pode ser desejável usar todos os dados disponíveis
        para treinar um modelo final. Note que fazer isso significa que o
        estimador não pode mais ser avaliado no conjunto de teste. O
        estimador retreinado substituirá o atributo `estimator`. Se
        houver um experimento mlflow ativo, uma nova execução é iniciada
        com o nome `[model_name]_full_train`. Como o estimador foi
        alterado, o modelo é limpo.

        !!! warning
            Embora o modelo seja treinado no conjunto de dados completo, o
            pipeline não é. Para obter um pipeline totalmente treinado, use:
            `pipeline = experionml.export_pipeline().fit(experionml.X, experionml.y)`.

        Parâmetros
        ----------
        include_holdout: bool, default=False
            Se deve incluir o conjunto holdout (se disponível) no
            treinamento do estimador. É desencorajado usar esta
            opção pois significa que o modelo não pode mais ser avaliado
            em nenhum conjunto.

        """
        if include_holdout and self.holdout is None:
            raise ValueError("Nenhum conjunto holdout disponível.")

        if not include_holdout:
            X, y = self.X, self.y
        else:
            X = pd.concat([self.X, self.X_holdout])
            y = pd.concat([self.y, self.y_holdout])

        # Atribui uma execução mlflow ao novo estimador
        if self.experiment:
            self._run = mlflow.start_run(run_name=f"{self.name}_full_train")
            mlflow.end_run()

        self.fit(X, y)

    @available_if(lambda self: self.task is Task.binary_classification)
    @composed(crash, beartype)
    def get_best_threshold(
        self,
        metric: int | str | None = None,
        *,
        train_on_test: bool = False,
    ) -> Float:
        """Obter o threshold que maximiza uma métrica.

        Usa o [TunedThresholdClassifierCV][] do sklearn para ajustar
        posteriormente o threshold de decisão (ponto de corte) usado para
        converter estimativas de probabilidade posterior (i.e., saída de
        `predict_proba`) ou pontuações de decisão (i.e., saída de
        `decision_function`) em um rótulo de classe. O ajuste é feito
        otimizando uma das métricas do experionml. O estimador ajustado é
        armazenado no atributo `tuned_threshold`. Disponível apenas para
        classificadores binários.

        !!! note
            Por padrão, o threshold é otimizado usando o conjunto de
            treinamento (que já é usado para o treinamento inicial). Esta
            abordagem está sujeita a overfitting indesejado. É preferível
            usar `train_on_test=True`, que usa o conjunto de teste para
            ajuste, mas somente se houver outro conjunto independente para
            teste ([conjunto holdout][data-sets]).

        !!! tip
            Use o método [plot_threshold][] para visualizar o efeito
            de diferentes thresholds em uma métrica.

        Parâmetros
        ----------
        metric: int, str or None, default=None
            Métrica para otimizar. Se None, a métrica principal é usada.

        train_on_test: bool, default=False
            Se deve treinar o calibrador no conjunto de teste.

        Retorna
        -------
        float
            Valor de threshold otimizado.

        """
        if metric is None:
            scorer = self._metric[0]
        else:
            scorer = self._metric[self._get_metric(metric)[0]]

        self.tuned_threshold = TunedThresholdClassifierCV(
            estimator=self.estimator,
            scoring=scorer,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        if not train_on_test:
            self.tuned_threshold.fit(self.X_train, self.y_train)
        else:
            self.tuned_threshold.fit(self.X_test, self.y_test)

        return self.tuned_threshold.best_threshold_

    @composed(crash, method_to_log, beartype)
    def inverse_transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        *,
        verbose: Verbose | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Transformar inversamente novos dados pelo pipeline.

        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. Os demais devem implementar o método `inverse_transform`.
        Se apenas `X` ou apenas `y` for fornecido, ignora transformadores
        que exigem o outro parâmetro. Isso pode ser útil para, por exemplo,
        transformar inversamente apenas a coluna alvo. Se chamado a partir
        de um modelo que usou escalonamento automático de features, o
        escalonamento também é invertido.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features transformado com shape=(n_samples, n_features).
            Se None, `X` é ignorado nos transformadores.

        y: int, str, sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

            - If None: `y` é ignorado.
            - If int: Posição da coluna alvo em `X`.
            - If str: Nome da coluna alvo em `X`.
            - If sequence: Coluna alvo com shape=(n_samples,) ou
              sequência de nomes ou posições de colunas para tarefas multioutput.
            - If dataframe-like: Colunas alvo para tarefas multioutput.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Conjunto de features original. Retornado apenas se fornecido.

        series or dataframe
            Coluna alvo original. Retornada apenas se fornecida.

        """
        Xt, yt = self._check_input(X, y, columns=self.branch.features, name=self.branch.target)

        with adjust(self.pipeline, transform=self.engine.data, verbose=verbose) as pl:
            return pl.inverse_transform(Xt, yt)

    @composed(crash, method_to_log, beartype)
    def register(
        self,
        name: str | None = None,
        stage: Stages = "None",
        *,
        archive_existing_versions: Bool = False,
    ):
        """Registrar o modelo no [registro de modelos do mlflow][registry].

        Este método só está disponível quando o [rastreamento][tracking] do
        modelo está habilitado usando um dos seguintes esquemas de URI:
        databricks, http, https, postgresql, mysql, sqlite, mssql.

        Parâmetros
        ----------
        name: str or None, default=None
            Nome para o modelo registrado. Se None, o nome completo do modelo
            é usado. Se o nome do modelo já existir, uma nova versão é criada.

        stage: str, default="None"
            Novo estágio desejado para o modelo.

        archive_existing_versions: bool, default=False
            Se todas as versões existentes do modelo no `stage` serão movidas
            para o estágio "Archived". Válido apenas quando `stage` é
            "Staging" ou "Production", caso contrário um erro será lançado.

        """
        if not self.experiment:
            raise PermissionError(
                "O método register só está disponível quando " "há um experimento mlflow ativo."
            )

        model = mlflow.register_model(
            model_uri=f"runs:/{self.run.info.run_id}/{self._est_class.__name__}",
            name=name or self.fullname,
            tags=self._ht["tags"] or None,
        )

        MlflowClient().transition_model_version_stage(
            name=model.name,
            version=model.version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    @composed(crash, method_to_log, beartype)
    def save_estimator(self, filename: str | Path = "auto"):
        """Salvar o estimador em um arquivo pickle.

        Parâmetros
        ----------
        filename: str or Path, default="auto"
            Nome do arquivo ou [pathlib.Path][] para salvar. Use
            "auto" para nomeação automática.

        """
        if (path := Path(filename)).suffix != ".pkl":
            path = path.with_suffix(".pkl")

        if path.name == "auto.pkl":
            path = path.with_name(f"{self.estimator.__class__.__name__}.pkl")

        with open(path, "wb") as f:
            pickle.dump(self.estimator, f)

        self._log(f"Estimador {self.fullname} salvo com sucesso.", 1)

    @composed(crash, method_to_log, beartype)
    def serve(self, method: str = "predict"):
        """Servir o modelo como endpoint REST para inferência.

        O pipeline completo é servido junto com o modelo. Os dados de
        inferência devem ser fornecidos como json na requisição HTTP, ex.:
        `requests.get("http://127.0.0.1:8000/", json=X.to_json())`.
        O deployment é feito em um cluster [ray][]. Os parâmetros padrão
        `host` e `port` fazem o deploy em localhost.

        !!! tip
            Use `import ray; ray.serve.shutdown()` para encerrar o
            endpoint ao terminar.

        Parâmetros
        ----------
        method: str, default="predict"
            Método do estimador para realizar a inferência.

        """
        check_dependency("ray")
        import ray
        from ray.serve import deployment, run

        @deployment
        class ServeModel:
            """Classe de deployment do modelo.

            Parâmetros
            ----------
            pipeline: Pipeline
                Transformadores + estimador para realizar inferência.

            method: str, default="predict"
                Método do estimador para realizar a inferência.

            """

            def __init__(self, pipeline: Pipeline, method: str = "predict"):
                self.pipeline = pipeline
                self.method = method

            async def __call__(self, request: Request) -> np.ndarray:
                """Chamada de inferência.

                Parâmetros
                ----------
                request: Request.
                    Requisição HTTP. Deve conter as linhas a prever
                    em um corpo json.

                Retorna
                -------
                np.ndarray
                    Previsões do modelo.

                """
                payload = await request.json()
                return getattr(self.pipeline, self.method)(pd.read_json(payload))

        if not ray.is_initialized():
            ray.init(log_to_driver=False)

        run(ServeModel.bind(pipeline=self.export_pipeline(), method=method))

        self._log(f"Servindo modelo {self.fullname}...", 1)

    @available_if(lambda self: self.task is Task.binary_classification)
    @composed(crash, method_to_log)
    def set_threshold(self, threshold: Float):
        """Definir o threshold binário do estimador.

        Um novo classificador com o novo threshold substitui o atributo
        `estimator`. Se houver um experimento mlflow ativo, uma nova
        execução é iniciada com o nome `[model_name]_threshold_X`. Como
        o estimador foi alterado, o modelo é [limpo][self-clear]. Apenas
        para classificadores binários.

        !!! tip
            Use o método [get_best_threshold][self-get_best_threshold]
            para encontrar o threshold ideal para uma métrica específica.

        Parâmetros
        ----------
        threshold: float
            Threshold binário para classificar a classe positiva.

        """
        if not 0 < threshold < 1:
            raise ValueError(
                "Valor inválido para o parâmetro threshold. O valor "
                f"deve estar entre 0 e 1, obtido {threshold}."
            )

        self._estimator = FixedThresholdClassifier(
            estimator=(est := deepcopy(self.estimator)),
            threshold=threshold,
            pos_label=self._config.pos_label,
        )

        # Adiciona o estimador ajustado manualmente para evitar retreinamento
        self._estimator.estimator_ = est  # type: ignore[union-attr]

        # Atribui uma execução mlflow ao novo estimador
        if self.experiment:
            self._run = mlflow.start_run(run_name=f"{self.name}_threshold_{threshold}")
            mlflow.end_run()

        self.fit(self.X_train, self.y_train, prefit=True)

    @composed(crash, method_to_log, beartype)
    def transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        *,
        verbose: Verbose | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Transformar novos dados pelo pipeline.

        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. Se apenas `X` ou apenas `y` for fornecido, ignora
        transformadores que exigem o outro parâmetro. Isso pode ser útil
        para, por exemplo, transformar apenas a coluna alvo. Se chamado
        a partir de um modelo que usou escalonamento automático de
        features, os dados também são escalados.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Se None,
            `X` é ignorado nos transformadores.

        y: int, str, sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

            - If None: `y` é ignorado.
            - If int: Posição da coluna alvo em `X`.
            - If str: Nome da coluna alvo em `X`.
            - If sequence: Coluna alvo com shape=(n_samples,) ou
              sequência de nomes ou posições de colunas para tarefas multioutput.
            - If dataframe-like: Colunas alvo para tarefas multioutput.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Conjunto de features transformado. Retornado apenas se fornecido.

        series or dataframe
            Coluna alvo transformada. Retornada apenas se fornecida.

        """
        Xt, yt = self._check_input(X, y, columns=self.og.features, name=self.og.target)

        with adjust(self.pipeline, transform=self.engine.data, verbose=verbose) as pl:
            return pl.transform(Xt, yt)


class ClassRegModel(BaseModel):
    """Modelos de classificação e regressão."""

    @crash
    def get_tags(self) -> dict[str, Any]:
        """Obter as tags do modelo.

        Retorna parâmetros da classe que fornecem informações gerais sobre
        as características do modelo.

        Retorna
        -------
        dict
            Tags do modelo.

        """
        return {
            "acronym": self.acronym,
            "fullname": self.fullname,
            "estimator": self._est_class.__name__,
            "module": self._est_class.__module__,
            "handles_missing": getattr(self, "handles_missing", None),
            "needs_scaling": self.needs_scaling,
            "accepts_sparse": getattr(self, "accepts_sparse", None),
            "native_multilabel": self.native_multilabel,
            "native_multioutput": self.native_multioutput,
            "validation": self.validation,
            "supports_engines": ", ".join(getattr(self, "supports_engines", [])),
        }

    @overload
    def _prediction(
        self,
        X: RowSelector | XSelector,
        y: YSelector | None = ...,
        metric: str | MetricFunction | Scorer | None = ...,
        sample_weight: Sequence[Scalar] | None = ...,
        verbose: Verbose | None = ...,
        method: Literal[
            "decision_function",
            "predict",
            "predict_log_proba",
            "predict_proba",
        ] = ...,
    ) -> Pandas: ...

    @overload
    def _prediction(
        self,
        X: RowSelector | XSelector,
        y: YSelector | None,
        metric: str | MetricFunction | Scorer | None,
        sample_weight: Sequence[Scalar] | None,
        verbose: Verbose | None,
        method: Literal["score"],
    ) -> Float: ...

    def _prediction(
        self,
        X: RowSelector | XSelector,
        y: YSelector | None = None,
        metric: str | MetricFunction | Scorer | None = None,
        sample_weight: Sequence[Scalar] | None = None,
        verbose: Verbose | None = None,
        method: PredictionMethods = "predict",
    ) -> Float | Pandas:
        """Obtém previsões em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O modelo deve implementar o método fornecido.

        Parâmetros
        ----------
        X: hashable, segment, sequence or dataframe-like
            [Seleção de linhas][row-and-column-selection] ou conjunto de
            features com shape=(n_samples, n_features) para fazer previsões.

        y: int, str, sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

            - Se None: `y` é ignorado.
            - Se int: Posição da coluna alvo em `X`.
            - Se str: Nome da coluna alvo em `X`.
            - Se sequence: Coluna alvo com shape=(n_samples,) ou
              sequência de nomes ou posições de colunas para tarefas
              multioutput.
            - Se dataframe: Colunas alvo para tarefas multioutput.

        metric: str, func, scorer or None, default=None
            Métrica a calcular. Escolha entre qualquer um dos scorers do sklearn,
            uma função com assinatura metric(y_true, y_pred) ou um objeto scorer.
            Se None, retorna acurácia média para tarefas de classificação
            e r2 para tarefas de regressão. Somente para method="score".

        sample_weight: sequence or None, default=None
            Pesos das amostras para o método `score`.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        method: str, default="predict"
            Método de previsão a ser aplicado ao estimador.

        Retorna
        -------
        float, series or dataframe
            Previsões calculadas. O tipo de retorno depende do método
            chamado.

        """

        def get_transform_X_y(
            X: XSelector,
            y: YSelector | None,
        ) -> tuple[pd.DataFrame, Pandas | None]:
            """Obtém X e y a partir da transformação do pipeline.

            Parâmetros
            ----------
            X: dataframe-like
                Conjunto de features.

            y: int, str, sequence, dataframe-like or None
                Coluna(s) alvo correspondente(s) a `X`.

            Retorna
            -------
            dataframe
                Conjunto de features transformado.

            series, dataframe or None
                Coluna alvo transformada.

            """
            Xt, yt = self._check_input(X, y, columns=self.og.features, name=self.og.target)

            with adjust(self.pipeline, verbose=verbose) as pl:
                out = pl.transform(Xt, yt)

            return out if isinstance(out, tuple) else (out, yt)

        def assign_prediction_columns() -> list[str]:
            """Atribui nomes de colunas para os métodos de previsão.

            Retorna
            -------
            list of str
                Colunas para o dataframe.

            """
            if self.task.is_multioutput:
                return self.target  # When multioutput, target is list of str
            else:
                return self.mapping.get(self.target, np.unique(self.y).astype(str))

        try:
            if isinstance(X, pd.DataFrame):
                # O DataFrame deve vir primeiro, pois podemos esperar
                # chamadas de previsão de dataframes com índices reiniciados
                Xt, yt = get_transform_X_y(X, y)
            else:
                Xt, yt = self.branch._get_rows(X, return_X_y=True)  # type: ignore[call-overload]

                if self.scaler:
                    Xt = cast(pd.DataFrame, self.scaler.transform(Xt))

        except Exception:  # noqa: BLE001
            Xt, yt = get_transform_X_y(X, y)  # type: ignore[arg-type]

        if method != "score":
            pred = np.array(self.memory.cache(getattr(self.estimator, method))(Xt[self.features]))

            if pred.ndim == 1 or pred.shape[1] == 1:
                return to_series(pred, index=Xt.index, name=self.target)
            elif pred.ndim < 3:
                return to_df(pred, index=Xt.index, columns=assign_prediction_columns())
            elif self.task is Task.multilabel_classification:
                # Converte para (n_samples, n_targets)
                return pd.DataFrame(
                    data=np.array([d[:, 1] for d in pred]).T,
                    index=Xt.index,
                    columns=assign_prediction_columns(),
                )
            else:
                # Converte para (n_samples * n_classes, n_targets)
                return pd.DataFrame(
                    data=pred.reshape(-1, pred.shape[2]),
                    index=pd.MultiIndex.from_tuples(
                        [(col, idx) for col in np.unique(self.y) for idx in Xt.index]
                    ),
                    columns=assign_prediction_columns(),
                )

        else:
            if metric is None:
                scorer = self._metric[0]
            else:
                scorer = get_custom_scorer(metric, pos_label=self._config.pos_label)

            return self._score_from_est(
                scorer=scorer,
                estimator=self.estimator,
                X=Xt,
                y=yt,  # type: ignore[arg-type]
                sample_weight=sample_weight,
            )

    @available_if(estimator_has_attr("decision_function"))
    @composed(crash, method_to_log, beartype)
    def decision_function(
        self,
        X: RowSelector | XSelector,
        *,
        verbose: Verbose | None = None,
    ) -> YReturn:
        """Obtém pontuações de confiança em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `decision_function`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        X: hashable, segment, sequence or dataframe-like
            [Seleção de linhas][row-and-column-selection] ou conjunto de
            features com shape=(n_samples, n_features) para fazer previsões.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        series or dataframe
            Pontuações de confiança previstas com shape=(n_samples,) para
            tarefas de classificação binária (razão log de verossimilhança da
            classe positiva) ou shape=(n_samples, n_classes) para tarefas de
            classificação multiclasse.

        """
        return self._convert(self._prediction(X, verbose=verbose, method="decision_function"))

    @available_if(estimator_has_attr("predict"))
    @composed(crash, method_to_log, beartype)
    def predict(
        self,
        X: RowSelector | XSelector,
        *,
        inverse: Bool = True,
        verbose: Verbose | None = None,
    ) -> YReturn:
        """Obtém previsões em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        X: hashable, segment, sequence or dataframe-like
            [Seleção de linhas][row-and-column-selection] ou conjunto de
            features com shape=(n_samples, n_features) para fazer previsões.

        inverse: bool, default=True
            Se deve transformar inversamente a saída pelo pipeline.
            Isso não afeta as previsões se não houver transformadores no
            pipeline ou se os transformadores não tiverem um método
            `inverse_transform` ou não se aplicarem a `y`.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        series or dataframe
            Previsões com shape=(n_samples,) ou shape=(n_samples,
            n_targets) para [tarefas multioutput][].

        """
        pred = self._prediction(X, verbose=verbose, method="predict")

        if inverse:
            return self.inverse_transform(y=pred)
        else:
            return self._convert(pred)

    @available_if(estimator_has_attr("predict_log_proba"))
    @composed(crash, method_to_log, beartype)
    def predict_log_proba(
        self,
        X: RowSelector | XSelector,
        *,
        verbose: Verbose | None = None,
    ) -> XReturn:
        """Obtém log-probabilidades das classes em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict_log_proba`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        X: hashable, segment, sequence or dataframe-like
            [Seleção de linhas][row-and-column-selection] ou conjunto de
            features com shape=(n_samples, n_features) para fazer previsões.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Log-probabilidades previstas das classes com shape=(n_samples,
            n_classes) ou shape=(n_samples * n_classes, n_targets) com
            formato multiindex para [tarefas multioutput][].

        """
        return self._convert(self._prediction(X, verbose=verbose, method="predict_log_proba"))

    @available_if(estimator_has_attr("predict_proba"))
    @composed(crash, method_to_log, beartype)
    def predict_proba(
        self,
        X: RowSelector | XSelector,
        *,
        verbose: Verbose | None = None,
    ) -> XReturn:
        """Obtém probabilidades das classes em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict_proba`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        X: hashable, segment, sequence or dataframe-like
            [Seleção de linhas][row-and-column-selection] ou conjunto de
            features com shape=(n_samples, n_features) para fazer previsões.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Probabilidades previstas das classes com shape=(n_samples,
            n_classes) ou shape=(n_samples * n_classes, n_targets) com
            formato multiindex para [tarefas multioutput][].

        """
        return self._convert(self._prediction(X, verbose=verbose, method="predict_proba"))

    @available_if(estimator_has_attr("score"))
    @composed(crash, method_to_log, beartype)
    def score(
        self,
        X: RowSelector | XSelector,
        y: YSelector | None = None,
        *,
        metric: str | MetricFunction | Scorer | None = None,
        sample_weight: Sequence[Scalar] | None = None,
        verbose: Verbose | None = None,
    ) -> Float:
        """Obtém uma pontuação de métrica em novos dados.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados.

        Leia mais no [guia do usuário][predicting].

        !!! info
            Se o parâmetro `metric` for deixado com seu valor padrão, o
            método retorna a pontuação da métrica do experionml, não a métrica retornada
            pelo método score do sklearn para estimadores.

        Parâmetros
        ----------
        X: hashable, segment, sequence or dataframe-like
            [Seleção de linhas][row-and-column-selection] ou conjunto de
            features com shape=(n_samples, n_features) para fazer previsões.

        y: int, str, sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

            - Se None: `X` deve ser uma seleção de linhas no dataset.
            - Se int: Posição da coluna alvo em `X`.
            - Se str: Nome da coluna alvo em `X`.
            - Se sequence: Coluna alvo com shape=(n_samples,) ou
              sequência de nomes ou posições de colunas para tarefas
              multioutput.
            - Se dataframe: Colunas alvo para tarefas multioutput.

        metric: str, func, scorer or None, default=None
            Métrica a calcular. Escolha entre qualquer um dos scorers do sklearn,
            uma função com assinatura `metric(y_true, y_pred) -> score`
            ou um objeto scorer. Se None, usa a métrica do experionml (a métrica
            principal para [execuções multi-métrica][]).

        sample_weight: sequence or None, default=None
            Pesos das amostras correspondentes a `y`.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        float
            Pontuação da métrica de X em relação a y.

        """
        return self._prediction(
            X=X,
            y=y,
            metric=metric,
            sample_weight=sample_weight,
            verbose=verbose,
            method="score",
        )


class ForecastModel(BaseModel):
    """Modelos de previsão de séries temporais."""

    @crash
    def get_tags(self) -> dict[str, Any]:
        """Obtém as tags do modelo.

        Retorna os parâmetros da classe que fornecem informações gerais sobre
        as características do modelo.

        Retorna
        -------
        dict
            Tags do modelo.

        """
        return {
            "acronym": self.acronym,
            "fullname": self.fullname,
            "estimator": self._est_class.__name__,
            "module": self._est_class.__module__,
            "handles_missing": getattr(self, "handles_missing", None),
            "multiple_seasonality": getattr(self, "multiple_seasonality", None),
            "native_multioutput": self.native_multioutput,
            "supports_engines": ", ".join(getattr(self, "supports_engines", ())),
        }

    @overload
    def _prediction(
        self,
        fh: RowSelector | ForecastingHorizon | None = ...,
        y: RowSelector | YSelector | None = ...,
        X: XSelector | None = ...,
        metric: str | MetricFunction | Scorer | None = ...,
        verbose: Verbose | None = ...,
        method: Literal[
            "predict",
            "predict_interval",
            "predict_proba",
            "predict_quantiles",
            "predict_residuals",
            "predict_var",
        ] = ...,
        **kwargs,
    ) -> Normal | Pandas: ...

    @overload
    def _prediction(
        self,
        fh: RowSelector | ForecastingHorizon | None,
        y: RowSelector | YSelector | None,
        X: XSelector | None,
        metric: str | MetricFunction | Scorer | None,
        verbose: Verbose | None,
        method: Literal["score"],
        **kwargs,
    ) -> Float: ...

    def _prediction(
        self,
        fh: RowSelector | ForecastingHorizon | None = None,
        y: RowSelector | YSelector | None = None,
        X: XSelector | None = None,
        metric: str | MetricFunction | Scorer | None = None,
        verbose: Verbose | None = None,
        method: PredictionMethodsTS = "predict",
        **kwargs,
    ) -> Float | Normal | Pandas:
        """Obtém previsões em novos dados ou linhas existentes.

        Se `fh` não for um [ForecastingHorizon][], obtém as linhas do
        branch. Se `fh` for um [ForecastingHorizon][] ou não for fornecido,
        converte `X` e `y` pelo pipeline. O modelo deve implementar o
        método fornecido.

        Parâmetros
        ----------
        fh: hashable, segment, sequence, dataframe, [ForecastingHorizon][] or None, default=None
            O [horizonte de previsão][row-and-column-selection] que codifica
            os timestamps para prever.

        y: int, str, sequence, dataframe-like or None, default=None
            Observações de ground truth.

        X: hashable, segment, sequence, dataframe-like or None, default=None
            Séries temporais exógenas correspondentes a `fh`.

        metric: str, func, scorer or None, default=None
            Métrica a calcular. Escolha entre qualquer um dos scorers do sklearn,
            uma função com assinatura metric(y_true, y_pred) ou um objeto scorer.
            Se None, retorna acurácia média para tarefas de classificação
            e r2 para tarefas de regressão. Somente para method="score".

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        method: str, default="predict"
            Método de previsão a ser aplicado ao estimador.

        **kwargs
            Argumentos de palavras-chave adicionais para o método.

        Retorna
        -------
        float, sktime.proba.[Normal][], series or dataframe
            Previsões calculadas. O tipo de retorno depende do método
            chamado.

        """
        Xt: pd.DataFrame | None
        yt: Pandas | None

        if not isinstance(fh, ForecastingHorizon | None):
            try:
                Xt, yt = self.branch._get_rows(fh, return_X_y=True)

                if self.scaler:
                    Xt = cast(pd.DataFrame, self.scaler.transform(Xt))

                fh = ForecastingHorizon(Xt.index, is_relative=False)
            except IndexError:
                raise ValueError(
                    f"Uso ambíguo do parâmetro fh, recebido {fh}. Use um objeto ForecastingHorizon "
                    "para fazer previsões em amostras futuras fora do dataset."
                ) from None

        elif y is not None:
            try:
                Xt, yt = self.branch._get_rows(y, return_X_y=True)  # type: ignore[call-overload]

                if self.scaler and Xt is not None:
                    Xt = cast(pd.DataFrame, self.scaler.transform(Xt))

            except Exception:  # noqa: BLE001
                Xt, yt = self._check_input(X, y, columns=self.og.features, name=self.og.target)  # type: ignore[arg-type]

                with adjust(self.pipeline, verbose=verbose) as pl:
                    out = pl.transform(Xt, yt)

                Xt, yt = out if isinstance(out, tuple) else (Xt, out)

        elif X is not None:
            Xt, _ = self._check_input(X, columns=self.og.features, name=self.og.target)  # type: ignore[call-overload, arg-type]

            with adjust(self.pipeline, verbose=verbose) as pl:
                Xt = pl.transform(Xt)

        else:
            Xt, yt = X, y

        if method != "score":
            if "y" in sign(func := getattr(self.estimator, method)):
                return self.memory.cache(func)(y=yt, X=check_empty(Xt), **kwargs)
            else:
                return self.memory.cache(func)(fh=fh, X=check_empty(Xt), **kwargs)
        else:
            if metric is None:
                scorer = self._metric[0]
            else:
                scorer = get_custom_scorer(metric, pos_label=self._config.pos_label)

            return self._score_from_est(scorer, self.estimator, Xt, yt, **kwargs)  # type: ignore[arg-type]

    @composed(crash, method_to_log, beartype)
    def predict(
        self,
        fh: RowSelector | ForecastingHorizon,
        X: XSelector | None = None,
        *,
        inverse: Bool = True,
        verbose: Verbose | None = None,
    ) -> YReturn:
        """Obtém previsões em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        fh: hashable, segment, sequence, dataframe or [ForecastingHorizon][]
            O [horizonte de previsão][row-and-column-selection] que codifica
            os timestamps para prever.

        X: hashable, segment, sequence, dataframe-like or None, default=None
            Séries temporais exógenas correspondentes a `fh`.

        inverse: bool, default=True
            Se deve transformar inversamente a saída pelo pipeline.
            Isso não afeta as previsões se não houver transformadores no
            pipeline ou se os transformadores não tiverem um método
            `inverse_transform` ou não se aplicarem a `y`.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        series or dataframe
            Previsões com shape=(n_samples,) ou shape=(n_samples,
            n_targets) para tarefas [multivariadas][].

        """
        pred = self._prediction(fh=fh, X=X, verbose=verbose, method="predict")

        if inverse:
            return self.inverse_transform(y=pred)
        else:
            return self._convert(pred)

    @composed(crash, method_to_log, beartype)
    def predict_interval(
        self,
        fh: RowSelector | ForecastingHorizon,
        X: XSelector | None = None,
        *,
        coverage: Float | Sequence[Float] = 0.9,
        inverse: Bool = True,
        verbose: Verbose | None = None,
    ) -> XReturn:
        """Obtém intervalos de previsão em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict_interval`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        fh: hashable, segment, sequence, dataframe or [ForecastingHorizon][]
            O [horizonte de previsão][row-and-column-selection] que codifica
            os timestamps para prever.

        X: hashable, segment, sequence, dataframe-like or None, default=None
            Séries temporais exógenas correspondentes a `fh`.

        coverage: float or sequence, default=0.9
            Cobertura(s) nominal(is) do(s) intervalo(s) de previsão.

        inverse: bool, default=True
            Se deve transformar inversamente a saída pelo pipeline.
            Isso não afeta as previsões se não houver transformadores no
            pipeline ou se os transformadores não tiverem um método
            `inverse_transform` ou não se aplicarem a `y`.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Previsões de intervalo calculadas.

        """
        pred = self._prediction(
            fh=fh,
            X=X,
            coverage=coverage,
            verbose=verbose,
            method="predict_interval",
        )

        if inverse:
            new_interval = pd.DataFrame(index=pred.index, columns=pred.columns)

            # Passamos cada nível do multiindex para inverse_transform...
            for cover in pred.columns.levels[1]:  # type: ignore[union-attr]
                for level in pred.columns.levels[2]:  # type: ignore[union-attr]
                    # Seleciona apenas as colunas do nível inferior ou superior
                    curr_cover = pred.columns.get_level_values(1)
                    curr_level = pred.columns.get_level_values(2)
                    df = pred.loc[:, (curr_cover == cover) & (curr_level == level)]

                    # Usa os nomes originais das colunas
                    df.columns = df.columns.droplevel(level=(1, 2))

                    # Aplica a transformação inversa
                    for name, column in self.inverse_transform(y=df).items():
                        new_interval.loc[:, (name, cover, level)] = column

            return self._convert(new_interval)
        else:
            return self._convert(pred)

    @composed(crash, method_to_log, beartype)
    def predict_proba(
        self,
        fh: RowSelector | ForecastingHorizon,
        X: XSelector | None = None,
        *,
        marginal: Bool = True,
        verbose: Verbose | None = None,
    ) -> Normal:
        """Obtém previsões probabilísticas em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict_proba`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        fh: hashable, segment, sequence, dataframe or [ForecastingHorizon][]
            O [horizonte de previsão][row-and-column-selection] que codifica
            os timestamps para prever.

        X: hashable, segment, sequence, dataframe-like or None, default=None
            Séries temporais exógenas correspondentes a `fh`.

        marginal: bool, default=True
            Se a distribuição retornada é marginal por índice de tempo.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        sktime.proba.[Normal][]
            Objeto de distribuição.

        """
        return self._prediction(
            fh=fh,
            X=X,
            marginal=marginal,
            verbose=verbose,
            method="predict_proba",
        )

    @composed(crash, method_to_log, beartype)
    def predict_quantiles(
        self,
        fh: RowSelector | ForecastingHorizon,
        X: XSelector | None = None,
        *,
        alpha: Float | Sequence[Float] = (0.05, 0.95),
        verbose: Verbose | None = None,
    ) -> XReturn:
        """Obtém previsões de quantis em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict_quantiles`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        fh: hashable, segment, sequence, dataframe or [ForecastingHorizon][]
            O [horizonte de previsão][row-and-column-selection] que codifica
            os timestamps para prever.

        X: hashable, segment, sequence, dataframe-like or None, default=None
            Séries temporais exógenas correspondentes a `fh`.

        alpha: float or sequence, default=(0.05, 0.95)
            Uma probabilidade ou lista de probabilidades nas quais as
            previsões de quantis são calculadas.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Previsões de quantis calculadas.

        """
        return self._convert(
            self._prediction(
                fh=fh,
                X=X,
                alpha=alpha,
                verbose=verbose,
                method="predict_quantiles",
            )
        )

    @composed(crash, method_to_log, beartype)
    def predict_residuals(
        self,
        y: RowSelector | YConstructor,
        X: XSelector | None = None,
        *,
        verbose: Verbose | None = None,
    ) -> YReturn:
        """Obtém resíduos de previsões em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict_residuals`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        y: hashable, segment, sequence or dataframe-like
            [Seleção de linhas][row-and-column-selection] ou observações
            de ground truth.

        X: dataframe-like or None, default=None
            Séries temporais exógenas correspondentes a `y`. Este parâmetro
            é ignorado se `y` for uma seleção de linhas no dataset.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        series or dataframe
            Resíduos com shape=(n_samples,) ou shape=(n_samples,
            n_targets) para tarefas [multivariadas][].

        """
        return self._convert(
            self._prediction(y=y, X=X, verbose=verbose, method="predict_residuals")
        )

    @composed(crash, method_to_log, beartype)
    def predict_var(
        self,
        fh: RowSelector | ForecastingHorizon,
        X: XSelector | None = None,
        *,
        cov: Bool = False,
        verbose: Verbose | None = None,
    ) -> XReturn:
        """Obtém previsões de variância em novos dados ou linhas existentes.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados. O estimador deve ter um método `predict_var`.

        Leia mais no [guia do usuário][predicting].

        Parâmetros
        ----------
        fh: hashable, segment, sequence, dataframe or [ForecastingHorizon][]
            O [horizonte de previsão][row-and-column-selection] que codifica
            os timestamps para prever.

        X: hashable, segment, sequence, dataframe-like or None, default=None
            Séries temporais exógenas correspondentes a `fh`.

        cov: bool, default=False
            Se deve calcular a previsão da matriz de covariância ou
            previsões de variância marginal.

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Previsões de variância calculadas.

        """
        return self._convert(
            self._prediction(
                fh=fh,
                X=X,
                cov=cov,
                verbose=verbose,
                method="predict_var",
            )
        )

    @composed(crash, method_to_log, beartype)
    def score(
        self,
        y: RowSelector | YSelector,
        X: XSelector | None = None,
        fh: RowSelector | ForecastingHorizon | None = None,
        *,
        metric: str | MetricFunction | Scorer | None = None,
        verbose: Verbose | None = None,
    ) -> Float:
        """Obtém uma pontuação de métrica em novos dados.

        Novos dados são primeiro transformados pelo pipeline do modelo.
        Transformadores aplicados apenas no conjunto de treinamento são
        ignorados.

        Leia mais no [guia do usuário][predicting].

        !!! info
            Se o parâmetro `metric` for deixado com seu valor padrão, o
            método retorna a pontuação da métrica do experionml, não a métrica usada pelo
            método score do sktime para estimadores.

        Parâmetros
        ----------
        y: int, str, sequence or dataframe-like
            [Seleção de linhas][row-and-column-selection] ou observações
            de ground truth.

        X: dataframe-like or None, default=None
            Séries temporais exógenas correspondentes a `fh`. Este parâmetro
            é ignorado se `y` for uma seleção de linhas no dataset.

        fh: hashable, segment, sequence, dataframe, [ForecastingHorizon][] or None, default=None
            Não faz nada. O horizonte de previsão é obtido do índice de
            `y`. Implementado para continuidade da API do sktime.

        metric: str, func, scorer or None, default=None
            Métrica a calcular. Escolha entre qualquer um dos scorers do sklearn,
            uma função com assinatura `metric(y_true, y_pred) -> score`
            ou um objeto scorer. Se None, usa a métrica do experionml (a métrica
            principal para [execuções multi-métrica][]).

        verbose: int or None, default=None
            Nível de verbosidade para os transformadores no pipeline. Se
            None, usa a verbosidade do pipeline.

        Retorna
        -------
        float
            Pontuação da métrica de `y` em relação ao ground truth.

        """
        return self._prediction(fh=None, y=y, X=X, metric=metric, verbose=verbose, method="score")
        return self._prediction(fh=None, y=y, X=X, metric=metric, verbose=verbose, method="score")
        return self._prediction(fh=None, y=y, X=X, metric=metric, verbose=verbose, method="score")
