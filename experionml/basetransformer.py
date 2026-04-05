from __future__ import annotations

import os
import random
import re
import tempfile
import warnings
from collections.abc import Hashable
from datetime import datetime as dt
from importlib import import_module
from importlib.util import find_spec
from logging import DEBUG, FileHandler, Formatter, Logger, NullHandler, getLogger
from multiprocessing import cpu_count
from pathlib import Path
from typing import Literal, NoReturn, TypeVar, overload

import joblib
import mlflow
import numpy as np
import pandas as pd
from beartype import beartype
from joblib.memory import Memory
from pandas._typing import Axes
from sklearn.utils.validation import check_memory

from experionml.integrations import INTEGRATIONS
from experionml.utils.types import (
    Backend,
    Bool,
    Engine,
    EngineDataOptions,
    EngineEstimatorOptions,
    EngineTuple,
    Estimator,
    FeatureNamesOut,
    Int,
    IntLargerEqualZero,
    Pandas,
    Severity,
    Verbose,
    Warnings,
    XReturn,
    XSelector,
    YReturn,
    YSelector,
    bool_t,
    int_t,
)
from experionml.utils.utils import check_dependency, crash, lst, make_sklearn, to_df, to_tabular


T_Estimator = TypeVar("T_Estimator", bound=Estimator)


class BaseTransformer:
    """Classe base para transformadores no pacote.

    Inclui experionml e runners. Contém propriedades compartilhadas,
    métodos de preparação de dados e métodos utilitários.

    Parâmetros
    ----------
    **kwargs
        Argumentos nomeados padrão para as classes. Pode incluir:

        - n_jobs: Número de núcleos para processamento paralelo.
        - device: Dispositivo no qual executar os estimadores.
        - engine: Motor de execução para dados e estimadores.
        - backend: Backend de paralelização.
        - verbose: Nível de verbosidade da saída.
        - warnings: Se deve exibir ou suprimir avisos encontrados.
        - logger: Nome do arquivo de log, objeto Logger ou None.
        - experiment: Nome do experimento mlflow usado para rastreamento.
        - random_state: Semente usada pelo gerador de números aleatórios.

    """

    attrs = (
        "n_jobs",
        "device",
        "engine",
        "backend",
        "memory",
        "verbose",
        "warnings",
        "logger",
        "experiment",
        "random_state",
    )

    def __init__(self, **kwargs):
        """Atualiza as propriedades com os kwargs fornecidos."""
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Propriedades ================================================= >>

    @property
    def n_jobs(self) -> int:
        """Número de núcleos para processamento paralelo."""
        return self._n_jobs

    @n_jobs.setter
    @beartype
    def n_jobs(self, value: Int):
        # Verifica o número de núcleos para multiprocessamento
        if value > (n_cores := cpu_count()):
            self._n_jobs = n_cores
        else:
            self._n_jobs = int(n_cores + 1 + value if value < 0 else value)

    @property
    def device(self) -> str:
        """Dispositivo no qual executar os estimadores."""
        return self._device

    @device.setter
    @beartype
    def device(self, value: str):
        self._device = value
        if "gpu" in value.lower():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self._device_id)

    @property
    def engine(self) -> EngineTuple:
        """Motor de execução para dados e estimadores."""
        return self._engine

    @engine.setter
    @beartype
    def engine(self, value: Engine):
        if value is None:
            engine = EngineTuple()
        elif value in EngineDataOptions.__args__:
            engine = EngineTuple(data=value)  # type: ignore[arg-type]
        elif value in EngineEstimatorOptions.__args__:
            engine = EngineTuple(estimator=value)  # type: ignore[arg-type]
        elif isinstance(value, dict):
            engine = EngineTuple(
                data=value.get("data", EngineTuple().data),
                estimator=value.get("estimator", EngineTuple().estimator),
            )
        elif isinstance(value, EngineTuple):
            engine = value

        # Garante que a biblioteca do motor de dados está instalada
        check_dependency(engine.data_engine.library)

        if engine.estimator == "sklearnex":
            check_dependency("sklearnex")
            import sklearnex

            sklearnex.set_config(self.device.lower() if self._gpu else "auto")

        elif engine.estimator == "cuml":
            if not find_spec("cuml"):
                raise ModuleNotFoundError(
                    "Falha ao importar cuml. Pacote não instalado. "
                    "Consulte: https://rapids.ai/start.html#install."
                )
            else:
                from cuml.common.device_selection import set_global_device_type

                set_global_device_type("gpu" if self._gpu else "cpu")

                # Ver https://github.com/rapidsai/cuml/issues/5564
                from cuml.internals.memory_utils import set_global_output_type

                set_global_output_type("numpy")

        self._engine = engine

    @property
    def backend(self) -> Backend:
        """Backend de paralelização."""
        return self._backend

    @backend.setter
    @beartype
    def backend(self, value: Backend):
        if value == "ray":
            check_dependency("ray")
            import ray
            from ray.util.joblib import register_ray

            register_ray()  # Registra ray como backend do joblib
            if not ray.is_initialized():
                ray.init(log_to_driver=False)

        elif value == "dask":
            check_dependency("dask")
            from dask.distributed import Client

            try:
                Client.current()
            except ValueError:
                Client(processes=False)

        joblib.parallel_config(backend=value)

        self._backend = value

    @property
    def memory(self) -> Memory:
        """Retorna o objeto de memória interno."""
        return self._memory

    @memory.setter
    @beartype
    def memory(self, value: Bool | str | Path | Memory):
        """Cria um novo objeto de memória interno."""
        if value is False:
            value = None
        elif value is True:
            value = tempfile.gettempdir()
        elif isinstance(value, Path):
            value = str(value)

        self._memory = check_memory(value)

    @property
    def verbose(self) -> Verbose:
        """Nível de verbosidade da saída."""
        return self._verbose

    @verbose.setter
    @beartype
    def verbose(self, value: Verbose):
        self._verbose = value

    @property
    def warnings(self) -> Warnings:
        """Se deve exibir ou suprimir avisos encontrados."""
        return self._warnings

    @warnings.setter
    @beartype
    def warnings(self, value: Bool | Warnings):
        if isinstance(value, bool_t):
            self._warnings: Warnings = "once" if value else "ignore"
        else:
            self._warnings = value

        warnings.filterwarnings(self._warnings)  # Altera o filtro neste processo
        warnings.filterwarnings("ignore", category=FutureWarning, module=".*pandas.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module=".*imblearn.*")
        warnings.filterwarnings("ignore", category=UserWarning, module=".*sktime.*")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=".*shap.*")
        warnings.filterwarnings("ignore", category=ResourceWarning, module=".*ray.*")
        os.environ["PYTHONWARNINGS"] = self._warnings  # Afeta subprocessos (joblib)

    @property
    def logger(self) -> Logger | None:
        """Logger desta instância."""
        return self._logger

    @logger.setter
    @beartype
    def logger(self, value: str | Path | Logger | None):
        external_loggers = [
            "dagshub",
            "mlflow",
            "optuna",
            "ray",
            "featuretools",
            "prophet",
            "cmdstanpy",
        ]

        # Limpa handlers existentes para loggers externos
        for name in external_loggers:
            for handler in (log := getLogger(name)).handlers:
                handler.close()
            log.handlers.clear()
            log.addHandler(
                NullHandler()
            )  # Adiciona handler fictício para evitar logging.lastResort

        if not value:
            logger = None
        else:
            if isinstance(value, Logger):
                logger = value
            else:
                logger = getLogger(self.__class__.__name__)
                logger.setLevel(DEBUG)

                # Limpa handlers existentes para o logger atual
                for handler in logger.handlers:
                    handler.close()
                logger.handlers.clear()

                # Prepara o FileHandler
                if (path := Path(value)).suffix != ".log":
                    path = path.with_suffix(".log")
                if path.name == "auto.log":
                    now = dt.now().strftime("%d%b%y_%Hh%Mm%Ss")
                    path = path.with_name(f"{self.__class__.__name__}_{now}.log")

                fh = FileHandler(path)
                fh.setFormatter(Formatter("%(asctime)s - %(levelname)s: %(message)s"))

                # Redireciona loggers para o handler de arquivo
                for name in [logger.name, *external_loggers]:
                    log = getLogger(name)
                    log.setLevel(DEBUG)
                    log.addHandler(fh)

        self._logger = logger

    @property
    def experiment(self) -> str | None:
        """Nome do experimento mlflow usado para rastreamento."""
        return self._experiment

    @experiment.setter
    @beartype
    def experiment(self, value: str | None):
        self._experiment = value
        if value:
            if ":" in value:
                integrator, experiment_name = value.split(":", 1)
                if integrator in INTEGRATIONS:
                    INTEGRATIONS[integrator](project_name=experiment_name)
                else:
                    raise ValueError(
                        "Valor inválido para o parâmetro experiment. O caractere ':' deve "
                        f"ser precedido por uma plataforma de integração válida, recebido {integrator}. "
                        f"Opções disponíveis: {','.join(INTEGRATIONS)}."
                    )
            else:
                if any(k in mlflow.get_tracking_uri() for k in INTEGRATIONS):
                    mlflow.set_tracking_uri("")  # Redefine URI para ./mlruns

                experiment_name = value

            mlflow.sklearn.autolog(disable=True)
            mlflow.set_experiment(experiment_name)

    @property
    def random_state(self) -> int | None:
        """Semente usada pelo gerador de números aleatórios."""
        return self._random_state

    @random_state.setter
    @beartype
    def random_state(self, value: IntLargerEqualZero | None):
        if value is not None:
            value = int(value)

        random.seed(value)
        np.random.seed(value)  # noqa: NPY002
        self._random_state = value

    @property
    def _gpu(self) -> bool:
        """Retorna se a instância usa uma implementação em GPU."""
        return "gpu" in self.device.lower()

    @property
    def _device_id(self) -> int:
        """Qual dispositivo GPU usar."""
        if len(value := self.device.split(":")) == 1:
            return 0  # Valor padrão
        else:
            try:
                return int(value[-1])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Valor inválido para o parâmetro device. O dispositivo GPU {value[-1]} "
                    "não é reconhecido. Use um único inteiro para indicar um dispositivo específico. "
                    "Observe que o ExperionML não suporta treinamento multi-GPU."
                ) from None

    # Métodos ====================================================== >>

    @staticmethod
    @overload
    def _check_input(
        X: Literal[None],
        y: Literal[None],
        *,
        columns: Axes | None = ...,
        name: str | Axes | None = ...,
    ) -> NoReturn: ...

    @staticmethod
    @overload
    def _check_input(
        X: XSelector,
        y: Literal[None],
        *,
        columns: Axes | None = ...,
        name: str | Axes | None = ...,
    ) -> tuple[pd.DataFrame, None]: ...

    @staticmethod
    @overload
    def _check_input(
        X: Literal[None],
        y: YSelector,
        *,
        columns: Axes | None = ...,
        name: str | Axes | None = ...,
    ) -> tuple[None, Pandas]: ...

    @staticmethod
    @overload
    def _check_input(
        X: XSelector,
        y: YSelector,
        *,
        columns: Axes | None = ...,
        name: str | Axes | None = ...,
    ) -> tuple[pd.DataFrame, Pandas]: ...

    @staticmethod
    def _check_input(
        X: XSelector | None = None,
        y: YSelector | None = None,
        *,
        columns: Axes | None = None,
        name: str | Axes | None = None,
    ) -> tuple[pd.DataFrame | None, Pandas | None]:
        """Prepara os dados de entrada.

        Converte X e y para pandas e realiza verificações padrão de
        compatibilidade (dimensões, comprimento, índices, etc...).

        Parâmetros
        ----------
        X: dataframe-like ou None, default=None
            Conjunto de features com shape=(n_samples, n_features). Se None,
            `X` é ignorado.

        y: int, str, sequence, dataframe-like ou None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

            - Se None: `y` é ignorado.
            - Se int: Posição da coluna alvo em `X`.
            - Se str: Nome da coluna alvo em `X`.
            - Se sequence: Coluna alvo com shape=(n_samples,) ou
              sequência de nomes ou posições de colunas para tarefas
              de múltiplas saídas.
            - Se dataframe-like: Colunas alvo para tarefas de múltiplas saídas.

        columns: sequência de str ou None, default=None
            Nomes de colunas para o conjunto de features. Se None, nomes
            padrão são usados.

        name: str, sequência ou None, default=None
            Nome da(s) coluna(s) alvo. Se None, um nome padrão é usado.

        Retorna
        -------
        pd.DataFrame ou None
            Conjunto de features.

        pd.Series, pd.DataFrame ou None
            Coluna(s) alvo correspondente(s) a `X`.

        """
        if X is None and y is None:
            raise ValueError("X e y não podem ser ambos None!")
        else:
            Xt = to_df(X() if callable(X) else X, columns=columns)

        # Prepara a coluna alvo
        yt: Pandas | None
        if y is None:
            yt = None
        elif isinstance(y, int_t):
            if Xt is None:
                raise ValueError("X não pode ser None quando y é um inteiro.")

            Xt, yt = Xt.drop(columns=Xt.columns[int(y)]), Xt[Xt.columns[int(y)]]
        elif isinstance(y, str):
            if Xt is not None:
                if y not in Xt.columns:
                    raise ValueError(f"Coluna {y} não encontrada em X!")

                Xt, yt = Xt.drop(columns=y), Xt[y]

            else:
                raise ValueError("X não pode ser None quando y é uma string.")
        else:
            # Se X e y têm número diferente de linhas, tenta multioutput
            if Xt is not None and not isinstance(y, dict) and len(Xt) != len(y):
                try:
                    targets: list[Hashable] = []
                    for col in y:
                        if isinstance(col, str) and col in Xt.columns:
                            targets.append(col)
                        elif isinstance(col, int_t):
                            if -Xt.shape[1] <= col < Xt.shape[1]:
                                targets.append(Xt.columns[int(col)])
                            else:
                                raise IndexError(
                                    "Valor inválido para o parâmetro y. O valor "
                                    f"{col} está fora do intervalo para dados com "
                                    f"{Xt.shape[1]} colunas."
                                )

                    Xt, yt = Xt.drop(columns=targets), Xt[targets]

                except (TypeError, IndexError, KeyError):
                    raise ValueError(
                        "X e y não têm o mesmo número de linhas,"
                        f" obtidos len(X)={len(Xt)} e len(y)={len(y)}."
                    ) from None
            else:
                yt = to_tabular(y, index=getattr(Xt, "index", None), columns=name)

            # Verifica se X e y têm os mesmos índices
            if Xt is not None and not Xt.index.equals(yt.index):
                raise ValueError("X e y não têm os mesmos índices!")

        return Xt, yt

    @overload
    def _convert(self, obj: Literal[None]) -> None: ...

    @overload
    def _convert(self, obj: pd.DataFrame) -> XReturn: ...

    @overload
    def _convert(self, obj: pd.Series) -> YReturn: ...

    def _convert(self, obj: Pandas | None) -> YReturn | None:
        """Converte dados para o tipo definido no motor de dados.

        Tipos que não são pandas são retornados como estão.

        Parâmetros
        ----------
        obj: object
            Objeto a ser convertido.

        Retorna
        -------
        object
            Dados convertidos ou objeto inalterado.

        """
        # Aplica transformações apenas quando o motor está definido
        if hasattr(self, "_engine") and isinstance(obj, pd.Series | pd.DataFrame):
            return self._engine.data_engine.convert(obj)
        else:
            return obj

    def _get_est_class(self, name: str, module: str) -> type[Estimator]:
        """Importa uma classe de um módulo.

        Quando a importação falha, por exemplo, se o experionml usa sklearnex e
        isso é passado para um transformador, usa o sklearn (motor padrão).

        Parâmetros
        ----------
        name: str
            Nome da classe a obter.

        module: str
            Módulo do qual obter a classe.

        Retorna
        -------
        Estimator
            Classe do estimador.

        """
        try:
            mod = import_module(f"{self.engine.estimator}.{module}")
        except (ModuleNotFoundError, AttributeError):
            mod = import_module(f"sklearn.{module}")

        return make_sklearn(getattr(mod, name))

    def _inherit(
        self,
        obj: T_Estimator,
        fixed: tuple[str, ...] = (),
        feature_names_out: FeatureNamesOut = "one-to-one",
    ) -> T_Estimator:
        """Herda parâmetros do pai.

        Método utilitário para definir os parâmetros sp (período sazonal), n_jobs e
        random_state de um estimador (se disponíveis) iguais aos desta instância.
        Se `obj` é um meta-estimador, também ajusta os parâmetros do estimador base.

        Parâmetros
        ----------
        obj: Estimator
            Instância para a qual alterar os parâmetros.

        fixed: tuple de str, default=()
            Parâmetros fixos que não devem ser sobrescritos.

        feature_names_out: "one-to-one", callable ou None, default="one-to-one"
            Determina a lista de nomes de features que serão retornados
            pelo método `get_feature_names_out`.

            - Se None: O método `get_feature_names_out` não é definido.
            - Se "one-to-one": Os nomes de features de saída serão iguais
              aos nomes de features de entrada.
            - Se callable: Função que recebe argumentos posicionais self
              e uma sequência de nomes de features de entrada. Deve retornar
              uma sequência de nomes de features de saída.

        Retorna
        -------
        Estimator
            Mesmo objeto com parâmetros alterados.

        """
        for p in obj.get_params():
            if p in fixed:
                continue
            elif match := re.search("^(n_jobs|random_state)$|__\1$", p):
                obj.set_params(**{p: getattr(self, match.group())})
            elif re.search(r"^sp$|__sp$", p) and hasattr(self, "_config"):
                if sp := self._config.sp.get("sp"):
                    if self.multiple_seasonality:
                        obj.set_params(**{p: sp})
                    else:
                        obj.set_params(**{p: lst(sp)[0]})

        return make_sklearn(obj, feature_names_out=feature_names_out)

    @crash
    def _log(self, msg: str, level: Int = 0, severity: Severity = "info"):
        """Exibe mensagem e salva no arquivo de log.

        Parâmetros
        ----------
        msg: str
            Mensagem a salvar no logger e exibir no stdout.

        level: int, default=0
            Nível mínimo de verbosidade para exibir a mensagem.

        severity: str, default="info"
            Nível de severidade da mensagem. Escolha entre: debug, info,
            warning, error, critical.

        """
        if severity in ("error", "critical"):
            raise UserWarning(msg)
        elif severity == "warning":
            warnings.warn(msg, category=UserWarning, stacklevel=2)
        elif severity == "info" and self.verbose >= level:
            print(msg)  # noqa: T201

        if getattr(self, "logger", None):
            for text in str(msg).split("\n"):
                getattr(self.logger, severity)(str(text))
