from __future__ import annotations

from logging import Logger
from pathlib import Path
from typing import Any, Literal

from beartype import beartype
from joblib.memory import Memory

from experionml.utils.types import (
    Backend,
    Bool,
    Engine,
    FloatLargerZero,
    IntLargerEqualZero,
    MetricConstructor,
    ModelsConstructor,
    NItems,
    NJobs,
    Sequence,
    Verbose,
    Warnings,
)
from experionml.utils.utils import (
    Goal,
)


from experionml.runners.base import Direct, SuccessiveHalving, TrainSizing


@beartype
class DirectForecaster(Direct):
    r"""Treina e avalia os modelos de forma direta.

    As etapas a seguir são aplicadas a cada modelo:

    1. Aplicar [ajuste de hiperparâmetros][] (opcional).
    2. Ajustar o modelo no conjunto de treino com a melhor combinação
       de hiperparâmetros encontrada.
    3. Avaliar o modelo no conjunto de teste.
    4. Treinar o estimador em amostras [bootstrapped][bootstrapping]
       do conjunto de treino e avaliar novamente no teste (opcional).

    Parâmetros
    ----------
    models: str, estimator or sequence, default=None
        Modelos a ajustar nos dados. As entradas permitidas são: uma sigla de
        qualquer um dos [modelos predefinidos][], um [ExperionMLModel][] ou um
        preditor personalizado como classe ou instância. Se None, todos os modelos
        predefinidos são usados.

    metric: str, func, scorer, sequence or None, default=None
        Métrica para ajuste dos modelos. Escolha entre qualquer um dos [scorers][] do
        sklearn, uma função com assinatura `function(y_true, y_pred, **kwargs) -> score`,
        um objeto scorer ou uma sequência destes. Se None, a métrica padrão `mean_absolute_percentage_error` é selecionada.

    n_trials: int, dict ou sequence, default=0
        Número máximo de iterações para o [ajuste de hiperparâmetros][].
        Se 0, o ajuste é ignorado e o modelo é ajustado com seus parâmetros
        padrão. Se sequence, o n-ésimo valor aplica-se ao n-ésimo modelo.

    est_params: dict ou None, default=None
        Parâmetros adicionais para os modelos. Consulte a documentação
        correspondente para as opções disponíveis. Para múltiplos modelos,
        use as siglas como chave (ou 'all' para todos) e um dict de parâmetros
        como valor. Adicione `_fit` ao nome do parâmetro para passá-lo ao
        método fit do estimador em vez do construtor.

    ht_params: dict ou None, default=None
        Parâmetros adicionais para o ajuste de hiperparâmetros. Se None,
        usa os mesmos parâmetros da primeira execução. Pode incluir:

        - **cv: int, cv-generator, dict ou sequence, default=1**<br>
          Objeto de validação cruzada ou número de divisões. Se 1, os dados
          são divididos aleatoriamente em subconjunto de treino e validação.
        - **plot: bool, dict ou sequence, default=False**<br>
          Se deve plotar o progresso da otimização em tempo real.
          Cria um canvas com dois gráficos: o primeiro exibe a pontuação de
          cada trial e o segundo mostra a distância entre os últimos passos
          consecutivos. Consulte o método [plot_trials][].
        - **distributions: dict, sequence ou None, default=None**<br>
          Distribuições de hiperparâmetros personalizadas. Se None, usa as
          distribuições predefinidas do modelo. Leia mais no
          [guia do usuário][hyperparameter-tuning].
        - **tags: dict, sequence ou None, default=None**<br>
          Tags personalizadas para o trial e a [execução do mlflow][tracking].
        - **\*\*kwargs**<br>
          Argumentos de palavra-chave adicionais para o construtor da classe
          [study][] ou do método [optimize][].

    n_bootstrap: int ou sequence, default=0
        Número de conjuntos de dados usados para [bootstrapping][]. Se 0, nenhum
        bootstrapping é realizado. Se sequence, o n-ésimo valor aplica-se
        ao n-ésimo modelo.

    parallel: bool, default=False
        Se os modelos devem ser treinados em paralelo ou sequencialmente.
        Usar `parallel=True` desativa a verbosidade dos modelos durante o
        treinamento. Observe que muitos modelos também possuem paralelização
        nativa (geralmente quando o estimador tem o parâmetro `n_jobs`).

    errors: str, default="skip"
        Como lidar com exceções encontradas durante o [treinamento][training] dos modelos.
        Escolha entre:

        - "raise": Lança qualquer exceção encontrada.
        - "skip": Ignora um modelo com falha. Este modelo não fica acessível
          após o treinamento.
        - "keep": Mantém o modelo no estado em que falhou. Este modelo pode
          quebrar outros métodos após o treinamento. Esta opção é útil para
          retomar a otimização de hiperparâmetros sem perder trials anteriores.

    n_jobs: int, default=1
        Número de núcleos a usar para processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usa todos os núcleos disponíveis.
        - Se <-1: Usa número de núcleos - 1 + `n_jobs`.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], por exemplo,
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str, dict ou None, default=None
        Engine de execução para [dados][data-engines] e
        [estimadores][estimator-acceleration]. O valor deve ser
        uma das opções possíveis para alterar uma das duas engines,
        ou um dicionário com chaves `data` e `estimator`, com as
        escolhas correspondentes como valores. Se None, os valores
        padrão são usados. Escolha entre:

        - "data":

            - "pandas" (padrão)
            - "pyarrow"
            - "modin"

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

    memory: bool, str, Path ou Memory, default=False
        Habilita cache para otimização de memória. Leia mais no
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

    warnings: bool ou str, default=False
        - Se True: Ação padrão de aviso (equivalente a "once").
        - Se False: Suprime todos os avisos (equivalente a "ignore").
        - Se str: Um dos [filtros de aviso][warnings] do Python.

        Alterar este parâmetro afeta o ambiente `PYTHONWarnings`.
        O ExperionML não consegue gerenciar avisos que vão de código C/C++ para stdout.

    logger: str, Logger ou None, default=None
        - Se None: Logging não é usado.
        - Se str: Nome do arquivo de log. Use "auto" para nome automático.
        - Se Path: Um [pathlib.Path][] para o arquivo de log.
        - Caso contrário: Instância de `logging.Logger` do Python.

    experiment: str ou None, default=None
        Nome do [experimento mlflow][experiment] a usar para rastreamento.
        Se None, nenhum rastreamento mlflow é realizado.

    random_state: int ou None, default=None
        Semente usada pelo gerador de números aleatórios. Se None, o gerador
        de números aleatórios é o `RandomState` usado por `np.random`.

    Veja também
    -----------
    experionml.api:ExperionMLForecaster
    experionml.training:SuccessiveHalvingForecaster
    experionml.training:TrainSizingForecaster

    Exemplos
    --------
    ```pycon
    from experionml.training import DirectForecaster
    from sktime.datasets import load_airline
    from sktime.split import temporal_train_test_split

    y = load_airline()

    train, test = temporal_train_test_split(y, test_size=0.2)

    runner = DirectForecaster(models=["ES", "ETS"], verbose=2)
    runner.run(train, test)

    # Analisa os resultados
    runner.results
    ```

    """

    _goal = Goal.forecast

    def __init__(
        self,
        models: ModelsConstructor = None,
        metric: MetricConstructor = None,
        *,
        est_params: dict[str, Any] | None = None,
        n_trials: NItems = 0,
        ht_params: dict[str, Any] | None = None,
        n_bootstrap: NItems = 0,
        parallel: Bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
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
            models,
            metric,
            est_params,
            n_trials,
            ht_params,
            n_bootstrap,
            parallel,
            errors,
            n_jobs,
            device,
            engine,
            backend,
            memory,
            verbose,
            warnings,
            logger,
            experiment,
            random_state,
        )


@beartype
class SuccessiveHalvingForecaster(SuccessiveHalving):
    r"""Treina e avalia os modelos no formato [successive halving][].

    As etapas a seguir são aplicadas a cada modelo (por iteração):

    1. Aplicar [ajuste de hiperparâmetros][] (opcional).
    2. Ajustar o modelo no conjunto de treino com a melhor combinação
       de hiperparâmetros encontrada.
    3. Avaliar o modelo no conjunto de teste.
    4. Treinar o estimador em amostras [bootstrapped][bootstrapping]
       do conjunto de treino e avaliar novamente no teste (opcional).

    Parâmetros
    ----------
    models: str, estimator or sequence, default=None
        Modelos a ajustar nos dados. As entradas permitidas são: uma sigla de
        qualquer um dos [modelos predefinidos][], um [ExperionMLModel][] ou um
        preditor personalizado como classe ou instância. Se None, todos os modelos
        predefinidos são usados.

    metric: str, func, scorer, sequence or None, default=None
        Métrica para ajuste dos modelos. Escolha entre qualquer um dos [scorers][] do
        sklearn, uma função com assinatura `function(y_true, y_pred, **kwargs) -> score`,
        um objeto scorer ou uma sequência destes. Se None, a métrica padrão `mean_absolute_percentage_error` é selecionada.

    skip_runs: int, default=0
        Ignora as últimas `skip_runs` execuções do successive halving.

    n_trials: int, dict ou sequence, default=0
        Número máximo de iterações para o [ajuste de hiperparâmetros][].
        Se 0, o ajuste é ignorado e o modelo é ajustado com seus parâmetros
        padrão. Se sequence, o n-ésimo valor aplica-se ao n-ésimo modelo.

    est_params: dict ou None, default=None
        Parâmetros adicionais para os modelos. Consulte a documentação
        correspondente para as opções disponíveis. Para múltiplos modelos,
        use as siglas como chave (ou 'all' para todos) e um dict de parâmetros
        como valor. Adicione `_fit` ao nome do parâmetro para passá-lo ao
        método fit do estimador em vez do construtor.

    ht_params: dict ou None, default=None
        Parâmetros adicionais para o ajuste de hiperparâmetros. Se None,
        usa os mesmos parâmetros da primeira execução. Pode incluir:

        - **cv: int, cv-generator, dict ou sequence, default=1**<br>
          Objeto de validação cruzada ou número de divisões. Se 1, os dados
          são divididos aleatoriamente em subconjunto de treino e validação.
        - **plot: bool, dict ou sequence, default=False**<br>
          Se deve plotar o progresso da otimização em tempo real.
          Cria um canvas com dois gráficos: o primeiro exibe a pontuação de
          cada trial e o segundo mostra a distância entre os últimos passos
          consecutivos. Consulte o método [plot_trials][].
        - **distributions: dict, sequence ou None, default=None**<br>
          Distribuições de hiperparâmetros personalizadas. Se None, usa as
          distribuições predefinidas do modelo. Leia mais no
          [guia do usuário][hyperparameter-tuning].
        - **tags: dict, sequence ou None, default=None**<br>
          Tags personalizadas para o trial e a [execução do mlflow][tracking].
        - **\*\*kwargs**<br>
          Argumentos de palavra-chave adicionais para o construtor da classe
          [study][] ou do método [optimize][].

    n_bootstrap: int ou sequence, default=0
        Número de conjuntos de dados usados para [bootstrapping][]. Se 0, nenhum
        bootstrapping é realizado. Se sequence, o n-ésimo valor aplica-se
        ao n-ésimo modelo.

    parallel: bool, default=False
        Se os modelos devem ser treinados em paralelo ou sequencialmente.
        Usar `parallel=True` desativa a verbosidade dos modelos durante o
        treinamento. Observe que muitos modelos também possuem paralelização
        nativa (geralmente quando o estimador tem o parâmetro `n_jobs`).

    errors: str, default="skip"
        Como lidar com exceções encontradas durante o [treinamento][training] dos modelos.
        Escolha entre:

        - "raise": Lança qualquer exceção encontrada.
        - "skip": Ignora um modelo com falha. Este modelo não fica acessível
          após o treinamento.
        - "keep": Mantém o modelo no estado em que falhou. Este modelo pode
          quebrar outros métodos após o treinamento. Esta opção é útil para
          retomar a otimização de hiperparâmetros sem perder trials anteriores.

    n_jobs: int, default=1
        Número de núcleos a usar para processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usa todos os núcleos disponíveis.
        - Se <-1: Usa número de núcleos - 1 + `n_jobs`.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], por exemplo,
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str, dict ou None, default=None
        Engine de execução para [dados][data-engines] e
        [estimadores][estimator-acceleration]. O valor deve ser
        uma das opções possíveis para alterar uma das duas engines,
        ou um dicionário com chaves `data` e `estimator`, com as
        escolhas correspondentes como valores. Se None, os valores
        padrão são usados. Escolha entre:

        - "data":

            - "pandas" (padrão)
            - "pyarrow"
            - "modin"

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

    memory: bool, str, Path ou Memory, default=False
        Habilita cache para otimização de memória. Leia mais no
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

    warnings: bool ou str, default=False
        - Se True: Ação padrão de aviso (equivalente a "once").
        - Se False: Suprime todos os avisos (equivalente a "ignore").
        - Se str: Um dos [filtros de aviso][warnings] do Python.

        Alterar este parâmetro afeta o ambiente `PYTHONWarnings`.
        O ExperionML não consegue gerenciar avisos que vão de código C/C++ para stdout.

    logger: str, Logger ou None, default=None
        - Se None: Logging não é usado.
        - Se str: Nome do arquivo de log. Use "auto" para nome automático.
        - Se Path: Um [pathlib.Path][] para o arquivo de log.
        - Caso contrário: Instância de `logging.Logger` do Python.

    experiment: str ou None, default=None
        Nome do [experimento mlflow][experiment] a usar para rastreamento.
        Se None, nenhum rastreamento mlflow é realizado.

    random_state: int ou None, default=None
        Semente usada pelo gerador de números aleatórios. Se None, o gerador
        de números aleatórios é o `RandomState` usado por `np.random`.

    Veja também
    -----------
    experionml.api:ExperionMLForecaster
    experionml.training:DirectForecaster
    experionml.training:TrainSizingForecaster

    Exemplos
    --------
    ```pycon
    from experionml.training import SuccessiveHalvingForecaster
    from sktime.datasets import load_airline
    from sktime.split import temporal_train_test_split

    y = load_airline()

    train, test = temporal_train_test_split(y, test_size=0.2)

    runner = SuccessiveHalvingForecaster(["Croston", "PT"], verbose=2)
    runner.run(train, test)

    # Analisa os resultados
    runner.results
    ```

    """

    _goal = Goal.forecast

    def __init__(
        self,
        models: ModelsConstructor = None,
        metric: MetricConstructor = None,
        *,
        skip_runs: IntLargerEqualZero = 0,
        est_params: dict[str, Any] | None = None,
        n_trials: NItems = 0,
        ht_params: dict[str, Any] | None = None,
        n_bootstrap: NItems = 0,
        parallel: Bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | str = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            models,
            metric,
            skip_runs,
            est_params,
            n_trials,
            ht_params,
            n_bootstrap,
            parallel,
            errors,
            n_jobs,
            device,
            engine,
            backend,
            memory,
            verbose,
            warnings,
            logger,
            experiment,
            random_state,
        )


@beartype
class TrainSizingForecaster(TrainSizing):
    r"""Treina e avalia os modelos no formato [train sizing][].

    As etapas a seguir são aplicadas a cada modelo (por iteração):

    1. Aplicar [ajuste de hiperparâmetros][] (opcional).
    2. Ajustar o modelo no conjunto de treino com a melhor combinação
       de hiperparâmetros encontrada.
    3. Avaliar o modelo no conjunto de teste.
    4. Treinar o estimador em amostras [bootstrapped][bootstrapping]
       do conjunto de treino e avaliar novamente no teste (opcional).

    Parâmetros
    ----------
    models: str, estimator or sequence, default=None
        Modelos a ajustar nos dados. As entradas permitidas são: uma sigla de
        qualquer um dos [modelos predefinidos][], um [ExperionMLModel][] ou um
        preditor personalizado como classe ou instância. Se None, todos os modelos
        predefinidos são usados.

    metric: str, func, scorer, sequence or None, default=None
        Métrica para ajuste dos modelos. Escolha entre qualquer um dos [scorers][] do
        sklearn, uma função com assinatura `function(y_true, y_pred, **kwargs) -> score`,
        um objeto scorer ou uma sequência destes. Se None, a métrica padrão `mean_absolute_percentage_error` é selecionada.

    train_sizes: int ou sequence, default=5
        Tamanhos de conjuntos de treino usados nas execuções.

        - Se int: Número de divisões igualmente distribuídas, ou seja, para
          valor `N`, equivale a `np.linspace(1.0/N, 1.0, N)`.
        - Se sequence: Fração do conjunto de treino quando <=1; caso
          contrário, número total de amostras.

    n_trials: int, dict ou sequence, default=0
        Número máximo de iterações para o [ajuste de hiperparâmetros][].
        Se 0, o ajuste é ignorado e o modelo é ajustado com seus parâmetros
        padrão. Se sequence, o n-ésimo valor aplica-se ao n-ésimo modelo.

    est_params: dict ou None, default=None
        Parâmetros adicionais para os modelos. Consulte a documentação
        correspondente para as opções disponíveis. Para múltiplos modelos,
        use as siglas como chave (ou 'all' para todos) e um dict de parâmetros
        como valor. Adicione `_fit` ao nome do parâmetro para passá-lo ao
        método fit do estimador em vez do construtor.

    ht_params: dict ou None, default=None
        Parâmetros adicionais para o ajuste de hiperparâmetros. Se None,
        usa os mesmos parâmetros da primeira execução. Pode incluir:

        - **cv: int, cv-generator, dict ou sequence, default=1**<br>
          Objeto de validação cruzada ou número de divisões. Se 1, os dados
          são divididos aleatoriamente em subconjunto de treino e validação.
        - **plot: bool, dict ou sequence, default=False**<br>
          Se deve plotar o progresso da otimização em tempo real.
          Cria um canvas com dois gráficos: o primeiro exibe a pontuação de
          cada trial e o segundo mostra a distância entre os últimos passos
          consecutivos. Consulte o método [plot_trials][].
        - **distributions: dict, sequence ou None, default=None**<br>
          Distribuições de hiperparâmetros personalizadas. Se None, usa as
          distribuições predefinidas do modelo. Leia mais no
          [guia do usuário][hyperparameter-tuning].
        - **tags: dict, sequence ou None, default=None**<br>
          Tags personalizadas para o trial e a [execução do mlflow][tracking].
        - **\*\*kwargs**<br>
          Argumentos de palavra-chave adicionais para o construtor da classe
          [study][] ou do método [optimize][].

    n_bootstrap: int ou sequence, default=0
        Número de conjuntos de dados usados para [bootstrapping][]. Se 0, nenhum
        bootstrapping é realizado. Se sequence, o n-ésimo valor aplica-se
        ao n-ésimo modelo.

    parallel: bool, default=False
        Se os modelos devem ser treinados em paralelo ou sequencialmente.
        Usar `parallel=True` desativa a verbosidade dos modelos durante o
        treinamento. Observe que muitos modelos também possuem paralelização
        nativa (geralmente quando o estimador tem o parâmetro `n_jobs`).

    errors: str, default="skip"
        Como lidar com exceções encontradas durante o [treinamento][training] dos modelos.
        Escolha entre:

        - "raise": Lança qualquer exceção encontrada.
        - "skip": Ignora um modelo com falha. Este modelo não fica acessível
          após o treinamento.
        - "keep": Mantém o modelo no estado em que falhou. Este modelo pode
          quebrar outros métodos após o treinamento. Esta opção é útil para
          retomar a otimização de hiperparâmetros sem perder trials anteriores.

    n_jobs: int, default=1
        Número de núcleos a usar para processamento paralelo.

        - Se >0: Número de núcleos a usar.
        - Se -1: Usa todos os núcleos disponíveis.
        - Se <-1: Usa número de núcleos - 1 + `n_jobs`.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], por exemplo,
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str, dict ou None, default=None
        Engine de execução para [dados][data-engines] e
        [estimadores][estimator-acceleration]. O valor deve ser
        uma das opções possíveis para alterar uma das duas engines,
        ou um dicionário com chaves `data` e `estimator`, com as
        escolhas correspondentes como valores. Se None, os valores
        padrão são usados. Escolha entre:

        - "data":

            - "pandas" (padrão)
            - "pyarrow"
            - "modin"

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

    memory: bool, str, Path ou Memory, default=False
        Habilita cache para otimização de memória. Leia mais no
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

    warnings: bool ou str, default=False
        - Se True: Ação padrão de aviso (equivalente a "once").
        - Se False: Suprime todos os avisos (equivalente a "ignore").
        - Se str: Um dos [filtros de aviso][warnings] do Python.

        Alterar este parâmetro afeta o ambiente `PYTHONWarnings`.
        O ExperionML não consegue gerenciar avisos que vão de código C/C++ para stdout.

    logger: str, Logger ou None, default=None
        - Se None: Logging não é usado.
        - Se str: Nome do arquivo de log. Use "auto" para nome automático.
        - Se Path: Um [pathlib.Path][] para o arquivo de log.
        - Caso contrário: Instância de `logging.Logger` do Python.

    experiment: str ou None, default=None
        Nome do [experimento mlflow][experiment] a usar para rastreamento.
        Se None, nenhum rastreamento mlflow é realizado.

    random_state: int ou None, default=None
        Semente usada pelo gerador de números aleatórios. Se None, o gerador
        de números aleatórios é o `RandomState` usado por `np.random`.

    Veja também
    -----------
    experionml.api:ExperionMLForecaster
    experionml.training:DirectForecaster
    experionml.training:SuccessiveHalvingForecaster

    Exemplos
    --------
    ```pycon
    from experionml.training import TrainSizingForecaster
    from sktime.datasets import load_airline
    from sktime.split import temporal_train_test_split

    y = load_airline()

    train, test = temporal_train_test_split(y, test_size=0.2)

    runner = TrainSizingForecaster(["Croston", "PT"], verbose=2)
    runner.run(train, test)

    # Analisa os resultados
    runner.results
    ```

    """

    _goal = Goal.forecast

    def __init__(
        self,
        models: ModelsConstructor = None,
        metric: MetricConstructor = None,
        *,
        train_sizes: FloatLargerZero | Sequence[FloatLargerZero] = 5,
        est_params: dict[str, Any] | None = None,
        n_trials: NItems = 0,
        ht_params: dict[str, Any] | None = None,
        n_bootstrap: NItems = 0,
        parallel: Bool = False,
        errors: Literal["raise", "skip", "keep"] = "skip",
        n_jobs: NJobs = 1,
        device: str = "cpu",
        engine: Engine = None,
        backend: Backend = "loky",
        memory: Bool | str | Path | Memory = False,
        verbose: Verbose = 0,
        warnings: Bool | str = False,
        logger: str | Path | Logger | None = None,
        experiment: str | None = None,
        random_state: IntLargerEqualZero | None = None,
    ):
        super().__init__(
            models,
            metric,
            train_sizes,
            est_params,
            n_trials,
            ht_params,
            n_bootstrap,
            parallel,
            errors,
            n_jobs,
            device,
            engine,
            backend,
            memory,
            verbose,
            warnings,
            logger,
            experiment,
            random_state,
        )
