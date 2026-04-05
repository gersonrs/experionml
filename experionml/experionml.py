from __future__ import annotations

import os
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterator
from copy import deepcopy
from logging import Logger
from pathlib import Path
from platform import machine, platform, python_build, python_version
from types import MappingProxyType
from typing import Any, Literal, TypeVar

import dill as pickle
import numpy as np
import pandas as pd
from beartype import beartype
from joblib.memory import Memory
from pandas._typing import DtypeObj
from scipy import stats
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.utils import Bunch
from sklearn.utils.metaestimators import available_if

from experionml.baserunner import BaseRunner
from experionml.basetransformer import BaseTransformer
from experionml.data import Branch, BranchManager
from experionml.data_cleaning import (
    Balancer,
    Cleaner,
    Decomposer,
    Discretizer,
    Encoder,
    Imputer,
    Normalizer,
    Pruner,
    Scaler,
    TransformerMixin,
)
from experionml.feature_engineering import (
    FeatureExtractor,
    FeatureGenerator,
    FeatureGrouper,
    FeatureSelector,
)
from experionml.nlp import TextCleaner, TextNormalizer, Tokenizer, Vectorizer
from experionml.plots import ExperionMLPlot
from experionml.training import (
    DirectClassifier,
    DirectForecaster,
    DirectRegressor,
    SuccessiveHalvingClassifier,
    SuccessiveHalvingForecaster,
    SuccessiveHalvingRegressor,
    TrainSizingClassifier,
    TrainSizingForecaster,
    TrainSizingRegressor,
)
from experionml.utils.constants import CAT_TYPES, DEFAULT_MISSING, __version__
from experionml.utils.types import (
    Backend,
    Bins,
    Bool,
    CategoricalStrats,
    ColumnSelector,
    DiscretizerStrats,
    Engine,
    EngineTuple,
    Estimator,
    FeatureNamesOut,
    FeatureSelectionSolvers,
    FeatureSelectionStrats,
    FloatLargerEqualZero,
    FloatLargerZero,
    FloatZeroToOneInc,
    IndexSelector,
    Int,
    IntLargerEqualZero,
    IntLargerTwo,
    IntLargerZero,
    MetadataDict,
    MetricConstructor,
    ModelsConstructor,
    NItems,
    NJobs,
    NormalizerStrats,
    NumericalStrats,
    Operators,
    PosLabel,
    Predictor,
    PrunerStrats,
    RowSelector,
    Scalar,
    ScalerStrats,
    Seasonality,
    Sequence,
    SPDict,
    TargetSelector,
    Transformer,
    VectorizerStarts,
    Verbose,
    Warnings,
    XReturn,
    XSelector,
    YReturn,
    YSelector,
    sequence_t,
)
from experionml.utils.utils import (
    ClassMap,
    DataConfig,
    DataContainer,
    Goal,
    Task,
    adjust,
    check_dependency,
    composed,
    crash,
    fit_one,
    flt,
    get_cols,
    get_custom_scorer,
    has_task,
    is_sparse,
    lst,
    make_sklearn,
    merge,
    method_to_log,
    n_cols,
    replace_missing,
    sign,
    to_series,
)


T_Transformer = TypeVar("T_Transformer", bound=Transformer)


@beartype
class ExperionML(BaseRunner, ExperionMLPlot, metaclass=ABCMeta):
    """Classe base abstrata do ExperionML.

    A classe ExperionML é um invólucro conveniente para todas as classes
    de limpeza de dados, engenharia de atributos e treinamento deste
    pacote. Forneça o conjunto de dados à classe e aplique a partir dela
    todas as transformações e o gerenciamento de modelos.

    !!! warning
        Esta classe não pode ser instanciada diretamente. Use, em vez
        disso, as classes derivadas definidas em api.py.

    """

    @property
    @abstractmethod
    def _goal(self) -> Goal: ...

    def __init__(
        self,
        arrays,
        *,
        y: YSelector = -1,
        index: IndexSelector = False,
        metadata: MetadataDict | None = None,
        ignore: ColumnSelector | None = None,
        sp: Seasonality | SPDict = None,
        shuffle: Bool = True,
        stratify: Int | str | None = -1,
        n_rows: Scalar = 1,
        test_size: Scalar = 0.2,
        holdout_size: Scalar | None = None,
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

        self._config = DataConfig(
            index=index is not False,
            metadata=Bunch(
                **{
                    k: to_series(v, name=k, index=getattr(arrays[0], "index", None))  # type: ignore[call-overload]
                    for k, v in (metadata or {}).items()
                }
            ),
            shuffle=shuffle,
            stratify=stratify if shuffle else None,
            n_rows=n_rows,
            test_size=test_size,
            holdout_size=holdout_size,
        )

        # Inicializa o sistema de branches e o preenche com os dados
        self._branches = BranchManager(memory=self.memory)
        self._branches.fill(*self._get_data(arrays, y=y, index=index))

        self.ignore = ignore  # type: ignore[assignment]
        self.sp = sp  # type: ignore[assignment]

        if self.task is Task.binary_classification:
            self.pos_label = self.branch.dataset[lst(self.branch.target)[0]].max()

        self.missing = DEFAULT_MISSING

        self._models = ClassMap()
        self._metric = ClassMap()

        self._log("<< ================== ExperionML ================== >>", 1)
        self._log("\nConfiguração ==================== >>", 1)
        self._log(f"Tarefa do algoritmo: {self.task}.", 1)
        if self.n_jobs > 1:
            self._log(f"Processamento paralelo com {self.n_jobs} núcleos.", 1)
        elif self.backend != "loky":
            self._log(
                "Manter n_jobs=1 ignora toda paralelização. Defina n_jobs>1 para "
                f"usar o backend de paralelização {self.backend}.",
                1,
                severity="warning",
            )
        if "cpu" not in self.device.lower():
            self._log(f"Dispositivo: {self.device}", 1)
        if self.engine.data != EngineTuple().data:
            self._log(f"Engine de dados: {self.engine.data}", 1)
        if self.engine.estimator != EngineTuple().estimator:
            self._log(f"Engine de estimadores: {self.engine.estimator}", 1)
        if self.backend != "loky" and self.n_jobs > 1:
            self._log(f"Backend de paralelização: {self.backend}", 1)
        if self.memory.location is not None:
            self._log(f"Armazenamento de cache: {os.path.join(self.memory.location, 'joblib')}", 1)
        if self.experiment:
            self._log(f"Experimento do Mlflow: {self.experiment}", 1)

        # Configurações do sistema apenas no logger
        self._log("\nInformações do sistema =========== >>", 3)
        self._log(f"Máquina: {machine()}", 3)
        self._log(f"SO: {platform()}", 3)
        self._log(f"Versão do Python: {python_version()}", 3)
        self._log(f"Build do Python: {python_build()}", 3)
        self._log(f"Versão do ExperionML: {__version__}", 3)

        # Adiciona linhas vazias em volta das estatísticas para melhor leitura
        self._log("", 1)
        self.stats(1)
        self._log("", 1)

    def __repr__(self) -> str:
        """Exibe uma visão geral das branches, dos modelos e das métricas."""
        out = f"{self.__class__.__name__}"
        out += "\n --> Branches:"
        if len(branches := self._branches.branches) == 1:
            out += f" {self.branch.name}"
        else:
            for branch in branches:
                out += f"\n   --> {branch.name}{' !' if branch is self.branch else ''}"
        out += f"\n --> Modelos: {', '.join(lst(self.models)) if self.models else None}"
        out += f"\n --> Métrica: {', '.join(lst(self.metric)) if self.metric else None}"

        return out

    def __iter__(self) -> Iterator[Transformer]:
        """Percorre os transformadores no pipeline."""
        yield from self.pipeline.named_steps.values()

    # Utility properties =========================================== >>

    @property
    def branch(self) -> Branch:
        """Branch ativa atual.

        Use o `@setter` da propriedade para mudar de branch ou criar
        uma nova. Se o valor for o nome de uma branch existente, a
        instância troca para ela. Caso contrário, uma nova branch é
        criada com esse nome. A nova branch é derivada da branch atual.
        Use `_from_` para derivar a nova branch a partir de qualquer
        outra branch existente. Leia mais no [guia do usuário][branches].

        """
        return super().branch

    @branch.setter
    def branch(self, name: str):
        """Troca de branch ou cria uma nova."""
        if name in self._branches:
            if self.branch is self._branches[name]:
                self._log(f"Já está na branch {self.branch.name}.", 1)
            else:
                self._branches.current = name  # type: ignore[assignment]
                self._log(f"Alternado para a branch {self.branch.name}.", 1)
        else:
            # A branch pode ser criada a partir da atual ou de outra
            if "_from_" in name:
                new_name, parent_name = name.split("_from_")

                # Verifica se a branch pai existe
                if parent_name not in self._branches:
                    raise ValueError(
                        "A branch selecionada para derivação não existe. Use "
                        "experionml.status() para ver uma visão geral das branches disponíveis."
                    )
                else:
                    parent = self._branches[parent_name]

            else:
                new_name, parent = name, self.branch

            # Verifica se a nova branch ainda não existe
            if new_name in self._branches:  # Pode acontecer ao usar _from_
                raise ValueError(
                    f"A branch {new_name} já existe. Tente usar um nome "
                    "diferente. Note que nomes de branch não diferenciam maiúsculas de minúsculas."
                )

            self._branches.add(name=new_name, parent=parent)
            self._log(f"Nova branch criada com sucesso: {new_name}.", 1)

    @branch.deleter
    def branch(self):
        """Exclui a branch ativa atual."""
        if len(self._branches) == 1:
            raise PermissionError("Não é possível excluir a última branch.")

        # Delete all depending models
        for model in self._models:
            if model.branch is self.branch:
                self._delete_models(model.name)

        current = self.branch.name
        self._branches.branches.remove(current)
        self._branches.current = self._branches[0].name
        self._log(
            f"A branch {current} foi excluída com sucesso. "
            f"Alternado para a branch {self.branch.name}.",
            1,
        )

    @property
    def pos_label(self) -> PosLabel:
        """Rótulo positivo para tarefas de classificação binária/multirrótulo."""
        return self._config.pos_label

    @pos_label.setter
    @beartype
    def pos_label(self, value: PosLabel):
        if not self.task.is_binary:
            raise ValueError(
                "A propriedade pos_label só pode ser definida para "
                "tarefas de classificação binária/multirrótulo."
            )

        self._config.pos_label = value

    @property
    def metadata(self) -> Bunch:
        """Metadados do conjunto de dados.

        Leia mais no [guia do usuário][metadata].

        """
        return self._config.metadata

    @metadata.setter
    def metadata(self, value: MetadataDict | None):
        self._config.metadata = Bunch(
            **{k: to_series(v, index=self.y.index, name=k) for k, v in (value or {}).items()}  # type: ignore[call-overload]
        )

    @property
    def ignore(self) -> tuple[str, ...]:
        """Nomes das colunas ignoradas.

        Essas colunas não são usadas no pipeline de transformadores nem
        no treinamento dos modelos.

        """
        return self._config.ignore

    @ignore.setter
    def ignore(self, value: ColumnSelector | None):
        if value is not None:
            self._config.ignore = tuple(self.branch._get_columns(value, include_target=False))
        else:
            self._config.ignore = ()

    @property
    def missing(self) -> list[Any]:
        """Valores considerados "ausentes".

        Esses valores são usados pelos métodos [clean][self-clean] e
        [impute][self-impute]. Os valores padrão são: None, NaN,
        NA, NaT, +inf, -inf, "", "?", "NA", "nan", "NaN", "NaT", "none",
        "None", "inf", "-inf". Observe que None, NaN, NA, +inf e -inf
        são sempre considerados ausentes, pois são incompatíveis com
        estimadores do sklearn.

        """
        return self._missing

    @missing.setter
    def missing(self, value: Sequence[Any]):
        self._missing = list(value)

    @property
    def scaled(self) -> bool:
        """Indica se o conjunto de atributos está escalonado.

        Um conjunto de dados é considerado escalonado quando tem média~0
        e desvio padrão~1, ou quando existe um escalonador no pipeline.
        Colunas categóricas e binárias (apenas zeros e uns) são
        excluídas do cálculo. [Conjuntos de dados esparsos][] sempre
        retornam False.

        """
        return self.branch.check_scaling()

    @property
    def duplicates(self) -> int:
        """Número de linhas duplicadas no conjunto de dados."""
        return int(self.branch.dataset.duplicated().sum())

    @property
    def nans(self) -> pd.Series:
        """Colunas com a quantidade de valores ausentes.

        Esta propriedade não está disponível para [conjuntos de dados esparsos][].

        """
        if not is_sparse(self.branch.dataset):
            return replace_missing(self.branch.dataset, self.missing).isna().sum()

        raise AttributeError(
            "Esta propriedade não está disponível para conjuntos de dados esparsos."
        )

    @property
    def n_nans(self) -> int:
        """Número de linhas que contêm valores ausentes.

        Esta propriedade não está disponível para [conjuntos de dados esparsos][].

        """
        if not is_sparse(self.branch.dataset):
            nans = replace_missing(self.branch.dataset, self.missing).isna().sum(axis=1)
            return len(nans[nans > 0])

        raise AttributeError(
            "Esta propriedade não está disponível para conjuntos de dados esparsos."
        )

    @property
    def numerical(self) -> pd.Index:
        """Nomes dos atributos numéricos no conjunto de dados."""
        return self.branch.X.select_dtypes(include=["number"]).columns

    @property
    def n_numerical(self) -> int:
        """Número de atributos numéricos no conjunto de dados."""
        return len(self.numerical)

    @property
    def categorical(self) -> pd.Index:
        """Nomes dos atributos categóricos no conjunto de dados."""
        return self.branch.X.select_dtypes(include=CAT_TYPES).columns  # type: ignore[arg-type]

    @property
    def n_categorical(self) -> int:
        """Número de atributos categóricos no conjunto de dados."""
        return len(self.categorical)

    @property
    def outliers(self) -> pd.Series:
        """Colunas no conjunto de treino com quantidade de valores atípicos.

        Esta propriedade não está disponível para [conjuntos de dados esparsos][].

        """
        if not is_sparse(self.branch.X):
            data = self.branch.train.select_dtypes(include=["number"])
            z_scores = np.abs(stats.zscore(data.to_numpy(float, na_value=np.nan))) > 3
            z_scores = pd.Series(z_scores.sum(axis=0), index=data.columns)
            return z_scores[z_scores > 0]

        raise AttributeError(
            "Esta propriedade não está disponível para conjuntos de dados esparsos."
        )

    @property
    def n_outliers(self) -> int:
        """Número de amostras no conjunto de treino que contêm valores atípicos.

        Esta propriedade não está disponível para [conjuntos de dados esparsos][].

        """
        if not is_sparse(self.branch.X):
            data = self.branch.train.select_dtypes(include=["number"])
            z_scores = np.abs(stats.zscore(data.to_numpy(float, na_value=np.nan))) > 3
            return int(z_scores.any(axis=1).sum())

        raise AttributeError(
            "Esta propriedade não está disponível para conjuntos de dados esparsos."
        )

    @property
    def classes(self) -> pd.DataFrame:
        """Distribuição das classes-alvo por conjunto de dados.

        Esta propriedade só está disponível para tarefas de classificação.

        """
        if self.task.is_classification:
            index = []
            data = defaultdict(list)

            for col in lst(self.target):
                for ds in ("dataset", "train", "test"):
                    values, counts = np.unique(getattr(self, ds)[col], return_counts=True)
                    data[ds].extend(list(counts))
                index.extend([(col, i) for i in values])

            df = pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index))

            # Non-multioutput has single level index (for simplicity)
            if not self.task.is_multioutput:
                df.index = df.index.droplevel(0)

            return df.fillna(0).astype(
                int
            )  # Se não houver contagens, retorna NaN -> preencher com 0

        raise AttributeError("Esta propriedade não está disponível para tarefas de regressão.")

    @property
    def n_classes(self) -> Int | pd.Series:
        """Número de classes na(s) coluna(s)-alvo.

        Esta propriedade só está disponível para tarefas de classificação.

        """
        if self.task.is_classification:
            return self.branch.y.nunique(dropna=False)

        raise AttributeError("Esta propriedade não está disponível para tarefas de regressão.")

    # Utility methods =============================================== >>

    @available_if(has_task("forecast"))
    @crash
    def checks(self, *, columns: ColumnSelector | None = None) -> pd.DataFrame:
        """Obtém estatísticas sobre estacionariedade e ruído branco.

        Calcula vários testes estatísticos para verificar
        estacionariedade e ruído branco em uma coluna do conjunto de
        dados. Apenas para colunas numéricas. Valores ausentes são
        ignorados. Os testes realizados são:

        - [Teste Augmented Dickey-Fuller][adf] (adf) para
          estacionariedade por diferenciação.
        - [Teste Kwiatkowski-Phillips-Schmidt-Shin][kpss] (kpss) para
          estacionariedade em torno de tendência.
        - [Teste Ljung-Box][lb] (lb). É retornado o atraso na função de
          autocorrelação com o menor p-value. Se o p-value for maior que
          0.05, isso sugere que os dados são consistentes com ruído branco.

        !!! tip
            Use os métodos [plot_acf][] e [plot_acf][] para inspecionar
            visualmente quaisquer correlações defasadas significativas.

        Parâmetros
        ----------
        columns: int, str, segment, sequence, dataframe or None, default=None
            [Seleção de colunas][row-and-column-selection] nas quais os
            testes serão executados. Se None, usa a coluna-alvo.

        Retorna
        -------
        pd.DataFrame
            Resultados estatísticos com níveis de multiíndice:

            - **test**: Sigla do teste ('adf', 'kpss' ou 'lb').
            - **stat:** Resultados estatísticos:

                - **score:** Pontuação do teste.
                - **p_value:** p-value correspondente.

        """
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.stattools import adfuller, kpss

        columns_c = self.branch._get_columns(columns, only_numerical=True)

        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                iterables=(("adf", "kpss", "lb"), ("score", "p_value")),
                names=("test", "stat"),
            ),
            columns=columns_c,
        )

        for col in columns_c:
            # Remove valores ausentes da coluna antes de testar
            X = replace_missing(self[col], self.missing).dropna().to_numpy(dtype=float)

            for test in ("adf", "kpss", "lb"):
                if test == "adf":
                    stat = adfuller(X, maxlag=None, autolag="AIC")
                elif test == "kpss":
                    # regression='ct' corresponde à estacionariedade em torno de tendência
                    stat = kpss(X, regression="ct", nlags="auto")
                elif test == "lb":
                    l_jung = acorr_ljungbox(X, lags=None, period=lst(self.sp.get("sp"))[0])
                    stat = l_jung.loc[l_jung["lb_pvalue"].idxmin()]

                # Adiciona como coluna ao dataframe
                df.loc[(test, "score"), col] = round(stat[0], 4)
                df.loc[(test, "p_value"), col] = round(stat[1], 4)

        return df

    @crash
    def distributions(
        self,
        distributions: str | Sequence[str] | None = None,
        *,
        columns: ColumnSelector | None = None,
    ) -> pd.DataFrame:
        """Obtém estatísticas sobre distribuições das colunas.

        Calcula o [teste de Kolmogorov-Smirnov][kstest] para várias
        distribuições em colunas do conjunto de dados. Apenas para
        colunas numéricas. Valores ausentes são ignorados.

        !!! tip
            Use o método [plot_distribution][] para visualizar a
            distribuição de uma coluna.

        Parâmetros
        ----------
        distributions: str, sequence or None, default=None
            Nomes das distribuições em `scipy.stats` para as quais as
            estatísticas serão calculadas. Se None, é usada uma seleção
            das distribuições mais comuns.

        columns: int, str, segment, sequence, dataframe or None, default=None
            [Seleção de colunas][row-and-column-selection] nas quais o
            teste será executado. Se None, seleciona todas as colunas numéricas.

        Retorna
        -------
        pd.DataFrame
            Resultados estatísticos com níveis de multiíndice:

            - **dist:** Nome da distribuição.
            - **stat:** Resultados estatísticos:

                - **score:** Pontuação do teste KS.
                - **p_value:** p-value correspondente.

        """
        if distributions is None:
            distributions_c = [
                "beta",
                "expon",
                "gamma",
                "invgauss",
                "lognorm",
                "norm",
                "pearson3",
                "triang",
                "uniform",
                "weibull_min",
                "weibull_max",
            ]
        else:
            distributions_c = lst(distributions)

        columns_c = self.branch._get_columns(columns, only_numerical=True)

        df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                iterables=(distributions_c, ["score", "p_value"]),
                names=["dist", "stat"],
            ),
            columns=columns_c,
        )

        for col in columns_c:
            # Remove valores ausentes da coluna antes de testar
            X = replace_missing(self[col], self.missing).dropna().to_numpy(dtype=float)

            for dist in distributions_c:
                # Obtém a estatística KS com os parâmetros ajustados da distribuição
                stat = stats.kstest(X, dist, args=getattr(stats, dist).fit(X))

                # Adiciona como coluna ao dataframe
                df.loc[(dist, "score"), col] = round(stat[0], 4)
                df.loc[(dist, "p_value"), col] = round(stat[1], 4)

        return df

    @crash
    def eda(
        self,
        rows: str | Sequence[str] | dict[str, RowSelector] = "dataset",
        *,
        target: TargetSelector = 0,
        filename: str | Path | None = None,
    ):
        """Cria um relatório de Análise Exploratória de Dados.

        O ExperionML usa o pacote [sweetviz][] para EDA. O [relatório][] é
        renderizado diretamente no notebook. Ele também pode ser acessado
        pelo atributo `report`. É possível gerar o relatório de um único
        conjunto de dados ou comparar dois conjuntos entre si.

        !!! warning
            Este método pode ser lento em conjuntos de dados grandes.

        Parâmetros
        ----------
        rows: str, sequence or dict, default="dataset"
                        Seleção de linhas sobre as quais o relatório será calculado.

                        - Se str: Nome do conjunto de dados a ser reportado.
                        - Se sequence: Nomes de dois conjuntos de dados a comparar.
                        - Se dict: Nomes de até dois conjuntos de dados com a
                            [seleção de linhas][row-and-column-selection] correspondente.

        target: int or str, default=0
                        Coluna-alvo a ser analisada. Apenas para tarefas [multilabel][].
                        Somente atributos booleanos e numéricos podem ser usados como alvo.

        filename: str, Path or None, default=None
                        Nome do arquivo ou [pathlib.Path][] do arquivo (html) a ser salvo.
                        Se None, nada é salvo.

        """
        check_dependency("sweetviz")
        import sweetviz as sv

        self._log("Criando relatório de EDA...", 1)

        if isinstance(rows, str):
            rows_c = [(self.branch._get_rows(rows), rows)]
        elif isinstance(rows, sequence_t):
            rows_c = [(self.branch._get_rows(r), r) for r in rows]
        elif isinstance(rows, dict):
            rows_c = [(self.branch._get_rows(v), k) for k, v in rows.items()]

        if len(rows_c) == 1:
            self.report = sv.analyze(
                source=rows_c[0],
                target_feat=self.branch._get_target(target, only_columns=True),
            )
        elif len(rows_c) == 2:
            self.report = sv.compare(
                source=rows_c[0],
                compare=rows_c[1],
                target_feat=self.branch._get_target(target, only_columns=True),
            )
        else:
            raise ValueError(
                "Valor inválido para o parâmetro rows. O número máximo "
                f"de conjuntos de dados a usar é 2, recebido {len(rows_c)}."
            )

        if filename:
            if (path := Path(filename)).suffix != ".html":
                path = path.with_suffix(".html")

        self.report.show_notebook(filepath=path if filename else None)

    @composed(crash, method_to_log)
    def inverse_transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        *,
        verbose: Verbose | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Aplica a transformação inversa a novos dados pelo pipeline.

        Transformadores aplicados apenas ao conjunto de treino são
        ignorados. Os demais devem implementar um método
        `inverse_transform`. Se apenas `X` ou apenas `y` for fornecido,
        são ignorados os transformadores que exigem o outro parâmetro.
        Isso pode ser útil para transformar apenas a coluna-alvo.

        Parâmetros
        ----------
                X: conjunto de atributos transformado com shape=(n_samples, n_features).
                        Se None, `X` é ignorado pelos transformadores.

        y: int, str, sequence, dataframe-like or None, default=None
                        Coluna-alvo transformada correspondente a `X`.

                        - Se None: `y` é ignorado.
                        - Se int: Posição da coluna-alvo em `X`.
                        - Se str: Nome da coluna-alvo em `X`.
                        - Se sequence: Coluna-alvo com shape=(n_samples,) ou
                            sequência de nomes/posições de colunas para tarefas multioutput.
                        - Se dataframe-like: Colunas-alvo para tarefas multioutput.

        verbose: int or None, default=None
                        Nível de verbosidade dos transformadores no pipeline. Se
                        None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Conjunto de atributos original. Só é retornado se fornecido.

        series or dataframe
            Coluna-alvo original. Só é retornada se fornecida.

        """
        Xt, yt = self._check_input(X, y, columns=self.branch.features, name=self.branch.target)

        with adjust(self.pipeline, transform=self.engine.data, verbose=verbose) as pl:
            return pl.inverse_transform(Xt, yt)

    @classmethod
    def load(cls, filename: str | Path, data: tuple[Any, ...] | None = None) -> ExperionML:
        """Carrega uma instância de experionml a partir de um arquivo pickle.

        Se a instância foi [salva][self-save] usando `save_data=False`,
        é possível carregar novos dados nela e aplicar todas as
        transformações de dados.

        !!! info
            A branch atual da instância carregada é a mesma branch que
            estava ativa no momento em que ela foi salva.

        Parâmetros
        ----------
        filename: str or Path
            Nome do arquivo ou [pathlib.Path][] do arquivo pickle.

        data: tuple of indexables or None, default=None
            Conjunto de dados original como foi fornecido ao construtor
            da instância. Use este parâmetro apenas se o arquivo
            carregado foi salvo usando `save_data=False`. Os formatos
            permitidos são:

            - X
            - X, y
            - train, test
            - train, test, holdout
            - X_train, X_test, y_train, y_test
            - X_train, X_test, X_holdout, y_train, y_test, y_holdout
            - (X_train, y_train), (X_test, y_test)
            - (X_train, y_train), (X_test, y_test), (X_holdout, y_holdout)

                        **X, train, test: dataframe-like**<br>
                        Conjunto de atributos com shape=(n_samples, n_features).

            **y: int, str, sequence or dataframe**<br>
                        Coluna(s)-alvo correspondente(s) a `X`.

                        - Se int: Posição da coluna-alvo em `X`.
                        - Se str: Nome da coluna-alvo em `X`.
                        - Se sequence: Coluna-alvo com shape=(n_samples,) ou
                            sequência de nomes/posições de colunas para tarefas
                            multioutput.
                        - Se dataframe: Colunas-alvo para tarefas multioutput.

        Retorna
        -------
        experionml
            Instância experionml desserializada.

        """
        if (path := Path(filename)).suffix != ".pkl":
            path = path.with_suffix(".pkl")

        with open(path, "rb") as f:
            experionml = pickle.load(f)

        # Check if it's an experionml instance
        if not experionml.__class__.__name__.startswith("ExperionML"):
            raise ValueError(
                "A classe carregada não é uma instância de ExperionMLClassifier, "
                "ExperionMLRegressor nem ExperionMLForecaster, mas sim "
                f"{experionml.__class__.__name__}."
            )

        # Reatribui os atributos do transformador (warnings, random_state etc.)
        BaseTransformer.__init__(
            experionml,
            **{x: getattr(experionml, x) for x in BaseTransformer.attrs},
        )

        if data is not None:
            # Prepara os dados fornecidos
            container, holdout = experionml._get_data(data)

            # Atribui os dados à branch original
            if experionml._branches._og is not None:
                experionml._branches._og._container = container

            # Aplica as transformações por branch
            for branch in experionml._branches:
                if branch._container is None:
                    branch._container = deepcopy(container)
                    branch._holdout = holdout
                else:
                    raise ValueError(
                        f"A instância carregada de {experionml.__class__.__name__} "
                        f"já contém dados na branch {branch.name}."
                    )

                if len(experionml._branches) > 2 and branch.pipeline:
                    experionml._log(f"Transformando dados para a branch {branch.name}:", 1)

                X_train, y_train = branch.pipeline.transform(
                    X=branch.X_train,
                    y=branch.y_train,
                    filter_train_only=False,
                )
                X_test, y_test = branch.pipeline.transform(branch.X_test, branch.y_test)

                # Atualiza o conjunto de dados completo
                branch._container.data = pd.concat([merge(X_train, y_train), merge(X_test, y_test)])

                if experionml._config.index is False:
                    branch._container = DataContainer(
                        data=(dataset := branch._container.data.reset_index(drop=True)),
                        train_idx=dataset.index[: len(branch._container.train_idx)],
                        test_idx=dataset.index[-len(branch._container.test_idx) :],
                        n_targets=branch._container.n_targets,
                    )

                # Armazena branches inativas em memória
                if branch is not experionml.branch:
                    branch.store()

        experionml._log(f"{experionml.__class__.__name__} carregado com sucesso.", 1)

        return experionml

    @composed(crash, method_to_log)
    def reset(self, *, hard: Bool = False):
        """Redefine a instância para seu estado inicial.

        Exclui todas as branches e modelos. O conjunto de dados também
        é restaurado para o estado logo após a inicialização.

        Parâmetros
        ----------
        hard: bool, default=False
            Se True, limpa completamente o cache.

        """
        self._delete_models(self._get_models())
        self._branches.reset(hard=hard)
        self._log(f"{self.__class__.__name__} redefinido com sucesso.", 1)

    @composed(crash, method_to_log)
    def save_data(
        self,
        filename: str | Path = "auto",
        *,
        rows: RowSelector = "dataset",
        **kwargs,
    ):
        """Salva os dados da branch atual em um arquivo `.csv`.

        Parâmetros
        ----------
        filename: str or Path, default="auto"
            Nome do arquivo ou [pathlib.Path][] do arquivo a salvar.
            Use "auto" para nomeação automática.

        rows: hashable, segment, sequence or dataframe, default="dataset"
            [Seleção de linhas][row-and-column-selection] a ser salva.

        **kwargs
            Argumentos nomeados adicionais para o método [to_csv][] do pandas.

        """
        if (path := Path(filename)).suffix != ".csv":
            path = path.with_suffix(".csv")

        if path.name == "auto.csv":
            if isinstance(rows, str):
                path = path.with_name(f"{self.__class__.__name__}_{rows}.csv")
            else:
                path = path.with_name(f"{self.__class__.__name__}.csv")

        self.branch._get_rows(rows).to_csv(path, **kwargs)
        self._log("Conjunto de dados salvo com sucesso.", 1)

    @composed(crash, method_to_log)
    def shrink(
        self,
        *,
        int2bool: Bool = False,
        int2uint: Bool = False,
        str2cat: Bool = False,
        dense2sparse: Bool = False,
        columns: ColumnSelector | None = None,
    ):
        """Converte as colunas para o menor dtype compatível possível.

        Exemplos: float64 -> float32, int64 -> int8, etc. Arrays esparsos
        também transformam seu valor de preenchimento. Use este método para
        otimizar memória antes de [salvar][self-save_data] o conjunto de dados.
        Note que aplicar transformadores aos dados pode alterar os tipos
        novamente.

        Parâmetros
        ----------
        int2bool: bool, default=False
            Indica se colunas `int` devem ser convertidas para `bool`.
            Apenas se os valores na coluna estiverem estritamente em
            (0, 1) ou (-1, 1).

        int2uint: bool, default=False
            Indica se `int` deve ser convertido para `uint` (inteiro sem sinal).
            Apenas se os valores na coluna forem estritamente positivos.

        str2cat: bool, default=False
            Indica se `string` deve ser convertido para `category`.
            Apenas se o número de categorias for menor que 30% do
            comprimento da coluna.

        dense2sparse: bool, default=False
            Indica se todos os atributos devem ser convertidos para formato
            esparso. O valor comprimido é o valor mais frequente na coluna.

        columns: int, str, segment, sequence, dataframe or None, default=None
            [Seleção de colunas][row-and-column-selection] a reduzir. Se
            None, transforma todas as colunas.

        """

        def get_data(new_t: DtypeObj) -> pd.Series:
            """Obtém a série no formato de dados correto.

            Também converte para formato esparso se `dense2sparse=True`.

            Parâmetros
            ----------
            new_t: DtypeObj
                Objeto de tipo de dado para o qual converter.

            Retorna
            -------
            pd.Series
                Objeto com o novo tipo de dado.

            """
            new_t_np = str(new_t).lower()

            # Se já for um array esparso, converte diretamente para um novo tipo esparso
            if isinstance(column.dtype, pd.SparseDtype):
                # O subtipo de SparseDtype deve ser um dtype do numpy
                return column.astype(pd.SparseDtype(new_t_np, column.dtype.fill_value))

            if dense2sparse and name not in lst(self.target):  # Ignora colunas-alvo
                # Seleciona o valor mais frequente para preencher o array esparso
                fill_value = column.mode()[0]

                # Converte primeiro para um array esparso, senão falha para tipos anuláveis do pandas
                sparse_col = pd.arrays.SparseArray(column, fill_value=fill_value)

                return sparse_col.astype(pd.SparseDtype(new_t_np, fill_value=fill_value))
            else:
                return column.astype(new_t)

        t1 = (pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype)
        t2 = (pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype)
        t3 = (pd.Float32Dtype, pd.Float64Dtype)

        types = {
            "int": [(x.name, np.iinfo(x.type).min, np.iinfo(x.type).max) for x in t1],
            "uint": [(x.name, np.iinfo(x.type).min, np.iinfo(x.type).max) for x in t2],
            "float": [(x.name, np.finfo(x.type).min, np.finfo(x.type).max) for x in t3],
        }

        data = self.branch.dataset[self.branch._get_columns(columns)]

        # Converte para o melhor dtype anulável
        data = data.convert_dtypes()

        for name, column in data.items():
            # Obtém o subtipo de dtypes esparsos
            old_t = getattr(column.dtype, "subtype", column.dtype)

            if old_t.name.startswith("string"):
                if str2cat and column.nunique() <= int(len(column) * 0.3):
                    self.branch._data.data[name] = get_data(pd.CategoricalDtype())
                    continue

            try:
                # Obtém os tipos a considerar
                t = next(v for k, v in types.items() if old_t.name.lower().startswith(k))
            except StopIteration:
                self.branch._data.data[name] = get_data(column.dtype)
                continue

            # Usa bool se os valores estiverem em (0, 1)
            if int2bool and (t == types["int"] or t == types["uint"]):
                if column.isin([0, 1]).all() or column.isin([-1, 1]).all():
                    self.branch._data.data[name] = get_data(pd.BooleanDtype())
                    continue

            # Usa uint se os valores forem estritamente positivos
            if int2uint and t == types["int"] and column.min() >= 0:
                t = types["uint"]

            # Encontra o menor tipo compatível
            self.branch._data.data[name] = next(
                get_data(r[0]) for r in t if r[1] <= column.min() and r[2] >= column.max()
            )

        self._log("Os dtypes das colunas foram convertidos com sucesso.", 1)

    @composed(crash, method_to_log)
    def stats(self, _vb: Int = -2, /):
        """Exibe informações básicas sobre o conjunto de dados.

        Parâmetros
        ----------
        _vb: int, default=-2
            Parâmetro interno para sempre imprimir quando chamado pelo usuário.

        """
        self._log("Estatísticas do conjunto de dados " + "=" * 20 + " >>", _vb)
        self._log(f"Formato: {self.branch.shape}", _vb)
        if self.task.is_forecast and self.sp.get("sp"):
            self._log(f"Período sazonal: {self.sp.sp}", _vb)

        for ds in ("train", "test", "holdout"):
            if (data := getattr(self.branch, ds)) is not None:
                self._log(f"Tamanho do conjunto de {ds}: {len(data)}", _vb)
                if self.task.is_forecast:
                    self._log(f" --> De: {min(data.index)}  Até: {max(data.index)}", _vb)

        self._log("-" * 37, _vb)
        if (memory := self.branch.dataset.memory_usage().sum()) < 1e6:
            self._log(f"Memória: {memory / 1e3:.2f} kB", _vb)
        else:
            self._log(f"Memória: {memory / 1e6:.2f} MB", _vb)

        if is_sparse(self.branch.X):
            self._log("Esparso: True", _vb)
            if hasattr(self.branch.X, "sparse"):  # All columns are sparse
                self._log(f"Densidade: {100. * self.branch.X.sparse.density:.2f}%", _vb)
            else:  # Not all columns are sparse
                n_sparse = sum(isinstance(self[c].dtype, pd.SparseDtype) for c in self.features)
                n_dense = self.n_features - n_sparse
                p_sparse = round(100 * n_sparse / self.n_features, 1)
                p_dense = round(100 * n_dense / self.n_features, 1)
                self._log(f"Atributos densos: {n_dense} ({p_dense}%)", _vb)
                self._log(f"Atributos esparsos: {n_sparse} ({p_sparse}%)", _vb)
        else:
            nans = self.nans.sum()
            n_categorical = self.n_categorical
            outliers = self.outliers.sum()
            try:  # Can fail for unhashable columns (e.g., multilabel with lists)
                duplicates = self.branch.dataset.duplicated().sum()
            except TypeError:
                duplicates = None
                self._log(
                    "Não foi possível calcular o número de linhas duplicadas "
                    "porque uma coluna não suporta hash.",
                    3,
                )

            if not self.branch.X.empty:
                self._log(f"Escalonado: {self.scaled}", _vb)
            if nans:
                p_nans = round(100 * nans / self.branch.dataset.size, 1)
                self._log(f"Valores ausentes: {nans} ({p_nans}%)", _vb)
            if n_categorical:
                p_cat = round(100 * n_categorical / self.n_features, 1)
                self._log(f"Atributos categóricos: {n_categorical} ({p_cat}%)", _vb)
            if outliers:
                p_out = round(100 * outliers / self.branch.train.size, 1)
                self._log(f"Valores atípicos: {outliers} ({p_out}%)", _vb)
            if duplicates:
                p_dup = round(100 * duplicates / len(self.branch.dataset), 1)
                self._log(f"Duplicatas: {duplicates} ({p_dup}%)", _vb)

    @composed(crash, method_to_log)
    def status(self):
        r"""Obtém uma visão geral das branches e dos modelos.

        Este método imprime as mesmas informações de \__repr__ e
        também as salva no logger.

        """
        self._log(str(self))

    @composed(crash, method_to_log)
    def transform(
        self,
        X: XSelector | None = None,
        y: YSelector | None = None,
        *,
        verbose: Verbose | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Transforma novos dados pelo pipeline.

        Transformadores aplicados apenas ao conjunto de treino são
        ignorados. Se apenas `X` ou apenas `y` for fornecido, são
        ignorados os transformadores que exigem o outro parâmetro.
        Isso pode ser útil, por exemplo, para transformar apenas a coluna-alvo.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
                        Conjunto de atributos com shape=(n_samples, n_features). Se None,
                        `X` é ignorado.

        y: int, str, sequence, dataframe-like or None, default=None
                        Coluna(s)-alvo correspondente(s) a `X`.

                        - Se None: `y` é ignorado.
                        - Se int: Posição da coluna-alvo em `X`.
                        - Se str: Nome da coluna-alvo em `X`.
                        - Se sequence: Coluna-alvo com shape=(n_samples,) ou
                            sequência de nomes/posições de colunas para tarefas multioutput.
                        - Se dataframe-like: Colunas-alvo para tarefas multioutput.

        verbose: int or None, default=None
                        Nível de verbosidade dos transformadores no pipeline. Se
                        None, usa a verbosidade do pipeline.

        Retorna
        -------
        dataframe
            Conjunto de atributos transformado. Só é retornado se fornecido.

        series or dataframe
            Coluna-alvo transformada. Só é retornada se fornecida.

        """
        Xt, yt = self._check_input(X, y, columns=self.og.features, name=self.og.target)

        with adjust(self.pipeline, transform=self.engine.data, verbose=verbose) as pl:
            return pl.transform(Xt, yt)

    # Base transformers ============================================ >>

    def _prepare_kwargs(
        self,
        kwargs: dict[str, Any],
        params: MappingProxyType | None = None,
        *,
        is_runner: Bool = False,
    ) -> dict[str, Any]:
        """Retorna kwargs com os valores do experionml quando não especificados.

        Este método é usado por todos os transformadores e runners para
        repassar às classes as propriedades do BaseTransformer do experionml.
        O parâmetro engine é o único modificado para não runners, já que os
        transformadores do ExperionML aceitam apenas a engine de estimadores.

        Parâmetros
        ----------
        kwargs: dict
            Argumentos nomeados especificados na chamada da função.

        params: mappingproxy or None, default=None
            Parâmetros na assinatura da classe.

        is_runner: bool, default=False
            Indica se os parâmetros são passados para um runner.

        Retorna
        -------
        dict
            Propriedades convertidas.

        """
        for attr in BaseTransformer.attrs:
            if (not params or attr in params) and attr not in kwargs:
                if attr == "engine" and not is_runner:
                    # O parâmetro engine é especial, pois não queremos
                    # alterar as engines de dados no pipeline
                    kwargs[attr] = getattr(self, attr).estimator
                else:
                    kwargs[attr] = getattr(self, attr)

        return kwargs

    def _add_transformer(
        self,
        transformer: T_Transformer,
        *,
        columns: ColumnSelector | None = None,
        train_only: Bool = False,
        feature_names_out: FeatureNamesOut = None,
        **fit_params,
    ) -> T_Transformer:
        """Adiciona um transformador ao pipeline.

        Se o transformador não estiver ajustado, ele é ajustado no
        conjunto de treino completo. Depois disso, o conjunto de dados
        é transformado e o transformador é adicionado ao pipeline do experionml.

        Parâmetros
        ----------
        transformer: Transformer
            Estimador a adicionar. Deve implementar um método `transform`.
            Se uma classe for fornecida (em vez de uma instância) e ela
            tiver os parâmetros `n_jobs` e/ou `random_state`, ela adota
            os valores do experionml.

        columns: int, str, segment, sequence, dataframe or None, default=None
            Colunas do conjunto de dados a transformar. Se None,
            transforma todos os atributos.

        train_only: bool, default=False
            Indica se o transformador deve ser aplicado apenas ao
            conjunto de treino ou ao conjunto de dados completo.

        feature_names_out: "one-to-one", callable or None, default=None
                        Determina a lista de nomes de atributos que será retornada
                        pelo método `get_feature_names_out`.

                        - Se None: O método `get_feature_names_out` não é definido.
                        - Se "one-to-one": Os nomes de saída serão iguais aos nomes
                            de entrada.
                        - Se callable: Função que recebe os argumentos posicionais self
                            e uma sequência de nomes de atributos de entrada. Deve
                            retornar uma sequência com os nomes de saída.

        **fit_params
            Argumentos nomeados adicionais para o método fit do transformador.

        Retorna
        -------
        Transformer
            Transformador ajustado.

        """
        if callable(transformer):
            transformer_c = self._inherit(transformer(), feature_names_out=feature_names_out)
        else:
            transformer_c = make_sklearn(transformer, feature_names_out=feature_names_out)

        if any(m.branch is self.branch for m in self._models):
            raise PermissionError(
                "Não é permitido adicionar transformadores à branch "
                "depois que ela foi usada para treinar modelos. Crie "
                "uma nova branch para continuar o pipeline."
            )

        if not hasattr(transformer_c, "_train_only"):
            transformer_c._train_only = train_only

        if columns is not None:
            cols = self.branch._get_columns(columns)
        else:
            cols = list(self.branch.features)

        # Colunas em self.ignore não são transformadas
        if self.ignore:
            cols = [c for c in cols if c not in self.ignore]

        if cols != list(self.branch.features):
            if any(c in self.features for c in cols) and any(c in lst(self.target) for c in cols):
                self._log(
                    "Colunas de atributos e alvo foram passadas ao transformador "
                    f"{transformer_c.__class__.__name__}. Selecione apenas atributos "
                    "ou a coluna-alvo, não ambos ao mesmo tempo. A transformação "
                    "da coluna-alvo será ignorada.",
                    1,
                    severity="warning",
                )
            transformer_c._cols = cols

        # Adiciona método de clonagem customizado para manter atributos internos
        transformer_c.__class__.__sklearn_clone__ = TransformerMixin.__sklearn_clone__

        if hasattr(transformer_c, "fit"):
            if not transformer_c.__module__.startswith("experionml"):
                self._log(f"Ajustando {transformer_c.__class__.__name__}...", 1)

            # Memoiza o transformador ajustado para instanciações repetidas do experionml
            fit = self.memory.cache(fit_one)

            if (sample_weight := self._config.get_sample_weight(self.branch.X_train)) is not None:
                if "sample_weight" in sign(transformer_c.fit):
                    fit_params["sample_weight"] = sample_weight
                if hasattr(transformer_c, "set_fit_request"):
                    transformer_c.set_fit_request(sample_weight=sample_weight is not None)

            kwargs = {
                "estimator": transformer_c,
                "X": self.branch.X_train,
                "y": self.branch.y_train,
                **fit_params,
            }

            # Verifica se o estimador ajustado foi recuperado do cache para
            # informar o usuário, caso contrário pode parecer que nada foi impresso
            if fit.check_call_in_cache(**kwargs):
                self._log(
                    f"Carregando resultados em cache para {transformer_c.__class__.__name__}...",
                    1,
                )

            transformer_c = fit(**kwargs)

        # Se esta for a última branch vazia, cria uma nova branch 'og'
        if len([b for b in self._branches if not b.pipeline.steps]) == 1:
            self._branches.add("og")

        if transformer_c._train_only:
            X, y = self.pipeline._mem_transform(
                transformer=transformer_c,
                X=self.branch.X_train,
                y=self.branch.y_train,
            )

            self.branch.train = merge(
                self.branch.X_train if X is None else X,
                self.branch.y_train if y is None else y,
            )
        else:
            X, y = self.pipeline._mem_transform(transformer_c, self.branch.X, self.branch.y)
            data = merge(self.branch.X if X is None else X, self.branch.y if y is None else y)

            # y pode alterar o número de colunas ou remover linhas -> reatribui o índice
            self._branches.fill(
                DataContainer(
                    data=data,
                    train_idx=self.branch._data.train_idx.intersection(data.index),
                    test_idx=self.branch._data.test_idx.intersection(data.index),
                    n_targets=self.branch._data.n_targets if y is None else n_cols(y),
                )
            )

        if self._config.index is False:
            self._branches.fill(
                DataContainer(
                    data=(data := self.branch.dataset.reset_index(drop=True)),
                    train_idx=data.index[: len(self.branch._data.train_idx)],
                    test_idx=data.index[-len(self.branch._data.test_idx) :],
                    n_targets=self.branch._data.n_targets,
                )
            )
            if self.branch._holdout is not None:
                self.branch._holdout.index = pd.Index(
                    range(len(data), len(data) + len(self.branch._holdout))
                )
        elif self.branch.dataset.index.duplicated().any():
            raise ValueError(
                "Foram encontrados índices duplicados no conjunto de dados. "
                "Tente inicializar o experionml com `index=False`."
            )

        # Adiciona o transformador ao pipeline
        # Verifica se já existe um estimador com esse nome.
        # Se existir, adiciona um contador ao final do nome
        counter = 1
        name = transformer_c.__class__.__name__.lower()
        while name in self.pipeline:
            counter += 1
            name = f"{transformer_c.__class__.__name__.lower()}-{counter}"

        self.branch.pipeline.steps.append((name, transformer_c))

        # Anexa os atributos do transformador do experionml à branch
        if "experionml" in transformer_c.__module__:
            attrs = ("mapping_", "feature_names_in_", "n_features_in_")
            for name, value in vars(transformer_c).items():
                if not name.startswith("_") and name.endswith("_") and name not in attrs:
                    setattr(self.branch, name, value)

        return transformer_c

    @composed(crash, method_to_log)
    def add(
        self,
        transformer: Transformer,
        *,
        columns: ColumnSelector | None = None,
        train_only: Bool = False,
        feature_names_out: FeatureNamesOut = None,
        **fit_params,
    ):
        """Adiciona um transformador ao pipeline.

        Se o transformador não estiver ajustado, ele é ajustado no conjunto
        de treino completo. Em seguida, o conjunto de dados é transformado e
        o estimador é adicionado ao pipeline do experionml. Se o estimador for
        um Pipeline do sklearn, cada estimador é incorporado individualmente.

        !!! warning

                        * O transformador deve ter métodos fit e/ou transform com
                            argumentos `X` (aceitando um objeto dataframe-like com
                            shape=(n_samples, n_features)) e/ou `y` (aceitando uma
                            sequência com shape=(n_samples,)).
                        * O método transform deve retornar um conjunto de atributos
                            como objeto dataframe-like com shape=(n_samples, n_features)
                            e/ou uma coluna-alvo como sequência com shape=(n_samples,).

        !!! note
                        Se o método transform não retornar um dataframe:

                        * A nomeação das colunas acontece da seguinte forma. Se o
                            transformador tiver um método `get_feature_names_out`, ele
                            será usado. Caso contrário, se o número de colunas for o
                            mesmo, os nomes são preservados. Se o número de colunas
                            mudar, colunas antigas mantêm seus nomes (desde que a coluna
                            permaneça inalterada) e novas colunas recebem o nome
                            `x[N-1]`, em que N representa o enésimo atributo. Isso
                            significa que um transformador deve apenas transformar,
                            adicionar ou remover colunas, e não combinar essas ações.
                        * O índice permanece o mesmo de antes da transformação. Isso
                            significa que o transformador não deve adicionar, remover
                            ou embaralhar linhas, a menos que retorne um dataframe.

        Parâmetros
        ----------
        transformer: Transformer
            Estimador a adicionar ao pipeline. Deve implementar um
            método `transform`. Se uma classe for fornecida (em vez de
            uma instância) e ela tiver os parâmetros `n_jobs` e/ou
            `random_state`, ela adota os valores do experionml.

        columns: int, str, segment, sequence, dataframe or None, default=None
            [Seleção de colunas][row-and-column-selection] a
            transformar. Selecione apenas atributos ou a coluna-alvo,
            não ambos ao mesmo tempo (se isso acontecer, a coluna-alvo
            será ignorada). Se None, transforma todas as colunas.

        train_only: bool, default=False
            Indica se o estimador deve ser aplicado apenas ao conjunto
            de treino ou ao conjunto de dados completo. Note que, se
            True, a transformação é ignorada ao fazer previsões em novos dados.

        feature_names_out: "one-to-one", callable or None, default=None
                        Determina a lista de nomes de atributos retornada pelo
                        método `get_feature_names_out`.

                        - Se None: O método `get_feature_names_out` não é definido.
                        - Se "one-to-one": Os nomes de saída serão iguais aos de entrada.
                        - Se callable: Função que recebe os argumentos posicionais self
                            e uma sequência de nomes de atributos de entrada. Deve
                            retornar uma sequência com os nomes de atributos de saída.

        **fit_params
            Argumentos nomeados adicionais para o método fit do transformador.

        """
        if isinstance(transformer, SkPipeline):
            # Adiciona recursivamente todos os transformadores ao pipeline
            for est in transformer.named_steps.values():
                self._log(f"Adicionando {est.__class__.__name__} ao pipeline...", 1)
                self._add_transformer(
                    transformer=est,
                    columns=columns,
                    train_only=train_only,
                    feature_names_out=feature_names_out,
                    **fit_params,
                )
        else:
            self._log(f"Adicionando {transformer.__class__.__name__} ao pipeline...", 1)
            self._add_transformer(
                transformer=transformer,
                columns=columns,
                train_only=train_only,
                feature_names_out=feature_names_out,
                **fit_params,
            )

    @composed(crash, method_to_log)
    def apply(
        self,
        func: Callable[..., pd.DataFrame],
        inverse_func: Callable[..., pd.DataFrame] | None = None,
        *,
        feature_names_out: FeatureNamesOut = None,
        kw_args: dict[str, Any] | None = None,
        inv_kw_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Aplica uma função ao conjunto de dados.

        Este método é útil para transformações sem estado, como aplicar
        log, fazer escalonamento customizado etc.

        !!! note
            Esta abordagem é preferível a alterar o conjunto de dados
            diretamente pelo `@setter` da propriedade, já que a
            transformação fica armazenada no pipeline.

        !!! tip
            Use `#!python experionml.apply(lambda df: df.drop("column_name",
            axis=1))` para armazenar a remoção de colunas no pipeline.

        Parâmetros
        ----------
        func: callable
            Função a aplicar com assinatura `func(dataframe, **kw_args)
            -> dataframe-like`.

        inverse_func: callable or None, default=None
            Função inversa de `func`. Se None, o método inverse_transform
            retorna a entrada sem alterações.

        feature_names_out: "one-to-one", callable or None, default=None
                        Determina a lista de nomes de atributos retornada pelo
                        método `get_feature_names_out`.

                        - Se None: O método `get_feature_names_out` não é definido.
                        - Se "one-to-one": Os nomes de saída serão iguais aos de entrada.
                        - Se callable: Função que recebe os argumentos posicionais self
                            e uma sequência de nomes de atributos de entrada. Deve
                            retornar uma sequência com os nomes de atributos de saída.


        kw_args: dict or None, default=None
            Argumentos nomeados adicionais para a função.

        inv_kw_args: dict or None, default=None
            Argumentos nomeados adicionais para a função inversa.

        """
        FunctionTransformer = self._get_est_class("FunctionTransformer", "preprocessing")

        columns = kwargs.pop("columns", None)
        transformer = FunctionTransformer(
            func=func,
            inverse_func=inverse_func,
            feature_names_out=feature_names_out,
            kw_args=kw_args,
            inv_kw_args=inv_kw_args,
        )

        self._add_transformer(transformer, columns=columns)  # type: ignore[type-var]

    # Data cleaning transformers =================================== >>

    @available_if(has_task(["classification", "!multioutput"]))
    @composed(crash, method_to_log)
    def balance(self, strategy: str | Estimator = "adasyn", **kwargs):
        """Balanceia o número de linhas por classe na coluna-alvo.

        Ao fazer oversampling, as amostras recém-criadas recebem um índice
        inteiro crescente para índices numéricos, e um índice no formato
        [estimator]_N para índices não numéricos, em que N representa a
        enésima amostra no conjunto de dados.

        Consulte a classe [Balancer][] para a descrição dos parâmetros.

        !!! warning
                        * O método balance não dá suporte a [tarefas multioutput][].
                        * O método balance não dá suporte a `sample_weights` passados
                            por roteamento de [metadata][].
                        * Esta transformação é aplicada apenas ao conjunto de treino
                            para manter a distribuição original das classes-alvo no teste.

        !!! tip
                        Use o atributo [classes][self-classes] do experionml para ver
                        a distribuição das classes-alvo por conjunto de dados.

        """
        if self._config.get_sample_weight() is not None:
            raise PermissionError(
                "O método balance não dá suporte a sample weights "
                "passados por roteamento de metadata."
            )

        columns = kwargs.pop("columns", None)
        balancer = Balancer(
            strategy=strategy,
            **self._prepare_kwargs(kwargs, sign(Balancer)),
        )

        # Adiciona o mapeamento da coluna-alvo para uma exibição mais limpa
        if mapping := self.mapping.get(self.target):
            balancer.mapping_ = mapping

        self._add_transformer(balancer, columns=columns)

    @composed(crash, method_to_log)
    def clean(
        self,
        *,
        convert_dtypes: Bool = True,
        drop_dtypes: str | Sequence[str] | None = None,
        drop_chars: str | None = None,
        strip_categorical: Bool = True,
        drop_duplicates: Bool = False,
        drop_missing_target: Bool = True,
        encode_target: Bool = True,
        **kwargs,
    ):
        """Aplica etapas padrão de limpeza de dados ao conjunto de dados.

        Use os parâmetros para escolher quais transformações executar.
        As etapas disponíveis são:

        - Converter dtypes para os melhores tipos possíveis.
        - Remover colunas com tipos de dados específicos.
        - Remover caracteres dos nomes das colunas.
        - Remover espaços de atributos categóricos.
        - Remover linhas duplicadas.
        - Remover linhas com valores ausentes na coluna-alvo.
        - Codificar a coluna-alvo (apenas para tarefas de classificação).

        Consulte a classe [Cleaner][] para a descrição dos parâmetros.

        """
        columns = kwargs.pop("columns", None)
        cleaner = Cleaner(
            convert_dtypes=convert_dtypes,
            drop_dtypes=drop_dtypes,
            drop_chars=drop_chars,
            strip_categorical=strip_categorical,
            drop_duplicates=drop_duplicates,
            drop_missing_target=drop_missing_target,
            encode_target=encode_target if self.task.is_classification else False,
            **self._prepare_kwargs(kwargs, sign(Cleaner)),
        )

        # Passa os valores ausentes do experionml ao cleaner antes da transformação
        cleaner.missing_ = self.missing

        cleaner = self._add_transformer(cleaner, columns=columns)
        self.branch._mapping.update(cleaner.mapping_)

        if self.task.is_binary and encode_target:
            self.pos_label = 1

    @composed(crash, method_to_log)
    def decompose(
        self,
        *,
        model: str | Predictor | None = None,
        test_seasonality: Bool = True,
        **kwargs,
    ):
        """Remove tendência e sazonalidade da série temporal.

        Esta classe faz duas coisas:

        - Remove a tendência de cada coluna, retornando os resíduos
          in-sample dos valores previstos pelo modelo.
        - Remove o componente sazonal de cada coluna, sujeito a um
          teste de sazonalidade.

        Colunas categóricas são ignoradas.

        Consulte a classe [Decomposer][] para a descrição dos parâmetros.
        O ExperionML injeta automaticamente na classe os parâmetros `sp`,
        `trend_model` e `seasonal_model`. Veja a seção [seasonality][]
        no guia do usuário para aprender a ajustar esses valores.

        !!! tip
            * Use o parâmetro `columns` para decompor apenas a coluna-alvo,
              por exemplo `experionml.decompose(columns=experionml.target)`.
            * Use o método [plot_decomposition][] para visualizar a
              tendência, sazonalidade e resíduos da série temporal.

        """
        columns = kwargs.pop("columns", None)
        decomposer = Decomposer(
            model=model,
            trend_model=self.sp.get("trend_model", "additive"),
            test_seasonality=test_seasonality,
            sp=lst(self.sp.get("sp"))[0],
            seasonal_model=self.sp.get("seasonal_model", "additive"),
            **self._prepare_kwargs(kwargs, sign(Decomposer)),
        )

        self._add_transformer(decomposer, columns=columns)

    @composed(crash, method_to_log)
    def discretize(
        self,
        strategy: DiscretizerStrats = "quantile",
        *,
        bins: Bins = 5,
        labels: Sequence[str] | dict[str, Sequence[str]] | None = None,
        **kwargs,
    ):
        """Agrupa dados contínuos em intervalos.

        Para cada atributo, os limites dos intervalos são calculados
        durante o ajuste e, juntamente com o número de bins, definem os
        intervalos. Ignora colunas numéricas.

        Consulte a classe [Discretizer][] para a descrição dos parâmetros.

        !!! tip
            Use o método [plot_distribution][] para visualizar a
            distribuição de uma coluna e decidir os bins.

        """
        columns = kwargs.pop("columns", None)
        discretizer = Discretizer(
            strategy=strategy,
            bins=bins,
            labels=labels,
            **self._prepare_kwargs(kwargs, sign(Discretizer)),
        )

        self._add_transformer(discretizer, columns=columns)

    @composed(crash, method_to_log)
    def encode(
        self,
        strategy: str = "Target",
        *,
        max_onehot: IntLargerTwo | None = 10,
        ordinal: dict[str, Sequence[Any]] | None = None,
        infrequent_to_value: FloatLargerZero | None = None,
        value: str = "infrequent",
        **kwargs,
    ):
        """Executa a codificação de atributos categóricos.

        O tipo de codificação depende do número de classes na coluna:

        - Se n_classes=2 ou o atributo for ordinal, usa codificação ordinal.
        - Se 2 < n_classes <= `max_onehot`, usa one-hot encoding.
        - Se n_classes > `max_onehot`, usa a codificação definida em `strategy`.

        Valores ausentes são propagados para a coluna de saída. Classes
        desconhecidas encontradas durante a transformação são imputadas
        de acordo com a estratégia selecionada. Classes raras podem ser
        substituídas por um valor para evitar cardinalidade excessiva.

        Consulte a classe [Encoder][] para a descrição dos parâmetros.

        !!! note
            Este método codifica apenas os atributos categóricos. Ele não
            codifica a coluna-alvo. Para isso, use o método [clean][self-clean].

        !!! tip
            Use o atributo [categorical][self-categorical] para obter a
            lista de atributos categóricos no conjunto de dados.

        """
        columns = kwargs.pop("columns", None)
        encoder = Encoder(
            strategy=strategy,
            max_onehot=max_onehot,
            ordinal=ordinal,
            infrequent_to_value=infrequent_to_value,
            value=value,
            **self._prepare_kwargs(kwargs, sign(Encoder)),
        )

        encoder = self._add_transformer(encoder, columns=columns)
        self.branch._mapping.update(encoder.mapping_)

    @composed(crash, method_to_log)
    def impute(
        self,
        strat_num: Scalar | NumericalStrats = "mean",
        strat_cat: str | CategoricalStrats = "most_frequent",
        *,
        max_nan_rows: FloatLargerZero | None = None,
        max_nan_cols: FloatLargerZero | None = None,
        **kwargs,
    ):
        """Trata valores ausentes no conjunto de dados.

        Imputa ou remove valores ausentes de acordo com a estratégia
        selecionada. Também remove linhas e colunas com valores ausentes demais.

        Consulte a classe [Imputer][] para a descrição dos parâmetros.

        !!! tip
            - Use o atributo [nans][self-nans] para verificar a quantidade
              de valores ausentes por coluna.
            - Use o atributo [`missing`][self-missing] para customizar o
              que é considerado "valor ausente".

        """
        columns = kwargs.pop("columns", None)
        imputer = Imputer(
            strat_num=strat_num,
            strat_cat=strat_cat,
            max_nan_rows=max_nan_rows,
            max_nan_cols=max_nan_cols,
            **self._prepare_kwargs(kwargs, sign(Imputer)),
        )

        # Passa os valores ausentes do experionml ao imputer antes da transformação
        imputer.missing = self.missing

        self._add_transformer(imputer, columns=columns)

    @composed(crash, method_to_log)
    def normalize(self, strategy: NormalizerStrats = "yeojohnson", **kwargs):
        """Transforma os dados para seguirem uma distribuição Normal/Gaussiana.

        Esta transformação é útil para problemas de modelagem relacionados
        à heterocedasticidade (variância não constante) ou outras
        situações em que a normalidade é desejada. Valores ausentes são
        desconsiderados no ajuste e mantidos na transformação. Ignora
        colunas categóricas.

        Consulte a classe [Normalizer][] para a descrição dos parâmetros.

        !!! tip
            Use o método [plot_distribution][] para examinar a
            distribuição de uma coluna.

        """
        columns = kwargs.pop("columns", None)
        normalizer = Normalizer(
            strategy=strategy,
            **self._prepare_kwargs(kwargs, sign(Normalizer)),
        )

        self._add_transformer(normalizer, columns=columns)

    @composed(crash, method_to_log)
    def prune(
        self,
        strategy: PrunerStrats | Sequence[PrunerStrats] = "zscore",
        *,
        method: Scalar | Literal["drop", "minmax"] = "drop",
        max_sigma: FloatLargerZero = 3,
        include_target: Bool = False,
        **kwargs,
    ):
        """Remove valores atípicos do conjunto de treino.

        Substitui ou remove valores atípicos. A definição de outlier
        depende da estratégia selecionada e pode variar bastante de uma
        para outra. Ignora colunas categóricas.

        Consulte a classe [Pruner][] para a descrição dos parâmetros.

        !!! note
            Esta transformação é aplicada apenas ao conjunto de treino
            para manter a distribuição original das amostras no teste.

        !!! tip
            Use o atributo [outliers][self-outliers] para verificar a
            quantidade de valores atípicos por coluna.

        """
        columns = kwargs.pop("columns", None)
        pruner = Pruner(
            strategy=strategy,
            method=method,
            max_sigma=max_sigma,
            include_target=include_target,
            **self._prepare_kwargs(kwargs, sign(Pruner)),
        )

        self._add_transformer(pruner, columns=columns)

    @composed(crash, method_to_log)
    def scale(
        self,
        strategy: ScalerStrats = "standard",
        *,
        include_binary: Bool = False,
        **kwargs,
    ):
        """Escalona os dados.

        Aplica uma das estratégias de escalonamento do sklearn. Colunas
        categóricas são ignoradas.

        Consulte a classe [Scaler][] para a descrição dos parâmetros.

        !!! tip
            Use o atributo [scaled][self-scaled] para verificar se o
            conjunto de dados está escalonado.

        """
        columns = kwargs.pop("columns", None)
        scaler = Scaler(
            strategy=strategy,
            include_binary=include_binary,
            **self._prepare_kwargs(kwargs, sign(Scaler)),
        )

        self._add_transformer(scaler, columns=columns)

    # NLP transformers ============================================= >>

    @composed(crash, method_to_log)
    def textclean(
        self,
        *,
        decode: Bool = True,
        lower_case: Bool = True,
        drop_email: Bool = True,
        regex_email: str | None = None,
        drop_url: Bool = True,
        regex_url: str | None = None,
        drop_html: Bool = True,
        regex_html: str | None = None,
        drop_emoji: Bool = True,
        regex_emoji: str | None = None,
        drop_number: Bool = True,
        regex_number: str | None = None,
        drop_punctuation: Bool = True,
        **kwargs,
    ):
        """Aplica uma limpeza textual padrão ao corpus.

        As transformações incluem normalizar caracteres e remover ruídos
        do texto (e-mails, tags HTML, URLs etc.). As transformações são
        aplicadas à coluna chamada `corpus`, na mesma ordem em que os
        parâmetros são apresentados. Se não houver uma coluna com esse
        nome, uma exceção é lançada.

        Consulte a classe [TextCleaner][] para a descrição dos
        parâmetros.

        """
        columns = kwargs.pop("columns", None)
        textcleaner = TextCleaner(
            decode=decode,
            lower_case=lower_case,
            drop_email=drop_email,
            regex_email=regex_email,
            drop_url=drop_url,
            regex_url=regex_url,
            drop_html=drop_html,
            regex_html=regex_html,
            drop_emoji=drop_emoji,
            regex_emoji=regex_emoji,
            drop_number=drop_number,
            regex_number=regex_number,
            drop_punctuation=drop_punctuation,
            **self._prepare_kwargs(kwargs, sign(TextCleaner)),
        )

        self._add_transformer(textcleaner, columns=columns)

    @composed(crash, method_to_log)
    def textnormalize(
        self,
        *,
        stopwords: Bool | str = True,
        custom_stopwords: Sequence[str] | None = None,
        stem: Bool | str = False,
        lemmatize: Bool = True,
        **kwargs,
    ):
        """Normaliza o corpus.

        Converte as palavras para um padrão mais uniforme. As
        transformações são aplicadas à coluna chamada `corpus`, na mesma
        ordem em que os parâmetros são apresentados. Se não houver uma
        coluna com esse nome, uma exceção é lançada. Se os documentos
        fornecidos forem strings, as palavras são separadas por espaços.

        Consulte a classe [TextNormalizer][] para a descrição dos
        parâmetros.

        """
        columns = kwargs.pop("columns", None)
        normalizer = TextNormalizer(
            stopwords=stopwords,
            custom_stopwords=custom_stopwords,
            stem=stem,
            lemmatize=lemmatize,
            **self._prepare_kwargs(kwargs, sign(TextNormalizer)),
        )

        self._add_transformer(normalizer, columns=columns)

    @composed(crash, method_to_log)
    def tokenize(
        self,
        bigram_freq: FloatLargerZero | None = None,
        trigram_freq: FloatLargerZero | None = None,
        quadgram_freq: FloatLargerZero | None = None,
        **kwargs,
    ):
        """Tokeniza o corpus.

        Converte documentos em sequências de palavras. Além disso,
        cria n-gramas (representados por palavras unidas por underscore,
        por exemplo "New_York") com base em sua frequência no corpus. As
        transformações são aplicadas à coluna chamada `corpus`. Se não
        houver uma coluna com esse nome, uma exceção é lançada.

        Consulte a classe [Tokenizer][] para a descrição dos parâmetros.

        """
        columns = kwargs.pop("columns", None)
        tokenizer = Tokenizer(
            bigram_freq=bigram_freq,
            trigram_freq=trigram_freq,
            quadgram_freq=quadgram_freq,
            **self._prepare_kwargs(kwargs, sign(Tokenizer)),
        )

        self._add_transformer(tokenizer, columns=columns)

    @composed(crash, method_to_log)
    def vectorize(
        self,
        strategy: VectorizerStarts = "bow",
        *,
        return_sparse: Bool = True,
        **kwargs,
    ):
        """Vetoriza o corpus.

        Transforma o corpus em vetores numéricos significativos. A
        transformação é aplicada à coluna chamada `corpus`. Se não
        houver uma coluna com esse nome, uma exceção é lançada.

        Se strategy="bow" ou "tfidf", as colunas transformadas recebem
        o nome da palavra que representam com o prefixo `corpus_`. Se
        strategy="hashing", as colunas recebem o nome hash[N], em que N
        representa a enésima coluna hash.

        Consulte a classe [Vectorizer][] para a descrição dos
        parâmetros.

        """
        columns = kwargs.pop("columns", None)
        vectorizer = Vectorizer(
            strategy=strategy,
            return_sparse=return_sparse,
            **self._prepare_kwargs(kwargs, sign(Vectorizer)),
        )

        self._add_transformer(vectorizer, columns=columns)

    # Feature engineering transformers ============================= >>

    @composed(crash, method_to_log)
    def feature_extraction(
        self,
        features: str | Sequence[str] = ("day", "month", "year"),
        fmt: str | dict[str, str] | None = None,
        *,
        encoding_type: Literal["ordinal", "cyclic"] = "ordinal",
        drop_columns: Bool = True,
        from_index: Bool = False,
        **kwargs,
    ):
        """Extrai atributos de colunas datetime.

        Cria novos atributos extraindo elementos de data e hora (dia,
        mês, ano etc.) das colunas fornecidas. Colunas do tipo
        `datetime64` são usadas diretamente. Colunas categóricas que
        possam ser convertidas com sucesso para um formato datetime
        (menos de 30% de valores NaT após a conversão) também são usadas.

        Consulte a classe [FeatureExtractor][] para a descrição dos
        parâmetros.

        """
        columns = kwargs.pop("columns", None)
        feature_extractor = FeatureExtractor(
            features=features,
            fmt=fmt,
            encoding_type=encoding_type,
            drop_columns=drop_columns,
            from_index=from_index,
            **self._prepare_kwargs(kwargs, sign(FeatureExtractor)),
        )

        self._add_transformer(feature_extractor, columns=columns)

    @composed(crash, method_to_log)
    def feature_generation(
        self,
        strategy: Literal["dfs", "gfg"] = "dfs",
        *,
        n_features: IntLargerZero | None = None,
        operators: Operators | Sequence[Operators] | None = None,
        **kwargs,
    ):
        """Gera novos atributos.

        Cria novas combinações de atributos existentes para capturar
        relações não lineares entre os atributos originais.

        Consulte a classe [FeatureGenerator][] para a descrição dos
        parâmetros.

        """
        columns = kwargs.pop("columns", None)
        feature_generator = FeatureGenerator(
            strategy=strategy,
            n_features=n_features,
            operators=operators,
            **self._prepare_kwargs(kwargs, sign(FeatureGenerator)),
        )

        self._add_transformer(feature_generator, columns=columns)

    @composed(crash, method_to_log)
    def feature_grouping(
        self,
        groups: dict[str, ColumnSelector],
        *,
        operators: str | Sequence[str] | None = None,
        drop_columns: Bool = True,
        **kwargs,
    ):
        """Extrai estatísticas de atributos semelhantes.

        Substitui grupos de atributos com características relacionadas
        por novos atributos que resumem propriedades estatísticas do
        grupo. Os operadores estatísticos são calculados em cada linha
        do grupo. Os nomes dos grupos e seus atributos podem ser
        acessados pelo método `groups`.

        Consulte a classe [FeatureGrouper][] para a descrição dos
        parâmetros.

        !!! tip
            Use um padrão regex no parâmetro `groups` para selecionar
            grupos mais facilmente, por exemplo
            `experionml.feature_grouping({"group1": "var_.+"})`
            para selecionar todos os atributos que começam com `var_`.

        """
        columns = kwargs.pop("columns", None)
        feature_grouper = FeatureGrouper(
            groups={
                name: self.branch._get_columns(fxs, include_target=False)
                for name, fxs in groups.items()
            },
            operators=operators,
            drop_columns=drop_columns,
            **self._prepare_kwargs(kwargs, sign(FeatureGrouper)),
        )

        self._add_transformer(feature_grouper, columns=columns)

    @composed(crash, method_to_log)
    def feature_selection(
        self,
        strategy: FeatureSelectionStrats | None = None,
        *,
        solver: FeatureSelectionSolvers = None,
        n_features: FloatLargerZero | None = None,
        min_repeated: FloatLargerEqualZero | None = 2,
        max_repeated: FloatLargerEqualZero | None = 1.0,
        max_correlation: FloatZeroToOneInc | None = 1.0,
        **kwargs,
    ):
        """Reduz o número de atributos nos dados.

        Aplica seleção de atributos ou redução de dimensionalidade, seja
        para melhorar a acurácia dos estimadores, seja para aumentar o
        desempenho em conjuntos de dados de alta dimensionalidade.
        Além disso, remove atributos multicolineares e de baixa variância.

        Consulte a classe [FeatureSelector][] para a descrição dos
        parâmetros.

        !!! note
            * Quando strategy="univariate" e solver=None, [f_classif][]
              ou [f_regression][] é usado como solver padrão.
            * Quando a estratégia é "sfs", "rfecv" ou qualquer uma das
              [estratégias avançadas][] e nenhum scoring é especificado,
              a métrica do experionml (se existir) é usada como scoring.

        """
        if isinstance(strategy, str):
            if strategy == "univariate" and solver is None:
                solver = "f_classif" if self.task.is_classification else "f_regression"
            elif (
                strategy not in ("univariate", "pca")
                and isinstance(solver, str)
                and (not solver.endswith("_class") and not solver.endswith("_reg"))
            ):
                solver += f"_{'class' if self.task.is_classification else 'reg'}"

            # Se o método run foi chamado antes, usa a métrica principal
            if strategy not in ("univariate", "pca", "sfm", "rfe"):
                if self._metric and "scoring" not in kwargs:
                    kwargs["scoring"] = self._metric[0]

        columns = kwargs.pop("columns", None)
        feature_selector = FeatureSelector(
            strategy=strategy,
            solver=solver,
            n_features=n_features,
            min_repeated=min_repeated,
            max_repeated=max_repeated,
            max_correlation=max_correlation,
            **self._prepare_kwargs(kwargs, sign(FeatureSelector)),
        )

        self._add_transformer(feature_selector, columns=columns)

    # Training methods ============================================= >>

    def _check_metric(self, metric: MetricConstructor) -> MetricConstructor:
        """Verifica se a métrica fornecida é válida.

        Se houve uma execução anterior, verifica se a métrica fornecida
        é a mesma.

        Parâmetros
        ----------
        metric: str, func, scorer, sequence or None
            Métrica fornecida para a execução.

        Retorna
        -------
        str, func, scorer, sequence or None
            Métrica da execução.

        """
        if self._metric:
            # Se a métrica estiver vazia, reutiliza a existente
            if metric is None:
                metric = list(self._metric)
            else:
                # Se houver métrica, ela deve ser a mesma da execução anterior
                new_metric = [get_custom_scorer(m).name for m in lst(metric)]
                if new_metric != self._metric.keys():
                    raise ValueError(
                        "Valor inválido para o parâmetro metric. A métrica "
                        "deve ser a mesma da execução anterior. Esperado "
                        f"{self.metric}, recebido {flt(new_metric)}."
                    )

        return metric

    def _run(self, trainer: BaseRunner):
        """Treina e avalia os modelos.

        Se todos os modelos falharem, captura os erros e os repassa ao
        experionml antes de lançar a exceção. Se a execução for bem-sucedida,
        atualiza todos os atributos e métodos relevantes.

        Parâmetros
        ----------
        trainer: Runner
            Instância responsável pelo treinamento efetivo dos modelos.

        """
        if any(col.dtype.kind not in "ifu" for col in get_cols(self.branch.y)):
            raise ValueError(
                "A coluna-alvo não é numérica. Use experionml.clean() "
                "para codificá-la em valores numéricos."
            )

        # Transfere atributos
        trainer._config = self._config
        trainer._branches = self._branches

        trainer.run()

        # Sobrescreve modelos com o mesmo nome dos novos
        for model in trainer._models:
            if model.name in self._models:
                self._delete_models(model.name)
                self._log(
                    f"Execuções consecutivas do modelo {model.name}. "
                    "O modelo anterior foi sobrescrito.",
                    3,
                )

        self._models.extend(trainer._models)
        self._metric = trainer._metric

    @composed(crash, method_to_log)
    def run(
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
        **kwargs,
    ):
        """Treina e avalia os modelos de forma direta.

        Ao contrário de [successive_halving][self-successive_halving] e
        [train_sizing][self-train_sizing], a abordagem direta itera
        apenas uma vez sobre os modelos, usando o conjunto completo.

        As etapas a seguir são aplicadas a cada modelo:

        1. Aplicar [hyperparameter tuning][] (opcional).
        2. Ajustar o modelo no conjunto de treino usando a melhor
           combinação de hiperparâmetros encontrada.
        3. Avaliar o modelo no conjunto de teste.
        4. Treinar o estimador em várias amostras [bootstrapped][bootstrapping]
           do conjunto de treino e avaliá-lo novamente no teste (opcional).

        Consulte a classe [DirectClassifier][] ou [DirectRegressor][] para
        a descrição dos parâmetros.

        """
        trainer = {
            "classification": DirectClassifier,
            "regression": DirectRegressor,
            "forecast": DirectForecaster,
        }

        self._run(
            trainer[self._goal.name](
                models=models,
                metric=self._check_metric(metric),
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                parallel=parallel,
                errors=errors,
                **self._prepare_kwargs(kwargs, is_runner=True),
            )
        )

    @composed(crash, method_to_log)
    def successive_halving(
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
        **kwargs,
    ):
        """Ajusta os modelos no formato successive halving.

        A técnica successive halving é um algoritmo baseado em bandits
        que ajusta N modelos em 1/N dos dados. A melhor metade é
        selecionada para a próxima iteração, em que o processo se
        repete. Isso continua até restar apenas um modelo, que então é
        ajustado no conjunto completo. Tenha em mente que o desempenho
        de um modelo pode depender bastante da quantidade de dados com
        que ele foi treinado. Por isso, recomenda-se usar essa técnica
        apenas com modelos semelhantes, por exemplo apenas modelos baseados em árvore.

        As etapas a seguir são aplicadas a cada modelo (por iteração):

        1. Aplicar [hyperparameter tuning][] (opcional).
        2. Ajustar o modelo no conjunto de treino usando a melhor
           combinação de hiperparâmetros encontrada.
        3. Avaliar o modelo no conjunto de teste.
        4. Treinar o estimador em várias amostras [bootstrapped][bootstrapping]
           do conjunto de treino e avaliá-lo novamente no teste (opcional).

        Consulte a classe [SuccessiveHalvingClassifier][] ou
        [SuccessiveHalvingRegressor][] para a descrição dos parâmetros.

        """
        trainer = {
            "classification": SuccessiveHalvingClassifier,
            "regression": SuccessiveHalvingRegressor,
            "forecast": SuccessiveHalvingForecaster,
        }

        self._run(
            trainer[self._goal.name](
                models=models,
                metric=self._check_metric(metric),
                skip_runs=skip_runs,
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                parallel=parallel,
                errors=errors,
                **self._prepare_kwargs(kwargs, is_runner=True),
            )
        )

    @composed(crash, method_to_log)
    def train_sizing(
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
        **kwargs,
    ):
        """Treina e avalia os modelos no formato train sizing.

        Ao treinar modelos, normalmente existe um trade-off entre
        desempenho e tempo de computação, regulado pela quantidade de
        amostras no conjunto de treino. Este método pode ser usado para
        gerar insights sobre esse trade-off e ajudar a determinar o
        tamanho ótimo do conjunto de treino. Os modelos são ajustados
        múltiplas vezes, aumentando progressivamente o número de amostras.

        As etapas a seguir são aplicadas a cada modelo (por iteração):

        1. Aplicar [hyperparameter tuning][] (opcional).
        2. Ajustar o modelo no conjunto de treino usando a melhor
           combinação de hiperparâmetros encontrada.
        3. Avaliar o modelo no conjunto de teste.
        4. Treinar o estimador em várias amostras [bootstrapped][bootstrapping]
           do conjunto de treino e avaliá-lo novamente no teste (opcional).

        Consulte a classe [TrainSizingClassifier][] ou [TrainSizingRegressor][]
        para a descrição dos parâmetros.

        """
        trainer = {
            "classification": TrainSizingClassifier,
            "regression": TrainSizingRegressor,
            "forecast": TrainSizingForecaster,
        }

        self._run(
            trainer[self._goal.name](
                models=models,
                metric=self._check_metric(metric),
                train_sizes=train_sizes,
                est_params=est_params,
                n_trials=n_trials,
                ht_params=ht_params,
                n_bootstrap=n_bootstrap,
                parallel=parallel,
                errors=errors,
                **self._prepare_kwargs(kwargs, is_runner=True),
            )
        )
