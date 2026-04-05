from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.pipeline import _final_estimator_has
from sklearn.utils import Bunch
from sklearn.utils._metadata_requests import MetadataRouter, MethodMapping
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import _raise_for_params, process_routing
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_memory
from sktime.forecasting.base import BaseForecaster
from typing_extensions import Self

from experionml.utils.types import (
    Bool,
    EngineDataOptions,
    EngineTuple,
    Estimator,
    FHConstructor,
    Float,
    Pandas,
    Scalar,
    Sequence,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
    YReturn,
)
from experionml.utils.utils import (
    NotFittedError,
    adjust,
    check_is_fitted,
    fit_one,
    fit_transform_one,
    to_df,
    to_tabular,
    transform_one,
    variable_return,
)


if TYPE_CHECKING:
    from sktime.proba.normal import Normal


T = TypeVar("T")


class Pipeline(SkPipeline):
    """Pipeline de transformações com um estimador final.

    Aplica sequencialmente uma lista de transformações e um estimador final.
    As etapas intermediárias do pipeline devem ser transformadores, ou
    seja, devem implementar os métodos `fit` e `transform`. O estimador
    final precisa apenas implementar `fit`. Os transformadores do
    pipeline podem ser armazenados em cache usando o parâmetro `memory`.

    O estimador de uma etapa pode ser completamente substituído definindo
    o parâmetro com seu nome para outro estimador, ou um transformador
    pode ser removido definindo-o como `passthrough` ou `None`.

    Leia mais no [guia do usuário][pipelinedocs] do sklearn.

    !!! info
        Esta classe se comporta de forma semelhante ao [pipeline][skpipeline]
        do sklearn, e adicionalmente:

        - Pode ser inicializada com um pipeline vazio.
        - Sempre retorna objetos 'pandas'.
        - Aceita transformadores que eliminam linhas.
        - Aceita transformadores que são ajustados apenas em um subconjunto
          do conjunto de dados fornecido.
        - Aceita transformadores que se aplicam apenas à coluna alvo.
        - Usa transformadores aplicados apenas no conjunto de treinamento
          para ajustar o pipeline, não para fazer previsões em novos dados.
        - A instância é considerada ajustada na inicialização se todos os
          transformadores/estimadores subjacentes do pipeline estiverem.
        - Retorna atributos do estimador final se não forem do Pipeline.
        - O último estimador também é armazenado em cache.
        - Suporta modelos de séries temporais seguindo a API do sktime.

    !!! warning
        Este Pipeline só funciona com estimadores cujos parâmetros
        para fit, transform, predict, etc... são nomeados `X` e/ou `y`.

    Parâmetros
    ----------
    steps: list of tuple
        Lista de tuplas (nome, transformador) (implementando `fit`/`transform`)
        encadeadas em ordem sequencial.

    memory: str, [Memory][joblibmemory] or None, default=None
        Usado para armazenar em cache os transformadores ajustados do pipeline.
        Ativar o cache aciona um clone dos transformadores antes do ajuste.
        Portanto, a instância do transformador fornecida ao pipeline não pode
        ser inspecionada diretamente. Use o atributo `named_steps` ou `steps`
        para inspecionar os estimadores no pipeline. O cache dos
        transformadores é vantajoso quando o ajuste consome muito tempo.

    verbose: int or None, default=0
        Nível de verbosidade dos transformadores no pipeline. Se None,
        mantém a verbosidade original. Se >0, o tempo decorrido durante
        o ajuste de cada etapa é impresso. Observe que isso não é o
        mesmo que o parâmetro `verbose` do sklearn. Use o atributo verbose
        do pipeline para modificar aquele (padrão False).

    Atributos
    ----------
    named_steps: [Bunch][]
        Objeto semelhante a dicionário com os seguintes atributos. Atributo
        somente leitura para acessar qualquer parâmetro de etapa pelo nome
        fornecido pelo usuário. As chaves são os nomes das etapas e os
        valores são os parâmetros das etapas.

    classes_: np.ndarray of shape (n_classes,)
        Rótulos das classes. Existem apenas se a última etapa do pipeline
        for um classificador.

    feature_names_in_: np.ndarray
        Nomes das features observadas durante o método `fit` da primeira etapa.

    n_features_in_: int
        Número de features observadas durante o método `fit` da primeira etapa.

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Inicializa o experionml
    experionml = ExperionMLClassifier(X, y, verbose=2)

    # Aplica métodos de limpeza de dados e engenharia de features
    experionml.scale()
    experionml.balance(strategy="smote")
    experionml.feature_selection(strategy="rfe", solver="lr", n_features=22)

    # Treina os modelos
    experionml.run(models="LR")

    # Obtém o objeto pipeline
    pipeline = experionml.lr.export_pipeline()
    print(pipeline)
    ```

    """

    def __init__(
        self,
        steps: list[tuple[str, Estimator]],
        *,
        memory: str | Memory | None = None,
        verbose: Verbose | None = 0,
    ):
        super().__init__(steps=steps, memory=memory, verbose=False)
        self._verbose = verbose

    def __bool__(self):
        """Se o pipeline contém pelo menos um estimador."""
        return len(self.steps) > 0

    def __contains__(self, item: str | Any):
        """Se o nome ou estimador está no pipeline."""
        if isinstance(item, str):
            return item in self.named_steps
        else:
            return item in self.named_steps.values()

    def __getattr__(self, item: str):
        """Obtém o atributo do estimador final."""
        try:
            return getattr(self._final_estimator, item)
        except (AttributeError, IndexError):
            raise AttributeError(f"'Pipeline' object has no attribute '{item}'.") from None

    def __sklearn_is_fitted__(self):
        """Se o pipeline foi ajustado."""
        try:
            # verifica se a última etapa do pipeline foi ajustada
            # verificamos apenas a última etapa pois se ela estiver ajustada,
            # significa que as etapas anteriores também deveriam estar. Isso é
            # mais rápido do que verificar se cada etapa do pipeline está ajustada.
            check_is_fitted(self.steps[-1][1])
            return True
        except (NotFittedError, IndexError):
            return False

    @property
    def memory(self) -> Memory:
        """Obtém o objeto de memória interno."""
        return self._memory

    @memory.setter
    def memory(self, value: str | Memory | None):
        """Cria um novo objeto de memória interno."""
        self._memory = check_memory(value)
        self._mem_fit = self._memory.cache(fit_one)
        self._mem_fit_transform = self._memory.cache(fit_transform_one)
        self._mem_transform = self._memory.cache(transform_one)

    @property
    def _final_estimator(self) -> Literal["passthrough"] | Estimator | None:
        """Retorna o último estimador no pipeline.

        Se o pipeline estiver vazio, retorna None. Se o estimador for
        None, retorna "passthrough".

        """
        try:
            estimator = self.steps[-1][1]
            return "passthrough" if estimator is None else estimator
        except (ValueError, AttributeError, TypeError, IndexError):
            # Esta condição ocorre quando o pipeline está vazio ou uma chamada
            # a um método está chamando `_available_if` primeiro e `fit` ainda
            # não validou `steps`.
            return None

    def _can_transform(self) -> bool:
        """Verifica se o pipeline pode usar o método transform."""
        return (
            self._final_estimator is None
            or self._final_estimator == "passthrough"
            or hasattr(self._final_estimator, "transform")
        )

    def _can_inverse_transform(self) -> bool:
        """Verifica se o pipeline pode usar o método inverse_transform."""
        return all(
            est is None or est == "passthrough" or hasattr(est, "inverse_transform")
            for _, _, est in self._iter()
        )

    @overload
    def _convert(self, obj: Literal[None]) -> None: ...

    @overload
    def _convert(self, obj: pd.DataFrame) -> XReturn: ...

    @overload
    def _convert(self, obj: pd.Series) -> YReturn: ...

    def _convert(self, obj: Pandas | None) -> YReturn | None:
        """Converte os dados para o tipo definido no motor de dados.

        Parâmetros
        ----------
        obj: pd.Series, pd.DataFrame or None
            Objeto a ser convertido. Se None, retorna como está.

        Retorna
        -------
        object
            Dados convertidos.

        """
        # Aplica transformações apenas quando o motor está definido
        if hasattr(self, "_engine") and isinstance(obj, pd.Series | pd.DataFrame):
            return self._engine.data_engine.convert(obj)
        else:
            return obj

    def _iter(
        self,
        *,
        with_final: Bool = True,
        filter_passthrough: Bool = True,
        filter_train_only: Bool = True,
    ) -> Iterator[tuple[int, str, Estimator]]:
        """Gera tuplas (idx, nome, estimador) a partir de self.steps.

        Por padrão, estimadores que são aplicados apenas no conjunto de
        treinamento são filtrados para previsões.

        Parâmetros
        ----------
        with_final: bool, default=True
            Se deve incluir o estimador final.

        filter_passthrough: bool, default=True
            Se deve excluir elementos `passthrough`.

        filter_train_only: bool, default=True
            Se deve excluir estimadores que devem ser usados apenas para
            treinamento (com atributo `_train_only=True`).

        Produz
        ------
        int
            Posição do índice no pipeline.

        str
            Nome do estimador.

        Estimator
            Instância do transformador ou preditor.

        """
        stop = len(self.steps)
        if not with_final and stop > 0:
            stop -= 1

        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if (not filter_passthrough or (trans is not None and trans != "passthrough")) and (
                not filter_train_only or not getattr(trans, "_train_only", False)
            ):
                yield idx, name, trans

    def _fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        routed_params: dict[str, Bunch] | None = None,
    ) -> tuple[pd.DataFrame | None, Pandas | None]:
        """Obtém os dados transformados pelo pipeline.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Se None,
            `X` é ignorado. None se o pipeline usa apenas y.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        routed_params: dict or None, default=None
            Parâmetros de metadados roteados para o método fit.

        Retorna
        -------
        dataframe or None
            Conjunto de features transformado.

        series, dataframe or None
            Coluna alvo transformada.

        """
        self.steps: list[tuple[str, Estimator]] = list(self.steps)
        self._validate_steps()

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        for step, name, transformer in self._iter(
            with_final=False, filter_passthrough=False, filter_train_only=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step)):
                    continue

            # Não clona quando o cache está desativado para preservar compatibilidade retroativa
            if self.memory.location is None:
                cloned = transformer
            else:
                cloned = clone(transformer)

            with adjust(cloned, verbose=self._verbose):
                # Ajusta ou carrega o estimador atual do cache
                # Type ignore porque routed_params nunca é None, mas
                # a assinatura de _fit precisa estar em conformidade com o sklearn
                Xt, yt, fitted_transformer = self._mem_fit_transform(
                    transformer=cloned,
                    X=Xt,
                    y=yt,
                    message=self._log_message(step),
                    **routed_params[name].fit_transform,  # type: ignore[index]
                )

            # Substitui o estimador da etapa pelo estimador ajustado
            # (necessário ao carregar do cache)
            self.steps[step] = (name, fitted_transformer)

        return Xt, yt

    def get_metadata_routing(self):
        """Obtém o roteamento de metadados deste objeto.

        Consulte a [documentação do sklearn][metadatarouting] sobre como
        o mecanismo de roteamento funciona.

        Retorna
        -------
        MetadataRouter
            Um [MetadataRouter][] encapsulando informações de roteamento.

        """
        router = MetadataRouter(owner=self.__class__.__name__)

        # Primeiro, adicionamos todas as etapas exceto a última
        for _, name, trans in self._iter(with_final=False, filter_train_only=False):
            method_mapping = MethodMapping()
            # fit, fit_predict e fit_transform chamam fit_transform se existir,
            # caso contrário chamam fit e transform
            if hasattr(trans, "fit_transform"):
                (
                    method_mapping.add(caller="fit", callee="fit_transform")
                    .add(caller="fit_transform", callee="fit_transform")
                    .add(caller="fit_predict", callee="fit_transform")
                )
            else:
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="fit", callee="transform")
                    .add(caller="fit_transform", callee="fit")
                    .add(caller="fit_transform", callee="transform")
                    .add(caller="fit_predict", callee="fit")
                    .add(caller="fit_predict", callee="transform")
                )

            (
                method_mapping.add(caller="predict", callee="transform")
                .add(caller="predict", callee="transform")
                .add(caller="predict_proba", callee="transform")
                .add(caller="decision_function", callee="transform")
                .add(caller="predict_log_proba", callee="transform")
                .add(caller="transform", callee="transform")
                .add(caller="inverse_transform", callee="inverse_transform")
                .add(caller="score", callee="transform")
            )

            router.add(method_mapping=method_mapping, **{name: trans})

        # Em seguida, adicionamos a última etapa
        if len(self.steps) > 0:
            final_name, final_est = self.steps[-1]
            if final_est is not None and final_est != "passthrough":
                # então adicionamos a última etapa
                method_mapping = MethodMapping()
                if hasattr(final_est, "fit_transform"):
                    method_mapping.add(caller="fit_transform", callee="fit_transform")
                else:
                    method_mapping.add(caller="fit", callee="fit").add(
                        caller="fit", callee="transform"
                    )
                (
                    method_mapping.add(caller="fit", callee="fit")
                    .add(caller="predict", callee="predict")
                    .add(caller="fit_predict", callee="fit_predict")
                    .add(caller="predict_proba", callee="predict_proba")
                    .add(caller="decision_function", callee="decision_function")
                    .add(caller="predict_log_proba", callee="predict_log_proba")
                    .add(caller="transform", callee="transform")
                    .add(caller="inverse_transform", callee="inverse_transform")
                    .add(caller="score", callee="score")
                )

                router.add(method_mapping=method_mapping, **{final_name: final_est})

        return router

    def fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **params,
    ) -> Self:
        """Ajusta o pipeline.

        Ajusta todos os transformadores um após o outro e transforma
        sequencialmente os dados. Por fim, ajusta os dados transformados
        usando o estimador final.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        **params
            Parâmetros solicitados e aceitos pelas etapas. Cada etapa deve
            ter solicitado certos metadados para que esses parâmetros sejam
            encaminhados a elas.

        Retorna
        -------
        self
            Pipeline com etapas ajustadas.

        """
        routed_params = self._check_method_params(method="fit", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator is not None and self._final_estimator != "passthrough":
                with adjust(self._final_estimator, verbose=self._verbose):
                    self._mem_fit(
                        estimator=self._final_estimator,
                        X=Xt,
                        y=yt,
                        **routed_params[self.steps[-1][0]].fit,
                    )

        return self

    @available_if(_can_transform)
    def fit_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Ajusta o pipeline e transforma os dados.

        Chama `fit` seguido de `transform` em cada transformador do
        pipeline. Os dados transformados são finalmente passados ao
        estimador final que chama o método `transform`. Válido apenas se
        o estimador final implementar `transform`. Também funciona quando
        o estimador final é `None`, caso em que todas as transformações
        anteriores são aplicadas.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Se None,
            `X` é ignorado. None se o estimador usa apenas y.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        **params
            Parâmetros solicitados e aceitos pelas etapas. Cada etapa deve
            ter solicitado certos metadados para que esses parâmetros sejam
            encaminhados a elas.

        Retorna
        -------
        dataframe
            Conjunto de features transformado. Retornado apenas se fornecido.

        series or dataframe
            Coluna alvo transformada. Retornada apenas se fornecida.

        """
        routed_params = self._check_method_params(method="fit_transform", props=params)
        Xt, yt = self._fit(X, y, routed_params)

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator is None or self._final_estimator == "passthrough":
                return variable_return(Xt, yt)

            with adjust(self._final_estimator, verbose=self._verbose):
                Xt, yt, _ = self._mem_fit_transform(
                    transformer=self._final_estimator,
                    X=Xt,
                    y=yt,
                    **routed_params[self.steps[-1][0]].fit_transform,
                )

        return variable_return(self._convert(Xt), self._convert(yt))

    @available_if(_can_transform)
    def transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        *,
        filter_train_only: Bool = True,
        **params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Transforma os dados.

        Chama `transform` em cada transformador do pipeline. Os dados
        transformados são finalmente passados ao estimador final que
        chama o método `transform`. Válido apenas se o estimador final
        implementar `transform`. Também funciona quando o estimador
        final é `None`, caso em que todas as transformações anteriores
        são aplicadas.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Se None,
            `X` é ignorado. None se o pipeline usa apenas y.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        filter_train_only: bool, default=True
            Se deve excluir transformadores que devem ser usados apenas
            no conjunto de treinamento.

        **params
            Parâmetros solicitados e aceitos pelas etapas. Cada etapa deve
            ter solicitado certos metadados para que esses parâmetros sejam
            encaminhados a elas.

        Retorna
        -------
        dataframe
            Conjunto de features transformado. Retornado apenas se fornecido.

        series or dataframe
            Coluna alvo transformada. Retornada apenas se fornecida.

        """
        if X is None and y is None:
            raise ValueError("X e y não podem ser ambos None.")

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        _raise_for_params(params, self, "transform")

        routed_params = process_routing(self, "transform", **params)
        for _, name, transformer in self._iter(filter_train_only=filter_train_only):
            with adjust(transformer, verbose=self._verbose):
                Xt, yt = self._mem_transform(
                    transformer=transformer,
                    X=Xt,
                    y=yt,
                    **routed_params[name].transform,
                )

        return variable_return(self._convert(Xt), self._convert(yt))

    @available_if(_can_inverse_transform)
    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        *,
        filter_train_only: Bool = True,
        **params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Transformação inversa para cada etapa em ordem reversa.

        Todos os estimadores no pipeline devem implementar o método
        `inverse_transform`.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Se None,
            `X` é ignorado. None se o pipeline usa apenas y.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        filter_train_only: bool, default=True
            Se deve excluir transformadores que devem ser usados apenas
            no conjunto de treinamento.

        **params
            Parâmetros solicitados e aceitos pelas etapas. Cada etapa deve
            ter solicitado certos metadados para que esses parâmetros sejam
            encaminhados a elas.

        Retorna
        -------
        dataframe
            Conjunto de features transformado. Retornado apenas se fornecido.

        series or dataframe
            Coluna alvo transformada. Retornada apenas se fornecida.

        """
        if X is None and y is None:
            raise ValueError("X e y não podem ser ambos None.")

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        _raise_for_params(params, self, "inverse_transform")

        routed_params = process_routing(self, "inverse_transform", **params)
        reverse_iter = reversed(list(self._iter(filter_train_only=filter_train_only)))
        for _, name, transformer in reverse_iter:
            with adjust(transformer, verbose=self._verbose):
                Xt, yt = self._mem_transform(
                    transformer=transformer,
                    X=Xt,
                    y=yt,
                    method="inverse_transform",
                    **routed_params[name].inverse_transform,
                )

        return variable_return(self._convert(Xt), self._convert(yt))

    @available_if(_final_estimator_has("decision_function"))
    def decision_function(self, X: XConstructor, **params) -> np.ndarray:
        """Transforma e depois aplica decision_function do estimador final.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de features com shape=(n_samples, n_features).

        **params
            Parâmetros solicitados e aceitos pelas etapas. Cada etapa deve
            ter solicitado certos metadados para que esses parâmetros sejam
            encaminhados a elas.

        Retorna
        -------
        np.ndarray
            Escores de confiança previstos com shape=(n_samples,) para
            tarefas de classificação binária (razão de verossimilhança
            logarítmica da classe positiva) ou shape=(n_samples, n_classes)
            para tarefas de classificação multiclasse.

        """
        Xt = to_df(X)

        _raise_for_params(params, self, "decision_function")

        routed_params = process_routing(self, "decision_function", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(
                    transformer=transformer,
                    X=Xt,
                    **routed_params.get(name, {}).get("transform", {}),
                )

        return self.steps[-1][1].decision_function(
            Xt, **routed_params.get(self.steps[-1][0], {}).get("decision_function", {})
        )

    @available_if(_final_estimator_has("predict"))
    def predict(
        self,
        X: XConstructor | None = None,
        fh: FHConstructor | None = None,
        **params,
    ) -> np.ndarray | Pandas:
        """Transforma e depois aplica predict do estimador final.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Pode ser
            `None` apenas para tarefas de [previsão][time-series].

        fh: int, sequence or [ForecastingHorizon][] or None, default=None
            O horizonte de previsão codificando os timestamps nos quais
            realizar as previsões. Apenas para tarefas de [previsão][time-series].

        **params
            Parâmetros solicitados e aceitos pelas etapas. Cada etapa deve
            ter solicitado certos metadados para que esses parâmetros sejam
            encaminhados a elas. Observe que, embora isso possa ser usado para
            retornar incertezas de alguns modelos com `return_std` ou
            `return_cov`, as incertezas geradas pelas transformações no
            pipeline não são propagadas ao estimador final.

        Retorna
        -------
        np.ndarray, series or dataframe
            Previsões com shape=(n_samples,) ou shape=(n_samples,
            n_targets) para [tarefas multi-saída][].

        """
        if X is None and fh is None:
            raise ValueError("X e fh não podem ser ambos None.")

        Xt = to_df(X)

        routed_params = process_routing(self, "predict", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt, **routed_params[name].transform)

        if isinstance(self._final_estimator, BaseForecaster):
            if fh is None:
                raise ValueError("O parâmetro fh não pode ser None para estimadores de previsão.")

            return self.steps[-1][1].predict(fh=fh, X=Xt)
        else:
            return self.steps[-1][1].predict(Xt, **routed_params[self.steps[-1][0]].predict)

    @available_if(_final_estimator_has("predict_interval"))
    def predict_interval(
        self,
        fh: FHConstructor,
        X: XConstructor | None = None,
        *,
        coverage: Float | Sequence[Float] = 0.9,
    ) -> pd.DataFrame:
        """Transforma e depois aplica predict_interval do estimador final.

        Parâmetros
        ----------
        fh: int, sequence or [ForecastingHorizon][]
            O horizonte de previsão codificando os timestamps nos quais
            realizar as previsões.

        X: dataframe-like or None, default=None
            Série temporal exógena correspondente a `fh`.

        coverage: float or sequence, default=0.9
            Cobertura(s) nominal(is) do(s) intervalo(s) preditivo(s).

        Retorna
        -------
        dataframe
            Previsões de intervalo calculadas.

        """
        Xt = to_df(X)

        for _, _, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt)

        return self.steps[-1][1].predict_interval(fh=fh, X=Xt, coverage=coverage)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X: XConstructor, **params) -> np.ndarray:
        """Transforma e depois aplica predict_log_proba do estimador final.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de features com shape=(n_samples, n_features).

        **params
            Parâmetros solicitados e aceitos pelas etapas. Cada etapa deve
            ter solicitado certos metadados para que esses parâmetros sejam
            encaminhados a elas.

        Retorna
        -------
        list or np.ndarray
            Log-probabilidades previstas com shape=(n_samples,
            n_classes) ou lista de arrays para [tarefas multi-saída][].

        """
        Xt = to_df(X)

        routed_params = process_routing(self, "predict_log_proba", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt, **routed_params[name].transform)

        return self.steps[-1][1].predict_log_proba(
            Xt, **routed_params[self.steps[-1][0]].predict_log_proba
        )

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(
        self,
        X: XConstructor | None = None,
        fh: FHConstructor | None = None,
        *,
        marginal: Bool = True,
        **params,
    ) -> list[np.ndarray] | np.ndarray | Normal:
        """Transforma e depois aplica predict_proba do estimador final.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Pode ser
            `None` apenas para tarefas de [previsão][time-series].

        fh: int, sequence, [ForecastingHorizon][] or None, default=None
            O horizonte de previsão codificando os timestamps nos quais
            realizar as previsões. Apenas para tarefas de [previsão][time-series].

        marginal: bool, default=True
            Se a distribuição retornada é marginal por índice de tempo.
            Apenas para tarefas de [previsão][time-series].

        **params
            Parâmetros solicitados e aceitos pelas etapas. Cada etapa deve
            ter solicitado certos metadados para que esses parâmetros sejam
            encaminhados a elas.

        Retorna
        -------
        list, np.ndarray or sktime.proba.[Normal][]

            - Para tarefas de classificação: probabilidades previstas com
              shape=(n_samples, n_classes).
            - Para [tarefas multi-saída][]: lista de arrays com
              shape=(n_samples, n_classes).
            - Para tarefas de [previsão][time-series]: objeto de distribuição.

        """
        if X is None and fh is None:
            raise ValueError("X e fh não podem ser ambos None.")

        Xt = to_df(X)

        routed_params = process_routing(self, "predict_proba", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt, **routed_params[name].transform)

        if isinstance(self._final_estimator, BaseForecaster):
            if fh is None:
                raise ValueError("O parâmetro fh não pode ser None para estimadores de previsão.")

            return self.steps[-1][1].predict_proba(fh=fh, X=Xt, marginal=marginal)
        else:
            return self.steps[-1][1].predict_proba(
                Xt, **routed_params[self.steps[-1][0]].predict_proba
            )

    @available_if(_final_estimator_has("predict_quantiles"))
    def predict_quantiles(
        self,
        fh: FHConstructor,
        X: XConstructor | None = None,
        *,
        alpha: Float | Sequence[Float] = (0.05, 0.95),
    ) -> Pandas:
        """Transforma e depois aplica predict_quantiles do estimador final.

        Parâmetros
        ----------
        fh: int, sequence or [ForecastingHorizon][]
            O horizonte de previsão codificando os timestamps nos quais
            realizar as previsões.

        X: dataframe-like or None, default=None
            Série temporal exógena correspondente a `fh`.

        alpha: float or sequence, default=(0.05, 0.95)
            Uma probabilidade ou lista delas em que as previsões de quantil
            são calculadas.

        Retorna
        -------
        dataframe
            Previsões de quantil calculadas.

        """
        Xt = to_df(X)

        for _, _, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt)

        return self.steps[-1][1].predict_quantiles(fh=fh, X=Xt, alpha=alpha)

    @available_if(_final_estimator_has("predict_residuals"))
    def predict_residuals(
        self,
        y: YConstructor,
        X: XConstructor | None = None,
    ) -> Pandas:
        """Transforma e depois aplica predict_residuals do estimador final.

        Parâmetros
        ----------
        y: sequence or dataframe
            Observações reais.

        X: dataframe-like or None, default=None
            Série temporal exógena correspondente a `y`.

        Retorna
        -------
        series or dataframe
            Resíduos com shape=(n_samples,) ou shape=(n_samples,
            n_targets) para tarefas [multivariadas][].

        """
        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        for _, _, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, yt = self._mem_transform(transformer, Xt, yt)

        return self.steps[-1][1].predict_residuals(y=yt, X=Xt)

    @available_if(_final_estimator_has("predict_var"))
    def predict_var(
        self,
        fh: FHConstructor,
        X: XConstructor | None = None,
        *,
        cov: Bool = False,
    ) -> pd.DataFrame:
        """Transforma e depois aplica predict_var do estimador final.

        Parâmetros
        ----------
        fh: int, sequence or [ForecastingHorizon][]
            O horizonte de previsão codificando os timestamps nos quais
            realizar as previsões.

        X: dataframe-like or None, default=None
            Série temporal exógena correspondente a `fh`.

        cov: bool, default=False
            Se deve calcular a previsão da matriz de covariância ou as
            previsões de variância marginal.

        Retorna
        -------
        dataframe
            Previsões de variância calculadas.

        """
        Xt = to_df(X)

        for _, _, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, _ = self._mem_transform(transformer, Xt)

        return self.steps[-1][1].predict_var(fh=fh, X=Xt, cov=cov)

    def set_output(self, *, transform: EngineDataOptions | None = None) -> Self:
        """Define o contêiner de saída.

        Consulte o [guia do usuário][set_output] do sklearn sobre como usar
        a API `set_output`. Veja [aqui][data-engines] uma descrição
        das opções.

        Parâmetros
        ----------
        transform: str or None, default=None
            Configura a saída dos métodos `transform`, `fit_transform`
            e `inverse_transform`. Se None, a configuração não é
            alterada. Escolha entre:

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

        Retorna
        -------
        Self
            Instância do estimador.

        """
        if transform is not None:
            self._engine = EngineTuple(data=transform)

        return self

    @available_if(_final_estimator_has("score"))
    def score(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        fh: FHConstructor | None = None,
        *,
        sample_weight: Sequence[Scalar] | None = None,
        **params,
    ) -> Float:
        """Transforma e depois aplica score do estimador final.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de features com shape=(n_samples, n_features). Pode ser
            `None` apenas para tarefas de [previsão][time-series].

        y: sequence, dataframe-like or None, default=None
            Valores alvo correspondentes a `X`.

        fh: int, sequence, [ForecastingHorizon][] or None, default=None
            O horizonte de previsão codificando os timestamps a pontuar.

        sample_weight: sequence or None, default=None
            Pesos de amostra correspondentes a `y` passados ao método `score`
            do estimador final. Se None, nenhuma ponderagem é realizada.
            Apenas para tarefas que não são de previsão.

        Retorna
        -------
        float
            Acurácia média, r2 ou mape de self.predict(X) em relação a
            `y` (dependendo da tarefa).

        """
        if X is None and y is None:
            raise ValueError("X e y não podem ser ambos None.")

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        # Descarta pesos de amostra se for estimador sktime
        if not isinstance(self._final_estimator, BaseForecaster):
            params["sample_weight"] = sample_weight

        routed_params = process_routing(self, "score", **params)

        for _, name, transformer in self._iter(with_final=False):
            with adjust(transformer, verbose=self._verbose):
                Xt, yt = self._mem_transform(transformer, Xt, yt, **routed_params[name].transform)

        if isinstance(self._final_estimator, BaseForecaster):
            return self.steps[-1][1].score(y=yt, X=Xt, fh=fh)
        else:
            return self.steps[-1][1].score(Xt, yt, **routed_params[self.steps[-1][0]].score)
