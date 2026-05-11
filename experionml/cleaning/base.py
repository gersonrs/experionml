from __future__ import annotations

import re
from typing import TypeVar, overload

import sklearn
from beartype import beartype
from sklearn.base import BaseEstimator, _clone_parametrized
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from typing_extensions import Self

from experionml.basetransformer import BaseTransformer
from experionml.utils.types import (
    EngineDataOptions,
    EngineTuple,
    Int,
    Transformer,
    XConstructor,
    XReturn,
    YConstructor,
    YReturn,
)
from experionml.utils.utils import (
    check_is_fitted,
    to_df,
    to_tabular,
    variable_return,
)


T_Transformer = TypeVar("T_Transformer", bound=Transformer)


@beartype
class TransformerMixin(BaseEstimator, BaseTransformer):
    """Classe mixin para todos os transformadores do ExperionML.

    Diferente do sklearn nas seguintes formas:

    - Considera a transformação de y.
    - Sempre adiciona um método fit.
    - Encapsula o método fit com atributos e verificação dos dados.
    - Encapsula os métodos de transformação com verificação dos dados.
    - Mantém os atributos internos ao ser clonado.

    """

    def __repr__(self, N_CHAR_MAX: Int = 700) -> str:
        """Remove named tuples com valores padrão da representação em string."""
        out = super().__repr__(N_CHAR_MAX)

        # Remove o engine padrão para uma representação mais limpa
        if hasattr(self, "engine") and sklearn.get_config()["print_changed_only"]:
            if self.engine.data == EngineTuple().data:
                out = re.sub(f"'data': '{self.engine.data}'", "", out)
            if self.engine.estimator == EngineTuple().estimator:
                out = re.sub(f", 'estimator': '{self.engine.estimator}'", "", out)
            out = re.sub("engine={}", "", out)
            out = re.sub(
                r"((?<=[{(]),\s|,\s(?=[})])|,\s(?=,\s))", "", out
            )  # Remove vírgulas e espaços

        return out

    def __sklearn_clone__(self) -> Self:
        """Encapsula o método de clonagem para anexar atributos internos."""
        cloned = _clone_parametrized(self)

        for attr in ("_cols", "_train_only"):
            if hasattr(self, attr):
                setattr(cloned, attr, getattr(self, attr))

        return cloned

    def fit(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> Self:
        """Não faz nada.

        Implementado para continuidade da API.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`. Se None, `y` é
            ignorado.

        **fit_params
            Argumentos de palavra-chave adicionais para o método fit.

        Retorna
        -------
        self
            Instância do estimador.

        """
        Xt = to_df(X)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self._log(f"Fitting {self.__class__.__name__}...", 1)

        return self

    @overload
    def fit_transform(
        self,
        X: None,
        y: YConstructor,
        **fit_params,
    ) -> YReturn: ...

    @overload
    def fit_transform(
        self,
        X: XConstructor,
        y: None = ...,
        **fit_params,
    ) -> XReturn: ...

    @overload
    def fit_transform(
        self,
        X: XConstructor,
        y: YConstructor,
        **fit_params,
    ) -> tuple[XReturn, YReturn]: ...

    def fit_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Ajusta aos dados e depois os transforma.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`. Se None, `y` é
            ignorado.

        **fit_params
            Argumentos de palavra-chave adicionais para o método fit.

        Retorna
        -------
        dataframe
            Conjunto de variáveis transformado. Retornado somente se fornecido.

        series or dataframe
            Coluna alvo transformada. Retornada somente se fornecida.

        """
        return self.fit(X, y, **fit_params).transform(X, y)

    @overload
    def inverse_transform(
        self,
        X: None,
        y: YConstructor,
        **fit_params,
    ) -> YReturn: ...

    @overload
    def inverse_transform(
        self,
        X: XConstructor,
        y: None = ...,
        **fit_params,
    ) -> XReturn: ...

    @overload
    def inverse_transform(
        self,
        X: XConstructor,
        y: YConstructor,
        **fit_params,
    ) -> tuple[XReturn, YReturn]: ...

    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
        **fit_params,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Não faz nada.

        Retorna a entrada sem alterações. Implementado para continuidade da API.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`. Se None, `y` é
            ignorado.

        Retorna
        -------
        dataframe
            Conjunto de variáveis. Retornado somente se fornecido.

        series or dataframe
            Coluna(s) alvo. Retornada(s) somente se fornecida(s).

        """
        check_is_fitted(self)

        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        return variable_return(self._convert(Xt), self._convert(yt))

    def set_output(self, *, transform: EngineDataOptions | None = None) -> Self:
        """Define o contêiner de saída.

        Consulte o [guia do usuário][set_output] do sklearn sobre como usar a
        API `set_output`. Veja [aqui][data-engines] uma descrição
        das opções disponíveis.

        Parâmetros
        ----------
        transform: str or None, default=None
            Configura a saída dos métodos `transform`, `fit_transform`
            e `inverse_transform`. Se None, a configuração não é alterada.
            Escolha entre:

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
        if not hasattr(self, "_engine"):
            self.engine = EngineTuple()

        if transform is not None:
            self.engine = EngineTuple(estimator=self.engine.estimator, data=transform)

        return self
