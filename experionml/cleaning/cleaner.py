from __future__ import annotations

import re
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.utils.validation import _check_feature_names_in
from typing_extensions import Self

from experionml.utils.constants import CAT_TYPES, DEFAULT_MISSING
from experionml.utils.types import (
    Bool,
    Engine,
    Sequence,
    Transformer,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
    YReturn,
    sequence_t,
)
from experionml.utils.utils import (
    check_is_fitted,
    get_col_names,
    get_cols,
    it,
    lst,
    merge,
    n_cols,
    replace_missing,
    to_df,
    to_series,
    to_tabular,
    variable_return,
)


from experionml.cleaning.base import TransformerMixin

T_Transformer = TypeVar("T_Transformer", bound=Transformer)


@beartype
class Cleaner(TransformerMixin):
    """Aplica etapas padrão de limpeza de dados a um conjunto de dados.

    Use os parâmetros para escolher quais transformações realizar.
    As etapas disponíveis são:

    - Converter dtypes para os melhores tipos possíveis.
    - Remover colunas com tipos de dados específicos.
    - Remover caracteres dos nomes das colunas.
    - Remover espaços de colunas categóricas.
    - Remover linhas duplicadas.
    - Remover linhas com valores ausentes na coluna alvo.
    - Codificar a coluna alvo.

    Esta classe pode ser acessada pelo experionml através do método [clean]
    [experionmlclassifier-clean]. Leia mais no [guia do usuário]
    [standard-data-cleaning].

    Parâmetros
    ----------
    convert_dtypes: bool, default=True
        Converte os tipos de dados das colunas para os melhores tipos possíveis
        que suportam `pd.NA`.

    drop_dtypes: str, sequence or None, default=None
        Colunas com esses tipos de dados são removidas do conjunto de dados.

    drop_chars: str or None, default=None
        Remove o padrão regex especificado dos nomes das colunas, ex.:
        `[^A-Za-z0-9]+` para remover todos os caracteres não alfanuméricos.

    strip_categorical: bool, default=True
        Se deve remover espaços das colunas categóricas.

    drop_duplicates: bool, default=False
        Se deve remover linhas duplicadas. Apenas a primeira ocorrência de
        cada linha duplicada é mantida.

    drop_missing_target: bool, default=True
        Se deve remover linhas com valores ausentes na coluna alvo.
        Esta transformação é ignorada se `y` não for fornecido.

    encode_target: bool, default=True
        Se deve codificar a(s) coluna(s) alvo. Isso inclui
        converter colunas categóricas para numérico e binarizar
        colunas [multilabel][]. Esta transformação é ignorada se `y`
        não for fornecido.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], ex.:
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Engine de execução para [estimadores][estimator-acceleration].
        Se None, o valor padrão é usado. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não exibir nada.
        - 1 para exibir informações básicas.
        - 2 para exibir informações detalhadas.

    Atributos
    ----------
    missing_: list
        Valores considerados "ausentes". Os valores padrão são: None,
        NaN, NA, NaT, +inf, -inf, "", "?", "NA", "nan", "NaN", "NaT",
        "none", "None", "inf", "-inf". Note que None, NaN, NA, +inf e
        -inf são sempre considerados ausentes pois são incompatíveis
        com estimadores sklearn.

    mapping_: dict
        Valores alvo mapeados para seus respectivos inteiros codificados. Apenas
        disponível se encode_target=True.

    feature_names_in_: np.ndarray
        Nomes das variáveis observadas durante o `fit`.

    target_names_in_: np.ndarray
        Nomes da(s) coluna(s) alvo observados durante o `fit`.

    n_features_in_: int
        Número de variáveis observadas durante o `fit`.

    Veja também
    --------
    experionml.data_cleaning:Encoder
    experionml.data_cleaning:Discretizer
    experionml.data_cleaning:Scaler

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        y = ["a" if i else "b" for i in y]

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.y)

        experionml.clean(verbose=2)

        print(experionml.y)
        ```

    === "stand-alone"
        ```pycon
        from experionml.data_cleaning import Cleaner
        from numpy.random import randint

        y = ["a" if i else "b" for i in range(randint(100))]

        cleaner = Cleaner(verbose=2)
        y = cleaner.fit_transform(y=y)

        print(y)
        ```

    """

    def __init__(
        self,
        *,
        convert_dtypes: Bool = True,
        drop_dtypes: str | Sequence[str] | None = None,
        drop_chars: str | None = None,
        strip_categorical: Bool = True,
        drop_duplicates: Bool = False,
        drop_missing_target: Bool = True,
        encode_target: Bool = True,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose)
        self.convert_dtypes = convert_dtypes
        self.drop_dtypes = drop_dtypes
        self.drop_chars = drop_chars
        self.strip_categorical = strip_categorical
        self.drop_duplicates = drop_duplicates
        self.drop_missing_target = drop_missing_target
        self.encode_target = encode_target

    def fit(self, X: XConstructor | None = None, y: YConstructor | None = None) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        Self
            Instância do estimador.

        """
        Xt = to_df(X)
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        self.mapping_: dict[str, Any] = {}
        self.target_names_in_ = np.array([])
        self._drop_cols = []
        self._estimators = {}

        if not hasattr(self, "missing_"):
            self.missing_ = DEFAULT_MISSING

        self._log("Ajustando Cleaner...", 1)

        if Xt is not None and self.drop_dtypes is not None:
            self._drop_cols = list(Xt.select_dtypes(include=lst(self.drop_dtypes)).columns)

        if yt is not None:
            self.target_names_in_ = np.array(get_col_names(yt))

            if self.drop_chars:
                if isinstance(yt, pd.DataFrame):
                    yt = yt.rename(lambda x: re.sub(self.drop_chars, "", str(x)), axis=1)
                else:
                    yt.name = re.sub(self.drop_chars, "", str(yt.name))

            if self.drop_missing_target:
                yt = replace_missing(yt, self.missing_).dropna(axis=0)

            if self.encode_target:
                for col in get_cols(yt):
                    if isinstance(col.iloc[0], sequence_t):  # Multilabel (múltiplos rótulos)
                        MultiLabelBinarizer = self._get_est_class(
                            name="MultiLabelBinarizer",
                            module="preprocessing",
                        )
                        self._estimators[col.name] = MultiLabelBinarizer().fit(col)
                    elif list(uq := np.unique(col)) != list(range(col.nunique())):
                        LabelEncoder = self._get_est_class("LabelEncoder", "preprocessing")
                        self._estimators[col.name] = LabelEncoder().fit(col)
                        self.mapping_.update(
                            {str(col.name): {str(it(v)): i for i, v in enumerate(uq)}}
                        )

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Retorna os nomes das variáveis após a transformação.

        Parâmetros
        ----------
        input_features: sequence or None, default=None
            Usado apenas para validar os nomes das variáveis com os nomes
            observados durante o `fit`.

        Retorna
        -------
        np.ndarray
            Nomes das variáveis transformadas.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        columns = [col for col in self.feature_names_in_ if col not in self._drop_cols]

        if self.drop_chars:
            # Remove caracteres proibidos dos nomes das colunas
            columns = [re.sub(self.drop_chars, "", str(c)) for c in columns]

        return np.array(columns)

    def transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Aplica as etapas de limpeza de dados.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Conjunto de variáveis com shape=(n_amostras, n_variáveis). Se None,
            `X` é ignorado.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        dataframe
            Conjunto de variáveis transformado. Retornado somente se fornecido.

        series or dataframe
            Coluna alvo transformada. Retornada somente se fornecida.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        yt = to_tabular(y, index=getattr(Xt, "index", None), columns=self.target_names_in_)

        self._log("Limpando os dados...", 1)

        if Xt is not None:
            # Unifica todos os valores ausentes
            Xt = replace_missing(Xt, self.missing_)

            for name, column in Xt.items():
                # Remove variáveis com tipo de dado inválido
                if name in self._drop_cols:
                    self._log(
                        f" --> Removendo a variável {name} por ter o tipo: {column.dtype.name}.",
                        2,
                    )
                    Xt = Xt.drop(columns=name)

                elif column.dtype.name in CAT_TYPES:
                    if self.strip_categorical:
                        # Remove espaços em branco de strings
                        Xt[name] = column.apply(
                            lambda val: val.strip() if isinstance(val, str) else val
                        )

            # Remove caracteres proibidos dos nomes das colunas
            if self.drop_chars:
                Xt = Xt.rename(columns=lambda x: re.sub(self.drop_chars, "", str(x)))

            # Remove amostras duplicadas
            if self.drop_duplicates:
                Xt = Xt.drop_duplicates(ignore_index=True)

            if self.convert_dtypes:
                Xt = Xt.convert_dtypes()

        if yt is not None:
            if self.drop_chars:
                if isinstance(y, pd.Series):
                    yt.name = re.sub(self.drop_chars, "", str(yt.name))
                else:
                    yt = yt.rename(lambda x: re.sub(self.drop_chars, "", str(x)), axis=1)

            # Remove amostras com valores ausentes no alvo
            if self.drop_missing_target:
                length = len(yt)  # Salva o comprimento original para contar as linhas removidas
                yt = replace_missing(yt, self.missing_).dropna()

                if Xt is not None:
                    Xt = Xt[Xt.index.isin(yt.index)]  # Seleciona apenas os índices restantes

                if (d := length - len(yt)) > 0:
                    self._log(f" --> Removendo {d} linhas com valores ausentes no alvo.", 2)

            if self.encode_target and self._estimators:
                y_new = yt.__class__(dtype="object")
                for col in get_cols(yt):
                    if est := self._estimators.get(col.name):
                        if n_cols(out := est.transform(col)) == 1:
                            self._log(f" --> Aplicando label encoding na coluna {col.name}.", 2)
                            out = to_series(out, yt.index, str(col.name))
                        else:
                            self._log(
                                f" --> Aplicando label binarization na coluna {col.name}.", 2
                            )
                            out = to_df(
                                data=out,
                                index=yt.index,
                                columns=[f"{col.name}_{c}" for c in est.classes_],
                            )

                        # Substitui o alvo pela(s) coluna(s) codificada(s)
                        if isinstance(yt, pd.Series):
                            y_new = out
                        else:
                            y_new = merge(y_new, out)

                    else:  # Adiciona coluna inalterada
                        y_new = merge(y_new, col)

                yt = y_new

            if self.convert_dtypes:
                yt = yt.convert_dtypes()

        return variable_return(self._convert(Xt), self._convert(yt))

    def inverse_transform(
        self,
        X: XConstructor | None = None,
        y: YConstructor | None = None,
    ) -> YReturn | tuple[XReturn, YReturn]:
        """Reverte a codificação de rótulos.

        Este método reverte apenas a codificação do alvo.
        As demais transformações não podem ser revertidas. Se
        `encode_target=False`, os dados são retornados como estão.

        Parâmetros
        ----------
        X: dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        y: sequence, dataframe-like or None, default=None
            Coluna(s) alvo correspondente(s) a `X`.

        Retorna
        -------
        dataframe
            Conjunto de variáveis inalterado. Retornado somente se fornecido.

        series or dataframe
            Coluna alvo original. Retornada somente se fornecida.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        yt = to_tabular(y, index=getattr(Xt, "index", None))

        self._log("Revertendo a limpeza dos dados...", 1)

        if yt is not None and self._estimators:
            y_new = yt.__class__(dtype="object")
            for col in self.target_names_in_:
                if est := self._estimators.get(col):
                    if est.__class__.__name__ == "LabelEncoder":
                        self._log(f" --> Revertendo label encoding da coluna {col}.", 2)
                        out = est.inverse_transform(pd.DataFrame(yt)[col])

                    elif isinstance(yt, pd.DataFrame):
                        self._log(f" --> Revertendo label binarization da coluna {col}.", 2)
                        out = est.inverse_transform(
                            yt.loc[:, yt.columns.str.startswith(f"{col}_")].to_numpy()
                        )

                    # Substitui as colunas codificadas pela coluna alvo
                    if isinstance(yt, pd.Series):
                        y_new = to_series(out, yt.index, col)
                    else:
                        y_new = merge(y_new, to_series(out, yt.index, col))

                else:  # Adiciona coluna inalterada
                    y_new = merge(y_new, pd.DataFrame(yt)[col])

            yt = y_new

        return variable_return(self._convert(Xt), self._convert(yt))
