from __future__ import annotations

from abc import ABCMeta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from beartype import beartype
from nltk.collocations import (
    BigramCollocationFinder,
    QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from scipy import stats
from scipy.fft import fft
from scipy.signal import periodogram
from sklearn.base import is_classifier
from sklearn.utils.metaestimators import available_if
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, ccf, pacf

from experionml.plots.baseplot import BasePlot
from experionml.utils.constants import PALETTE
from experionml.utils.types import (
    Bool,
    ColumnSelector,
    Int,
    IntLargerZero,
    Legend,
    PACFMethods,
    RowSelector,
    Segment,
    Sequence,
    TargetSelector,
)
from experionml.utils.utils import (
    check_dependency,
    crash,
    divide,
    get_cols,
    get_corpus,
    has_task,
    lst,
    replace_missing,
    rnd,
)


@beartype
class DataPlot(BasePlot, metaclass=ABCMeta):
    """Gráficos de dados.

    Gráficos usados para entendimento e interpretação do conjunto de dados.
    Eles só são acessíveis pelo ExperionML, pois os demais runners devem
    ser usados apenas para treinamento de modelos, não para manipulação de dados.

    """

    @available_if(has_task("forecast"))
    @crash
    def plot_acf(
        self,
        columns: ColumnSelector | None = None,
        *,
        nlags: IntLargerZero | None = None,
        plot_interval: Bool = True,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a função de autocorrelação.

        A função de autocorrelação (ACF) mede a correlação
        entre uma série temporal e versões defasadas de si mesma. A ACF pode
        ajudar a identificar a ordem do processo de média móvel (MA)
        em um modelo de série temporal. Este gráfico está disponível apenas para
        tarefas de [previsão][time-series].

        Parâmetros
        ----------
        columns: int, str, segment, sequence, dataframe or None, default=None
            Colunas das quais plotar a ACF. Se None, seleciona a coluna
            alvo.

        nlags: int or None, default=None
            Número de defasagens para as quais retornar a
            autocorrelação. Se None, usa
            `min(10 * np.log10(len(y)), len(y) // 2 - 1)`. O valor
            retornado inclui a defasagem 0, então o tamanho do vetor é
            `(nlags + 1,)`.

        plot_interval: bool, default=True
            Se deve plotar o intervalo de confiança de 95%.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição detalhada das opções.

            - Se None: nenhuma legenda é exibida.
            - Se str: posição em que a legenda será exibida.
            - Se dict: configuração da legenda.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de defasagens exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_series
        experionml.plots:DataPlot.plot_decomposition
        experionml.plots:DataPlot.plot_pacf

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        experionml = ExperionMLForecaster(y, random_state=1)
        experionml.plot_acf()
        ```

        """
        if columns is None:
            columns_c = lst(self.branch.target)
        else:
            columns_c = self.branch._get_columns(columns)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        if nlags is None:
            nlags = min(int(10 * np.log10(self.branch.shape[0])), self.branch.shape[0] // 2 - 1)

        for col in columns_c:
            # Retorna o array de correlação e o intervalo de confiança
            corr, conf = acf(self.branch.dataset[col], nlags=nlags, alpha=0.05)

            for pos in (x := np.arange(len(corr))):
                fig.add_scatter(
                    x=(pos, pos),
                    y=(0, corr[pos]),
                    mode="lines",
                    line={"width": self.line_width, "color": BasePlot._fig.get_elem(col)},
                    hoverinfo="skip",
                    hovertemplate=None,
                    legendgroup=col,
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

            self._draw_line(
                x=x,
                y=corr,
                parent=col,
                mode="markers",
                legend=legend,
                xaxis=xaxis,
                yaxis=yaxis,
            )

            if plot_interval:
                fig.add_traces(
                    [
                        go.Scatter(
                            x=x,
                            y=conf[:, 1] - corr,
                            mode="lines",
                            line={"width": 1, "color": BasePlot._fig.get_elem(col)},
                            hovertemplate="%{y}<extra>upper bound</extra>",
                            legendgroup=col,
                            showlegend=False,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        ),
                        go.Scatter(
                            x=x,
                            y=conf[:, 0] - corr,
                            mode="lines",
                            line={"width": 1, "color": BasePlot._fig.get_elem(col)},
                            fill="tonexty",
                            fillcolor=f"rgba({BasePlot._fig.get_elem(col)[4:-1]}, 0.2)",
                            hovertemplate="%{y}<extra>lower bound</extra>",
                            legendgroup=col,
                            showlegend=False,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        ),
                    ]
                )

        fig.update_yaxes(zerolinecolor="black")
        fig.update_layout(hovermode="x unified")

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            xlabel="Lag",
            ylabel="Correlation",
            xlim=(-1, nlags + 1),
            title=title,
            legend=legend,
            figsize=figsize or (700 + nlags * 10, 600),
            plotname="plot_acf",
            filename=filename,
            display=display,
        )

    @available_if(has_task("forecast"))
    @crash
    def plot_ccf(
        self,
        columns: ColumnSelector = 0,
        target: TargetSelector = 0,
        *,
        nlags: IntLargerZero | None = None,
        plot_interval: Bool = False,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a correlação cruzada entre duas séries temporais.

        A função de correlação cruzada (CCF) mede a similaridade
        entre as features e a coluna alvo em função do deslocamento
        de uma série em relação à outra. É similar ao gráfico
        [acf][plot_acf], onde a correlação é plotada em relação às
        versões defasadas de si mesma. A faixa transparente representa
        o intervalo de confiança de 95%. Este gráfico está disponível
        apenas para tarefas de [previsão][time-series].

        Parâmetros
        ----------
        columns: int, str, segment, sequence or dataframe, default=0
            Colunas para plotar a CCF. Se None, seleciona
            todas as features numéricas.

        target: int or str, default=0
            Coluna alvo contra a qual calcular as correlações.
            Apenas para tarefas [multivariadas][].

        nlags: int or None, default=None
            Número de defasagens para as quais retornar a autocorrelação.
            Se None, usa `min(10 * np.log10(len(y)), len(y) // 2 - 1)`. O
            valor retornado inclui a defasagem 0 (ou seja, 1), então o tamanho
            do vetor é `(nlags + 1,)`.

        plot_interval: bool, default=False
            Se deve plotar o intervalo de confiança de 95%.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição detalhada das opções.

            - Se None: nenhuma legenda é exibida.
            - Se str: posição em que a legenda será exibida.
            - Se dict: configuração da legenda.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de defasagens exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_series
        experionml.plots:DataPlot.plot_decomposition
        experionml.plots:DataPlot.plot_periodogram

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLForecaster
        from sktime.datasets import load_macroeconomic

        X = load_macroeconomic()

        experionml = ExperionMLForecaster(X, random_state=1)
        experionml.plot_ccf()
        ```

        """
        if self.branch.dataset.shape[1] < 2:
            raise ValueError(
                "O método plot_ccf requer pelo menos duas colunas no conjunto de dados, encontrou 1. "
                "Leia mais sobre o uso de variáveis exógenas no guia do usuário."
            )

        columns_c = self.branch._get_columns(columns, only_numerical=True)
        target_c = self.branch._get_target(target, only_columns=True)

        if nlags is None:
            nlags = min(int(10 * np.log10(self.branch.shape[0])), self.branch.shape[0] // 2 - 1)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for col in columns_c:
            corr, conf = ccf(
                x=self.branch.dataset[target_c],
                y=self.branch.dataset[col],
                nlags=nlags,
                alpha=0.05,
            )

            for pos in (x := np.arange(len(corr))):
                fig.add_scatter(
                    x=(pos, pos),
                    y=(0, corr[pos]),
                    mode="lines",
                    line={"width": self.line_width, "color": BasePlot._fig.get_elem(col)},
                    hoverinfo="skip",
                    hovertemplate=None,
                    legendgroup=col,
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

            self._draw_line(
                x=x,
                y=corr,
                parent=col,
                mode="markers",
                legend=legend,
                xaxis=xaxis,
                yaxis=yaxis,
            )

            if plot_interval:
                fig.add_traces(
                    [
                        go.Scatter(
                            x=x,
                            y=conf[:, 1] - corr,
                            mode="lines",
                            line={"width": 1, "color": BasePlot._fig.get_elem(col)},
                            hovertemplate="%{y}<extra>upper bound</extra>",
                            legendgroup=col,
                            showlegend=False,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        ),
                        go.Scatter(
                            x=x,
                            y=conf[:, 0] - corr,
                            mode="lines",
                            line={"width": 1, "color": BasePlot._fig.get_elem(col)},
                            fill="tonexty",
                            fillcolor=f"rgba({BasePlot._fig.get_elem(col)[4:-1]}, 0.2)",
                            hovertemplate="%{y}<extra>lower bound</extra>",
                            legendgroup=col,
                            showlegend=False,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        ),
                    ]
                )

        fig.update_yaxes(zerolinecolor="black")
        fig.update_layout(hovermode="x unified")

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            xlabel="Lag",
            ylabel="Correlation",
            xlim=(-1, nlags),
            title=title,
            legend=legend,
            figsize=figsize or (700 + nlags * 10, 600),
            plotname="plot_ccf",
            filename=filename,
            display=display,
        )

    @crash
    def plot_components(
        self,
        show: IntLargerZero | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a razão de variância explicada por componente.

        Os componentes mantidos são coloridos e os descartados são
        transparentes. Este gráfico está disponível apenas quando a
        seleção de features foi aplicada com strategy="pca".

        Parâmetros
        ----------
        show: int or None, default=None
            Número de componentes a exibir. None para exibir todos.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="lower right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição detalhada das opções.

            - Se None: nenhuma legenda é exibida.
            - Se str: posição em que a legenda será exibida.
            - Se dict: configuração da legenda.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de componentes exibidos.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_pca
        experionml.plots:DataPlot.plot_rfecv

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.feature_selection("pca", n_features=5)
        experionml.plot_components(show=10)
        ```

        """
        if not hasattr(self, "pca_"):
            raise PermissionError(
                "O método plot_pca está disponível apenas para instâncias "
                "que executaram a seleção de features com a estratégia 'pca', "
                "por exemplo: experionml.feature_selection(strategy='pca')."
            )

        # Obtém a razão de variância por componente
        variance = np.array(self.pca_.explained_variance_ratio_)

        show_c = self._get_show(show, len(variance))

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        # Cria um esquema de cores: primeiro normal e depois totalmente transparente
        color = BasePlot._fig.get_elem("components")
        opacity = [0.2] * self.pca_._comps + [0] * (len(variance) - self.pca_._comps)

        fig.add_bar(
            x=variance,
            y=[f"pca{i}" for i in range(len(variance))],
            orientation="h",
            marker={
                "color": [f"rgba({color[4:-1]}, {o})" for o in opacity],
                "line": {"width": 2, "color": color},
            },
            hovertemplate="%{x}<extra></extra>",
            name=f"Variance retained: {variance[:self.pca_._comps].sum():.3f}",
            legendgroup="components",
            showlegend=BasePlot._fig.showlegend("components", legend),
            xaxis=xaxis,
            yaxis=yaxis,
        )

        fig.update_layout({f"yaxis{yaxis[1:]}": {"categoryorder": "total ascending"}})

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Explained variance ratio",
            ylim=(len(variance) - show_c - 0.5, len(variance) - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_components",
            filename=filename,
            display=display,
        )

    @crash
    def plot_correlation(
        self,
        columns: Segment | Sequence[Int | str] | pd.DataFrame | None = None,
        method: Literal["pearson", "kendall", "spearman"] = "pearson",
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (800, 700),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota uma matriz de correlação.

        Exibe um mapa de calor mostrando a correlação entre colunas do
        conjunto de dados. As cores vermelho, azul e branco representam
        correlação positiva, negativa e nula, respectivamente.

        Parâmetros
        ----------
        columns: segment, sequence, dataframe or None, default=None
            Colunas a plotar. Se None, plota todas as colunas do conjunto
            de dados. Colunas categóricas selecionadas são ignoradas.

        method: str, default="pearson"
            Método de correlação. Escolha entre: pearson, kendall ou
            spearman.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Sem efeito. Implementado para continuidade da API.

        figsize: tuple, default=(800, 700)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_distribution
        experionml.plots:DataPlot.plot_qq
        experionml.plots:DataPlot.plot_relationships

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.plot_correlation()
        ```

        """
        columns_c = self.branch._get_columns(columns, only_numerical=True)

        # Calcula a matriz de correlação
        corr = self.branch.dataset[columns_c].corr(method=method)

        # Gera uma máscara para o triângulo inferior
        # k=1 significa manter a linha diagonal mais externa
        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(
            x=(0, 0.87),
            coloraxis={
                "colorscale": "rdbu_r",
                "cmin": -1,
                "cmax": 1,
                "title": f"{method} correlation",
                "font_size": self.label_fontsize,
            },
        )

        fig.add_heatmap(
            z=corr.mask(mask),
            x=columns_c,
            y=columns_c,
            coloraxis=f"coloraxis{xaxis[1:]}",
            hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
            hoverongaps=False,
            showlegend=False,
            xaxis=xaxis,
            yaxis=yaxis,
        )

        fig.update_layout(
            {
                "template": "plotly_white",
                f"yaxis{yaxis[1:]}_autorange": "reversed",
                f"xaxis{xaxis[1:]}_showgrid": False,
                f"yaxis{yaxis[1:]}_showgrid": False,
            }
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_correlation",
            filename=filename,
            display=display,
        )

    @crash
    def plot_data_splits(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Visualiza as divisões dos dados.

        Plota as divisões treino/teste/holdout. O eixo x mostra o
        número de linhas, onde cada ponto corresponde à n-ésima
        amostra. Adicionalmente, rótulos de classe e [grupos][metadata]
        são plotados quando relevantes.

        Parâmetros
        ----------
        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição detalhada das opções.

            - Se None: nenhuma legenda é exibida.
            - Se str: posição em que a legenda será exibida.
            - Se dict: configuração da legenda.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:PredictionPlot.plot_cv_splits
        experionml.plots:DataPlot.plot_decomposition
        experionml.plots:DataPlot.plot_relationships

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier, ExperionMLForecaster
        from random import choices
        from sklearn.datasets import load_breast_cancer
        from sktime.datasets import load_airline

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        groups = choices(["A", "B", "C", "D"], k=X.shape[0])

        experionml = ExperionMLClassifier(
            X,
            y=y,
            metadata={"groups": groups},
            n_rows=0.2,
            holdout_size=0.1,
            random_state=1,
        )
        experionml.run("LR")
        experionml.plot_data_splits()
        ```

        """
        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        all_reset = self.branch._all.reset_index()
        for ds in ["train", "test"] + ([] if self.branch.holdout is None else ["holdout"]):
            if self.task.is_forecast:
                x = self._get_plot_index(getattr(self.branch, ds))
            else:
                x = all_reset[all_reset["index"].isin(getattr(self.branch, ds).index)].index

            self._draw_line(
                x=x,
                y=["data"] * len(x),
                parent=ds,
                mode="markers",
                marker={
                    "symbol": "line-ns",
                    "size": 25,
                    "line": {
                        "width": self.marker_size,
                        "color": f"rgba({BasePlot._fig.get_elem(ds)[4:-1]}, 1)",
                    },
                },
                hovertemplate=f"%{{y}}: {ds}<extra></extra>",
                legend=legend,
                xaxis=xaxis,
                yaxis=yaxis,
            )

        if self.task.is_classification:
            for col in get_cols(self.branch._all[self.branch.target]):
                mapping = self.branch.mapping.get(col.name, {k: k for k in np.unique(col)})
                inverse_mapping = {v: k for k, v in mapping.items()}

                self._draw_line(
                    x=(x2 := list(range(self.branch._all.shape[0]))),
                    y=[col.name] * len(x2),
                    parent=str(col.name),
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 25,
                        "line": {
                            "width": self.marker_size,
                            "color": [f"rgba({BasePlot._fig.get_elem(i)[4:-1]}, 1)" for i in col],
                        },
                    },
                    customdata=[inverse_mapping[i] for i in self.branch._all[col.name]],
                    hovertemplate=f"{col.name}: %{{customdata}}<extra></extra>",
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        if (groups := self._config.get_groups()) is not None:
            self._draw_line(
                x=(x3 := list(range(self.branch._all.shape[0]))),
                y=["group"] * len(x3),
                parent="group",
                mode="markers",
                marker={
                    "symbol": "line-ns",
                    "size": 25,
                    "line": {
                        "width": self.marker_size,
                        "color": [
                            f"rgba({BasePlot._fig.get_elem(f'g{i}')[4:-1]}, 1)" for i in groups
                        ],
                    },
                },
                customdata=groups,
                hovertemplate="group: %{customdata}<extra></extra>",
                legend=legend,
                xaxis=xaxis,
                yaxis=yaxis,
            )

        fig.update_yaxes(autorange="reversed")
        fig.update_layout(hovermode="x unified")

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            xlabel="Rows",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_data_splits",
            filename=filename,
            display=display,
        )

    @available_if(has_task("forecast"))
    @crash
    def plot_decomposition(
        self,
        columns: ColumnSelector | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 900),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a tendência, sazonalidade e resíduos de uma série temporal.

        Este gráfico está disponível apenas para tarefas de [previsão][time-series].

        !!! tip
            Use o método [decompose][experionmlforecaster-decompose] do ExperionML
            para remover tendência e sazonalidade dos dados.

        Parâmetros
        ----------
        columns: int, str, segment, sequence, dataframe or None, default=None
            [Seleção de colunas][row-and-column-selection] para plotar.
            Se None, a coluna alvo é selecionada.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper left"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição detalhada das opções.

            - Se None: nenhuma legenda é exibida.
            - Se str: posição em que a legenda será exibida.
            - Se dict: configuração da legenda.

        figsize: tuple, default=(900, 900)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_acf
        experionml.plots:DataPlot.plot_pacf
        experionml.plots:DataPlot.plot_series

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        experionml = ExperionMLForecaster(y, random_state=1)
        experionml.plot_decomposition()
        ```

        """
        if columns is None:
            columns_c = lst(self.branch.target)
        else:
            columns_c = self.branch._get_columns(columns)

        self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(y=(0.76, 1.0))
        xaxis2, yaxis2 = BasePlot._fig.get_axes(y=(0.51, 0.74))
        xaxis3, yaxis3 = BasePlot._fig.get_axes(y=(0.26, 0.49))
        xaxis4, yaxis4 = BasePlot._fig.get_axes(y=(0.0, 0.24))

        for col in columns_c:
            # Retorna o array de correlação e o intervalo de confiança
            decompose = seasonal_decompose(
                x=self.branch.dataset[col],
                model=self.sp.get("seasonal_model", "additive"),
                period=self.sp.get("sp", self._get_sp(self.branch.dataset.index.freqstr)),
            )

            self._draw_line(
                x=(x := self._get_plot_index(decompose.trend)),
                y=decompose.observed,
                parent=col,
                legend=legend,
                xaxis=xaxis4,
                yaxis=yaxis,
            )

            self._draw_line(x=x, y=decompose.trend, parent=col, xaxis=xaxis4, yaxis=yaxis2)
            self._draw_line(x=x, y=decompose.seasonal, parent=col, xaxis=xaxis4, yaxis=yaxis3)
            self._draw_line(x=x, y=decompose.resid, parent=col, xaxis=xaxis4, yaxis=yaxis4)

        self._plot(ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"), ylabel="Trend")
        self._plot(ax=(f"xaxis{xaxis3[1:]}", f"yaxis{yaxis3[1:]}"), ylabel="Seasonal")
        self._plot(
            ax=(f"xaxis{xaxis4[1:]}", f"yaxis{yaxis4[1:]}"),
            ylabel="Residual",
            xlabel=self.branch.dataset.index.name or "index",
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            ylabel="Observed",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_decomposition",
            filename=filename,
            display=display,
        )

    @crash
    def plot_distribution(
        self,
        columns: ColumnSelector = 0,
        distributions: str | Sequence[str] | None = "kde",
        show: IntLargerZero | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota as distribuições das colunas.

        - Para colunas numéricas, plota a distribuição de densidade de
          probabilidade. Adicionalmente, é possível plotar qualquer
          distribuição de `scipy.stats` ajustada à coluna.
        - Para colunas categóricas, plota a distribuição de classes.
          Apenas uma coluna categórica pode ser plotada por vez.

        !!! tip
            Use o método [distributions][experionmlclassifier-distributions] do
            ExperionML para verificar qual distribuição melhor se ajusta à coluna.

        Parâmetros
        ----------
        columns: int, str, slice or sequence, default=0
            Colunas a plotar. Só é possível plotar uma coluna categórica.
            Se mais de uma coluna categórica for selecionada,
            todas as colunas categóricas são ignoradas.

        distributions: str, sequence or None, default="kde"
            Distribuições a ajustar. Apenas para colunas numéricas.

            - Se None: Nenhuma distribuição é ajustada.
            - Se "kde": Ajusta uma [distribuição Gaussian kde][kde].
            - Caso contrário: Nome de uma distribuição de `scipy.stats`.

        show: int or None, default=None
            Número de classes (ordenadas por número de ocorrências) a
            exibir no gráfico. Se None, exibe todas as classes. Apenas
            para colunas categóricas.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None: Nenhum título é exibido.
            - Se str: Texto do título.
            - Se dict: [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição detalhada das opções.

            - Se None: nenhuma legenda é exibida.
            - Se str: posição em que a legenda será exibida.
            - Se dict: configuração da legenda.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao tipo do gráfico.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_correlation
        experionml.plots:DataPlot.plot_qq
        experionml.plots:DataPlot.plot_relationships

        Exemplos
        --------
        ```pycon
        import numpy as np
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        # Add a categorical feature
        animals = ["cat", "dog", "bird", "lion", "zebra"]
        probabilities = [0.001, 0.1, 0.2, 0.3, 0.399]
        X["animals"] = np.random.choice(animals, size=len(X), p=probabilities)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.plot_distribution(columns=[0, 1])
        experionml.plot_distribution(columns=0, distributions=["norm", "invgauss"])
        experionml.plot_distribution(columns="animals")
        ```

        """
        columns_c = self.branch._get_columns(columns)
        num_columns = self.branch.dataset.select_dtypes(include="number").columns

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        if len(columns_c) == 1 and columns_c[0] not in num_columns:
            series = self.branch.dataset[columns_c[0]].value_counts(ascending=True)
            show_c = self._get_show(show, len(series))

            color = BasePlot._fig.get_elem()
            fig.add_bar(
                x=series,
                y=series.index,
                orientation="h",
                marker={
                    "color": f"rgba({color[4:-1]}, 0.2)",
                    "line": {"width": 2, "color": color},
                },
                hovertemplate="%{x}<extra></extra>",
                name=f"{columns_c[0]}: {len(series)} classes",
                showlegend=BasePlot._fig.showlegend("dist", legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )

            return self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel="Counts",
                ylim=(len(series) - show_c - 0.5, len(series) - 0.5),
                title=title,
                legend=legend,
                figsize=figsize or (900, 400 + show_c * 50),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

        else:
            for col in [c for c in columns_c if c in num_columns]:
                fig.add_histogram(
                    x=self.branch.dataset[col],
                    histnorm="probability density",
                    marker={
                        "color": f"rgba({BasePlot._fig.get_elem(col)[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": BasePlot._fig.get_elem(col)},
                    },
                    nbinsx=40,
                    name="dist",
                    legendgroup=col,
                    legendgrouptitle={"text": col, "font_size": self.label_fontsize},
                    showlegend=BasePlot._fig.showlegend(f"{col}-dist", legend),
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                x = np.linspace(
                    start=self.branch.dataset[col].min(),
                    stop=self.branch.dataset[col].max(),
                    num=200,
                )

                # Remove valores ausentes para compatibilidade com scipy.stats
                values = replace_missing(self.branch.dataset[col], self.missing).dropna()
                values = values.to_numpy(dtype=float)

                if distributions is not None:
                    # Obtém uma linha para cada distribuição
                    for dist in lst(distributions):
                        if dist == "kde":
                            y = stats.gaussian_kde(values)(x)
                        else:
                            params = getattr(stats, dist).fit(values)
                            y = getattr(stats, dist).pdf(x, *params)

                        self._draw_line(
                            x=x,
                            y=y,
                            parent=col,
                            child=dist,
                            legend=legend,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        )

            fig.update_layout(barmode="overlay")

            return self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel="Values",
                ylabel="Probability density",
                title=title,
                legend=legend,
                figsize=figsize or (900, 600),
                plotname="plot_distribution",
                filename=filename,
                display=display,
            )

    @available_if(has_task("forecast"))
    @crash
    def plot_fft(
        self,
        columns: ColumnSelector | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a transformada de Fourier de uma série temporal.

        Um gráfico de Transformada Rápida de Fourier (FFT) visualiza a
        representação no domínio da frequência de um sinal, transformando-o
        do domínio temporal para o domínio da frequência usando o algoritmo
        FFT. O eixo x mostra as frequências, normalizadas para a
        [frequência de Nyquist][nyquist], e o eixo y mostra a densidade
        espectral de potência ou amplitude quadrática por unidade de
        frequência em escala logarítmica. Este gráfico está disponível
        apenas para tarefas de [previsão][time-series].

        !!! tip
            - Se o gráfico apresentar pico em f~0, pode indicar o
              comportamento errante característico de um
              [passeio aleatório][random_walk] que precisa ser diferenciado.
              Também pode ser indicativo de um processo estacionário
              [ARMA][] com alto valor positivo de phi.
            - Pico em uma frequência e seus múltiplos é indicativo de
              sazonalidade. A menor frequência nesse caso é chamada de
              frequência fundamental, e o inverso dessa frequência é o
              período sazonal dos dados.

        Parâmetros
        ----------
        columns: int, str, segment, sequence, dataframe or None, default=None
            Colunas para plotar o periodograma. Se None, seleciona
            a coluna alvo.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição detalhada das opções.

            - Se None: nenhuma legenda é exibida.
            - Se str: posição em que a legenda será exibida.
            - Se dict: configuração da legenda.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_series
        experionml.plots:DataPlot.plot_decomposition
        experionml.plots:DataPlot.plot_periodogram

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        experionml = ExperionMLForecaster(y, random_state=1)
        experionml.plot_fft()
        ```

        """
        if columns is None:
            columns_c = lst(self.branch.target)
        else:
            columns_c = self.branch._get_columns(columns)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for col in columns_c:
            fft_values = fft(self.branch.dataset[col].to_numpy(), workers=self.n_jobs)
            psd = np.abs(fft_values) ** 2
            freq = np.fft.fftfreq(len(psd))

            self._draw_line(
                x=freq[freq >= 0],  # Desenha apenas >0 pois a FFT é espelhada em torno de x=0
                y=psd[freq >= 0],
                parent=col,
                mode="lines+markers",
                legend=legend,
                xaxis=xaxis,
                yaxis=yaxis,
            )

        fig.update_yaxes(type="log")

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Frequency",
            ylabel="PSD",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_fft",
            filename=filename,
            display=display,
        )

    @crash
    def plot_ngrams(
        self,
        ngram: Literal[1, 2, 3, 4, "word", "bigram", "trigram", "quadgram"] = "bigram",
        rows: RowSelector | None = "dataset",
        show: IntLargerZero | None = 10,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota as frequências de n-gramas.

        O texto para o gráfico é extraído da coluna chamada
        `corpus`. Se não houver coluna com esse nome, uma exceção
        é lançada. Se os documentos não estiverem tokenizados, as palavras são
        separadas por espaços.

        !!! tip
            Use o método [tokenize][experionmlclassifier-tokenize] do experionml para
            separar as palavras criando n-gramas com base em sua frequência
            no corpus.

        Parâmetros
        ----------
        ngram: str or int, default="bigram"
            Número de palavras contíguas a procurar (tamanho do n-grama).
            Escolha entre: word (1), bigram (2), trigram (3), quadgram (4).

        rows: hashable, segment, sequence or dataframe, default="dataset"
            [Seleção de linhas][row-and-column-selection] no corpus
            a incluir na busca.

        show: int or None, default=10
            Número de n-gramas (ordenados por número de ocorrências) a
            exibir no gráfico. Se none, exibe todos os n-gramas (até 200).

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="lower right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters] para
            uma descrição detalhada das opções.

            - Se None: Nenhuma legenda é exibida.
            - Se str: Posição para exibir a legenda.
            - Se dict: Configuração da legenda.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            ajusta o tamanho ao número de n-gramas exibidos.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_wordcloud

        Exemplos
        --------
        ```pycon
        import numpy as np
        from experionml import ExperionMLClassifier
        from sklearn.datasets import fetch_20newsgroups

        X, y = fetch_20newsgroups(
            return_X_y=True,
            categories=["alt.atheism", "sci.med", "comp.windows.x"],
            shuffle=True,
            random_state=1,
        )
        X = np.array(X).reshape(-1, 1)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.textclean()
        experionml.textnormalize()
        experionml.plot_ngrams()
        ```

        """

        def get_text(column: pd.Series) -> pd.Series:
            """Obtém o corpus completo como sequência de tokens.

            Parâmetros
            ----------
            column: series
                Coluna contendo o corpus.

            Retorna
            -------
            series
                Corpus de tokens.

            """
            if isinstance(column.iloc[0], str):
                return column.apply(lambda row: row.split())
            else:
                return column

        corpus = get_corpus(self.branch.X)
        rows_c = self.branch._get_rows(rows)
        show_c = self._get_show(show)

        if str(ngram) in ("1", "word"):
            ngram_c = "words"
            series = pd.Series(
                [word for row in get_text(rows_c[corpus]) for word in row]
            ).value_counts(ascending=True)
        else:
            if str(ngram) in ("2", "bigram"):
                ngram_c, finder = "bigrams", BigramCollocationFinder
            elif str(ngram) in ("3", "trigram"):
                ngram_c, finder = "trigrams", TrigramCollocationFinder
            elif str(ngram) in ("4", "quadgram"):
                ngram_c, finder = "quadgrams", QuadgramCollocationFinder

            ngram_fd = finder.from_documents(get_text(rows_c[corpus])).ngram_fd
            series = pd.Series(
                data=[x[1] for x in ngram_fd.items()],
                index=[" ".join(x[0]) for x in ngram_fd.items()],
            ).sort_values(ascending=True)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        fig.add_bar(
            x=(data := series[-self._get_show(show, len(series)) :]),
            y=data.index,
            orientation="h",
            marker={
                "color": f"rgba({BasePlot._fig.get_elem(ngram_c)[4:-1]}, 0.2)",
                "line": {"width": 2, "color": BasePlot._fig.get_elem(ngram_c)},
            },
            hovertemplate="%{x}<extra></extra>",
            name=f"Total {ngram_c}: {len(series)}",
            legendgroup=ngram_c,
            showlegend=BasePlot._fig.showlegend(ngram_c, legend),
            xaxis=xaxis,
            yaxis=yaxis,
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Counts",
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_ngrams",
            filename=filename,
            display=display,
        )

    @available_if(has_task("forecast"))
    @crash
    def plot_pacf(
        self,
        columns: ColumnSelector | None = None,
        *,
        nlags: IntLargerZero | None = None,
        method: PACFMethods = "ywadjusted",
        plot_interval: Bool = True,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a função de autocorrelação parcial.

        A função de autocorrelação parcial (PACF) mede a
        correlação entre uma série temporal e versões defasadas de
        si mesma, após remover os efeitos de valores com defasagens menores.
        Em outras palavras, representa a correlação entre duas
        variáveis controlando a influência de outras variáveis.
        O PACF pode ajudar a identificar a ordem do processo
        autorregressivo (AR) em um modelo de série temporal. Este
        gráfico está disponível apenas para tarefas de [previsão][time-series].

        Parâmetros
        ----------
        columns: int, str, segment, sequence, dataframe or None, default=None
            Colunas para plotar o pacf. Se None, seleciona a
            coluna alvo.

        nlags: int or None, default=None
            Número de defasagens para retornar a autocorrelação. Se None,
            usa `min(10 * np.log10(len(y)), len(y) // 2 - 1)`. O
            valor retornado inclui a defasagem 0 (i.e., 1), portanto o tamanho do
            vetor é `(nlags + 1,)`.

        method : str, default="ywadjusted"
            Especifica qual método usar para os cálculos.

            - "yw" ou "ywadjusted": Yule-Walker com ajuste pelo tamanho da
              amostra no denominador para acovf.
            - "ywm" ou "ywmle": Yule-Walker sem ajuste.
            - "ols": Regressão da série temporal sobre suas defasagens e uma
              constante.
            - "ols-inefficient": Regressão da série temporal sobre defasagens usando
              uma única amostra comum para estimar todos os coeficientes pacf.
            - "ols-adjusted": Regressão da série temporal sobre defasagens com
              ajuste de viés.
            - "ld" ou "ldadjusted": Recursão de Levinson-Durbin com
              correção de viés.
            - "ldb" ou "ldbiased": Recursão de Levinson-Durbin sem
              correção de viés.
            - "burg": Estimador de autocorrelação parcial de Burg.

        plot_interval: bool, default=True
            Se deve plotar o intervalo de confiança de 95%.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters] para
            uma descrição detalhada das opções.

            - Se None: Nenhuma legenda é exibida.
            - Se str: Posição para exibir a legenda.
            - Se dict: Configuração da legenda.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            ajusta o tamanho ao número de defasagens exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_acf
        experionml.plots:DataPlot.plot_decomposition
        experionml.plots:DataPlot.plot_series

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        experionml = ExperionMLForecaster(y, random_state=1)
        experionml.plot_pacf()
        ```

        """
        if columns is None:
            columns_c = lst(self.branch.target)
        else:
            columns_c = self.branch._get_columns(columns)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        if nlags is None:
            nlags = min(int(10 * np.log10(self.branch.shape[0])), self.branch.shape[0] // 2 - 1)

        for col in columns_c:
            # Retorna o array de correlação e o intervalo de confiança
            corr, conf = pacf(self.branch.dataset[col], nlags=nlags, method=method, alpha=0.05)

            for pos in (x := np.arange(len(corr))):
                fig.add_scatter(
                    x=(pos, pos),
                    y=(0, corr[pos]),
                    mode="lines",
                    line={"width": self.line_width, "color": BasePlot._fig.get_elem(col)},
                    hoverinfo="skip",
                    hovertemplate=None,
                    legendgroup=col,
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

            self._draw_line(
                x=x,
                y=corr,
                parent=col,
                mode="markers",
                legend=legend,
                xaxis=xaxis,
                yaxis=yaxis,
            )

            if plot_interval:
                fig.add_traces(
                    [
                        go.Scatter(
                            x=x,
                            y=conf[:, 1] - corr,
                            mode="lines",
                            line={"width": 1, "color": BasePlot._fig.get_elem(col)},
                            hovertemplate="%{y}<extra>upper bound</extra>",
                            legendgroup=col,
                            showlegend=False,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        ),
                        go.Scatter(
                            x=x,
                            y=conf[:, 0] - corr,
                            mode="lines",
                            line={"width": 1, "color": BasePlot._fig.get_elem(col)},
                            fill="tonexty",
                            fillcolor=f"rgba({BasePlot._fig.get_elem(col)[4:-1]}, 0.2)",
                            hovertemplate="%{y}<extra>lower bound</extra>",
                            legendgroup=col,
                            showlegend=False,
                            xaxis=xaxis,
                            yaxis=yaxis,
                        ),
                    ]
                )

        fig.update_yaxes(zerolinecolor="black")
        fig.update_layout(hovermode="x unified")

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            xlabel="Lag",
            ylabel="Correlation",
            xlim=(-1, nlags + 1),
            title=title,
            legend=legend,
            figsize=figsize or (700 + nlags * 10, 600),
            plotname="plot_pacf",
            filename=filename,
            display=display,
        )

    @crash
    def plot_pca(
        self,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a razão de variância explicada em função do número de componentes.

        Se o estimador subjacente for [PCA][] (para conjuntos de dados densos),
        todos os componentes possíveis são plotados. Se o estimador subjacente
        for [TruncatedSVD][] (para conjuntos de dados esparsos), apenas os
        componentes selecionados são exibidos. A estrela marca o número de componentes
        selecionados pelo usuário. Este gráfico está disponível apenas quando a
        seleção de features foi aplicada com strategy="pca".

        Parâmetros
        ----------
        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para continuidade da API.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_components
        experionml.plots:DataPlot.plot_rfecv

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.feature_selection("pca", n_features=5)
        experionml.plot_pca()
        ```

        """
        if not hasattr(self, "pca_"):
            raise PermissionError(
                "O método plot_components está disponível apenas para instâncias "
                "que executaram seleção de features usando a estratégia 'pca', "
                "ex.: experionml.feature_selection(strategy='pca')."
            )

        # Cria o símbolo de estrela no número de componentes selecionados
        symbols = ["circle"] * self.pca_.n_features_in_
        symbols[self.pca_._comps - 1] = "star"
        sizes = [self.marker_size] * self.pca_.n_features_in_
        sizes[self.pca_._comps - 1] = self.marker_size * 1.5

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()
        fig.add_scatter(
            x=tuple(range(1, self.pca_.n_features_in_ + 1)),
            y=np.cumsum(self.pca_.explained_variance_ratio_),
            mode="lines+markers",
            line={"width": self.line_width, "color": BasePlot._fig.get_elem("pca")},
            marker={
                "symbol": symbols,
                "size": sizes,
                "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                "opacity": 1,
            },
            hovertemplate="%{y}<extra></extra>",
            showlegend=False,
            xaxis=xaxis,
            yaxis=yaxis,
        )

        fig.update_layout(
            {
                "hovermode": "x",
                f"xaxis{xaxis[1:]}_showspikes": True,
                f"yaxis{yaxis[1:]}_showspikes": True,
            }
        )

        margin = self.pca_.n_features_in_ / 30
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="First N principal components",
            ylabel="Cumulative variance ratio",
            xlim=(1 - margin, self.pca_.n_features_in_ - 1 + margin),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_pca",
            filename=filename,
            display=display,
        )

    @available_if(has_task("forecast"))
    @crash
    def plot_periodogram(
        self,
        columns: ColumnSelector | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a densidade espectral de uma série temporal.

        Um gráfico de periodograma é usado para visualizar o conteúdo de frequência
        de um sinal de série temporal. É particularmente útil na análise de
        séries temporais para identificar frequências dominantes, padrões
        periódicos e características espectrais gerais dos dados.
        O eixo x mostra as frequências, normalizadas pela
        [frequência de Nyquist][nyquist], e o eixo y mostra a densidade
        espectral de potência ou amplitude ao quadrado por unidade de frequência
        em escala logarítmica. Este gráfico está disponível apenas para
        tarefas de [previsão][time-series].

        !!! tip
            - Se o gráfico apresentar pico em f~0, pode indicar o comportamento
              errante característico de um [passeio aleatório][random_walk]
              que precisa ser diferenciado. Também pode ser indicativo
              de um processo [ARMA][] estacionário com valor phi positivo alto.
            - Pico em uma frequência e seus múltiplos é indicativo de
              sazonalidade. A frequência mais baixa nesse caso é chamada
              de frequência fundamental, e o inverso dessa
              frequência é o período sazonal dos dados.

        Parâmetros
        ----------
        columns: int, str, segment, sequence, dataframe or None, default=None
            Colunas para plotar o periodograma. Se None, seleciona
            a coluna alvo.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters] para
            uma descrição detalhada das opções.

            - Se None: Nenhuma legenda é exibida.
            - Se str: Posição para exibir a legenda.
            - Se dict: Configuração da legenda.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_series
        experionml.plots:DataPlot.plot_decomposition
        experionml.plots:DataPlot.plot_fft

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        experionml = ExperionMLForecaster(y, random_state=1)
        experionml.plot_periodogram()
        ```

        """
        if columns is None:
            columns_c = lst(self.branch.target)
        else:
            columns_c = self.branch._get_columns(columns)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for col in columns_c:
            freq, psd = periodogram(self.branch.dataset[col], window="parzen")

            self._draw_line(
                x=freq,
                y=psd,
                parent=col,
                mode="lines+markers",
                legend=legend,
                xaxis=xaxis,
                yaxis=yaxis,
            )

        fig.update_yaxes(type="log")

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Frequency",
            ylabel="PSD",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_periodogram",
            filename=filename,
            display=display,
        )

    @crash
    def plot_qq(
        self,
        columns: ColumnSelector = 0,
        distributions: str | Sequence[str] = "norm",
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota um gráfico quantil-quantil.

        As colunas são distinguidas por cor e as distribuições são
        distinguidas pelo tipo de marcador. Valores ausentes são ignorados.

        Parâmetros
        ----------
        columns: int, str, segment, sequence or dataframe, default=0
            Colunas para plotar. Colunas categóricas selecionadas são ignoradas.

        distributions: str or sequence, default="norm"
            Nomes das distribuições `scipy.stats` para ajustar às
            colunas.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="lower right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters] para
            uma descrição detalhada das opções.

            - Se None: Nenhuma legenda é exibida.
            - Se str: Posição para exibir a legenda.
            - Se dict: Configuração da legenda.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_correlation
        experionml.plots:DataPlot.plot_distribution
        experionml.plots:DataPlot.plot_relationships

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.plot_qq(columns=[5, 6])
        experionml.plot_qq(columns=0, distributions=["norm", "invgauss", "triang"])
        ```

        """
        columns_c = self.branch._get_columns(columns)

        self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        percentiles = np.linspace(0, 100, 101)
        for col in columns_c:
            # Remove valores ausentes para compatibilidade com scipy.stats
            values = replace_missing(self.branch.dataset[col], self.missing).dropna()
            values = values.to_numpy(dtype=float)

            for dist in lst(distributions):
                stat = getattr(stats, dist)
                params = stat.fit(values)
                samples = stat.rvs(*params, size=101, random_state=self.random_state)

                self._draw_line(
                    x=(x := np.percentile(samples, percentiles)),
                    y=(y := np.percentile(values, percentiles)),
                    mode="markers",
                    parent=col,
                    child=dist,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        self._draw_diagonal_line((x, y), xaxis=xaxis, yaxis=yaxis)

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Theoretical quantiles",
            ylabel="Observed quantiles",
            title=title,
            legend=legend,
            figsize=figsize or (900, 600),
            plotname="plot_qq",
            filename=filename,
            display=display,
        )

    @crash
    def plot_relationships(
        self,
        columns: Segment | Sequence[Int | str] | pd.DataFrame = (0, 1, 2),
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 900),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota relações pareadas em um conjunto de dados.

        Cria uma grade de eixos de forma que cada coluna numérica apareça
        uma vez nos eixos x e uma vez nos eixos y. O triângulo inferior
        contém gráficos de dispersão (no máximo 250 amostras aleatórias), a diagonal
        contém distribuições das colunas e o triângulo superior
        contém histogramas de contorno para todas as amostras das colunas.

        Parâmetros
        ----------
        columns: segment, sequence or dataframe, default=(0, 1, 2)
            Colunas para plotar. Colunas categóricas selecionadas são ignoradas.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para continuidade da API.

        figsize: tuple, default=(900, 900)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_correlation
        experionml.plots:DataPlot.plot_distribution
        experionml.plots:DataPlot.plot_qq

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.plot_relationships(columns=[0, 4, 5])
        ```

        """
        columns_c = self.branch._get_columns(columns, only_numerical=True)

        # Usa no máximo 250 amostras para não poluir o gráfico
        sample = lambda col: self.branch.dataset[col].sample(
            n=min(len(self.branch.dataset), 250), random_state=self.random_state
        )

        fig = self._get_figure()
        color = BasePlot._fig.get_elem()
        for i in range(len(columns_c) ** 2):
            x, y = i // len(columns_c), i % len(columns_c)

            # Calcula a distância entre os subgráficos
            offset = divide(0.0125, (len(columns_c) - 1))

            # Calcula o tamanho do subgráfico
            size = (1 - ((offset * 2) * (len(columns_c) - 1))) / len(columns_c)

            # Determina a posição dos eixos
            x_pos = y * (size + 2 * offset)
            y_pos = (len(columns_c) - x - 1) * (size + 2 * offset)

            xaxis, yaxis = BasePlot._fig.get_axes(
                x=(x_pos, rnd(x_pos + size)),
                y=(y_pos, rnd(y_pos + size)),
                coloraxis={
                    "colorscale": PALETTE.get(color, "Blues"),
                    "cmin": 0,
                    "cmax": len(self.branch.dataset),
                    "showscale": False,
                },
            )

            if x == y:
                fig.add_histogram(
                    x=self.branch.dataset[columns_c[x]],
                    marker={
                        "color": f"rgba({color[4:-1]}, 0.2)",
                        "line": {"width": 2, "color": color},
                    },
                    name=columns_c[x],
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            elif x > y:
                fig.add_scatter(
                    x=sample(columns_c[y]),
                    y=sample(columns_c[x]),
                    mode="markers",
                    marker={"color": color},
                    hovertemplate="(%{x}, %{y})<extra></extra>",
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
            elif y > x:
                fig.add_histogram2dcontour(
                    x=self.branch.dataset[columns_c[y]],
                    y=self.branch.dataset[columns_c[x]],
                    coloraxis=f"coloraxis{xaxis[1:]}",
                    hovertemplate="x:%{x}<br>y:%{y}<br>z:%{z}<extra></extra>",
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

            if x < len(columns_c) - 1:
                fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
            if y > 0:
                fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

            self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel=columns_c[y] if x == len(columns_c) - 1 else None,
                ylabel=columns_c[x] if y == 0 else None,
            )

        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (900, 900),
            plotname="plot_relationships",
            filename=filename,
            display=display,
        )

    @crash
    def plot_rfecv(
        self,
        *,
        plot_interval: Bool = True,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota os resultados do RFECV.

        Plota as pontuações obtidas pelo estimador ajustado em cada
        subconjunto do dataset. Disponível apenas quando a seleção de
        features foi aplicada com strategy="rfecv".

        Parâmetros
        ----------
        plot_interval: bool, default=True
            Se deve plotar o intervalo de confiança de 1 sigma.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters] para
            uma descrição detalhada das opções.

            - Se None: Nenhuma legenda é exibida.
            - Se str: Posição para exibir a legenda.
            - Se dict: Configuração da legenda.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_components
        experionml.plots:DataPlot.plot_pca

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.feature_selection("rfecv", solver="Tree")
        experionml.plot_rfecv()
        ```

        """
        if not hasattr(self, "rfecv_"):
            raise PermissionError(
                "O método plot_rfecv está disponível apenas para instâncias "
                "que executaram seleção de features usando a estratégia 'rfecv', "
                "ex.: experionml.feature_selection(strategy='rfecv')."
            )

        try:  # Define o rótulo do eixo y para o gráfico
            ylabel = self.rfecv_.get_params()["scoring"].name
        except AttributeError:
            ylabel = "accuracy" if is_classifier(self.rfecv_.estimator_) else "r2"

        x = np.arange(self.rfecv_.min_features_to_select, self.rfecv_.n_features_in_ + 1)

        # Cria o símbolo de estrela no número de features selecionadas
        sizes = [6] * len(x)
        sizes[self.rfecv_.n_features_ - self.rfecv_.min_features_to_select] = 12
        symbols = ["circle"] * len(x)
        symbols[self.rfecv_.n_features_ - self.rfecv_.min_features_to_select] = "star"

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        mean = self.rfecv_.cv_results_["mean_test_score"]
        std = self.rfecv_.cv_results_["std_test_score"]

        self._draw_line(
            x=x,
            y=mean,
            parent=ylabel,
            mode="lines+markers",
            marker={
                "symbol": symbols,
                "size": sizes,
                "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                "opacity": 1,
            },
            legend=legend,
            xaxis=xaxis,
            yaxis=yaxis,
        )

        if plot_interval:
            fig.add_traces(
                [
                    go.Scatter(
                        x=x,
                        y=mean + std,
                        mode="lines",
                        line={"width": 1, "color": BasePlot._fig.get_elem(ylabel)},
                        hovertemplate="%{y}<extra>upper bound</extra>",
                        legendgroup=ylabel,
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    ),
                    go.Scatter(
                        x=x,
                        y=mean - std,
                        mode="lines",
                        line={"width": 1, "color": BasePlot._fig.get_elem(ylabel)},
                        fill="tonexty",
                        fillcolor=f"rgba{BasePlot._fig.get_elem(ylabel)[3:-1]}, 0.2)",
                        hovertemplate="%{y}<extra>lower bound</extra>",
                        legendgroup=ylabel,
                        showlegend=False,
                        xaxis=xaxis,
                        yaxis=yaxis,
                    ),
                ]
            )

        fig.update_layout(hovermode="x unified")

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            xlabel="Number of features",
            ylabel=ylabel,
            xlim=(min(x) - len(x) / 30, max(x) + len(x) / 30),
            ylim=(min(mean) - 3 * max(std), max(mean) + 3 * max(std)),
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_rfecv",
            filename=filename,
            display=display,
        )

    @available_if(has_task("forecast"))
    @crash
    def plot_series(
        self,
        rows: str | Sequence[str] | dict[str, RowSelector] = ("train", "test"),
        columns: ColumnSelector | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota uma série de dados.

        Este gráfico está disponível apenas para tarefas de [previsão][time-series].

        Parâmetros
        ----------
        rows: str, sequence or dict, default=("train", "test")
            Seleção de linhas para plotar.

            - Se str: Nome do conjunto de dados para plotar.
            - Se sequence: Nomes dos conjuntos de dados para plotar.
            - Se dict: Nomes dos conjuntos com a
              [seleção de linhas][row-and-column-selection] correspondente como valores.

        columns: int, str, segment, sequence, dataframe or None, default=None
            [Colunas][row-and-column-selection] para plotar. Se None, todas
            as colunas alvo são selecionadas.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper left"
            Legenda do gráfico. Consulte o [guia do usuário][parameters] para
            uma descrição detalhada das opções.

            - Se None: Nenhuma legenda é exibida.
            - Se str: Posição para exibir a legenda.
            - Se dict: Configuração da legenda.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_distribution
        experionml.plots:DataPlot.plot_relationships
        experionml.plots:DataPlot.plot_qq

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLForecaster
        from sktime.datasets import load_airline

        y = load_airline()

        experionml = ExperionMLForecaster(y, random_state=1)
        experionml.plot_series()
        ```

        """
        if columns is None:
            columns_c = lst(self.target)
        else:
            columns_c = self.branch._get_columns(columns, include_target=True)

        self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for col in columns_c:
            for child, ds in self._get_set(rows):
                self._draw_line(
                    x=self._get_plot_index(y := self.branch._get_rows(ds)[col]),
                    y=y,
                    parent=col,
                    child=child,
                    mode="lines+markers",
                    marker={
                        "size": self.marker_size,
                        "color": BasePlot._fig.get_elem(col),
                        "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                    },
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel=self.branch.dataset.index.name or "index",
            ylabel="Values",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_series",
            filename=filename,
            display=display,
        )

    @crash
    def plot_wordcloud(
        self,
        rows: RowSelector = "dataset",
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
        **kwargs,
    ) -> go.Figure | None:
        """Plota uma nuvem de palavras a partir do corpus.

        O texto para o gráfico é extraído da coluna chamada
        `corpus`. Se não houver coluna com esse nome, uma exceção
        é lançada.

        Parâmetros
        ----------
        rows: hashable, segment, sequence or dataframe, default="dataset"
            [Seleção de linhas][row-and-column-selection] no corpus
            a incluir na nuvem de palavras.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto para o título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para continuidade da API.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomenclatura
            automática. O tipo de arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver tipo de arquivo,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        **kwargs
            Argumentos de palavra-chave adicionais para o objeto [Wordcloud][].

        Retorna
        -------
        [go.Figure][] or None
            Objeto do gráfico. Retornado apenas se `display=None`.

        Veja também
        --------
        experionml.plots:DataPlot.plot_ngrams
        experionml.plots:PredictionPlot.plot_pipeline

        Exemplos
        --------
        ```pycon
        import numpy as np
        from experionml import ExperionMLClassifier
        from sklearn.datasets import fetch_20newsgroups

        X, y = fetch_20newsgroups(
            return_X_y=True,
            categories=["alt.atheism", "sci.med", "comp.windows.x"],
            shuffle=True,
            random_state=1,
        )
        X = np.array(X).reshape(-1, 1)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.textclean()
        experionml.textnormalize()
        experionml.plot_wordcloud()
        ```

        """

        def get_text(column):
            """Obtém o corpus completo como uma única string longa."""
            if isinstance(column.iloc[0], str):
                return " ".join(column)
            else:
                return " ".join([" ".join(row) for row in column])

        check_dependency("wordcloud")
        from wordcloud import WordCloud

        corpus = get_corpus(self.branch.X)
        rows_c = self.branch._get_rows(rows)

        wordcloud = WordCloud(
            width=figsize[0],
            height=figsize[1],
            background_color=kwargs.pop("background_color", "white"),
            random_state=kwargs.pop("random_state", self.random_state),
            **kwargs,
        )

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        fig.add_image(
            z=wordcloud.generate(get_text(rows_c[corpus])),
            hoverinfo="skip",
            xaxis=xaxis,
            yaxis=yaxis,
        )

        fig.update_layout(
            {
                f"xaxis{xaxis[1:]}_showticklabels": False,
                f"yaxis{xaxis[1:]}_showticklabels": False,
            }
        )

        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize or (900, 600),
            plotname="plot_wordcloud",
            filename=filename,
            display=display,
        )
