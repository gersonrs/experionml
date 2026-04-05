from __future__ import annotations

from abc import ABCMeta
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, Literal, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from beartype import beartype
from mlflow.tracking import MlflowClient

from experionml.basetracker import BaseTracker
from experionml.basetransformer import BaseTransformer
from experionml.plots.basefigure import BaseFigure
from experionml.utils.constants import PALETTE
from experionml.utils.types import (
    Bool,
    FloatLargerZero,
    FloatZeroToOneExc,
    Int,
    IntLargerZero,
    Legend,
    MetricSelector,
    Model,
    ModelsSelector,
    Pandas,
    PlotBackend,
    RowSelector,
    Scalar,
    Sequence,
    int_t,
    sequence_t,
)
from experionml.utils.utils import (
    Aesthetics,
    check_is_fitted,
    composed,
    crash,
    get_custom_scorer,
    lst,
)


class BasePlot(BaseTransformer, BaseTracker, metaclass=ABCMeta):
    """Classe base abstrata para todos os métodos de plotagem.

    Esta classe base define as propriedades que podem ser alteradas
    para personalizar a estética dos gráficos.

    """

    _fig = BaseFigure()
    _custom_layout: ClassVar[dict[str, Any]] = {}
    _custom_traces: ClassVar[dict[str, Any]] = {}
    _aesthetics = Aesthetics(
        palette=list(PALETTE),
        title_fontsize=24,
        label_fontsize=16,
        tick_fontsize=12,
        line_width=2,
        marker_size=8,
    )

    # Properties =================================================== >>

    @property
    def aesthetics(self) -> Aesthetics:
        """Todos os atributos estéticos dos gráficos."""
        return self._aesthetics

    @aesthetics.setter
    def aesthetics(self, value: dict):
        self.palette = value.get("palette", self.palette)
        self.title_fontsize = value.get("title_fontsize", self.title_fontsize)
        self.label_fontsize = value.get("label_fontsize", self.label_fontsize)
        self.tick_fontsize = value.get("tick_fontsize", self.tick_fontsize)
        self.line_width = value.get("line_width", self.line_width)
        self.marker_size = value.get("marker_size", self.marker_size)

    @property
    def palette(self) -> str | Sequence[str]:
        """Paleta de cores.

        Specify one of plotly's [built-in palettes][palette] or create
        a custom one, e.g., `experionml.palette = ["red", "green", "blue"]`.

        """
        return self._aesthetics.palette

    @palette.setter
    def palette(self, value: str | Sequence[str]):
        if isinstance(value, str) and not hasattr(px.colors.qualitative, value):
            raise ValueError(
                f"Valor inválido para o parâmetro palette: {value}. Escolha "
                f"uma das sequências qualitativas internas do Plotly no módulo "
                f"px.colors.qualitative ou defina sua própria sequência."
            )

        self._aesthetics.palette = value

    @property
    def title_fontsize(self) -> Scalar:
        """Tamanho da fonte do título do gráfico."""
        return self._aesthetics.title_fontsize

    @title_fontsize.setter
    def title_fontsize(self, value: FloatLargerZero):
        self._aesthetics.title_fontsize = value

    @property
    def label_fontsize(self) -> Scalar:
        """Tamanho da fonte dos rótulos, da legenda e do hover."""
        return self._aesthetics.label_fontsize

    @label_fontsize.setter
    def label_fontsize(self, value: FloatLargerZero):
        self._aesthetics.label_fontsize = value

    @property
    def tick_fontsize(self) -> Scalar:
        """Tamanho da fonte das marcações dos eixos do gráfico."""
        return self._aesthetics.tick_fontsize

    @tick_fontsize.setter
    def tick_fontsize(self, value: FloatLargerZero):
        self._aesthetics.tick_fontsize = value

    @property
    def line_width(self) -> Scalar:
        """Largura das linhas do gráfico."""
        return self._aesthetics.line_width

    @line_width.setter
    def line_width(self, value: FloatLargerZero):
        self._aesthetics.line_width = value

    @property
    def marker_size(self) -> Scalar:
        """Tamanho dos marcadores."""
        return self._aesthetics.marker_size

    @marker_size.setter
    def marker_size(self, value: FloatLargerZero):
        self._aesthetics.marker_size = value

    # Methods ====================================================== >>

    @staticmethod
    def _get_plot_index(obj: Pandas) -> pd.Index:
        """Retorna o índice do conjunto em um formato plottável.

        O Plotly aceita apenas formatos desserializáveis por JSON para os eixos.
        Use este método utilitário para converter para timestamp os índices
        que permitem isso (ex.: pd.Period). Caso contrário, retorna como está.

        Parâmetros
        ----------
        obj: pd.Series or pd.DataFrame
            Conjunto de dados de onde o índice será obtido.

        Retorna
        -------
        pd.Index
            Índice em formato desserializável por JSON.

        """
        if hasattr(obj.index, "to_timestamp"):
            return obj.index.to_timestamp()
        else:
            return obj.index

    @staticmethod
    def _get_show(show: IntLargerZero | None, maximum: IntLargerZero = 200) -> Int:
        """Obtém o número de elementos a exibir.

        Sempre limita o máximo de elementos exibidos a 200 para evitar
        um erro de tamanho máximo de figura.

        Parâmetros
        ----------
        show: int or None
            Número de elementos a exibir. Se None, seleciona até 200.

        maximum: int, default=200
            Número máximo de recursos permitidos.

        Retorna
        -------
        int
            Número de recursos a exibir.

        """
        if show is None or show > maximum:
            show_c = min(200, maximum)
        else:
            show_c = show

        return show_c

    @staticmethod
    def _get_set(
        rows: str | Sequence[str] | dict[str, RowSelector],
    ) -> Iterator[tuple[str, RowSelector]]:
        """Obtém a seleção de linhas.

        Converte as linhas fornecidas para um dict onde as chaves são os nomes
        dos conjuntos de dados, e os valores correspondentes são as linhas de
        seleção passadas para branch._get_rows().

        Parâmetros
        ----------
        rows: str, sequence or dict
            Seleção de linhas a plotar.

        Produz
        ------
        str
            Nome do conjunto de linhas.

        RowSelector
            Seleção de linhas.

        """
        if isinstance(rows, sequence_t):
            rows_c = {row: row for row in rows}
        elif isinstance(rows, str):
            rows_c = {rows: rows}
        elif isinstance(rows, dict):
            rows_c = rows

        yield from rows_c.items()

    def _get_metric(self, metric: MetricSelector, *, max_one: Bool = False) -> list[str]:
        """Valida e retorna o índice de métrica fornecido.

        Parâmetros
        ----------
        metric: int, str, sequence or None
            Métrica a recuperar. Se None, todas as métricas são retornadas.

        max_one: bool, default=False
            Se uma ou múltiplas métricas são permitidas. Se True, lança
            uma exceção se mais de uma métrica for selecionada.

        Retorna
        -------
        list of str
            Nomes das métricas selecionadas.

        """
        if metric is None:
            return self._metric.keys()
        else:
            inc: list[str] = []
            for met in lst(metric):
                if isinstance(met, int_t):
                    if int(met) < len(self._metric):
                        inc.append(self._metric[met].name)
                    else:
                        raise ValueError(
                            f"Valor inválido para o parâmetro metric. O valor {met} está "
                            f"fora do intervalo para um pipeline com {len(self._metric)} métricas."
                        )
                elif isinstance(met, str):
                    for m in met.split("+"):
                        if m in ("time_ht", "time_fit", "time_bootstrap", "time"):
                            inc.append(m)
                        elif (name := get_custom_scorer(m).name) in self._metric:
                            inc.append(name)
                        else:
                            raise ValueError(
                                "Valor inválido para o parâmetro metric. A métrica "
                                f"{name} não foi usada para ajustar os modelos."
                            )

        if max_one and len(inc) > 1:
            raise ValueError(
                f"Valor inválido para o parâmetro metric: {metric}. "
                f"Este método de plotagem aceita apenas uma métrica."
            )

        return inc

    def _get_plot_models(
        self,
        models: ModelsSelector,
        *,
        max_one: Bool = False,
        ensembles: Bool = True,
        check_fitted: Bool = True,
    ) -> list[Model]:
        """Se um gráfico for chamado a partir de um modelo, adapta o parâmetro `models`.

        Parâmetros
        ----------
        func: func or None
            Função a decorar. Quando o decorador é chamado sem argumentos
            opcionais, a função é passada como primeiro argumento e o
            decorador apenas retorna a função decorada.

        max_one: bool, default=False
            Se um ou múltiplos modelos são permitidos. Se True, lança
            uma exceção se mais de um modelo for selecionado.

        ensembles: bool, default=True
            Se False, remove modelos ensemble da seleção.

        check_fitted: bool, default=True
            Lança uma exceção se o runner não estiver ajustado (sem modelos).

        Retorna
        -------
        list
            Modelos a plotar.

        """
        if hasattr(self, "_get_models"):
            if check_fitted:
                check_is_fitted(self)

            models_c = self._get_models(models, ensembles=ensembles)
            if max_one and len(models_c) > 1:
                raise ValueError(
                    f"Valor inválido para o parâmetro models: {models_c}. "
                    f"Este método de plotagem aceita apenas um modelo."
                )

            return models_c
        else:
            return [self]  # type: ignore[list-item]

    @overload
    def _get_figure(
        self,
        backend: Literal["plotly"] = ...,
        *,
        create_figure: Literal[True] = ...,
    ) -> go.Figure: ...

    @overload
    def _get_figure(
        self,
        backend: Literal["matplotlib"],
        *,
        create_figure: Literal[True] = ...,
    ) -> plt.Figure: ...

    @overload
    def _get_figure(
        self,
        backend: PlotBackend,
        *,
        create_figure: Literal[False],
    ) -> None: ...

    def _get_figure(
        self,
        backend: PlotBackend = "plotly",
        *,
        create_figure: Bool = True,
    ) -> go.Figure | plt.Figure | None:
        """Retorna uma figura existente se estiver no canvas, caso contrário cria uma nova.

        Cada vez que este método é chamado a partir de um canvas, o índice
        do gráfico é incrementado em um para rastrear em qual subplot o
        BaseFigure se encontra.

        Parâmetros
        ----------
        backend: str, default="plotly"
            Backend da figura. Escolha entre plotly ou matplotlib.

        create_figure: bool, default=True
            Se uma nova figura deve ser criada.

        Retorna
        -------
        [go.Figure][], [plt.Figure][] or None
            Figura existente ou recém-criada. Retorna None se o argumento
            `create_figure=False`.

        """
        if BasePlot._fig and BasePlot._fig.is_canvas:
            return BasePlot._fig.next_subplot
        else:
            BasePlot._fig = BaseFigure(
                palette=self.palette,
                backend=backend,
                create_figure=create_figure,
            )
            return BasePlot._fig.next_subplot

    @staticmethod
    def _draw_diagonal_line(values: tuple, xaxis: str, yaxis: str):
        """Desenha uma linha diagonal ao longo do eixo.

        A linha deve ser usada como referência. Não é adicionada à
        legenda e não exibe nenhuma informação ao passar o cursor.

        Parâmetros
        ----------
        values: tuple of sequence
            Valores dos pontos de dados necessários para determinar os intervalos,
            no formato (x, y).

        xaxis: str
            Nome do eixo x onde será desenhada.

        yaxis: str
            Nome do eixo y onde será desenhada.

        """
        # Obtém os intervalos com margem de 5%, exceto quando o intervalo é exatamente 0-1
        y_min, y_max = min(values[1]), max(values[1])
        if np.issubdtype(type(y_min), np.number) and y_min != 0 and y_max != 1:
            margin = (y_max - y_min) * 0.05
            y_min = y_min - margin
            y_max = y_max + margin

        BasePlot._fig.figure.add_shape(
            type="line",
            x0=y_min,
            x1=y_max,
            y0=y_min,
            y1=y_max,
            xref=xaxis,
            yref=yaxis,
            line={"width": 1, "color": "black"},
            opacity=0.5,
        )

    def _draw_line(
        self,
        parent: str,
        child: str | None = None,
        legend: Legend | dict[str, Any] | None = None,
        **kwargs,
    ):
        """Desenha uma linha na figura atual.

        Unifica o estilo para desenhar uma linha, onde parent e child
        (ex.: modelo - conjunto de dados ou coluna - distribuição) mantêm
        o mesmo estilo (cor ou traço). Um título de legendgroup é adicionado
        apenas quando há um elemento filho.

        Parâmetros
        ----------
        parent: str
            Nome do atributo principal.

        child: str or None, default=None
            Nome do atributo secundário.

        legend: str, dict or None, default=None
            Argumento de legenda fornecido pelo usuário.

        **kwargs
            Argumentos de palavra-chave adicionais para o trace.

        """
        BasePlot._fig.figure.add_scatter(
            name=kwargs.pop("name", child or parent),
            mode=kwargs.pop("mode", "lines"),
            line=kwargs.pop(
                "line",
                {
                    "width": self.line_width,
                    "color": BasePlot._fig.get_elem(parent),
                    "dash": BasePlot._fig.get_elem(child, "dash"),
                },
            ),
            marker=kwargs.pop(
                "marker",
                {
                    "symbol": BasePlot._fig.get_elem(child, "marker"),
                    "size": self.marker_size,
                    "color": BasePlot._fig.get_elem(parent),
                    "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                },
            ),
            hovertemplate=kwargs.pop(
                "hovertemplate",
                f"(%{{x}}, %{{y}})<extra>{parent}{f' - {child}' if child else ''}</extra>",
            ),
            legendgroup=kwargs.pop("legendgroup", parent),
            legendgrouptitle=kwargs.pop(
                "legendgrouptitle",
                {"text": parent, "font_size": self.label_fontsize} if child else None,
            ),
            showlegend=kwargs.pop(
                "showlegend",
                BasePlot._fig.showlegend(f"{parent}-{child}" if child else parent, legend),
            ),
            **kwargs,
        )

    def _plot(
        self,
        fig: go.Figure | plt.Figure | None = None,
        ax: plt.Axes | tuple[str, str] | None = None,
        **kwargs,
    ) -> go.Figure | plt.Figure | None:
        """Gera o gráfico.

        Personaliza os eixos para o layout padrão e plota a figura
        se não for parte de um canvas.

        Parâmetros
        ----------
        fig: go.Figure, plt.Figure or None
            Figura atual. Se None, usa `plt.gcf()`.

        ax: plt.Axes, tuple or None, default=None
            Objeto de eixo ou nomes dos eixos a atualizar. Se None, ignora
            a atualização.

        **kwargs
            Argumentos de palavra-chave contendo os parâmetros da figura.

            - title: Nome do título ou configuração personalizada.
            - legend: Se deve exibir a legenda ou configuração personalizada.
            - xlabel: Rótulo do eixo x.
            - ylabel: Rótulo do eixo y.
            - xlim: Limites do eixo x.
            - ylim: Limites do eixo y.
            - figsize: Tamanho da figura.
            - filename: Nome do arquivo salvo.
            - plotname: Nome do gráfico.
            - display: Se deve exibir o gráfico. Se None, retorna a figura.

        Retorna
        -------
        plt.Figure, go.Figure or None
            Figura criada. Retornada apenas se `display=None`.

        """
        # Define um Path para salvar o arquivo
        if kwargs.get("filename"):
            if (path := Path(kwargs["filename"])).name == "auto":
                path = path.with_name(kwargs["plotname"])
        else:
            path = Path(kwargs.get("plotname", ""))

        fig = fig or BasePlot._fig.figure
        if isinstance(fig, go.Figure):
            if isinstance(ax, tuple):
                # Oculta o rótulo e as marcações dos eixos em subplots não-bordas
                if not BasePlot._fig.sharex or self._fig.grid[0] == self._fig.rows:
                    fig.update_layout(
                        {
                            f"{ax[0]}_title": {
                                "text": kwargs.get("xlabel"),
                                "font_size": self.label_fontsize,
                            }
                        }
                    )
                else:
                    fig.update_layout({f"{ax[0]}_showticklabels": False})

                if not BasePlot._fig.sharey or self._fig.grid[1] == 1:
                    fig.update_layout(
                        {
                            f"{ax[1]}_title": {
                                "text": kwargs.get("ylabel"),
                                "font_size": self.label_fontsize,
                            }
                        }
                    )
                else:
                    fig.update_layout({f"{ax[1]}_showticklabels": False})

                fig.update_layout(
                    {
                        f"{ax[0]}_range": kwargs.get("xlim"),
                        f"{ax[1]}_range": kwargs.get("ylim"),
                        f"{ax[0]}_automargin": True,
                        f"{ax[1]}_automargin": True,
                    }
                )

                if BasePlot._fig.is_canvas and (title := kwargs.get("title")):
                    # Adiciona um subtítulo a um gráfico no canvas
                    default_title = {
                        "x": BasePlot._fig.pos[ax[0][5:] or "1"][0],
                        "y": BasePlot._fig.pos[ax[0][5:] or "1"][1] + 0.005,
                        "xref": "paper",
                        "yref": "paper",
                        "xanchor": "center",
                        "yanchor": "bottom",
                        "showarrow": False,
                        "font_size": self.title_fontsize - 4,
                    }

                    if isinstance(title, dict):
                        title = default_title | title
                    else:
                        title = {"text": title, **default_title}

                    fig.update_layout(annotations=(*fig.layout.annotations, title))

            if not BasePlot._fig.is_canvas and kwargs.get("plotname"):
                default_title = {
                    "x": 0.5,
                    "y": 1,
                    "pad": {"t": 15, "b": 15},
                    "xanchor": "center",
                    "yanchor": "top",
                    "xref": "paper",
                    "font_size": self.title_fontsize,
                }
                if isinstance(title := kwargs.get("title"), dict):
                    title = default_title | title
                else:
                    title = {"text": title, **default_title}

                default_legend = {
                    "traceorder": "grouped",
                    "groupclick": kwargs.get("groupclick", "toggleitem"),
                    "font_size": self.label_fontsize,
                    "bgcolor": "rgba(255, 255, 255, 0.5)",
                }
                if isinstance(legend := kwargs.get("legend"), str):
                    position = {}
                    if legend == "upper left":
                        position = {"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"}
                    elif legend == "lower left":
                        position = {"x": 0.01, "y": 0.01, "xanchor": "left", "yanchor": "bottom"}
                    elif legend == "upper right":
                        position = {"x": 0.99, "y": 0.99, "xanchor": "right", "yanchor": "top"}
                    elif legend == "lower right":
                        position = {"x": 0.99, "y": 0.01, "xanchor": "right", "yanchor": "bottom"}
                    elif legend == "upper center":
                        position = {"x": 0.5, "y": 0.99, "xanchor": "center", "yanchor": "top"}
                    elif legend == "lower center":
                        position = {"x": 0.5, "y": 0.01, "xanchor": "center", "yanchor": "bottom"}
                    elif legend == "center left":
                        position = {"x": 0.01, "y": 0.5, "xanchor": "left", "yanchor": "middle"}
                    elif legend == "center right":
                        position = {"x": 0.99, "y": 0.5, "xanchor": "right", "yanchor": "middle"}
                    elif legend == "center":
                        position = {"x": 0.5, "y": 0.5, "xanchor": "center", "yanchor": "middle"}

                    legend = default_legend | position

                elif isinstance(legend, dict):
                    legend = default_legend | legend

                # Atualiza o layout com as configurações predefinidas
                space1 = self.title_fontsize if title.get("text") else 10
                space2 = self.title_fontsize * int(bool(fig.layout.annotations))
                fig.update_layout(
                    title=title,
                    legend=legend,
                    showlegend=bool(kwargs.get("legend")),
                    hoverlabel={"font_size": self.label_fontsize},
                    font_size=self.tick_fontsize,
                    margin={"l": 50, "b": 50, "r": 0, "t": 25 + space1 + space2, "pad": 0},
                    width=kwargs["figsize"][0],
                    height=kwargs["figsize"][1],
                )

                # Atualiza o gráfico com configurações personalizadas
                fig.update_traces(**self._custom_traces)
                fig.update_layout(**self._custom_layout)

                if kwargs.get("filename"):
                    if path.suffix in ("", ".html"):
                        fig.write_html(path.with_suffix(".html"))
                    else:
                        fig.write_image(path)

                # Registra o gráfico na execução do mlflow para cada modelo visualizado
                if getattr(self, "experiment", None) and self.log_plots:
                    for m in set(BasePlot._fig.used_models):
                        MlflowClient().log_figure(
                            run_id=m.run.info.run_id,
                            figure=fig,
                            artifact_file=str(path.with_suffix(".html")),
                        )

                if kwargs.get("display") is True:
                    fig.show()
                elif kwargs.get("display") is None:
                    return fig

        elif isinstance(fig, plt.Figure):
            if isinstance(ax, plt.Axes):
                if title := kwargs.get("title"):
                    ax.set_title(title, fontsize=self.title_fontsize, pad=20)
                if xlabel := kwargs.get("xlabel"):
                    ax.set_xlabel(xlabel, fontsize=self.label_fontsize, labelpad=12)
                if ylabel := kwargs.get("ylabel"):
                    ax.set_ylabel(ylabel, fontsize=self.label_fontsize, labelpad=12)
                ax.tick_params(axis="both", labelsize=self.tick_fontsize)

            if size := kwargs.get("figsize"):
                # Converte de pixels para polegadas
                fig.set_size_inches(size[0] // fig.get_dpi(), size[1] // fig.get_dpi())

            plt.tight_layout()
            if kwargs.get("filename"):
                fig.savefig(path.with_suffix(".png"))

            # Registra o gráfico na execução do mlflow para cada modelo visualizado
            if self.experiment and self.log_plots:
                for m in set(BasePlot._fig.used_models):
                    MlflowClient().log_figure(
                        run_id=m.run.info.run_id,
                        figure=fig,
                        artifact_file=str(path.with_suffix(".png")),
                    )

            plt.show() if kwargs.get("display") else plt.close()
            if kwargs.get("display") is None:
                return fig

        return None  # display!=None ou não são figuras finais

    @composed(beartype, contextmanager, crash)
    def canvas(
        self,
        rows: IntLargerZero = 1,
        cols: IntLargerZero = 2,
        *,
        sharex: Bool = False,
        sharey: Bool = False,
        hspace: FloatZeroToOneExc = 0.05,
        vspace: FloatZeroToOneExc = 0.07,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "out",
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool = True,
    ):
        """Cria uma figura com múltiplos gráficos.

        Este `@contextmanager` permite desenhar muitos gráficos em uma
        única figura. A opção padrão é adicionar dois gráficos lado a lado.
        Consulte o [guia do usuário][canvas] para um exemplo.

        Parâmetros
        ----------
        rows: int, default=1
            Número de gráficos na altura.

        cols: int, default=2
            Número de gráficos na largura.

        sharex: bool, default=False
            Se True, oculta o rótulo e as marcações dos subplots não-bordas
            no eixo x.

        sharey: bool, default=False
            Se True, oculta o rótulo e as marcações dos subplots não-bordas
            no eixo y.

        hspace: float, default=0.05
            Espaço entre linhas de subplot em coordenadas normalizadas.
            O espaçamento é relativo ao tamanho da figura.

        vspace: float, default=0.07
            Espaço entre colunas de subplot em coordenadas normalizadas.
            O espaçamento é relativo ao tamanho da figura.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: bool, str or dict, default="out"
            Legenda do gráfico. Consulte o [guia do usuário][parameters] para
            uma descrição detalhada das opções.

            - Se None: Nenhuma legenda é exibida.
            - Se str: Posição para exibir a legenda.
            - Se dict: Configuração da legenda.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de gráficos no canvas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf, etc...). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool, default=True
            Se deve renderizar o gráfico.

        Produz
        ------
        [go.Figure][]
            Objeto do gráfico.

        """
        BasePlot._fig = BaseFigure(
            rows=rows,
            cols=cols,
            sharex=sharex,
            sharey=sharey,
            hspace=hspace,
            vspace=vspace,
            palette=self.palette,
            is_canvas=True,
        )

        try:
            yield BasePlot._fig.figure
        finally:
            BasePlot._fig.is_canvas = False  # Fecha o canvas
            self._plot(
                groupclick="togglegroup",
                title=title,
                legend=legend,
                figsize=figsize or (550 + 350 * cols, 200 + 400 * rows),
                plotname="canvas",
                filename=filename,
                display=display,
            )

    @classmethod
    def reset_aesthetics(cls):
        """Redefine os [aesthetics][] do gráfico para seus valores padrão."""
        cls._custom_layout = {}
        cls._custom_traces = {}
        cls._aesthetics = Aesthetics(
            palette=list(PALETTE),
            title_fontsize=24,
            label_fontsize=16,
            tick_fontsize=12,
            line_width=2,
            marker_size=8,
        )

    @classmethod
    def update_layout(cls, **kwargs):
        """Atualiza as propriedades do layout do gráfico.

        Atualiza recursivamente a estrutura do layout original com
        os valores dos argumentos.

        Parâmetros
        ----------
        **kwargs
            Argumentos de palavra-chave para o método [update_layout][] da figura.

        """
        cls._custom_layout = kwargs

    @classmethod
    def update_traces(cls, **kwargs):
        """Atualiza as propriedades dos traces do gráfico.

        Atualiza recursivamente a estrutura dos traces originais com
        os valores dos argumentos.

        Parâmetros
        ----------
        **kwargs
            Argumentos de palavra-chave para o método [update_traces][] da figura.

        """
        cls._custom_traces = kwargs
