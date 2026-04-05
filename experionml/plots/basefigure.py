from __future__ import annotations

from itertools import cycle
from typing import Any, Literal

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from experionml.utils.types import (
    Bool,
    FloatZeroToOneExc,
    Int,
    IntLargerZero,
    Legend,
    Model,
    PlotBackend,
    Scalar,
    Sequence,
    Style,
    sequence_t,
)
from experionml.utils.utils import divide, rnd, to_rgb


class BaseFigure:
    """Figura base do Plotly.

    A instância armazena a posição dos eixos atuais na grade, bem como
    os modelos usados no gráfico (para rastreamento no mlflow).

    Parâmetros
    ----------
    rows: int, default=1
        Número de linhas de subplots no canvas.

    cols: int, default=1
        Número de colunas de subplots no canvas.

    sharex: bool, default=False
        Se True, oculta o rótulo e os ticks dos subplots que não estão
        na borda do eixo x.

    sharey: bool, default=False
        Se True, oculta o rótulo e os ticks dos subplots que não estão
        na borda do eixo y.

    hspace: float, default=0.05
        Espaço entre linhas de subplots em coordenadas normalizadas.
        O espaçamento é relativo ao tamanho da figura.

    vspace: float, default=0.07
        Espaço entre colunas de subplots em coordenadas normalizadas.
        O espaçamento é relativo ao tamanho da figura.

    palette: str or sequence, default="Prism"
        Nome ou sequência de cores para a paleta.

    is_canvas: bool, default=False
        Se o gráfico exibe múltiplos plots.

    backend: str, default="plotly"
        Backend do gráfico. Escolha entre plotly ou matplotlib.

    create_figure: bool, default=True
        Se deve criar uma nova figura.

    """

    _marker = ("circle", "x", "diamond", "pentagon", "star", "hexagon")
    _dash = ("solid", "dashdot", "dash", "dot", "longdash", "longdashdot")
    _shape = ("", "/", "x", "\\", "-", "|", "+", ".")

    def __init__(
        self,
        rows: IntLargerZero = 1,
        cols: IntLargerZero = 1,
        *,
        sharex: Bool = False,
        sharey: Bool = False,
        hspace: FloatZeroToOneExc = 0.05,
        vspace: FloatZeroToOneExc = 0.07,
        palette: str | Sequence[str] = "Prism",
        is_canvas: Bool = False,
        backend: PlotBackend = "plotly",
        create_figure: Bool = True,
    ):
        self.rows = rows
        self.cols = cols
        self.sharex = sharex
        self.sharey = sharey
        self.hspace = hspace
        self.vspace = vspace
        if isinstance(palette, str):
            self._palette = getattr(px.colors.qualitative, palette)
        elif isinstance(palette, sequence_t):
            # Converte nomes de cores ou hex para rgb
            self._palette = list(map(to_rgb, palette))
        self.is_canvas = is_canvas
        self.backend = backend
        self.create_figure = create_figure

        self.idx = 0  # N-th plot in the canvas
        self.axes = 0  # N-th axis in the canvas
        if self.create_figure:
            if self.backend == "plotly":
                self.figure = go.Figure()
            else:
                self.figure, _ = plt.subplots()

        self.groups: list[str] = []
        self.style: Style = {"palette": {}, "marker": {}, "dash": {}, "shape": {}}
        self.palette = cycle(self._palette)
        self.marker = cycle(self._marker)
        self.dash = cycle(self._dash)
        self.shape = cycle(self._shape)

        self.pos: dict[str, tuple[Scalar, Scalar]] = {}  # Subplot position for title
        self.custom_layout: dict[str, Any] = {}  # Layout params specified by user
        self.used_models: list[Model] = []  # Models plotted in this figure

    @property
    def grid(self) -> tuple[Int, Int]:
        """Posição dos eixos atuais na grade.

        Retorna
        -------
        int
            Posição em X.

        int
            Posição em Y.

        """
        return (self.idx - 1) // self.cols + 1, self.idx % self.cols or self.cols

    @property
    def next_subplot(self) -> go.Figure | plt.Figure | None:
        """Avança o índice do subplot.

        Retorna
        -------
        go.Figure, plt.Figure or None
            Figura atual. Retorna None se `create_figure=False`.

        """
        # Check if there are too many plots in the canvas
        if self.idx >= self.rows * self.cols:
            raise ValueError(
                "Número inválido de gráficos na tela. Aumente o número "
                "de linhas e colunas para adicionar mais gráficos."
            )
        else:
            self.idx += 1

        if self.create_figure:
            return self.figure
        else:
            return None

    def get_elem(
        self,
        name: str | None = None,
        element: Literal["palette", "marker", "dash", "shape"] = "palette",
    ) -> str:
        """Obtém o elemento do gráfico para um nome específico.

        Este método é usado para atribuir o mesmo elemento (cor, marcador,
        etc.) às mesmas colunas e modelos em um plot.

        Parâmetros
        ----------
        name: int, float or str or None, default=None
            Nome para o qual obter o elemento do plot. O nome é armazenado
            nos atributos de elemento para atribuir o mesmo elemento a todas
            as chamadas com o mesmo nome. Se None, retorna o primeiro elemento.

        element: str, default="palette"
            Elemento do plot a obter. Escolha entre: palette, marker, dash, shape.

        Retorna
        -------
        str
            Código do elemento.

        """
        if name is None:
            return getattr(self, f"_{element}")[0]  # Get first element (default)
        elif name in self.style[element]:
            return self.style[element][name]
        else:
            return self.style[element].setdefault(name, next(getattr(self, element)))

    def showlegend(self, name: str, legend: Legend | dict[str, Any] | None) -> bool:
        """Retorna se o trace deve ser exibido na legenda.

        Se já houver um trace com o mesmo nome, não é necessário
        exibi-lo novamente na legenda do plot.

        Parâmetros
        ----------
        name: str
            Nome do trace.

        legend: str, dict or None
            Parâmetro de legenda.

        Retorna
        -------
        bool
            Se o trace deve aparecer na legenda.

        """
        if name in self.groups:
            return False
        else:
            self.groups.append(name)
            return legend is not None

    def get_axes(
        self,
        x: tuple[Scalar, Scalar] = (0, 1),
        y: tuple[Scalar, Scalar] = (0, 1),
        coloraxis: dict[str, Any] | None = None,
    ) -> tuple[str, str]:
        """Cria e atualiza os eixos do gráfico.

        Parâmetros
        ----------
        x: tuple
            Tamanho relativo no eixo x do plot.

        y: tuple
            Tamanho relativo no eixo y do plot.

        coloraxis: dict or None
            Propriedades do coloraxis a criar. None para ignorar.

        Retorna
        -------
        str
            Nome do eixo x.

        str
            Nome do eixo y.

        """
        self.axes += 1

        # Calcula a distância entre os subplots
        x_offset = divide(self.hspace, (self.cols - 1))
        y_offset = divide(self.vspace, (self.rows - 1))

        # Calcula o tamanho do subplot
        x_size = (1 - ((x_offset * 2) * (self.cols - 1))) / self.cols
        y_size = (1 - ((y_offset * 2) * (self.rows - 1))) / self.rows

        # Calcula o tamanho dos eixos
        ax_size = (x[1] - x[0]) * x_size
        ay_size = (y[1] - y[0]) * y_size

        # Determina a posição dos eixos
        x_pos = (self.grid[1] - 1) * (x_size + 2 * x_offset) + x[0] * x_size
        y_pos = (self.rows - self.grid[0]) * (y_size + 2 * y_offset) + y[0] * y_size

        # Armazena posições para o título do subplot
        self.pos[str(self.axes)] = (x_pos + ax_size / 2, rnd(y_pos + ay_size))

        # Atualiza a figura com os novos eixos
        self.figure.update_layout(
            {
                f"xaxis{self.axes}": {
                    "domain": (x_pos, rnd(x_pos + ax_size)),
                    "anchor": f"y{self.axes}",
                },
                f"yaxis{self.axes}": {
                    "domain": (y_pos, rnd(y_pos + ay_size)),
                    "anchor": f"x{self.axes}",
                },
            }
        )

        # Coloca uma barra de cores à direita dos eixos
        if coloraxis:
            if title := coloraxis.pop("title", None):
                coloraxis["colorbar_title"] = {
                    "text": title,
                    "side": "right",
                    "font_size": coloraxis.pop("font_size"),
                }

            coloraxis["colorbar_x"] = rnd(x_pos + ax_size) + ax_size / 40
            coloraxis["colorbar_xanchor"] = "left"
            coloraxis["colorbar_y"] = y_pos + ay_size / 2
            coloraxis["colorbar_yanchor"] = "middle"
            coloraxis["colorbar_len"] = ay_size * 0.9
            coloraxis["colorbar_thickness"] = ax_size * 30  # Largura padrão em pixels
            self.figure.update_layout({f"coloraxis{coloraxis.pop('axes', self.axes)}": coloraxis})

        xaxis = f"x{self.axes if self.axes > 1 else ''}"
        yaxis = f"y{self.axes if self.axes > 1 else ''}"
        return xaxis, yaxis
