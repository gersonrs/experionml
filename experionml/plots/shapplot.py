from __future__ import annotations

from abc import ABCMeta
from collections.abc import Hashable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import shap
from beartype import beartype
from sklearn.utils.metaestimators import available_if

from experionml.plots.baseplot import BasePlot
from experionml.utils.types import (
    Bool,
    ColumnSelector,
    Int,
    IntLargerZero,
    Legend,
    ModelSelector,
    RowSelector,
    TargetsSelector,
)
from experionml.utils.utils import check_canvas, crash, has_task


@beartype
class ShapPlot(BasePlot, metaclass=ABCMeta):
    """Gráficos SHAP.

    ExperionML wrapper for plots made by the shap package, using Shapley
    values for model interpretation. These plots are accessible from
    the runners or from the models. Only one model can be plotted at
    the same time since the plots are not made by ExperionML.

    """

    @available_if(has_task("!forecast"))
    @crash
    def plot_shap_bar(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plota o gráfico de barras do SHAP.

        Create a bar plot of a set of SHAP values. If a single sample
        is passed, then the SHAP values are plotted. If many samples
        are passed, then the mean absolute value for each feature
        column is plotted. Read more about SHAP plots in the
        [user guide][shap].

        Parâmetros
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_shap_bar()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] a plotar.

        show: int or None, default=None
            Número de features, ordenadas por importância, a mostrar. Se
            None, mostra todas as features.

        target: int, str or tuple, default=1
            Classe da coluna alvo a ser considerada. Para tarefas com
            múltiplas saídas, o valor deve ser uma tupla no formato
            (coluna, classe). Note que, em tarefas binárias e
            multirrótulo, a classe selecionada é sempre a positiva.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de features exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como png. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [plt.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        See Also
        --------
        experionml.plots:PredictionPlot.plot_parshap
        experionml.plots:ShapPlot.plot_shap_beeswarm
        experionml.plots:ShapPlot.plot_shap_scatter

        Examples
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("LR")
        experionml.plot_shap_bar(show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_bar")

        shap.plots.bar(explanation, max_display=show_c, show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_shap_bar",
            filename=filename,
            display=display,
        )

    @available_if(has_task("!forecast"))
    @crash
    def plot_shap_beeswarm(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plota o gráfico beeswarm do SHAP.

        O gráfico é colorido pelos valores das features. Leia mais sobre
        gráficos SHAP no [guia do usuário][shap].

        Parâmetros
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_shap_beeswarm()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] a plotar. O
            método plot_shap_beeswarm não suporta plotar uma única
            amostra.

        show: int or None, default=None
            Número de features, ordenadas por importância, a mostrar. Se
            None, mostra todas as features.

        target: int, str or tuple, default=1
            Classe da coluna alvo a ser considerada. Para tarefas com
            múltiplas saídas, o valor deve ser uma tupla no formato
            (coluna, classe). Note que, em tarefas binárias e
            multirrótulo, a classe selecionada é sempre a positiva.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de features exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como png. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [plt.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        See Also
        --------
        experionml.plots:PredictionPlot.plot_parshap
        experionml.plots:ShapPlot.plot_shap_bar
        experionml.plots:ShapPlot.plot_shap_scatter

        Examples
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("LR")
        experionml.plot_shap_beeswarm(show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_beeswarm")

        shap.plots.beeswarm(explanation, max_display=show_c, show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            filename=filename,
            display=display,
        )

    @available_if(has_task("!forecast"))
    @crash
    def plot_shap_decision(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plota o gráfico de decisão do SHAP.

        Visualiza decisões do modelo usando valores SHAP cumulativos.
        Cada linha plotada explica uma única predição do modelo. Se uma
        única predição for plotada, os valores das features são
        impressos no gráfico, se fornecidos. Se múltiplas predições
        forem plotadas juntas, os valores das features não serão
        exibidos. Plotar predições demais ao mesmo tempo torna o gráfico
        ininteligível. Leia mais sobre gráficos SHAP no [guia do
        usuário][shap].

        Parâmetros
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_shap_decision()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] a plotar.

        show: int or None, default=None
            Número de features, ordenadas por importância, a mostrar. Se
            None, mostra todas as features.

        target: int, str or tuple, default=1
            Classe da coluna alvo a ser considerada. Para tarefas com
            múltiplas saídas, o valor deve ser uma tupla no formato
            (coluna, classe). Note que, em tarefas binárias e
            multirrótulo, a classe selecionada é sempre a positiva.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de features exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como png. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [plt.Figure][] or None
            Objeto do gráfico. Só é retornado se `display=None`.

        See Also
        --------
        experionml.plots:ShapPlot.plot_shap_bar
        experionml.plots:ShapPlot.plot_shap_beeswarm
        experionml.plots:ShapPlot.plot_shap_force

        Examples
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("LR")
        experionml.plot_shap_decision(show=10)
        experionml.plot_shap_decision(rows=-1, show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_decision")

        shap.decision_plot(
            base_value=explanation.base_values,
            shap_values=explanation.values,
            features=X.columns,
            feature_display_range=slice(-1, -show_c - 1, -1),
            auto_size_plot=False,
            show=False,
        )

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_shap_decision",
            filename=filename,
            display=display,
        )

    @available_if(has_task("!forecast"))
    @crash
    def plot_shap_force(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 300),
        filename: str | Path | None = None,
        display: Bool | None = True,
        **kwargs,
    ) -> plt.Figure | None:
        """Plota o gráfico de força do SHAP.

        Visualiza os valores SHAP fornecidos com um layout de força
        aditivo. Note que, por padrão, este gráfico será renderizado
        usando javascript. Para uma figura comum, use
        `matplotlib=True`, opção disponível apenas quando uma única
        amostra é plotada. Leia mais sobre gráficos SHAP no [guia do
        usuário][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_shap_force()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] a plotar.

        target: int, str or tuple, default=1
            Classe da coluna alvo a ser considerada. Para tarefas com
            múltiplas saídas, o valor deve ser uma tupla no formato
            (coluna, classe). Note que, em tarefas binárias e
            multirrótulo, a classe selecionada é sempre a positiva.

        title: str, dict or None, default=None
            Title for the plot.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=(900, 300)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como png. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura,
            apenas se `matplotlib=True` em `kwargs`.

        **kwargs
            Argumentos nomeados adicionais para
            [shap.plots.force][force].

        Retorna
        -------
        [plt.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        See Also
        --------
        experionml.plots:ShapPlot.plot_shap_beeswarm
        experionml.plots:ShapPlot.plot_shap_scatter
        experionml.plots:ShapPlot.plot_shap_decision

        Examples
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("LR")
        experionml.plot_shap_force(rows=-2, matplotlib=True, figsize=(1800, 300))
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(create_figure=False, backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_force")

        plot = shap.force_plot(
            base_value=explanation.base_values,
            shap_values=explanation.values,
            features=X.columns,
            show=False,
            **kwargs,
        )

        if kwargs.get("matplotlib"):
            BasePlot._fig.used_models.append(models_c)
            return self._plot(
                fig=plt.gcf(),
                ax=plt.gca(),
                title=title,
                legend=legend,
                figsize=figsize,
                plotname="plot_shap_force",
                filename=filename,
                display=display,
            )
        else:
            if filename:
                if (path := Path(filename)).suffix != ".html":
                    path = path.with_suffix(".html")
                shap.save_html(str(path), plot)
            if display and find_spec("IPython"):
                from IPython.display import display as ipydisplay

                shap.initjs()
                ipydisplay(plot)

            return None

    @available_if(has_task("!forecast"))
    @crash
    def plot_shap_heatmap(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plota o mapa de calor do SHAP.

        Este gráfico foi projetado para mostrar a subestrutura da
        população de um conjunto de dados usando agrupamento
        supervisionado e um mapa de calor. O agrupamento supervisionado
        agrupa os pontos de dados não pelos valores originais das
        features, mas por suas explicações. Leia mais sobre gráficos
        SHAP no [guia do usuário][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_shap_heatmap()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] a plotar. O
            método plot_shap_heatmap não suporta plotar uma única
            amostra.

        show: int or None, default=None
            Número de features, ordenadas por importância, a mostrar. Se
            None, mostra todas as features.

        target: int, str or tuple, default=1
            Classe da coluna alvo a ser considerada. Para tarefas com
            múltiplas saídas, o valor deve ser uma tupla no formato
            (coluna, classe). Note que, em tarefas binárias e
            multirrótulo, a classe selecionada é sempre a positiva.

        title: str, dict or None, default=None
            Title for the plot.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de features exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como png. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [plt.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        See Also
        --------
        experionml.plots:ShapPlot.plot_shap_decision
        experionml.plots:ShapPlot.plot_shap_force
        experionml.plots:ShapPlot.plot_shap_waterfall

        Examples
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("LR")
        experionml.plot_shap_heatmap(show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_heatmap")

        shap.plots.heatmap(explanation, max_display=show_c, show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_shap_heatmap",
            filename=filename,
            display=display,
        )

    @available_if(has_task("!forecast"))
    @crash
    def plot_shap_scatter(
        self,
        models: ModelSelector | None = None,
        rows: RowSelector = "test",
        columns: ColumnSelector = 0,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plota o gráfico de dispersão do SHAP.

        Plots the value of the feature on the x-axis and the SHAP value
        of the same feature on the y-axis. This shows how the model
        depends on the given feature, and is like a richer extension of
        the classical partial dependence plots. Vertical dispersion of
        the data points represents interaction effects. Read more about
        SHAP plots in the [user guide][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_shap_scatter()`.

        rows: hashable, segment, sequence or dataframe, default="test"
            [Seleção de linhas][row-and-column-selection] a plotar. O
            método plot_shap_scatter não suporta plotar uma única
            amostra.

        columns: int, str, segment, sequence or dataframe, default=0
            [Feature][row-and-column-selection] to plot. Only a single
            feature can be selected.

        target: int, str or tuple, default=1
            Class in the target column to target. For multioutput tasks,
            the value should be a tuple of the form (column, class).
            Note that for binary and multilabel tasks, the selected
            class is always the positive one.

        title: str, dict or None, default=None
            Title for the plot.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y).

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como png. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [plt.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        See Also
        --------
        experionml.plots:ShapPlot.plot_shap_beeswarm
        experionml.plots:ShapPlot.plot_shap_decision
        experionml.plots:ShapPlot.plot_shap_force

        Examples
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("LR")
        experionml.plot_shap_scatter(columns="symmetry error")
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        columns_c = models_c.branch._get_columns(columns, include_target=False)

        if len(columns_c) > 1:
            raise ValueError(
                f"Valor inválido para o parâmetro columns, recebeu {columns_c}. "
                f"Selecione no máximo uma feature, recebeu {len(columns_c)}."
            )
        else:
            col = columns_c[0]

        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        # Get explanation for a specific column
        explanation = explanation[:, models_c.branch.columns.get_loc(col)]

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_scatter")

        shap.plots.scatter(explanation, color=explanation, ax=plt.gca(), show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            xlabel=plt.gca().get_xlabel(),
            ylabel=plt.gca().get_ylabel(),
            title=title,
            legend=legend,
            plotname="plot_shap_scatter",
            figsize=figsize,
            filename=filename,
            display=display,
        )

    @available_if(has_task("!forecast"))
    @crash
    def plot_shap_waterfall(
        self,
        models: ModelSelector | None = None,
        rows: Hashable = 0,
        show: Int | None = None,
        target: TargetsSelector = 1,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> plt.Figure | None:
        """Plota o gráfico waterfall do SHAP.

        O valor SHAP de uma feature representa o impacto da evidência
        fornecida por essa feature na saída do modelo. O gráfico
        waterfall foi projetado para mostrar visualmente como os valores
        SHAP, isto é, as evidências de cada feature, movem a saída do
        modelo desde nossa expectativa prévia sob a distribuição dos
        dados de fundo até a predição final do modelo, dada a evidência
        de todas as features. As features são ordenadas pela magnitude
        dos valores SHAP, com as de menor magnitude agrupadas na parte
        inferior do gráfico quando o número de features do modelo excede
        o parâmetro `show`. Leia mais sobre gráficos SHAP no [guia do
        usuário][shap].

        Parameters
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_shap_waterfall()`.

        rows: int or str, default=0
            [Seleção de linhas][row-and-column-selection] a plotar. O
            método plot_shap_waterfall não suporta plotar múltiplas
            amostras.

        show: int or None, default=None
            Número de features, ordenadas por importância, a mostrar. Se
            None, mostra todas as features.

        target: int, str or tuple, default=1
            Classe da coluna alvo a ser considerada. Para tarefas com
            múltiplas saídas, o valor deve ser uma tupla no formato
            (coluna, classe). Note que, em tarefas binárias e
            multirrótulo, a classe selecionada é sempre a positiva.

        title: str, dict or None, default=None
            Title for the plot.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de features exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como png. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [plt.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        See Also
        --------
        experionml.plots:ShapPlot.plot_shap_bar
        experionml.plots:ShapPlot.plot_shap_beeswarm
        experionml.plots:ShapPlot.plot_shap_heatmap

        Examples
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("LR")
        experionml.plot_shap_waterfall(show=10)
        ```

        """
        models_c = self._get_plot_models(models, max_one=True)[0]
        X, _ = models_c.branch._get_rows(rows, return_X_y=True)
        if len(X) > 1:
            raise ValueError(
                f"Valor inválido para o parâmetro rows, recebeu {rows}. "
                "O método plot_shap_waterfall não suporta "
                f"plotar múltiplas amostras, recebeu {len(X)}."
            )

        show_c = self._get_show(show, models_c.branch.n_features)
        target_c = self.branch._get_target(target)
        explanation = models_c._shap.get_explanation(X, target_c)

        # Waterfall accepts only one row
        explanation.values = explanation.values[0]
        explanation.data = explanation.data[0]

        self._get_figure(backend="matplotlib")
        check_canvas(BasePlot._fig.is_canvas, "plot_shap_waterfall")

        shap.plots.waterfall(explanation, max_display=show_c, show=False)

        BasePlot._fig.used_models.append(models_c)
        return self._plot(
            ax=plt.gca(),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_shap_waterfall",
            filename=filename,
            display=display,
        )
