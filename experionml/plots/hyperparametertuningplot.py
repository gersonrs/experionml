from __future__ import annotations

from abc import ABCMeta
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from optuna.importance import FanovaImportanceEvaluator
from optuna.trial import TrialState
from optuna.visualization._parallel_coordinate import (
    _get_dims_from_info,
    _get_parallel_coordinate_info,
)
from optuna.visualization._terminator_improvement import _get_improvement_info
from optuna.visualization._utils import _is_log_scale
from sklearn.utils._bunch import Bunch

from experionml.plots.baseplot import BasePlot
from experionml.utils.constants import PALETTE
from experionml.utils.types import (
    Bool,
    Int,
    IntLargerEqualZero,
    IntLargerZero,
    Legend,
    MetricSelector,
    Model,
    ModelSelector,
    ModelsSelector,
    ParamsSelector,
    Scalar,
    Sequence,
    int_t,
    segment_t,
)
from experionml.utils.utils import check_dependency, crash, divide, get_segment, it, lst, rnd


class HyperparameterTuningPlot(BasePlot, metaclass=ABCMeta):
    """Gráficos de ajuste de hiperparâmetros.

    Gráficos que ajudam a interpretar o estudo do modelo e seus trials
    correspondentes. Esses gráficos podem ser acessados a partir dos
    runners ou dos modelos. Se forem chamados a partir de um runner, o
    parâmetro `models` precisa ser especificado; se for None, usa todos
    os modelos. Se forem chamados a partir de um modelo, esse modelo é
    usado e o parâmetro `models` deixa de estar disponível.

    """

    @staticmethod
    def _check_hyperparams(models: list[Model]) -> list[Model]:
        """Filtra os modelos que executaram ajuste de hiperparâmetros.

        Se nenhum dos modelos fornecidos executou ajuste de
        hiperparâmetros, levanta uma exceção.

        Parâmetros
        ----------
        models: list of Model
            Modelos a verificar.

        Retorna
        -------
        list of Model
            Modelos que executaram ajuste de hiperparâmetros.

        """
        if not (models_c := [m for m in models if m._study is not None]):
            raise PermissionError(
                "Este método de plotagem só está disponível para modelos "
                "que executaram ajuste de hiperparâmetros."
            )

        return models_c

    @staticmethod
    def _get_hyperparams(params: ParamsSelector | None, model: Model) -> list[str]:
        """Valida e retorna os hiperparâmetros de um modelo.

        Parâmetros
        ----------
        params: str, segment, sequence or None
            Hiperparâmetros a obter. Use uma sequência ou adicione `+`
            entre as opções para selecionar mais de um. Se None, todos
            os hiperparâmetros do modelo são selecionados.

        model: Model
            Obtém os parâmetros deste modelo.

        Retorna
        -------
        list of str
            Hiperparâmetros selecionados.

        """
        if params is None:
            params_c = list(model._ht["distributions"])
        elif isinstance(params, segment_t):
            params_c = get_segment(list(model._ht["distributions"]), params)
        else:
            params_c = []
            for param in lst(params):
                if isinstance(param, int_t):
                    params_c.append(list(model._ht["distributions"])[param])
                elif isinstance(param, str):
                    for p in param.split("+"):
                        if p not in model._ht["distributions"]:
                            raise ValueError(
                                "Valor inválido para o parâmetro params. "
                                f"O hiperparâmetro {p} não foi usado durante a "
                                f"otimização do modelo {model.name}."
                            )
                        else:
                            params_c.append(p)

        if not params_c:
            raise ValueError(f"Nenhum hiperparâmetro foi encontrado para o modelo {model.name}.")

        return params_c

    def _optuna_target(self, metric: str) -> Callable | None:
        """Valor para o parâmetro target nas classes do Optuna.

        Parâmetros
        ----------
        metric: str
            Nome da métrica a obter.

        Retorna
        -------
        lambda or None
            Retorna None para execuções com uma única métrica e uma
            lambda que retorna a métrica fornecida para execuções com
            múltiplas métricas.

        """
        if len(self._metric) == 1:
            return None
        else:
            return lambda x: x.values[self._metric.index(metric)]

    @crash
    def plot_edf(
        self,
        models: ModelsSelector = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a função de distribuição empírica de um estudo.

        Use este gráfico para analisar e melhorar espaços de busca de
        hiperparâmetros. A EDF assume que o valor da função objetivo está
        de acordo com a distribuição uniforme no espaço objetivo. Este
        gráfico está disponível apenas para modelos que executaram
        [ajuste de hiperparâmetros][].

        !!! note
            Apenas trials completos são considerados ao plotar a EDF.

        Parâmetros
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Modelos a plotar. Se None, todos os modelos que usaram
            ajuste de hiperparâmetros são selecionados.

        metric: int, str, sequence or None, default=None
            Métrica a plotar, apenas em execuções com múltiplas métricas.
            Se str, adicione `+` entre as opções para selecionar mais de
            uma. Se None, é selecionada a métrica usada para rodar o
            pipeline.

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
        experionml.plots:HyperparameterTuningPlot.plot_hyperparameters
        experionml.plots:HyperparameterTuningPlot.plot_trials

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from optuna.distributions import IntDistribution
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        experionml = ExperionMLClassifier(X, y, random_state=1)

        # Run three models with different search spaces
        experionml.run(
            models="RF_1",
            n_trials=20,
            ht_params={"distributions": {"n_estimators": IntDistribution(6, 10)}},
        )
        experionml.run(
            models="RF_2",
            n_trials=20,
            ht_params={"distributions": {"n_estimators": IntDistribution(11, 15)}},
        )
        experionml.run(
            models="RF_3",
            n_trials=20,
            ht_params={"distributions": {"n_estimators": IntDistribution(16, 20)}},
        )

        experionml.plot_edf()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)
        metric_c = self._get_metric(metric)

        x_min = pd.concat([m.trials[metric_c] for m in models_c]).min(axis=None)
        x_max = pd.concat([m.trials[metric_c] for m in models_c]).max(axis=None)
        x = np.linspace(x_min, x_max, 100)

        self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            for met in metric_c:
                y = np.sum(m.trials[met].values[:, np.newaxis] <= x, axis=0)
                self._draw_line(
                    x=x,
                    y=y / len(m.trials),
                    parent=m.name,
                    child=met,
                    legend=legend,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            ylim=(0, 1),
            xlabel="Score",
            ylabel="Cumulative Probability",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_edf",
            filename=filename,
            display=display,
        )

    @crash
    def plot_hyperparameter_importance(
        self,
        models: ModelsSelector = None,
        metric: IntLargerEqualZero | str = 0,
        show: IntLargerZero | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a importância dos hiperparâmetros de um modelo.

        As importâncias dos hiperparâmetros são calculadas usando o
        avaliador de importância [fANOVA][]. A soma de todas as
        importâncias de todos os parâmetros, por modelo, é 1. Este
        gráfico está disponível apenas para modelos que executaram
        [ajuste de hiperparâmetros][].

        Parâmetros
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Modelos a plotar. Se None, todos os modelos que usaram
            ajuste de hiperparâmetros são selecionados.

        metric: int or str, default=0
            Métrica a plotar, apenas em execuções com múltiplas métricas.

        show: int or None, default=None
            Número de hiperparâmetros, ordenados por importância, a
            mostrar. Use None para mostrar todos.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição detalhada das opções.

            - Se None: nenhuma legenda é exibida.
            - Se str: posição em que a legenda será exibida.
            - Se dict: configuração da legenda.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de hiperparâmetros exibidos.

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
        experionml.plots:PredictionPlot.plot_feature_importance
        experionml.plots:HyperparameterTuningPlot.plot_hyperparameters
        experionml.plots:HyperparameterTuningPlot.plot_trials

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run(["ET", "RF"], n_trials=10)
        experionml.plot_hyperparameter_importance()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)
        metric_c = self._get_metric(metric, max_one=True)[0]
        params_c = len({k for m in models_c for k in m._ht["distributions"]})
        show_c = self._get_show(show, params_c)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            fanova = FanovaImportanceEvaluator(seed=self.random_state)
            importances = fanova.evaluate(m.study, target=self._optuna_target(metric_c))

            fig.add_bar(
                x=np.array(list(importances.values())) / sum(importances.values()),
                y=list(importances),
                orientation="h",
                marker={
                    "color": f"rgba({BasePlot._fig.get_elem(m.name)[4:-1]}, 0.2)",
                    "line": {"width": 2, "color": BasePlot._fig.get_elem(m.name)},
                },
                hovertemplate="%{x}<extra></extra>",
                name=m.name,
                legendgroup=m.name,
                showlegend=BasePlot._fig.showlegend(m.name, legend),
                xaxis=xaxis,
                yaxis=yaxis,
            )

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}": {"categoryorder": "total ascending"},
                "bargroupgap": 0.05,
            }
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Normalized hyperparameter importance",
            ylim=(params_c - show_c - 0.5, params_c - 0.5),
            title=title,
            legend=legend,
            figsize=figsize or (900, 400 + show_c * 50),
            plotname="plot_hyperparameter_importance",
            filename=filename,
            display=display,
        )

    @crash
    def plot_hyperparameters(
        self,
        models: ModelSelector | None = None,
        params: ParamsSelector = (0, 1),
        metric: IntLargerEqualZero | str = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota as relações entre hiperparâmetros em um estudo.

        Os hiperparâmetros de um modelo são plotados uns contra os
        outros. As pontuações da métrica correspondente são exibidas em
        um gráfico de contorno. Os marcadores representam os trials do
        estudo. Este gráfico está disponível apenas para modelos que
        executaram [ajuste de hiperparâmetros][].

        Parâmetros
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_hyperparameters()`.

        params: str, segment or sequence, default=(0, 1)
            Hiperparâmetros a plotar. Use uma sequência ou adicione `+`
            entre as opções para selecionar mais de um.

        metric: int or str, default=0
            Métrica a plotar, apenas para execuções com múltiplas métricas.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de hiperparâmetros exibidos.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:HyperparameterTuningPlot.plot_hyperparameter_importance
        experionml.plots:HyperparameterTuningPlot.plot_parallel_coordinate
        experionml.plots:HyperparameterTuningPlot.plot_trials

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("LR", n_trials=15)
        experionml.plot_hyperparameters(params=(0, 1, 2))
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False, max_one=True)
        model = self._check_hyperparams(models_c)[0]
        metric_c = self._get_metric(metric, max_one=True)[0]

        if len(params_c := self._get_hyperparams(params, model)) < 2:
            raise ValueError(
                "Valor inválido para o parâmetro hyperparameters. É necessário "
                f"um mínimo de dois parâmetros, mas foram recebidos {len(params_c)}."
            )

        fig = self._get_figure()
        for i in range((length := len(params_c) - 1) ** 2):
            x, y = i // length, i % length

            if y <= x:
                # Calcula o tamanho do subplot
                size = 1 / length

                # Determina a posição dos eixos
                x_pos = y * size
                y_pos = (length - x - 1) * size

                xaxis, yaxis = BasePlot._fig.get_axes(
                    x=(x_pos, rnd(x_pos + size)),
                    y=(y_pos, rnd(y_pos + size)),
                    coloraxis={
                        "axes": "99",
                        "colorscale": PALETTE.get(BasePlot._fig.get_elem(model.name), "Blues"),
                        "cmin": model.trials[metric_c].min(),
                        "cmax": model.trials[metric_c].max(),
                        "showscale": False,
                    },
                )

                fig.add_scatter(
                    x=model.trials[params_c[y]],
                    y=model.trials[params_c[x + 1]],
                    mode="markers",
                    marker={
                        "size": self.marker_size,
                        "color": BasePlot._fig.get_elem(model.name),
                        "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                    },
                    customdata=list(zip(model.trials.index, model.trials[metric_c], strict=True)),
                    hovertemplate=(
                        f"{params_c[y]}:%{{x}}<br>"
                        f"{params_c[x + 1]}:%{{y}}<br>"
                        f"{metric_c}:%{{customdata[1]:.4f}}"
                        "<extra>Trial %{customdata[0]}</extra>"
                    ),
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                fig.add_contour(
                    x=model.trials[params_c[y]],
                    y=model.trials[params_c[x + 1]],
                    z=model.trials[metric_c],
                    contours={
                        "showlabels": True,
                        "labelfont": {"size": self.tick_fontsize, "color": "white"},
                    },
                    coloraxis="coloraxis99",
                    hoverinfo="skip",
                    showlegend=False,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                if _is_log_scale(model.study.trials, params_c[y]):
                    fig.update_layout({f"xaxis{xaxis[1:]}_type": "log"})
                if _is_log_scale(model.study.trials, params_c[x + 1]):
                    fig.update_layout({f"yaxis{xaxis[1:]}_type": "log"})

                if x < length - 1:
                    fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
                if y > 0:
                    fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

                fig.update_layout(
                    {
                        "template": "plotly_white",
                        f"xaxis{xaxis[1:]}_showgrid": False,
                        f"yaxis{yaxis[1:]}_showgrid": False,
                        f"xaxis{yaxis[1:]}_zeroline": False,
                        f"yaxis{yaxis[1:]}_zeroline": False,
                    }
                )

                self._plot(
                    ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                    xlabel=params_c[y] if x == length - 1 else None,
                    ylabel=params_c[x + 1] if y == 0 else None,
                )

        BasePlot._fig.used_models.append(model)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (800 + 100 * length, 500 + 100 * length),
            plotname="plot_hyperparameters",
            filename=filename,
            display=display,
        )

    @crash
    def plot_parallel_coordinate(
        self,
        models: ModelSelector | None = None,
        params: ParamsSelector | None = None,
        metric: IntLargerEqualZero | str = 0,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota relações de parâmetros em alta dimensão em um estudo.

        Cada linha do gráfico representa um trial. Este gráfico está
        disponível apenas para modelos que executaram [ajuste de
        hiperparâmetros][].

        Parâmetros
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_parallel_coordinate()`.

        params: str, segment, sequence or None, default=None
            Hiperparâmetros a plotar. Use uma sequência ou adicione `+`
            entre as opções para selecionar mais de um. Se None, todos
            os hiperparâmetros do modelo são selecionados.

        metric: int or str, default=0
            Métrica a plotar, apenas para execuções com múltiplas métricas.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de hiperparâmetros exibidos.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:HyperparameterTuningPlot.plot_edf
        experionml.plots:HyperparameterTuningPlot.plot_hyperparameter_importance
        experionml.plots:HyperparameterTuningPlot.plot_hyperparameters

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("RF", n_trials=15)
        experionml.plot_parallel_coordinate(params=slice(1, 5))
        ```

        """

        def sort_mixed_types(values: list[str]) -> list[str]:
            """Ordena uma sequência de números e strings.

            Números são convertidos e têm precedência sobre strings.

            Parâmetros
            ----------
            values: list
                Valores a ordenar.

            Retorna
            -------
            list of str
                Valores ordenados.

            """
            numbers: list[Scalar] = []
            categorical: list[str] = []
            for elem in values:
                try:
                    numbers.append(it(float(elem)))
                except (TypeError, ValueError):  # noqa: PERF203
                    categorical.append(str(elem))

            return list(map(str, sorted(numbers))) + sorted(categorical)

        models_c = self._get_plot_models(models, max_one=True, ensembles=False)
        model = self._check_hyperparams(models_c)[0]
        params_c = self._get_hyperparams(params, model)
        metric_c = self._get_metric(metric, max_one=True)[0]

        dims = _get_dims_from_info(
            _get_parallel_coordinate_info(
                study=model.study,
                params=params_c,
                target=self._optuna_target(metric_c),
                target_name=metric_c,
            )
        )

        # Limpa e ordena as dimensões para uma visualização melhor
        dims = [dims[0], *sorted(dims[1:], key=lambda x: params_c.index(x["label"]))]
        for d in dims:
            if "ticktext" in d:
                # Pula o processamento para parâmetros logarítmicos
                if all(isinstance(i, int_t) for i in d["values"]):
                    # Ordena valores categóricos
                    mapping = [d["ticktext"][i] for i in d["values"]]
                    d["ticktext"] = sort_mixed_types(d["ticktext"])
                    d["values"] = [d["ticktext"].index(v) for v in mapping]
            else:
                # Arredonda valores numéricos
                d["tickvals"] = [rnd(v) for v in np.linspace(min(d["values"]), max(d["values"]), 5)]

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(
            coloraxis={
                "colorscale": PALETTE.get(BasePlot._fig.get_elem(model.name), "Blues"),
                "cmin": min(dims[0]["values"]),
                "cmax": max(dims[0]["values"]),
                "title": metric_c,
                "font_size": self.label_fontsize,
            }
        )

        fig.add_parcoords(
            dimensions=dims,
            line={
                "color": dims[0]["values"],
                "coloraxis": f"coloraxis{xaxis[1:]}",
            },
            unselected={"line": {"color": "gray", "opacity": 0.5}},
            labelside="bottom",
            labelfont={"size": self.label_fontsize},
        )

        BasePlot._fig.used_models.append(model)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            title=title,
            legend=legend,
            figsize=figsize or (700 + len(params_c) * 50, 600),
            plotname="plot_parallel_coordinate",
            filename=filename,
            display=display,
        )

    @crash
    def plot_pareto_front(
        self,
        models: ModelSelector | None = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a fronteira de Pareto de um estudo.

        Mostra as pontuações dos trials plotadas umas contra as outras.
        As cores dos marcadores indicam o número do trial. Este gráfico
        está disponível apenas para modelos com [execuções com múltiplas
        métricas][] e [ajuste de hiperparâmetros][].

        Parâmetros
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_pareto_front()`.

        metric: str, sequence or None, default=None
            Métricas a plotar. Use uma sequência ou adicione `+` entre
            as opções para selecionar mais de uma. Se None, as métricas
            usadas para executar o pipeline são selecionadas.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de métricas exibidas.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:HyperparameterTuningPlot.plot_edf
        experionml.plots:HyperparameterTuningPlot.plot_slice
        experionml.plots:HyperparameterTuningPlot.plot_trials

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run(
            models="RF",
            metric=["f1", "accuracy", "recall"],
            n_trials=15,
         )
        experionml.plot_pareto_front()
        ```

        """
        if len(self._metric) == 1:
            raise PermissionError(
                "O método plot_pareto_front só está disponível para "
                "modelos com execuções de múltiplas métricas."
            )

        models_c = self._get_plot_models(models, max_one=True, ensembles=False)
        model = self._check_hyperparams(models_c)[0]

        if len(metric_c := self._get_metric(metric)) < 2:
            raise ValueError(
                "Valor inválido para o parâmetro metric. É necessário "
                f"um mínimo de duas métricas, mas foi recebido {metric_c}."
            )

        fig = self._get_figure()
        for i in range((length := len(metric_c) - 1) ** 2):
            x, y = i // length, i % length

            if y <= x:
                # Calcula a distância entre subplots
                offset = divide(0.0125, length - 1)

                # Calcula o tamanho do subplot
                size = (1 - ((offset * 2) * (length - 1))) / length

                # Determina a posição dos eixos
                x_pos = y * (size + 2 * offset)
                y_pos = (length - x - 1) * (size + 2 * offset)

                xaxis, yaxis = BasePlot._fig.get_axes(
                    x=(x_pos, rnd(x_pos + size)),
                    y=(y_pos, rnd(y_pos + size)),
                )

                fig.add_scatter(
                    x=model.trials[metric_c[y]],
                    y=model.trials[metric_c[x + 1]],
                    mode="markers",
                    marker={
                        "size": self.marker_size,
                        "color": model.trials.index,
                        "colorscale": "Teal",
                        "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                    },
                    customdata=model.trials.index,
                    hovertemplate="(%{x}, %{y})<extra>Trial %{customdata}</extra>",
                    xaxis=xaxis,
                    yaxis=yaxis,
                )

                if x < length - 1:
                    fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
                if y > 0:
                    fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

                self._plot(
                    ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                    xlabel=metric_c[y] if x == length - 1 else None,
                    ylabel=metric_c[x + 1] if y == 0 else None,
                )

        BasePlot._fig.used_models.append(model)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (500 + 100 * length, 500 + 100 * length),
            plotname="plot_pareto_front",
            filename=filename,
            display=display,
        )

    @crash
    def plot_slice(
        self,
        models: ModelSelector | None = None,
        params: ParamsSelector | None = None,
        metric: MetricSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = None,
        figsize: tuple[IntLargerZero, IntLargerZero] | None = None,
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a relação entre parâmetros em um estudo.

        A cor dos marcadores indica o trial. Este gráfico está
        disponível apenas para modelos que executaram [ajuste de
        hiperparâmetros][].

        Parâmetros
        ----------
        models: int, str, Model or None, default=None
            Modelo a plotar. Se None, todos os modelos são
            selecionados. Note que manter a opção padrão pode levantar
            uma exceção se houver múltiplos modelos. Para evitar isso,
            chame o gráfico diretamente a partir de um modelo, por
            exemplo, `experionml.lr.plot_slice()`.

        params: str, segment, sequence or None, default=None
            Hiperparâmetros a plotar. Use uma sequência ou adicione `+`
            entre as opções para selecionar mais de um. Se None, todos
            os hiperparâmetros do modelo são selecionados.

        metric: int or str, default=None
            Métrica a plotar, apenas para execuções com múltiplas
            métricas. Se for str, adicione `+` entre as opções para
            selecionar mais de uma. Se None, a métrica usada para
            executar o pipeline é selecionada.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default=None
            Não faz nada. Implementado para manter continuidade da API.

        figsize: tuple or None, default=None
            Tamanho da figura em pixels, no formato (x, y). Se None,
            adapta o tamanho ao número de hiperparâmetros exibidos.

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:HyperparameterTuningPlot.plot_edf
        experionml.plots:HyperparameterTuningPlot.plot_hyperparameters
        experionml.plots:HyperparameterTuningPlot.plot_parallel_coordinate

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import load_breast_cancer

        X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run(
            models="RF",
            metric=["f1", "recall"],
            n_trials=15,
        )
        experionml.plot_slice(params=(0, 1, 2))
        ```

        """
        models_c = self._get_plot_models(models, max_one=True, ensembles=False)
        model = self._check_hyperparams(models_c)[0]
        params_c = self._get_hyperparams(params, model)
        metric_c = self._get_metric(metric)

        fig = self._get_figure()
        for i in range(len(params_c) * len(metric_c)):
            x, y = i // len(params_c), i % len(params_c)

            # Calcula a distância entre subplots
            x_offset = divide(0.0125, (len(params_c) - 1))
            y_offset = divide(0.0125, (len(metric_c) - 1))

            # Calcula o tamanho do subplot
            x_size = (1 - ((x_offset * 2) * (len(params_c) - 1))) / len(params_c)
            y_size = (1 - ((y_offset * 2) * (len(metric_c) - 1))) / len(metric_c)

            # Determina a posição dos eixos
            x_pos = y * (x_size + 2 * x_offset)
            y_pos = (len(metric_c) - x - 1) * (y_size + 2 * y_offset)

            xaxis, yaxis = BasePlot._fig.get_axes(
                x=(x_pos, rnd(x_pos + x_size)),
                y=(y_pos, rnd(y_pos + y_size)),
            )

            fig.add_scatter(
                x=model.trials[params_c[y]],
                y=model.trials[metric_c[x]],
                mode="markers",
                marker={
                    "size": self.marker_size,
                    "color": model.trials.index,
                    "colorscale": "Teal",
                    "line": {"width": 1, "color": "rgba(255, 255, 255, 0.9)"},
                },
                customdata=model.trials.index,
                hovertemplate="(%{x}, %{y})<extra>Trial %{customdata}</extra>",
                xaxis=xaxis,
                yaxis=yaxis,
            )

            if _is_log_scale(model.study.trials, params_c[y]):
                fig.update_layout({f"xaxis{xaxis[1:]}_type": "log"})

            if x < len(metric_c) - 1:
                fig.update_layout({f"xaxis{xaxis[1:]}_showticklabels": False})
            if y > 0:
                fig.update_layout({f"yaxis{yaxis[1:]}_showticklabels": False})

            self._plot(
                ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
                xlabel=params_c[y] if x == len(metric_c) - 1 else None,
                ylabel=metric_c[x] if y == 0 else None,
            )

        BasePlot._fig.used_models.append(model)
        return self._plot(
            title=title,
            legend=legend,
            figsize=figsize or (800 + 100 * len(params_c), 500 + 100 * len(metric_c)),
            plotname="plot_slice",
            filename=filename,
            display=display,
        )

    @crash
    def plot_terminator_improvement(
        self,
        models: ModelsSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota os potenciais de melhoria futura do objetivo.

                Esta função visualiza os potenciais de melhoria do objetivo.
                Ela ajuda a determinar se você deve continuar a otimização ou
                não. O erro avaliado também é plotado. Observe que essa função
                pode levar algum tempo para calcular os potenciais de melhoria.
                Este gráfico está disponível apenas para modelos que executaram
                [ajuste de hiperparâmetros][].

        !!! warning
                        * O método plot_terminator_improvement só está disponível
                            para modelos que executaram ajuste de hiperparâmetros com
                            validação cruzada, por exemplo, usando `ht_params={'cv': 5}`.
                        * Este método não oferece suporte a
                            [otimizações multiobjetivo][multi-metric runs].
                        * O cálculo da melhoria pode ser lento. Defina o parâmetro
                            [`memory`][experionmlclassifier-memory] para armazenar os
                            resultados em cache e acelerar chamadas repetidas.

        Parâmetros
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Modelos a plotar. Se None, todos os modelos que usaram
            ajuste de hiperparâmetros são selecionados.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição mais detalhada das opções.

            - Se None, nenhuma legenda é exibida.
            - Se str, posição em que a legenda é exibida.
            - Se dict, configuração da legenda.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y)

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:HyperparameterTuningPlot.plot_pareto_front
        experionml.plots:HyperparameterTuningPlot.plot_timeline
        experionml.plots:HyperparameterTuningPlot.plot_trials

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, flip_y=0.2, random_state=1)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run("RF", n_trials=10, ht_params={"cv": 5})
        experionml.plot_terminator_improvement()
        ```

        """
        check_dependency("botorch")

        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)

        self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        for m in models_c:
            if m._ht["cv"] > 1 and len(self._metric) == 1:
                info = self.memory.cache(_get_improvement_info)(m.study, get_error=True)
            else:
                raise PermissionError(
                    "O método plot_terminator_improvement só está disponível para "
                    "modelos que executaram ajuste de hiperparâmetros usando validação "
                    "cruzada (por exemplo, com ht_params={'cv': 5}) em uma otimização "
                    "com uma única métrica."
                )

            self._draw_line(
                x=m.trials.index,
                y=info.improvements,
                error_y={"type": "data", "array": info.errors},
                mode="markers+lines",
                parent=m.name,
                legend=legend,
                xaxis=xaxis,
                yaxis=yaxis,
            )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Trial",
            ylabel="Melhoria do terminator",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_terminator_improvement",
            filename=filename,
            display=display,
        )

    @crash
    def plot_timeline(
        self,
        models: ModelsSelector = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "lower right",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 600),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota a linha do tempo de um estudo.

        Este gráfico está disponível apenas para modelos que executaram
        [ajuste de hiperparâmetros][].

        Parâmetros
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Modelos a plotar. Se None, todos os modelos que usaram
            ajuste de hiperparâmetros são selecionados.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="lower right"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição mais detalhada das opções.

            - Se None, nenhuma legenda é exibida.
            - Se str, posição em que a legenda é exibida.
            - Se dict, configuração da legenda.

        figsize: tuple, default=(900, 600)
            Tamanho da figura em pixels, no formato (x, y)

        filename: str, Path or None, default=None
            Salva o gráfico com este nome. Use "auto" para nomeação
            automática. O tipo do arquivo depende do nome fornecido
            (.html, .png, .pdf etc.). Se `filename` não tiver extensão,
            o gráfico é salvo como html. Se None, o gráfico não é salvo.

        display: bool or None, default=True
            Se deve renderizar o gráfico. Se None, retorna a figura.

        Retorna
        -------
        [go.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:HyperparameterTuningPlot.plot_edf
        experionml.plots:HyperparameterTuningPlot.plot_slice
        experionml.plots:HyperparameterTuningPlot.plot_terminator_improvement

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from optuna.pruners import PatientPruner
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=1000, flip_y=0.2, random_state=1)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run(
            models="LGB",
            n_trials=15,
            ht_params={"pruner": PatientPruner(None, patience=2)},
        )
        experionml.plot_timeline()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes()

        _cm = {
            "COMPLETE": "rgb(173, 116, 230)",  # Azul
            "FAIL": "rgb(255, 0, 0)",  # Vermelho
            "PRUNED": "rgb(255, 165, 0)",  # Laranja
            "RUNNING": "rgb(124, 252, 0)",  # Verde
            "WAITING": "rgb(220, 220, 220)",  # Cinza
        }

        for m in models_c:
            info = []
            for trial in m.study.get_trials(deepcopy=False):
                date_complete = trial.datetime_complete or datetime.now()
                date_start = trial.datetime_start or date_complete

                # Cria uma representação legível de pontuações e parâmetros para o hover
                s = [f"{m}: {trial.values[i]}" for i, m in enumerate(self._metric.keys())]
                p = [f" --> {k}: {v}" for k, v in trial.params.items()]

                info.append(
                    Bunch(
                        number=trial.number,
                        start=date_start,
                        duration=1000 * (date_complete - date_start).total_seconds(),
                        state=trial.state,
                        hovertext=(
                            f"Trial: {trial.number}<br>"
                            f"{'<br>'.join(s)}"
                            f"Parâmetros:<br>{'<br>'.join(p)}"
                        ),
                    )
                )

            for state in sorted(TrialState, key=lambda x: x.name):
                if bars := list(filter(lambda x: x.state == state, info)):
                    fig.add_bar(
                        name=state.name,
                        x=[b.duration for b in bars],
                        y=[b.number for b in bars],
                        base=[b.start.isoformat() for b in bars],
                        text=[b.hovertext for b in bars],
                        textposition="none",
                        hovertemplate=f"%{{text}}<extra>{m.name}</extra>",
                        orientation="h",
                        marker={
                            "color": f"rgba({_cm[state.name][4:-1]}, 0.2)",
                            "line": {"width": 2, "color": _cm[state.name]},
                        },
                        showlegend=BasePlot._fig.showlegend(_cm[state.name], legend),
                        xaxis=xaxis,
                        yaxis=yaxis,
                    )

        fig.update_layout({f"xaxis{yaxis[1:]}_type": "date", "barmode": "group"})

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            xlabel="Datetime",
            ylabel="Trial",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_timeline",
            filename=filename,
            display=display,
        )

    @crash
    def plot_trials(
        self,
        models: ModelsSelector = None,
        metric: Int | str | Sequence[Int | str] | None = None,
        *,
        title: str | dict[str, Any] | None = None,
        legend: Legend | dict[str, Any] | None = "upper left",
        figsize: tuple[IntLargerZero, IntLargerZero] = (900, 800),
        filename: str | Path | None = None,
        display: Bool | None = True,
    ) -> go.Figure | None:
        """Plota os trials de ajuste de hiperparâmetros.

        Cria uma figura com dois gráficos: o primeiro mostra a
        pontuação de cada trial e o segundo mostra a distância entre os
        últimos passos consecutivos. O melhor trial é indicado com uma
        estrela. Este é o mesmo gráfico produzido por
        `ht_params={"plot": True}`. Este gráfico está disponível apenas
        para modelos que executaram [ajuste de hiperparâmetros][].

        Parâmetros
        ----------
        models: int, str, Model, segment, sequence or None, default=None
            Modelos a plotar. Se None, todos os modelos que usaram
            ajuste de hiperparâmetros são selecionados.

        metric: int, str, sequence or None, default=None
            Métrica a plotar, apenas para execuções com múltiplas
            métricas. Adicione `+` entre as opções para selecionar mais
            de uma. Se None, todas as métricas são selecionadas.

        title: str, dict or None, default=None
            Título do gráfico.

            - Se None, nenhum título é exibido.
            - Se str, texto do título.
            - Se dict, [configuração do título][parameters].

        legend: str, dict or None, default="upper left"
            Legenda do gráfico. Consulte o [guia do usuário][parameters]
            para uma descrição mais detalhada das opções.

            - Se None, nenhuma legenda é exibida.
            - Se str, posição em que a legenda é exibida.
            - Se dict, configuração da legenda.

        figsize: tuple, default=(900, 800)
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
        [go.Figure][] ou None
            Objeto do gráfico. Só é retornado se `display=None`.

        Veja também
        --------
        experionml.plots:PredictionPlot.plot_evals
        experionml.plots:HyperparameterTuningPlot.plot_hyperparameters
        experionml.plots:PredictionPlot.plot_results

        Exemplos
        --------
        ```pycon
        from experionml import ExperionMLClassifier
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=100, flip_y=0.2, random_state=1)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        experionml.run(["ET", "RF"], n_trials=15)
        experionml.plot_trials()
        ```

        """
        models_c = self._get_plot_models(models, ensembles=False)
        models_c = self._check_hyperparams(models_c)
        metric_c = self._get_metric(metric)

        fig = self._get_figure()
        xaxis, yaxis = BasePlot._fig.get_axes(y=(0.31, 1.0))
        xaxis2, yaxis2 = BasePlot._fig.get_axes(y=(0.0, 0.29))

        for m in models_c:
            for met in metric_c:
                # Posiciona um símbolo de estrela no melhor trial
                symbols = ["circle"] * len(m.trials)
                symbols[m.best_trial.number] = "star"
                sizes = [self.marker_size] * len(m.trials)
                sizes[m.best_trial.number] = self.marker_size * 1.5

                self._draw_line(
                    x=m.trials.index,
                    y=m.trials[met],
                    mode="lines+markers",
                    marker_symbol=symbols,
                    marker_size=sizes,
                    hovertemplate=None,
                    parent=m.name,
                    child=self._metric[met].name,
                    legend=legend,
                    xaxis=xaxis2,
                    yaxis=yaxis,
                )

                self._draw_line(
                    x=m.trials.index,
                    y=m.trials[met].diff(),
                    mode="lines+markers",
                    marker_symbol="circle",
                    parent=m.name,
                    child=self._metric[met].name,
                    legend=legend,
                    xaxis=xaxis2,
                    yaxis=yaxis2,
                )

        fig.update_layout(
            {
                f"yaxis{yaxis[1:]}_anchor": f"x{xaxis2[1:]}",
                f"xaxis{xaxis[1:]}_showticklabels": False,
                "hovermode": "x unified",
            },
        )

        self._plot(
            ax=(f"xaxis{xaxis2[1:]}", f"yaxis{yaxis2[1:]}"),
            xlabel="Trial",
            ylabel="d",
        )

        BasePlot._fig.used_models.extend(models_c)
        return self._plot(
            ax=(f"xaxis{xaxis[1:]}", f"yaxis{yaxis[1:]}"),
            groupclick="togglegroup",
            ylabel="Score",
            title=title,
            legend=legend,
            figsize=figsize,
            plotname="plot_trials",
            filename=filename,
            display=display,
        )
