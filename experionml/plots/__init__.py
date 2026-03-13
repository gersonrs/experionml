from abc import ABCMeta

from experionml.plots.dataplot import DataPlot
from experionml.plots.hyperparametertuningplot import HyperparameterTuningPlot
from experionml.plots.predictionplot import PredictionPlot
from experionml.plots.shapplot import ShapPlot


class ExperionMLPlot(
    DataPlot,
    HyperparameterTuningPlot,
    PredictionPlot,
    ShapPlot,
    metaclass=ABCMeta,
):
    """Classes de plot herdadas pelas classes principais do ExperionML."""


class RunnerPlot(HyperparameterTuningPlot, PredictionPlot, ShapPlot, metaclass=ABCMeta):
    """Classes de plot herdadas pelos runners e chamáveis a partir dos modelos."""
