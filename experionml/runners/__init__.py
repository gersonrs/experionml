"""experionml.runners — runners concretos de treinamento.

Subpacote extraído de `training.py` para melhorar navegabilidade.
O módulo `experionml.training` continua funcionando como shim de
compatibilidade retroativa.
"""

from __future__ import annotations

from experionml.runners.base import Direct, SuccessiveHalving, TrainSizing
from experionml.runners.classifier import (
    DirectClassifier,
    SuccessiveHalvingClassifier,
    TrainSizingClassifier,
)
from experionml.runners.forecaster import (
    DirectForecaster,
    SuccessiveHalvingForecaster,
    TrainSizingForecaster,
)
from experionml.runners.regressor import (
    DirectRegressor,
    SuccessiveHalvingRegressor,
    TrainSizingRegressor,
)


__all__ = [
    "Direct",
    "DirectClassifier",
    "DirectForecaster",
    "DirectRegressor",
    "SuccessiveHalving",
    "SuccessiveHalvingClassifier",
    "SuccessiveHalvingForecaster",
    "SuccessiveHalvingRegressor",
    "TrainSizing",
    "TrainSizingClassifier",
    "TrainSizingForecaster",
    "TrainSizingRegressor",
]
