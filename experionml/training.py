"""Shim de compatibilidade retroativa.

O conteúdo deste módulo foi movido para o subpacote `experionml.runners`.
Importar daqui continua funcionando — apenas re-exportamos do novo local.

Migrar para:
    from experionml.runners import DirectClassifier, ...
"""

from __future__ import annotations

from experionml.runners import (
    Direct,
    DirectClassifier,
    DirectForecaster,
    DirectRegressor,
    SuccessiveHalving,
    SuccessiveHalvingClassifier,
    SuccessiveHalvingForecaster,
    SuccessiveHalvingRegressor,
    TrainSizing,
    TrainSizingClassifier,
    TrainSizingForecaster,
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
