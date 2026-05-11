from __future__ import annotations

from experionml._show_versions import show_versions
from experionml.api import (
    ExperionMLClassifier,
    ExperionMLForecaster,
    ExperionMLModel,
    ExperionMLRegressor,
)
from experionml.utils.constants import __version__


__all__ = [
    "ExperionMLClassifier",
    "ExperionMLForecaster",
    "ExperionMLModel",
    "ExperionMLRegressor",
    "__version__",
    "show_versions",
]


def _configure_sklearn() -> None:
    # Evita efeito colateral global no momento do `import experionml`.
    # Chamada (idempotente) por `ExperionML.__init__` quando o usuário
    # realmente instancia a fachada.
    import sklearn

    sklearn.set_config(transform_output="pandas", enable_metadata_routing=True)
