"""Shim de compatibilidade retroativa.

O conteúdo deste módulo foi movido para o subpacote `experionml.cleaning`.
Importar daqui continua funcionando — apenas re-exportamos do novo local.

Migrar para:
    from experionml.cleaning import Balancer, Cleaner, ...
"""

from __future__ import annotations

from experionml.cleaning import (
    Balancer,
    Cleaner,
    Decomposer,
    Discretizer,
    Encoder,
    Imputer,
    Normalizer,
    Pruner,
    Scaler,
    TransformerMixin,
)


__all__ = [
    "Balancer",
    "Cleaner",
    "Decomposer",
    "Discretizer",
    "Encoder",
    "Imputer",
    "Normalizer",
    "Pruner",
    "Scaler",
    "TransformerMixin",
]
