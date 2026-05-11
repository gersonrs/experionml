"""experionml.cleaning — transformadores de pré-processamento de dados.

Subpacote extraído de `data_cleaning.py` para melhorar navegabilidade.
O módulo `experionml.data_cleaning` continua funcionando como shim de
compatibilidade retroativa — qualquer `from experionml.data_cleaning import X`
continua válido.
"""

from __future__ import annotations

from experionml.cleaning.balancer import Balancer
from experionml.cleaning.base import TransformerMixin
from experionml.cleaning.cleaner import Cleaner
from experionml.cleaning.decomposer import Decomposer
from experionml.cleaning.discretizer import Discretizer
from experionml.cleaning.encoder import Encoder
from experionml.cleaning.imputer import Imputer
from experionml.cleaning.normalizer import Normalizer
from experionml.cleaning.pruner import Pruner
from experionml.cleaning.scaler import Scaler


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
