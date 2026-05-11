"""Testes de contrato da API pública.

Esta suíte protege invariantes críticos do ponto de entrada do
`experionml`: superfície pública exportada, efeitos colaterais no
`import`, reprodutibilidade de split, e o comportamento documentado de
`ExperionMLModel`.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys

import dill
import numpy as np
import pytest
from sklearn.linear_model import HuberRegressor

import experionml
from experionml import (
    ExperionMLClassifier,
    ExperionMLForecaster,
    ExperionMLModel,
    ExperionMLRegressor,
)

from .conftest import X_bin, X_reg, y_bin, y_fc, y_reg


# ---------------------------------------------------------------------------
# Superfície pública
# ---------------------------------------------------------------------------


def test_public_api_symbols_exported():
    """Nomes públicos documentados devem estar em `__all__`."""
    expected = {
        "ExperionMLClassifier",
        "ExperionMLForecaster",
        "ExperionMLModel",
        "ExperionMLRegressor",
        "__version__",
        "show_versions",
    }
    assert expected.issubset(set(experionml.__all__))


def test_version_is_valid_pep440():
    """`__version__` deve ser uma string não-vazia no formato MAJOR.MINOR.PATCH."""
    assert isinstance(experionml.__version__, str)
    assert experionml.__version__
    # PEP 440 simplificado: X.Y.Z[.suffix]
    assert re.match(r"^\d+\.\d+\.\d+", experionml.__version__)


# ---------------------------------------------------------------------------
# Regressão: import não deve mutar config global do sklearn
# ---------------------------------------------------------------------------


def test_import_does_not_mutate_sklearn_config():
    """`import experionml` NÃO pode mutar `sklearn.set_config` globalmente.

    Regressão de um bug antigo onde o `__init__.py` chamava
    `sklearn.set_config(transform_output="pandas", enable_metadata_routing=True)`
    no momento do import, afetando todo o processo Python. Rodamos o
    teste em subprocesso para ter garantia de ambiente limpo.
    """
    code = (
        "import json, sklearn; "
        "before = sklearn.get_config(); "
        "import experionml; "
        "after = sklearn.get_config(); "
        "print(json.dumps({k: [before.get(k), after.get(k)] "
        "for k in ('transform_output', 'enable_metadata_routing')}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    import json

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    # transform_output e enable_metadata_routing permanecem no default.
    assert payload["transform_output"][0] == payload["transform_output"][1]
    assert payload["enable_metadata_routing"][0] == payload["enable_metadata_routing"][1]


def test_sklearn_config_applied_on_instantiation():
    """Após instanciar a fachada, a config relevante deve estar ativa."""
    import sklearn

    ExperionMLClassifier(X_bin, y_bin, random_state=1)
    cfg = sklearn.get_config()
    assert cfg["transform_output"] == "pandas"
    assert cfg["enable_metadata_routing"] is True


# ---------------------------------------------------------------------------
# ExperionMLModel
# ---------------------------------------------------------------------------


def test_experionmlmodel_clones_and_tags():
    """ExperionMLModel deve retornar um *clone* do estimador com tags."""
    original = HuberRegressor()
    model = ExperionMLModel(
        estimator=original,
        name="huber1",
        acronym="huber",
        needs_scaling=True,
    )
    ExperionMLRegressor(X_reg, y_reg, random_state=1).run(model)

    assert model is not original
    assert model.name == "huber1"
    assert model.acronym == "huber"
    assert model.needs_scaling is True
    assert model.native_multioutput is False
    assert model.validation is None


def test_experionmlmodel_defaults_acronym_from_name():
    """Quando `acronym` não é informado, deve derivar de `name`."""
    model = ExperionMLModel(estimator=HuberRegressor(), name="MyHuber")
    assert model.acronym


# ---------------------------------------------------------------------------
# Goals e tipos de tarefa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "args", "expected_goal"),
    [
        (ExperionMLClassifier, (X_bin, y_bin), "classification"),
        (ExperionMLRegressor, (X_reg, y_reg), "regression"),
        (ExperionMLForecaster, (y_fc,), "forecast"),
    ],
)
def test_goal_for_each_variant(factory, args, expected_goal):
    """Cada fachada deve declarar o `_goal` correto."""
    instance = factory(*args, random_state=1)
    assert instance._goal.name == expected_goal


# ---------------------------------------------------------------------------
# Reprodutibilidade determinística do split
# ---------------------------------------------------------------------------


def test_same_random_state_gives_same_split():
    """Mesmo `random_state` deve produzir exatamente o mesmo split train/test."""
    a = ExperionMLClassifier(X_bin, y_bin, random_state=42)
    b = ExperionMLClassifier(X_bin, y_bin, random_state=42)

    np.testing.assert_array_equal(a.train.index, b.train.index)
    np.testing.assert_array_equal(a.test.index, b.test.index)


def test_different_random_state_gives_different_split():
    """`random_state` diferente deve produzir splits distintos."""
    a = ExperionMLClassifier(X_bin, y_bin, random_state=1)
    b = ExperionMLClassifier(X_bin, y_bin, random_state=2)
    assert not np.array_equal(a.train.index, b.train.index)


# ---------------------------------------------------------------------------
# Serialização (dill / pickle) — reconstrução do estado
# ---------------------------------------------------------------------------


def test_dill_roundtrip_preserves_basic_state():
    """`ExperionML*` deve ser serializável via dill preservando dataset."""
    experionml_obj = ExperionMLClassifier(X_bin, y_bin, random_state=7)
    blob = dill.dumps(experionml_obj)
    restored = dill.loads(blob)

    assert restored._goal.name == experionml_obj._goal.name
    np.testing.assert_array_equal(restored.train.index, experionml_obj.train.index)
    np.testing.assert_array_equal(restored.test.index, experionml_obj.test.index)


# ---------------------------------------------------------------------------
# Reimport é idempotente
# ---------------------------------------------------------------------------


def test_reimport_is_idempotent():
    """`importlib.reload(experionml)` não deve quebrar nada nem duplicar estado."""
    mod = importlib.reload(experionml)
    assert mod is experionml
    assert "ExperionMLClassifier" in dir(mod)
