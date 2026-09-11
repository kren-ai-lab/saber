"""Phase 0 scope and optional dependency contract tests."""

from __future__ import annotations

import subprocess
import sys

from mlcore import MODEL_REGISTRY


def test_only_frozen_task_families_are_registered() -> None:
    assert MODEL_REGISTRY.tasks() == {"classification", "regression"}


def test_neural_network_estimators_are_out_of_scope() -> None:
    assert "mlp_regressor" not in MODEL_REGISTRY
    assert all("neural_network" not in spec.tags for spec in MODEL_REGISTRY)


def test_registry_tags_are_normalized() -> None:
    for tag in MODEL_REGISTRY.tags():
        assert tag == tag.lower()
        assert " " not in tag
        assert "-" not in tag


def test_core_import_can_skip_optional_estimator_providers() -> None:
    code = r'''
import importlib.util

original_find_spec = importlib.util.find_spec


def core_only_find_spec(name, package=None):
    if name in {"xgboost", "lightgbm"}:
        return None
    return original_find_spec(name, package)


importlib.util.find_spec = core_only_find_spec

import mlcore

assert mlcore.MODEL_REGISTRY.backends() == {"sklearn"}
assert "xgb_classifier" not in mlcore.MODEL_REGISTRY
assert "lgbm_classifier" not in mlcore.MODEL_REGISTRY
'''

    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
