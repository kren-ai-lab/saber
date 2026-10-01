"""Optional dependency contract tests."""

from __future__ import annotations

import subprocess
import sys


def test_core_import_can_skip_optional_estimator_providers() -> None:
    code = r"""
import importlib.util

original_find_spec = importlib.util.find_spec


def core_only_find_spec(name, package=None):
    if name in {"xgboost", "lightgbm"}:
        return None
    return original_find_spec(name, package)


importlib.util.find_spec = core_only_find_spec

import saber

assert {spec.provider for spec in saber.ALGORITHMS.values()} == {"sklearn"}
assert "xgb_classifier" not in saber.ALGORITHMS
assert "lgbm_classifier" not in saber.ALGORITHMS
"""

    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
