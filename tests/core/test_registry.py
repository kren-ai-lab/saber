"""Static algorithm catalog."""

from __future__ import annotations

import pytest

from saber._optional import is_dependency_available
from saber.core.registry import ALGORITHMS, get_algorithm
from saber.exceptions import AlgorithmNotFoundError


def test_get_algorithm_unknown_name_lists_available() -> None:
    with pytest.raises(AlgorithmNotFoundError, match="not_a_model") as excinfo:
        get_algorithm("not_a_model")
    assert "random_forest_classifier" in str(excinfo.value)


def test_catalog_keys_match_spec_names_and_are_read_only() -> None:
    assert all(name == spec.name for name, spec in ALGORITHMS.items())
    with pytest.raises(TypeError):
        ALGORITHMS["x"] = get_algorithm("random_forest_classifier")  # type: ignore[index]


def test_sklearn_specs_present() -> None:
    for name, task in (
        ("random_forest_classifier", "classification"),
        ("random_forest_regressor", "regression"),
    ):
        spec = get_algorithm(name)
        assert spec.provider == "sklearn"
        assert spec.task == task


@pytest.mark.parametrize("provider", ["xgboost", "lightgbm"])
def test_optional_providers_present_iff_installed(provider: str) -> None:
    present = any(spec.provider == provider for spec in ALGORITHMS.values())
    assert present == is_dependency_available(provider)
