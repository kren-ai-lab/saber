"""Tests for the canonical algorithm registry."""

from __future__ import annotations

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from saber.core.registry import AlgorithmRegistry
from saber.core.specs import AlgorithmSpec
from saber.exceptions import AlgorithmAlreadyRegisteredError, AlgorithmNotFoundError


def _spec(name: str = "random_forest", aliases: tuple[str, ...] = ("rf",)) -> AlgorithmSpec:
    estimator = RandomForestClassifier if name == "random_forest" else LogisticRegression
    return AlgorithmSpec(
        provider="sklearn",
        task="classification",
        name=name,
        estimator_cls=estimator,
        aliases=aliases,
        tags=("classification", "tree" if name == "random_forest" else "linear"),
        supports_cv=True,
    )


def test_register_get_alias_and_exists() -> None:
    registry = AlgorithmRegistry()
    registry.register(_spec())

    assert registry.count() == 1
    assert registry.get("random_forest").name == "random_forest"
    assert registry.get("rf").name == "random_forest"
    assert registry.exists("rf")
    assert not registry.exists("missing")


def test_duplicate_names_and_aliases_are_rejected() -> None:
    registry = AlgorithmRegistry()
    registry.register(_spec())

    with pytest.raises(AlgorithmAlreadyRegisteredError):
        registry.register(_spec())
    with pytest.raises(AlgorithmAlreadyRegisteredError):
        registry.register(_spec("logistic_regression", aliases=("rf",)))


def test_filtering_uses_provider_task_and_tags() -> None:
    registry = AlgorithmRegistry()
    registry.register_many([_spec(), _spec("logistic_regression", aliases=("logreg",))])

    assert {spec.name for spec in registry.filter(task="classification")} == {
        "random_forest",
        "logistic_regression",
    }
    assert {spec.name for spec in registry.filter(provider="sklearn")} == {
        "random_forest",
        "logistic_regression",
    }
    assert [spec.name for spec in registry.filter(tags=("tree",))] == ["random_forest"]
    assert registry.get_by_provider("sklearn") == registry.filter(provider="sklearn")
    assert registry.providers() == {"sklearn"}


def test_registry_metadata_factory_and_summary() -> None:
    registry = AlgorithmRegistry()
    registry.register(_spec())

    metadata = registry.describe("rf")
    estimator = registry.build_estimator("rf", random_state=23, n_estimators=5)
    summary = registry.summary()

    assert metadata["provider"] == "sklearn"
    assert metadata["has_estimator_factory"] is True
    assert estimator.random_state == 23
    assert estimator.n_estimators == 5
    assert summary["providers"] == {"sklearn": 1}
    assert summary["available_providers"] == ["sklearn"]
    assert "backends" not in summary


def test_remove_clear_and_missing_errors() -> None:
    registry = AlgorithmRegistry()
    registry.register(_spec())

    registry.remove("rf")
    assert registry.count() == 0
    with pytest.raises(AlgorithmNotFoundError):
        registry.get("rf")
    with pytest.raises(AlgorithmNotFoundError):
        registry.remove("rf")

    registry.register(_spec())
    registry.clear()
    assert len(registry) == 0
