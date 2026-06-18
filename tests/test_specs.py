"""
tests.test_specs
================

Unit tests for AlgorithmSpec.
"""

from __future__ import annotations

from mlcore.core.specs import AlgorithmSpec


# ============================================================
# Fixtures
# ============================================================

def dummy_runner(*args, **kwargs) -> None:
    """Dummy runner."""
    return None


class DummyBackend:
    """Dummy backend."""
    pass


# ============================================================
# Tests
# ============================================================

def test_algorithm_spec_creation() -> None:
    """
    AlgorithmSpec should be instantiated correctly.
    """

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
    )

    assert spec.backend == "sklearn"
    assert spec.task == "classification"
    assert spec.name == "random_forest"

    assert spec.runner is dummy_runner
    assert spec.backend_cls is DummyBackend


def test_algorithm_spec_default_aliases() -> None:
    """
    Aliases should default to an empty tuple.
    """

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
    )

    assert isinstance(
        spec.aliases,
        tuple,
    )

    assert len(spec.aliases) == 0


def test_algorithm_spec_custom_aliases() -> None:
    """
    Custom aliases should be stored.
    """

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
        aliases=(
            "rf",
            "randomforest",
        ),
    )

    assert spec.aliases == (
        "rf",
        "randomforest",
    )


def test_algorithm_spec_default_tags() -> None:
    """
    Tags should default to an empty tuple.
    """

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
    )

    assert isinstance(
        spec.tags,
        tuple,
    )


def test_algorithm_spec_custom_tags() -> None:
    """
    Tags should be stored.
    """

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
        tags=(
            "classification",
            "tree",
        ),
    )

    assert spec.tags == (
        "classification",
        "tree",
    )


def test_algorithm_spec_default_flags() -> None:
    """
    Capability flags should have expected defaults.
    """

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
    )

    assert spec.supports_cv is False
    assert spec.supports_proba is False


def test_algorithm_spec_custom_flags() -> None:
    """
    Capability flags should be configurable.
    """

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
        supports_cv=True,
        supports_proba=True,
    )

    assert spec.supports_cv is True
    assert spec.supports_proba is True


def test_algorithm_spec_default_params() -> None:
    """
    Default params should be stored.
    """

    params = {
        "n_estimators": 100,
        "max_depth": 10,
    }

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
        default_params=params,
    )

    assert spec.default_params == params


def test_algorithm_spec_search_space() -> None:
    """
    Search space should be stored.
    """

    search_space = {
        "n_estimators": [
            100,
            200,
            500,
        ],
    }

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
        search_space=search_space,
    )

    assert spec.search_space == search_space


def test_algorithm_spec_repr() -> None:
    """
    __repr__ should contain useful information.
    """

    spec = AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
    )

    representation = repr(spec)

    assert "random_forest" in representation
    assert "classification" in representation
    assert "sklearn" in representation