"""
tests.test_registry
===================

Unit tests for AlgorithmRegistry.
"""

from __future__ import annotations

import pytest

from mlcore.core.registry import AlgorithmRegistry
from mlcore.core.specs import AlgorithmSpec

from mlcore.exceptions import (
    AlgorithmAlreadyRegisteredError,
    AlgorithmNotFoundError,
)


# ============================================================
# Fixtures
# ============================================================

def dummy_runner(*args, **kwargs) -> None:
    """Dummy runner."""
    return None


class DummyBackend:
    """Dummy backend."""
    pass


@pytest.fixture
def registry() -> AlgorithmRegistry:
    """
    Fresh registry for each test.
    """

    return AlgorithmRegistry()


@pytest.fixture
def random_forest_spec() -> AlgorithmSpec:
    """
    Example AlgorithmSpec.
    """

    return AlgorithmSpec(
        backend="sklearn",
        task="classification",
        name="random_forest",
        runner=dummy_runner,
        backend_cls=DummyBackend,
        aliases=("rf",),
        tags=("classification", "tree"),
    )


# ============================================================
# Registration
# ============================================================

def test_register_algorithm(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    assert registry.count() == 1


def test_register_duplicate_algorithm_raises(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    with pytest.raises(
        AlgorithmAlreadyRegisteredError,
    ):
        registry.register(random_forest_spec)


def test_register_many(
    registry: AlgorithmRegistry,
) -> None:

    specs = [
        AlgorithmSpec(
            backend="sklearn",
            task="classification",
            name="rf",
            runner=dummy_runner,
            backend_cls=DummyBackend,
        ),
        AlgorithmSpec(
            backend="sklearn",
            task="classification",
            name="svm",
            runner=dummy_runner,
            backend_cls=DummyBackend,
        ),
    ]

    registry.register_many(specs)

    assert registry.count() == 2


# ============================================================
# Retrieval
# ============================================================

def test_get_algorithm(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    spec = registry.get("random_forest")

    assert spec.name == "random_forest"


def test_get_algorithm_by_alias(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    spec = registry.get("rf")

    assert spec.name == "random_forest"


def test_get_missing_algorithm_raises(
    registry: AlgorithmRegistry,
) -> None:

    with pytest.raises(
        AlgorithmNotFoundError,
    ):
        registry.get("does_not_exist")


# ============================================================
# Exists
# ============================================================

def test_exists(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    assert registry.exists("random_forest")
    assert registry.exists("rf")

    assert not registry.exists(
        "foobar"
    )


# ============================================================
# Remove
# ============================================================

def test_remove_algorithm(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    registry.remove(
        "random_forest",
    )

    assert registry.count() == 0


def test_remove_missing_algorithm_raises(
    registry: AlgorithmRegistry,
) -> None:

    with pytest.raises(
        AlgorithmNotFoundError,
    ):
        registry.remove(
            "foobar",
        )


# ============================================================
# Filtering
# ============================================================

def test_filter_by_task(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    results = registry.filter(
        task="classification",
    )

    assert len(results) == 1


def test_filter_by_backend(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    results = registry.filter(
        backend="sklearn",
    )

    assert len(results) == 1


def test_filter_by_tag(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    results = registry.filter(
        tags=["tree"],
    )

    assert len(results) == 1


# ============================================================
# Metadata
# ============================================================

def test_list_algorithms(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    names = registry.list()

    assert "random_forest" in names


def test_tasks(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    assert (
        "classification"
        in registry.tasks()
    )


def test_backends(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    assert (
        "sklearn"
        in registry.backends()
    )


def test_summary(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    summary = registry.summary()

    assert summary["n_algorithms"] == 1


# ============================================================
# Maintenance
# ============================================================

def test_clear(
    registry: AlgorithmRegistry,
    random_forest_spec: AlgorithmSpec,
) -> None:

    registry.register(random_forest_spec)

    registry.clear()

    assert registry.count() == 0