"""
tests.test_base_optimizer
=========================

Tests for BaseOptimizer.
"""

from __future__ import annotations

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.tuning.base import BaseOptimizer
from mlcore.tuning.results import OptimizationResult


# ============================================================
# Dummy optimizer
# ============================================================

class DummyOptimizer(BaseOptimizer):
    """
    Minimal optimizer implementation
    used for testing.
    """

    def optimize(
        self,
        algorithm: str,
        X,
        y,
        **kwargs,
    ) -> OptimizationResult:

        return OptimizationResult(
            algorithm=algorithm,
            best_score=0.0,
            best_params={},
            optimizer="dummy",
        )


# ============================================================
# Fixtures
# ============================================================

def build_optimizer() -> DummyOptimizer:

    return DummyOptimizer(
        registry=MODEL_REGISTRY,
    )


# ============================================================
# Registry
# ============================================================

def test_algorithm_exists() -> None:

    optimizer = build_optimizer()

    assert optimizer.algorithm_exists(
        "random_forest",
    )

    assert optimizer.algorithm_exists(
        "random_forest_regressor",
    )

    assert not optimizer.algorithm_exists(
        "papucho_model",
    )


# ============================================================
# Specs
# ============================================================

def test_get_spec_classification() -> None:

    optimizer = build_optimizer()

    spec = optimizer.get_spec(
        "random_forest",
    )

    assert spec.name == "random_forest"

    assert spec.task == "classification"


def test_get_spec_regression() -> None:

    optimizer = build_optimizer()

    spec = optimizer.get_spec(
        "random_forest_regressor",
    )

    assert spec.name == "random_forest_regressor"

    assert spec.task == "regression"


# ============================================================
# Search Spaces
# ============================================================

def test_get_search_space_classification() -> None:

    optimizer = build_optimizer()

    space = optimizer.get_search_space(
        "random_forest",
    )

    assert space is not None

    assert (
        "n_estimators"
        in space.parameters
    )


def test_get_search_space_regression() -> None:

    optimizer = build_optimizer()

    space = optimizer.get_search_space(
        "random_forest_regressor",
    )

    assert space is not None

    assert (
        "n_estimators"
        in space.parameters
    )


def test_get_search_space_parameters_classification() -> None:

    optimizer = build_optimizer()

    params = optimizer.get_search_space_parameters(
        "random_forest",
    )

    assert isinstance(
        params,
        dict,
    )

    assert (
        "n_estimators"
        in params
    )


def test_get_search_space_parameters_regression() -> None:

    optimizer = build_optimizer()

    params = optimizer.get_search_space_parameters(
        "ridge_regressor",
    )

    assert isinstance(
        params,
        dict,
    )

    assert (
        "alpha"
        in params
    )


# ============================================================
# Optimization API
# ============================================================

def test_optimize_returns_result() -> None:

    optimizer = build_optimizer()

    result = optimizer.optimize(
        algorithm="random_forest",
        X=None,
        y=None,
    )

    assert isinstance(
        result,
        OptimizationResult,
    )

    assert result.algorithm == "random_forest"

    assert result.optimizer == "dummy"


# ============================================================
# Repr
# ============================================================

def test_optimizer_repr() -> None:

    optimizer = build_optimizer()

    representation = repr(
        optimizer,
    )

    assert "DummyOptimizer" in representation

    assert "registry=" in representation