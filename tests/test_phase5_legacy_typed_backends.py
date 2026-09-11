from __future__ import annotations

import pytest
from sklearn.datasets import make_classification

from mlcore import MODEL_REGISTRY
from mlcore.core.search_space import Integer, LogFloat, SearchSpace
from mlcore.tuning.sklearn.grid import GridSearchOptimizer
from mlcore.tuning.sklearn.random import RandomSearchOptimizer


def test_legacy_grid_optimizer_consumes_typed_integer_space() -> None:
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    result = GridSearchOptimizer(MODEL_REGISTRY).optimize(
        "random_forest",
        X,
        y,
        metric="accuracy",
        cv=2,
        n_jobs=1,
        random_state=42,
        search_space=SearchSpace("rf", {"n_estimators": Integer(5, 10, step=5)}),
    )
    assert result.best_params["n_estimators"] in {5, 10}


def test_legacy_random_optimizer_consumes_log_float_space() -> None:
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    result = RandomSearchOptimizer(MODEL_REGISTRY).optimize(
        "logistic_regression",
        X,
        y,
        metric="accuracy",
        cv=2,
        n_jobs=1,
        n_iter=3,
        random_state=42,
        search_space=SearchSpace("lr", {"C": LogFloat(1e-3, 10.0)}),
    )
    assert result.best_params["C"] > 0


def test_halving_random_optimizer_consumes_log_float_space() -> None:
    from mlcore.tuning.sklearn.halving_random import HalvingRandomSearchOptimizer

    X, y = make_classification(n_samples=60, n_features=5, random_state=42)
    result = HalvingRandomSearchOptimizer(MODEL_REGISTRY).optimize(
        "logistic_regression",
        X,
        y,
        metric="accuracy",
        cv=2,
        n_jobs=1,
        random_state=42,
        search_space=SearchSpace("lr", {"C": LogFloat(1e-3, 10.0)}),
    )
    assert result.best_params["C"] > 0
