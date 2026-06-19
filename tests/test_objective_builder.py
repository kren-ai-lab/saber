"""
tests.test_objective_builder
============================

Tests for ObjectiveBuilder.
"""

from __future__ import annotations

from optuna.trial import FixedTrial

from sklearn.datasets import (
    make_classification,
    make_regression,
)

import mlcore.classification
import mlcore.regression

from mlcore import MODEL_REGISTRY

from mlcore.core.search_space import SearchSpace

from mlcore.tuning.objective import (
    ObjectiveBuilder,
)


# ============================================================
# Classification
# ============================================================

def test_build_classification_objective() -> None:

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42,
    )

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    search_space = SearchSpace(
        name="rf_test",
        parameters={
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        },
    )

    builder = ObjectiveBuilder(
        spec=spec,
        search_space=search_space,
        metric="accuracy",
        cv=3,
        X=X,
        y=y,
    )

    objective = builder.build()

    assert callable(
        objective,
    )


def test_classification_objective_returns_float() -> None:

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42,
    )

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    search_space = SearchSpace(
        name="rf_test",
        parameters={
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        },
    )

    builder = ObjectiveBuilder(
        spec=spec,
        search_space=search_space,
        metric="accuracy",
        cv=3,
        X=X,
        y=y,
    )

    objective = builder.build()

    trial = FixedTrial(
        {
            "n_estimators": 10,
            "max_depth": 3,
        }
    )

    score = objective(
        trial,
    )

    assert isinstance(
        score,
        float,
    )


# ============================================================
# Regression
# ============================================================

def test_build_regression_objective() -> None:

    X, y = make_regression(
        n_samples=100,
        n_features=10,
        random_state=42,
    )

    spec = MODEL_REGISTRY.get(
        "random_forest_regressor",
    )

    search_space = SearchSpace(
        name="rf_reg_test",
        parameters={
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        },
    )

    builder = ObjectiveBuilder(
        spec=spec,
        search_space=search_space,
        metric="r2",
        cv=3,
        X=X,
        y=y,
    )

    objective = builder.build()

    assert callable(
        objective,
    )


def test_regression_objective_returns_float() -> None:

    X, y = make_regression(
        n_samples=100,
        n_features=10,
        random_state=42,
    )

    spec = MODEL_REGISTRY.get(
        "random_forest_regressor",
    )

    search_space = SearchSpace(
        name="rf_reg_test",
        parameters={
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        },
    )

    builder = ObjectiveBuilder(
        spec=spec,
        search_space=search_space,
        metric="r2",
        cv=3,
        X=X,
        y=y,
    )

    objective = builder.build()

    trial = FixedTrial(
        {
            "n_estimators": 10,
            "max_depth": 3,
        }
    )

    score = objective(
        trial,
    )

    assert isinstance(
        score,
        float,
    )


# ============================================================
# Parameter sampling
# ============================================================

def test_sample_params() -> None:

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42,
    )

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    search_space = SearchSpace(
        name="rf_test",
        parameters={
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        },
    )

    builder = ObjectiveBuilder(
        spec=spec,
        search_space=search_space,
        metric="accuracy",
        cv=3,
        X=X,
        y=y,
    )

    trial = FixedTrial(
        {
            "n_estimators": 20,
            "max_depth": 5,
        }
    )

    params = builder.sample_params(
        trial,
    )

    assert params["n_estimators"] == 20

    assert params["max_depth"] == 5


# ============================================================
# Search space integrity
# ============================================================

def test_all_parameters_are_sampled() -> None:

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42,
    )

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    search_space = SearchSpace(
        name="rf_test",
        parameters={
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
            "criterion": [
                "gini",
                "entropy",
            ],
        },
    )

    builder = ObjectiveBuilder(
        spec=spec,
        search_space=search_space,
        metric="accuracy",
        cv=3,
        X=X,
        y=y,
    )

    trial = FixedTrial(
        {
            "n_estimators": 10,
            "max_depth": 3,
            "criterion": "gini",
        }
    )

    params = builder.sample_params(
        trial,
    )

    assert len(
        params
    ) == 3

    assert (
        "criterion"
        in params
    )