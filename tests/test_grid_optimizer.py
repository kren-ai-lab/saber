"""
tests.test_grid_optimizer
=========================

Integration tests for GridSearchOptimizer.
"""

from __future__ import annotations

import pytest

from sklearn.datasets import (
    make_classification,
    make_regression,
)

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.search_space import SearchSpace

from mlcore.tuning.sklearn.grid import (
    GridSearchOptimizer,
)

from mlcore.tuning.results import (
    OptimizationResult,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def classification_dataset():

    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

    return X, y


@pytest.fixture
def regression_dataset():

    X, y = make_regression(
        n_samples=150,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42,
    )

    return X, y


@pytest.fixture
def optimizer():

    return GridSearchOptimizer(
        registry=MODEL_REGISTRY,
    )


# ============================================================
# Tiny search spaces
# ============================================================

RF_TEST_SPACE = SearchSpace(
    name="rf_test",
    parameters={
        "n_estimators": [10, 20],
        "max_depth": [3, 5],
    },
)

LOGREG_TEST_SPACE = SearchSpace(
    name="logreg_test",
    parameters={
        "C": [0.1, 1.0],
    },
)

RIDGE_TEST_SPACE = SearchSpace(
    name="ridge_test",
    parameters={
        "alpha": [0.1, 1.0],
    },
)

RF_REG_TEST_SPACE = SearchSpace(
    name="rf_reg_test",
    parameters={
        "n_estimators": [10, 20],
        "max_depth": [3, 5],
    },
)


# ============================================================
# Classification
# ============================================================

def test_random_forest_grid_search(
    optimizer,
    classification_dataset,
) -> None:

    X, y = classification_dataset

    result = optimizer.optimize(
        algorithm="random_forest",
        X=X,
        y=y,
        metric="accuracy",
        cv=3,
        search_space=RF_TEST_SPACE,
    )

    assert isinstance(
        result,
        OptimizationResult,
    )

    assert result.algorithm == "random_forest"

    assert result.has_model()

    assert result.best_score > 0.0

    assert len(
        result.best_params
    ) > 0


def test_logistic_regression_grid_search(
    optimizer,
    classification_dataset,
) -> None:

    X, y = classification_dataset

    result = optimizer.optimize(
        algorithm="logistic_regression",
        X=X,
        y=y,
        metric="accuracy",
        cv=3,
        search_space=LOGREG_TEST_SPACE,
    )

    assert result.has_model()

    assert result.best_score > 0.0


# ============================================================
# Regression
# ============================================================

def test_random_forest_regressor_grid_search(
    optimizer,
    regression_dataset,
) -> None:

    X, y = regression_dataset

    result = optimizer.optimize(
        algorithm="random_forest_regressor",
        X=X,
        y=y,
        metric="r2",
        cv=3,
        search_space=RF_REG_TEST_SPACE,
    )

    assert result.has_model()

    assert len(
        result.best_params
    ) > 0


def test_ridge_regressor_grid_search(
    optimizer,
    regression_dataset,
) -> None:

    X, y = regression_dataset

    result = optimizer.optimize(
        algorithm="ridge_regressor",
        X=X,
        y=y,
        metric="r2",
        cv=3,
        search_space=RIDGE_TEST_SPACE,
    )

    assert result.has_model()

    assert result.best_score > 0.0


# ============================================================
# History
# ============================================================

def test_history_is_created(
    optimizer,
    classification_dataset,
) -> None:

    X, y = classification_dataset

    result = optimizer.optimize(
        algorithm="random_forest",
        X=X,
        y=y,
        metric="accuracy",
        cv=3,
        search_space=RF_TEST_SPACE,
    )

    assert result.has_history()

    assert len(
        result.history
    ) > 0


# ============================================================
# Search space
# ============================================================

def test_best_params_are_from_search_space(
    optimizer,
    classification_dataset,
) -> None:

    X, y = classification_dataset

    result = optimizer.optimize(
        algorithm="random_forest",
        X=X,
        y=y,
        metric="accuracy",
        cv=3,
        search_space=RF_TEST_SPACE,
    )

    assert (
        result.best_params["n_estimators"]
        in [10, 20]
    )

    assert (
        result.best_params["max_depth"]
        in [3, 5]
    )


# ============================================================
# Study object
# ============================================================

def test_study_object_is_available(
    optimizer,
    classification_dataset,
) -> None:

    X, y = classification_dataset

    result = optimizer.optimize(
        algorithm="random_forest",
        X=X,
        y=y,
        metric="accuracy",
        cv=3,
        search_space=RF_TEST_SPACE,
    )

    assert result.has_study()


# ============================================================
# Error handling
# ============================================================

def test_unknown_algorithm_raises(
    optimizer,
    classification_dataset,
) -> None:

    X, y = classification_dataset

    with pytest.raises(
        Exception,
    ):
        optimizer.optimize(
            algorithm="papucho_forest",
            X=X,
            y=y,
            metric="accuracy",
        )