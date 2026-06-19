"""
tests.test_optuna_optimizer
===========================

Integration tests for OptunaOptimizer.
"""

from __future__ import annotations

import pytest

from sklearn.datasets import (
    make_classification,
    make_regression,
)

import mlcore.classification
import mlcore.regression

from mlcore import MODEL_REGISTRY

from mlcore.core.search_space import SearchSpace

from mlcore.tuning.optuna import (
    OptunaOptimizer,
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

    return OptunaOptimizer(
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

RF_REG_TEST_SPACE = SearchSpace(
    name="rf_reg_test",
    parameters={
        "n_estimators": [10, 20],
        "max_depth": [3, 5],
    },
)

RIDGE_TEST_SPACE = SearchSpace(
    name="ridge_test",
    parameters={
        "alpha": [0.1, 1.0],
    },
)


# ============================================================
# Classification
# ============================================================

def test_random_forest_optuna(
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
        n_trials=5,
        search_space=RF_TEST_SPACE,
    )

    assert isinstance(
        result,
        OptimizationResult,
    )

    assert result.has_model()

    assert result.best_score > 0.0


def test_logistic_regression_optuna(
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
        n_trials=5,
        search_space=LOGREG_TEST_SPACE,
    )

    assert result.has_model()

    assert result.best_score > 0.0


# ============================================================
# Regression
# ============================================================

def test_random_forest_regressor_optuna(
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
        n_trials=5,
        search_space=RF_REG_TEST_SPACE,
    )

    assert result.has_model()

    assert result.best_score > 0.0


def test_ridge_regressor_optuna(
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
        n_trials=5,
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
        n_trials=5,
        search_space=RF_TEST_SPACE,
    )

    assert result.has_history()

    assert len(
        result.history
    ) > 0


# ============================================================
# Best params
# ============================================================

def test_best_params_are_available(
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
        n_trials=5,
        search_space=RF_TEST_SPACE,
    )

    assert (
        "n_estimators"
        in result.best_params
    )

    assert (
        "max_depth"
        in result.best_params
    )


# ============================================================
# Study
# ============================================================

def test_study_is_available(
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
        n_trials=5,
        search_space=RF_TEST_SPACE,
    )

    assert result.has_study()


# ============================================================
# Search space consistency
# ============================================================

def test_best_params_belong_to_search_space(
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
        n_trials=5,
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
# Errors
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
            algorithm="unknown_algorithm",
            X=X,
            y=y,
            metric="accuracy",
        )