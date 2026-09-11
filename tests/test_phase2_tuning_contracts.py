from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from mlcore import MODEL_REGISTRY
from mlcore.core.search_space import SearchSpace
from mlcore.exceptions import MetricTaskMismatchError, NonFiniteScoreError
from mlcore.tuning.results import OptimizationResult
from mlcore.tuning.sklearn.grid import GridSearchOptimizer


LOGREG_SPACE = SearchSpace(
    name="phase2_logreg",
    parameters={"C": [0.1, 1.0]},
)

RIDGE_SPACE = SearchSpace(
    name="phase2_ridge",
    parameters={"alpha": [0.1, 1.0]},
)


def test_grid_roc_auc_returns_finite_score() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        random_state=42,
    )
    result = GridSearchOptimizer(MODEL_REGISTRY).optimize(
        "logistic_regression",
        X,
        y,
        metric="roc_auc",
        cv=3,
        search_space=LOGREG_SPACE,
    )
    assert np.isfinite(result.best_score)
    assert result.best_score > 0.5


def test_grid_refit_false_keeps_best_params_without_model() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        random_state=42,
    )
    result = GridSearchOptimizer(MODEL_REGISTRY).optimize(
        "logistic_regression",
        X,
        y,
        metric="accuracy",
        cv=3,
        refit=False,
        search_space=LOGREG_SPACE,
    )
    assert not result.has_model()
    assert result.refit is False
    assert result.best_params
    assert np.isfinite(result.best_score)


def test_optimizer_rejects_metric_from_wrong_task() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        random_state=42,
    )
    with pytest.raises(MetricTaskMismatchError):
        GridSearchOptimizer(MODEL_REGISTRY).optimize(
            "logistic_regression",
            X,
            y,
            metric="rmse",
            cv=3,
            search_space=LOGREG_SPACE,
        )


def test_regression_loss_display_score_is_natural_positive_value() -> None:
    X, y = make_regression(
        n_samples=120,
        n_features=8,
        noise=0.1,
        random_state=42,
    )
    result = GridSearchOptimizer(MODEL_REGISTRY).optimize(
        "ridge_regressor",
        X,
        y,
        metric="rmse",
        cv=3,
        search_space=RIDGE_SPACE,
    )
    assert result.best_score < 0.0
    assert result.display_score > 0.0
    assert np.isclose(result.display_score, -result.best_score)


def test_optimization_result_rejects_non_finite_best_score() -> None:
    with pytest.raises(NonFiniteScoreError):
        OptimizationResult(
            algorithm="logistic_regression",
            best_score=float("nan"),
            best_params={"C": 1.0},
            optimizer="grid_search",
            metric="accuracy",
        )
