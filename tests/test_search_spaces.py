"""
tests.test_search_spaces
========================

Tests for search space integration.
"""

from __future__ import annotations

import mlcore.classification
import mlcore.regression

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.search_space import SearchSpace


# ============================================================
# Classification
# ============================================================

def test_random_forest_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    assert spec.has_search_space()

    space = spec.get_search_space()

    assert isinstance(
        space,
        SearchSpace,
    )

    assert (
        "n_estimators"
        in space.parameters
    )

    assert (
        "max_depth"
        in space.parameters
    )


def test_logistic_regression_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "logistic_regression",
    )

    assert spec.has_search_space()

    params = spec.get_search_space_parameters()

    assert "C" in params

    assert "solver" in params


def test_xgb_classifier_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "xgb_classifier",
    )

    assert spec.has_search_space()

    params = spec.get_search_space_parameters()

    assert "n_estimators" in params

    assert "learning_rate" in params


def test_lgbm_classifier_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "lgbm_classifier",
    )

    assert spec.has_search_space()

    params = spec.get_search_space_parameters()

    assert "num_leaves" in params

    assert "learning_rate" in params


# ============================================================
# Regression
# ============================================================

def test_random_forest_regressor_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "random_forest_regressor",
    )

    assert spec.has_search_space()

    params = spec.get_search_space_parameters()

    assert "n_estimators" in params

    assert "max_depth" in params


def test_ridge_regressor_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "ridge_regressor",
    )

    assert spec.has_search_space()

    params = spec.get_search_space_parameters()

    assert "alpha" in params


def test_xgb_regressor_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "xgb_regressor",
    )

    assert spec.has_search_space()

    params = spec.get_search_space_parameters()

    assert "n_estimators" in params

    assert "learning_rate" in params


def test_lgbm_regressor_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "lgbm_regressor",
    )

    assert spec.has_search_space()

    params = spec.get_search_space_parameters()

    assert "num_leaves" in params

    assert "learning_rate" in params


def test_svr_has_search_space() -> None:

    spec = MODEL_REGISTRY.get(
        "svr",
    )

    assert spec.has_search_space()

    params = spec.get_search_space_parameters()

    assert "C" in params

    assert "epsilon" in params


# ============================================================
# Generic API
# ============================================================

def test_search_space_is_accessible() -> None:

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    space = spec.get_search_space()

    assert space is not None

    assert len(space) > 0


def test_search_space_parameters_returns_dict() -> None:

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    params = spec.get_search_space_parameters()

    assert isinstance(
        params,
        dict,
    )

    assert (
        "n_estimators"
        in params
    )


def test_regression_search_space_parameters_returns_dict() -> None:

    spec = MODEL_REGISTRY.get(
        "random_forest_regressor",
    )

    params = spec.get_search_space_parameters()

    assert isinstance(
        params,
        dict,
    )

    assert (
        "n_estimators"
        in params
    )