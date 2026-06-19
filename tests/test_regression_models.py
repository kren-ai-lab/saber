"""
tests.test_regression_models
============================

Integration tests for regression models.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.datasets import make_regression

from mlcore.core.registry import MODEL_REGISTRY


@pytest.fixture
def regression_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )

    return X, y


def test_regression_models_are_registered() -> None:
    assert MODEL_REGISTRY.exists("linear_regression")
    assert MODEL_REGISTRY.exists("random_forest_regressor")
    assert MODEL_REGISTRY.exists("xgb_regressor")
    assert MODEL_REGISTRY.exists("lgbm_regressor")


def test_linear_regression_runner(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_dataset

    spec = MODEL_REGISTRY.get("linear_regression")
    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
    )

    assert backend.has_model()
    assert backend.has_predictions()
    assert backend.get_predictions() is not None
    assert backend.get_predictions().shape[0] == len(y)


def test_random_forest_regressor_runner(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_dataset

    spec = MODEL_REGISTRY.get("random_forest_regressor")
    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
        n_estimators=10,
        random_state=42,
    )

    assert backend.has_model()
    assert backend.has_predictions()
    assert backend.get_predictions() is not None
    assert backend.get_predictions().shape[0] == len(y)


def test_xgb_regressor_runner(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_dataset

    spec = MODEL_REGISTRY.get("xgb_regressor")
    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
        n_estimators=10,
        max_depth=3,
        random_state=42,
        verbosity=0,
    )

    assert backend.has_model()
    assert backend.has_predictions()
    assert backend.get_predictions() is not None
    assert backend.get_predictions().shape[0] == len(y)


def test_lgbm_regressor_runner(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_dataset

    spec = MODEL_REGISTRY.get("lgbm_regressor")
    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
        n_estimators=10,
        random_state=42,
        verbose=-1,
    )

    assert backend.has_model()
    assert backend.has_predictions()
    assert backend.get_predictions() is not None
    assert backend.get_predictions().shape[0] == len(y)


def test_regression_metadata_storage(
    regression_dataset: tuple[np.ndarray, np.ndarray],
) -> None:
    X, y = regression_dataset

    spec = MODEL_REGISTRY.get("random_forest_regressor")
    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
        n_estimators=10,
        random_state=42,
    )

    metadata = backend.get_metadata()

    assert "n_features" in metadata
    assert metadata["n_features"] == X.shape[1]