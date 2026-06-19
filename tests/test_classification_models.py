"""
tests.test_classification_models
================================

Integration tests for classification models.
"""

from __future__ import annotations

import numpy as np
import pytest

from sklearn.datasets import make_classification

from mlcore.core.registry import MODEL_REGISTRY


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def classification_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Create a small synthetic classification dataset.
    """

    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )

    return X, y


# ============================================================
# Registry
# ============================================================

def test_models_are_registered() -> None:
    """
    Verify auto-registration.
    """

    assert MODEL_REGISTRY.exists("logistic_regression")
    assert MODEL_REGISTRY.exists("random_forest")
    assert MODEL_REGISTRY.exists("xgb_classifier")
    assert MODEL_REGISTRY.exists("lgbm_classifier")


# ============================================================
# Sklearn
# ============================================================

def test_logistic_regression_runner(
    classification_dataset: tuple[np.ndarray, np.ndarray],
) -> None:

    X, y = classification_dataset

    spec = MODEL_REGISTRY.get(
        "logistic_regression",
    )

    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
    )

    assert backend.has_model()
    assert backend.has_predictions()

    assert len(
        backend.get_predictions(),
    ) == len(y)


def test_random_forest_runner(
    classification_dataset: tuple[np.ndarray, np.ndarray],
) -> None:

    X, y = classification_dataset

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
        n_estimators=10,
        random_state=42,
    )

    assert backend.has_model()

    assert (
        backend.get_predictions().shape[0]
        == len(y)
    )


# ============================================================
# XGBoost
# ============================================================

def test_xgb_classifier_runner(
    classification_dataset: tuple[np.ndarray, np.ndarray],
) -> None:

    X, y = classification_dataset

    spec = MODEL_REGISTRY.get(
        "xgb_classifier",
    )

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

    assert (
        backend.get_predictions().shape[0]
        == len(y)
    )


# ============================================================
# LightGBM
# ============================================================

def test_lgbm_classifier_runner(
    classification_dataset: tuple[np.ndarray, np.ndarray],
) -> None:

    X, y = classification_dataset

    spec = MODEL_REGISTRY.get(
        "lgbm_classifier",
    )

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

    assert (
        backend.get_predictions().shape[0]
        == len(y)
    )


# ============================================================
# Probabilities
# ============================================================

def test_probability_storage(
    classification_dataset: tuple[np.ndarray, np.ndarray],
) -> None:

    X, y = classification_dataset

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
        n_estimators=10,
        random_state=42,
    )

    assert backend.has_probabilities()

    probabilities = backend.get_probabilities()

    assert probabilities is not None

    assert probabilities.shape[0] == len(y)


# ============================================================
# Metadata
# ============================================================

def test_class_storage(
    classification_dataset: tuple[np.ndarray, np.ndarray],
) -> None:

    X, y = classification_dataset

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
        n_estimators=10,
        random_state=42,
    )

    metadata = backend.get_metadata()

    assert "classes" in metadata

    assert len(
        metadata["classes"],
    ) == 2


def test_feature_count_storage(
    classification_dataset: tuple[np.ndarray, np.ndarray],
) -> None:

    X, y = classification_dataset

    spec = MODEL_REGISTRY.get(
        "random_forest",
    )

    backend = spec.backend_cls()

    spec.runner(
        backend,
        X,
        y,
        n_estimators=10,
        random_state=42,
    )

    metadata = backend.get_metadata()

    assert (
        metadata["n_features"]
        == X.shape[1]
    )