"""
tests.test_trainer
==================

Integration tests for Trainer.
"""

from __future__ import annotations

import pytest

from sklearn.datasets import make_classification

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.trainer import Trainer

from mlcore.exceptions import (
    AlgorithmNotFoundError,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def dataset():

    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=42,
    )

    return X, y


@pytest.fixture
def trainer():

    return Trainer(
        registry=MODEL_REGISTRY,
    )


# ============================================================
# Basic training
# ============================================================

def test_random_forest_training(
    trainer,
    dataset,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="random_forest",
        X=X,
        y=y,
        n_estimators=10,
        random_state=42,
    )

    backend = result.backend

    assert backend.has_model()
    assert backend.has_predictions()

    assert result.model is not None


def test_logistic_regression_training(
    trainer,
    dataset,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="logistic_regression",
        X=X,
        y=y,
    )

    backend = result.backend

    assert backend.has_model()
    assert backend.has_predictions()

    assert result.model is not None


# ============================================================
# XGBoost
# ============================================================

def test_xgb_training(
    trainer,
    dataset,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="xgb_classifier",
        X=X,
        y=y,
        n_estimators=10,
        verbosity=0,
    )

    backend = result.backend

    assert backend.has_model()

    assert result.model is not None


# ============================================================
# LightGBM
# ============================================================

def test_lgbm_training(
    trainer,
    dataset,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="lgbm_classifier",
        X=X,
        y=y,
        n_estimators=10,
        verbose=-1,
    )

    backend = result.backend

    assert backend.has_model()

    assert result.model is not None


# ============================================================
# Metadata
# ============================================================

def test_metadata_is_available(
    trainer,
    dataset,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="random_forest",
        X=X,
        y=y,
        n_estimators=10,
        random_state=42,
    )

    metadata = result.backend.get_metadata()

    assert "classes" in metadata
    assert "n_features" in metadata


# ============================================================
# Probabilities
# ============================================================

def test_probabilities_are_available(
    trainer,
    dataset,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="random_forest",
        X=X,
        y=y,
        n_estimators=10,
        random_state=42,
        return_probabilities=True,
    )

    backend = result.backend

    assert backend.has_probabilities()

    assert result.probabilities is not None


# ============================================================
# Predictions
# ============================================================

def test_predictions_are_available(
    trainer,
    dataset,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="random_forest",
        X=X,
        y=y,
        n_estimators=10,
        random_state=42,
        return_predictions=True,
    )

    assert result.predictions is not None

    assert len(result.predictions) == len(y)


# ============================================================
# Error handling
# ============================================================

def test_unknown_algorithm_raises(
    trainer,
    dataset,
) -> None:

    X, y = dataset

    with pytest.raises(
        AlgorithmNotFoundError,
    ):

        trainer.fit(
            algorithm="unknown_algorithm",
            X=X,
            y=y,
        )