"""
tests.test_regression_trainer
=============================

Integration tests for regression training with Trainer.
"""

from __future__ import annotations

import pytest

from sklearn.datasets import make_regression

from mlcore.core.registry import MODEL_REGISTRY
from mlcore.core.trainer import Trainer
from mlcore.exceptions import AlgorithmNotFoundError


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def dataset() -> tuple:
    """
    Synthetic regression dataset.
    """

    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )

    return X, y


@pytest.fixture
def trainer() -> Trainer:
    """
    Trainer instance.
    """

    return Trainer(
        registry=MODEL_REGISTRY,
    )


# ============================================================
# Registry
# ============================================================

def test_regression_models_are_registered() -> None:
    """
    Ensure regression models were registered.
    """

    assert MODEL_REGISTRY.exists(
        "linear_regression",
    )

    assert MODEL_REGISTRY.exists(
        "random_forest_regressor",
    )

    assert MODEL_REGISTRY.exists(
        "xgb_regressor",
    )

    assert MODEL_REGISTRY.exists(
        "lgbm_regressor",
    )


# ============================================================
# Linear regression
# ============================================================

def test_linear_regression_training(
    trainer: Trainer,
    dataset: tuple,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="linear_regression",
        X=X,
        y=y,
    )

    backend = result.backend

    assert backend.has_model()

    assert backend.has_predictions()

    assert result.model is not None

    assert result.spec.name == "linear_regression"


# ============================================================
# Random forest
# ============================================================

def test_random_forest_regression_training(
    trainer: Trainer,
    dataset: tuple,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="random_forest_regressor",
        X=X,
        y=y,
        n_estimators=10,
        random_state=42,
    )

    backend = result.backend

    assert backend.has_model()

    assert backend.has_predictions()

    assert result.model is not None

    assert result.spec.name == "random_forest_regressor"


# ============================================================
# XGBoost
# ============================================================

def test_xgb_regression_training(
    trainer: Trainer,
    dataset: tuple,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="xgb_regressor",
        X=X,
        y=y,
        n_estimators=10,
        max_depth=3,
        verbosity=0,
        random_state=42,
    )

    backend = result.backend

    assert backend.has_model()

    assert backend.has_predictions()

    assert result.model is not None

    assert result.spec.name == "xgb_regressor"


# ============================================================
# LightGBM
# ============================================================

def test_lgbm_regression_training(
    trainer: Trainer,
    dataset: tuple,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="lgbm_regressor",
        X=X,
        y=y,
        n_estimators=10,
        verbose=-1,
        random_state=42,
    )

    backend = result.backend

    assert backend.has_model()

    assert backend.has_predictions()

    assert result.model is not None

    assert result.spec.name == "lgbm_regressor"


# ============================================================
# Metadata
# ============================================================

def test_regression_metadata_is_available(
    trainer: Trainer,
    dataset: tuple,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="random_forest_regressor",
        X=X,
        y=y,
        n_estimators=10,
        random_state=42,
    )

    metadata = result.backend.get_metadata()

    assert "n_features" in metadata

    assert metadata["n_features"] == X.shape[1]


# ============================================================
# Predictions
# ============================================================

def test_regression_prediction_with_trainer(
    trainer: Trainer,
    dataset: tuple,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="random_forest_regressor",
        X=X,
        y=y,
        n_estimators=10,
        random_state=42,
    )

    predictions = trainer.predict(
        result,
        X,
    )

    assert predictions is not None

    assert predictions.shape[0] == len(y)


def test_train_result_contains_spec(
    trainer: Trainer,
    dataset: tuple,
) -> None:

    X, y = dataset

    result = trainer.fit(
        algorithm="random_forest_regressor",
        X=X,
        y=y,
        n_estimators=10,
        random_state=42,
    )

    assert result.spec.name == "random_forest_regressor"

    assert result.spec.task == "regression"


# ============================================================
# Error handling
# ============================================================

def test_unknown_regression_algorithm_raises(
    trainer: Trainer,
    dataset: tuple,
) -> None:

    X, y = dataset

    with pytest.raises(
        AlgorithmNotFoundError,
    ):
        trainer.fit(
            algorithm="unknown_regressor",
            X=X,
            y=y,
        )