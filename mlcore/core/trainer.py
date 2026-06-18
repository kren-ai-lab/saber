"""
mlcore.core.trainer
===================

Core training engine for mlcore.

The Trainer orchestrates:
- algorithm selection from registry
- execution via AlgorithmSpec runners
- backend lifecycle management
- prediction workflows
- evaluation workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mlcore.core.registry import AlgorithmRegistry
from mlcore.core.specs import AlgorithmSpec


# ============================================================
# Training result container
# ============================================================

@dataclass(slots=True)
class TrainResult:
    """
    Container for training outputs.
    """

    model: Any
    backend: Any
    spec: AlgorithmSpec

    metrics: dict[str, float] | None = None

    predictions: np.ndarray | None = None
    probabilities: np.ndarray | None = None


# ============================================================
# Trainer
# ============================================================

class Trainer:
    """
    Core training engine for machine learning models.

    Notes
    -----
    The Trainer is responsible for:

    - retrieving algorithms from the registry
    - instantiating backends
    - executing runners
    - generating predictions
    - returning a standardized TrainResult
    """

    def __init__(
        self,
        registry: AlgorithmRegistry,
    ) -> None:
        """
        Initialize trainer.

        Parameters
        ----------
        registry : AlgorithmRegistry
            Registry containing available algorithms.
        """

        self.registry = registry

    # ============================================================
    # Training
    # ============================================================

    def fit(
        self,
        algorithm: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        return_predictions: bool = False,
        return_probabilities: bool = False,
        **params: Any,
    ) -> TrainResult:
        """
        Train a registered algorithm.

        Parameters
        ----------
        algorithm : str
            Algorithm name or alias.

        X : np.ndarray
            Training features.

        y : np.ndarray
            Training targets.

        return_predictions : bool, default=False
            Whether to compute predictions after fitting.

        return_probabilities : bool, default=False
            Whether to compute probabilities after fitting.

        **params
            Hyperparameters passed to the model.

        Returns
        -------
        TrainResult
            Training result container.
        """

        spec = self.registry.get(
            algorithm,
        )

        backend = spec.backend_cls()

        runner_params = {
            **spec.default_params,
            **params,
        }

        spec.runner(
            backend,
            X,
            y,
            **runner_params,
        )

        model = backend.get_model()

        predictions = None
        probabilities = None

        if (
            return_predictions
            and model is not None
            and hasattr(model, "predict")
        ):
            predictions = model.predict(X)

        if (
            return_probabilities
            and model is not None
            and hasattr(model, "predict_proba")
        ):
            try:

                probabilities = model.predict_proba(X)

            except Exception:
                probabilities = None

        return TrainResult(
            model=model,
            backend=backend,
            spec=spec,
            metrics=backend.get_metrics(),
            predictions=predictions,
            probabilities=probabilities,
        )

    # ============================================================
    # Evaluation
    # ============================================================

    def evaluate(
        self,
        result: TrainResult,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric_fn: Any,
    ) -> dict[str, float]:
        """
        Evaluate a trained model.

        Parameters
        ----------
        result : TrainResult
            Training result.

        X_test : np.ndarray
            Test features.

        y_test : np.ndarray
            Test targets.

        metric_fn : callable
            Evaluation metric.

        Returns
        -------
        dict[str, float]
            Metric dictionary.
        """

        model = result.model

        if model is None:
            raise ValueError(
                "Model is not available for evaluation.",
            )

        predictions = model.predict(
            X_test,
        )

        score = metric_fn(
            y_test,
            predictions,
        )

        return {
            "score": float(score),
        }

    # ============================================================
    # Prediction
    # ============================================================

    def predict(
        self,
        result: TrainResult,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        result : TrainResult
            Training result.

        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted values.
        """

        model = result.model

        if model is None:
            raise ValueError(
                "Model is not available.",
            )

        return model.predict(
            X,
        )

    def predict_proba(
        self,
        result: TrainResult,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Generate class probabilities.

        Parameters
        ----------
        result : TrainResult
            Training result.

        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Class probabilities.
        """

        model = result.model

        if model is None:
            raise ValueError(
                "Model is not available.",
            )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise ValueError(
                "Model does not support probability prediction.",
            )

        return model.predict_proba(
            X,
        )