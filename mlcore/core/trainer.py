"""
mlcore.core.trainer
====================

Core training engine for mlcore.

The Trainer orchestrates:
- algorithm selection from registry
- execution via AlgorithmSpec runners
- backend lifecycle management
- optional validation workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .registry import AlgorithmRegistry
from .specs import AlgorithmSpec


# ============================================================
# Training result container
# ============================================================

@dataclass
class TrainResult:
    """
    Container for training outputs.
    """

    model: Any
    backend: Any
    spec: AlgorithmSpec

    metrics: Optional[Dict[str, float]] = None
    predictions: Optional[np.ndarray] = None


# ============================================================
# Trainer
# ============================================================

class Trainer:
    """
    Core training engine for classical ML models.

    This class is responsible for:
    - retrieving algorithms from registry
    - executing training via runners
    - managing backend state
    """

    def __init__(
        self,
        registry: AlgorithmRegistry,
    ) -> None:

        self.registry = registry

    # --------------------------------------------------------
    # Main API
    # --------------------------------------------------------

    def fit(
        self,
        algorithm: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        params: Optional[Dict[str, Any]] = None,
        return_predictions: bool = False,
    ) -> TrainResult:
        """
        Fit a model using a registered algorithm.

        Parameters
        ----------
        algorithm : str
            Algorithm name or alias.

        X : np.ndarray
            Training features.

        y : np.ndarray
            Training labels.

        params : dict, optional
            Hyperparameters passed to runner.

        return_predictions : bool
            Whether to compute training predictions.

        Returns
        -------
        TrainResult
        """

        spec = self.registry.get(algorithm)

        backend = spec.backend_cls()

        params = params or {}

        # ----------------------------------------------------
        # Execute runner
        # ----------------------------------------------------

        spec.runner(
            backend,
            X,
            y,
            **{**spec.default_params, **params},
        )

        model = backend.model if hasattr(backend, "model") else None

        predictions = None

        if return_predictions and model is not None:

            if hasattr(model, "predict"):
                predictions = model.predict(X)

        return TrainResult(
            model=model,
            backend=backend,
            spec=spec,
            predictions=predictions,
        )

    # --------------------------------------------------------
    # Simple evaluation hook (minimal, extensible)
    # --------------------------------------------------------

    def evaluate(
        self,
        result: TrainResult,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric_fn: Any,
    ) -> Dict[str, float]:
        """
        Evaluate a trained model using a metric function.

        Parameters
        ----------
        result : TrainResult
            Output from training.

        X_test : np.ndarray
            Test features.

        y_test : np.ndarray
            Test labels.

        metric_fn : callable
            Metric function (e.g., accuracy_score).

        Returns
        -------
        dict
        """

        model = result.model

        if model is None:
            raise ValueError("Model is not available for evaluation.")

        preds = model.predict(X_test)

        score = metric_fn(y_test, preds)

        return {
            "score": float(score),
        }

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    def predict(
        self,
        result: TrainResult,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Generate predictions from a trained model.
        """

        model = result.model

        if model is None:
            raise ValueError("Model is not available.")

        return model.predict(X)