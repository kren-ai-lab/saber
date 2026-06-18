"""
mlcore.core.base
=================

Base backend abstraction for all machine learning models.

This module defines the execution state contract for all
algorithm backends used in mlcore.

Backends are responsible ONLY for storing state:
- trained model
- predictions
- probabilities
- metrics
- parameters
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ============================================================
# Backend base class
# ============================================================

class BackendBase:
    """
    Base class for all ML backends.

    A backend does NOT execute training logic.

    Instead, it stores the results produced by a runner:
    - model
    - predictions
    - probabilities
    - metrics
    - parameters
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.predictions: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None

        self.metrics: Dict[str, float] = {}
        self.params: Dict[str, Any] = {}

    # ============================================================
    # State setters (used by runners)
    # ============================================================

    def set_model(self, model: Any) -> None:
        """Store trained model."""
        self.model = model

    def set_predictions(self, preds: np.ndarray) -> None:
        """Store predictions."""
        self.predictions = preds

    def set_probabilities(self, probs: np.ndarray) -> None:
        """Store probability outputs (if available)."""
        self.probabilities = probs

    def set_metrics(self, metrics: Dict[str, float]) -> None:
        """Store evaluation metrics."""
        self.metrics = dict(metrics)

    def set_params(self, params: Dict[str, Any]) -> None:
        """Store hyperparameters used in training."""
        self.params = dict(params)

    # ============================================================
    # Accessors
    # ============================================================

    def get_model(self) -> Any:
        """Return trained model."""
        return self.model

    def get_predictions(self) -> Optional[np.ndarray]:
        """Return predictions."""
        return self.predictions

    def get_probabilities(self) -> Optional[np.ndarray]:
        """Return probability outputs."""
        return self.probabilities

    def get_metrics(self) -> Dict[str, float]:
        """Return evaluation metrics."""
        return self.metrics

    def get_params(self) -> Dict[str, Any]:
        """Return training parameters."""
        return self.params

    # ============================================================
    # Utility methods
    # ============================================================

    def has_model(self) -> bool:
        """Check if model exists."""
        return self.model is not None

    def clear(self) -> None:
        """Reset backend state."""
        self.model = None
        self.predictions = None
        self.probabilities = None
        self.metrics = {}
        self.params = {}

    # ============================================================
    # Hooks (future extensibility)
    # ============================================================

    def post_fit_hook(self) -> None:
        """
        Hook executed after training.

        Intended for:
        - logging
        - persistence
        - experiment tracking
        """
        pass

    def post_predict_hook(self) -> None:
        """
        Hook executed after prediction.

        Intended for:
        - logging
        - monitoring
        """
        pass

    # ============================================================
    # Representation
    # ============================================================

    def __repr__(self) -> str:
        status = "trained" if self.model is not None else "untrained"

        return (
            f"{self.__class__.__name__}("
            f"status={status}, "
            f"metrics={list(self.metrics.keys())}"
            f")"
        )