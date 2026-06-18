"""
mlcore.core.base
================

Base backend abstraction for all machine learning models.

This module defines the execution state contract for all
algorithm backends used in mlcore.

Backends are responsible ONLY for storing state:
- trained model
- predictions
- probabilities
- metrics
- parameters
- metadata
"""

from __future__ import annotations

from typing import Any

import numpy as np


class BackendBase:
    """
    Base class for all machine learning backends.

    Notes
    -----
    A backend does NOT perform training.

    Instead, it stores the execution state produced
    by a runner:

    - trained model
    - predictions
    - probabilities
    - metrics
    - hyperparameters
    - metadata
    """

    def __init__(self) -> None:
        """
        Initialize backend state.
        """

        self.model: Any = None

        self.predictions: np.ndarray | None = None
        self.probabilities: np.ndarray | None = None

        self.metrics: dict[str, float] = {}
        self.params: dict[str, Any] = {}

        self.metadata: dict[str, Any] = {}

    # ============================================================
    # State setters
    # ============================================================

    def set_model(
        self,
        model: Any,
    ) -> None:
        """
        Store trained model.
        """

        self.model = model

    def set_predictions(
        self,
        predictions: np.ndarray,
    ) -> None:
        """
        Store predictions.
        """

        self.predictions = predictions

    def set_probabilities(
        self,
        probabilities: np.ndarray,
    ) -> None:
        """
        Store probability outputs.
        """

        self.probabilities = probabilities

    def set_metrics(
        self,
        metrics: dict[str, float],
    ) -> None:
        """
        Store evaluation metrics.
        """

        self.metrics = dict(metrics)

    def set_params(
        self,
        params: dict[str, Any],
    ) -> None:
        """
        Store training parameters.
        """

        self.params = dict(params)

    def set_metadata(
        self,
        metadata: dict[str, Any],
    ) -> None:
        """
        Store metadata.
        """

        self.metadata = dict(metadata)

    def update_metadata(
        self,
        **metadata: Any,
    ) -> None:
        """
        Update metadata dictionary.
        """

        self.metadata.update(metadata)

    # ============================================================
    # Accessors
    # ============================================================

    def get_model(self) -> Any:
        """
        Return trained model.
        """

        return self.model

    def get_predictions(
        self,
    ) -> np.ndarray | None:
        """
        Return predictions.
        """

        return self.predictions

    def get_probabilities(
        self,
    ) -> np.ndarray | None:
        """
        Return probabilities.
        """

        return self.probabilities

    def get_metrics(
        self,
    ) -> dict[str, float]:
        """
        Return evaluation metrics.
        """

        return dict(self.metrics)

    def get_params(
        self,
    ) -> dict[str, Any]:
        """
        Return training parameters.
        """

        return dict(self.params)

    def get_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return metadata.
        """

        return dict(self.metadata)

    # ============================================================
    # State inspection
    # ============================================================

    def has_model(self) -> bool:
        """
        Check whether a model exists.
        """

        return self.model is not None

    def has_predictions(self) -> bool:
        """
        Check whether predictions exist.
        """

        return self.predictions is not None

    def has_probabilities(self) -> bool:
        """
        Check whether probabilities exist.
        """

        return self.probabilities is not None

    # ============================================================
    # Utility methods
    # ============================================================

    def clear(self) -> None:
        """
        Reset backend state.
        """

        self.model = None

        self.predictions = None
        self.probabilities = None

        self.metrics = {}
        self.params = {}
        self.metadata = {}

    # ============================================================
    # Hooks
    # ============================================================

    def post_fit_hook(self) -> None:
        """
        Hook executed after training.

        Intended for:

        - logging
        - persistence
        - experiment tracking
        - monitoring
        """

        return None

    def post_predict_hook(self) -> None:
        """
        Hook executed after prediction.

        Intended for:

        - logging
        - monitoring
        """

        return None

    # ============================================================
    # Magic methods
    # ============================================================

    def __len__(self) -> int:
        """
        Number of stored metrics.
        """

        return len(self.metrics)

    def __repr__(self) -> str:
        """
        Backend representation.
        """

        status = (
            "trained"
            if self.has_model()
            else "untrained"
        )

        return (
            f"{self.__class__.__name__}("
            f"status={status}, "
            f"n_metrics={len(self.metrics)}, "
            f"n_metadata={len(self.metadata)}"
            f")"
        )