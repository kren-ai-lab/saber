"""
mlcore.core.prediction
======================

Structured prediction contracts shared by training and evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mlcore.core.task import TaskType
from mlcore.exceptions import PredictionContractError


@dataclass(slots=True)
class PredictionResult:
    """Structured predictions with explicit class/probability semantics."""

    task: TaskType
    predictions: np.ndarray
    probabilities: np.ndarray | None = None
    decision_scores: np.ndarray | None = None
    classes: np.ndarray | None = None
    positive_class: Any | None = None
    sample_ids: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predictions = np.asarray(self.predictions)

        if self.predictions.ndim != 1:
            raise PredictionContractError(
                "predictions must be a one-dimensional array."
            )

        if self.probabilities is not None:
            self.probabilities = np.asarray(self.probabilities)

        if self.decision_scores is not None:
            self.decision_scores = np.asarray(self.decision_scores)

        if self.classes is not None:
            self.classes = np.asarray(self.classes)
            if self.classes.ndim != 1:
                raise PredictionContractError(
                    "classes must be a one-dimensional array."
                )

        if self.sample_ids is not None:
            self.sample_ids = np.asarray(self.sample_ids)
            if self.sample_ids.ndim != 1:
                raise PredictionContractError(
                    "sample_ids must be a one-dimensional array."
                )
            if self.sample_ids.shape[0] != self.n_samples:
                raise PredictionContractError(
                    "sample_ids length must match predictions length."
                )

        self._validate_response_shapes()
        self._validate_task_semantics()

    @property
    def n_samples(self) -> int:
        """Number of predicted samples."""

        return int(self.predictions.shape[0])

    @property
    def n_classes(self) -> int | None:
        """Number of known classes for classification results."""

        if self.classes is None:
            return None
        return int(self.classes.shape[0])

    @property
    def is_binary(self) -> bool:
        """Whether this is an explicitly binary classification result."""

        return self.task == "classification" and self.n_classes == 2

    def class_index(self, label: Any) -> int:
        """Return the probability-column index associated with a class label."""

        if self.classes is None:
            raise PredictionContractError(
                "Class order is unavailable for this prediction result."
            )

        matches = np.flatnonzero(self.classes == label)
        if matches.size != 1:
            raise PredictionContractError(
                f"Class {label!r} is not present in classes."
            )
        return int(matches[0])

    def probabilities_for(self, label: Any) -> np.ndarray:
        """Return one-dimensional probabilities for a requested class."""

        if self.probabilities is None:
            raise PredictionContractError(
                "Probability predictions are unavailable."
            )

        probabilities = self.probabilities

        if probabilities.ndim == 1:
            if self.positive_class is None:
                raise PredictionContractError(
                    "One-dimensional probabilities require positive_class."
                )
            if label != self.positive_class:
                if self.is_binary:
                    return 1.0 - probabilities
                raise PredictionContractError(
                    "One-dimensional probabilities only describe positive_class."
                )
            return probabilities

        if probabilities.ndim != 2:
            raise PredictionContractError(
                "Probabilities must be one- or two-dimensional."
            )

        return probabilities[:, self.class_index(label)]

    def positive_probabilities(self) -> np.ndarray:
        """Return binary positive-class probabilities as a one-dimensional array."""

        if not self.is_binary:
            raise PredictionContractError(
                "Positive-class probabilities are only defined for binary classification."
            )

        if self.positive_class is None:
            raise PredictionContractError(
                "positive_class is not defined."
            )

        return self.probabilities_for(self.positive_class)

    def positive_decision_scores(self) -> np.ndarray:
        """Return decision scores oriented toward the configured positive class."""

        if not self.is_binary:
            raise PredictionContractError(
                "Positive-class decision scores are only defined for binary classification."
            )
        if self.decision_scores is None:
            raise PredictionContractError(
                "Decision scores are unavailable."
            )
        if self.positive_class is None or self.classes is None:
            raise PredictionContractError(
                "Binary decision scores require classes and positive_class."
            )

        scores = self.decision_scores
        positive_index = self.class_index(self.positive_class)

        if scores.ndim == 1:
            # scikit-learn binary decision_function values are oriented toward
            # classes_[1]. Negate them when the configured positive class is
            # classes_[0].
            if positive_index == 1:
                return scores
            return -scores

        if scores.ndim == 2 and scores.shape[1] == 2:
            return scores[:, positive_index]

        raise PredictionContractError(
            "Binary decision scores must have shape (n_samples,) or (n_samples, 2)."
        )

    def _validate_response_shapes(self) -> None:
        for name, values in (
            ("probabilities", self.probabilities),
            ("decision_scores", self.decision_scores),
        ):
            if values is None:
                continue
            if values.ndim not in (1, 2):
                raise PredictionContractError(
                    f"{name} must be one- or two-dimensional."
                )
            if values.shape[0] != self.n_samples:
                raise PredictionContractError(
                    f"{name} rows must match predictions length."
                )

    def _validate_task_semantics(self) -> None:
        if self.task == "regression":
            if self.classes is not None:
                raise PredictionContractError(
                    "Regression predictions cannot define classes."
                )
            if self.probabilities is not None:
                raise PredictionContractError(
                    "Regression predictions cannot define probabilities."
                )
            if self.positive_class is not None:
                raise PredictionContractError(
                    "Regression predictions cannot define positive_class."
                )
            return

        if self.task != "classification":
            raise PredictionContractError(
                f"Unknown prediction task: {self.task}"
            )

        if self.probabilities is not None and self.probabilities.ndim == 2:
            if self.classes is None:
                raise PredictionContractError(
                    "Two-dimensional probabilities require explicit class order."
                )
            if self.probabilities.shape[1] != self.classes.shape[0]:
                raise PredictionContractError(
                    "Probability columns must match the number of classes."
                )

        if self.classes is not None and self.classes.size == 2:
            if self.positive_class is None:
                self.positive_class = self.classes[-1]
            else:
                self.class_index(self.positive_class)
        elif self.positive_class is not None:
            raise PredictionContractError(
                "positive_class is only valid for binary classification."
            )
