"""
mlcore.tuning.results
=====================

Result objects returned by optimization methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mlcore.core.metrics import get_metric_spec
from mlcore.core.specs import AlgorithmSpec
from mlcore.exceptions import NonFiniteScoreError


@dataclass(slots=True)
class OptimizationResult:
    """Stores the result of a hyperparameter optimization process."""

    algorithm: str
    best_score: float
    best_params: dict[str, Any]
    best_model: Any = None
    spec: AlgorithmSpec | None = None
    optimizer: str | None = None
    metric: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    study: Any = None
    refit: bool | None = None

    def __post_init__(self) -> None:
        self.best_score = float(self.best_score)
        if not np.isfinite(self.best_score):
            raise NonFiniteScoreError(
                algorithm=self.algorithm,
                metric=self.metric or "unknown",
                optimizer=self.optimizer or "unknown",
                score=self.best_score,
            )

    @property
    def display_score(self) -> float:
        """User-facing score in the metric's natural direction."""

        if self.metric is None:
            return self.best_score
        return get_metric_spec(self.metric).to_natural_score(self.best_score)

    @property
    def score_is_finite(self) -> bool:
        """Whether the stored best score is finite."""

        return bool(np.isfinite(self.best_score))

    def has_model(self) -> bool:
        """Check whether a best fitted model is available."""

        return self.best_model is not None

    def has_history(self) -> bool:
        """Check whether optimization history exists."""

        return len(self.history) > 0

    def has_study(self) -> bool:
        """Check whether a backend-specific study object exists."""

        return self.study is not None

    def get_best_param(self, name: str) -> Any:
        """Retrieve a best parameter value."""

        return self.best_params[name]

    def get_history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of optimization history."""

        return list(self.history)

    def to_dict(self) -> dict[str, Any]:
        """Export result as dictionary."""

        return {
            "algorithm": self.algorithm,
            "optimizer": self.optimizer,
            "metric": self.metric,
            "best_score": self.best_score,
            "display_score": self.display_score,
            "best_params": self.best_params,
            "refit": self.refit,
        }

    def summary(self) -> dict[str, Any]:
        """Return concise optimization summary."""

        return self.to_dict()
