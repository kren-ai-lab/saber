"""saber.tuning.results
=====================

Result objects returned by optimization methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from saber.core.metrics import get_metric_spec
from saber.core.specs import AlgorithmSpec
from saber.exceptions import NonFiniteScoreError


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
    metrics: tuple[str, ...] = field(default_factory=tuple)
    refit_metric: str | None = None
    best_scores: dict[str, float] = field(default_factory=dict)
    partition_plan: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.best_score = float(self.best_score)
        self.metrics = tuple(self.metrics)
        self.best_scores = {name: float(value) for name, value in self.best_scores.items()}
        self.metadata = dict(self.metadata)
        if self.metric is not None and not self.metrics:
            self.metrics = (self.metric,)
        if self.refit_metric is None:
            self.refit_metric = self.metric
        if self.metric is not None and self.metric not in self.best_scores:
            self.best_scores[self.metric] = self.best_score
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

    @property
    def display_scores(self) -> dict[str, float]:
        """Best candidate scores converted to each metric's natural direction."""
        return {
            name: get_metric_spec(name).to_natural_score(value) for name, value in self.best_scores.items()
        }

    @property
    def failures(self) -> list[dict[str, Any]]:
        """Return candidate/trial history entries explicitly marked as failed."""
        return [entry for entry in self.history if entry.get("status") == "failed"]

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

    def history_frame(self):
        """Return optimization history as a flat pandas DataFrame."""
        import pandas as pd

        rows: list[dict[str, Any]] = []
        for entry in self.history:
            row: dict[str, Any] = {
                key: value for key, value in entry.items() if key not in {"params", "metrics"}
            }
            for name, value in entry.get("params", {}).items():
                row[f"param__{name}"] = value
            for metric, payload in entry.get("metrics", {}).items():
                if isinstance(payload, dict):
                    for statistic, value in payload.items():
                        row[f"metric__{metric}__{statistic}"] = value
                else:
                    row[f"metric__{metric}"] = payload
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Export result as dictionary."""
        return {
            "algorithm": self.algorithm,
            "optimizer": self.optimizer,
            "metric": self.metric,
            "metrics": self.metrics,
            "refit_metric": self.refit_metric,
            "best_score": self.best_score,
            "display_score": self.display_score,
            "best_scores": dict(self.best_scores),
            "display_scores": self.display_scores,
            "best_params": self.best_params,
            "refit": self.refit,
            "n_failures": len(self.failures),
            "metadata": dict(self.metadata),
        }

    def summary(self) -> dict[str, Any]:
        """Return concise optimization summary."""
        return self.to_dict()
