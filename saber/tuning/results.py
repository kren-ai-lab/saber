"""saber.tuning.results: Result objects returned by optimization methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from saber.exceptions import NonFiniteScoreError
from saber.utils.tabular import records_frame

if TYPE_CHECKING:
    import polars as pl

    from saber.core.specs import AlgorithmSpec
    from saber.datasets.schemas import FeatureSchema


@dataclass(slots=True)
class OptimizationResult:
    """Stores the result of a hyperparameter optimization process.

    ``best_score`` (for ``refit_metric``), ``best_scores`` and the history
    report each metric in its natural direction (e.g. RMSE is positive, lower
    is better); ranks in the history keep 1 as the best candidate.
    """

    algorithm: str
    best_score: float
    best_params: dict[str, Any]
    best_model: Any = None
    spec: AlgorithmSpec | None = None
    optimizer: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    study: Any = None
    refit: bool | None = None
    metrics: tuple[str, ...] = field(default_factory=tuple)
    refit_metric: str | None = None
    best_scores: dict[str, float] = field(default_factory=dict)
    partition_plan: Any = None
    feature_schema: FeatureSchema | None = None
    positive_class: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize numeric fields and validate that the best score is finite."""
        self.best_score = float(self.best_score)
        self.metrics = tuple(self.metrics)
        self.best_scores = {name: float(value) for name, value in self.best_scores.items()}
        self.metadata = dict(self.metadata)
        if self.refit_metric is not None and self.refit_metric not in self.best_scores:
            self.best_scores[self.refit_metric] = self.best_score
        if not np.isfinite(self.best_score):
            raise NonFiniteScoreError(
                algorithm=self.algorithm,
                metric=self.refit_metric or "unknown",
                optimizer=self.optimizer or "unknown",
                score=self.best_score,
            )

    @property
    def failures(self) -> list[dict[str, Any]]:
        """Return candidate/trial history entries explicitly marked as failed."""
        return [entry for entry in self.history if entry.get("status") == "failed"]

    def history_frame(self) -> pl.DataFrame:
        """Return optimization history as a flat Polars DataFrame."""
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
        return records_frame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly summary; ``metrics`` holds the best candidate's scores."""
        return {
            "algorithm": self.algorithm,
            "task": None if self.spec is None else self.spec.task,
            "dataset_fingerprint": self.metadata.get("dataset_fingerprint"),
            "partition_fingerprint": self.metadata.get("partition_fingerprint"),
            "optimizer": self.optimizer,
            "refit_metric": self.refit_metric,
            "best_score": self.best_score,
            "best_params": dict(self.best_params),
            "metrics": dict(self.best_scores),
            "n_candidates": len(self.history),
            "n_failures": len(self.failures),
            "refit": self.refit,
        }
