"""Structured validation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from saber.core.prediction import PredictionResult
from saber.datasets.folds import PartitionPlan
from saber.evaluation.results import EvaluationResult


@dataclass(slots=True)
class FoldValidationResult:
    """Outputs from one explicit validation split."""

    split_name: str
    evaluation_role: str
    train_ids: tuple[Any, ...]
    evaluation_ids: tuple[Any, ...]
    prediction: PredictionResult
    evaluation: EvaluationResult
    estimator: Any | None
    fit_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationResult:
    """Aggregate result across explicit holdout/CV partitions."""

    algorithm: str
    task: str
    partition_plan: PartitionPlan
    folds: tuple[FoldValidationResult, ...]
    aggregate_metrics: dict[str, float]
    metric_summary: dict[str, dict[str, float]]
    oof_prediction: PredictionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_splits(self) -> int:
        return len(self.folds)

    @property
    def total_fit_seconds(self) -> float:
        return float(sum(fold.fit_seconds for fold in self.folds))


def aggregate_fold_metrics(
    folds: tuple[FoldValidationResult, ...],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Aggregate finite metrics across folds."""
    metric_names = sorted({name for fold in folds for name in fold.evaluation.metrics})
    means: dict[str, float] = {}
    summary: dict[str, dict[str, float]] = {}

    for name in metric_names:
        values = np.asarray(
            [
                fold.evaluation.metrics[name]
                for fold in folds
                if name in fold.evaluation.metrics and np.isfinite(fold.evaluation.metrics[name])
            ],
            dtype=float,
        )
        if not values.size:
            continue
        means[name] = float(np.mean(values))
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": float(values.size),
        }

    return means, summary
