"""Structured validation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from saber.utils.tabular import records_frame

if TYPE_CHECKING:
    import polars as pl

    from saber.core.prediction import PredictionResult
    from saber.datasets.folds import PartitionPlan


@dataclass(slots=True)
class FoldValidationResult:
    """Outputs from one explicit validation split."""

    split: str
    evaluation_role: str
    train_ids: tuple[Any, ...]
    evaluation_ids: tuple[Any, ...]
    prediction: PredictionResult
    metrics: dict[str, float]
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
    oof_prediction: PredictionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_splits(self) -> int:
        """Return the number of folds in this result."""
        return len(self.folds)

    @property
    def total_fit_seconds(self) -> float:
        """Return the summed fit time across all folds."""
        return float(sum(fold.fit_seconds for fold in self.folds))

    def metrics_frame(self) -> pl.DataFrame:
        """Return metrics in long form: one ``aggregate`` row per metric, then one ``fold`` row per split.

        Columns: ``level``, ``split``, ``evaluation_role``, ``metric``, ``score``
        (fold score, or the mean of finite fold scores), ``std``/``min``/``max``/``n``
        (aggregate rows only; sample std, ``n`` finite folds) and ``fit_seconds``
        (fold rows only).
        """
        rows: list[dict[str, Any]] = [
            {
                "level": "aggregate",
                "split": None,
                "evaluation_role": None,
                "metric": metric,
                "score": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "max": stats["max"],
                "n": stats["n"],
                "fit_seconds": None,
            }
            for metric, stats in aggregate_fold_metrics(self.folds).items()
        ]
        rows += [
            {
                "level": "fold",
                "split": fold.split,
                "evaluation_role": fold.evaluation_role,
                "metric": metric,
                "score": float(score),
                "std": None,
                "min": None,
                "max": None,
                "n": None,
                "fit_seconds": float(fold.fit_seconds),
            }
            for fold in self.folds
            for metric, score in fold.metrics.items()
        ]
        return records_frame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly summary."""
        return {
            "algorithm": self.algorithm,
            "task": self.task,
            "dataset_fingerprint": self.metadata.get("dataset_fingerprint"),
            "partition_fingerprint": self.metadata.get("partition_fingerprint"),
            "n_splits": self.n_splits,
            "total_fit_seconds": self.total_fit_seconds,
            "oof_available": self.oof_prediction is not None,
            "metrics": dict(self.aggregate_metrics),
        }


def aggregate_fold_metrics(folds: tuple[FoldValidationResult, ...]) -> dict[str, dict[str, Any]]:
    """Summarize finite fold scores per metric: ``mean``, sample ``std``, ``min``, ``max`` and ``n``."""
    metric_names = sorted({name for fold in folds for name in fold.metrics})
    summary: dict[str, dict[str, Any]] = {}

    for name in metric_names:
        values = np.asarray(
            [
                fold.metrics[name]
                for fold in folds
                if name in fold.metrics and np.isfinite(fold.metrics[name])
            ],
            dtype=float,
        )
        if not values.size:
            continue
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n": int(values.size),
        }

    return summary
