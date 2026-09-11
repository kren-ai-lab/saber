"""
mlcore.evaluation.registry
==========================

Metric-name registry for user-facing evaluation helpers.
"""

from __future__ import annotations

from mlcore.evaluation.classification import CLASSIFICATION_METRICS
from mlcore.evaluation.regression import REGRESSION_METRICS


METRIC_REGISTRY: dict[str, tuple[str, ...]] = {
    "classification": tuple(CLASSIFICATION_METRICS.keys()),
    "regression": tuple(REGRESSION_METRICS),
}


def available_tasks() -> tuple[str, ...]:
    """Return supported evaluation tasks."""

    return tuple(METRIC_REGISTRY.keys())


def get_metrics(task: str) -> tuple[str, ...]:
    """Return metric names for a given task."""

    return METRIC_REGISTRY[task]


def has_metric(task: str, metric: str) -> bool:
    """Check whether an evaluation metric exists for a task."""

    return metric in METRIC_REGISTRY.get(task, ())
