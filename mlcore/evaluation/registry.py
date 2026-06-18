"""
mlcore.evaluation.registry
==========================

Metric registry for all evaluation modules.
"""

from __future__ import annotations

from mlcore.evaluation.classification import (
    CLASSIFICATION_METRICS,
)

from mlcore.evaluation.regression import (
    REGRESSION_METRICS,
)


METRIC_REGISTRY: dict[str, tuple[str, ...]] = {
    "classification": CLASSIFICATION_METRICS,
    "regression": REGRESSION_METRICS,
}


def available_tasks() -> tuple[str, ...]:
    """
    Return supported evaluation tasks.
    """

    return tuple(
        METRIC_REGISTRY.keys(),
    )


def get_metrics(
    task: str,
) -> tuple[str, ...]:
    """
    Return metrics for a given task.

    Parameters
    ----------
    task : str
        Task name.

    Returns
    -------
    tuple[str, ...]
        Available metrics.
    """

    return METRIC_REGISTRY[task]


def has_metric(
    task: str,
    metric: str,
) -> bool:
    """
    Check whether a metric exists.

    Parameters
    ----------
    task : str
        Task name.

    metric : str
        Metric name.

    Returns
    -------
    bool
    """

    return metric in METRIC_REGISTRY.get(
        task,
        (),
    )