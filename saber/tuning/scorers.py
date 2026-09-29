"""saber.tuning.scorers: Compatibility scoring API backed by the canonical ``MetricSpec`` registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saber.core.metrics import (
    METRIC_SPECS,
    get_metric_spec,
    list_metric_specs,
    validate_metric,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

CLASSIFICATION_SCORERS = {spec.name: spec.make_scorer() for spec in list_metric_specs(task="classification")}

REGRESSION_SCORERS = {spec.name: spec.make_scorer() for spec in list_metric_specs(task="regression")}

SCORERS = {
    **CLASSIFICATION_SCORERS,
    **REGRESSION_SCORERS,
}

LOSS_SCORERS = {name for name, spec in METRIC_SPECS.items() if spec.is_loss}


def get_scorer(
    name: str,
    *,
    task: str | None = None,
    y: Sequence[Any] | np.ndarray | None = None,
) -> Any:
    """Retrieve a scorer, optionally validating task and target regime."""
    spec = get_metric_spec(name) if task is None else validate_metric(name, task=task, y=y)

    return spec.make_scorer()


def list_scorers() -> list[str]:
    """List available scorer identifiers."""
    return sorted(SCORERS)


def is_classification_scorer(name: str) -> bool:
    """Check whether a scorer belongs to classification."""
    return name in CLASSIFICATION_SCORERS


def is_regression_scorer(name: str) -> bool:
    """Check whether a scorer belongs to regression."""
    return name in REGRESSION_SCORERS


def is_loss_scorer(name: str) -> bool:
    """Check whether lower natural values are better."""
    return name in LOSS_SCORERS


def is_gain_scorer(name: str) -> bool:
    """Check whether larger natural values are better."""
    return name in SCORERS and name not in LOSS_SCORERS


def normalize_score(metric: str, score: float) -> float:
    """Convert an optimization score into the metric's natural representation."""
    return get_metric_spec(metric).to_natural_score(score)
