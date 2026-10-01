"""Regression evaluation metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)

if TYPE_CHECKING:
    import numpy as np

_METRIC_FUNCS = {
    "mae": mean_absolute_error,
    "median_ae": median_absolute_error,
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "mape": mean_absolute_percentage_error,
    "r2": r2_score,
    "explained_variance": explained_variance_score,
    "pearson": lambda y_true, y_pred: pearsonr(y_true, y_pred)[0],
    "spearman": lambda y_true, y_pred: spearmanr(y_true, y_pred)[0],
}
REGRESSION_METRICS = tuple(_METRIC_FUNCS)


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metrics: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """Evaluate requested regression metrics without computing unused statistics."""
    requested = REGRESSION_METRICS if metrics is None else tuple(dict.fromkeys(metrics))
    unknown = set(requested) - set(REGRESSION_METRICS)
    if unknown:
        raise ValueError(f"Unknown regression metrics: {sorted(unknown)!r}.")

    return {name: float(fn(y_true, y_pred)) for name, fn in _METRIC_FUNCS.items() if name in requested}
