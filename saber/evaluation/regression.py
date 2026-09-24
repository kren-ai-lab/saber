"""saber.evaluation.regression
============================

Regression evaluation metrics.
"""

from __future__ import annotations

import numpy as np
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

REGRESSION_METRICS = (
    "mae",
    "median_ae",
    "mse",
    "rmse",
    "mape",
    "r2",
    "explained_variance",
    "pearson",
    "spearman",
)


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

    values: dict[str, float] = {}

    if "mae" in requested:
        values["mae"] = float(mean_absolute_error(y_true, y_pred))
    if "median_ae" in requested:
        values["median_ae"] = float(median_absolute_error(y_true, y_pred))
    if "mse" in requested:
        values["mse"] = float(mean_squared_error(y_true, y_pred))
    if "rmse" in requested:
        values["rmse"] = float(root_mean_squared_error(y_true, y_pred))
    if "mape" in requested:
        values["mape"] = float(mean_absolute_percentage_error(y_true, y_pred))
    if "r2" in requested:
        values["r2"] = float(r2_score(y_true, y_pred))
    if "explained_variance" in requested:
        values["explained_variance"] = float(explained_variance_score(y_true, y_pred))
    if "pearson" in requested:
        pearson_value, _ = pearsonr(y_true, y_pred)
        values["pearson"] = float(pearson_value)
    if "spearman" in requested:
        spearman_value, _ = spearmanr(y_true, y_pred)
        values["spearman"] = float(spearman_value)

    return values


def metric_names() -> tuple[str, ...]:
    """Return available metric names.
    """
    return REGRESSION_METRICS
