"""
mlcore.evaluation.regression
============================

Regression evaluation metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
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
) -> dict[str, float]:
    """
    Evaluate regression predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.

    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict[str, float]
        Dictionary containing regression metrics.
    """

    metrics: dict[str, float] = {}

    metrics["mae"] = float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

    metrics["median_ae"] = float(
        median_absolute_error(
            y_true,
            y_pred,
        )
    )

    metrics["mse"] = float(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    metrics["rmse"] = float(
        np.sqrt(
            metrics["mse"],
        )
    )

    metrics["mape"] = float(
        mean_absolute_percentage_error(
            y_true,
            y_pred,
        )
    )

    metrics["r2"] = float(
        r2_score(
            y_true,
            y_pred,
        )
    )

    metrics["explained_variance"] = float(
        explained_variance_score(
            y_true,
            y_pred,
        )
    )

    pearson_value, _ = pearsonr(
        y_true,
        y_pred,
    )

    metrics["pearson"] = float(
        pearson_value,
    )

    spearman_value, _ = spearmanr(
        y_true,
        y_pred,
    )

    metrics["spearman"] = float(
        spearman_value,
    )

    return metrics


def metric_names() -> tuple[str, ...]:
    """
    Return available metric names.
    """

    return REGRESSION_METRICS