"""
tests.test_regression_metrics
=============================

Unit tests for regression evaluation metrics.
"""

from __future__ import annotations

import numpy as np

from saber.evaluation.regression import (
    REGRESSION_METRICS,
    evaluate_regression,
    metric_names,
)


def test_regression_metric_names() -> None:
    names = metric_names()

    assert isinstance(names, tuple)
    assert "mae" in names
    assert "rmse" in names
    assert "r2" in names
    assert names == REGRESSION_METRICS


def test_regression_metrics_output_keys() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    results = evaluate_regression(
        y_true=y_true,
        y_pred=y_pred,
    )

    for metric in REGRESSION_METRICS:
        assert metric in results


def test_regression_metrics_are_floats() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])

    results = evaluate_regression(
        y_true=y_true,
        y_pred=y_pred,
    )

    for value in results.values():
        assert isinstance(value, float)


def test_perfect_regression_predictions() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    results = evaluate_regression(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert results["mae"] == 0.0
    assert results["mse"] == 0.0
    assert results["rmse"] == 0.0
    assert results["r2"] == 1.0
    assert results["explained_variance"] == 1.0
    assert results["pearson"] == 1.0
    assert results["spearman"] == 1.0


def test_rmse_is_square_root_of_mse() -> None:
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.2, 1.8, 3.1, 4.1])

    results = evaluate_regression(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert np.isclose(
        results["rmse"],
        np.sqrt(results["mse"]),
    )