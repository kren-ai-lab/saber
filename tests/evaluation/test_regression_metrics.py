"""Unit tests for regression evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from saber.evaluation.regression import REGRESSION_METRICS, evaluate_regression


def test_each_regression_metric_name_maps_to_its_own_computation() -> None:
    # Non-perfect predictions with distinct errors, so swapping any two metric
    # implementations (e.g. MAE and MSE) changes the result.
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.0, 3.0, 5.0])

    results = evaluate_regression(y_true=y_true, y_pred=y_pred)

    assert set(results) == set(REGRESSION_METRICS)
    assert all(isinstance(value, float) for value in results.values())
    assert results["mae"] == pytest.approx(0.375)
    assert results["median_ae"] == pytest.approx(0.25)
    assert results["mape"] == pytest.approx(0.1875)
    assert results["mse"] == pytest.approx(0.3125)
    assert results["rmse"] == pytest.approx(np.sqrt(0.3125))
    assert results["r2"] == pytest.approx(1.0 - 1.25 / 5.0)
