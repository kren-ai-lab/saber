from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from saber.core.metrics import get_metric_spec, validate_metric
from saber.exceptions import MetricProblemTypeError, MetricTaskMismatchError
from saber.tuning.scorers import get_scorer, normalize_score


def test_roc_auc_uses_modern_response_method() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        random_state=42,
    )
    scores = cross_val_score(
        LogisticRegression(max_iter=1000),
        X,
        y,
        scoring=get_scorer("roc_auc", task="classification", y=y),
        cv=3,
    )
    assert np.all(np.isfinite(scores))


def test_metric_task_mismatch_fails_before_search() -> None:
    with pytest.raises(MetricTaskMismatchError):
        validate_metric("rmse", task="classification", y=np.array([0, 1]))


def test_binary_only_roc_auc_rejects_multiclass() -> None:
    with pytest.raises(MetricProblemTypeError):
        validate_metric(
            "roc_auc",
            task="classification",
            y=np.array([0, 1, 2, 0, 1, 2]),
        )


def test_loss_metric_has_natural_score_conversion() -> None:
    spec = get_metric_spec("rmse")
    assert spec.is_loss
    assert spec.optimization_direction == "maximize"
    assert normalize_score("rmse", -2.5) == 2.5
