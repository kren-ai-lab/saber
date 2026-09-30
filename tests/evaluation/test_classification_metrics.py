"""Unit tests for classification evaluation metrics."""

from __future__ import annotations

import numpy as np

from saber.evaluation.classification import (
    evaluate_binary_classification,
    evaluate_classification,
    specificity_score,
)


def test_auto_dispatch_selects_binary_or_multiclass_metric_sets() -> None:
    binary = evaluate_classification(y_true=np.array([0, 0, 1, 1]), y_pred=np.array([0, 1, 1, 1]))
    assert {"specificity", "sensitivity"} <= set(binary)
    assert binary["accuracy"] == 0.75

    multiclass = evaluate_classification(
        y_true=np.array([0, 1, 2, 0, 1, 2]),
        y_pred=np.array([0, 1, 1, 0, 2, 2]),
    )
    assert "f1_macro" in multiclass
    assert "specificity" not in multiclass


def test_specificity_score() -> None:
    score = specificity_score(y_true=np.array([0, 0, 1, 1]), y_pred=np.array([0, 1, 1, 1]))
    assert score == 0.5


def test_binary_probability_matrix_uses_explicit_positive_class() -> None:
    y_true = np.array(["active", "inactive", "active", "inactive"])
    y_pred = np.array(["active", "inactive", "active", "inactive"])
    classes = np.array(["active", "inactive"])
    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
        ]
    )

    results = evaluate_binary_classification(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=probabilities,
        classes=classes,
        positive_class="active",
    )

    assert results["roc_auc"] == 1.0
    assert results["pr_auc"] == 1.0
    assert results["brier_score"] < 0.1
