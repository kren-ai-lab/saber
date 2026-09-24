"""
tests.test_classification_metrics
=================================

Unit tests for classification evaluation metrics.
"""

from __future__ import annotations

import numpy as np

from saber.evaluation.classification import (
    evaluate_binary_classification,
    evaluate_multiclass_classification,
    evaluate_classification,
    sensitivity_score,
    specificity_score,
)


# ============================================================
# Binary classification
# ============================================================

def test_binary_classification_metrics() -> None:
    """
    Binary evaluation should return expected metrics.
    """

    y_true = np.array(
        [0, 0, 1, 1],
    )

    y_pred = np.array(
        [0, 1, 1, 1],
    )

    results = evaluate_binary_classification(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "specificity" in results
    assert "sensitivity" in results

    assert results["accuracy"] == 0.75


# ============================================================
# Multiclass classification
# ============================================================

def test_multiclass_classification_metrics() -> None:
    """
    Multiclass evaluation should return expected metrics.
    """

    y_true = np.array(
        [0, 1, 2, 0, 1, 2],
    )

    y_pred = np.array(
        [0, 1, 2, 0, 2, 1],
    )

    results = evaluate_multiclass_classification(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert "accuracy" in results
    assert "precision_macro" in results
    assert "recall_macro" in results
    assert "f1_macro" in results

    assert results["accuracy"] > 0.0


# ============================================================
# Automatic dispatch
# ============================================================

def test_auto_binary_dispatch() -> None:
    """
    Automatic evaluation should detect binary problems.
    """

    y_true = np.array(
        [0, 0, 1, 1],
    )

    y_pred = np.array(
        [0, 0, 1, 0],
    )

    results = evaluate_classification(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert "specificity" in results
    assert "sensitivity" in results


def test_auto_multiclass_dispatch() -> None:
    """
    Automatic evaluation should detect multiclass problems.
    """

    y_true = np.array(
        [0, 1, 2, 0, 1, 2],
    )

    y_pred = np.array(
        [0, 1, 1, 0, 2, 2],
    )

    results = evaluate_classification(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert "f1_macro" in results


# ============================================================
# Sensitivity
# ============================================================

def test_sensitivity_score() -> None:
    """
    Sensitivity should match expected value.
    """

    y_true = np.array(
        [0, 0, 1, 1],
    )

    y_pred = np.array(
        [0, 1, 1, 1],
    )

    score = sensitivity_score(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert score == 1.0


# ============================================================
# Specificity
# ============================================================

def test_specificity_score() -> None:
    """
    Specificity should match expected value.
    """

    y_true = np.array(
        [0, 0, 1, 1],
    )

    y_pred = np.array(
        [0, 1, 1, 1],
    )

    score = specificity_score(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert score == 0.5


# ============================================================
# Perfect predictions
# ============================================================

def test_perfect_binary_predictions() -> None:
    """
    Perfect predictions should yield perfect metrics.
    """

    y_true = np.array(
        [0, 0, 1, 1],
    )

    y_pred = np.array(
        [0, 0, 1, 1],
    )

    results = evaluate_binary_classification(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert results["accuracy"] == 1.0
    assert results["precision"] == 1.0
    assert results["recall"] == 1.0
    assert results["f1"] == 1.0
    assert results["specificity"] == 1.0
    assert results["sensitivity"] == 1.0


# ============================================================
# Shape consistency
# ============================================================

def test_metric_outputs_are_numeric() -> None:
    """
    All returned metrics should be numeric.
    """

    y_true = np.array(
        [0, 0, 1, 1],
    )

    y_pred = np.array(
        [0, 1, 1, 1],
    )

    results = evaluate_binary_classification(
        y_true=y_true,
        y_pred=y_pred,
    )

    for value in results.values():

        assert isinstance(
            value,
            float,
        )

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
