"""
mlcore.evaluation.classification
================================

Classification evaluation utilities.

This module provides:
- Binary classification metrics
- Multiclass classification metrics
- Unified evaluation interface
"""

from __future__ import annotations

from typing import Callable
from typing import Optional
from typing import Sequence

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def specificity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute binary classification specificity.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    float
        Specificity score.
    """

    tn, fp, _, _ = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    denominator = tn + fp

    if denominator == 0:
        return 0.0

    return float(tn / denominator)


def sensitivity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute binary classification sensitivity.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Predicted labels.

    Returns
    -------
    float
        Sensitivity score.
    """

    return float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    )


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Evaluate binary classification predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Predicted labels.

    y_proba : np.ndarray, optional
        Positive-class probabilities.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """

    results = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            ),
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        ),
        "sensitivity": float(
            sensitivity_score(
                y_true,
                y_pred,
            ),
        ),
        "specificity": float(
            specificity_score(
                y_true,
                y_pred,
            ),
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            ),
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            ),
        ),
    }

    if y_proba is not None:

        results["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_proba,
            ),
        )

        results["pr_auc"] = float(
            average_precision_score(
                y_true,
                y_proba,
            ),
        )

        results["log_loss"] = float(
            log_loss(
                y_true,
                y_proba,
            ),
        )

        results["brier_score"] = float(
            brier_score_loss(
                y_true,
                y_proba,
            ),
        )

    return results


def evaluate_multiclass_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Evaluate multiclass classification predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Predicted labels.

    y_proba : np.ndarray, optional
        Class probabilities.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """

    results = {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            ),
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),
        ),
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
        ),
        "precision_micro": float(
            precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            ),
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
        ),
        "recall_micro": float(
            recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            ),
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
        ),
        "f1_micro": float(
            f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            ),
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
        ),
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            ),
        ),
    }

    if y_proba is not None:

        results["log_loss"] = float(
            log_loss(
                y_true,
                y_proba,
            ),
        )

    return results


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    metrics: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """
    Unified classification evaluation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.

    y_pred : np.ndarray
        Predicted labels.

    y_proba : np.ndarray, optional
        Predicted probabilities.

    metrics : Sequence[str], optional
        Subset of metrics to return.

    Returns
    -------
    dict[str, float]
        Evaluation metrics.
    """

    n_classes = np.unique(y_true).shape[0]

    if n_classes == 2:

        results = evaluate_binary_classification(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    else:

        results = evaluate_multiclass_classification(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

    if metrics is None:
        return results

    return {
        metric: results[metric]
        for metric in metrics
        if metric in results
    }


CLASSIFICATION_METRICS: dict[str, Callable[..., float]] = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "mcc": matthews_corrcoef,
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "log_loss": log_loss,
    "brier_score": brier_score_loss,
    "specificity": specificity_score,
    "sensitivity": sensitivity_score,
}