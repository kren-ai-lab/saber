"""saber.evaluation.classification
================================

Classification evaluation utilities with explicit binary class semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

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


def _resolve_binary_classes(
    y_true: np.ndarray,
    *,
    y_pred: np.ndarray | None = None,
    classes: np.ndarray | Sequence[Any] | None = None,
    positive_class: Any | None = None,
) -> tuple[np.ndarray, Any, Any]:
    """Resolve binary class order and positive/negative labels deterministically."""
    y_true = np.asarray(y_true)

    if classes is None:
        if y_pred is None:
            resolved = np.unique(y_true)
        else:
            resolved = np.unique(np.concatenate([y_true, np.asarray(y_pred)]))
    else:
        resolved = np.asarray(classes)

    if resolved.ndim != 1 or resolved.size != 2:
        raise ValueError("Binary classification requires exactly two ordered classes.")

    if positive_class is None:
        positive_class = resolved[-1]

    matches = np.flatnonzero(resolved == positive_class)
    if matches.size != 1:
        raise ValueError(f"positive_class {positive_class!r} is not present in classes.")

    positive_index = int(matches[0])
    negative_index = 1 - positive_index
    negative_class = resolved[negative_index]

    return resolved, positive_class, negative_class


def _positive_probability_vector(
    y_proba: np.ndarray,
    *,
    classes: np.ndarray,
    positive_class: Any,
) -> np.ndarray:
    """Normalize binary probabilities to the positive-class one-dimensional form."""
    probabilities = np.asarray(y_proba)

    if probabilities.ndim == 1:
        return probabilities

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Binary probabilities must have shape (n_samples,) or (n_samples, 2).")

    matches = np.flatnonzero(classes == positive_class)
    if matches.size != 1:
        raise ValueError(f"positive_class {positive_class!r} is not present in classes.")

    return probabilities[:, int(matches[0])]


def specificity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_class: Any | None = None,
    classes: np.ndarray | Sequence[Any] | None = None,
) -> float:
    """Compute binary specificity for an explicit positive class."""
    resolved, positive_class, negative_class = _resolve_binary_classes(
        y_true,
        y_pred=y_pred,
        classes=classes,
        positive_class=positive_class,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[negative_class, positive_class],
    )

    tn, fp, _, _ = matrix.ravel()
    denominator = tn + fp
    if denominator == 0:
        return 0.0

    return float(tn / denominator)


def sensitivity_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    positive_class: Any | None = None,
    classes: np.ndarray | Sequence[Any] | None = None,
) -> float:
    """Compute binary sensitivity/recall for an explicit positive class."""
    _, positive_class, _ = _resolve_binary_classes(
        y_true,
        y_pred=y_pred,
        classes=classes,
        positive_class=positive_class,
    )

    return float(
        recall_score(
            y_true,
            y_pred,
            pos_label=positive_class,
            zero_division=0,
        )
    )


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    y_score: np.ndarray | None = None,
    *,
    classes: np.ndarray | Sequence[Any] | None = None,
    positive_class: Any | None = None,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Evaluate binary predictions with explicit class/probability semantics.

    When ``metrics`` is provided, every requested metric must be both valid and
    computable from the supplied prediction responses. With ``metrics=None``,
    saber returns every metric that is well-defined for the observed fold and
    available estimator responses; fold-local undefined ranking metrics are
    omitted rather than crashing unrelated evaluation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    resolved_classes, positive_class, _ = _resolve_binary_classes(
        y_true,
        y_pred=y_pred,
        classes=classes,
        positive_class=positive_class,
    )

    all_metrics = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "sensitivity",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier_score",
    )
    requested = all_metrics if metrics is None else tuple(dict.fromkeys(metrics))
    unknown = set(requested) - set(all_metrics)
    if unknown:
        raise ValueError(f"Unknown binary classification metrics: {sorted(unknown)!r}.")
    explicit = metrics is not None
    requested_set = set(requested)
    results: dict[str, float] = {}

    if "accuracy" in requested_set:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "balanced_accuracy" in requested_set:
        results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    if "precision" in requested_set:
        results["precision"] = float(
            precision_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
        )
    if "recall" in requested_set:
        results["recall"] = float(recall_score(y_true, y_pred, pos_label=positive_class, zero_division=0))
    if "sensitivity" in requested_set:
        results["sensitivity"] = float(
            sensitivity_score(
                y_true,
                y_pred,
                positive_class=positive_class,
                classes=resolved_classes,
            )
        )
    if "specificity" in requested_set:
        results["specificity"] = float(
            specificity_score(
                y_true,
                y_pred,
                positive_class=positive_class,
                classes=resolved_classes,
            )
        )
    if "f1" in requested_set:
        results["f1"] = float(f1_score(y_true, y_pred, pos_label=positive_class, zero_division=0))
    if "mcc" in requested_set:
        results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    y_true_binary = (y_true == positive_class).astype(int)
    ranking_score = None
    positive_proba = None

    if y_proba is not None:
        positive_proba = _positive_probability_vector(
            y_proba,
            classes=resolved_classes,
            positive_class=positive_class,
        )
        if positive_proba.shape[0] != y_true.shape[0]:
            raise ValueError("Probability rows must match y_true length.")
        ranking_score = positive_proba
    elif y_score is not None:
        ranking_score = np.asarray(y_score)
        if ranking_score.ndim != 1:
            raise ValueError("Binary decision scores must be one-dimensional.")
        if ranking_score.shape[0] != y_true.shape[0]:
            raise ValueError("Decision-score rows must match y_true length.")

    ranking_requested = requested_set & {"roc_auc", "pr_auc"}
    if ranking_requested:
        if ranking_score is None:
            if explicit:
                raise ValueError(
                    "Requested binary ranking metrics require predict_proba or decision_function responses."
                )
        elif np.unique(y_true_binary).size < 2:
            if explicit:
                raise ValueError(
                    "roc_auc/pr_auc are undefined when the evaluation fold "
                    "contains fewer than two observed classes."
                )
        else:
            if "roc_auc" in ranking_requested:
                results["roc_auc"] = float(roc_auc_score(y_true_binary, ranking_score))
            if "pr_auc" in ranking_requested:
                results["pr_auc"] = float(average_precision_score(y_true_binary, ranking_score))

    probability_requested = requested_set & {"log_loss", "brier_score"}
    if probability_requested:
        if positive_proba is None:
            if explicit:
                raise ValueError("Requested probability metrics require predict_proba responses.")
        else:
            positive_index = int(np.flatnonzero(resolved_classes == positive_class)[0])
            probability_matrix = np.empty((y_true.shape[0], 2), dtype=float)
            probability_matrix[:, positive_index] = positive_proba
            probability_matrix[:, 1 - positive_index] = 1.0 - positive_proba
            if "log_loss" in probability_requested:
                results["log_loss"] = float(
                    log_loss(y_true, probability_matrix, labels=list(resolved_classes))
                )
            if "brier_score" in probability_requested:
                results["brier_score"] = float(brier_score_loss(y_true_binary, positive_proba))

    return results


def evaluate_multiclass_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    *,
    classes: np.ndarray | Sequence[Any] | None = None,
    metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Evaluate multiclass predictions, computing only requested metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    all_metrics = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "precision_macro",
        "precision_micro",
        "precision_weighted",
        "recall",
        "recall_macro",
        "recall_micro",
        "recall_weighted",
        "f1",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "mcc",
        "log_loss",
    )
    requested = all_metrics if metrics is None else tuple(dict.fromkeys(metrics))
    unknown = set(requested) - set(all_metrics)
    if unknown:
        raise ValueError(f"Unknown multiclass classification metrics: {sorted(unknown)!r}.")
    explicit = metrics is not None
    requested_set = set(requested)
    results: dict[str, float] = {}

    if "accuracy" in requested_set:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "balanced_accuracy" in requested_set:
        results["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    # Canonical MetricSpec names use weighted averaging for multiclass
    # precision/recall/F1. Explicit *_macro/*_micro/*_weighted names remain
    # available for the lower-level evaluator.
    if "precision" in requested_set:
        results["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    if "recall" in requested_set:
        results["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    if "f1" in requested_set:
        results["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    for average in ("macro", "micro", "weighted"):
        key = f"precision_{average}"
        if key in requested_set:
            results[key] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
        key = f"recall_{average}"
        if key in requested_set:
            results[key] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
        key = f"f1_{average}"
        if key in requested_set:
            results[key] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    if "mcc" in requested_set:
        results["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    if "log_loss" in requested_set:
        if y_proba is None:
            if explicit:
                raise ValueError("Requested multiclass log_loss requires predict_proba responses.")
        else:
            probabilities = np.asarray(y_proba)
            if probabilities.ndim != 2:
                raise ValueError("Multiclass probabilities must be a two-dimensional matrix.")
            if probabilities.shape[0] != y_true.shape[0]:
                raise ValueError("Probability rows must match y_true length.")
            resolved_classes = (
                np.asarray(classes) if classes is not None else np.unique(np.concatenate([y_true, y_pred]))
            )
            if probabilities.shape[1] != resolved_classes.size:
                raise ValueError("Probability columns must match the number of classes.")
            results["log_loss"] = float(log_loss(y_true, probabilities, labels=list(resolved_classes)))

    return results


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    metrics: Sequence[str] | None = None,
    *,
    y_score: np.ndarray | None = None,
    classes: np.ndarray | Sequence[Any] | None = None,
    positive_class: Any | None = None,
) -> dict[str, float]:
    """Unified classification evaluation using fitted class semantics when available."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    resolved_classes = (
        np.asarray(classes) if classes is not None else np.unique(np.concatenate([y_true, y_pred]))
    )
    n_classes = resolved_classes.shape[0]

    if n_classes == 2:
        return evaluate_binary_classification(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            y_score=y_score,
            classes=resolved_classes,
            positive_class=positive_class,
            metrics=metrics,
        )
    if n_classes > 2:
        return evaluate_multiclass_classification(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            classes=resolved_classes,
            metrics=metrics,
        )
    raise ValueError("Classification evaluation requires fitted class semantics with at least two classes.")


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
