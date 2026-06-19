"""
mlcore.tuning.scorers
=====================

Scoring utilities used by optimization methods.

This module provides a unified interface for retrieving
scikit-learn compatible scorers.
"""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# Classification scorers
# ============================================================

CLASSIFICATION_SCORERS = {
    "accuracy": make_scorer(
        accuracy_score,
    ),
    "balanced_accuracy": make_scorer(
        balanced_accuracy_score,
    ),
    "precision": make_scorer(
        precision_score,
        average="weighted",
        zero_division=0,
    ),
    "recall": make_scorer(
        recall_score,
        average="weighted",
        zero_division=0,
    ),
    "f1": make_scorer(
        f1_score,
        average="weighted",
        zero_division=0,
    ),
    "mcc": make_scorer(
        matthews_corrcoef,
    ),
    "roc_auc": make_scorer(
        roc_auc_score,
        needs_proba=True,
    ),
}

# ============================================================
# Regression scorers
# ============================================================

REGRESSION_SCORERS = {
    "mae": make_scorer(
        mean_absolute_error,
        greater_is_better=False,
    ),
    "mse": make_scorer(
        mean_squared_error,
        greater_is_better=False,
    ),
    "rmse": make_scorer(
        root_mean_squared_error,
        greater_is_better=False,
    ),
    "median_ae": make_scorer(
        median_absolute_error,
        greater_is_better=False,
    ),
    "r2": make_scorer(
        r2_score,
    ),
    "explained_variance": make_scorer(
        explained_variance_score,
    ),
}

# ============================================================
# Regression loss metrics
# ============================================================

LOSS_SCORERS = {
    "mae",
    "mse",
    "rmse",
    "median_ae",
}

# ============================================================
# Combined registry
# ============================================================

SCORERS = {
    **CLASSIFICATION_SCORERS,
    **REGRESSION_SCORERS,
}

# ============================================================
# Public API
# ============================================================

def get_scorer(
    name: str,
):
    """
    Retrieve a scorer by name.

    Parameters
    ----------
    name : str
        Scorer identifier.

    Returns
    -------
    callable
        Scikit-learn compatible scorer.
    """

    if name not in SCORERS:

        raise ValueError(
            f"Unknown scorer: {name}"
        )

    return SCORERS[name]


def list_scorers() -> list[str]:
    """
    List available scorers.
    """

    return sorted(
        SCORERS.keys()
    )


def is_classification_scorer(
    name: str,
) -> bool:
    """
    Check whether scorer belongs
    to classification.
    """

    return (
        name
        in CLASSIFICATION_SCORERS
    )


def is_regression_scorer(
    name: str,
) -> bool:
    """
    Check whether scorer belongs
    to regression.
    """

    return (
        name
        in REGRESSION_SCORERS
    )


def is_loss_scorer(
    name: str,
) -> bool:
    """
    Check whether scorer represents
    a loss/error metric.

    Notes
    -----
    Loss scorers are internally transformed
    by sklearn into maximization objectives
    when ``greater_is_better=False`` is used.
    """

    return (
        name
        in LOSS_SCORERS
    )


def is_gain_scorer(
    name: str,
) -> bool:
    """
    Check whether scorer represents
    a metric where larger values are better.
    """

    return (
        name
        in SCORERS
        and name not in LOSS_SCORERS
    )

def normalize_score(
    metric: str,
    score: float,
) -> float:
    """
    Convert internal optimization score into user-facing score.
    """

    if is_loss_scorer(metric):
        return abs(score)

    return score