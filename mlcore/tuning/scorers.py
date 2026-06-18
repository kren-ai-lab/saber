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
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
    make_scorer,
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
        mean_squared_error,
        squared=False,
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
        Scikit-learn scorer.
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