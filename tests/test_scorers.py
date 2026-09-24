"""
tests.test_scorers
==================

Tests for tuning scorers.
"""

from __future__ import annotations

import pytest

from saber.tuning.scorers import (
    CLASSIFICATION_SCORERS,
    REGRESSION_SCORERS,
    SCORERS,
    get_scorer,
    list_scorers,
    is_classification_scorer,
    is_regression_scorer,
)


# ============================================================
# Registry
# ============================================================

def test_classification_scorers_exist() -> None:

    assert "accuracy" in CLASSIFICATION_SCORERS

    assert "precision" in CLASSIFICATION_SCORERS

    assert "recall" in CLASSIFICATION_SCORERS

    assert "f1" in CLASSIFICATION_SCORERS

    assert "mcc" in CLASSIFICATION_SCORERS


def test_regression_scorers_exist() -> None:

    assert "mae" in REGRESSION_SCORERS

    assert "mse" in REGRESSION_SCORERS

    assert "rmse" in REGRESSION_SCORERS

    assert "r2" in REGRESSION_SCORERS


def test_combined_registry_contains_all() -> None:

    assert "accuracy" in SCORERS

    assert "f1" in SCORERS

    assert "mae" in SCORERS

    assert "r2" in SCORERS


# ============================================================
# get_scorer
# ============================================================

def test_get_classification_scorer() -> None:

    scorer = get_scorer(
        "accuracy",
    )

    assert scorer is not None


def test_get_regression_scorer() -> None:

    scorer = get_scorer(
        "r2",
    )

    assert scorer is not None


def test_unknown_scorer_raises() -> None:

    with pytest.raises(
        ValueError,
    ):
        get_scorer(
            "unknown_score",
        )


# ============================================================
# Listing
# ============================================================

def test_list_scorers_returns_list() -> None:

    scorers = list_scorers()

    assert isinstance(
        scorers,
        list,
    )

    assert len(
        scorers,
    ) > 0


def test_list_scorers_contains_expected_metrics() -> None:

    scorers = list_scorers()

    assert "accuracy" in scorers

    assert "f1" in scorers

    assert "r2" in scorers

    assert "mae" in scorers


# ============================================================
# Classification helpers
# ============================================================

def test_is_classification_scorer() -> None:

    assert is_classification_scorer(
        "accuracy",
    )

    assert is_classification_scorer(
        "f1",
    )

    assert not is_classification_scorer(
        "r2",
    )


def test_is_regression_scorer() -> None:

    assert is_regression_scorer(
        "r2",
    )

    assert is_regression_scorer(
        "mae",
    )

    assert not is_regression_scorer(
        "accuracy",
    )


# ============================================================
# Scorer objects
# ============================================================

def test_accuracy_returns_sklearn_scorer() -> None:

    scorer = get_scorer(
        "accuracy",
    )

    assert callable(
        scorer,
    )


def test_r2_returns_sklearn_scorer() -> None:

    scorer = get_scorer(
        "r2",
    )

    assert callable(
        scorer,
    )