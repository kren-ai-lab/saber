"""
tests.test_optimization_result
==============================

Tests for OptimizationResult.
"""

from __future__ import annotations

from saber.tuning.results import OptimizationResult


def test_optimization_result_creation() -> None:
    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={
            "n_estimators": 100,
        },
        optimizer="grid_search",
        metric="f1",
    )

    assert result.algorithm == "random_forest"
    assert result.best_score == 0.91
    assert result.best_params == {
        "n_estimators": 100,
    }
    assert result.optimizer == "grid_search"
    assert result.metric == "f1"


def test_optimization_result_has_model_false_by_default() -> None:
    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={},
    )

    assert not result.has_model()


def test_optimization_result_has_model_true() -> None:
    model = object()

    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={},
        best_model=model,
    )

    assert result.has_model()


def test_optimization_result_has_history_false_by_default() -> None:
    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={},
    )

    assert not result.has_history()


def test_optimization_result_has_history_true() -> None:
    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={},
        history=[
            {
                "params": {
                    "n_estimators": 100,
                },
                "score": 0.9,
            },
        ],
    )

    assert result.has_history()


def test_optimization_result_has_study_false_by_default() -> None:
    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={},
    )

    assert not result.has_study()


def test_optimization_result_has_study_true() -> None:
    study = object()

    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={},
        study=study,
    )

    assert result.has_study()


def test_optimization_result_get_best_param() -> None:
    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={
            "n_estimators": 100,
            "max_depth": 5,
        },
    )

    assert result.get_best_param(
        "n_estimators",
    ) == 100

    assert result.get_best_param(
        "max_depth",
    ) == 5


def test_optimization_result_get_history() -> None:
    history = [
        {
            "params": {
                "n_estimators": 100,
            },
            "score": 0.9,
        },
    ]

    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={},
        history=history,
    )

    returned_history = result.get_history()

    assert returned_history == history
    assert returned_history is not history


def test_optimization_result_to_dict() -> None:
    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={
            "n_estimators": 100,
        },
        optimizer="grid_search",
        metric="f1",
    )

    data = result.to_dict()

    assert data["algorithm"] == "random_forest"
    assert data["optimizer"] == "grid_search"
    assert data["metric"] == "f1"
    assert data["best_score"] == 0.91
    assert data["best_params"] == {
        "n_estimators": 100,
    }


def test_optimization_result_summary() -> None:
    result = OptimizationResult(
        algorithm="random_forest",
        best_score=0.91,
        best_params={
            "n_estimators": 100,
        },
        optimizer="grid_search",
        metric="f1",
    )

    assert result.summary() == result.to_dict()