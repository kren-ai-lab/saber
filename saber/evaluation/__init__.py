"""Evaluation helpers and structured results."""

from saber.evaluation.classification import (
    evaluate_binary_classification,
    evaluate_classification,
    evaluate_multiclass_classification,
)
from saber.evaluation.evaluator import evaluate_prediction
from saber.evaluation.regression import REGRESSION_METRICS, evaluate_regression
from saber.evaluation.results import EvaluationResult

__all__ = [
    "REGRESSION_METRICS",
    "EvaluationResult",
    "evaluate_binary_classification",
    "evaluate_classification",
    "evaluate_multiclass_classification",
    "evaluate_prediction",
    "evaluate_regression",
]
