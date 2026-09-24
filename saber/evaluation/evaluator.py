"""saber.evaluation.evaluator
===========================

Evaluation entry point for structured prediction results.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from saber.core.prediction import PredictionResult
from saber.evaluation.classification import evaluate_classification
from saber.evaluation.regression import evaluate_regression
from saber.evaluation.results import EvaluationResult


def evaluate_prediction(
    y_true: np.ndarray,
    prediction: PredictionResult,
    *,
    metrics: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate a :class:`PredictionResult` using task-aware semantics."""
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError("y_true must be a one-dimensional array.")
    if y_true.shape[0] != prediction.n_samples:
        raise ValueError("y_true length must match the prediction result length.")

    if prediction.task == "classification":
        y_score = None
        if (
            prediction.probabilities is None
            and prediction.is_binary
            and prediction.decision_scores is not None
        ):
            y_score = prediction.positive_decision_scores()

        values = evaluate_classification(
            y_true=y_true,
            y_pred=prediction.predictions,
            y_proba=prediction.probabilities,
            metrics=metrics,
            y_score=y_score,
            classes=prediction.classes,
            positive_class=prediction.positive_class,
        )
    else:
        values = evaluate_regression(
            y_true=y_true,
            y_pred=prediction.predictions,
            metrics=None if metrics is None else tuple(metrics),
        )

    return EvaluationResult(
        task=prediction.task,
        metrics=values,
        prediction=prediction,
    )
