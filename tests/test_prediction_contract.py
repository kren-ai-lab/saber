from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from mlcore import MODEL_REGISTRY
from mlcore.core.prediction import PredictionResult
from mlcore.core.trainer import Trainer
from mlcore.evaluation import evaluate_prediction
from mlcore.exceptions import PredictionContractError


def test_binary_prediction_result_extracts_positive_probability() -> None:
    result = PredictionResult(
        task="classification",
        predictions=np.array(["neg", "pos", "pos"]),
        probabilities=np.array(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        ),
        classes=np.array(["neg", "pos"]),
        positive_class="pos",
    )
    np.testing.assert_allclose(
        result.positive_probabilities(),
        np.array([0.2, 0.8, 0.9]),
    )


def test_binary_positive_class_defaults_to_estimator_class_order() -> None:
    result = PredictionResult(
        task="classification",
        predictions=np.array([0, 1]),
        probabilities=np.array([[0.9, 0.1], [0.2, 0.8]]),
        classes=np.array([0, 1]),
    )
    assert result.positive_class == 1


def test_probability_matrix_requires_explicit_class_order() -> None:
    with pytest.raises(PredictionContractError):
        PredictionResult(
            task="classification",
            predictions=np.array([0, 1]),
            probabilities=np.array([[0.9, 0.1], [0.2, 0.8]]),
        )


def test_trainer_predict_result_is_directly_evaluable() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        random_state=42,
    )
    trainer = Trainer(MODEL_REGISTRY)
    trained = trainer.fit("logistic_regression", X, y, random_state=42)
    prediction = trainer.predict_result(trained, X)
    evaluation = evaluate_prediction(y, prediction)

    assert prediction.classes is not None
    assert prediction.probabilities is not None
    assert prediction.positive_probabilities().ndim == 1
    assert np.isfinite(evaluation.metrics["roc_auc"])
    assert np.isfinite(evaluation.metrics["brier_score"])


def test_decision_function_only_classifier_is_directly_evaluable() -> None:
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        random_state=7,
    )
    trainer = Trainer(MODEL_REGISTRY)
    trained = trainer.fit("linear_svc", X, y, random_state=7)
    prediction = trainer.predict_result(trained, X)
    evaluation = evaluate_prediction(y, prediction)

    assert prediction.probabilities is None
    assert prediction.decision_scores is not None
    assert prediction.positive_decision_scores().ndim == 1
    assert np.isfinite(evaluation.metrics["roc_auc"])
    assert np.isfinite(evaluation.metrics["pr_auc"])
    assert "brier_score" not in evaluation.metrics
