from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from saber import MODEL_REGISTRY
from saber.core.prediction import PredictionResult
from saber.datasets import DatasetBundle, PartitionPlan, PartitionSplit
from saber.evaluation import evaluate_prediction
from saber.exceptions import PredictionContractError, ValidationContractError
from saber.validation import ValidationEngine


def test_multiclass_prediction_is_not_misclassified_as_binary_when_fold_lacks_one_class():
    prediction = PredictionResult(
        task="classification",
        predictions=np.array([0, 1, 1, 0]),
        probabilities=np.array(
            [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
                [0.7, 0.2, 0.1],
            ]
        ),
        classes=np.array([0, 1, 2]),
    )
    result = evaluate_prediction(
        np.array([0, 1, 1, 0]),
        prediction,
        metrics=("accuracy", "log_loss"),
    )
    assert set(result.metrics) == {"accuracy", "log_loss"}
    assert np.isfinite(result.metrics["log_loss"])


def test_binary_fold_with_single_observed_class_can_compute_accuracy_only():
    prediction = PredictionResult(
        task="classification",
        predictions=np.array([0, 0, 0, 0]),
        probabilities=np.array([[0.9, 0.1]] * 4),
        classes=np.array([0, 1]),
        positive_class=1,
    )
    result = evaluate_prediction(np.array([0, 0, 0, 0]), prediction, metrics=("accuracy",))
    assert result.metrics == {"accuracy": 1.0}


def test_binary_fold_with_single_observed_class_rejects_explicit_roc_auc():
    prediction = PredictionResult(
        task="classification",
        predictions=np.array([0, 0, 0, 0]),
        probabilities=np.array([[0.9, 0.1]] * 4),
        classes=np.array([0, 1]),
        positive_class=1,
    )
    with pytest.raises(ValueError, match=r"roc_auc|two classes|undefined"):
        evaluate_prediction(np.array([0, 0, 0, 0]), prediction, metrics=("roc_auc",))


@pytest.mark.parametrize("metric", ["totally_unknown", "rmse"])
def test_classification_evaluation_rejects_unknown_or_wrong_task_metric(metric):
    prediction = PredictionResult(
        task="classification",
        predictions=np.array([0, 1, 0, 1]),
        probabilities=np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]]),
        classes=np.array([0, 1]),
    )
    with pytest.raises(ValueError, match=r"metric|Unknown|available"):
        evaluate_prediction(np.array([0, 1, 0, 1]), prediction, metrics=(metric,))


def test_custom_positive_class_flips_decision_function_orientation():
    raw = np.array([-2.0, -0.5, 0.5, 2.0])
    prediction = PredictionResult(
        task="classification",
        predictions=np.array(["neg", "neg", "pos", "pos"]),
        decision_scores=raw,
        classes=np.array(["neg", "pos"]),
        positive_class="neg",
    )
    np.testing.assert_allclose(prediction.positive_decision_scores(), -raw)


def test_one_dimensional_probabilities_can_recover_negative_class_probability():
    p = np.array([0.1, 0.3, 0.8])
    prediction = PredictionResult(
        task="classification",
        predictions=np.array([0, 0, 1]),
        probabilities=p,
        classes=np.array([0, 1]),
        positive_class=1,
    )
    np.testing.assert_allclose(prediction.probabilities_for(0), 1.0 - p)


def test_probability_row_count_mismatch_is_rejected():
    with pytest.raises(PredictionContractError, match="rows"):
        PredictionResult(
            task="classification",
            predictions=np.array([0, 1, 0]),
            probabilities=np.array([[0.8, 0.2], [0.2, 0.8]]),
            classes=np.array([0, 1]),
        )


def test_probability_class_count_mismatch_is_rejected():
    with pytest.raises(PredictionContractError, match="columns"):
        PredictionResult(
            task="classification",
            predictions=np.array([0, 1]),
            probabilities=np.array([[0.4, 0.3, 0.3], [0.2, 0.6, 0.2]]),
            classes=np.array([0, 1]),
        )


def test_repeated_heldout_membership_disables_oof_instead_of_merging_predictions():
    X, y = make_classification(n_samples=30, n_features=5, random_state=12)
    ids = [f"s{i}" for i in range(30)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    split_a = PartitionSplit(
        name="a",
        train_ids=tuple(ids[10:]),
        test_ids=tuple(ids[:10]),
    )
    split_b = PartitionSplit(
        name="b",
        train_ids=tuple(ids[:10] + ids[20:]),
        test_ids=tuple(ids[10:20]),
    )
    split_c = PartitionSplit(
        name="c",
        train_ids=tuple(ids[5:20] + ids[25:]),
        test_ids=tuple(ids[:5] + ids[20:25]),
    )
    plan = PartitionPlan(splits=(split_a, split_b, split_c), kind="cross_validation")
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=("accuracy",),
        require_complete=False,
    )
    assert result.oof_prediction is None
    assert result.metadata["oof_available"] is False
    assert "repeat" in result.metadata["oof_reason"]


def test_incomplete_external_plan_reports_partial_oof_coverage_when_allowed():
    X, y = make_classification(n_samples=30, n_features=5, random_state=13)
    ids = [f"s{i}" for i in range(30)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.holdout(
        train_ids=ids[:15],
        test_ids=ids[15:20],
        dataset_fingerprint=dataset.fingerprint,
    )
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=("accuracy",),
        evaluation_role="test",
        require_complete=False,
    )
    assert result.oof_prediction is not None
    assert result.metadata["oof_complete"] is False
    assert result.metadata["oof_coverage"] == pytest.approx(5 / 30)


def test_multiclass_cv_with_string_labels_returns_complete_oof():
    X, y_int = make_classification(
        n_samples=75,
        n_features=8,
        n_informative=6,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=8,
    )
    labels = np.asarray(["alpha", "beta", "gamma"])[y_int]
    ids = [f"s{i}" for i in range(75)]
    dataset = DatasetBundle(X=X, y=labels, sample_ids=ids)
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=np.arange(75) % 3,
        dataset_fingerprint=dataset.fingerprint,
    )
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=("accuracy", "mcc"),
        random_state=2,
    )
    assert result.oof_prediction is not None
    assert result.oof_prediction.n_classes == 3
    assert result.oof_prediction.classes is not None
    assert set(result.oof_prediction.classes) == {"alpha", "beta", "gamma"}


def test_invalid_requested_evaluation_role_fails_before_fit():
    X, y = make_classification(n_samples=24, n_features=4, random_state=2)
    ids = [f"s{i}" for i in range(24)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.holdout(train_ids=ids[:18], test_ids=ids[18:])
    with pytest.raises(ValidationContractError, match="validation"):
        ValidationEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            partition_plan=plan,
            metrics=("accuracy",),
            evaluation_role="validation",
        )


def test_validation_rejects_training_fold_with_single_class_before_estimator_fit():
    X = np.arange(48, dtype=float).reshape(16, 3)
    y = np.array([0] * 8 + [1] * 8)
    ids = [f"s{i}" for i in range(16)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.holdout(
        train_ids=ids[:8],
        test_ids=ids[8:],
        dataset_fingerprint=dataset.fingerprint,
    )
    with pytest.raises(Exception, match=r"at least two|binary|multiclass"):
        ValidationEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            partition_plan=plan,
            metrics=("accuracy",),
            evaluation_role="test",
        )
