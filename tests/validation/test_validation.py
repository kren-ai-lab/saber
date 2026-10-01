from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression

from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import ValidationContractError
from saber.preprocessing import PreprocessingConfig
from saber.validation import validate
from saber.validation.results import aggregate_fold_metrics


def test_validation_fits_scaler_only_on_training_partition():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [1000.0, 1000.0],
            [2000.0, 2000.0],
        ]
    )
    dataset = DatasetBundle(
        X=X,
        y=[0, 1, 0, 1, 0, 1],
        sample_ids=list("abcdef"),
    )
    plan = PartitionPlan.holdout(
        train_ids=list("abcd"),
        test_ids=list("ef"),
        dataset=dataset,
    )
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        preprocessing=PreprocessingConfig(imputation=None, scaler="standard"),
        evaluation_role="test",
        random_state=42,
        return_estimators=True,
    )
    assert result.folds[0].estimator is not None
    scaler = result.folds[0].estimator.named_steps["scaler"]
    np.testing.assert_allclose(scaler.mean_, np.array([1.5, 1.5]))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))


def test_validation_fits_imputer_only_on_training_partition():
    X = np.array(
        [
            [np.nan, 0.0],
            [1.0, 1.0],
            [3.0, 2.0],
            [5.0, 3.0],
            [1000.0, 1000.0],
            [2000.0, 2000.0],
        ]
    )
    dataset = DatasetBundle(
        X=X,
        y=[0, 1, 0, 1, 0, 1],
        sample_ids=list("abcdef"),
    )
    plan = PartitionPlan.holdout(
        train_ids=list("abcd"),
        test_ids=list("ef"),
    )
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        preprocessing=PreprocessingConfig(imputation="median", scaler=None),
        evaluation_role="test",
        random_state=42,
        return_estimators=True,
    )
    assert result.folds[0].estimator is not None
    imputer = result.folds[0].estimator.named_steps["imputer"]
    np.testing.assert_allclose(imputer.statistics_, np.array([3.0, 1.5]))


def test_predefined_folds_produce_complete_identity_preserving_oof_predictions():
    X, y = make_classification(
        n_samples=60,
        n_features=8,
        n_informative=5,
        random_state=12,
    )
    ids = [f"sample_{i}" for i in range(60)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    assignments = np.arange(60) % 5
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=assignments,
        dataset=dataset,
    )
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        random_state=5,
    )

    assert result.n_splits == 5
    assert result.oof_prediction is not None
    assert result.oof_prediction.n_samples == dataset.n_samples
    assert result.oof_prediction.sample_ids is not None
    assert list(result.oof_prediction.sample_ids) == ids
    assert "accuracy" in result.aggregate_metrics


def test_regression_validation_returns_fold_and_oof_metrics():
    X, y = make_regression(  # pyrefly: ignore[bad-unpacking]
        n_samples=50,
        n_features=5,
        noise=0.5,
        random_state=4,
    )
    ids = [f"r{i}" for i in range(50)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=np.arange(50) % 5,
    )
    result = validate(
        dataset=dataset,
        algorithm="ridge_regressor",
        partition_plan=plan,
    )
    assert result.oof_prediction is not None
    assert result.oof_prediction.task == "regression"
    assert "rmse" in result.aggregate_metrics
    assert len(result.folds) == 5


def test_unpartitioned_data_require_explicit_biosieve_configuration():
    X, y = make_classification(n_samples=30, n_features=5, random_state=2)
    dataset = DatasetBundle(X=X, y=y)
    with pytest.raises(ValidationContractError, match="BioSievePartitionConfig"):
        validate(
            dataset=dataset,
            algorithm="logistic_regression",
        )


def test_sample_weights_are_forwarded_aligned_with_the_training_fold():
    X, y = make_classification(n_samples=40, n_features=6, random_state=7)
    weights = np.linspace(0.05, 5.0, 40)
    ids = [f"s{i}" for i in range(40)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, sample_weight=weights)
    # Train on a non-prefix block so a misaligned weight slice would be detected.
    plan = PartitionPlan.holdout(train_ids=ids[10:], test_ids=ids[:10])
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        evaluation_role="test",
        return_estimators=True,
    )
    fitted = result.folds[0].estimator
    assert fitted is not None
    weighted = clone(fitted).fit(X[10:], y[10:], estimator__sample_weight=weights[10:])
    unweighted = clone(fitted).fit(X[10:], y[10:])
    coef = fitted.named_steps["estimator"].coef_
    np.testing.assert_allclose(coef, weighted.named_steps["estimator"].coef_)
    assert not np.allclose(coef, unweighted.named_steps["estimator"].coef_)


def test_sample_weights_fail_for_estimator_without_weight_support():
    X, y = make_classification(n_samples=40, n_features=6, random_state=7)
    dataset = DatasetBundle(
        X=X,
        y=y,
        sample_ids=[f"s{i}" for i in range(40)],
        sample_weight=np.ones(40),
    )
    assert dataset.sample_ids is not None
    plan = PartitionPlan.holdout(
        train_ids=dataset.sample_ids[:30],
        test_ids=dataset.sample_ids[30:],
    )
    with pytest.raises(ValidationContractError, match="sample-weight support"):
        validate(
            dataset=dataset,
            algorithm="knn_classifier",
            partition_plan=plan,
            evaluation_role="test",
        )


def test_holdout_auto_role_prefers_validation_over_final_test():
    X, y = make_classification(n_samples=50, n_features=6, random_state=10)
    ids = [f"s{i}" for i in range(50)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.holdout(
        train_ids=ids[:30],
        validation_ids=ids[30:40],
        test_ids=ids[40:],
    )
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        random_state=3,
    )
    assert result.folds[0].evaluation_role == "validation"
    assert result.folds[0].evaluation_ids == tuple(ids[30:40])


def test_cross_validation_auto_role_uses_biosieve_style_test_fold():
    X, y = make_classification(n_samples=30, n_features=6, random_state=11)
    ids = [f"s{i}" for i in range(30)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    splits = []
    from saber.datasets import PartitionSplit

    for fold in range(3):
        test_ids = tuple(ids[fold * 10 : (fold + 1) * 10])
        train_ids = tuple(sample_id for sample_id in ids if sample_id not in test_ids)
        splits.append(
            PartitionSplit(
                name=f"fold_{fold}",
                train_ids=train_ids,
                test_ids=test_ids,
                metadata={"source": "biosieve"},
            )
        )
    plan = PartitionPlan(splits=tuple(splits), kind="cross_validation")

    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        random_state=3,
    )
    assert all(fold.evaluation_role == "test" for fold in result.folds)
    assert result.oof_prediction is not None


def _fold(metrics):
    return SimpleNamespace(metrics=metrics)


def test_fold_aggregation_skips_non_finite_values_and_uses_sample_std():
    folds = (
        _fold({"accuracy": 0.6, "roc_auc": float("nan")}),
        _fold({"accuracy": 0.8, "roc_auc": 0.9}),
        _fold({"accuracy": 1.0}),
    )
    summary = aggregate_fold_metrics(folds)  # pyrefly: ignore[bad-argument-type]
    assert summary["accuracy"] == pytest.approx({"mean": 0.8, "std": 0.2, "min": 0.6, "max": 1.0, "n": 3})
    assert summary["roc_auc"] == pytest.approx({"mean": 0.9, "std": 0.0, "min": 0.9, "max": 0.9, "n": 1})
    assert type(summary["accuracy"]["n"]) is int


def test_oof_prediction_carries_probabilities_decision_scores_and_classes():
    X, y = make_classification(n_samples=60, n_features=6, random_state=3)
    ids = [f"s{i}" for i in range(60)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=np.arange(60) % 3,
        dataset=dataset,
    )
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=("accuracy",),
        random_state=3,
    )
    oof = result.oof_prediction
    assert oof is not None
    assert oof.probabilities is not None
    assert oof.probabilities.shape == (60, 2)
    assert oof.decision_scores is not None
    np.testing.assert_array_equal(oof.classes, [0, 1])


def test_metrics_frame_is_long_form_with_aggregate_and_fold_rows():
    X, y = make_classification(n_samples=60, n_features=6, random_state=3)
    ids = [f"s{i}" for i in range(60)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids, fold_assignments=np.arange(60) % 3, dataset=dataset
    )
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=("accuracy", "mcc"),
        random_state=3,
    )
    frame = result.metrics_frame()
    assert frame.columns == [
        "level",
        "split",
        "evaluation_role",
        "metric",
        "score",
        "std",
        "min",
        "max",
        "n",
        "fit_seconds",
    ]
    aggregate = frame.filter(pl.col("level") == "aggregate")
    assert dict(zip(aggregate["metric"], aggregate["score"], strict=True)) == pytest.approx(
        result.aggregate_metrics
    )
    assert aggregate["n"].to_list() == [3, 3]
    folds = frame.filter(pl.col("level") == "fold")
    assert folds["split"].to_list() == [fold.split for fold in result.folds for _ in range(2)]
    assert folds["score"].to_list() == [score for fold in result.folds for score in fold.metrics.values()]
    assert result.to_dict()["metrics"] == result.aggregate_metrics
