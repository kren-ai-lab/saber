from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from mlcore import MODEL_REGISTRY
from mlcore.datasets import DatasetBundle, PartitionPlan
from mlcore.datasets.biosieve import BioSievePartitionConfig
from mlcore.exceptions import ValidationContractError
from mlcore.preprocessing import PreprocessingConfig
from mlcore.validation import ValidationEngine


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
        dataset_fingerprint=dataset.fingerprint,
    )
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        preprocessing=PreprocessingConfig(imputation=None, scaler="standard"),
        evaluation_role="test",
        random_state=42,
    )
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
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        preprocessing=PreprocessingConfig(imputation="median", scaler=None),
        evaluation_role="test",
        random_state=42,
    )
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
        dataset_fingerprint=dataset.fingerprint,
    )
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        random_state=5,
    )

    assert result.n_splits == 5
    assert result.oof_prediction is not None
    assert result.oof_prediction.n_samples == dataset.n_samples
    assert list(result.oof_prediction.sample_ids) == ids
    assert result.metadata["oof_complete"] is True
    assert result.metadata["oof_coverage"] == 1.0
    assert "accuracy" in result.aggregate_metrics


def test_regression_validation_returns_fold_and_oof_metrics():
    X, y = make_regression(n_samples=50, n_features=5, noise=0.5, random_state=4)
    ids = [f"r{i}" for i in range(50)]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=np.arange(50) % 5,
    )
    result = ValidationEngine(MODEL_REGISTRY).run(
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
        ValidationEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
        )


def test_partition_plan_and_biosieve_configuration_are_mutually_exclusive():
    X, y = make_classification(n_samples=30, n_features=5, random_state=2)
    dataset = DatasetBundle(X=X, y=y)
    plan = PartitionPlan.holdout(
        train_ids=range(20),
        test_ids=range(20, 30),
    )
    with pytest.raises(ValidationContractError, match="not both"):
        ValidationEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="logistic_regression",
            partition_plan=plan,
            partitioning=BioSievePartitionConfig(strategy="random"),
        )


def test_sample_weights_are_forwarded_when_supported():
    X, y = make_classification(n_samples=40, n_features=6, random_state=7)
    dataset = DatasetBundle(
        X=X,
        y=y,
        sample_ids=[f"s{i}" for i in range(40)],
        sample_weight=np.linspace(1.0, 2.0, 40),
    )
    plan = PartitionPlan.holdout(
        train_ids=dataset.sample_ids[:30],
        test_ids=dataset.sample_ids[30:],
    )
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        evaluation_role="test",
    )
    assert result.folds[0].estimator is not None


def test_sample_weights_fail_for_estimator_without_weight_support():
    X, y = make_classification(n_samples=40, n_features=6, random_state=7)
    dataset = DatasetBundle(
        X=X,
        y=y,
        sample_ids=[f"s{i}" for i in range(40)],
        sample_weight=np.ones(40),
    )
    plan = PartitionPlan.holdout(
        train_ids=dataset.sample_ids[:30],
        test_ids=dataset.sample_ids[30:],
    )
    with pytest.raises(ValidationContractError, match="sample-weight support"):
        ValidationEngine(MODEL_REGISTRY).run(
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
    result = ValidationEngine(MODEL_REGISTRY).run(
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
    from mlcore.datasets import PartitionSplit

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

    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        random_state=3,
    )
    assert all(fold.evaluation_role == "test" for fold in result.folds)
    assert result.oof_prediction is not None
    assert result.metadata["oof_complete"] is True
