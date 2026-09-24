from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from saber import MODEL_REGISTRY
from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import DatasetValidationError, PreprocessingContractError
from saber.preprocessing import PreprocessingConfig
from saber.validation import ValidationEngine


def _cv_plan(dataset: DatasetBundle, n_splits: int = 3) -> PartitionPlan:
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=np.arange(dataset.n_samples) % n_splits,
        dataset_fingerprint=dataset.fingerprint,
    )


def test_continuous_target_is_rejected_for_classification_before_fit():
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.linspace(0.01, 0.99, 20)
    dataset = DatasetBundle(X=X, y=y)
    with pytest.raises(DatasetValidationError, match="classification|target|continuous"):
        dataset.validate(task="classification")


@pytest.mark.parametrize(
    "labels, expected",
    [
        (["neg", "pos"] * 10, "binary"),
        ([10, 20] * 10, "binary"),
        ([False, True] * 10, "binary"),
        (["a", "b", "c", "a", "b"] * 4, "multiclass"),
        ([0, 1, 2, 0, 1] * 4, "multiclass"),
    ],
)
def test_supported_class_label_regimes_are_detected(labels, expected):
    X = np.arange(60, dtype=float).reshape(20, 3)
    dataset = DatasetBundle(X=X, y=labels)
    assert dataset.validate(task="classification") == expected


def test_single_class_target_is_rejected_for_classification():
    dataset = DatasetBundle(X=np.ones((12, 2)), y=["only"] * 12)
    with pytest.raises(DatasetValidationError, match="at least two|binary|multiclass"):
        dataset.validate(task="classification")


def test_string_target_is_rejected_for_regression():
    dataset = DatasetBundle(X=np.ones((12, 2)), y=["x", "y"] * 6)
    with pytest.raises(DatasetValidationError, match="numeric"):
        dataset.validate(task="regression")


@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_infinite_features_are_rejected_at_dataset_boundary(bad_value):
    X = np.zeros((8, 3), dtype=float)
    X[2, 1] = bad_value
    with pytest.raises(DatasetValidationError, match="infinite"):
        DatasetBundle(X=X, y=np.arange(8, dtype=float))


def test_nan_features_remain_allowed_for_fold_local_imputation():
    X, y = make_classification(n_samples=45, n_features=6, n_informative=4, random_state=4)
    X[::7, 2] = np.nan
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(45)])
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=_cv_plan(dataset),
        preprocessing=PreprocessingConfig(imputation="median", scaler="standard"),
        metrics=("accuracy", "balanced_accuracy"),
        random_state=42,
    )
    assert result.n_splits == 3
    assert all(np.isfinite(list(fold.evaluation.metrics.values())).all() for fold in result.folds)


def test_high_dimensional_p_greater_than_n_classification_runs():
    X, y = make_classification(
        n_samples=36,
        n_features=180,
        n_informative=12,
        n_redundant=8,
        random_state=9,
    )
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(36)])
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=_cv_plan(dataset),
        preprocessing=PreprocessingConfig(scaler="standard"),
        metrics=("accuracy", "mcc"),
        random_state=5,
    )
    assert set(result.aggregate_metrics) == {"accuracy", "mcc"}


def test_constant_feature_columns_do_not_break_scaled_linear_classification():
    X, y = make_classification(n_samples=60, n_features=5, n_informative=3, random_state=1)
    X = np.column_stack([X, np.ones(60), np.zeros(60)])
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(60)])
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=_cv_plan(dataset),
        metrics=("accuracy",),
        random_state=3,
    )
    assert np.isfinite(result.aggregate_metrics["accuracy"])


def test_outlier_heldout_samples_do_not_change_robust_scaler_fit():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10000.0], [-10000.0]])
    y = np.array([0, 1, 0, 1, 0, 1])
    ids = tuple(f"s{i}" for i in range(6))
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids)
    plan = PartitionPlan.holdout(train_ids=ids[:4], test_ids=ids[4:], dataset_fingerprint=dataset.fingerprint)
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=plan,
        preprocessing=PreprocessingConfig(imputation=None, scaler="robust"),
        metrics=("accuracy",),
        evaluation_role="test",
    )
    scaler = result.folds[0].estimator.named_steps["scaler"]
    np.testing.assert_allclose(scaler.center_, [1.5])


def test_negative_features_are_made_safe_for_multinomial_nb_by_auto_preprocessing():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(48, 8))
    y = np.array([0, 1] * 24)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(48)])
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="multinomial_nb",
        partition_plan=_cv_plan(dataset),
        metrics=("accuracy",),
    )
    assert np.isfinite(result.aggregate_metrics["accuracy"])
    assert all(f.estimator.named_steps["scaler"].clip for f in result.folds)


def test_negative_features_without_nonnegative_transform_fail_for_multinomial_nb():
    X = np.array([[-1.0, 2.0], [0.0, 1.0], [1.0, 3.0], [2.0, 4.0], [3.0, 2.0], [4.0, 1.0]])
    dataset = DatasetBundle(X=X, y=[0, 1, 0, 1, 0, 1], sample_ids=list("abcdef"))
    plan = PartitionPlan.holdout(train_ids=list("abcd"), test_ids=list("ef"))
    with pytest.raises(PreprocessingContractError, match="non-negative"):
        ValidationEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="multinomial_nb",
            partition_plan=plan,
            preprocessing=PreprocessingConfig(imputation=None, scaler=None),
            metrics=("accuracy",),
            evaluation_role="test",
        )


def test_gamma_regression_rejects_nonpositive_training_targets_early():
    X, y = make_regression(n_samples=30, n_features=4, random_state=2)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(30)])
    plan = PartitionPlan.holdout(train_ids=dataset.sample_ids[:20], test_ids=dataset.sample_ids[20:])
    with pytest.raises(PreprocessingContractError, match="positive target"):
        ValidationEngine(MODEL_REGISTRY).run(
            dataset=dataset,
            algorithm="gamma_regression",
            partition_plan=plan,
            metrics=("rmse",),
            evaluation_role="test",
        )


def test_high_dimensional_regression_runs_with_regularization():
    X, y = make_regression(n_samples=35, n_features=120, n_informative=15, noise=0.5, random_state=7)
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"r{i}" for i in range(35)])
    result = ValidationEngine(MODEL_REGISTRY).run(
        dataset=dataset,
        algorithm="ridge_regressor",
        partition_plan=_cv_plan(dataset, n_splits=5),
        metrics=("rmse", "mae"),
    )
    assert result.aggregate_metrics["rmse"] >= 0.0
    assert result.aggregate_metrics["mae"] >= 0.0


def test_dataframe_nondefault_index_does_not_affect_sample_identity():
    X, y = make_classification(n_samples=24, n_features=4, random_state=5)
    frame = pd.DataFrame(X, columns=list("abcd"), index=np.arange(100, 124))
    ids = [f"id_{i}" for i in range(24)]
    dataset = DatasetBundle(frame, y, sample_ids=ids)
    subset = dataset.subset([ids[10], ids[2], ids[18]])
    assert subset.sample_ids == (ids[2], ids[10], ids[18])
    assert subset.X.index.tolist() == [102, 110, 118]
