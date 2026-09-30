from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

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
    with pytest.raises(DatasetValidationError, match=r"classification|target|continuous"):
        dataset.validate(task="classification")


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["neg", "pos"] * 10, "binary"),
        ([10, 20] * 10, "binary"),
        (["a", "b", "c", "a", "b"] * 4, "multiclass"),
    ],
)
def test_supported_class_label_regimes_are_detected(labels, expected):
    X = np.arange(60, dtype=float).reshape(20, 3)
    dataset = DatasetBundle(X=X, y=labels)
    assert dataset.validate(task="classification") == expected


def test_single_class_target_is_rejected_for_classification():
    dataset = DatasetBundle(X=np.ones((12, 2)), y=["only"] * 12)
    with pytest.raises(DatasetValidationError, match=r"at least two|binary|multiclass"):
        dataset.validate(task="classification")


def test_negative_features_without_nonnegative_transform_fail_for_multinomial_nb():
    X = np.array([[-1.0, 2.0], [0.0, 1.0], [1.0, 3.0], [2.0, 4.0], [3.0, 2.0], [4.0, 1.0]])
    dataset = DatasetBundle(X=X, y=[0, 1, 0, 1, 0, 1], sample_ids=list("abcdef"))
    plan = PartitionPlan.holdout(train_ids=list("abcd"), test_ids=list("ef"))
    with pytest.raises(PreprocessingContractError, match="non-negative"):
        ValidationEngine().run(
            dataset=dataset,
            algorithm="multinomial_nb",
            partition_plan=plan,
            preprocessing=PreprocessingConfig(imputation=None, scaler=None),
            metrics=("accuracy",),
            evaluation_role="test",
        )


def test_gamma_regression_rejects_nonpositive_training_targets_early():
    X, y = make_regression(n_samples=30, n_features=4, random_state=2)  # pyrefly: ignore[bad-unpacking]
    dataset = DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(30)])
    assert dataset.sample_ids is not None
    plan = PartitionPlan.holdout(train_ids=dataset.sample_ids[:20], test_ids=dataset.sample_ids[20:])
    with pytest.raises(PreprocessingContractError, match="positive target"):
        ValidationEngine().run(
            dataset=dataset,
            algorithm="gamma_regression",
            partition_plan=plan,
            metrics=("rmse",),
            evaluation_role="test",
        )


def test_dataframe_nondefault_index_does_not_affect_sample_identity():
    X, y = make_classification(n_samples=24, n_features=4, random_state=5)
    frame = pd.DataFrame(X, columns=list("abcd"), index=np.arange(100, 124))
    ids = [f"id_{i}" for i in range(24)]
    dataset = DatasetBundle(frame, y, sample_ids=ids)
    subset = dataset.subset([ids[10], ids[2], ids[18]])
    assert subset.sample_ids == (ids[2], ids[10], ids[18])
    assert subset.X["a"].to_list() == frame["a"].to_numpy()[[2, 10, 18]].tolist()
