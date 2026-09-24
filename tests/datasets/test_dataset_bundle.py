from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saber.datasets import DatasetBundle, FeatureSchema
from saber.exceptions import DatasetValidationError, FeatureSchemaMismatchError


def _dataset(**overrides):
    values = {
        "X": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        "y": np.array([0, 1, 0, 1]),
        "sample_ids": ("s1", "s2", "s3", "s4"),
    }
    values.update(overrides)
    return DatasetBundle(**values)


def test_dataset_bundle_constructs_numerical_contract():
    dataset = _dataset()
    assert dataset.n_samples == 4
    assert dataset.n_features == 2
    assert dataset.feature_names == ("feature_0", "feature_1")
    assert dataset.feature_schema.n_features == 2


def test_dataframe_feature_names_are_preserved():
    X = pd.DataFrame({"esm_1": [1.0, 2.0], "esm_2": [3.0, 4.0]})
    dataset = DatasetBundle(X=X, y=[0, 1], sample_ids=["a", "b"])
    assert dataset.feature_names == ("esm_1", "esm_2")
    assert dataset.feature_schema.dtypes == ("float64", "float64")


def test_generated_sample_ids_are_explicitly_reported():
    dataset = DatasetBundle(X=np.eye(3), y=[0, 1, 0])
    assert dataset.sample_ids == (0, 1, 2)
    assert dataset.generated_sample_ids is True


def test_nan_features_are_allowed_for_future_fold_safe_imputation():
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    dataset = DatasetBundle(X=X, y=[0, 1])
    assert dataset.n_samples == 2
    with pytest.raises(DatasetValidationError):
        dataset.validate(require_finite_features=True)


def test_infinite_features_are_rejected():
    with pytest.raises(DatasetValidationError):
        DatasetBundle(X=[[1.0, np.inf], [2.0, 3.0]], y=[0, 1])


def test_non_numeric_features_are_rejected():
    with pytest.raises(DatasetValidationError):
        DatasetBundle(X=pd.DataFrame({"x": ["a", "b"]}), y=[0, 1])


def test_multioutput_target_is_rejected():
    with pytest.raises(DatasetValidationError):
        DatasetBundle(X=np.eye(2), y=[[0, 1], [1, 0]])


def test_sample_ids_must_be_unique_and_aligned():
    with pytest.raises(DatasetValidationError):
        _dataset(sample_ids=("s1", "s1", "s3", "s4"))
    with pytest.raises(DatasetValidationError):
        _dataset(sample_ids=("s1", "s2"))


def test_mixed_type_sample_ids_are_rejected():
    with pytest.raises(DatasetValidationError) as exc_info:
        _dataset(sample_ids=(1, "2", 3, 4))
    assert "sample_ids must all share one type" in str(exc_info.value)
    assert "found: int, str" in str(exc_info.value)


def test_homogeneous_sample_id_types_are_accepted():
    assert _dataset(sample_ids=(1, 2, 3, 4)).sample_ids == (1, 2, 3, 4)
    assert _dataset(sample_ids=(1.0, 2.0, 3.0, 4.0)).sample_ids == (1.0, 2.0, 3.0, 4.0)
    assert _dataset(sample_ids=np.array([1, 2, 3, 4], dtype=np.int64)).sample_ids == (1, 2, 3, 4)


def test_groups_and_sample_weights_are_validated():
    dataset = _dataset(groups=("g1", "g1", "g2", "g2"), sample_weight=[1, 2, 1, 3])
    assert dataset.groups == ("g1", "g1", "g2", "g2")
    assert np.allclose(dataset.sample_weight, [1, 2, 1, 3])

    with pytest.raises(DatasetValidationError):
        _dataset(sample_weight=[1, -1, 1, 1])


def test_task_specific_target_validation():
    classification = _dataset()
    assert classification.validate(task="classification") == "binary"

    multiclass = _dataset(y=np.array([0, 1, 2, 1]))
    assert multiclass.validate(task="classification") == "multiclass"

    regression = _dataset(y=np.array([1.2, 3.4, 2.1, 9.0]))
    assert regression.validate(task="regression") == "regression"

    with pytest.raises(DatasetValidationError):
        _dataset(y=np.array(["a", "b", "c", "d"])).validate(task="regression")


def test_dataset_fingerprint_is_deterministic_and_metadata_independent():
    first = _dataset(metadata={"note": "first"})
    second = _dataset(metadata={"note": "different"})
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_dataset_fingerprint_changes_when_scientific_content_changes():
    first = _dataset()
    changed_target = _dataset(y=np.array([0, 1, 1, 1]))
    changed_ids = _dataset(sample_ids=("s4", "s2", "s3", "s1"))
    assert first.fingerprint != changed_target.fingerprint
    assert first.fingerprint != changed_ids.fingerprint


def test_subset_preserves_identity_features_groups_and_weights():
    dataset = _dataset(
        groups=("g1", "g1", "g2", "g2"),
        sample_weight=[1, 2, 3, 4],
    )
    subset = dataset.subset(["s4", "s2"])

    # Membership order in a partition is not data order; dataset order is stable.
    assert subset.sample_ids == ("s2", "s4")
    assert subset.feature_names == dataset.feature_names
    assert subset.groups == ("g1", "g2")
    assert np.allclose(subset.sample_weight, [2, 4])
    assert subset.metadata["parent_dataset_fingerprint"] == dataset.fingerprint


def test_subset_rejects_unknown_or_duplicate_requested_ids():
    dataset = _dataset()
    with pytest.raises(DatasetValidationError):
        dataset.subset(["missing"])
    with pytest.raises(DatasetValidationError):
        dataset.subset(["s1", "s1"])


def test_feature_schema_checks_order_and_optional_dtypes():
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    schema = FeatureSchema.from_data(train)
    schema.validate_compatible(train.copy())

    with pytest.raises(FeatureSchemaMismatchError):
        schema.validate_compatible(train[["b", "a"]])

    integer_frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    schema.validate_compatible(integer_frame, check_dtypes=False)
    with pytest.raises(FeatureSchemaMismatchError):
        schema.validate_compatible(integer_frame, check_dtypes=True)
