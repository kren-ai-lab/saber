from __future__ import annotations

import numpy as np
import pytest

from mlcore.datasets import DatasetBundle, PartitionPlan, PartitionSplit
from mlcore.exceptions import DatasetFingerprintMismatchError, PartitionValidationError


def _dataset():
    return DatasetBundle(
        X=np.arange(24, dtype=float).reshape(6, 4),
        y=[0, 1, 0, 1, 0, 1],
        sample_ids=["a", "b", "c", "d", "e", "f"],
    )


def test_partition_split_rejects_overlap():
    with pytest.raises(PartitionValidationError):
        PartitionSplit(
            name="bad",
            train_ids=("a", "b"),
            validation_ids=("b", "c"),
        )


def test_partition_split_rejects_duplicate_ids():
    with pytest.raises(PartitionValidationError):
        PartitionSplit(name="bad", train_ids=("a", "a"), test_ids=("b",))


def test_holdout_plan_validates_and_resolves_dataset_identity():
    dataset = _dataset()
    plan = PartitionPlan.holdout(
        train_ids=["a", "b", "c"],
        validation_ids=["d"],
        test_ids=["e", "f"],
        dataset_fingerprint=dataset.fingerprint,
    )
    plan.validate_against(dataset)
    resolved = plan.resolve(dataset, "holdout")

    assert resolved.train.sample_ids == ("a", "b", "c")
    assert resolved.validation is not None
    assert resolved.validation.sample_ids == ("d",)
    assert resolved.test is not None
    assert resolved.test.sample_ids == ("e", "f")


def test_partition_unknown_samples_fail_loudly():
    dataset = _dataset()
    plan = PartitionPlan.holdout(
        train_ids=["a", "b", "c"],
        test_ids=["d", "e", "unknown"],
    )
    with pytest.raises(PartitionValidationError, match="unknown sample IDs"):
        plan.validate_against(dataset)


def test_partition_missing_coverage_fails_by_default():
    dataset = _dataset()
    plan = PartitionPlan.holdout(
        train_ids=["a", "b", "c"],
        test_ids=["d", "e"],
    )
    with pytest.raises(PartitionValidationError, match="does not cover all"):
        plan.validate_against(dataset)

    plan.validate_against(dataset, require_complete=False)


def test_dataset_fingerprint_mismatch_is_detected():
    dataset = _dataset()
    plan = PartitionPlan.holdout(
        train_ids=["a", "b", "c"],
        test_ids=["d", "e", "f"],
        dataset_fingerprint="wrong-fingerprint",
    )
    with pytest.raises(DatasetFingerprintMismatchError):
        plan.validate_against(dataset)


def test_predefined_fold_assignments_become_explicit_memberships():
    dataset = _dataset()
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=[0, 0, 1, 1, 2, 2],
        dataset_fingerprint=dataset.fingerprint,
    )

    assert plan.kind == "predefined"
    assert plan.n_splits == 3
    assert plan.get_split("fold_0").validation_ids == ("a", "b")
    assert plan.get_split("fold_0").train_ids == ("c", "d", "e", "f")
    assert plan.validation_counts() == {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "e": 1,
        "f": 1,
    }
    plan.validate_against(dataset)


def test_predefined_always_train_samples_never_enter_validation():
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=["always", "a", "b", "c", "d"],
        fold_assignments=[-1, 0, 0, 1, 1],
    )
    assert all("always" in split.train_ids for split in plan.splits)
    assert all("always" not in split.validation_ids for split in plan.splits)


def test_partition_fingerprint_is_deterministic_and_membership_order_independent():
    first = PartitionPlan.holdout(
        train_ids=["a", "b", "c"],
        test_ids=["d", "e", "f"],
    )
    reordered = PartitionPlan.holdout(
        train_ids=["c", "a", "b"],
        test_ids=["f", "d", "e"],
    )
    changed = PartitionPlan.holdout(
        train_ids=["a", "b", "d"],
        test_ids=["c", "e", "f"],
    )

    assert first.fingerprint == reordered.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_partition_plan_dict_roundtrip_preserves_fingerprint():
    plan = PartitionPlan.holdout(
        train_ids=["a", "b", "c"],
        validation_ids=["d"],
        test_ids=["e", "f"],
        dataset_fingerprint="dataset-abc",
        metadata={"source": "unit-test"},
    )
    restored = PartitionPlan.from_dict(plan.to_dict())
    assert restored.fingerprint == plan.fingerprint
    assert restored.get_split("holdout").test_ids == ("e", "f")
