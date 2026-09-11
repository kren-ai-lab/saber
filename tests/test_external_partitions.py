from __future__ import annotations

import json

import pandas as pd
import pytest

from mlcore.datasets import load_partition_plan, partition_plan_from_frame
from mlcore.exceptions import PartitionValidationError


def test_membership_frame_supports_external_biosieve_style_contract():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d", "a", "b", "c", "d"],
            "split": ["fold0"] * 4 + ["fold1"] * 4,
            "role": [
                "validation",
                "validation",
                "train",
                "train",
                "train",
                "train",
                "validation",
                "validation",
            ],
            "dataset_fingerprint": ["abc"] * 8,
        }
    )

    plan = partition_plan_from_frame(frame)
    assert plan.kind == "external"
    assert plan.dataset_fingerprint == "abc"
    assert plan.n_splits == 2
    assert plan.get_split("fold0").validation_ids == ("a", "b")
    assert plan.get_split("fold1").validation_ids == ("c", "d")


def test_membership_frame_accepts_role_aliases_and_custom_columns():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "subset": ["training", "val", "testing"],
        }
    )
    plan = partition_plan_from_frame(
        frame,
        sample_id_col="id",
        role_col="subset",
        split_col=None,
    )
    split = plan.get_split("split_0")
    assert split.train_ids == (1,)
    assert split.validation_ids == (2,)
    assert split.test_ids == (3,)


def test_fold_assignment_frame_is_supported_without_biosieve_dependency():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "fold_id": [0, 0, 1, 1],
        }
    )
    plan = partition_plan_from_frame(frame, fold_col="fold_id")
    assert plan.kind == "predefined"
    assert plan.get_split("fold_0").validation_ids == ("a", "b")


def test_external_frame_rejects_unknown_roles():
    frame = pd.DataFrame({"sample_id": ["a"], "role": ["mystery"]})
    with pytest.raises(PartitionValidationError, match="Unknown partition role"):
        partition_plan_from_frame(frame)


def test_external_frame_rejects_conflicting_dataset_fingerprints():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "role": ["train", "test"],
            "dataset_fingerprint": ["one", "two"],
        }
    )
    with pytest.raises(PartitionValidationError, match="multiple dataset fingerprints"):
        partition_plan_from_frame(frame)


def test_json_partition_plan_roundtrip(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "role": ["train", "validation", "test"],
        }
    )
    plan = partition_plan_from_frame(frame)
    path = tmp_path / "partitions.json"
    path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

    restored = load_partition_plan(path)
    assert restored.fingerprint == plan.fingerprint


def test_csv_partition_loader(tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "role": ["train", "validation", "test"],
        }
    )
    path = tmp_path / "partitions.csv"
    frame.to_csv(path, index=False)

    plan = load_partition_plan(path)
    assert plan.get_split("split_0").train_ids == ("a",)
    assert plan.get_split("split_0").test_ids == ("c",)
