from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from saber.datasets import DatasetBundle
from saber.datasets import biosieve as adapter
from saber.datasets.biosieve import BioSievePartitionConfig, partition_with_biosieve
from saber.exceptions import OptionalDependencyError, PartitionIntegrationError


class FakeSeries:
    def __init__(self, values):
        self.values = list(values)

    def to_list(self):
        return list(self.values)


class FakeFrame:
    def __init__(self, payload):
        self.payload = {name: list(values) for name, values in payload.items()}
        self.columns = list(self.payload)
        self.height = len(next(iter(self.payload.values()))) if self.payload else 0

    def __getitem__(self, key):
        if isinstance(key, str):
            return FakeSeries(self.payload[key])
        indices = list(key)
        return FakeFrame(
            {name: [values[index] for index in indices] for name, values in self.payload.items()}
        )

    def clone(self):
        return FakeFrame(self.payload)


class FakePolars:
    DataFrame = FakeFrame


@dataclass(frozen=True)
class FakeColumns:
    id_col: str = "id"
    seq_col: str = "sequence"
    label_col: str | None = "label"
    group_col: str | None = None
    cluster_col: str | None = None
    date_col: str | None = None


class FakeKFoldSplitter:
    strategy = "random_kfold"

    def run_folds(self, frame, cols):
        assert cols.id_col == "__saber_sample_id__"
        results = []
        folds = ([0, 1], [2, 3], [4, 5])
        all_indices = set(range(6))
        for fold_index, test_indices in enumerate(folds):
            train_indices = sorted(all_indices - set(test_indices))
            results.append(
                SimpleNamespace(
                    train=frame[train_indices],
                    test=frame[test_indices],
                    val=None,
                    strategy="random_kfold",
                    params={"n_splits": 3, "fold_index": fold_index},
                    stats={"fold_index": fold_index, "n_test": 2},
                )
            )
        return results


class FakeSingleSplitter:
    strategy = "group"

    def run(self, frame, cols):
        assert cols.group_col == "__saber_group__"
        return SimpleNamespace(
            train=frame[[0, 1, 2, 3]],
            test=frame[[4, 5]],
            val=None,
            strategy="group",
            params={"group_col": cols.group_col, "seed": 13},
            stats={"leak_groups_train_test": 0},
        )


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(adapter, "_import_biosieve_runtime", lambda: (FakePolars, FakeColumns))
    monkeypatch.setattr(adapter, "_biosieve_version", lambda: "0.1.2")


def _dataset(*, with_groups=False):
    kwargs = {}
    if with_groups:
        kwargs["groups"] = ["g1", "g1", "g2", "g2", "g3", "g3"]
    return DatasetBundle(
        X=np.arange(18, dtype=float).reshape(6, 3),
        y=[0, 1, 0, 1, 0, 1],
        sample_ids=[f"s{i}" for i in range(6)],
        **kwargs,
    )


def test_biosieve_kfold_results_are_adapted_without_regenerating_membership(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    dataset = _dataset()
    config = BioSievePartitionConfig(strategy="random_kfold", params={"n_splits": 3})
    plan = partition_with_biosieve(
        dataset,
        config,
        splitter=FakeKFoldSplitter(),
    )

    assert plan.kind == "cross_validation"
    assert plan.n_splits == 3
    assert plan.metadata["source"] == "biosieve"
    assert plan.metadata["biosieve_version"] == "0.1.2"
    assert plan.get_split("fold_0").test_ids == ("s0", "s1")
    assert plan.get_split("fold_1").test_ids == ("s2", "s3")
    assert plan.get_split("fold_2").test_ids == ("s4", "s5")
    assert plan.get_split("fold_0").metadata["stats"]["fold_index"] == 0
    plan.validate_against(dataset)


def test_biosieve_group_split_preserves_groups_and_strategy_provenance(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    dataset = _dataset(with_groups=True)
    plan = partition_with_biosieve(
        dataset,
        BioSievePartitionConfig(strategy="group"),
        splitter=FakeSingleSplitter(),
    )
    split = plan.splits[0]
    assert split.train_ids == ("s0", "s1", "s2", "s3")
    assert split.test_ids == ("s4", "s5")
    assert split.metadata["strategy"] == "group"
    assert split.metadata["stats"]["leak_groups_train_test"] == 0


def test_biosieve_extra_columns_must_be_aligned(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    dataset = _dataset()
    with pytest.raises(PartitionIntegrationError, match="must contain 6"):
        partition_with_biosieve(
            dataset,
            BioSievePartitionConfig(strategy="random"),
            extra_columns={"sequence": ["AAA"]},
            splitter=FakeSingleSplitter(),
        )


def test_group_strategy_requires_dataset_groups(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    dataset = _dataset(with_groups=False)
    with pytest.raises(PartitionIntegrationError, match=re.escape("requires DatasetBundle.groups")):
        partition_with_biosieve(
            dataset,
            BioSievePartitionConfig(strategy="group"),
            splitter=FakeSingleSplitter(),
        )


def test_missing_biosieve_dependency_has_actionable_error(monkeypatch):
    original = adapter.import_module

    def fail(name):
        if name in {"polars", "biosieve.types"}:
            raise ImportError(name)
        return original(name)

    monkeypatch.setattr(adapter, "import_module", fail)
    with pytest.raises(OptionalDependencyError, match="saberlib\\[biosieve\\]"):
        adapter._import_biosieve_runtime()  # noqa: SLF001  test verifies internal state directly


def test_unknown_biosieve_strategy_fails_before_execution():
    with pytest.raises(PartitionIntegrationError, match="Unsupported BioSieve strategy"):
        BioSievePartitionConfig(strategy="invented_splitter")


class CaptureSplitter:
    strategy = "capture"

    def __init__(self):
        self.columns_seen = None

    def run(self, frame, cols):  # noqa: ARG002  fixed signature required by splitter protocol
        self.columns_seen = tuple(frame.columns)
        return SimpleNamespace(
            train=frame[[0, 1, 2, 3]],
            test=frame[[4, 5]],
            val=None,
            strategy="capture",
            params={},
            stats={},
        )


def test_regular_biosieve_partition_does_not_copy_full_feature_matrix(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    dataset = _dataset()
    splitter = CaptureSplitter()
    partition_with_biosieve(
        dataset,
        BioSievePartitionConfig(strategy="random"),
        splitter=splitter,
    )
    assert splitter.columns_seen == (
        "__saber_sample_id__",
        "__saber_target__",
    )


def test_distance_descriptor_partition_exposes_prepared_numeric_features(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    dataset = _dataset()
    splitter = CaptureSplitter()
    partition_with_biosieve(
        dataset,
        BioSievePartitionConfig(
            strategy="distance_aware",
            params={"feature_mode": "descriptors"},
        ),
        splitter=splitter,
    )
    assert "feature_0" in splitter.columns_seen
    assert "feature_1" in splitter.columns_seen
    assert "feature_2" in splitter.columns_seen


class CapturePolarsSplitter:
    strategy = "capture"

    def __init__(self):
        self.frame_seen = None

    def run(self, frame, cols):  # noqa: ARG002  fixed signature required by splitter protocol
        self.frame_seen = frame
        return SimpleNamespace(
            train=frame[[0, 1, 2, 3]],
            test=frame[[4, 5]],
            val=None,
            strategy="capture",
            params={},
            stats={},
        )


def test_polars_dataset_preserves_int_and_bool_column_dtypes(monkeypatch):
    monkeypatch.setattr(adapter, "_import_biosieve_runtime", lambda: (pl, FakeColumns))
    monkeypatch.setattr(adapter, "_biosieve_version", lambda: "0.1.2")

    X = pl.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5, 6],
            "bool_col": [True, False, True, False, True, False],
        }
    )
    dataset = DatasetBundle(X=X, y=[0, 1, 0, 1, 0, 1], sample_ids=[f"s{i}" for i in range(6)])
    splitter = CapturePolarsSplitter()

    partition_with_biosieve(
        dataset,
        BioSievePartitionConfig(strategy="distance_aware", params={"feature_mode": "descriptors"}),
        splitter=splitter,
    )

    assert splitter.frame_seen["int_col"].dtype == pl.Int64
    assert splitter.frame_seen["bool_col"].dtype == pl.Boolean
