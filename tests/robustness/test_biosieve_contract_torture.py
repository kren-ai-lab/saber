from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import saber.validation.partitioning as partitioning_module
from saber.datasets import DatasetBundle
from saber.datasets import biosieve as adapter
from saber.datasets.biosieve import BioSievePartitionConfig, partition_with_biosieve
from saber.exceptions import PartitionIntegrationError, PartitionValidationError
from saber.validation import validate


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
        return FakeFrame({name: [values[i] for i in indices] for name, values in self.payload.items()})


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


@pytest.fixture
def fake_runtime(monkeypatch):
    monkeypatch.setattr(adapter, "_import_biosieve_runtime", lambda: (FakePolars, FakeColumns))
    monkeypatch.setattr(adapter, "_biosieve_version", lambda: "0.1.2-test")


def _dataset(n=36, *, groups=False):
    rng = np.random.default_rng(8)
    X = rng.normal(size=(n, 6))
    y = np.array([0, 1] * (n // 2))
    kwargs: dict[str, Any] = {}
    if groups:
        kwargs["groups"] = [f"g{i // 3}" for i in range(n)]
    return DatasetBundle(X=X, y=y, sample_ids=[f"s{i}" for i in range(n)], **kwargs)


class CompleteKFoldSplitter:
    strategy = "stratified_kfold"

    def run_folds(self, frame, cols):  # noqa: ARG002  fixed signature required by splitter protocol
        n = frame.height
        folds = [list(range(i, n, 3)) for i in range(3)]
        all_idx = set(range(n))
        return [
            SimpleNamespace(
                train=frame[sorted(all_idx - set(test_idx))],
                test=frame[test_idx],
                val=None,
                strategy=self.strategy,
                params={"n_splits": 3, "fold_index": i},
                stats={"fold_index": i},
            )
            for i, test_idx in enumerate(folds)
        ]


class EmptySplitter:
    strategy = "random"

    def run_folds(self, frame, cols):  # noqa: ARG002  fixed signature required by splitter protocol
        return []


class MissingIdColumnSplitter:
    strategy = "random"

    def run(self, frame, cols):  # noqa: ARG002  fixed signature required by splitter protocol
        bad = FakeFrame({"wrong": list(range(frame.height))})
        return SimpleNamespace(train=bad, test=bad, val=None, strategy="random", params={}, stats={})


class UnknownIdSplitter:
    strategy = "random"

    def run(self, frame, cols):
        payload = dict(frame.payload)
        payload[cols.id_col] = list(payload[cols.id_col])
        payload[cols.id_col][-1] = "foreign_id"
        altered = FakeFrame(payload)
        return SimpleNamespace(
            train=altered[list(range(frame.height - 2))],
            test=altered[list(range(frame.height - 2, frame.height))],
            val=None,
            strategy="random",
            params={},
            stats={},
        )


def test_validation_engine_can_consume_biosieve_generated_memberships_end_to_end(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
    monkeypatch,
):
    dataset = _dataset()

    def fake_partition(ds, config):
        return partition_with_biosieve(ds, config, splitter=CompleteKFoldSplitter())

    monkeypatch.setattr(partitioning_module, "partition_with_biosieve", fake_partition)
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=BioSievePartitionConfig(strategy="stratified_kfold", params={"n_splits": 3}),
        metrics=("accuracy", "mcc"),
        random_state=42,
    )
    assert result.partition_plan.metadata["source"] == "biosieve"
    assert result.partition_plan.metadata["biosieve_version"] == "0.1.2-test"
    assert result.oof_prediction is not None
    assert result.metadata["oof_complete"] is True


def test_biosieve_empty_fold_output_is_rejected(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    with pytest.raises(PartitionIntegrationError, match="no split results"):
        partition_with_biosieve(
            _dataset(),
            BioSievePartitionConfig(strategy="random_kfold"),
            splitter=EmptySplitter(),
        )


def test_biosieve_output_missing_sample_id_column_is_rejected(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    with pytest.raises(PartitionIntegrationError, match="sample ID column"):
        partition_with_biosieve(
            _dataset(),
            BioSievePartitionConfig(strategy="random"),
            splitter=MissingIdColumnSplitter(),
        )


def test_biosieve_output_with_unknown_sample_id_is_rejected(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    with pytest.raises(PartitionValidationError, match=r"unknown|Unknown"):
        partition_with_biosieve(
            _dataset(),
            BioSievePartitionConfig(strategy="random"),
            splitter=UnknownIdSplitter(),
        )


def test_biosieve_extra_column_cannot_override_reserved_target(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    dataset = _dataset()
    with pytest.raises(PartitionIntegrationError, match="overwrite"):
        partition_with_biosieve(
            dataset,
            BioSievePartitionConfig(
                strategy="random",
                extra_columns={"__saber_target__": np.zeros(dataset.n_samples).tolist()},
            ),
            splitter=CompleteKFoldSplitter(),
        )


def test_biosieve_feature_name_collision_with_reserved_columns_is_rejected(
    fake_runtime,  # noqa: ARG001  fixture required for setup, value unused
):
    dataset = DatasetBundle(
        X=np.ones((8, 2)),
        y=[0, 1] * 4,
        feature_names=["__saber_sample_id__", "x"],
        sample_ids=[f"s{i}" for i in range(8)],
    )
    with pytest.raises(PartitionIntegrationError, match="collide"):
        partition_with_biosieve(
            dataset,
            BioSievePartitionConfig(strategy="distance_aware", params={"feature_mode": "descriptors"}),
            splitter=CompleteKFoldSplitter(),
        )
