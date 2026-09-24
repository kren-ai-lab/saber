"""Load external partition contracts without provider dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from saber.datasets.folds import PartitionPlan, PartitionSplit
from saber.exceptions import PartitionValidationError

_ROLE_ALIASES = {
    "train": "train",
    "training": "train",
    "validation": "validation",
    "valid": "validation",
    "val": "validation",
    "test": "test",
    "testing": "test",
}


def partition_plan_from_frame(
    frame: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    role_col: str | None = "role",
    split_col: str | None = "split",
    fold_col: str | None = None,
    dataset_fingerprint: str | None = None,
    dataset_fingerprint_col: str = "dataset_fingerprint",
    always_train_value: Any = -1,
) -> PartitionPlan:
    """Create a partition plan from a dependency-free interchange DataFrame.

    Two normalized external layouts are supported:

    1. membership layout: ``sample_id``, ``role`` and optional ``split``;
    2. predefined-fold layout: ``sample_id`` and a caller-selected ``fold_col``.

    This is the BioSieve interoperability boundary: BioSieve only needs to
    export one of these tabular contracts; saber never imports BioSieve.
    """
    if sample_id_col not in frame.columns:
        raise PartitionValidationError(f"External partition table is missing '{sample_id_col}'.")

    inferred_fingerprint = _extract_dataset_fingerprint(
        frame,
        column=dataset_fingerprint_col,
    )
    if dataset_fingerprint is not None and inferred_fingerprint is not None:
        if dataset_fingerprint != inferred_fingerprint:
            raise PartitionValidationError("Explicit dataset_fingerprint disagrees with the partition table.")
    resolved_fingerprint = dataset_fingerprint or inferred_fingerprint

    if fold_col is not None:
        if fold_col not in frame.columns:
            raise PartitionValidationError(f"External partition table is missing fold column '{fold_col}'.")
        return PartitionPlan.from_predefined_folds(
            sample_ids=frame[sample_id_col].tolist(),
            fold_assignments=frame[fold_col].tolist(),
            always_train_value=always_train_value,
            dataset_fingerprint=resolved_fingerprint,
            metadata={"source": "external_frame", "layout": "fold_assignment"},
        )

    if role_col is None or role_col not in frame.columns:
        raise PartitionValidationError(
            "Membership partition tables require a role column, or provide fold_col."
        )

    working = frame.copy()
    if split_col is None or split_col not in working.columns:
        internal_split_col = "__saber_split__"
        working[internal_split_col] = "split_0"
    else:
        internal_split_col = split_col

    splits: list[PartitionSplit] = []
    for split_name, split_frame in working.groupby(internal_split_col, sort=False, dropna=False):
        memberships: dict[str, list[Any]] = {
            "train": [],
            "validation": [],
            "test": [],
        }

        for _, row in split_frame.iterrows():
            raw_role = str(row[role_col]).strip().lower()
            role = _ROLE_ALIASES.get(raw_role)
            if role is None:
                raise PartitionValidationError(f"Unknown partition role '{row[role_col]}'.")
            memberships[role].append(row[sample_id_col])

        splits.append(
            PartitionSplit(
                name=str(split_name),
                train_ids=tuple(memberships["train"]),
                validation_ids=tuple(memberships["validation"]),
                test_ids=tuple(memberships["test"]),
            )
        )

    return PartitionPlan(
        kind="external",
        dataset_fingerprint=resolved_fingerprint,
        splits=tuple(splits),
        metadata={"source": "external_frame", "layout": "membership"},
    )


def load_partition_plan(
    source: str | Path | dict[str, Any] | pd.DataFrame,
    **frame_kwargs: Any,
) -> PartitionPlan:
    """Load a partition plan from JSON mapping/file or a tabular frame/CSV."""
    if isinstance(source, PartitionPlan):
        return source
    if isinstance(source, dict):
        return PartitionPlan.from_dict(source)
    if isinstance(source, pd.DataFrame):
        return partition_plan_from_frame(source, **frame_kwargs)

    path = Path(source)
    if not path.exists():
        raise PartitionValidationError(f"Partition source does not exist: {path}.")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise PartitionValidationError("Partition JSON must contain an object payload.")
        return PartitionPlan.from_dict(payload)

    if suffix in {".csv", ".tsv"}:
        separator = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator)
        return partition_plan_from_frame(frame, **frame_kwargs)

    raise PartitionValidationError(
        "Unsupported partition source. Use JSON, CSV, TSV, dict, or pandas DataFrame."
    )


def _extract_dataset_fingerprint(frame: pd.DataFrame, *, column: str) -> str | None:
    if column not in frame.columns:
        return None

    values = [str(value) for value in frame[column].dropna().unique()]
    if not values:
        return None
    if len(values) != 1:
        raise PartitionValidationError(f"Column '{column}' contains multiple dataset fingerprints.")
    return values[0]
