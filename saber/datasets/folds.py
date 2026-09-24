"""Explicit partition and fold contracts for saber."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from saber.datasets._fingerprint import partition_fingerprint
from saber.datasets.schemas import DatasetBundle
from saber.exceptions import (
    DatasetFingerprintMismatchError,
    PartitionValidationError,
)

PartitionKind = Literal["holdout", "cross_validation", "predefined", "external"]


@dataclass(frozen=True)
class PartitionSplit:
    """One explicit train/validation/test membership definition."""

    name: str
    train_ids: tuple[Any, ...]
    validation_ids: tuple[Any, ...] = ()
    test_ids: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise PartitionValidationError("Partition split names cannot be empty.")

        train_ids = tuple(_to_python_scalar(value) for value in self.train_ids)
        validation_ids = tuple(_to_python_scalar(value) for value in self.validation_ids)
        test_ids = tuple(_to_python_scalar(value) for value in self.test_ids)

        object.__setattr__(self, "train_ids", train_ids)
        object.__setattr__(self, "validation_ids", validation_ids)
        object.__setattr__(self, "test_ids", test_ids)
        object.__setattr__(self, "metadata", dict(self.metadata))

        if not train_ids:
            raise PartitionValidationError(
                f"Split '{self.name}' must contain at least one training sample."
            )

        for role, values in (
            ("train", train_ids),
            ("validation", validation_ids),
            ("test", test_ids),
        ):
            _validate_unique_ids(values, split=self.name, role=role)

        train = set(train_ids)
        validation = set(validation_ids)
        test = set(test_ids)

        overlaps = {
            "train/validation": train & validation,
            "train/test": train & test,
            "validation/test": validation & test,
        }
        bad = {name: values for name, values in overlaps.items() if values}
        if bad:
            details = "; ".join(
                f"{pair}: {sorted(values, key=repr)!r}" for pair, values in bad.items()
            )
            raise PartitionValidationError(
                f"Split '{self.name}' contains overlapping memberships ({details})."
            )

    @property
    def all_ids(self) -> tuple[Any, ...]:
        return self.train_ids + self.validation_ids + self.test_ids

    def ids_for(self, role: str) -> tuple[Any, ...]:
        if role == "train":
            return self.train_ids
        if role in {"validation", "val"}:
            return self.validation_ids
        if role == "test":
            return self.test_ids
        raise PartitionValidationError(
            "role must be 'train', 'validation', or 'test'."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train_ids": list(self.train_ids),
            "validation_ids": list(self.validation_ids),
            "test_ids": list(self.test_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PartitionSplit":
        return cls(
            name=str(payload["name"]),
            train_ids=tuple(payload.get("train_ids", ())),
            validation_ids=tuple(payload.get("validation_ids", ())),
            test_ids=tuple(payload.get("test_ids", ())),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ResolvedPartition:
    """Dataset bundles resolved from one explicit split."""

    name: str
    train: DatasetBundle
    validation: DatasetBundle | None = None
    test: DatasetBundle | None = None


@dataclass(frozen=True)
class PartitionPlan:
    """Reproducible collection of explicit partition memberships."""

    splits: tuple[PartitionSplit, ...]
    kind: PartitionKind = "external"
    dataset_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        splits = tuple(self.splits)
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "metadata", dict(self.metadata))

        if not splits:
            raise PartitionValidationError("PartitionPlan must contain at least one split.")

        valid_kinds = {"holdout", "cross_validation", "predefined", "external"}
        if self.kind not in valid_kinds:
            raise PartitionValidationError(
                f"Unsupported partition kind '{self.kind}'."
            )

        names = [split.name for split in splits]
        if len(names) != len(set(names)):
            raise PartitionValidationError("Partition split names must be unique.")

    @property
    def fingerprint(self) -> str:
        return partition_fingerprint(
            kind=self.kind,
            splits=[split.to_dict() for split in self.splits],
            dataset_fingerprint_value=self.dataset_fingerprint,
        )

    @property
    def n_splits(self) -> int:
        return len(self.splits)

    def get_split(self, name: str) -> PartitionSplit:
        for split in self.splits:
            if split.name == name:
                return split
        raise PartitionValidationError(f"Partition split '{name}' was not found.")

    def validate_against(
        self,
        dataset: DatasetBundle,
        *,
        require_complete: bool = True,
        verify_dataset_fingerprint: bool = True,
    ) -> None:
        """Validate memberships against a concrete dataset."""

        if (
            verify_dataset_fingerprint
            and self.dataset_fingerprint is not None
            and self.dataset_fingerprint != dataset.fingerprint
        ):
            raise DatasetFingerprintMismatchError(
                expected=self.dataset_fingerprint,
                observed=dataset.fingerprint,
            )

        known_ids = set(dataset.sample_ids)

        for split in self.splits:
            split_ids = set(split.all_ids)
            unknown = split_ids - known_ids
            if unknown:
                raise PartitionValidationError(
                    f"Split '{split.name}' contains unknown sample IDs: "
                    f"{sorted(unknown, key=repr)!r}."
                )

            if require_complete:
                missing = known_ids - split_ids
                if missing:
                    raise PartitionValidationError(
                        f"Split '{split.name}' does not cover all dataset samples; "
                        f"missing {sorted(missing, key=repr)!r}."
                    )

    def resolve(
        self,
        dataset: DatasetBundle,
        split_name: str,
        *,
        require_complete: bool = True,
    ) -> ResolvedPartition:
        """Resolve one split to identity-preserving dataset subsets."""

        self.validate_against(dataset, require_complete=require_complete)
        split = self.get_split(split_name)

        validation = None
        if split.validation_ids:
            validation = dataset.subset(split.validation_ids)

        test = None
        if split.test_ids:
            test = dataset.subset(split.test_ids)

        return ResolvedPartition(
            name=split.name,
            train=dataset.subset(split.train_ids),
            validation=validation,
            test=test,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_fingerprint": self.dataset_fingerprint,
            "splits": [split.to_dict() for split in self.splits],
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PartitionPlan":
        return cls(
            kind=payload.get("kind", "external"),
            dataset_fingerprint=payload.get("dataset_fingerprint"),
            splits=tuple(
                PartitionSplit.from_dict(split_payload)
                for split_payload in payload.get("splits", ())
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def holdout(
        cls,
        *,
        train_ids: Any,
        test_ids: Any,
        validation_ids: Any = (),
        dataset_fingerprint: str | None = None,
        name: str = "holdout",
        metadata: dict[str, Any] | None = None,
    ) -> "PartitionPlan":
        return cls(
            kind="holdout",
            dataset_fingerprint=dataset_fingerprint,
            splits=(
                PartitionSplit(
                    name=name,
                    train_ids=tuple(train_ids),
                    validation_ids=tuple(validation_ids),
                    test_ids=tuple(test_ids),
                ),
            ),
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_predefined_folds(
        cls,
        *,
        sample_ids: Any,
        fold_assignments: Any,
        always_train_value: Any = -1,
        dataset_fingerprint: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "PartitionPlan":
        """Build explicit CV splits from sklearn-style fold assignments.

        Samples assigned ``always_train_value`` are included in every training
        fold and never used as validation samples.
        """

        ids = tuple(sample_ids)
        assignments = tuple(fold_assignments)
        if len(ids) != len(assignments):
            raise PartitionValidationError(
                "sample_ids and fold_assignments must have identical lengths."
            )
        _validate_unique_ids(ids, split="predefined", role="sample_ids")

        fold_values = []
        for fold in assignments:
            if fold == always_train_value:
                continue
            if fold not in fold_values:
                fold_values.append(fold)

        if not fold_values:
            raise PartitionValidationError(
                "Predefined folds must contain at least one validation fold."
            )

        splits = []
        for fold in fold_values:
            validation_ids = tuple(
                sample_id
                for sample_id, assignment in zip(ids, assignments, strict=True)
                if assignment == fold
            )
            train_ids = tuple(
                sample_id
                for sample_id, assignment in zip(ids, assignments, strict=True)
                if assignment != fold
            )
            splits.append(
                PartitionSplit(
                    name=f"fold_{fold}",
                    train_ids=train_ids,
                    validation_ids=validation_ids,
                )
            )

        plan_metadata = {} if metadata is None else dict(metadata)
        plan_metadata.setdefault("always_train_value", always_train_value)

        return cls(
            kind="predefined",
            dataset_fingerprint=dataset_fingerprint,
            splits=tuple(splits),
            metadata=plan_metadata,
        )

    def validation_counts(self) -> Counter[Any]:
        """Return how often each sample appears in validation roles."""

        counts: Counter[Any] = Counter()
        for split in self.splits:
            counts.update(split.validation_ids)
        return counts


def _validate_unique_ids(values: tuple[Any, ...], *, split: str, role: str) -> None:
    for value in values:
        if not isinstance(value, (str, int, float, bool)):
            raise PartitionValidationError(
                f"Split '{split}' role '{role}' must use scalar str/int/float/bool sample IDs."
            )
        if isinstance(value, float) and not np.isfinite(value):
            raise PartitionValidationError(
                f"Split '{split}' role '{role}' contains a non-finite sample ID."
            )

    try:
        unique = set(values)
    except TypeError as exc:
        raise PartitionValidationError(
            f"Split '{split}' role '{role}' contains non-hashable sample IDs."
        ) from exc

    if len(unique) != len(values):
        raise PartitionValidationError(
            f"Split '{split}' role '{role}' contains duplicate sample IDs."
        )


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
