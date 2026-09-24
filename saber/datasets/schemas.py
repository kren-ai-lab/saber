"""Dataset and feature-schema contracts for saber."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from saber.datasets._fingerprint import (
    dataset_fingerprint,
    feature_schema_fingerprint,
)
from saber.datasets.validation import (
    validate_feature_matrix,
    validate_groups,
    validate_sample_ids,
    validate_sample_weight,
    validate_target,
    validate_target_for_task,
)
from saber.exceptions import DatasetValidationError, FeatureSchemaMismatchError
from saber.utils.tabular import as_frame, canonical_dtype


@dataclass(frozen=True)
class FeatureSchema:
    """Ordered numerical feature schema associated with a dataset."""

    names: tuple[str, ...]
    dtypes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that names and dtypes are non-empty, aligned, and unique."""
        if not self.names:
            raise DatasetValidationError("FeatureSchema must contain at least one feature.")
        if len(self.names) != len(self.dtypes):
            raise DatasetValidationError("FeatureSchema names and dtypes must have identical lengths.")
        if len(set(self.names)) != len(self.names):
            raise DatasetValidationError("FeatureSchema names must be unique.")
        if any(not name for name in self.names):
            raise DatasetValidationError("FeatureSchema names cannot be empty.")

    @property
    def n_features(self) -> int:
        """Return the number of features in the schema."""
        return len(self.names)

    @property
    def fingerprint(self) -> str:
        """Return a content fingerprint of the feature names and dtypes."""
        return feature_schema_fingerprint(names=self.names, dtypes=self.dtypes)

    def to_dict(self) -> dict[str, Any]:
        """Return the schema as a plain JSON-serializable dictionary."""
        return {
            "names": list(self.names),
            "dtypes": list(self.dtypes),
            "n_features": self.n_features,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_data(
        cls,
        X: Any,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> FeatureSchema:
        """Infer a feature schema from a feature matrix and optional names."""
        X = as_frame(X)
        _, n_features = validate_feature_matrix(X)

        if feature_names is None:
            if isinstance(X, pl.DataFrame):
                names = tuple(str(column) for column in X.columns)
            else:
                names = tuple(f"feature_{index}" for index in range(n_features))
        else:
            names = tuple(str(name) for name in feature_names)
            if len(names) != n_features:
                raise DatasetValidationError(
                    f"feature_names must contain {n_features} names; received {len(names)}."
                )

        if len(set(names)) != len(names):
            raise DatasetValidationError("feature_names must be unique.")

        if isinstance(X, pl.DataFrame):
            dtypes = tuple(canonical_dtype(dtype) for dtype in X.dtypes)
        else:
            dtype = str(np.asarray(X).dtype)
            dtypes = tuple(dtype for _ in range(n_features))

        return cls(names=names, dtypes=dtypes)

    def validate_compatible(
        self,
        X: Any,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
        check_dtypes: bool = False,
    ) -> None:
        """Validate that another feature matrix matches this schema."""
        candidate = FeatureSchema.from_data(X, feature_names=feature_names)
        if candidate.names != self.names:
            raise FeatureSchemaMismatchError("Feature names/order do not match the expected training schema.")
        if check_dtypes and candidate.dtypes != self.dtypes:
            raise FeatureSchemaMismatchError("Feature dtypes do not match the expected training schema.")


@dataclass
class DatasetBundle:
    """Explicit supervised dataset with stable sample and feature identity."""

    X: Any
    y: Any
    sample_ids: Any | None = None
    feature_names: Any | None = None
    groups: Any | None = None
    sample_weight: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _feature_schema: FeatureSchema = field(init=False, repr=False)
    _generated_sample_ids: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize the dataset's features, target, and identifiers."""
        self.X = as_frame(self.X)
        n_samples, _ = validate_feature_matrix(self.X)
        self.y = validate_target(self.y, n_samples=n_samples)

        self._feature_schema = FeatureSchema.from_data(
            self.X,
            feature_names=self.feature_names,
        )
        self.feature_names = self._feature_schema.names

        self._generated_sample_ids = self.sample_ids is None
        if self.sample_ids is None:
            self.sample_ids = tuple(range(n_samples))
        else:
            self.sample_ids = validate_sample_ids(self.sample_ids, n_samples=n_samples)

        if self.groups is not None:
            self.groups = validate_groups(self.groups, n_samples=n_samples)

        if self.sample_weight is not None:
            self.sample_weight = validate_sample_weight(
                self.sample_weight,
                n_samples=n_samples,
            )

        self.metadata = dict(self.metadata)

    @property
    def n_samples(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.y)

    @property
    def n_features(self) -> int:
        """Return the number of features in the dataset."""
        return self._feature_schema.n_features

    @property
    def feature_schema(self) -> FeatureSchema:
        """Return the dataset's feature schema."""
        return self._feature_schema

    @property
    def generated_sample_ids(self) -> bool:
        """Return whether sample identifiers were auto-generated."""
        return self._generated_sample_ids

    @property
    def fingerprint(self) -> str:
        """Return a content fingerprint excluding free-form metadata."""
        return dataset_fingerprint(
            X=self.X,
            y=self.y,
            sample_ids=self.sample_ids,
            feature_names=self.feature_names,
            groups=self.groups,
            sample_weight=self.sample_weight,
        )

    def validate(
        self,
        *,
        task: str | None = None,
        require_finite_features: bool = False,
    ) -> str | None:
        """Revalidate dataset integrity and optional task semantics."""
        n_samples, n_features = validate_feature_matrix(
            self.X,
            require_finite=require_finite_features,
        )
        if n_samples != self.n_samples or n_features != self.n_features:
            raise DatasetValidationError("Dataset X was structurally modified after DatasetBundle creation.")
        validate_target(self.y, n_samples=n_samples)
        validate_sample_ids(self.sample_ids, n_samples=n_samples)

        if task is None:
            return None
        return validate_target_for_task(self.y, task)

    def indices_for(self, sample_ids: Any) -> np.ndarray:
        """Return positions for identifiers in original dataset order."""
        requested = tuple(sample_ids)
        requested_set = set(requested)
        known = set(self.sample_ids)
        unknown = requested_set - known
        if unknown:
            formatted = ", ".join(repr(value) for value in sorted(unknown, key=repr))
            raise DatasetValidationError(f"Unknown sample_ids requested: {formatted}.")
        if len(requested_set) != len(requested):
            raise DatasetValidationError("Requested sample_ids must be unique.")

        return np.asarray(
            [index for index, sample_id in enumerate(self.sample_ids) if sample_id in requested_set],
            dtype=int,
        )

    def subset(self, sample_ids: Any) -> DatasetBundle:
        """Create a sample-identity-preserving subset in original dataset order."""
        indices = self.indices_for(sample_ids)

        X_subset = self.X[indices] if isinstance(self.X, pl.DataFrame) else np.asarray(self.X)[indices].copy()

        y_subset = np.asarray(self.y)[indices].copy()
        ids_subset = tuple(self.sample_ids[index] for index in indices)

        groups_subset = None
        if self.groups is not None:
            groups_subset = tuple(self.groups[index] for index in indices)

        weights_subset = None
        if self.sample_weight is not None:
            weights_subset = np.asarray(self.sample_weight)[indices].copy()

        metadata = dict(self.metadata)
        metadata.setdefault("parent_dataset_fingerprint", self.fingerprint)

        return DatasetBundle(
            X=X_subset,
            y=y_subset,
            sample_ids=ids_subset,
            feature_names=self.feature_names,
            groups=groups_subset,
            sample_weight=weights_subset,
            metadata=metadata,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return serialization-friendly structural metadata."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "sample_ids_generated": self.generated_sample_ids,
            "has_groups": self.groups is not None,
            "has_sample_weight": self.sample_weight is not None,
            "feature_schema": self.feature_schema.to_dict(),
            "fingerprint": self.fingerprint,
            "metadata": dict(self.metadata),
        }
