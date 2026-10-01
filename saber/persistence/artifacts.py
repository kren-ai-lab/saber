"""Loaded persistence-artifact objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from saber.core.prediction import PredictionResult, collect_model_outputs
from saber.preprocessing.pipeline import pipeline_input

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import polars as pl

    from saber.core.task import TaskType
    from saber.datasets import FeatureSchema
    from saber.persistence.metadata import ArtifactManifest


@dataclass(slots=True)
class LoadedModelArtifact:
    """Validated fitted model/pipeline plus inference provenance."""

    path: Path
    model: Any
    manifest: ArtifactManifest
    feature_schema: FeatureSchema
    provenance: dict[str, Any]
    environment: dict[str, Any]
    compatibility_warnings: tuple[str, ...] = ()

    @property
    def algorithm(self) -> str | None:
        """Return the algorithm name recorded in the artifact provenance."""
        return self.provenance.get("algorithm")

    @property
    def task(self) -> str:
        """Return the supervised task recorded in the artifact provenance."""
        return str(self.provenance["task"])

    @property
    def positive_class(self) -> Any | None:
        """Return the positive class recorded in the artifact provenance."""
        return self.provenance.get("positive_class")

    def validate_features(
        self,
        X: Any,
        *,
        feature_names: Sequence[str] | None = None,
        check_dtypes: bool = False,
    ) -> None:
        """Validate that ``X`` is compatible with the artifact's feature schema."""
        self.feature_schema.validate_compatible(
            X,
            feature_names=feature_names,
            check_dtypes=check_dtypes,
        )

    def predict(
        self,
        X: Any,
        *,
        feature_names: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Validate ``X`` and return the model's predictions."""
        self.validate_features(X, feature_names=feature_names)
        return np.asarray(self.model.predict(pipeline_input(self.model, X)))

    def _predict_result(
        self,
        X: Any,
        *,
        feature_names: Sequence[str] | None = None,
        sample_ids: Any | None = None,
        positive_class: Any | None = None,
    ) -> PredictionResult:
        """Validate ``X`` and return a structured prediction result."""
        self.validate_features(X, feature_names=feature_names)
        X = pipeline_input(self.model, X)
        outputs = collect_model_outputs(self.model, X, task=self.task, tolerant=True)

        ids = None if sample_ids is None else np.asarray(sample_ids, dtype=object)
        return PredictionResult(
            # The persisted manifest always records "classification"/"regression";
            # pyrefly can't see that invariant through the deserialized dict.
            task=cast("TaskType", self.task),
            **outputs,
            positive_class=(self.positive_class if positive_class is None else positive_class),
            sample_ids=ids,
            metadata={"algorithm": self.algorithm},
        )


@dataclass(slots=True)
class LoadedBenchmarkArtifact:
    """Analysis-ready benchmark tables and optional serialized result object."""

    path: Path
    manifest: ArtifactManifest
    metadata: dict[str, Any]
    environment: dict[str, Any]
    tables: dict[str, pl.DataFrame] = field(default_factory=dict)
    result: Any | None = None
    compatibility_warnings: tuple[str, ...] = ()

    def table(self, name: str) -> pl.DataFrame:
        """Return a clone of the named benchmark table."""
        try:
            return self.tables[name].clone()
        except KeyError as exc:
            raise KeyError(f"Benchmark artifact does not contain table '{name}'.") from exc
