"""Loaded persistence-artifact objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from saber.core.prediction import PredictionResult
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
        return self.provenance.get("algorithm")

    @property
    def task(self) -> str:
        return str(self.provenance["task"])

    @property
    def positive_class(self) -> Any | None:
        return self.provenance.get("positive_class")

    def validate_features(
        self,
        X: Any,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
        check_dtypes: bool = False,
    ) -> None:
        self.feature_schema.validate_compatible(
            X,
            feature_names=feature_names,
            check_dtypes=check_dtypes,
        )

    def predict(
        self,
        X: Any,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        self.validate_features(X, feature_names=feature_names)
        return np.asarray(self.model.predict(X))

    def predict_result(
        self,
        X: Any,
        *,
        feature_names: tuple[str, ...] | list[str] | None = None,
        sample_ids: Any | None = None,
        positive_class: Any | None = None,
    ) -> PredictionResult:
        self.validate_features(X, feature_names=feature_names)
        predictions = np.asarray(self.model.predict(X))

        probabilities = None
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            try:
                probabilities = np.asarray(self.model.predict_proba(X))
            except (AttributeError, NotImplementedError):
                probabilities = None

        decision_scores = None
        if self.task == "classification" and hasattr(self.model, "decision_function"):
            try:
                decision_scores = np.asarray(self.model.decision_function(X))
            except (AttributeError, NotImplementedError):
                decision_scores = None

        classes = None
        if self.task == "classification" and hasattr(self.model, "classes_"):
            classes = np.asarray(self.model.classes_)

        ids = None if sample_ids is None else np.asarray(sample_ids, dtype=object)
        return PredictionResult(
            task=self.task,
            predictions=predictions,
            probabilities=probabilities,
            decision_scores=decision_scores,
            classes=classes,
            positive_class=(self.positive_class if positive_class is None else positive_class),
            sample_ids=ids,
            metadata={
                "artifact_path": str(self.path),
                "algorithm": self.algorithm,
                "provider": self.provenance.get("provider"),
            },
        )


@dataclass(slots=True)
class LoadedBenchmarkArtifact:
    """Analysis-ready benchmark tables and optional serialized result object."""

    path: Path
    manifest: ArtifactManifest
    metadata: dict[str, Any]
    environment: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    result: Any | None = None
    compatibility_warnings: tuple[str, ...] = ()

    def table(self, name: str) -> pd.DataFrame:
        try:
            return self.tables[name].copy()
        except KeyError as exc:
            raise KeyError(f"Benchmark artifact does not contain table '{name}'.") from exc
