"""High-level structured prediction API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from mlcore.api._common import prediction_from_model
from mlcore.core.prediction import PredictionResult
from mlcore.core.trainer import TrainResult
from mlcore.datasets import DatasetBundle
from mlcore.persistence import LoadedModelArtifact, load_model_artifact


def predict(
    model: str | Path | TrainResult | LoadedModelArtifact,
    *,
    dataset: DatasetBundle | None = None,
    X: Any | None = None,
    feature_names: Sequence[str] | None = None,
    sample_ids: Sequence[Any] | None = None,
    positive_class: Any | None = None,
    strict_environment: bool = False,
) -> PredictionResult:
    """Generate a structured PredictionResult from a fitted/persisted model."""

    active = model
    if isinstance(model, (str, Path)):
        active = load_model_artifact(model, strict_environment=strict_environment)

    if dataset is not None:
        X = dataset.X
        feature_names = dataset.feature_names
        sample_ids = dataset.sample_ids

    if isinstance(active, LoadedModelArtifact):
        if X is None:
            raise ValueError("Prediction requires dataset=... or X=....")
        return active.predict_result(
            X,
            feature_names=feature_names,
            sample_ids=sample_ids,
            positive_class=positive_class,
        )

    if isinstance(active, TrainResult):
        return prediction_from_model(
            result=active,
            dataset=dataset,
            X=X,
            sample_ids=sample_ids,
            positive_class=positive_class,
        )

    raise TypeError("model must be an artifact path, LoadedModelArtifact, or TrainResult.")


__all__ = ["predict"]
