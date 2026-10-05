"""High-level direct evaluation API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saber._api._common import load_active_model, predict_model
from saber.core.results import TrainResult
from saber.evaluation import EvaluationResult, evaluate_prediction

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from saber.datasets import DatasetBundle
    from saber.persistence import LoadedModelArtifact


def evaluate(
    model: str | Path | TrainResult | LoadedModelArtifact,
    dataset: DatasetBundle,
    *,
    metrics: Sequence[str] | None = None,
    positive_class: Any | None = None,
) -> EvaluationResult:
    """Evaluate one already-fitted model on a labeled prepared dataset."""
    active = load_active_model(model)
    dataset.validate(task=active.spec.task if isinstance(active, TrainResult) else active.task)
    prediction = predict_model(
        active,
        dataset.X,
        feature_names=dataset.feature_names,
        sample_ids=dataset.sample_ids,
        positive_class=positive_class,
    )
    result = evaluate_prediction(dataset.y, prediction, metrics=metrics)
    result.metadata.update(
        algorithm=prediction.metadata.get("algorithm"), dataset_fingerprint=dataset.fingerprint
    )
    return result


__all__ = ["evaluate"]
