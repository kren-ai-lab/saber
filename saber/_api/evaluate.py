"""High-level direct evaluation API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from saber._api._common import prediction_from_model
from saber.core.results import TrainResult
from saber.evaluation import EvaluationResult, evaluate_prediction
from saber.persistence import LoadedModelArtifact, load_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from saber.datasets import DatasetBundle


def evaluate(
    *,
    dataset: DatasetBundle,
    model: str | Path | TrainResult | LoadedModelArtifact,
    metrics: Sequence[str] | None = None,
    positive_class: Any | None = None,
) -> EvaluationResult:
    """Evaluate one already-fitted model on a labeled prepared dataset."""
    if isinstance(model, (str, Path)):
        model = load_model(model)

    if isinstance(model, LoadedModelArtifact):
        dataset.validate(task=model.task)
        prediction = model.predict_result(
            dataset.X,
            feature_names=dataset.feature_names,
            sample_ids=dataset.sample_ids,
            positive_class=positive_class,
        )
    elif isinstance(model, TrainResult):
        dataset.validate(task=model.spec.task)
        prediction = prediction_from_model(
            result=model,
            dataset=dataset,
            positive_class=positive_class,
        )
    else:
        raise TypeError("model must be TrainResult or LoadedModelArtifact.")

    return evaluate_prediction(dataset.y, prediction, metrics=metrics)


__all__ = ["evaluate"]
