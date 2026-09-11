"""High-level validation and direct evaluation APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mlcore.api._common import prediction_from_model
from mlcore.core.registry import MODEL_REGISTRY, AlgorithmRegistry
from mlcore.core.trainer import TrainResult
from mlcore.datasets import BioSievePartitionConfig, DatasetBundle, PartitionPlan
from mlcore.evaluation import EvaluationResult, evaluate_prediction
from mlcore.persistence import LoadedModelArtifact, load_model_artifact
from mlcore.preprocessing import PreprocessingConfig
from mlcore.validation import ValidationEngine, ValidationResult


def validate(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    partition_plan: PartitionPlan | None = None,
    partitioning: BioSievePartitionConfig | None = None,
    biosieve_extra_columns: Mapping[str, Sequence[Any]] | None = None,
    preprocessing: PreprocessingConfig | Any | None = None,
    metrics: Sequence[str] | None = None,
    evaluation_role: str = "auto",
    positive_class: Any | None = None,
    random_state: int | None = None,
    return_estimators: bool = False,
    require_complete: bool = True,
    model_params: Mapping[str, Any] | None = None,
    registry: AlgorithmRegistry = MODEL_REGISTRY,
) -> ValidationResult:
    """Validate a registered model using explicit/BioSieve partitions."""

    return ValidationEngine(registry).run(
        dataset=dataset,
        algorithm=algorithm,
        partition_plan=partition_plan,
        partitioning=partitioning,
        biosieve_extra_columns=biosieve_extra_columns,
        preprocessing=preprocessing,
        metrics=metrics,
        evaluation_role=evaluation_role,
        positive_class=positive_class,
        random_state=random_state,
        return_estimators=return_estimators,
        require_complete=require_complete,
        **dict(model_params or {}),
    )


def evaluate(
    *,
    dataset: DatasetBundle,
    model: str | Path | TrainResult | LoadedModelArtifact,
    metrics: Sequence[str] | None = None,
    positive_class: Any | None = None,
) -> EvaluationResult:
    """Evaluate one already-fitted model on a labeled prepared dataset."""

    if isinstance(model, (str, Path)):
        model = load_model_artifact(model)

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


__all__ = ["evaluate", "validate"]
