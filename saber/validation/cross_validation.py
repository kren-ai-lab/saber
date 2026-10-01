"""Partition-driven leakage-safe model validation."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from saber.core.prediction import PredictionResult, collect_model_outputs
from saber.core.registry import get_algorithm
from saber.evaluation import evaluate_prediction
from saber.exceptions import DatasetValidationError, ValidationContractError
from saber.preprocessing.pipeline import PreprocessingConfig, build_model_pipeline, pipeline_input
from saber.validation.partitioning import (
    EvaluationRole,
    resolve_evaluation_dataset,
    resolve_partition_plan,
)
from saber.validation.results import (
    FoldValidationResult,
    ValidationResult,
    aggregate_fold_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from saber.core.task import TaskType
    from saber.datasets import DatasetBundle, PartitionPlan
    from saber.datasets.biosieve import BioSievePartitionConfig


def validate(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    model_params: Mapping[str, Any] | None = None,
    preprocessing: PreprocessingConfig | None = None,
    partition_plan: PartitionPlan | BioSievePartitionConfig | None = None,
    metrics: Sequence[str] | None = None,
    evaluation_role: EvaluationRole = "auto",
    require_complete: bool = True,
    positive_class: Any | None = None,
    return_estimators: bool = False,
    random_state: int | None = None,
) -> ValidationResult:
    """Validate one registered algorithm over an explicit or BioSieve-generated plan.

    ``partition_plan`` is either an explicit :class:`PartitionPlan` or a
    :class:`BioSievePartitionConfig` that delegates split generation to BioSieve.
    """
    spec = get_algorithm(algorithm)
    dataset.validate(task=spec.task)

    plan = resolve_partition_plan(dataset=dataset, partition_plan=partition_plan)
    plan.validate_against(dataset, require_complete=require_complete)

    folds: list[FoldValidationResult] = []
    for split in plan.splits:
        resolved = plan.resolve(
            dataset,
            split.name,
            require_complete=require_complete,
        )
        role, evaluation_data = resolve_evaluation_dataset(
            resolved,
            requested=evaluation_role,
            plan_kind=plan.kind,
        )

        try:
            resolved.train.validate(task=spec.task)
        except DatasetValidationError as exc:
            raise ValidationContractError(
                f"Training membership for split '{split.name}' is invalid for task '{spec.task}': {exc}"
            ) from exc

        estimator = spec.build_estimator(
            random_state=random_state,
            **(model_params or {}),
        )
        pipeline = build_model_pipeline(
            spec=spec,
            estimator=estimator,
            training_data=resolved.train,
            preprocessing=preprocessing,
        )

        fit_kwargs = spec.sample_weight_fit_params(resolved.train.sample_weight)

        start = perf_counter()
        pipeline.fit(pipeline_input(pipeline, resolved.train.X), resolved.train.y, **fit_kwargs)
        fit_seconds = perf_counter() - start

        outputs = collect_model_outputs(
            pipeline, pipeline_input(pipeline, evaluation_data.X), task=spec.task, tolerant=True
        )
        prediction = PredictionResult(
            task=spec.task,
            **outputs,
            positive_class=positive_class,
            sample_ids=np.asarray(evaluation_data.sample_ids, dtype=object),
            metadata={"algorithm": spec.name, "provider": spec.provider},
        )
        evaluation = evaluate_prediction(
            evaluation_data.y,
            prediction,
            metrics=metrics,
        )

        folds.append(
            FoldValidationResult(
                split=split.name,
                evaluation_role=role,
                train_ids=tuple(resolved.train.resolved_sample_ids),
                evaluation_ids=tuple(evaluation_data.sample_ids),
                prediction=prediction,
                metrics=evaluation.metrics,
                estimator=pipeline if return_estimators else None,
                fit_seconds=float(fit_seconds),
                metadata={
                    "partition_metadata": dict(split.metadata),
                    "n_train": resolved.train.n_samples,
                    "n_evaluation": evaluation_data.n_samples,
                },
            )
        )

    fold_tuple = tuple(folds)
    aggregate_metrics = {name: stats["mean"] for name, stats in aggregate_fold_metrics(fold_tuple).items()}
    oof_prediction, oof_metadata = _build_oof_prediction(
        dataset=dataset,
        folds=fold_tuple,
        task=spec.task,
    )

    return ValidationResult(
        algorithm=spec.name,
        task=spec.task,
        partition_plan=plan,
        folds=fold_tuple,
        aggregate_metrics=aggregate_metrics,
        oof_prediction=oof_prediction,
        metadata={
            "partition_source": plan.metadata.get("source", "external"),
            "partition_fingerprint": plan.fingerprint,
            "dataset_fingerprint": dataset.fingerprint,
            **oof_metadata,
        },
    )


def _build_oof_prediction(
    *,
    dataset: DatasetBundle,
    folds: tuple[FoldValidationResult, ...],
    task: TaskType,
) -> tuple[PredictionResult | None, dict[str, Any]]:
    ids = [sample_id for fold in folds for sample_id in fold.evaluation_ids]
    if len(ids) != len(set(ids)):
        return None, {
            "oof_available": False,
            "oof_reason": "held-out sample IDs repeat across splits",
        }

    if not ids:
        return None, {
            "oof_available": False,
            "oof_reason": "no held-out predictions",
        }

    prediction_by_id: dict[Any, tuple[FoldValidationResult, int]] = {}
    for fold in folds:
        for local_index, sample_id in enumerate(fold.evaluation_ids):
            prediction_by_id[sample_id] = (fold, local_index)

    ordered_ids = [sample_id for sample_id in dataset.resolved_sample_ids if sample_id in prediction_by_id]
    predictions = np.asarray(
        [
            prediction_by_id[sample_id][0].prediction.predictions[prediction_by_id[sample_id][1]]
            for sample_id in ordered_ids
        ]
    )

    classes = _consistent_classes(folds)
    positive_class = _consistent_positive_class(folds)
    probabilities = _stack_optional_response(
        folds,
        ordered_ids,
        prediction_by_id,
        attribute="probabilities",
    )
    decision_scores = _stack_optional_response(
        folds,
        ordered_ids,
        prediction_by_id,
        attribute="decision_scores",
    )

    result = PredictionResult(
        task=task,
        predictions=predictions,
        probabilities=probabilities,
        decision_scores=decision_scores,
        classes=classes,
        positive_class=positive_class,
        sample_ids=np.asarray(ordered_ids, dtype=object),
        metadata={"kind": "out_of_fold"},
    )
    return result, {
        "oof_available": True,
        "oof_n_samples": len(ordered_ids),
        "oof_coverage": float(len(ordered_ids) / dataset.n_samples),
        "oof_complete": len(ordered_ids) == dataset.n_samples,
    }


def _consistent_classes(folds: tuple[FoldValidationResult, ...]) -> np.ndarray | None:
    values = [fold.prediction.classes for fold in folds]
    if any(value is None for value in values):
        return None
    first = np.asarray(values[0])
    if any(not np.array_equal(first, np.asarray(value)) for value in values[1:]):
        raise ValidationContractError(
            "Cannot aggregate OOF probabilities because fitted class order differs across folds."
        )
    return first


def _consistent_positive_class(folds: tuple[FoldValidationResult, ...]) -> Any | None:
    values = [fold.prediction.positive_class for fold in folds]
    non_null = [value for value in values if value is not None]
    if not non_null:
        return None
    first = non_null[0]
    if any(value != first for value in non_null[1:]):
        raise ValidationContractError(
            "Cannot aggregate OOF predictions with inconsistent positive_class values."
        )
    return first


def _stack_optional_response(
    folds: tuple[FoldValidationResult, ...],
    ordered_ids: list[Any],
    prediction_by_id: dict[Any, tuple[FoldValidationResult, int]],
    *,
    attribute: str,
) -> np.ndarray | None:
    if any(getattr(fold.prediction, attribute) is None for fold in folds):
        return None

    rows = []
    for sample_id in ordered_ids:
        fold, local_index = prediction_by_id[sample_id]
        values = np.asarray(getattr(fold.prediction, attribute))
        rows.append(values[local_index])

    array = np.asarray(rows)
    if array.ndim == 2 and array.shape[1] == 1:
        return array[:, 0]
    return array
