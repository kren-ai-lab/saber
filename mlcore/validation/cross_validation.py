"""Partition-driven leakage-safe model validation."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from mlcore.core.prediction import PredictionResult
from mlcore.core.registry import AlgorithmRegistry
from mlcore.datasets import DatasetBundle, PartitionPlan
from mlcore.datasets.biosieve import BioSievePartitionConfig
from mlcore.evaluation import evaluate_prediction
from mlcore.exceptions import ValidationContractError
from mlcore.preprocessing import PreprocessingConfig, build_model_pipeline
from mlcore.validation.partitioning import (
    EvaluationRole,
    resolve_evaluation_dataset,
    resolve_partition_plan,
)
from mlcore.validation.results import (
    FoldValidationResult,
    ValidationResult,
    aggregate_fold_metrics,
)



class ValidationEngine:
    """Execute explicit partition plans with fold-local preprocessing."""

    def __init__(self, registry: AlgorithmRegistry) -> None:
        self.registry = registry

    def run(
        self,
        *,
        dataset: DatasetBundle,
        algorithm: str,
        partition_plan: PartitionPlan | None = None,
        partitioning: BioSievePartitionConfig | None = None,
        biosieve_extra_columns: Mapping[str, Sequence[Any]] | None = None,
        preprocessing: PreprocessingConfig | Any | None = None,
        metrics: Sequence[str] | None = None,
        evaluation_role: EvaluationRole = "auto",
        positive_class: Any | None = None,
        random_state: int | None = None,
        return_estimators: bool = True,
        require_complete: bool = True,
        **model_params: Any,
    ) -> ValidationResult:
        """Validate one registered algorithm over an explicit/generated plan."""

        spec = self.registry.get(algorithm)
        dataset.validate(task=spec.task)

        plan = resolve_partition_plan(
            dataset=dataset,
            partition_plan=partition_plan,
            partitioning=partitioning,
            biosieve_extra_columns=biosieve_extra_columns,
        )
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

            estimator = spec.build_estimator(
                random_state=random_state,
                **model_params,
            )
            pipeline = build_model_pipeline(
                spec=spec,
                estimator=estimator,
                training_data=resolved.train,
                preprocessing=preprocessing,
            )

            fit_kwargs: dict[str, Any] = {}
            if resolved.train.sample_weight is not None:
                if not spec.capabilities.sample_weight:
                    raise ValidationContractError(
                        f"Dataset supplies sample_weight but algorithm '{spec.name}' "
                        "does not advertise sample-weight support."
                    )
                fit_kwargs["estimator__sample_weight"] = np.asarray(
                    resolved.train.sample_weight,
                    dtype=float,
                )

            start = perf_counter()
            pipeline.fit(resolved.train.X, resolved.train.y, **fit_kwargs)
            fit_seconds = perf_counter() - start

            prediction = _prediction_from_pipeline(
                pipeline,
                spec_task=spec.task,
                X=evaluation_data.X,
                sample_ids=evaluation_data.sample_ids,
                positive_class=positive_class,
                algorithm=spec.name,
                provider=spec.provider,
            )
            evaluation = evaluate_prediction(
                evaluation_data.y,
                prediction,
                metrics=metrics,
            )

            folds.append(
                FoldValidationResult(
                    split_name=split.name,
                    evaluation_role=role,
                    train_ids=tuple(resolved.train.sample_ids),
                    evaluation_ids=tuple(evaluation_data.sample_ids),
                    prediction=prediction,
                    evaluation=evaluation,
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
        aggregate_metrics, metric_summary = aggregate_fold_metrics(fold_tuple)
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
            metric_summary=metric_summary,
            oof_prediction=oof_prediction,
            metadata={
                "partition_source": plan.metadata.get("source", "external"),
                "partition_fingerprint": plan.fingerprint,
                "dataset_fingerprint": dataset.fingerprint,
                **oof_metadata,
            },
        )


def validate_model(
    registry: AlgorithmRegistry,
    **kwargs: Any,
) -> ValidationResult:
    """Functional convenience wrapper around :class:`ValidationEngine`."""

    return ValidationEngine(registry).run(**kwargs)


def _prediction_from_pipeline(
    pipeline: Any,
    *,
    spec_task: str,
    X: Any,
    sample_ids: Sequence[Any],
    positive_class: Any | None,
    algorithm: str,
    provider: str,
) -> PredictionResult:
    predictions = np.asarray(pipeline.predict(X))

    probabilities = None
    if spec_task == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probabilities = np.asarray(pipeline.predict_proba(X))
        except (AttributeError, NotImplementedError):
            probabilities = None

    decision_scores = None
    if spec_task == "classification" and hasattr(pipeline, "decision_function"):
        try:
            decision_scores = np.asarray(pipeline.decision_function(X))
        except (AttributeError, NotImplementedError):
            decision_scores = None

    classes = None
    if spec_task == "classification" and hasattr(pipeline, "classes_"):
        classes = np.asarray(pipeline.classes_)

    return PredictionResult(
        task=spec_task,
        predictions=predictions,
        probabilities=probabilities,
        decision_scores=decision_scores,
        classes=classes,
        positive_class=positive_class,
        sample_ids=np.asarray(sample_ids, dtype=object),
        metadata={
            "algorithm": algorithm,
            "provider": provider,
        },
    )


def _build_oof_prediction(
    *,
    dataset: DatasetBundle,
    folds: tuple[FoldValidationResult, ...],
    task: str,
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

    ordered_ids = [sample_id for sample_id in dataset.sample_ids if sample_id in prediction_by_id]
    predictions = np.asarray(
        [
            prediction_by_id[sample_id][0].prediction.predictions[
                prediction_by_id[sample_id][1]
            ]
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
    if all(value is None for value in values):
        return None
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
