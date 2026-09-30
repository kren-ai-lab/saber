"""Shared helpers for the public high-level API."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from saber.core.prediction import PredictionResult, collect_model_outputs
from saber.core.registry import get_algorithm
from saber.core.results import TrainResult
from saber.exceptions import ValidationContractError
from saber.persistence import LoadedModelArtifact, load_model
from saber.preprocessing.pipeline import (
    PreprocessingConfig,
    build_model_pipeline,
    pipeline_input,
    preprocessing_summary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from saber.datasets import DatasetBundle, FeatureSchema


def fit_dataset(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    preprocessing: PreprocessingConfig | None = None,
    random_state: int | None = None,
    model_params: Mapping[str, Any] | None = None,
) -> TrainResult:
    """Fit one final model on all supplied prepared samples."""
    spec = get_algorithm(algorithm)
    dataset.validate(task=spec.task)

    estimator = spec.build_estimator(
        random_state=random_state,
        **dict(model_params or {}),
    )
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=estimator,
        training_data=dataset,
        preprocessing=preprocessing,
    )

    fit_kwargs: dict[str, Any] = {}
    if dataset.sample_weight is not None:
        if not spec.resolved_capabilities.sample_weight:
            raise ValidationContractError(f"Algorithm '{spec.name}' does not support sample weights.")
        fit_kwargs["estimator__sample_weight"] = np.asarray(dataset.sample_weight)

    pipeline.fit(pipeline_input(pipeline, dataset.X), dataset.y, **fit_kwargs)

    estimator_params = {}
    fitted_estimator = pipeline.named_steps.get("estimator")
    if fitted_estimator is not None and hasattr(fitted_estimator, "get_params"):
        estimator_params = dict(fitted_estimator.get_params(deep=False))

    return TrainResult(
        model=pipeline,
        spec=spec,
        parameters=estimator_params,
        feature_schema=dataset.feature_schema,
        metadata={
            "algorithm": spec.name,
            "provider": spec.provider,
            "task": spec.task,
            "dataset_fingerprint": dataset.fingerprint,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "random_state": random_state,
            "preprocessing": preprocessing_summary(preprocessing),
        },
    )


def _validate_feature_schema(
    schema: FeatureSchema | None,
    X: Any,
    *,
    feature_names: Sequence[str] | None = None,
) -> None:
    """Validate X against the training feature schema before the NumPy conversion."""
    if schema is not None:
        schema.validate_compatible(X, feature_names=feature_names)


def load_active_model(model: Any) -> TrainResult | LoadedModelArtifact:
    """Resolve an artifact path, TrainResult, or LoadedModelArtifact to a fitted model."""
    if isinstance(model, (str, Path)):
        return load_model(model)
    if isinstance(model, (TrainResult, LoadedModelArtifact)):
        return model
    raise ValidationContractError("model must be an artifact path, LoadedModelArtifact, or TrainResult.")


def model_task(model: TrainResult | LoadedModelArtifact) -> str:
    """Return the supervised task of a resolved model."""
    return model.spec.task if isinstance(model, TrainResult) else model.task


def predict_model(
    model: TrainResult | LoadedModelArtifact,
    X: Any,
    *,
    feature_names: Sequence[str] | None = None,
    sample_ids: Sequence[Any] | None = None,
    positive_class: Any | None = None,
) -> PredictionResult:
    """Generate a structured prediction from a resolved fitted model."""
    if X is None:
        raise ValidationContractError("Prediction requires a feature matrix or DatasetBundle.")
    if isinstance(model, LoadedModelArtifact):
        return model._predict_result(  # noqa: SLF001  # the artifact's own inference path
            X,
            feature_names=feature_names,
            sample_ids=sample_ids,
            positive_class=positive_class,
        )

    fitted = model.model
    if fitted is None:
        raise ValidationContractError("TrainResult does not contain a fitted model.")

    _validate_feature_schema(model.feature_schema, X, feature_names=feature_names)
    X = pipeline_input(fitted, X)
    capabilities = model.spec.resolved_capabilities
    outputs = collect_model_outputs(
        fitted,
        X,
        task=model.spec.task,
        use_proba=capabilities.predict_proba,
        use_decision=capabilities.decision_function,
    )

    return PredictionResult(
        task=model.spec.task,
        **outputs,
        positive_class=model.positive_class if positive_class is None else positive_class,
        sample_ids=None if sample_ids is None else np.asarray(sample_ids),
        metadata={
            "algorithm": model.spec.name,
            "provider": model.spec.provider,
        },
    )
