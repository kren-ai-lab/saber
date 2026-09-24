"""Shared helpers for the public high-level API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from saber.core.prediction import PredictionResult
from saber.core.registry import MODEL_REGISTRY, AlgorithmRegistry
from saber.core.results import TrainResult
from saber.datasets import DatasetBundle
from saber.exceptions import ValidationContractError
from saber.preprocessing import PreprocessingConfig, build_model_pipeline


def fit_dataset(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    registry: AlgorithmRegistry = MODEL_REGISTRY,
    preprocessing: PreprocessingConfig | Any | None = None,
    random_state: int | None = None,
    model_params: Mapping[str, Any] | None = None,
) -> TrainResult:
    """Fit one final model on all supplied prepared samples."""

    spec = registry.get(algorithm)
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
        if not spec.capabilities.sample_weight:
            raise ValidationContractError(
                f"Algorithm '{spec.name}' does not support sample weights."
            )
        fit_kwargs["estimator__sample_weight"] = np.asarray(dataset.sample_weight)

    pipeline.fit(dataset.X, dataset.y, **fit_kwargs)

    estimator_params = {}
    fitted_estimator = pipeline.named_steps.get("estimator")
    if fitted_estimator is not None and hasattr(fitted_estimator, "get_params"):
        estimator_params = dict(fitted_estimator.get_params(deep=False))

    return TrainResult(
        model=pipeline,
        spec=spec,
        parameters=estimator_params,
        metadata={
            "algorithm": spec.name,
            "provider": spec.provider,
            "task": spec.task,
            "dataset_fingerprint": dataset.fingerprint,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "random_state": random_state,
            "public_api": True,
        },
    )


def prediction_from_model(
    *,
    result: TrainResult,
    dataset: DatasetBundle | None = None,
    X: Any | None = None,
    sample_ids: Sequence[Any] | None = None,
    positive_class: Any | None = None,
) -> PredictionResult:
    """Generate a structured prediction from a high-level TrainResult."""

    if dataset is not None:
        X = dataset.X
        sample_ids = dataset.sample_ids
    if X is None:
        raise ValidationContractError("Prediction requires dataset=... or X=....")

    model = result.model
    if model is None:
        raise ValidationContractError("TrainResult does not contain a fitted model.")

    predictions = np.asarray(model.predict(X))
    probabilities = None
    if (
        result.spec.task == "classification"
        and result.spec.capabilities.predict_proba
        and hasattr(model, "predict_proba")
    ):
        probabilities = np.asarray(model.predict_proba(X))

    decision_scores = None
    if (
        result.spec.task == "classification"
        and result.spec.capabilities.decision_function
        and hasattr(model, "decision_function")
    ):
        decision_scores = np.asarray(model.decision_function(X))

    classes = None
    if result.spec.task == "classification" and hasattr(model, "classes_"):
        classes = np.asarray(model.classes_)

    return PredictionResult(
        task=result.spec.task,
        predictions=predictions,
        probabilities=probabilities,
        decision_scores=decision_scores,
        classes=classes,
        positive_class=positive_class,
        sample_ids=None if sample_ids is None else tuple(sample_ids),
        metadata={
            "algorithm": result.spec.name,
            "provider": result.spec.provider,
            "public_api": True,
        },
    )
