"""
mlcore.core.trainer
===================

Core training engine for mlcore.

All estimator construction is delegated to ``AlgorithmSpec.build_estimator``.
Legacy backend objects are populated only as compatibility state containers;
they no longer execute model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mlcore.core.prediction import PredictionResult
from mlcore.core.registry import AlgorithmRegistry
from mlcore.core.specs import AlgorithmSpec


@dataclass(slots=True)
class TrainResult:
    """Container for direct-training outputs."""

    model: Any
    backend: Any | None
    spec: AlgorithmSpec

    metrics: dict[str, float] | None = None
    predictions: np.ndarray | None = None
    probabilities: np.ndarray | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class Trainer:
    """Core training engine using the canonical estimator factory."""

    def __init__(self, registry: AlgorithmRegistry) -> None:
        self.registry = registry

    def fit(
        self,
        algorithm: str,
        X: np.ndarray,
        y: np.ndarray,
        *,
        return_predictions: bool = False,
        return_probabilities: bool = False,
        random_state: int | None = None,
        **params: Any,
    ) -> TrainResult:
        """Train a registered algorithm through the unified construction path."""

        spec = self.registry.get(algorithm)

        model = spec.build_estimator(
            random_state=random_state,
            **params,
        )

        model.fit(X, y)

        all_predictions = None
        if hasattr(model, "predict"):
            all_predictions = model.predict(X)

        all_probabilities = None
        if spec.capabilities.predict_proba and hasattr(model, "predict_proba"):
            try:
                all_probabilities = model.predict_proba(X)
            except Exception:
                all_probabilities = None

        resolved_params = _get_model_params(model)
        metadata = _get_model_metadata(
            model=model,
            spec=spec,
        )

        backend = _populate_legacy_backend(
            spec=spec,
            model=model,
            params=resolved_params,
            predictions=all_predictions,
            probabilities=all_probabilities,
            metadata=metadata,
        )

        metrics = None
        if backend is not None and hasattr(backend, "get_metrics"):
            metrics = backend.get_metrics()

        return TrainResult(
            model=model,
            backend=backend,
            spec=spec,
            metrics=metrics,
            predictions=(
                all_predictions if return_predictions else None
            ),
            probabilities=(
                all_probabilities if return_probabilities else None
            ),
            parameters=resolved_params,
            metadata=metadata,
        )

    def evaluate(
        self,
        result: TrainResult,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric_fn: Any,
    ) -> dict[str, float]:
        """Evaluate a trained model with a prediction-based metric."""

        model = result.model

        if model is None:
            raise ValueError("Model is not available for evaluation.")

        predictions = model.predict(X_test)
        score = metric_fn(y_test, predictions)

        return {"score": float(score)}

    def predict(
        self,
        result: TrainResult,
        X: np.ndarray,
    ) -> np.ndarray:
        """Generate predictions."""

        model = result.model

        if model is None:
            raise ValueError("Model is not available.")

        return model.predict(X)

    def predict_proba(
        self,
        result: TrainResult,
        X: np.ndarray,
    ) -> np.ndarray:
        """Generate class probabilities."""

        model = result.model

        if model is None:
            raise ValueError("Model is not available.")

        if not result.spec.capabilities.predict_proba:
            raise ValueError(
                "Model does not support probability prediction."
            )

        return model.predict_proba(X)

    def predict_result(
        self,
        result: TrainResult,
        X: np.ndarray,
        *,
        positive_class: Any | None = None,
    ) -> PredictionResult:
        """Generate structured predictions with explicit response semantics."""

        model = result.model
        if model is None:
            raise ValueError("Model is not available.")

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
            metadata={
                "algorithm": result.spec.name,
                "provider": result.spec.provider,
            },
        )


def _get_model_params(model: Any) -> dict[str, Any]:
    get_params = getattr(model, "get_params", None)
    if not callable(get_params):
        return {}

    try:
        return dict(get_params(deep=False))
    except TypeError:
        return dict(get_params())


def _get_model_metadata(
    *,
    model: Any,
    spec: AlgorithmSpec,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "algorithm": spec.name,
        "provider": spec.provider,
        "task": spec.task,
    }

    if hasattr(model, "classes_"):
        metadata["classes"] = model.classes_

    if hasattr(model, "n_features_in_"):
        metadata["n_features"] = model.n_features_in_

    return metadata


def _populate_legacy_backend(
    *,
    spec: AlgorithmSpec,
    model: Any,
    params: dict[str, Any],
    predictions: np.ndarray | None,
    probabilities: np.ndarray | None,
    metadata: dict[str, Any],
):
    """Populate the legacy backend container without using it for execution."""

    if spec.backend_cls is None:
        return None

    backend = spec.backend_cls()

    if hasattr(backend, "set_model"):
        backend.set_model(model)

    if hasattr(backend, "set_params"):
        backend.set_params(params)

    if predictions is not None and hasattr(backend, "set_predictions"):
        backend.set_predictions(predictions)

    if probabilities is not None and hasattr(backend, "set_probabilities"):
        backend.set_probabilities(probabilities)

    if metadata and hasattr(backend, "set_metadata"):
        backend.set_metadata(metadata)

    if hasattr(backend, "post_fit_hook"):
        backend.post_fit_hook()

    return backend
