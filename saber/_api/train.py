"""High-level final-model training API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saber.core.registry import get_algorithm
from saber.core.results import TrainResult
from saber.preprocessing.pipeline import build_model_pipeline, pipeline_input, preprocessing_summary

if TYPE_CHECKING:
    from collections.abc import Mapping

    from saber.datasets import DatasetBundle
    from saber.preprocessing import PreprocessingConfig


def train(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    model_params: Mapping[str, Any] | None = None,
    preprocessing: PreprocessingConfig | None = None,
    random_state: int | None = None,
    positive_class: Any | None = None,
) -> TrainResult:
    """Fit a final supervised model on all supplied samples.

    This is a final-fit operation.  Model assessment belongs to ``validate`` or
    ``benchmark``; hyperparameter selection belongs to ``tune``.  Persist the
    result with :func:`saber.save_model`.
    """
    spec = get_algorithm(algorithm)
    dataset.validate(task=spec.task)

    estimator = spec.build_estimator(random_state=random_state, **(model_params or {}))
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=estimator,
        training_data=dataset,
        preprocessing=preprocessing,
    )
    pipeline.fit(
        pipeline_input(pipeline, dataset.X),
        dataset.y,
        **spec.sample_weight_fit_params(dataset.sample_weight),
    )

    return TrainResult(
        model=pipeline,
        spec=spec,
        parameters=dict(pipeline.named_steps["estimator"].get_params(deep=False)),
        feature_schema=dataset.feature_schema,
        positive_class=positive_class,
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


__all__ = ["train"]
