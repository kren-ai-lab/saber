"""High-level final-model training API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saber._api._common import fit_dataset
from saber.persistence import save_model

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from saber.core.results import TrainResult
    from saber.datasets import DatasetBundle, PartitionPlan
    from saber.preprocessing import PreprocessingConfig


def train(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    preprocessing: PreprocessingConfig | None = None,
    random_state: int | None = None,
    model_params: Mapping[str, Any] | None = None,
    artifact_path: str | Path | None = None,
    partition_plan: PartitionPlan | None = None,
    artifact_overwrite: bool = False,
    metadata: Mapping[str, Any] | None = None,
    positive_class: Any | None = None,
) -> TrainResult:
    """Fit a final supervised model and optionally persist it.

    This is a final-fit operation.  Model assessment belongs to ``validate`` or
    ``benchmark``; hyperparameter selection belongs to ``tune``.
    """
    result = fit_dataset(
        dataset=dataset,
        algorithm=algorithm,
        preprocessing=preprocessing,
        random_state=random_state,
        model_params=model_params,
    )

    result.positive_class = positive_class

    if artifact_path is not None:
        save_model(
            artifact_path,
            model=result.model,
            algorithm=result.spec.name,
            task=result.spec.task,
            dataset=dataset,
            partition_plan=partition_plan,
            provider=result.spec.provider,
            parameters=result.parameters,
            training_config={
                "random_state": random_state,
                "preprocessing": _preprocessing_metadata(preprocessing),
            },
            positive_class=positive_class,
            metadata=dict(metadata or {}),
            overwrite=artifact_overwrite,
        )

    return result


def _preprocessing_metadata(preprocessing: PreprocessingConfig | None) -> dict[str, Any]:
    if preprocessing is None:
        return {"imputation": "auto", "scaler": "auto"}
    if preprocessing.transformer is not None:
        return {"custom_transformer": type(preprocessing.transformer).__name__}
    return {
        "imputation": preprocessing.imputation,
        "scaler": preprocessing.scaler,
        "fill_value": preprocessing.fill_value,
    }


__all__ = ["train"]
