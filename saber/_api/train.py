"""High-level final-model training API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from saber._api._common import fit_dataset

if TYPE_CHECKING:
    from collections.abc import Mapping

    from saber.core.results import TrainResult
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
    result = fit_dataset(
        dataset=dataset,
        algorithm=algorithm,
        preprocessing=preprocessing,
        random_state=random_state,
        model_params=model_params,
    )
    result.positive_class = positive_class
    return result


__all__ = ["train"]
