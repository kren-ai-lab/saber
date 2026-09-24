"""High-level hyperparameter optimization API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from saber.core import SearchSpace
from saber.core.registry import MODEL_REGISTRY, AlgorithmRegistry
from saber.datasets import BioSievePartitionConfig, DatasetBundle, PartitionPlan
from saber.preprocessing import PreprocessingConfig
from saber.tuning import OptimizationResult, TuningConfig, TuningEngine


def tune(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    config: TuningConfig,
    partition_plan: PartitionPlan | None = None,
    partitioning: BioSievePartitionConfig | None = None,
    biosieve_extra_columns: Mapping[str, Sequence[Any]] | None = None,
    preprocessing: PreprocessingConfig | Any | None = None,
    search_space: SearchSpace | None = None,
    evaluation_role: str = "auto",
    require_complete: bool = True,
    model_params: Mapping[str, Any] | None = None,
    registry: AlgorithmRegistry = MODEL_REGISTRY,
) -> OptimizationResult:
    """Optimize a model on explicit/BioSieve partitions."""

    return TuningEngine(registry).run(
        dataset=dataset,
        algorithm=algorithm,
        config=config,
        partition_plan=partition_plan,
        partitioning=partitioning,
        biosieve_extra_columns=biosieve_extra_columns,
        preprocessing=preprocessing,
        search_space=search_space,
        evaluation_role=evaluation_role,
        require_complete=require_complete,
        model_params=model_params,
    )


optimize = tune

__all__ = ["optimize", "tune"]
