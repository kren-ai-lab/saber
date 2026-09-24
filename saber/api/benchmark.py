"""High-level benchmark API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from saber.benchmark import BenchmarkConfig, BenchmarkDataset, BenchmarkEngine, BenchmarkPartition, BenchmarkResult
from saber.core.registry import MODEL_REGISTRY, AlgorithmRegistry
from saber.datasets import BioSievePartitionConfig, DatasetBundle, PartitionPlan
from saber.preprocessing import PreprocessingConfig


def benchmark(
    *,
    datasets: DatasetBundle | BenchmarkDataset | Mapping[str, DatasetBundle] | Sequence[BenchmarkDataset],
    algorithms: Sequence[str],
    config: BenchmarkConfig,
    partitions: PartitionPlan | BenchmarkPartition | Mapping[str, PartitionPlan] | Sequence[BenchmarkPartition] | None = None,
    partitioning: BioSievePartitionConfig | None = None,
    partitioning_reference: str | None = None,
    biosieve_extra_columns: Mapping[str, Sequence[Any]] | None = None,
    preprocessing: PreprocessingConfig | Any | None = None,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    search_spaces: Mapping[str, Any] | None = None,
    positive_class: Any | None = None,
    registry: AlgorithmRegistry = MODEL_REGISTRY,
) -> BenchmarkResult:
    """Execute a systematic supervised benchmark matrix."""

    return BenchmarkEngine(registry).run(
        datasets=datasets,
        algorithms=algorithms,
        config=config,
        partitions=partitions,
        partitioning=partitioning,
        partitioning_reference=partitioning_reference,
        biosieve_extra_columns=biosieve_extra_columns,
        preprocessing=preprocessing,
        model_params=model_params,
        search_spaces=search_spaces,
        positive_class=positive_class,
    )


__all__ = ["benchmark"]
