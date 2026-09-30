"""Versioned declarative workflow configuration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from saber.benchmark import BenchmarkConfig
from saber.datasets import BioSievePartitionConfig
from saber.exceptions import ConfigurationError
from saber.preprocessing import PreprocessingConfig
from saber.tuning import TuningConfig

CONFIG_SCHEMA_VERSION = "1.0"
WORKFLOWS = {"train", "evaluate", "validate", "tune", "benchmark", "predict"}

_COMMON_KEYS = {
    "schema_version",
    "workflow",
    "metadata",
    "output",
}
_ALLOWED_KEYS = {
    "train": _COMMON_KEYS
    | {
        "dataset",
        "algorithm",
        "preprocessing",
        "random_state",
        "model_params",
        "artifact",
        "positive_class",
        "partition",
    },
    "evaluate": _COMMON_KEYS
    | {
        "dataset",
        "artifact",
        "metrics",
        "positive_class",
        "strict_environment",
    },
    "validate": _COMMON_KEYS
    | {
        "dataset",
        "algorithm",
        "partition",
        "partitioning",
        "preprocessing",
        "metrics",
        "evaluation_role",
        "positive_class",
        "random_state",
        "return_estimators",
        "require_complete",
        "model_params",
        "biosieve_extra_columns",
    },
    "tune": _COMMON_KEYS
    | {
        "dataset",
        "algorithm",
        "partition",
        "partitioning",
        "preprocessing",
        "tuning",
        "search_space",
        "evaluation_role",
        "require_complete",
        "model_params",
        "positive_class",
        "biosieve_extra_columns",
        "artifact",
    },
    "benchmark": _COMMON_KEYS
    | {
        "dataset",
        "datasets",
        "algorithms",
        "partitions",
        "partitioning",
        "partitioning_reference",
        "biosieve_extra_columns",
        "preprocessing",
        "benchmark",
        "model_params",
        "search_spaces",
        "positive_class",
        "artifact",
    },
    "predict": _COMMON_KEYS
    | {
        "dataset",
        "artifact",
        "positive_class",
        "strict_environment",
    },
}
_REQUIRED_KEYS = {
    "train": {"dataset", "algorithm"},
    "evaluate": {"dataset", "artifact"},
    "validate": {"dataset", "algorithm"},
    "tune": {"dataset", "algorithm", "tuning"},
    "benchmark": {"algorithms", "benchmark"},
    "predict": {"dataset", "artifact"},
}


def _field_names(config_cls: type) -> set[str]:
    return {item.name for item in fields(config_cls)}


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """Validated workflow configuration with source-relative path context."""

    workflow: str
    payload: dict[str, Any]
    schema_version: str = CONFIG_SCHEMA_VERSION
    source: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Normalize and validate the workflow configuration after construction."""
        workflow = str(self.workflow).strip().lower()
        if workflow not in WORKFLOWS:
            raise ConfigurationError(f"Unsupported workflow '{workflow}'. Supported: {sorted(WORKFLOWS)!r}.")
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ConfigurationError(
                f"Unsupported config schema '{self.schema_version}'. "
                f"This saber build supports '{CONFIG_SCHEMA_VERSION}'."
            )
        object.__setattr__(self, "workflow", workflow)
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "metadata", dict(self.metadata))
        self._validate()

    @property
    def base_dir(self) -> Path:
        """Return the directory that relative config paths resolve against."""
        if self.source is None:
            return Path.cwd()
        return self.source.parent

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a config-relative path against base_dir."""
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    def to_dict(self) -> dict[str, Any]:
        """Serialize this config back into a plain dict for storage."""
        result = dict(self.payload)
        result["schema_version"] = self.schema_version
        result["workflow"] = self.workflow
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result

    def _validate(self) -> None:
        keys = set(self.payload) | {"workflow", "schema_version"}
        unknown = keys - _ALLOWED_KEYS[self.workflow]
        if unknown:
            raise ConfigurationError(f"Unknown keys for workflow '{self.workflow}': {sorted(unknown)!r}.")
        missing = _REQUIRED_KEYS[self.workflow] - set(self.payload)
        if missing:
            raise ConfigurationError(
                f"Workflow '{self.workflow}' is missing required keys: {sorted(missing)!r}."
            )

        if self.workflow in {"validate", "tune"}:
            if "partition" not in self.payload and "partitioning" not in self.payload:
                raise ConfigurationError(
                    f"Workflow '{self.workflow}' requires either 'partition' or 'partitioning'."
                )
            if "partition" in self.payload and "partitioning" in self.payload:
                raise ConfigurationError("Provide only one of 'partition' or 'partitioning'.")

        if self.workflow == "benchmark":
            if ("dataset" in self.payload) == ("datasets" in self.payload):
                raise ConfigurationError("Benchmark config requires exactly one of 'dataset' or 'datasets'.")
            if "partitions" in self.payload and "partitioning" in self.payload:
                raise ConfigurationError(
                    "Benchmark config cannot define both 'partitions' and 'partitioning'."
                )
            if "partitions" not in self.payload and "partitioning" not in self.payload:
                raise ConfigurationError(
                    "Benchmark config requires either 'partitions' or BioSieve 'partitioning'."
                )

        if self.workflow == "train" and "partition" in self.payload:
            partition = self.payload["partition"]
            if not isinstance(partition, Mapping):
                raise ConfigurationError("'partition' must be a mapping.")

        _validate_nested(self.workflow, self.payload)


def workflow_config_from_mapping(
    mapping: Mapping[str, Any],
    *,
    source: str | Path | None = None,
) -> WorkflowConfig:
    """Validate a raw mapping into a versioned WorkflowConfig."""
    payload = dict(mapping)
    workflow = payload.pop("workflow", None)
    if workflow is None:
        raise ConfigurationError("Configuration must define 'workflow'.")
    schema_version = str(payload.pop("schema_version", CONFIG_SCHEMA_VERSION))
    metadata = payload.pop("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("'metadata' must be a mapping.")
    resolved_source = None if source is None else Path(source).resolve()
    return WorkflowConfig(
        workflow=str(workflow),
        payload=payload,
        schema_version=schema_version,
        source=resolved_source,
        metadata=dict(metadata),
    )


DATASET_KEYS = {"path", "target", "sample_id", "features", "groups", "sample_weight", "sep"}
PARTITION_KEYS = {"path", "sample_id_col", "role_col", "split_col", "fold_col", "always_train_value"}
PARTITIONING_KEYS = _field_names(BioSievePartitionConfig)
PREPROCESSING_KEYS = _field_names(PreprocessingConfig) - {"transformer"}
TUNING_KEYS = _field_names(TuningConfig)
BENCHMARK_KEYS = _field_names(BenchmarkConfig)
_ARTIFACT_KEYS = {"path", "overwrite", "include_object"}
_OUTPUT_KEYS = {"path", "directory", "summary"}


def _validate_nested(workflow: str, payload: Mapping[str, Any]) -> None:
    if "dataset" in payload:
        validate_mapping_keys(payload["dataset"], DATASET_KEYS, "dataset")
    if "datasets" in payload:
        datasets = as_mapping(payload["datasets"], "datasets")
        if not datasets:
            raise ConfigurationError("'datasets' cannot be empty.")
        for label, item in datasets.items():
            validate_mapping_keys(item, DATASET_KEYS, f"datasets.{label}")

    if "partition" in payload:
        validate_mapping_keys(payload["partition"], PARTITION_KEYS, "partition")
    if "partitions" in payload:
        partitions = as_mapping(payload["partitions"], "partitions")
        for label, item in partitions.items():
            validate_mapping_keys(item, PARTITION_KEYS, f"partitions.{label}")
    if "partitioning" in payload:
        validate_mapping_keys(payload["partitioning"], PARTITIONING_KEYS, "partitioning")

    if "preprocessing" in payload:
        validate_mapping_keys(payload["preprocessing"], PREPROCESSING_KEYS, "preprocessing")
    if "tuning" in payload:
        tuning = validate_mapping_keys(payload["tuning"], TUNING_KEYS, "tuning")
        if "metrics" not in tuning or not tuning["metrics"]:
            raise ConfigurationError("'tuning.metrics' must contain at least one metric.")
    if "benchmark" in payload:
        benchmark = validate_mapping_keys(payload["benchmark"], BENCHMARK_KEYS, "benchmark")
        if "metrics" not in benchmark or not benchmark["metrics"]:
            raise ConfigurationError("'benchmark.metrics' must contain at least one metric.")
        if benchmark.get("tuning") is not None:
            validate_mapping_keys(benchmark["tuning"], TUNING_KEYS, "benchmark.tuning")

    if "artifact" in payload:
        artifact_value = payload["artifact"]
        if workflow in {"train", "tune", "benchmark"} and not isinstance(artifact_value, Mapping):
            raise ConfigurationError(f"Workflow '{workflow}' requires 'artifact' to be a mapping.")
        if isinstance(artifact_value, Mapping):
            artifact = validate_mapping_keys(artifact_value, _ARTIFACT_KEYS, "artifact")
            if "path" not in artifact:
                raise ConfigurationError("'artifact.path' is required.")
        elif workflow in {"evaluate", "predict"} and not isinstance(artifact_value, (str, Path)):
            raise ConfigurationError("'artifact' must be a path string or mapping with path.")
    if "output" in payload:
        validate_mapping_keys(payload["output"], _OUTPUT_KEYS, "output")

    if workflow == "benchmark":
        algorithms = payload.get("algorithms")
        if not isinstance(algorithms, (list, tuple)) or not algorithms:
            raise ConfigurationError("'algorithms' must be a non-empty list.")

    if "search_space" in payload:
        search_space = as_mapping(payload["search_space"], "search_space")
        if not search_space:
            raise ConfigurationError("'search_space' cannot be empty.")
    if "search_spaces" in payload:
        search_spaces = as_mapping(payload["search_spaces"], "search_spaces")
        for label, item in search_spaces.items():
            if not as_mapping(item, f"search_spaces.{label}"):
                raise ConfigurationError(f"'search_spaces.{label}' cannot be empty.")

    if workflow in {"train", "evaluate", "validate", "tune"}:
        dataset = as_mapping(payload["dataset"], "dataset")
        if "path" not in dataset:
            raise ConfigurationError("'dataset.path' is required.")
        if "target" not in dataset:
            raise ConfigurationError(f"Workflow '{workflow}' requires 'dataset.target'.")
    if workflow == "predict":
        dataset = as_mapping(payload["dataset"], "dataset")
        if "path" not in dataset:
            raise ConfigurationError("'dataset.path' is required.")

    if "partitioning" in payload:
        partitioning = as_mapping(payload["partitioning"], "partitioning")
        if "strategy" not in partitioning:
            raise ConfigurationError("'partitioning.strategy' is required.")
    if "partition" in payload:
        partition = as_mapping(payload["partition"], "partition")
        if "path" not in partition:
            raise ConfigurationError("'partition.path' is required.")


def validate_mapping_keys(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    """Return ``value`` as a dict after rejecting keys outside ``allowed``."""
    mapping = as_mapping(value, label)
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown {label} keys: {sorted(unknown)!r}.")
    return mapping


def as_mapping(value: Any, label: str) -> dict[str, Any]:
    """Return ``value`` as a dict, or raise ConfigurationError if it is not a mapping."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{label}' must be a mapping.")
    return dict(value)
