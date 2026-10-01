"""Versioned declarative workflow configuration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from saber.benchmark import BenchmarkConfig
from saber.exceptions import ConfigurationError
from saber.preprocessing import PreprocessingConfig
from saber.tuning import TuningConfig

CONFIG_SCHEMA_VERSION = "2.0"

_COMMON_KEYS = {"schema_version", "workflow", "metadata", "output", "overwrite"}
_FIT_KEYS = {"dataset", "algorithm", "model_params", "preprocessing", "positive_class", "random_state"}
_CV_KEYS = _FIT_KEYS | {"partition_plan", "partitioning", "metrics", "evaluation_role", "require_complete"}
_ALLOWED_KEYS = {
    "train": _COMMON_KEYS | _FIT_KEYS | {"partition_plan", "artifact"},
    "evaluate": _COMMON_KEYS | {"dataset", "model", "metrics", "positive_class", "strict_environment"},
    "predict": _COMMON_KEYS | {"dataset", "model", "positive_class", "strict_environment"},
    "validate": _COMMON_KEYS | _CV_KEYS,
    "tune": _COMMON_KEYS | _CV_KEYS | {"tuning", "search_space", "artifact"},
    "benchmark": _COMMON_KEYS
    | {
        "datasets",
        "algorithms",
        "model_params",
        "preprocessing",
        "partitions",
        "partitioning",
        "partitioning_reference",
        "benchmark",
        "search_spaces",
        "metrics",
        "evaluation_role",
        "require_complete",
        "positive_class",
    },
}
_REQUIRED_KEYS = {
    "train": {"dataset", "algorithm"},
    "evaluate": {"dataset", "model"},
    "predict": {"dataset", "model"},
    "validate": {"dataset", "algorithm"},
    "tune": {"dataset", "algorithm", "tuning", "metrics"},
    "benchmark": {"datasets", "algorithms", "metrics"},
}
# ``partition_plan`` (validate/tune) and ``partitions`` (benchmark) are
# mutually exclusive with BioSieve ``partitioning``; exactly one is required.
_PARTITION_SOURCE = {"validate": "partition_plan", "tune": "partition_plan", "benchmark": "partitions"}


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
        if workflow not in _ALLOWED_KEYS:
            raise ConfigurationError(
                f"Unsupported workflow '{workflow}'. Supported: {sorted(_ALLOWED_KEYS)!r}."
            )
        if self.schema_version == "1.0":
            raise ConfigurationError(
                f"Config schema '1.0' is no longer supported; this saber build reads schema "
                f"'{CONFIG_SCHEMA_VERSION}'. Migrate the config following docs/configuration.md "
                "(for example 'partition' -> 'partition_plan', 'dataset.target' -> 'dataset.target_col', "
                "'output' -> a directory string), then use schema_version: \"2.0\"."
            )
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
        source = _PARTITION_SOURCE.get(self.workflow)
        if source is not None and (source in self.payload) == ("partitioning" in self.payload):
            raise ConfigurationError(
                f"Workflow '{self.workflow}' requires exactly one of '{source}' or 'partitioning'."
            )
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


# Column keys follow the ``*_col`` names of saber.load_partition_plan.
DATASET_KEYS = {
    "path",
    "target_col",
    "sample_id_col",
    "feature_cols",
    "group_col",
    "sample_weight_col",
    "sep",
}
PARTITION_KEYS = {"path", "sample_id_col", "role_col", "split_col", "fold_col", "always_train_value"}
# ``extra_columns`` and the ``*_col`` keys name dataset-file columns; the loader
# turns them into the aligned arrays of BioSievePartitionConfig.extra_columns.
PARTITIONING_ROLE_COLUMNS = ("seq_col", "cluster_col", "date_col")
PARTITIONING_KEYS = {"strategy", "params", "extra_columns", *PARTITIONING_ROLE_COLUMNS}
PREPROCESSING_KEYS = _field_names(PreprocessingConfig) - {"transformer"}
TUNING_KEYS = _field_names(TuningConfig)
# The top-level ``metadata`` is the benchmark's metadata.
BENCHMARK_KEYS = _field_names(BenchmarkConfig) - {"metadata"}
_PATH_KEYS = ("output", "artifact", "model")
_BOOL_KEYS = ("overwrite", "require_complete", "strict_environment")


def _validate_nested(workflow: str, payload: Mapping[str, Any]) -> None:
    for key in _PATH_KEYS:
        if key in payload and (not isinstance(payload[key], (str, Path)) or not str(payload[key])):
            raise ConfigurationError(f"'{key}' must be a path string.")
    for key in _BOOL_KEYS:
        if key in payload and not isinstance(payload[key], bool):
            raise ConfigurationError(f"'{key}' must be true or false.")

    if "dataset" in payload:
        _validate_dataset(payload["dataset"], "dataset", labeled=workflow != "predict")
    if "datasets" in payload:
        datasets = as_mapping(payload["datasets"], "datasets")
        if not datasets:
            raise ConfigurationError("'datasets' cannot be empty.")
        for label, item in datasets.items():
            _validate_dataset(item, f"datasets.{label}", labeled=True)
    if "partitioning_reference" in payload:
        if "partitioning" not in payload:
            raise ConfigurationError("'partitioning_reference' requires 'partitioning'.")
        if str(payload["partitioning_reference"]) not in {str(label) for label in payload["datasets"]}:
            raise ConfigurationError(
                f"partitioning_reference '{payload['partitioning_reference']}' is not a 'datasets' label."
            )

    if "partition_plan" in payload:
        _validate_partition_file(payload["partition_plan"], "partition_plan")
        if workflow == "train" and "artifact" not in payload:
            raise ConfigurationError("Train 'partition_plan' is artifact provenance; it requires 'artifact'.")
    if "partitions" in payload:
        partitions = as_mapping(payload["partitions"], "partitions")
        if not partitions:
            raise ConfigurationError("'partitions' cannot be empty.")
        for label, item in partitions.items():
            _validate_partition_file(item, f"partitions.{label}")
    if "partitioning" in payload:
        partitioning = validate_mapping_keys(payload["partitioning"], PARTITIONING_KEYS, "partitioning")
        if "strategy" not in partitioning:
            raise ConfigurationError("'partitioning.strategy' is required.")
        as_mapping(partitioning.get("params", {}), "partitioning.params")
        _names(partitioning.get("extra_columns", []), "partitioning.extra_columns")

    if "preprocessing" in payload:
        validate_mapping_keys(payload["preprocessing"], PREPROCESSING_KEYS, "preprocessing")
    if "tuning" in payload:
        tuning = validate_mapping_keys(payload["tuning"], TUNING_KEYS, "tuning")
        if "artifact" in payload and tuning.get("refit") is False:
            raise ConfigurationError("'artifact' needs a refitted model; remove 'tuning.refit: false'.")
    if "benchmark" in payload:
        benchmark = validate_mapping_keys(payload["benchmark"], BENCHMARK_KEYS, "benchmark")
        if benchmark.get("tuning") is not None:
            validate_mapping_keys(benchmark["tuning"], TUNING_KEYS, "benchmark.tuning")

    for key in ("metrics", "algorithms"):
        if key in payload and not _names(payload[key], key):
            raise ConfigurationError(f"'{key}' must contain at least one entry.")

    if "model_params" in payload:
        model_params = as_mapping(payload["model_params"], "model_params")
        if workflow == "benchmark":
            for algorithm, params in model_params.items():
                as_mapping(params, f"model_params.{algorithm}")
    if "search_space" in payload and not as_mapping(payload["search_space"], "search_space"):
        raise ConfigurationError("'search_space' cannot be empty.")
    if "search_spaces" in payload:
        search_spaces = as_mapping(payload["search_spaces"], "search_spaces")
        for label, item in search_spaces.items():
            if not as_mapping(item, f"search_spaces.{label}"):
                raise ConfigurationError(f"'search_spaces.{label}' cannot be empty.")


def _validate_dataset(value: Any, label: str, *, labeled: bool) -> None:
    dataset = validate_mapping_keys(value, DATASET_KEYS, label)
    if "path" not in dataset:
        raise ConfigurationError(f"'{label}.path' is required.")
    if labeled and "target_col" not in dataset:
        raise ConfigurationError(f"'{label}.target_col' is required for labeled workflows.")
    if "feature_cols" in dataset:
        _names(dataset["feature_cols"], f"{label}.feature_cols")
    sep = dataset.get("sep")
    if sep is not None and (not isinstance(sep, str) or len(sep) != 1):
        raise ConfigurationError(f"'{label}.sep' must be a single character; received {sep!r}.")


def _validate_partition_file(value: Any, label: str) -> None:
    if "path" not in validate_mapping_keys(value, PARTITION_KEYS, label):
        raise ConfigurationError(f"'{label}.path' is required.")


def _names(value: Any, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"'{label}' must be a list of names.")
    return list(value)


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
