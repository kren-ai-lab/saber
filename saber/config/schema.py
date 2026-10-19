"""Versioned declarative workflow configuration contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from saber.benchmark import BenchmarkConfig
from saber.core.registry import ALGORITHMS
from saber.exceptions import ConfigurationError
from saber.preprocessing import PreprocessingConfig
from saber.tuning import TuningConfig

CONFIG_SCHEMA_VERSION = "1.0"

_COMMON_KEYS = {"schema_version", "workflow", "metadata", "output", "overwrite"}
_FIT_KEYS = {"dataset", "algorithm", "model_params", "preprocessing", "positive_class", "random_state"}
_CV_KEYS = _FIT_KEYS | {"partition_plan", "partitioning", "metrics", "evaluation_role", "require_complete"}
_ALLOWED_KEYS = {
    "train": _COMMON_KEYS | _FIT_KEYS | {"partition_plan", "artifact"},
    "evaluate": _COMMON_KEYS | {"dataset", "model", "metrics", "positive_class", "strict_environment"},
    "predict": _COMMON_KEYS | {"dataset", "model", "positive_class", "strict_environment"},
    "cross-validate": _COMMON_KEYS | _CV_KEYS,
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
    "cross-validate": {"dataset", "algorithm"},
    "tune": {"dataset", "algorithm", "tuning", "metrics"},
    "benchmark": {"datasets", "algorithms", "metrics"},
}
# ``partition_plan`` (cross-validate/tune) and ``partitions`` (benchmark) are
# mutually exclusive with BioSieve ``partitioning``; exactly one is required.
_PARTITION_SOURCE = {"cross-validate": "partition_plan", "tune": "partition_plan", "benchmark": "partitions"}


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
        unknown = sorted((set(self.payload) | {"workflow", "schema_version"}) - _ALLOWED_KEYS[self.workflow])
        missing = sorted(_REQUIRED_KEYS[self.workflow] - set(self.payload))
        issues = [(key, f"Unknown key for workflow '{self.workflow}'.") for key in unknown]
        issues += [(key, f"Required by workflow '{self.workflow}'.") for key in missing]
        source = _PARTITION_SOURCE.get(self.workflow)
        if source is not None and (source in self.payload) == ("partitioning" in self.payload):
            issues.append(
                ("", f"Workflow '{self.workflow}' requires exactly one of '{source}' or 'partitioning'.")
            )
        issues += _nested_issues(self.workflow, self.payload)
        if issues:
            message = "\n".join(f"{path or '<config>'}: {text}" for path, text in issues)
            raise ConfigurationError(message, issues=issues)


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


def _nested_issues(workflow: str, payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Collect every problem below the top-level keys; each check reports at most its first problem."""
    issues: list[tuple[str, str]] = []

    def check(path: str, fn: Callable[[], object]) -> None:
        try:
            fn()
        except ConfigurationError as exc:
            issues.extend(exc.issues if exc.issues[0][0] else [(path, str(exc))])

    issues += [
        (key, "Must be a path string.")
        for key in _PATH_KEYS
        if key in payload and (not isinstance(payload[key], (str, Path)) or not str(payload[key]))
    ]
    issues += [
        (key, "Must be true or false.")
        for key in _BOOL_KEYS
        if key in payload and not isinstance(payload[key], bool)
    ]

    if "dataset" in payload:
        check(
            "dataset", lambda: _validate_dataset(payload["dataset"], "dataset", labeled=workflow != "predict")
        )
    if "datasets" in payload:
        check("datasets", lambda: _non_empty_mapping(payload["datasets"], "datasets"))
        for label, item in (payload["datasets"] if isinstance(payload["datasets"], Mapping) else {}).items():
            check(
                f"datasets.{label}",
                lambda item=item, label=label: _validate_dataset(item, f"datasets.{label}", labeled=True),
            )
    if "partitioning_reference" in payload:
        if "partitioning" not in payload:
            issues.append(("partitioning_reference", "Requires 'partitioning'."))
        elif str(payload["partitioning_reference"]) not in {
            str(label) for label in payload.get("datasets", ())
        }:
            issues.append(
                (
                    "partitioning_reference",
                    f"'{payload['partitioning_reference']}' is not a 'datasets' label.",
                )
            )

    if "partition_plan" in payload:
        check("partition_plan", lambda: _validate_partition_file(payload["partition_plan"], "partition_plan"))
        if workflow == "train" and "artifact" not in payload:
            issues.append(
                ("artifact", "Train 'partition_plan' is artifact provenance; it requires 'artifact'.")
            )
    if "partitions" in payload:
        check("partitions", lambda: _non_empty_mapping(payload["partitions"], "partitions"))
        for label, item in (
            payload["partitions"] if isinstance(payload["partitions"], Mapping) else {}
        ).items():
            check(
                f"partitions.{label}",
                lambda item=item, label=label: _validate_partition_file(item, f"partitions.{label}"),
            )
    if "partitioning" in payload:
        check("partitioning", lambda: _validate_partitioning(payload["partitioning"]))

    if "preprocessing" in payload:
        check(
            "preprocessing",
            lambda: validate_mapping_keys(payload["preprocessing"], PREPROCESSING_KEYS, "preprocessing"),
        )
    if "tuning" in payload:
        check("tuning", lambda: _validate_tuning(payload))
    if "benchmark" in payload:
        check("benchmark", lambda: _validate_benchmark(payload["benchmark"]))

    for key in ("metrics", "algorithms"):
        if key in payload:
            check(key, lambda key=key: _names(payload[key], key))
            if isinstance(payload[key], (list, tuple)) and not payload[key]:
                issues.append((key, "Must contain at least one entry."))

    named = [("algorithm", payload.get("algorithm"))]
    if isinstance(payload.get("algorithms"), (list, tuple)):
        named += [(f"algorithms[{i}]", name) for i, name in enumerate(payload["algorithms"])]
    issues.extend(
        (path, f"Unknown algorithm {name!r}; see 'saber models list'.")
        for path, name in named
        if isinstance(name, str) and name not in ALGORITHMS
    )

    if "model_params" in payload:
        check("model_params", lambda: _validate_model_params(workflow, payload["model_params"]))
    if "search_space" in payload:
        check("search_space", lambda: _non_empty_mapping(payload["search_space"], "search_space"))
    if "search_spaces" in payload:
        check("search_spaces", lambda: _validate_search_spaces(payload["search_spaces"]))
    return issues


def _validate_partitioning(value: Any) -> None:
    partitioning = validate_mapping_keys(value, PARTITIONING_KEYS, "partitioning")
    if "strategy" not in partitioning:
        raise ConfigurationError("'partitioning.strategy' is required.", path="partitioning.strategy")
    as_mapping(partitioning.get("params", {}), "partitioning.params")
    _names(partitioning.get("extra_columns", []), "partitioning.extra_columns")


def _validate_tuning(payload: Mapping[str, Any]) -> None:
    tuning = validate_mapping_keys(payload["tuning"], TUNING_KEYS, "tuning")
    if "artifact" in payload and tuning.get("refit") is False:
        raise ConfigurationError(
            "'artifact' needs a refitted model; remove 'tuning.refit: false'.", path="tuning.refit"
        )


def _validate_benchmark(value: Any) -> None:
    benchmark = validate_mapping_keys(value, BENCHMARK_KEYS, "benchmark")
    if benchmark.get("tuning") is not None:
        validate_mapping_keys(benchmark["tuning"], TUNING_KEYS, "benchmark.tuning")


def _validate_model_params(workflow: str, value: Any) -> None:
    model_params = as_mapping(value, "model_params")
    if workflow == "benchmark":
        for algorithm, params in model_params.items():
            as_mapping(params, f"model_params.{algorithm}")


def _non_empty_mapping(value: Any, label: str) -> None:
    if not as_mapping(value, label):
        raise ConfigurationError(f"'{label}' cannot be empty.", path=label)


def _validate_search_spaces(value: Any) -> None:
    for label, item in as_mapping(value, "search_spaces").items():
        _non_empty_mapping(item, f"search_spaces.{label}")


def _validate_dataset(value: Any, label: str, *, labeled: bool) -> None:
    dataset = validate_mapping_keys(value, DATASET_KEYS, label)
    if "path" not in dataset:
        raise ConfigurationError(f"'{label}.path' is required.", path=f"{label}.path")
    if labeled and "target_col" not in dataset:
        raise ConfigurationError(
            f"'{label}.target_col' is required for labeled workflows.", path=f"{label}.target_col"
        )
    if "feature_cols" in dataset:
        _names(dataset["feature_cols"], f"{label}.feature_cols")
    sep = dataset.get("sep")
    if sep is not None and (not isinstance(sep, str) or len(sep) != 1):
        raise ConfigurationError(
            f"'{label}.sep' must be a single character; received {sep!r}.", path=f"{label}.sep"
        )


def _validate_partition_file(value: Any, label: str) -> None:
    if "path" not in validate_mapping_keys(value, PARTITION_KEYS, label):
        raise ConfigurationError(f"'{label}.path' is required.", path=f"{label}.path")


def _names(value: Any, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ConfigurationError(f"'{label}' must be a list of names.", path=label)
    return list(value)


def validate_mapping_keys(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    """Return ``value`` as a dict after rejecting keys outside ``allowed``."""
    mapping = as_mapping(value, label)
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigurationError(
            f"Unknown {label} keys: {unknown!r}.",
            issues=[(f"{label}.{key}", "Unknown key.") for key in unknown],
        )
    return mapping


def as_mapping(value: Any, label: str) -> dict[str, Any]:
    """Return ``value`` as a dict, or raise ConfigurationError if it is not a mapping."""
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{label}' must be a mapping.", path=label)
    return dict(value)
