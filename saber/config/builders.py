"""Translate validated declarative configuration into saber contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from saber.benchmark import BenchmarkConfig
from saber.config.schema import (
    BENCHMARK_KEYS,
    BENCHMARK_KWARG_KEYS,
    DATASET_KEYS,
    PARTITION_KEYS,
    PARTITIONING_KEYS,
    PREPROCESSING_KEYS,
    TUNING_KEYS,
    TUNING_KWARG_KEYS,
    validate_mapping_keys,
)
from saber.core import Categorical, Float, Integer, LogFloat, SearchSpace
from saber.datasets import (
    BioSievePartitionConfig,
    DatasetBundle,
    PartitionPlan,
    load_partition_plan,
)
from saber.exceptions import ConfigurationError
from saber.preprocessing import PreprocessingConfig
from saber.tuning import TuningConfig
from saber.utils.tabular import read_table

if TYPE_CHECKING:
    import polars as pl

    from saber.config.schema import WorkflowConfig


def load_dataset(
    config: WorkflowConfig,
    payload: Mapping[str, Any],
) -> DatasetBundle:
    """Load one prepared numeric tabular dataset from CSV/TSV."""
    validate_mapping_keys(payload, DATASET_KEYS, "dataset")
    if "path" not in payload:
        raise ConfigurationError("Dataset config requires 'path'.")
    if "target" not in payload:
        raise ConfigurationError("Labeled workflows require dataset.target.")

    path = config.resolve_path(payload["path"])
    if not path.exists():
        raise ConfigurationError(f"Dataset file does not exist: {path}.")
    suffix = path.suffix.lower()
    separator = _dataset_separator(payload, suffix)
    if suffix not in {".csv", ".tsv", ".txt"}:
        raise ConfigurationError("CLI/YAML dataset files currently support CSV, TSV, or TXT.")
    frame = read_table(path, separator=separator)

    target_col = payload.get("target")
    id_col = payload.get("sample_id")
    group_col = payload.get("groups")
    weight_col = payload.get("sample_weight")
    excluded = {name for name in (target_col, id_col, group_col, weight_col) if name is not None}

    features = payload.get("features")
    if features is None:
        feature_cols = [column for column in frame.columns if column not in excluded]
    else:
        if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
            raise ConfigurationError("dataset.features must be a list of column names.")
        feature_cols = [str(column) for column in features]
    if not feature_cols:
        raise ConfigurationError("Dataset configuration resolves to zero feature columns.")

    required_columns = set(feature_cols) | excluded
    missing = required_columns - set(frame.columns)
    if missing:
        raise ConfigurationError(f"Dataset is missing configured columns: {sorted(missing)!r}.")

    if target_col is None:
        # Prediction configs do not use DatasetBundle because y is absent.
        raise ConfigurationError("Internal error: use load_prediction_frame for unlabeled data.")

    return DatasetBundle(
        X=frame.select(feature_cols),
        y=frame[target_col].to_numpy(),
        sample_ids=None if id_col is None else frame[id_col].to_list(),
        feature_names=feature_cols,
        groups=None if group_col is None else frame[group_col].to_list(),
        sample_weight=None if weight_col is None else frame[weight_col].to_numpy(),
        metadata={"source": str(path)},
    )


def load_prediction_frame(
    config: WorkflowConfig,
    payload: Mapping[str, Any],
) -> tuple[pl.DataFrame, tuple[Any, ...] | None]:
    """Load features/sample IDs for prediction without requiring a target."""
    validate_mapping_keys(payload, DATASET_KEYS, "dataset")
    if "path" not in payload:
        raise ConfigurationError("Prediction dataset config requires 'path'.")
    path = config.resolve_path(payload["path"])
    frame = read_table(path, separator=_dataset_separator(payload, path.suffix.lower()))
    id_col = payload.get("sample_id")
    excluded = {
        name
        for name in (payload.get("target"), id_col, payload.get("groups"), payload.get("sample_weight"))
        if name is not None
    }
    features = payload.get("features")
    feature_cols = (
        [column for column in frame.columns if column not in excluded]
        if features is None
        else [str(column) for column in features]
    )
    missing = set(feature_cols) - set(frame.columns)
    if missing:
        raise ConfigurationError(f"Prediction data are missing features: {sorted(missing)!r}.")
    sample_ids = None if id_col is None else tuple(frame[id_col].to_list())
    return frame.select(feature_cols), sample_ids


def build_preprocessing(payload: Mapping[str, Any] | None) -> PreprocessingConfig:
    """Build a PreprocessingConfig from a validated preprocessing payload."""
    if payload is None:
        return PreprocessingConfig()
    validate_mapping_keys(payload, PREPROCESSING_KEYS, "preprocessing")
    return PreprocessingConfig(
        imputation=payload.get("imputation", "auto"),
        scaler=payload.get("scaler", "auto"),
        fill_value=payload.get("fill_value", 0.0),
    )


def build_partition_inputs(
    config: WorkflowConfig,
    *,
    partition: Mapping[str, Any] | None,
    partitioning: Mapping[str, Any] | None,
    extra_columns: Mapping[str, Any] | None = None,
) -> PartitionPlan | BioSievePartitionConfig | None:
    """Build an external partition plan or a BioSieve partitioning config from validated payloads."""
    if partition is not None:
        validate_mapping_keys(partition, PARTITION_KEYS, "partition")
        if "path" not in partition:
            raise ConfigurationError("External partition config requires 'path'.")
        kwargs = {key: value for key, value in partition.items() if key != "path"}
        return load_partition_plan(config.resolve_path(partition["path"]), **kwargs)
    if partitioning is not None:
        validate_mapping_keys(partitioning, PARTITIONING_KEYS, "partitioning")
        if "strategy" not in partitioning:
            raise ConfigurationError("BioSieve partitioning requires 'strategy'.")
        return BioSievePartitionConfig(**dict(partitioning), extra_columns=extra_columns)
    return None


def build_tuning_config(payload: Mapping[str, Any]) -> TuningConfig:
    """Build a TuningConfig from a validated tuning payload.

    ``metrics`` and ``random_state`` are tune() keyword arguments; read them
    from the payload with :func:`tuning_kwargs`.
    """
    validate_mapping_keys(payload, TUNING_KEYS, "tuning")
    return TuningConfig(**{key: value for key, value in payload.items() if key not in TUNING_KWARG_KEYS})


def tuning_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the tune() keyword arguments nested in a tuning payload."""
    return {
        "metrics": tuple(payload.get("metrics", ())),
        "random_state": payload.get("random_state"),
    }


def build_benchmark_config(payload: Mapping[str, Any]) -> BenchmarkConfig:
    """Build a BenchmarkConfig from a validated benchmark payload.

    ``metrics``, ``evaluation_role``, ``require_complete`` and
    ``return_estimators`` are benchmark() keyword arguments; read them with
    :func:`benchmark_kwargs`.  Inside ``benchmark.tuning``, tuned runs select
    on the benchmark metrics, so ``tuning.metrics`` only supplies the default
    ``refit_metric`` (its first entry) and ``tuning.random_state`` is rejected:
    each benchmark seed is the tuning seed.
    """
    validate_mapping_keys(payload, BENCHMARK_KEYS, "benchmark")
    if "metrics" not in payload:
        raise ConfigurationError("Benchmark config requires 'metrics'.")
    values = {key: value for key, value in payload.items() if key not in BENCHMARK_KWARG_KEYS}
    if "seeds" in values:
        values["seeds"] = tuple(values["seeds"])
    if "modes" in values:
        values["modes"] = tuple(values["modes"])
    if values.get("tuning") is not None:
        tuning = dict(values["tuning"])
        if tuning.get("random_state") is not None:
            raise ConfigurationError(
                "'benchmark.tuning.random_state' is not supported; each benchmark seed is the tuning seed."
            )
        tuning_metrics = tuple(tuning.get("metrics", ()))
        if tuning_metrics and tuning.get("refit_metric") is None:
            tuning["refit_metric"] = tuning_metrics[0]
        values["tuning"] = build_tuning_config(tuning)
    return BenchmarkConfig(**values)


def benchmark_kwargs(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the benchmark() keyword arguments nested in a benchmark payload."""
    return {
        "metrics": tuple(payload["metrics"]),
        "evaluation_role": payload.get("evaluation_role", "auto"),
        "require_complete": bool(payload.get("require_complete", True)),
        "return_estimators": bool(payload.get("return_estimators", False)),
    }


def build_search_space(name: str, payload: Mapping[str, Any] | None) -> SearchSpace | None:
    """Build a SearchSpace from a validated search-space payload, or None if absent."""
    if payload is None:
        return None
    parameters: dict[str, Any] = {}
    for parameter, domain in payload.items():
        parameters[str(parameter)] = _build_domain(domain)
    return SearchSpace(name=name, parameters=parameters)


def _build_domain(domain: Any) -> Any:
    if isinstance(domain, list):
        return domain
    if not isinstance(domain, Mapping):
        raise ConfigurationError("Search-space domains must be lists or typed mappings.")
    kind = str(domain.get("type", "")).lower()
    if kind == "categorical":
        return Categorical(domain.get("values", ()))
    if kind == "integer":
        return Integer(int(domain["low"]), int(domain["high"]), int(domain.get("step", 1)))
    if kind == "float":
        step = domain.get("step")
        return Float(float(domain["low"]), float(domain["high"]), None if step is None else float(step))
    if kind == "log_float":
        return LogFloat(float(domain["low"]), float(domain["high"]))
    raise ConfigurationError(f"Unknown search-space domain type '{kind}'.")


def _dataset_separator(payload: Mapping[str, Any], suffix: str) -> str:
    separator = payload.get("sep")
    if separator is None:
        separator = "\t" if suffix == ".tsv" else ","
    if len(separator) != 1:
        raise ConfigurationError(f"dataset.sep must be a single character; received {separator!r}.")
    return separator
