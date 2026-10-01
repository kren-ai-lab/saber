"""Translate validated declarative configuration into saber contracts.

Payloads reaching these builders were validated by :mod:`saber.config.schema`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from saber.benchmark import BenchmarkConfig
from saber.config.schema import PARTITIONING_ROLE_COLUMNS
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
from saber.utils.tabular import as_frame, read_table

if TYPE_CHECKING:
    from saber.config.schema import WorkflowConfig

# Default separator per text suffix; ``None`` marks Parquet.
_DATASET_FORMATS = {".csv": ",", ".tsv": "\t", ".txt": "\t", ".parquet": None}


def read_dataset(
    config: WorkflowConfig,
    payload: Mapping[str, Any],
    *,
    extra_columns: Sequence[str] = (),
) -> tuple[pl.DataFrame, list[str]]:
    """Read one dataset file and return it with its resolved feature columns.

    Target, sample-id, group, sample-weight and (when present) ``extra_columns``
    columns are never features unless ``feature_cols`` names them explicitly.
    """
    path = config.resolve_path(payload["path"])
    if not path.exists():
        raise ConfigurationError(f"Dataset file does not exist: {path}.")
    suffix = path.suffix.lower()
    if suffix not in _DATASET_FORMATS:
        raise ConfigurationError("Dataset files must be CSV, TSV, TXT (tab-separated) or Parquet.")
    default_separator = _DATASET_FORMATS[suffix]
    if default_separator is None:
        frame = as_frame(pl.read_parquet(path))
    else:
        frame = read_table(path, separator=payload.get("sep", default_separator))

    roles = ("target_col", "sample_id_col", "group_col", "sample_weight_col")
    role_cols = {payload[key] for key in roles if payload.get(key) is not None}
    features = payload.get("feature_cols")
    feature_cols = (
        [column for column in frame.columns if column not in role_cols | set(extra_columns)]
        if features is None
        else list(features)
    )
    if not feature_cols:
        raise ConfigurationError("Dataset configuration resolves to zero feature columns.")
    missing = (set(feature_cols) | role_cols) - set(frame.columns)
    if missing:
        raise ConfigurationError(f"Dataset {path.name} is missing configured columns: {sorted(missing)!r}.")
    return frame, feature_cols


def load_dataset(
    config: WorkflowConfig,
    payload: Mapping[str, Any],
    *,
    extra_columns: Sequence[str] = (),
) -> tuple[DatasetBundle, dict[str, list[Any]]]:
    """Load a labeled dataset plus the ``extra_columns`` it contains as aligned lists."""
    frame, feature_cols = read_dataset(config, payload, extra_columns=extra_columns)
    id_col = payload.get("sample_id_col")
    group_col = payload.get("group_col")
    weight_col = payload.get("sample_weight_col")
    dataset = DatasetBundle(
        X=frame.select(feature_cols),
        y=frame[payload["target_col"]].to_numpy(),
        sample_ids=None if id_col is None else frame[id_col].to_list(),
        feature_names=feature_cols,
        groups=None if group_col is None else frame[group_col].to_list(),
        sample_weight=None if weight_col is None else frame[weight_col].to_numpy(),
        metadata={"source": str(config.resolve_path(payload["path"]))},
    )
    return dataset, {name: frame[name].to_list() for name in extra_columns if name in frame.columns}


def load_prediction_frame(
    config: WorkflowConfig,
    payload: Mapping[str, Any],
) -> tuple[pl.DataFrame, tuple[Any, ...] | None]:
    """Load features/sample IDs for prediction without requiring a target."""
    frame, feature_cols = read_dataset(config, payload)
    id_col = payload.get("sample_id_col")
    sample_ids = None if id_col is None else tuple(frame[id_col].to_list())
    return frame.select(feature_cols), sample_ids


def build_preprocessing(payload: Mapping[str, Any] | None) -> PreprocessingConfig:
    """Build a PreprocessingConfig from a validated preprocessing payload."""
    return PreprocessingConfig(**dict(payload or {}))


def build_partition_plan(config: WorkflowConfig, payload: Mapping[str, Any]) -> PartitionPlan:
    """Load an external partition plan from a validated partition-file payload."""
    kwargs = {key: value for key, value in payload.items() if key != "path"}
    return load_partition_plan(config.resolve_path(payload["path"]), **kwargs)


def build_partitioning(
    payload: Mapping[str, Any],
    extra_columns: Mapping[str, Sequence[Any]],
) -> BioSievePartitionConfig:
    """Build a BioSieve partitioning config; ``extra_columns`` holds the loaded columns."""
    return BioSievePartitionConfig(
        strategy=payload["strategy"],
        params=payload.get("params", {}),
        **{key: payload[key] for key in PARTITIONING_ROLE_COLUMNS if key in payload},
        extra_columns=extra_columns or None,
    )


def build_tuning_config(payload: Mapping[str, Any]) -> TuningConfig:
    """Build a TuningConfig from a validated tuning payload."""
    return TuningConfig(**dict(payload))


def build_benchmark_config(payload: Mapping[str, Any] | None) -> BenchmarkConfig:
    """Build a BenchmarkConfig from a validated benchmark payload."""
    values = dict(payload or {})
    for key in ("seeds", "modes"):
        if key in values:
            values[key] = tuple(values[key])
    if values.get("tuning") is not None:
        values["tuning"] = build_tuning_config(values["tuning"])
    return BenchmarkConfig(**values)


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
