"""Translate validated declarative configuration into saber contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from saber.benchmark import BenchmarkConfig
from saber.config.schema import WorkflowConfig
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


def load_dataset(
    config: WorkflowConfig,
    payload: Mapping[str, Any],
    *,
    require_target: bool = True,
) -> DatasetBundle:
    """Load one prepared numeric tabular dataset from CSV/TSV."""
    allowed = {
        "path",
        "target",
        "sample_id",
        "features",
        "groups",
        "sample_weight",
        "sep",
    }
    _reject_unknown(payload, allowed, "dataset")
    if "path" not in payload:
        raise ConfigurationError("Dataset config requires 'path'.")
    if require_target and "target" not in payload:
        raise ConfigurationError("Labeled workflows require dataset.target.")

    path = config.resolve_path(payload["path"])
    if not path.exists():
        raise ConfigurationError(f"Dataset file does not exist: {path}.")
    suffix = path.suffix.lower()
    separator = payload.get("sep")
    if separator is None:
        separator = "\t" if suffix == ".tsv" else ","
    if suffix not in {".csv", ".tsv", ".txt"}:
        raise ConfigurationError("CLI/YAML dataset files currently support CSV, TSV, or TXT.")
    frame = pd.read_csv(path, sep=separator)

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
        X=frame.loc[:, feature_cols].copy(),
        y=frame[target_col].to_numpy(),
        sample_ids=None if id_col is None else frame[id_col].tolist(),
        feature_names=feature_cols,
        groups=None if group_col is None else frame[group_col].tolist(),
        sample_weight=None if weight_col is None else frame[weight_col].to_numpy(),
        metadata={"source": str(path)},
    )


def load_prediction_frame(
    config: WorkflowConfig,
    payload: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[Any, ...] | None]:
    """Load features/sample IDs for prediction without requiring a target."""
    allowed = {"path", "target", "sample_id", "features", "groups", "sample_weight", "sep"}
    _reject_unknown(payload, allowed, "dataset")
    if "path" not in payload:
        raise ConfigurationError("Prediction dataset config requires 'path'.")
    path = config.resolve_path(payload["path"])
    suffix = path.suffix.lower()
    separator = payload.get("sep") or ("\t" if suffix == ".tsv" else ",")
    frame = pd.read_csv(path, sep=separator)
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
    sample_ids = None if id_col is None else tuple(frame[id_col].tolist())
    return frame.loc[:, feature_cols].copy(), sample_ids


def build_preprocessing(payload: Mapping[str, Any] | None) -> PreprocessingConfig:
    if payload is None:
        return PreprocessingConfig()
    allowed = {"imputation", "scaler", "fill_value"}
    _reject_unknown(payload, allowed, "preprocessing")
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
) -> tuple[PartitionPlan | None, BioSievePartitionConfig | None]:
    plan = None
    biosieve = None
    if partition is not None:
        allowed = {"path", "sample_id_col", "role_col", "split_col", "fold_col", "always_train_value"}
        _reject_unknown(partition, allowed, "partition")
        if "path" not in partition:
            raise ConfigurationError("External partition config requires 'path'.")
        kwargs = {key: value for key, value in partition.items() if key != "path"}
        plan = load_partition_plan(config.resolve_path(partition["path"]), **kwargs)
    if partitioning is not None:
        allowed = {
            "strategy",
            "params",
            "id_col",
            "label_col",
            "group_col",
            "seq_col",
            "cluster_col",
            "date_col",
        }
        _reject_unknown(partitioning, allowed, "partitioning")
        if "strategy" not in partitioning:
            raise ConfigurationError("BioSieve partitioning requires 'strategy'.")
        biosieve = BioSievePartitionConfig(**dict(partitioning))
    return plan, biosieve


def build_tuning_config(payload: Mapping[str, Any]) -> TuningConfig:
    allowed = {
        "optimizer",
        "metrics",
        "refit_metric",
        "refit",
        "n_jobs",
        "random_state",
        "n_iter",
        "n_trials",
        "timeout",
        "factor",
        "resource",
        "max_resources",
        "min_resources",
        "aggressive_elimination",
        "error_score",
        "optuna_storage",
        "optuna_study_name",
        "optuna_load_if_exists",
    }
    _reject_unknown(payload, allowed, "tuning")
    values = dict(payload)
    values["metrics"] = tuple(values.get("metrics", ()))
    return TuningConfig(**values)


def build_benchmark_config(payload: Mapping[str, Any]) -> BenchmarkConfig:
    allowed = {
        "metrics",
        "seeds",
        "modes",
        "include_baselines",
        "fail_fast",
        "evaluation_role",
        "require_complete",
        "return_estimators",
        "tuning",
        "metadata",
    }
    _reject_unknown(payload, allowed, "benchmark")
    if "metrics" not in payload:
        raise ConfigurationError("Benchmark config requires 'metrics'.")
    values = dict(payload)
    values["metrics"] = tuple(values["metrics"])
    if "seeds" in values:
        values["seeds"] = tuple(values["seeds"])
    if "modes" in values:
        values["modes"] = tuple(values["modes"])
    if values.get("tuning") is not None:
        values["tuning"] = build_tuning_config(values["tuning"])
    return BenchmarkConfig(**values)


def build_search_space(name: str, payload: Mapping[str, Any] | None) -> SearchSpace | None:
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


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown {label} keys: {sorted(unknown)!r}.")
