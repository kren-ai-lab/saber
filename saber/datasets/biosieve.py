"""BioSieve integration for partition generation.

saber never reimplements BioSieve split-generation algorithms. This module is
an adapter between :class:`DatasetBundle` / :class:`PartitionPlan` and the
public BioSieve splitting protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import polars as pl

from saber.datasets.folds import PartitionPlan, PartitionSplit
from saber.datasets.schemas import DatasetBundle
from saber.exceptions import OptionalDependencyError, PartitionIntegrationError

_STRATEGIES: dict[str, tuple[str, str]] = {
    "random": ("biosieve.splitting.random", "RandomSplitter"),
    "stratified": ("biosieve.splitting.stratified", "StratifiedSplitter"),
    "stratified_numeric": (
        "biosieve.splitting.stratified_numeric",
        "StratifiedNumericSplitter",
    ),
    "group": ("biosieve.splitting.group", "GroupSplitter"),
    "time": ("biosieve.splitting.time_based", "TimeSplitter"),
    "cluster_aware": ("biosieve.splitting.cluster", "ClusterAwareSplitter"),
    "distance_aware": (
        "biosieve.splitting.distance_aware",
        "DistanceAwareSplitter",
    ),
    "homology_aware": (
        "biosieve.splitting.homology_aware",
        "HomologyAwareSplitter",
    ),
    "random_kfold": (
        "biosieve.splitting.random_kfold",
        "RandomKFoldSplitter",
    ),
    "stratified_kfold": (
        "biosieve.splitting.stratified_kfold",
        "StratifiedKFoldSplitter",
    ),
    "group_kfold": (
        "biosieve.splitting.group_kfold",
        "GroupKFoldSplitter",
    ),
    "stratified_numeric_kfold": (
        "biosieve.splitting.stratified_numeric_kfold",
        "StratifiedNumericKFoldSplitter",
    ),
    "distance_aware_kfold": (
        "biosieve.splitting.distance_aware_kfold",
        "DistanceAwareKFoldSplitter",
    ),
}

_KFOLD_STRATEGIES = {
    "random_kfold",
    "stratified_kfold",
    "group_kfold",
    "stratified_numeric_kfold",
    "distance_aware_kfold",
}


@dataclass(frozen=True, slots=True)
class BioSievePartitionConfig:
    """Configuration for BioSieve-backed partition generation.

    Parameters are forwarded to the corresponding BioSieve splitter. saber
    only adds canonical column names when a strategy needs labels/groups and
    those parameters were not supplied explicitly.
    """

    strategy: str
    params: Mapping[str, Any] = field(default_factory=dict)
    id_col: str = "__saber_sample_id__"
    label_col: str = "__saber_target__"
    group_col: str = "__saber_group__"
    seq_col: str = "sequence"
    cluster_col: str | None = None
    date_col: str | None = None

    def __post_init__(self) -> None:
        strategy = str(self.strategy).strip().lower()
        if not strategy:
            raise PartitionIntegrationError("BioSieve strategy cannot be empty.")
        if strategy not in _STRATEGIES:
            supported = ", ".join(sorted(_STRATEGIES))
            raise PartitionIntegrationError(
                f"Unsupported BioSieve strategy '{strategy}'. Supported: {supported}."
            )
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "params", dict(self.params))


def available_biosieve_strategies() -> tuple[str, ...]:
    """Return BioSieve strategies supported by this adapter."""
    return tuple(sorted(_STRATEGIES))


def partition_with_biosieve(
    dataset: DatasetBundle,
    config: BioSievePartitionConfig,
    *,
    extra_columns: Mapping[str, Sequence[Any]] | None = None,
    splitter: Any | None = None,
) -> PartitionPlan:
    """Generate a :class:`PartitionPlan` through BioSieve.

    BioSieve remains the sole split-generation engine. saber converts the
    resulting sample memberships into its internal identity-based contract.
    ``extra_columns`` can provide prepared aligned information required by
    BioSieve strategies (for example sequence, cluster, time, or descriptor
    columns) without expanding ``DatasetBundle`` into a domain-specific data
    container.
    """
    dataset.validate()
    pl, Columns = _import_biosieve_runtime()

    params = _resolved_strategy_params(dataset, config)
    include_features = (
        config.strategy in {"distance_aware", "distance_aware_kfold"}
        and str(params.get("feature_mode", "embeddings")).lower() == "descriptors"
    )
    if include_features and not params.get("descriptor_cols"):
        params["descriptor_cols"] = list(dataset.feature_names)

    frame = _dataset_to_polars(
        dataset,
        config=config,
        extra_columns=extra_columns,
        polars_module=pl,
        include_features=include_features,
    )
    active_splitter = splitter or _build_splitter(config.strategy, params)

    columns = Columns(
        id_col=config.id_col,
        seq_col=config.seq_col,
        label_col=config.label_col,
        group_col=config.group_col if dataset.groups is not None else None,
        cluster_col=config.cluster_col,
        date_col=config.date_col,
    )

    try:
        if hasattr(active_splitter, "run_folds"):
            raw_results = list(active_splitter.run_folds(frame, columns))
        elif hasattr(active_splitter, "run"):
            raw_results = [active_splitter.run(frame, columns)]
        else:
            raise PartitionIntegrationError("BioSieve splitter must implement run(...) or run_folds(...).")
    except PartitionIntegrationError:
        raise
    except Exception as exc:
        raise PartitionIntegrationError(f"BioSieve strategy '{config.strategy}' failed: {exc}") from exc

    if not raw_results:
        raise PartitionIntegrationError(f"BioSieve strategy '{config.strategy}' produced no split results.")

    splits = tuple(
        _split_result_to_partition(
            result,
            id_col=config.id_col,
            fallback_index=index,
        )
        for index, result in enumerate(raw_results)
    )

    kind = "cross_validation" if len(splits) > 1 or config.strategy in _KFOLD_STRATEGIES else "holdout"
    metadata = {
        "source": "biosieve",
        "biosieve_version": _biosieve_version(),
        "strategy": config.strategy,
        "requested_params": dict(config.params),
    }

    plan = PartitionPlan(
        splits=splits,
        kind=kind,
        dataset_fingerprint=dataset.fingerprint,
        metadata=metadata,
    )
    plan.validate_against(dataset)
    return plan


def _import_biosieve_runtime():
    try:
        pl = import_module("polars")
        types = import_module("biosieve.types")
    except ImportError as exc:
        raise OptionalDependencyError(
            dependency="biosieve",
            extra="biosieve",
            purpose="partition generation",
        ) from exc
    return pl, types.Columns


def _build_splitter(strategy: str, params: Mapping[str, Any]):
    module_name, class_name = _STRATEGIES[strategy]
    try:
        module = import_module(module_name)
        splitter_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise OptionalDependencyError(
            dependency="biosieve",
            extra="biosieve",
            purpose=f"'{strategy}' partition generation",
        ) from exc

    try:
        return splitter_cls(**dict(params))
    except TypeError as exc:
        raise PartitionIntegrationError(
            f"Invalid parameters for BioSieve strategy '{strategy}': {exc}"
        ) from exc


def _resolved_strategy_params(
    dataset: DatasetBundle,
    config: BioSievePartitionConfig,
) -> dict[str, Any]:
    params = dict(config.params)

    if config.strategy in {"stratified", "stratified_kfold"}:
        params.setdefault("label_col", config.label_col)

    if config.strategy in {"stratified_numeric", "stratified_numeric_kfold"}:
        params.setdefault("label_col", config.label_col)

    if config.strategy in {"group", "group_kfold"}:
        if dataset.groups is None:
            raise PartitionIntegrationError(
                f"BioSieve strategy '{config.strategy}' requires DatasetBundle.groups."
            )
        params.setdefault("group_col", config.group_col)

    if config.strategy == "cluster_aware" and config.cluster_col is not None:
        params.setdefault("cluster_col", config.cluster_col)

    if config.strategy == "time" and config.date_col is not None:
        params.setdefault("time_col", config.date_col)

    return params


def _dataset_to_polars(
    dataset: DatasetBundle,
    *,
    config: BioSievePartitionConfig,
    extra_columns: Mapping[str, Sequence[Any]] | None,
    polars_module: Any,
    include_features: bool,
):
    reserved = {config.id_col, config.label_col, config.group_col}
    feature_names = tuple(dataset.feature_names)
    collisions = reserved & set(feature_names)
    if collisions:
        raise PartitionIntegrationError(
            "Feature names collide with saber/BioSieve integration columns: " + ", ".join(sorted(collisions))
        )

    features: dict[str, Any] = {}
    if include_features:
        if isinstance(dataset.X, pl.DataFrame):
            features = {
                name: dataset.X[column].to_numpy() for name, column in zip(feature_names, dataset.X.columns)
            }
        else:
            values = np.asarray(dataset.X)
            features = {name: values[:, index].copy() for index, name in enumerate(feature_names)}

    payload: dict[str, Any] = {
        config.id_col: list(dataset.sample_ids),
        config.label_col: np.asarray(dataset.y).tolist(),
        **features,
    }
    if dataset.groups is not None:
        payload[config.group_col] = list(dataset.groups)

    for name, values in dict(extra_columns or {}).items():
        if name in payload:
            raise PartitionIntegrationError(f"extra_columns cannot overwrite existing column '{name}'.")
        sequence = list(values)
        if len(sequence) != dataset.n_samples:
            raise PartitionIntegrationError(
                f"extra column '{name}' must contain {dataset.n_samples} values; received {len(sequence)}."
            )
        payload[str(name)] = sequence

    try:
        return polars_module.DataFrame(payload)
    except Exception as exc:
        raise PartitionIntegrationError(
            f"Could not convert DatasetBundle to a BioSieve DataFrame: {exc}"
        ) from exc


def _split_result_to_partition(
    result: Any,
    *,
    id_col: str,
    fallback_index: int,
) -> PartitionSplit:
    strategy = str(getattr(result, "strategy", "biosieve"))
    params = dict(getattr(result, "params", {}) or {})
    stats = dict(getattr(result, "stats", {}) or {})

    fold_index = stats.get("fold_index", params.get("fold_index"))
    if fold_index is None:
        name = strategy if fallback_index == 0 else f"{strategy}_{fallback_index}"
    else:
        name = f"fold_{int(fold_index)}"

    train_ids = _frame_ids(getattr(result, "train", None), id_col=id_col, role="train")
    validation_ids = _frame_ids(getattr(result, "val", None), id_col=id_col, role="validation")
    test_ids = _frame_ids(getattr(result, "test", None), id_col=id_col, role="test")

    return PartitionSplit(
        name=name,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        metadata={
            "source": "biosieve",
            "strategy": strategy,
            "params": params,
            "stats": stats,
        },
    )


def _frame_ids(frame: Any, *, id_col: str, role: str) -> tuple[Any, ...]:
    if frame is None:
        return ()
    try:
        columns = tuple(frame.columns)
    except Exception as exc:
        raise PartitionIntegrationError(
            f"BioSieve {role} output is not a supported DataFrame-like object."
        ) from exc

    if id_col not in columns:
        raise PartitionIntegrationError(
            f"BioSieve {role} output does not contain sample ID column '{id_col}'."
        )

    series = frame[id_col]
    to_list = getattr(series, "to_list", None)
    values = to_list() if callable(to_list) else list(series)
    return tuple(_python_scalar(value) for value in values)


def _biosieve_version() -> str:
    try:
        return version("biosieve")
    except PackageNotFoundError:
        try:
            module = import_module("biosieve")
            return str(getattr(module, "__version__", "unknown"))
        except ImportError:
            return "unknown"


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
