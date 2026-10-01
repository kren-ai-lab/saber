"""Execute validated declarative workflows through the public Python API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from saber._api import evaluate, predict, train
from saber.benchmark import benchmark
from saber.config.builders import (
    build_benchmark_config,
    build_partition_plan,
    build_partitioning,
    build_preprocessing,
    build_search_space,
    build_tuning_config,
    load_dataset,
    load_prediction_frame,
)
from saber.config.io import load_config
from saber.config.schema import PARTITIONING_ROLE_COLUMNS
from saber.exceptions import ConfigurationError
from saber.persistence import load_model, save_benchmark, save_model
from saber.tuning import tune
from saber.utils.serialization import to_jsonable
from saber.utils.tabular import records_frame
from saber.validation import validate

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    import polars as pl

    from saber.config.schema import WorkflowConfig
    from saber.datasets import BioSievePartitionConfig, DatasetBundle, PartitionPlan
    from saber.persistence import LoadedModelArtifact
    from saber.validation.partitioning import EvaluationRole


@dataclass(slots=True)
class WorkflowExecution:
    """Result of a config-driven workflow plus deterministic output paths."""

    config: WorkflowConfig
    result: Any
    summary: dict[str, Any]
    outputs: dict[str, str] = field(default_factory=dict)


def run_config(source: str | Path | Mapping[str, Any] | WorkflowConfig) -> WorkflowExecution:
    """Run one YAML/JSON workflow (a config path, mapping or WorkflowConfig) through the public API."""
    config = load_config(source)
    _check_targets(config)
    handler = {
        "train": _run_train,
        "evaluate": _run_evaluate,
        "validate": _run_validate,
        "tune": _run_tune,
        "benchmark": _run_benchmark,
        "predict": _run_predict,
    }[config.workflow]
    return handler(config)


def _run_train(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset, _ = load_dataset(config, payload["dataset"])
    result = train(
        dataset=dataset,
        algorithm=payload["algorithm"],
        model_params=payload.get("model_params"),
        preprocessing=build_preprocessing(payload.get("preprocessing")),
        random_state=payload.get("random_state"),
        positive_class=payload.get("positive_class"),
    )
    outputs: dict[str, str] = {}
    if "artifact" in payload:
        plan = payload.get("partition_plan")
        partition_plan = None if plan is None else build_partition_plan(config, plan)
        outputs["artifact"] = _save_model(config, result, dataset, partition_plan)
    summary = _summary(config, result)
    outputs |= _write_output(config, summary)
    return WorkflowExecution(config, result, summary, outputs)


def _run_evaluate(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset, _ = load_dataset(config, payload["dataset"])
    model_path, model = _load_model(config)
    result = evaluate(
        model,
        dataset,
        metrics=payload.get("metrics"),
        positive_class=payload.get("positive_class"),
    )
    summary = _summary(config, result, model=str(model_path))
    metrics = records_frame([{"metric": name, "score": score} for name, score in result.metrics.items()])
    outputs = _write_output(config, summary, {"metrics": metrics})
    return WorkflowExecution(config, result, summary, outputs)


def _evaluation_role(payload: Mapping[str, Any]) -> EvaluationRole:
    # Config-supplied strings are validated downstream by
    # resolve_evaluation_dataset, which raises ValidationContractError for
    # anything other than "auto"/"validation"/"test".
    return cast("EvaluationRole", str(payload.get("evaluation_role", "auto")))


def _search_kwargs(
    config: WorkflowConfig, dataset: DatasetBundle, extras: Mapping[str, list[Any]]
) -> dict[str, Any]:
    """Keyword arguments shared by the validate and tune workflows."""
    payload = config.payload
    return {
        "dataset": dataset,
        "algorithm": payload["algorithm"],
        "model_params": payload.get("model_params"),
        "preprocessing": build_preprocessing(payload.get("preprocessing")),
        "partition_plan": _partition_plan(config, extras),
        "evaluation_role": _evaluation_role(payload),
        "require_complete": payload.get("require_complete", True),
        "positive_class": payload.get("positive_class"),
        "random_state": payload.get("random_state"),
    }


def _run_validate(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset, extras = load_dataset(config, payload["dataset"], extra_columns=_extra_column_names(config))
    result = validate(
        **_search_kwargs(config, dataset, extras),
        metrics=payload.get("metrics"),
    )
    tables = {"metrics": result.metrics_frame()}
    oof = result.oof_prediction
    if oof is not None:
        targets = dict(zip(dataset.resolved_sample_ids, dataset.y.tolist(), strict=True))
        sample_ids = [] if oof.sample_ids is None else oof.sample_ids.tolist()
        tables["predictions"] = oof.to_frame(y_true=[targets[sample_id] for sample_id in sample_ids])
    summary = _summary(config, result)
    outputs = _write_output(config, summary, tables)
    return WorkflowExecution(config, result, summary, outputs)


def _run_tune(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset, extras = load_dataset(config, payload["dataset"], extra_columns=_extra_column_names(config))
    result = tune(
        **_search_kwargs(config, dataset, extras),
        config=build_tuning_config(payload["tuning"]),
        search_space=build_search_space(payload.get("search_space")),
        metrics=tuple(payload["metrics"]),
    )
    summary = _summary(config, result)
    outputs: dict[str, str] = {}
    if "artifact" in payload:
        outputs["artifact"] = _save_model(config, result, dataset)
    outputs |= _write_output(config, summary, {"optimization_history": result.history_frame()})
    return WorkflowExecution(config, result, summary, outputs)


def _run_benchmark(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    names = _extra_column_names(config)
    datasets: dict[str, DatasetBundle] = {}
    extras: dict[str, dict[str, list[Any]]] = {}
    for label, spec in payload["datasets"].items():
        datasets[str(label)], extras[str(label)] = load_dataset(config, spec, extra_columns=names)
    # benchmark() generates BioSieve partitions from the first dataset.
    reference = str(payload.get("partitioning_reference", next(iter(datasets))))
    datasets = {reference: datasets[reference]} | datasets

    partitions: PartitionPlan | BioSievePartitionConfig | dict[str, PartitionPlan]
    if "partitions" in payload:
        partitions = {
            str(label): build_partition_plan(config, spec) for label, spec in payload["partitions"].items()
        }
    else:
        partitions = _partition_plan(config, extras[reference])

    search_spaces = None
    if "search_spaces" in payload:
        search_spaces = {
            str(algorithm): build_search_space(space) for algorithm, space in payload["search_spaces"].items()
        }

    result = benchmark(
        datasets=datasets,
        algorithms=tuple(payload["algorithms"]),
        model_params=payload.get("model_params"),
        preprocessing=build_preprocessing(payload.get("preprocessing")),
        partitions=partitions,
        config=build_benchmark_config(payload.get("benchmark")),
        search_spaces=search_spaces,
        metrics=tuple(payload["metrics"]),
        evaluation_role=_evaluation_role(payload),
        require_complete=payload.get("require_complete", True),
        positive_class=payload.get("positive_class"),
    )
    summary = _summary(config, result)
    outputs: dict[str, str] = {}
    if "output" in payload:
        # The output directory is the checksummed benchmark artifact; its
        # benchmark_metadata.json carries the summary counts.
        path = save_benchmark(
            config.resolve_path(payload["output"]),
            result,
            metadata=config.metadata,
            overwrite=_overwrite(config),
        )
        outputs["benchmark"] = str(path)
    return WorkflowExecution(config, result, summary, outputs)


def _run_predict(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    X, sample_ids = load_prediction_frame(config, payload["dataset"])
    model_path, model = _load_model(config)
    result = predict(
        model,
        X,
        sample_ids=sample_ids,
        positive_class=payload.get("positive_class"),
    )
    summary = _summary(config, result, model=str(model_path))
    outputs = _write_output(config, summary, {"predictions": result.to_frame()})
    return WorkflowExecution(config, result, summary, outputs)


def _summary(config: WorkflowConfig, result: Any, **extra: Any) -> dict[str, Any]:
    """Return the ``summary.json`` envelope: ``workflow`` then ``result.to_dict()`` and ``extra``."""
    return {"workflow": config.workflow, **result.to_dict(), **extra}


def _write_output(
    config: WorkflowConfig,
    summary: dict[str, Any],
    tables: Mapping[str, pl.DataFrame] | None = None,
) -> dict[str, str]:
    """Write ``summary.json`` and ``<name>.csv`` tables into the ``output`` directory, if any."""
    if "output" not in config.payload:
        return {}
    directory = config.resolve_path(config.payload["output"])
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for name, frame in (tables or {}).items():
        path = directory / f"{name}.csv"
        frame.write_csv(path)
        outputs[name] = str(path)
    path = directory / "summary.json"
    path.write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    outputs["summary"] = str(path)
    return outputs


def _check_targets(config: WorkflowConfig) -> None:
    """Fail before running when ``artifact``/``output`` exists and ``overwrite`` is false."""
    if _overwrite(config):
        return
    for key in ("artifact", "output"):
        if key in config.payload:
            path = config.resolve_path(config.payload[key])
            if path.exists():
                raise ConfigurationError(
                    f"'{key}' path already exists: {path}. Set 'overwrite: true' to replace it."
                )


def _overwrite(config: WorkflowConfig) -> bool:
    return bool(config.payload.get("overwrite", False))


def _save_model(
    config: WorkflowConfig,
    result: Any,
    dataset: DatasetBundle,
    partition_plan: PartitionPlan | None = None,
) -> str:
    path = save_model(
        config.resolve_path(config.payload["artifact"]),
        result,
        dataset=dataset,
        partition_plan=partition_plan,
        metadata=config.metadata,
        overwrite=_overwrite(config),
    )
    return str(path)


def _load_model(config: WorkflowConfig) -> tuple[Path, LoadedModelArtifact]:
    path = config.resolve_path(config.payload["model"])
    return path, load_model(path, strict_environment=config.payload.get("strict_environment", False))


def _extra_column_names(config: WorkflowConfig) -> list[str]:
    partitioning = config.payload.get("partitioning", {})
    names = list(partitioning.get("extra_columns", ()))
    role_cols = (partitioning[key] for key in PARTITIONING_ROLE_COLUMNS if key in partitioning)
    return names + [name for name in role_cols if name not in names]


def _partition_plan(
    config: WorkflowConfig,
    extras: Mapping[str, list[Any]],
) -> PartitionPlan | BioSievePartitionConfig:
    payload = config.payload
    if "partition_plan" in payload:
        return build_partition_plan(config, payload["partition_plan"])
    missing = set(_extra_column_names(config)) - set(extras)
    if missing:
        raise ConfigurationError(f"Dataset is missing partitioning.extra_columns: {sorted(missing)!r}.")
    return build_partitioning(payload["partitioning"], extras)
