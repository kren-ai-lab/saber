"""Execute validated declarative workflows through the public Python API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

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
from saber.exceptions import ConfigurationError
from saber.persistence import load_model, save_benchmark, save_model
from saber.tuning import tune
from saber.utils.serialization import to_jsonable
from saber.utils.tabular import records_frame
from saber.validation import ValidationResult, validate

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

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
    summary = {
        "workflow": "train",
        "algorithm": result.spec.name,
        "task": result.spec.task,
        "provider": result.spec.provider,
        "dataset_fingerprint": dataset.fingerprint,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
    }
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
    summary = {
        "workflow": "evaluate",
        "task": result.task,
        "metrics": dict(result.metrics),
        "model": str(model_path),
    }
    metrics = records_frame([{"metric": name, "score": score} for name, score in result.metrics.items()])
    outputs = _write_output(config, summary, {"metrics": metrics})
    return WorkflowExecution(config, result, summary, outputs)


def _run_validate(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset, extras = load_dataset(config, payload["dataset"], extra_columns=_extra_column_names(config))
    result = validate(
        dataset=dataset,
        algorithm=payload["algorithm"],
        model_params=payload.get("model_params"),
        preprocessing=build_preprocessing(payload.get("preprocessing")),
        partition_plan=_partition_plan(config, extras),
        metrics=payload.get("metrics"),
        # Config-supplied strings are validated downstream by
        # resolve_evaluation_dataset, which raises ValidationContractError for
        # anything other than "auto"/"validation"/"test".
        evaluation_role=cast("EvaluationRole", str(payload.get("evaluation_role", "auto"))),
        require_complete=payload.get("require_complete", True),
        positive_class=payload.get("positive_class"),
        random_state=payload.get("random_state"),
    )
    rows = [
        {"split": fold.split_name, "metric": metric, "score": score}
        for fold in result.folds
        for metric, score in fold.evaluation.metrics.items()
    ]
    tables = {"metrics": records_frame(rows)}
    if result.oof_prediction is not None:
        tables["predictions"] = _prediction_frame(result.oof_prediction)
    summary = _validation_summary(result)
    outputs = _write_output(config, summary, tables)
    return WorkflowExecution(config, result, summary, outputs)


def _run_tune(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset, extras = load_dataset(config, payload["dataset"], extra_columns=_extra_column_names(config))
    result = tune(
        dataset=dataset,
        algorithm=payload["algorithm"],
        model_params=payload.get("model_params"),
        preprocessing=build_preprocessing(payload.get("preprocessing")),
        partition_plan=_partition_plan(config, extras),
        config=build_tuning_config(payload["tuning"]),
        search_space=build_search_space(payload["algorithm"], payload.get("search_space")),
        metrics=tuple(payload["metrics"]),
        # Config-supplied strings are validated downstream by
        # resolve_evaluation_dataset, which raises ValidationContractError for
        # anything other than "auto"/"validation"/"test".
        evaluation_role=cast("EvaluationRole", str(payload.get("evaluation_role", "auto"))),
        require_complete=payload.get("require_complete", True),
        positive_class=payload.get("positive_class"),
        random_state=payload.get("random_state"),
    )
    summary = {
        "workflow": "tune",
        "algorithm": result.algorithm,
        "optimizer": result.optimizer,
        "refit_metric": result.refit_metric,
        "best_score": result.display_score,
        "best_scores": result.display_scores,
        "best_params": dict(result.best_params),
    }
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
            str(algorithm): build_search_space(str(algorithm), space)
            for algorithm, space in payload["search_spaces"].items()
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
        evaluation_role=cast("EvaluationRole", str(payload.get("evaluation_role", "auto"))),
        require_complete=payload.get("require_complete", True),
        positive_class=payload.get("positive_class"),
    )
    summary = {
        "workflow": "benchmark",
        "n_runs": result.n_runs,
        "n_successes": len(result.successes),
        "n_failures": len(result.failures),
    }
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
    _, model = _load_model(config)
    result = predict(
        model,
        X,
        sample_ids=sample_ids,
        positive_class=payload.get("positive_class"),
    )
    summary = {
        "workflow": "predict",
        "task": result.task,
        "n_samples": result.n_samples,
    }
    outputs = _write_output(config, summary, {"predictions": _prediction_frame(result)})
    return WorkflowExecution(config, result, summary, outputs)


def _validation_summary(result: ValidationResult) -> dict[str, Any]:
    return {
        "workflow": "validate",
        "algorithm": result.algorithm,
        "task": result.task,
        "n_splits": result.n_splits,
        "aggregate_metrics": dict(result.aggregate_metrics),
        "total_fit_seconds": result.total_fit_seconds,
    }


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
    role_cols = (partitioning[key] for key in ("seq_col", "cluster_col", "date_col") if key in partitioning)
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


def _prediction_frame(result: Any) -> pl.DataFrame:
    sample_ids = list(result.sample_ids) if result.sample_ids is not None else list(range(result.n_samples))
    frame: dict[str, Any] = {
        "sample_id": sample_ids,
        "prediction": result.predictions,
    }
    if result.probabilities is not None:
        probabilities = np.asarray(result.probabilities)
        if probabilities.ndim == 1:
            frame["probability"] = probabilities
        elif result.classes is not None:
            for index, label in enumerate(result.classes):
                frame[f"probability__{label}"] = probabilities[:, index]
    if result.decision_scores is not None:
        scores = np.asarray(result.decision_scores)
        if scores.ndim == 1:
            frame["decision_score"] = scores
    return pl.DataFrame(frame)
