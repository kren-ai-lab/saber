"""Execute validated declarative workflows through the public Python API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

from saber._api import evaluate, predict, train
from saber.benchmark import BenchmarkResult, benchmark
from saber.config.builders import (
    benchmark_kwargs,
    build_benchmark_config,
    build_partition_inputs,
    build_preprocessing,
    build_search_space,
    build_tuning_config,
    load_dataset,
    load_prediction_frame,
    tuning_kwargs,
)
from saber.config.io import load_config
from saber.config.schema import as_mapping
from saber.exceptions import ConfigurationError
from saber.persistence import load_model, save_benchmark, save_model
from saber.tuning import tune
from saber.utils.serialization import to_jsonable
from saber.utils.tabular import records_frame
from saber.validation import ValidationResult, validate

if TYPE_CHECKING:
    from pathlib import Path

    from saber.config.schema import WorkflowConfig
    from saber.datasets import BioSievePartitionConfig, PartitionPlan
    from saber.validation.partitioning import EvaluationRole


@dataclass(slots=True)
class WorkflowExecution:
    """Result of a config-driven workflow plus deterministic output paths."""

    config: WorkflowConfig
    result: Any
    summary: dict[str, Any]
    outputs: dict[str, str] = field(default_factory=dict)


def run_config(source: str | Path | Mapping[str, Any] | WorkflowConfig) -> WorkflowExecution:
    """Run one YAML/JSON workflow using the same public API as Python callers."""
    config = load_config(source)
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
    dataset = load_dataset(config, as_mapping(payload["dataset"], "dataset"))
    preprocessing = build_preprocessing(_optional_mapping(payload.get("preprocessing"), "preprocessing"))
    artifact = _optional_mapping(payload.get("artifact"), "artifact")
    artifact_path = None if artifact is None else config.resolve_path(_required(artifact, "path", "artifact"))

    result = train(
        dataset=dataset,
        algorithm=str(payload["algorithm"]),
        model_params=_optional_mapping(payload.get("model_params"), "model_params"),
        preprocessing=preprocessing,
        random_state=payload.get("random_state"),
        positive_class=payload.get("positive_class"),
    )
    if artifact is not None and artifact_path is not None:
        partition_plan = build_partition_inputs(
            config,
            partition=_optional_mapping(payload.get("partition"), "partition"),
            partitioning=None,
        )
        save_model(
            artifact_path,
            result,
            dataset=dataset,
            partition_plan=cast("PartitionPlan | None", partition_plan),
            metadata=config.metadata,
            overwrite=bool(artifact.get("overwrite", False)),
        )
    summary = {
        "workflow": "train",
        "algorithm": result.spec.name,
        "task": result.spec.task,
        "provider": result.spec.provider,
        "dataset_fingerprint": dataset.fingerprint,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
    }
    outputs: dict[str, str] = {}
    if artifact_path is not None:
        outputs["artifact"] = str(artifact_path)
    _write_summary(config, summary, outputs)
    return WorkflowExecution(config, result, summary, outputs)


def _run_evaluate(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset = load_dataset(config, as_mapping(payload["dataset"], "dataset"))
    artifact = payload["artifact"]
    if isinstance(artifact, Mapping):
        artifact_path = config.resolve_path(_required(artifact, "path", "artifact"))
    else:
        artifact_path = config.resolve_path(str(artifact))
    model = load_model(
        artifact_path,
        strict_environment=bool(payload.get("strict_environment", False)),
    )
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
        "artifact": str(artifact_path),
    }
    outputs: dict[str, str] = {}
    _write_summary(config, summary, outputs)
    return WorkflowExecution(config, result, summary, outputs)


def _run_validate(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset = load_dataset(config, as_mapping(payload["dataset"], "dataset"))
    result = validate(
        dataset=dataset,
        algorithm=str(payload["algorithm"]),
        model_params=_optional_mapping(payload.get("model_params"), "model_params"),
        preprocessing=build_preprocessing(_optional_mapping(payload.get("preprocessing"), "preprocessing")),
        partition_plan=_partition_input(config),
        metrics=payload.get("metrics"),
        # Config-supplied strings are validated downstream by
        # resolve_evaluation_dataset, which raises ValidationContractError for
        # anything other than "auto"/"validation"/"test".
        evaluation_role=cast("EvaluationRole", str(payload.get("evaluation_role", "auto"))),
        require_complete=bool(payload.get("require_complete", True)),
        positive_class=payload.get("positive_class"),
        return_estimators=bool(payload.get("return_estimators", False)),
        random_state=payload.get("random_state"),
    )
    summary = _validation_summary(result)
    outputs: dict[str, str] = {}
    _write_result_tables(config, result, outputs)
    _write_summary(config, summary, outputs)
    return WorkflowExecution(config, result, summary, outputs)


def _run_tune(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset = load_dataset(config, as_mapping(payload["dataset"], "dataset"))
    tuning_payload = as_mapping(payload["tuning"], "tuning")
    result = tune(
        dataset=dataset,
        algorithm=str(payload["algorithm"]),
        model_params=_optional_mapping(payload.get("model_params"), "model_params"),
        preprocessing=build_preprocessing(_optional_mapping(payload.get("preprocessing"), "preprocessing")),
        partition_plan=_partition_input(config),
        config=build_tuning_config(tuning_payload),
        search_space=build_search_space(
            str(payload["algorithm"]), _optional_mapping(payload.get("search_space"), "search_space")
        ),
        **tuning_kwargs(tuning_payload),
        # Config-supplied strings are validated downstream by
        # resolve_evaluation_dataset, which raises ValidationContractError for
        # anything other than "auto"/"validation"/"test".
        evaluation_role=cast("EvaluationRole", str(payload.get("evaluation_role", "auto"))),
        require_complete=bool(payload.get("require_complete", True)),
        positive_class=payload.get("positive_class"),
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
    output = _output_mapping(config)
    if output and output.get("directory"):
        directory = config.resolve_path(output["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        history_path = directory / "optimization_history.csv"
        result.history_frame().write_csv(history_path)
        outputs["optimization_history"] = str(history_path)

    artifact = _optional_mapping(payload.get("artifact"), "artifact")
    if artifact is not None:
        if result.best_model is None:
            raise ConfigurationError("Cannot save tuned artifact because tuning used refit=False.")
        artifact_path = config.resolve_path(_required(artifact, "path", "artifact"))
        save_model(
            artifact_path,
            result,
            dataset=dataset,
            metadata=config.metadata,
            overwrite=bool(artifact.get("overwrite", False)),
        )
        outputs["artifact"] = str(artifact_path)

    _write_summary(config, summary, outputs)
    return WorkflowExecution(config, result, summary, outputs)


def _run_benchmark(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    if "datasets" in payload:
        datasets_payload = as_mapping(payload["datasets"], "datasets")
        datasets = {
            str(label): load_dataset(config, as_mapping(spec, f"datasets.{label}"))
            for label, spec in datasets_payload.items()
        }
    else:
        datasets = load_dataset(config, as_mapping(payload["dataset"], "dataset"))

    reference = payload.get("partitioning_reference")
    if reference is not None:
        # benchmark() generates BioSieve partitions from the first dataset.
        if not isinstance(datasets, dict) or str(reference) not in datasets:
            raise ConfigurationError(
                f"partitioning_reference '{reference}' is not a benchmark dataset label."
            )
        datasets = {str(reference): datasets[str(reference)]} | datasets

    partitions: Any = None
    if payload.get("partitions") is not None:
        partitions = {
            str(label): build_partition_inputs(
                config,
                partition=as_mapping(spec, f"partitions.{label}"),
                partitioning=None,
            )
            for label, spec in as_mapping(payload["partitions"], "partitions").items()
        }
    elif payload.get("partitioning") is not None:
        partitions = build_partition_inputs(
            config,
            partition=None,
            partitioning=as_mapping(payload["partitioning"], "partitioning"),
            extra_columns=_optional_mapping(payload.get("biosieve_extra_columns"), "biosieve_extra_columns"),
        )

    search_spaces = None
    if payload.get("search_spaces") is not None:
        search_spaces = {
            str(algorithm): build_search_space(
                str(algorithm), as_mapping(space, f"search_spaces.{algorithm}")
            )
            for algorithm, space in as_mapping(payload["search_spaces"], "search_spaces").items()
        }

    benchmark_payload = as_mapping(payload["benchmark"], "benchmark")
    result = benchmark(
        datasets=datasets,
        algorithms=tuple(str(value) for value in payload["algorithms"]),
        model_params=_optional_mapping(payload.get("model_params"), "model_params"),
        preprocessing=build_preprocessing(_optional_mapping(payload.get("preprocessing"), "preprocessing")),
        partitions=partitions,
        config=build_benchmark_config(benchmark_payload),
        search_spaces=search_spaces,
        positive_class=payload.get("positive_class"),
        **benchmark_kwargs(benchmark_payload),
    )
    summary = {
        "workflow": "benchmark",
        "n_runs": result.n_runs,
        "n_successes": len(result.successes),
        "n_failures": len(result.failures),
    }
    outputs: dict[str, str] = {}
    artifact = _optional_mapping(payload.get("artifact"), "artifact")
    if artifact is not None:
        artifact_path = config.resolve_path(_required(artifact, "path", "artifact"))
        save_benchmark(
            artifact_path,
            result,
            metadata=config.metadata,
            include_object=bool(artifact.get("include_object", False)),
            overwrite=bool(artifact.get("overwrite", False)),
        )
        outputs["artifact"] = str(artifact_path)
    _write_result_tables(config, result, outputs)
    _write_summary(config, summary, outputs)
    return WorkflowExecution(config, result, summary, outputs)


def _run_predict(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    X, sample_ids = load_prediction_frame(config, as_mapping(payload["dataset"], "dataset"))
    artifact = payload["artifact"]
    if isinstance(artifact, Mapping):
        artifact_path = config.resolve_path(_required(artifact, "path", "artifact"))
    else:
        artifact_path = config.resolve_path(str(artifact))
    model = load_model(
        artifact_path,
        strict_environment=bool(payload.get("strict_environment", False)),
    )
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
    outputs: dict[str, str] = {}
    output = _output_mapping(config)
    if output and (output.get("path") or output.get("directory")):
        target = (
            config.resolve_path(output["path"])
            if output.get("path")
            else config.resolve_path(output["directory"]) / "predictions.csv"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        _prediction_frame(result).write_csv(target)
        outputs["predictions"] = str(target)
    _write_summary(config, summary, outputs)
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


def _write_result_tables(config: WorkflowConfig, result: Any, outputs: dict[str, str]) -> None:
    output = _output_mapping(config)
    if not output or not output.get("directory"):
        return
    directory = config.resolve_path(output["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(result, ValidationResult):
        rows = []
        for fold in result.folds:
            for metric, score in fold.evaluation.metrics.items():
                rows.append({"split": fold.split_name, "metric": metric, "score": score})
        path = directory / "metrics.csv"
        records_frame(rows).write_csv(path)
        outputs["metrics"] = str(path)
        if result.oof_prediction is not None:
            pred_path = directory / "predictions.csv"
            _prediction_frame(result.oof_prediction).write_csv(pred_path)
            outputs["predictions"] = str(pred_path)
    elif isinstance(result, BenchmarkResult):
        tables = {
            "metrics": result.metrics_frame(),
            "predictions": result.predictions_frame(),
            "failures": result.failures_frame(),
            "optimization_history": result.optimization_history_frame(),
        }
        for name, frame in tables.items():
            path = directory / f"{name}.csv"
            frame.write_csv(path)
            outputs[name] = str(path)


def _write_summary(config: WorkflowConfig, summary: dict[str, Any], outputs: dict[str, str]) -> None:
    output = _output_mapping(config)
    if not output:
        return
    target = None
    if output.get("summary"):
        target = config.resolve_path(output["summary"])
    elif output.get("directory"):
        target = config.resolve_path(output["directory"]) / "summary.json"
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(to_jsonable(summary), indent=2) + "\n", encoding="utf-8")
        outputs["summary"] = str(target)


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


def _partition_input(config: WorkflowConfig) -> PartitionPlan | BioSievePartitionConfig | None:
    payload = config.payload
    return build_partition_inputs(
        config,
        partition=_optional_mapping(payload.get("partition"), "partition"),
        partitioning=_optional_mapping(payload.get("partitioning"), "partitioning"),
        extra_columns=_optional_mapping(payload.get("biosieve_extra_columns"), "biosieve_extra_columns"),
    )


def _optional_mapping(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return as_mapping(value, label)


def _required(payload: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in payload:
        raise ConfigurationError(f"'{label}' requires '{key}'.")
    return payload[key]


def _output_mapping(config: WorkflowConfig) -> dict[str, Any] | None:
    return _optional_mapping(config.payload.get("output"), "output")
