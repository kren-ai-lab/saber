"""Execute validated declarative workflows through the public Python API."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from mlcore.api import benchmark, evaluate, predict, train, tune, validate
from mlcore.benchmark import BenchmarkResult
from mlcore.config.builders import (
    build_benchmark_config,
    build_partition_inputs,
    build_preprocessing,
    build_search_space,
    build_tuning_config,
    load_dataset,
    load_prediction_frame,
)
from mlcore.config.io import load_config
from mlcore.config.schema import WorkflowConfig
from mlcore.exceptions import ConfigurationError
from mlcore.persistence import load_model_artifact, save_benchmark_artifact, save_model_artifact
from mlcore.tuning import OptimizationResult
from mlcore.utils.serialization import to_jsonable
from mlcore.validation import ValidationResult


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
    dataset = load_dataset(config, _mapping(payload["dataset"], "dataset"))
    preprocessing = build_preprocessing(_optional_mapping(payload.get("preprocessing"), "preprocessing"))
    partition_plan = None
    if payload.get("partition") is not None:
        partition_plan, _ = build_partition_inputs(
            config,
            partition=_mapping(payload["partition"], "partition"),
            partitioning=None,
        )
    artifact = _optional_mapping(payload.get("artifact"), "artifact")
    artifact_path = None if artifact is None else config.resolve_path(_required(artifact, "path", "artifact"))

    result = train(
        dataset=dataset,
        algorithm=str(payload["algorithm"]),
        preprocessing=preprocessing,
        random_state=payload.get("random_state"),
        model_params=_optional_mapping(payload.get("model_params"), "model_params"),
        artifact_path=artifact_path,
        partition_plan=partition_plan,
        artifact_overwrite=bool(artifact.get("overwrite", False)) if artifact else False,
        metadata=config.metadata,
        positive_class=payload.get("positive_class"),
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
    dataset = load_dataset(config, _mapping(payload["dataset"], "dataset"))
    artifact = payload["artifact"]
    if isinstance(artifact, Mapping):
        artifact_path = config.resolve_path(_required(artifact, "path", "artifact"))
    else:
        artifact_path = config.resolve_path(str(artifact))
    model = load_model_artifact(
        artifact_path,
        strict_environment=bool(payload.get("strict_environment", False)),
    )
    result = evaluate(
        dataset=dataset,
        model=model,
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
    dataset = load_dataset(config, _mapping(payload["dataset"], "dataset"))
    plan, partitioning = build_partition_inputs(
        config,
        partition=_optional_mapping(payload.get("partition"), "partition"),
        partitioning=_optional_mapping(payload.get("partitioning"), "partitioning"),
    )
    result = validate(
        dataset=dataset,
        algorithm=str(payload["algorithm"]),
        partition_plan=plan,
        partitioning=partitioning,
        biosieve_extra_columns=_optional_mapping(payload.get("biosieve_extra_columns"), "biosieve_extra_columns"),
        preprocessing=build_preprocessing(_optional_mapping(payload.get("preprocessing"), "preprocessing")),
        metrics=payload.get("metrics"),
        evaluation_role=str(payload.get("evaluation_role", "auto")),
        positive_class=payload.get("positive_class"),
        random_state=payload.get("random_state"),
        return_estimators=bool(payload.get("return_estimators", False)),
        require_complete=bool(payload.get("require_complete", True)),
        model_params=_optional_mapping(payload.get("model_params"), "model_params"),
    )
    summary = _validation_summary(result)
    outputs: dict[str, str] = {}
    _write_result_tables(config, result, outputs)
    _write_summary(config, summary, outputs)
    return WorkflowExecution(config, result, summary, outputs)


def _run_tune(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    dataset = load_dataset(config, _mapping(payload["dataset"], "dataset"))
    plan, partitioning = build_partition_inputs(
        config,
        partition=_optional_mapping(payload.get("partition"), "partition"),
        partitioning=_optional_mapping(payload.get("partitioning"), "partitioning"),
    )
    tuning_config = build_tuning_config(_mapping(payload["tuning"], "tuning"))
    result = tune(
        dataset=dataset,
        algorithm=str(payload["algorithm"]),
        config=tuning_config,
        partition_plan=plan,
        partitioning=partitioning,
        biosieve_extra_columns=_optional_mapping(payload.get("biosieve_extra_columns"), "biosieve_extra_columns"),
        preprocessing=build_preprocessing(_optional_mapping(payload.get("preprocessing"), "preprocessing")),
        search_space=build_search_space(str(payload["algorithm"]), _optional_mapping(payload.get("search_space"), "search_space")),
        evaluation_role=str(payload.get("evaluation_role", "auto")),
        require_complete=bool(payload.get("require_complete", True)),
        model_params=_optional_mapping(payload.get("model_params"), "model_params"),
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
        result.history_frame().to_csv(history_path, index=False)
        outputs["optimization_history"] = str(history_path)

    artifact = _optional_mapping(payload.get("artifact"), "artifact")
    if artifact is not None:
        if result.best_model is None:
            raise ConfigurationError("Cannot save tuned artifact because tuning used refit=False.")
        artifact_path = config.resolve_path(_required(artifact, "path", "artifact"))
        save_model_artifact(
            artifact_path,
            model=result.best_model,
            algorithm=result.algorithm,
            task=result.spec.task if result.spec is not None else "classification",
            dataset=dataset,
            partition_plan=result.partition_plan,
            provider=result.spec.provider if result.spec is not None else None,
            parameters=dict(result.best_params),
            metrics=result.display_scores,
            training_config={"tuning": _mapping(payload["tuning"], "tuning")},
            metadata=config.metadata,
            overwrite=bool(artifact.get("overwrite", False)),
        )
        outputs["artifact"] = str(artifact_path)

    _write_summary(config, summary, outputs)
    return WorkflowExecution(config, result, summary, outputs)


def _run_benchmark(config: WorkflowConfig) -> WorkflowExecution:
    payload = config.payload
    if "datasets" in payload:
        datasets_payload = _mapping(payload["datasets"], "datasets")
        datasets = {
            str(label): load_dataset(config, _mapping(spec, f"datasets.{label}"))
            for label, spec in datasets_payload.items()
        }
    else:
        datasets = load_dataset(config, _mapping(payload["dataset"], "dataset"))

    partitions = None
    if payload.get("partitions") is not None:
        partitions = {}
        for label, spec in _mapping(payload["partitions"], "partitions").items():
            plan, _ = build_partition_inputs(
                config,
                partition=_mapping(spec, f"partitions.{label}"),
                partitioning=None,
            )
            partitions[str(label)] = plan

    partitioning = None
    if payload.get("partitioning") is not None:
        _, partitioning = build_partition_inputs(
            config,
            partition=None,
            partitioning=_mapping(payload["partitioning"], "partitioning"),
        )

    search_spaces = None
    if payload.get("search_spaces") is not None:
        search_spaces = {
            str(algorithm): build_search_space(str(algorithm), _mapping(space, f"search_spaces.{algorithm}"))
            for algorithm, space in _mapping(payload["search_spaces"], "search_spaces").items()
        }

    result = benchmark(
        datasets=datasets,
        algorithms=tuple(str(value) for value in payload["algorithms"]),
        config=build_benchmark_config(_mapping(payload["benchmark"], "benchmark")),
        partitions=partitions,
        partitioning=partitioning,
        partitioning_reference=payload.get("partitioning_reference"),
        biosieve_extra_columns=_optional_mapping(payload.get("biosieve_extra_columns"), "biosieve_extra_columns"),
        preprocessing=build_preprocessing(_optional_mapping(payload.get("preprocessing"), "preprocessing")),
        model_params=_optional_mapping(payload.get("model_params"), "model_params"),
        search_spaces=search_spaces,
        positive_class=payload.get("positive_class"),
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
        save_benchmark_artifact(
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
    X, sample_ids = load_prediction_frame(config, _mapping(payload["dataset"], "dataset"))
    artifact = payload["artifact"]
    if isinstance(artifact, Mapping):
        artifact_path = config.resolve_path(_required(artifact, "path", "artifact"))
    else:
        artifact_path = config.resolve_path(str(artifact))
    result = predict(
        artifact_path,
        X=X,
        feature_names=list(X.columns),
        sample_ids=sample_ids,
        positive_class=payload.get("positive_class"),
        strict_environment=bool(payload.get("strict_environment", False)),
    )
    summary = {
        "workflow": "predict",
        "task": result.task,
        "n_samples": result.n_samples,
    }
    outputs: dict[str, str] = {}
    output = _output_mapping(config)
    if output and output.get("path"):
        target = config.resolve_path(output["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        _prediction_frame(result).to_csv(target, index=False)
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
        pd.DataFrame(rows).to_csv(path, index=False)
        outputs["metrics"] = str(path)
        if result.oof_prediction is not None:
            pred_path = directory / "predictions.csv"
            _prediction_frame(result.oof_prediction).to_csv(pred_path, index=False)
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
            frame.to_csv(path, index=False)
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


def _prediction_frame(result: Any) -> pd.DataFrame:
    frame: dict[str, Any] = {
        "sample_id": result.sample_ids if result.sample_ids is not None else np.arange(result.n_samples),
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
    return pd.DataFrame(frame)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"'{label}' must be a mapping.")
    return dict(value)


def _optional_mapping(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _mapping(value, label)


def _required(payload: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in payload:
        raise ConfigurationError(f"'{label}' requires '{key}'.")
    return payload[key]


def _output_mapping(config: WorkflowConfig) -> dict[str, Any] | None:
    return _optional_mapping(config.payload.get("output"), "output")
