"""Persistence writers for model and benchmark artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from saber.benchmark.results import BenchmarkResult
from saber.datasets import DatasetBundle, FeatureSchema, PartitionPlan
from saber.exceptions import PersistenceError
from saber.persistence.checksums import write_checksums
from saber.persistence.environment import environment_snapshot
from saber.persistence.metadata import ArtifactManifest
from saber.utils.serialization import stable_json_dumps, write_json


def save_model_artifact(
    path: str | Path,
    *,
    model: Any,
    algorithm: str,
    task: str,
    dataset: DatasetBundle | None = None,
    feature_schema: FeatureSchema | None = None,
    partition_plan: PartitionPlan | None = None,
    provider: str | None = None,
    parameters: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    training_config: dict[str, Any] | None = None,
    positive_class: Any | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist one fitted estimator/pipeline with reproducibility metadata."""
    if task not in {"classification", "regression"}:
        raise PersistenceError("task must be 'classification' or 'regression'.")
    if dataset is None and feature_schema is None:
        raise PersistenceError(
            "save_model_artifact requires dataset or feature_schema for inference validation."
        )
    schema = dataset.feature_schema if dataset is not None else feature_schema
    if schema is None:
        raise PersistenceError("Feature schema could not be resolved.")
    if (
        dataset is not None
        and feature_schema is not None
        and feature_schema.fingerprint != dataset.feature_schema.fingerprint
    ):
        raise PersistenceError("Explicit feature_schema does not match the supplied dataset.")
    if partition_plan is not None and dataset is not None:
        partition_plan.validate_against(dataset, require_complete=False)

    observed_n_features = getattr(model, "n_features_in_", None)
    if observed_n_features is not None and int(observed_n_features) != schema.n_features:
        raise PersistenceError("Fitted model feature count does not match the persisted FeatureSchema.")

    classes = getattr(model, "classes_", None)
    if task == "classification" and positive_class is None and classes is not None and len(classes) == 2:
        positive_class = classes[-1]

    target = Path(path)
    with _atomic_artifact_directory(target, overwrite=overwrite) as root:
        files = {
            "model": "model.joblib",
            "manifest": "manifest.json",
            "environment": "environment.json",
            "feature_schema": "feature_schema.json",
            "provenance": "provenance.json",
            "parameters": "parameters.json",
            "metrics": "metrics.json",
            "training_config": "training_config.json",
        }
        if partition_plan is not None:
            files["partition_plan"] = "partition_plan.json"

        joblib.dump(model, root / files["model"])
        environment = environment_snapshot()
        write_json(root / files["environment"], environment)
        write_json(root / files["feature_schema"], schema.to_dict())
        write_json(root / files["parameters"], parameters or {})
        write_json(root / files["metrics"], metrics or {})
        write_json(root / files["training_config"], training_config or {})

        dataset_fingerprint = dataset.fingerprint if dataset is not None else None
        partition_fingerprint = partition_plan.fingerprint if partition_plan is not None else None
        provenance = {
            "algorithm": algorithm,
            "task": task,
            "provider": provider,
            "positive_class": positive_class,
            "classes": classes,
            "dataset_fingerprint": dataset_fingerprint,
            "partition_fingerprint": partition_fingerprint,
            "feature_schema_fingerprint": schema.fingerprint,
            "n_features": schema.n_features,
            "metadata": dict(metadata or {}),
        }
        write_json(root / files["provenance"], provenance)
        if partition_plan is not None:
            write_json(root / files["partition_plan"], partition_plan.to_dict())

        manifest = ArtifactManifest(
            artifact_type="model",
            saber_version=_saber_version(),
            files=files,
            metadata={
                "algorithm": algorithm,
                "task": task,
                "provider": provider,
                "dataset_fingerprint": dataset_fingerprint,
                "partition_fingerprint": partition_fingerprint,
                "feature_schema_fingerprint": schema.fingerprint,
            },
        )
        write_json(root / files["manifest"], manifest.to_dict())
        write_checksums(root)

    return target


def save_benchmark_artifact(
    path: str | Path,
    result: BenchmarkResult,
    *,
    metadata: dict[str, Any] | None = None,
    include_object: bool = False,
    overwrite: bool = False,
) -> Path:
    """Persist analysis-ready benchmark outputs and provenance.

    The fitted fold estimators are intentionally not required for the default
    benchmark artifact. ``include_object=True`` additionally stores the full
    Python result object with joblib for trusted same-ecosystem round trips.
    """
    target = Path(path)
    with _atomic_artifact_directory(target, overwrite=overwrite) as root:
        files = {
            "manifest": "manifest.json",
            "environment": "environment.json",
            "benchmark_metadata": "benchmark_metadata.json",
            "runs": "runs.csv",
            "metrics": "metrics.csv",
            "predictions": "predictions.csv",
            "failures": "failures.csv",
            "optimization_history": "optimization_history.csv",
        }
        if include_object:
            files["benchmark_object"] = "benchmark.joblib"

        environment = environment_snapshot()
        write_json(root / files["environment"], environment)
        benchmark_metadata = {
            "result_metadata": dict(result.metadata),
            "n_runs": result.n_runs,
            "n_successes": len(result.successes),
            "n_failures": len(result.failures),
            "metadata": dict(metadata or {}),
        }
        write_json(root / files["benchmark_metadata"], benchmark_metadata)

        _write_frame(result.runs_frame(), root / files["runs"])
        _write_frame(result.metrics_frame(), root / files["metrics"])
        _write_frame(result.predictions_frame(), root / files["predictions"])
        _write_frame(result.failures_frame(), root / files["failures"])
        _write_frame(
            result.optimization_history_frame(),
            root / files["optimization_history"],
        )
        if include_object:
            joblib.dump(result, root / files["benchmark_object"])

        manifest = ArtifactManifest(
            artifact_type="benchmark",
            saber_version=_saber_version(),
            files=files,
            metadata={
                "n_runs": result.n_runs,
                "n_successes": len(result.successes),
                "n_failures": len(result.failures),
                "includes_python_object": include_object,
            },
        )
        write_json(root / files["manifest"], manifest.to_dict())
        write_checksums(root)

    return target


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    normalized = frame.copy()
    for column in normalized.columns:
        if normalized[column].dtype == object:
            normalized[column] = normalized[column].map(_csv_cell)
    normalized.to_csv(path, index=False)


def _csv_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return stable_json_dumps(value)
    return value


@contextmanager
def _atomic_artifact_directory(target: Path, *, overwrite: bool):
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise PersistenceError(f"Artifact path already exists: '{target}'. Use overwrite=True to replace it.")

    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        yield temp
        backup = None
        if target.exists():
            backup = target.with_name(f".{target.name}.backup-{os.getpid()}")
            if backup.exists():
                shutil.rmtree(backup)
            target.rename(backup)
        try:
            temp.rename(target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                backup.rename(target)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
    except Exception:
        if temp.exists():
            shutil.rmtree(temp, ignore_errors=True)
        raise


def _saber_version() -> str | None:
    try:
        return importlib_metadata.version("saberlib")
    except importlib_metadata.PackageNotFoundError:
        try:
            from saber import __version__

            return __version__
        except Exception:
            return None
