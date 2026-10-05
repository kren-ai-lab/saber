"""Persistence writers for model and benchmark artifacts."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

from saber.core.results import TrainResult
from saber.exceptions import PersistenceError
from saber.persistence.checksums import write_checksums
from saber.persistence.environment import environment_snapshot, package_version
from saber.persistence.metadata import ArtifactManifest
from saber.utils.serialization import write_json

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from saber.benchmark.results import BenchmarkResult
    from saber.datasets import DatasetBundle, FeatureSchema, PartitionPlan
    from saber.tuning.results import OptimizationResult


def save_model(
    path: str | Path,
    result: TrainResult | OptimizationResult,
    *,
    dataset: DatasetBundle,
    partition_plan: PartitionPlan | None = None,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist a trained or tuned model with reproducibility metadata.

    Everything the artifact records (fitted model, algorithm, parameters,
    feature schema, positive class, random state and preprocessing) is taken
    from ``result``.  ``dataset`` is the data the model was fitted on; its
    fingerprint travels with the artifact.  For an :class:`OptimizationResult`
    the cross-validated scores are stored as ``selection_scores`` — they
    selected the hyperparameters and are not an unbiased performance estimate —
    and ``partition_plan`` defaults to the tuning plan.
    """
    training_config: dict[str, Any] = {
        "random_state": result.metadata.get("random_state"),
        "preprocessing": result.metadata.get("preprocessing"),
    }
    selection_scores = None
    spec = result.spec
    if isinstance(result, TrainResult):
        model = result.model
        parameters = dict(result.parameters)
    else:
        if result.best_model is None:
            raise PersistenceError("Cannot save an OptimizationResult tuned with refit=False.")
        model = result.best_model
        parameters = dict(result.best_params)
        training_config["tuning"] = result.metadata.get("tuning_config")
        selection_scores = {
            "refit_metric": result.refit_metric,
            "selection_scores": dict(result.best_scores),
        }
        if partition_plan is None:
            partition_plan = result.partition_plan
    if spec is None:
        raise PersistenceError("The result does not record its AlgorithmSpec.")

    return _write_model_artifact(
        path,
        model=model,
        algorithm=spec.name,
        task=spec.task,
        provider=spec.provider,
        dataset=dataset,
        feature_schema=result.feature_schema,
        partition_plan=partition_plan,
        parameters=parameters,
        selection_scores=selection_scores,
        training_config=training_config,
        positive_class=result.positive_class,
        metadata=dict(metadata or {}),
        overwrite=overwrite,
    )


def _write_model_artifact(
    path: str | Path,
    *,
    model: Any,
    algorithm: str,
    task: str,
    provider: str | None,
    dataset: DatasetBundle,
    feature_schema: FeatureSchema | None,
    partition_plan: PartitionPlan | None,
    parameters: dict[str, Any],
    selection_scores: dict[str, Any] | None,
    training_config: dict[str, Any],
    positive_class: Any | None,
    metadata: dict[str, Any],
    overwrite: bool,
) -> Path:
    """Write one fitted estimator/pipeline artifact with checksums."""
    if task not in {"classification", "regression"}:
        raise PersistenceError("task must be 'classification' or 'regression'.")
    schema = dataset.feature_schema
    if feature_schema is not None and feature_schema.fingerprint != schema.fingerprint:
        raise PersistenceError("The result's feature schema does not match the supplied dataset.")
    if partition_plan is not None:
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
            "training_config": "training_config.json",
        }
        if selection_scores is not None:
            files["selection_scores"] = "selection_scores.json"
        if partition_plan is not None:
            files["partition_plan"] = "partition_plan.json"

        joblib.dump(model, root / files["model"])
        environment = environment_snapshot()
        write_json(root / files["environment"], environment)
        write_json(root / files["feature_schema"], schema.to_dict())
        write_json(root / files["parameters"], parameters)
        write_json(root / files["training_config"], training_config)
        if selection_scores is not None:
            write_json(root / files["selection_scores"], selection_scores)

        dataset_fingerprint = dataset.fingerprint
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
            "metadata": metadata,
        }
        write_json(root / files["provenance"], provenance)
        if partition_plan is not None:
            write_json(root / files["partition_plan"], partition_plan.to_dict())

        manifest = ArtifactManifest(
            artifact_type="model",
            saber_version=package_version("saberlib"),
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


def save_benchmark(
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

        result.runs_frame().write_csv(root / files["runs"])
        result.metrics_frame().write_csv(root / files["metrics"])
        result.predictions_frame().write_csv(root / files["predictions"])
        result.optimization_history_frame().write_csv(root / files["optimization_history"])
        if include_object:
            joblib.dump(result, root / files["benchmark_object"])

        manifest = ArtifactManifest(
            artifact_type="benchmark",
            saber_version=package_version("saberlib"),
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


@contextmanager
def _atomic_artifact_directory(target: Path, *, overwrite: bool) -> Iterator[Path]:
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
