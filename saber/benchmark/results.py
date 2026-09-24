"""Structured benchmark results and long-form exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from saber.core.prediction import PredictionResult
from saber.tuning.results import OptimizationResult
from saber.validation.results import ValidationResult

BenchmarkStatus = Literal["complete", "failed"]


@dataclass(slots=True)
class BenchmarkRun:
    """One algorithm × representation × partition × seed × mode execution."""

    run_id: str
    dataset_label: str
    representation: str
    partition_label: str
    algorithm: str
    provider: str | None
    task: str | None
    mode: str
    seed: int | None
    status: BenchmarkStatus
    validation: ValidationResult | None = None
    optimization: OptimizationResult | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    targets: dict[Any, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate_metrics(self) -> dict[str, float]:
        if self.validation is None:
            return {}
        return dict(self.validation.aggregate_metrics)

    @property
    def oof_prediction(self) -> PredictionResult | None:
        if self.validation is None:
            return None
        return self.validation.oof_prediction


@dataclass(slots=True)
class BenchmarkResult:
    """Collection of benchmark runs with analysis-ready exports."""

    runs: tuple[BenchmarkRun, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def successes(self) -> tuple[BenchmarkRun, ...]:
        return tuple(run for run in self.runs if run.status == "complete")

    @property
    def failures(self) -> tuple[BenchmarkRun, ...]:
        return tuple(run for run in self.runs if run.status == "failed")

    @property
    def n_runs(self) -> int:
        return len(self.runs)

    def aggregate_metrics_frame(self) -> pd.DataFrame:
        """Return one long-form row per completed run and aggregate metric."""

        rows: list[dict[str, Any]] = []
        for run in self.successes:
            for metric, value in run.aggregate_metrics.items():
                rows.append(
                    {
                        **_run_identity(run),
                        "level": "aggregate",
                        "split": None,
                        "evaluation_role": None,
                        "metric": metric,
                        "score": float(value),
                        "elapsed_seconds": float(run.elapsed_seconds),
                    }
                )
        return pd.DataFrame(rows)

    def fold_metrics_frame(self) -> pd.DataFrame:
        """Return one long-form row per fold/split and metric."""

        rows: list[dict[str, Any]] = []
        for run in self.successes:
            if run.validation is None:
                continue
            for fold in run.validation.folds:
                for metric, value in fold.evaluation.metrics.items():
                    rows.append(
                        {
                            **_run_identity(run),
                            "level": "fold",
                            "split": fold.split_name,
                            "evaluation_role": fold.evaluation_role,
                            "metric": metric,
                            "score": float(value),
                            "fit_seconds": float(fold.fit_seconds),
                            "elapsed_seconds": float(run.elapsed_seconds),
                        }
                    )
        return pd.DataFrame(rows)

    def metrics_frame(self) -> pd.DataFrame:
        """Return aggregate and fold metrics in one analysis-ready table."""

        frames = [self.aggregate_metrics_frame(), self.fold_metrics_frame()]
        non_empty = [frame for frame in frames if not frame.empty]
        if not non_empty:
            return pd.DataFrame()
        return pd.concat(non_empty, ignore_index=True, sort=False)

    def predictions_frame(self) -> pd.DataFrame:
        """Return sample-level held-out predictions for every successful run."""

        rows: list[dict[str, Any]] = []
        for run in self.successes:
            validation = run.validation
            if validation is None:
                continue
            for fold in validation.folds:
                prediction = fold.prediction
                for index, sample_id in enumerate(prediction.sample_ids):
                    row: dict[str, Any] = {
                        **_run_identity(run),
                        "split": fold.split_name,
                        "evaluation_role": fold.evaluation_role,
                        "sample_id": sample_id,
                        "y_true": run.targets.get(sample_id),
                        "y_pred": prediction.predictions[index],
                    }
                    if prediction.probabilities is not None:
                        values = np.asarray(prediction.probabilities)
                        if values.ndim == 1:
                            row["probability"] = float(values[index])
                        elif prediction.classes is not None:
                            for class_index, class_label in enumerate(prediction.classes):
                                row[f"probability__{class_label}"] = float(
                                    values[index, class_index]
                                )
                    if prediction.decision_scores is not None:
                        values = np.asarray(prediction.decision_scores)
                        if values.ndim == 1:
                            row["decision_score"] = float(values[index])
                        elif prediction.classes is not None and values.shape[1] == len(prediction.classes):
                            for class_index, class_label in enumerate(prediction.classes):
                                row[f"decision_score__{class_label}"] = float(
                                    values[index, class_index]
                                )
                    rows.append(row)
        return pd.DataFrame(rows)


    def optimization_history_frame(self) -> pd.DataFrame:
        """Return tuning candidate/trial history annotated with benchmark identity."""

        frames: list[pd.DataFrame] = []
        for run in self.successes:
            if run.optimization is None:
                continue
            frame = run.optimization.history_frame().copy()
            if frame.empty:
                continue
            identity = _run_identity(run)
            for column, value in reversed(tuple(identity.items())):
                frame.insert(0, column, value)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def failures_frame(self) -> pd.DataFrame:
        """Return failed runs without discarding successful benchmark results."""

        return pd.DataFrame(
            [
                {
                    **_run_identity(run),
                    "error": run.error,
                    "elapsed_seconds": float(run.elapsed_seconds),
                }
                for run in self.failures
            ]
        )

    def runs_frame(self) -> pd.DataFrame:
        """Return one row per requested benchmark run."""

        rows = []
        for run in self.runs:
            row = {
                **_run_identity(run),
                "status": run.status,
                "error": run.error,
                "elapsed_seconds": float(run.elapsed_seconds),
            }
            row.update({f"metric__{name}": value for name, value in run.aggregate_metrics.items()})
            rows.append(row)
        return pd.DataFrame(rows)


def _run_identity(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "dataset": run.dataset_label,
        "representation": run.representation,
        "partition": run.partition_label,
        "algorithm": run.algorithm,
        "provider": run.provider,
        "task": run.task,
        "mode": run.mode,
        "seed": run.seed,
        "configuration_id": _configuration_id(run.parameters),
        "parameters": dict(run.parameters),
    }


def _configuration_id(parameters: dict[str, Any]) -> str:
    from hashlib import sha256

    payload = repr(sorted(parameters.items(), key=lambda item: item[0]))
    return sha256(payload.encode("utf-8")).hexdigest()[:12]
