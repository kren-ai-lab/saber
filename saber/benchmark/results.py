"""Structured benchmark results and long-form exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

import polars as pl

from saber.utils.tabular import records_frame

if TYPE_CHECKING:
    from saber.core.prediction import PredictionResult
    from saber.tuning.results import OptimizationResult
    from saber.validation.results import ValidationResult

BenchmarkStatus = Literal["complete", "failed"]


@dataclass(slots=True)
class BenchmarkRun:
    """One algorithm x representation x partition x seed x mode execution."""

    run_id: str
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
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate_metrics(self) -> dict[str, float]:
        """Return the run's aggregate metrics, or an empty dict if unvalidated."""
        if self.validation is None:
            return {}
        return dict(self.validation.aggregate_metrics)

    @property
    def oof_prediction(self) -> PredictionResult | None:
        """Return the run's out-of-fold prediction, if validation was performed."""
        if self.validation is None:
            return None
        return self.validation.oof_prediction


@dataclass(slots=True)
class BenchmarkResult:
    """Collection of benchmark runs with analysis-ready exports."""

    runs: tuple[BenchmarkRun, ...]
    targets: dict[Any, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def successes(self) -> tuple[BenchmarkRun, ...]:
        """Return runs that completed successfully."""
        return tuple(run for run in self.runs if run.status == "complete")

    @property
    def failures(self) -> tuple[BenchmarkRun, ...]:
        """Return runs that failed."""
        return tuple(run for run in self.runs if run.status == "failed")

    @property
    def n_runs(self) -> int:
        """Return the total number of requested runs."""
        return len(self.runs)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly summary; per-run results are in the ``*_frame()`` tables."""
        return {
            "algorithms": list(self.metadata.get("algorithms", ())),
            "task": self.metadata.get("task"),
            "dataset_fingerprints": {
                run.representation: run.metadata.get("dataset_fingerprint") for run in self.runs
            },
            "n_runs": self.n_runs,
            "n_successes": len(self.successes),
            "n_failures": len(self.failures),
        }

    def aggregate_metrics_frame(self) -> pl.DataFrame:
        """Return one long-form row per completed run and aggregate metric."""
        return self._metrics_frame("aggregate")

    def fold_metrics_frame(self) -> pl.DataFrame:
        """Return one long-form row per fold/split and metric."""
        return self._metrics_frame("fold")

    def metrics_frame(self) -> pl.DataFrame:
        """Return aggregate and fold metrics in one analysis-ready table.

        Rows are :meth:`ValidationResult.metrics_frame` rows prefixed with the
        run identity and suffixed with the run's ``elapsed_seconds``.
        """
        return _concat([self.aggregate_metrics_frame(), self.fold_metrics_frame()])

    def _metrics_frame(self, level: str) -> pl.DataFrame:
        return _concat(
            [
                _with_identity(
                    run.validation.metrics_frame().filter(pl.col("level") == level),
                    run,
                    elapsed_seconds=float(run.elapsed_seconds),
                )
                for run in self.successes
                if run.validation is not None
            ]
        )

    def predictions_frame(self) -> pl.DataFrame:
        """Return sample-level held-out predictions for every successful run.

        Columns are the run identity, ``split`` and ``evaluation_role``
        followed by :meth:`PredictionResult.to_frame` columns.
        """
        frames: list[pl.DataFrame] = []
        for run in self.successes:
            if run.validation is None:
                continue
            for fold in run.validation.folds:
                y_true = [self.targets.get(sample_id) for sample_id in fold.evaluation_ids]
                frame = fold.prediction.to_frame(y_true=y_true)
                frames.append(
                    _with_identity(frame, run, split=fold.split, evaluation_role=fold.evaluation_role)
                )
        return _concat(frames)

    def optimization_history_frame(self) -> pl.DataFrame:
        """Return tuning candidate/trial history annotated with benchmark identity."""
        return _concat(
            [
                _with_identity(run.optimization.history_frame(), run)
                for run in self.successes
                if run.optimization is not None
            ]
        )

    def runs_frame(self) -> pl.DataFrame:
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
        return records_frame(rows)


def _with_identity(frame: pl.DataFrame, run: BenchmarkRun, **context: Any) -> pl.DataFrame:
    """Prefix ``frame`` with the run identity and ``context`` columns (``elapsed_seconds`` goes last)."""
    if frame.is_empty():
        return frame
    elapsed = context.pop("elapsed_seconds", None)
    frame = records_frame([_run_identity(run) | context]).join(frame, how="cross")
    if elapsed is not None:
        frame = frame.with_columns(elapsed_seconds=pl.lit(elapsed))
    return frame


def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return pl.DataFrame()
    return pl.concat(non_empty, how="diagonal_relaxed")


def _run_identity(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
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
    payload = repr(sorted(parameters.items(), key=lambda item: item[0]))
    return sha256(payload.encode("utf-8")).hexdigest()[:12]
