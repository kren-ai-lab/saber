"""Rich rendering helpers for the saber command-line interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from saber.benchmark import BenchmarkResult
from saber.config.runner import WorkflowExecution
from saber.config.schema import WorkflowConfig
from saber.tuning import OptimizationResult
from saber.utils.serialization import to_jsonable
from saber.validation import ValidationResult


def render_preflight(console: Console, config: WorkflowConfig) -> None:
    """Render a concise execution plan without duplicating workflow logic."""
    payload = config.payload
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column()
    table.add_row("Workflow", config.workflow)
    if config.source is not None:
        table.add_row("Config", str(config.source))

    algorithm = payload.get("algorithm")
    if algorithm is not None:
        table.add_row("Algorithm", str(algorithm))
    algorithms = payload.get("algorithms")
    if algorithms is not None:
        table.add_row("Algorithms", ", ".join(str(item) for item in algorithms))

    if "datasets" in payload:
        table.add_row("Datasets", ", ".join(str(key) for key in payload["datasets"]))
    elif "dataset" in payload:
        dataset = payload["dataset"]
        if isinstance(dataset, Mapping):
            table.add_row("Dataset", str(dataset.get("path", "prepared input")))

    if "partitioning" in payload:
        partitioning = payload["partitioning"]
        if isinstance(partitioning, Mapping):
            table.add_row("Partitioning", f"BioSieve / {partitioning.get('strategy', 'configured')}")
    elif "partitions" in payload:
        table.add_row("Partitions", ", ".join(str(key) for key in payload["partitions"]))
    elif "partition" in payload:
        partition = payload["partition"]
        if isinstance(partition, Mapping):
            table.add_row("Partition", str(partition.get("path", "external")))

    metrics = _configured_metrics(config)
    if metrics:
        table.add_row("Metrics", ", ".join(metrics))

    output = payload.get("output")
    if isinstance(output, Mapping):
        destination = output.get("directory") or output.get("path")
        if destination:
            table.add_row("Output", str(destination))
    artifact = payload.get("artifact")
    if artifact is not None:
        if isinstance(artifact, Mapping):
            artifact = artifact.get("path")
        if artifact:
            table.add_row("Artifact", str(artifact))

    console.print(Panel(table, title="[bold]saber execution plan[/bold]", border_style="cyan"))


def render_dry_run(console: Console, config: WorkflowConfig) -> None:
    render_preflight(console, config)
    console.print("[green]✓[/green] Configuration is valid. No workflow was executed.")


def render_execution(console: Console, execution: WorkflowExecution) -> None:
    """Render one completed public/config workflow."""
    workflow = execution.config.workflow
    result = execution.result
    console.print()
    console.print(
        Panel(
            Text(f"{workflow.upper()} completed", style="bold green", justify="center"),
            border_style="green",
        )
    )

    if isinstance(result, ValidationResult):
        _render_validation(console, result)
    elif isinstance(result, OptimizationResult):
        _render_optimization(console, result)
    elif isinstance(result, BenchmarkResult):
        _render_benchmark(console, result)
    else:
        _render_summary_mapping(console, execution.summary)

    if execution.outputs:
        _render_outputs(console, execution.outputs)


def render_model_list(console: Console, rows: Sequence[Mapping[str, Any]]) -> None:
    table = Table(title=f"Registered models ({len(rows)})", header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Task")
    table.add_column("Provider")
    table.add_column("Prob.", justify="center")
    table.add_column("Weights", justify="center")
    table.add_column("Scaling")
    table.add_column("Tags")
    for row in rows:
        capabilities = row.get("capabilities", {})
        requirements = row.get("requirements", {})
        table.add_row(
            str(row["name"]),
            str(row["task"]),
            str(row["provider"]),
            _yes_no(bool(capabilities.get("predict_proba"))),
            _yes_no(bool(capabilities.get("sample_weight"))),
            str(requirements.get("scaling", "-")),
            ", ".join(str(tag) for tag in row.get("tags", ())),
        )
    console.print(table)


def render_model(console: Console, metadata: Mapping[str, Any]) -> None:
    title = f"{metadata.get('name')}  [{metadata.get('provider')}]"
    basics = Table.grid(padding=(0, 2))
    basics.add_column(style="bold cyan")
    basics.add_column()
    for key in ("task", "estimator", "description"):
        if metadata.get(key) is not None:
            basics.add_row(key.replace("_", " ").title(), str(metadata[key]))
    aliases = metadata.get("aliases") or ()
    tags = metadata.get("tags") or ()
    if aliases:
        basics.add_row("Aliases", ", ".join(str(value) for value in aliases))
    if tags:
        basics.add_row("Tags", ", ".join(str(value) for value in tags))
    console.print(Panel(basics, title=title, border_style="cyan"))

    capability_table = Table(title="Capabilities & requirements", header_style="bold cyan")
    capability_table.add_column("Property")
    capability_table.add_column("Value")
    for key, value in (metadata.get("capabilities") or {}).items():
        capability_table.add_row(key, _yes_no(value) if isinstance(value, bool) else str(value))
    for key, value in (metadata.get("requirements") or {}).items():
        capability_table.add_row(key, str(value))
    console.print(capability_table)

    defaults = metadata.get("default_params") or {}
    search = "available" if metadata.get("has_search_space") else "not defined"
    footer = f"Default params: {len(defaults)}  •  Search space: {search}  •  CV: {_yes_no(metadata.get('supports_cv', False))}"
    console.print(footer)


def render_artifact(console: Console, payload: Mapping[str, Any], path: str) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("Path", path)
    for key in ("artifact_type", "schema_version", "created_at", "saber_version"):
        if key in payload:
            table.add_row(key.replace("_", " ").title(), str(payload.get(key)))
    files = payload.get("files", {})
    table.add_row("Files", str(len(files)))
    console.print(Panel(table, title="Artifact", border_style="cyan"))


def render_doctor(console: Console, payload: Mapping[str, Any]) -> None:
    table = Table(title="saber doctor", header_style="bold cyan")
    table.add_column("Component")
    table.add_column("Status")
    table.add_column("Version / detail")
    for item in payload.get("components", []):
        available = bool(item.get("available"))
        table.add_row(
            str(item.get("name")),
            "[green]available[/green]" if available else "[yellow]not installed[/yellow]",
            str(item.get("version") or item.get("detail") or "-"),
        )
    console.print(table)


def _render_validation(console: Console, result: ValidationResult) -> None:
    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold cyan")
    overview.add_column()
    overview.add_row("Algorithm", result.algorithm)
    overview.add_row("Task", result.task)
    overview.add_row("Splits", str(result.n_splits))
    overview.add_row("Fit time", f"{result.total_fit_seconds:.3f} s")
    if result.oof_prediction is not None:
        overview.add_row("OOF samples", str(result.oof_prediction.n_samples))
    console.print(overview)
    _render_metric_table(console, result.aggregate_metrics, result.metric_summary)


def _render_optimization(console: Console, result: OptimizationResult) -> None:
    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold cyan")
    overview.add_column()
    overview.add_row("Algorithm", result.algorithm)
    overview.add_row("Optimizer", str(result.optimizer))
    overview.add_row("Refit metric", str(result.refit_metric))
    overview.add_row("Best score", _format_number(result.display_score))
    overview.add_row("Candidates", str(len(result.history)))
    overview.add_row("Failed", str(len(result.failures)))
    overview.add_row("Refit model", _yes_no(result.best_model is not None))
    console.print(overview)

    if result.display_scores:
        _render_metric_table(console, result.display_scores, None, title="Best candidate metrics")
    if result.best_params:
        params = Table(title="Best parameters", header_style="bold cyan")
        params.add_column("Parameter")
        params.add_column("Value")
        for name, value in sorted(result.best_params.items()):
            params.add_row(str(name), str(value))
        console.print(params)


def _render_benchmark(console: Console, result: BenchmarkResult) -> None:
    overview = Table.grid(padding=(0, 2))
    overview.add_column(style="bold cyan")
    overview.add_column()
    overview.add_row("Runs", str(result.n_runs))
    overview.add_row("Successful", str(len(result.successes)))
    overview.add_row("Failed", str(len(result.failures)))
    console.print(overview)

    frame = result.aggregate_metrics_frame()
    if frame.is_empty():
        return
    preview = frame.sort(
        [pl.col("metric"), pl.col("score").fill_nan(None)], descending=[False, True], nulls_last=True
    ).head(16)
    table = Table(title="Aggregate results (preview)", header_style="bold cyan")
    for column in ("representation", "partition", "algorithm", "mode", "seed", "metric", "score"):
        table.add_column(column.replace("_", " ").title())
    for row in preview.iter_rows(named=True):
        table.add_row(
            str(row["representation"]),
            str(row["partition"]),
            str(row["algorithm"]),
            str(row["mode"]),
            str(row["seed"]),
            str(row["metric"]),
            _format_number(row["score"]),
        )
    console.print(table)
    if len(frame) > len(preview):
        console.print(f"[dim]Showing {len(preview)} of {len(frame)} aggregate metric rows.[/dim]")

    if result.failures:
        failures = Table(title="Failed runs", header_style="bold red")
        failures.add_column("Algorithm")
        failures.add_column("Representation")
        failures.add_column("Mode")
        failures.add_column("Error")
        for run in result.failures[:8]:
            failures.add_row(run.algorithm, run.representation, run.mode, _truncate(run.error or "unknown"))
        console.print(failures)


def _render_metric_table(
    console: Console,
    metrics: Mapping[str, float],
    summary: Mapping[str, Mapping[str, float]] | None,
    *,
    title: str = "Metrics",
) -> None:
    table = Table(title=title, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    if summary:
        table.add_column("Std", justify="right")
        table.add_column("Range", justify="right")
    for name, value in metrics.items():
        cells = [name, _format_number(value)]
        if summary:
            stats = summary.get(name, {})
            cells.extend(
                [
                    _format_number(stats.get("std")),
                    f"{_format_number(stats.get('min'))} – {_format_number(stats.get('max'))}",
                ]
            )
        table.add_row(*cells)
    console.print(table)


def _render_summary_mapping(console: Console, summary: Mapping[str, Any]) -> None:
    table = Table(title="Summary", header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in summary.items():
        if key == "workflow":
            continue
        table.add_row(key.replace("_", " ").title(), _display_value(value))
    console.print(table)


def _render_outputs(console: Console, outputs: Mapping[str, str]) -> None:
    table = Table(title="Outputs", header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Path")
    for name, path in sorted(outputs.items()):
        table.add_row(name, path)
    console.print(table)


def _configured_metrics(config: WorkflowConfig) -> list[str]:
    payload = config.payload
    if isinstance(payload.get("metrics"), Sequence) and not isinstance(payload.get("metrics"), (str, bytes)):
        return [str(value) for value in payload["metrics"]]
    tuning = payload.get("tuning")
    if isinstance(tuning, Mapping):
        return [str(value) for value in tuning.get("metrics", ())]
    benchmark = payload.get("benchmark")
    if isinstance(benchmark, Mapping):
        return [str(value) for value in benchmark.get("metrics", ())]
    return []


def _display_value(value: Any) -> str:
    converted = to_jsonable(value)
    if isinstance(converted, Mapping):
        return ", ".join(f"{key}={_format_number(item)}" for key, item in converted.items())
    if isinstance(converted, list):
        return ", ".join(str(item) for item in converted)
    return _format_number(converted)


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) >= 1000 or (value != 0 and abs(value) < 1e-4):
            return f"{value:.4g}"
        return f"{value:.4f}"
    return str(value)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _truncate(value: str, length: int = 80) -> str:
    text = str(value).replace("\n", " ")
    if len(text) <= length:
        return text
    return text[: length - 1] + "…"
