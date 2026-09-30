"""Command-line interface for saber's public API/config layer."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console

from saber import __version__
from saber.cli.render import (
    render_artifact,
    render_dry_run,
    render_execution,
    render_model,
    render_model_list,
    render_preflight,
)
from saber.config import load_config, run_config
from saber.core.registry import ALGORITHMS, get_algorithm
from saber.exceptions import ConfigurationError, SaberError
from saber.persistence import inspect_artifact
from saber.utils.serialization import to_jsonable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from saber.core.specs import AlgorithmSpec

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_WORKFLOW = 3
EXIT_INTERNAL = 4

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    name="saber",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Classical supervised ML workflows with reproducible partitions, tuning, benchmarking, and artifacts."
    ),
)
models_app = typer.Typer(no_args_is_help=True, help="Discover registered algorithms and their capabilities.")
artifact_app = typer.Typer(no_args_is_help=True, help="Inspect or verify persistence artifacts.")
app.add_typer(models_app, name="models")
app.add_typer(artifact_app, name="artifact")


class TaskChoice(StrEnum):
    """Restricts the ``--task`` option to classification or regression."""

    classification = "classification"
    regression = "regression"


ConfigArg = Annotated[Path, typer.Argument(help="YAML or JSON workflow configuration.")]
DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Validate and show the execution plan without running it.")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON instead of rich output.")]
TaskOpt = Annotated[
    TaskChoice | None, typer.Option("--task", help="Restrict to classification or regression models.")
]
ProviderOpt = Annotated[str | None, typer.Option("--provider")]
TagOpt = Annotated[str | None, typer.Option("--tag")]


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"saber {__version__}")
        raise typer.Exit


@app.callback()
def root(
    _version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """Saber CLI root."""


@app.command(
    "run",
    help="Run the workflow declared by a validated YAML/JSON config.",
    epilog="Examples: saber run experiment.yaml · saber run experiment.yaml --dry-run · --json",
)
def run(config: ConfigArg, dry_run: DryRunOpt = False, json_output: JsonOpt = False) -> None:
    """Run a workflow config, or preview it with ``--dry-run``."""
    console = Console()
    loaded = load_config(config)
    if dry_run:
        if json_output:
            _print_json(console, {"status": "valid", "config": loaded.to_dict()})
        else:
            render_dry_run(console, loaded)
        return
    if json_output:
        execution = run_config(loaded)
        _print_json(console, {"status": "ok", "summary": execution.summary, "outputs": execution.outputs})
        return
    render_preflight(console, loaded)
    with console.status(f"[bold cyan]Running {loaded.workflow} workflow…[/bold cyan]", spinner="dots"):
        execution = run_config(loaded)
    render_execution(console, execution)


@models_app.command("list", help="List available algorithms.")
def models_list(
    task: TaskOpt = None, provider: ProviderOpt = None, tag: TagOpt = None, json_output: JsonOpt = False
) -> None:
    """List registered models, optionally filtered by task, provider, or tag."""
    _print_model_rows(Console(), _filtered_specs(task, provider, tag), json_output=json_output)


@models_app.command("show", help="Show detailed metadata for one algorithm.")
def models_show(name: str, json_output: JsonOpt = False) -> None:
    """Show metadata for one registered model."""
    console = Console()
    metadata = get_algorithm(name).metadata()
    if json_output:
        _print_json(console, metadata)
    else:
        render_model(console, metadata)


@artifact_app.command("inspect", help="Inspect an artifact manifest without loading its model.")
def artifact_inspect(
    path: Path,
    no_verify: Annotated[bool, typer.Option("--no-verify")] = False,
    json_output: JsonOpt = False,
) -> None:
    """Print an artifact's manifest, verifying checksums by default."""
    console = Console()
    payload = inspect_artifact(path, verify=not no_verify).to_dict()
    if json_output:
        _print_json(console, payload)
    else:
        render_artifact(console, payload, str(Path(path).resolve()))
        if not no_verify:
            console.print("[green]✓[/green] Checksums and artifact schema verified.")


@artifact_app.command("verify", help="Verify artifact schema and checksums.")
def artifact_verify(path: Path) -> None:
    """Verify an artifact's schema and checksums."""
    inspect_artifact(path)
    Console().print(f"[green]✓[/green] Artifact verified: {Path(path).resolve()}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and map domain failures to the documented exit codes."""
    error_console = Console(stderr=True)
    try:
        # Click exits with 0 on success and 2 on usage errors, which is EXIT_CONFIG.
        typer.main.get_command(app).main(args=None if argv is None else list(argv), prog_name="saber")
    except SystemExit as exc:
        if exc.code is None:
            return EXIT_OK
        return exc.code if isinstance(exc.code, int) else EXIT_INTERNAL
    except ConfigurationError as exc:
        error_console.print(f"[bold red]Configuration error[/bold red]\n{exc}")
        return EXIT_CONFIG
    except SaberError as exc:
        error_console.print(f"[bold red]saber workflow failed[/bold red]\n{exc}")
        return EXIT_WORKFLOW
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        error_console.print(f"[bold red]Error[/bold red]\n{exc}")
        return EXIT_WORKFLOW
    except Exception as exc:  # noqa: BLE001 - last-resort CLI boundary
        error_console.print(f"[bold red]Unexpected error[/bold red]\n{exc}")
        return EXIT_INTERNAL
    return EXIT_OK


def _filtered_specs(task: TaskChoice | None, provider: str | None, tag: str | None) -> list[AlgorithmSpec]:
    return [
        spec
        for spec in ALGORITHMS.values()
        if (task is None or spec.task == task.value)
        and (provider is None or spec.provider == provider)
        and (tag is None or tag in spec.tags)
    ]


def _print_model_rows(console: Console, specs: list[AlgorithmSpec], *, json_output: bool) -> None:
    rows = [_model_row(spec) for spec in sorted(specs, key=lambda item: item.name)]
    if json_output:
        _print_json(console, rows)
    else:
        render_model_list(console, rows)
        if not rows:
            console.print("[yellow]No models matched the requested filters.[/yellow]")


def _model_row(spec: Any) -> dict[str, Any]:
    return {
        "name": spec.name,
        "task": spec.task,
        "provider": spec.provider,
        "tags": list(spec.tags),
        "capabilities": spec.capabilities.to_dict(),
        "requirements": spec.requirements.to_dict(),
    }


def _print_json(console: Console, payload: Any) -> None:
    console.print_json(json.dumps(to_jsonable(payload), ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
