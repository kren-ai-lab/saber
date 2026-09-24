"""Command-line interface for saber's public API/config layer."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console

from saber import __version__
from saber.cli.render import (
    render_artifact,
    render_doctor,
    render_dry_run,
    render_execution,
    render_model,
    render_model_list,
    render_preflight,
)
from saber.config import dump_config, load_config, run_config
from saber.core.registry import MODEL_REGISTRY
from saber.exceptions import ConfigurationError, SaberError
from saber.persistence import inspect_artifact, verify_artifact
from saber.utils.serialization import to_jsonable

if TYPE_CHECKING:
    from collections.abc import Sequence

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
config_app = typer.Typer(no_args_is_help=True, help="Inspect, validate, or normalize workflow config files.")
app.add_typer(models_app, name="models")
app.add_typer(artifact_app, name="artifact")
app.add_typer(config_app, name="config")

WORKFLOW_HELP = {
    "run": "Run the workflow declared by a validated YAML/JSON config.",
    "train": "Fit a final model and optionally persist an artifact.",
    "evaluate": "Evaluate a persisted model artifact on labeled data.",
    "validate": "Run holdout/CV validation using explicit or BioSieve partitions.",
    "tune": "Optimize hyperparameters using the configured partition plan.",
    "optimize": "Alias for tune.",
    "benchmark": "Run a reproducible benchmark matrix.",
    "predict": "Run inference from a persisted model artifact.",
}


class TaskChoice(StrEnum):
    """Restricts the ``--task`` option to classification or regression."""

    classification = "classification"
    regression = "regression"


ConfigArg = Annotated[Path, typer.Argument(help="YAML or JSON workflow configuration.")]
DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Validate and show the execution plan without running it.")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON instead of rich output.")]
QuietOpt = Annotated[
    bool, typer.Option("--quiet", help="Suppress normal CLI output; errors still go to stderr.")
]
NoProgressOpt = Annotated[
    bool, typer.Option("--no-progress", help="Disable the interactive progress/status indicator.")
]
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


def _register_workflow(name: str, help_text: str) -> None:
    def command(
        config: ConfigArg,
        dry_run: DryRunOpt = False,
        json_output: JsonOpt = False,
        quiet: QuietOpt = False,
        no_progress: NoProgressOpt = False,
    ) -> None:
        _workflow_command(
            name,
            config,
            dry_run=dry_run,
            json_output=json_output,
            quiet=quiet,
            no_progress=no_progress,
            console=Console(),
        )

    epilog = (
        f"Examples: saber {name} experiment.yaml · saber {name} experiment.yaml --dry-run · "
        f"saber {name} experiment.yaml --json"
    )
    app.command(name, help=help_text, epilog=epilog)(command)


for _name, _help in WORKFLOW_HELP.items():
    _register_workflow(_name, _help)


@models_app.command("list", help="List available algorithms.")
def models_list(
    task: TaskOpt = None, provider: ProviderOpt = None, tag: TagOpt = None, json_output: JsonOpt = False
) -> None:
    """List registered models, optionally filtered by task, provider, or tag."""
    _models_command("list", console=Console(), task=task, provider=provider, tag=tag, json_output=json_output)


@models_app.command("show", help="Show detailed metadata for one algorithm or alias.")
def models_show(name: str, json_output: JsonOpt = False) -> None:
    """Show metadata for one registered model, by name or alias."""
    _models_command("show", console=Console(), name=name, json_output=json_output)


@models_app.command("search", help="Search names, aliases, tags, and descriptions.")
def models_search(
    query: str,
    task: TaskOpt = None,
    provider: ProviderOpt = None,
    tag: TagOpt = None,
    json_output: JsonOpt = False,
) -> None:
    """Search registered models by name, alias, tag, or description."""
    _models_command(
        "search",
        console=Console(),
        query=query,
        task=task,
        provider=provider,
        tag=tag,
        json_output=json_output,
    )


@artifact_app.command("inspect", help="Inspect an artifact manifest without loading its model.")
def artifact_inspect(
    path: Path,
    no_verify: Annotated[bool, typer.Option("--no-verify")] = False,
    json_output: JsonOpt = False,
) -> None:
    """Print an artifact's manifest, verifying checksums by default."""
    _artifact_command("inspect", console=Console(), path=path, no_verify=no_verify, json_output=json_output)


@artifact_app.command("verify", help="Verify artifact schema and checksums.")
def artifact_verify(path: Path) -> None:
    """Verify an artifact's schema and checksums."""
    _artifact_command("verify", console=Console(), path=path)


@config_app.command("validate", help="Validate a YAML/JSON config without executing it.")
def config_validate(path: Path) -> None:
    """Validate a workflow config without executing it."""
    _config_command("validate", console=Console(), path=path)


@config_app.command("show", help="Render a validated execution plan.")
def config_show(path: Path, json_output: JsonOpt = False) -> None:
    """Render the execution plan a config would run."""
    _config_command("show", console=Console(), path=path, json_output=json_output)


@config_app.command("normalize", help="Write a normalized versioned config.")
def config_normalize(path: Path, output: Annotated[Path, typer.Option("-o", "--output")]) -> None:
    """Write a normalized, versioned copy of a config file."""
    _config_command("normalize", console=Console(), path=path, output=output)


@app.command(
    "doctor",
    help="Show runtime and optional-provider availability.",
    epilog="Examples: saber doctor · saber doctor --json",
)
def doctor(json_output: JsonOpt = False) -> None:
    """Show which optional providers are installed and available."""
    payload = _doctor_payload()
    console = Console()
    if json_output:
        _print_json(console, payload)
    else:
        render_doctor(console, payload)


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


def _workflow_command(
    name: str,
    config_path: Path,
    *,
    dry_run: bool,
    json_output: bool,
    quiet: bool,
    no_progress: bool,
    console: Console,
) -> int:
    config = load_config(config_path)
    expected_workflow = "tune" if name == "optimize" else name
    if name != "run" and config.workflow != expected_workflow:
        raise ConfigurationError(
            f"Command '{name}' requires workflow='{expected_workflow}', "
            f"but config declares '{config.workflow}'."
        )

    if dry_run:
        if json_output:
            _print_json(console, {"status": "valid", "config": config.to_dict()})
        elif not quiet:
            render_dry_run(console, config)
        return EXIT_OK

    if not json_output and not quiet:
        render_preflight(console, config)

    if json_output or quiet or no_progress:
        execution = run_config(config)
    else:
        with console.status(f"[bold cyan]Running {config.workflow} workflow…[/bold cyan]", spinner="dots"):
            execution = run_config(config)

    if json_output:
        _print_json(console, {"summary": execution.summary, "outputs": execution.outputs})
    elif not quiet:
        render_execution(console, execution)
    return EXIT_OK


def _models_command(
    subcommand: str,
    *,
    console: Console,
    json_output: bool,
    name: str | None = None,
    query: str | None = None,
    task: TaskChoice | None = None,
    provider: str | None = None,
    tag: str | None = None,
) -> int:
    if subcommand == "show":
        metadata = MODEL_REGISTRY.describe(name)
        if json_output:
            _print_json(console, metadata)
        else:
            render_model(console, metadata)
        return EXIT_OK

    specs = MODEL_REGISTRY.filter(task=task.value if task else None, provider=provider)
    if tag:
        specs = [spec for spec in specs if tag in spec.tags]
    if subcommand == "search":
        needle = query.strip().lower()
        specs = [
            spec
            for spec in specs
            if needle in " ".join([spec.name, *spec.aliases, *spec.tags, spec.description or ""]).lower()
        ]

    rows = [_model_row(spec) for spec in sorted(specs, key=lambda item: item.name)]
    if json_output:
        _print_json(console, rows)
    else:
        render_model_list(console, rows)
        if not rows:
            console.print("[yellow]No models matched the requested filters.[/yellow]")
    return EXIT_OK


def _artifact_command(
    subcommand: str,
    *,
    console: Console,
    path: Path,
    no_verify: bool = False,
    json_output: bool = False,
) -> int:
    if subcommand == "verify":
        verify_artifact(path)
        console.print(f"[green]✓[/green] Artifact verified: {Path(path).resolve()}")
        return EXIT_OK

    manifest = inspect_artifact(path, verify=not no_verify)
    payload = manifest.to_dict()
    if json_output:
        _print_json(console, payload)
    else:
        render_artifact(console, payload, str(Path(path).resolve()))
        if not no_verify:
            console.print("[green]✓[/green] Checksums and artifact schema verified.")
    return EXIT_OK


def _config_command(
    subcommand: str,
    *,
    console: Console,
    path: Path,
    json_output: bool = False,
    output: Path | None = None,
) -> int:
    config = load_config(path)
    if subcommand == "validate":
        console.print(
            f"[green]✓[/green] Config valid  "
            f"[dim]workflow={config.workflow}  schema={config.schema_version}[/dim]"
        )
        return EXIT_OK
    if subcommand == "show":
        if json_output:
            _print_json(console, config.to_dict())
        else:
            render_preflight(console, config)
        return EXIT_OK
    written = dump_config(config, output)
    console.print(f"[green]✓[/green] Normalized config written to {written.resolve()}")
    return EXIT_OK


def _doctor_payload() -> dict[str, Any]:
    components = [
        {"name": "saber", "available": True, "version": __version__},
        {"name": "python", "available": True, "version": platform.python_version()},
    ]
    for package, import_name in (
        ("scikit-learn", "sklearn"),
        ("biosieve", "biosieve"),
        ("xgboost", "xgboost"),
        ("lightgbm", "lightgbm"),
        ("optuna", "optuna"),
    ):
        available = importlib.util.find_spec(import_name) is not None
        version = None
        if available:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "available from source"
        components.append({"name": package, "available": available, "version": version})
    return {"components": components}


def _model_row(spec: Any) -> dict[str, Any]:
    return {
        "name": spec.name,
        "task": spec.task,
        "provider": spec.provider,
        "aliases": list(spec.aliases),
        "tags": list(spec.tags),
        "capabilities": spec.capabilities.to_dict(),
        "requirements": spec.requirements.to_dict(),
    }


def _print_json(console: Console, payload: Any) -> None:
    console.print_json(json.dumps(to_jsonable(payload), ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
