"""Command-line interface for saber's public API/config layer."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.table import Table

from saber import __version__
from saber.cli.render import (
    render_cli_help,
    render_cli_usage_error,
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

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_WORKFLOW = 3
EXIT_INTERNAL = 4

_WORKFLOW_COMMANDS = {"run", "train", "evaluate", "validate", "tune", "optimize", "benchmark", "predict"}


class RichArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that keeps argparse semantics but renders Rich help."""

    def print_help(self, file: Any | None = None) -> None:
        console = Console(file=file or sys.stdout)
        render_cli_help(console, self)

    def error(self, message: str) -> None:
        console = Console(stderr=True)
        render_cli_usage_error(console, self, message)
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = RichArgumentParser(
        prog="saber",
        description="Classical supervised ML workflows with reproducible partitions, tuning, benchmarking, and artifacts.",
    )
    parser._saber_examples = (
        "saber validate experiment.yaml",
        "saber benchmark study.yaml --dry-run",
        "saber models search forest --task classification",
        "saber artifact verify artifacts/model",
        "saber doctor",
    )
    parser.add_argument("--version", action="version", version=f"saber {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run the workflow declared by a validated YAML/JSON config.", description="Run the workflow declared by a validated YAML/JSON config.")
    _add_workflow_arguments(run_parser)
    run_parser._saber_examples = ("saber run experiment.yaml", "saber run experiment.yaml --dry-run")

    help_text = {
        "train": "Fit a final model and optionally persist an artifact.",
        "evaluate": "Evaluate a persisted model artifact on labeled data.",
        "validate": "Run holdout/CV validation using explicit or BioSieve partitions.",
        "tune": "Optimize hyperparameters using the configured partition plan.",
        "optimize": "Alias for tune.",
        "benchmark": "Run a reproducible benchmark matrix.",
        "predict": "Run inference from a persisted model artifact.",
    }
    for workflow, description in help_text.items():
        item = sub.add_parser(workflow, help=description, description=description)
        _add_workflow_arguments(item)
        item._saber_examples = (
            f"saber {workflow} experiment.yaml",
            f"saber {workflow} experiment.yaml --dry-run",
            f"saber {workflow} experiment.yaml --json",
        )

    models = sub.add_parser("models", help="Discover registered algorithms and their capabilities.", description="Discover registered algorithms and their capabilities.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    list_parser = models_sub.add_parser("list", help="List available algorithms.")
    _add_model_filters(list_parser)
    list_parser.add_argument("--json", action="store_true")
    show_parser = models_sub.add_parser("show", help="Show detailed metadata for one algorithm or alias.")
    show_parser.add_argument("name")
    show_parser.add_argument("--json", action="store_true")
    search_parser = models_sub.add_parser("search", help="Search names, aliases, tags, and descriptions.")
    search_parser.add_argument("query")
    _add_model_filters(search_parser)
    search_parser.add_argument("--json", action="store_true")
    models._saber_examples = (
        "saber models list --task classification",
        "saber models search forest --provider sklearn",
        "saber models show random_forest",
    )

    artifact = sub.add_parser("artifact", help="Inspect or verify persistence artifacts.", description="Inspect or verify persistence artifacts.")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    inspect_parser = artifact_sub.add_parser("inspect", help="Inspect an artifact manifest without loading its model.")
    inspect_parser.add_argument("path")
    inspect_parser.add_argument("--no-verify", action="store_true")
    inspect_parser.add_argument("--json", action="store_true")
    verify_parser = artifact_sub.add_parser("verify", help="Verify artifact schema and checksums.")
    verify_parser.add_argument("path")
    artifact._saber_examples = (
        "saber artifact inspect artifacts/model",
        "saber artifact verify artifacts/model",
    )

    config = sub.add_parser("config", help="Inspect, validate, or normalize workflow config files.", description="Inspect, validate, or normalize workflow config files.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate_parser = config_sub.add_parser("validate", help="Validate a YAML/JSON config without executing it.")
    validate_parser.add_argument("path")
    show_config_parser = config_sub.add_parser("show", help="Render a validated execution plan.")
    show_config_parser.add_argument("path")
    show_config_parser.add_argument("--json", action="store_true")
    normalize_parser = config_sub.add_parser("normalize", help="Write a normalized versioned config.")
    normalize_parser.add_argument("path")
    normalize_parser.add_argument("-o", "--output", required=True)
    config._saber_examples = (
        "saber config validate experiment.yaml",
        "saber config show experiment.yaml",
        "saber config normalize experiment.yaml -o normalized.yaml",
    )

    doctor = sub.add_parser("doctor", help="Show runtime and optional-provider availability.", description="Show runtime and optional-provider availability.")
    doctor.add_argument("--json", action="store_true")
    doctor._saber_examples = ("saber doctor", "saber doctor --json")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    console = Console()
    error_console = Console(stderr=True)
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        if args.command in _WORKFLOW_COMMANDS:
            return _workflow_command(args, console)
        if args.command == "models":
            return _models_command(args, console)
        if args.command == "artifact":
            return _artifact_command(args, console)
        if args.command == "config":
            return _config_command(args, console)
        if args.command == "doctor":
            return _doctor_command(args, console)
        parser.error("Unknown command.")
    except ConfigurationError as exc:
        error_console.print(f"[bold red]Configuration error[/bold red]\n{exc}")
        return EXIT_CONFIG
    except SaberError as exc:
        error_console.print(f"[bold red]saber workflow failed[/bold red]\n{exc}")
        return EXIT_WORKFLOW
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        error_console.print(f"[bold red]Error[/bold red]\n{exc}")
        return EXIT_WORKFLOW
    except Exception as exc:  # pragma: no cover - last-resort CLI boundary
        error_console.print(f"[bold red]Unexpected error[/bold red]\n{exc}")
        return EXIT_INTERNAL
    return EXIT_INTERNAL


def _workflow_command(args: Any, console: Console) -> int:
    config = load_config(args.config)
    expected_workflow = "tune" if args.command == "optimize" else args.command
    if args.command != "run" and config.workflow != expected_workflow:
        raise ConfigurationError(
            f"Command '{args.command}' requires workflow='{expected_workflow}', "
            f"but config declares '{config.workflow}'."
        )

    if args.dry_run:
        if args.json:
            _print_json(console, {"status": "valid", "config": config.to_dict()})
        elif not args.quiet:
            render_dry_run(console, config)
        return EXIT_OK

    if not args.json and not args.quiet:
        render_preflight(console, config)

    if args.json or args.quiet or args.no_progress:
        execution = run_config(config)
    else:
        with console.status(f"[bold cyan]Running {config.workflow} workflow…[/bold cyan]", spinner="dots"):
            execution = run_config(config)

    if args.json:
        _print_json(console, {"summary": execution.summary, "outputs": execution.outputs})
    elif not args.quiet:
        render_execution(console, execution)
    return EXIT_OK


def _models_command(args: Any, console: Console) -> int:
    if args.models_command == "show":
        metadata = MODEL_REGISTRY.describe(args.name)
        if args.json:
            _print_json(console, metadata)
        else:
            render_model(console, metadata)
        return EXIT_OK

    specs = MODEL_REGISTRY.filter(task=args.task, provider=args.provider)
    if args.tag:
        specs = [spec for spec in specs if args.tag in spec.tags]
    if args.models_command == "search":
        needle = args.query.strip().lower()
        specs = [
            spec
            for spec in specs
            if needle in " ".join(
                [spec.name, *spec.aliases, *spec.tags, spec.description or ""]
            ).lower()
        ]

    rows = [_model_row(spec) for spec in sorted(specs, key=lambda item: item.name)]
    if args.json:
        _print_json(console, rows)
    else:
        render_model_list(console, rows)
        if not rows:
            console.print("[yellow]No models matched the requested filters.[/yellow]")
    return EXIT_OK


def _artifact_command(args: Any, console: Console) -> int:
    if args.artifact_command == "verify":
        verify_artifact(args.path)
        console.print(f"[green]✓[/green] Artifact verified: {Path(args.path).resolve()}")
        return EXIT_OK

    manifest = inspect_artifact(args.path, verify=not args.no_verify)
    payload = manifest.to_dict()
    if args.json:
        _print_json(console, payload)
    else:
        render_artifact(console, payload, str(Path(args.path).resolve()))
        if not args.no_verify:
            console.print("[green]✓[/green] Checksums and artifact schema verified.")
    return EXIT_OK


def _config_command(args: Any, console: Console) -> int:
    config = load_config(args.path)
    if args.config_command == "validate":
        console.print(
            f"[green]✓[/green] Config valid  [dim]workflow={config.workflow}  schema={config.schema_version}[/dim]"
        )
        return EXIT_OK
    if args.config_command == "show":
        if args.json:
            _print_json(console, config.to_dict())
        else:
            render_preflight(console, config)
        return EXIT_OK
    output = dump_config(config, args.output)
    console.print(f"[green]✓[/green] Normalized config written to {output.resolve()}")
    return EXIT_OK


def _doctor_command(args: Any, console: Console) -> int:
    payload = _doctor_payload()
    if args.json:
        _print_json(console, payload)
    else:
        render_doctor(console, payload)
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


def _add_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("config", help="YAML or JSON workflow configuration.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show the execution plan without running it.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of rich output.")
    parser.add_argument("--quiet", action="store_true", help="Suppress normal CLI output; errors still go to stderr.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the interactive progress/status indicator.")


def _add_model_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=("classification", "regression"))
    parser.add_argument("--provider")
    parser.add_argument("--tag")


def _print_json(console: Console, payload: Any) -> None:
    console.print_json(json.dumps(to_jsonable(payload), ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
