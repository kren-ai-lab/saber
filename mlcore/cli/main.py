"""Command-line interface for mlcore's public API/config layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.table import Table

from mlcore import __version__
from mlcore.config import dump_config, load_config, run_config
from mlcore.core.registry import MODEL_REGISTRY
from mlcore.exceptions import ConfigurationError, MLCoreError
from mlcore.persistence import inspect_artifact, verify_artifact
from mlcore.utils.serialization import to_jsonable

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_WORKFLOW = 3
EXIT_INTERNAL = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlcore", description="Classical supervised ML workflows.")
    parser.add_argument("--version", action="version", version=f"mlcore {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a validated YAML/JSON workflow config.")
    run_parser.add_argument("config")

    for workflow in ("train", "evaluate", "validate", "tune", "optimize", "benchmark", "predict"):
        item = sub.add_parser(workflow, help=f"Run a {workflow} workflow config.")
        item.add_argument("config")

    models = sub.add_parser("models", help="Discover registered algorithms.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    list_parser = models_sub.add_parser("list")
    list_parser.add_argument("--task", choices=("classification", "regression"))
    list_parser.add_argument("--provider")
    list_parser.add_argument("--json", action="store_true")
    show_parser = models_sub.add_parser("show")
    show_parser.add_argument("name")
    show_parser.add_argument("--json", action="store_true")

    artifact = sub.add_parser("artifact", help="Inspect or verify persistence artifacts.")
    artifact_sub = artifact.add_subparsers(dest="artifact_command", required=True)
    inspect_parser = artifact_sub.add_parser("inspect")
    inspect_parser.add_argument("path")
    inspect_parser.add_argument("--no-verify", action="store_true")
    inspect_parser.add_argument("--json", action="store_true")
    verify_parser = artifact_sub.add_parser("verify")
    verify_parser.add_argument("path")

    config = sub.add_parser("config", help="Validate or normalize workflow config files.")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    validate_parser = config_sub.add_parser("validate")
    validate_parser.add_argument("path")
    normalize_parser = config_sub.add_parser("normalize")
    normalize_parser.add_argument("path")
    normalize_parser.add_argument("-o", "--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    console = Console()
    error_console = Console(stderr=True)
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        if args.command in {"run", "train", "evaluate", "validate", "tune", "optimize", "benchmark", "predict"}:
            config = load_config(args.config)
            expected_workflow = "tune" if args.command == "optimize" else args.command
            if args.command != "run" and config.workflow != expected_workflow:
                raise ConfigurationError(
                    f"Command '{args.command}' requires workflow='{expected_workflow}', "
                    f"but config declares '{config.workflow}'."
                )
            execution = run_config(config)
            _print_json(console, execution.summary)
            if execution.outputs:
                console.print("Outputs:")
                for name, path in sorted(execution.outputs.items()):
                    console.print(f"  {name}: {path}")
            return EXIT_OK

        if args.command == "models":
            return _models_command(args, console)
        if args.command == "artifact":
            return _artifact_command(args, console)
        if args.command == "config":
            return _config_command(args, console)
        parser.error("Unknown command.")
    except ConfigurationError as exc:
        error_console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        return EXIT_CONFIG
    except MLCoreError as exc:
        error_console.print(f"[bold red]mlcore error:[/bold red] {exc}")
        return EXIT_WORKFLOW
    except (FileNotFoundError, OSError, ValueError, TypeError) as exc:
        error_console.print(f"[bold red]Error:[/bold red] {exc}")
        return EXIT_WORKFLOW
    except Exception as exc:  # pragma: no cover - last-resort CLI boundary
        error_console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        return EXIT_INTERNAL
    return EXIT_INTERNAL


def _models_command(args: Any, console: Console) -> int:
    if args.models_command == "show":
        metadata = MODEL_REGISTRY.describe(args.name)
        if args.json:
            _print_json(console, metadata)
        else:
            table = Table(title=args.name)
            table.add_column("Field")
            table.add_column("Value")
            for key, value in metadata.items():
                table.add_row(str(key), _short(value))
            console.print(table)
        return EXIT_OK

    specs = MODEL_REGISTRY.filter(task=args.task, backend=args.provider)
    rows = [
        {
            "name": spec.name,
            "task": spec.task,
            "provider": spec.provider,
            "aliases": list(spec.aliases),
            "tags": list(spec.tags),
        }
        for spec in sorted(specs, key=lambda item: item.name)
    ]
    if args.json:
        _print_json(console, rows)
    else:
        table = Table(title="mlcore models")
        table.add_column("Name")
        table.add_column("Task")
        table.add_column("Provider")
        table.add_column("Aliases")
        for row in rows:
            table.add_row(row["name"], row["task"], row["provider"], ", ".join(row["aliases"]))
        console.print(table)
    return EXIT_OK


def _artifact_command(args: Any, console: Console) -> int:
    if args.artifact_command == "verify":
        verify_artifact(args.path)
        console.print(f"Artifact OK: {Path(args.path).resolve()}")
        return EXIT_OK

    manifest = inspect_artifact(args.path, verify=not args.no_verify)
    payload = manifest.to_dict()
    if args.json:
        _print_json(console, payload)
    else:
        table = Table(title="Artifact")
        table.add_column("Field")
        table.add_column("Value")
        for key in ("artifact_type", "schema_version", "created_at", "mlcore_version"):
            table.add_row(key, _short(payload.get(key)))
        table.add_row("files", ", ".join(sorted(payload.get("files", {}).keys())))
        console.print(table)
    return EXIT_OK


def _config_command(args: Any, console: Console) -> int:
    config = load_config(args.path)
    if args.config_command == "validate":
        console.print(f"Config OK: workflow={config.workflow}, schema={config.schema_version}")
        return EXIT_OK
    output = dump_config(config, args.output)
    console.print(f"Normalized config: {output.resolve()}")
    return EXIT_OK


def _print_json(console: Console, payload: Any) -> None:
    console.print_json(json.dumps(to_jsonable(payload), ensure_ascii=False))


def _short(value: Any) -> str:
    converted = to_jsonable(value)
    if isinstance(converted, (dict, list)):
        return json.dumps(converted, ensure_ascii=False)
    return str(converted)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
