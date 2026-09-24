"""YAML/JSON configuration loading and normalized serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from saber.config.schema import WorkflowConfig, workflow_config_from_mapping
from saber.exceptions import ConfigurationError


def load_config(source: str | Path | Mapping[str, Any] | WorkflowConfig) -> WorkflowConfig:
    """Load and validate a YAML/JSON workflow configuration."""

    if isinstance(source, WorkflowConfig):
        return source
    if isinstance(source, Mapping):
        return workflow_config_from_mapping(source)

    path = Path(source).resolve()
    if not path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {path}.")

    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
        if suffix == ".json":
            payload = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(text)
        else:
            raise ConfigurationError("Configuration files must be YAML or JSON.")
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Could not parse configuration '{path}': {exc}") from exc

    if not isinstance(payload, Mapping):
        raise ConfigurationError("Configuration root must be a mapping/object.")
    return workflow_config_from_mapping(payload, source=path)


def dump_config(
    config: WorkflowConfig | Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Serialize a normalized config for reproducible reruns."""

    document = load_config(config)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = document.to_dict()

    suffix = target.suffix.lower()
    if suffix == ".json":
        target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    elif suffix in {".yaml", ".yml"}:
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        raise ConfigurationError("Normalized configs must be written as YAML or JSON.")
    return target
