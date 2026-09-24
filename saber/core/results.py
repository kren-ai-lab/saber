"""Core result contracts shared by the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from saber.core.specs import AlgorithmSpec


@dataclass(slots=True)
class TrainResult:
    """Result of fitting one final supervised model."""

    model: Any
    spec: AlgorithmSpec
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
