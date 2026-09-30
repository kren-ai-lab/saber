"""Core result contracts shared by the public API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saber.core.specs import AlgorithmSpec
    from saber.datasets.schemas import FeatureSchema


@dataclass(slots=True)
class TrainResult:
    """Result of fitting one final supervised model."""

    model: Any
    spec: AlgorithmSpec
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_schema: FeatureSchema | None = None
    positive_class: Any | None = None
