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

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly summary."""
        return {
            "algorithm": self.spec.name,
            "task": self.spec.task,
            "dataset_fingerprint": self.metadata.get("dataset_fingerprint"),
            "provider": self.spec.provider,
            "n_samples": self.metadata.get("n_samples"),
            "n_features": self.metadata.get("n_features"),
            "parameters": dict(self.parameters),
            "positive_class": self.positive_class,
        }
