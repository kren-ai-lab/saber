"""Structured evaluation result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saber.core.prediction import PredictionResult
    from saber.core.task import TaskType


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation metrics paired with the prediction contract that produced them."""

    task: TaskType
    metrics: dict[str, float]
    prediction: PredictionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly summary."""
        return {
            "task": self.task,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }
