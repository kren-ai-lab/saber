"""
mlcore.evaluation.results
=========================

Structured evaluation result objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlcore.core.prediction import PredictionResult
from mlcore.core.task import TaskType


@dataclass(slots=True)
class EvaluationResult:
    """Evaluation metrics paired with the prediction contract that produced them."""

    task: TaskType
    metrics: dict[str, float]
    prediction: PredictionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_metric(self, name: str) -> float:
        """Retrieve one metric by name."""

        return self.metrics[name]

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly summary."""

        return {
            "task": self.task,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }
