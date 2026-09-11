"""Predefined/external fold validation convenience wrappers."""

from __future__ import annotations

from typing import Any

from mlcore.core.registry import AlgorithmRegistry
from mlcore.datasets import PartitionPlan
from mlcore.validation.cross_validation import ValidationEngine
from mlcore.validation.results import ValidationResult


def validate_predefined_folds(
    registry: AlgorithmRegistry,
    *,
    partition_plan: PartitionPlan,
    **kwargs: Any,
) -> ValidationResult:
    """Validate over an already supplied fold plan without regenerating splits."""

    return ValidationEngine(registry).run(
        partition_plan=partition_plan,
        **kwargs,
    )
