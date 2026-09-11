"""Holdout validation convenience wrapper."""

from __future__ import annotations

from typing import Any

from mlcore.core.registry import AlgorithmRegistry
from mlcore.datasets import PartitionPlan
from mlcore.exceptions import ValidationContractError
from mlcore.validation.cross_validation import ValidationEngine
from mlcore.validation.results import ValidationResult


def validate_holdout(
    registry: AlgorithmRegistry,
    *,
    partition_plan: PartitionPlan,
    **kwargs: Any,
) -> ValidationResult:
    """Validate a single explicit holdout partition."""

    if partition_plan.n_splits != 1:
        raise ValidationContractError(
            "validate_holdout requires a PartitionPlan with exactly one split."
        )
    return ValidationEngine(registry).run(
        partition_plan=partition_plan,
        **kwargs,
    )
