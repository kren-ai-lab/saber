"""Shared partition resolution utilities for validation and tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from saber.datasets import DatasetBundle, PartitionPlan
from saber.datasets.biosieve import BioSievePartitionConfig, partition_with_biosieve
from saber.exceptions import ValidationContractError

EvaluationRole = Literal["auto", "validation", "test"]


def resolve_partition_plan(
    *,
    dataset: DatasetBundle,
    partition_plan: PartitionPlan | None,
    partitioning: BioSievePartitionConfig | None,
    biosieve_extra_columns: Mapping[str, Sequence[Any]] | None,
) -> PartitionPlan:
    """Resolve supplied partitions or delegate generation exclusively to BioSieve."""
    if partition_plan is not None and partitioning is not None:
        raise ValidationContractError(
            "Provide either partition_plan (already partitioned data) or "
            "partitioning (BioSieve generation), not both."
        )
    if partition_plan is not None:
        return partition_plan
    if partitioning is None:
        raise ValidationContractError(
            "Unpartitioned data require an explicit BioSievePartitionConfig. "
            "saber does not generate its own fallback splits."
        )
    return partition_with_biosieve(
        dataset,
        partitioning,
        extra_columns=biosieve_extra_columns,
    )


def resolve_evaluation_dataset(
    resolved: Any,
    *,
    requested: EvaluationRole,
    plan_kind: str,
):
    """Choose the held-out role while protecting a final test set by default."""
    if requested == "test":
        if resolved.test is None:
            raise ValidationContractError(f"Split '{resolved.name}' has no test membership.")
        return "test", resolved.test

    if requested == "validation":
        if resolved.validation is None:
            raise ValidationContractError(f"Split '{resolved.name}' has no validation membership.")
        return "validation", resolved.validation

    if requested != "auto":
        raise ValidationContractError("evaluation_role must be 'auto', 'validation', or 'test'.")

    # BioSieve k-fold protocols expose the held-out fold as ``test``. For a
    # single holdout, however, an explicit validation set should be preferred
    # so the final test set remains protected during development/tuning.
    if plan_kind == "cross_validation":
        if resolved.test is not None:
            return "test", resolved.test
        if resolved.validation is not None:
            return "validation", resolved.validation
    else:
        if resolved.validation is not None:
            return "validation", resolved.validation
        if resolved.test is not None:
            return "test", resolved.test
    raise ValidationContractError(f"Split '{resolved.name}' does not define a held-out evaluation role.")


def build_explicit_cv(
    *,
    dataset: DatasetBundle,
    plan: PartitionPlan,
    evaluation_role: EvaluationRole = "auto",
    require_complete: bool = True,
) -> tuple[DatasetBundle, tuple[tuple[np.ndarray, np.ndarray], ...], tuple[str, ...]]:
    """Build explicit sklearn-compatible CV indices from a PartitionPlan.

    Only samples participating in training or the chosen evaluation role are
    included in the returned search dataset. This keeps a separate final test
    set outside hyperparameter search/refit when a holdout provides validation
    and test memberships.
    """
    plan.validate_against(dataset, require_complete=require_complete)

    memberships: list[tuple[tuple[Any, ...], tuple[Any, ...], str]] = []
    used_ids: set[Any] = set()

    for split in plan.splits:
        resolved = plan.resolve(
            dataset,
            split.name,
            require_complete=require_complete,
        )
        role, evaluation_data = resolve_evaluation_dataset(
            resolved,
            requested=evaluation_role,
            plan_kind=plan.kind,
        )
        train_ids = tuple(resolved.train.sample_ids)
        evaluation_ids = tuple(evaluation_data.sample_ids)
        memberships.append((train_ids, evaluation_ids, role))
        used_ids.update(train_ids)
        used_ids.update(evaluation_ids)

    ordered_ids = tuple(sample_id for sample_id in dataset.sample_ids if sample_id in used_ids)
    search_dataset = dataset.subset(ordered_ids)
    id_to_index = {sample_id: index for index, sample_id in enumerate(search_dataset.sample_ids)}

    cv: list[tuple[np.ndarray, np.ndarray]] = []
    roles: list[str] = []
    for train_ids, evaluation_ids, role in memberships:
        train_index = np.asarray([id_to_index[sample_id] for sample_id in train_ids], dtype=int)
        evaluation_index = np.asarray(
            [id_to_index[sample_id] for sample_id in evaluation_ids],
            dtype=int,
        )
        cv.append((train_index, evaluation_index))
        roles.append(role)

    return search_dataset, tuple(cv), tuple(roles)
