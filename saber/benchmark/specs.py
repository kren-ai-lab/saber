"""Benchmark input contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from saber.datasets import DatasetBundle, PartitionPlan


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """One prepared numerical representation participating in a benchmark.

    ``saber`` does not generate representations.  ``label`` and ``metadata``
    only describe a prepared :class:`DatasetBundle` so downstream tables can
    trace scores back to the representation that produced them.
    """

    label: str
    dataset: DatasetBundle
    representation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Validate the label and copy metadata into an owned dict."""
        if not self.label.strip():
            raise ValueError("Benchmark dataset labels cannot be empty.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def representation_label(self) -> str:
        """Return the representation name, falling back to the dataset label."""
        return self.representation or self.label


@dataclass(frozen=True, slots=True)
class BenchmarkPartition:
    """Named partition scenario used by a benchmark matrix."""

    label: str
    plan: PartitionPlan
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Validate the label and copy metadata into an owned dict."""
        if not self.label.strip():
            raise ValueError("Benchmark partition labels cannot be empty.")
        object.__setattr__(self, "metadata", dict(self.metadata))
