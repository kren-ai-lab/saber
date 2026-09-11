"""Benchmark execution configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from mlcore.tuning import TuningConfig

BenchmarkMode = Literal["untuned", "tuned"]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Controls the algorithm × representation × partition benchmark matrix."""

    metrics: tuple[str, ...]
    seeds: tuple[int | None, ...] = (42,)
    modes: tuple[BenchmarkMode, ...] = ("untuned",)
    include_baselines: bool = True
    fail_fast: bool = False
    evaluation_role: str = "auto"
    require_complete: bool = True
    return_estimators: bool = False
    tuning: TuningConfig | None = None
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        metrics = tuple(dict.fromkeys(self.metrics))
        seeds = tuple(dict.fromkeys(self.seeds))
        modes = tuple(dict.fromkeys(self.modes))

        if not metrics:
            raise ValueError("BenchmarkConfig.metrics must contain at least one metric.")
        if not seeds:
            raise ValueError("BenchmarkConfig.seeds must contain at least one seed.")
        if not modes:
            raise ValueError("BenchmarkConfig.modes must contain at least one mode.")
        invalid = set(modes) - {"untuned", "tuned"}
        if invalid:
            raise ValueError(f"Unsupported benchmark modes: {sorted(invalid)!r}.")
        if "tuned" in modes and self.tuning is None:
            raise ValueError("Tuned benchmarks require an explicit TuningConfig.")

        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "metadata", dict(self.metadata))
