"""Benchmark execution configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from saber.tuning import TuningConfig

BenchmarkMode = Literal["untuned", "tuned"]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Benchmark-specific controls of the algorithm x representation x partition matrix.

    Workflow knobs shared with ``validate``/``tune`` (``metrics``,
    ``evaluation_role``, ...) are keyword arguments of :func:`benchmark`.
    Each seed in ``seeds`` is the ``random_state`` of one set of runs.
    """

    seeds: tuple[int | None, ...] = (42,)
    modes: tuple[BenchmarkMode, ...] = ("untuned",)
    include_baselines: bool = True
    fail_fast: bool = False
    tuning: TuningConfig | None = None
    metadata: dict[str, object] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        """Deduplicate seeds/modes and validate the tuned-mode contract."""
        seeds = tuple(dict.fromkeys(self.seeds))
        modes = tuple(dict.fromkeys(self.modes))

        if not seeds:
            raise ValueError("BenchmarkConfig.seeds must contain at least one seed.")
        if not modes:
            raise ValueError("BenchmarkConfig.modes must contain at least one mode.")
        invalid = set(modes) - {"untuned", "tuned"}
        if invalid:
            raise ValueError(f"Unsupported benchmark modes: {sorted(invalid)!r}.")
        if "tuned" in modes and self.tuning is None:
            raise ValueError("Tuned benchmarks require an explicit TuningConfig.")

        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "modes", modes)
        object.__setattr__(self, "metadata", dict(self.metadata))
