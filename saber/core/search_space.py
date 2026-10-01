"""Typed, backend-agnostic hyperparameter search-space contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import loguniform, uniform


@dataclass(frozen=True, slots=True)
class Categorical:
    """Finite categorical parameter domain."""

    values: tuple[Any, ...]

    def __init__(self, values: Any) -> None:
        """Validate and store the finite set of categorical values."""
        values = tuple(values)
        if not values:
            raise ValueError("Categorical values cannot be empty.")
        object.__setattr__(self, "values", values)

    def grid_values(self) -> list[Any]:
        """Return the categorical values as a list."""
        return list(self.values)

    def random_distribution(self) -> list[Any]:
        """Return the categorical values as a list."""
        return list(self.values)

    def suggest(self, trial: Any, name: str) -> Any:
        """Sample one categorical value using an Optuna trial."""
        return trial.suggest_categorical(name, list(self.values))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of this domain."""
        return {"type": "categorical", "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class Integer:
    """Inclusive integer parameter domain."""

    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        """Validate that the integer bounds and step are well-formed."""
        if self.low > self.high:
            raise ValueError("Integer.low cannot exceed Integer.high.")
        if self.step <= 0:
            raise ValueError("Integer.step must be positive.")

    def grid_values(self) -> list[int]:
        """Return the inclusive, stepped range of integer values."""
        return list(range(self.low, self.high + 1, self.step))

    def random_distribution(self) -> list[int]:
        """Return the inclusive, stepped range of integer values."""
        # A finite stepped list preserves exactly the same domain across
        # RandomizedSearchCV, halving-random, and Optuna.
        return self.grid_values()

    def suggest(self, trial: Any, name: str) -> int:
        """Sample one integer value using an Optuna trial."""
        return int(trial.suggest_int(name, self.low, self.high, step=self.step))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of this domain."""
        return {"type": "integer", **asdict(self)}


@dataclass(frozen=True, slots=True)
class Float:
    """Continuous or stepped floating-point parameter domain."""

    low: float
    high: float
    step: float | None = None

    def __post_init__(self) -> None:
        """Validate that the float bounds and optional step are well-formed."""
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("Float bounds must be finite.")
        if self.low >= self.high:
            raise ValueError("Float.low must be smaller than Float.high.")
        if self.step is not None and self.step <= 0:
            raise ValueError("Float.step must be positive when supplied.")
        if self.step is not None:
            n_steps = (self.high - self.low) / self.step
            # Grid and Optuna must see the same lattice; Optuna truncates a
            # non-divisible range while a grid would add the off-step bound.
            if not np.isclose(n_steps, round(n_steps)):
                raise ValueError(
                    f"Float range [{self.low}, {self.high}] must be divisible by step={self.step}."
                )

    def grid_values(self) -> list[float]:
        """Return stepped float values, when a step is defined."""
        if self.step is None:
            raise ValueError(
                "Continuous Float parameters are not directly enumerable for grid search. "
                "Provide step=... or use random/Optuna optimization."
            )
        count = round((self.high - self.low) / self.step)
        return [float(value) for value in np.linspace(self.low, self.high, count + 1)]

    def random_distribution(self) -> Any:
        """Return stepped values when defined, otherwise a continuous uniform distribution."""
        if self.step is not None:
            return self.grid_values()
        return uniform(loc=self.low, scale=self.high - self.low)

    def suggest(self, trial: Any, name: str) -> float:
        """Sample one float value using an Optuna trial."""
        return float(
            trial.suggest_float(
                name,
                self.low,
                self.high,
                step=self.step,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of this domain."""
        payload: dict[str, Any] = {
            "type": "float",
            "low": self.low,
            "high": self.high,
        }
        if self.step is not None:
            payload["step"] = self.step
        return payload


@dataclass(frozen=True, slots=True)
class LogFloat:
    """Continuous log-uniform floating-point parameter domain."""

    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate that the log-uniform bounds are strictly positive and ordered."""
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogFloat bounds must be strictly positive.")
        if self.low >= self.high:
            raise ValueError("LogFloat.low must be smaller than LogFloat.high.")

    def grid_values(self) -> list[float]:
        """Raise, since a continuous log-uniform domain is not enumerable for grid search."""
        raise ValueError(
            "Continuous LogFloat parameters are not directly enumerable for grid search. "
            "Use categorical values for an explicit logarithmic grid, or use random/Optuna."
        )

    def random_distribution(self) -> Any:
        """Return a log-uniform distribution over the configured bounds."""
        return loguniform(self.low, self.high)

    def suggest(self, trial: Any, name: str) -> float:
        """Sample one log-uniform float value using an Optuna trial."""
        return float(trial.suggest_float(name, self.low, self.high, log=True))

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of this domain."""
        return {"type": "log_float", **asdict(self)}


ParameterDomain = Categorical | Integer | Float | LogFloat | list[Any] | tuple[Any, ...]


def _coerce_domain(value: ParameterDomain) -> Categorical | Integer | Float | LogFloat:
    if isinstance(value, (Categorical, Integer, Float, LogFloat)):
        return value
    if isinstance(value, (list, tuple)):
        return Categorical(value)
    raise TypeError(
        "Search-space parameters must be Categorical, Integer, Float, LogFloat, or a finite list/tuple."
    )


@dataclass(slots=True)
class SearchSpace:
    """Backend-agnostic hyperparameter search-space definition.

    Plain lists and tuples are interpreted as finite categorical domains.
    """

    parameters: dict[str, ParameterDomain]

    def __post_init__(self) -> None:
        """Validate and coerce every parameter domain."""
        self.parameters = dict(self.parameters)
        for name, domain in self.parameters.items():
            if not str(name).strip():
                raise ValueError("Search-space parameter names cannot be empty.")
            _coerce_domain(domain)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of this search space."""
        serialized: dict[str, Any] = {}
        for name, domain in self.parameters.items():
            if isinstance(domain, (list, tuple)):
                serialized[name] = list(domain)
            else:
                serialized[name] = _coerce_domain(domain).to_dict()
        return {"parameters": serialized}

    def to_grid(self, *, prefix: str = "") -> dict[str, list[Any]]:
        """Translate the logical search space to sklearn grid domains."""
        return {
            f"{prefix}{name}": _coerce_domain(domain).grid_values()
            for name, domain in self.parameters.items()
        }

    def to_random(self, *, prefix: str = "") -> dict[str, Any]:
        """Translate the logical search space to sklearn random domains."""
        return {
            f"{prefix}{name}": _coerce_domain(domain).random_distribution()
            for name, domain in self.parameters.items()
        }

    def sample_optuna(self, trial: Any, *, prefix: str = "") -> dict[str, Any]:
        """Sample one parameter configuration using an Optuna trial."""
        return {
            name: _coerce_domain(domain).suggest(trial, f"{prefix}{name}")
            for name, domain in self.parameters.items()
        }

    def __len__(self) -> int:
        """Return the number of parameters in this search space."""
        return len(self.parameters)
