"""
saber.core.search_space
========================

Typed, backend-agnostic hyperparameter search-space contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.stats import loguniform, uniform


class SearchParameter(Protocol):
    """Protocol implemented by typed search-space primitives."""

    def grid_values(self) -> list[Any]: ...
    def random_distribution(self) -> Any: ...
    def suggest(self, trial: Any, name: str) -> Any: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class Categorical:
    """Finite categorical parameter domain."""

    values: tuple[Any, ...]

    def __init__(self, values: Any) -> None:
        values = tuple(values)
        if not values:
            raise ValueError("Categorical values cannot be empty.")
        object.__setattr__(self, "values", values)

    def grid_values(self) -> list[Any]:
        return list(self.values)

    def random_distribution(self) -> list[Any]:
        return list(self.values)

    def suggest(self, trial: Any, name: str) -> Any:
        return trial.suggest_categorical(name, list(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "categorical", "values": list(self.values)}


@dataclass(frozen=True, slots=True)
class Integer:
    """Inclusive integer parameter domain."""

    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("Integer.low cannot exceed Integer.high.")
        if self.step <= 0:
            raise ValueError("Integer.step must be positive.")

    def grid_values(self) -> list[int]:
        return list(range(self.low, self.high + 1, self.step))

    def random_distribution(self) -> list[int]:
        # A finite stepped list preserves exactly the same domain across
        # RandomizedSearchCV, halving-random, and Optuna.
        return self.grid_values()

    def suggest(self, trial: Any, name: str) -> int:
        return int(trial.suggest_int(name, self.low, self.high, step=self.step))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "integer",
            "low": self.low,
            "high": self.high,
            "step": self.step,
        }


@dataclass(frozen=True, slots=True)
class Float:
    """Continuous or stepped floating-point parameter domain."""

    low: float
    high: float
    step: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("Float bounds must be finite.")
        if self.low >= self.high:
            raise ValueError("Float.low must be smaller than Float.high.")
        if self.step is not None and self.step <= 0:
            raise ValueError("Float.step must be positive when supplied.")

    def grid_values(self) -> list[float]:
        if self.step is None:
            raise ValueError(
                "Continuous Float parameters are not directly enumerable for grid search. "
                "Provide step=... or use random/Optuna optimization."
            )
        count = int(np.floor((self.high - self.low) / self.step))
        values = [self.low + index * self.step for index in range(count + 1)]
        if not np.isclose(values[-1], self.high) and values[-1] < self.high:
            values.append(self.high)
        return [float(value) for value in values]

    def random_distribution(self) -> Any:
        if self.step is not None:
            return self.grid_values()
        return uniform(loc=self.low, scale=self.high - self.low)

    def suggest(self, trial: Any, name: str) -> float:
        return float(
            trial.suggest_float(
                name,
                self.low,
                self.high,
                step=self.step,
            )
        )

    def to_dict(self) -> dict[str, Any]:
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
        if self.low <= 0 or self.high <= 0:
            raise ValueError("LogFloat bounds must be strictly positive.")
        if self.low >= self.high:
            raise ValueError("LogFloat.low must be smaller than LogFloat.high.")

    def grid_values(self) -> list[float]:
        raise ValueError(
            "Continuous LogFloat parameters are not directly enumerable for grid search. "
            "Use categorical values for an explicit logarithmic grid, or use random/Optuna."
        )

    def random_distribution(self) -> Any:
        return loguniform(self.low, self.high)

    def suggest(self, trial: Any, name: str) -> float:
        return float(trial.suggest_float(name, self.low, self.high, log=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "log_float",
            "low": self.low,
            "high": self.high,
        }


ParameterDomain = Categorical | Integer | Float | LogFloat | list[Any] | tuple[Any, ...]


def _coerce_domain(value: ParameterDomain) -> Categorical | Integer | Float | LogFloat:
    if isinstance(value, (Categorical, Integer, Float, LogFloat)):
        return value
    if isinstance(value, (list, tuple)):
        return Categorical(value)
    raise TypeError(
        "Search-space parameters must be Categorical, Integer, Float, LogFloat, "
        "or a finite list/tuple."
    )


def _domain_from_dict(payload: Any) -> ParameterDomain:
    # Legacy JSON files stored parameters directly as lists.
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, Mapping):
        raise TypeError("Serialized search-space parameter must be a list or mapping.")

    kind = payload.get("type")
    if kind == "categorical":
        return Categorical(payload["values"])
    if kind == "integer":
        return Integer(
            int(payload["low"]),
            int(payload["high"]),
            step=int(payload.get("step", 1)),
        )
    if kind == "float":
        step = payload.get("step")
        return Float(
            float(payload["low"]),
            float(payload["high"]),
            step=None if step is None else float(step),
        )
    if kind == "log_float":
        return LogFloat(float(payload["low"]), float(payload["high"]))
    raise ValueError(f"Unknown search-space parameter type '{kind}'.")


@dataclass(slots=True)
class SearchSpace:
    """Backend-agnostic hyperparameter search-space definition.

    Plain lists remain fully supported for backward compatibility and are
    interpreted as finite categorical domains.
    """

    name: str
    parameters: dict[str, ParameterDomain]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("SearchSpace.name cannot be empty.")
        self.parameters = dict(self.parameters)
        for name, domain in self.parameters.items():
            if not str(name).strip():
                raise ValueError("Search-space parameter names cannot be empty.")
            _coerce_domain(domain)

    @classmethod
    def from_dict(
        cls,
        name: str,
        parameters: dict[str, ParameterDomain],
    ) -> "SearchSpace":
        return cls(name=name, parameters=parameters)

    @classmethod
    def from_json(cls, path: str | Path) -> "SearchSpace":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        parameters = {
            name: _domain_from_dict(domain)
            for name, domain in data["parameters"].items()
        }
        return cls(name=data["name"], parameters=parameters)

    def to_dict(self) -> dict[str, Any]:
        serialized: dict[str, Any] = {}
        for name, domain in self.parameters.items():
            if isinstance(domain, list):
                # Preserve the established public representation for legacy
                # finite spaces.
                serialized[name] = list(domain)
            elif isinstance(domain, tuple):
                serialized[name] = list(domain)
            else:
                serialized[name] = _coerce_domain(domain).to_dict()
        return {"name": self.name, "parameters": serialized}

    def to_json(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=4)

    def get(self, parameter: str) -> ParameterDomain:
        return self.parameters[parameter]

    def exists(self, parameter: str) -> bool:
        return parameter in self.parameters

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

    def prefixed(self, prefix: str) -> "SearchSpace":
        """Return an equivalent space with parameter names prefixed."""

        return SearchSpace(
            name=self.name,
            parameters={f"{prefix}{name}": domain for name, domain in self.parameters.items()},
        )

    def __len__(self) -> int:
        return len(self.parameters)
