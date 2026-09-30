"""Algorithm specifications used by the registry and execution engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

from saber.core.capabilities import (
    EstimatorCapabilities,
    EstimatorRequirements,
    infer_estimator_capabilities,
)

if TYPE_CHECKING:
    from saber.core.search_space import SearchSpace
    from saber.core.task import TaskType


@dataclass(frozen=True)
class AlgorithmSpec:
    """Immutable specification of a registered machine-learning algorithm."""

    provider: str
    task: TaskType
    name: str

    estimator_cls: type

    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: SearchSpace | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    capabilities: EstimatorCapabilities | None = None
    requirements: EstimatorRequirements = field(default_factory=EstimatorRequirements)
    description: str | None = None

    def __post_init__(self) -> None:
        """Resolve capabilities and description defaults."""
        object.__setattr__(self, "default_params", dict(self.default_params))

        if self.capabilities is None:
            object.__setattr__(self, "capabilities", infer_estimator_capabilities(self.estimator_cls))

        if self.description is None:
            object.__setattr__(self, "description", _description_from_estimator(self.estimator_cls))

    @property
    def resolved_capabilities(self) -> EstimatorCapabilities:
        """Return capabilities, always resolved by __post_init__."""
        if self.capabilities is None:
            raise AssertionError(f"AlgorithmSpec {self.name!r} has unresolved capabilities.")
        return self.capabilities

    def build_estimator(
        self,
        *,
        random_state: int | None = None,
        **params: Any,
    ) -> Any:
        """Construct a fresh estimator from defaults, overrides and an optional seed.

        ``random_state`` is injected only when given and when the estimator
        exposes that parameter; an explicit seed overrides any default.
        """
        resolved = {**self.default_params, **params}
        if random_state is not None and _supports_parameter(self.estimator_cls, "random_state"):
            resolved["random_state"] = random_state
        return self.estimator_cls(**resolved)

    def metadata(self) -> dict[str, Any]:
        """Return serialization-friendly registry metadata."""
        return {
            "name": self.name,
            "task": self.task,
            "provider": self.provider,
            "estimator": self.estimator_cls.__name__,
            "tags": self.tags,
            "description": self.description,
            "capabilities": self.resolved_capabilities.to_dict(),
            "requirements": self.requirements.to_dict(),
            "default_params": dict(self.default_params),
            "has_search_space": self.search_space is not None,
        }


def _description_from_estimator(estimator_cls: type) -> str | None:
    doc = getattr(estimator_cls, "__doc__", None)
    if not doc:
        return None
    for line in doc.splitlines():
        text = line.strip()
        if text:
            return text
    return None


def _supports_parameter(estimator_cls: type, name: str) -> bool:
    """Return whether the estimator advertises a constructor parameter."""
    names = _parameter_names(estimator_cls)
    return name in names or "**kwargs" in names


def _parameter_names(estimator_cls: type) -> set[str]:
    try:
        get_params = getattr(estimator_cls(), "get_params", None)
        if callable(get_params):
            return set(get_params(deep=False))
    except Exception:  # noqa: S110, BLE001  # falls back to signature inspection below
        pass

    try:
        constructor = signature(estimator_cls)
    except (TypeError, ValueError):
        return set()

    names: set[str] = set()
    for name, parameter in constructor.parameters.items():
        if parameter.kind is Parameter.VAR_KEYWORD:
            # Third-party sklearn-compatible providers may accept arbitrary keywords.
            names.add("**kwargs")
        elif parameter.kind is not Parameter.VAR_POSITIONAL:
            names.add(name)
    return names
