"""Algorithm specifications used by the registry and execution engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from saber.core.capabilities import (
    EstimatorCapabilities,
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from saber.core.estimator import EstimatorFactory

if TYPE_CHECKING:
    from saber.core.search_space import SearchSpace
    from saber.core.task import TaskType


@dataclass(frozen=True)
class AlgorithmSpec:
    """Immutable specification of a registered machine-learning algorithm."""

    provider: str
    task: TaskType
    name: str

    estimator_cls: type | None = None
    estimator_factory: EstimatorFactory | None = None

    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: SearchSpace | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    capabilities: EstimatorCapabilities | None = None
    requirements: EstimatorRequirements = field(default_factory=EstimatorRequirements)
    supports_cv: bool = False
    description: str | None = None

    def __post_init__(self) -> None:
        """Resolve the estimator factory, capabilities, and description defaults."""
        default_params = dict(self.default_params)
        object.__setattr__(self, "default_params", default_params)

        factory = self.estimator_factory
        estimator_cls = self.estimator_cls

        if factory is None and estimator_cls is not None:
            factory = EstimatorFactory(
                estimator_cls=estimator_cls,
                default_params=default_params,
            )
            object.__setattr__(self, "estimator_factory", factory)
        elif factory is not None:
            if default_params:
                factory = factory.with_defaults(default_params)
                object.__setattr__(self, "estimator_factory", factory)

            if estimator_cls is None:
                estimator_cls = factory.estimator_cls
                object.__setattr__(self, "estimator_cls", estimator_cls)
            elif estimator_cls is not factory.estimator_cls:
                raise ValueError(
                    "estimator_cls and estimator_factory.estimator_cls "
                    "must reference the same estimator class."
                )

        capabilities = self.capabilities
        if capabilities is None and estimator_cls is not None:
            capabilities = infer_estimator_capabilities(estimator_cls)
        if capabilities is None:
            capabilities = EstimatorCapabilities()
        object.__setattr__(self, "capabilities", capabilities)

        if self.description is None and estimator_cls is not None:
            object.__setattr__(self, "description", _description_from_estimator(estimator_cls))

    def has_tag(self, tag: str) -> bool:
        """Return whether this algorithm is tagged with the given tag."""
        return tag in self.tags

    def get_default_params(self) -> dict[str, Any]:
        """Return a copy of the algorithm's default parameters."""
        return dict(self.default_params)

    def has_search_space(self) -> bool:
        """Return whether this algorithm defines a search space."""
        return self.search_space is not None

    def get_search_space(self) -> SearchSpace | None:
        """Return this algorithm's search space, if defined."""
        return self.search_space

    def get_search_space_parameters(self) -> dict[str, Any]:
        """Return a copy of the search space's parameters, if defined."""
        if self.search_space is None:
            return {}
        return dict(self.search_space.parameters)

    def has_estimator_factory(self) -> bool:
        """Return whether this algorithm defines an estimator factory."""
        return self.estimator_factory is not None

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
        """Build an estimator through the canonical factory."""
        if self.estimator_factory is None:
            raise ValueError(f"Algorithm '{self.name}' does not define an estimator factory.")
        return self.estimator_factory.build(random_state=random_state, **params)

    def metadata(self) -> dict[str, Any]:
        """Return serialization-friendly registry metadata."""
        estimator_name = None
        if self.estimator_cls is not None:
            estimator_name = self.estimator_cls.__name__

        return {
            "name": self.name,
            "task": self.task,
            "provider": self.provider,
            "estimator": estimator_name,
            "tags": self.tags,
            "description": self.description,
            "supports_cv": self.supports_cv,
            "capabilities": self.resolved_capabilities.to_dict(),
            "requirements": self.requirements.to_dict(),
            "default_params": dict(self.default_params),
            "has_search_space": self.has_search_space(),
            "has_estimator_factory": self.has_estimator_factory(),
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
