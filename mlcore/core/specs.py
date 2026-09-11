"""
mlcore.core.specs
=================

Core algorithm specifications used by the registry and execution engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from mlcore.core.capabilities import (
    EstimatorCapabilities,
    EstimatorRequirements,
    infer_estimator_capabilities,
)
from mlcore.core.estimator import EstimatorFactory
from mlcore.core.search_space import SearchSpace


@dataclass(frozen=True)
class AlgorithmSpec:
    """Immutable specification of a registered machine-learning algorithm.

    ``estimator_factory`` is the canonical construction path. ``runner`` and
    ``backend_cls`` remain only as compatibility fields while legacy callers
    migrate to the unified engine.
    """

    backend: str
    task: str
    name: str

    estimator_cls: type | None = None
    estimator_factory: EstimatorFactory | None = None

    # Legacy execution fields retained temporarily for compatibility.
    runner: Callable[..., None] | None = None
    backend_cls: type | None = None

    aliases: tuple[str, ...] = field(default_factory=tuple)
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: SearchSpace | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    capabilities: EstimatorCapabilities | None = None
    requirements: EstimatorRequirements = field(
        default_factory=EstimatorRequirements,
    )

    supports_cv: bool = False
    supports_proba: bool | None = None

    description: str | None = None

    def __post_init__(self) -> None:
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

        if self.supports_proba is not None:
            capabilities = replace(
                capabilities,
                predict_proba=bool(self.supports_proba),
            )

        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(
            self,
            "supports_proba",
            capabilities.predict_proba,
        )

        if self.description is None and estimator_cls is not None:
            object.__setattr__(
                self,
                "description",
                _description_from_estimator(estimator_cls),
            )

    @property
    def provider(self) -> str:
        """Canonical provider identifier.

        ``backend`` is retained as the legacy field name for compatibility.
        """

        return self.backend

    def matches_name(self, query: str) -> bool:
        """Check whether query matches the algorithm name or aliases."""

        return query == self.name or query in self.aliases

    def has_tag(self, tag: str) -> bool:
        """Check whether algorithm contains a specific tag."""

        return tag in self.tags

    def get_default_params(self) -> dict[str, Any]:
        """Return default hyperparameters."""

        return dict(self.default_params)

    def has_search_space(self) -> bool:
        """Check whether a search space is available."""

        return self.search_space is not None

    def get_search_space(self) -> SearchSpace | None:
        """Return the search space object."""

        return self.search_space

    def get_search_space_parameters(self) -> dict[str, list[Any]]:
        """Return search space parameters as a dictionary."""

        if self.search_space is None:
            return {}

        return dict(self.search_space.parameters)

    def has_estimator_factory(self) -> bool:
        """Return whether canonical estimator construction is available."""

        return self.estimator_factory is not None

    def build_estimator(
        self,
        *,
        random_state: int | None = None,
        **params: Any,
    ):
        """Build an estimator through the canonical factory."""

        if self.estimator_factory is None:
            raise ValueError(
                f"Algorithm '{self.name}' does not define an estimator factory."
            )

        return self.estimator_factory.build(
            random_state=random_state,
            **params,
        )

    def metadata(self) -> dict[str, Any]:
        """Return serialization-friendly registry metadata."""

        estimator_name = None
        if self.estimator_cls is not None:
            estimator_name = self.estimator_cls.__name__

        return {
            "name": self.name,
            "task": self.task,
            "provider": self.provider,
            "backend": self.backend,
            "estimator": estimator_name,
            "aliases": self.aliases,
            "tags": self.tags,
            "description": self.description,
            "supports_cv": self.supports_cv,
            "supports_proba": bool(self.supports_proba),
            "capabilities": self.capabilities.to_dict(),
            "requirements": self.requirements.to_dict(),
            "default_params": dict(self.default_params),
            "has_search_space": self.has_search_space(),
            "has_estimator_factory": self.has_estimator_factory(),
        }


def _description_from_estimator(estimator_cls: type) -> str | None:
    """Extract the first useful line from an estimator docstring."""

    doc = getattr(estimator_cls, "__doc__", None)
    if not doc:
        return None

    for line in doc.splitlines():
        text = line.strip()
        if text:
            return text

    return None


def make_spec(
    *,
    backend: str,
    task: str,
    name: str,
    estimator_cls: type | None = None,
    estimator_factory: EstimatorFactory | None = None,
    runner: Callable[..., None] | None = None,
    backend_cls: type | None = None,
    aliases: tuple[str, ...] | None = None,
    default_params: dict[str, Any] | None = None,
    search_space: SearchSpace | None = None,
    tags: tuple[str, ...] | None = None,
    capabilities: EstimatorCapabilities | None = None,
    requirements: EstimatorRequirements | None = None,
    supports_cv: bool = False,
    supports_proba: bool | None = None,
    description: str | None = None,
) -> AlgorithmSpec:
    """Convenience constructor for :class:`AlgorithmSpec`."""

    return AlgorithmSpec(
        backend=backend,
        task=task,
        name=name,
        estimator_cls=estimator_cls,
        estimator_factory=estimator_factory,
        runner=runner,
        backend_cls=backend_cls,
        aliases=aliases or tuple(),
        default_params=default_params or {},
        search_space=search_space,
        tags=tags or tuple(),
        capabilities=capabilities,
        requirements=requirements or EstimatorRequirements(),
        supports_cv=supports_cv,
        supports_proba=supports_proba,
        description=description,
    )
