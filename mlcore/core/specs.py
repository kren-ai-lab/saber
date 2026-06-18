"""
mlcore.core.specs
=================

Defines the core specification for all algorithms in mlcore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from mlcore.core.search_space import SearchSpace


@dataclass(frozen=True)
class AlgorithmSpec:
    """
    Immutable specification of a machine learning algorithm.
    """

    backend: str
    task: str
    name: str

    runner: Callable[..., None]
    backend_cls: type

    aliases: tuple[str, ...] = field(default_factory=tuple)
    default_params: dict[str, Any] = field(default_factory=dict)
    search_space: SearchSpace | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    supports_cv: bool = False
    supports_proba: bool = False

    description: str | None = None

    def matches_name(
        self,
        query: str,
    ) -> bool:
        """Check whether query matches the algorithm name or aliases."""

        return query == self.name or query in self.aliases

    def has_tag(
        self,
        tag: str,
    ) -> bool:
        """Check whether algorithm contains a specific tag."""

        return tag in self.tags

    def get_default_params(
        self,
    ) -> dict[str, Any]:
        """Return default hyperparameters."""

        return dict(self.default_params)

    def has_search_space(
        self,
    ) -> bool:
        """Check whether a search space is available."""

        return self.search_space is not None

    def get_search_space(
        self,
    ) -> SearchSpace | None:
        """Return the search space object."""

        return self.search_space

    def get_search_space_parameters(
        self,
    ) -> dict[str, list[Any]]:
        """Return search space parameters as a dictionary."""

        if self.search_space is None:
            return {}

        return dict(self.search_space.parameters)


def make_spec(
    *,
    backend: str,
    task: str,
    name: str,
    runner: Callable[..., None],
    backend_cls: type,
    aliases: tuple[str, ...] | None = None,
    default_params: dict[str, Any] | None = None,
    search_space: SearchSpace | None = None,
    tags: tuple[str, ...] | None = None,
    supports_cv: bool = False,
    supports_proba: bool = False,
    description: str | None = None,
) -> AlgorithmSpec:
    """
    Convenience constructor for AlgorithmSpec.
    """

    return AlgorithmSpec(
        backend=backend,
        task=task,
        name=name,
        runner=runner,
        backend_cls=backend_cls,
        aliases=aliases or tuple(),
        default_params=default_params or {},
        search_space=search_space,
        tags=tags or tuple(),
        supports_cv=supports_cv,
        supports_proba=supports_proba,
        description=description,
    )