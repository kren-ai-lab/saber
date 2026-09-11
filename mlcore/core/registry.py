"""
mlcore.core.registry
====================

Central registry for machine learning algorithms.

This module provides the registry infrastructure used to
register, query, filter, and retrieve algorithm specifications
throughout the mlcore ecosystem.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Sequence

from mlcore.core.specs import AlgorithmSpec

from mlcore.exceptions import (
    AlgorithmAlreadyRegisteredError,
    AlgorithmNotFoundError,
)


class AlgorithmRegistry:
    """
    Registry for algorithm specifications.

    Notes
    -----
    The registry acts as the central discovery mechanism
    for all machine learning algorithms available in mlcore.

    Algorithms are registered using an AlgorithmSpec object.
    """

    def __init__(self) -> None:
        self._algorithms: Dict[str, AlgorithmSpec] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        spec: AlgorithmSpec,
        overwrite: bool = False,
    ) -> None:
        """
        Register a single algorithm.

        Parameters
        ----------
        spec : AlgorithmSpec
            Algorithm specification.

        overwrite : bool, default=False
            Whether to overwrite existing entries.

        Raises
        ------
        AlgorithmAlreadyRegisteredError
        """

        if (
            spec.name in self._algorithms
            and not overwrite
        ):
            raise AlgorithmAlreadyRegisteredError(
                f"Algorithm '{spec.name}' is already registered."
            )

        if (
            spec.name in self._aliases
            and not overwrite
        ):
            raise AlgorithmAlreadyRegisteredError(
                f"Algorithm name '{spec.name}' already exists as alias."
            )

        for alias in spec.aliases:

            if (
                alias in self._algorithms
                and not overwrite
            ):
                raise AlgorithmAlreadyRegisteredError(
                    f"Alias '{alias}' already exists as algorithm name."
                )

            if (
                alias in self._aliases
                and not overwrite
            ):
                raise AlgorithmAlreadyRegisteredError(
                    f"Alias '{alias}' is already registered."
                )

        self._algorithms[spec.name] = spec

        for alias in spec.aliases:
            self._aliases[alias] = spec.name

    def register_many(
        self,
        specs: Sequence[AlgorithmSpec],
        overwrite: bool = False,
    ) -> None:
        """
        Register multiple algorithm specifications.

        Parameters
        ----------
        specs : Sequence[AlgorithmSpec]
            Algorithm specifications.

        overwrite : bool, default=False
            Whether to overwrite existing entries.
        """

        for spec in specs:
            self.register(
                spec=spec,
                overwrite=overwrite,
            )

    def get(
        self,
        name: str,
    ) -> AlgorithmSpec:
        """
        Retrieve an algorithm specification.

        Parameters
        ----------
        name : str
            Algorithm name or alias.

        Returns
        -------
        AlgorithmSpec

        Raises
        ------
        AlgorithmNotFoundError
        """

        if name in self._algorithms:
            return self._algorithms[name]

        if name in self._aliases:
            canonical_name = self._aliases[name]
            return self._algorithms[canonical_name]

        raise AlgorithmNotFoundError(
            f"Algorithm '{name}' not found."
        )

    def exists(
        self,
        name: str,
    ) -> bool:
        """
        Check whether an algorithm exists.

        Parameters
        ----------
        name : str
            Algorithm name or alias.

        Returns
        -------
        bool
        """

        try:
            self.get(name)
            return True

        except AlgorithmNotFoundError:
            return False

    def filter(
        self,
        *,
        task: Optional[str] = None,
        backend: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
    ) -> list[AlgorithmSpec]:
        """
        Filter registered algorithms.

        Parameters
        ----------
        task : str, optional
            Task type.

        backend : str, optional
            Backend identifier.

        tags : Sequence[str], optional
            Required tags.

        Returns
        -------
        list[AlgorithmSpec]
        """

        results: list[AlgorithmSpec] = []

        for spec in self._algorithms.values():

            if (
                task is not None
                and spec.task != task
            ):
                continue

            if (
                backend is not None
                and spec.backend != backend
            ):
                continue

            if tags is not None:

                if not all(
                    tag in spec.tags
                    for tag in tags
                ):
                    continue

            results.append(spec)

        return results

    def get_by_task(
        self,
        task: str,
    ) -> list[AlgorithmSpec]:
        """
        Retrieve algorithms by task.
        """

        return self.filter(task=task)

    def get_by_backend(
        self,
        backend: str,
    ) -> list[AlgorithmSpec]:
        """
        Retrieve algorithms by backend.
        """

        return self.filter(
            backend=backend
        )

    def get_by_tag(
        self,
        tag: str,
    ) -> list[AlgorithmSpec]:
        """
        Retrieve algorithms by tag.
        """

        return self.filter(
            tags=[tag]
        )

    def list(
        self,
        *,
        task: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> list[str]:
        """
        List algorithm names.

        Parameters
        ----------
        task : str, optional
            Task filter.

        backend : str, optional
            Backend filter.

        Returns
        -------
        list[str]
        """

        return sorted(
            spec.name
            for spec in self.filter(
                task=task,
                backend=backend,
            )
        )

    def count(self) -> int:
        """
        Return the number of registered algorithms.

        Returns
        -------
        int
        """

        return len(self._algorithms)

    def tasks(self) -> set[str]:
        """
        Return registered tasks.

        Returns
        -------
        set[str]
        """

        return {
            spec.task
            for spec in self._algorithms.values()
        }

    def backends(self) -> set[str]:
        """
        Return registered backends.

        Returns
        -------
        set[str]
        """

        return {
            spec.backend
            for spec in self._algorithms.values()
        }

    def tags(self) -> set[str]:
        """
        Return all registered tags.

        Returns
        -------
        set[str]
        """

        tags: set[str] = set()

        for spec in self._algorithms.values():
            tags.update(spec.tags)

        return tags

    def aliases_for(
        self,
        name: str,
    ) -> tuple[str, ...]:
        """
        Retrieve aliases associated with an algorithm.

        Parameters
        ----------
        name : str
            Algorithm name.

        Returns
        -------
        tuple[str, ...]
        """

        spec = self.get(name)

        return spec.aliases

    def to_dict(self) -> dict[str, Any]:
        """Export registry metadata for all registered algorithms."""

        return {
            name: spec.metadata()
            for name, spec in self._algorithms.items()
        }

    def describe(
        self,
        name: str,
    ) -> dict[str, Any]:
        """Return metadata for one algorithm or alias."""

        return self.get(name).metadata()

    def get_factory(
        self,
        name: str,
    ):
        """Return the canonical estimator factory for an algorithm."""

        spec = self.get(name)

        if spec.estimator_factory is None:
            raise ValueError(
                f"Algorithm '{spec.name}' does not define an estimator factory."
            )

        return spec.estimator_factory

    def build_estimator(
        self,
        name: str,
        *,
        random_state: int | None = None,
        **params: Any,
    ):
        """Construct an estimator through the registry's canonical path."""

        return self.get(name).build_estimator(
            random_state=random_state,
            **params,
        )

    def summary(self) -> dict[str, Any]:
        """
        Generate registry summary.

        Returns
        -------
        dict[str, Any]
        """

        by_task = defaultdict(int)
        by_backend = defaultdict(int)

        for spec in self._algorithms.values():

            by_task[spec.task] += 1
            by_backend[spec.backend] += 1

        return {
            "n_algorithms": self.count(),
            "n_aliases": len(self._aliases),
            "n_with_estimator_factory": sum(
                spec.has_estimator_factory()
                for spec in self._algorithms.values()
            ),
            "tasks": dict(by_task),
            "backends": dict(by_backend),
            "providers": dict(by_backend),
            "available_tasks": sorted(self.tasks()),
            "available_backends": sorted(self.backends()),
            "available_providers": sorted(self.backends()),
            "available_tags": sorted(self.tags()),
        }

    def clear(self) -> None:
        """
        Remove all registered algorithms.
        """

        self._algorithms.clear()
        self._aliases.clear()

    def remove(
        self,
        name: str,
        missing_ok: bool = False,
    ) -> None:
        """
        Remove an algorithm from the registry.

        Parameters
        ----------
        name : str
            Algorithm name or alias.

        missing_ok : bool, default=False
            If True, silently ignore missing algorithms.

        Raises
        ------
        AlgorithmNotFoundError
            If the algorithm does not exist and
            ``missing_ok=False``.
        """

        try:
            spec = self.get(name)

        except AlgorithmNotFoundError:

            if missing_ok:
                return

            raise

        for alias in spec.aliases:
            self._aliases.pop(alias, None)

        self._algorithms.pop(spec.name, None)

    def get_runner(
        self,
        name: str,
    ):
        """Return the legacy runner compatibility field."""

        return self.get(name).runner

    def get_backend(
        self,
        name: str,
    ):
        """Return the legacy backend compatibility class."""

        return self.get(name).backend_cls

    @property
    def algorithms(self) -> dict[str, AlgorithmSpec]:
        """
        Registered algorithms.

        Returns
        -------
        dict[str, AlgorithmSpec]
        """

        return dict(self._algorithms)

    @property
    def aliases(self) -> dict[str, str]:
        """
        Registered aliases.

        Returns
        -------
        dict[str, str]
        """

        return dict(self._aliases)

    def __len__(self) -> int:
        return len(self._algorithms)

    def __contains__(
        self,
        item: str,
    ) -> bool:
        return self.exists(item)

    def __iter__(
        self,
    ) -> Iterator[AlgorithmSpec]:
        return iter(
            self._algorithms.values()
        )

    def __repr__(
        self,
    ) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(n_algorithms={len(self)})"
        )
    
# ============================================================
# Global registry
# ============================================================

MODEL_REGISTRY = AlgorithmRegistry()