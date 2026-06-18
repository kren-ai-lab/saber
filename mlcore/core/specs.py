"""
mlcore.core.specs
==================

Defines the core specification for all algorithms in mlcore.

The AlgorithmSpec acts as the central contract between:
- Registry system
- Trainer
- Runners
- Evaluation layer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple, Optional


# ============================================================
# Algorithm Specification
# ============================================================

@dataclass(frozen=True)
class AlgorithmSpec:
    """
    Immutable specification of a machine learning algorithm.

    This object defines how an algorithm is executed, configured,
    and registered within the mlcore framework.

    Attributes
    ----------
    backend : str
        Backend family identifier (e.g., "scikit", "xgboost").

    task : str
        Task type ("classification" or "regression").

    name : str
        Unique algorithm identifier used in the registry.

    runner : Callable
        Function that executes training logic.
        Signature:
            runner(backend, X, y, **params) -> None

    backend_cls : type
        Backend class used to store state and outputs.

    aliases : Tuple[str, ...]
        Alternative names for the algorithm.

    default_params : Dict[str, Any]
        Default hyperparameters.

    search_space : Dict[str, Any]
        Hyperparameter search space (Optuna / Grid / TPOT).

    tags : Tuple[str, ...]
        Metadata tags for filtering (e.g., "tree", "linear").

    supports_cv : bool
        Whether algorithm supports internal CV execution.

    supports_proba : bool
        Whether algorithm supports predict_proba.

    description : Optional[str]
        Human-readable description of the algorithm.
    """

    backend: str
    task: str
    name: str

    runner: Callable[..., None]
    backend_cls: type

    aliases: Tuple[str, ...] = field(default_factory=tuple)

    default_params: Dict[str, Any] = field(default_factory=dict)

    search_space: Dict[str, Any] = field(default_factory=dict)

    tags: Tuple[str, ...] = field(default_factory=tuple)

    supports_cv: bool = False
    supports_proba: bool = False

    description: Optional[str] = None

    # --------------------------------------------------------
    # Utility methods
    # --------------------------------------------------------

    def matches_name(self, query: str) -> bool:
        """
        Check if a name matches the algorithm or any alias.
        """
        return query == self.name or query in self.aliases

    def has_tag(self, tag: str) -> bool:
        """
        Check if algorithm contains a specific tag.
        """
        return tag in self.tags

    def get_default_params(self) -> Dict[str, Any]:
        """
        Return default hyperparameters.
        """
        return dict(self.default_params)

    def get_search_space(self) -> Dict[str, Any]:
        """
        Return hyperparameter search space.
        """
        return dict(self.search_space)


# ============================================================
# Helper constructors
# ============================================================

def make_spec(
    *,
    backend: str,
    task: str,
    name: str,
    runner: Callable[..., None],
    backend_cls: type,
    aliases: Optional[Tuple[str, ...]] = None,
    default_params: Optional[Dict[str, Any]] = None,
    search_space: Optional[Dict[str, Any]] = None,
    tags: Optional[Tuple[str, ...]] = None,
    supports_cv: bool = False,
    supports_proba: bool = False,
    description: Optional[str] = None,
) -> AlgorithmSpec:
    """
    Convenience constructor for AlgorithmSpec.

    This helps reduce boilerplate when registering algorithms.
    """

    return AlgorithmSpec(
        backend=backend,
        task=task,
        name=name,
        runner=runner,
        backend_cls=backend_cls,
        aliases=aliases or tuple(),
        default_params=default_params or {},
        search_space=search_space or {},
        tags=tags or tuple(),
        supports_cv=supports_cv,
        supports_proba=supports_proba,
        description=description,
    )
