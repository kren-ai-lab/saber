"""
mlcore.tuning.base
==================

Base classes for hyperparameter optimization.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from typing import Any

from mlcore.core.registry import AlgorithmRegistry
from mlcore.tuning.results import OptimizationResult


# ============================================================
# Base Optimizer
# ============================================================

class BaseOptimizer(ABC):
    """
    Abstract optimizer interface.

    All optimization backends must inherit from this class.
    Examples include:

    - GridSearchOptimizer
    - RandomSearchOptimizer
    - HalvingGridSearchOptimizer
    - HalvingRandomSearchOptimizer
    - OptunaOptimizer
    - TPOTOptimizer
    """

    def __init__(
        self,
        registry: AlgorithmRegistry,
    ) -> None:
        """
        Parameters
        ----------
        registry : AlgorithmRegistry
            Global algorithm registry.
        """

        self.registry = registry

    # --------------------------------------------------------
    # Registry helpers
    # --------------------------------------------------------

    def algorithm_exists(
        self,
        algorithm: str,
    ) -> bool:
        """
        Check whether an algorithm exists.
        """

        return self.registry.exists(
            algorithm,
        )

    def get_spec(
        self,
        algorithm: str,
    ):
        """
        Retrieve AlgorithmSpec.
        """

        return self.registry.get(
            algorithm,
        )

    # --------------------------------------------------------
    # Search-space helpers
    # --------------------------------------------------------

    def get_search_space(
        self,
        algorithm: str,
    ):
        """
        Retrieve search space associated
        with an algorithm.
        """

        spec = self.get_spec(
            algorithm,
        )

        return spec.get_search_space()

    def get_search_space_parameters(
        self,
        algorithm: str,
    ) -> dict[str, list[Any]]:
        """
        Retrieve search-space parameters.
        """

        spec = self.get_spec(
            algorithm,
        )

        return spec.get_search_space_parameters()

    # --------------------------------------------------------
    # Main API
    # --------------------------------------------------------

    @abstractmethod
    def optimize(
        self,
        algorithm: str,
        X,
        y,
        **kwargs,
    ) -> OptimizationResult:
        """
        Execute optimization.
        """
        raise NotImplementedError

    # --------------------------------------------------------
    # Representation
    # --------------------------------------------------------

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}"
            f"(registry={len(self.registry)})"
        )