"""
mlcore.exceptions.base
======================

Core exception hierarchy for mlcore.
"""

from __future__ import annotations


class MLCoreError(Exception):
    """Base exception for mlcore."""


class RegistryError(MLCoreError):
    """Base registry exception."""


class AlgorithmAlreadyRegisteredError(RegistryError):
    """Raised when an algorithm is already registered."""

    def __init__(
        self,
        name: str,
    ) -> None:

        super().__init__(
            f"Algorithm '{name}' is already registered.",
        )


class AlgorithmNotFoundError(RegistryError):
    """Raised when an algorithm cannot be found."""

    def __init__(
        self,
        name: str,
    ) -> None:

        super().__init__(
            f"Algorithm '{name}' was not found in the registry.",
        )

class MetricError(MLCoreError, ValueError):
    """Base exception for metric contract errors."""


class MetricNotFoundError(MetricError):
    """Raised when a metric cannot be found in the canonical registry."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Metric '{name}' was not found in the metric registry.")


class MetricTaskMismatchError(MetricError):
    """Raised when a metric is used with the wrong supervised task."""

    def __init__(
        self,
        *,
        metric: str,
        metric_task: str,
        requested_task: str,
    ) -> None:
        super().__init__(
            f"Metric '{metric}' supports task '{metric_task}', "
            f"not '{requested_task}'."
        )


class MetricProblemTypeError(MetricError):
    """Raised when a metric does not support the requested class regime."""

    def __init__(
        self,
        *,
        metric: str,
        problem_type: str,
        supported: tuple[str, ...],
    ) -> None:
        supported_text = ", ".join(supported)
        super().__init__(
            f"Metric '{metric}' does not support problem type "
            f"'{problem_type}'. Supported: {supported_text}."
        )


class PredictionContractError(MLCoreError, ValueError):
    """Raised when structured predictions violate the prediction contract."""


class OptimizationError(MLCoreError, RuntimeError):
    """Base exception for optimization failures."""


class NonFiniteScoreError(OptimizationError):
    """Raised when an optimizer produces a NaN or infinite objective score."""

    def __init__(
        self,
        *,
        algorithm: str,
        metric: str,
        optimizer: str,
        score: float,
    ) -> None:
        super().__init__(
            f"Optimizer '{optimizer}' produced a non-finite score ({score}) "
            f"for algorithm '{algorithm}' and metric '{metric}'."
        )
