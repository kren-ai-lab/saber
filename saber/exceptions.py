"""Saber exception hierarchy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


class SaberError(Exception):
    """Base exception for saber."""


class AlgorithmNotFoundError(SaberError):
    """Raised when an algorithm cannot be found."""

    def __init__(
        self,
        name: str,
        available: Sequence[str] = (),
    ) -> None:
        """Initialize the error with the not-found algorithm name."""
        message = f"Algorithm '{name}' was not found."
        if available:
            message += f" Available algorithms: {', '.join(available)}."
        super().__init__(message)


class MetricError(SaberError, ValueError):
    """Base exception for metric contract errors."""


class MetricNotFoundError(MetricError):
    """Raised when a metric cannot be found in the canonical registry."""

    def __init__(self, name: str) -> None:
        """Initialize the error with the not-found metric name."""
        super().__init__(f"Metric '{name}' was not found in the metric registry.")


class MetricIncompatibleError(MetricError):
    """Raised when a metric does not support the requested task or problem type."""


class PredictionContractError(SaberError, ValueError):
    """Raised when structured predictions violate the prediction contract."""


class OptimizationError(SaberError, RuntimeError):
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
        """Initialize the error with the offending optimizer, algorithm, metric, and score."""
        super().__init__(
            f"Optimizer '{optimizer}' produced a non-finite score ({score}) "
            f"for algorithm '{algorithm}' and metric '{metric}'."
        )


class DatasetError(SaberError, ValueError):
    """Base exception for dataset contract errors."""


class DatasetValidationError(DatasetError):
    """Raised when a dataset violates structural or semantic contracts."""


class FeatureSchemaMismatchError(DatasetError):
    """Raised when feature identity/order/dtypes do not match a schema."""


class PartitionError(SaberError, ValueError):
    """Base exception for partition contract errors."""


class PartitionValidationError(PartitionError):
    """Raised when explicit partition membership is invalid."""


class DatasetFingerprintMismatchError(PartitionError):
    """Raised when a partition artifact targets a different dataset."""

    def __init__(self, *, expected: str, observed: str) -> None:
        """Initialize the error with the expected and observed fingerprints."""
        super().__init__(
            "Partition dataset fingerprint does not match the supplied dataset: "
            f"expected '{expected}', observed '{observed}'."
        )


class OptionalDependencyError(SaberError, ImportError):
    """Raised when an optional integration dependency is required but absent."""

    def __init__(self, *, dependency: str, extra: str, purpose: str) -> None:
        """Initialize the error with the missing dependency, extra, and purpose."""
        super().__init__(
            f"Optional dependency '{dependency}' is required for {purpose}. "
            f"Install with: pip install 'saberlib[{extra}]'."
        )


class PartitionIntegrationError(PartitionError):
    """Raised when an external partition engine cannot be adapted safely."""


class PreprocessingContractError(SaberError, ValueError):
    """Raised when preprocessing cannot satisfy estimator/data requirements."""


class ValidationContractError(SaberError, ValueError):
    """Raised when a validation workflow violates an explicit contract."""


class BenchmarkContractError(SaberError, ValueError):
    """Raised when a benchmark matrix violates a scientific/workflow contract."""


class PersistenceError(SaberError, RuntimeError):
    """Base exception for persistence/artifact failures."""


class ArtifactIntegrityError(PersistenceError):
    """Raised when a persistence artifact is missing, malformed, or corrupted."""


class ArtifactCompatibilityError(PersistenceError):
    """Raised when an artifact schema/environment is incompatible."""


class ConfigurationError(SaberError, ValueError):
    """Raised when a declarative workflow configuration is invalid."""


__all__ = [
    "AlgorithmNotFoundError",
    "ArtifactCompatibilityError",
    "ArtifactIntegrityError",
    "BenchmarkContractError",
    "ConfigurationError",
    "DatasetError",
    "DatasetFingerprintMismatchError",
    "DatasetValidationError",
    "FeatureSchemaMismatchError",
    "MetricError",
    "MetricIncompatibleError",
    "MetricNotFoundError",
    "NonFiniteScoreError",
    "OptimizationError",
    "OptionalDependencyError",
    "PartitionError",
    "PartitionIntegrationError",
    "PartitionValidationError",
    "PersistenceError",
    "PredictionContractError",
    "PreprocessingContractError",
    "SaberError",
    "ValidationContractError",
]
