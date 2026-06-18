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