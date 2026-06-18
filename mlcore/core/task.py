"""
mlcore.core.task
=================

Task abstraction layer for mlcore.

This module provides semantic utilities for handling
machine learning problem types across the framework.

It is NOT required for core execution, but provides
structure and validation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


# ============================================================
# Core task type (canonical definition)
# ============================================================

TaskType = Literal["classification", "regression"]


# ============================================================
# Task wrapper (optional semantic layer)
# ============================================================

@dataclass(frozen=True)
class Task:
    """
    Lightweight representation of a machine learning task.

    This class is optional and used mainly for:
    - validation
    - readability
    - future pipeline orchestration (DL / PU / contrastive)
    """

    name: TaskType

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def is_classification(self) -> bool:
        """Check if task is classification."""
        return self.name == "classification"

    def is_regression(self) -> bool:
        """Check if task is regression."""
        return self.name == "regression"

    # --------------------------------------------------------
    # String representation
    # --------------------------------------------------------

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Task(name={self.name})"