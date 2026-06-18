"""
mlcore.core.typing
===================

Shared type definitions for the mlcore ecosystem.
"""

from __future__ import annotations

from typing import Any, TypeAlias, Literal

import numpy as np

# ============================================================
# Core array types
# ============================================================

ArrayLike = Any  # flexible input (lists, np arrays, pd frames)
ArrayLike1D: TypeAlias = np.ndarray
ArrayLike2D: TypeAlias = np.ndarray

# ============================================================
# Labels
# ============================================================

LabelArray: TypeAlias = np.ndarray

# ============================================================
# Features / datasets
# ============================================================

FeatureMatrix: TypeAlias = np.ndarray
TargetVector: TypeAlias = np.ndarray

# ============================================================
# Model types
# ============================================================

ModelType: TypeAlias = Any

BackendType: TypeAlias = Any

# ============================================================
# Metrics
# ============================================================

MetricValue: TypeAlias = float

MetricDict: TypeAlias = dict[str, float]

# ============================================================
# Registry types
# ============================================================

AlgorithmName: TypeAlias = str

TaskType: TypeAlias = Literal["classification", "regression"]
BackendName: TypeAlias = str