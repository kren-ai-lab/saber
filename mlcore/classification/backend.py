"""
mlcore.classification.backend
=============================

Backend implementations for classification tasks.
"""

from __future__ import annotations

from mlcore.core.base import BackendBase


class ClassificationBackend(BackendBase):
    """
    Backend for classification models.

    This backend stores:

    - trained model
    - predictions
    - probabilities
    - metrics
    - parameters

    and can be shared across:

    - scikit-learn
    - XGBoost
    - LightGBM
    - CatBoost (future)
    - other classification frameworks
    """

    pass