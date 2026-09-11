"""
mlcore.tuning._validation
=========================

Internal validation helpers shared by optimization backends.
"""

from __future__ import annotations

import numpy as np

from mlcore.exceptions import NonFiniteScoreError


def ensure_finite_score(
    score: float,
    *,
    algorithm: str,
    metric: str,
    optimizer: str,
) -> float:
    """Return a finite score or raise a domain-specific optimization error."""

    value = float(score)
    if not np.isfinite(value):
        raise NonFiniteScoreError(
            algorithm=algorithm,
            metric=metric,
            optimizer=optimizer,
            score=value,
        )
    return value


def best_estimator_or_none(search, *, refit: bool):
    """Return a search object's fitted best estimator only when refit was requested."""

    if not refit:
        return None
    return search.best_estimator_
