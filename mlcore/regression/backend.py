"""
mlcore.regression.backend
=========================

Backend implementation for regression models.
"""

from __future__ import annotations

from mlcore.core.base import BackendBase


class RegressionBackend(BackendBase):
    """
    Backend for regression models.

    Notes
    -----
    This backend extends BackendBase without adding
    additional functionality.

    It exists primarily to provide semantic separation
    between regression and classification workflows and
    to support future task-specific extensions.
    """

    pass