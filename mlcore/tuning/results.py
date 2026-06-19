"""
mlcore.tuning.results
=====================

Result objects returned by optimization methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlcore.core.specs import AlgorithmSpec

from mlcore.tuning.scorers import (
    is_loss_scorer,
)


# ============================================================
# Optimization Result
# ============================================================

@dataclass(slots=True)
class OptimizationResult:
    """
    Stores the result of a hyperparameter optimization process.

    Attributes
    ----------
    algorithm : str
        Optimized algorithm name.

    best_score : float
        Best optimization score.

    best_params : dict[str, Any]
        Best parameter configuration.

    best_model : Any
        Best fitted model.

    spec : AlgorithmSpec | None
        Associated algorithm specification.

    optimizer : str
        Optimizer name.

    metric : str
        Optimization metric.

    history : list[dict[str, Any]]
        Optimization history.

    study : Any
        Backend-specific study object
        (Optuna study, GridSearchCV object, etc.).
    """

    algorithm: str

    best_score: float

    best_params: dict[str, Any]

    best_model: Any = None

    spec: AlgorithmSpec | None = None

    optimizer: str | None = None

    metric: str | None = None

    history: list[dict[str, Any]] = field(
        default_factory=list,
    )

    study: Any = None

    # --------------------------------------------------------
    # Display helpers
    # --------------------------------------------------------

    @property
    def display_score(
        self,
    ) -> float:
        """
        User-facing score.

        Loss metrics are internally optimized
        as negative values by sklearn scorers.
        This property converts them back to
        their natural positive representation.
        """

        if (
            self.metric is not None
            and is_loss_scorer(
                self.metric,
            )
        ):
            return abs(
                self.best_score,
            )

        return self.best_score

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def has_model(
        self,
    ) -> bool:
        """
        Check whether a best model is available.
        """

        return self.best_model is not None

    def has_history(
        self,
    ) -> bool:
        """
        Check whether optimization history exists.
        """

        return len(
            self.history
        ) > 0

    def has_study(
        self,
    ) -> bool:
        """
        Check whether study object exists.
        """

        return self.study is not None

    def get_best_param(
        self,
        name: str,
    ) -> Any:
        """
        Retrieve a best parameter value.
        """

        return self.best_params[name]

    def get_history(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return optimization history.
        """

        return list(
            self.history
        )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Export result as dictionary.
        """

        return {
            "algorithm": self.algorithm,
            "optimizer": self.optimizer,
            "metric": self.metric,
            "best_score": self.best_score,
            "display_score": self.display_score,
            "best_params": self.best_params,
        }

    def summary(
        self,
    ) -> dict[str, Any]:
        """
        Return concise optimization summary.
        """

        return self.to_dict()