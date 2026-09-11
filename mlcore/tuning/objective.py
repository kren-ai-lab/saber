"""
mlcore.tuning.objective
=======================

Objective builders for Optuna optimization.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.model_selection import cross_val_score

from mlcore.core.search_space import SearchSpace
from mlcore.core.specs import AlgorithmSpec
from mlcore.exceptions import NonFiniteScoreError
from mlcore.tuning.scorers import get_scorer


class ObjectiveBuilder:
    """Build Optuna objective functions from algorithm and search-space specs."""

    def __init__(
        self,
        *,
        spec: AlgorithmSpec,
        search_space: SearchSpace,
        metric: str,
        cv: int,
        X,
        y,
        n_jobs: int = -1,
        random_state: int | None = None,
    ) -> None:
        self.spec = spec
        self.search_space = search_space
        self.metric = metric
        self.cv = cv
        self.X = X
        self.y = y
        self.n_jobs = n_jobs
        self.random_state = random_state

    def sample_params(self, trial) -> dict[str, Any]:
        """Sample parameters from the current categorical SearchSpace contract."""

        params: dict[str, Any] = {}
        for parameter_name, parameter_values in self.search_space.parameters.items():
            params[parameter_name] = trial.suggest_categorical(
                parameter_name,
                parameter_values,
            )
        return params

    def build(self) -> Callable:
        """Build an Optuna objective with task-aware metric validation."""

        scorer = get_scorer(
            self.metric,
            task=self.spec.task,
            y=self.y,
        )

        if not self.spec.has_estimator_factory():
            raise ValueError(
                f"Algorithm '{self.spec.name}' does not define an estimator factory."
            )

        X = self.X
        y = self.y
        cv = self.cv
        n_jobs = self.n_jobs
        random_state = self.random_state

        def objective(trial) -> float:
            params = self.sample_params(trial)
            model = self.spec.build_estimator(
                random_state=random_state,
                **params,
            )

            scores = np.asarray(
                cross_val_score(
                    estimator=model,
                    X=X,
                    y=y,
                    scoring=scorer,
                    cv=cv,
                    n_jobs=n_jobs,
                ),
                dtype=float,
            )

            if not np.all(np.isfinite(scores)):
                raise NonFiniteScoreError(
                    algorithm=self.spec.name,
                    metric=self.metric,
                    optimizer="optuna",
                    score=float(np.mean(scores)),
                )

            return float(np.mean(scores))

        return objective
