"""
mlcore.tuning.objective
=======================

Objective builders for Optuna optimization.
"""

from __future__ import annotations

from typing import Any
from typing import Callable

import numpy as np

from sklearn.model_selection import cross_val_score

from mlcore.core.search_space import SearchSpace
from mlcore.core.specs import AlgorithmSpec

from mlcore.tuning.scorers import get_scorer


class ObjectiveBuilder:
    """
    Build Optuna objective functions from
    AlgorithmSpec and SearchSpace objects.
    """

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
    ) -> None:

        self.spec = spec
        self.search_space = search_space

        self.metric = metric
        self.cv = cv

        self.X = X
        self.y = y

        self.n_jobs = n_jobs

    # ========================================================
    # Parameter sampling
    # ========================================================

    def sample_params(
        self,
        trial,
    ) -> dict[str, Any]:
        """
        Sample parameters from SearchSpace.

        Currently all parameters are treated
        as categorical variables.
        """

        params: dict[str, Any] = {}

        for (
            parameter_name,
            parameter_values,
        ) in self.search_space.parameters.items():

            params[
                parameter_name
            ] = trial.suggest_categorical(
                parameter_name,
                parameter_values,
            )

        return params

    # ========================================================
    # Objective factory
    # ========================================================

    def build(
        self,
    ) -> Callable:
        """
        Build Optuna objective function.
        """

        scorer = get_scorer(
            self.metric,
        )

        estimator_cls = (
            self.spec.estimator_cls
        )

        if estimator_cls is None:

            raise ValueError(
                f"Algorithm '{self.spec.name}' "
                "does not define estimator_cls."
            )

        X = self.X
        y = self.y

        cv = self.cv

        n_jobs = self.n_jobs

        def objective(
            trial,
        ) -> float:

            params = self.sample_params(
                trial,
            )

            model = estimator_cls(
                **params,
            )

            scores = cross_val_score(
                estimator=model,
                X=X,
                y=y,
                scoring=scorer,
                cv=cv,
                n_jobs=n_jobs,
            )

            return float(
                np.mean(scores)
            )

        return objective