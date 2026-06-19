"""
mlcore.tuning.optuna
====================

Optuna-based hyperparameter optimization.
"""

from __future__ import annotations

from typing import Any

import optuna

from mlcore.tuning.base import BaseOptimizer
from mlcore.tuning.objective import (
    ObjectiveBuilder,
)
from mlcore.tuning.results import (
    OptimizationResult,
)


class OptunaOptimizer(BaseOptimizer):
    """
    Hyperparameter optimization using Optuna.
    """

    def optimize(
        self,
        algorithm: str,
        X,
        y,
        *,
        metric: str,
        cv: int = 5,
        n_trials: int = 50,
        timeout: float | None = None,
        n_jobs: int = -1,
        search_space=None,
        study_name: str | None = None,
        direction: str | None = None,
        sampler=None,
        pruner=None,
    ) -> OptimizationResult:
        """
        Optimize hyperparameters using Optuna.

        Parameters
        ----------
        algorithm : str
            Registered algorithm.

        X :
            Feature matrix.

        y :
            Target vector.

        metric : str
            Optimization metric.

        cv : int, default=5
            Number of folds.

        n_trials : int, default=50
            Number of Optuna trials.

        timeout : float | None
            Maximum optimization time in seconds.

        n_jobs : int, default=-1
            Parallel jobs used during CV.

        search_space : SearchSpace | None
            Custom search space.

        study_name : str | None
            Optuna study name.

        direction : str | None
            Optimization direction.

        sampler :
            Optional Optuna sampler.

        pruner :
            Optional Optuna pruner.
        """

        spec = self.get_spec(
            algorithm,
        )

        if spec.estimator_cls is None:

            raise ValueError(
                f"Algorithm '{algorithm}' "
                "does not define estimator_cls."
            )

        if search_space is None:

            search_space = (
                spec.get_search_space()
            )

        if search_space is None:

            raise ValueError(
                f"No search space defined for "
                f"algorithm '{algorithm}'."
            )

        if direction is None:

            # All scorers returned by get_scorer()
            # are converted into maximization objectives.
            direction = "maximize"

        builder = ObjectiveBuilder(
            spec=spec,
            search_space=search_space,
            metric=metric,
            cv=cv,
            X=X,
            y=y,
            n_jobs=n_jobs,
        )

        objective = builder.build()

        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
        )

        best_params = dict(
            study.best_params
        )

        best_model = (
            spec.estimator_cls(
                **best_params
            )
        )

        best_model.fit(
            X,
            y,
        )

        history = []

        for trial in study.trials:

            if trial.value is None:

                continue

            history.append(
                {
                    "trial": trial.number,
                    "score": float(
                        trial.value
                    ),
                    "params": dict(
                        trial.params
                    ),
                    "state": str(
                        trial.state
                    ),
                }
            )

        return OptimizationResult(
            algorithm=algorithm,
            optimizer="optuna",
            metric=metric,
            best_score=float(
                study.best_value
            ),
            best_params=best_params,
            best_model=best_model,
            spec=spec,
            history=history,
            study=study,
        )