"""
mlcore.tuning.optuna
====================

Optuna-based hyperparameter optimization.
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna.trial import TrialState

from mlcore.exceptions import NonFiniteScoreError
from mlcore.tuning._validation import ensure_finite_score
from mlcore.tuning.base import BaseOptimizer
from mlcore.tuning.objective import ObjectiveBuilder
from mlcore.tuning.results import OptimizationResult


class OptunaOptimizer(BaseOptimizer):
    """Hyperparameter optimization using Optuna."""

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
        random_state: int | None = None,
    ) -> OptimizationResult:
        spec = self.get_spec(algorithm)

        if not spec.has_estimator_factory():
            raise ValueError(
                f"Algorithm '{algorithm}' does not define an estimator factory."
            )

        if search_space is None:
            search_space = spec.get_search_space()
        if search_space is None:
            raise ValueError(
                f"No search space defined for algorithm '{algorithm}'."
            )
        if len(search_space) == 0:
            raise ValueError(f"Search space for '{algorithm}' is empty.")

        # Every mlcore scorer is transformed to a maximization objective.
        if direction is None:
            direction = "maximize"
        if direction != "maximize":
            raise ValueError(
                "mlcore scorer objectives must use direction='maximize'. "
                "Loss metrics are already negated by the scorer contract."
            )

        builder = ObjectiveBuilder(
            spec=spec,
            search_space=search_space,
            metric=metric,
            cv=cv,
            X=X,
            y=y,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        objective = builder.build()

        if sampler is None and random_state is not None:
            sampler = optuna.samplers.TPESampler(seed=random_state)

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
            catch=(NonFiniteScoreError,),
        )

        completed = [
            trial
            for trial in study.trials
            if (
                trial.state == TrialState.COMPLETE
                and trial.value is not None
                and np.isfinite(float(trial.value))
            )
        ]
        if not completed:
            raise NonFiniteScoreError(
                algorithm=algorithm,
                metric=metric,
                optimizer="optuna",
                score=float("nan"),
            )

        best_trial = max(completed, key=lambda trial: float(trial.value))
        best_score = ensure_finite_score(
            float(best_trial.value),
            algorithm=algorithm,
            metric=metric,
            optimizer="optuna",
        )
        best_params = dict(best_trial.params)

        best_model = spec.build_estimator(
            random_state=random_state,
            **best_params,
        )
        best_model.fit(X, y)

        history = []
        for trial in study.trials:
            history.append(
                {
                    "trial": trial.number,
                    "score": (
                        None if trial.value is None else float(trial.value)
                    ),
                    "params": dict(trial.params),
                    "state": str(trial.state),
                }
            )

        return OptimizationResult(
            algorithm=algorithm,
            optimizer="optuna",
            metric=metric,
            best_score=best_score,
            best_params=best_params,
            best_model=best_model,
            spec=spec,
            history=history,
            study=study,
            refit=True,
        )
