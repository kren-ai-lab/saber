"""
mlcore.tuning.sklearn.random
============================

RandomizedSearchCV optimizer.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import RandomizedSearchCV

from mlcore.tuning._validation import (
    best_estimator_or_none,
    ensure_finite_score,
)
from mlcore.tuning.base import BaseOptimizer
from mlcore.tuning.results import OptimizationResult
from mlcore.tuning.scorers import get_scorer


class RandomSearchOptimizer(BaseOptimizer):
    """Randomized hyperparameter optimization using RandomizedSearchCV."""

    def optimize(
        self,
        algorithm: str,
        X,
        y,
        *,
        metric: str,
        cv: int = 5,
        n_iter: int = 20,
        n_jobs: int = -1,
        refit: bool = True,
        random_state: int | None = 42,
        search_space=None,
        **fit_params: Any,
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

        scorer = get_scorer(metric, task=spec.task, y=y)
        estimator = spec.build_estimator(random_state=random_state)

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_space.to_random(),
            scoring=scorer,
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            refit=refit,
            random_state=random_state,
            return_train_score=True,
        )
        search.fit(X, y, **fit_params)

        best_score = ensure_finite_score(
            search.best_score_,
            algorithm=algorithm,
            metric=metric,
            optimizer="random_search",
        )

        results = search.cv_results_
        history: list[dict[str, Any]] = []
        for idx in range(len(results["params"])):
            history.append(
                {
                    "params": results["params"][idx],
                    "mean_test_score": float(results["mean_test_score"][idx]),
                    "std_test_score": float(results["std_test_score"][idx]),
                    "rank_test_score": int(results["rank_test_score"][idx]),
                }
            )

        return OptimizationResult(
            algorithm=algorithm,
            optimizer="random_search",
            metric=metric,
            best_score=best_score,
            best_params=dict(search.best_params_),
            best_model=best_estimator_or_none(search, refit=refit),
            spec=spec,
            history=history,
            study=search,
            refit=refit,
        )
