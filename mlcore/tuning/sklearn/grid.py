"""
mlcore.tuning.sklearn.grid
==========================

GridSearchCV optimizer.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV

from mlcore.tuning._validation import (
    best_estimator_or_none,
    ensure_finite_score,
)
from mlcore.tuning.base import BaseOptimizer
from mlcore.tuning.results import OptimizationResult
from mlcore.tuning.scorers import get_scorer


class GridSearchOptimizer(BaseOptimizer):
    """Exhaustive hyperparameter optimization using GridSearchCV."""

    def optimize(
        self,
        algorithm: str,
        X,
        y,
        *,
        metric: str,
        cv: int = 5,
        n_jobs: int = -1,
        refit: bool = True,
        random_state: int | None = None,
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

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=search_space.to_grid(),
            scoring=scorer,
            cv=cv,
            n_jobs=n_jobs,
            refit=refit,
            return_train_score=True,
        )
        grid.fit(X, y, **fit_params)

        best_score = ensure_finite_score(
            grid.best_score_,
            algorithm=algorithm,
            metric=metric,
            optimizer="grid_search",
        )

        results = grid.cv_results_
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
            optimizer="grid_search",
            metric=metric,
            best_score=best_score,
            best_params=dict(grid.best_params_),
            best_model=best_estimator_or_none(grid, refit=refit),
            spec=spec,
            history=history,
            study=grid,
            refit=refit,
        )
