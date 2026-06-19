"""
mlcore.tuning.sklearn.grid
==========================

GridSearchCV optimizer.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import GridSearchCV

from mlcore.tuning.base import BaseOptimizer
from mlcore.tuning.results import OptimizationResult
from mlcore.tuning.scorers import get_scorer


class GridSearchOptimizer(BaseOptimizer):
    """
    Exhaustive hyperparameter optimization using GridSearchCV.
    """

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
        search_space=None,
        **fit_params: Any,
    ) -> OptimizationResult:
        """
        Execute exhaustive grid search.

        Parameters
        ----------
        algorithm : str
            Registered algorithm name.

        X :
            Training features.

        y :
            Training targets.

        metric : str
            Optimization metric.

        cv : int, default=5
            Number of cross-validation folds.

        n_jobs : int, default=-1
            Number of parallel jobs.

        refit : bool, default=True
            Whether to refit the best model.

        search_space : SearchSpace | None
            Optional custom search space.

        **fit_params
            Additional parameters passed to fit().
        """

        spec = self.get_spec(
            algorithm,
        )

        if spec.estimator_cls is None:

            raise ValueError(
                f"Algorithm '{algorithm}' "
                "does not define an estimator_cls."
            )

        if search_space is None:

            search_space = spec.get_search_space()

        if search_space is None:

            raise ValueError(
                f"No search space defined for "
                f"algorithm '{algorithm}'."
            )

        if len(search_space) == 0:

            raise ValueError(
                f"Search space for '{algorithm}' "
                "is empty."
            )

        scorer = get_scorer(
            metric,
        )

        estimator = spec.estimator_cls()

        grid = GridSearchCV(
            estimator=estimator,
            param_grid=search_space.parameters,
            scoring=scorer,
            cv=cv,
            n_jobs=n_jobs,
            refit=refit,
            return_train_score=True,
        )

        grid.fit(
            X,
            y,
            **fit_params,
        )

        history: list[dict[str, Any]] = []

        results = grid.cv_results_

        for idx in range(
            len(
                results["params"]
            )
        ):

            history.append(
                {
                    "params": results["params"][idx],
                    "mean_test_score": float(
                        results["mean_test_score"][idx]
                    ),
                    "std_test_score": float(
                        results["std_test_score"][idx]
                    ),
                    "rank_test_score": int(
                        results["rank_test_score"][idx]
                    ),
                }
            )

        return OptimizationResult(
            algorithm=algorithm,
            optimizer="grid_search",
            metric=metric,
            best_score=float(
                grid.best_score_
            ),
            best_params=dict(
                grid.best_params_
            ),
            best_model=grid.best_estimator_,
            spec=spec,
            history=history,
            study=grid,
        )