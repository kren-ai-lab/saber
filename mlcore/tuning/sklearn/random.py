"""
mlcore.tuning.sklearn.random
============================

RandomizedSearchCV optimizer.
"""

from __future__ import annotations

from typing import Any

from sklearn.model_selection import RandomizedSearchCV

from mlcore.tuning.base import BaseOptimizer
from mlcore.tuning.results import OptimizationResult
from mlcore.tuning.scorers import get_scorer


class RandomSearchOptimizer(BaseOptimizer):
    """
    Randomized hyperparameter optimization using
    sklearn.model_selection.RandomizedSearchCV.
    """

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
        """
        Execute randomized search.

        Parameters
        ----------
        algorithm : str
            Registered algorithm.

        X :
            Training features.

        y :
            Training targets.

        metric : str
            Optimization metric.

        cv : int, default=5
            Number of CV folds.

        n_iter : int, default=20
            Number of parameter combinations sampled.

        n_jobs : int, default=-1
            Number of parallel jobs.

        refit : bool, default=True
            Refit best estimator.

        random_state : int | None
            Random seed.

        search_space : SearchSpace | None
            Optional custom search space.
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

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=search_space.parameters,
            scoring=scorer,
            cv=cv,
            n_iter=n_iter,
            n_jobs=n_jobs,
            refit=refit,
            random_state=random_state,
            return_train_score=True,
        )

        search.fit(
            X,
            y,
            **fit_params,
        )

        history: list[dict[str, Any]] = []

        results = search.cv_results_

        for idx in range(
            len(results["params"])
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
            optimizer="random_search",
            metric=metric,
            best_score=float(
                search.best_score_
            ),
            best_params=dict(
                search.best_params_
            ),
            best_model=search.best_estimator_,
            spec=spec,
            history=history,
            study=search,
        )