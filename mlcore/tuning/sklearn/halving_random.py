"""
mlcore.tuning.sklearn.halving_random
====================================

Successive halving random search optimizer.
"""

from __future__ import annotations

from typing import Any

from mlcore.tuning.sklearn import _experimental

from sklearn.model_selection import (
    HalvingRandomSearchCV,
)

from mlcore.tuning.base import BaseOptimizer
from mlcore.tuning.results import OptimizationResult
from mlcore.tuning.scorers import get_scorer


class HalvingRandomSearchOptimizer(BaseOptimizer):
    """
    Successive Halving optimization using
    HalvingRandomSearchCV.
    """

    def optimize(
        self,
        algorithm: str,
        X,
        y,
        *,
        metric: str,
        cv: int = 5,
        factor: int = 3,
        resource: str = "n_samples",
        max_resources: str | int = "auto",
        min_resources: str | int = "smallest",
        n_jobs: int = -1,
        refit: bool = True,
        random_state: int | None = 42,
        search_space=None,
        **fit_params: Any,
    ) -> OptimizationResult:

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

        search = HalvingRandomSearchCV(
            estimator=estimator,
            param_distributions=search_space.parameters,
            scoring=scorer,
            cv=cv,
            factor=factor,
            resource=resource,
            max_resources=max_resources,
            min_resources=min_resources,
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

        history = []

        results = search.cv_results_

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
                    "iter": int(
                        results["iter"][idx]
                    ),
                }
            )

        return OptimizationResult(
            algorithm=algorithm,
            optimizer="halving_random_search",
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