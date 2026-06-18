"""
mlcore.regression.runners
=========================

Shared runner factories for regression models.
"""

from __future__ import annotations

from typing import Any
from typing import Callable

import numpy as np

from mlcore.regression.backend import RegressionBackend


def make_regression_runner(
    model_cls: type,
) -> Callable[..., None]:
    """
    Create a generic regression runner.

    Parameters
    ----------
    model_cls : type
        Estimator class.

    Returns
    -------
    Callable[..., None]
        Runner function compatible with AlgorithmSpec.
    """

    def runner(
        backend: RegressionBackend,
        X: np.ndarray,
        y: np.ndarray,
        **params: Any,
    ) -> None:
        """
        Execute training for a regression model.

        Parameters
        ----------
        backend : RegressionBackend
            Backend instance.

        X : np.ndarray
            Feature matrix.

        y : np.ndarray
            Target vector.

        **params
            Estimator parameters.
        """

        model = model_cls(**params)

        model.fit(
            X,
            y,
        )

        backend.set_model(model)

        backend.set_params(params)

        metadata: dict[str, Any] = {}

        if hasattr(model, "n_features_in_"):
            metadata["n_features"] = model.n_features_in_

        if metadata:
            backend.update_metadata(**metadata)

        predictions = model.predict(
            X,
        )

        backend.set_predictions(
            predictions,
        )

        backend.post_fit_hook()

    return runner