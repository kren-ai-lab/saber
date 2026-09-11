"""
mlcore.classification.runners
=============================

Legacy runner factories for classification compatibility.
"""

from __future__ import annotations

from typing import Any
from typing import Callable

import numpy as np

from mlcore.classification.backend import ClassificationBackend


def make_classifier_runner(
    model_cls: type,
) -> Callable[..., None]:
    """
    Create a generic classification runner.

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
        backend: ClassificationBackend,
        X: np.ndarray,
        y: np.ndarray,
        **params: Any,
    ) -> None:
        """
        Execute training for a classification model.
        """

        model = model_cls(**params)

        model.fit(
            X,
            y,
        )

        backend.set_model(model)

        backend.set_params(params)

        metadata: dict[str, Any] = {}

        if hasattr(model, "classes_"):
            metadata["classes"] = model.classes_

        if hasattr(model, "n_features_in_"):
            metadata["n_features"] = model.n_features_in_

        if metadata:
            backend.update_metadata(**metadata)

        predictions = model.predict(X)

        backend.set_predictions(predictions)

        if hasattr(model, "predict_proba"):

            try:

                probabilities = model.predict_proba(X)

                backend.set_probabilities(probabilities)

            except Exception:
                pass

        backend.post_fit_hook()

    return runner