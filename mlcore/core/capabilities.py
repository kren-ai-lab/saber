"""
mlcore.core.capabilities
========================

Static estimator capability and requirement contracts.

Capabilities describe interfaces that orchestration layers may use without
probing fitted estimators at runtime. Requirements describe known constraints
that later preprocessing/validation phases can enforce explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Literal


ScalingRecommendation = Literal[
    "optional",
    "recommended",
    "not_required",
]


@dataclass(frozen=True, slots=True)
class EstimatorCapabilities:
    """Static capabilities exposed by a registered estimator."""

    predict_proba: bool = False
    decision_function: bool = False
    sample_weight: bool = False
    native_missing_values: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Return a serialization-friendly capability mapping."""

        return {
            "predict_proba": self.predict_proba,
            "decision_function": self.decision_function,
            "sample_weight": self.sample_weight,
            "native_missing_values": self.native_missing_values,
        }


@dataclass(frozen=True, slots=True)
class EstimatorRequirements:
    """Known input/preprocessing requirements for an estimator."""

    non_negative_X: bool = False
    positive_y: bool = False
    scaling: ScalingRecommendation = "optional"

    def to_dict(self) -> dict[str, bool | str]:
        """Return a serialization-friendly requirement mapping."""

        return {
            "non_negative_X": self.non_negative_X,
            "positive_y": self.positive_y,
            "scaling": self.scaling,
        }


def infer_estimator_capabilities(
    estimator_cls: type,
    *,
    native_missing_values: bool = False,
) -> EstimatorCapabilities:
    """Infer stable interface capabilities from an estimator class.

    The inference is intentionally conservative. Only capabilities that can be
    determined from the public estimator interface without fitting data are
    inferred automatically. Provider modules may explicitly supply additional
    information, such as native missing-value handling.
    """

    predict_proba = hasattr(estimator_cls, "predict_proba")
    decision_function = hasattr(estimator_cls, "decision_function")

    sample_weight = False

    try:
        fit_signature = inspect.signature(estimator_cls.fit)
        sample_weight = "sample_weight" in fit_signature.parameters
    except (AttributeError, TypeError, ValueError):
        sample_weight = False

    return EstimatorCapabilities(
        predict_proba=predict_proba,
        decision_function=decision_function,
        sample_weight=sample_weight,
        native_missing_values=native_missing_values,
    )
