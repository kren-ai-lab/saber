"""Numerical scaling helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from saber.exceptions import PreprocessingContractError

if TYPE_CHECKING:
    from saber.core.capabilities import EstimatorRequirements

_ALLOWED_SCALERS = {"standard", "robust", "minmax"}


def resolve_scaler_name(
    scaler: str | None,
    *,
    requirements: EstimatorRequirements,
) -> str | None:
    """Resolve explicit/automatic scaler policy for an estimator."""
    if scaler is None or scaler == "none":
        return None
    if scaler == "auto":
        if requirements.non_negative_X:
            return "minmax"
        if requirements.scaling == "recommended":
            return "standard"
        return None
    if scaler not in _ALLOWED_SCALERS:
        allowed = ", ".join(sorted(_ALLOWED_SCALERS))
        raise PreprocessingContractError(f"Unsupported scaler '{scaler}'. Supported: auto, {allowed}, none.")
    if requirements.non_negative_X and scaler in {"standard", "robust"}:
        raise PreprocessingContractError(
            "Standard/robust scaling can create negative values for an estimator "
            "that requires non-negative features. Use scaler='minmax' or 'auto'."
        )
    return scaler


def build_scaler(
    scaler: str | None,
    *,
    requirements: EstimatorRequirements,
) -> Any:
    """Build scaler or passthrough marker."""
    resolved = resolve_scaler_name(scaler, requirements=requirements)
    if resolved is None:
        return "passthrough"
    if resolved == "standard":
        return StandardScaler()
    if resolved == "robust":
        return RobustScaler()
    if resolved == "minmax":
        return MinMaxScaler(clip=True)
    raise AssertionError(f"Unhandled scaler: {resolved}")
