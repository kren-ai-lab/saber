"""Numerical imputation helpers."""

from __future__ import annotations

from typing import Any

from sklearn.impute import SimpleImputer

from saber.exceptions import PreprocessingContractError


_ALLOWED_IMPUTATION = {"mean", "median", "most_frequent", "constant"}


def build_imputer(
    strategy: str | None,
    *,
    fill_value: float | int | str | None = 0.0,
) -> Any:
    """Build a numerical SimpleImputer or passthrough marker."""

    if strategy is None or strategy == "none":
        return "passthrough"
    if strategy not in _ALLOWED_IMPUTATION:
        allowed = ", ".join(sorted(_ALLOWED_IMPUTATION))
        raise PreprocessingContractError(
            f"Unsupported imputation strategy '{strategy}'. Supported: {allowed}, none."
        )
    kwargs = {"strategy": strategy}
    if strategy == "constant":
        kwargs["fill_value"] = fill_value
    return SimpleImputer(**kwargs)
