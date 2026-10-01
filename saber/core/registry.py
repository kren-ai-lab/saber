"""Static catalog of the algorithms shipped with saber."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from saber._optional import is_dependency_available
from saber.classification import sklearn as _classification_sklearn
from saber.exceptions import AlgorithmNotFoundError
from saber.regression import sklearn as _regression_sklearn

if TYPE_CHECKING:
    from collections.abc import Mapping

    from saber.core.specs import AlgorithmSpec


def _collect_specs() -> tuple[AlgorithmSpec, ...]:
    specs = [*_classification_sklearn.SPECS, *_regression_sklearn.SPECS]

    if is_dependency_available("xgboost"):
        from saber.classification import xgboost as clf_xgboost  # noqa: PLC0415  # optional extra
        from saber.regression import xgboost as reg_xgboost  # noqa: PLC0415  # optional extra

        specs += [*clf_xgboost.SPECS, *reg_xgboost.SPECS]

    if is_dependency_available("lightgbm"):
        from saber.classification import lightgbm as clf_lightgbm  # noqa: PLC0415  # optional extra
        from saber.regression import lightgbm as reg_lightgbm  # noqa: PLC0415  # optional extra

        specs += [*clf_lightgbm.SPECS, *reg_lightgbm.SPECS]

    return tuple(specs)


def _build_catalog() -> Mapping[str, AlgorithmSpec]:
    catalog: dict[str, AlgorithmSpec] = {}
    for spec in _collect_specs():
        if spec.name in catalog:
            raise ValueError(f"Duplicate algorithm name '{spec.name}'.")
        catalog[spec.name] = spec
    return MappingProxyType(catalog)


ALGORITHMS: Mapping[str, AlgorithmSpec] = _build_catalog()


def get_algorithm(name: str) -> AlgorithmSpec:
    """Return the algorithm specification registered under ``name``.

    Raises
    ------
    AlgorithmNotFoundError
        If ``name`` is not in :data:`ALGORITHMS`.

    """
    try:
        return ALGORITHMS[name]
    except KeyError:
        raise AlgorithmNotFoundError(name, available=sorted(ALGORITHMS)) from None
