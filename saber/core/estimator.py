"""Canonical estimator construction for saber."""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class EstimatorFactory:
    """Build estimator instances through one deterministic construction path.

    Parameters
    ----------
    estimator_cls:
        Scikit-learn compatible estimator class.
    default_params:
        Parameters applied before call-specific overrides.

    Notes
    -----
    ``random_state`` is injected only when explicitly requested and when the
    estimator exposes that parameter. An explicit seed overrides a factory
    default; other call-specific parameters override their defaults normally.

    """

    estimator_cls: type
    default_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize default_params into a plain dict."""
        object.__setattr__(
            self,
            "default_params",
            dict(self.default_params),
        )

    def parameter_names(self) -> set[str]:
        """Return constructor parameters accepted by the estimator."""
        try:
            estimator = self.estimator_cls()
            get_params = getattr(estimator, "get_params", None)
            if callable(get_params):
                return set(get_params(deep=False))
        except Exception:  # noqa: S110, BLE001  # falls back to signature inspection below
            pass

        try:
            constructor = signature(self.estimator_cls)
        except (TypeError, ValueError):
            return set()

        names: set[str] = set()
        accepts_kwargs = False

        for name, parameter in constructor.parameters.items():
            if parameter.kind is Parameter.VAR_KEYWORD:
                accepts_kwargs = True
            elif parameter.kind is not Parameter.VAR_POSITIONAL:
                names.add(name)

        if accepts_kwargs:
            # Unknown keyword parameters may be accepted by third-party
            # sklearn-compatible providers. Known names remain useful for seed
            # propagation; arbitrary user parameters are still passed through.
            names.add("**kwargs")

        return names

    def supports_parameter(self, name: str) -> bool:
        """Return whether the estimator advertises a constructor parameter."""
        names = self.parameter_names()
        return name in names or "**kwargs" in names

    def resolved_params(
        self,
        *,
        random_state: int | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        """Resolve defaults, overrides, and optional random-state injection."""
        resolved = {
            **dict(self.default_params),
            **params,
        }

        if random_state is not None and self.supports_parameter("random_state"):
            resolved["random_state"] = random_state

        return resolved

    def build(
        self,
        *,
        random_state: int | None = None,
        **params: Any,
    ) -> Any:
        """Construct a fresh estimator instance."""
        resolved = self.resolved_params(
            random_state=random_state,
            **params,
        )

        return self.estimator_cls(**resolved)

    def with_defaults(
        self,
        default_params: Mapping[str, Any],
    ) -> EstimatorFactory:
        """Return a new factory with merged defaults."""
        merged = {
            **dict(self.default_params),
            **dict(default_params),
        }

        return EstimatorFactory(
            estimator_cls=self.estimator_cls,
            default_params=merged,
        )
