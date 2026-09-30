"""Registered search spaces must only name parameters their estimator accepts."""

from __future__ import annotations

from saber.core.registry import MODEL_REGISTRY


def test_search_space_parameters_are_valid_estimator_parameters() -> None:
    unknown = {}
    for spec in MODEL_REGISTRY:
        if not spec.has_search_space():
            continue
        accepted = set(spec.build_estimator().get_params())
        invalid = set(spec.get_search_space_parameters()) - accepted
        if invalid:
            unknown[spec.name] = sorted(invalid)
    assert not unknown, f"Search spaces name unknown estimator parameters: {unknown}"
