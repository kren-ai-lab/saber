"""
tests.test_search_space
=======================

Tests for SearchSpace.
"""

from __future__ import annotations

from saber.core.search_space import SearchSpace


def test_search_space_creation() -> None:

    space = SearchSpace(
        name="rf",
        parameters={
            "n_estimators": [100, 200],
        },
    )

    assert space.name == "rf"

    assert "n_estimators" in space.parameters


def test_search_space_exists() -> None:

    space = SearchSpace(
        name="rf",
        parameters={
            "n_estimators": [100],
        },
    )

    assert space.exists(
        "n_estimators",
    )

    assert not space.exists(
        "max_depth",
    )


def test_search_space_get() -> None:

    space = SearchSpace(
        name="rf",
        parameters={
            "n_estimators": [100, 200],
        },
    )

    values = space.get(
        "n_estimators",
    )

    assert values == [
        100,
        200,
    ]


def test_search_space_len() -> None:

    space = SearchSpace(
        name="rf",
        parameters={
            "a": [1],
            "b": [2],
        },
    )

    assert len(space) == 2


def test_search_space_dict_roundtrip() -> None:

    space = SearchSpace(
        name="rf",
        parameters={
            "n_estimators": [100],
        },
    )

    data = space.to_dict()

    assert data["name"] == "rf"

    assert "n_estimators" in data["parameters"]
