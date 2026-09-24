from __future__ import annotations

import json

import numpy as np
import pytest
from optuna.trial import FixedTrial

from saber.core.search_space import (
    Categorical,
    Float,
    Integer,
    LogFloat,
    SearchSpace,
)


def test_legacy_lists_remain_supported() -> None:
    space = SearchSpace("legacy", {"C": [0.1, 1.0]})
    assert space.to_grid() == {"C": [0.1, 1.0]}
    assert space.to_random() == {"C": [0.1, 1.0]}


def test_integer_is_shared_across_grid_random_and_optuna() -> None:
    space = SearchSpace("typed", {"depth": Integer(2, 6, step=2)})
    assert space.to_grid()["depth"] == [2, 4, 6]
    assert space.to_random()["depth"] == [2, 4, 6]
    assert space.sample_optuna(FixedTrial({"depth": 4})) == {"depth": 4}


def test_categorical_is_shared_across_backends() -> None:
    space = SearchSpace("typed", {"kernel": Categorical(["linear", "rbf"])})
    assert space.to_grid()["kernel"] == ["linear", "rbf"]
    assert space.sample_optuna(FixedTrial({"kernel": "rbf"}))["kernel"] == "rbf"


def test_continuous_float_requires_discretization_for_grid() -> None:
    space = SearchSpace("typed", {"alpha": Float(0.0, 1.0)})
    with pytest.raises(ValueError, match="not directly enumerable"):
        space.to_grid()
    samples = space.to_random()["alpha"].rvs(size=20, random_state=42)
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_stepped_float_can_be_gridded() -> None:
    space = SearchSpace("typed", {"alpha": Float(0.0, 1.0, step=0.5)})
    assert space.to_grid()["alpha"] == [0.0, 0.5, 1.0]


def test_log_float_supports_random_and_optuna_but_not_grid() -> None:
    space = SearchSpace("typed", {"C": LogFloat(1e-3, 1e2)})
    with pytest.raises(ValueError, match="not directly enumerable"):
        space.to_grid()
    values = space.to_random()["C"].rvs(size=20, random_state=42)
    assert np.all(values > 0)
    sampled = space.sample_optuna(FixedTrial({"C": 0.1}))
    assert sampled["C"] == pytest.approx(0.1)


def test_typed_search_space_json_roundtrip(tmp_path) -> None:
    path = tmp_path / "space.json"
    original = SearchSpace(
        "typed",
        {
            "kind": Categorical(["a", "b"]),
            "depth": Integer(1, 5, step=2),
            "rate": Float(0.1, 0.5, step=0.2),
            "C": LogFloat(1e-4, 1e2),
        },
    )
    original.to_json(path)
    restored = SearchSpace.from_json(path)
    assert restored.to_dict() == original.to_dict()
    assert json.loads(path.read_text())["parameters"]["C"]["type"] == "log_float"


def test_parameter_prefixing_for_pipeline_backends() -> None:
    space = SearchSpace("typed", {"C": [0.1, 1.0]})
    assert set(space.to_grid(prefix="estimator__")) == {"estimator__C"}
    assert set(space.to_random(prefix="estimator__")) == {"estimator__C"}
