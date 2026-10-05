"""Non-finite scores are rejected before an OptimizationResult is built."""

import numpy as np
import pytest

from saber.exceptions import NonFiniteScoreError
from saber.tuning.engine import _best_index
from saber.tuning.results import OptimizationResult


def test_best_index_rejects_all_non_finite_scores():
    with pytest.raises(NonFiniteScoreError):
        _best_index(np.array([np.nan, np.inf]), "ridge", "r2", "grid")


def test_best_index_skips_non_finite_candidates():
    assert _best_index(np.array([np.nan, 0.2, 0.5]), "ridge", "r2", "grid") == 2


def test_optimization_result_rejects_non_finite_best_score():
    with pytest.raises(NonFiniteScoreError):
        OptimizationResult(
            algorithm="ridge", refit_metric="r2", best_scores={"r2": float("nan")}, best_params={}
        )
