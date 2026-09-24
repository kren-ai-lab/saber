"""Golden fingerprints: provenance contracts that infrastructure work must not change."""

import numpy as np
import pandas as pd

from mlcore.datasets import DatasetBundle, PartitionPlan

IDS = ["s0", "s1", "s2", "s3"]


def _float_bundle() -> DatasetBundle:
    X = np.array([[0.5, 1.0, -2.25], [1.5, np.nan, 3.0], [2.5, 4.0, 0.0], [3.5, 5.0, 1.0]])
    return DatasetBundle(
        X=X,
        y=np.array(["a", "b", "a", "b"], dtype=object),
        sample_ids=IDS,
        feature_names=["f0", "f1", "f2"],
        groups=["g0", "g0", "g1", "g1"],
        sample_weight=[1.0, 2.0, 1.0, 0.5],
    )


def _mixed_frame_bundle() -> DatasetBundle:
    frame = pd.DataFrame(
        {
            "f0": [0.5, 1.5, 2.5, 3.5],
            "f1": [1.0, np.nan, 4.0, 5.0],
            "n": np.array([1, 2, 3, 4], dtype=np.int64),
            "flag": [True, False, True, False],
        }
    )
    return DatasetBundle(X=frame, y=np.array([0, 1, 0, 1]), sample_ids=[10, 11, 12, 13])


def test_numpy_float_dataset_fingerprint_is_stable():
    bundle = _float_bundle()
    assert bundle.fingerprint == "562121162148b257046f489cd705cf8f1d66e0af23ca4ee4b753fd0bb48aff8e"
    assert bundle.feature_schema.fingerprint == "8466944c7c0684941b75d7f114cc0f67568f61db6655817eae086a61243ccde9"
    assert bundle.feature_schema.dtypes == ("float64", "float64", "float64")


def test_numpy_int_dataset_fingerprint_is_stable():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    bundle = DatasetBundle(X=X, y=np.array([0.1, 0.2, 0.3, 0.4]))
    assert bundle.fingerprint == "e86b2bff106eb60c46176f4da0695f673925fd6df6118dc299ab5fc958649590"
    assert bundle.feature_schema.fingerprint == "3ca741ca1c716dcf842363775555c066ccbef48b8bdd0501feeea0b499d9bda0"


def test_dataframe_dataset_fingerprint_is_stable():
    bundle = _mixed_frame_bundle()
    assert bundle.fingerprint == "fc5243d21da04edb3d8ea7bbf56b08450e1229f80a5872d4f2253cffe13827fb"
    assert bundle.feature_schema.fingerprint == "96544a4a17e868ef3f4b7074511d9601bbc2555db5cb0443fdf5076e7e3a38e9"
    assert bundle.feature_schema.dtypes == ("float64", "float64", "int64", "bool")


def test_subset_fingerprint_and_order_are_stable():
    subset = _mixed_frame_bundle().subset([13, 11])
    assert subset.sample_ids == (11, 13)
    assert subset.fingerprint == "a2d19f21f3dc283d2cc66c9f3f7a89f4366fcc3e86e1f3e78364b23664418973"


def test_partition_fingerprint_is_stable():
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=IDS,
        fold_assignments=[0, 1, 0, 1],
        dataset_fingerprint=_float_bundle().fingerprint,
    )
    assert plan.fingerprint == "41bac11aa63dd37e5695299a4abe4ad87def22503be7e613b0f55c18ffa6bc53"
