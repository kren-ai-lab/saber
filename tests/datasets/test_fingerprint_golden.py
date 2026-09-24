"""Golden fingerprints: provenance contracts that infrastructure work must not change."""

import numpy as np
import pandas as pd

from saber.datasets import DatasetBundle, PartitionPlan

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
    assert bundle.fingerprint == "34127c6582310a87e9ddd3ce2927dcdf49a23ab4009e36b495a557b9044874a0"
    assert (
        bundle.feature_schema.fingerprint
        == "03fa76b63bacc4d4d53b2127c58970c5e8ddc00b686e4c8224eacbf0c25b9200"
    )
    assert bundle.feature_schema.dtypes == ("float64", "float64", "float64")


def test_numpy_int_dataset_fingerprint_is_stable():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int64)
    bundle = DatasetBundle(X=X, y=np.array([0.1, 0.2, 0.3, 0.4]))
    assert bundle.fingerprint == "c19c53e24268620e11e933c65a11b6d84e28fca31e88b5d121bf49ee9e80c887"
    assert (
        bundle.feature_schema.fingerprint
        == "4a78ce8db8553917f40da9c5b7d646d6a8b0758d87e0eedd42825c48492882c2"
    )


def test_dataframe_dataset_fingerprint_is_stable():
    bundle = _mixed_frame_bundle()
    assert bundle.fingerprint == "0f04b24a4970a562567118699e657301a01b5e55ebc7c96ecdfab64be2bafdaf"
    assert (
        bundle.feature_schema.fingerprint
        == "c9109f309306648713fd4a28cb537cbed0295653f6fd38baa9e8bbdfb3dab994"
    )
    assert bundle.feature_schema.dtypes == ("float64", "float64", "int64", "bool")


def test_subset_fingerprint_and_order_are_stable():
    subset = _mixed_frame_bundle().subset([13, 11])
    assert subset.sample_ids == (11, 13)
    assert subset.fingerprint == "a8c55bdbc5ab820dc2405b51bbf487a806af478a2ccd511d0c1397271b165b97"


def test_partition_fingerprint_is_stable():
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=IDS,
        fold_assignments=[0, 1, 0, 1],
        dataset_fingerprint=_float_bundle().fingerprint,
    )
    assert plan.fingerprint == "eaf826d66fed2201f5202b68f235cf97ee587ff88f9b2fb828ee33bebcc133f0"
