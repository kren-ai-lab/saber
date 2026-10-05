from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

import saber
from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import FeatureSchemaMismatchError, ValidationContractError
from saber.preprocessing import PreprocessingConfig


def _classification_dataset(n=72):
    X, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        random_state=7,
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    return DatasetBundle(frame, y, sample_ids=[f"s{i}" for i in range(n)])


def _cv_plan(dataset):
    folds = [i % 3 for i in range(dataset.n_samples)]
    return PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=folds,
        dataset=dataset,
    )


def test_train_predict_and_evaluate_share_prediction_contract():
    dataset = _classification_dataset()
    result = saber.train(
        dataset=dataset,
        algorithm="logistic_regression",
        preprocessing=PreprocessingConfig(scaler="standard"),
        random_state=42,
    )
    prediction = saber.predict(result, dataset)
    evaluation = saber.evaluate(result, dataset, metrics=("accuracy", "roc_auc"))

    assert prediction.n_samples == dataset.n_samples
    assert prediction.sample_ids is not None
    assert dataset.sample_ids is not None
    assert prediction.sample_ids.tolist() == list(dataset.sample_ids)
    assert set(evaluation.metrics) == {"accuracy", "roc_auc"}
    assert evaluation.prediction is not None


def test_predict_keeps_native_dtype_for_integer_sample_ids():
    dataset = _classification_dataset()
    result = saber.train(
        dataset=dataset,
        algorithm="logistic_regression",
        preprocessing=PreprocessingConfig(scaler="standard"),
        random_state=42,
    )
    prediction = saber.predict(result, dataset.X, sample_ids=list(range(dataset.n_samples)))

    assert prediction.sample_ids is not None
    assert prediction.sample_ids.dtype.kind == "i"


def _named_classification_dataset(n=60):
    X, y = make_classification(n_samples=n, n_features=4, n_informative=3, n_redundant=0, random_state=11)
    frame = pd.DataFrame(X, columns=list("abcd"))
    return DatasetBundle(frame, y, sample_ids=[f"s{i}" for i in range(n)])


def test_predict_rejects_reordered_dataframe_columns():
    dataset = _named_classification_dataset()
    result = saber.train(dataset=dataset, algorithm="logistic_regression", random_state=42)
    reordered = dataset.X[list("dcba")]
    with pytest.raises(FeatureSchemaMismatchError):
        saber.predict(result, reordered)


def test_evaluate_rejects_reordered_dataframe_columns():
    dataset = _named_classification_dataset()
    result = saber.train(dataset=dataset, algorithm="logistic_regression", random_state=42)
    reordered = DatasetBundle(dataset.X[list("dcba")], dataset.y, sample_ids=dataset.sample_ids)
    with pytest.raises(FeatureSchemaMismatchError):
        saber.evaluate(result, reordered, metrics=("accuracy",))


def test_predict_and_evaluate_accept_dataset_feature_name_overrides():
    dataset = _named_classification_dataset()
    result = saber.train(dataset=dataset, algorithm="logistic_regression", random_state=42)

    renamed = dataset.X.rename({"a": "x0", "b": "x1", "c": "x2", "d": "x3"})
    overridden = DatasetBundle(renamed, dataset.y, sample_ids=dataset.sample_ids, feature_names=list("abcd"))

    prediction = saber.predict(result, overridden)
    evaluation = saber.evaluate(result, overridden, metrics=("accuracy",))

    reference = saber.evaluate(result, dataset, metrics=("accuracy",))
    np.testing.assert_array_equal(prediction.predictions, saber.predict(result, dataset).predictions)
    assert evaluation.metrics == reference.metrics


def test_predict_rejects_renamed_dataframe_columns_without_override():
    dataset = _named_classification_dataset()
    result = saber.train(dataset=dataset, algorithm="logistic_regression", random_state=42)

    renamed = dataset.X.rename({"a": "x0", "b": "x1", "c": "x2", "d": "x3"})
    with pytest.raises(FeatureSchemaMismatchError):
        saber.predict(result, renamed)


def test_predict_accepts_same_order_dataframe_and_matching_width_numpy_array():
    dataset = _named_classification_dataset()
    result = saber.train(dataset=dataset, algorithm="logistic_regression", random_state=42)

    same_order = saber.predict(result, dataset.X)
    from_numpy = saber.predict(result, np.asarray(dataset.X))

    np.testing.assert_array_equal(same_order.predictions, from_numpy.predictions)


def test_predict_rejects_unsupported_model_type():
    with pytest.raises(ValidationContractError):
        saber.predict(object(), np.zeros((2, 3)))  # pyrefly: ignore[bad-argument-type]


def test_public_api_regression_end_to_end():
    X, y = make_regression(  # pyrefly: ignore[bad-unpacking]
        n_samples=60,
        n_features=5,
        noise=0.2,
        random_state=3,
    )
    dataset = DatasetBundle(
        pd.DataFrame(X, columns=[f"x{i}" for i in range(5)]),
        y,
        sample_ids=[f"r{i}" for i in range(60)],
    )
    result = saber.train(dataset=dataset, algorithm="ridge_regressor", random_state=42)
    evaluation = saber.evaluate(result, dataset, metrics=("rmse", "mae"))
    assert evaluation.metrics["rmse"] >= 0.0
    assert evaluation.metrics["mae"] >= 0.0


def test_saber_works_without_pandas():
    # Blocks pandas the way an environment without it would; optional providers
    # (xgboost/lightgbm) import it opportunistically and must tolerate its absence.
    code = (
        "import sys; sys.modules['pandas'] = None\n"
        "import numpy as np, saber\n"
        "from saber.datasets import DatasetBundle\n"
        "DatasetBundle(X=np.ones((4, 2)), y=np.array([0, 1, 0, 1]))\n"
        "print('ok')"
    )
    completed = subprocess.run(  # noqa: S603  trusted, fixed argument list, no shell interpolation
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_train_positive_class_matches_loaded_artifact(tmp_path) -> None:
    X, y = make_classification(n_samples=60, n_features=5, random_state=3)  # pyrefly: ignore[bad-unpacking]
    dataset = DatasetBundle(X=X, y=y)
    trained = saber.train(dataset=dataset, algorithm="logistic_regression", positive_class=0)
    saber.save_model(tmp_path / "m", trained, dataset=dataset)
    loaded = saber.load_model(tmp_path / "m")
    in_memory = saber.predict(trained, dataset)
    from_disk = saber.predict(loaded, dataset)
    assert in_memory.positive_class == from_disk.positive_class == 0
    np.testing.assert_allclose(in_memory.positive_probabilities(), from_disk.positive_probabilities())
    assert saber.evaluate(trained, dataset).metrics == saber.evaluate(loaded, dataset).metrics
