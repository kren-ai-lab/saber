from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from saber import predict
from saber.core import get_algorithm
from saber.core.results import TrainResult
from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import (
    ArtifactIntegrityError,
    DatasetFingerprintMismatchError,
    FeatureSchemaMismatchError,
    PersistenceError,
)
from saber.persistence import (
    ARTIFACT_SCHEMA_VERSION,
    inspect_artifact,
    load_model,
    save_model,
)
from saber.persistence import load as load_module
from saber.preprocessing import PreprocessingConfig
from saber.preprocessing.pipeline import build_model_pipeline
from saber.utils.tabular import to_numpy


def _forbidden_joblib_load(*_args, **_kwargs):
    raise AssertionError("joblib.load must not be called")


def _fitted_fixture():
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        random_state=42,
    )
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(X, columns=columns)
    dataset = DatasetBundle(
        X=frame,
        y=y,
        sample_ids=[f"sample_{i}" for i in range(len(y))],
    )
    assert dataset.sample_ids is not None
    plan = PartitionPlan.holdout(
        train_ids=dataset.sample_ids[:40],
        validation_ids=dataset.sample_ids[40:50],
        test_ids=dataset.sample_ids[50:],
        dataset=dataset,
    )
    spec = get_algorithm("logistic_regression")
    estimator = spec.build_estimator(random_state=42)
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=estimator,
        training_data=dataset,
    )
    pipeline.fit(to_numpy(dataset.X), dataset.y)
    return dataset, plan, pipeline


def _train_result(pipeline, dataset, *, positive_class=None, metadata=None, parameters=None):
    return TrainResult(
        model=pipeline,
        spec=get_algorithm("logistic_regression"),
        parameters=parameters or {},
        metadata=metadata or {},
        feature_schema=dataset.feature_schema,
        positive_class=positive_class,
    )


def test_model_artifact_round_trip_preserves_predictions_and_probabilities(tmp_path):
    dataset, plan, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    expected_prediction = pipeline.predict(to_numpy(dataset.X))
    expected_probability = pipeline.predict_proba(to_numpy(dataset.X))

    save_model(
        artifact_path,
        _train_result(
            pipeline,
            dataset,
            positive_class=1,
            parameters={"random_state": 42},
            metadata={"random_state": 42, "preprocessing": "auto"},
        ),
        dataset=dataset,
        partition_plan=plan,
    )

    loaded = load_model(artifact_path)
    result = predict(loaded, dataset.X, sample_ids=dataset.sample_ids)
    # A TrainResult artifact carries no metrics file; the training config records the inputs.
    assert not (artifact_path / "metrics.json").exists()
    training_config = json.loads((artifact_path / "training_config.json").read_text())
    assert training_config["random_state"] == 42
    assert training_config["preprocessing"] == "auto"

    np.testing.assert_array_equal(result.predictions, expected_prediction)
    assert result.probabilities is not None
    np.testing.assert_allclose(result.probabilities, expected_probability)
    assert result.positive_class == 1
    np.testing.assert_array_equal(result.classes, pipeline.classes_)
    assert result.decision_scores is not None
    np.testing.assert_allclose(result.decision_scores, pipeline.decision_function(to_numpy(dataset.X)))
    assert loaded.provenance["positive_class"] == 1
    assert loaded.feature_schema.fingerprint == dataset.feature_schema.fingerprint
    assert loaded.provenance["dataset_fingerprint"] == dataset.fingerprint
    assert loaded.provenance["partition_fingerprint"] == plan.fingerprint
    assert loaded.manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert inspect_artifact(artifact_path).artifact_type == "model"


def test_feature_schema_mismatch_is_detected_before_prediction(tmp_path):
    dataset, _, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    save_model(artifact_path, _train_result(pipeline, dataset), dataset=dataset)
    loaded = load_model(artifact_path)
    wrong_order = dataset.X[list(reversed(dataset.X.columns))]

    with pytest.raises(FeatureSchemaMismatchError):
        loaded.predict(wrong_order)


def test_corrupted_model_is_rejected_before_deserialization(tmp_path, monkeypatch):
    dataset, _, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    save_model(artifact_path, _train_result(pipeline, dataset), dataset=dataset)
    with (artifact_path / "model.joblib").open("ab") as handle:
        handle.write(b"corruption")
    monkeypatch.setattr(load_module.joblib, "load", _forbidden_joblib_load)

    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        load_model(artifact_path)


def test_model_artifact_requires_schema_and_protects_existing_path(tmp_path):
    dataset, _, pipeline = _fitted_fixture()
    other = DatasetBundle(X=np.asarray(dataset.X)[:, :3], y=dataset.y)
    with pytest.raises(PersistenceError, match="feature schema does not match"):
        save_model(tmp_path / "wrong_schema", _train_result(pipeline, other), dataset=dataset)

    artifact_path = tmp_path / "model_artifact"
    save_model(artifact_path, _train_result(pipeline, dataset), dataset=dataset)
    with pytest.raises(PersistenceError, match="already exists"):
        save_model(artifact_path, _train_result(pipeline, dataset), dataset=dataset)


def test_model_artifact_rejects_partition_from_another_dataset(tmp_path):
    dataset, _, pipeline = _fitted_fixture()
    foreign = DatasetBundle(
        X=np.asarray(dataset.X) + 0.01,
        y=dataset.y,
        sample_ids=dataset.sample_ids,
        feature_names=dataset.feature_names,
    )
    assert foreign.sample_ids is not None
    foreign_plan = PartitionPlan.holdout(
        train_ids=foreign.sample_ids[:40],
        test_ids=foreign.sample_ids[40:],
        dataset=foreign,
    )
    with pytest.raises(DatasetFingerprintMismatchError, match="fingerprint"):
        save_model(
            tmp_path / "bad_partition",
            _train_result(pipeline, dataset),
            dataset=dataset,
            partition_plan=foreign_plan,
        )


def test_round_trip_works_in_fresh_python_process(tmp_path):
    dataset, _, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    expected_path = tmp_path / "expected.npy"
    data_path = tmp_path / "features.csv"
    np.save(expected_path, pipeline.predict(to_numpy(dataset.X)))
    dataset.X.write_csv(data_path)

    save_model(artifact_path, _train_result(pipeline, dataset, positive_class=1), dataset=dataset)

    code = f"""
import numpy as np
import pandas as pd
import saber
from saber.persistence import load_model
artifact = load_model(r'{artifact_path}')
X = pd.read_csv(r'{data_path}')
expected = np.load(r'{expected_path}')
observed = artifact.predict(X)
assert np.array_equal(observed, expected)
result = saber.predict(artifact, X)
assert result.probabilities.shape == (60, 2)
print('PHASE7_SUBPROCESS_OK')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(  # noqa: S603  trusted, fixed argument list, no shell interpolation
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "PHASE7_SUBPROCESS_OK" in completed.stdout


def test_inspect_does_not_need_to_load_joblib(tmp_path, monkeypatch):
    dataset, _, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    save_model(artifact_path, _train_result(pipeline, dataset), dataset=dataset)
    monkeypatch.setattr(load_module.joblib, "load", _forbidden_joblib_load)
    manifest = inspect_artifact(artifact_path)
    assert manifest.metadata["algorithm"] == "logistic_regression"


def test_optimization_result_artifact_records_selection_scores_and_training_config(tmp_path):
    from saber import SearchSpace, TuningConfig, tune

    dataset, plan, _ = _fitted_fixture()
    result = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=TuningConfig(optimizer="grid", n_jobs=1),
        partition_plan=plan,
        search_space=SearchSpace("lr", {"C": [0.1, 1.0]}),
        preprocessing=PreprocessingConfig(scaler="standard"),
        metrics=("accuracy",),
        random_state=5,
    )
    path = save_model(tmp_path / "tuned", result, dataset=dataset)

    assert not (path / "metrics.json").exists()
    scores = json.loads((path / "selection_scores.json").read_text())
    assert set(scores["selection_scores"]) == {"accuracy"}
    assert "selection_scores" in load_model(path).manifest.files
    training_config = json.loads((path / "training_config.json").read_text())
    assert training_config["random_state"] == 5
    assert training_config["preprocessing"]["scaler"] == "standard"
