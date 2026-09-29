from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from saber import MODEL_REGISTRY
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
    load_model_artifact,
    save_model_artifact,
    verify_artifact,
)
from saber.preprocessing import build_model_pipeline
from saber.utils.tabular import to_numpy


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
    plan = PartitionPlan.holdout(
        train_ids=dataset.sample_ids[:40],
        validation_ids=dataset.sample_ids[40:50],
        test_ids=dataset.sample_ids[50:],
        dataset_fingerprint=dataset.fingerprint,
    )
    spec = MODEL_REGISTRY.get("logistic_regression")
    estimator = spec.build_estimator(random_state=42)
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=estimator,
        training_data=dataset,
    )
    pipeline.fit(to_numpy(dataset.X), dataset.y)
    return dataset, plan, pipeline


def test_model_artifact_round_trip_preserves_predictions_and_probabilities(tmp_path):
    dataset, plan, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    expected_prediction = pipeline.predict(to_numpy(dataset.X))
    expected_probability = pipeline.predict_proba(to_numpy(dataset.X))

    save_model_artifact(
        artifact_path,
        model=pipeline,
        algorithm="logistic_regression",
        task="classification",
        dataset=dataset,
        partition_plan=plan,
        provider="sklearn",
        parameters={"random_state": 42},
        metrics={"accuracy": 0.9},
        training_config={"preprocessing": "auto"},
        positive_class=1,
    )

    loaded = load_model_artifact(artifact_path)
    result = loaded.predict_result(dataset.X, sample_ids=dataset.sample_ids)

    np.testing.assert_array_equal(result.predictions, expected_prediction)
    np.testing.assert_allclose(result.probabilities, expected_probability)
    assert result.positive_class == 1
    assert loaded.provenance["positive_class"] == 1
    assert loaded.feature_schema.fingerprint == dataset.feature_schema.fingerprint
    assert loaded.provenance["dataset_fingerprint"] == dataset.fingerprint
    assert loaded.provenance["partition_fingerprint"] == plan.fingerprint
    assert loaded.manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert verify_artifact(artifact_path).artifact_type == "model"


def test_feature_schema_mismatch_is_detected_before_prediction(tmp_path):
    dataset, _, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    save_model_artifact(
        artifact_path,
        model=pipeline,
        algorithm="logistic_regression",
        task="classification",
        dataset=dataset,
    )
    loaded = load_model_artifact(artifact_path)
    wrong_order = dataset.X[list(reversed(dataset.X.columns))]

    with pytest.raises(FeatureSchemaMismatchError):
        loaded.predict(wrong_order)


def test_corrupted_model_is_rejected_before_deserialization(tmp_path):
    dataset, _, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    save_model_artifact(
        artifact_path,
        model=pipeline,
        algorithm="logistic_regression",
        task="classification",
        dataset=dataset,
    )
    with (artifact_path / "model.joblib").open("ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        load_model_artifact(artifact_path)


def test_model_artifact_requires_schema_and_protects_existing_path(tmp_path):
    _, _, pipeline = _fitted_fixture()
    with pytest.raises(PersistenceError, match="dataset or feature_schema"):
        save_model_artifact(
            tmp_path / "missing_schema",
            model=pipeline,
            algorithm="logistic_regression",
            task="classification",
        )

    dataset, _, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    save_model_artifact(
        artifact_path,
        model=pipeline,
        algorithm="logistic_regression",
        task="classification",
        dataset=dataset,
    )
    with pytest.raises(PersistenceError, match="already exists"):
        save_model_artifact(
            artifact_path,
            model=pipeline,
            algorithm="logistic_regression",
            task="classification",
            dataset=dataset,
        )


def test_model_artifact_rejects_partition_from_another_dataset(tmp_path):
    dataset, _, pipeline = _fitted_fixture()
    foreign = DatasetBundle(
        X=np.asarray(dataset.X) + 0.01,
        y=dataset.y,
        sample_ids=dataset.sample_ids,
        feature_names=dataset.feature_names,
    )
    foreign_plan = PartitionPlan.holdout(
        train_ids=foreign.sample_ids[:40],
        test_ids=foreign.sample_ids[40:],
        dataset_fingerprint=foreign.fingerprint,
    )
    with pytest.raises(DatasetFingerprintMismatchError, match="fingerprint"):
        save_model_artifact(
            tmp_path / "bad_partition",
            model=pipeline,
            algorithm="logistic_regression",
            task="classification",
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

    save_model_artifact(
        artifact_path,
        model=pipeline,
        algorithm="logistic_regression",
        task="classification",
        dataset=dataset,
        positive_class=1,
    )

    code = f"""
import numpy as np
import pandas as pd
from saber.persistence import load_model_artifact
artifact = load_model_artifact(r'{artifact_path}')
X = pd.read_csv(r'{data_path}')
expected = np.load(r'{expected_path}')
observed = artifact.predict(X)
assert np.array_equal(observed, expected)
result = artifact.predict_result(X)
assert result.probabilities.shape == (60, 2)
print('PHASE7_SUBPROCESS_OK')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert "PHASE7_SUBPROCESS_OK" in completed.stdout


def test_inspect_does_not_need_to_load_joblib(tmp_path):
    dataset, _, pipeline = _fitted_fixture()
    artifact_path = tmp_path / "model_artifact"
    save_model_artifact(
        artifact_path,
        model=pipeline,
        algorithm="logistic_regression",
        task="classification",
        dataset=dataset,
    )
    manifest = inspect_artifact(artifact_path)
    assert manifest.metadata["algorithm"] == "logistic_regression"
