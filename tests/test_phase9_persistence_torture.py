from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

import saber
from saber.datasets import DatasetBundle
from saber.exceptions import ArtifactIntegrityError, FeatureSchemaMismatchError, PersistenceError, PredictionContractError
from saber.persistence import inspect_artifact, load_model_artifact, save_model_artifact, verify_artifact
from saber.persistence.checksums import write_checksums


def _trained(tmp_path: Path):
    X, y = make_classification(n_samples=50, n_features=5, n_informative=3, random_state=4)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    dataset = DatasetBundle(frame, y, sample_ids=[f"s{i}" for i in range(50)])
    trained = saber.train(dataset=dataset, algorithm="logistic_regression", random_state=42)
    path = tmp_path / "model"
    save_model_artifact(path, model=trained.model, algorithm="logistic_regression", task="classification", dataset=dataset)
    return dataset, trained, path


def test_missing_checksum_file_is_rejected(tmp_path):
    _, _, path = _trained(tmp_path)
    (path / "checksums.sha256").unlink()
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        verify_artifact(path)


def test_missing_manifest_is_rejected_even_when_checksum_verification_disabled(tmp_path):
    _, _, path = _trained(tmp_path)
    (path / "manifest.json").unlink()
    with pytest.raises(ArtifactIntegrityError, match="manifest"):
        inspect_artifact(path, verify=False)


def test_manifest_missing_required_model_entry_is_rejected(tmp_path):
    _, _, path = _trained(tmp_path)
    manifest_path = path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    del payload["files"]["model"]
    manifest_path.write_text(json.dumps(payload))
    write_checksums(path)
    with pytest.raises(ArtifactIntegrityError, match="required file 'model'"):
        load_model_artifact(path)


def test_tampered_feature_schema_is_rejected_even_if_checksums_are_rewritten(tmp_path):
    _, _, path = _trained(tmp_path)
    schema_path = path / "feature_schema.json"
    payload = json.loads(schema_path.read_text())
    payload["names"][0] = "tampered"
    schema_path.write_text(json.dumps(payload))
    write_checksums(path)
    with pytest.raises(ArtifactIntegrityError, match="fingerprint"):
        load_model_artifact(path)


def test_wrong_feature_order_is_rejected_before_model_prediction(tmp_path):
    dataset, _, path = _trained(tmp_path)
    loaded = load_model_artifact(path)
    wrong = dataset.X[list(reversed(dataset.X.columns))]
    with pytest.raises(FeatureSchemaMismatchError):
        loaded.predict(wrong)


def test_missing_feature_is_rejected_before_model_prediction(tmp_path):
    dataset, _, path = _trained(tmp_path)
    loaded = load_model_artifact(path)
    with pytest.raises(FeatureSchemaMismatchError):
        loaded.predict(dataset.X.iloc[:, :-1])


def test_extra_feature_is_rejected_before_model_prediction(tmp_path):
    dataset, _, path = _trained(tmp_path)
    loaded = load_model_artifact(path)
    extra = dataset.X.copy()
    extra["extra"] = 0.0
    with pytest.raises(FeatureSchemaMismatchError):
        loaded.predict(extra)


def test_prediction_sample_ids_must_match_inference_row_count(tmp_path):
    dataset, _, path = _trained(tmp_path)
    loaded = load_model_artifact(path)
    with pytest.raises(PredictionContractError, match="sample_ids"):
        loaded.predict_result(dataset.X, sample_ids=["too_short"])


def test_artifact_overwrite_replaces_previous_model_atomically(tmp_path):
    dataset, trained, path = _trained(tmp_path)
    first_manifest = inspect_artifact(path)
    save_model_artifact(
        path,
        model=trained.model,
        algorithm="logistic_regression",
        task="classification",
        dataset=dataset,
        metadata={"revision": 2},
        overwrite=True,
    )
    second_manifest = verify_artifact(path)
    assert second_manifest.artifact_type == first_manifest.artifact_type == "model"
    assert not any(path.parent.glob(f".{path.name}.backup-*"))


def test_invalid_task_is_rejected_before_writing_artifact(tmp_path):
    dataset, trained, _ = _trained(tmp_path)
    with pytest.raises(PersistenceError, match="classification.*regression"):
        save_model_artifact(
            tmp_path / "bad",
            model=trained.model,
            algorithm="logistic_regression",
            task="clustering",
            dataset=dataset,
        )
