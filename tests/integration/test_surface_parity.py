from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

import saber
from saber.cli.main import EXIT_OK, main
from saber.config import load_config, run_config
from saber.datasets import DatasetBundle, PartitionPlan
from saber.utils.tabular import read_table


def _write_classification_case(tmp_path: Path):
    X, y = make_classification(n_samples=45, n_features=5, n_informative=3, random_state=12)
    ids = [f"s{i}" for i in range(45)]
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    frame.insert(0, "sample_id", ids)
    frame["label"] = y
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)
    loaded = read_table(data_path, separator=",")
    dataset = DatasetBundle(
        loaded.select([f"f{i}" for i in range(5)]),
        loaded["label"].to_numpy(),
        sample_ids=loaded["sample_id"].to_list(),
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=np.arange(45) % 3,
        dataset=dataset,
    )
    part_path = tmp_path / "folds.json"
    part_path.write_text(json.dumps(plan.to_dict(), indent=2))
    return dataset, plan, data_path, part_path


def test_config_relative_paths_do_not_depend_on_current_working_directory(tmp_path, monkeypatch):
    case = tmp_path / "case"
    case.mkdir()
    _, _, data_path, part_path = _write_classification_case(case)
    config_path = case / "validate.yaml"
    config_path.write_text(
        f"""schema_version: "1.0"\nworkflow: validate\ndataset:\n  path: {data_path.name}\n"""
        """  target: label\n  sample_id: sample_id\nalgorithm: logistic_regression\npartition:\n"""
        f"""  path: {part_path.name}\nmetrics: [accuracy]\n"""
    )
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    result = run_config(load_config(config_path))
    assert result.result.n_splits == 3


def test_yaml_train_artifact_predict_roundtrip(tmp_path):
    X, y = make_classification(n_samples=40, n_features=4, random_state=4)
    ids = [f"s{i}" for i in range(40)]
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    train_frame = frame.copy()
    train_frame.insert(0, "sample_id", ids)
    train_frame["label"] = y
    train_frame.to_csv(tmp_path / "train.csv", index=False)

    train_config = tmp_path / "train.yaml"
    train_config.write_text(
        """schema_version: "1.0"\nworkflow: train\ndataset:\n  path: train.csv\n  target: label\n"""
        """  sample_id: sample_id\nalgorithm: logistic_regression\nrandom_state: 42\nartifact:\n"""
        """  path: artifact\n"""
    )
    assert main(["run", str(train_config)]) == EXIT_OK

    pred_frame = frame.copy()
    pred_frame.insert(0, "sample_id", ids)
    pred_frame.to_csv(tmp_path / "predict.csv", index=False)
    predict_config = tmp_path / "predict.yaml"
    predict_config.write_text(
        """schema_version: "1.0"\nworkflow: predict\nartifact: artifact\ndataset:\n  path: predict.csv\n"""
        """  sample_id: sample_id\noutput:\n  path: predictions.csv\n"""
    )
    assert main(["run", str(predict_config)]) == EXIT_OK
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert len(predictions) == 40
    assert predictions["sample_id"].tolist() == ids


def test_regression_yaml_evaluate_artifact_roundtrip(tmp_path):
    X, y = make_regression(  # pyrefly: ignore[bad-unpacking]
        n_samples=45,
        n_features=4,
        noise=0.2,
        random_state=7,
    )
    ids = [f"r{i}" for i in range(45)]
    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(4)])
    dataset = DatasetBundle(frame, y, sample_ids=ids)
    trained = saber.train(dataset=dataset, algorithm="ridge_regressor")
    saber.save_model(tmp_path / "artifact", trained, dataset=dataset)

    eval_frame = frame.copy()
    eval_frame.insert(0, "sample_id", ids)
    eval_frame["target"] = y
    eval_frame.to_csv(tmp_path / "eval.csv", index=False)
    config_path = tmp_path / "evaluate.yaml"
    config_path.write_text(
        """schema_version: "1.0"\nworkflow: evaluate\nartifact: artifact\ndataset:\n  path: eval.csv\n"""
        """  target: target\n  sample_id: sample_id\nmetrics: [rmse, mae]\n"""
    )
    result = run_config(load_config(config_path))
    assert set(result.result.metrics) == {"rmse", "mae"}
    assert result.result.metrics["rmse"] >= 0.0
