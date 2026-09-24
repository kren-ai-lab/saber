from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import yaml
from sklearn.datasets import make_classification

from saber import train
from saber.cli.main import EXIT_CONFIG, EXIT_OK, main
from saber.config import run_config
from saber.datasets import DatasetBundle, PartitionPlan
from saber.persistence import save_model_artifact
from saber.utils.tabular import read_table


def _workflow_files(tmp_path):
    X, y = make_classification(n_samples=45, n_features=4, n_informative=3, n_redundant=0, random_state=5)
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    frame.insert(0, "sample_id", [f"s{i}" for i in range(45)])
    frame["label"] = y
    frame.to_csv(tmp_path / "data.csv", index=False)
    loaded = read_table(tmp_path / "data.csv", separator=",")
    bundle = DatasetBundle(
        loaded.select(["f0", "f1", "f2", "f3"]),
        loaded["label"].to_numpy(),
        sample_ids=loaded["sample_id"].to_list(),
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=bundle.sample_ids,
        fold_assignments=[i % 3 for i in range(45)],
        dataset_fingerprint=bundle.fingerprint,
    )
    (tmp_path / "folds.json").write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    config = {
        "workflow": "validate",
        "dataset": {"path": "data.csv", "target": "label", "sample_id": "sample_id"},
        "algorithm": "logistic_regression",
        "partition": {"path": "folds.json"},
        "metrics": ["accuracy"],
        "random_state": 42,
        "output": {"directory": "cli_results"},
    }
    path = tmp_path / "validate.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, bundle


def test_cli_models_discovery_commands():
    assert main(["models", "list", "--task", "classification", "--json"]) == EXIT_OK
    assert main(["models", "show", "logistic_regression", "--json"]) == EXIT_OK


def test_cli_config_validation_and_wrong_workflow(tmp_path):
    path, _ = _workflow_files(tmp_path)
    assert main(["config", "validate", str(path)]) == EXIT_OK
    assert main(["train", str(path)]) == EXIT_CONFIG


def test_cli_train_rejects_multi_character_dataset_separator(tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("sample_id::target::f0\ns0::0::1.0\ns1::1::2.0\n", encoding="utf-8")
    config = {
        "workflow": "train",
        "dataset": {"path": "data.csv", "target": "target", "sample_id": "sample_id", "sep": "::"},
        "algorithm": "ridge_regressor",
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assert main(["train", str(path)]) == EXIT_CONFIG


def test_cli_validate_uses_same_config_runner(tmp_path):
    path, _ = _workflow_files(tmp_path)
    direct = run_config(path)
    assert main(["validate", str(path)]) == EXIT_OK
    summary = json.loads((tmp_path / "cli_results" / "summary.json").read_text())
    assert summary["aggregate_metrics"] == direct.summary["aggregate_metrics"]


def test_python_module_entry_point(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "saber", "--version"],
        cwd=str(tmp_path),
        env={**__import__("os").environ, "PYTHONPATH": str(__import__("pathlib").Path(__file__).parents[2])},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "saber 0.1.0" in completed.stdout


def test_cli_artifact_inspection_and_verification(tmp_path):
    _, bundle = _workflow_files(tmp_path)
    trained = train(dataset=bundle, algorithm="logistic_regression", random_state=42)
    artifact = tmp_path / "artifact"
    save_model_artifact(
        artifact,
        model=trained.model,
        algorithm=trained.spec.name,
        task=trained.spec.task,
        dataset=bundle,
        provider=trained.spec.provider,
    )
    assert main(["artifact", "verify", str(artifact)]) == EXIT_OK
    assert main(["artifact", "inspect", str(artifact), "--json"]) == EXIT_OK


def test_cli_optimize_is_thin_alias_for_tune_config(tmp_path):
    path, _ = _workflow_files(tmp_path)
    payload = yaml.safe_load(path.read_text())
    payload["workflow"] = "tune"
    payload["tuning"] = {
        "optimizer": "grid",
        "metrics": ["accuracy"],
        "refit_metric": "accuracy",
    }
    payload["search_space"] = {"C": [0.1, 1.0]}
    payload.pop("metrics", None)
    payload.pop("random_state", None)
    payload.pop("output", None)
    tune_path = tmp_path / "tune.yaml"
    tune_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    assert main(["optimize", str(tune_path)]) == EXIT_OK
