from __future__ import annotations

import json

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


def test_cli_run_uses_same_config_runner(tmp_path):
    path, _ = _workflow_files(tmp_path)
    direct = run_config(path)
    assert main(["run", str(path)]) == EXIT_OK
    summary = json.loads((tmp_path / "cli_results" / "summary.json").read_text())
    assert summary["aggregate_metrics"] == direct.summary["aggregate_metrics"]
    assert summary["workflow"] == "validate"
    assert summary["n_splits"] == 3


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


def test_cli_error_exit_codes(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("workflow: nope\n", encoding="utf-8")
    assert main(["run", str(bad)]) == EXIT_CONFIG
    assert main(["artifact", "verify", str(tmp_path / "missing")]) != EXIT_OK
    assert main(["models", "show", "no_such_model"]) != EXIT_OK
