from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_classification

from saber.cli.main import EXIT_CONFIG, EXIT_OK, _doctor_payload, main
from saber.datasets import DatasetBundle, PartitionPlan

ROOT = Path(__file__).parents[2]


def _validate_config(tmp_path: Path) -> Path:
    X, y = make_classification(
        n_samples=48,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=11,
    )
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    frame.insert(0, "sample_id", [f"s{i}" for i in range(len(frame))])
    frame["label"] = y
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)

    loaded = pd.read_csv(data_path)
    dataset = DatasetBundle(
        loaded[[f"f{i}" for i in range(5)]],
        loaded["label"].to_numpy(),
        sample_ids=loaded["sample_id"].tolist(),
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=[i % 3 for i in range(dataset.n_samples)],
        dataset_fingerprint=dataset.fingerprint,
    )
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

    config = {
        "workflow": "validate",
        "dataset": {"path": "data.csv", "target": "label", "sample_id": "sample_id"},
        "algorithm": "logistic_regression",
        "partition": {"path": "folds.json"},
        "metrics": ["accuracy", "balanced_accuracy", "mcc"],
        "random_state": 42,
        "output": {"directory": "results"},
    }
    config_path = tmp_path / "validate.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_dry_run_validates_without_creating_outputs(tmp_path, capsys):
    config = _validate_config(tmp_path)
    assert main(["validate", str(config), "--dry-run"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "execution plan" in output.lower()
    assert "No workflow was executed" in output
    assert not (tmp_path / "results").exists()


def test_workflow_json_mode_is_machine_readable(tmp_path, capsys):
    config = _validate_config(tmp_path)
    assert main(["validate", str(config), "--json"]) == EXIT_OK
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["summary"]["workflow"] == "validate"
    assert payload["summary"]["algorithm"] == "logistic_regression"
    assert payload["summary"]["n_splits"] == 3
    assert "metrics" in payload["outputs"]


def test_quiet_mode_executes_without_normal_output(tmp_path, capsys):
    config = _validate_config(tmp_path)
    assert main(["validate", str(config), "--quiet"]) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (tmp_path / "results" / "summary.json").exists()


def test_config_show_renders_execution_plan(tmp_path, capsys):
    config = _validate_config(tmp_path)
    assert main(["config", "show", str(config)]) == EXIT_OK
    output = capsys.readouterr().out
    assert "Workflow" in output
    assert "logistic_regression" in output
    assert "accuracy" in output


def test_models_search_and_tag_filter(capsys):
    assert main(["models", "search", "forest", "--task", "classification", "--json"]) == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    assert rows
    assert all(row["task"] == "classification" for row in rows)
    assert any("forest" in row["name"] for row in rows)

    assert main(["models", "list", "--tag", "baseline", "--json"]) == EXIT_OK
    baselines = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in baselines} >= {"dummy_classifier", "dummy_regressor"}


def test_model_show_human_output_includes_capabilities(capsys):
    assert main(["models", "show", "logistic_regression"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "Capabilities & requirements" in output
    assert "sample_weight" in output
    assert "Scaling" not in output or "recommended" in output


def test_doctor_payload_and_cli(capsys):
    payload = _doctor_payload()
    names = {item["name"] for item in payload["components"]}
    assert {"saber", "python", "scikit-learn", "biosieve", "xgboost", "lightgbm", "optuna"} <= names

    assert main(["doctor", "--json"]) == EXIT_OK
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["components"][0]["name"] == "saber"


def test_wrong_workflow_still_uses_configuration_exit_code(tmp_path):
    config = _validate_config(tmp_path)
    assert main(["train", str(config), "--dry-run"]) == EXIT_CONFIG


def test_module_help_contains_cli2_commands(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "saber", "--help"],
        cwd=tmp_path,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    for token in ("doctor", "models", "artifact", "config", "benchmark"):
        assert token in completed.stdout
    assert "saber benchmark study.yaml --dry-run" in completed.stdout


def test_cli2_contains_no_scientific_engine_imports_or_splitters():
    source = (ROOT / "saber" / "cli" / "main.py").read_text(encoding="utf-8")
    renderer = (ROOT / "saber" / "cli" / "render.py").read_text(encoding="utf-8")
    combined = source + renderer
    forbidden = (
        "train_test_split",
        "StratifiedKFold",
        "GroupKFold",
        "GridSearchCV",
        "RandomizedSearchCV",
        ".fit(",
    )
    for token in forbidden:
        assert token not in combined


def test_top_level_help_uses_rich_cli2_layout(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "saber", "--help"],
        cwd=tmp_path,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "saber  CLI" in completed.stdout
    assert "Commands" in completed.stdout
    assert "Options" in completed.stdout
    assert "Examples" in completed.stdout
    assert "Run saber COMMAND --help" in completed.stdout
    assert "usage:" not in completed.stdout.lower()


def test_subcommand_help_uses_rich_cli2_layout(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "saber", "benchmark", "--help"],
        cwd=tmp_path,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "saber benchmark  CLI" in completed.stdout
    assert "Arguments" in completed.stdout
    assert "Options" in completed.stdout
    assert "--dry-run" in completed.stdout
    assert "--json" in completed.stdout
    assert "Examples" in completed.stdout
    assert "usage:" not in completed.stdout.lower()


def test_invalid_command_uses_rich_usage_error(tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", "saber", "definitely-not-a-command"],
        cwd=tmp_path,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "CLI usage error" in completed.stderr
    assert "invalid choice" in completed.stderr
    assert "Run saber --help for details" in completed.stderr
    assert "usage:" not in completed.stderr.lower()
