from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_classification

import saber
from saber.cli.main import EXIT_CONFIG, EXIT_OK, main
from saber.datasets import DatasetBundle, PartitionPlan
from saber.utils.tabular import read_table

# saber.cli re-exports the main() function under the module name, so import the module explicitly.
cli_main = importlib.import_module("saber.cli.main")


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

    loaded = read_table(data_path, separator=",")
    dataset = DatasetBundle(
        loaded.select([f"f{i}" for i in range(5)]),
        loaded["label"].to_numpy(),
        sample_ids=loaded["sample_id"].to_list(),
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=[i % 3 for i in range(dataset.n_samples)],
        dataset=dataset,
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


def _benchmark_config(tmp_path: Path) -> Path:
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

    loaded = read_table(data_path, separator=",")
    dataset = DatasetBundle(
        loaded.select([f"f{i}" for i in range(5)]),
        loaded["label"].to_numpy(),
        sample_ids=loaded["sample_id"].to_list(),
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=[i % 3 for i in range(dataset.n_samples)],
        dataset=dataset,
    )
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

    config = {
        "workflow": "benchmark",
        "datasets": {"rep_a": {"path": "data.csv", "target": "label", "sample_id": "sample_id"}},
        "algorithms": ["logistic_regression"],
        "partitions": {"cv": {"path": "folds.json"}},
        "benchmark": {
            "metrics": ["accuracy"],
            "seeds": [42],
            "modes": ["untuned"],
            "include_baselines": False,
        },
    }
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_benchmark_human_output_renders_metric_table(tmp_path, capsys):
    config = _benchmark_config(tmp_path)
    assert main(["run", str(config)]) == EXIT_OK
    output = capsys.readouterr().out
    assert "Aggregate results (preview)" in output
    assert "logistic_regression" in output


def test_dry_run_validates_without_creating_outputs(tmp_path, capsys, monkeypatch):
    config = _validate_config(tmp_path)
    monkeypatch.setattr(cli_main, "run_config", _forbidden_run_config)
    assert main(["run", str(config), "--dry-run"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "execution plan" in output.lower()
    assert "No workflow was executed" in output
    assert not (tmp_path / "results").exists()


def _forbidden_run_config(*_args, **_kwargs):
    raise AssertionError("dry-run must not execute the workflow")


def test_workflow_json_mode_is_machine_readable(tmp_path, capsys):
    config = _validate_config(tmp_path)
    assert main(["run", str(config), "--json"]) == EXIT_OK
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["summary"]["workflow"] == "validate"
    assert payload["summary"]["algorithm"] == "logistic_regression"
    assert payload["summary"]["n_splits"] == 3
    assert "metrics" in payload["outputs"]


def test_models_list_filters(capsys):
    assert main(["models", "list", "--task", "classification", "--json"]) == EXIT_OK
    rows = json.loads(capsys.readouterr().out)
    assert rows
    assert all(row["task"] == "classification" for row in rows)

    assert main(["models", "list", "--tag", "baseline", "--json"]) == EXIT_OK
    baselines = json.loads(capsys.readouterr().out)
    assert {row["name"] for row in baselines} >= {"dummy_classifier", "dummy_regressor"}


def test_model_show_human_output_includes_capabilities(capsys):
    assert main(["models", "show", "logistic_regression"]) == EXIT_OK
    output = capsys.readouterr().out
    assert "Capabilities & requirements" in output
    assert "sample_weight" in output


COMMANDS = ("run", "models", "artifact")


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "COLUMNS": "200", "NO_COLOR": "1"}
    env.pop("GITHUB_ACTIONS", None)
    env.pop("FORCE_COLOR", None)
    env.pop("PY_COLORS", None)
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "saber", *args], text=True, capture_output=True, env=env, check=False
    )


def test_top_level_help_lists_every_command():
    completed = _cli("--help")
    assert completed.returncode == 0
    for command in COMMANDS:
        assert command in completed.stdout


def test_run_help_lists_options():
    completed = _cli("run", "--help")
    assert completed.returncode == 0
    for option in ("--dry-run", "--json"):
        assert option in completed.stdout


def test_invalid_command_is_a_configuration_exit_code():
    assert main(["definitely-not-a-command"]) == EXIT_CONFIG


def test_version_flag():
    completed = _cli("--version")
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"saber {saber.__version__}"


def test_benchmark_preview_lists_nan_scores_last():
    from types import SimpleNamespace

    import polars as pl
    from rich.console import Console

    from saber.cli.render import _render_benchmark

    frame = pl.DataFrame(
        {
            "representation": ["r"] * 3,
            "partition": ["p"] * 3,
            "algorithm": ["nan_model", "low_model", "high_model"],
            "mode": ["untuned"] * 3,
            "seed": [0] * 3,
            "metric": ["accuracy"] * 3,
            "score": [float("nan"), 0.5, 0.9],
        }
    )
    result = SimpleNamespace(n_runs=3, successes=(), failures=(), aggregate_metrics_frame=lambda: frame)
    console = Console(record=True, width=200)
    _render_benchmark(console, result)  # pyrefly: ignore[bad-argument-type] - duck-typed stand-in for BenchmarkResult
    text = console.export_text()
    assert text.index("high_model") < text.index("low_model") < text.index("nan_model")
