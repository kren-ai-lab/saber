"""Shared ecosystem CLI contract: ``validate-config``, exit codes, record ids and provenance sidecars."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from jsonschema import Draft202012Validator
from sklearn.datasets import make_classification

from saber.cli.main import EXIT_CONFIG, EXIT_OK, EXIT_WORKFLOW, main

SCHEMAS = Path(__file__).parent


def _schema(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((SCHEMAS / name).read_text(encoding="utf-8")))


def _write(tmp_path: Path, config: dict, id_col: str = "id") -> Path:
    X, y = make_classification(n_samples=30, n_features=3, n_informative=2, n_redundant=0, random_state=1)
    frame = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    frame.insert(0, id_col, [f"s{i}" for i in range(30)])
    frame["label"] = y
    frame.to_csv(tmp_path / "data.csv", index=False)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


TRAIN = {
    "workflow": "train",
    "dataset": {"path": "data.csv", "target_col": "label"},
    "algorithm": "logistic_regression",
    "artifact": "model",
    "output": "out",
}


def test_validate_config_reports_every_problem_as_json(tmp_path, capsys):
    bad = {
        **TRAIN,
        "dataset": {"path": "data.csv", "bogus": 1},
        "mystery": 1,
        "metrics": 3,
        "algorithm": "nope",
    }
    path = _write(tmp_path, bad)
    assert main(["validate-config", str(path), "--json"]) == EXIT_CONFIG
    errors = json.loads(capsys.readouterr().out)
    assert list(_schema("validate-errors.schema.json").iter_errors(errors)) == []
    assert {"mystery", "dataset.bogus", "algorithm"} <= {item["path"] for item in errors}

    good = _write(tmp_path, TRAIN)
    assert main(["validate-config", str(good), "--json"]) == EXIT_OK
    assert json.loads(capsys.readouterr().out) == []


def test_run_writes_provenance_sidecars_for_output_and_artifact(tmp_path):
    path = _write(tmp_path, TRAIN)
    assert main(["run", str(path)]) == EXIT_OK
    validator = _schema("provenance.schema.json")
    for name in ("out", "model"):
        document = json.loads((tmp_path / f"{name}.provenance.json").read_text(encoding="utf-8"))
        assert list(validator.iter_errors(document)) == []
        assert document["command"] == "train"
        assert document["extensions"]["algorithm"] == "logistic_regression"
        assert document["extensions"]["dataset_fingerprint"]
        assert [Path(item["path"]).name for item in document["inputs"]] == ["data.csv"]
        assert "path" not in document["parameters"]["dataset"]
        assert "output" not in document["parameters"]


def test_missing_id_column_is_a_runtime_error(tmp_path):
    path = _write(tmp_path, TRAIN, id_col="name")
    assert main(["run", str(path)]) == EXIT_WORKFLOW
    assert not (tmp_path / "out.provenance.json").exists()
