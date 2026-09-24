from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_classification, make_regression

import saber
from saber.config import CONFIG_SCHEMA_VERSION, dump_config, load_config, run_config
from saber.config.builders import load_dataset, load_prediction_frame
from saber.datasets import DatasetBundle, PartitionPlan
from saber.exceptions import ConfigurationError
from saber.utils.tabular import read_table


def _write_classification_inputs(tmp_path):
    X, y = make_classification(n_samples=60, n_features=5, n_informative=3, random_state=11)
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    frame.insert(0, "sample_id", [f"s{i}" for i in range(60)])
    frame["label"] = y
    data_path = tmp_path / "data.csv"
    frame.to_csv(data_path, index=False)

    loaded = read_table(data_path, separator=",")
    bundle = DatasetBundle(
        loaded.select([f"f{i}" for i in range(5)]),
        loaded["label"].to_numpy(),
        sample_ids=loaded["sample_id"].to_list(),
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=bundle.sample_ids,
        fold_assignments=[i % 3 for i in range(60)],
        dataset_fingerprint=bundle.fingerprint,
    )
    partition_path = tmp_path / "folds.json"
    partition_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    return data_path, partition_path, bundle, plan


def test_yaml_validation_workflow_matches_direct_python_api(tmp_path):
    _, _, bundle, plan = _write_classification_inputs(tmp_path)
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": CONFIG_SCHEMA_VERSION,
                "workflow": "validate",
                "dataset": {
                    "path": "data.csv",
                    "target": "label",
                    "sample_id": "sample_id",
                },
                "algorithm": "logistic_regression",
                "partition": {"path": "folds.json"},
                "preprocessing": {"scaler": "auto", "imputation": "auto"},
                "metrics": ["accuracy", "balanced_accuracy"],
                "random_state": 42,
                "output": {"directory": "results"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    execution = run_config(config_path)
    direct = saber.validate(
        dataset=bundle,
        algorithm="logistic_regression",
        partition_plan=plan,
        metrics=("accuracy", "balanced_accuracy"),
        random_state=42,
    )
    assert execution.config.workflow == "validate"
    assert execution.result.aggregate_metrics == direct.aggregate_metrics
    assert (tmp_path / "results" / "summary.json").exists()
    assert (tmp_path / "results" / "metrics.csv").exists()
    assert (tmp_path / "results" / "predictions.csv").exists()


def test_config_normalization_round_trip(tmp_path):
    config = load_config(
        {
            "workflow": "train",
            "dataset": {"path": "data.csv", "target": "y"},
            "algorithm": "ridge_regressor",
        }
    )
    path = dump_config(config, tmp_path / "normalized.yaml")
    reloaded = load_config(path)
    assert reloaded.to_dict() == config.to_dict()
    assert reloaded.schema_version == CONFIG_SCHEMA_VERSION


def test_config_rejects_unknown_top_level_keys():
    with pytest.raises(ConfigurationError, match="Unknown keys"):
        load_config(
            {
                "workflow": "train",
                "dataset": {"path": "data.csv", "target": "y"},
                "algorithm": "ridge_regressor",
                "mystery": 1,
            }
        )


def test_validation_config_requires_partition_or_biosieve():
    with pytest.raises(ConfigurationError, match="requires either"):
        load_config(
            {
                "workflow": "validate",
                "dataset": {"path": "data.csv", "target": "y"},
                "algorithm": "ridge_regressor",
            }
        )


def test_tuning_yaml_executes_typed_search_space(tmp_path):
    _write_classification_inputs(tmp_path)
    config = {
        "workflow": "tune",
        "dataset": {"path": str(tmp_path / "data.csv"), "target": "label", "sample_id": "sample_id"},
        "algorithm": "logistic_regression",
        "partition": {"path": str(tmp_path / "folds.json")},
        "tuning": {
            "optimizer": "grid",
            "metrics": ["accuracy"],
            "refit_metric": "accuracy",
            "random_state": 42,
        },
        "search_space": {
            "C": {"type": "categorical", "values": [0.1, 1.0]},
        },
    }
    execution = run_config(config)
    assert execution.result.best_params["C"] in {0.1, 1.0}
    assert execution.summary["workflow"] == "tune"


def test_train_yaml_artifact_then_predict_yaml_round_trip(tmp_path):
    X, y = make_regression(n_samples=40, n_features=4, random_state=19)
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame.insert(0, "sample_id", [f"r{i}" for i in range(40)])
    frame["target"] = y
    frame.to_csv(tmp_path / "regression.csv", index=False)

    train_execution = run_config(
        {
            "workflow": "train",
            "dataset": {
                "path": str(tmp_path / "regression.csv"),
                "target": "target",
                "sample_id": "sample_id",
            },
            "algorithm": "ridge_regressor",
            "artifact": {"path": str(tmp_path / "model"), "overwrite": True},
        }
    )
    assert train_execution.outputs["artifact"].endswith("model")

    predict_execution = run_config(
        {
            "workflow": "predict",
            "dataset": {
                "path": str(tmp_path / "regression.csv"),
                "target": "target",
                "sample_id": "sample_id",
            },
            "artifact": str(tmp_path / "model"),
            "output": {"path": str(tmp_path / "predictions.csv")},
        }
    )
    assert predict_execution.result.n_samples == 40
    assert (tmp_path / "predictions.csv").exists()


def test_benchmark_yaml_runs_same_public_engine(tmp_path):
    _, _, bundle, plan = _write_classification_inputs(tmp_path)
    config = {
        "workflow": "benchmark",
        "datasets": {
            "rep_a": {
                "path": str(tmp_path / "data.csv"),
                "target": "label",
                "sample_id": "sample_id",
            }
        },
        "algorithms": ["logistic_regression"],
        "partitions": {"cv": {"path": str(tmp_path / "folds.json")}},
        "benchmark": {
            "metrics": ["accuracy"],
            "seeds": [42],
            "modes": ["untuned"],
            "include_baselines": False,
        },
    }
    execution = run_config(config)
    direct = saber.benchmark(
        datasets={"rep_a": bundle},
        algorithms=("logistic_regression",),
        config=__import__("saber.benchmark", fromlist=["BenchmarkConfig"]).BenchmarkConfig(
            metrics=("accuracy",), seeds=(42,), modes=("untuned",), include_baselines=False
        ),
        partitions={"cv": plan},
    )
    assert execution.result.n_runs == direct.n_runs == 1
    assert (
        execution.result.aggregate_metrics_frame()["score"].to_list()
        == direct.aggregate_metrics_frame()["score"].to_list()
    )


def test_config_validate_rejects_unknown_nested_keys():
    with pytest.raises(ConfigurationError, match="Unknown dataset keys"):
        load_config(
            {
                "workflow": "train",
                "dataset": {"path": "data.csv", "target": "y", "magic": True},
                "algorithm": "ridge_regressor",
            }
        )


def test_evaluate_yaml_uses_persisted_artifact(tmp_path):
    X, y = make_regression(n_samples=36, n_features=3, random_state=31)
    frame = pd.DataFrame(X, columns=["x0", "x1", "x2"])
    frame["target"] = y
    frame.to_csv(tmp_path / "eval.csv", index=False)

    run_config(
        {
            "workflow": "train",
            "dataset": {"path": str(tmp_path / "eval.csv"), "target": "target"},
            "algorithm": "ridge_regressor",
            "artifact": {"path": str(tmp_path / "eval_model"), "overwrite": True},
        }
    )
    execution = run_config(
        {
            "workflow": "evaluate",
            "dataset": {"path": str(tmp_path / "eval.csv"), "target": "target"},
            "artifact": str(tmp_path / "eval_model"),
            "metrics": ["rmse", "mae"],
        }
    )
    assert execution.summary["workflow"] == "evaluate"
    assert set(execution.result.metrics) == {"rmse", "mae"}


def test_csv_and_tsv_dataset_loading_give_same_fingerprint(tmp_path):
    rows = [
        "sample_id,target,group,weight,f0,f1",
        "s0,0,g1,1.0,0.5,NA",
        "s1,1,g1,2.0,1.5,3.0",
        "s2,0,g2,1.0,2.5,4.0",
    ]
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    tsv_path = tmp_path / "data.tsv"
    tsv_path.write_text("\n".join(line.replace(",", "\t") for line in rows) + "\n", encoding="utf-8")

    def _bundle(path):
        config = load_config(
            {
                "workflow": "train",
                "dataset": {
                    "path": str(path),
                    "target": "target",
                    "sample_id": "sample_id",
                    "groups": "group",
                    "sample_weight": "weight",
                },
                "algorithm": "ridge_regressor",
            }
        )
        return load_dataset(config, config.payload["dataset"])

    csv_bundle = _bundle(csv_path)
    tsv_bundle = _bundle(tsv_path)

    assert csv_bundle.fingerprint == tsv_bundle.fingerprint
    assert csv_bundle.feature_names == ("f0", "f1")
    assert np.isnan(csv_bundle.X["f1"].to_numpy()[0])
    assert np.isnan(tsv_bundle.X["f1"].to_numpy()[0])


def test_dataset_sep_rejects_multi_character_separator(tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("sample_id::target::f0\ns0::0::1.0\n", encoding="utf-8")
    config = load_config(
        {
            "workflow": "train",
            "dataset": {
                "path": str(data_path),
                "target": "target",
                "sample_id": "sample_id",
                "sep": "::",
            },
            "algorithm": "ridge_regressor",
        }
    )
    with pytest.raises(ConfigurationError, match="single character"):
        load_dataset(config, config.payload["dataset"])


def test_prediction_frame_sep_rejects_multi_character_separator(tmp_path):
    data_path = tmp_path / "data.csv"
    data_path.write_text("sample_id::f0\ns0::1.0\n", encoding="utf-8")
    config = load_config(
        {
            "workflow": "predict",
            "dataset": {"path": str(data_path), "sample_id": "sample_id", "sep": "::"},
            "artifact": str(tmp_path / "model"),
        }
    )
    with pytest.raises(ConfigurationError, match="single character"):
        load_prediction_frame(config, config.payload["dataset"])


def test_benchmark_config_requires_partitions_or_biosieve():
    with pytest.raises(ConfigurationError, match="requires either 'partitions'"):
        load_config(
            {
                "workflow": "benchmark",
                "dataset": {"path": "data.csv", "target": "y"},
                "algorithms": ["ridge_regressor"],
                "benchmark": {"metrics": ["rmse"]},
            }
        )
