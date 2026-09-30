from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml
from sklearn.datasets import make_classification, make_regression

import saber
from saber.config import CONFIG_SCHEMA_VERSION, dump_config, load_config, run_config
from saber.config.builders import load_dataset
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
        dataset=bundle,
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
                    "target_col": "label",
                    "sample_id_col": "sample_id",
                },
                "algorithm": "logistic_regression",
                "partition_plan": {"path": "folds.json"},
                "preprocessing": {"scaler": "auto", "imputation": "auto"},
                "metrics": ["accuracy", "balanced_accuracy"],
                "random_state": 42,
                "output": "results",
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
    assert execution.result.aggregate_metrics == pytest.approx(direct.aggregate_metrics)
    assert direct.oof_prediction is not None
    assert execution.result.oof_prediction.predictions.tolist() == direct.oof_prediction.predictions.tolist()
    assert (tmp_path / "results" / "summary.json").exists()
    assert (tmp_path / "results" / "metrics.csv").exists()
    assert (tmp_path / "results" / "predictions.csv").exists()


def test_config_normalization_round_trip(tmp_path):
    config = load_config(
        {
            "workflow": "train",
            "dataset": {"path": "data.csv", "target_col": "y"},
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
                "dataset": {"path": "data.csv", "target_col": "y"},
                "algorithm": "ridge_regressor",
                "mystery": 1,
            }
        )


def test_validation_config_requires_partition_or_biosieve():
    with pytest.raises(
        ConfigurationError, match="requires exactly one of 'partition_plan' or 'partitioning'"
    ):
        load_config(
            {
                "workflow": "validate",
                "dataset": {"path": "data.csv", "target_col": "y"},
                "algorithm": "ridge_regressor",
            }
        )


def test_tuning_yaml_executes_typed_search_space_and_writes_history_csv(tmp_path):
    _write_classification_inputs(tmp_path)
    config = {
        "workflow": "tune",
        "dataset": {"path": str(tmp_path / "data.csv"), "target_col": "label", "sample_id_col": "sample_id"},
        "algorithm": "logistic_regression",
        "partition_plan": {"path": str(tmp_path / "folds.json")},
        "tuning": {"optimizer": "grid", "refit_metric": "accuracy"},
        "metrics": ["accuracy"],
        "random_state": 42,
        "search_space": {
            "C": {"type": "categorical", "values": [0.1, 1.0]},
        },
        "output": str(tmp_path / "results"),
    }
    execution = run_config(config)
    assert execution.result.best_params["C"] in {0.1, 1.0}
    assert execution.summary["workflow"] == "tune"
    history_path = tmp_path / "results" / "optimization_history.csv"
    assert history_path.exists()
    assert not read_table(history_path, separator=",").is_empty()
    assert execution.outputs["optimization_history"] == str(history_path)


def test_train_yaml_artifact_then_predict_yaml_round_trip(tmp_path):
    X, y = make_regression(n_samples=40, n_features=4, random_state=19)  # pyrefly: ignore[bad-unpacking]
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame.insert(0, "sample_id", [f"r{i}" for i in range(40)])
    frame["target"] = y
    frame.to_csv(tmp_path / "regression.csv", index=False)

    train_execution = run_config(
        {
            "workflow": "train",
            "dataset": {
                "path": str(tmp_path / "regression.csv"),
                "target_col": "target",
                "sample_id_col": "sample_id",
            },
            "algorithm": "ridge_regressor",
            "artifact": str(tmp_path / "model"),
        }
    )
    assert train_execution.outputs["artifact"].endswith("model")

    predict_execution = run_config(
        {
            "workflow": "predict",
            "dataset": {
                "path": str(tmp_path / "regression.csv"),
                "target_col": "target",
                "sample_id_col": "sample_id",
            },
            "model": str(tmp_path / "model"),
            "output": str(tmp_path / "out"),
        }
    )
    assert predict_execution.result.n_samples == 40
    assert (tmp_path / "out" / "predictions.csv").exists()


def test_predict_output_directory_writes_predictions(tmp_path):
    X, y = make_regression(n_samples=40, n_features=4, random_state=19)  # pyrefly: ignore[bad-unpacking]
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    frame["target"] = y
    frame.to_csv(tmp_path / "r.csv", index=False)
    dataset = {"path": str(tmp_path / "r.csv"), "target_col": "target"}
    run_config(
        {
            "workflow": "train",
            "dataset": dataset,
            "algorithm": "ridge_regressor",
            "artifact": str(tmp_path / "model"),
        }
    )
    execution = run_config(
        {
            "workflow": "predict",
            "dataset": dataset,
            "model": str(tmp_path / "model"),
            "output": str(tmp_path / "out"),
        }
    )
    assert (tmp_path / "out" / "predictions.csv").exists()
    assert execution.outputs["predictions"] == str(tmp_path / "out" / "predictions.csv")


def test_benchmark_yaml_runs_same_public_engine(tmp_path):
    _, _, bundle, plan = _write_classification_inputs(tmp_path)
    config = {
        "workflow": "benchmark",
        "datasets": {
            "rep_a": {
                "path": str(tmp_path / "data.csv"),
                "target_col": "label",
                "sample_id_col": "sample_id",
            }
        },
        "algorithms": ["logistic_regression"],
        "partitions": {"cv": {"path": str(tmp_path / "folds.json")}},
        "metrics": ["accuracy"],
        "benchmark": {"seeds": [42], "modes": ["untuned"], "include_baselines": False},
        "output": str(tmp_path / "results"),
        "metadata": {"study": "config-test"},
    }
    execution = run_config(config)
    # The output directory is exactly the save_benchmark artifact (checksums verify).
    assert execution.outputs == {"benchmark": str(tmp_path / "results")}
    loaded = saber.load_benchmark(tmp_path / "results")
    assert loaded.metadata["metadata"] == {"study": "config-test"}
    for name in ("runs", "metrics", "predictions", "failures", "optimization_history"):
        read_table(tmp_path / "results" / f"{name}.csv", separator=",")  # must exist and parse
    assert not (tmp_path / "results" / "summary.json").exists()
    direct = saber.benchmark(
        datasets={"rep_a": bundle},
        algorithms=("logistic_regression",),
        config=__import__("saber.benchmark", fromlist=["BenchmarkConfig"]).BenchmarkConfig(
            seeds=(42,),
            modes=("untuned",),
            include_baselines=False,
        ),
        partitions={"cv": plan},
        metrics=("accuracy",),
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
                "dataset": {"path": "data.csv", "target_col": "y", "magic": True},
                "algorithm": "ridge_regressor",
            }
        )


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
                    "target_col": "target",
                    "sample_id_col": "sample_id",
                    "group_col": "group",
                    "sample_weight_col": "weight",
                },
                "algorithm": "ridge_regressor",
            }
        )
        return load_dataset(config, config.payload["dataset"])[0]

    csv_bundle = _bundle(csv_path)
    tsv_bundle = _bundle(tsv_path)

    assert csv_bundle.fingerprint == tsv_bundle.fingerprint
    assert csv_bundle.feature_names == ("f0", "f1")
    assert np.isnan(csv_bundle.X["f1"].to_numpy()[0])
    assert np.isnan(tsv_bundle.X["f1"].to_numpy()[0])


def test_dataset_sep_rejects_multi_character_separator():
    with pytest.raises(ConfigurationError, match="single character"):
        load_config(
            {
                "workflow": "train",
                "dataset": {
                    "path": "data.csv",
                    "target_col": "target",
                    "sample_id_col": "sample_id",
                    "sep": "::",
                },
                "algorithm": "ridge_regressor",
            }
        )


def test_benchmark_config_requires_partitions_or_biosieve():
    with pytest.raises(ConfigurationError, match="requires exactly one of 'partitions' or 'partitioning'"):
        load_config(
            {
                "workflow": "benchmark",
                "datasets": {"rep": {"path": "data.csv", "target_col": "y"}},
                "algorithms": ["ridge_regressor"],
                "metrics": ["rmse"],
            }
        )


def test_all_empty_feature_column_is_typed_float64_and_workflow_runs(tmp_path):
    rows = ["sample_id,f0,f1,label"]
    rows += [f"s{i},{float(i)},,{i % 2}" for i in range(20)]
    data_path = tmp_path / "data.csv"
    data_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    frame = read_table(data_path, separator=",")
    assert frame["f1"].dtype == pl.Float64
    assert frame["f1"].null_count() == frame.height

    with warnings.catch_warnings():
        # SimpleImputer warns that the all-null column has no observed values
        # to impute from, regardless of strategy; the workflow still runs.
        warnings.simplefilter("ignore")
        execution = run_config(
            {
                "workflow": "train",
                "dataset": {"path": str(data_path), "target_col": "label", "sample_id_col": "sample_id"},
                "algorithm": "logistic_regression",
                "preprocessing": {"imputation": "constant", "fill_value": 0.0},
            }
        )
    assert execution.result.model is not None


def test_schema_version_1_0_is_rejected_with_migration_hint():
    with pytest.raises(
        ConfigurationError, match=r"'1\.0' is no longer supported.*'2\.0'.*docs/configuration\.md"
    ):
        load_config(
            {
                "schema_version": "1.0",
                "workflow": "train",
                "dataset": {"path": "data.csv", "target_col": "y"},
                "algorithm": "ridge_regressor",
            }
        )


def _write_membership_partition(tmp_path, n_samples=60, *, holdout=None):
    """Write one train/validation/test split with non-default column names."""
    roles = ["train"] * 40 + ["validation"] * 10 + ["test"] * 10
    rows = ["id\tset\tfold_name"] + [f"s{i}\t{roles[i]}\tonly" for i in range(n_samples) if i != holdout]
    path = tmp_path / "membership.tsv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"path": str(path), "sample_id_col": "id", "role_col": "set", "split_col": "fold_name"}


def _validate_config(tmp_path, partition_plan, **extra):
    return {
        "workflow": "validate",
        "dataset": {"path": str(tmp_path / "data.csv"), "target_col": "label", "sample_id_col": "sample_id"},
        "algorithm": "logistic_regression",
        "partition_plan": partition_plan,
        "metrics": ["accuracy"],
        "random_state": 42,
        **extra,
    }


@pytest.mark.parametrize("role", ["validation", "test"])
def test_evaluation_role_selects_the_evaluated_membership(tmp_path, role):
    _write_classification_inputs(tmp_path)
    partition_plan = _write_membership_partition(tmp_path)
    execution = run_config(_validate_config(tmp_path, partition_plan, evaluation_role=role))
    (fold,) = execution.result.folds
    assert fold.evaluation_role == role
    assert len(fold.evaluation_ids) == 10
    assert fold.evaluation_ids[0] == ("s40" if role == "validation" else "s50")


def test_require_complete_controls_partial_partition_plans(tmp_path):
    _write_classification_inputs(tmp_path)
    partition_plan = _write_membership_partition(tmp_path, holdout=0)
    with pytest.raises(saber.exceptions.PartitionValidationError, match="does not cover all dataset samples"):
        run_config(_validate_config(tmp_path, partition_plan))
    execution = run_config(_validate_config(tmp_path, partition_plan, require_complete=False))
    assert "s0" not in execution.result.folds[0].train_ids


def _write_sequence_dataset(path, *, with_sequence=True, seed=3):
    X, y = make_classification(n_samples=40, n_features=4, n_informative=3, n_redundant=0, random_state=seed)
    frame = pl.DataFrame({f"f{i}": X[:, i] for i in range(4)}).with_columns(
        pl.Series("sample_id", [f"s{i}" for i in range(40)]),
        pl.Series("label", y),
    )
    if with_sequence:
        frame = frame.with_columns(pl.Series("sequence", [f"ACD{'K' * (i % 5)}" for i in range(40)]))
    frame.write_csv(path)
    return frame


def test_partitioning_extra_columns_are_passed_to_biosieve_and_not_features(tmp_path, monkeypatch):
    pytest.importorskip("biosieve")
    from saber.config import runner

    frame = _write_sequence_dataset(tmp_path / "data.csv")
    captured = {}
    real_validate = runner.validate

    def spy(**kwargs):
        captured.update(kwargs)
        return real_validate(**kwargs)

    monkeypatch.setattr(runner, "validate", spy)
    execution = run_config(
        {
            "workflow": "validate",
            "dataset": {
                "path": str(tmp_path / "data.csv"),
                "target_col": "label",
                "sample_id_col": "sample_id",
            },
            "algorithm": "logistic_regression",
            "partitioning": {
                "strategy": "random_kfold",
                "params": {"n_splits": 3, "seed": 1},
                "extra_columns": ["sequence"],
            },
            "metrics": ["accuracy"],
        }
    )
    assert captured["dataset"].feature_names == ("f0", "f1", "f2", "f3")
    assert captured["partition_plan"].extra_columns == {"sequence": frame["sequence"].to_list()}
    assert execution.result.n_splits == 3


def test_partitioning_role_columns_are_loaded_and_forwarded(tmp_path, monkeypatch):
    pytest.importorskip("biosieve")
    from saber.config import runner

    frame = _write_sequence_dataset(tmp_path / "data.csv")
    captured = {}
    real_validate = runner.validate

    def spy(**kwargs):
        captured.update(kwargs)
        return real_validate(**kwargs)

    monkeypatch.setattr(runner, "validate", spy)
    run_config(
        {
            "workflow": "validate",
            "dataset": {
                "path": str(tmp_path / "data.csv"),
                "target_col": "label",
                "sample_id_col": "sample_id",
            },
            "algorithm": "logistic_regression",
            "partitioning": {
                "strategy": "random_kfold",
                "params": {"n_splits": 3, "seed": 1},
                "seq_col": "sequence",
            },
            "metrics": ["accuracy"],
        }
    )
    plan = captured["partition_plan"]
    assert plan.seq_col == "sequence"
    assert plan.extra_columns == {"sequence": frame["sequence"].to_list()}
    assert "sequence" not in captured["dataset"].feature_names


def test_partitioning_extra_columns_must_exist_in_the_dataset(tmp_path):
    _write_sequence_dataset(tmp_path / "data.csv", with_sequence=False)
    config = {
        "workflow": "validate",
        "dataset": {"path": str(tmp_path / "data.csv"), "target_col": "label", "sample_id_col": "sample_id"},
        "algorithm": "logistic_regression",
        "partitioning": {"strategy": "random_kfold", "extra_columns": ["sequence"]},
    }
    with pytest.raises(ConfigurationError, match=r"missing partitioning.extra_columns: \['sequence'\]"):
        run_config(config)


def test_partitioning_reference_generates_partitions_from_the_named_dataset(tmp_path, monkeypatch):
    from saber.config import runner

    _write_sequence_dataset(tmp_path / "a.csv", with_sequence=False)
    reference = _write_sequence_dataset(tmp_path / "b.csv")
    captured = {}
    monkeypatch.setattr(runner, "benchmark", lambda **kwargs: captured.update(kwargs) or _FakeBenchmark())
    dataset = {"target_col": "label", "sample_id_col": "sample_id"}
    run_config(
        {
            "workflow": "benchmark",
            "datasets": {
                "rep_a": {"path": str(tmp_path / "a.csv"), **dataset},
                "rep_b": {"path": str(tmp_path / "b.csv"), **dataset},
            },
            "algorithms": ["logistic_regression"],
            "partitioning": {"strategy": "random_kfold", "extra_columns": ["sequence"]},
            "partitioning_reference": "rep_b",
            "metrics": ["accuracy"],
        }
    )
    assert list(captured["datasets"]) == ["rep_b", "rep_a"]
    assert captured["datasets"]["rep_b"].feature_names == ("f0", "f1", "f2", "f3")
    assert captured["partitions"].extra_columns == {"sequence": reference["sequence"].to_list()}

    with pytest.raises(ConfigurationError, match="partitioning_reference 'rep_c' is not a 'datasets' label"):
        load_config(
            {
                "workflow": "benchmark",
                "datasets": {"rep_a": {"path": "a.csv", **dataset}},
                "algorithms": ["logistic_regression"],
                "partitioning": {"strategy": "random_kfold"},
                "partitioning_reference": "rep_c",
                "metrics": ["accuracy"],
            }
        )


class _FakeBenchmark:
    n_runs = 0
    successes = ()
    failures = ()


@pytest.mark.parametrize("workflow", ["evaluate", "predict"])
@pytest.mark.parametrize("strict", [True, False])
def test_strict_environment_is_forwarded_to_load_model(tmp_path, monkeypatch, workflow, strict):
    from saber.config import runner

    _write_classification_inputs(tmp_path)
    dataset = {"path": str(tmp_path / "data.csv"), "target_col": "label", "sample_id_col": "sample_id"}
    run_config(
        {
            "workflow": "train",
            "dataset": dataset,
            "algorithm": "logistic_regression",
            "artifact": str(tmp_path / "model"),
        }
    )
    seen = []
    real_load_model = runner.load_model

    def spy(path, **kwargs):
        seen.append(kwargs)
        return real_load_model(path, **kwargs)

    monkeypatch.setattr(runner, "load_model", spy)
    run_config(
        {
            "workflow": workflow,
            "dataset": dataset,
            "model": str(tmp_path / "model"),
            "strict_environment": strict,
        }
    )
    assert seen == [{"strict_environment": strict}]


def test_txt_and_parquet_datasets_load_like_csv(tmp_path):
    rows = ["sample_id,target,f0,f1", "s0,0,0.5,NA", "s1,1,1.5,3.0", "s2,0,2.5,4.0"]
    (tmp_path / "data.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "data.txt").write_text(
        "\n".join(row.replace(",", "\t") for row in rows) + "\n", encoding="utf-8"
    )
    read_table(tmp_path / "data.csv", separator=",").write_parquet(tmp_path / "data.parquet")

    def _bundle(name):
        config = load_config(
            {
                "workflow": "train",
                "dataset": {
                    "path": str(tmp_path / name),
                    "target_col": "target",
                    "sample_id_col": "sample_id",
                },
                "algorithm": "ridge_regressor",
            }
        )
        return load_dataset(config, config.payload["dataset"])[0]

    csv_bundle = _bundle("data.csv")
    for name in ("data.txt", "data.parquet"):
        bundle = _bundle(name)
        assert bundle.feature_names == ("f0", "f1")
        assert bundle.fingerprint == csv_bundle.fingerprint
        assert bundle.X["f1"].null_count() == 1


def test_unsupported_dataset_suffix_is_rejected_for_labeled_and_unlabeled_data(tmp_path):
    (tmp_path / "data.xlsx").write_text("x", encoding="utf-8")
    for workflow, extra in (("train", {"algorithm": "ridge_regressor"}), ("predict", {"model": "m"})):
        with pytest.raises(ConfigurationError, match="CSV, TSV, TXT"):
            run_config(
                {
                    "workflow": workflow,
                    "dataset": {"path": str(tmp_path / "data.xlsx"), "target_col": "y"},
                    **extra,
                }
            )
    with pytest.raises(ConfigurationError, match="does not exist"):
        run_config({"workflow": "predict", "dataset": {"path": str(tmp_path / "none.csv")}, "model": "m"})


def test_every_workflow_writes_fixed_file_names_into_output(tmp_path):
    _write_classification_inputs(tmp_path)
    dataset = {"path": str(tmp_path / "data.csv"), "target_col": "label", "sample_id_col": "sample_id"}
    folds = {"path": str(tmp_path / "folds.json")}
    configs = {
        "train": {
            "algorithm": "logistic_regression",
            "artifact": str(tmp_path / "model"),
            "partition_plan": folds,
        },
        "validate": {"algorithm": "logistic_regression", "partition_plan": folds},
        "tune": {
            "algorithm": "logistic_regression",
            "partition_plan": folds,
            "tuning": {"optimizer": "grid"},
            "search_space": {"C": [0.1, 1.0]},
            "metrics": ["accuracy"],
        },
        "evaluate": {"model": str(tmp_path / "model"), "metrics": ["accuracy"]},
        "predict": {"model": str(tmp_path / "model")},
    }
    expected = {
        "train": {"summary.json"},
        "validate": {"summary.json", "metrics.csv", "predictions.csv"},
        "tune": {"summary.json", "optimization_history.csv"},
        "evaluate": {"summary.json", "metrics.csv"},
        "predict": {"summary.json", "predictions.csv"},
    }
    for workflow, extra in configs.items():
        output = tmp_path / f"out_{workflow}"
        execution = run_config({"workflow": workflow, "dataset": dataset, "output": str(output), **extra})
        assert {path.name for path in output.iterdir()} == expected[workflow], workflow
        assert json.loads((output / "summary.json").read_text())["workflow"] == workflow
        assert {name for name in execution.outputs if name != "artifact"} == {
            name.split(".")[0] for name in expected[workflow]
        }


def test_overwrite_governs_existing_artifact_and_output(tmp_path):
    _write_classification_inputs(tmp_path)
    config = {
        "workflow": "train",
        "dataset": {"path": str(tmp_path / "data.csv"), "target_col": "label", "sample_id_col": "sample_id"},
        "algorithm": "logistic_regression",
        "artifact": str(tmp_path / "model"),
        "output": str(tmp_path / "out"),
    }
    run_config(config)
    with pytest.raises(ConfigurationError, match="'artifact' path already exists"):
        run_config(config)
    with pytest.raises(ConfigurationError, match="'output' path already exists"):
        run_config({**config, "artifact": str(tmp_path / "model2")})
    execution = run_config({**config, "overwrite": True})
    assert execution.outputs["artifact"] == str(tmp_path / "model")


def test_benchmark_tuning_block_takes_tuning_config_fields_only():
    base = {
        "workflow": "benchmark",
        "datasets": {"rep": {"path": "data.csv", "target_col": "y"}},
        "algorithms": ["ridge_regressor"],
        "partitions": {"cv": {"path": "folds.json"}},
        "metrics": ["rmse"],
    }
    with pytest.raises(ConfigurationError, match=r"Unknown benchmark.tuning keys: \['metrics'\]"):
        load_config({**base, "benchmark": {"modes": ["tuned"], "tuning": {"metrics": ["rmse"]}}})
    with pytest.raises(ConfigurationError, match=r"Unknown benchmark keys: \['metadata'\]"):
        load_config({**base, "benchmark": {"metadata": {}}})
