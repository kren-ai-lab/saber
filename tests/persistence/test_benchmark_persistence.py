from __future__ import annotations

from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal
from sklearn.datasets import make_classification

from saber.benchmark import BenchmarkConfig, benchmark
from saber.datasets import DatasetBundle, PartitionPlan
from saber.persistence import load_benchmark, save_benchmark, verify_artifact


def _benchmark_result():
    X, y = make_classification(
        n_samples=50,
        n_features=5,
        n_informative=3,
        random_state=7,
    )
    dataset = DatasetBundle(
        X=X,
        y=y,
        sample_ids=[f"s{i}" for i in range(len(y))],
    )
    assert dataset.sample_ids is not None
    plan = PartitionPlan.holdout(
        train_ids=dataset.sample_ids[:35],
        test_ids=dataset.sample_ids[35:],
        dataset_fingerprint=dataset.fingerprint,
    )
    return benchmark(
        datasets={"representation_a": dataset},
        algorithms=("logistic_regression",),
        partitions={"holdout": plan},
        config=BenchmarkConfig(
            metrics=("accuracy", "mcc"),
            seeds=(42,),
            include_baselines=False,
            return_estimators=False,
        ),
    )


def test_benchmark_artifact_persists_analysis_tables(tmp_path):
    result = _benchmark_result()
    artifact_path = tmp_path / "benchmark_artifact"
    save_benchmark(
        artifact_path,
        result,
        metadata={"study": "phase7"},
    )
    loaded = load_benchmark(artifact_path)

    assert loaded.manifest.artifact_type == "benchmark"
    assert loaded.metadata["n_runs"] == 1
    assert loaded.metadata["metadata"]["study"] == "phase7"
    assert not loaded.table("runs").is_empty()
    assert not loaded.table("metrics").is_empty()
    assert not loaded.table("predictions").is_empty()
    assert loaded.table("failures").is_empty()


def test_benchmark_artifact_can_optionally_round_trip_python_object(tmp_path):
    result = _benchmark_result()
    artifact_path = tmp_path / "benchmark_artifact"
    save_benchmark(
        artifact_path,
        result,
        include_object=True,
    )
    loaded = load_benchmark(artifact_path, load_object=True)
    assert loaded.result is not None
    assert loaded.result.n_runs == result.n_runs
    assert loaded.result.runs[0].run_id == result.runs[0].run_id


def test_loaded_benchmark_table_is_polars_and_matches_result_frame(tmp_path):
    result = _benchmark_result()
    artifact_path = tmp_path / "benchmark_artifact"
    save_benchmark(artifact_path, result)
    loaded = load_benchmark(artifact_path)

    for name, expected in (
        ("metrics", result.metrics_frame()),
        ("predictions", result.predictions_frame()),
        ("runs", result.runs_frame()),
    ):
        table = loaded.table(name)
        assert isinstance(table, pl.DataFrame)
        # CSV does not keep exact dtypes, but every value must survive the round trip.
        assert_frame_equal(table, expected, check_dtypes=False)
    assert verify_artifact(artifact_path).artifact_type == "benchmark"


def test_benchmark_artifact_file_set_is_human_inspectable(tmp_path):
    result = _benchmark_result()
    artifact_path = tmp_path / "benchmark_artifact"
    save_benchmark(artifact_path, result)
    expected = {
        "manifest.json",
        "environment.json",
        "benchmark_metadata.json",
        "runs.csv",
        "metrics.csv",
        "predictions.csv",
        "failures.csv",
        "optimization_history.csv",
        "checksums.sha256",
    }
    assert expected <= {path.name for path in Path(artifact_path).iterdir()}
