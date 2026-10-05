from __future__ import annotations

import saber

EXPECTED = {
    "train", "validate", "tune", "benchmark", "evaluate", "predict", "run_config",
    "save_model", "load_model", "save_benchmark", "load_benchmark", "inspect_artifact",
    "DatasetBundle", "PartitionPlan", "BioSievePartitionConfig", "PreprocessingConfig",
    "TuningConfig", "BenchmarkConfig", "SearchSpace", "Categorical", "Integer", "Float", "LogFloat",
    "TrainResult", "PredictionResult", "EvaluationResult", "ValidationResult",
    "OptimizationResult", "BenchmarkResult", "LoadedModelArtifact",
    "ALGORITHMS", "SaberError", "__version__",
}  # fmt: skip


def test_top_level_all_is_exact_and_resolves():
    assert set(saber.__all__) == EXPECTED
    assert len(saber.__all__) == len(EXPECTED)
    for name in saber.__all__:
        assert getattr(saber, name) is not None
