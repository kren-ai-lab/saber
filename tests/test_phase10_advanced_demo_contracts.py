from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").rglob("*.ipynb"))


def _code(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "code"
    )


def _markdown(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", [])) for cell in payload["cells"] if cell.get("cell_type") == "markdown"
    )


def test_demo_suite_is_substantive_not_minimal() -> None:
    assert len(NOTEBOOKS) >= 13
    for notebook in NOTEBOOKS:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        code = _code(notebook)
        markdown = _markdown(notebook)
        assert len(payload["cells"]) >= 6, notebook
        assert len(code) >= 1800, notebook
        assert len(markdown) >= 120, notebook
        assert "DEMO_TEST" in code, notebook
        assert "DEMO_CHECKS" in code, notebook


def test_suite_covers_complex_scientific_workflows() -> None:
    code = "\n".join(_code(path) for path in NOTEBOOKS)
    required = (
        "sample_weight",
        "pr_auc",
        "log_loss",
        "f1_macro",
        "optimization_history_frame",
        "PartitionPlan.holdout",
        "BioSievePartitionConfig",
        "BenchmarkConfig",
        "SearchSpace",
        "positive_probabilities",
        "predictions_frame",
        "verify_artifact",
    )
    for token in required:
        assert token in code, token


def test_benchmark_and_reporting_demos_exist() -> None:
    expected = {
        ROOT / "examples/benchmark/01_multi_representation_multi_partition.ipynb",
        ROOT / "examples/reporting/01_benchmark_reporting.ipynb",
        ROOT / "examples/validation/02_partition_strategy_comparison.ipynb",
        ROOT / "examples/tuning/03_optimizer_comparison.ipynb",
    }
    assert expected.issubset(set(NOTEBOOKS))


def test_reporting_demo_exports_portable_tables() -> None:
    code = _code(ROOT / "examples/reporting/01_benchmark_reporting.ipynb")
    for name in ("leaderboard.csv", "metric_summary.csv", "sample_error_audit.csv", "report.md"):
        assert name in code
