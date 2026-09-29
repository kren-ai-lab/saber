from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
FORBIDDEN = (
    "Trainer(",
    "GridSearchOptimizer(",
    "RandomSearchOptimizer(",
    "HalvingGridSearchOptimizer(",
    "HalvingRandomSearchOptimizer(",
    "train_test_split(",
    "KFold(",
    "StratifiedKFold(",
    "GroupKFold(",
)
REQUIRED_ACROSS_SUITE = (
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


def test_thirteen_examples_exist():
    assert len(EXAMPLES) == 13


def test_examples_self_check_and_use_public_workflows():
    for path in EXAMPLES:
        code = path.read_text(encoding="utf-8")
        assert "import marimo" in code, path.name
        assert "assert all(DEMO_CHECKS.values())" in code, path.name
        assert "import pandas" not in code, path.name
        for token in FORBIDDEN:
            assert token not in code, f"{path.name} reintroduces split/legacy logic via {token}"


def test_svc_examples_do_not_enable_deprecated_probability():
    for path in EXAMPLES:
        code = path.read_text(encoding="utf-8")
        assert '"probability": True' not in code
        assert "'probability': True" not in code
        assert "Categorical([True])" not in code


def test_suite_covers_complex_scientific_workflows():
    code = "\n".join(path.read_text(encoding="utf-8") for path in EXAMPLES)
    for token in REQUIRED_ACROSS_SUITE:
        assert token in code, token


def test_reporting_example_exports_portable_tables():
    code = (ROOT / "examples" / "11_benchmark_reporting.py").read_text(encoding="utf-8")
    for name in ("leaderboard.csv", "metric_summary.csv", "sample_error_audit.csv", "report.md"):
        assert name in code


def test_examples_display_generated_figures():
    """marimo only shows a cell's last expression; a figure left as a bare
    assignment never renders in the notebook. Every cell that creates a
    figure must emit it explicitly."""
    figure_markers = ("plt.subplots(", "plt.figure(", "ConfusionMatrixDisplay")
    for path in EXAMPLES:
        code = path.read_text(encoding="utf-8")
        if any(marker in code for marker in figure_markers):
            assert "mo.output.append(" in code, path.name
