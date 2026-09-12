from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NOTEBOOKS = {
    "examples/classification/01_binary_classification.ipynb",
    "examples/classification/02_multiclass_classification.ipynb",
    "examples/classification/03_model_comparison.ipynb",
    "examples/regression/01_regression_workflow.ipynb",
    "examples/tuning/01_hyperparameter_optimization.ipynb",
    "examples/tuning/02_optuna_optimization.ipynb",
    "examples/tuning/03_optimizer_comparison.ipynb",
    "examples/validation/01_biosieve_validation.ipynb",
    "examples/validation/02_partition_strategy_comparison.ipynb",
    "examples/benchmark/01_multi_representation_multi_partition.ipynb",
    "examples/reporting/01_benchmark_reporting.ipynb",
    "examples/persistence/01_model_persistence.ipynb",
    "examples/end_to_end/01_data_centric_benchmark.ipynb",
}

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(nb: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in nb["cells"]
        if cell.get("cell_type") == "code"
    )


def test_phase10_demo_inventory_is_complete() -> None:
    observed = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "examples").rglob("*.ipynb")
    }
    assert observed == EXPECTED_NOTEBOOKS


def test_every_demo_has_assertions_visualization_and_phase_metadata() -> None:
    for relative in sorted(EXPECTED_NOTEBOOKS):
        nb = _load(ROOT / relative)
        code = _source(nb)
        assert nb["metadata"]["mlcore"]["phase"] == 10
        assert nb["metadata"]["mlcore"]["level"] == "advanced"
        assert len(nb["cells"]) >= 6
        assert "DEMO_CHECKS" in code
        assert "assert all(DEMO_CHECKS.values())" in code
        assert "plt." in code or "ConfusionMatrixDisplay" in code
        assert all(cell.get("execution_count") is None for cell in nb["cells"] if cell.get("cell_type") == "code")
        assert all(not cell.get("outputs") for cell in nb["cells"] if cell.get("cell_type") == "code")


def test_notebooks_use_public_workflows_and_do_not_reintroduce_splitters() -> None:
    forbidden = (
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
    for relative in sorted(EXPECTED_NOTEBOOKS):
        code = _source(_load(ROOT / relative))
        for token in forbidden:
            assert token not in code, f"{relative} reintroduces legacy/split logic via {token}"


def test_demo_index_references_every_notebook() -> None:
    text = (ROOT / "examples" / "README.md").read_text(encoding="utf-8")
    for relative in EXPECTED_NOTEBOOKS:
        short = relative.removeprefix("examples/")
        assert short in text


def test_svc_demos_do_not_enable_deprecated_probability_parameter() -> None:
    """SVC probability=True is deprecated in sklearn >=1.9.

    Phase 10 demos rely on decision_function for ranking metrics such as
    ROC-AUC. Probability calibration, when scientifically required, should be
    demonstrated explicitly with CalibratedClassifierCV instead of the
    deprecated SVC(probability=True) path.
    """

    for relative in sorted(EXPECTED_NOTEBOOKS):
        code = _source(_load(ROOT / relative))
        if "svc" not in code.lower():
            continue
        assert '"probability": True' not in code
        assert "'probability': True" not in code
        assert "Categorical([True])" not in code
