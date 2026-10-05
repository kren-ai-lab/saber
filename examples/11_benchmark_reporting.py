import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Benchmark reporting: leaderboard, stability, error audit and export
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    This notebook turns a `BenchmarkResult` into a report with code that lives outside the `saber` core. It builds a leaderboard with mean, SD and rank, a wide table of all metrics, a runtime table and a per-sample error audit, then exports them as CSV and Markdown files.
    """)
    return


@app.cell
def _():
    import os

    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"

    from pathlib import Path
    import tempfile
    from sklearn.datasets import make_classification
    from saber import benchmark, BenchmarkConfig, DatasetBundle, PartitionPlan

    return (
        BenchmarkConfig,
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        Path,
        benchmark,
        cs,
        make_classification,
        np,
        pl,
        plt,
        tempfile,
    )


@app.cell
def _(
    BenchmarkConfig,
    DEMO_TEST,
    DatasetBundle,
    PartitionPlan,
    benchmark,
    make_classification,
    np,
):
    X, y = make_classification(
        n_samples=120 if DEMO_TEST else 260, n_features=12, n_informative=8, class_sep=1.05, random_state=404
    )
    ids = [f"report_{i:04d}" for i in range(len(y))]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"f{i}" for i in range(X.shape[1])])
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids, fold_assignments=np.arange(len(y)) % 5, dataset=dataset
    )
    bench = benchmark(
        datasets={"prepared": dataset},
        algorithms=("logistic_regression", "random_forest_classifier", "svc"),
        partitions={"fivefold": plan},
        metrics=("mcc", "balanced_accuracy", "f1", "roc_auc"),
        config=BenchmarkConfig(
            seeds=(42,) if DEMO_TEST else (42, 123, 777), modes=("untuned",), include_baselines=True
        ),
        model_params={"random_forest_classifier": {"n_estimators": 45 if DEMO_TEST else 120, "max_depth": 7}},
    )
    assert not bench.failures
    metrics = bench.aggregate_metrics_frame()
    runs = bench.runs_frame()
    preds = bench.predictions_frame()
    return dataset, metrics, preds, runs


@app.cell
def _(metrics, pl, preds, runs):
    # Leaderboard and stability report.
    leader = (
        metrics.filter(pl.col("metric") == "mcc")
        .group_by("algorithm")
        .agg(
            mean=pl.col("score").mean(),
            std=pl.col("score").std(),
            min=pl.col("score").min(),
            max=pl.col("score").max(),
            count=pl.col("score").count(),
        )
        .sort("mean", descending=True)
        .with_row_index("rank", offset=1)
    )
    metric_wide = metrics.pivot(on="metric", index=["algorithm", "seed"], values="score")
    runtime = (
        runs.group_by("algorithm")
        .agg(mean=pl.col("elapsed_seconds").mean(), std=pl.col("elapsed_seconds").std())
        .sort("mean")
    )
    # Sample-level audit: how often is each sample misclassified across runs?
    error_audit = (
        preds.group_by("sample_id")
        .agg(
            error_rate=(pl.col("y_true") != pl.col("y_pred")).mean(),
            n_predictions=pl.len(),
            y_true=pl.col("y_true").first(),
        )
        .sort("error_rate", descending=True)
    )
    (leader, error_audit.head(10))
    return error_audit, leader, metric_wide, runtime


@app.cell
def _(Path, cs, error_audit, leader, metric_wide, runtime, tempfile):
    report_text = (
        "# saber benchmark report\n\n## Leaderboard\n\n```text\n"
        + str(leader.with_columns(cs.float().round(4)))
        + "\n```\n\n## Mean runtime\n\n```text\n"
        + str(runtime.with_columns(cs.float().round(4)))
        + "\n```\n"
    )
    # Write portable report artifacts to a temporary directory.
    with tempfile.TemporaryDirectory(prefix="saber-example-") as tmp:
        out = Path(tmp)
        leader.write_csv(out / "leaderboard.csv")
        metric_wide.write_csv(out / "metric_summary.csv")
        error_audit.write_csv(out / "sample_error_audit.csv")
        (out / "report.md").write_text(report_text, encoding="utf-8")
        exported = sorted(p.name for p in out.iterdir())
    print(report_text[:900])
    return (exported,)


@app.cell
def _(error_audit, leader, metric_wide, mo, pl, plt):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].bar(leader["algorithm"], leader["mean"])
    axes[0].set(title="Leaderboard — MCC", ylabel="mean MCC")
    profile_cols = ["balanced_accuracy", "f1", "roc_auc"]
    profile = metric_wide.group_by("algorithm").agg(pl.col(profile_cols).mean())
    for col in profile_cols:
        axes[1].plot(profile["algorithm"], profile[col], "o", label=col)
    axes[1].legend()
    axes[1].set(title="Metric profile")
    axes[1].tick_params(axis="x", rotation=35)
    axes[2].hist(error_audit["error_rate"], bins=12)
    axes[2].set(title="Sample-level error frequency", xlabel="error rate across runs")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(dataset, error_audit, exported, leader):
    DEMO_CHECKS = {
        "leaderboard_has_baseline_and_models": len(leader) == 4,
        "sample_audit_complete": len(error_audit) == dataset.n_samples,
        "report_exports": exported
        == ["leaderboard.csv", "metric_summary.csv", "report.md", "sample_error_audit.csv"],
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
