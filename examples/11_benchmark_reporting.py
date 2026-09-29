import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Benchmark reporting — leaderboard, stability, error audit and export
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Turn structured `BenchmarkResult` outputs into an analysis-ready report **outside the saber core**. We generate a leaderboard with mean/SD/rank, metric-wide summary, runtime table, sample-level error audit and portable CSV/Markdown report files.
    """)
    return


@app.cell
def _():
    import os
    os.environ.setdefault("MPLBACKEND", "Agg")

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"

    def balanced_fold_labels(y, n_splits):
        """Demo-only external memberships. Production partition generation belongs to BioSieve."""
        y = np.asarray(y)
        folds = np.empty(len(y), dtype=int)
        for label in np.unique(y):
            idx = np.flatnonzero(y == label)
            folds[idx] = np.arange(len(idx)) % n_splits
        return folds

    from pathlib import Path
    import tempfile
    from sklearn.datasets import make_classification
    from saber import benchmark
    from saber.benchmark import BenchmarkConfig
    from saber.datasets import DatasetBundle, PartitionPlan

    return (
        BenchmarkConfig,
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        Path,
        balanced_fold_labels,
        benchmark,
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
    balanced_fold_labels,
    benchmark,
    make_classification,
):
    X,y=make_classification(n_samples=120 if DEMO_TEST else 260,n_features=12,n_informative=8,class_sep=1.05,random_state=404)
    ids=[f"report_{i:04d}" for i in range(len(y))]
    dataset=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"f{i}" for i in range(X.shape[1])])
    plan=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=balanced_fold_labels(y,5),dataset_fingerprint=dataset.fingerprint)
    bench=benchmark(datasets={"prepared":dataset},algorithms=("logistic_regression","random_forest","svc"),partitions={"fivefold":plan},
                    config=BenchmarkConfig(metrics=("mcc","balanced_accuracy","f1","roc_auc"),seeds=(42,) if DEMO_TEST else (42,123,777),modes=("untuned",),include_baselines=True),
                    model_params={"random_forest":{"n_estimators":45 if DEMO_TEST else 120,"max_depth":7}})
    assert not bench.failures
    metrics=bench.aggregate_metrics_frame(); runs=bench.runs_frame(); preds=bench.predictions_frame()
    return dataset, metrics, preds, runs


@app.cell
def _(metrics, pl, preds, runs):
    # Leaderboard and stability report.
    mcc = metrics.filter(pl.col("metric") == "mcc")
    leader = (
        mcc.group_by("algorithm")
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").max().alias("max"),
            pl.col("score").count().alias("count"),
        )
        .sort("mean", descending=True)
        .with_row_index("rank", offset=1)
    )
    metric_wide = metrics.pivot(on="metric", index=["algorithm", "seed"], values="score")
    runtime = (
        runs.group_by("algorithm")
        .agg(pl.col("elapsed_seconds").mean().alias("mean"), pl.col("elapsed_seconds").std().alias("std"))
        .sort("mean")
    )
    # Sample-level audit: how often is each sample misclassified across runs?
    preds_1 = preds.with_columns((pl.col("y_true") != pl.col("y_pred")).cast(pl.Int64).alias("error"))
    error_audit = (
        preds_1.group_by("sample_id")
        .agg(
            pl.col("error").mean().alias("error_rate"),
            pl.col("error").len().alias("n_predictions"),
            pl.col("y_true").first().alias("y_true"),
        )
        .sort("error_rate", descending=True)
    )
    (leader, error_audit.head(10))
    return error_audit, leader, metric_wide, runtime


@app.cell
def _(Path, error_audit, leader, metric_wide, pl, runtime, tempfile):
    def round_frame(frame, decimals=4):
        numeric_cols = [c for c, dt in zip(frame.columns, frame.dtypes) if dt.is_numeric()]
        return frame.with_columns([pl.col(c).round(decimals) for c in numeric_cols])

    # Produce portable report artifacts in a temporary directory.
    out = Path(tempfile.mkdtemp(prefix="saber-example-"))
    leader.write_csv(out/"leaderboard.csv")
    metric_wide.write_csv(out/"metric_summary.csv")
    error_audit.write_csv(out/"sample_error_audit.csv")
    report_text = (
        "# saber benchmark report\n\n## Leaderboard\n\n```text\n" + str(round_frame(leader))
        + "\n```\n\n## Mean runtime\n\n```text\n" + str(round_frame(runtime)) + "\n```\n"
    )
    (out/"report.md").write_text(report_text,encoding="utf-8")
    exported=sorted(p.name for p in out.iterdir())
    print(report_text[:900])
    return exported, report_text


@app.cell
def _(error_audit, leader, metric_wide, mo, np, pl, plt):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    axes[0].bar(leader.get_column("algorithm").to_list(), leader.get_column("mean").to_numpy())
    axes[0].set(title="Leaderboard — MCC",ylabel="mean MCC")
    profile_cols = ["balanced_accuracy", "f1", "roc_auc"]
    profile = metric_wide.group_by("algorithm").agg([pl.col(c).mean() for c in profile_cols])
    grouped_bar(axes[1], profile.get_column("algorithm").to_list(), {col: profile.get_column(col).to_numpy() for col in profile_cols})
    axes[1].set(title="Metric profile"); axes[1].tick_params(axis="x",rotation=35)
    axes[2].hist(error_audit.get_column("error_rate").to_numpy(),bins=12); axes[2].set(title="Sample-level error frequency",xlabel="error rate across runs")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(
    FIGURE_COUNT,
    dataset,
    error_audit,
    exported,
    leader,
    metric_wide,
    report_text,
):
    DEMO_CHECKS={
        "leaderboard_has_baseline_and_models":len(leader)==4,
        "metric_report_complete":set(["mcc","balanced_accuracy","f1","roc_auc"]).issubset(metric_wide.columns),
        "sample_audit_complete":len(error_audit)==dataset.n_samples,
        "traceable_prediction_count":error_audit.get_column("n_predictions").min()>0,
        "report_exports":set(exported)=={"leaderboard.csv","metric_summary.csv","report.md","sample_error_audit.csv"},
        "markdown_report":report_text.startswith("# saber benchmark report"),
        "figure_created":FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
