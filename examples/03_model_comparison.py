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
    # Model comparison — representation × model × seed benchmark
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Run a reproducible benchmark matrix across three prepared representations, multiple classical model families, repeated seeds and a dummy baseline. The notebook builds a leaderboard, stability table, rank analysis and runtime comparison from `BenchmarkResult` rather than hand-assembling model loops.
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

    from sklearn.datasets import make_classification
    from saber import benchmark, BenchmarkConfig, DatasetBundle, PartitionPlan

    return (
        BenchmarkConfig,
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        balanced_fold_labels,
        benchmark,
        make_classification,
        np,
        pl,
        plt,
    )


@app.cell
def _(
    DEMO_TEST,
    DatasetBundle,
    PartitionPlan,
    balanced_fold_labels,
    make_classification,
    np,
):
    rng = np.random.default_rng(21)
    n_samples = 120 if DEMO_TEST else 300
    X, y = make_classification(n_samples=n_samples, n_features=12, n_informative=8, n_redundant=2,
                               class_sep=1.1, random_state=21)
    ids = [f"cmp_{i:04d}" for i in range(len(y))]
    compact = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"compact_{i}" for i in range(X.shape[1])])
    expanded_X = np.column_stack([X, rng.normal(size=(len(y), 18))])
    expanded = DatasetBundle(X=expanded_X, y=y, sample_ids=ids, feature_names=[f"expanded_{i}" for i in range(expanded_X.shape[1])])
    compressed_X = X[:, :6] + 0.05*rng.normal(size=(len(y),6))
    compressed = DatasetBundle(X=compressed_X, y=y, sample_ids=ids, feature_names=[f"compressed_{i}" for i in range(6)])
    plan = PartitionPlan.from_predefined_folds(sample_ids=ids, fold_assignments=balanced_fold_labels(y,4),
                                               dataset=compact)
    return compact, compressed, expanded, plan


@app.cell
def _(
    BenchmarkConfig,
    DEMO_TEST,
    benchmark,
    compact,
    compressed,
    expanded,
    pl,
    plan,
):
    seeds = (42,) if DEMO_TEST else (42,123,777)
    bench = benchmark(
        datasets={"compact":compact, "expanded_noisy":expanded, "compressed":compressed},
        algorithms=("logistic_regression", "random_forest_classifier", "svc"),
        partitions={"prepared_4fold":plan},
        metrics=("mcc","balanced_accuracy","f1","roc_auc"),config=BenchmarkConfig(
                               seeds=seeds, modes=("untuned",), include_baselines=True),
        model_params={"random_forest_classifier":{"n_estimators":50 if DEMO_TEST else 120,"max_depth":7}},
    )
    assert not bench.failures
    metrics = bench.aggregate_metrics_frame(); runs = bench.runs_frame(); preds = bench.predictions_frame()
    leader = (
        metrics.filter(pl.col("metric") == "mcc")
        .group_by(["representation", "algorithm"])
        .agg(
            pl.col("score").mean().alias("mean"),
            pl.col("score").std().alias("std"),
            pl.col("score").min().alias("min"),
            pl.col("score").max().alias("max"),
        )
    )
    leader = leader.with_columns(
        pl.col("mean").rank(method="dense", descending=True).over("representation").alias("rank_within_representation")
    ).sort(["representation", "rank_within_representation"])
    leader
    return bench, leader, metrics, preds, runs, seeds


@app.cell
def _(leader, mo, np, pl, plt, runs):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    fig, axes = plt.subplots(1,3,figsize=(17,4.5))
    mean_wide = leader.pivot(on="representation", index="algorithm", values="mean")
    grouped_bar(
        axes[0],
        mean_wide.get_column("algorithm").to_list(),
        {col: mean_wide.get_column(col).to_numpy() for col in mean_wide.columns if col != "algorithm"},
    )
    axes[0].set(title="Mean MCC", ylabel="MCC"); axes[0].tick_params(axis="x",rotation=30)
    std_wide = leader.pivot(on="representation", index="algorithm", values="std")
    grouped_bar(
        axes[1],
        std_wide.get_column("algorithm").to_list(),
        {col: std_wide.get_column(col).to_numpy() for col in std_wide.columns if col != "algorithm"},
    )
    axes[1].set(title="Seed stability (MCC SD)", ylabel="SD"); axes[1].tick_params(axis="x",rotation=30)
    runtime = runs.group_by("algorithm").agg(pl.col("elapsed_seconds").mean().alias("mean")).sort("mean")
    axes[2].bar(runtime.get_column("algorithm").to_list(), runtime.get_column("mean").to_numpy())
    axes[2].set(title="Mean runtime",ylabel="seconds"); axes[2].tick_params(axis="x",rotation=30)
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(FIGURE_COUNT, bench, leader, metrics, preds, runs, seeds):
    expected = 3 * 4 * len(seeds)  # three representations × baseline+3 models × seeds
    DEMO_CHECKS = {
        "matrix_complete": bench.n_runs == expected,
        "no_failures": len(bench.failures)==0,
        "metric_panel": set(metrics.get_column("metric").unique().to_list()) == {"mcc","balanced_accuracy","f1","roc_auc"},
        "predictions_traceable": set(preds.get_column("run_id")) == set(runs.get_column("run_id")),
        "leaderboard_complete": len(leader)==12,
        "rank_available": leader.get_column("rank_within_representation").is_not_null().all(),
        "figure_created": FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
