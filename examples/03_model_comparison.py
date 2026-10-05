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
    # Model comparison: representation × model × seed benchmark
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    One `benchmark` call runs three prepared representations, three classical models, repeated seeds and a dummy baseline. The leaderboard, seed stability, ranks within each representation and runtimes all come from the returned `BenchmarkResult`, with no hand-written model loop.
    """)
    return


@app.cell
def _():
    import os

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"

    from sklearn.datasets import make_classification
    from saber import benchmark, BenchmarkConfig, DatasetBundle, PartitionPlan

    return (
        BenchmarkConfig,
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        benchmark,
        make_classification,
        np,
        pl,
        plt,
    )


@app.cell
def _(DEMO_TEST, DatasetBundle, PartitionPlan, make_classification, np):
    rng = np.random.default_rng(21)
    n_samples = 120 if DEMO_TEST else 300
    X, y = make_classification(
        n_samples=n_samples, n_features=12, n_informative=8, n_redundant=2, class_sep=1.1, random_state=21
    )
    ids = [f"cmp_{i:04d}" for i in range(len(y))]
    compact = DatasetBundle(
        X=X, y=y, sample_ids=ids, feature_names=[f"compact_{i}" for i in range(X.shape[1])]
    )
    expanded_X = np.column_stack([X, rng.normal(size=(len(y), 18))])
    expanded = DatasetBundle(
        X=expanded_X, y=y, sample_ids=ids, feature_names=[f"expanded_{i}" for i in range(expanded_X.shape[1])]
    )
    compressed_X = X[:, :6] + 0.05 * rng.normal(size=(len(y), 6))
    compressed = DatasetBundle(
        X=compressed_X, y=y, sample_ids=ids, feature_names=[f"compressed_{i}" for i in range(6)]
    )
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids, fold_assignments=np.arange(len(y)) % 4, dataset=compact
    )
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
    seeds = (42,) if DEMO_TEST else (42, 123, 777)
    bench = benchmark(
        datasets={"compact": compact, "expanded_noisy": expanded, "compressed": compressed},
        algorithms=("logistic_regression", "random_forest_classifier", "svc"),
        partitions={"prepared_4fold": plan},
        metrics=("mcc", "balanced_accuracy", "f1", "roc_auc"),
        config=BenchmarkConfig(seeds=seeds, modes=("untuned",), include_baselines=True),
        model_params={"random_forest_classifier": {"n_estimators": 50 if DEMO_TEST else 120, "max_depth": 7}},
    )
    assert not bench.failures
    metrics = bench.aggregate_metrics_frame()
    runs = bench.runs_frame()
    preds = bench.predictions_frame()
    leader = (
        metrics.filter(pl.col("metric") == "mcc")
        .group_by(["representation", "algorithm"])
        .agg(
            mean=pl.col("score").mean(),
            std=pl.col("score").std(),
            min=pl.col("score").min(),
            max=pl.col("score").max(),
        )
        .with_columns(
            rank_within_representation=pl.col("mean")
            .rank(method="dense", descending=True)
            .over("representation")
        )
        .sort(["representation", "rank_within_representation"])
    )
    leader
    return bench, leader, metrics, preds, runs, seeds


@app.cell
def _(leader, mo, pl, plt, runs):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for ax, stat, title in ((axes[0], "mean", "Mean MCC"), (axes[1], "std", "Seed stability (MCC SD)")):
        for (representation,), part in leader.group_by("representation", maintain_order=True):
            ax.plot(part["algorithm"], part[stat], "o", label=representation)
        ax.set(title=title, ylabel=stat)
        ax.legend()
    runtime = runs.group_by("algorithm").agg(mean=pl.col("elapsed_seconds").mean()).sort("mean")
    axes[2].bar(runtime["algorithm"], runtime["mean"])
    axes[2].set(title="Mean runtime", ylabel="seconds")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(bench, leader, metrics, preds, runs, seeds):
    expected = 3 * 4 * len(seeds)  # three representations × baseline+3 models × seeds
    DEMO_CHECKS = {
        "matrix_complete": bench.n_runs == expected,
        "metric_panel": set(metrics.get_column("metric").unique().to_list())
        == {"mcc", "balanced_accuracy", "f1", "roc_auc"},
        "predictions_traceable": set(preds.get_column("run_id")) == set(runs.get_column("run_id")),
        "leaderboard_complete": len(leader) == 12,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
