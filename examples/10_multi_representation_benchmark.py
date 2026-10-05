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
    # Benchmark matrix: representations × partitions × models × seeds
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    This benchmark crosses three prepared feature spaces, two partition plans prepared outside `saber`, three models plus a dummy baseline, repeated seeds and four metrics. The best model can change with the representation and with the partition plan. The tables report ranks, seed stability and how the two factors interact.
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
def _(
    DEMO_TEST,
    DatasetBundle,
    PartitionPlan,
    make_classification,
    np,
):
    rng = np.random.default_rng(303)
    n = 120 if DEMO_TEST else 280
    X, y = make_classification(
        n_samples=n, n_features=14, n_informative=9, n_redundant=2, class_sep=1.0, random_state=303
    )
    ids = np.asarray([f"matrix_{i:04d}" for i in range(n)], dtype=object)
    rep1 = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"R1_{i}" for i in range(X.shape[1])])
    rep2_X = np.column_stack([X[:, :10], rng.normal(size=(n, 14))])
    rep2 = DatasetBundle(
        X=rep2_X, y=y, sample_ids=ids, feature_names=[f"R2_{i}" for i in range(rep2_X.shape[1])]
    )
    rep3_X = np.tanh(X[:, :8])
    rep3 = DatasetBundle(
        X=rep3_X, y=y, sample_ids=ids, feature_names=[f"R3_{i}" for i in range(rep3_X.shape[1])]
    )
    plan_a = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=np.arange(n) % 4,
        dataset=rep1,
        metadata={"source": "external", "strategy": "random_like"},
    )
    # Folds follow the order of the first feature, standing in for a second upstream strategy.
    alt = np.empty(n, dtype=int)
    alt[np.argsort(X[:, 0])] = np.arange(n) % 4
    plan_b = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=alt,
        dataset=rep1,
        metadata={"source": "external", "strategy": "feature_blocked_like"},
    )
    return plan_a, plan_b, rep1, rep2, rep3


@app.cell
def _(BenchmarkConfig, DEMO_TEST, benchmark, pl, plan_a, plan_b, rep1, rep2, rep3):
    seeds = (42,) if DEMO_TEST else (42, 123)
    bench = benchmark(
        datasets={"rep_core": rep1, "rep_expanded": rep2, "rep_nonlinear": rep3},
        algorithms=("logistic_regression", "random_forest_classifier", "svc"),
        partitions={"random": plan_a, "feature_blocked": plan_b},
        metrics=("mcc", "balanced_accuracy", "f1", "roc_auc"),
        config=BenchmarkConfig(seeds=seeds, modes=("untuned",), include_baselines=True),
        model_params={"random_forest_classifier": {"n_estimators": 45 if DEMO_TEST else 110, "max_depth": 7}},
    )
    assert not bench.failures
    metrics = bench.aggregate_metrics_frame()
    runs = bench.runs_frame()
    folds = bench.fold_metrics_frame()
    report = (
        metrics.filter(pl.col("metric") == "mcc")
        .group_by(["representation", "partition", "algorithm"])
        .agg(mean=pl.col("score").mean(), std=pl.col("score").std())
        .with_columns(
            rank=pl.col("mean").rank(method="dense", descending=True).over(["representation", "partition"])
        )
        .sort("algorithm")
    )
    best = report.filter(pl.col("rank") == 1).sort(["partition", "representation"])
    best
    return bench, folds, report, runs, seeds


@app.cell
def _(mo, pl, plt, report, runs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for ax, part in zip(axes, ("random", "feature_blocked")):
        chunk = report.filter(pl.col("partition") == part)
        for (representation,), sub in chunk.group_by("representation", maintain_order=True):
            ax.plot(sub["algorithm"], sub["mean"], "o", label=representation)
        ax.legend()
        ax.set_title(f"Mean MCC — {part}")
    runtime = runs.group_by("algorithm").agg(pl.col("elapsed_seconds").mean()).sort("elapsed_seconds")
    axes[2].bar(runtime["algorithm"], runtime["elapsed_seconds"])
    axes[2].set(title="Mean runtime", ylabel="seconds")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(bench, folds, report, seeds):
    expected = 3 * 2 * 4 * len(seeds)
    DEMO_CHECKS = {
        "full_matrix": bench.n_runs == expected,
        "two_partitions": set(report.get_column("partition").unique().to_list())
        == {"random", "feature_blocked"},
        "three_representations": report.get_column("representation").n_unique() == 3,
        "fold_metrics_available": not folds.is_empty(),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
