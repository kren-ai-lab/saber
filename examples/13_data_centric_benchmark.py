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
    # End-to-end study: prepared representations, tuning and a protected-test benchmark
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scientific question
    How much do the choice of representation and hyperparameter tuning change performance on a protected test set? The benchmark compares two prepared representations, two classical models in untuned and tuned modes, repeated seeds and a dummy baseline. Tuning only sees the train and validation data, and every reported score comes from the protected test.
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
    from saber import (
        benchmark,
        BenchmarkConfig,
        SearchSpace,
        Categorical,
        DatasetBundle,
        PartitionPlan,
        TuningConfig,
    )

    return (
        BenchmarkConfig,
        Categorical,
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        SearchSpace,
        TuningConfig,
        benchmark,
        make_classification,
        np,
        pl,
        plt,
    )


@app.cell
def _(DEMO_TEST, DatasetBundle, PartitionPlan, make_classification, np):
    rng = np.random.default_rng(2026)
    n_samples = 150 if DEMO_TEST else 360
    X, y = make_classification(
        n_samples=n_samples,
        n_features=14,
        n_informative=9,
        n_redundant=2,
        weights=[0.6, 0.4],
        class_sep=1.05,
        random_state=2026,
    )
    ids = np.asarray([f"study_{i:04d}" for i in range(len(y))], dtype=object)
    rep_a = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"A_{i}" for i in range(X.shape[1])])
    rep_b_X = np.column_stack(
        [X[:, :9], 0.35 * X[:, 0:3] + rng.normal(scale=0.4, size=(len(y), 3)), rng.normal(size=(len(y), 8))]
    )
    rep_b = DatasetBundle(
        X=rep_b_X, y=y, sample_ids=ids, feature_names=[f"B_{i}" for i in range(rep_b_X.shape[1])]
    )
    train_ids, val_ids, test_ids = np.split(ids, [int(0.6 * len(ids)), int(0.8 * len(ids))])
    plan = PartitionPlan.holdout(
        train_ids=train_ids,
        validation_ids=val_ids,
        test_ids=test_ids,
        dataset=rep_a,
        metadata={"source": "prepared_holdout", "test_role": "protected"},
    )
    return plan, rep_a, rep_b


@app.cell
def _(
    BenchmarkConfig,
    Categorical,
    DEMO_TEST,
    SearchSpace,
    TuningConfig,
    benchmark,
    pl,
    plan,
    rep_a,
    rep_b,
):
    seeds = (42,) if DEMO_TEST else (42, 123)
    tuning = TuningConfig(optimizer="grid", refit_metric="mcc", n_jobs=1)
    search_spaces = {
        "logistic_regression": SearchSpace(
            {"C": Categorical([0.1, 1.0, 3.0]), "solver": Categorical(["lbfgs"])}
        ),
        "random_forest_classifier": SearchSpace(
            {
                "n_estimators": Categorical([40, 80] if DEMO_TEST else [80, 160]),
                "max_depth": Categorical([4, 8]),
            }
        ),
    }
    bench = benchmark(
        datasets={"representation_A": rep_a, "representation_B": rep_b},
        algorithms=("logistic_regression", "random_forest_classifier"),
        partitions={"development_plus_protected_test": plan},
        metrics=("mcc", "balanced_accuracy", "f1", "roc_auc"),
        config=BenchmarkConfig(
            seeds=seeds, modes=("untuned", "tuned"), include_baselines=True, tuning=tuning
        ),
        search_spaces=search_spaces,
    )
    assert not bench.failures
    runs = bench.runs_frame()
    metrics = bench.aggregate_metrics_frame()
    history = bench.optimization_history_frame()
    preds = bench.predictions_frame()
    leader = (
        metrics.filter(pl.col("metric") == "mcc")
        .group_by(["representation", "algorithm", "mode"])
        .agg(mean=pl.col("score").mean(), std=pl.col("score").std())
        .with_columns(label=pl.col("algorithm") + " / " + pl.col("mode"))
        .sort("label")
    )
    leader.sort("mean", descending=True).head(10)
    return bench, history, leader, preds, runs, seeds


@app.cell
def _(history, leader, mo, pl, plt, runs):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for (representation,), part in leader.group_by("representation", maintain_order=True):
        axes[0].plot(part["label"], part["mean"], "o", label=representation)
    axes[0].legend()
    axes[0].set(title="Protected-test MCC", ylabel="MCC")
    axes[0].tick_params(axis="x", rotation=45)
    counts = history.filter(pl.col("status") == "complete").group_by("algorithm").len()
    axes[1].bar(counts["algorithm"], counts["len"])
    axes[1].set(title="Tuning candidates evaluated", ylabel="count")
    runtime = (
        runs.group_by(["algorithm", "mode"])
        .agg(mean=pl.col("elapsed_seconds").mean())
        .with_columns(label=pl.col("algorithm") + " / " + pl.col("mode"))
    )
    axes[2].bar(runtime["label"], runtime["mean"])
    axes[2].set(title="Workflow runtime", ylabel="seconds")
    axes[2].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(bench, history, preds, runs, seeds):
    expected = len(seeds) * 2 * (1 + 2 * 2)  # seeds × representations × (baseline + 2 models × 2 modes)
    DEMO_CHECKS = {
        "benchmark_matrix": bench.n_runs == expected,
        "tuned_runs_have_history": not history.is_empty(),
        "protected_test_only": set(preds.get_column("evaluation_role").unique().to_list()) == {"test"},
        "run_prediction_traceability": set(preds.get_column("run_id")) == set(runs.get_column("run_id")),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
