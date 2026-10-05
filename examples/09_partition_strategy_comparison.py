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
    # Partition scenarios: how validation design changes conclusions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scientific question
    How much does model performance depend on the partitioning scheme? Two fold plans prepared outside `saber` are compared on the same samples: random folds and group-blocked folds. In a real project these plans come from upstream, for example from BioSieve, and `saber` only consumes them.
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
    rng = np.random.default_rng(90)
    n_groups = 20 if DEMO_TEST else 40
    per_group = 8
    n = n_groups * per_group
    groups = np.repeat(np.arange(n_groups), per_group)
    # Global predictive signal + group-specific nuisance dimensions.
    X, y = make_classification(
        n_samples=n, n_features=10, n_informative=6, n_redundant=1, class_sep=0.9, random_state=90
    )
    group_signal = rng.normal(size=(n_groups, 3))[groups]
    X = np.column_stack([X, group_signal])
    ids = np.asarray([f"part_{i:04d}" for i in range(n)], dtype=object)
    dataset = DatasetBundle(
        X=X, y=y, sample_ids=ids, groups=groups, feature_names=[f"f{i}" for i in range(X.shape[1])]
    )
    # Scenario A: round-robin memberships that ignore groups.
    random_plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=np.arange(n) % 4,
        dataset=dataset,
        metadata={"source": "external", "strategy": "random_like"},
    )
    # Scenario B: whole groups are held out together; no group spans train/evaluation.
    group_blocked = PartitionPlan.from_predefined_folds(
        sample_ids=ids,
        fold_assignments=groups % 4,
        dataset=dataset,
        metadata={"source": "external", "strategy": "group_blocked_like"},
    )
    return dataset, group_blocked, random_plan


@app.cell
def _(BenchmarkConfig, DEMO_TEST, benchmark, dataset, group_blocked, pl, random_plan):
    bench = benchmark(
        datasets={"prepared": dataset},
        algorithms=("logistic_regression", "random_forest_classifier"),
        partitions={"random": random_plan, "group_blocked": group_blocked},
        metrics=("mcc", "balanced_accuracy", "f1", "roc_auc"),
        config=BenchmarkConfig(seeds=(42,), modes=("untuned",), include_baselines=True),
        model_params={"random_forest_classifier": {"n_estimators": 50 if DEMO_TEST else 120, "max_depth": 7}},
    )
    assert not bench.failures
    metrics = bench.aggregate_metrics_frame()
    folds = bench.fold_metrics_frame()
    comparison = (
        metrics.filter(pl.col("metric") == "mcc")
        .pivot(on="partition", index="algorithm", values="score")
        .with_columns(delta_group_minus_random=pl.col("group_blocked") - pl.col("random"))
    )
    comparison
    return comparison, folds, metrics


@app.cell
def _(comparison, folds, mo, pl, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for col in ("random", "group_blocked"):
        axes[0].plot(comparison["algorithm"], comparison[col], "o", label=col)
    axes[0].legend()
    axes[0].set(title="MCC by partition regime", ylabel="MCC")
    axes[0].tick_params(axis="x", rotation=30)
    fold_mcc = folds.filter(pl.col("metric") == "mcc")
    for (name,), part in fold_mcc.group_by("partition"):
        axes[1].plot(part["split"], part["score"], marker="o", label=name)
    axes[1].set(title="Fold-level sensitivity to partition design", ylabel="MCC")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend()
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(dataset, folds, group_blocked, metrics):
    DEMO_CHECKS = {
        "two_partition_scenarios": set(metrics.get_column("partition").unique().to_list())
        == {"random", "group_blocked"},
        "group_isolation": all(
            not set(dataset.subset(split.train_ids).groups) & set(dataset.subset(split.validation_ids).groups)
            for split in group_blocked.splits
        ),
        "fold_results": folds.get_column("split").n_unique() == 4,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
