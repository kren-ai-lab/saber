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
    # Partition scenarios — how validation design changes conclusions
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scientific question
    How sensitive is model performance to the partitioning regime? We compare two **externally prepared membership plans** on the same samples: a balanced fold plan and a group-blocked plan. In production these plans should be generated upstream (for example with BioSieve). `saber` only consumes the memberships.
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
    from saber import benchmark
    from saber.benchmark import BenchmarkConfig
    from saber.datasets import DatasetBundle, PartitionPlan

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
    rng=np.random.default_rng(90)
    n_groups=20 if DEMO_TEST else 40; per_group=8
    n=n_groups*per_group
    groups=np.repeat(np.arange(n_groups),per_group)
    # Global predictive signal + group-specific nuisance dimensions.
    X,y=make_classification(n_samples=n,n_features=10,n_informative=6,n_redundant=1,class_sep=.9,random_state=90)
    group_signal=rng.normal(size=(n_groups,3))[groups]
    X=np.column_stack([X,group_signal])
    ids=np.asarray([f"part_{i:04d}" for i in range(n)],dtype=object)
    dataset=DatasetBundle(X=X,y=y,sample_ids=ids,groups=groups,feature_names=[f"f{i}" for i in range(X.shape[1])])
    # Scenario A: balanced memberships across labels.
    balanced=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=balanced_fold_labels(y,4),dataset_fingerprint=dataset.fingerprint,
                                                  metadata={"source":"external","strategy":"balanced_random_like"})
    # Scenario B: whole groups are held out together; no group spans train/evaluation.
    group_folds=groups%4
    group_blocked=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=group_folds,dataset_fingerprint=dataset.fingerprint,
                                                       metadata={"source":"external","strategy":"group_blocked_like"})
    return balanced, dataset, group_blocked


@app.cell
def _(BenchmarkConfig, DEMO_TEST, balanced, benchmark, dataset, group_blocked, pl):
    bench=benchmark(datasets={"prepared":dataset},algorithms=("logistic_regression","random_forest"),
                    partitions={"balanced":balanced,"group_blocked":group_blocked},
                    config=BenchmarkConfig(metrics=("mcc","balanced_accuracy","f1","roc_auc"),seeds=(42,),modes=("untuned",),include_baselines=True),
                    model_params={"random_forest":{"n_estimators":50 if DEMO_TEST else 120,"max_depth":7}})
    assert not bench.failures
    metrics=bench.aggregate_metrics_frame(); folds=bench.fold_metrics_frame()
    comparison=metrics.filter(pl.col("metric")=="mcc").pivot(on="partition",index="algorithm",values="score")
    comparison=comparison.with_columns((pl.col("group_blocked")-pl.col("balanced")).alias("delta_group_minus_balanced"))
    comparison
    return bench, comparison, folds, metrics


@app.cell
def _(comparison, dataset, folds, group_blocked, np, pl, plt):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    # Confirm group isolation for the blocked plan.
    leak_checks=[]
    for split in group_blocked.splits:
        train_groups=set(dataset.subset(split.train_ids).groups)
        eval_groups=set(dataset.subset(split.validation_ids).groups)
        leak_checks.append(len(train_groups & eval_groups)==0)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    grouped_bar(
        axes[0],
        comparison.get_column("algorithm").to_list(),
        {col: comparison.get_column(col).to_numpy() for col in ("balanced","group_blocked")},
    )
    axes[0].set(title="MCC by partition regime",ylabel="MCC"); axes[0].tick_params(axis="x",rotation=30)
    fold_mcc=folds.filter(pl.col("metric")=="mcc")
    for (name,),part in fold_mcc.group_by("partition"):
        axes[1].plot(part.get_column("split").to_numpy(),part.get_column("score").to_numpy(),marker="o",label=name)
    axes[1].set(title="Fold-level sensitivity to partition design",ylabel="MCC"); axes[1].tick_params(axis="x",rotation=30); axes[1].legend()
    plt.tight_layout(); FIGURE_COUNT=1
    return FIGURE_COUNT, leak_checks


@app.cell
def _(FIGURE_COUNT, bench, comparison, folds, leak_checks, metrics):
    DEMO_CHECKS={
        "two_partition_scenarios":set(metrics.get_column("partition").unique().to_list())=={"balanced","group_blocked"},
        "group_isolation":all(leak_checks),
        "no_failures":len(bench.failures)==0,
        "comparison_report":set(comparison.columns)>={"balanced","group_blocked","delta_group_minus_balanced"},
        "fold_results":folds.get_column("split").n_unique()==4,
        "figure_created":FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
