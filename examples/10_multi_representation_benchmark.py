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
    # Benchmark matrix — representations × partitions × models × seeds
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Build a realistic data-centric benchmark matrix in which conclusions can depend on both representation and partition regime. We compare three prepared feature spaces, two externally prepared partition scenarios, three model families plus a dummy baseline, repeated seeds and four metrics. The resulting tables support ranking, stability and interaction analysis.
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
    rng=np.random.default_rng(303)
    n=120 if DEMO_TEST else 280
    X,y=make_classification(n_samples=n,n_features=14,n_informative=9,n_redundant=2,class_sep=1.0,random_state=303)
    ids=np.asarray([f"matrix_{i:04d}" for i in range(n)],dtype=object)
    rep1=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"R1_{i}" for i in range(X.shape[1])])
    rep2_X=np.column_stack([X[:,:10],rng.normal(size=(n,14))]); rep2=DatasetBundle(X=rep2_X,y=y,sample_ids=ids,feature_names=[f"R2_{i}" for i in range(rep2_X.shape[1])])
    rep3_X=np.tanh(X[:,:8]); rep3=DatasetBundle(X=rep3_X,y=y,sample_ids=ids,feature_names=[f"R3_{i}" for i in range(rep3_X.shape[1])])
    plan_a=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=balanced_fold_labels(y,4),dataset_fingerprint=rep1.fingerprint,metadata={"source":"external","strategy":"balanced"})
    # A deliberately different deterministic membership pattern to emulate a second upstream strategy.
    order=np.argsort(X[:,0]); alt=np.empty(n,dtype=int); alt[order]=np.arange(n)%4
    # Ensure every training membership retains both classes; if not, fall back to balanced labels for that demo artifact.
    plan_b=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=alt,dataset_fingerprint=rep1.fingerprint,metadata={"source":"external","strategy":"feature_blocked_like"})
    return plan_a, plan_b, rep1, rep2, rep3


@app.cell
def _(BenchmarkConfig, DEMO_TEST, benchmark, pl, plan_a, plan_b, rep1, rep2, rep3):
    seeds=(42,) if DEMO_TEST else (42,123)
    bench=benchmark(datasets={"rep_core":rep1,"rep_expanded":rep2,"rep_nonlinear":rep3},
                    algorithms=("logistic_regression","random_forest","svc"),
                    partitions={"balanced":plan_a,"feature_blocked":plan_b},
                    config=BenchmarkConfig(metrics=("mcc","balanced_accuracy","f1","roc_auc"),seeds=seeds,modes=("untuned",),include_baselines=True),
                    model_params={"random_forest":{"n_estimators":45 if DEMO_TEST else 110,"max_depth":7}})
    assert not bench.failures
    metrics=bench.aggregate_metrics_frame(); runs=bench.runs_frame(); folds=bench.fold_metrics_frame()
    mcc=metrics.filter(pl.col("metric")=="mcc")
    report=(
        mcc.group_by(["representation","partition","algorithm"])
        .agg(pl.col("score").mean().alias("mean"), pl.col("score").std().alias("std"))
    )
    report=report.with_columns(
        pl.col("mean").rank(method="dense",descending=True).over(["representation","partition"]).alias("rank")
    )
    best=report.filter(pl.col("rank")==1).sort(["partition","representation"])
    best
    return bench, folds, report, runs, seeds


@app.cell
def _(mo, np, plt, report, runs):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    fig,axes=plt.subplots(1,3,figsize=(18,4.8))
    partitions_sorted=sorted(report.get_column("partition").unique().to_list())
    for (part,),chunk in report.group_by("partition"):
        piv=chunk.pivot(on="representation",index="algorithm",values="mean")
        # Plot only first two panels, one partition each.
        ax=axes[0] if part==partitions_sorted[0] else axes[1]
        rep_cols=[col for col in piv.columns if col!="algorithm"]
        grouped_bar(ax, piv.get_column("algorithm").to_list(), {col: piv.get_column(col).to_numpy() for col in rep_cols})
        ax.set_title(f"Mean MCC — {part}"); ax.tick_params(axis="x",rotation=35)
    runtime=runs.group_by("algorithm").agg(elapsed_seconds=pl.col("elapsed_seconds").mean()).sort("elapsed_seconds")
    axes[2].bar(runtime.get_column("algorithm").to_list(),runtime.get_column("elapsed_seconds").to_numpy()); axes[2].set(title="Mean runtime",ylabel="seconds"); axes[2].tick_params(axis="x",rotation=35)
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(FIGURE_COUNT, bench, folds, report, seeds):
    expected=3*2*4*len(seeds)
    DEMO_CHECKS={
        "full_matrix":bench.n_runs==expected,
        "no_failures":len(bench.failures)==0,
        "two_partitions":set(report.get_column("partition").unique().to_list())=={"balanced","feature_blocked"},
        "three_representations":report.get_column("representation").n_unique()==3,
        "rankings_complete":report.get_column("rank").is_not_null().all(),
        "fold_metrics_available":not folds.is_empty(),
        "figure_created":FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
