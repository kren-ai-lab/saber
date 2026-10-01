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
    # End-to-end study — prepared representations, tuning and protected-test benchmark
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Scientific question
    How much do representation choice and hyperparameter optimization change performance on a **protected external test set**? We compare two prepared representations, two classical models, untuned and tuned modes, repeated seeds, and a dummy baseline. Tuning is confined to train/validation; all reported benchmark scores come from the protected test.
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
    from saber import benchmark, BenchmarkConfig, SearchSpace, Categorical, DatasetBundle, PartitionPlan, TuningConfig

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
    rng=np.random.default_rng(2026)
    n_samples=150 if DEMO_TEST else 360
    X,y=make_classification(n_samples=n_samples,n_features=14,n_informative=9,n_redundant=2,weights=[.6,.4],class_sep=1.05,random_state=2026)
    ids=np.asarray([f"study_{i:04d}" for i in range(len(y))],dtype=object)
    rep_a=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"A_{i}" for i in range(X.shape[1])])
    rep_b_X=np.column_stack([X[:,:9],0.35*X[:,0:3]+rng.normal(scale=.4,size=(len(y),3)),rng.normal(size=(len(y),8))])
    rep_b=DatasetBundle(X=rep_b_X,y=y,sample_ids=ids,feature_names=[f"B_{i}" for i in range(rep_b_X.shape[1])])
    train_ids=[]; val_ids=[]; test_ids=[]
    for cls in np.unique(y):
        cls_ids=ids[y==cls]; n=len(cls_ids); a,b=int(.60*n),int(.80*n)
        train_ids.extend(cls_ids[:a]); val_ids.extend(cls_ids[a:b]); test_ids.extend(cls_ids[b:])
    plan=PartitionPlan.holdout(train_ids=train_ids,validation_ids=val_ids,test_ids=test_ids,dataset=rep_a,
                               metadata={"source":"prepared_holdout","test_role":"protected"})
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
    seeds=(42,) if DEMO_TEST else (42,123)
    tuning=TuningConfig(optimizer="grid",refit_metric="mcc",n_jobs=1)
    search_spaces={
        "logistic_regression":SearchSpace({"C":Categorical([0.1,1.0,3.0]),"solver":Categorical(["lbfgs"])}),
        "random_forest_classifier":SearchSpace({"n_estimators":Categorical([40,80] if DEMO_TEST else [80,160]),"max_depth":Categorical([4,8])}),
    }
    bench=benchmark(
        datasets={"representation_A":rep_a,"representation_B":rep_b},
        algorithms=("logistic_regression","random_forest_classifier"), partitions={"development_plus_protected_test":plan},
        metrics=("mcc","balanced_accuracy","f1","roc_auc"),config=BenchmarkConfig(seeds=seeds,modes=("untuned","tuned"),
                               include_baselines=True,tuning=tuning),
        search_spaces=search_spaces,
    )
    assert not bench.failures
    runs=bench.runs_frame(); metrics=bench.aggregate_metrics_frame(); history=bench.optimization_history_frame(); preds=bench.predictions_frame()
    leader=(
        metrics.filter(pl.col("metric")=="mcc")
        .group_by(["representation","algorithm","mode"])
        .agg(pl.col("score").mean().alias("mean"), pl.col("score").std().alias("std"))
    )
    leader.sort("mean",descending=True).head(10)
    return bench, history, leader, metrics, preds, runs, seeds


@app.cell
def _(history, leader, mo, np, pl, plt, runs):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    fig,axes=plt.subplots(1,3,figsize=(17,4.5))
    plot=leader.with_columns((pl.col("algorithm")+" / "+pl.col("mode")).alias("label"))
    piv=plot.pivot(on="representation",index="label",values="mean")
    rep_cols=[col for col in piv.columns if col!="label"]
    grouped_bar(axes[0], piv.get_column("label").to_list(), {col: piv.get_column(col).to_numpy() for col in rep_cols})
    axes[0].set(title="Protected-test MCC",ylabel="MCC"); axes[0].tick_params(axis="x",rotation=45)
    if not history.is_empty():
        h=history.filter(pl.col("status")=="complete")
        counts=h.group_by("algorithm").agg(pl.len().alias("count"))
        axes[1].bar(counts.get_column("algorithm").to_list(),counts.get_column("count").to_numpy()); axes[1].set(title="Tuning candidates evaluated",ylabel="count")
    runtime=runs.group_by(["algorithm","mode"]).agg(pl.col("elapsed_seconds").mean().alias("mean"))
    runtime=runtime.with_columns((pl.col("algorithm")+" / "+pl.col("mode")).alias("label"))
    axes[2].bar(runtime.get_column("label").to_list(),runtime.get_column("mean").to_numpy()); axes[2].set(title="Workflow runtime",ylabel="seconds"); axes[2].tick_params(axis="x",rotation=45)
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(FIGURE_COUNT, bench, history, leader, metrics, preds, runs, seeds):
    nonbaseline={"logistic_regression","random_forest_classifier"}
    expected=len(seeds)*2*(1+2*len(nonbaseline))
    DEMO_CHECKS={
        "benchmark_matrix":bench.n_runs==expected,
        "no_failures":len(bench.failures)==0,
        "tuned_runs_have_history":not history.is_empty(),
        "protected_test_only":set(preds.get_column("evaluation_role").unique().to_list())=={"test"},
        "all_metrics":set(metrics.get_column("metric").unique().to_list())=={"mcc","balanced_accuracy","f1","roc_auc"},
        "run_prediction_traceability":set(preds.get_column("run_id"))==set(runs.get_column("run_id")),
        "leaderboard_available":not leader.is_empty(),
        "figure_created":FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
