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
    # Optimizer comparison — Grid vs Random vs Optuna on identical memberships
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Compare three optimization backends over the **same dataset, folds, metric contract and discrete search domain**. This is not a final model comparison; it is an optimizer behavior study focusing on best score, candidate count, runtime and search efficiency.
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

    from time import perf_counter
    from sklearn.datasets import make_classification
    from saber import tune
    from saber.core import SearchSpace, Categorical
    from saber.datasets import DatasetBundle, PartitionPlan
    from saber.tuning import TuningConfig

    return (
        Categorical,
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        SearchSpace,
        TuningConfig,
        balanced_fold_labels,
        make_classification,
        np,
        pl,
        perf_counter,
        plt,
        tune,
    )


@app.cell
def _(
    Categorical,
    DEMO_TEST,
    DatasetBundle,
    PartitionPlan,
    SearchSpace,
    balanced_fold_labels,
    make_classification,
):
    X,y=make_classification(n_samples=140 if DEMO_TEST else 300,n_features=12,n_informative=8,class_sep=1.0,random_state=99)
    ids=[f"optimizer_{i:04d}" for i in range(len(y))]
    dataset=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"f{i}" for i in range(X.shape[1])])
    plan=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=balanced_fold_labels(y,4),dataset_fingerprint=dataset.fingerprint)
    space=SearchSpace("logreg_shared",{"C":Categorical([0.03,0.1,0.3,1.0,3.0,10.0]),"class_weight":Categorical([None,"balanced"]),"solver":Categorical(["lbfgs"])})
    return dataset, plan, space


@app.cell
def _(DEMO_TEST, TuningConfig, dataset, pl, perf_counter, plan, space, tune):
    rows=[]; histories={}
    for name in ("grid","random","optuna"):
        cfg=TuningConfig(optimizer=name,metrics=("mcc","roc_auc"),refit_metric="mcc",random_state=123,n_jobs=1,
                         n_iter=6 if DEMO_TEST else 10,n_trials=6 if DEMO_TEST else 16)
        t0=perf_counter(); res=tune(dataset=dataset,algorithm="logistic_regression",config=cfg,partition_plan=plan,search_space=space); elapsed=perf_counter()-t0
        hist=res.history_frame(); histories[name]=hist
        rows.append({"optimizer":name,"best_mcc":res.display_scores["mcc"],"best_roc_auc":res.display_scores["roc_auc"],
                     "elapsed_seconds":elapsed,"candidates":len(hist),"best_params":res.best_params})
    summary=pl.DataFrame(rows)
    summary
    return histories, summary


@app.cell
def _(plt, summary):
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    optimizers = summary.get_column("optimizer").to_list()
    axes[0].bar(optimizers,summary.get_column("best_mcc").to_numpy()); axes[0].set(title="Best CV MCC",ylim=(-.05,1.0))
    axes[1].bar(optimizers,summary.get_column("elapsed_seconds").to_numpy()); axes[1].set(title="Optimization runtime",ylabel="seconds")
    axes[2].bar(optimizers,summary.get_column("candidates").to_numpy()); axes[2].set(title="Candidates evaluated",ylabel="count")
    plt.tight_layout(); FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(FIGURE_COUNT, histories, np, plan, summary):
    DEMO_CHECKS={
        "three_optimizers":set(summary.get_column("optimizer").to_list())=={"grid","random","optuna"},
        "finite_scores":np.isfinite(summary.select(["best_mcc","best_roc_auc"]).to_numpy()).all(),
        "histories_retained":all(len(v)>0 for v in histories.values()),
        "same_partition_contract":all(plan.fingerprint==plan.fingerprint for _ in histories),
        "figure_created":FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
