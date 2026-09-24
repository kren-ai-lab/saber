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
    # Optuna optimization — typed continuous spaces and trial diagnostics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Run a reproducible typed Optuna study over continuous/log-scaled SVC parameters. We inspect convergence, explored parameter space, trial ranking and secondary metrics. SVC probabilities are intentionally **not** enabled; ROC-AUC uses the decision function.
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
    from saber import tune
    from saber.core import SearchSpace, LogFloat, Categorical
    from saber.datasets import DatasetBundle, PartitionPlan
    from saber.tuning import TuningConfig

    return (
        Categorical,
        DEMO_TEST,
        DatasetBundle,
        LogFloat,
        PartitionPlan,
        SearchSpace,
        TuningConfig,
        balanced_fold_labels,
        make_classification,
        np,
        pl,
        plt,
        tune,
    )


@app.cell
def _(
    Categorical,
    DEMO_TEST,
    DatasetBundle,
    LogFloat,
    PartitionPlan,
    SearchSpace,
    balanced_fold_labels,
    make_classification,
):
    n_samples=140 if DEMO_TEST else 320
    X,y=make_classification(n_samples=n_samples,n_features=12,n_informative=8,class_sep=1.05,random_state=54)
    ids=[f"opt_{i:04d}" for i in range(len(y))]
    dataset=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"f{i}" for i in range(X.shape[1])])
    plan=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=balanced_fold_labels(y,4),dataset_fingerprint=dataset.fingerprint)
    space=SearchSpace("svc_optuna",{"C":LogFloat(1e-2,30.0),"gamma":LogFloat(1e-4,1.0),"kernel":Categorical(["rbf"])})
    return dataset, plan, space


@app.cell
def _(DEMO_TEST, TuningConfig, dataset, pl, plan, space, tune):
    n_trials=8 if DEMO_TEST else 28
    config=TuningConfig(optimizer="optuna",metrics=("mcc","roc_auc","balanced_accuracy"),refit_metric="mcc",n_trials=n_trials,random_state=123,n_jobs=1)
    opt=tune(dataset=dataset,algorithm="svc",config=config,partition_plan=plan,search_space=space)
    history=opt.history_frame(); completed=history.filter(pl.col("status")=="complete")
    score_col=next(c for c in completed.columns if c in {"metric__mcc","metric__mcc__mean"})
    completed=completed.with_columns(pl.col(score_col).cum_max().alias("best_so_far"))
    top=completed.sort(score_col,descending=True).head(min(5,len(completed)))
    print("Best:",opt.best_params,opt.display_scores)
    top.select(["param__C","param__gamma",score_col]).head()
    return completed, history, n_trials, opt, score_col


@app.cell
def _(completed, np, opt, plt, score_col):
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    scores = completed.get_column(score_col).to_numpy()
    best_so_far = completed.get_column("best_so_far").to_numpy()
    axes[0].plot(np.arange(len(completed)),scores,"o-",alpha=.6,label="trial"); axes[0].plot(best_so_far,label="best so far"); axes[0].set(title="Optimization convergence",xlabel="trial",ylabel="MCC"); axes[0].legend()
    sc=axes[1].scatter(completed.get_column("param__C").to_numpy(),completed.get_column("param__gamma").to_numpy(),c=scores); axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set(title="Explored search space",xlabel="C",ylabel="gamma"); fig.colorbar(sc,ax=axes[1],label="MCC")
    secondary_scores={k:v for k,v in opt.display_scores.items() if k!="mcc"}
    axes[2].bar(list(secondary_scores.keys()),list(secondary_scores.values()))
    axes[2].set(title="Secondary metrics for selected trial",ylabel="score",ylim=(0,1.05))
    plt.tight_layout(); FIGURE_COUNT=1
    return FIGURE_COUNT, secondary_scores


@app.cell
def _(FIGURE_COUNT, completed, history, n_trials, np, opt, secondary_scores):
    _best_so_far = completed.get_column("best_so_far").to_numpy()
    DEMO_CHECKS={
        "requested_trials": len(history)==n_trials,
        "all_complete": (history.get_column("status")=="complete").all(),
        "finite_best": np.isfinite(opt.best_score),
        "study_available": opt.study is not None,
        "typed_parameters": {"param__C","param__gamma"}.issubset(history.columns),
        "monotonic_best": bool(np.all(np.diff(_best_so_far)>=0)),
        "secondary_metrics": len(secondary_scores)>=2,
        "figure_created": FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
