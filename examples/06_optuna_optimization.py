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
    # Optuna optimization: typed continuous spaces and trial diagnostics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    A seeded Optuna study searches log-scaled `C` and `gamma` for an RBF SVC. The notebook shows convergence, the explored region of the search space, the top trials and the secondary metrics of the best one. Probability outputs stay off for the SVC, so ROC AUC is computed from the decision function.
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
    from saber import tune, SearchSpace, LogFloat, Categorical, DatasetBundle, PartitionPlan, TuningConfig

    return (
        Categorical,
        DEMO_TEST,
        DatasetBundle,
        LogFloat,
        PartitionPlan,
        SearchSpace,
        TuningConfig,
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
    make_classification,
    np,
):
    n_samples = 140 if DEMO_TEST else 320
    X, y = make_classification(
        n_samples=n_samples, n_features=12, n_informative=8, class_sep=1.05, random_state=54
    )
    ids = [f"opt_{i:04d}" for i in range(len(y))]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"f{i}" for i in range(X.shape[1])])
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids, fold_assignments=np.arange(len(y)) % 4, dataset=dataset
    )
    space = SearchSpace(
        {"C": LogFloat(1e-2, 30.0), "gamma": LogFloat(1e-4, 1.0), "kernel": Categorical(["rbf"])}
    )
    return dataset, plan, space


@app.cell
def _(DEMO_TEST, TuningConfig, dataset, pl, plan, space, tune):
    n_trials = 8 if DEMO_TEST else 28
    config = TuningConfig(optimizer="optuna", refit_metric="mcc", n_trials=n_trials, n_jobs=1)
    opt = tune(
        dataset=dataset,
        algorithm="svc",
        config=config,
        partition_plan=plan,
        search_space=space,
        metrics=("mcc", "roc_auc", "balanced_accuracy"),
        random_state=123,
    )
    history = opt.history_frame()
    completed = history.filter(pl.col("status") == "complete")
    score_col = "metric__mcc"
    completed = completed.with_columns(best_so_far=pl.col(score_col).cum_max())
    top = completed.sort(score_col, descending=True).head(5)
    print("Best:", opt.best_params, opt.best_scores)
    top.select(["param__C", "param__gamma", score_col]).head()
    return completed, history, n_trials, opt, score_col


@app.cell
def _(completed, mo, opt, plt, score_col):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(completed[score_col], "o-", alpha=0.6, label="trial")
    axes[0].plot(completed["best_so_far"], label="best so far")
    axes[0].set(title="Optimization convergence", xlabel="trial", ylabel="MCC")
    axes[0].legend()
    sc = axes[1].scatter(completed["param__C"], completed["param__gamma"], c=completed[score_col])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(title="Explored search space", xlabel="C", ylabel="gamma")
    fig.colorbar(sc, ax=axes[1], label="MCC")
    secondary_scores = {k: v for k, v in opt.best_scores.items() if k != "mcc"}
    axes[2].bar(list(secondary_scores), list(secondary_scores.values()))
    axes[2].set(title="Secondary metrics for selected trial", ylabel="score", ylim=(0, 1.05))
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(history, n_trials, opt):
    DEMO_CHECKS = {
        "requested_trials": len(history) == n_trials,
        "all_complete": (history.get_column("status") == "complete").all(),
        "study_available": opt.study is not None,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
