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
    # Optimizer comparison: grid vs random vs Optuna on the same folds
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Grid, random and Optuna search run on the same dataset, folds, metrics and discrete search space. The comparison looks at the optimizers themselves: best score, number of candidates and runtime. It does not choose a final model.
    """)
    return


@app.cell
def _():
    import os

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"

    from time import perf_counter
    from sklearn.datasets import make_classification
    from saber import tune, SearchSpace, Categorical, DatasetBundle, PartitionPlan, TuningConfig

    return (
        Categorical,
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        SearchSpace,
        TuningConfig,
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
    make_classification,
    np,
):
    X, y = make_classification(
        n_samples=140 if DEMO_TEST else 300, n_features=12, n_informative=8, class_sep=1.0, random_state=99
    )
    ids = [f"optimizer_{i:04d}" for i in range(len(y))]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"f{i}" for i in range(X.shape[1])])
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=ids, fold_assignments=np.arange(len(y)) % 4, dataset=dataset
    )
    space = SearchSpace(
        {
            "C": Categorical([0.03, 0.1, 0.3, 1.0, 3.0, 10.0]),
            "class_weight": Categorical([None, "balanced"]),
            "solver": Categorical(["lbfgs"]),
        }
    )
    return dataset, plan, space


@app.cell
def _(DEMO_TEST, TuningConfig, dataset, pl, perf_counter, plan, space, tune):
    rows = []
    for name in ("grid", "random", "optuna"):
        cfg = TuningConfig(
            optimizer=name,
            refit_metric="mcc",
            n_jobs=1,
            n_iter=6 if DEMO_TEST else 10,
            n_trials=6 if DEMO_TEST else 16,
        )
        t0 = perf_counter()
        res = tune(
            dataset=dataset,
            algorithm="logistic_regression",
            config=cfg,
            partition_plan=plan,
            search_space=space,
            metrics=("mcc", "roc_auc"),
            random_state=123,
        )
        rows.append(
            {
                "optimizer": name,
                "best_mcc": res.best_scores["mcc"],
                "best_roc_auc": res.best_scores["roc_auc"],
                "elapsed_seconds": perf_counter() - t0,
                "candidates": len(res.history),
                "best_params": res.best_params,
            }
        )
    summary = pl.DataFrame(rows)
    summary
    return (summary,)


@app.cell
def _(mo, plt, summary):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(summary["optimizer"], summary["best_mcc"])
    axes[0].set(title="Best CV MCC", ylim=(-0.05, 1.0))
    axes[1].bar(summary["optimizer"], summary["elapsed_seconds"])
    axes[1].set(title="Optimization runtime", ylabel="seconds")
    axes[2].bar(summary["optimizer"], summary["candidates"])
    axes[2].set(title="Candidates evaluated", ylabel="count")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(np, summary):
    DEMO_CHECKS = {
        "finite_scores": np.isfinite(summary.select(["best_mcc", "best_roc_auc"]).to_numpy()).all(),
        "candidates_evaluated": (summary["candidates"] > 0).all(),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
