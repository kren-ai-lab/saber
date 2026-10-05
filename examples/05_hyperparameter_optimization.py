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
    # Grid search: selection report with a protected final test
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    A grid search over `C` and `class_weight` selects hyperparameters on a train/validation split and scores each candidate on several metrics. The notebook ranks the candidates, then compares the untuned and tuned models on a final test set that the search never sees.
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
    from saber import tune, validate, SearchSpace, Categorical, DatasetBundle, PartitionPlan, TuningConfig

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
        plt,
        tune,
        validate,
    )


@app.cell
def _(DEMO_TEST, DatasetBundle, PartitionPlan, make_classification, np):
    n_samples = 180 if DEMO_TEST else 360
    X, y = make_classification(
        n_samples=n_samples,
        n_features=14,
        n_informative=9,
        weights=[0.62, 0.38],
        class_sep=1.0,
        random_state=31,
    )
    ids = np.asarray([f"tune_{i:04d}" for i in range(len(y))], dtype=object)
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"f{i}" for i in range(X.shape[1])])
    train_ids, val_ids, test_ids = np.split(ids, [int(0.6 * len(ids)), int(0.8 * len(ids))])
    plan = PartitionPlan.holdout(
        train_ids=train_ids, validation_ids=val_ids, test_ids=test_ids, dataset=dataset
    )
    return dataset, plan, test_ids, train_ids, val_ids


@app.cell
def _(Categorical, SearchSpace, TuningConfig, dataset, pl, plan, tune):
    space = SearchSpace(
        {
            "C": Categorical([0.03, 0.1, 0.3, 1.0, 3.0]),
            "class_weight": Categorical([None, "balanced"]),
            "solver": Categorical(["lbfgs"]),
        }
    )
    config = TuningConfig(optimizer="grid", refit_metric="mcc", n_jobs=1)
    opt = tune(
        dataset=dataset,
        algorithm="logistic_regression",
        config=config,
        partition_plan=plan,
        search_space=space,
        metrics=("mcc", "roc_auc", "balanced_accuracy"),
        random_state=42,
    )
    history = opt.history_frame()
    complete = history.filter(pl.col("status") == "complete")
    score_col = "metric__mcc__mean"
    top = complete.sort(score_col, descending=True).head(5)
    print("Best parameters:", opt.best_params)
    top.select(["param__C", "param__class_weight", score_col]).head()
    return complete, opt, score_col


@app.cell
def _(PartitionPlan, dataset, opt, pl, test_ids, train_ids, val_ids, validate):
    # Baseline and tuned configurations are both evaluated only on the protected test.
    final_plan = PartitionPlan.holdout(
        train_ids=(*train_ids, *val_ids), test_ids=test_ids, dataset=dataset, name="protected_test"
    )
    final = {
        name: validate(
            dataset=dataset,
            algorithm="logistic_regression",
            partition_plan=final_plan,
            metrics=("mcc", "roc_auc", "balanced_accuracy"),
            evaluation_role="test",
            random_state=42,
            model_params=params,
        )
        for name, params in (("untuned", None), ("tuned", opt.best_params))
    }
    comparison = pl.DataFrame([{"config": name, **res.aggregate_metrics} for name, res in final.items()])
    tuned = final["tuned"]
    comparison
    return comparison, tuned


@app.cell
def _(comparison, complete, mo, opt, plt, score_col, tuned):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for (class_weight,), part in complete.sort("param__C").group_by(
        "param__class_weight", maintain_order=True
    ):
        axes[0].plot(part["param__C"], part[score_col], marker="o", label=str(class_weight))
    axes[0].set_xscale("log")
    axes[0].set(title="Grid response", xlabel="C", ylabel="validation MCC")
    axes[0].legend()
    for row in comparison.iter_rows(named=True):
        axes[1].plot(
            ["mcc", "balanced_accuracy", "roc_auc"],
            [row["mcc"], row["balanced_accuracy"], row["roc_auc"]],
            "o",
            label=row["config"],
        )
    axes[1].legend()
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Protected test")
    axes[2].bar(["search", "protected test"], [opt.best_scores["mcc"], tuned.aggregate_metrics["mcc"]])
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Selection vs final estimate")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(complete, np, test_ids, tuned):
    DEMO_CHECKS = {
        "test_protected": set(tuned.oof_prediction.sample_ids) == set(test_ids),
        "ten_candidates": len(complete) == 10,
        "finite_final": np.isfinite(tuned.aggregate_metrics["mcc"]),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
