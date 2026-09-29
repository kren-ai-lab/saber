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
    # Grid optimization — selection report with protected final test
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Demonstrate a safe train/validation/test optimization study with multiple metrics, a two-dimensional search space, explicit candidate ranking, untuned-vs-tuned comparison and a protected final test that is never used during hyperparameter selection.
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
    from saber import tune, validate
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
        make_classification,
        np,
        pl,
        plt,
        tune,
        validate,
    )


@app.cell
def _(DEMO_TEST, DatasetBundle, PartitionPlan, make_classification, np):
    n_samples=180 if DEMO_TEST else 360
    X,y=make_classification(n_samples=n_samples,n_features=14,n_informative=9,weights=[.62,.38],class_sep=1.0,random_state=31)
    ids=np.asarray([f"tune_{i:04d}" for i in range(len(y))],dtype=object)
    dataset=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"f{i}" for i in range(X.shape[1])])
    train_ids=[]; val_ids=[]; test_ids=[]
    for cls in np.unique(y):
        cls_ids=ids[y==cls]; n=len(cls_ids); a,b=int(.60*n),int(.80*n)
        train_ids.extend(cls_ids[:a]); val_ids.extend(cls_ids[a:b]); test_ids.extend(cls_ids[b:])
    plan=PartitionPlan.holdout(train_ids=train_ids,validation_ids=val_ids,test_ids=test_ids,dataset_fingerprint=dataset.fingerprint)
    return dataset, plan, test_ids, train_ids, val_ids


@app.cell
def _(Categorical, SearchSpace, TuningConfig, dataset, pl, plan, tune):
    space=SearchSpace("logreg_grid",{
        "C":Categorical([0.03,0.1,0.3,1.0,3.0]),
        "class_weight":Categorical([None,"balanced"]),
        "solver":Categorical(["lbfgs"]),
    })
    config=TuningConfig(optimizer="grid",metrics=("mcc","roc_auc","balanced_accuracy"),refit_metric="mcc",random_state=42,n_jobs=1)
    opt=tune(dataset=dataset,algorithm="logistic_regression",config=config,partition_plan=plan,search_space=space)
    history=opt.history_frame(); complete=history.filter(pl.col("status")=="complete")
    score_col="metric__mcc__mean"
    top=complete.sort(score_col,descending=True).head(5)
    print("Best parameters:",opt.best_params)
    top.select(["param__C","param__class_weight",score_col]).head()
    return complete, opt, score_col, top


@app.cell
def _(PartitionPlan, dataset, opt, pl, test_ids, train_ids, val_ids, validate):
    # Baseline and tuned configurations are both evaluated only on the protected test.
    final_plan=PartitionPlan.holdout(train_ids=tuple(train_ids)+tuple(val_ids),test_ids=test_ids,
                                     dataset_fingerprint=dataset.fingerprint,name="protected_test")
    untuned=validate(dataset=dataset,algorithm="logistic_regression",partition_plan=final_plan,
                     metrics=("mcc","roc_auc","balanced_accuracy"),evaluation_role="test",random_state=42)
    tuned=validate(dataset=dataset,algorithm="logistic_regression",partition_plan=final_plan,
                   metrics=("mcc","roc_auc","balanced_accuracy"),evaluation_role="test",random_state=42,
                   model_params=opt.best_params)
    comparison=pl.DataFrame([
        {"config":"untuned",**untuned.aggregate_metrics},
        {"config":"tuned",**tuned.aggregate_metrics},
    ])
    comparison
    return comparison, tuned


@app.cell
def _(comparison, complete, mo, np, opt, pl, plt, score_col, tuned):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for cw,label in [(None,"None"),("balanced","balanced")]:
        part = complete.filter(pl.col("param__class_weight").is_null()) if cw is None else complete.filter(pl.col("param__class_weight")==cw)
        if len(part): axes[0].plot(part.get_column("param__C").to_numpy(),part.get_column(score_col).to_numpy(),marker="o",label=label)
    axes[0].set_xscale("log"); axes[0].set(title="Grid response",xlabel="C",ylabel="validation MCC"); axes[0].legend()
    grouped_bar(
        axes[1],
        comparison.get_column("config").to_list(),
        {metric: comparison.get_column(metric).to_numpy() for metric in ("mcc","balanced_accuracy","roc_auc")},
    )
    axes[1].set_ylim(-.05,1.05); axes[1].set_title("Protected test")
    axes[2].bar(["search","protected test"],[opt.display_scores["mcc"],tuned.aggregate_metrics["mcc"]]); axes[2].set_ylim(-.05,1.05); axes[2].set_title("Selection vs final estimate")
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(FIGURE_COUNT, comparison, complete, np, opt, test_ids, top, tuned):
    DEMO_CHECKS={
        "test_protected": opt.metadata["protected_samples"]==len(test_ids),
        "ten_candidates": len(complete)==10,
        "multi_metric": set(opt.metrics)=={"mcc","roc_auc","balanced_accuracy"},
        "top_report": len(top)==5,
        "protected_comparison": set(comparison.get_column("config").to_list())=={"untuned","tuned"},
        "finite_final": np.isfinite(tuned.aggregate_metrics["mcc"]),
        "figure_created": FIGURE_COUNT==1,
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
