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
    # BioSieve integration — live partitioning, provenance and fold diagnostics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    Exercise the canonical partition boundary. If BioSieve is installed, `saber` delegates partition generation to BioSieve and preserves provenance. If the optional dependency is absent, this executable documentation uses a clearly labeled pre-partitioned fallback; **there is no internal saber fallback splitter**.
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
    from saber import validate
    from saber.datasets import DatasetBundle, PartitionPlan, BioSievePartitionConfig
    from saber.exceptions import OptionalDependencyError

    return (
        BioSievePartitionConfig,
        DEMO_TEST,
        DatasetBundle,
        OptionalDependencyError,
        PartitionPlan,
        balanced_fold_labels,
        make_classification,
        np,
        pl,
        plt,
        validate,
    )


@app.cell
def _(
    BioSievePartitionConfig,
    DEMO_TEST,
    DatasetBundle,
    OptionalDependencyError,
    PartitionPlan,
    balanced_fold_labels,
    make_classification,
    validate,
):
    n_samples=120 if DEMO_TEST else 300
    X,y=make_classification(n_samples=n_samples,n_features=11,n_informative=7,weights=[.65,.35],class_sep=1.0,random_state=71)
    ids=[f"bio_{i:04d}" for i in range(len(y))]
    dataset=DatasetBundle(X=X,y=y,sample_ids=ids,feature_names=[f"f{i}" for i in range(X.shape[1])])
    LIVE_BIOSIEVE=False
    try:
        result=validate(dataset=dataset,algorithm="logistic_regression",
                        partitioning=BioSievePartitionConfig(strategy="stratified_kfold",params={"n_splits":5,"seed":42}),
                        metrics=("mcc","balanced_accuracy","f1","roc_auc","pr_auc"),random_state=42)
        LIVE_BIOSIEVE=True
    except OptionalDependencyError:
        plan=PartitionPlan.from_predefined_folds(sample_ids=ids,fold_assignments=balanced_fold_labels(y,5),
            dataset_fingerprint=dataset.fingerprint,metadata={"source":"demo_prepartitioned_fallback","strategy":"stratified_like_5fold"})
        result=validate(dataset=dataset,algorithm="logistic_regression",partition_plan=plan,
                        metrics=("mcc","balanced_accuracy","f1","roc_auc","pr_auc"),random_state=42)
    print("Live BioSieve:",LIVE_BIOSIEVE)
    print("Partition metadata:",result.partition_plan.metadata)
    return LIVE_BIOSIEVE, dataset, result, y


@app.cell
def _(dataset, np, pl, result):
    fold_rows=[]
    for fold in result.folds:
        eval_y=dataset.subset(fold.evaluation_ids).y
        fold_rows.append({"fold":fold.split_name,"n_eval":len(eval_y),"positive_rate":float(np.mean(eval_y)),**fold.evaluation.metrics})
    fold_df=pl.DataFrame(fold_rows)
    fold_df
    return (fold_df,)


@app.cell
def _(fold_df, mo, np, plt, y):
    def grouped_bar(ax, categories, series):
        x = np.arange(len(categories))
        width = 0.8 / len(series)
        for i, (label, values) in enumerate(series.items()):
            ax.bar(x + i * width, values, width, label=label)
        ax.set_xticks(x + width * (len(series) - 1) / 2, categories)
        ax.legend()

    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    fold_cols = ["mcc", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]
    fold_names = fold_df.get_column("fold").to_list()
    grouped_bar(axes[0], fold_names, {col: fold_df.get_column(col).to_numpy() for col in fold_cols})
    axes[0].set_ylim(-.05,1.05); axes[0].set_title("Fold performance")
    axes[1].bar(fold_names,fold_df.get_column("positive_rate").to_numpy()); axes[1].axhline(np.mean(y),linestyle="--"); axes[1].set(title="Class balance per held-out fold",ylabel="positive rate")
    axes[2].bar(fold_names,fold_df.get_column("n_eval").to_numpy()); axes[2].set(title="Held-out fold size",ylabel="samples")
    for ax in axes: ax.tick_params(axis="x",rotation=35)
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    FIGURE_COUNT=1
    return (FIGURE_COUNT,)


@app.cell
def _(FIGURE_COUNT, LIVE_BIOSIEVE, fold_df, np, result):
    DEMO_CHECKS={
        "five_splits":result.n_splits==5,
        "complete_oof":result.metadata["oof_complete"] is True,
        "source_explicit":result.partition_plan.metadata.get("source") in {"biosieve","demo_prepartitioned_fallback"},
        "fold_diagnostics":len(fold_df)==5,
        "finite_metrics":np.isfinite(fold_df.select(["mcc","balanced_accuracy","f1","roc_auc","pr_auc"]).to_numpy()).all(),
        "figure_created":FIGURE_COUNT==1,
    }
    if LIVE_BIOSIEVE: DEMO_CHECKS["live_source_is_biosieve"]=result.partition_plan.metadata.get("source")=="biosieve"
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
