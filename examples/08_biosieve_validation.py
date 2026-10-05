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
    # BioSieve integration: live partitioning, provenance and fold diagnostics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    When BioSieve is installed, `saber` hands partition generation to it and keeps BioSieve's provenance in the partition metadata. Without that optional dependency, the notebook uses precomputed fold memberships labeled `demo_prepartitioned_fallback`. `saber` has no splitter of its own to fall back on.
    """)
    return


@app.cell
def _():
    import os
    from importlib.util import find_spec

    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt

    DEMO_TEST = os.getenv("SABER_DEMO_TEST") == "1"
    LIVE_BIOSIEVE = find_spec("biosieve") is not None

    from sklearn.datasets import make_classification
    from saber import validate, DatasetBundle, PartitionPlan, BioSievePartitionConfig

    return (
        BioSievePartitionConfig,
        DEMO_TEST,
        DatasetBundle,
        LIVE_BIOSIEVE,
        PartitionPlan,
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
    LIVE_BIOSIEVE,
    PartitionPlan,
    make_classification,
    np,
    validate,
):
    n_samples = 120 if DEMO_TEST else 300
    X, y = make_classification(
        n_samples=n_samples,
        n_features=11,
        n_informative=7,
        weights=[0.65, 0.35],
        class_sep=1.0,
        random_state=71,
    )
    ids = [f"bio_{i:04d}" for i in range(len(y))]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"f{i}" for i in range(X.shape[1])])
    if LIVE_BIOSIEVE:
        partition = BioSievePartitionConfig(strategy="stratified_kfold", params={"n_splits": 5, "seed": 42})
    else:
        partition = PartitionPlan.from_predefined_folds(
            sample_ids=ids,
            fold_assignments=np.arange(len(y)) % 5,
            dataset=dataset,
            metadata={"source": "demo_prepartitioned_fallback", "strategy": "round_robin_5fold"},
        )
    result = validate(
        dataset=dataset,
        algorithm="logistic_regression",
        partition_plan=partition,
        metrics=("mcc", "balanced_accuracy", "f1", "roc_auc", "pr_auc"),
        random_state=42,
    )
    print("Live BioSieve:", LIVE_BIOSIEVE)
    print("Partition metadata:", result.partition_plan.metadata)
    return dataset, result, y


@app.cell
def _(dataset, np, pl, result):
    fold_rows = []
    for fold in result.folds:
        eval_y = dataset.subset(fold.evaluation_ids).y
        fold_rows.append({"fold": fold.split, "n_eval": len(eval_y), "positive_rate": float(np.mean(eval_y))})
    fold_df = pl.DataFrame(fold_rows).join(
        result.metrics_frame()
        .filter(pl.col("level") == "fold")
        .pivot("metric", index="split", values="score")
        .rename({"split": "fold"}),
        on="fold",
        how="left",
    )
    fold_df
    return (fold_df,)


@app.cell
def _(fold_df, mo, np, plt, y):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for col in ("mcc", "balanced_accuracy", "f1", "roc_auc", "pr_auc"):
        axes[0].plot(fold_df["fold"], fold_df[col], "o-", label=col)
    axes[0].legend()
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("Fold performance")
    axes[1].bar(fold_df["fold"], fold_df["positive_rate"])
    axes[1].axhline(np.mean(y), linestyle="--")
    axes[1].set(title="Class balance per held-out fold", ylabel="positive rate")
    axes[2].bar(fold_df["fold"], fold_df["n_eval"])
    axes[2].set(title="Held-out fold size", ylabel="samples")
    for ax in axes:
        ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(LIVE_BIOSIEVE, fold_df, np, result):
    DEMO_CHECKS = {
        "five_splits": result.n_splits == 5,
        "source_explicit": result.partition_plan.metadata.get("source")
        == ("biosieve" if LIVE_BIOSIEVE else "demo_prepartitioned_fallback"),
        "finite_metrics": np.isfinite(
            fold_df.select(["mcc", "balanced_accuracy", "f1", "roc_auc", "pr_auc"]).to_numpy()
        ).all(),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
