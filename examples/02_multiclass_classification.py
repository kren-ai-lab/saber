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
    # Multiclass classification: metric semantics and class-level diagnostics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Goal
    A three-class problem with imbalanced classes goes through `validate`. The notebook reports a wide metric panel, OOF probabilities, a per-class report and fold stability. The plain names `f1`, `precision` and `recall` are weighted averages, and the panel also includes explicit macro and micro variants.
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
    from sklearn.metrics import ConfusionMatrixDisplay, classification_report
    from saber import validate, DatasetBundle, PartitionPlan

    return (
        ConfusionMatrixDisplay,
        DEMO_TEST,
        DatasetBundle,
        PartitionPlan,
        classification_report,
        make_classification,
        np,
        pl,
        plt,
        validate,
    )


@app.cell
def _(DEMO_TEST, DatasetBundle, PartitionPlan, make_classification, np):
    n_samples = 210 if DEMO_TEST else 480
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=12,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.55, 0.30, 0.15],
        class_sep=1.15,
        random_state=7,
    )
    ids = [f"multi_{i:04d}" for i in range(len(y))]
    dataset = DatasetBundle(X=X, y=y, sample_ids=ids, feature_names=[f"f{i:02d}" for i in range(X.shape[1])])
    plan = PartitionPlan.from_predefined_folds(
        sample_ids=dataset.sample_ids,
        fold_assignments=np.arange(len(y)) % 5,
        dataset=dataset,
        metadata={"source": "demo_external_memberships"},
    )
    return dataset, plan, y


@app.cell
def _(DEMO_TEST, classification_report, dataset, pl, plan, validate, y):
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_micro",
        "mcc",
        "log_loss",
    )
    result = validate(
        dataset=dataset,
        algorithm="random_forest_classifier",
        partition_plan=plan,
        metrics=metrics,
        random_state=17,
        model_params={"n_estimators": 80 if DEMO_TEST else 180, "max_depth": 7},
    )
    oof = result.oof_prediction
    assert oof is not None and tuple(oof.classes) == (0, 1, 2)
    fold_metrics = (
        result.metrics_frame()
        .filter(pl.col("level") == "fold")
        .pivot("metric", index="split", values="score")
        .rename({"split": "fold"})
    )
    class_report_dict = classification_report(y, oof.predictions, output_dict=True, zero_division=0)
    per_class = pl.DataFrame(
        [{"class": key, **value} for key, value in class_report_dict.items() if isinstance(value, dict)]
    )
    print({k: round(result.aggregate_metrics[k], 4) for k in sorted(result.aggregate_metrics)})
    per_class
    return fold_metrics, metrics, oof, per_class, result


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Diagnostic report
    The normalized confusion matrix shows the recall of each class. The second panel plots weighted and macro F1 per fold, a comparison that helps under class imbalance. The last panel is a rough calibration check: the probability the model gave to the true class, split by class.
    """)
    return


@app.cell
def _(ConfusionMatrixDisplay, fold_metrics, mo, np, oof, plt, y):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, oof.predictions, normalize="true", ax=axes[0], colorbar=False)
    axes[0].set_title("Normalized OOF confusion matrix")
    for col in ("f1", "f1_macro", "balanced_accuracy", "mcc"):
        axes[1].plot(fold_metrics["fold"], fold_metrics[col], "o-", label=col)
    axes[1].legend()
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Fold metric stability")
    proba = np.asarray(oof.probabilities)
    for cls in oof.classes:
        axes[2].hist(proba[y == cls, int(cls)], bins=15, alpha=0.55, label=f"true {cls}")
    axes[2].set(title="Probability assigned to true class", xlabel="P(true class)")
    axes[2].legend()
    plt.tight_layout()
    mo.output.append(mo.as_html(fig))
    return


@app.cell
def _(fold_metrics, metrics, np, per_class, result):
    DEMO_CHECKS = {
        "five_folds": result.n_splits == 5,
        "extended_metrics": set(metrics) == set(result.aggregate_metrics),
        "canonical_f1_weighted": np.isclose(
            result.aggregate_metrics["f1"], fold_metrics.get_column("f1").mean()
        ),
        "class_report_has_all_classes": all(
            str(c) in per_class.get_column("class").to_list() for c in (0, 1, 2)
        ),
    }
    assert all(DEMO_CHECKS.values()), DEMO_CHECKS
    DEMO_CHECKS
    return


if __name__ == "__main__":
    app.run()
