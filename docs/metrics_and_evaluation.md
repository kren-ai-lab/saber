# Metrics and evaluation

Evaluation is built around `PredictionResult`, not around guessing what an estimator's second probability column means.

```text
PredictionResult
├── predictions
├── probabilities
├── decision_scores
├── classes
├── positive_class
├── sample_ids
└── metadata
```

## Binary classification

Available evaluation metrics include:

| Metric | Requires |
|---|---|
| `accuracy` | predictions |
| `balanced_accuracy` | predictions |
| `precision`, `recall`, `sensitivity`, `specificity`, `f1` | predictions + explicit positive-class semantics |
| `mcc` | predictions |
| `roc_auc`, `pr_auc` | probability **or** decision score |
| `log_loss`, `brier_score` | probabilities |

`positive_class` is explicit. A two-column probability matrix is mapped through the fitted class order rather than assuming that column 1 is always scientifically positive.

## Multiclass classification

Available metrics include accuracy, balanced accuracy, MCC, log-loss, and precision/recall/F1 using canonical weighted semantics plus explicit `*_macro`, `*_micro`, and `*_weighted` variants.

The problem type is determined from the fitted model's class semantics where possible, not merely from whichever classes happen to occur in one held-out fold.

## Regression

Available evaluation metrics are:

```text
mae
median_ae
mse
rmse
mape
r2
explained_variance
pearson
spearman
```

Only requested metrics are computed. This matters for statistics such as correlations that may be undefined for constant predictions.

## Metrics used for tuning

The canonical tuning registry currently exposes task-aware scorer specifications for:

**Classification:** `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1`, `mcc`, `roc_auc`.

**Regression:** `mae`, `mse`, `rmse`, `median_ae`, `r2`, `explained_variance`.

Losses keep their natural positive values in user-facing results while sklearn scorers internally negate them for maximization.

## Fold-local undefined metrics

A valid global binary task may have a held-out fold containing only one observed class. Prediction metrics such as accuracy remain defined, while ranking metrics such as ROC-AUC do not. When such a metric is explicitly requested, `saber` fails clearly rather than returning a misleading number.

## OOF predictions

For complete non-repeated CV, validation reconstructs a sample-aligned `PredictionResult` in original dataset order. OOF outputs can be used for threshold diagnostics, calibration analysis, error auditing, and downstream statistical comparisons without refitting the model.
