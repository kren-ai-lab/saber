# Command-line interface

The CLI uses the same configuration and execution engine as the Python API. It contains no independent estimator, splitting, preprocessing, tuning, or benchmarking logic.

## Discover commands

```bash
mlcore --help
mlcore doctor
```

## Run workflows

```bash
mlcore run experiment.yaml
mlcore train train.yaml
mlcore evaluate evaluate.yaml
mlcore validate validate.yaml
mlcore tune tune.yaml
mlcore optimize tune.yaml
mlcore benchmark benchmark.yaml
mlcore predict predict.yaml
```

Every workflow command supports:

```text
--dry-run       validate and show what would execute
--json          machine-readable summary
--quiet         suppress normal human-oriented output
--no-progress   disable Rich progress/spinner output
```

## Model discovery

```bash
mlcore models list
mlcore models list --task classification
mlcore models list --provider sklearn
mlcore models list --tag baseline
mlcore models search forest
mlcore models show random_forest
```

Use `--json` for programmatic discovery.

## Configuration utilities

```bash
mlcore config validate experiment.yaml
mlcore config show experiment.yaml
mlcore config normalize experiment.yaml -o normalized.yaml
```

## Artifacts

```bash
mlcore artifact inspect artifacts/model
mlcore artifact verify artifacts/model
```

## Runtime diagnostics

```bash
mlcore doctor
mlcore doctor --json
```

This reports Python, mlcore, scikit-learn, BioSieve, XGBoost, LightGBM, and Optuna availability/version information where applicable.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | success |
| `2` | invalid configuration or CLI usage contract |
| `3` | mlcore workflow/domain failure |
| `4` | unexpected internal CLI failure |

Human-readable Rich output is intended for interactive use; `--json` is the stable choice for shell automation and surrounding workflow systems.
