# Command-line interface

Each workflow command takes a [YAML or JSON config](configuration.md) and runs it through the same code as the Python API.

## Discover commands

```bash
saber --help
saber doctor
```

## Run workflows

```bash
saber run experiment.yaml
saber train train.yaml
saber evaluate evaluate.yaml
saber validate validate.yaml
saber tune tune.yaml
saber optimize tune.yaml
saber benchmark benchmark.yaml
saber predict predict.yaml
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
saber models list
saber models list --task classification
saber models list --provider sklearn
saber models list --tag baseline
saber models search forest
saber models show random_forest
```

Use `--json` for programmatic discovery.

## Configuration utilities

```bash
saber config validate experiment.yaml
saber config show experiment.yaml
saber config normalize experiment.yaml -o normalized.yaml
```

## Artifacts

```bash
saber artifact inspect artifacts/model
saber artifact verify artifacts/model
```

## Runtime diagnostics

```bash
saber doctor
saber doctor --json
```

This reports Python, saber, scikit-learn, BioSieve, XGBoost, LightGBM, and Optuna availability/version information where applicable.

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | success |
| `2` | invalid configuration or CLI usage contract |
| `3` | saber workflow/domain failure |
| `4` | unexpected internal CLI failure |

Use `--json` for scripts; the default human-readable output may change.
