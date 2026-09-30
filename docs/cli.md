# Command-line interface

`saber run` takes a [YAML or JSON config](configuration.md) and runs it through the same code as the Python API.

```bash
saber --help
saber --version
```

## Run a workflow

```bash
saber run experiment.yaml
saber run experiment.yaml --dry-run
saber run experiment.yaml --json
```

```text
--dry-run   validate and show what would execute
--json      machine-readable summary
```

## Model discovery

```bash
saber models list
saber models list --task classification
saber models list --provider sklearn
saber models list --tag baseline
saber models show random_forest_classifier
```

Use `--json` for programmatic discovery.

## Artifacts

```bash
saber artifact inspect artifacts/model
saber artifact verify artifacts/model
```

## Exit codes

| Code | Meaning |
|---:|---|
| `0` | success |
| `2` | invalid configuration or CLI usage contract |
| `3` | saber workflow/domain failure |
| `4` | unexpected internal CLI failure |

Use `--json` for scripts; the default human-readable output may change.
