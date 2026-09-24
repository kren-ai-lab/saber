# Development Guide

## Prerequisites

- Python 3.11–3.14
- [uv](https://docs.astral.sh/uv/) (package manager)

## Dependencies

Core dependencies are managed via `uv`:
- **Polars** (`polars>=1.44,<2`) is the tabular data engine: dataset loading, result tables, and CSV I/O.
- **NumPy**, **SciPy**, and **scikit-learn** provide the numerical and estimator layer; NumPy is the boundary every estimator receives.
- **Typer** and **Rich** power the CLI.
- **pandas** is not a runtime dependency; it is only in the `dev` group, to test the pandas-input adapter (a pandas DataFrame passed as `X` is converted to Polars once, at the boundary).
- Optional extras: `biosieve` (partition generation), `xgboost`, `lightgbm`, `optuna`, and `all` (every optional integration). The `examples` group installs marimo/matplotlib for the notebooks.

## Setup

```bash
git clone https://github.com/kren-ai-lab/saber
cd saber
uv sync --all-extras
# optional: dependencies for the examples
uv sync --all-extras --group examples
```

## Common Tasks

```bash
uv run task format    # sort imports + ruff format
uv run task lint      # ruff check (no fixes)
uv run task lint-fix  # ruff check --fix
uv run task test      # pytest -q
uv run task test-v    # pytest -v
uv run task test-cov  # pytest + HTML coverage report
uv run task pyrefly   # pyrefly check
```

## Running the CLI

```bash
uv run saber --version
uv run saber --help
```

## Examples

The examples under `examples/` are marimo notebooks stored as plain Python
files (requires the `examples` group):

```bash
uv sync --all-extras --group examples
bash examples/run_ci_examples.sh                          # run them all, as CI does
uv run marimo edit examples/01_binary_classification.py   # open one interactively
```

## Project Structure

The package uses a flat layout:

```text
saber/       # Flat package layout
tests/       # One directory per saber/ package + integration, robustness, examples
examples/    # marimo examples (plain .py) and configs
docs/        # Technical documentation
```

## Making changes

Read [AGENTS.md](AGENTS.md) for the scientific invariants. Until the first
release, the set of scientific methods is closed; fixes and refactors are
welcome. A change to numerical results needs a regression test that pins the
new behavior.

Keep scientific calculations in `saber/` and plotting in `examples/`. The core
must not import Matplotlib or Plotly. The Python API, config-driven execution,
and CLI must agree for the same inputs and options.

Polars is the tabular contract and NumPy is the boundary towards scikit-learn,
XGBoost, and LightGBM estimators. When changing dataset loading or result
tables, preserve sample identity, dataset/partition fingerprints, and the
public Polars result schemas.

## Choosing checks

Run `uv run task lint`, `uv run task pyrefly`, and `uv run task test` before
submitting code changes. Use the suite nearest the affected behavior while
iterating:

| Location | What it checks |
| --- | --- |
| `tests/<package>/` | Public behavior of the corresponding package |
| `tests/integration/` | Composition of workflows across packages |
| `tests/robustness/` | Missing/constant data, degenerate folds, edge-case inputs |
| `tests/examples/` | Static contracts of the example notebooks (execution runs in the examples workflow / `bash examples/run_ci_examples.sh`) |
| `tests/datasets/test_fingerprint_golden.py` | Provenance contract: never update these hashes to make a change pass |

For scientific changes, cover the relevant degenerate case as well as an
estimable case. Verify sample counts/exclusions, statuses, nulls, schemas and
provenance where affected. Stochastic checks should use explicit seeds.

When changing examples or the API they consume, run
`bash examples/run_ci_examples.sh`. This executes all 13 notebooks.

For documentation-only changes, verify local links, referenced public names,
CLI options and runnable snippets. Update the guide for the affected behavior
and keep it about the current API. The [documentation index](docs/README.md)
is the user entry point; examples are catalogued only in
[examples/README.md](examples/README.md).
