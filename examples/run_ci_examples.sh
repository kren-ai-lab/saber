#!/usr/bin/env bash
set -euo pipefail

for example in examples/[0-9][0-9]_*.py; do
    SABER_DEMO_TEST=1 MPLBACKEND=Agg uv run python "$example"
done
