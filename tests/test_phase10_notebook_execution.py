from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").rglob("*.ipynb"))


def _script_from_notebook(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = [
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    ]
    return "\n\n# ---- notebook cell ----\n\n".join(cells)


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_executes_in_clean_subprocess(notebook: Path) -> None:
    script = _script_from_notebook(notebook)
    with tempfile.TemporaryDirectory() as directory:
        script_path = Path(directory) / "demo.py"
        script_path.write_text(script, encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["MPLBACKEND"] = "Agg"
        env["SABER_DEMO_TEST"] = "1"
        env["PYTHONWARNINGS"] = "error"
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=directory,
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
    assert completed.returncode == 0, (
        f"Notebook failed: {notebook.relative_to(ROOT)}\n"
        f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
    )
