"""Environment capture and compatibility checks for persisted artifacts."""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any

from saber.exceptions import ArtifactCompatibilityError

_TRACKED_PACKAGES = (
    "saberlib",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "xgboost",
    "lightgbm",
    "optuna",
    "biosieve",
)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        if name == "saberlib":
            try:
                from saber import __version__

                return __version__
            except Exception:
                return None
        return None


def environment_snapshot() -> dict[str, Any]:
    """Capture runtime/package versions without importing optional providers."""
    packages = {name: version for name in _TRACKED_PACKAGES if (version := package_version(name)) is not None}
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": packages,
    }


def compatibility_warnings(
    expected: dict[str, Any],
    *,
    strict: bool = False,
) -> tuple[str, ...]:
    """Compare artifact environment with the active runtime.

    Major/minor changes in Python, scikit-learn, or saber are considered
    compatibility-sensitive. Other package-version changes are reported as
    warnings but remain loadable unless ``strict`` is enabled.
    """
    current = environment_snapshot()
    warnings: list[str] = []

    expected_python = str(expected.get("python", {}).get("version", ""))
    current_python = str(current["python"]["version"])
    if expected_python and _major_minor(expected_python) != _major_minor(current_python):
        warnings.append(f"Python version differs: artifact={expected_python}, current={current_python}.")

    expected_packages = expected.get("packages", {})
    current_packages = current.get("packages", {})
    for name, expected_version in expected_packages.items():
        current_version = current_packages.get(name)
        if current_version is None:
            warnings.append(
                f"Package '{name}' was present in the artifact environment "
                "but is not installed in the current environment."
            )
            continue
        if str(expected_version) != str(current_version):
            warnings.append(
                f"Package '{name}' version differs: artifact={expected_version}, current={current_version}."
            )

    if strict and warnings:
        raise ArtifactCompatibilityError(" ".join(warnings))
    return tuple(warnings)


def _major_minor(version: str) -> tuple[int, int] | None:
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None
