"""Internal helpers for optional dependency discovery."""

from __future__ import annotations

from importlib.util import find_spec


def is_dependency_available(module_name: str) -> bool:
    """Return whether an optional top-level Python module can be discovered."""
    return find_spec(module_name) is not None
