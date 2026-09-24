from __future__ import annotations

import pytest

from saber.exceptions import ArtifactCompatibilityError
from saber.persistence.environment import compatibility_warnings, environment_snapshot
from saber.persistence.metadata import ARTIFACT_SCHEMA_VERSION, ArtifactManifest


def test_environment_snapshot_contains_reproducibility_core():
    snapshot = environment_snapshot()
    assert "version" in snapshot["python"]
    assert "saberlib" in snapshot["packages"]
    assert "scikit-learn" in snapshot["packages"]
    assert "numpy" in snapshot["packages"]


def test_environment_differences_can_be_warnings_or_errors():
    snapshot = environment_snapshot()
    expected = {
        **snapshot,
        "packages": {
            **snapshot["packages"],
            "scikit-learn": "0.0.0",
        },
    }
    warnings = compatibility_warnings(expected)
    assert any("scikit-learn" in warning for warning in warnings)
    with pytest.raises(ArtifactCompatibilityError):
        compatibility_warnings(expected, strict=True)


def test_artifact_schema_version_is_independent_contract():
    manifest = ArtifactManifest(
        artifact_type="model",
        files={"manifest": "manifest.json"},
    )
    payload = manifest.to_dict()
    assert payload["schema_version"] == ARTIFACT_SCHEMA_VERSION

    payload["schema_version"] = "999.0"
    with pytest.raises(ArtifactCompatibilityError, match="Unsupported artifact schema"):
        ArtifactManifest.from_dict(payload)
