"""Versioned artifact manifest contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from saber.exceptions import ArtifactCompatibilityError

ARTIFACT_SCHEMA_VERSION = "1.0"
ArtifactType = Literal["model", "benchmark"]


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Small stable manifest independent of joblib/internal result layouts."""

    artifact_type: ArtifactType
    files: dict[str, str]
    schema_version: str = ARTIFACT_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    saber_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the artifact type and file listing."""
        if self.artifact_type not in {"model", "benchmark"}:
            raise ArtifactCompatibilityError(f"Unsupported artifact type '{self.artifact_type}'.")
        if not self.files:
            raise ArtifactCompatibilityError("Artifact manifest must list files.")

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a plain JSON-serializable dictionary."""
        return {
            "schema_version": self.schema_version,
            "artifact_type": self.artifact_type,
            "created_at": self.created_at,
            "saber_version": self.saber_version,
            "files": dict(self.files),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactManifest:
        """Build a manifest from a dictionary produced by :meth:`to_dict`."""
        schema_version = str(payload.get("schema_version", ""))
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactCompatibilityError(
                f"Unsupported artifact schema '{schema_version}'. "
                f"This saber build supports '{ARTIFACT_SCHEMA_VERSION}'."
            )
        return cls(
            schema_version=schema_version,
            artifact_type=payload["artifact_type"],
            created_at=str(payload.get("created_at", "")),
            saber_version=payload.get("saber_version"),
            files=dict(payload.get("files", {})),
            metadata=dict(payload.get("metadata", {})),
        )
